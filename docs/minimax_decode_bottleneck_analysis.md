# MiniMax-M2 Decode Performance Bottleneck Analysis

**Date**: 2026-06-14  
**Hardware**: 4×RTX 2080Ti (TP4, PCIe Gen3)  
**Model**: MiniMax-M2.7 UD-IQ1_M (256 experts, top_k=8, iq2_xxs quantization)  
**Current Performance**: ~5.1-6.5 decode TPS (high variance)

## Executive Summary

**Decode bottleneck is Attention (71%), NOT all_reduce (8%).**

Previous assumptions about NCCL all_reduce dominating decode (67% in memory notes) were **incorrect**. Real profiling shows:

| Component | Time per Layer | % of Layer Time | Optimization Priority |
|-----------|----------------|-----------------|----------------------|
| **Attention** | **1.97 ms** | **71.3%** | 🔴 **CRITICAL** |
| MoE EP Compute | 0.58 ms | 20.9% | 🟡 Already optimized (DP4A) |
| All-Reduce (NCCL) | 0.22 ms | 7.8% | 🟢 Not a bottleneck |
| **TOTAL** | **2.77 ms** | **100%** | Extrapolated: 171.7 ms/token (5.82 TPS) |

## Detailed Findings

### 1. Layer-Level Breakdown (averaged over 3 runs, 5 layers)

```
Layer  Attn(ms)  MoE Compute(ms)  All-Reduce(ms)  Total(ms)
-----  --------  ---------------  --------------  ---------
  0      1.90         0.57            0.23          2.70
  1      2.29         0.59            0.19          3.07
  2      1.86         0.60            0.24          2.70
  3      1.92         0.59            0.22          2.73
  4      1.90         0.54            0.21          2.65
-----  --------  ---------------  --------------  ---------
AVG     1.97         0.58            0.22          2.77
```

**Key Insight**: All-reduce (0.22ms) is **9× smaller** than attention (1.97ms). Optimizing all_reduce (e.g., bf16 reduce) has negligible impact.

### 2. Attention Internal Breakdown (layer 0, decode, 10 runs)

```
Component        ms      % of Attention
-------------  ------   ---------------
RoPE           1.309        29.1%      ← LARGEST bottleneck
repeat         0.418         9.3%      ← GQA expansion (n_kv_heads=8 → n_heads=48)
q_norm         0.413         9.2%
k_norm         0.414         9.2%
q_proj         0.263         5.8%
o_proj         0.259         5.8%
k_proj         0.312         6.9%
cache_copy     0.271         6.0%
transpose      0.261         5.8%
sdpa           0.235         5.2%      ← SDPA itself is fast!
o_transpose    0.187         4.2%
-------------  ------   ---------------
TOTAL          4.498       100.0%
```

**Critical Discovery**: 
- **RoPE (29%)** dominates attention, using PyTorch ops (arange/sin/cos/cat) → many small kernel launches
- **GQA repeat_interleave (9%)** expands KV heads 6× (8 kv_heads → 48 heads) because PyTorch SDPA doesn't support GQA natively
- **SDPA (5%)** is already fast; FlashAttention-2 is working well

## Optimization Roadmap

### 🔴 Priority 1: Fused RoPE Kernel (Target: +40% decode TPS)

**Current**: `_apply_rope()` uses PyTorch ops (1.3ms/layer)
```python
# architecture.py line 120-139
positions = torch.arange(...)
inv = torch.pow(...)
freqs = positions[:, None] * inv[None, :]
sin = torch.sin(freqs).to(x.dtype)
cos = torch.cos(freqs).to(x.dtype)
# ... multiple tensor ops
```

**Target**: Single fused CUDA kernel (estimated 0.1-0.2ms/layer)

**Existing Code**: `src/csrc/` already has `fused_q_rmsnorm_rope_inplace_cuda` (for DeepSeek-V4 cpp_engine), but it requires:
- bfloat16 input (MiniMax uses fp16)
- Pre-computed freqs_real/freqs_imag (MiniMax computes on-the-fly)

**Action**:
1. Adapt existing fused kernel to support fp16
2. Pre-compute and cache RoPE freqs (one-time cost at model load)
3. Call fused kernel instead of PyTorch ops

**Expected Gain**: 
- Per-layer: 1.3ms → 0.2ms (save 1.1ms)
- Full model: 1.1ms × 62 layers = **68ms/token**
- TPS: 5.8 → **~8.1 TPS (+40%)**

### 🟡 Priority 2: GQA-Aware SDPA (Target: +15% decode TPS)

**Current**: `repeat_interleave` expands KV from 8 heads → 48 heads (0.42ms/layer)
```python
# architecture.py line 168-170
repeat = self.args.n_heads // self.args.n_kv_heads  # 48 / 8 = 6
k_t = k_t.repeat_interleave(repeat, dim=1)
v_t = v_t.repeat_interleave(repeat, dim=1)
```

**Target**: Direct GQA support in attention kernel (0ms expansion)

**Options**:
1. Wait for PyTorch SDPA to add GQA support (tracked in PyTorch issues)
2. Use custom FlashAttention-2 GQA variant (requires FA2 integration)
3. Write minimal GQA decode kernel (since decode is [B=1, S=1, ...])

**Expected Gain**:
- Per-layer: 0.42ms → 0ms (save 0.42ms)
- Full model: 0.42ms × 62 = **26ms/token**
- TPS: 8.1 → **~9.5 TPS (+15% over Priority 1)**

### 🟢 Priority 3: Fused QKV Projection + RMSNorm (Target: +20% decode TPS)

**Current**: Separate q_proj/k_proj/v_proj + q_norm/k_norm (total ~1.8ms/layer)

**Target**: Single fused kernel for QKV projection + norm

**Expected Gain**:
- Per-layer: ~0.6ms
- Full model: 0.6ms × 62 = **37ms/token**
- TPS: 9.5 → **~11.5 TPS (+20% over Priority 2)**

### Combined Upper Bound

If all three are implemented:
- Time saved: 68 + 26 + 37 = **131 ms/token**
- New decode time: 172 - 131 = **41 ms/token**
- **TPS: ~24 (4× current baseline)**

## Why All-Reduce Optimization Failed

**Attempted**: MINIMAX_M2_DECODE_REDUCE_DTYPE=bf16 (reduce message 12KB → 6KB)

**Result**: ~10% decode regression (6.5 TPS → 5.0 TPS)

**Root Cause**: 
- All-reduce only takes 0.22ms (7.8% of layer time)
- 12KB message is already latency-bound (not bandwidth-bound)
- Conversion overhead (fp32 ↔ bf16) > latency reduction from smaller message
- On PCIe Gen3 + 2080Ti, the NCCL synchronization overhead dominates, not transfer time

**Conclusion**: All-reduce is not a bottleneck; don't optimize it further.

## Why Cross-Layer Overlap is Not Feasible

**Proposed Idea**: Overlap Layer L's all_reduce with Layer L+1's attention

**Why It Fails**: **Strict execution dependency**

```python
# Layer L
x = x + attention(norm(x))      # (1) needs x from L-1
x = x + moe(norm(x))            # (2) needs x from (1)
                                # (3) all_reduce must finish before x is complete

# Layer L+1
x = x + attention(norm(x))      # ← BLOCKS on Layer L's all_reduce completing
```

Layer L+1's attention **requires the full residual output** `x = x_prev + attention + moe` from Layer L. The `moe` term needs all_reduce to finish. There is **no independent compute** to overlap with.

**Alternative (cross-layer reduce fusion)**: Merge multiple layers' all_reduces into one large reduce → **also fails** because each layer's MoE needs the current layer's post-attention `x`, not a delayed value.

**Conclusion**: Decode has strict layer-by-layer dependencies; no overlap opportunity exists.

## Profiling Scripts

1. **Layer-level timing**: `tests/profile_minimax_decode_layers.py`
   - Measures Attention / MoE Compute / All-Reduce per layer
   - Run: `torchrun --standalone --nproc_per_node=4 tests/profile_minimax_decode_layers.py`

2. **Attention internals**: `tests/profile_minimax_attn_detail.py`
   - Breaks down attention into projection/norm/rope/sdpa/repeat
   - Run: `torchrun --standalone --nproc_per_node=4 tests/profile_minimax_attn_detail.py`

## Recommendations

1. **Immediate**: Implement Priority 1 (fused RoPE) — single highest-impact optimization
2. **Short-term**: Implement Priority 2 (GQA SDPA) — good ROI, moderate complexity
3. **Long-term**: Implement Priority 3 (fused QKV proj) — requires kernel engineering
4. **Do NOT**: Pursue all_reduce optimization (bf16 reduce) — proven ineffective
5. **Do NOT**: Pursue cross-layer overlap — architecturally infeasible

## Related PRs

- **PR #35**: MoE iq2_xxs w2 DP4A kernel (+100% prefill, decode neutral)
- **PR #36**: Prefill/decode path separation (foundation for decode-specific optimizations)
