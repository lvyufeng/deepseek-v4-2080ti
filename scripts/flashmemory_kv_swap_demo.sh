#!/bin/bash
# FlashMemory + KV_SWAP 演示脚本
# Phase-2 验证通过版本

set -e

CKPT=/mnt/data1/dsv4_inference/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
BIN=/mnt/data1/dsv4_inference/.claude/worktrees/flashmemory-kv-iq1m/cpp_engine/build/tests/test_gguf_generate
NCCL_ID=/tmp/pocketllm_fm_swap_demo_nccl.id
LOG_DIR=/tmp/fm_swap_demo

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       FlashMemory + KV_SWAP 长上下文演示                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "功能："
echo "  - FlashMemory 3-layer ensemble scoring (l10/l12/l20)"
echo "  - KV_SWAP Host-GPU swap with LRU cache"
echo "  - 长上下文支持 + GPU 显存优化"
echo ""

# 选择测试场景
echo "选择测试场景："
echo "  1) 2K context (快速演示, ~5 分钟)"
echo "  2) 8K context (显存节省演示, ~15 分钟)"
echo "  3) Custom (自定义 context length)"
echo ""
read -p "选择 [1-3]: " choice

case $choice in
  1)
    CONTEXT=2048
    GPU_CHUNKS=520
    ;;
  2)
    CONTEXT=8192
    GPU_CHUNKS=512
    ;;
  3)
    read -p "输入 context length (tokens): " CONTEXT
    read -p "输入 GPU_CHUNKS [推荐 512]: " GPU_CHUNKS
    GPU_CHUNKS=${GPU_CHUNKS:-512}
    ;;
  *)
    CONTEXT=2048
    GPU_CHUNKS=520
    ;;
esac

read -p "生成多少 tokens? [默认 64]: " MAX_NEW
MAX_NEW=${MAX_NEW:-64}

echo ""
echo "配置："
echo "  Context: $CONTEXT tokens"
echo "  GPU_CHUNKS: $GPU_CHUNKS"
echo "  Generate: $MAX_NEW tokens"
echo ""

# 生成测试输入
echo "生成测试输入..."
python3 << EOF
import random
random.seed(42)

tokens = []
for i in range($CONTEXT):
    token_id = random.randint(1000, 100000)
    tokens.append(str(token_id))

with open('/tmp/fm_swap_input.txt', 'w') as f:
    f.write(' '.join(tokens))

print(f"✓ Generated {$CONTEXT} tokens")

# 预期 compressed_cap
cap_4 = ($CONTEXT + 3) // 4
cap_128 = ($CONTEXT + 127) // 128

print()
print(f"预期 compressed_cap:")
print(f"  Ratio=4 (20 layers): {cap_4}")
print(f"  Ratio=128 (21 layers): {cap_128}")
print()

if cap_4 > $GPU_CHUNKS:
    print(f"✓ Ratio=4 layers: {cap_4} > {$GPU_CHUNKS} → KV_SWAP 将触发")
else:
    print(f"⚠ Ratio=4 layers: {cap_4} ≤ {$GPU_CHUNKS} → 不会触发 swap (全驻留更快)")
EOF

mkdir -p "$LOG_DIR"
rm -f "$NCCL_ID"

# FlashMemory + KV_SWAP 配置
export POCKETLLM_GGUF_FLASHMEMORY_PLUGIN=1
export POCKETLLM_GGUF_FLASHMEMORY_LOAD_DEVICE=1
export POCKETLLM_GGUF_FLASHMEMORY_RUNTIME_SCORING=1
export POCKETLLM_GGUF_FLASHMEMORY_ENSEMBLE=max
export POCKETLLM_GGUF_FLASHMEMORY_CKPT=/tmp/FlashMemory-Deepseek-V4/weights/flashmemory_ds_v4.safetensors
export POCKETLLM_GGUF_SPARSE_COMPRESSOR=1
export POCKETLLM_GGUF_SPARSE_INDEXER=1
export POCKETLLM_GGUF_KV_SWAP=1
export POCKETLLM_GGUF_KV_SWAP_GPU_CHUNKS=$GPU_CHUNKS
export POCKETLLM_GGUF_MEM_PROFILE=1

echo ""
echo "启动推理..."
echo ""

# 运行 TP4 推理
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank $BIN "$CKPT" $MAX_NEW --seed-file /tmp/fm_swap_input.txt \
    --tp-world 4 --tp-rank $rank --device 0 --nccl-id-path "$NCCL_ID" \
    > "$LOG_DIR/rank${rank}.log" 2>&1 &
done

wait

echo "════════════════════════════════════════════════════════════════"
echo "  推理完成"
echo "════════════════════════════════════════════════════════════════"
echo ""

RANK0="$LOG_DIR/rank0.log"

# 显示结果
echo "⚡ FlashMemory Configuration:"
grep "gguf_flashmemory_runtime_scoring=" "$RANK0"
echo ""

echo "🔄 KV_SWAP Status:"
grep "gguf_kv_swap=" "$RANK0"
echo ""

if grep -q "gguf_kv_swap_stats" "$RANK0"; then
  echo "💾 Swap Statistics:"
  grep "gguf_kv_swap_stats" "$RANK0"
  echo ""

  # 计算命中率
  STATS=$(grep "gguf_kv_swap_stats" "$RANK0")
  HITS=$(echo "$STATS" | grep -oP 'hits=\K[0-9]+')
  MISSES=$(echo "$STATS" | grep -oP 'misses=\K[0-9]+')

  if [ -n "$HITS" ] && [ -n "$MISSES" ]; then
    TOTAL=$((HITS + MISSES))
    if [ $TOTAL -gt 0 ]; then
      HIT_RATE=$(python3 -c "print(f'{$HITS * 100 / $TOTAL:.2f}')")
      echo "📊 LRU Cache Hit Rate: ${HIT_RATE}%"
      echo "   Hits: $HITS, Misses: $MISSES"
      echo ""
    fi
  fi
fi

echo "📦 KV Cache:"
grep "gguf_kv_cache " "$RANK0"
echo ""

echo "📊 Memory:"
grep "gguf_mem tag=after_kv_cache" "$RANK0"
echo ""

echo "📈 Performance:"
grep "prefill_seconds\|decode_seconds" "$RANK0"
echo ""

echo "✅ Status:"
grep "PASS\|FAIL" "$RANK0"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "完整日志: $LOG_DIR/rank*.log"
echo ""
echo "总结:"
echo "  - 如果看到 gguf_kv_swap=1 → KV_SWAP 已启用"
echo "  - 如果看到 swap stats → Host-GPU swap 正在工作"
echo "  - 高 hit rate (>99%) → LRU cache 效果很好"
echo "  - h2d_mib 小 → swap 开销很低"
echo ""
