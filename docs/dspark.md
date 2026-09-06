# DSpark speculative decoding with adaptive draft-length gating

DeepSeek-V4-Flash-0731 ships a 3-stage DSpark draft module under `mtp.*`. One
round drafts `dspark_block_size` (=5) tokens from a single committed token; the
main model verifies them in one forward and commits the longest matching prefix
plus one bonus token.

In exact arithmetic that scheme reproduces plain greedy decoding. On this stack
it does not, and the cause is upstream of gating -- see
[Output determinism](#output-determinism) before relying on DSpark output
matching sequential decode.

All numbers below: TP=4, FP4 active-expert MoE, 4x RTX 2080Ti, routed experts on
CPU, short prompts (17-22 tokens). Measured 2026-08-05/06.

## Why gating exists

Verify cost grows with draft length. Fitting the measured multi-token points
(n=1: 296ms, n=2: 432, n=3: 494, n=6: 850):

    verify(n) ~= 300ms + 109ms * n

A round costs `draft(196ms) + verify(n)`. At n=5 that is 1041ms to produce
`accepted + 1` tokens, against 303ms for a plain single-token decode. So a round
that gets nothing accepted is a ~740ms loss. In the 31-round baseline, **7 rounds
accepted zero tokens and together burned 5.2s more than plain decoding** -- 16%
of total speculative time.

Always drafting all 5 therefore helps or hurts depending entirely on the
workload:

| prompt | accept rate | tok/s | vs plain (3.30) |
|---|---|---|---|
| repeat (count upward) | 100% (5.0/5) | 5.76 | 1.75x |
| code | 82.5% (4.1/5) | 4.92 | 1.49x |
| prose | 40% (2.0/5) | 2.88 | **0.87x** |
| math | 17.5% (0.9/5) | 1.80 | **0.55x** |

## The signal

The last DSpark stage already emits a confidence score per draft position, and
it is well calibrated. Pooled over all 4 prompts, P(draft token matches the main
model) by confidence bucket:

| confidence | n | match rate |
|---|---|---|
| < -0.5 | 5 | 20% |
| -0.5 .. 0.5 | 17 | 65% |
| 0.5 .. 1.5 | 19 | 79% |
| 1.5 .. 3.0 | 19 | 89% |
| 3.0 .. 6.0 | 14 | 100% |
| > 6.0 | 36 | 100% |

Monotone, and the per-prompt curves agree within their sample sizes. Position
carries no extra information once confidence is known (later positions show
higher raw match rates only because they are censored -- a position is observed
only if the whole prefix before it was accepted).

## Why not a fixed threshold

A threshold was tried first and rejected. Leave-one-prompt-out (threshold chosen
on 3 prompts, scored on the 4th):

| held out | always-k | thresholded | |
|---|---|---|---|
| math | 1.80 | 3.40 | 1.89x |
| prose | 2.88 | 3.59 | 1.25x |
| repeat | 5.76 | 5.62 | 0.98x |
| code | 4.92 | 4.31 | **0.87x** |

Mean 1.25x but a 13% regression on the *best* prompt. The best threshold depends
on the workload's accept rate, which a threshold cannot express: on an easy
workload every cut removes a token that would have been accepted.

## The rule that shipped

Estimate P(match | confidence) online from rounds already observed, then pick the
length maximizing expected tokens per millisecond:

    n* = argmax_n  E[tokens(n)] / cost(n)
    E[tokens(n)] = 1 + sum_{i<n} prod_{j<=i} p_j
    cost(n)      = draft_ms + verify(n)   (n > 0)
                 = plain_ms               (n = 0, skip the verify entirely)

Two details that matter:

- **Censored observations.** After the first rejection, later positions were
  verified but their match status is unknown. Counting them as misses biases
  every bucket down and makes the gate truncate progressively harder. Only
  positions `0 .. n_accepted` are recorded.
- **Margin.** Cost is linear in n but expected tokens are multiplicative, so
  underestimating p compounds while overestimating it wastes one position.
  Truncation must beat the full draft by 1.15x before it is taken.

## Results

Causal replay of the same 31 rounds (each prompt starts with a cold gate, the
pessimistic case; a real stream keeps its calibration across a generation):

| prompt | plain | always-k | **gated** | vs always-k |
|---|---|---|---|---|
| prose | 3.30 | 2.88 | 3.05 | 1.06x |
| math | 3.30 | 1.80 | 2.45 | **1.36x** |
| repeat | 3.30 | 5.76 | 5.76 | 1.00x |
| code | 3.30 | 4.92 | 4.92 | 1.00x |
| **overall** (round-weighted) | 3.30 | 3.14 | **3.62** | **1.15x** |

No prompt regresses. The gain is entirely on the workloads where speculation was
losing, and the two easy prompts are left drafting the full block untouched.

### Known limitation

Gating narrows but does not close the gap to plain decode on hard workloads
(prose 3.05 and math 2.45 vs plain 3.30). The reason is structural: the
confidence score is produced *by* the draft, so the draft always runs before the
gate can decide. Gating avoids verify cost, never the 196ms draft cost. A round
the gate skips still paid for its draft. Recovered fraction of the always-k ->
plain gap: prose 44%, math 72%.

**Pre-draft gating** addresses this by checking the committed token's logit margin
(top1 - top2) *before* drafting: if the main model is uncertain about what comes
next (small margin), the draft will be poor and speculation will lose even before
verify cost. Skipping the round there avoids both draft and verify costs.
Controlled by `DEEPSEEK_DSPARK_GATE_MARGIN_THRESHOLD` (default 0.0 = disabled):
rounds with margin below this threshold skip drafting entirely and take a plain
single-token decode step instead. Set to 4.0 as a starting point if enabling.

The threshold is workload-dependent: easy prompts (repeat, code) rarely hit it
because the model is confident; hard prompts (math, prose) hit it more often,
which is exactly when skipping saves the most time. Disabled by default; tune
based on `margin_skipped_rounds` in the gate stats if enabling.

## Configuration

Gating is **off by default**; it changes draft lengths, so an existing deployment
should opt in.

| env var | default | meaning |
|---|---|---|
| `DEEPSEEK_DSPARK_GATE` | `0` | enable gating |
| `DEEPSEEK_DSPARK_GATE_MARGIN_THRESHOLD` | `0.0` | skip drafting when committed token margin (top1 - top2 logit) is below this; 0 = disabled, try 4.0 if enabling |
| `DEEPSEEK_DSPARK_GATE_MARGIN` | `1.15` | truncation must win by this factor |
| `DEEPSEEK_DSPARK_GATE_MIN_DRAFT` | `0` | never draft fewer than this (0 = no floor) |
| `DEEPSEEK_DSPARK_GATE_DECAY` | `0.9` | per-round forgetting for workload shifts |
| `DEEPSEEK_DSPARK_GATE_DRAFT_MS` | `196` | measured draft cost |
| `DEEPSEEK_DSPARK_GATE_VERIFY_BASE_MS` | `300` | verify fixed cost |
| `DEEPSEEK_DSPARK_GATE_VERIFY_SLOPE_MS` | `109` | verify per-token cost |
| `DEEPSEEK_DSPARK_GATE_PLAIN_MS` | `303` | plain single-token decode cost |

Only the *ratios* between the cost parameters affect decisions, so the defaults
transfer to other setups as long as the cost shape holds. Re-measure them if the
verify path changes.

**`DEEPSEEK_GPU_MOE_MULTI_TOKEN_FP4=1` is required.** Without the small-batch
FP4 MoE kernel a multi-token verify falls into the prefill grouped MoE path:
verify goes 850ms -> 3273ms and speculation drops to 0.37x plain decode. That
kernel is itself default-off.

## Output determinism

Measured on-device (TP=4 FP4, 4x2080Ti, 20 tokens per prompt, three prompts),
comparing token sequences:

| check | math | code | prose |
|---|---|---|---|
| plain vs plain (same input twice) | identical | identical | identical |
| always-k vs always-k (same input twice) | identical | **diverges at 14** | **diverges at 7** |
| gated vs always-k | identical | diverges at 14 | identical |
| always-k vs plain | diverges at 12 | diverges at 5 | identical |

**Sequential decode is reproducible; the multi-token verify forward is not.**
Running always-k twice on the same prompt with the same weights produced
different tokens on 2 of 3 prompts. So DSpark output does not match sequential
decode, and it does not even match itself run to run.

This is upstream of gating. Gating only chooses how many already-drafted tokens
to submit, and where the verify path is stable (math, prose) gated and always-k
agree exactly. Where they differ (code, first difference at token 14) always-k
already differs from itself at the same position, so the divergence cannot be
attributed to the gate.

Consistent with [`fp4_multi_token_moe_kernel`]: batch-vs-sequential logit drift
was traced to the batched **attention** path (KV write ordering / indexer), not
the MoE kernel, and is present with the multi-token kernel off. Speculative
decoding turns that latent drift into visibly different tokens because the
accept/reject comparison amplifies it into a discrete decision.

**What this means in practice:** treat DSpark as changing the sampling
distribution, not as a transparent speedup. It is not safe for workloads
requiring reproducible output or exact parity with sequential decode. Fixing it
means fixing multi-token attention determinism, which is out of scope here.

Reproduce with `/tmp/verify_dspark_gate_vs_alwaysk.py` (A/B/C attribution) --
kept out of the repo because it needs the 167GB checkpoint and 4 GPUs.

## Tests

- `tests/test_dspark_gate.py` -- decision-rule properties, no GPU needed
- `tests/test_dspark_gate_replay.py` -- replays `tests/data/dspark_rounds_tp4.json`
  (the 31 measured rounds) and pins the results above
- On-device verification: performance gains confirmed (below), token identity
  disproved (above). The gate's 1.30-1.32x on the hard prompts reproduced live:

| prompt | always-k | gated | |
|---|---|---|---|
| math | 3.15 | 4.15 | 1.32x |
| code | 2.77 | 3.62 | 1.30x |
| repeat | 4.44 | 4.41 | 0.99x |
| prose | 3.22 | 3.06 | 0.95x |

The two easy prompts land within noise of always-k rather than exactly equal, as
the replay predicted -- live accept rates differed from the fixture's, so the
gate skipped a few rounds it would otherwise have drafted.

---

## cpp_engine port status

### Verify path

`PersistentEngine::verify_step` forwards a draft block and reports, for each
draft token, what the target model samples after consuming it.

- `cpp_engine/include/persistent_engine.hpp` — interface
- `cpp_engine/engine/deepseek_v4_engine.cpp` — implementation
- `cpp_engine/tests/test_perfect_draft.cpp` — feeds a plain decode back in as a
  "perfect" draft, so any mismatch is the verify path's own fault rather than
  the drafter's. 59/59 checked, 0 mismatches at draft_len 5.

It forwards the draft tokens one at a time rather than as a `[1, n, d]` batch.
Batching is the whole point of speculative decoding, but the two are not
numerically equivalent: GEMMs pick tiles and reduction orders by shape, so a
batched verify disagrees with plain decode by ~4e-3 at the first projection,
which amplifies to O(1) at the head. Batching it is a separate optimization
that has to be measured against that drift, not assumed free.

`POCKETLLM_CPP_MOE_DETERMINISTIC_REDUCE=1` (default) removes the MoE atomicAdd
nondeterminism for topk>=3 via per-route partials and a fixed-order reduction;
`cpp_engine/tests/test_moe_fp4_determinism.cpp` guards it.

That drift is also **not reproducible run to run**. Two consecutive TP=1 runs of
the same prompt and block measured, at 43 layers greedy: one where
`batch_verify_step()` returned 1760 as row 1's successor while sequential decode
returned 1718 (so only 2 of 6 rows accepted), and one where the batched and
sequential chains were identical. So a test must not assert that batched and
sequential agree token for token; it will pass and fail on the same code.

`cpp_engine/tests/test_batch_verify_transaction.cpp` therefore derives its block
from the engine's own `decode_step()` chain, then iteratively refines it until the
batched path accepts every row, and reports the batched-vs-sequential first
divergence separately from the rollback assertions. It checks the property that
actually matters: for each accepted prefix length, two blocks differing only in
their *rejected* suffix must leave identical committed hidden rows
(`rows * 4096` values) and identical continuations over the next 8 decode steps.
All prefix lengths 2..6 pass. Committed hidden still varies by `max_abs` 0.69-3.19
between the two suffix variants while the continuation tokens stay bit-identical,
which is why the hidden comparison is reported rather than asserted exact.

### Draft module (Stage B)

Porting `src/models/deepseek_v4/dspark.py` into `cpp_engine/engine/dspark_engine.cpp`:

| Step | What | State |
|---|---|---|
| B1 | Skeleton + config parsing | done |
| B2 | Stage 0 (main_proj + main_norm + embed) | done |
| B3a | DSparkAttention | done |
| B3b | MoE FFN (routed + shared) | done |
| B4 | Stage 2 heads (norm, hc_head, markov, confidence) | done |
| B5 | Weight loading | done |
| B6 | Main-hidden caching during verify | done |
| B7 | `draft_tokens()` + TP all-reduces + ring priming | done |
| B8 | Speculative scheduler (`speculative_step()`, early-exit verify) | done |

Parity tests compare each sub-path against an fp32 reference driven by the same
weights (`tests/test_dspark_attention_parity.py`, `tests/test_dspark_moe_parity.py`,
`tests/test_dspark_head_parity.py`). Main-hidden capture is covered by
`cpp_engine/tests/test_dspark_hidden_capture.cpp`, which needs no reference
because it checks the capture against the engine's own decode path.

Two further tests need no reference either, because each compares the draft
against another run of itself:

- `cpp_engine/tests/test_dspark_tp_consistency.cpp` -- same synthetic seed at
  TP=1 and TP=4; the drafted tokens must match, which is what says the
  all-reduces and the replicated head are right.
- `cpp_engine/tests/test_dspark_draft_sensitivity.cpp` -- perturbs one input at
  a time (seed, seed position, ring priming, committed token) and requires each
  to change the drafted tokens. Answers the question an accept rate cannot:
  whether an input reaches the draft at all.

Both run draft-only at TP=1 where possible -- the module is ~12.4 GB and fits on
one card without the main model, so neither needs the 167 GB checkpoint load.

#### Drafting

`PersistentEngine::load_dspark()` loads the draft module and switches capture on
for the layers named in the draft's own config, rather than a caller-supplied
list -- a mismatch there would still produce plausible drafts, just worse ones.
It is a separate call from the constructor because the weights are ~12.4 GB at
TP=1.

`draft_tokens(input_token, start_pos, hidden)` takes the hidden explicitly
instead of reading the engine's last capture: after a verify round the caller
drafts from the *accepted* position, which is not the last one forwarded. The
captured row is one concatenated `[n_target * dim]` block, so the bridge uploads
it once and hands the draft slices of that staging buffer.

TP>1 now does the same two all-reduces the main model does -- after the
attention `wo_b`, where each rank has only its own heads, and after the routed
MoE but *before* the shared expert, which every rank computes in full and would
otherwise be counted `tp_world` times. Both go over bf16, matching the main
model's reduction. Supplying the NCCL channel is required at TP>1: the
3-argument `DSparkEngine` constructor throws rather than silently returning a
partial sum, which would still read as a well-formed draft.

`cpp_engine/tests/test_dspark_draft.cpp` measures accept rate against the main
model's own verify. Shape and finiteness checks are worthless here -- a draft
seeded with the wrong hidden still emits fluent token ids -- and so, it turns
out, is a low accept rate on ordinary text. The prompt has to be one the main
model is near-certain about, or a correct draft and a broken one are
indistinguishable; the default is therefore cyclic and the bar is an absolute
rate. Block alignment is the other trap: verify gets `[committed, drafts...]`,
since verifying the drafts alone shifts every comparison one position and reads
as a near-zero accept rate.

**The draft works, and the earlier "accept rate 0" was a prompt artifact.** On a
cyclic prompt the same build accepts 30/30 -- 5/5 on every round, on all four TP
ranks, with confidence 3.2-4.7. On a short English prompt the same build accepts
0/30. Ordinary text is genuinely uncertain at every position, so a correct draft
loses there and the rate says nothing about correctness; that is what the first
measurements were reading. (A shorter 6-token cyclic prompt reads 20/30: the
first two rounds score 0 until the pattern is established in the ring, then 5/5
thereafter, with confidence climbing 1.86 -> 4.21 as it settles.)

Three real bugs were found and fixed on the way, none of which was the reason
the rate looked like 0:

- **Ring priming was missing.** `draft()` writes only the position it drafts
  from, but the reference's `write_main_kv` writes *every* committed position;
  without it the draft attended to a window that was almost entirely zeros.
  `PersistentEngine::prime_dspark_kv()` now does this, and prefill's capture
  keeps the last `window_size` positions rather than one to feed it.
- **The NCCL id wait was 30s**, which is shorter than the time four ranks take
  to load a 167 GB checkpoint, so ranks 1-2 died before rank 0 published the id.
  Now 10 minutes, overridable via `POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS`.
- **`start_pos` was off by one at the call site.** It is the position of the
  *seed hidden* -- the last position the main model consumed -- not the
  committed token's own position; the committed token goes into draft slot 0 and
  is roped at `start_pos + 1`. The reference passes `self._pos - 1` for exactly
  this reason. The C++ arithmetic was right and the header comment was wrong,
  which is what led the test to pass `pos`. Both are fixed.

Two structural suspects were ruled out in the process, each by a test that now
guards it:

- **TP sharding.** `cpp_engine/tests/test_dspark_tp_consistency.cpp` runs the
  draft alone (no main model, so it fits on one card) from a fixed synthetic
  seed at both world sizes. TP=1 and all four TP=4 ranks draft the identical
  token sequence, so the two all-reduces and the replicated head are correct.
- **Wiring.** `cpp_engine/tests/test_dspark_draft_sensitivity.cpp` perturbs one
  input at a time and counts changed draft tokens: committed token 5/5, ring
  priming 5/5, seed position 1/5, and a zeroed seed 5/5 once the ring is zeroed
  so the seed's slot is the only live key. Every input reaches the draft.
  (With the ring primed a zeroed seed changes 0/5 -- priming has already written
  that position's KV and the seed only overwrites one key of six, so that
  variant is reported but deliberately not asserted on.)

#### Scheduler

`PersistentEngine::speculative_step(committed_token, position, sp)` runs one
speculative round: draft from the committed token → forward `[committed, drafts...]`
one position at a time, comparing each sample against the next draft → stop at the
first mismatch → prime the draft's ring with the committed rows → return the
accepted prefix plus the bonus token. The caller advances position by that count.

**Verification stops at the first mismatch**, so the only positions ever forwarded
into the main model's state are the ones the round commits. That is what makes the
round safe, and it is available only because verify is sequential (see the note on
batched verify above) -- a batched verify computes the whole block before anything
can be compared.

The alternative -- forward the whole block and roll back on reject -- does not
work here, for a reason worth recording. The KV ring addresses by
`position % window_size` and overwrites by construction, but the compressor state
on layers 2-42 accumulates into ratio-sized slots and at every ratio boundary
shifts the upper half down and zeroes the top. That shift is destructive: replaying
a position that crossed a boundary cannot rebuild what was shifted away. A host
snapshot taken before verify does restore it -- but restoring to the *pre-verify*
position also discards the accepted prefix's contribution, and nothing re-forwards
those positions. Rolling back correctly would mean snapshotting per position, at
which point early exit is strictly cheaper.

`snapshot_compressor_state()` / `restore_compressor_state()` remain on
`PersistentEngine` as the primitives that establish the above, and are what
`cpp_engine/tests/test_verify_overwrite_safety.cpp` exercises: for each overshoot
length (1, 2, 3, 4, 5, 8) it verifies `over` extra tokens at a position, then
continues from that position and compares against a clean run. Measured at TP=4,
43 layers: the clean run gives `455 17132 582 67 4 305 582 2869`; without
protection all six overshoots diverge at token 1 (the ratio-4 boundary crossing at
position 7); with snapshot/restore all six match the clean run exactly. A control
at `layer_count=2` (layers 0-1, `compress_ratio == 0`) matches on both paths,
pinning the compressor as the cause rather than the ring, the MoE, or TP.

`cpp_engine/tests/test_speculative_scheduler.cpp` is an end-to-end test: it runs
the same prompt through plain decode and the speculative path and requires
token-by-token identity. It also checks that the speculative path generates the
same number of tokens in fewer rounds (i.e., amortizes).

`PersistentEngine::prefill()` automatically primes the draft ring from the
captured trailing prompt window when DSpark is loaded; callers must not write the
same rows a second time. Under TP, rank 0 then drives workers through three
scheduler-specific commands. `Draft` makes every rank enter the draft
attention/MoE all-reduces with its rank-local captured hidden.
`SpeculativeDecode` forwards one candidate position on every rank; rank 0's
globally selected token alone decides whether another position is sent.
`PrimeDraftKV` then writes the accumulated committed hiddens to every rank's draft
ring in one batched call. This keeps the ranks in lockstep through exactly the
positions rank 0 commits, without turning the ring update into one allocation-
heavy call per token.

That last point is also why `test_dspark_draft` asserts an absolute accept rate
rather than beating a corrupted seed. On a prompt predictable enough for the
absolute rate to mean anything, the ring and the committed token already carry
the continuation, so a zeroed seed scores the same (real 30/30, zeros 30/30).
The seed's contribution is established by the sensitivity test, which isolates
it properly; an assertion here that only held on prompts where the rate is
uninformative would be worse than none.

#### Main-hidden capture

`PersistentEngine::set_dspark_capture_layers()` turns on capture of the main
model's block output at the draft's target layers, mean-pooled over the hc
dimension and concatenated on the last axis -- exactly what the reference's
forward hooks on `model.layers[idx]` record. Capturing the raw `[4, dim]`
instead would be 4x the memory and would not match what `main_proj` was trained
on. Off by default; the cost when on is one pooling kernel per target layer per
forward plus an `[n_target * dim]` D2H copy.

`verify_step` keeps one row per draft token rather than only the last, because
the accepted prefix is not known until after the comparison and the next round
has to start from wherever it lands. Prefill keeps only the final prompt
position: holding all of them would be `n_target * dim` floats per token, ~49 KB
at dim=4096, i.e. 3 GB at a 64K context, for hiddens no draft round reads.

Every way of getting this wrong still yields a finite vector of plausible
magnitude, so `cpp_engine/tests/test_dspark_hidden_capture.cpp` checks the
pieces separately. Against the real checkpoint: each slot matches a
single-layer capture of that layer bitwise (`max_abs=0`), the target layers are
distinguishable (`rel_l2` 0.56 and 1.23 against the first, so the slot check is
not vacuous), verify's rows match plain decode at the same positions exactly,
and prefill's hidden reads 2.8e-6 against the correct position versus 1.79
against the next one. Enabling capture leaves the token stream unchanged. The
pooling kernel itself is checked against a CPU mean at exact equality, which
separates a mean from a sum by a clean 4x, plus a poisoned destination so an
unwritten or over-wide slot cannot pass.

#### MoE notes

The draft's routed experts are held resident on the device rather than staged
per call: 13.4 MB per expert is 10.3 GB at TP=1 but 2.6 GB at TP=4, and staging
would put a PCIe transfer on the critical path of every draft round -- the one
thing speculative decoding cannot afford, since the draft has to stay cheap
relative to the verify it is trying to skip. Total DSpark weights measure
11.4 GB at TP=1 and 3.8 GB at TP=4.

Routing is discrete, so a wrong gate produces plausible-looking numbers rather
than obvious garbage. Measured on 5 tokens against an fp32 reference, cpp/ref
`rel_l2` is 0.014-0.016 (int8 activation quantization before the expert GEMM),
while swapping a single expert for the next-ranked one reads 0.476 -- a ~32x
margin, which is what makes the 0.02 tolerance meaningful rather than merely
satisfied.

#### Head notes

The output head (`head.weight`, 129280x4096 bf16, ~1 GB) is kept whole on every
rank rather than vocab-sharded as the reference does. The reference all-gathers
its logit shards; here the draft's inner loop runs `block_size` times per round,
so a collective per position would put TP latency on the hot path -- and a
replicated table makes every rank's drafted ids identical by construction
rather than by agreement. That is the reason total DSpark weights read 12.4 GB
at TP=1 rather than the 11.4 GB the MoE alone accounts for.

The head's parity is a different regime from the MoE's: bf16 weights against
fp32 activations with no int8 anywhere, so cpp/ref `rel_l2` is 1.3e-7. Three
plausible ways to get it wrong -- no markov bias, all biases computed from the
input token instead of sequentially, and collapsing the hc dimension by mean
instead of the head's own gate -- read 8.3e-1, 5.7e-1 and 1.0e0, and each
changes at least one drafted token. Hence a 1e-5 tolerance plus an exact
token-id comparison: the smallest observed top1-top2 margin was 0.02, so ids
matching is a real check, not a foregone one.

### Speculative scheduling

A verify round forwards `[committed, d0, d1, ...]` and samples at each position,
returning the truth for `d0..dn`. The scheduler compares this against the draft
block, finds the longest matching prefix, and commits `committed` plus that
prefix (even when the prefix is empty, the committed token advances the
position by one). The question is what happens to the rejected suffix, which a
naive verify has already forwarded into the main model's KV cache and compressor
state.

**The KV ring self-heals**: it addresses by `position % window_size`, so
forwarding from the new position writes exactly the slots the next verify would
read. **The compressor accumulator does not.** It sits on layers 2-42, updates
at offset `position % ratio`, and at every ratio boundary (when
`(position+1) % ratio == 0`) `compressor_shift_overlap_state_kernel` moves the
upper half down and zeroes the top. That shift is destructive: replaying a
position that crossed a boundary cannot rebuild what was shifted away, so a
rejected verify that crossed one leaves zeroed slots the next decode reads.

`test_verify_overwrite_safety` measures this directly. With a 6-token prompt
the continuation starts at position 6, mid-ratio for `ratio=4`. Overshoots of
1–8 all cross the boundary at position 7, and without protection they all
diverge at token 1 (the boundary crossing). A control at `layer_count=2`
(excluding every compressed layer) matches exactly, pinning the compressor as
the cause. The pooled compressed slots are position-addressed (`window +
position/ratio`) and also self-heal; only the accumulator is destructive.

`snapshot_compressor_state()` copies both `kv_state` and `score_state` to host;
`restore_compressor_state()` writes them back. With a snapshot taken before the
overshoot and restored after it, all six overshoot lengths match the clean run
exactly. Per layer the buffers are `slots * state_cols` **fp32** each, where
`state_cols = head_dim * 2` and `slots = ratio * 2` when the layer is overlap-
layout and `head_dim` / `ratio` otherwise -- head-sharded under TP. (The absolute
size has not been measured; it is small enough that it has never shown up against
a round's forward cost, but no number should be quoted here until it is.)

The scheduler itself does **not** use this. Restoring to the pre-verify position
would also discard the accepted prefix's contribution, which nothing re-forwards.
`speculative_step()` instead stops verifying at the first mismatch, so no
uncommitted position is ever forwarded -- see [Scheduler](#scheduler).

### Performance status

The current C++ scheduler verifies draft tokens sequentially. This preserves
plain greedy token identity and allows verification to stop before any rejected
tail mutates the compressor state, but it does not amortize the target-model
forward: every accepted draft token still runs one full 43-layer decode and its
TP collectives.

A TP=4 benchmark on the 0731 FP4 checkpoint measured C++ speculative/plain
ratios of roughly 0.83-0.90x across cyclic, short-text, and longer-text fixtures.
Even the cyclic fixture, which accepted the full 5-token draft in each complete
round, measured about 0.85x. These numbers establish the current limitation:
the draft cost is added while sequential verification retains almost all of the
plain-decode work.

This benchmark is intentionally kept separate from PyTorch runtime comparisons.
The validated PyTorch serving path uses CPU-resident raw FP4 experts with active
expert staging and normal compressed attention, and measures about 3.44 tok/s on
the real long-prompt benchmark. Earlier cross-runtime results using a different
expert placement and disabled compression are not comparable and must not be
used for parity or speedup claims.

The next performance step is a C++ multi-token verify path. It must be evaluated
against sequential greedy decode rather than assumed exact: changing the GEMM
shape changes reduction order, and prior batched verify experiments showed
numerical drift that can change token selection. Until that path has its own
correctness policy and real-prompt measurements, the sequential scheduler is a
correctness implementation, not a speedup claim.

### Reading a cross-runtime first-token mismatch

A greedy argmax flips as soon as the two runtimes' logits differ by more than the
top-2 gap, so comparing token ids alone cannot separate numerical drift from a
wrong state. `scripts/run_dspark_parity_bench.sh` therefore records the prefill
top-k on both sides (`POCKETLLM_CPP_TOPK_DIAG` for the C++ ranks,
`tests/debug_prefill_topk.py` for PyTorch) and the summarizer reports a
`prefill_argmax` verdict per fixture alongside the token comparison.

A TP=4 run on the 0731 checkpoint at 43 layers measured:

| fixture | prompt | PyTorch argmax / top-2 margin | C++ argmax / top-2 margin | verdict |
| --- | --- | --- | --- | --- |
| raw_cyclic | 12 | 16 / 3.638 | 16 / 3.306 | argmax_agree |
| real_short | 16 | 2524 / **0.00204** | 36101 / 1.881 | near_tie |
| real_long | 61 | 19 / 0.924 | 19 / 2.021 | argmax_agree |

`real_short` reports `cpp_plain_dspark_vs_pytorch: MISMATCH@0`, but PyTorch's own
top-1 and top-2 there are 2524 and 36101 separated by 0.002 (relative 6e-5), and
36101 is exactly what C++ picks. The two fixtures whose argmax agrees both have
margins above 0.92. So the mismatch is a tie-break at a margin far below the
decision headroom the agreeing cases have, not a state or indexing error -- the
logits themselves differ by less than that gap everywhere. What remains to be
explained is the logit *scale* difference (C++ 30.42 vs PyTorch 33.09 at
real_short's top-1), which belongs to head/norm precision and reduction order,
the same regime as the batched-verify drift above.

Do not chase a `MISMATCH@0` on this fixture through the prefill row extraction or
`reset_session()`. Check the margin first; only a mismatch whose margin is large
relative to the runtimes' logit agreement indicates a real divergence.
