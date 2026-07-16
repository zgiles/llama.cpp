# Fable (max-creativity) — speeding up frontier MoE (GLM-5.2 class) on our CPU+small-GPU cluster (2026-07-15)

Deliverable from a Fable subagent asked for hard-but-high-payoff ways to make big-active frontier MoE
(GLM-5.2-REAP 504B / glm-dsa, DeepSeek, Kimi) decode/prefill fast on our bandwidth-poor, capacity-rich,
GPU-poor box. It READ THE TREE and corrected two of our prior conclusions. Decode ceiling = socket_BW /
active_bytes_per_token; today B~=17GB (30B active x Q4), ceiling ~6 t/s/socket, measured 2.2.

## TWO CODE-GROUNDED CORRECTIONS (change our priors)
1. The DSA long-context "collapse" is a REAL, FIXABLE code inefficiency -- NOT O(ctx^2), NOT random
   gathers. src/llama-graph.cpp:2573-2603: the sparse main attention builds a full [n_kv,n_batch] KQ
   mask (-INF except top-2048) then runs build_attn_mha over the ENTIRE K/V cache (dense masked GEMM);
   flash disabled at :2780. And the indexer score GEMM (deepseek32.cpp:310) is f16 with NO int8/VNNI.
   So the wall is an O(ctx) masked-GEMM + O(ctx) un-VNNI'd indexer, both fixable.
2. *** GLM-5.2 DOES have an MTP head in the GGUF *** -- our memory's "no MTP head" is WRONG for this
   checkpoint. nextn tensors load (glm-dsa.cpp:135-145, deepseek32.cpp:141-151) but are never computed
   in the main graph (output-only hook at llama-graph.cpp:953). common/speculative.cpp already has
   COMMON_SPECULATIVE_TYPE_DRAFT_MTP (PR #22673). So spec decode is a LIVE lever for GLM/DeepSeek.

## TOP 3 (order-of-magnitude-ish, ranked payoff x feasibility)
**#1 Sub-2-bit expert re-quant (Unsloth-Dynamic-style) + TEAL activation sparsity.** Attack B from both
sides. Re-quant ONLY routed experts to ~1.6-2.0 bit (IQ2_XS/TQ), keep down_proj + first ~3 dense layers
+ DSA indexer + MLA-latent at >=Q6. B 17GB -> ~9GB (2bit) or ~7.7GB (1.6bit) -> ceiling ~11-13 t/s vs 6.
Stack TEAL (training-free per-layer activation thresholds -> skip ~40% of expert FFN rows) -> B ~6.4GB
-> combined ceiling ~2.5-3x, measured 2.2 -> ~5-6 t/s. Bonus: 332GB -> ~150-180GB (load time halves).
Model change = re-quantize only (llama-quantize + imatrix, NO retrain). Risk: REAP is already
expert-pruned = less redundancy; validate KL-div vs Q4 on held-out set. FIRST STEP: imatrix from ~512
in-domain tokens -> llama-quantize experts IQ2/TQ w/ protected tensors -> measure KL vs Q4. **Biggest
single win, days not months, untried.**
**#2 Make DSA truly O(k): gather top-2048 K/V + int8/VNNI indexer.** Replace the dense-masked GEMM over
all n_kv with a ggml_get_rows gather of the selected 2048 rows (O(ctx)->O(k): ~64x fewer key-dots at
128K, ~500x at 1M) + an int8 indexer GEMM (~4x on VNNI). Turns 1M prefill days->hours. THE deep-context
lever. Pure systems (inference-graph rewrite, no weights). Hard (ggml surgery; per-query different 2048
keys = batched get_rows + block-diag attn; for prefill use block-shared selection to stay GEMM-shaped --
exactly the "blocked-int8 + block-sparse" we intuited). FIRST STEP: prototype the gather at DECODE
(single query), A/B token-identical vs the dense-mask path at d=128K, measure attn speedup vs depth.
**#3 Light up the MTP head already in the GLM-5.2 GGUF (self-speculation, no draft model).** nextn weights
exist, just not run. ~1 extra layer (~0.4B) draft, trained MTP head alpha~0.8 -> +30-60% decode for
near-zero marginal bytes; stacks on #1. Pure systems (port PR #22673's MTP path to the DSA archs; advance
the DSA indexer/KV for the drafted position). Risk: REAP pruning may have degraded nextn -- check norms +
empirical alpha. FIRST STEP: dump nextn norms (non-degenerate?), port draft-mtp to glm-dsa, measure alpha
on reasoning output; alpha>0.7 = clean win.

## Tier 2 (high, harder/narrower)
#4 = same as #2 (the O(k) fix, listed as the deep-context unlock). #5 KV persistence/reuse/delta: prefill
big shared prefixes ONCE (system prompt/codebase/corpus), save MLA+indexer KV, mmap back per query; 1.5TB
RAM holds many 100K+ caches; per-query prefill days->minutes for shared-prefix (RAG/agent/codebase)
workloads. Pure systems (extend prompt-cache save/restore to llama-kv-cache-dsa; quantize saved KV q8/q4).
#6 Ring/context-parallel prefill across 4 nodes: shard the SEQUENCE (bandwidth-bound comms, IB-matched,
unlike latency-bound decode AR) -> ~4-6x prefill; needs a 4th homogeneous node + global top-k reduction
across ring hops. #7 GPU (B580) as indexer/MLA-latent accelerator: indexer key cache at 256K ~5.2GB FITS
12GB; int8 XMX ~4x CPU on that O(ctx) kernel; experts stay on CPU. Prefill play (PCIe latency kills it at
batch-1 decode). This is the ONLY productive role for the 12GB card on this model.

## Tier 3 (lower / build-on)
Low-rank/SVD expert compression + cross-expert weight sharing (2-3x beyond quant, needs offline
factorization + validation; fast-follow to #1). Jacobi/lookahead decoding (model-agnostic, no head, but
~+10-20% on bandwidth-bound big-active -- fallback to #3). KV quant to q4 (already supported, do w/ #5).
Distill a 30-50B student (highest ceiling, but a real multi-GPU training project). Non-autoregressive/
diffusion/PIM/analog = not real on this box.

*** CAVEAT throughout: target is GLM-5.2-REAP (already expert-pruned) -> every "spare redundancy"
assumption (sub-2bit, activation sparsity, MTP quality) has LESS margin than the un-pruned models these
were validated on. Validate KL/acceptance empirically before committing. ***

Key files: DSA dense-masked attn llama-graph.cpp:2573-2603 (flash off :2780); indexer f16 GEMM
deepseek32.cpp:310,343; nextn loaded-not-computed glm-dsa.cpp:135-145 + deepseek32.cpp:141-151, hook
llama-graph.cpp:953; MTP spec common/speculative.cpp (DRAFT_MTP); DSA KV caches llama-kv-cache-dsa.cpp:32
(quantizable via type_k/type_v). Refs: Unsloth Dynamic 2.0, TEAL (arXiv 2408.14690), KTransformers SOSP25,
llama.cpp PR #22673, DeepSeek-V3.2 DSA (vLLM blog).
