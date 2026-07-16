# CPU tensor-parallelism — performance tracking

Collected perf data across the project so we can see "where we are" and (later) plot progression.

**Important caveat.** These are **not a controlled series** — each phase measured a *different model
on different hardware*, because the work was mostly *breadth* (new architectures/capabilities: FFN →
attn → MoE-EP → MoE-tensor → MLA → Mamba-2 → GDN → multi-node) rather than *depth* (one number getting
faster). So there is no single "the speedup" that climbs monotonically. A true iteration-over-iteration
curve needs the *same* (model, hardware, config) benchmarked at each commit — see the harness note below.

All numbers are `llama-bench` t/s unless marked `[simple]` (single-run `llama-simple`, noisier).
pp = prefill (`-p512`), tg = decode (`-n128`), `-mmp 0`. Speedups are vs the row's stated baseline.

## Capability / perf milestones

| date | phase (commit) | model | HW | config | pp t/s | tg t/s | vs baseline |
|------|----------------|-------|----|--------|-------:|-------:|-------------|
| 06-22 | Ph1 FFN-only (232c8d2f5) | Llama-70B Q8 | 2× Skylake IB | rank=node FFN | — | 1.24 | **1.44× dec** vs 1-node 0.86 |
| 06-22 | Ph2 attn shard (a52832bd7) | Llama-70B Q8 | 2× Skylake IB | +attn | 34.83 | — | **1.81× pp / 1.53× dec** |
| 06-22 | Ph3 4-way (289306b08) | Llama-70B Q8 | 2 nodes / 4 sockets | rank=socket | — | 3.44 | **4.0× dec** vs 1-socket |
| 06-23 | MoE EP (05ef89ec8) | Qwen3.6-35B-A3B Q4 | 2× Skylake IB | EP | — | ~5.5 | 0.87× vs 1-node 6.34 (capacity, not speed) |
| 06-24 | MoE tensor (408ee119a) | DeepSeek-V3.1 Q4 | 2× Skylake IB | tensor | 15.5 | 3.32 | **+36% pp / +27% dec** vs 1-sock 11.3/2.61 |
| 06-24 | MoE tensor N4 | DeepSeek-V3.1 Q4 | 4 sockets | tensor | 18.9 | — | **+66% pp** |
| 06-24 | MLA attn (f772875da) | DeepSeek-V3.1 Q4 | 1 node 2-sock sm | tensor+MLA | 19.16 | 3.68 | **+25% pp / +13% dec** over tensor-only |
| 06-26 | tensor-TP (408ee119a) | Qwen3-235B-A22B Q8 | tonto92 2-sock | tensor | 21.10 | 3.72 | **+36% dec** vs 1-sock 13.32/2.74 |
| 06-25 | spec decode (validated) | R1-70B Q6 + 8B draft | 1 socket | draft-simple | — | 2.39 | **1.56× dec** vs 1.53 baseline |
| 07-01 | GDN + repack (38f47f52c) | Qwen3.5-27B Q4 | tonto92 2-sock | attn+GDN | — | — | **1.80× pp / 1.71× dec** (repack fix) |
| 07-02 | GDN full (aad985ef0) | Qwen3.6-35B-A3B Q4 | Cascade 92 2-sock sm | attn+GDN+MoE | 131.27 | 12.01 | **+36% pp / +17% dec** vs 1-sock 96.74/10.27 |
| 07-02 | GDN multinode (232a3d8dc) | Qwen3.6-35B-A3B Q4 | 2× Skylake IB | attn+GDN+MoE | 246.11 | 7.89 | **+84% pp / −27% dec** vs 1-sock 133.55/10.80 |
| 07-02 | qwen3next (232a3d8dc) | Qwen3-Next-80B Q4 | 2× Skylake IB | attn+GDN+MoE | — | 4.54 | `[simple]` coherent, no clean bench yet |
| 07-03 | GLM-5.2 multinode (404be19a6) | GLM-5.2-REAP-504B Q4 (glm-dsa) | 2× Skylake IB | expert-TP tp | 24.36 | 1.57 | 2-way @ d0; vs Cascade 1-node 14.11/1.30 |
| 07-03 | GLM-5.2 4-way (404be19a6) | GLM-5.2-REAP-504B Q4 (glm-dsa) | 2 nodes / 4 sockets | expert-TP rank=socket | 16.92 | 3.09 | 4-way @ d0; **2× decode** vs 2-way, lower pp |
| 07-04 | MiniMax-M2.7 (404be19a6) | MiniMax-M2.7 Q8 (minimax-m2, 10B act) | Cascade 92 1-sock | CPU -t24 | 27.96 | 5.99 | @ d0; collapses at depth (see below) |

## The recurring lesson (holds across every phase)
- **Prefill scales** with sockets/nodes (compute-bound, batched GEMM): +36% to +84% typical, near-linear
  within-node, sublinear cross-node.
- **Decode is the hard part** (batch-1, latency/bandwidth-bound): within-node shared-memory TP *helps or is
  free* (+17% GDN on Cascade); multi-node IB TP *hurts* (−27%) because every layer pays IB all-reduce
  latency. `sm` reduce ≈ microseconds; `rc`/IB reduce ≈ latency+skew-bound. → multi-node is a
  prefill/throughput/capacity play; batch-1 decode wants one NUMA-local socket or speculative decode.

## Long-context: decode-vs-depth + GPU-attn offload (2026-07-03/04)

New axis of work: **long-context decode**, and offloading attention (+KV) to the tonto92 Intel B580
(12 GB) via Vulkan while experts stay on CPU (`-ngl 99 -ncmoe N`). tg = `-n32` at prompt depth `d`.

**qwen35moe 35B Q4 (GDN hybrid), tonto92 — GPU-attn FLATTENS the decode drop.** CPU-only baseline is
`build-native` (has AVX512-VNNI Q4_K repack; the Vulkan build's CPU path is ~½ speed, don't use it as
the CPU baseline). GPU-attn = `build-vk -ngl 99 -ncmoe 64 -t48`, KV on B580.
| tg @ depth | d0 | d8K | d16K | d32K |
|---|---:|---:|---:|---:|
| CPU-only 1-sock (repack) | 9.84 | 10.24 | 8.89 | 6.19 |
| GPU-attn + CPU-experts | 9.77 | 8.26 | 8.17 | 7.33 |
CPU decode drops −40% (d8K→d32K, the KV-scan hitting DDR4); GPU-attn drops only −11%. They cross ~20K,
GPU wins at 32K (**+18%**) and the gap widens with depth. Short/mid context: the strong VNNI CPU path wins.

**MiniMax-M2.7 Q8 (minimax-m2, ~10B active, FULL GQA attn), tonto92 CPU 1-sock -t24 — decode collapses
at depth.** Smart + light, but full attention (8 KV heads × 62 layers ≈ 250 KiB/token KV) → the KV scan
kills long-context decode on CPU (worse than GLM's DSA at 32K).
| depth | pp512 | tg32 |
|---|---:|---:|
| d0 | 27.96 | 5.99 |
| d8K | 16.00 | 2.87 |
| d32K | 6.60 | 0.56 |
(Q8 also caps decode; a Q4_K would ~2× it.) **GPU-attn on M2.7 = NET LOSS** (build-vk, KV on B580):
pp 2.71 / tg 2.31 @ d0, tg 1.54 @ d8K, **OOM at d32K**. ~10× slower prefill / ~2× slower decode than CPU.
Why: (1) it's **Q8_0** and Battlemage is ~4× slower on Q8 than Q4_K; (2) per-layer GPU<->CPU shuttle +
12 GB VRAM ceiling (device-memory alloc fail at 32K). The qwen35moe GPU-attn win needed **Q4_K + GDN**.

**GLM-5.2-REAP-504B (glm-dsa = MLA + DSA sparse), the 1M-context target — prefill-walled.** Cascade
1-node -t48: pp512 14.11 @ d0 → **2.0 @ d32K**; tg 1.30 → 0.81. Multi-node expert-TP over IB works and
scales (2-way pp 24.36 / tg 1.57; 4-way rank=socket pp 16.92 / tg 3.09), but a *fresh* 1M prefill is a
**multi-day** job (prefill is ~O(ctx²) dense compute; the 12 GB GPU can't help — 331 GB experts don't
fit; KV at 1M ≈ 116 GB). Verdict: 1M on a 504B model here is impractical without GPU-accelerated prefill
or a linear-attention model.

**Cross-model lesson for deep context (CPU + 1× 12 GB GPU):** every full-attention / MLA frontier model
collapses at depth — GLM-5.2 via *prefill*, MiniMax-M2.7 via *decode KV-scan*. Only linear/GDN attention
(qwen3next) stays flat. GPU-attn rescues full-attention decode only up to the depth whose KV fits 12 GB
(~48K for M2.7; GLM's KV never fits past ~100K).

## Decode cycle-accounting -- the batch-1 decode is BARRIER-bound, not comms-bound (2026-07-15)

Fable (xhigh) breakthrough analysis (hpc/FABLE_TP_BREAKTHROUGH.md) claimed the intra-node all-reduce
was a 170us wall. Instrumented it (llama-tp-net.c: LLAMA_TP_TRACE per-call CSV, size-bucketed
LLAMA_TP_TIME) and MEASURED on tonto92 (Cascade 8260, 2-sock), qwen35moe 35B-A3B Q4, -p0 -n128 -t24:
- **Intra-node all-reduce is only ~38us, NOT 170us**, and comms is only **~5% of decode** (NOCOMMS
  17.53 vs comms 16.68 t/s). Same on OpenMP (39.5us) and pthreads (37.4us) -> the 170us was an older
  build / the inter-node rc path, not intra-node. Cross-rank join closes: |D|/2+x+s=39.78us. The AR
  window is ~63% random partner-skew, ~28% UCX protocol, ~9% sum. So idea #1 (fused all-reduce) has a
  ~5% ceiling intra-node = NOT worth building. Its real prize is inter-node rc (deferred).
- **perf: the 3x gap to the DRAM roof (~17.5 vs ~52 t/s) is BARRIER-bound.** perf stat: only 15.5/24
  CPUs utilized, IPC 0.4. Top self-time: gomp_team_barrier_wait ~40% (!), vec_dot_q8_0_q8_0 39%
  (generic non-VNNI path), VNNI-repacked gemv_q4_K_8x8 only ~8%.
- **FREE WIN: -t 16/socket beats -t 24 (barrier oversubscription on tiny decode ops).** qwen35moe-UD:
  1-sock tg 11.64@t24 -> 12.70@t16 (+9%); 2-sock TP 15.42@t24 -> **17.56@t16 (+14%)** (t16+comms even
  beats t24 NOCOMMS). BANKED, zero code -- use -t 16 (=2/3 of a 24c socket) for decode.
- Quant finding: 39% of decode self-time is generic Q8_0 vec_dot (the "UD" Unsloth-dynamic model keeps
  tensors at Q8_0, missing the VNNI repack). Uniform non-UD Q4_K_M: 2-sock TP 16.62@t24 / 16.53@t16 --
  faster at t24 (+8%, fast path) but does NOT benefit from -t16 and does NOT beat UD@t16 (17.56).
  So the quant swap is not a net win over UD+t16. *** perf CONFIRMS the mechanism + clinches the
  diagnosis: non-UD moved vec_dot_q8_0 39%->11% and raised VNNI gemv_q4_K 8%->20% (fast path DID
  engage) BUT gomp_team_barrier_wait rose 40%->46% and a new vec_dot_q6_K appeared at 18%; net decode
  unchanged. => Making kernels FASTER just increases the idle-at-barrier fraction. Decode is
  fundamentally batch-1 SYNCHRONIZATION-bound (hundreds of tiny ops each barriering 24 threads at
  different finish times), NOT compute-bound. Faster kernels can't close the 3x roof gap; only fewer
  threads (-t16, banked) or fewer/bigger ops help. *** This is the strongest possible motivation for
  speculative decode (idea #2): batching K draft tokens per forward turns K barrier-bound batch-1
  passes into ONE pass with K-bigger ops (GEMV->GEMM, barriers amortized over K accepted tokens).
- Fable verdict: op-fusion in ggml-core = NO (the 40% barrier-wait was t24 oversubscription; -t16
  already reclaimed it; residual is genuine load-imbalance, not fusable; invasive core change for a
  single-digit ceiling). Next big lever = idea #2 speculative decode (~2x, multiplies with +14%).

## Speculative decode (idea #2) -- Phase 0 measured, +32% single-socket (2026-07-15)

Fable design (grounded in the tree): spec decode for GDN hybrids is ALREADY in upstream; {qwen35,
qwen35moe} are the arch whitelist for recurrent-state rollback. common/speculative.cpp has draft-mtp
(qwen35moe's trained nextn/MTP head) + ngram self-draft + EAGLE3. The official
Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf HAS the nextn tensors; --spec-type draft-mtp drafts against the target
model itself (no 2nd load). Tool = llama-cli (needs -DLLAMA_BUILD_SERVER=ON + openssl; links
server-context). *** GOTCHA: llama-cli is conversational and IGNORES -n without -st/--single-turn ->
without it, it runs away unbounded (a run generated a 136GB log / 6h before I killed it). ALWAYS use
-st + a timeout backstop. ***
ROI model (T(K) curve, llama-bench -p 1..16 -n0): rho(K)=T(K)/T(1) = cost of a K-token verify pass:
rho(2)=1.31 rho(3)=1.62 rho(4)=1.81 rho(6)=2.29 rho(8)=2.71 (t16, 1-sock). So a 4-token verify is only
~1.8x a single token -> spec speedup = E[accepted]/rho(K).
RESULT (1-sock t16 non-TP, official Q4_K_M, temp 0, code+reasoning prompt, -n160):
  ctrl 15.2 t/s | draft-mtp n-max4 **20.1 t/s (+32%)** | ngram-mod 15.3 (no gain on reasoning, expected).
+32% => E[accepted] ~2.4/pass => alpha ~0.62 (matches Fable's ~0.65). COMPOSES with the banked +14%
thread taper. K SWEEP (single-sock t16): k2 18.7 / k4 19.4 / **k6 20.2 (PEAK, +33%)** / k8 15.4 (cliff)
-- interior optimum at K=6, K=8 regresses (single MTP head, alpha decays with draft depth), exactly as
Fable predicted. Correctness VERIFIED by Fable (read sample_and_accept_n: emits target argmax, accept
iff draft matches -> exactness-preserving vs the batched forward; divergence from single-token ctrl =
benign reduction-order). Gate corrected: use batched-control-identity, never single-token.
*** PHASE 2 (MTP spec under 2-socket TP) -- WORKS (2026-07-15). *** The GGML_ASSERT at ggml.c:3903
ggml_set_rows was NOT the recurrent snapshot (hypothesis wrong); Fable located it: the KV write at
llama-kv-cache.cpp:1323 (cpy_k). Root cause = hparams-coverage gap: the loader shards the MTP/nextn
block's attn_k/attn_v by tensor name (no layer gate) but the head-count DIVISION loop in src/llama.cpp
ran only il<n_layer(), skipping the nextn block at index n_layer() -> MTP KV cache allocated full-width
vs sharded half-width k_cur. FIX (1 line, src/llama.cpp): change the attn head-count division loop bound
from hp.n_layer() to hp.n_layer_all so the nextn block's n_head_arr/n_head_kv_arr also /= tp.size. Plus
the graph_mtp all-reduce (qwen35moe.cpp + qwen35.cpp). RESULT (2-sock TP t16, official Q4_K_M, K=6):
TP-control 20.4 t/s -> **TP+MTP-spec 23.3 t/s (+14%)**, lockstep held (both ranks transport up, no
deadlock), output COHERENT. *** ALL THREE LEVERS STACK: from 2-sock TP t24 no-spec ~15.4 -> t16+MTP-spec
23.3 = ~+51% decode, measurement-driven + validated. *** Spec speedup is smaller under TP (+14%) than
single-sock (+33%) because the TP baseline is already faster per-token and the per-pass AR amortizes.
Remaining: TP+cli aborts on TEARDOWN (server_context+TP cleanup crash, non-fatal to the measurement);
correctness gate = TP+MTP output should be batched-control-identical to single-sock+MTP (deeper check,
not yet run; output is coherent). qwen3next (not in rs-rollback whitelist) + EAGLE3 later. CAVEAT: mtp output NOT bit-identical to single-token ctrl -- diverges at an early near-tie
then cascades (both coherent) = likely benign batched-verify reduction-order (same near-tie flips as the
TP work); correctness verification (re-feed output, check greedy argmax) pending with Fable. NEXT: Phase
2 = the ONE code change (2-line all-reduce after manual wo in graph_mtp qwen35moe.cpp:683 + qwen35.cpp
twin, matching the 3 main GDN graphs we already patched) -> MTP under 2-socket TP; then K sweep + multinode.

## MTP speculative decode WIRED for glm-dsa / GLM-5.2 (2026-07-16) -- frontier-model decode win

GLM-5.2's nextn/MTP tensors load in the GGUF but glm-dsa never computed them (loader TENSOR_SKIP'd
blk.n_layer; no graph_mtp). Implemented it (commit: graph_mtp = eh_proj seed + deepseek2 MLA attn + MoE
FFN + shared head; loader loads blk.78 when present; deepseek2 graph tail sets t_h_nextn gated;
KV-filter for the MTP layer). Pre-gate: the REAP nextn weights are BIT-IDENTICAL to the full UD-Q6's
(REAP didn't touch the MTP head). RESULT (GLM-5.2-REAP-504B, single-sock -t24, --spec-type draft-mtp):
  ctrl 2.3 | **K=2 3.2 (+39%, PEAK)** | K=4 3.1 (+35%) | K=6 2.4 (+4%).
tg tracks K -> CONFIRMED real spec (not a load confound). GLM's MTP alpha decays FASTER than qwen35moe's
(peak K=2 vs qwen's K=6) -- single MTP head + REAP-pruned MTP-block experts. graph_mtp graph-reserve is
CLEAN on the real 504B (MLA splice correct). *** So both frontier-MoE decode levers now proven on
GLM-5.2: MTP spec +39% (this) and (pending) Unsloth IQ2_M 2-bit ~2x bandwidth -- they STACK (fewer passes
x fewer bytes/pass) -> plausibly ~2.5-2.7x decode at full-model quality. This is the real answer to
"speed up frontier SOTA models" -- not comms/1M. *** GPU-for-MTP idea (user): low upside, draft already
~1-2% of a pass; aim a real GPU (5090) at the DSA indexer offload / KV instead.

## Machine-readable (for plotting)
```csv
date,commit,model,hw,config,pp_ts,tg_ts,pp_base,tg_base,pp_speedup,tg_speedup
2026-06-22,232c8d2f5,llama70b-q8,2xskylake-ib,ffn-2node,,1.24,,0.86,,1.44
2026-06-22,a52832bd7,llama70b-q8,2xskylake-ib,attn-2node,34.83,,19.24,,1.81,1.53
2026-06-24,408ee119a,deepseek-v31-q4,2xskylake-ib,moe-tensor-2node,15.5,3.32,11.3,2.61,1.36,1.27
2026-06-24,f772875da,deepseek-v31-q4,1node-2sock,tensor+mla,19.16,3.68,15.31,3.25,1.25,1.13
2026-06-26,408ee119a,qwen3-235b-q8,tonto92-2sock,tensor,21.10,3.72,13.32,2.74,1.58,1.36
2026-07-02,aad985ef0,qwen36-35b-a3b-q4,cascade92-2sock-sm,attn+gdn+moe,131.27,12.01,96.74,10.27,1.36,1.17
2026-07-02,232a3d8dc,qwen36-35b-a3b-q4,2xskylake-ib,attn+gdn+moe,246.11,7.89,133.55,10.80,1.84,0.73
2026-07-03,404be19a6,glm52-reap-504b-q4,cascade92-1node,glm-dsa-cpu-d0,14.11,1.30,,,,
2026-07-03,404be19a6,glm52-reap-504b-q4,cascade92-1node,glm-dsa-cpu-d32k,2.00,0.81,,,,
2026-07-03,404be19a6,glm52-reap-504b-q4,2xskylake-ib,expert-tp-2way-d0,24.36,1.57,14.11,1.30,1.73,1.21
2026-07-03,404be19a6,glm52-reap-504b-q4,2nodes-4sock,expert-tp-4way-rank-socket-d0,16.92,3.09,24.36,1.57,0.69,1.97
2026-07-04,404be19a6,minimax-m2.7-q8,cascade92-1sock,cpu-d0,27.96,5.99,,,,
2026-07-04,404be19a6,minimax-m2.7-q8,cascade92-1sock,cpu-d32k,6.60,0.56,,,,
```

## Depth-sweep CSV (long-context decode; tg = -n32 @ depth)
```csv
date,model,hw,config,depth,pp_ts,tg_ts
2026-07-03,qwen35moe-35b-q4,tonto92-1sock,cpu-repack,0,,9.84
2026-07-03,qwen35moe-35b-q4,tonto92-1sock,cpu-repack,8192,,10.24
2026-07-03,qwen35moe-35b-q4,tonto92-1sock,cpu-repack,16384,,8.89
2026-07-03,qwen35moe-35b-q4,tonto92-1sock,cpu-repack,32768,,6.19
2026-07-03,qwen35moe-35b-q4,tonto92-b580,gpu-attn-cpu-exp,0,,9.77
2026-07-03,qwen35moe-35b-q4,tonto92-b580,gpu-attn-cpu-exp,8192,,8.26
2026-07-03,qwen35moe-35b-q4,tonto92-b580,gpu-attn-cpu-exp,16384,,8.17
2026-07-03,qwen35moe-35b-q4,tonto92-b580,gpu-attn-cpu-exp,32768,,7.33
2026-07-04,minimax-m2.7-q8,tonto92-1sock,cpu,0,27.96,5.99
2026-07-04,minimax-m2.7-q8,tonto92-1sock,cpu,8192,16.00,2.87
2026-07-04,minimax-m2.7-q8,tonto92-1sock,cpu,32768,6.60,0.56
2026-07-04,minimax-m2.7-q8,tonto92-b580,gpu-attn-cpu-exp,0,2.71,2.31
2026-07-04,minimax-m2.7-q8,tonto92-b580,gpu-attn-cpu-exp,8192,2.79,1.54
2026-07-04,minimax-m2.7-q8,tonto92-b580,gpu-attn-cpu-exp,32768,,
```

## To make the actual "speedup vs commit" graph (controlled series)
1. Pick ONE fixed benchmark: e.g. `qwen35moe 35B Q4_K_M, tonto92 2-socket sm, -p512 -n128 -r3`.
2. `hpc/perf_series.sh` (to write): for each key commit — `git archive <sha> | build on node | run the
   fixed bench | append pp/tg to a TSV`. Most commits won't move this number (they added *other* archs),
   so expect a mostly-flat line with jumps only at commits that touch that path (repack fix, GDN fix).
3. Plot: `pp_speedup`/`tg_speedup` vs commit index (matplotlib step plot / waterfall). Better suited to a
   *sub-thread* where the same metric improves (e.g. MoE decode on one model: 1-sock → EP → tensor → …).
