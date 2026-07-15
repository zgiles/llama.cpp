# Fable (xhigh) analysis — breaking the CPU-TP LLM inference perf walls (2026-07-14)

Full deliverable from a Fable subagent asked to find breakthrough parallelism ideas for our CPU
tensor-parallel llama.cpp work. Briefed with our hardware, everything we built, and the measured walls
(batch-1 all-reduce latency floor, O(ctx^2) prefill, KV/VRAM limits, MoE sparsity). Analysis only, no code.

---

Below is the deliverable. I did no coding — this is analysis against your measured numbers. I first
re-derive where the true physical floors are, because two of your walls are provably *not* physics walls
(your own measurements show it), and that changes the ranking.

---

# Where the physics floors actually are

**Decode floor (per token):** `bytes_active / aggregate_local_BW + n_sync x sync_cost + graph_overhead`.
- For a 504B-A~30B at Q4 (~17 GB active reads): 2 sockets = 85 ms -> 11.7 t/s ceiling. You measure 3.68. **Gap ~3x.**
- For 35B-A3B at Q4 (~1.9 GB active): 1 socket = 19 ms -> ~52 t/s ceiling. You measure 10.3. **Gap ~5x.**
- Your all-reduce costs 170 us while the wire is 2-12 us. **Gap ~15-80x.**

So decode is NOT sitting on the DRAM-bandwidth wall yet. It is sitting on a *synchronization/overhead*
wall (AR skew + ggml per-op barriers + small-GEMV inefficiency), and that wall is implementation, not
physics. There is roughly 3-5x on the table for decode before you touch DRAM limits, and after that,
speculative decode moves the DRAM limit itself.

**Prefill floor (1M, full attention):** attention FLOPs ~= `4 * N^2 * d_attn * n_layer`. For a 504B-class
model at N=1e6 this is ~1e18 FLOPs. Your 8 sockets deliver maybe 16 TFLOPS sustained -> 17 hours *at
perfect efficiency*, days in practice. This one IS physics — for **dense** attention. The only exits are
(a) don't do N^2 (sparsity), (b) don't do it on CPU alone (GPU int8 for the quadratic part), (c) don't do
it twice (KV persistence), (d) don't do it at all (linear-attention hybrids). All four compose.

One honest note before the list: your DSA finding (#4, "still blows up") is very likely an implementation
artifact, not physics. A DSA indexer at prefill is `N^2/2 * d_idx` int8 MACs ~= 3-8e15 ops at 1M — that's
~10 minutes of VNNI GEMM across 8 sockets *if executed as blocked causal int8 GEMM*, versus days if
executed as per-token scans with per-token gathers. The 14 -> 2 t/s collapse by 32K smells like per-query
top-k gather (random 1-KB reads) killing bandwidth, not the indexer math. Verify this with a cycle
breakdown before believing the DSA wall.

---

# Ranked ideas

## 1. Kill the all-reduce skew: fuse the reduction into the producing matmul's epilogue, chunked, over spin-polled one-sided transport

**Idea.** Stop treating the AR as a separate phase after an op barrier. The AR input is the output of a
row-parallel GEMV whose output slices are produced by threads at different times. Today you pay: (op
barrier: wait for slowest of 24-48 threads) + (wake) + (AR: another rendezvous) + (wake). Instead: as each
thread finishes its output chunk, it immediately stores it into a shared-memory (intra-node) or
RDMA-written (inter-node) staging buffer and bumps a per-chunk counter; a designated spinning consumer (or
the peer's threads) reduces chunks as they arrive. The AR completes ~when the last GEMV chunk completes —
the sync cost collapses from additive to overlapped. Inter-node, use one-sided RDMA WRITE with polled
flags (no rendezvous, no progress-thread wakeups): 8-16 KB writes are 2-4 us on EDR, consistent with your
own 2-12 us wire measurement.

**Math.** ~100 ARs/token (60ish layers x ~2). Today: 100 x 170 us = 17 ms/token intra-node — that's
25-45% of the whole token budget for your big models. Physics floor per AR: 16 KB copy + reduce + poll
~= 5-10 us intra-node sm, ~15-30 us for a 2-step 4-rank butterfly over IB with one-sided writes. Chunked
epilogue fusion hides most of even that behind the GEMV tail. Target: 170 us -> 30-50 us effective.

- Intra-node 2-rank: 17 ms -> ~4 ms/token: +15-25% decode on big models.
- **The real prize: multi-node TP decode flips from a loss to a win.** 4 ranks over 2 nodes on the 504B:
  compute 42 ms + 100 x 50 us = 47 ms -> ~2x your current 2-rank decode instead of your measured -27%.
  This is the only path on your hardware to aggregate >200 GB/s of DRAM behind one decode stream.

**Speedup.** 1.2x intra-node; 1.8-2.5x by enabling 4-8 socket TP decode on big models.

**Novelty/difficulty.** The concept (comm fused into GEMM epilogue) is known on GPUs (Flux, async-TP,
NVLS); doing it in ggml's CPU threadpool with UCX/RDMA is a genuinely uncommon build. Difficulty: high —
you must break the ggml op barrier abstraction for exactly one op pattern (row-parallel matmul + AR) and
hand-roll chunk-grained signaling. But it's *your* infrastructure end to end; nothing upstream blocks you.

**Failure modes.** (a) Skew might be dominated by OS jitter/last-level-cache contention rather than
intra-op timing spread — then chunking helps less; mitigate with core isolation (`nohz_full`, no SMT
sharing, dedicated comm core spinning). (b) Non-matmul AR producers (GDN mixer outputs) need the same
treatment or become the new floor. (c) Getting the memory ordering right for polled flags (need
release/acquire + WRITE-with-immediate or fenced last-byte flag) is fiddly. Measure first: instrument the
170 us into (wait-for-last-thread / wake / copy / wait-for-peer) — if wait-for-last-thread is >70%, this
idea is the right one; if wake/futex dominates, pure spin-polling alone gets most of it cheaply.

## 2. Speculative decoding done properly: MTP self-draft + prompt-lookup drafting (this is the correct form of "speculative execution under an assumed result")

**Idea.** You asked whether a rank can speculate past the AR under an assumed reduction result. At the
*activation* level: no — there is no cheap verifier (checking the assumption requires the very computation
you skipped, and everything downstream is nonlinear). But at the *token* level there IS a cheap verifier:
the model itself, run batched. That is exactly speculative decoding — it is the sound version of the
speculation you're reaching for, and on your hardware it is unusually profitable because it amortizes BOTH
of your decode walls at once: one batched forward for K draft tokens = one set of weight reads AND one set
of ARs for E[accepted] tokens.

**Math.** CPU headroom check: your sockets have ~20 FLOP/byte of compute headroom; Q4 GEMV needs ~4
FLOP/byte per token, so verifying K=4-6 tokens per forward stays bandwidth-bound — verification is nearly
*free*. With an MTP head (GLM/DeepSeek-family ship them) acceptance ~60-80%/step: E[accepted] ~ 2-2.7 per
forward at K=4. Per-token cost of weights, ARs, and graph overhead all divide by E. Separately, at 1M
context, **prompt-lookup (n-gram) drafting is close to free and has high acceptance** on grounded tasks
(long-doc QA, summarization, code editing — output copies input spans); a hash index over 1M tokens is
trivial in your RAM. Tree/multi-candidate speculation raises E further at negligible bandwidth cost.

**Speedup.** 1.8-2.5x decode, multiplicative with idea 1 (they compose: fewer syncs per token x cheaper
syncs). Composite for the 504B: 3.68 -> ~12-18 t/s with ideas 1+2. That is an interactive frontier model
on CPUs.

**Novelty/difficulty.** Fully known (EAGLE/MTP/PLD). Medium effort: llama.cpp has spec-decode scaffolding;
your work is making it coexist with TP (drafts and verify batch must traverse the same sharded graph —
mostly free since AR size scales with batch and stays tiny) and wiring MTP heads for your models. Highest
certainty item on this list.

**Failure modes.** Models without MTP heads need a draft source (prompt-lookup covers long-context use; a
small draft model burns one socket's bandwidth). Acceptance collapses on free-form creative generation
(worst case: you keep baseline speed minus ~5% draft overhead). Verify-batch GEMM efficiency on CPU for
K<8 must actually reach bandwidth-bound — check llama.cpp's Q4 GEMM path at n_tokens=4.

## 3. Context parallelism by KV sharding + log-sum-exp merge: move queries, not KV (deep-context decode AND chunked prefill)

**Idea.** Shard the KV cache (and DSA index) across ranks by *sequence position*, not by head. For each
query block (a 512-token prefill chunk, or 1 decode token): broadcast Q (tiny), each rank runs flash
attention of Q against its local KV shard producing (partial_out, lse), then one AR-sized exchange merges
partials via the standard flash-decoding log-sum-exp rescale. KV never moves; queries (KB) move. This is
the inference-friendly alternative to ring attention — no bulk KV traffic on the wire at all, so it maps
onto your existing AR machinery (the merge is one more small reduce per attn layer, fusable with the
O-projection AR you already do).

**Math.** Today your MLA design *replicates* the compressed KV — at 1M that is fatal at decode: GLM-5.2's
~108 KB/token -> ~108 GB total; every socket scans a full ~70-108 GB per token -> 0.7-1.1 s/token ->
<=1.4 t/s even if everything else were free. CP over 8 sockets: ~9-14 GB/socket -> ~90-140 ms of
*parallel* scan — 8x on the attention term, and with DSA sparsity on top the scan term nearly vanishes
(see idea 4/5). Cost: ~1 extra AR-equivalent per attention layer (~50 us post-idea-1) — worth it precisely
when `KV_bytes_per_rank / 100 GB/s > AR_cost`, i.e., beyond a few K tokens of context; below that, switch
it off (make CP depth-gated). For prefill it parallelizes the O(N^2) term nearly linearly across all 8
sockets with comms of only ~`C*d` per chunk per layer (a few MB — thousands of times smaller than ring
attention's KV circulation), and it removes the "KV must fit one rank" constraint.

**Speedup.** Deep-context decode on full-attn/MLA models: from collapsed (your tg 6 -> 0.56 at 32K) to
roughly depth-flat until ~100K+, then degrading 8x more slowly. Prefill: ~1.9x/2 nodes, ~3.5x/4 nodes on
the attention-dominated regime (better than your +36-84% TP scaling exactly where it matters most, at depth).

**Novelty/difficulty.** Known technique (flash-decoding / inference context-parallel in vLLM/TRT-LLM);
novel in llama.cpp and in combination with your rank=socket NUMA scheme (KV shards become NUMA-local for
free — nobody else gets that on CPU). Difficulty: medium-high — per-rank KV cache indexing, the LSE-merge
op, and depth-gating. Note the two-pass max issue in the merge: use pre-scaled
`(out*exp(lse - m_local), sum_exp)` partials with a final rescale so it stays one exchange.

**Failure modes.** At batch-1 shallow context it's pure loss (extra sync per layer) — must be gated. Causal
imbalance at prefill (early shards see fewer queries): use zigzag/striped position assignment. Interacts
with your MLA head-sharding — you get a 2D shard (heads x positions); keep heads intra-node, positions
inter-node so the LSE merge crosses IB only once per layer.

## 4. Break the prefill quadratic: block-sparse attention with block-level selection (and re-do DSA as blocked GEMM)

**Idea.** Two halves. (a) For full-attention models: dynamic block-sparse prefill — MInference-style
per-head patterns (sink + local slash + top vertical columns) or Quest/NSA-style block-topk: select key
*blocks* (64-128 tokens) per query *tile* (128 queries) using cheap block summaries (mean/max pooled keys),
then run dense flash attention only on selected tiles. Published results show ~10x FLOP reduction at 1M
with near-lossless quality. Crucially, block granularity preserves GEMM structure and *shared* reads — the
selected KV block is read once for 128 queries, so bandwidth drops with FLOPs. (b) For GLM-5.2/DSA:
restructure indexer prefill as blocked causal int8 GEMM (VNNI's home turf) and take the *union* of top-k
over a query tile so the sparse attention gather is block-shaped, not 2048 random 1-KB reads per token.

**Math.** (a) 1e18 FLOPs -> ~1e17 at 10x sparsity; combined with idea 3's ~3.5x node scaling: 17 h ideal
-> ~1-2 h realistic for a 504B at 1M. (b) DSA math check: indexer prefill as GEMM ~= `N^2/2 * d_idx *
n_layer` int8 ops ~= 4-8e15 -> ~10-20 min on 8 VNNI sockets. Naive per-token sparse gather at prefill
would cost ~`N * k * bytes_tok * n_layer` ~= 100+ TB of random reads (days); tile-union gathering cuts this
by the tile size (~100x). This is why I believe your measured DSA collapse is fixable.

**Speedup.** The difference between "multiple days" and "a working day, or ~1-2 hours with idea 3" for
fresh 1M prefill. Nothing else on the prefill front matters until this exists.

**Novelty/difficulty.** Known family (MInference, Quest, NSA, DSA itself); the CPU/VNNI blocked realization
is engineering, not research. Difficulty: high (a real sparse flash-attention CPU kernel + selection
heuristics), but it's the only fundamental exit from N^2 for models you can't swap for hybrids.

**Failure modes.** Quality: block-sparse prefill can miss dispersed dependencies; validate on
needle/multi-hop tasks per model before trusting. Pattern-search overhead must be amortized (do it per
tile, not per query). For DSA specifically you're bound by whatever the model was trained to tolerate —
top-k=2048 is fixed; your freedom is only in execution order.

## 5. Repurpose the B580 as a retrieval/index engine, not a compute engine

**Idea.** Stop trying to make the GPU run layers (your data says it loses except Q4+small-KV). Give it the
two jobs where its 456 GB/s and int8 XMX beat a socket by 4-5x and the working set FITS in 12 GB: (a) **DSA
indexer residency at decode** — the index keys (~d_idx=128 int8/token/layer) for 1M tokens are ~5-10 GB:
hold them in VRAM, GPU scores q_idx against all cached keys and returns top-2048 ids per layer (KBs over
PCIe), CPU does the sparse MLA attention on 2048 x ~1 KB local reads. (b) **Full-attention-layer host for
GDN hybrids at 1M** — qwen3next has ~12 full-attn layers; their KV at q4 is ~6 KB/token -> ~6 GB at 1M:
fits. This extends your proven finding #6 (+18% at 32K) to the 1M regime where the CPU alternative doesn't
degrade gently — it collapses.

**Math.** (a) Indexer scan/token at 1M: ~5-10 GB. CPU replicated: 50-100 ms/token (a second wall). GPU at
456 GB/s: 11-22 ms + ~61 dispatches of Vulkan overhead (~50-100 us each) + tiny PCIe transfers -> ~20-40 ms/token
realistic. Combined with 4-socket weights (42 ms) + fixed ARs: **504B at 1M context decoding at ~5-10 t/s**,
versus ~1 t/s-or-collapse today. (Idea 3's CP-on-CPU is the fallback if Vulkan dispatch overhead eats the
win: 8-way sharded scan ~10 ms/token but burns CPU bandwidth.) (b) hybrid full-attn KV on GPU: 6 GB /
456 GB/s = 13 ms of scan spread over 12 layer calls — decode stays ~flat to 1M at your GDN-hybrid rates
(~8-12 t/s).

**Speedup.** (b) is the pragmatic 1M *product*: near-flat decode at 1M on the 80B-A3B hybrid. (a) is the
1M *frontier* play for the 504B.

**Novelty/difficulty.** (b) is incremental on your existing offload — medium. (a) "indexer-only offload"
is, to my knowledge, a genuinely fresh combination (DSA is new; nobody is running it split CPU/GPU) —
medium-high difficulty (a Vulkan top-k/score kernel, or abuse existing matmul+argsort ops; batch all
layers' dispatches to amortize launch overhead). Verify the exact index bytes/token for GLM-5.2 before
committing — if it's >12 GB at 1M, quantize the index to int4 (selection is rank-based, very
quantization-tolerant) or keep only deeper layers' indexes on GPU.

**Failure modes.** Vulkan dispatch latency at batch-1 (61 sequential small kernels) is the main risk for
(a) — mitigate by fusing all-layer scoring into one dispatch where dependencies allow (they don't across
layers — q_idx depends on layer input — so you're stuck with sequential; measure whether ~60 x 0.5 ms beats
CPU's CP alternative). PCIe transfer of per-layer q is negligible (KBs), so no shuttle problem this time.

## 6. Close the 3-5x decode overhead gap: cycle accounting, then barrier/op-granularity surgery

**Idea.** Before (or alongside) everything above: account for every microsecond of one decode token. The
A3B model runs 5x off its DRAM ceiling with NO comms — the loss is inside ggml: ~2-3k graph nodes x
per-node threadpool barrier (2-5 us each = 5-15 ms), small-GEMV bandwidth inefficiency on 512-wide expert
rows (scattered 1-KB row reads -> maybe 30-40% of stream BW), dequant overhead, TLB misses. Fixes: fuse
norm/elementwise chains into the adjacent matmul (fewer barriers), batch the 8 experts' GEMVs into one
interleaved pass (sequential streaming instead of 8 scattered scans), 1-GB hugepages for weight arenas,
software prefetch on expert rows, and thread-count tapering for tiny ops (24 threads on a 2048-wide op is
pure barrier tax).

**Math.** If overhead is ~50-60 ms of the A3B's 97 ms/token, halving it is +40-50% decode; the same fixes
shrink the skew that idea 1 fights (fewer, fatter ops = less spread). These gains apply to every model, TP
or not.

**Speedup.** +30-80% decode on small-active MoE; +10-20% on big models. Multiplies with ideas 1-2.

**Novelty/difficulty.** Zero novelty, pure engineering, medium effort, near-zero risk. The reason it ranks
this high is certainty x breadth.

**Failure modes.** The gap might partly be irreducible (dequant compute, unavoidable gather entropy in
MoE). The accounting itself will tell you within a day.

## 7. KV persistence + pipeline-across-nodes as the capacity topology

**Idea.** (a) Prefill 1M ONCE, snapshot KV/state to RAM/NVMe (llama.cpp has slot save/restore), and serve
many queries against it — turns an hours-long prefill into an amortized cost; with 1.5 TB/node you can keep
several 100-GB contexts hot. (b) For models/contexts that exceed one node, place *pipeline* stages across
nodes and TP within node: inter-node traffic at decode is one 8-16 KB activation hop per stage boundary
(~10 us) instead of ~60-120 ARs — PP crossing IB is nearly free.

**Math for (b), honest version:** PP gives NO batch-1 latency win — stages execute sequentially in time,
so per-token latency ~= single-node latency; DRAM bandwidth does not aggregate. Its value is capacity
without the decode tax you measured for inter-node TP (-27%) or EP (-13%): you run the 504B at Q8 (700 GB,
2 nodes) or hold 1M KV, at ~full single-node speed. Contrast with idea 1 which makes inter-node TP
*positive*: once idea 1 lands, TP beats PP across nodes; until then, PP is the correct inter-node default.

**Speedup.** (a) is a >100x wall-clock win for the repeated-query workflow (arguably the biggest practical
number in this document); (b) is 1.3-1.5x vs inter-node TP/EP decode today.

**Novelty/difficulty.** Known; low ((a)) to medium ((b) — llama.cpp layer-splitting exists via RPC but
you'd want your UCX path). Failure mode for (a): NFS load bandwidth (0.3-0.7 GB/s measured) makes a 108 GB
restore ~3-6 min — fine versus hours, but put snapshots on local NVMe if colder-than-RAM.

## 8. A shard/replicate cost model instead of all-or-nothing TP (formalizes your shexp lesson)

**Idea.** Per component, shard iff `saved_local_bytes / 100 GB/s > delta_ARs x AR_cost + GEMV_efficiency_loss`.
Your shexp experiment already proved the negative case; your GDN+attn sharding proved the positive case.
Make it a load-time decision per tensor-role per model (with AR_cost as a measured runtime constant), so
e.g. Qwen's 512-wide experts stay replicated at 4-way while DeepSeek's 2048-wide shard, attention shards at
depth but not at 2K, etc. Include depth-dependence: KV/scan terms grow with context, so the optimal plan
*changes* as the context deepens — re-plan at thresholds.

**Speedup.** 10-25% by avoiding today's negative-value shards; prevents idea-1's cheaper ARs from being
spent on worthless splits. Trivial-to-medium difficulty; known idea (auto-parallel cost models) applied at
unusually fine grain. Failure mode: model constants drift with quant/arch; keep it a heuristic, not a solver.

## 9. MoE-specific tricks: deterministic-owner reduction and expert compaction (modest, keep expectations low)

**Idea.** The router input is post-AR and replicated, so every rank computes identical routing
*deterministically* — every rank already knows which ranks own firing experts, with zero extra comms. So
in EP mode the per-layer combine need not be a symmetric AR: it's a gather from the (often 1-2) owning
ranks, ideally a one-sided RDMA put from owners into consumers, and ranks with zero hits skip sends
entirely. Plus your own queued idea: expert compaction (build a dense local `selected_experts` and feed
`mul_mat_id`) to recover GEMM efficiency for EP prefill. On *prefetch-ahead routing*: on CPU it's
near-worthless — all experts are already in local DRAM and decode pays the read regardless; there is no
PCIe cliff to hide. Comms-aware routing (biasing top-k toward local experts) changes model output —
near-tie experts flip and your own logs show this model family is routing-sensitive; don't.

**Speedup.** ~5-15% on EP decode; expert compaction maybe +20-30% EP prefill (your own earlier data
brackets it). Known techniques; low-medium difficulty. Failure mode: at tensor-mode MoE (your better mode)
none of this applies — which is why it ranks low.

## 10. Stale/approximate all-reduce — ranked last, on purpose (anti-recommendation with a cheap kill-test)

**Analysis.** (a) *Quantized/low-rank AR*: your ARs are 8-16 KB — latency/skew-bound, below any rendezvous
threshold; shrinking bytes saves ~nothing. Dead on arrival, skip. (b) *1-layer-stale AR* (proceed on the
local partial, apply the remote correction to the residual one layer late): the residual pass-through is
corrected exactly, but layer l+1's nonlinearity was evaluated on an input missing the other rank's entire
layer-l contribution — that is a ~50% perturbation of the layer output, not noise; it feeds norms, routing
(top-8-of-256 near-ties — your logs show flips from mere reduction-order changes at ~1e-3), and attention.
Expect a quality cliff, not graceful degradation. And the prize is small once idea 1 lands: you'd be hiding
30-50 us/layer, ~3-5 ms/token. (c) The salvageable fragment: use the *stale* pre-AR activation only for
side-channel hints with no correctness requirement — e.g., start routing/prefetch heuristics during the AR
window. Marginal on CPU (see idea 9). **Kill-test if curious:** your TP stack can simulate staleness in an
afternoon (delay the remote partial's residual add by one layer, measure perplexity on 1k tokens). If PPL
moves >1%, bury it permanently.

---

# TOP 3 — what I'd try next, most breakthrough-likely first

**1. Epilogue-fused, chunk-grained all-reduce with one-sided RDMA (idea 1) + the cycle-accounting sweep
(idea 6) as its first step.** Your own measurements prove the decode wall is synthetic: 170 us AR vs
2-12 us wire, and 3-5x headroom to the DRAM ceiling. This is the only item that flips your headline
negative result (multi-node TP decode loses) into the thing you actually want (aggregate 4-8 sockets of
DRAM bandwidth behind one decode stream). Expected composite: 2-2.5x decode on big models, and it raises
the ceiling every other idea pushes toward. Start with the instrumented breakdown of the 170 us — one day
of measurement decides the design.

**2. Speculative decoding: MTP self-draft plus prompt-lookup drafting at long context (idea 2).**
Highest-certainty multiplier on the list, ~2x decode, and it is *the* theoretically sound answer to
"speculate past the AR": token-level speculation is the only granularity with a cheap verifier, and on CPU
the verifier is nearly free (20 FLOP/byte of headroom means verifying 4-6 tokens costs the same DRAM
traffic as one). It amortizes weights, ARs, and graph overhead simultaneously, and prompt-lookup is
disproportionately strong exactly at your 1M-context target workloads.

**3. The 1M package: KV-sharded LSE-merge context parallelism (idea 3) + block-sparse/blocked-DSA prefill
(idea 4) + GPU-resident indexer (idea 5a).** This is the set that makes the stated goal — frontier model,
1M tokens — *exist at all* on this cluster: ~1-2 h fresh 1M prefill (amortized to minutes with KV
persistence) and ~5-10 t/s decode at full depth for the 504B, versus days and ~1 t/s today. It's the most
work, but each third is independently useful, and the first move is cheap: re-implement the DSA indexer as
blocked int8 GEMM and check whether the 14 -> 2 t/s collapse was ever real physics. Meanwhile, ship the
pragmatic variant — GDN hybrid + q4 full-attn KV on the B580 (idea 5b) — which gets near-flat 1M decode
with the least effort of anything in section B.

Everything composes multiplicatively where it matters: (1) x (2) alone is a plausible 4-5x on
frontier-model decode; adding (3) is what takes "fast" from 4K context to 1M.
