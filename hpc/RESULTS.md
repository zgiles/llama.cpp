# HPC CPU-inference experiments — results log

Hardware: 2× nodes, each 2× Intel Xeon Platinum 8168 (Skylake-SP, 24c/socket, no HT,
48c total), 1.5 TB DDR4-2666 (6ch/socket), NUMA node0=cpu0-23 / node1=cpu24-47.
Interconnect: Mellanox ConnectX-5 EDR InfiniBand (100 Gb/s), mlx5_0. IPoIB 172.16.0.121/124.
Note: Skylake-SP has AVX-512 but NO AVX512-VNNI (DL Boost) — that's Cascade Lake+.
Note: both nodes share hostname `nixos`; NixOS firewall (`nixos-fw`) drops non-SSH TCP,
so MPI/UCX bootstrap must be SSH-tunneled (RDMA data path bypasses the firewall).

## A2 — fabric all-reduce / RDMA latency (UCX 1.19, ucx_perftest tag_lat, RC over IB)

Bootstrap tunneled over SSH; data path native IB RC (UCX_TLS=rc,sm,self, mlx5_0:1).
One-way latency (ping-pong/2):

| bytes  | lat_us (1-way) | bw MB/s | msg/s   |
|--------|----------------|---------|---------|
| 8      | 0.918          | 8.3     | 1088878 |
| 64     | 1.204          | 50.7    | 830395  |
| 256    | 1.496          | 163     | 668282  |
| 1024   | 1.694          | 576     | 590250  |
| 2048   | 2.236          | 873     | 447137  |
| 4096   | 4.684          | 834     | 213479  |
| 8192   | 5.896          | 1325    | 169599  |
| 16384  | 5.963          | 2620    | 167689  |
| 32768  | 8.128          | 3845    | 123030  |
| 65536  | 9.807          | 6373    | 101970  |
| 131072 | 15.138         | 8257    | 66059   |
| 262144 | 25.961         | 9630    | 38520   |

Takeaways:
- Latency floor 0.92 us — healthy EDR RC. Half-roundtrip ~1 us small-message.
- TP per-layer activation all-reduce is hidden*batch*dtype. For hidden=8192:
  fp16=16KB -> ~6 us one-way; fp32=32KB -> ~8 us one-way.
- Rough cross-node TP comms budget (2 nodes, recursive-doubling allreduce ~= 2x one-way):
  80 layers * 2 allreduce/layer * ~12 us ~= ~1.9 ms/token of comms => few-hundred tok/s
  ceiling from comms alone. That is far above the single-node giant-model floor, so
  cross-node TP is viable. Levers to cut it: fp16 activations, SHARP in-network reduce,
  streaming/tile-wise partial reduce, comm/compute overlap.

## A1 — NUMA / decode-bandwidth characterization

Model: Llama-3.1-70B-Instruct Q4_K_M, 39.59 GiB (42.5 GB), 70.55B params. tg96/pp128, r=3.
--no-mmap (private placement) except the default case. weights read per token ~= 40 GB.

| config                      | threads | decode tg96 t/s | prefill pp128 t/s | eff BW (GB/s, decode) |
|-----------------------------|---------|-----------------|-------------------|-----------------------|
| default (mmap, no bind)     | 48      | 1.17            | 2.00              | ~50                   |
| socket0 fully local         | 24      | 0.77            | 0.97              | ~33                   |
| forced remote (cpu1/mem0)   | 24      | 0.67            | 1.00              | ~28                   |
| interleave all              | 48      | 1.19            | 2.00              | ~51                   |
| llama --numa distribute     | 48      | 1.12            | 1.94              | ~48                   |
| no-bind, no-mmap            | 48      | 1.17            | 1.93              | ~50                   |
| aggregate 2x per-socket     | 24+24   | 0.78+0.78=1.56  | 0.96+0.97=1.93    | ~66 (2 streams)       |

Dual-socket 8168 achievable mem BW ~170-190 GB/s (STREAM). Decode uses only ~28% of it.

### HEADLINE FINDING (Skylake): decode is COMPUTE-bound, not bandwidth-bound.
- Single-stream decode draws only ~50 GB/s of ~180 GB/s available (~28% util).
- Forcing ALL weight reads across UPI costs only ~13% (0.77 -> 0.67). If decode were
  bandwidth-bound, remote would be ~2x worse. Small penalty => BW is not the limiter.
- 24t->48t single-stream scales 1.52x (sublinear); aggregate 2-instance ceiling 1.56 t/s.
- Limiter is per-token compute: Q4_K superblock dequant + dot-product WITHOUT AVX512-VNNI.
  (AVX-512 IS compiled via -march=native; Skylake just lacks the INT8 DL Boost path.)

### Implications / strategy update
- Intra-node socket-local TP headroom on Skylake ~= 1.56/1.19 = ~1.3x (smaller than the
  2x we'd see if BW-bound), because the second socket mostly adds compute, not just BW.
- Multi-node TP still wins, but on Skylake the framing is "aggregate COMPUTE" not "aggregate
  bandwidth" — more cores doing dequant+FMA in parallel.
- CRITICAL CAVEAT: the target is Cascade Lake, which HAS AVX512-VNNI. VNNI accelerates the
  INT8 dot-product, cutting decode compute -> Cascade Lake will be FASTER per-token and more
  BANDWIDTH-bound, shifting the balance back toward our original NUMA/BW thesis. Skylake
  over-represents compute cost. Must re-run A1 on the Cascade Lake box to recalibrate.
- New cheap lever if compute-bound: a quant that is cheaper to dequant (e.g. Q4_0/Q8_0 simpler
  unpack) may decode FASTER per-token despite more bytes. Worth a quant sweep.
- Thread-count sweep (a1b), socket0-local decode tg32:
    6t=0.25  12t=0.48  18t=0.63  24t=0.78 t/s
    6->12 (2x cores) = 1.92x (near-linear => compute-bound);
    18->24 = 1.24x (vs 1.33 ideal => mild BW/sync pressure only near full occupancy).
  Confirms: predominantly COMPUTE-bound, with memory BW a secondary constraint that
  only appears as the socket fills. A BW-bound workload would saturate by ~12 threads.

## Strategy: two ORTHOGONAL, PORTABLE workstreams (no arch special-casing)
W1 Tensor parallelism — aggregates whatever resource is scarce per box (compute on
   VNNI-less Skylake, bandwidth on Cascade Lake, mix on AMX Emerald/Granite, or AMD).
   Identical design everywhere; only the expected speedup magnitude varies per box.
   In-process socket-local L1 shmem reduce + UCC L2 inter-node. Reduce cost is provably
   negligible: 160 reduces/token * <1us (32KB shmem add) << ~850ms/token.
W2 On-box math — best per-arch kernels (ggml already dispatches AVX512/VNNI/AMX/NEON).
   Compute-bound decode => faster Q4_K dequant+dot is a universal win; cheaper-to-unpack
   quant may decode faster per-token. Quant sweep probes this.
Both compose multiplicatively.

## L1 in-process shared-memory all-reduce (hpc/tp_allreduce.{h,c}) — built + validated
Thread-parallel hierarchical reduce (reduce-scatter + all-gather), the spine of CPU TP.
Correctness PASS. Two iterations of building taught us what theory didn't:
- The DATA movement is never the cost; the BARRIER is. A flat 48-way centralized barrier
  across UPI was ~38us FLAT regardless of vector size (false sharing on count/sense + cross
  -socket cacheline ping-pong).
- Fixes applied: cache-line-padded barrier state (arch-aware 64/128B), and a 2-LEVEL
  barrier (K threads reduce within a socket on shared L3; only the 2 socket-leaders sync
  across UPI). Also HT-aware pinning (one thread/physical core; skip SMT siblings).

Measured all-reduce latency (us) by threads-per-socket K (2 sockets, 8168), corrected build:
| hidden (bytes)   | K=1   | K=2   | K=4   | K=8   | K=24  |
|------------------|-------|-------|-------|-------|-------|
| 2048  (8 KB)     | 5.87  | 6.04  | 7.27  | 9.84  | 24.59 |
| 8192  (32 KB)    | 11.66 | 11.71 | 9.12  | 11.26 | 26.06 |
| 32768 (128 KB)   | 36.52 | 27.41 | 20.87 | 17.51 | 29.85 |

Takeaways:
- Sweet spot shifts with size: small vectors -> few threads (barrier-bound, K=1-2 ~6us);
  large -> more threads (data-bound, K=8). For 70B activations (hidden 8192) best ~9us (K=4).
- The all-reduce should use a TUNED SUBSET of threads, not the whole 48-thread pool.
- Hierarchical barrier cut the K=24 case ~38us -> ~25us. High K is still barrier-bound.
- Per-token comms (2 reduces x 80 layers) at the 70B sweet spot ~1.5 ms/token == ~0.2% of a
  ~850 ms/token decode. CONFIRMED: the in-process reduce is negligible for giant models;
  the barrier tuning only matters once we push toward small/fast models or high tok/s.
- Bug found while building: passing nth where the API wanted K -> sub-barrier waited on more
  threads than ever arrive -> deadlock. (Caught by a minimal pinned reproducer.)

L2 hook (inter-node UCC/IB) is stubbed in tp_allreduce_f32 at the point where `scratch`
holds the node-local sum — ready to wire to the A2 RDMA path. Firewall now flushed (native
UCC bootstrap available), hostname-collision worked around via direct IPs.

## Cross-arch probe: Skylake (8168, no VNNI) vs Cascade Lake (8260, VNNI)
Same /kimie model loaded directly (no copy): deepseek-r1-qwen-2.5-32B-ablated-Q8_0 (32GB).
Socket0-local decode tg32, threads sweep:

| threads | Skylake 8168 t/s | CSL 8260 t/s |
|---------|------------------|--------------|
| 6       | (n/a)            | 0.62         |
| 12      | 1.73             | 1.17         |
| 18      | 2.19             | 1.55         |
| 24      | 2.48             | 1.91         |
Skylake 24t Q8 ~= 80 GB/s eff (~88% of socket BW) => Q8_0 is BANDWIDTH-bound even w/o VNNI
(trivial dequant + 2x bytes). CSL 24t Q8 ~= 62 GB/s eff (~62% util) => slower, not saturated.

IMPORTANT — this CSL run is NOT a clean verdict on VNNI:
  - Node 92 is SHARED and began other work around the run; not isolated.
  - HT is ON; threads were not explicitly pinned to distinct PHYSICAL cores (numactl bound to
    the node cpuset incl. SMT siblings) — possible sibling packing.
  - 8260 base clock 2.4 GHz < 8168 2.7 GHz (-11% if compute-bound).
  - Q8_0 is a poor VNNI discriminator: its dequant is trivial, so it's BW-bound regardless;
    VNNI's headroom is on the Q4_K path (the quant that was COMPUTE-bound on Skylake).
Robust takeaways: (1) regime is QUANT-dependent — Q4_K compute-bound, Q8_0 BW-bound on the
same chip; (2) the VNNI-flips-regime hypothesis is UNCONFIRMED and needs a clean re-run:
physical-core-pinned, Q4_K_M path, idle node (do when 92 is free / post-reboot for IB).

## L2 inter-node all-reduce over UCX/IB (hpc/tp_net_ucx.c) — built + validated
Decision: build L2 on UCX directly, NOT MPI. Reason: cross-node MPI_Init hangs on these nodes
(shared hostname "nixos" -> PMIx thinks both ranks co-located -> wireup deadlock; confirmed:
local 2-rank MPI works, cross-node hangs even over plain TCP). UCX identifies peers by exchanged
worker addresses (no hostname dependency), is proven over IB, and avoids an MPI runtime dep in
llama.cpp. Bootstrap: plain TCP socket on the IB subnet (firewall flushed) to swap UCX worker
addresses; data path native IB RC (UCX_TLS=rc,sm,self mlx5_0:1).

2-node all-reduce (exchange + sum), correctness OK all sizes:
| floats | bytes  | lat_us |
|--------|--------|--------|
| 256    | 1 KB   | 3.47   |
| 1024   | 4 KB   | 4.02   |
| 4096   | 16 KB  | 8.10   |
| 8192   | 32 KB  | 11.99  |   <- 70B hidden activation
| 16384  | 64 KB  | 21.78  |
| 32768  | 128 KB | 36.18  |
| 65536  | 256 KB | 64.33  |

### Both levels now proven on real hardware
L1 (in-process shmem, 32KB) ~9 us  +  L2 (UCX/IB 2-node, 32KB) ~12 us  => full hierarchical
all-reduce ~21 us for a 4-way TP (2 nodes x 2 sockets). Per-token comms = 21us x 2 reduces x
80 layers ~= 3.4 ms/token. Against a ~850 ms/token single-socket 70B decode, a 4-way TP that
cuts compute ~4x lands near ~210 ms/token + 3.4 ms comms => comms is ~1.6% overhead.
CONCLUSION: the hierarchical local+remote all-reduce is validated and comms is negligible;
multi-node TP is viable. Next: wire tp_net_allreduce into tp_allreduce.c's L2 hook (unified
API), then the ggml graph integration (shard weights + insert all-reduce in build_lora_mm).

## UNIFIED L1+L2 primitive — wired + validated end-to-end (hpc/tp_full_test.c)
tp_net decoupled from core via a function pointer (c->net_allreduce); tp_allreduce_f32 now
transparently does local-only OR local+remote. Test: 2 nodes x 2 sockets, process-per-node,
ith==0 drives the UCX worker (single-thread mode), the rest wait. Correctness (GLOBAL sum
across all 4 socket-partials over both nodes): PASS.

Full 2-node x 2-socket all-reduce latency (fp32):
| bytes  | K=2 (4 thr/node) | K=24 (48 thr/node) |
|--------|------------------|--------------------|
| 16 KB  | 24.4 us          | 72.0 us            |
| 32 KB  | 40.8 us          | 120.7 us           |   <- 70B activation
| 64 KB  | 71.9 us          | 180.9 us           |

CORRECTION to the earlier ~21us / 1.6% estimate (which used best-case components in
isolation): the COMBINED path is barrier-bound at high thread counts. 4 hierarchical
barriers/all-reduce across 48 threads/node dominate; data+network are only ~20us of the 120us.
Realistic per-token comms (160 all-reduce/token):
  K=2:  160 x 40us  ~= 6.4 ms/token  -> ~3% of a ~210ms 4-way-TP 70B decode
  K=24: 160 x 120us ~= 19 ms/token   -> ~9%
Still a clear 4-way win, but comms is 3-9% (not 1.6%) depending on tuning. Caveat both ways:
this standalone bench WORST-CASES barriers (back-to-back reduces, no compute between). In real
inference there is heavy matmul compute between the 2 reduces/layer, so threads aren't in a
tight barrier loop and comms/compute OVERLAP can hide most of it.
Optimization levers (for later): fewer threads for the reduce (barrier-bound), fewer barriers,
fp16 activations (half bytes), and comms/compute overlap (the big one for single-stream).

## ggml integration progress (task #9, in progress)
Design LOCKED: rank=node (2-way first), all-reduce = ggml_map_custom1_inplace callback after
row-parallel matmuls (no core ggml changes). Pieces built:
- hpc/tp_ggml.{h,c}: the ggml<->tp_net all-reduce bridge op (userdata=NULL => no-op).
- hpc/tp_shard.{h,c}: load-time weight-sharding MATH + env config (LLAMA_TP_SIZE/RANK).
  Column-parallel (split ne[1], whole rows -> 1 contiguous read, block-safe) and row-parallel
  (split ne[0] contraction -> block-aligned, STRIDED per-row read plan; refuses misaligned
  splits that would cut a quant block). Unit-tested (hpc/tp_shard_test.c): ALL CHECKS PASSED
  for Q4_K (256/144B) and Q8_0 (32/34B), incl. alignment guard.
Remaining loader/graph work (next): wire tp_shard into create_tensor (reduce ne) at
llama-model-loader.cpp:1047 and into load_data_for (strided read via the plan) at :1385;
thread TP context into llm_graph_context (llama-graph.h:789) -> build_lora_mm
(llama-graph.cpp:1085) inserts the bridge after wo/ffn_down; process-per-node launch +
token-for-token validation vs single-node. Start with FFN-only sharding (simplest correct
slice: ffn_up/gate column, ffn_down row+all-reduce; attention replicated).

## Loader sharding WIRED INTO real llama.cpp + VALIDATED
- src/llama-tp-shard.{c,h} (the tested math, extern "C" for C++) added to src/CMakeLists.txt.
- llama-model-loader.h: tp_cfg (from env) + tp_plans map members.
- create_tensor: tp_role_for_tensor() maps ffn_up/gate->COLUMN, ffn_down->ROW; creates this
  rank's SHARDED tensor (reduced ne) + records the load plan.
- load_data_for: strided gather-read of the rank's slice (requires --no-mmap for TP; row-
  parallel data is non-contiguous in the file).
- VALIDATION (Llama-3.2-3B-Q6_K, --no-mmap, single node, TP_RANK=0 of SIZE=2):
    baseline peak RSS 2763 MB  vs  TP=2 rank0 1916 MB  (-31%, == FFN halved).
  Sharded graph RUNS (tg produced, no crash; strided quant load not corrupted). Output is
  numerically wrong until the graph all-reduce is added (expected).
- BUILD GOTCHA: flake devShell openssl path went STALE on the NFS nix store ("Stale file
  handle"). Build with fresh current-nixpkgs toolchain instead of `nix develop .#default`:
  nix shell nixpkgs#cmake nixpkgs#ninja nixpkgs#gcc nixpkgs#pkg-config nixpkgs#openssl
  --command cmake ... -DLLAMA_BUILD_SERVER=OFF --target llama-bench.

## *** END-TO-END MULTI-NODE TENSOR-PARALLEL INFERENCE WORKING + CORRECT ***
Graph all-reduce wired in and validated cross-node:
- src/llama-tp-net.{c,h}: UCX/IB inter-node transport (lazy-bootstrapped on ggml thread 0 from
  env LLAMA_TP_SIZE/RANK/PEER/PORT) + llama_tp_allreduce_op (ggml custom1 op) + llama_tp_enabled().
  Optional build: -DTP_UCX_INC=<ucx-dev>/include -DTP_UCX_LIB=<ucx>/lib (else no-op).
- llama-graph.cpp build_ffn: after ffn_down, `if (down && llama_tp_enabled()) cur =
  ggml_map_custom1_inplace(ctx0, cur, llama_tp_allreduce_op, 1, nullptr);`
- loader: tp auto-disables mmap; THE KEY BUG was that load_all_data (not load_data_for) is the
  real CPU load path — it read ggml_nbytes from weight->offs with NO per-rank base_off, so every
  rank loaded the FIRST shard (both ranks computed identical partials -> garbage). Fixed by
  applying the strided shard-read plan in load_all_data's host-buffer branch too.
- RESULT (Llama-3.2-3B-Q6_K, 2 nodes 121+124 over IB, FFN sharded, attention replicated):
  "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is
   Madrid. The capital of Germany is Berlin." -- COHERENT + CORRECT. Debug confirmed the two
  ranks compute complementary halves (mine != peer) and the all-reduce sums them.
  (Diverges slightly from single-node after "Paris": float non-associativity in the reduce vs
   the monolithic matmul flips greedy tokens -- numerically equivalent, not bit-exact; normal TP.)
Run: rank1/server on 124 (LLAMA_TP_RANK=1 LLAMA_TP_PORT=P), rank0/client on 121 (LLAMA_TP_RANK=0
  LLAMA_TP_PEER=172.16.0.124 LLAMA_TP_PORT=P), both UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc,sm,self.

Remaining (future): attention sharding (wq/wk/wv col + wo row+all-reduce, more comm), N>2 nodes
(recursive-doubling), NUMA replication within node, comms/compute overlap, perf tuning on large
models (3B is too small to show TP speedup -- it was for correctness).

## PHASE 1 — Large-model perf baseline (DONE 2026-06-22): 70B Q8_0, FFN-sharded 2-node TP
Model: Llama-3.3-70B-Instruct-ablated-Q8_0 (split GGUF, point at -00001; 69.82 GiB, 70.55B).
Nodes: Skylake 8168 pair 121+124, 48 threads/node, EDR IB. llama-bench -p128 -n64.
**The split GGUF + TP-shard combination WORKS** (was previously untested; only the single-file
3B had been validated). Both ranks bootstrapped transport, ran in lockstep (identical t/s),
tp_dbg confirmed complementary partials (mine != peer) -> all-reduce sums real halves.

| config                        | decode tg64 t/s | prefill pp128 t/s | peak RSS / node |
|-------------------------------|-----------------|-------------------|-----------------|
| single-node (mmap=0, 48t)     | 0.86 ± 0.00     | 19.22 ± 0.22      | 71.7 GB         |
| 2-node TP FFN-sharded (mmap=0)| 1.24 ± 0.00     | 30.93 ± 0.80      | 43.2 GB         |
| **speedup / ratio**           | **1.44x**       | **1.61x**         | **0.60 (-40%)** |

MEASUREMENT GOTCHA: the first single-node run used default mmap=1 and gave NOISY garbage
(tg64 0.40 ± 0.12, pp128 11.89 ± 3.15) -- mmap-over-NFS page-faults the 70 GB during the timed
runs. Re-run with `-mmp 0` (NOT `--no-mmap`, which llama-bench doesn't accept) -> fully resident,
rock-steady. ALWAYS bench CPU with mmap=0 when the model is on NFS. Decode 0.86 t/s = ~60 GB/s
effective (consistent with A1's Q4_K_M 70B scaled to Q8's 70 GB/token read).

ANALYSIS (the Phase-1 decision point):
- Memory: measured 0.60x/node EXACTLY matches theory. FFN ~80% of params; halving it ->
  per-node = 0.2 (attn/embed, replicated) + 0.8/2 (FFN) = 0.60 of full. -40% RAM/node confirmed.
- Decode 1.44x vs ideal 1.67x (= 1/0.60). The ~0.23 gap is NOT comms: at hidden=8192 the
  all-reduce is ~32 KB, 2 reduces x 80 layers ~= <2 ms/token, i.e. <0.3% of the 806 ms/token.
  The gap is the REPLICATED ATTENTION (full attn compute on BOTH nodes, zero parallelism there)
  plus barrier/sync. Attention is now the dominant non-parallelized cost.
- prefill 1.61x is closer to ideal (compute-bound regime, FFN dominates the matmul flops more).
- DECISION: attention sharding (Phase 2) is the clear next lever -- it both removes the
  replicated-attention ceiling (pushes decode toward and past 1.67x via head-parallel attn) AND
  shards the KV cache for per-node KV-memory savings at long context. Phase 1 confirms it's worth
  the work. N>2 nodes (Phase 3) then multiplies the FFN win further.
Run recipe used: rank1/server on 124 first (`LLAMA_TP_SIZE=2 LLAMA_TP_RANK=1 LLAMA_TP_PORT=13710`),
then rank0/client on 121 (`... RANK=0 LLAMA_TP_PEER=172.16.0.124 LLAMA_TP_PORT=13710`), both
`UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc,sm,self`, llama-bench `-mmp 0 -t 48 -p 128 -n 64 -r 2`.

## PHASE 2 — Attention sharding DONE + VALIDATED (2026-06-22)
Implemented head-parallel attention TP, gated behind a new `LLAMA_TP_ATTN=1` env flag (FFN-only
Phase-1 mode stays the default when unset). Design:
- wq/wk/wv COLUMN-sharded (split ne[1] = output rows = Q/KV heads); wo ROW-sharded (split ne[0]
  = the per-head contraction) + an all-reduce after the wo matmul (mirrors ffn_down). 2 all-
  reduces/layer now instead of 1.
- THE key subtlety (per the plan): local head counts. Solved with ONE lever — after load_tensors,
  divide hparams.n_head_arr / n_head_kv_arr by tp_size (llama.cpp ~line 333). Because n_embd_head_k
  is stored independently and n_embd_k_gqa = n_embd_head_k * n_head_kv, this automatically (a)
  shrinks the KV cache to local KV heads and (b) feeds local head counts to build_qkv / build_attn,
  while n_embd (residual stream) stays full. build_attn_mha is already head-count-agnostic (derives
  dims from tensor ne). Tensors are created/sharded against the FULL GGUF dims FIRST (file dim check
  passes), THEN the arrays are reduced — order matters.
- GQA stays correct per-rank: contiguous equal splits of both Q heads and KV heads map rank r's
  local Q heads exactly onto its local KV heads (proof: q head h -> kv head floor(h/g); rank r's
  q range [r*Hq/s,(r+1)*Hq/s) maps to kv range [r*Hkv/s,(r+1)*Hkv/s) when s | Hq and s | Hkv).
- Guards: refuse (clear error) if any layer's n_head/n_head_kv isn't divisible by tp_size, or if
  the model uses fused QKV / attention bias (column-splitting a 1-D bias is unsupported for now;
  Llama has neither). Also bumped the all-reduce scratch 1M->4M floats so larger prefill batches
  don't abort.
Files: src/llama-tp-shard.{c,h} (attn flag), src/llama-model-loader.cpp (tp_role_for_tensor:
ATTN_Q/K/V->COLUMN, ATTN_OUT->ROW), src/llama-tp-net.{c,h} (llama_tp_attn_enabled), src/llama-
graph.cpp (all-reduce after wo), src/llama.cpp (head-count reduction + guards).

CORRECTNESS (Llama-3.2-3B-Q6_K, llama-simple -n 32 "The capital of France is"):
  attn+FFN sharded 2-node output == single-node output TOKEN-FOR-TOKEN (exact, not just coherent):
  "...Paris. ...located in the Île-de-France region. ...also the most populous city in France."
  Log confirmed "attention sharded — local n_head=12 n_head_kv=4 (tp_size=2)" (24->12, 8->4).

PERF (Llama-3.3-70B-Q8_0, 2 nodes, llama-bench -mmp 0 -p128 -n64 -r2, LLAMA_TP_ATTN=1):

| config                          | decode tg64 | prefill pp128 | peak RSS / node |
|---------------------------------|-------------|---------------|-----------------|
| single-node                     | 0.86        | 19.22         | 71.7 GB         |
| Phase 1: FFN-only TP            | 1.24        | 30.93         | 43.2 GB         |
| **Phase 2: FFN + attn TP**      | **1.32**    | **34.83**     | **37.0 GB**     |
| Phase 2 vs single-node          | 1.53x       | 1.81x         | 0.52 (-48%)     |
| Phase 2 vs Phase 1              | +6.5%       | +12.6%        | -14%            |

ANALYSIS:
- Memory: -6.2 GB/node vs Phase 1, matches theory (attention ~13 GB total, halved -> ~6.5 GB/node).
  Now 0.52x single-node RAM. NOTE: this short-context bench barely exercises the KV cache, so the
  RSS drop is almost all ATTENTION WEIGHTS, not KV. The KV-cache halving (the other half of the
  win) shows up at LONG context — that's where Phase 2 pays off most (per-node KV memory).
- Decode only +6.5% over Phase 1 (1.24->1.32), below what byte-count predicts (~1.9x ideal vs
  single). At batch=1 decode the FFN weight reads (already halved in Phase 1) dominate; attention
  is a smaller slice, and there's a fixed serial floor: the REPLICATED output projection (lm_head
  [n_embd x 128k vocab], ~1.1 GB read + a 128k-wide matmul EVERY token) and tok_embd are not sharded.
  (EARLIER CLAIM HERE — "comms is ~0.25%, not the limiter" — was WRONG; it used the standalone
  microbench latency. The in-inference measurement below shows comms is ~5-10% and sync-bound.)
- Prefill +12.6% (compute-bound regime benefits more from splitting attention flops).
- Next levers for DECODE specifically: shard the output projection (column-parallel lm_head +
  all-gather logits), N>2 nodes (splits FFN further), comms/compute overlap. For MEMORY at scale:
  the KV-cache sharding already in place is the long-context win.
Run recipe: identical to Phase 1 plus `LLAMA_TP_ATTN=1` in the env on BOTH ranks (port 13730 used).

## IN-INFERENCE COMMS MEASUREMENT (2026-06-22) — comms is sync-bound, ~5-10% of decode at 2 nodes
Instrumented the all-reduce op (LLAMA_TP_TIME: count + us/call, printed atexit) and added a comms-off
timing mode (LLAMA_TP_NOCOMMS: skip the UCX exchange -> wrong output, pure-compute timing). 70B Q8_0,
2-node, attn+FFN sharded, llama-bench -mmp 0 -p128 -n64 -r2:

| run                  | decode tg64 | prefill pp128 | all-reduce avg/call | note                  |
|----------------------|-------------|---------------|---------------------|-----------------------|
| comms ON             | 1.24 ± 0.11 | 34.16         | ~400 us             | NOISY (sync jitter)   |
| comms OFF (NOCOMMS)  | 1.38 ± 0.00 | 35.96         | 0.21 us (no-op)     | dead steady           |

FINDINGS (these reshape the Phase 3 design):
1. Comms costs ~5-10% of DECODE even at just 2 nodes (1.38 -> 1.24). NOT 0.25% — the microbench
   was misleading. Direct A/B (comms on vs off) is the truth.
2. It's SYNCHRONIZATION WAIT, not bandwidth: a 32 KB all-reduce takes ~400 us in situ vs ~12 us on
   the wire (A2). Each node spins in ucp_worker_progress waiting for its PEER to reach the same
   all-reduce point. The tell: comms-ON decode is noisy (+/-0.11), comms-OFF is +/-0.00 — the
   jitter IS the cost. Even on identical, idle, balanced nodes the threadpools drift (OS noise,
   progress granularity) and with NO overlap every drift becomes a stall.
3. Scaling / the "~16-node limit" question, answered: the limiter as N grows is the BARRIER, not the
   link. Recursive-doubling makes each all-reduce log2(N) sequential exchanges, each waiting on the
   slowest of a growing group, while the compute it could overlap with SHRINKS as the FFN splits
   further. Sync-bound barriers are why ~16 nodes is where comms tips over.

DESIGN CONSEQUENCES:
- COMMS/COMPUTE OVERLAP is promoted from "advanced/later" to a TOP lever. Launch the all-reduce
  right after wo/ffn_down and only wait on it just before its result is consumed, so the next
  column-parallel matmul (needs no comms) runs underneath it. Recovers most of the ~10% now and is
  what keeps comms from dominating at scale. (Needs non-blocking UCX progressed off the compute
  thread, or a deferred-wait op in the graph.)
- rank=socket NUMA-local still matters independently: it lifts the COMPUTE ceiling (the 1.38).
- Algorithm choice (recursive-doubling vs ring) matters LESS than minimizing barrier count + overlap.
Instrumentation lives in src/llama-tp-net.c (g_time/g_nocomms, tp_print_stats); env LLAMA_TP_TIME=1,
LLAMA_TP_NOCOMMS=1.

## PHASE 3a+3b — N-way all-reduce + rank=socket (NUMA-local) — DONE + VALIDATED (2026-06-22)
Generalized the 2-node exchange-sum to N-way (power-of-2) RECURSIVE-DOUBLING all-reduce over UCX
(src/llama-tp-net.c rewrite): at step s a rank exchanges+sums with partner rank^(1<<s); after
log2(N) steps all hold the global sum. Distinct tag per step (TP_TAG+s) so a fast rank's step-(s+1)
message can't be grabbed by a slow rank's step-s recv. Bootstrap: rank 0 is a TCP coordinator on the
IB subnet that gathers every rank's UCX worker address and broadcasts the full list; each rank then
opens eps to its log2(N) partners. Rank LAYOUT is chosen by the launcher so step 0 pairs are
INTRA-node (UCX auto-selects sm) and later steps inter-node (rc) -> hierarchy for free, no special-
casing in code. Requires size power-of-2 in [2,16].

rank=SOCKET: run ONE process per socket (numactl --cpunodebind=N --membind=N -t 24) so each shard's
weights are NUMA-LOCAL. On the 2-node pair this is a 4-way TP (rank0=121/s0 [coordinator],
rank1=121/s1, rank2=124/s0, rank3=124/s1), all LLAMA_TP_SIZE=4 LLAMA_TP_PEER=172.16.0.121.

CORRECTNESS (3B, llama-simple): 4-way output TOKEN-FOR-TOKEN identical to single-node. Log:
"attention sharded — local n_head=6 n_head_kv=2 (tp_size=4)" (24/4, 8/4), "transport up (rank 0/4,
2 steps)". All 4 ranks ran exactly 21120 all-reduces (lockstep).

PERF (Llama-3.3-70B-Q8_0, llama-bench -mmp 0 -p128 -n64 -r2):

| config                              | decode tg64 | prefill pp128 | RSS/rank | total threads |
|-------------------------------------|-------------|---------------|----------|---------------|
| single-node                         | 0.86        | 19.22         | 71.7 GB  | 48            |
| 2-way rank=node                     | 1.32        | 34.83         | 37.0 GB  | 2x48 = 96     |
| 4-way rank=socket, -t 48 (BAD)      | 2.94        | 26.63         | 19.7 GB  | 4x48 = 192 (oversub) |
| **4-way rank=socket, -t 24**        | **3.44**    | **33.35**     | 19.7 GB  | 4x24 = 96     |

*** HEADLINE: rank=socket decode = 3.44 t/s = 4.0x single-node, 2.6x over 2-way rank=node. ***
THE KEY INSIGHT: 2-way (2x48) and 4-way (4x24) use the SAME 96 total threads — so the 2.6x decode
gain is PURELY NUMA-LOCALITY, not more cores. Confirms decode was UPI/NUMA-bound: a single 48-thread
process strdes weights across both NUMA nodes (~60 GB/s, 33% util, A1); rank=socket reads each 1/4
shard from LOCAL memory at full per-socket BW. This is the decode lever we'd been missing.
- GOTCHA: numactl restricts the cpuset but llama-bench's default -t still saw 48 -> 2 ranks/node x 48
  = 2x OVERSUBSCRIPTION, which clobbered compute-bound prefill (34.8->26.6) and capped decode (2.94).
  Setting -t 24 (= cores per socket) fixed BOTH: decode 2.94->3.44, prefill 26.6->33.4. ALWAYS pass
  -t = cores-per-socket for rank=socket.
- Memory: 19.7 GB/RANK (1/4 model + replicated embed/output), ~39 GB/node (replicated parts now 2x
  per node). Per-node ~unchanged vs 2-way; the win here is DECODE SPEED + the path to scale.
- Comms grew to ~18% of decode at 4-way (2 recursive-doubling steps + faster/smaller compute to hide
  it behind). avg all-reduce ~320 us/call (step0 sm + step1 rc). Comms-free ceiling ~4.2 t/s ->
  comms/compute OVERLAP (Phase 3c) is the clear next lever, and is what keeps comms from dominating
  as N grows toward the ~16-node barrier limit.
Run recipe: per-rank `numactl --cpunodebind=$SOCK --membind=$SOCK llama-bench ... -t 24 -mmp 0`,
env `LLAMA_TP_SIZE=4 LLAMA_TP_RANK=$r LLAMA_TP_ATTN=1 LLAMA_TP_PEER=172.16.0.121 LLAMA_TP_PORT=P`,
`UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc,sm,self`. rank0 (coordinator) on 121/socket0.
NOTE: this is 4 NUMA domains across 2 physical IB boxes (rank=socket), NOT 4 physical nodes — true
N-physical-node scaling awaits more IB hosts (node 92 IB still down). The N-way transport is proven.

## PHASE 3c — comms/compute overlap: INVESTIGATED, NOT VIABLE for decode (2026-06-22)
Goal was to hide the ~18% decode comms behind compute. Conclusion after measurement: NOT possible at
batch=1 decode; the comms is a SKEW-BOUND hard sync point, not a hideable transfer. Evidence:
- Dependency chain per layer is tight: norm -> wq/wk/wv(no comm) -> attn -> wo -> AR#1 -> +residual
  -> ffn_norm -> ffn_up/gate(no comm) -> ffn_down -> AR#2 -> +residual -> next layer. BOTH all-reduce
  outputs feed the residual immediately; everything after depends on it. At 1 token there is no
  sequence dimension to tile (the Megatron SP/overlap trick), so there is NO independent compute to
  overlap the reduce with.
- Force-eager (UCX_RNDV_THRESH=inf): 320 -> 295 us/call, decode unchanged (3.44->3.45). NOT rendezvous.
- Single-node 2-socket, intra-node sm ONLY, NO IB, 1 step: all-reduce avg = 171 us/call for a ~2 us
  shared-mem transfer. So ~170 us/step is pure ARRIVAL SKEW + progress spin, transport-independent:
  the two ranks don't finish their 24-thread matmul at the same instant, so every barrier waits for
  the laggard. (1 step intra ~170 us; 2-step intra+IB ~300 us => the rc step adds ~130 us.)
- Three-way kill: (a) no independent compute to overlap (above); (b) a dedicated UCX progress thread
  can't hide SKEW (the compute thread must still wait for the peer's data); (c) tiling the output to
  pipeline reduces is COUNTERPRODUCTIVE when skew-bound -- more, smaller messages = more barriers =
  more total skew.
The ~18% decode comms is therefore a hard floor for this design. Reducing it would require attacking
ARRIVAL SKEW itself (threadpool/OS jitter, NUMA contention between the 2 ranks sharing a node) -- a
deep, uncertain effort -- or a fundamentally different comms structure. Decode overlap is shelved.
(Overlap IS available for PREFILL -- 128 tokens are tileable -- but prefill is already ~33 t/s and
comms is a smaller fraction there; low priority.)

### Full decode picture (Llama-3.3-70B-Q8_0), the NUMA-locality story:
| config                                | total cores | decode t/s | vs single | note                    |
|---------------------------------------|-------------|------------|-----------|-------------------------|
| single-node, 1 proc, 48t (UPI-striped)| 48          | 0.86       | 1.00x     | ~60 GB/s, 33% BW util   |
| single-node, 2 sockets (sm, no IB)    | 48          | 2.20       | 2.56x     | NUMA-local, SAME cores  |
| 2 nodes rank=node, 2x48t              | 96          | 1.32       | 1.53x     | UPI-striped per node    |
| 2 nodes rank=socket (4-way)           | 96          | 3.44       | 4.00x     | NUMA-local + 2nd node   |
KEY: single-node 2.56x at the SAME 48 cores is pure NUMA locality (local vs UPI-striped weight reads).
rank=socket is the dominant decode lever; comms (~18% at 4-way) is the residual skew-bound floor.

## CROSS-ARCH: rank=socket NUMA-local win confirmed on Cascade Lake (2026-06-22/23)
Built CSL-native (GGML_NATIVE -> -march=cascadelake, VNNI) on node 92 (tonto92, 8260, 2x24c HT-on,
2x EDR rails mlx5_0/172.16.0.92 + mlx5_1/172.16.1.92). Single-node 2-socket TP via UCX sm only
(localhost bootstrap PEER=127.0.0.1, UCX_TLS=sm,self -- no IB needed). 70B Q8_0, -mmp 0, -p128 -n64.

| 70B Q8_0, single node       | single-proc 48t decode | 2-socket rank=socket decode | gain  |
|-----------------------------|------------------------|-----------------------------|-------|
| Skylake 8168 (no VNNI)      | 0.86                   | 2.20                        | 2.56x |
| Cascade Lake 8260 (VNNI)    | 0.82                   | 2.03                        | 2.48x |
The NUMA-local rank=socket decode win is ARCHITECTURAL (NUMA topology), ~2.5x on BOTH chips -- not a
Skylake quirk. CSL slightly slower absolute (Q8 is BW-bound; 8260 lower clock/BW than 8168). CSL
prefill ~flat 2-socket vs single (compute-bound, same 48 cores). CSL intra-node sm all-reduce ~130us
(< Skylake 171us). VNNI doesn't show on Q8 (BW-bound) -- would matter on the Q4_K compute-bound path.
HW note: node 92 has TWO active EDR rails and ib0=172.16.0.92 is on the SAME subnet as 121 -> a true
3-physical-node test (121+124+92) is now possible (92 is CSL though; mixed-arch TP is correctness-OK,
perf-limited by the slowest). Build gotcha: 92 has a SEPARATE /nix/store (not NFS-shared w/ 121);
must `nix build nixpkgs#ucx ucx.dev` (or include in the nix shell) to realize UCX before building.

## MoE EP debugging — mul_mat_id + sliced expert tensor is NOT buggy (2026-06-23)
Suspected ggml_mul_mat_id gave wrong output for token>=1 when src0 is a sharded (ne[2]-sliced)
expert tensor. Wrote a standalone repro (hpc/mul_mat_id_repro.c, no llama/TP code): build a full
[K,N,256] expert tensor + a [K,N,128] slice of its first 128 experts, run mul_mat_id on both with
identical src1 + ids (low experts), compare per token. RESULT: full == slice BIT-EXACT (maxdiff=0)
for F32, Q8_0, Q4_K; for both the up/gate pattern (src1 ne1=1) AND the down pattern (src1 ne1=
n_used); single- and multi-threaded. => slicing ne[2] is fully supported; NOT an upstream bug.
The real MoE-EP multi-token bug (token>=1 wrong in prefill) is therefore in OUR integration
(graph wiring / custom ops / scheduler interaction), NOT ggml. Every component verified correct in
isolation (shard load per-rank base_off, sel_mm local ids, weight mask complementary, all-reduce
combine mine+peer=sum, mul_mat_id) yet the composite fails — points to a subtle graph/scheduler
issue. NEXT: pivot to a custom local-expert op (replaces remap+mask+3x mul_mat_id) which sidesteps
the suspect chain and is the faster design anyway. Repro kept at hpc/mul_mat_id_repro.c.
