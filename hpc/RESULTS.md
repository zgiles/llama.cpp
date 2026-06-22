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
