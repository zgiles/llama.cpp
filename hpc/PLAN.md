# CPU Tensor Parallelism for llama.cpp — Next-Phase Plan

Resume doc for continuing the CPU multi-node tensor-parallelism work after a context clear.
Read together with `hpc/RESULTS.md` (full findings) and the memory file
`hpc-cpu-tensor-parallel-project.md`.

## DONE (working + validated)
- Hierarchical all-reduce primitive: L1 in-process shmem + L2 UCX/IB (hpc/tp_*.{c,h}).
- **FFN-only, 2-way TP inference works + is correct in real llama.cpp.** Model sharded across
  2 nodes (each holds half the FFN), ffn_down all-reduced over IB. Coherent output.
- Files: src/llama-tp-shard.{c,h} (sharding math), src/llama-tp-net.{c,h} (UCX transport +
  ggml custom op), edits in llama-model-loader.{h,cpp}, llama-graph.cpp, src/CMakeLists.txt.

## RESUME ESSENTIALS
Nodes: Skylake pair 10.70.132.121 (rank0) + 10.70.132.124 (rank1), live EDR IB 172.16.0.121/.124.
Models load directly from /kimie/projects/ai/models (read-only NFS, DO NOT copy). Test model:
Llama-3.2-3B-Instruct-Q6_K_L.gguf (small, correctness only).

Build (fresh nixpkgs toolchain — flake devShell openssl is STALE on the NFS store):
```
UCXDEV=/nix/store/nirp33rmwgl30p494jh43lsnh4nf67k8-ucx-1.19.0-dev
UCX=/nix/store/2dszw6n4fqdvw0zqdph4i487w7is1k95-ucx-1.19.0
cd /home/zrg/llama.cpp && nix shell nixpkgs#cmake nixpkgs#ninja nixpkgs#gcc nixpkgs#pkg-config nixpkgs#openssl --command \
  bash -c "cmake -B build-native -G Ninja -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DTP_UCX_INC=$UCXDEV/include -DTP_UCX_LIB=$UCX/lib && \
    cmake --build build-native -j48 --target llama-bench llama-simple"
# copy bin/ to peer (rpath needs same path): on 121:
#   cd build-native && tar cf - bin | ssh 172.16.0.124 'cd /home/zrg/llama.cpp/build-native && tar xf -'
```
Run 2-node (rank1=server on 124, rank0=client on 121), both with
`UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc,sm,self`:
```
# 124:  LLAMA_TP_SIZE=2 LLAMA_TP_RANK=1 LLAMA_TP_PORT=13710 ... llama-simple -m M -n 24 "prompt"
# 121:  LLAMA_TP_SIZE=2 LLAMA_TP_RANK=0 LLAMA_TP_PEER=172.16.0.124 LLAMA_TP_PORT=13710 ... llama-simple ...
```
ALWAYS kill stray procs after runs: `pgrep -x llama-simple|llama-bench|... | xargs -r kill -9`.
Local repo changes are UNCOMMITTED (working tree) — consider committing before big changes.

## PHASE 1 — Large-model perf baseline — DONE (2026-06-22). See RESULTS.md "PHASE 1" section.
Llama-3.3-70B-Q8_0 (69.82 GiB), Skylake pair, llama-bench -p128 -n64 -mmp 0:
  single-node decode 0.86 t/s / RSS 71.7 GB  vs  2-node TP decode 1.24 t/s / RSS 43.2 GB/node.
  => decode 1.44x, prefill 1.61x, RAM/node -40% (0.60x, exactly matches FFN=80% halved theory).
Split GGUF + TP-shard combo VALIDATED (point at -00001). Comms is <0.3% of decode; the 1.44x-vs-
ideal-1.67x gap is REPLICATED ATTENTION, not the network.
GOTCHA: bench with `-mmp 0` (not `--no-mmap`); mmap-over-NFS made the first run noisy garbage.
DECISION: attention is now the dominant non-parallelized cost -> Phase 2 (attention sharding) is
the clear next lever. PROCEED to Phase 2.

## PHASE 2 — Attention sharding — DONE + VALIDATED (2026-06-22). See RESULTS.md "PHASE 2".
Head-parallel attention TP, gated by `LLAMA_TP_ATTN=1` (FFN-only stays default). wq/wk/wv COLUMN,
wo ROW + all-reduce. SOLVED the local-head-count subtlety with ONE lever: after load_tensors,
divide hparams.n_head_arr / n_head_kv_arr by tp_size (n_embd_head_k is stored independently, so the
KV cache, build_qkv and build_attn all pick up local head counts automatically; n_embd stays full).
build_attn_mha is already head-agnostic. Guards: divisibility + refuse fused-QKV / attn-bias.
VALIDATED token-for-token vs single-node (3B). 70B 2-node: decode 1.32 t/s (1.53x single, +6.5% vs
Phase 1), prefill 34.83 (1.81x), RSS 37.0 GB/node (0.52x single, -14% vs Phase 1). Decode gain is
modest because the FFN (already halved in P1) dominates batch=1 decode and the REPLICATED lm_head
(128k-vocab matmul/token) is a fixed floor; KV-cache halving is the long-context memory win.
Files: llama-tp-shard.{c,h}, llama-model-loader.cpp, llama-tp-net.{c,h}, llama-graph.cpp, llama.cpp.
NEXT-decode levers identified: shard lm_head (column + all-gather), N>2 nodes, comms/compute overlap.

## PHASE 3a+3b — N-way all-reduce + rank=socket — DONE + VALIDATED (2026-06-22). See RESULTS.md.
N-way recursive-doubling all-reduce over UCX (rank^(1<<s) partners, log2 N steps, per-step tag) +
rank-0 TCP coordinator bootstrap (gather/broadcast worker addrs). Power-of-2 size in [2,16]. Rank
layout makes step0 intra-node (sm) and later steps inter-node (rc) -> free hierarchy.
rank=SOCKET (numactl --cpunodebind/--membind, -t = cores/socket) makes each shard NUMA-LOCAL.
*** 4-way rank=socket (2 nodes x 2 sockets): decode 3.44 t/s = 4.0x single-node, 2.6x over 2-way ***
PURELY from NUMA-locality (same 96 total threads as 2-way; decode was UPI-bound). Token-for-token
correct (3B). GOTCHA: pass -t 24 (= cores/socket) or you get 2x oversubscription. Comms now ~18% of
decode (2 steps) -> Phase 3c (overlap) is the next lever. True N-physical-node awaits more IB hosts.

## PHASE 3c — comms/compute overlap (NEXT; the measured top lever)
- Decode comms is ~18% at 4-way and SYNC-bound (each rank spins for its peer). Comms-free ceiling
  ~4.2 t/s. Overlap recovers it AND keeps comms from dominating toward the ~16-node barrier limit.
- Split the all-reduce into LAUNCH (post sends/recvs, return) + WAIT (block before the consumer),
  placing independent compute (next column-parallel matmul) between them. Needs UCX progressed OFF
  the compute thread (dedicated progress thread) since single-thread mode + busy compute = no
  progress. Careful ggml op ordering so the overlapped op has no data dep on the reduce result.

## PHASE 4 — Perf tuning
- NUMA replication within node (the ~13% UPI penalty on Skylake; larger on bandwidth-bound
  arches like Cascade Lake). Replicate weights per socket or run socket-local thread groups.
- Comms/compute overlap: overlap the all-reduce with the NEXT layer's column-parallel matmul
  (needs no comms). The big single-stream lever at scale. Needs non-blocking UCX + careful ggml
  op ordering. Advanced.
- SHARP in-network reduce (Switch-IB 2 supports SHARPv1), fp16 activations (half the bytes),
  larger node counts.

## PARALLEL TRACK — MoE expert parallelism ("host trillion-param, fast")
- /kimie has DeepSeek-V3, Qwen3.6-35B-A3B-UD-Q4_K_M (3B active!), GLM-4.5/4.6, KimiK2.
- Shard EXPERTS across nodes; route tokens via all-to-all dispatch/combine. Different mechanic
  from dense TP. See build_moe_ffn (llama-graph.cpp ~1454) + ggml_mul_mat_id. Highest leverage
  for giant models; a larger separate effort. Qwen3.6-35B-A3B is the ideal first target.

## KNOWN GOTCHAS (don't relearn)
- Build with fresh nixpkgs toolchain, NOT `nix develop .#default` (stale NFS openssl).
- TP auto-disables mmap (row-parallel data is non-contiguous).
- THE bug class: there are TWO load paths — load_data_for AND load_all_data; the CPU path is
  load_all_data. Both must apply the shard plan (we fixed both).
- Cross-node MPI is dead on these nodes (shared hostname "nixos" -> PMIx deadlock); use UCX.
- Copy bin/ to peers at the same absolute path (rpath); kill stray procs after every run.
