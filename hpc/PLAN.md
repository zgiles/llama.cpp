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

## PHASE 1 — Large-model perf baseline (DO FIRST; no new code)
The 3B was correctness-only. Run a real model big enough that compute/memory dominate.
- Dense 70B from /kimie: Llama-3.3-70B-Instruct-ablated-Q8_0 (split 00001/00002) or
  DeepSeek-R1-Distill-Llama-70B-Q8_0. Single-node (non-TP, mmap) decode t/s + RSS  vs
  2-node TP (--no-mmap) decode t/s + per-node RSS.
- Expectation: FFN-only halves FFN compute+memory/node; attention replicated. Modest decode
  speedup, ~halved FFN memory/node. Also measures all-reduce cost at hidden=8192 activations.
- Gotcha: split GGUF — point at the 00001 file; both parts must be on /kimie (they are).
- Decision: this tells us how much attention sharding matters before investing in it.

## PHASE 2 — Attention sharding (biggest remaining compute + KV-memory win)
- tp_role_for_tensor (llama-model-loader.cpp): add ATTN_Q/K/V -> COLUMN, ATTN_OUT -> ROW.
- HARD subtlety: heads. Column-split wq/wk/wv = split heads -> each rank gets n_head/tp query
  heads, n_head_kv/tp KV heads (GQA). The graph (build_attn, llama-graph.cpp ~2226) reshapes
  Q/K/V using hparams.n_head / n_head_kv and allocates the KV cache with n_head_kv. With sharded
  weights those counts must be the LOCAL (per-rank) values. Investigate whether build_attn /
  KV-cache derive head counts from the tensor ne or from hparams; likely need a per-rank
  "local n_head/n_head_kv" override (in hparams or a TP context) threaded through build_attn
  and the KV cache. This is the real work of this phase.
- RoPE is per-head (fine on local heads). wo = ROW-parallel + all-reduce (reuse the ffn_down op).
- KV cache becomes sharded -> per-node KV memory win (good for long context).
- Require n_head and n_head_kv divisible by tp_size; refuse/fall back otherwise.
- Validate: coherent token-for-token vs single-node.

## PHASE 3 — N>2 nodes (recursive-doubling all-reduce)
- src/llama-tp-net.c is currently a 2-node exchange-sum. Generalize to N:
  recursive-doubling (log2 N steps, power-of-2) or ring-allreduce. Need N-1 endpoints +
  a bootstrap rendezvous (rank 0 coordinator, or all-pairs TCP address exchange).
- Sharding math already general (divides by tp_size). Only the transport + bootstrap change.
- Validate 4-node coherence; measure all-reduce latency vs node count (compare to A2 prediction:
  log2(N) hops; SHARP would flatten it).

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
