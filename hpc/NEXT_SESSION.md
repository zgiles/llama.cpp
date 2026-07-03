# Next session — TODO

Two independent tracks. Nodes must be up (tonto92 has the Intel B580 GPU; use lab-manager
skill / IPMI to power on 121/122/124 if multi-node needed — ~10min boot, tmpfs so rebuild).

## Track A — codebase cleanup (upstream/fork hygiene, ~30min, LOW risk)
Goal: make the GDN-TP feature clean/mergeable. Strip debug scaffolding, keep the feature.
- STRIP: `llama_tp_dump_op` (+decl) in llama-tp-net.{c,h}; the `LLAMA_TP_ATTN_DUMP` meta/dump
  inserts in llama-graph.cpp (build_attn_mha) and the per-layer `mix_*` dumps in
  models/qwen35.cpp, qwen35moe.cpp, qwen3next.cpp; the `GDN_DIM` print in llama.cpp;
  `LLAMA_NO_FUSED_GDN` in llama-context.cpp (fused GDN works correctly now).
- KEEP: the fix; `LLAMA_TP_GDN_DEBUG` shard-plan logging; the LLAMA_LOG_INFO "…sharded" lines.
- VERIFY (the part that needs nodes): compile-check, then ONE coherence run per arch on tonto92
  2-socket sm — qwen35moe / qwen35(dense) / qwen3next must stay coherent (token-identical on the
  haiku prompt for qwen35moe). Then update fork cpu-tp (strip trailer, no hpc/, author Zachary Giles).
- Risk: low (env-gated/additive removal); the coherence run is the guard against a fat-fingered line.

## Track B — GPU offload of attention (the real decode lever for our usage)
Context: CPU long-context decode is bottlenecked by the attention KV-scan (DDR4 bandwidth) — measured
qwen35moe tg 11.71 (d0) -> 6.25 (d32K) on Cascade 92, driven by the periodic full-attention layers.
The B580 has far more bandwidth -> put attention (+ its KV) on the GPU, keep MoE experts on the CPU
sockets (our EP/tensor TP). This is the ktransformers / `--n-cpu-moe` / `--override-tensor` pattern.
SINGLE-NODE play (tonto92: 1 B580 + 2 CPU sockets) — NOT multi-node (one GPU). For models that fit one
node with fast decode + long context.
Steps:
1. Build Vulkan on 92 (build-vk; Mesa ANV; VK_ICD_FILENAMES=.../intel_icd.x86_64.json,
   LD_LIBRARY_PATH=<vulkan-loader>/lib inside `nix shell vulkan-loader`). Use LOCAL/tmpfs weights +
   `-fit off` (LLAMA_ARG_FIT=off) + `-dio` — NFS + fit pass is pathological for GPU load (regression #19912).
2. Baseline: qwen35moe 35B or qwen3next 80B, CPU-only long-context decode (tg at d0/8K/32K) — already have.
3. GPU-attn + CPU-experts: --n-cpu-moe (experts on CPU) / --override-tensor to pin attn+dense to Vulkan,
   experts to CPU. Measure tg at the same depths — expect the long-context decode drop to flatten (GPU KV bw).
4. Then compose with CPU expert-TP (2 sockets) if it helps; measure.
Open Qs: does llama.cpp's --n-cpu-moe interplay with our TP env flags? (attn on GPU while experts sharded
across CPU sockets/ranks). May need a small override-tensor recipe. B580 hangs loading big MoE over NFS
(fit pass) — local weights mandatory. Q8_0 ~4x slower than Q4_K on Battlemage — stay Q4_K.

## Also-ran (lower priority, note only)
- Speculative decode (MTP self-draft) on the GDN hybrids — 2nd decode lever after GPU; amortizes the
  per-layer all-reduces across K verified tokens. qwen35moe/deepseek have MTP wired. Needs
  llama-speculative-simple or server + --spec-type draft-simple.
- The "fold all-reduces" idea is DEAD — per-layer comms are sequential (mixer->norm->FFN), not foldable.

## Ready-to-paste prompt for the next session
> Resume the CPU-TP project. First do Track A (codebase cleanup) from hpc/NEXT_SESSION.md: strip the
> debug scaffolding, compile, and run ONE coherence check per GDN arch on tonto92 2-socket to confirm no
> regression, then refresh the fork. Then start Track B: build Vulkan on tonto92 and prototype
> GPU-offloaded attention (B580) + CPU experts (--n-cpu-moe/--override-tensor), measuring long-context
> decode (tg at d0/8K/32K) vs the CPU-only baseline to see if the GPU flattens the attention-driven decode
> drop. Use the lab-manager skill for power; tonto92 is usually already up. Keep changes opt-in and clean
> (upstream goal); no Co-Authored-By trailer.
