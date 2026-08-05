# CPU tensor-parallel — long-term follow-ups (TODO)

Parked items from the K3 8-way + DeepSeek-V4-Flash TP work (2026-08). None block current use —
everything we actually run (Kimi-K3, DeepSeek-V4-Flash, Kimi-K2, GLM, Qwen3-MoE) works and is correct.
These matter only if we (a) upstream the CPU-TP changes, (b) point CPU-TP at other architectures, or
(c) want to push decode past the current bandwidth ceiling.

## Correctness/robustness (dense-FFN replicate change) — non-K3 archs only
The dense-FFN replicate fallback (commit 0dd7b652f) is correct for every model we run (their dense
`ffn_down` width `== n_ff`, so the graph's `down->ne[0] == n_ff(il)` replication check holds). It has
latent issues for architectures we do NOT currently CPU-TP:

1. **Silent wrong output** on archs whose dense `ffn_down` width `!= n_ff` — Arctic (dense MLP
   `n_embd×n_embd`), old Qwen-1 (`n_ff/2`): the graph would all-reduce a replicated FFN → ×N garbage.
   This *regressed* a clean "cannot shard" refusal into silent garbage for those archs.
2. **Crash** on fused gate/up experts-less dense FFN (ChatGLM3-style): fused-up shards but down
   replicates → matmul dim mismatch abort.
3. **Over-replication (perf nit)** — the hardcoded `256` block replicates a dense FFN in a `block<256`
   quant (q4_0/q8_0/f16) even when it could shard; only 1–3 dense-lead layers so the cost is tiny.
   K3 is q4_K/block-256 so it's exact there.

**Cheapest fix if/when needed:** don't implement full per-arch handling — add a guard that makes the
replicate fallback **refuse loudly** (throw, as before) for the cases the graph heuristic can't verify,
restoring loud-failure instead of silent garbage (~30 min). Full fixes (propagate the loader's replicate
decision to `build_ffn` instead of the `n_ff(il)` heuristic; mirror the criterion into the fused-gate/up
path; key the block on `ffn_down`'s real type) are ~a day and only worth it for upstreaming.

## DeepSeek-V4-Flash `--tp-attn` — deferred (low ROI)
deepseek4's attention (Q-lora + O-lora + compressed latent KV + attn sinks + hyper-connections + DSA
lightning indexer) needs a K3-KDA-style per-tensor shard plan to shard by query head. It's a substantial
arch-specific job, and deepseek4 is **expert-bound, not attention-bound** (attention ~4% of the model),
so sharding attention buys almost nothing. Skip unless doing very-long-context (where sharding the KV
across ranks becomes a capacity play). Design sketch is in the `deepseek-v4-flash-model` memory note.

## Actually making decode faster (not more ranks)
Decode saturates ~4 ranks on this cluster for both models — cross-node TP is capacity, not speed. Real
levers, in rough order:
- **Shard the shared expert.** It's replicated on every rank and read every token → an Amdahl floor that
  doesn't shrink with rank count.
- **Speculative decode / MTP** — sidesteps the per-token barrier entirely (K3 MTP already gave +39%).
- **Big-VRAM GPU for the experts.** The bottleneck is CPU DDR bandwidth reading experts. A GPU only helps
  if the (quantized) experts fit VRAM — feasible for DeepSeek-V4-Flash (~75 GB @Q4 / ~40 GB @Q2, unlike
  K3's 930 GB), not for the small 12 GB B580. GPU-attention offload only helps *attention-bound* models
  like K3, not expert-bound ones like deepseek4.
