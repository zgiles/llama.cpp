# Sharding the replicated compute: attention, shared-expert, Mamba-2 (design)

Today CPU TP shards only the **MoE-FFN** experts. Everything else — attention, the shared expert, and
(for hybrids) the Mamba-2 layers — is **replicated** on every rank. At batch=1 that replicated compute
dominates, which is why tensor mode gave only +27% decode on DeepSeek (and nothing on Nemotron). To make
single-stream decode meaningfully faster we must shard the rest of the per-token compute too.

This is a survey of what each piece needs, ordered by effort.

## Infrastructure that already exists

* **Row-parallel all-reduce after `wo`.** `build_attn` for `llm_graph_input_attn_kv` (the standard KV-cache
  path, llama-graph.cpp ~2431) already does `if (wo && llama_tp_attn_enabled()) all_reduce`. Standard
  MHA/GQA models therefore already shard attention via `LLAMA_TP_ATTN` (`wq/wk/wv` column, `wo` row, heads
  split). Limit: the load-time guard (llama.cpp ~352) **refuses** fused-QKV, attention bias, **and any
  model with `wq_b/wk_b/wv_b`** — which is exactly MLA.
* **COLUMN/ROW shard plans** generalized to 3D tensors (used by tensor-mode experts) — reusable for the
  per-head 3D MLA tensors and for the shared-expert FFN.

## 1. Shared expert (shexp) — EASY, do first

DeepSeek (and others) compute a dense FFN every token from `ffn_{gate,up,down}_shexp`
(`LLM_TENSOR_FFN_{GATE,UP,DOWN}_SHEXP`). It is a plain gated FFN built with `build_ffn`, currently **not**
sharded.

* **Loader:** add the SHEXP tensor enums to `tp_role_for_tensor` → `gate/up` COLUMN, `down` ROW (identical
  to the dense FFN that already shards).
* **Graph:** `build_ffn`'s all-reduce is gated `... && !llama_tp_moe_enabled()` (so shexp stays whole in
  MoE mode). Change it to all-reduce the shexp output when MoE-TP is on. Activation/glue unchanged.
* **Effort:** small (reuses dense FFN sharding). **Payoff:** modest but free — the shexp runs on every
  token; on DeepSeek it is `n_ff_exp × n_expert_shared` wide.

## 2. MLA attention (DeepSeek) — DONE (2026-06-24), MODERATE, high value

> **Status: implemented + validated correct** on DeepSeek-V3.1-Q4 (2-rank, single node). Coherent output,
> log line `attention sharded (MLA) — local n_head=64 n_head_kv=1 (tp_size=2)`. Enable with
> `LLAMA_TP_ATTN=1` (alongside `LLAMA_TP_MOE=tp`). The crucial fact confirmed at load: DeepSeek MLA reports
> **`n_head_kv == 1`** (the absorbed-MQA latent), so only the **query heads** are sharded (`n_head` divided);
> `n_head_kv` and the latent KV cache are left exactly as single-node (replicated on every rank). Roles:
> `wq_b` COLUMN, `wk_b`/`wv_b` per-head `ne[2]=n_head` (EXPERT-style) split, `wo` ROW + all-reduce (added to
> the `attn_k` overload); `wq_a`/`wkv_a_mqa`/`*_a_norm` replicated. The plan below is what was built.
>
> **Perf (DeepSeek-V3.1-Q4, 2-rank single-node, `-p128 -n32 -t24 -mmp0`):** tensor-only `pp 15.31 / tg 3.25`
> → tensor+MLA-attn `pp 19.16 / tg 3.68` = **+25% prefill, +13% decode**, on top of tensor-mode MoE. Unlike
> the shared-expert experiment (§1, net loss because it is too small), MLA attention is a large enough
> replicated component that head-splitting it pays off.



MLA factors Q and KV through low-rank latents: `wq_a→q_a_norm→wq_b`, `wkv_a_mqa→kv_a_norm→wk_b/wv_b`, then
`wo`. The heads (`n_head`) still exist; the per-head tensors are `wq_b {q_lora, n_head·head_k}`,
`wk_b/wv_b {.., .., n_head}` (3D, per-head), `wo {n_head·head_v, n_embd}`.

* **Shard by head** (like standard attention): `wq_b` COLUMN (its output is `n_head·head_k`), `wk_b/wv_b`
  split `ne[2]=n_head`, `wo` ROW. The **latent compressions** `wq_a / wkv_a_mqa / q_a_norm / kv_a_norm`
  are shared across heads → **replicated** (small: `q_lora`/`kv_lora` ranks). The existing per-layer
  `n_head` division already produces this rank's head count.
* **Graph:** MLA uses the `llm_graph_input_attn_k` overload (llama-graph.cpp ~2472), which does **not**
  currently have the `wo` all-reduce — add the same one-liner there (and the `attn_k_dsa` / `kv_iswa`
  overloads for completeness).
* **Load guard:** the current guard refuses any model with `wq_b/wk_b/wv_b`. Replace "refuse" with "shard
  the MLA tensors" for MLA models.
* **Effort:** moderate (head/dim bookkeeping for the 3D `wk_b/wv_b`, the `head_k = nope+rope` split must
  stay intact per head, RoPE on the rope part). **Payoff:** attention is a real fraction of DeepSeek
  decode; combined with FFN tensor mode this is the path toward ~2× on the flagship MoE.

## 3. Mamba-2 SSM (Nemotron hybrid) — HARD, the frontier

Nemotron's recurrent layers (`is_recr` when `n_head_kv==0 && n_ff==0`) are Mamba-2 mixers:
`ssm_in {n_embd, d_in_proj}` with `d_in_proj = 2·d_inner + 2·n_group·d_state + n_ssm_head`, a depthwise
`ssm_conv1d`, per-head `ssm_a/ssm_dt`, the selective-scan, then `ssm_out {d_inner, n_embd}`.

* **Channel/head-parallel:** split `d_inner` (and the `n_ssm_head` heads) across ranks, on **`n_group`
  boundaries** so each rank owns whole state groups. `ssm_in` COLUMN (its `d_inner`+conv+head slices),
  `ssm_conv1d` depthwise-split, the scan runs **independently per channel** on each rank, `ssm_out` ROW +
  all-reduce. `ssm_a/ssm_dt_b/ssm_d` split per head.
* **Hard parts:** (a) the **recurrent state cache** must be sharded so each rank stores only its channels'
  state (it persists across tokens — the cache allocation + the `build_rs` path need rank-local sizes);
  (b) the selective-scan custom op and conv must operate on the sharded channel range; (c) `d_in_proj` is
  a concatenation of several differently-shaped slices, so the COLUMN split is **not** a simple even cut —
  it must split each sub-slice (`z`, `x`, `B`, `C`, `dt`) consistently, like a merged gate_up.
* **Effort:** high (new state-cache sharding + scan/conv slicing). **Payoff:** Mamba dominates Nemotron's
  compute, so it is the only lever that makes that class faster — but it's the biggest piece.

## Recommended order

1. **shexp** — cheap, reuses dense FFN sharding, helps every shared-expert MoE.
2. **MLA** — moderate, unlocks the DeepSeek "approach 2×" story (FFN-tensor + MLA + shexp all sharded).
3. **Mamba-2** — hard; needed for Nemotron-class hybrids, do last.

The general principle: *every* per-token matmul must shard for single-stream decode to scale. Dense models
get this for free (all weights active); MoE/hybrids need each component (experts, shexp, attention, SSM)
handled, and the attention/SSM type sets the difficulty.
