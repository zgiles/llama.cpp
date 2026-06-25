# Sharding Mamba-2 / SSM layers for CPU tensor parallelism (design)

> **Status: implemented + validated correct (2026-06-25)** on Falcon-H1-0.5B (2-rank, single node, sm).
> Enable with `LLAMA_TP_SSM=1` / `--tp-ssm`. Output coherent + correct vs single-process (small token
> divergence after a few steps is expected: the per-layer all-reduce sums partials in a different float
> order, amplified slightly by the recurrent state — not a bug). Log line:
> `SSM sharded — local n_head=12 d_inner=768 n_group=1 (groups replicated, tp_size=2)`.
> Validated path: **head + d_inner sharding with replicated groups** (Falcon-H1 has `n_group=1` and no
> `ssm_norm`). The **group-sharded path** (`n_group % size == 0`, e.g. Nemotron-H `n_group=8`, with the
> grouped `ssm_norm` split on `ne[1]`) is implemented but **not yet tested** — no small model with
> `n_group>=2` was available; validate on Nemotron-3 next.
> Bug found + fixed during bring-up: `tp_shard_plan_make` left `n_seg` uninitialized, so the generic
> COLUMN/ROW path's plans carried stack garbage and `tp_read_shard` mis-took them for SSM segment gathers
> — corrupting every FFN/ssm_out shard. Always zero-init `n_seg` for non-segment plans.


Target: make the *hybrid* models (Nemotron-3 `nemotron_h_moe`, Granite-hybrid, Falcon-H1, and any
`mamba2`) faster under CPU TP by sharding their recurrent SSM mixer — today it is replicated on every
rank and dominates single-stream decode, so TP gives them ~nothing. Dev model: **Falcon-H1-0.5B**
(uses the exact `build_mamba2_layer` path Nemotron-3 uses; tiny → fast iterate).

## The key insight: Mamba-2 shards like an FFN

A Mamba-2 mixer is structurally an FFN with a per-channel recurrence in the middle:

```
ssm_in {n_embd, d_in_proj}  →  [ z | x | B | C | dt ]          (like ffn_up: COLUMN, expands to d_inner)
   conv1d(x,B,C) → selective scan over heads                    (the per-channel "activation")
ssm_out {d_inner, n_embd}                                       (like ffn_down: ROW + all-reduce)
```

`d_inner` plays the role of `n_ff`. The selective scan evolves **each SSM head's state independently**
— there is *no cross-channel mixing inside the scan* — so if we give each rank a contiguous subset of
heads, each rank scans its own heads with its own slice of state and needs **no communication until
`ssm_out`**, exactly like a sharded FFN. One all-reduce per layer after `ssm_out`.

### Why the state cache shards "for free"

The recurrent state cache dims come from hparams:
* `n_embd_r()` (rolling conv buffer) = `(d_conv-1) * (d_inner + 2*n_group*d_state)`
* `n_embd_s()` (SSM state)            = `d_state * d_inner`

(`src/llama-hparams.cpp` ~183-222). The cache (`llama_memory_recurrent`) and the graph build
(`build_mamba2_layer`, `src/models/mamba-base.cpp:149`) **read `d_inner`/`n_group`/`n_head` straight
from hparams**. So if we divide those hparams by the rank count *at load* (the same lever used for MLA
`n_head`), then:
* the state cache is allocated rank-local automatically (no cache surgery),
* the scan / conv ops get rank-local shapes automatically,
* `ssm_norm`'s grouped reshape stays consistent.

Divide at load (require divisibility + group alignment):
* `ssm_dt_rank` (= n_head)  /= R     — the SSM heads
* `ssm_d_inner`            /= R       — `= head_dim * n_head`, so head_dim is preserved
* `ssm_n_group`            /= R       — shard on whole-group boundaries

Constraints: `n_head % R == 0`, `n_group % R == 0`, and the heads-per-group `n_head/n_group` must stay
integral (it does: both scale by R). `ssm_out` ROW split must keep `d_inner/R` quant-block aligned.
Heads map to groups, so splitting `n_group` into R parts assigns `n_head/R` heads to each rank
consistently with its groups.

## Loader: the one hard part — the concatenated projections

Most ssm tensors split trivially once hparams are halved, but **`ssm_in` and `ssm_conv1d` are
concatenations of differently-shaped sub-slices**, so a plain even column-cut would slice across the
`z|x|B|C|dt` seams. The loader needs a *multi-segment column gather*: for each segment, take this rank's
contiguous sub-range and pack the pieces into the rank-local tensor. Splitting `ne[1]` (columns) is
always quant-safe (each column is a whole quantized row along `ne[0]`).

`ssm_in {n_embd, d_in_proj}`, `d_in_proj = 2*d_inner + 2*n_group*d_state + n_head`, laid out as:

| segment | full width | per-rank slice (rank r of R) |
|---|---|---|
| `z`  | `d_inner`          | `[r·d_inner/R, (r+1)·d_inner/R)` |
| `x`  | `d_inner`          | same offset within the `x` block |
| `B`  | `n_group·d_state`  | `[r·(n_group/R)·d_state, …)` |
| `C`  | `n_group·d_state`  | same within the `C` block |
| `dt` | `n_head`           | `[r·n_head/R, …)` |

`ssm_conv1d {d_conv, d_inner + 2*n_group*d_state}` and its bias = the `[x|B|C]` segments only (same
per-rank slices as above for x/B/C). Depthwise → each channel independent, so the column gather is exact.

### Per-tensor shard plan

| tensor | shape | role |
|---|---|---|
| `ssm_in`      | `{n_embd, d_in_proj}`              | **SSM_PROJ** 5-segment column gather (z,x,B,C,dt) |
| `ssm_conv1d`  | `{d_conv, d_inner+2·n_group·d_state}` | **SSM_CONV** 3-segment column gather (x,B,C) |
| `ssm_conv1d_b`| `{d_inner+2·n_group·d_state}`      | SSM_CONV 3-segment gather (1-D) |
| `ssm_dt_b`    | `{n_head}`                         | split `n_head` (COLUMN on ne[0], 1-D) |
| `ssm_a`       | `{1, n_head}`                      | split ne[1] = n_head |
| `ssm_d`       | `{1, n_head}`                      | split ne[1] = n_head |
| `ssm_norm`    | `{d_inner/n_group, n_group}`       | split ne[1] = n_group (ne[0] unchanged) |
| `ssm_out`     | `{d_inner, n_embd}`                | ROW split ne[0] = d_inner + all-reduce |

Replicated (no role): any pre/post norms on `n_embd`, the attention tensors (handled by MLA/standard
attn sharding separately), the MoE/FFN tensors (their own roles).

## Graph

`build_mamba2_layer` needs no structural change (it reads rank-local hparams), **except** add the
row-parallel all-reduce after the `ssm_out` projection (`src/models/mamba-base.cpp:281`), gated by a new
`llama_tp_ssm_enabled()` (env `LLAMA_TP_SSM=1` + a `--tp-ssm` flag, mirroring `LLAMA_TP_ATTN`). Keep it a
separate flag from attention so a model can shard SSM without attention and vice-versa.

## Order of work

1. hparams division at load (guards: n_head%R, n_group%R, ssm_out quant alignment) — the single lever.
2. Loader: the SSM_PROJ / SSM_CONV multi-segment gather plans + the trivial splits.
3. `ssm_out` all-reduce + the `LLAMA_TP_SSM` flag/CLI.
4. Validate token-for-token on Falcon-H1-0.5B (2-rank single-node sm) vs single-node.
5. Perf on Nemotron-3 (where the SSM dominates decode — the actual payoff).

## Gated-delta-net (Qwen3.5/3.6) — separate, later

Qwen3-Next / Qwen3.6 use **gated delta-net** (`src/models/qwen3next.cpp` `build_layer_attn_linear`,
a `build_recurrent_attn` custom op + `ssm_beta_alpha`), not `ggml_ssm_scan`. Same channel-parallel
principle (split value heads + state), but different projections and a custom recurrent op, so it is a
second implementation. Do standard Mamba-2 first; revisit gated-delta-net after.
