# Kimi-K3 attention sharding under CPU tensor parallelism (`--tp-attn`)

Implementation notes for `LLAMA_TP_ATTN=1` on the Kimi-K3 (`kimi-k3`) architecture. See
[cpu-tensor-parallel.md](../cpu-tensor-parallel.md) for the user-facing feature and results.

## Why K3 needs it

K3 is a 2.78 T hybrid: ~¾ of its layers are **KDA** (Kimi delta-net, a recurrent gated-delta-net token
mixer) and ~¼ are **MLA** (DeepSeek-style latent attention). Decode is ~⅘ *attention* — the trunk
weights are read every token, while only 16/896 experts fire. With attention replicated on every rank,
that dominant cost does not shard, so decode does not scale (it regresses 2→4 ranks as expert-parallel
imbalance grows). Sharding the attention heads is the lever that makes K3 decode scale.

## The mechanism it rides

`--tp-attn` already divides `hparams.n_head_arr[il] /= tp_size` for all layers at load time
(`src/llama.cpp`, after `load_tensors`). Two consequences make K3 almost free:

- `build_kda_layer` / `build_mla_layer` read `hparams.n_head()` for their reshapes → the graph becomes
  per-rank automatically.
- K3's recurrent state is sized from `n_head()`: `n_embd_r() = 3·(d_conv−1)·n_head·head_dim` and
  `n_embd_s() = head_dim²·n_head` (`src/llama-hparams.cpp`). So the conv/delta-net state cache shards
  **automatically** — no separate SSM-dim division is needed (unlike the Mamba/GDN path).

So dividing `n_head` shards the graph, the MLA path, and the recurrent state. Only two things were
missing: the loader sharding K3's KDA *gate* tensors, and the `wo` all-reduces.

## What was added

1. **`tp_kda_plan_for`** (`src/llama-model-loader.cpp`) — head-shards K3's KDA gate tensors. K3's KDA is
   R=1 (no GQA), so every split is a single contiguous whole-head slice:

   | tensor | shape | split |
   |---|---|---|
   | `ssm_beta` | `{n_embd, n_head}` | COLUMN (`ne1` = heads) |
   | `ssm_f_b`, `ssm_g` | `{·, d_inner}` | COLUMN (`ne1` = `d_inner`) |
   | `ssm_conv1d_{q,k,v}` | `{d_conv, 1, d_inner}` | EXPERT (`ne2` depthwise channels) |
   | `ssm_dt` (bias) | `{d_inner}` | ROW (`ne0`, 1-D) |
   | `ssm_a` | `{n_head}` or `{1,n_head}` | ROW or COLUMN (shape-branch on `ne1`) |
   | `ssm_f_a`, `ssm_o_norm` | per-head-dim | REPLICATED |

   The head-carrying projections `wq`/`wk`/`wv` (`ATTN_Q/K/V`) and `wo` (`ATTN_OUT`) shard via their
   existing attention roles. `tp_kda_plan_for` is gated on `LLM_ARCH_KIMI_K3` and fires from a
   **new `attn`-gated load-site block** (not the `ssm`-gated one — K3 uses `--tp-attn`, not `--tp-ssm`).

2. **Two roles** in `tp_role_for_tensor`: `ATTN_KV_B` → COLUMN (fused MLA kv_b, when `wk_b`/`wv_b` are
   absent — without this it loads full-width and produces silently wrong output on rank > 0), and a
   K3-gated `ATTN_GATE` → COLUMN (the MLA output gate; arch-gated because `ATTN_GATE` is the GDN z-gate
   in Qwen3-Next/3.5 and a full-attention gate elsewhere).

3. **`wo` all-reduce** in `build_kda_layer` and `build_mla_layer` (`src/models/kimi-k3.cpp`). Both apply
   `wo` via a manual `ggml_mul_mat` (KDA has no `build_attn`; MLA multiplies the sigmoid gate in before
   `wo`, so it passes `wo=nullptr` to `build_attn`, bypassing its post-`wo` all-reduce). `wo` is
   row-parallel, so each rank's output is a partial sum over its local heads → one
   `llama_tp_allreduce_op` per attention layer recombines to full `n_embd`. The payload is
   `n_embd·4 B` (~28 KB) per layer — negligible against the per-token compute.

## Correctness & divisibility

- Sharded attention is **byte-equivalent** to the replicated path (the all-reduce sums the per-head
  contributions). Verified on-node: identical greedy output with and without `LLAMA_TP_ATTN=1`.
- `n_head = 96` divides by 2 and 4; `d_inner = head_dim·n_head = 12288` likewise. Any tensor that can't
  split fails **loudly** (`tp_refused` → throw) rather than loading full-width.
- Run with `LLAMA_TP_ATTN=1 LLAMA_TP_MOE=expert` (see the runbook in the user doc).
