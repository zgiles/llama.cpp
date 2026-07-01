# Gated-delta-net (linear attention) tensor-parallel sharding — design

## What & why
Gated-delta-net (GDN) is the linear-attention recurrent mixer in **Qwen3-Next, Qwen3.5, Qwen3.6**
(and Kimi-Linear). Today our TP shards these hybrids' FFN only (attn/GDN replicated → ~+23% decode).
Sharding the GDN mixer is the last big architecture gap and makes the most-used non-Llama family
fully TP-capable.

GDN is **fully head-parallel**: state `s` is `[S_v, S_v, H_v, n_seqs]` (independent per-head matrices),
output is `[S_v, H_v, …]`, every op in `build_delta_net_*` is per-head. So shard the heads across ranks
exactly like attention / Mamba-2; the output projection is row-parallel + all-reduce.

## Dimensions (qwen3next `build_layer_attn_linear`)
- `head_k_dim = head_v_dim = ssm_d_state` (per-head dim, S) — **invariant under sharding**
- `n_k_heads = ssm_n_group` (H_k), `n_v_heads = ssm_dt_rank` (H_v), `H_v % H_k == 0` (GQA-style)
- `key_dim = S·H_k`, `value_dim = d_inner = S·H_v`, `conv_dim = 2·key_dim + value_dim`

## Sharding plan

### (A) Hparams division — ALREADY DONE by the `tp.ssm` lever
`llama.cpp:389 if (tp.ssm)` already divides `ssm_dt_rank`, `ssm_d_inner`, and (when divisible)
`ssm_n_group` by `tp.size`. Because `n_embd_s/n_embd_r` derive from these, the state+conv caches
auto-shard per rank; `head_v_dim = d_inner/n_v_heads` is preserved (both divided). The GDN build code
reads `n_v_heads/n_k_heads/d_inner` straight from hparams, so the graph builds per-rank automatically —
the same "one lever" as attention head division. **Nothing to add here.**

### (B) Weight sharding — TO ADD (loader, mirrors `tp_ssm_plan_for`)
| tensor | shape | shard |
|---|---|---|
| `wqkv` (ATTN_QKV) | `{n_embd, 2·key_dim + value_dim}` | 3-seg gather: q(key_dim)·by-k-head, k(key_dim)·by-k-head, v(value_dim)·by-v-head |
| `wqkv_gate` (ATTN_GATE) | `{n_embd, value_dim}` | COLUMN by v-head |
| `ssm_conv1d` | `{d_conv, conv_dim}` | seg gather by channel matching q\|k\|v (same 3 segments) |
| `ssm_dt` (bias) | `{n_v_heads}` | 1-D split by v-head |
| `ssm_a` (A_NOSCAN) | `{n_v_heads}` | 1-D split by v-head |
| `ssm_beta_alpha` | `{n_embd, 2·n_v_heads}` | 2-seg gather: beta(n_v_heads)·by-v-head, alpha(n_v_heads)·by-v-head |
| `ssm_norm` | `{head_v_dim}` | **REPLICATE** (within-head-dim, identical across heads) |
| `ssm_out` (SSM_OUT) | `{value_dim, n_embd}` | ROW split by value_dim (per-v-head) + **all-reduce** |
| `ssm_in` (legacy qkvz) | `{n_embd, 4·…}` | REFUSE legacy path initially (require the optimized `wqkv` GGUF) |

### (C) All-reduce — TO ADD (build)
Insert `llama_tp_allreduce_op` after `cur = build_lora_mm(ssm_out, final_output)` in the GDN build
functions: `qwen3next.cpp`, `qwen35.cpp`, `qwen35moe.cpp`, `kimi-linear.cpp` (guard on
`llama_tp_ssm_enabled()`, as Mamba-2 does in `mamba-base.cpp`).

## Reuse ledger
- hparams lever: **done** (tp.ssm).
- segment-gather read path (`tp_read_shard`, `n_seg`): **reuse**.
- all-reduce op (`llama_tp_allreduce_op`): **reuse**.
- Net-new ≈ GDN tensor shard-plan (~150 LOC, patterned on `tp_ssm_plan_for`) + 4 one-line all-reduce insertions.

## Open questions / risks (verify before/while coding)
1. **Divisibility of `n_k_heads = ssm_n_group`.** RESOLVED — confirmed against the actual GGUFs:
   Qwen3.6-35B-A3B (qwen35moe): n_k=16, n_v=32, d_inner=4096. Qwen3.5-27B (qwen35 dense): n_k=16,
   n_v=48, d_inner=6144. Both divide cleanly by 2/4/8; full-attn layers too (16/2, 24/4). No
   asymmetric fallback needed for our 2–4-way regime. **Primary target: Qwen3.5-27B (dense hybrid)** —
   all weights active → best TP regime and cleanest correctness check.
2. **`ssm_conv1d` is depthwise** — the channel split must match the q\|k\|v layout exactly
   (conv_dim = 2·key_dim + value_dim; 3 channel segments). Off-by-one here = garbage (same class of
   bug as the fused gate\|up seam).
3. **GQA repeat** (`n_k_heads != n_v_heads`): after dividing both by size the ratio is preserved, so
   the `repeat_interleave` in build stays correct per-rank. (Confirmed by reading the build.)
4. **Legacy `ssm_in`**: refuse initially; add the 4-seg (q\|k\|v\|z) gather later if a target model
   only ships the legacy layout.

## Validation — DONE (2026-07-01, tonto92 Cascade Lake, 2-socket)
**VALIDATED CORRECT** on Qwen3.5-27B **Q8_0**, both GDN-only (`LLAMA_TP_SSM=1`) and full
(`+LLAMA_TP_ATTN=1`) — coherent output ("...Paris. This is a well-established geographical fact...").
Weight shard plans verified byte-exact via LLAMA_TP_GDN_DEBUG. Two bugs found + fixed during bring-up:
1. `tn.arch` is bound at model construction (before arch is known) → was UNKNOWN in the loader; use the
   loader's `get_arch()` instead (from the GGUF metadata).
2. **qwen35/qwen35moe use SEPARATE `ssm_beta` + `ssm_alpha` `{n_embd,n_v}`** (qwen3next has fused
   `ssm_beta_alpha`); added both to `tp_gdn_plan_for`. Also `ssm_beta_alpha` is per-k-group interleaved
   (contiguous split, not 2-seg).

### Base-TP bug FIXED: sharded tensors + online repack (was "Q4_K FFN garbage")
Root cause (general, not GDN): a sharded weight inherited the buffer type picked from its FULL metadata,
which on AVX-512-VNNI hosts is the online-repack extra buffer. Our shard loader wrote the raw slice
straight to tensor->data, bypassing the buffer's set_tensor transform, so a repack buffer read raw quant
bytes as repacked -> garbage. Only repack-path quants (Q4_K etc.) on a repack-capable host hit it;
Q8_0 / non-VNNI worked, hiding it. FIX (commit 38f47f52c): stage the shard and hand it to
ggml_backend_tensor_set (the same path a full tensor uses) — host buffers memcpy, repack buffers repack
in place, so the shards KEEP the SIMD layout. This is a TP-integration fix, not an upstream bug (normal
loading already repacks correctly). **Fixes Q4 TP for ALL models on VNNI hosts (dense/MoE too).**

### Perf (Qwen3.5-27B Q4_K_M, tonto92 Cascade Lake, 2-socket full TP vs 1-socket)
prefill 25.46 -> **45.74 t/s (+80%, 1.80x)**; decode 4.15 -> **7.08 t/s (+71%, 1.71x)**. (With the shards
skipping repack the numbers were +0%/-5% and +21% — repack matters a lot for both.)

TODO: token-for-token vs single-rank (currently "coherent" check); 2-socket decode/prefill perf numbers.
