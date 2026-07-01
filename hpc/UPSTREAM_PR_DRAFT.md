# DRAFT PR — CPU tensor parallelism for large models (experimental)

> Status: **draft for internal review — do NOT submit.** Framed as an RFC/feature proposal.

## Summary

Adds optional **CPU tensor parallelism (TP)**: shard a model across N processes (typically
rank = NUMA socket, or rank = node) and all-reduce the row-parallel results over a pluggable
transport. Targets large MoE/dense models on CPU clusters, where the existing RPC backend is a
serial layer pipeline (no compute scaling) and per-node RAM is the limiting resource.

The transport (UCX/IB) is **behind a build flag** (`-DTP_UCX_*`); without it the all-reduce op is a
no-op and the build is unaffected. Activated per model load via params/CLI (off by default).

## Motivation

- SoTA models (DeepSeek-V3, GLM, Qwen3-235B, Nemotron-3) are too big / too slow on one CPU node.
- The RPC backend pipelines layers over TCP serially → adds latency, no throughput scaling.
- TP aggregates the scarce resources — memory **bandwidth** across sockets/nodes and RAM **capacity**
  — and shards the weights: faster prefill, lower per-rank RAM, and (NUMA-local) faster decode.

## What gets sharded

| Component | Split |
|---|---|
| Dense FFN | gate/up COLUMN, down ROW; **fused gate\|up** split per-half (2-segment gather, keeps swiglu channel-aligned) |
| Attention (GQA) | q/k/v COLUMN, o ROW, heads divided |
| **Fused QKV** attention (Phi-3, GPT-NeoX, InternLM2, …) | per-head 3-segment gather of `[Wq\|Wk\|Wv]` (+ fused bias); `wo` ROW |
| MLA attention (DeepSeek) | query heads (wq_b col, wk_b/wv_b per-head, wo row); latent KV replicated |
| MoE routed experts | selectable **EP** (shard expert set) or **tensor** (split each expert's n_ff) |
| Mamba-2 / SSM mixer | channel/head-parallel, group-aligned; ssm_in/conv segment-gather; ssm_out row |
| Replicated | router, embeddings, lm_head, norms, shared expert |

## Design

- **Load-time weight sharding** (`llama-model-loader`): each rank materializes only its slice
  (column/row/expert/SSM-segment), quant-block-aligned; sequential span-read (NFS-friendly).
- **One all-reduce after each row-parallel matmul**, as a `ggml` custom op. N-way recursive-doubling;
  hierarchical (intra-node shared-memory + inter-node IB).
- **rank = socket** → NUMA-local shards — this is the decode win (avoids UPI thrash).
- **Surface:** first-class `llama_model_params` fields + CLI (`--tp-size/--tp-rank/--moe-parallel/
  --tp-attn/--tp-ssm/--tp-peer/--tp-port`), mirroring `--split-mode`. Env vars kept only as a fallback
  (to be dropped for upstream, or gated).

## Performance (CPU, Xeon Skylake/Cascade, rank=socket, validated token-for-token correct)

| Model | Config | Result |
|---|---|---|
| 70B dense Q8 | 4-way rank=socket (1 node) | **decode 4.0× single**, RAM −48% |
| DeepSeek-V3.1 MoE Q4 | tensor mode + MLA, N=4 (2 nodes IB) | **+66% prefill, +13% decode**; MLA is load-bearing for prefill scaling |
| Qwen3-235B-A22B Q8 | 2-socket tensor-TP (1 node) | **+36% decode, 2.1× over naive** full-node |
| Nemotron-3-Ultra (Mamba2+MoE) | MoE + SSM TP (1 node) | **+49% decode over 1-socket** (+32% from the SSM sharding alone) |
| Phi-3-mini (dense, fused QKV + fused gate\|up) | 2-socket TP | **2.18× decode, 1.84× prefill** (fully dense → near-linear) |

Notes for reviewers (honest limits): single-stream decode is bandwidth/latency-bound — multi-node
helps **prefill + capacity**, but decode is best on a single NUMA-local node (inter-node all-reduce
latency dominates at batch=1). MoE decode does not benefit from speculation on CPU (verify scatters
across experts); dense does.

## Limitations & future work

Architecture coverage (attention sharding refuses models it can't yet split — fails loudly, never
silently mis-shards):
- **Gated-delta-net / linear-attention not sharded.** Only standard Mamba-2 (`build_mamba2_layer`,
  Nemotron-H/Granite/Falcon-H1) is sharded. The Qwen3-Next/3.5/3.6 gated-delta-net mixer
  (`build_recurrent_attn`) is a different op and stays replicated — so those hybrids only get FFN TP.
- **Attention bias** (bq/bk/bv) is refused alongside fused-QKV.

Performance envelope (be upfront — sets reviewer expectations):
- **Single-stream decode is bandwidth/latency-bound.** Multi-node helps **prefill + capacity**; decode is
  best on a single NUMA-local node (inter-node all-reduce latency dominates at batch=1, can regress).
- **MoE decode does not benefit from speculation/MTP on CPU** (the K-token verify scatters across experts,
  losing the weight-reuse amortization). Dense does (validated ~1.4–1.6×). Spec×TP isn't wired (the
  speculative loop would need to be rank-0-driven + broadcast across TP ranks).

## Suggested landing (series, not one mega-PR)

1. **Core dense TP** — loader sharding + all-reduce op (UCX optional) + params/CLI. The foundation; clearest win (near-linear dense, RAM).
2. **MoE EP + tensor modes.**
3. **MLA attention sharding.**
4. **Mamba-2 / SSM sharding** (validated correct + +32% decode).

## Strip before submitting (currently on our branch)

- `hpc/` experiment scripts, results logs, and design docs (ours, not for upstream — fold essentials
  into `docs/cpu-tensor-parallel.md`).
- Env-var fallbacks (`LLAMA_TP_*`) — params-only for a clean PR, or keep behind the build flag.
- Any debug knobs / `simple.cpp` test helper.

## Open questions for maintainers (post in the RFC first)

- Appetite for a multi-process CPU TP path in tree?
- Transport: keep UCX optional-behind-build-flag, or define a generic transport interface?
- One feature branch vs the staged series above?
