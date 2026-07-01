# RFC / draft PR: CPU tensor parallelism — a scaling alternative to the RPC backend

**Status:** RFC + draft of the core. Seeking direction on scope/transport before finishing the series.
Nothing here changes default behavior — the feature is off unless requested, and the inter-node
transport is behind a build flag.

## TL;DR

Adds **multi-process CPU tensor parallelism (TP)**: shard a model across N processes (typically
rank = NUMA socket, or rank = node) and all-reduce the row-parallel results. It targets large MoE/dense
models on CPU clusters. This is the same problem the **RPC backend** already addresses — distributed CPU
inference — but the RPC backend is a *serial layer pipeline* (no compute scaling), whereas TP shards the
weights and scales prefill near-linearly, cuts per-node RAM, and (NUMA-local) speeds up decode.

I'd like feedback on **(1)** whether this belongs in-tree, **(2)** the transport (UCX behind a build
flag vs a generic interface), and **(3)** how to stage it, before I finish and clean the full series.

## Why

- SoTA models (DeepSeek-V3, Qwen3-235B, GLM, Nemotron) are slow / don't fit on a single CPU node.
- The RPC backend pipelines layers over TCP serially → adds latency, no throughput scaling.
- TP aggregates the scarce resources — memory **bandwidth** across sockets/nodes and RAM **capacity** —
  and shards the weights: faster prefill, lower per-rank RAM, faster NUMA-local decode.

## What it does

- **Load-time weight sharding** (`llama-model-loader`): each rank materializes only its slice
  (column / row / expert / per-head-segment), quant-block-aligned, sequential span-read (NFS-friendly).
  Fails loudly if a shape can't be split rather than silently mis-sharding.
- **One all-reduce after each row-parallel matmul**, as a `ggml` custom op (no core ggml changes).
  N-way recursive-doubling; hierarchical (intra-node shared-memory + inter-node IB).
- **rank = socket** → NUMA-local shards (the decode win; avoids UPI thrash).
- **Surface:** first-class `llama_model_params` fields + CLI (`--tp-size/--tp-rank/--moe-parallel/
  --tp-attn/--tp-ssm/--tp-peer/--tp-port`), mirroring `--split-mode`. The UCX transport is gated behind
  `-DTP_UCX_*`; without it the all-reduce op is a no-op and nothing else is affected.

### What gets sharded
| Component | Split |
|---|---|
| Dense FFN | gate/up COLUMN, down ROW; **fused gate\|up** split per-half (keeps swiglu channel-aligned) |
| Attention (GQA) | q/k/v COLUMN, o ROW, heads divided |
| **Fused QKV** (Phi-3, GPT-NeoX, InternLM2, DBRX, …) | per-head 3-segment gather of `[Wq\|Wk\|Wv]` (+fused bias) |
| MLA attention (DeepSeek) | query heads; latent KV replicated |
| MoE routed experts | selectable **EP** (shard expert set) or **tensor** (split each expert's n_ff) |
| Mamba-2 / SSM mixer | channel/head-parallel, group-aligned |
| Replicated | router, embeddings, lm_head, norms, shared expert |

## Performance (CPU, Xeon, rank=socket, single stream, validated token-for-token correct)

| Model | config | result |
|---|---|---|
| Llama-70B dense Q8 | 4-way rank=socket (1 node) | **decode 4.0× single**, RAM −48% |
| Phi-3-mini dense (fused QKV+gate\|up) Q4 | 2-socket TP | **2.18× decode, 1.84× prefill** (near-linear) |
| DeepSeek-V3.1 MoE Q4 | tensor + MLA, N=4 (2 nodes IB) | **+66% prefill, +13% decode** |
| Qwen3-235B-A22B MoE Q8 | 2-socket tensor-TP | **+36% decode, 2.1× over naive** full-node |
| Nemotron-3-Ultra (Mamba2+MoE) Q4 | MoE + SSM TP | **+49% decode** (+32% from the SSM sharding) |

Honest envelope: single-stream **decode** is bandwidth/latency-bound — multi-node helps **prefill +
capacity**, but decode is best on a single NUMA-local node (inter-node all-reduce latency dominates at
batch=1). MoE decode doesn't benefit from speculation on CPU (verify scatters across experts); dense does.

## Proposed staging (each stage independently useful)

1. **Core** — loader sharding + all-reduce op (UCX optional) + params/CLI + dense/GQA/fused-QKV/gate\|up.
2. **MoE** — EP + tensor modes.
3. **MLA attention** (DeepSeek).
4. **Mamba-2 / SSM** (Nemotron/Granite/Falcon-H1).

## Known limitations / future work
- **Gated-delta-net** (Qwen3-Next/3.5/3.6 linear-attention mixer) not yet sharded — separate op.
- **Custom-QKV-split** archs (plamo3/openelm) and **attention bias** not yet handled.
- Speculation × TP not wired (would need a rank-0-driven, broadcast speculative loop).

## Questions for maintainers
1. Is a multi-process CPU-TP path something you'd want in-tree (as a better-scaling sibling to RPC)?
2. Transport: keep UCX optional-behind-a-build-flag, or define a generic transport interface (so it
   can run over plain sockets / MPI / etc.)?
3. Staging: the series above, or a different split?

Happy to shape it to your preferences before submitting the real PRs.
