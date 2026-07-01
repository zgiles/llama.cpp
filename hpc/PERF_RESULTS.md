# CPU tensor-parallelism — consolidated performance results

All numbers are `llama-bench` (pp = prefill t/s, tg = decode t/s) unless noted; CPU, `-mmp 0`,
rank=socket (NUMA-local) for TP, single stream (batch=1). Xeon Skylake 8168 (nodes 121/124) or
Cascade Lake 8260 (node 92). "N-way" = N NUMA sockets; ≤2-way is single-node (`sm`), 4-way is
2 nodes over 100 Gb IB (`rc`).

## Dense models — TP scales well (all weights active)

| Model | quant | config | pp | tg | vs base |
|---|---|---|---|---|---|
| Llama-70B | Q8 | 1 socket → 4-way rank=socket | — | — → — | **decode 4.0×**, RAM −48% |
| Llama-70B | Q8 | 2-node IB | 1.81× | 1.53× | prefill/decode over 1 node |
| Phi-3-mini (fused QKV + fused gate\|up) | Q4 | 1-sock → 2-sock TP | 77.3→142.0 | 15.4→**33.6** | **2.18× tg, 1.84× pp** (near-linear) |
| InternLM2-7B (GQA) | Q4 | 1-sock → 2-sock TP | 46.3→86.6 | 8.95→**14.2** | **1.59× tg, 1.87× pp** |
| Qwen3.6-27B (dense hybrid, gated-delta-net) | Q4 | 1-sock → 2-sock FFN-only TP | — | 3.07→3.78 | +23% (attn/gdn replicated) |
| **Qwen3.5-27B (hybrid, GDN sharded)** | Q4_K | 1-sock → 2-sock full TP (GDN+attn+FFN) | 25.5→**45.7** | 4.15→**7.08** | **+80% pp, +71% tg** (GDN TP + repack fix) |

## MoE models — decode is bandwidth-bound (~3–4 t/s ceiling); tensor-TP is the lever

| Model | quant | config | pp | tg | note |
|---|---|---|---|---|---|
| DeepSeek-V3.1 (256×2048, MLA) | Q4 | 1-sock | 11.3 | 2.61 | baseline |
| DeepSeek-V3.1 | Q4 | tensor-TP N=2 | 15.5 | 3.32 | +27% dec / +36% pp |
| DeepSeek-V3.1 | Q4 | tensor+MLA-attn N=4 (IB) | 30.15 | 3.37 | MLA: +66% pp @N4, +13% tg |
| Qwen3-235B-A22B | Q8 | 1-sock | 13.3 | 2.74 | baseline |
| Qwen3-235B-A22B | Q8 | naive 2-sock (-t48) | 22.6 | **1.74** | UPI trap — worse than 1-sock |
| Qwen3-235B-A22B | Q8 | **tensor-TP 2-sock** | 21.1 | **3.72** | **+36% tg, 2.1× over naive** |
| Nemotron-3-Ultra (Mamba2+MoE) | Q4 | 1-sock | 6.14 | 1.61 | baseline |
| Nemotron-3-Ultra | Q4 | MoE-only TP | 8.56 | 1.82 | +13% (Mamba replicated) |
| Nemotron-3-Ultra | Q4 | **MoE + SSM TP** | 10.45 | **2.40** | **+49% tg** (+32% from SSM) |
| GLM-5.2-REAP-504B (pruned, dsa) | Q4 | 1-sock → tensor-TP | 6.42→8.76 | 2.05→2.43 | +19% |
| Qwen3.6-A3B (256×512) | Q4 | 1-sock / tensor N=2 | — | 12.35 / 11.62 | experts too small — ties 1-sock |

Key: smaller-**active** wins — Qwen3-235B-A22B Q8 (tg 3.72) beats REAP-504B Q4 (2.43) despite Q4.

## Speculation / MTP — dense only

| Target | draft | config | tg base → spec | accept |
|---|---|---|---|---|
| R1-Distill-Llama-70B (dense) | R1-Distill-8B | 1-sock | 1.53 → **2.39 (1.56×)** | 61% (code) |
| Qwen3.6-27B (dense) | **native MTP** (self) | 1-sock, server | 3.08 → **4.34 (1.41×)** | 71%, mean 3.82/4 |
| Qwen3-235B-A22B (**MoE**) | Qwen3-0.6B | 1-sock | 2.74 → 2.75 | **neutral** (expert scatter) |

MoE decode gets ~nothing from spec on CPU (K-token verify scatters across experts, no weight-reuse);
GLM-5.2's MTP isn't wired in llama.cpp at all.

## GPU (Intel Arc B580, Vulkan/Mesa on node 92)

| Case | result |
|---|---|
| Small dense (Qwen3-0.6B Q8) on GPU | **95 t/s** — works great |
| Big-MoE load (235B, Vulkan) | stalls — `--fit` NFS-thrash + non-mmap staging (fix: local weights + `-fit off`) |
| Dense-offload (235B, experts CPU + rest GPU) | tg **0.99** — net **loss** (per-layer CPU↔GPU latency at batch=1) |
| Dense-offload (REAP-504B) | OOM (dense part > 12 GB VRAM) |

Verdict: B580 helps small dense; **dead end for big-MoE single-stream**. SYCL won't fix the load
(backend-agnostic) + Q8 is ~4× slower than Q4 on Battlemage — not worth pursuing.

## Hardware note — Cascade Lake vs Skylake (DeepSeek-V3.1 Q4)

| | Skylake 8168 | Cascade 8260 (VNNI) |
|---|---|---|
| 1-sock pp / tg | 11.34 / 2.59 | 7.52 / 2.51 |
| 2-sock tensor+attn pp / tg | 19.16 / 3.68 | 12.94 / 3.81 |

VNNI does **not** help Q4_K (not INT8); Skylake's higher clock wins prefill. Decode ~equal.

## Durable conclusions

- **Max single-stream decode** = smallest-**active** MoE + smallest quant + **tensor-TP rank=socket**, single NUMA-local node. Decode is bandwidth-bound.
- **Dense** is the only regime where **TP × spec/MTP compound** (needs the spec↔TP loop, not yet built).
- **Dead ends** for big-MoE single-stream decode: GPU offload, spec/MTP, multi-node (decode).
- **Wins that shipped:** dense/GQA/**fused-QKV**/MLA attention sharding, **fused gate\|up** FFN, MoE EP/tensor, Mamba-2 SSM. See `UPSTREAM_PR_DRAFT.md`.
