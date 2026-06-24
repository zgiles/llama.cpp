# Overnight CPU-TP MoE campaign — summary (2026-06-24)

## TL;DR

* Added a **selectable MoE parallel mode** (`LLAMA_TP_MOE=ep|tp`) — committed.
* **Tensor mode (`tp`) wins on big-expert models**: DeepSeek-V3.1 decode **+27%** / prefill **+36%** at 2
  sockets over one socket, **+66%** prefill at 4 — while expert mode (`ep`) barely helps. On fine-grained
  Qwen (tiny experts) tensor mode ≈ one socket (experts too small), as predicted.
* Added a **loader optimization** so tensor-mode sharded load is sequential (NFS-friendly) instead of
  millions of strided reads — made the big-model tensor runs feasible.
* Wrote **`docs/cpu-tensor-parallel.md`** (upstream-style: concepts, modes, usage, glossary, results).

## Results (tokens/s; rank=socket; Skylake-SP, no VNNI; 100Gb IB; weights on NFS)

Terminology: **S1** = one socket (NUMA-local baseline). **S2** = one process over both sockets (naive,
cross-socket UPI). **tp-N**/**ep-N** = tensor/expert parallel over N sockets. `pp`=prefill, `tg`=decode.

### Qwen3.6-35B-A3B — 256 experts × n_ff 512 (fine-grained)
| config | sockets | pp | tg |
|---|---|---|---|
| S1 | 1 | 133.0 | 12.16 |
| S2 (naive 2-sock) | 2 | 171.8 | 5.78 |
| tp-N2 (sm) | 2 | 150.8 | 12.05 |
| ep-N2 (sm) | 2 | 128.1 | 10.34 |
| tp-N4 (IB) | 4 | (refused — n_ff 512 not 4-way block-aligned; loader abort, by design) |
| ep-N4 (IB) | 4 | 129.0 | 10.83 |

### DeepSeek-V3.1 (Q4_K_M, 671B/37B-active) — 256 experts × n_ff 2048 (medium)
| config | sockets | pp | tg |
|---|---|---|---|
| S1 | 1 | 11.34 | 2.61 |
| tp-N2 (sm) | 2 | 15.45 | **3.32** (+27%) |
| ep-N2 (sm) | 2 | 11.91 | 2.70 |
| tp-N4 (IB) | 4 | **18.86** (+66%) | 3.00 |

### Nemotron-3-Ultra (550B/55B-active, Q4_K_M 334 GiB, nemotron_h_moe — Mamba-2 + MoE hybrid; 512 experts, 22 active)
| config | sockets | tool | pp | tg |
|---|---|---|---|---|
| S2 (naive 2-sock) | 2 | llama-bench | 9.75 | 1.17 |
| tp-N2 (sm) | 2 | llama-simple | — | ~0.54 |

* **TP now applies to `nemotron_h_moe` and is CORRECT.** It routes through the shared `build_moe_ffn` but
  with **no gate experts** (up → relu² → down; a LatentMoE with a latent projection around it). My `moe_par`
  gate required `gate_exps`; relaxing it to `up_exps && down_exps` (one line) makes ep/tensor modes apply.
  Tensor N2 output is coherent ("...Paris. ...Berlin."). This is the "flesh out a component for the hybrid"
  step the model needed.
* **But tensor mode does NOT speed Nemotron up** (tg ~0.54 vs S2 1.17; tool mismatch caveat — tensor was
  measured with llama-simple, baseline with llama-bench, and there is no clean S1 number). Root cause:
  Nemotron is a **Mamba-2 hybrid** — only the MoE-FFN layers shard; the **Mamba-2/attention layers are
  replicated** and now run on half the threads per rank, plus per-MoE-layer all-reduce. The non-FFN
  compute dominates, so FFN-only parallelism doesn't pay off.
* **Lesson (good for the doc):** TP helps when the **MoE-FFN is the bottleneck** (DeepSeek). For
  Mamba/attention-heavy hybrids, the SSM/attention layers must also be sharded — otherwise TP is overhead.
* Practical: also load-heavy — tensor mode's loader reads ~full expert tensors per rank to extract the
  `n_ff` slice; 334 GiB over shared NFS (2 ranks) took ~40 min. Local/NVMe weights or a chunked-strided
  loader would make big-model tensor runs quick.

## Findings

1. **NUMA pinning is the first-order win.** Naive 2-socket (S2) decode is < half of one NUMA-local socket
   (S1) — a single process streams weights across UPI. `rank=socket` fixes it.
2. **Tensor mode scales with expert size; expert mode is for capacity.** Confirmed across the spread:
   fine-grained Qwen (n_ff 512) → tensor ≈ S1; medium DeepSeek (n_ff 2048) → tensor clearly > S1 and > ep.
3. **The loader matters on NFS.** Strided per-row shard reads are pathological at 400 GB; one sequential
   span-read + in-RAM extract fixed it (`tp_read_shard`).
4. **GLM-5.2 (684 GB Q6_K_XL, dsa) is too slow to benchmark in budget** — single-config load+compute
   exceeded the 90-min cap (logged ERR). Arch loads fine; just too large/slow on these CPUs for a sweep.

## State

* Commits on `cpu-tensor-parallel`: EP mode (`05ef89ec8`), selectable ep/tensor mode (`408ee119a`).
* **Uncommitted** (working tree): the loader optimization (`tp_read_shard` in llama-model-loader.cpp) and
  `docs/cpu-tensor-parallel.md`. Raw data: `hpc/overnight_results.tsv`. Sweep harness: `hpc/overnight_sweep.sh`.

## Suggested next steps

* Commit the loader opt + doc.
* If pursuing upstream: turn the env-var knobs into proper params/CLI (mirror `--split-mode`), and add a
  small diagram. Decide whether to support tensor mode for merged `gate_up_exps` and for `LLAMA_TP_ATTN`
  fused-QKV models.
* Expert-compaction for EP prefill (keep `mul_mat_id` GEMM, drop the redundant-expert compute) if the
  EP-capacity prefill case matters.
