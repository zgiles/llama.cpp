# CPU tensor-parallelism — performance tracking

Collected perf data across the project so we can see "where we are" and (later) plot progression.

**Important caveat.** These are **not a controlled series** — each phase measured a *different model
on different hardware*, because the work was mostly *breadth* (new architectures/capabilities: FFN →
attn → MoE-EP → MoE-tensor → MLA → Mamba-2 → GDN → multi-node) rather than *depth* (one number getting
faster). So there is no single "the speedup" that climbs monotonically. A true iteration-over-iteration
curve needs the *same* (model, hardware, config) benchmarked at each commit — see the harness note below.

All numbers are `llama-bench` t/s unless marked `[simple]` (single-run `llama-simple`, noisier).
pp = prefill (`-p512`), tg = decode (`-n128`), `-mmp 0`. Speedups are vs the row's stated baseline.

## Capability / perf milestones

| date | phase (commit) | model | HW | config | pp t/s | tg t/s | vs baseline |
|------|----------------|-------|----|--------|-------:|-------:|-------------|
| 06-22 | Ph1 FFN-only (232c8d2f5) | Llama-70B Q8 | 2× Skylake IB | rank=node FFN | — | 1.24 | **1.44× dec** vs 1-node 0.86 |
| 06-22 | Ph2 attn shard (a52832bd7) | Llama-70B Q8 | 2× Skylake IB | +attn | 34.83 | — | **1.81× pp / 1.53× dec** |
| 06-22 | Ph3 4-way (289306b08) | Llama-70B Q8 | 2 nodes / 4 sockets | rank=socket | — | 3.44 | **4.0× dec** vs 1-socket |
| 06-23 | MoE EP (05ef89ec8) | Qwen3.6-35B-A3B Q4 | 2× Skylake IB | EP | — | ~5.5 | 0.87× vs 1-node 6.34 (capacity, not speed) |
| 06-24 | MoE tensor (408ee119a) | DeepSeek-V3.1 Q4 | 2× Skylake IB | tensor | 15.5 | 3.32 | **+36% pp / +27% dec** vs 1-sock 11.3/2.61 |
| 06-24 | MoE tensor N4 | DeepSeek-V3.1 Q4 | 4 sockets | tensor | 18.9 | — | **+66% pp** |
| 06-24 | MLA attn (f772875da) | DeepSeek-V3.1 Q4 | 1 node 2-sock sm | tensor+MLA | 19.16 | 3.68 | **+25% pp / +13% dec** over tensor-only |
| 06-26 | tensor-TP (408ee119a) | Qwen3-235B-A22B Q8 | tonto92 2-sock | tensor | 21.10 | 3.72 | **+36% dec** vs 1-sock 13.32/2.74 |
| 06-25 | spec decode (validated) | R1-70B Q6 + 8B draft | 1 socket | draft-simple | — | 2.39 | **1.56× dec** vs 1.53 baseline |
| 07-01 | GDN + repack (38f47f52c) | Qwen3.5-27B Q4 | tonto92 2-sock | attn+GDN | — | — | **1.80× pp / 1.71× dec** (repack fix) |
| 07-02 | GDN full (aad985ef0) | Qwen3.6-35B-A3B Q4 | Cascade 92 2-sock sm | attn+GDN+MoE | 131.27 | 12.01 | **+36% pp / +17% dec** vs 1-sock 96.74/10.27 |
| 07-02 | GDN multinode (232a3d8dc) | Qwen3.6-35B-A3B Q4 | 2× Skylake IB | attn+GDN+MoE | 246.11 | 7.89 | **+84% pp / −27% dec** vs 1-sock 133.55/10.80 |
| 07-02 | qwen3next (232a3d8dc) | Qwen3-Next-80B Q4 | 2× Skylake IB | attn+GDN+MoE | — | 4.54 | `[simple]` coherent, no clean bench yet |

## The recurring lesson (holds across every phase)
- **Prefill scales** with sockets/nodes (compute-bound, batched GEMM): +36% to +84% typical, near-linear
  within-node, sublinear cross-node.
- **Decode is the hard part** (batch-1, latency/bandwidth-bound): within-node shared-memory TP *helps or is
  free* (+17% GDN on Cascade); multi-node IB TP *hurts* (−27%) because every layer pays IB all-reduce
  latency. `sm` reduce ≈ microseconds; `rc`/IB reduce ≈ latency+skew-bound. → multi-node is a
  prefill/throughput/capacity play; batch-1 decode wants one NUMA-local socket or speculative decode.

## Machine-readable (for plotting)
```csv
date,commit,model,hw,config,pp_ts,tg_ts,pp_base,tg_base,pp_speedup,tg_speedup
2026-06-22,232c8d2f5,llama70b-q8,2xskylake-ib,ffn-2node,,1.24,,0.86,,1.44
2026-06-22,a52832bd7,llama70b-q8,2xskylake-ib,attn-2node,34.83,,19.24,,1.81,1.53
2026-06-24,408ee119a,deepseek-v31-q4,2xskylake-ib,moe-tensor-2node,15.5,3.32,11.3,2.61,1.36,1.27
2026-06-24,f772875da,deepseek-v31-q4,1node-2sock,tensor+mla,19.16,3.68,15.31,3.25,1.25,1.13
2026-06-26,408ee119a,qwen3-235b-q8,tonto92-2sock,tensor,21.10,3.72,13.32,2.74,1.58,1.36
2026-07-02,aad985ef0,qwen36-35b-a3b-q4,cascade92-2sock-sm,attn+gdn+moe,131.27,12.01,96.74,10.27,1.36,1.17
2026-07-02,232a3d8dc,qwen36-35b-a3b-q4,2xskylake-ib,attn+gdn+moe,246.11,7.89,133.55,10.80,1.84,0.73
```

## To make the actual "speedup vs commit" graph (controlled series)
1. Pick ONE fixed benchmark: e.g. `qwen35moe 35B Q4_K_M, tonto92 2-socket sm, -p512 -n128 -r3`.
2. `hpc/perf_series.sh` (to write): for each key commit — `git archive <sha> | build on node | run the
   fixed bench | append pp/tg to a TSV`. Most commits won't move this number (they added *other* archs),
   so expect a mostly-flat line with jumps only at commits that touch that path (repack fix, GDN fix).
3. Plot: `pp_speedup`/`tg_speedup` vs commit index (matplotlib step plot / waterfall). Better suited to a
   *sub-thread* where the same metric improves (e.g. MoE decode on one model: 1-sock → EP → tensor → …).
