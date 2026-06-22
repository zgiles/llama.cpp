#!/usr/bin/env bash
# A1: NUMA / decode-bandwidth characterization on one dual-socket node.
# Decode (tg) is memory-bandwidth bound; this isolates the cross-socket UPI penalty
# and the aggregate per-socket ceiling that socket-local TP / replication can recover.
#
# Run inside: nix shell nixpkgs#numactl --command bash hpc/a1_numa_bench.sh
# Uses --no-mmap (-mmp 0) so numactl --membind controls private weight placement
# (mmap'd files share page cache and ignore membind).
set -u
MODEL="${MODEL:-/home/zrg/models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf}"
BENCH="${BENCH:-/home/zrg/llama.cpp/build-native/bin/llama-bench}"
NGEN="${NGEN:-96}"
NPROMPT="${NPROMPT:-128}"
REPS="${REPS:-3}"
OUT="${OUT:-/home/zrg/hpc/a1_results.txt}"

COMMON="-m $MODEL -p $NPROMPT -n $NGEN -r $REPS"
: > "$OUT"
log(){ echo "$@" | tee -a "$OUT"; }
run(){ local label="$1"; shift; log ""; log "##### $label"; log "CMD: $*"; "$@" 2>>"$OUT" | tee -a "$OUT"; }

log "# A1 NUMA characterization  model=$(basename "$MODEL")  ngen=$NGEN nprompt=$NPROMPT reps=$REPS"
log "# host=$(hostname)  date=$(date -u +%FT%TZ)"

# 1. real-world default: mmap on, all 48 threads, no binding
run "default-mmap-48t"               $BENCH $COMMON -t 48 -mmp 1

# 2. single socket 0, fully local (24 cores + node0 memory) -> per-socket local ceiling
run "socket0-local-24t"              numactl --cpunodebind=0 --membind=0 -- $BENCH $COMMON -t 24 -mmp 0

# 3. forced remote: threads on socket1, weights on node0 -> every read crosses UPI
run "remote-cpu1-mem0-24t"           numactl --cpunodebind=1 --membind=0 -- $BENCH $COMMON -t 24 -mmp 0

# 4. interleave pages across both nodes, 48 threads (~50% local)
run "interleave-all-48t"             numactl --interleave=all     -- $BENCH $COMMON -t 48 -mmp 0

# 5. llama.cpp's own NUMA distribute mode, 48 threads
run "llama-numa-distribute-48t"      $BENCH $COMMON -t 48 -mmp 0 --numa distribute

# 6. naive both sockets, no binding, private alloc (no-mmap) -> shows cross-NUMA hurt
run "nobind-nomm-48t"                $BENCH $COMMON -t 48 -mmp 0

log ""
log "##### aggregate-2x-per-socket (two concurrent instances, one fully local per socket)"
log "This approximates the ceiling that socket-local TP / per-socket weight replication targets."
numactl --cpunodebind=0 --membind=0 -- $BENCH $COMMON -t 24 -mmp 0 -o csv > /home/zrg/hpc/a1_agg_s0.csv 2>/home/zrg/hpc/a1_agg_s0.err &
P0=$!
numactl --cpunodebind=1 --membind=1 -- $BENCH $COMMON -t 24 -mmp 0 -o csv > /home/zrg/hpc/a1_agg_s1.csv 2>/home/zrg/hpc/a1_agg_s1.err &
P1=$!
wait $P0 $P1
log "--- socket0 instance (csv) ---"; cat /home/zrg/hpc/a1_agg_s0.csv | tee -a "$OUT"
log "--- socket1 instance (csv) ---"; cat /home/zrg/hpc/a1_agg_s1.csv | tee -a "$OUT"
log ""
log "Done. Full log: $OUT"
