#!/usr/bin/env bash
# Overnight CPU-TP MoE benchmark campaign.
# Each config runs llama-bench on one or more node:socket "ranks"; results -> overnight_results.tsv.
# Robust: per-config timeout, transient-SSH retries, skip-on-failure, everything logged.
set -uo pipefail

DIR=/home/zrg/git/llama.cpp/hpc
RES=$DIR/overnight_results.tsv
LOG=$DIR/overnight.log
BIN=/home/zrg/llama.cpp/build-nr/bin
PP=128; NG=32; REP=2     # modest: big MoEs are compute-slow on no-VNNI Skylake; tg is the key metric
PORT=15000

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
[ -f "$RES" ] || printf 'model\tconfig\tnuma_domains\ttransport\tpp_t/s\ttg_t/s\n' > "$RES"

# retry an ssh command a few times (nodes occasionally drop a connection)
rssh(){ local h=$1; shift; local i; for i in 1 2 3; do ssh -o ConnectTimeout=10 zrg@"$h" "$@" && return 0; sleep 5; done; return 1; }

# parse "pp tg" t/s from a llama-bench table piped on stdin
parse(){ awk -F'|' '/pp[0-9]/{gsub(/ /,"",$(NF-1));pp=$(NF-1)} /tg[0-9]/{gsub(/ /,"",$(NF-1));tg=$(NF-1)} END{print (pp==""?"ERR":pp)" "(tg==""?"ERR":tg)}'; }

# single-process baseline. args: tag model node threads numa(0|1|both) mmap numa_desc
single(){
  local tag=$1 model=$2 node=$3 th=$4 numa=$5 mmap=$6 nd=$7
  local nc=""; [ "$numa" != both ] && nc="numactl --cpunodebind=$numa --membind=$numa"
  log "RUN $tag [$nd] node=$node t=$th mmap=$mmap"
  rssh 10.70.132.$node "pgrep -x llama-bench|xargs -r kill -9 2>/dev/null; cd $BIN; timeout 5400 $nc ./llama-bench -m $model -p $PP -n $NG -r $REP -t $th -mmp $mmap >/tmp/sw.out 2>/tmp/sw.err; echo END\$? >>/tmp/sw.out"
  local out; out=$(rssh 10.70.132.$node "cat /tmp/sw.out")
  local r; r=$(printf '%s' "$out" | parse)
  printf '%s\t%s\t%s\t%s\t%s\n' "$TAGM" "$tag" "$nd" "-" "${r/ /	}" >> "$RES"
  log "  $tag -> $r"
}

# tensor/expert parallel. args: tag model mode(tp|ep) tls(sm|rc) peer numa_desc  "node:sock" ...
tp(){
  local tag=$1 model=$2 mode=$3 tls=$4 peer=$5 nd=$6; shift 6
  local ranks=("$@"); local size=${#ranks[@]}; local port=$((PORT++))
  log "RUN $tag [$nd] mode=$mode size=$size tls=$tls port=$port ranks=${ranks[*]}"
  for n in "${ranks[@]}"; do rssh 10.70.132.${n%%:*} 'pgrep -x llama-bench|xargs -r kill -9 2>/dev/null' ; done
  local rank=0
  for rs in "${ranks[@]}"; do
    local node=${rs%%:*} sock=${rs##*:}
    rssh 10.70.132.$node "cd $BIN; UCX_TLS=$tls,self UCX_NET_DEVICES=mlx5_0:1 LLAMA_TP_SIZE=$size LLAMA_TP_RANK=$rank LLAMA_TP_MOE=$mode LLAMA_TP_PEER=$peer LLAMA_TP_PORT=$port nohup numactl --cpunodebind=$sock --membind=$sock ./llama-bench -m $model -p $PP -n $NG -r $REP -t 24 -mmp 0 >/tmp/sw_$rank.out 2>/tmp/sw_$rank.err </dev/null & echo started"
    rank=$((rank+1))
    sleep 1
  done
  # wait for rank0 to finish (END marker) or time out (~90 min)
  local r0node=${ranks[0]%%:*}; local waited=0
  rssh 10.70.132.$r0node "(for i in \$(seq 1 540); do grep -qE 'tg[0-9]' /tmp/sw_0.out 2>/dev/null && break; grep -qE 'error|EXIT|GGML_ASSERT|cannot shard|terminate|what\\(\\)' /tmp/sw_0.err 2>/dev/null && break; sleep 10; done)"
  local out; out=$(rssh 10.70.132.$r0node "cat /tmp/sw_0.out")
  local r; r=$(printf '%s' "$out" | parse)
  printf '%s\t%s\t%s\t%s\t%s\n' "$TAGM" "$tag" "$nd" "$tls" "${r/ /	}" >> "$RES"
  log "  $tag -> $r"
  for n in "${ranks[@]}"; do rssh 10.70.132.${n%%:*} 'pgrep -x llama-bench|xargs -r kill -9 2>/dev/null'; done
}

log "=== campaign start ==="

# full config matrix for one model. rank=socket, -t 24. Failures (e.g. tensor split not block-aligned
# -> loader abort) are logged as ERR and the sweep continues.
run_model(){            # name model set(full|key)
  TAGM=$1; local M=$2 set=$3
  log "######## MODEL $TAGM  set=$set  ($M) ########"
  single S1       "$M" 121 24 0    0 1          # 1 socket (NUMA-local baseline)
  tp tp-N2-sm "$M" tp sm 127.0.0.1      2 121:0 121:1                  # tensor, 1 node, 2 sockets
  tp ep-N2-sm "$M" ep sm 127.0.0.1      2 121:0 121:1                  # expert, 1 node, 2 sockets
  tp tp-N4-rc "$M" tp rc 10.70.132.121  4 121:0 121:1 124:0 124:1      # tensor, 2 nodes, 4 sockets (IB)
  if [ "$set" = full ]; then
    single S2     "$M" 121 48 both 0 2          # 2 sockets, one process (naive, UPI)
    tp ep-N4-rc "$M" ep rc 10.70.132.121 4 121:0 121:1 124:0 124:1    # expert, 2 nodes, 4 sockets (IB)
  fi
}

M_QWEN=/kimie/projects/ai/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
M_DS=/kimie/projects/ai/models/deepseek-v31/DeepSeek-V3.1-Q4_K_M-00001-of-00009.gguf
M_GLM=/kimie/projects/ai/models/GLM-5.2/GLM-5.2-UD-Q6_K_XL-00001-of-00016.gguf

run_model qwen     "$M_QWEN" full     # small, fast loads -> full matrix incl S2 + ep-N4
run_model deepseek "$M_DS"   key      # huge -> S1, tp-N2, ep-N2, tp-N4 only (load-dominated)
run_model glm      "$M_GLM"  key

log "=== campaign done ==="
