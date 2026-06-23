// Inter-node transport for CPU tensor parallelism (rank=node, 2-way), used by build_ffn to
// all-reduce the row-parallel ffn_down output across nodes. UCX/IB, no MPI. Optional: when
// built without LLAMA_TP_UCX, llama_tp_enabled() is false and the op is a no-op.
#ifndef LLAMA_TP_NET_H
#define LLAMA_TP_NET_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cheap env check (LLAMA_TP_SIZE > 1) used at graph-build time to decide whether to insert the
// all-reduce op. Does NOT bootstrap the network.
int llama_tp_enabled(void);

// Cheap env check (LLAMA_TP_SIZE > 1 && LLAMA_TP_ATTN) — true when attention is sharded (Phase 2)
// so build_attn inserts an all-reduce after the row-parallel wo. Does NOT bootstrap the network.
int llama_tp_attn_enabled(void);

// Cheap env check (LLAMA_TP_SIZE > 1 && LLAMA_TP_MOE) — true when MoE experts are sharded across
// ranks (expert parallelism) so build_moe_ffn remaps to local experts and all-reduces the combine.
int llama_tp_moe_enabled(void);

// This rank's id and the TP world size (from LLAMA_TP_RANK / LLAMA_TP_SIZE; 0 / 1 if unset).
int llama_tp_rank(void);
int llama_tp_size(void);

// ggml custom1 op (n_tasks=1): in-place inter-node SUM all-reduce of `dst` (the partial
// ffn_down output). On its first call (ggml thread 0) it lazily bootstraps the transport from
// env (LLAMA_TP_SIZE / LLAMA_TP_RANK / LLAMA_TP_PEER / LLAMA_TP_PORT), so the UCX worker is
// created and used by the same thread.
void llama_tp_allreduce_op(struct ggml_tensor * dst, const struct ggml_tensor * a,
                           int ith, int nth, void * userdata);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_TP_NET_H
