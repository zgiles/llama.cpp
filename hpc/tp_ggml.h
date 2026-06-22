// Bridge: ggml custom-op <-> tp_net inter-node all-reduce.
// Insert after a ROW-PARALLEL matmul (attn-output wo, ffn_down) in build_lora_mm:
//   cur = ggml_map_custom1_inplace(ctx0, cur, tp_ggml_allreduce, 1, tp_net_handle);
// n_tasks=1 -> a single ggml worker thread drives the (single-thread-mode) UCX worker.
// userdata == NULL => no-op (local-only / TP disabled), so the same graph runs unsharded.
#ifndef TP_GGML_H
#define TP_GGML_H
#include "ggml.h"
void tp_ggml_allreduce(struct ggml_tensor * dst, const struct ggml_tensor * a,
                       int ith, int nth, void * userdata);
#endif
