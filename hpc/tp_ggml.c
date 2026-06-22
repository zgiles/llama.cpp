#include "tp_ggml.h"
#include "tp_net.h"

// In-place inter-node SUM all-reduce of the row-parallel matmul output.
// dst aliases `a` (inplace custom op); only thread 0 touches the UCX worker.
void tp_ggml_allreduce(struct ggml_tensor * dst, const struct ggml_tensor * a,
                       int ith, int nth, void * userdata) {
    (void)a; (void)nth;
    if (ith != 0)   return;                 // single-thread driver for the UCX worker
    if (!userdata)  return;                 // TP disabled => no-op (graph runs unsharded)
    // The row-parallel matmul output is f32 and contiguous.
    tp_net_allreduce((tp_net *)userdata, (float *)dst->data, (size_t)ggml_nelements(dst));
}
