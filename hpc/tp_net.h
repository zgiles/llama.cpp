// L2 inter-node transport for the hierarchical all-reduce (UCX/IB backend, no MPI).
// Decoupled from the L1 core via a function pointer on tp_comm, so tp_allreduce.c never
// links UCX for local-only builds.
#ifndef TP_NET_H
#define TP_NET_H

#include <stddef.h>
#include "tp_allreduce.h"

typedef struct tp_net tp_net;

// Bootstrap a 2-node (for now) inter-node group over UCX/IB.
//   node_rank 0 connects to `server_ip`; node_rank 1 binds and waits (server).
// Returns NULL on failure. Run with UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc,sm,self in env.
tp_net * tp_net_init(int n_nodes, int node_rank, const char * server_ip, int port);

// Attach this transport to a tp_comm: sets c->net, c->net_allreduce, c->n_nodes, c->node_rank
// so tp_allreduce_f32 transparently performs the L2 step.
void tp_net_attach(tp_net * net, tp_comm * c);

// In-place SUM all-reduce of `count` floats across nodes (same op the function pointer wraps).
// Exposed for the ggml custom-op bridge (single-thread driver).
void tp_net_allreduce(tp_net * net, float * buf, size_t count);

void tp_net_finalize(tp_net * net);

#endif // TP_NET_H
