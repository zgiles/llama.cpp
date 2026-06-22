// Hierarchical all-reduce for CPU tensor parallelism — THREAD-PARALLEL model.
//
// Matches how ggml runs ops: the whole thread pool (nth threads across n_sock sockets)
// executes the all-reduce collectively, each thread doing a slice. Not one thread/socket.
//
// Two physical levels behind one API:
//   L1 (intra-node): n_sock socket-local partials summed in shared memory by all threads.
//   L2 (inter-node): after the local reduce, `scratch` is the full node-local sum — exactly
//                    the buffer an inter-node UCC/IB all-reduce exchanges. Hook is marked in
//                    tp_allreduce_f32 (no-op until c->n_nodes>1 / c->net set).
//
// Per call, every thread passes: sid (its socket), lid (index within socket, 0..K-1),
// ith (global thread index, 0..nth-1), and its socket's send/recv buffers (identical
// across that socket's threads). recvbuf may alias sendbuf. Each thread keeps a private
// `sense` (sense-reversing barrier state, init 0 once).
#ifndef TP_ALLREDUCE_H
#define TP_ALLREDUCE_H

#include <stddef.h>
#include <stdatomic.h>

#define TP_MAX_LOCAL 8  // max sockets / NUMA domains sharing one address space

// Cache-line size is arch-dependent (x86 64B; Apple/some ARM64 use 128B). Over-padding is
// always safe, so the larger value on ARM64 avoids false sharing across the bigger line.
#if defined(__aarch64__) || defined(__arm__)
#define TP_CACHELINE 128
#else
#define TP_CACHELINE 64
#endif

// Cache-line padded so `count` (written by all) and `sense` (spun on by all) never share
// a line — false sharing on a centralized barrier is brutal across sockets.
typedef struct {
    _Alignas(TP_CACHELINE) _Atomic int count;
    char _pad0[TP_CACHELINE - sizeof(int)];
    _Alignas(TP_CACHELINE) _Atomic int sense;
    char _pad1[TP_CACHELINE - sizeof(int)];
    int n;
} tp_barrier;

// Per-thread barrier state (local + cross senses for the 2-level barrier).
typedef struct { int ls; int cs; } tp_sense;

typedef struct tp_comm {
    int n_sock;                              // # socket-local partials
    int K;                                   // threads per socket
    int nth;                                 // = n_sock * K
    const float * _Atomic send[TP_MAX_LOCAL]; // each socket's partial (published per call)
    float * scratch;                         // shared, >= count floats (node-local sum)
    size_t scratch_cap;
    tp_barrier sub[TP_MAX_LOCAL];            // per-socket sub-barrier (K threads, shared L3)
    tp_barrier cross;                        // across the n_sock socket leaders only

    // ---- L2 (inter-node) — set by tp_net_attach(); NULL/1 => local-only ----
    int node_rank;
    int n_nodes;                             // 1 => local-only
    void * net;                              // opaque transport handle (UCX)
    void (*net_allreduce)(void * net, float * buf, size_t count); // in-place SUM across nodes
} tp_comm;

// n_sock socket partials, K threads/socket. scratch: shared buffer >= max count floats.
int  tp_comm_init_local(tp_comm * c, int n_sock, int K, float * scratch, size_t scratch_cap);

// Collective SUM all-reduce of `count` floats, run by all nth threads.
//   sid: socket id (0..n_sock-1)   lid: thread index within socket (0..K-1)
//   ith: global thread index (0..nth-1)   st: this thread's private barrier state
void tp_allreduce_f32(tp_comm * c, int sid, int lid, int ith, tp_sense * st,
                      const float * sendbuf, float * recvbuf, size_t count);

void tp_barrier_wait(tp_barrier * b, int * sense);          // flat barrier over b->n threads
void tp_hier_barrier(tp_comm * c, int sid, int lid, tp_sense * st); // 2-level: socket then cross

#endif // TP_ALLREDUCE_H
