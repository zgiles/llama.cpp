#include "tp_allreduce.h"
#include <string.h>

static inline void tp_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

void tp_barrier_wait(tp_barrier * b, int * sense) {
    const int s = !*sense;
    *sense = s;
    if (atomic_fetch_add_explicit(&b->count, 1, memory_order_acq_rel) == b->n - 1) {
        atomic_store_explicit(&b->count, 0, memory_order_relaxed);
        atomic_store_explicit(&b->sense, s, memory_order_release);
    } else {
        while (atomic_load_explicit(&b->sense, memory_order_acquire) != s) {
            tp_cpu_relax();
        }
    }
}

// 2-level barrier: K threads meet within a socket (shared L3, no UPI), then the n_sock
// leaders meet across sockets, then release down. Cross-socket traffic is O(n_sock), not O(nth).
void tp_hier_barrier(tp_comm * c, int sid, int lid, tp_sense * st) {
    tp_barrier_wait(&c->sub[sid], &st->ls);
    if (lid == 0) {
        tp_barrier_wait(&c->cross, &st->cs);
    }
    tp_barrier_wait(&c->sub[sid], &st->ls);
}

static void tp_bar_init(tp_barrier * b, int n) {
    atomic_store_explicit(&b->count, 0, memory_order_relaxed);
    atomic_store_explicit(&b->sense, 0, memory_order_relaxed);
    b->n = n;
}

int tp_comm_init_local(tp_comm * c, int n_sock, int K, float * scratch, size_t scratch_cap) {
    if (n_sock < 1 || n_sock > TP_MAX_LOCAL || K < 1) return -1;
    c->n_sock      = n_sock;
    c->K           = K;
    c->nth         = n_sock * K;
    c->scratch     = scratch;
    c->scratch_cap = scratch_cap;
    for (int s = 0; s < TP_MAX_LOCAL; s++) {
        atomic_store_explicit(&c->send[s], (const float *)0, memory_order_relaxed);
        tp_bar_init(&c->sub[s], K);
    }
    tp_bar_init(&c->cross, n_sock);
    c->node_rank     = 0;
    c->n_nodes       = 1;
    c->net           = (void *)0;
    c->net_allreduce = (void *)0;
    return 0;
}

static inline void tp_slice(size_t count, int n, int idx, size_t * off, size_t * len) {
    const size_t base = count / (size_t)n;
    const size_t rem  = count % (size_t)n;
    *off = (size_t)idx * base + (idx < (int)rem ? (size_t)idx : rem);
    *len = base + (idx < (int)rem ? 1 : 0);
}

void tp_allreduce_f32(tp_comm * c, int sid, int lid, int ith, tp_sense * st,
                      const float * sendbuf, float * recvbuf, size_t count) {
    const int     S       = c->n_sock;
    const int     K       = c->K;
    float * const scratch = c->scratch;

    atomic_store_explicit(&c->send[sid], sendbuf, memory_order_release);
    tp_hier_barrier(c, sid, lid, st);

    // Reduce-scatter: global partition by ith/nth; each thread sums its slice across sockets.
    size_t go, gl;
    tp_slice(count, c->nth, ith, &go, &gl);
    if (gl) {
        const float * s0 = atomic_load_explicit(&c->send[0], memory_order_acquire);
        for (size_t i = go; i < go + gl; i++) scratch[i] = s0[i];
        for (int s = 1; s < S; s++) {
            const float * sp = atomic_load_explicit(&c->send[s], memory_order_acquire);
            for (size_t i = go; i < go + gl; i++) scratch[i] += sp[i];
        }
    }
    tp_hier_barrier(c, sid, lid, st);

    // ---- L2 (inter-node): scratch now holds the NODE-LOCAL sum. One thread drives the
    //      network all-reduce (UCX worker is single-threaded), the rest wait; afterwards
    //      scratch holds the GLOBAL sum. No-op when local-only (n_nodes==1). ----
    if (c->n_nodes > 1 && c->net_allreduce) {
        if (ith == 0) {
            c->net_allreduce(c->net, scratch, count);
        }
        tp_hier_barrier(c, sid, lid, st);
    }

    // All-gather: this socket's K threads copy the full scratch into recvbuf.
    size_t lo, ll;
    tp_slice(count, K, lid, &lo, &ll);
    if (ll) memcpy(recvbuf + lo, scratch + lo, ll * sizeof(float));
    tp_hier_barrier(c, sid, lid, st);
}
