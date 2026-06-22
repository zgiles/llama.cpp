// Reproducer: 4 threads pinned across 2 sockets, exercising the REAL allreduce like the bench.
#define _GNU_SOURCE
#include "tp_allreduce.h"
#include <stdio.h>
#include <pthread.h>
#include <sched.h>

#define N 32768
static tp_comm c;
static float scratch[N];
static float sendb[2][N];
static float recvb[2][N];
static int PIN[4] = {0, 1, 24, 25};
typedef struct { int sid, lid, ith; } A;

static void * w(void * p) {
    A * a = (A *)p;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(PIN[a->ith], &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    tp_sense st = {0, 0};
    int sid = a->sid, lid = a->lid, ith = a->ith;
    float * send = sendb[sid];
    float * recv = recvb[sid];
    // first-touch (K=2: lid in 0..1)
    size_t b = N / 2, lo = (size_t)lid * b, ll = (lid == 1) ? (N - lo) : b;
    for (size_t i = lo; i < lo + ll; i++) { send[i] = (float)(sid + 1); recv[i] = 0.0f; }
    fprintf(stderr, "ith%d arrived\n", ith); fflush(stderr);
    tp_hier_barrier(&c, sid, lid, &st);
    if (ith == 0) { fprintf(stderr, "first barrier OK\n"); fflush(stderr); }

    for (int r = 0; r < 5; r++) {
        tp_allreduce_f32(&c, sid, lid, ith, &st, send, recv, N);
        if (ith == 0) { fprintf(stderr, "allreduce %d OK recv[0]=%.0f recv[%d]=%.0f\n", r, recv[0], N-1, recv[N-1]); fflush(stderr); }
    }
    return NULL;
}

int main(void) {
    tp_comm_init_local(&c, 2, 2, scratch, N);
    pthread_t t[4];
    A a[4] = {{0,0,0},{0,1,1},{1,0,2},{1,1,3}};
    for (int i = 0; i < 4; i++) pthread_create(&t[i], NULL, w, &a[i]);
    for (int i = 0; i < 4; i++) pthread_join(t[i], NULL);
    printf("DONE\n");
    return 0;
}
