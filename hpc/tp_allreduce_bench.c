// Benchmark + correctness test for the thread-parallel (L1) shared-memory all-reduce.
// T = n_sock * K threads (K per socket), pinned per socket, buffers first-touched locally.
// Models how ggml runs the all-reduce: the whole pool executes it, each thread a slice.
//
//   gcc -O3 -march=native -std=c11 -pthread tp_allreduce_bench.c tp_allreduce.c -o tp_bench
//   ./tp_bench [K]     # threads per socket (default = #cpus on socket 0)
#define _GNU_SOURCE
#include "tp_allreduce.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>

#define MAXN  (32768)
#define NSIZE 6
static const size_t SIZES[NSIZE] = {1024, 2048, 4096, 8192, 16384, 32768};
#define ITERS 20000

static tp_comm  g_comm;
static float  * g_scratch;
static int      g_nsock, g_K, g_nth;
static int      g_cpu[TP_MAX_LOCAL][512];     // ordered cpu ids per socket
static int      g_ncpu[TP_MAX_LOCAL];
static float  * g_send[TP_MAX_LOCAL];          // one partial per socket (shared by its threads)
static float  * g_recv[TP_MAX_LOCAL];
static double   g_lat_us[TP_MAX_LOCAL * 512][NSIZE];
static int      g_ok[TP_MAX_LOCAL * 512];

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void parse_cpulist(const char * s, int * arr, int * n) {
    int k = 0;
    while (*s && *s != '\n') {
        int a = 0, b;
        while (*s >= '0' && *s <= '9') { a = a * 10 + (*s - '0'); s++; }
        b = a;
        if (*s == '-') { s++; b = 0; while (*s >= '0' && *s <= '9') { b = b * 10 + (*s - '0'); s++; } }
        for (int c = a; c <= b && k < 512; c++) arr[k++] = c;
        if (*s == ',') s++;
    }
    *n = k;
}

// Keep only the primary (lowest-id) hardware thread of each physical core, so two threads
// never land on hyperthread siblings of the same core — that wrecks spin-barrier latency.
static int tp_is_primary_core(int cpu) {
    char path[160]; snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    FILE * f = fopen(path, "r");
    if (!f) return 1;                       // no topology -> treat as primary
    char buf[256] = {0}; int first = -1;
    if (fgets(buf, sizeof buf, f)) sscanf(buf, "%d", &first);  // siblings sorted ascending
    fclose(f);
    return (first < 0 || first == cpu);
}

typedef struct { int sid, lid, ith; } targ;

static void * worker(void * p) {
    targ * t = (targ *)p;
    int sid = t->sid, lid = t->lid, ith = t->ith;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(g_cpu[sid][lid % g_ncpu[sid]], &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    sched_yield();

    tp_sense st = {0, 0};
    float * send = g_send[sid];
    float * recv = g_recv[sid];

    // first-touch our socket's send/recv slices and our global scratch slice
    size_t lo, ll; { size_t b = MAXN / g_K; lo = (size_t)lid * b; ll = (lid == g_K - 1) ? (MAXN - lo) : b; }
    for (size_t i = lo; i < lo + ll; i++) { send[i] = (float)(sid + 1); recv[i] = 0.0f; }
    { size_t b = MAXN / g_nth; size_t o = (size_t)ith * b, l = (ith == g_nth - 1) ? (MAXN - o) : b;
      for (size_t i = o; i < o + l; i++) g_scratch[i] = 0.0f; }
    tp_hier_barrier(&g_comm, sid, lid, &st);

    // correctness: send=sid+1 => recv == sum_s (s+1)
    tp_allreduce_f32(&g_comm, sid, lid, ith, &st, send, recv, SIZES[NSIZE - 1]);
    float expect = (float)(g_nsock * (g_nsock + 1) / 2);
    int ok = 1;
    for (size_t i = 0; i < SIZES[NSIZE - 1]; i++) if (recv[i] != expect) { ok = 0; break; }
    g_ok[ith] = ok;
    tp_hier_barrier(&g_comm, sid, lid, &st);

    for (int s = 0; s < NSIZE; s++) {
        size_t n = SIZES[s];
        for (int it = 0; it < 500; it++) tp_allreduce_f32(&g_comm, sid, lid, ith, &st, send, recv, n);
        tp_hier_barrier(&g_comm, sid, lid, &st);
        double t0 = now_us();
        for (int it = 0; it < ITERS; it++) tp_allreduce_f32(&g_comm, sid, lid, ith, &st, send, recv, n);
        double t1 = now_us();
        g_lat_us[ith][s] = (t1 - t0) / ITERS;
        tp_hier_barrier(&g_comm, sid, lid, &st);
    }
    return NULL;
}

int main(int argc, char ** argv) {
    for (int i = 0; i < TP_MAX_LOCAL; i++) {
        char path[128]; snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", i);
        FILE * f = fopen(path, "r"); if (!f) break;
        char buf[512]; if (fgets(buf, sizeof buf, f)) parse_cpulist(buf, g_cpu[g_nsock], &g_ncpu[g_nsock]), g_nsock++;
        fclose(f);
    }
    if (g_nsock == 0) { g_nsock = 1; g_ncpu[0] = 8; for (int c = 0; c < 8; c++) g_cpu[0][c] = c; }
    // drop hyperthread siblings: one thread per physical core
    for (int s = 0; s < g_nsock; s++) {
        int w = 0;
        for (int i = 0; i < g_ncpu[s]; i++) if (tp_is_primary_core(g_cpu[s][i])) g_cpu[s][w++] = g_cpu[s][i];
        g_ncpu[s] = w;
    }
    g_K   = (argc > 1) ? atoi(argv[1]) : g_ncpu[0];
    if (g_K < 1) g_K = 1;
    g_nth = g_nsock * g_K;

    g_scratch = aligned_alloc(64, MAXN * sizeof(float));
    for (int s = 0; s < g_nsock; s++) { g_send[s] = aligned_alloc(64, MAXN * sizeof(float)); g_recv[s] = aligned_alloc(64, MAXN * sizeof(float)); }
    tp_comm_init_local(&g_comm, g_nsock, g_K, g_scratch, MAXN);

    printf("# L1 parallel all-reduce: n_sock=%d  K=%d threads/socket  nth=%d  iters=%d\n", g_nsock, g_K, g_nth, ITERS);

    pthread_t th[TP_MAX_LOCAL * 512];
    targ    * ta = malloc(sizeof(targ) * g_nth);
    int      idx = 0;
    for (int s = 0; s < g_nsock; s++)
        for (int l = 0; l < g_K; l++) { ta[idx].sid = s; ta[idx].lid = l; ta[idx].ith = idx; pthread_create(&th[idx], NULL, worker, &ta[idx]); idx++; }
    for (int i = 0; i < g_nth; i++) pthread_join(th[i], NULL);

    int all_ok = 1; for (int i = 0; i < g_nth; i++) all_ok &= g_ok[i];
    printf("# correctness: %s\n", all_ok ? "PASS" : "FAIL");
    printf("# %-8s %-10s %-12s   per-token comms (2 x L=80)\n", "floats", "bytes", "lat_us(max)");
    for (int s = 0; s < NSIZE; s++) {
        double lat = 0; for (int i = 0; i < g_nth; i++) if (g_lat_us[i][s] > lat) lat = g_lat_us[i][s];
        printf("  %-8zu %-10zu %-12.3f   %.3f ms/token\n", SIZES[s], SIZES[s] * sizeof(float), lat, lat * 2.0 * 80.0 / 1000.0);
    }
    return 0;
}
