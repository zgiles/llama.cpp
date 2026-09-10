#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#if defined(__gnu_linux__)
#include <unistd.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#define GGML_NUMA_MIRROR_MAX_NODES 8
#define GGML_NUMA_MIRROR_MAX_BUFS  1024
#define GGML_NUMA_MIRROR_MAX_CPUS  1024

// mbind(2) — we call it directly rather than linking libnuma, which ggml does not otherwise need.
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE 2
#endif

struct ggml_numa_mirror_buf {
    void * base;
    size_t size;
    void * copy[GGML_NUMA_MIRROR_MAX_NODES];
};

static struct {
    int    initialized;   // 0 = not yet probed
    int    enabled;
    int    n_nodes;
    int    n_bufs;
    struct ggml_numa_mirror_buf bufs[GGML_NUMA_MIRROR_MAX_BUFS];
    // cpu -> node, built once at init. sched_getcpu() is vDSO-backed; SYS_getcpu via syscall() is
    // not, and this is on the per-op path for every thread.
    signed char cpu_node[GGML_NUMA_MIRROR_MAX_CPUS];
} g_mirror = {0};

// parse a Linux cpulist ("0,2,4-8") into the cpu -> node table
static void ggml_numa_mirror_parse_cpulist(const char * s, int node) {
    while (*s) {
        char * end;
        long a = strtol(s, &end, 10);
        if (end == s) { break; }
        long b = a;
        s = end;
        if (*s == '-') {
            s++;
            b = strtol(s, &end, 10);
            if (end == s) { break; }
            s = end;
        }
        for (long c = a; c <= b && c < GGML_NUMA_MIRROR_MAX_CPUS; c++) {
            g_mirror.cpu_node[c] = (signed char) node;
        }
        if (*s == ',') { s++; }
    }
}

static int ggml_numa_mirror_count_nodes(void) {
    int n = 0;
    for (;; ++n) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", n);
        if (access(path, F_OK) != 0) {
            break;
        }
        if (n + 1 >= GGML_NUMA_MIRROR_MAX_NODES) {
            return GGML_NUMA_MIRROR_MAX_NODES;
        }
    }
    return n;
}

static void ggml_numa_mirror_init(void) {
    if (g_mirror.initialized) {
        return;
    }
    g_mirror.initialized = 1;

    const char * env = getenv("GGML_NUMA_MIRROR");
    if (!env || atoi(env) == 0) {
        return;
    }
    g_mirror.n_nodes = ggml_numa_mirror_count_nodes();
    if (g_mirror.n_nodes < 2) {
        GGML_LOG_INFO("%s: only %d NUMA node(s), mirroring disabled\n", __func__, g_mirror.n_nodes);
        return;
    }
    memset(g_mirror.cpu_node, -1, sizeof(g_mirror.cpu_node));
    for (int n = 0; n < g_mirror.n_nodes; n++) {
        char path[128], buf[4096];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", n);
        FILE * f = fopen(path, "r");
        if (!f) { continue; }
        if (fgets(buf, sizeof(buf), f)) {
            ggml_numa_mirror_parse_cpulist(buf, n);
        }
        fclose(f);
    }
    g_mirror.enabled = 1;
    GGML_LOG_INFO("%s: NUMA weight mirroring enabled across %d nodes\n", __func__, g_mirror.n_nodes);
}

bool ggml_numa_mirror_enabled(void) {
    ggml_numa_mirror_init();
    return g_mirror.enabled != 0;
}

// which NUMA node is the calling thread running on right now
static inline int ggml_numa_mirror_this_node(void) {
    const int cpu = sched_getcpu();
    if (cpu < 0 || cpu >= GGML_NUMA_MIRROR_MAX_CPUS) {
        return -1;
    }
    return g_mirror.cpu_node[cpu];
}

// allocate `size` bytes bound to `node`, and copy `src` into it
static void * ggml_numa_mirror_alloc_on(int node, const void * src, size_t size) {
    void * p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        return NULL;
    }
    // ask for huge pages: distros commonly ship THP as "madvise", where it is never applied unless
    // requested. Advisory — if unavailable the kernel just keeps 4 KiB pages.
    (void) madvise(p, size, MADV_HUGEPAGE);

    // bind BEFORE touching, so the pages fault in on the node we want. mbind on an untouched
    // anonymous mapping sets the policy; the copy below then populates it locally.
    unsigned long nodemask[GGML_NUMA_MIRROR_MAX_NODES / (8 * sizeof(unsigned long)) + 1] = {0};
    nodemask[node / (8 * sizeof(unsigned long))] |= 1UL << (node % (8 * sizeof(unsigned long)));
    if (syscall(SYS_mbind, p, size, MPOL_BIND, nodemask,
                (unsigned long) GGML_NUMA_MIRROR_MAX_NODES + 1, MPOL_MF_MOVE) != 0) {
        // not fatal: without a binding this copy is just placed by first touch, which is still
        // correct, only not necessarily local. Report it so a silent perf loss is visible.
        GGML_LOG_WARN("%s: mbind to node %d failed, copy may not be node-local\n", __func__, node);
    }

    memcpy(p, src, size);
    return p;
}

bool ggml_numa_mirror_register(void * base, size_t size) {
    ggml_numa_mirror_init();
    if (!g_mirror.enabled || base == NULL || size == 0) {
        return false;
    }
    for (int i = 0; i < g_mirror.n_bufs; i++) {
        if (g_mirror.bufs[i].base == base) {
            return true; // already mirrored
        }
    }
    if (g_mirror.n_bufs >= GGML_NUMA_MIRROR_MAX_BUFS) {
        GGML_LOG_WARN("%s: too many mirrored buffers, skipping\n", __func__);
        return false;
    }

    struct ggml_numa_mirror_buf * b = &g_mirror.bufs[g_mirror.n_bufs];
    memset(b, 0, sizeof(*b));
    b->base = base;
    b->size = size;

    // Every node gets its own freshly-allocated, mbind-ed copy — including node 0.
    //
    // The tempting alternative is to migrate the ORIGINAL allocation onto node 0 with
    // mbind(MPOL_MF_MOVE) and use it as node 0's copy, saving one copy. That is implemented and
    // works in isolation (verified standalone, both for a whole allocation and for many
    // sub-ranges of one), but in the real load path it silently did not migrate: the split stayed
    // at N0=37/N1=83 instead of the balanced ~60/60 it should produce. The cause is still unknown,
    // and ggml's log output is swallowed by llama.cpp's log callback so the errno was never
    // visible. Correctness first: allocate per node, which is measured and verified balanced.
    // TODO: revisit the in-place migration — it would cut resident memory by one full copy.
    for (int n = 0; n < g_mirror.n_nodes; n++) {
        b->copy[n] = ggml_numa_mirror_alloc_on(n, base, size);
        if (!b->copy[n]) {
            GGML_LOG_WARN("%s: failed to allocate %.1f GiB mirror on node %d; "
                          "falling back to the shared copy\n",
                          __func__, size / (double) (1u << 30), n);
            for (int k = 0; k < n; k++) {
                munmap(b->copy[k], size);
            }
            return false;
        }
    }
    g_mirror.n_bufs++;
    GGML_LOG_INFO("%s: mirrored %.2f GiB across %d nodes \n",
                  __func__, size / (double) (1u << 30), g_mirror.n_nodes);
    return true;
}

const void * ggml_numa_mirror_local(const void * p) {
    if (!g_mirror.enabled || p == NULL) {
        return p;
    }
    const int node = ggml_numa_mirror_this_node();
    if (node < 0 || node >= g_mirror.n_nodes) {
        return p;
    }
    for (int i = 0; i < g_mirror.n_bufs; i++) {
        const struct ggml_numa_mirror_buf * b = &g_mirror.bufs[i];
        const size_t off = (const char *) p - (const char *) b->base;
        if (off < b->size && b->copy[node]) {
            return (const char *) b->copy[node] + off;
        }
    }
    return p;
}

void ggml_numa_mirror_free_all(void) {
    for (int i = 0; i < g_mirror.n_bufs; i++) {
        for (int n = 0; n < g_mirror.n_nodes; n++) {
            if (g_mirror.bufs[i].copy[n]) {
                munmap(g_mirror.bufs[i].copy[n], g_mirror.bufs[i].size);
            }
        }
    }
    g_mirror.n_bufs = 0;
}

#else // not linux

bool ggml_numa_mirror_enabled(void) { return false; }
bool ggml_numa_mirror_register(void * base, size_t size) { GGML_UNUSED(base); GGML_UNUSED(size); return false; }
const void * ggml_numa_mirror_local(const void * p) { return p; }
void ggml_numa_mirror_free_all(void) {}

#endif
