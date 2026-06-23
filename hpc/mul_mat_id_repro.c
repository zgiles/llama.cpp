// Standalone repro: does ggml_mul_mat_id give the SAME result when src0 (the expert tensor)
// is the FULL [K,N,E_full] tensor vs a SLICE [K,N,E_slice] containing the first E_slice experts?
//
// The ids reference only experts < E_slice, so for those experts the data is byte-identical in
// both tensors. Therefore mul_mat_id(full, src1, ids) and mul_mat_id(slice, src1, ids) MUST give
// identical outputs for every token. If they differ (esp. for token >= 1), that is a ggml bug
// exposed purely by slicing the n_expert (ne[2]) dimension — no llama.cpp / TP code involved.
//
// Build (on a node with the llama.cpp build tree):
//   gcc -O2 hpc/mul_mat_id_repro.c -I ggml/include -I ggml/src \
//       build-native/src/libggml*.a ... (see hpc/build_repro.sh)
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float frand(void) { return (float) rand() / (float) RAND_MAX - 0.5f; }

// s1ne1: src1->ne[1]. =1 mimics the up/gate projection (b=[K,1,T]); =n_used mimics the DOWN
// projection (b=[K,n_used,T], one input row per expert slot).
static void run_case2(enum ggml_type type, const char * tname,
                      int K, int N, int E_full, int E_slice,
                      int n_tok, int n_used, int nth, int distinct, int s1ne1) {
    struct ggml_init_params p = { (size_t) 1024*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(p);

    const int64_t nrows_full = (int64_t) N * E_full;
    float * w = malloc((size_t) K * nrows_full * sizeof(float));
    for (size_t i = 0; i < (size_t) K * nrows_full; i++) w[i] = frand();

    struct ggml_tensor * s0_full  = ggml_new_tensor_3d(ctx, type, K, N, E_full);
    struct ggml_tensor * s0_slice = ggml_new_tensor_3d(ctx, type, K, N, E_slice);
    if (type == GGML_TYPE_F32) {
        memcpy(s0_full->data,  w, (size_t) K * nrows_full * sizeof(float));
        memcpy(s0_slice->data, w, (size_t) K * (int64_t) N * E_slice * sizeof(float));
    } else {
        ggml_quantize_chunk(type, w, s0_full->data,  0, nrows_full,            K, NULL);
        ggml_quantize_chunk(type, w, s0_slice->data, 0, (int64_t) N * E_slice, K, NULL);
    }

    struct ggml_tensor * s1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, s1ne1, n_tok);
    float * s1d = (float *) s1->data;
    for (size_t i = 0; i < (size_t) K * s1ne1 * n_tok; i++) s1d[i] = frand();

    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    int32_t * idd = (int32_t *) ids->data;
    // experts referenced are all < E_slice. distinct=0 -> same expert (5) everywhere;
    // distinct=1 -> a spread of distinct low experts per (slot,token), like the real case.
    for (int t = 0; t < n_tok; t++)
        for (int u = 0; u < n_used; u++)
            idd[t*n_used + u] = distinct ? ((u*13 + t*7 + 3) % E_slice) : 5;

    struct ggml_tensor * r_full  = ggml_mul_mat_id(ctx, s0_full,  s1, ids); // [N, n_used, n_tok]
    struct ggml_tensor * r_slice = ggml_mul_mat_id(ctx, s0_slice, s1, ids);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r_full);
    ggml_build_forward_expand(gf, r_slice);
    ggml_graph_compute_with_ctx(ctx, gf, nth);

    const float * of = (const float *) r_full->data;
    const float * os = (const float *) r_slice->data;
    printf("[%-4s K=%d N=%d E:%d->%d ntok=%d nused=%d nth=%d s1ne1=%d %s]\n",
           tname, K, N, E_full, E_slice, n_tok, n_used, nth, s1ne1, distinct ? "distinct-ids" : "same-id");
    double worst = 0;
    for (int t = 0; t < n_tok; t++) {
        double td = 0;
        for (int u = 0; u < n_used; u++)
            for (int n = 0; n < N; n++) {
                int idx = n + u*N + t*N*n_used;
                double d = fabs((double) of[idx] - (double) os[idx]);
                if (d > td) td = d;
            }
        printf("   token %d: full[0,0]=%+.5f slice[0,0]=%+.5f  maxdiff=%.3g %s\n",
               t, of[t*N*n_used], os[t*N*n_used], td, td > 1e-4 ? "<-- MISMATCH" : "ok");
        if (td > worst) worst = td;
    }
    printf("   => %s (worst diff %.3g)\n\n", worst > 1e-4 ? "*** BUG: full != slice ***" : "OK: full == slice", worst);
    free(w);
    ggml_free(ctx);
}

static void run_case(enum ggml_type type, const char * tname, int K, int N, int E_full,
                     int E_slice, int n_tok, int n_used, int nth, int distinct) {
    run_case2(type, tname, K, N, E_full, E_slice, n_tok, n_used, nth, distinct, 1);
}

int main(void) {
    srand(1234);
    // --- up/gate projection style: src1->ne[1] = 1 ---
    run_case(GGML_TYPE_F32,  "F32",  256, 32, 256, 128, 2, 1, 4, 0);
    run_case(GGML_TYPE_Q8_0, "Q8_0", 256, 32, 256, 128, 2, 1, 4, 0);
    run_case(GGML_TYPE_Q4_K, "Q4_K", 256, 32, 256, 128, 5, 8, 4, 1);
    // --- DOWN projection style: src1->ne[1] = n_used (the untested case!) ---
    run_case2(GGML_TYPE_F32,  "F32-dn",  256, 32, 256, 128, 5, 8, 4, 1, 8);
    run_case2(GGML_TYPE_Q8_0, "Q8_0-dn", 256, 32, 256, 128, 5, 8, 4, 1, 8);
    run_case2(GGML_TYPE_Q4_K, "Q4_K-dn", 256, 32, 256, 128, 5, 8, 4, 1, 8);
    run_case2(GGML_TYPE_Q4_K, "Q4_K-dn", 256, 32, 256, 128, 5, 8, 1, 1, 8);  // single-thread
    return 0;
}
