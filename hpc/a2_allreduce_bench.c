// A2: small-message all-reduce latency benchmark (the number that gates cross-node TP).
// Measures MPI_Allreduce latency over message sizes spanning the TP activation regime
// (hidden_size * batch * dtype ~ a few KB to tens of KB). Build with the nix openmpi mpicc.
//
//   mpicc -O2 -o a2_allreduce_bench a2_allreduce_bench.c
//   mpirun --host 172.16.0.121,172.16.0.124 -np 2 \
//     -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc,sm,self ./a2_allreduce_bench
//
// Reports per-size avg (max across ranks) and min latency in microseconds.
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int iters = 2000, skip = 200;
    // bytes: 1K..256K, brackets the per-layer activation all-reduce for hidden 2k..32k
    size_t sizes[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    char name[MPI_MAX_PROCESSOR_NAME];
    int nlen;
    MPI_Get_processor_name(name, &nlen);
    if (rank == 0) {
        printf("# all-reduce latency  ranks=%d\n", size);
        printf("# %-10s %-12s %-12s\n", "bytes", "avg_us", "min_us");
    }
    fflush(stdout);

    for (int s = 0; s < nsizes; s++) {
        size_t bytes = sizes[s];
        size_t n = bytes / sizeof(float);
        float *sbuf = malloc(bytes);
        float *rbuf = malloc(bytes);
        for (size_t i = 0; i < n; i++) sbuf[i] = 1.0f;

        double tot = 0.0, mn = 1e30;
        for (int i = 0; i < iters + skip; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            MPI_Allreduce(sbuf, rbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            if (i >= skip) {
                double us = (t1 - t0) * 1e6;
                tot += us;
                if (us < mn) mn = us;
            }
        }
        double avg = tot / iters, gavg, gmin;
        MPI_Reduce(&avg, &gavg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mn, &gmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("  %-10zu %-12.2f %-12.2f\n", bytes, gavg, gmin);
            fflush(stdout);
        }
        free(sbuf);
        free(rbuf);
    }
    MPI_Finalize();
    return 0;
}
