// Unit test for tp_shard offset math. No ggml/UCX deps.
//   gcc -O2 -std=c11 tp_shard_test.c tp_shard.c -o tp_shard_test && ./tp_shard_test
#include "tp_shard.h"
#include <stdio.h>

static int fails = 0;
#define CHECK(cond, ...) do { if(!(cond)){ printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while(0)

int main(void) {
    // Q4_K: 256 elems/block, 144 bytes/block.  Q8_0: 32 elems/block, 34 bytes/block.
    const int64_t Q4K_B = 256; const size_t Q4K_TS = 144;
    const int64_t Q80_B = 32;  const size_t Q80_TS = 34;

    // FFN-ish tensor. ffn_up: [n_embd=4096, n_ff=14336]. column-parallel split ne1 across 2.
    tp_shard_plan p;
    int rc = tp_shard_plan_make(TP_SHARD_COLUMN, 0, 2, 4096, 14336, Q4K_B, Q4K_TS, &p);
    CHECK(rc==0, "column rc");
    CHECK(p.ne0==4096 && p.ne1==7168, "column dims %lld,%lld", (long long)p.ne0,(long long)p.ne1);
    CHECK(p.nrows==1, "column nrows %lld",(long long)p.nrows);
    size_t rb = (4096/256)*144;                       // 2304
    CHECK(p.chunk_bytes==(size_t)7168*rb, "column chunk");
    CHECK(p.base_off==0, "column rank0 off");
    tp_shard_plan_make(TP_SHARD_COLUMN, 1, 2, 4096, 14336, Q4K_B, Q4K_TS, &p);
    CHECK(p.base_off==(size_t)7168*rb, "column rank1 off %zu", p.base_off);
    // rank0+rank1 chunks must tile the whole tensor exactly
    CHECK(p.base_off + p.chunk_bytes == (size_t)14336*rb, "column coverage");

    // ffn_down: [n_ff=14336, n_embd=4096]. row-parallel split ne0 across 2.
    rc = tp_shard_plan_make(TP_SHARD_ROW, 1, 2, 14336, 4096, Q4K_B, Q4K_TS, &p);
    CHECK(rc==0, "row rc");
    CHECK(p.ne0==7168 && p.ne1==4096, "row dims %lld,%lld",(long long)p.ne0,(long long)p.ne1);
    CHECK(p.nrows==4096, "row nrows %lld",(long long)p.nrows);
    size_t full_rb = (14336/256)*144;                 // 8064
    size_t shard_rb = (7168/256)*144;                 // 4032
    CHECK(p.chunk_bytes==shard_rb, "row chunk %zu", p.chunk_bytes);
    CHECK(p.src_stride==full_rb, "row stride %zu", p.src_stride);
    CHECK(p.base_off==shard_rb, "row rank1 base %zu", p.base_off);
    CHECK(p.total_bytes==(size_t)4096*shard_rb, "row total");

    // block-alignment guard: ne0 not a multiple of size*block must be REFUSED.
    rc = tp_shard_plan_make(TP_SHARD_ROW, 0, 2, 14000, 4096, Q4K_B, Q4K_TS, &p);
    CHECK(rc < 0, "row misalignment should fail (rc=%d)", rc);
    // Q8_0 smaller blocks: 14336 % (2*32)=0 ok
    rc = tp_shard_plan_make(TP_SHARD_ROW, 0, 2, 14336, 4096, Q80_B, Q80_TS, &p);
    CHECK(rc==0 && p.ne0==7168, "q80 row ok");
    // 3-way where ne1 not divisible -> column refuse
    rc = tp_shard_plan_make(TP_SHARD_COLUMN, 0, 3, 4096, 14336, Q4K_B, Q4K_TS, &p);
    CHECK(rc < 0, "column 14336%%3 should fail");

    printf(fails ? "\n%d CHECK(S) FAILED\n" : "ALL CHECKS PASSED\n", fails);
    return fails ? 1 : 0;
}
