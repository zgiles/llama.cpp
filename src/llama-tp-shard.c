#include "llama-tp-shard.h"
#include <stdlib.h>

tp_shard_config tp_shard_from_env(void) {
    tp_shard_config c = {1, 0, 0};
    const char * s = getenv("LLAMA_TP_SIZE");
    const char * r = getenv("LLAMA_TP_RANK");
    if (s) c.size = atoi(s);
    if (r) c.rank = atoi(r);
    if (c.size < 1) c.size = 1;
    if (c.rank < 0 || c.rank >= c.size) c.rank = 0;
    c.enabled = (c.size > 1);
    return c;
}

// bytes occupied by a row of `n` elements of a type with `block` elems/block, `type_size` bytes/block.
static size_t row_bytes(int64_t n, int64_t block, size_t type_size) {
    return (size_t)(n / block) * type_size;
}

int tp_shard_plan_make(tp_shard_role role, int rank, int size,
                       int64_t ne0_full, int64_t ne1_full,
                       int64_t block, size_t type_size,
                       tp_shard_plan * out) {
    if (size <= 1 || role == TP_SHARD_NONE) {
        out->ne0 = ne0_full; out->ne1 = ne1_full; out->nrows = ne1_full;
        out->chunk_bytes = row_bytes(ne0_full, block, type_size);
        out->base_off = 0; out->src_stride = out->chunk_bytes;
        out->total_bytes = (size_t)ne1_full * out->chunk_bytes;
        return 0;
    }

    if (role == TP_SHARD_COLUMN) {
        // split ne[1] (output rows). Whole rows -> always block-safe. Single contiguous read.
        if (ne1_full % size != 0) return -1;
        int64_t ne1_s = ne1_full / size;
        size_t  rb    = row_bytes(ne0_full, block, type_size);
        out->ne0 = ne0_full; out->ne1 = ne1_s;
        out->nrows = 1;
        out->chunk_bytes = (size_t)ne1_s * rb;
        out->base_off = (size_t)rank * ne1_s * rb;
        out->src_stride = 0;
        out->total_bytes = out->chunk_bytes;
        return 0;
    }

    // TP_SHARD_ROW: split ne[0] (the contraction). Must be block-aligned per shard.
    if (ne0_full % (size * block) != 0) return -1;   // would cut a quant block -> corruption
    int64_t ne0_s = ne0_full / size;
    size_t  full_rb  = row_bytes(ne0_full, block, type_size);
    size_t  shard_rb = row_bytes(ne0_s,    block, type_size);
    out->ne0 = ne0_s; out->ne1 = ne1_full;
    out->nrows = ne1_full;                 // one strided chunk per output row
    out->chunk_bytes = shard_rb;
    out->base_off = (size_t)rank * shard_rb;
    out->src_stride = full_rb;
    out->total_bytes = (size_t)ne1_full * shard_rb;
    return 0;
}
