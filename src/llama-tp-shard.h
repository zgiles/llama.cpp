// Load-time weight sharding math for CPU tensor parallelism.
// The crux of loader sharding is computing, per rank, the sharded tensor shape and the
// byte layout of this rank's slice inside the GGUF tensor data — correctly for QUANTIZED
// weights (block-aligned), including the strided per-row read needed for row-parallel.
//
// ggml weight layout: ne[0]=contraction (cols), ne[1]=output rows; row-major; a row of n
// elements occupies ggml_row_size(type,n) = type_size * (n / block_size) bytes.
//
//   Column-parallel (wq/wk/wv/ffn_up/ffn_gate): split ne[1] (whole output rows). One
//     contiguous read. Always block-safe (rows are whole).
//   Row-parallel (wo/ffn_down): split ne[0] (the contraction). Each output row contributes
//     a sub-range; the rank's data is STRIDED (one chunk per row). Requires ne0/size to be a
//     multiple of the quant block size, else it cuts a block and corrupts the weights.
#ifndef TP_SHARD_H
#define TP_SHARD_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { TP_SHARD_NONE = 0, TP_SHARD_COLUMN, TP_SHARD_ROW } tp_shard_role;

typedef struct { int size; int rank; int enabled; int attn; } tp_shard_config;

// Reads LLAMA_TP_SIZE and LLAMA_TP_RANK from the environment. enabled = (size > 1).
// attn = (enabled && LLAMA_TP_ATTN); when set, attention (wq/wk/wv/wo) is sharded too
// (Phase 2). When unset, only the FFN is sharded (Phase 1, attention replicated).
tp_shard_config tp_shard_from_env(void);

// A load plan: `nrows` chunks of `chunk_bytes`, the i-th read from
//   src: base_off + i*src_stride   ->   dst: i*chunk_bytes (contiguous).
typedef struct {
    int64_t ne0, ne1;     // sharded dimensions
    int64_t nrows;        // number of chunks (1 for column-parallel; ne1 for row-parallel)
    size_t  chunk_bytes;  // bytes per chunk
    size_t  base_off;     // byte offset of first chunk within the source tensor data
    size_t  src_stride;   // source stride between chunks
    size_t  total_bytes;  // == nrows * chunk_bytes (the sharded tensor size)
} tp_shard_plan;

// Compute the sharded shape + load plan. block = ggml_blck_size(type) (elements/block),
// type_size = ggml_type_size(type) (bytes/block). Returns 0 on success, <0 on a divisibility
// / block-alignment violation (caller should refuse to shard and fall back).
int tp_shard_plan_make(tp_shard_role role, int rank, int size,
                       int64_t ne0_full, int64_t ne1_full,
                       int64_t block, size_t type_size,
                       tp_shard_plan * out);

#ifdef __cplusplus
}
#endif

#endif // TP_SHARD_H
