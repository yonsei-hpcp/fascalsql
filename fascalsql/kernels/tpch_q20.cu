// TPC-H Q20 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Rewritten: CPU-side lineitem_sum + partsupp pre-qualification,
//            GPU supplier scan with nation + qualifying-suppkey semi-joins.
//
// Q20 SQL:
//   SELECT s_name, s_address FROM supplier, nation
//   WHERE s_suppkey IN (
//       SELECT ps_suppkey FROM partsupp
//       WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
//         AND ps_availqty > 0.5 * (SELECT SUM(l_quantity) FROM lineitem
//                                   WHERE l_partkey=ps_partkey AND l_suppkey=ps_suppkey
//                                     AND l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01')
//   )
//   AND s_nationkey = n_nationkey AND n_name = 'CANADA'
//   ORDER BY s_name;

#define CUB_STDERR
#define FASCAL_TILE_SIZE 512

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <immintrin.h>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <cstring>
#include <limits>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <cuda/atomic>
#include "fascal/runtime/odzc.hpp"
#include "fascal/runtime/afp.hpp"
#include "fascal/optimizer/cqo.hpp"
#include "gpu_hashtable.cuh"
#include "fascal_generated_common.cuh"
#include "fascal_gpu_macros.cuh"
#include "fascal_query_runtime.cuh"
#include "fascal_cpu_pred_helpers.hpp"

using namespace std;

// --- Query configuration ---
#define BLOCK_THREADS 128
#define ITEMS_PER_THREAD 4
#define TILE_SIZE 512
#define MAX_PREDICATES 16

// Ablation flags
__device__ __constant__ unsigned char d_pred_flags[MAX_PREDICATES];
static unsigned char h_pred_flags[MAX_PREDICATES] = {0};

// ---- GPU SQL LIKE Matching ----
__device__ bool sql_like_match(const char* s, const char* p) {
    const char* cp = nullptr;
    const char* cs = nullptr;
    while (*s && *p != '*') {
        if (*p == '%') {
            while (*p == '%') p++;
            if (!*p) return true;
            cp = p;
            cs = s + 1;
        } else if (*p == '_' || *p == *s) {
            p++;
            s++;
        } else if (cp) {
            p = cp;
            s = cs++;
        } else {
            return false;
        }
    }
    while (*p == '%') p++;
    return !*p && !*s;
}

__global__ void regex_bitset_kernel(const char* dict_strings, const int* dict_offsets,
                                     int num_entries, const char* pattern, unsigned char* bitset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_entries) {
        const char* s = dict_strings + dict_offsets[idx];
        bitset[idx] = sql_like_match(s, pattern) ? 1 : 0;
    }
}

// =============================================================================
// Stage 2 GPU kernel: scan supplier, join nation + qualifying_suppkey
// =============================================================================

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q20_kernel_supplier(
    int num_tuples,
    int batch_offset,
    int *d_supplier_s_name,
    int *d_supplier_s_address,
    int *d_supplier_s_nationkey,
    int *d_supplier_s_suppkey,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_nation,
    uint32_t bloom_size_bits_nation,
    HtEntry *ht_nation,
    uint32_t ht_size_nation,
    const uint64_t *__restrict__ d_bloom_qual,
    uint32_t bloom_size_bits_qual,
    HtEntry *ht_qual,
    uint32_t ht_size_qual,
    int *d_out_count,
    int *d_out_col0,
    int *d_out_col1)
{
    FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

    int loc_s_nationkey[ITEMS_PER_THREAD_T];
    int loc_s_suppkey[ITEMS_PER_THREAD_T];
    int loc_s_name[ITEMS_PER_THREAD_T];
    int loc_s_address[ITEMS_PER_THREAD_T];

    if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {

    // Phase 1: Bloom filter probes
    // Load s_nationkey and probe nation BF
    BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        d_supplier_s_nationkey + tile_offset, loc_s_nationkey, selection_flags, num_tile_items);
    BlockBloomProbe<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        loc_s_nationkey, selection_flags, num_tile_items,
        bloom_size_bits_nation, d_bloom_nation, 3);

    // Load s_suppkey and probe qualifying-suppkey BF
    BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        d_supplier_s_suppkey + tile_offset, loc_s_suppkey, selection_flags, num_tile_items);
    BlockBloomProbe<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        loc_s_suppkey, selection_flags, num_tile_items,
        bloom_size_bits_qual, d_bloom_qual, 3);

    FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

    // Phase 2: HT joins
    // Join 0: s_nationkey -> nation (inner)
    BlockJoinProbeDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        loc_s_nationkey, selection_flags, num_tile_items,
        ht_nation, ht_size_nation, 0);

    // Join 1: s_suppkey -> qualifying_suppkey (semi-join)
    BlockJoinProbeSemi<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        loc_s_suppkey, selection_flags, num_tile_items,
        ht_qual, ht_size_qual);

    // Load output columns
    BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        d_supplier_s_name + tile_offset, loc_s_name, selection_flags, num_tile_items);
    BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        d_supplier_s_address + tile_offset, loc_s_address, selection_flags, num_tile_items);

    // Materialize output
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
            if (selection_flags[ITEM]) {
                int idx = atomicAdd(d_out_count, 1);
                d_out_col0[idx] = loc_s_name[ITEM];
                d_out_col1[idx] = loc_s_address[ITEM];
            }
    }
    } while(0);
}

// =============================================================================
// Nation build kernel (filter n_name == CANADA, insert into HT+BF)
// =============================================================================

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q20_kernel_nation(
    int num_tuples,
    const int *__restrict__ d_nation_n_nationkey,
    const int *__restrict__ d_nation_n_name,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
{
    FASCAL_BUILD_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
    int loc_nkey[ITEMS_PER_THREAD_T];
    int loc_nname[ITEMS_PER_THREAD_T];

    BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        (int*)(d_nation_n_nationkey + tile_offset), loc_nkey, selection_flags, num_tile_items);
    BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        (int*)(d_nation_n_name + tile_offset), loc_nname, selection_flags, num_tile_items);

    // n_name == 3 (CANADA)
    BlockPredAndEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_nname, 3, selection_flags, num_tile_items);

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
        if (!selection_flags[ITEM]) continue;
        int payload = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
        gpu_ht_insert_one_direct(loc_nkey[ITEM], payload, 1, 0, d_ht, ht_size);
        gpu_bloom_set_one(loc_nkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
    }
}

// Nation CPU prefilter
static void tpch_q20_cpu_prefilter_nation(
    int *h_n_nationkey, int *h_n_name, int num_tuples,
    uint32_t *bitmap, uint8_t *tile_summary) {
    fascal_cpu_prefilter_run(num_tuples, bitmap, tile_summary,
        [&](int offset, int cnt) {
            uint32_t word_start = (uint32_t)offset >> 5;
            uint32_t word_end   = ((uint32_t)(offset + cnt) + 31) >> 5;
            memset(&bitmap[word_start], 0, (word_end - word_start) * sizeof(uint32_t));
            for (int i = 0; i < cnt; ++i) {
                int idx = offset + i;
                if (h_n_name[idx] == 3)
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
            }
        });
}

// =============================================================================
// Qualifying-suppkey build kernel: bulk-insert pre-computed qualifying suppkeys
// =============================================================================

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q20_kernel_qual_suppkey(
    int num_tuples,
    const int *__restrict__ d_qual_suppkeys,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes)
{
    int tile_offset = blockIdx.x * (BLOCK_THREADS_T * ITEMS_PER_THREAD_T);
    int num_tile_items = BLOCK_THREADS_T * ITEMS_PER_THREAD_T;
    if ((tile_offset + BLOCK_THREADS_T * ITEMS_PER_THREAD_T) > num_tuples)
        num_tile_items = num_tuples - tile_offset;

    int loc_key[ITEMS_PER_THREAD_T];

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
        int idx = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
        if (idx < num_tuples)
            loc_key[ITEM] = d_qual_suppkeys[idx];
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
        int idx = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
        if (idx >= num_tuples) break;
        unsigned long long key = static_cast<unsigned int>(loc_key[ITEM]);
        gpu_ht_insert_one(key, idx, 1, d_ht, ht_size);
        gpu_bloom_set_one(loc_key[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
    }
}

// CPU prefilter for supplier fact table (Bloom probes only, no scalar predicates)
static void tpch_q20_cpu_predicate_supplier(
    int *h_data[], int morsel_offset, int elem_cnt, uint32_t *bitmap,
    int selection_step, int ablation_indicator,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0)
{
    uint8_t *bm = (uint8_t *)bitmap;
    int bs = morsel_offset >> 3, nb = (elem_cnt + 7) >> 3;
    fascal::cpu_pred::init_bitmap(bm, bs, nb, elem_cnt);

    if (bloom_filters && 0 < num_bloom_filters && bloom_filters[0])
        fascal::cpu_pred::bloom_probe_pass(bm, bs, h_data[2], morsel_offset, elem_cnt, bloom_filters[0]);
    if (bloom_filters && 1 < num_bloom_filters && bloom_filters[1])
        fascal::cpu_pred::bloom_probe_pass(bm, bs, h_data[3], morsel_offset, elem_cnt, bloom_filters[1]);
}

static void tpch_q20_cpu_prefilter_supplier(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0, int total_tuples = 0) {
    fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
        [&](int off, int cnt) { tpch_q20_cpu_predicate_supplier(h_data, off, cnt, bitmap, 0, 0, bloom_filters, num_bloom_filters); },
        batch_offset, total_tuples);
}

// =============================================================================
// Stage 1 GPU kernel: filter lineitem by shipdate, compact (packed_key, qty)
// =============================================================================

__global__ void tpch_q20_lineitem_filter_compact(
    int num_tuples,
    const int *__restrict__ d_l_partkey,
    const int *__restrict__ d_l_suppkey,
    const int *__restrict__ d_l_shipdate,
    const float *__restrict__ d_l_quantity,
    unsigned long long *__restrict__ d_out_keys,
    float *__restrict__ d_out_vals,
    int *__restrict__ d_compact_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tuples) return;
    int sd = d_l_shipdate[idx];
    if (sd >= 19940101 && sd < 19950101) {
        int pos = atomicAdd(d_compact_count, 1);
        unsigned long long pk = (unsigned long long)(unsigned int)d_l_partkey[idx];
        unsigned long long sk = (unsigned long long)(unsigned int)d_l_suppkey[idx];
        d_out_keys[pos] = (pk << 32) | sk;
        d_out_vals[pos] = d_l_quantity[idx];
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char** argv) {

    const char *data_dir = getenv("FASCALSQL_DATA_DIR");
    if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
    if (argc >= 3) data_dir = argv[2];
    if (!data_dir) { fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n"); return 1; }

    auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(
        new fascal::runtime::ODZCManager(0));

    printf("=== FaScalSQL: tpch_q20 ===\n");
    if (argc < 3) { printf("Usage: %s <num_tuples> <data_dir>\n", argv[0]); return 1; }
    data_dir = argv[2];

    // =========================================================================
    // Stage 1 (GPU): Compute lineitem_sum = SUM(l_quantity) GROUP BY (l_partkey, l_suppkey)
    //                WHERE l_shipdate >= 19940101 AND l_shipdate < 19950101
    //   GPU sort-based: filter+compact → thrust sort_by_key → thrust reduce_by_key
    // =========================================================================
    printf("--- Stage 1: lineitem_sum aggregation (GPU sort-based) ---\n");
    cudaEvent_t _s1_t0, _s1_t1;
    CUDA_CHECK(cudaEventCreate(&_s1_t0)); CUDA_CHECK(cudaEventCreate(&_s1_t1));

    int num_lineitem = 0;
    char path[512];

    snprintf(path, sizeof(path), "%s/l_partkey.bin", data_dir);
    int *h_l_partkey = load_binary_column_auto(path, &num_lineitem);
    if (!h_l_partkey) { fprintf(stderr, "Failed to load l_partkey\n"); return 1; }

    snprintf(path, sizeof(path), "%s/l_suppkey.bin", data_dir);
    int *h_l_suppkey = load_binary_column_auto(path, &num_lineitem);

    snprintf(path, sizeof(path), "%s/l_shipdate.bin", data_dir);
    int *h_l_shipdate = load_binary_column_auto(path, &num_lineitem);

    // l_quantity is stored as IEEE 754 float
    snprintf(path, sizeof(path), "%s/l_quantity.bin", data_dir);
    int *h_l_quantity_raw = load_binary_column_auto(path, &num_lineitem);

    printf("lineitem: %d rows\n", num_lineitem);

    // Lineitem columns are cudaMallocManaged (UVM) — same pointer works on both host and device.
    int *d_l_partkey = h_l_partkey;
    int *d_l_suppkey = h_l_suppkey;
    int *d_l_shipdate = h_l_shipdate;
    float *d_l_quantity = reinterpret_cast<float*>(h_l_quantity_raw);

    // Allocate GPU compaction output buffers (worst case: all lineitem rows qualify)
    unsigned long long *d_compact_keys = nullptr;
    float *d_compact_vals = nullptr;
    int *d_compact_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_compact_keys, (size_t)num_lineitem * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_compact_vals, (size_t)num_lineitem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_compact_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_compact_count, 0, sizeof(int)));

    // Launch filter+compact kernel
    CUDA_CHECK(cudaEventRecord(_s1_t0));
    {
        int threads_per_block = 256;
        int num_blocks_li = (num_lineitem + threads_per_block - 1) / threads_per_block;
        tpch_q20_lineitem_filter_compact<<<num_blocks_li, threads_per_block>>>(
            num_lineitem, d_l_partkey, d_l_suppkey, d_l_shipdate, d_l_quantity,
            d_compact_keys, d_compact_vals, d_compact_count);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
    }

    int h_compact_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_compact_count, d_compact_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("lineitem filter: %d / %d rows qualify (shipdate in [1994,1995))\n", h_compact_count, num_lineitem);

    // Thrust sort_by_key on compacted (key, val) pairs
    {
        thrust::device_ptr<unsigned long long> keys_ptr(d_compact_keys);
        thrust::device_ptr<float> vals_ptr(d_compact_vals);
        thrust::sort_by_key(keys_ptr, keys_ptr + h_compact_count, vals_ptr);
    }

    // Thrust reduce_by_key: SUM(l_quantity) per packed_key
    unsigned long long *d_unique_keys = nullptr;
    float *d_sum_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_unique_keys, (size_t)h_compact_count * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_sum_vals, (size_t)h_compact_count * sizeof(float)));

    int num_groups = 0;
    {
        thrust::device_ptr<unsigned long long> keys_in(d_compact_keys);
        thrust::device_ptr<float> vals_in(d_compact_vals);
        thrust::device_ptr<unsigned long long> keys_out(d_unique_keys);
        thrust::device_ptr<float> vals_out(d_sum_vals);
        auto end_pair = thrust::reduce_by_key(keys_in, keys_in + h_compact_count,
                                               vals_in, keys_out, vals_out);
        num_groups = (int)(end_pair.first - keys_out);
    }
    CUDA_CHECK(cudaEventRecord(_s1_t1));
    CUDA_CHECK(cudaDeviceSynchronize());
    { float _s1_ms = 0; CUDA_CHECK(cudaEventElapsedTime(&_s1_ms, _s1_t0, _s1_t1));
      printf("Kernel: %.3f ms\n", _s1_ms); }
    CUDA_CHECK(cudaEventDestroy(_s1_t0)); CUDA_CHECK(cudaEventDestroy(_s1_t1));
    printf("lineitem_sum: %d unique (partkey, suppkey) groups\n", num_groups);

    // Copy results back to host and build hash map
    std::vector<unsigned long long> h_unique_keys(num_groups);
    std::vector<float> h_sum_vals(num_groups);
    CUDA_CHECK(cudaMemcpy(h_unique_keys.data(), d_unique_keys,
                           (size_t)num_groups * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_vals.data(), d_sum_vals,
                           (size_t)num_groups * sizeof(float), cudaMemcpyDeviceToHost));

    // Build CPU hash map: packed_key → sum_qty (for Stage 1.5 lookup)
    std::unordered_map<unsigned long long, double> lineitem_sum_map;
    lineitem_sum_map.reserve(num_groups);
    for (int i = 0; i < num_groups; ++i)
        lineitem_sum_map[h_unique_keys[i]] = (double)h_sum_vals[i];

    // Free GPU compaction/reduction buffers
    CUDA_CHECK(cudaFree(d_compact_keys));
    CUDA_CHECK(cudaFree(d_compact_vals));
    CUDA_CHECK(cudaFree(d_compact_count));
    CUDA_CHECK(cudaFree(d_unique_keys));
    CUDA_CHECK(cudaFree(d_sum_vals));

    // Free lineitem columns (no longer needed)
    fascal_free_column(h_l_partkey);
    fascal_free_column(h_l_suppkey);
    fascal_free_column(h_l_shipdate);
    fascal_free_column(h_l_quantity_raw);

    // =========================================================================
    // Stage 1.5 (CPU): Identify forest% parts, then find qualifying suppkeys
    //   For each partsupp row:
    //     if ps_partkey IN forest_parts
    //     AND (ps_partkey, ps_suppkey) IN lineitem_sum
    //     AND ps_availqty > 0.5 * sum_qty
    //     => ps_suppkey qualifies
    // =========================================================================
    printf("--- Stage 1.5: qualifying suppkeys (CPU) ---\n");

    // Load p_name binary dict and find forest% entries
    // Binary dict: offsets.bin (int32 array) + strings.bin (null-terminated concatenated strings)
    snprintf(path, sizeof(path), "%s/part_p_name_dict_offsets.bin", data_dir);
    int dict_count = 0;
    int *dict_offsets = load_binary_column_auto(path, &dict_count);
    if (!dict_offsets) {
        // Fallback: try p_name_dict_offsets.bin
        snprintf(path, sizeof(path), "%s/p_name_dict_offsets.bin", data_dir);
        dict_offsets = load_binary_column_auto(path, &dict_count);
    }
    if (!dict_offsets) { fprintf(stderr, "Failed to load p_name dict offsets\n"); return 1; }

    snprintf(path, sizeof(path), "%s/part_p_name_dict_strings.bin", data_dir);
    FILE* fp_str = fopen(path, "rb");
    if (!fp_str) {
        snprintf(path, sizeof(path), "%s/p_name_dict_strings.bin", data_dir);
        fp_str = fopen(path, "rb");
    }
    if (!fp_str) { fprintf(stderr, "Failed to load p_name dict strings\n"); return 1; }
    fseek(fp_str, 0, SEEK_END); long str_size = ftell(fp_str); rewind(fp_str);
    char* dict_strings = (char*)malloc(str_size);
    fread(dict_strings, 1, str_size, fp_str); fclose(fp_str);

    // Find all dict entries starting with "forest"
    std::unordered_set<int> forest_enc_ids;
    for (int i = 0; i < dict_count; ++i) {
        const char* name = dict_strings + dict_offsets[i];
        if (strncasecmp(name, "forest", 6) == 0)
            forest_enc_ids.insert(i);
    }
    free(dict_strings);
    fascal_free_column(dict_offsets);
    printf("forest%% dict entries: %zu (from %d total)\n", forest_enc_ids.size(), dict_count);

    // Load part table to find forest partkeys
    int num_parts = 0;
    snprintf(path, sizeof(path), "%s/p_partkey.bin", data_dir);
    int *h_p_partkey = load_binary_column_auto(path, &num_parts);
    snprintf(path, sizeof(path), "%s/p_name.bin", data_dir);
    int *h_p_name = load_binary_column_auto(path, &num_parts);

    std::unordered_set<int> forest_partkeys;
    for (int i = 0; i < num_parts; ++i) {
        if (forest_enc_ids.count(h_p_name[i]))
            forest_partkeys.insert(h_p_partkey[i]);
    }
    printf("forest partkeys: %zu\n", forest_partkeys.size());
    fascal_free_column(h_p_partkey);
    fascal_free_column(h_p_name);

    // Load partsupp columns
    snprintf(path, sizeof(path), "%s/ps_partkey.bin", data_dir);
    int num_partsupp = 0;
    int *h_ps_partkey = load_binary_column_auto(path, &num_partsupp);
    snprintf(path, sizeof(path), "%s/ps_suppkey.bin", data_dir);
    int *h_ps_suppkey = load_binary_column_auto(path, &num_partsupp);
    snprintf(path, sizeof(path), "%s/ps_availqty.bin", data_dir);
    int *h_ps_availqty = load_binary_column_auto(path, &num_partsupp);
    printf("partsupp: %d rows\n", num_partsupp);

    // Hash-map lookup: lineitem_sum_map[packed_key] → SUM(l_quantity)
    std::unordered_set<int> qualifying_suppkeys_set;
    for (int i = 0; i < num_partsupp; ++i) {
        int pk = h_ps_partkey[i];
        int sk = h_ps_suppkey[i];
        int aq = h_ps_availqty[i];

        // ps_partkey must be a forest part
        if (forest_partkeys.find(pk) == forest_partkeys.end()) continue;

        // Look up SUM(l_quantity) for this (partkey, suppkey) pair
        unsigned long long packed = ((unsigned long long)(unsigned int)pk << 32)
                                  | (unsigned long long)(unsigned int)sk;
        auto it = lineitem_sum_map.find(packed);
        if (it == lineitem_sum_map.end()) continue;  // no matching lineitem
        double sq = it->second;
        if (sq == 0.0) continue;

        // ps_availqty > 0.5 * sum_qty
        if ((double)aq > 0.5 * sq)
            qualifying_suppkeys_set.insert(sk);
    }
    printf("qualifying suppkeys: %zu\n", qualifying_suppkeys_set.size());

    fascal_free_column(h_ps_partkey);
    fascal_free_column(h_ps_suppkey);
    fascal_free_column(h_ps_availqty);
    lineitem_sum_map.clear();  // hash map no longer needed

    // Convert to array for GPU upload
    std::vector<int> qual_suppkeys(qualifying_suppkeys_set.begin(), qualifying_suppkeys_set.end());
    int num_qual = (int)qual_suppkeys.size();

    // =========================================================================
    // Stage 2 (GPU): Scan supplier, join nation (CANADA) + qualifying suppkeys
    // =========================================================================
    printf("--- Stage 2: supplier scan (GPU) ---\n");

    int num_tuples = 0;  // supplier
    snprintf(path, sizeof(path), "%s/s_name.bin", data_dir);
    int *h_s_name = load_binary_column_auto(path, &num_tuples);
    if (!h_s_name) { fprintf(stderr, "Failed to load s_name\n"); return 1; }

    snprintf(path, sizeof(path), "%s/s_address.bin", data_dir);
    int *h_s_address = load_binary_column_auto(path, &num_tuples);

    snprintf(path, sizeof(path), "%s/s_nationkey.bin", data_dir);
    int *h_s_nationkey = load_binary_column_auto(path, &num_tuples);

    snprintf(path, sizeof(path), "%s/s_suppkey.bin", data_dir);
    int *h_s_suppkey = load_binary_column_auto(path, &num_tuples);

    printf("supplier: %d rows\n", num_tuples);

    // Load nation
    int num_nation = 0;
    snprintf(path, sizeof(path), "%s/n_nationkey.bin", data_dir);
    int *h_n_nationkey = load_binary_column_auto(path, &num_nation);
    snprintf(path, sizeof(path), "%s/n_name.bin", data_dir);
    int *h_n_name = load_binary_column_auto(path, &num_nation);
    printf("nation: %d rows\n", num_nation);

    // === Timing ===
    cudaEvent_t t_start, t_load, t_query, t_result;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_load));
    CUDA_CHECK(cudaEventCreate(&t_query));
    CUDA_CHECK(cudaEventCreate(&t_result));
    CUDA_CHECK(cudaEventRecord(t_start));

    // === CQO ===
    auto cqo_in = fascal_cqo_init(num_tuples);
    cqo_in.join_bf_selectivities.push_back(0.05);
    cqo_in.join_bf_selectivities.push_back(0.05);
    cqo_in.num_gpu_only_ops = 2;
    auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
    fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);
    int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr) || cqo_out.bloom_off;

    // === ODZC device buffers ===
    fascal::runtime::ODZCManager::MappedColumn mc_s_name, mc_s_address, mc_s_nationkey, mc_s_suppkey;
    int *d_s_name = nullptr, *d_s_address = nullptr, *d_s_nationkey = nullptr, *d_s_suppkey = nullptr;
    fascal_odzc_map(odzc_mgr.get(), h_s_name, num_tuples, "s_name", &mc_s_name, &d_s_name);
    fascal_odzc_map(odzc_mgr.get(), h_s_address, num_tuples, "s_address", &mc_s_address, &d_s_address);
    fascal_odzc_map(odzc_mgr.get(), h_s_nationkey, num_tuples, "s_nationkey", &mc_s_nationkey, &d_s_nationkey);
    fascal_odzc_map(odzc_mgr.get(), h_s_suppkey, num_tuples, "s_suppkey", &mc_s_suppkey, &d_s_suppkey);

    // Nation device columns
    fascal::runtime::ODZCManager::MappedColumn mc_n_nk, mc_n_nm;
    mc_n_nk = odzc_mgr->register_column(h_n_nationkey, num_nation * sizeof(int));
    int *d_n_nk = static_cast<int*>(mc_n_nk.device_ptr);
    if (!d_n_nk) { CUDA_CHECK(cudaMalloc(&d_n_nk, num_nation*sizeof(int)));
                    CUDA_CHECK(cudaMemcpy(d_n_nk, h_n_nationkey, num_nation*sizeof(int), cudaMemcpyHostToDevice)); }
    mc_n_nm = odzc_mgr->register_column(h_n_name, num_nation * sizeof(int));
    int *d_n_nm = static_cast<int*>(mc_n_nm.device_ptr);
    if (!d_n_nm) { CUDA_CHECK(cudaMalloc(&d_n_nm, num_nation*sizeof(int)));
                    CUDA_CHECK(cudaMemcpy(d_n_nm, h_n_name, num_nation*sizeof(int), cudaMemcpyHostToDevice)); }

    // Qualifying suppkeys device column
    int *d_qual_suppkeys = nullptr;
    if (num_qual > 0) {
        CUDA_CHECK(cudaMalloc(&d_qual_suppkeys, (size_t)num_qual * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_qual_suppkeys, qual_suppkeys.data(), (size_t)num_qual * sizeof(int), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
    CUDA_CHECK(cudaEventRecord(t_load));

    // === Output buffers ===
    int *d_out_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));
    int *d_out_col0 = nullptr, *d_out_col1 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_col0, (size_t)num_tuples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_col1, (size_t)num_tuples * sizeof(int)));

    // === Build HTs ===

    // Nation HT + BF
    GpuHashTable ht_nation = GpuHashTable::allocate_direct(num_nation);
    GpuBloomFilter gpu_bf_nation = GpuBloomFilter::allocate(num_nation, 1);
    fascal::runtime::BloomFilter bf_nation_host;
    if (!bloom_off && cqo_out.join_bf_active.size() > 0 && cqo_out.join_bf_active[0])
        bf_nation_host = fascal::runtime::BloomFilter::allocate(num_nation, 1);
    CUDA_CHECK(cudaMemset(ht_nation.d_entries, 0xff, ht_nation.ht_size * sizeof(HtEntry)));
    CUDA_CHECK(cudaMemset(gpu_bf_nation.d_bits, 0, ((size_t)gpu_bf_nation.size_bits + 63) / 64 * sizeof(uint64_t)));

    // Nation prefilter bitmap
    size_t nation_bm_words = ((size_t)num_nation + 31) / 32;
    size_t nation_tiles = ((size_t)num_nation + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
    uint32_t *h_nation_pf = nullptr; uint32_t *d_nation_pf = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_nation_pf, nation_bm_words * sizeof(uint32_t), cudaHostAllocMapped));
    std::memset(h_nation_pf, 0, nation_bm_words * sizeof(uint32_t));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_nation_pf, h_nation_pf, 0));
    uint8_t *h_nation_ts = nullptr; uint8_t *d_nation_ts = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_nation_ts, nation_tiles, cudaHostAllocMapped));
    std::memset(h_nation_ts, 0, nation_tiles);
    CUDA_CHECK(cudaHostGetDevicePointer(&d_nation_ts, h_nation_ts, 0));

    // Qualifying-suppkey HT + BF
    GpuHashTable ht_qual = GpuHashTable::allocate(std::max(num_qual, 1));
    GpuBloomFilter gpu_bf_qual = GpuBloomFilter::allocate(std::max(num_qual, 1), 1);
    fascal::runtime::BloomFilter bf_qual_host;
    if (!bloom_off && cqo_out.join_bf_active.size() > 1 && cqo_out.join_bf_active[1])
        bf_qual_host = fascal::runtime::BloomFilter::allocate(std::max(num_qual, 1), 1);
    CUDA_CHECK(cudaMemset(ht_qual.d_entries, 0xff, ht_qual.ht_size * sizeof(HtEntry)));
    CUDA_CHECK(cudaMemset(gpu_bf_qual.d_bits, 0, ((size_t)gpu_bf_qual.size_bits + 63) / 64 * sizeof(uint64_t)));

    // Seed bitmap for supplier
    uint32_t *h_seed_bitmap = nullptr, *d_seed_bitmap = nullptr;
    uint8_t *h_tile_summary = nullptr, *d_tile_summary = nullptr;
    fascal_alloc_seed_bitmap(num_tuples, &h_seed_bitmap, &d_seed_bitmap, &h_tile_summary, &d_tile_summary);

    // Bloom filter array for CPU prefilter
    fascal::runtime::BloomFilter *bf_array[2];
    bf_array[0] = (!bloom_off && cqo_out.join_bf_active.size() > 0 && cqo_out.join_bf_active[0]) ? &bf_nation_host : nullptr;
    bf_array[1] = (!bloom_off && cqo_out.join_bf_active.size() > 1 && cqo_out.join_bf_active[1]) ? &bf_qual_host : nullptr;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
    const bool pipeline_off = (getenv("FASCAL_PIPELINE_OFF") != nullptr);
    const int FASCAL_GPU_BATCH = []() -> int {
        if (const char *e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; }
        return 4194304;
    }();
    const int FASCAL_NUM_STREAMS = []() -> int {
        if (const char *e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; }
        return 4;
    }();
    cudaStream_t *fascal_streams = new cudaStream_t[FASCAL_NUM_STREAMS];
    for (int s = 0; s < FASCAL_NUM_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&fascal_streams[s]));
    printf("Launch: %d blocks x %d threads (CPU-first prefilter)\n", num_blocks, BLOCK_THREADS);
    if (!pipeline_off)
        printf("Pipeline: %d batches x %d tuples, %d streams\n",
               (num_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH, FASCAL_GPU_BATCH, FASCAL_NUM_STREAMS);
    else
        printf("Pipeline: OFF (sequential CPU-then-GPU)\n");

    int *h_data[] = {h_s_name, h_s_address, h_s_nationkey, h_s_suppkey};

    auto run_result_kernel = [&]() -> int {
        if (pipeline_off) {
            tpch_q20_cpu_prefilter_supplier(h_data, num_tuples, h_seed_bitmap, h_tile_summary, bf_array, 2);
            tpch_q20_kernel_supplier<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                num_tuples, 0,
                d_s_name, d_s_address, d_s_nationkey, d_s_suppkey,
                d_seed_bitmap, d_tile_summary,
                gpu_bf_nation.d_bits, (bloom_off ? 0u : gpu_bf_nation.size_bits),
                ht_nation.d_entries, ht_nation.ht_size,
                gpu_bf_qual.d_bits, (bloom_off ? 0u : gpu_bf_qual.size_bits),
                ht_qual.d_entries, ht_qual.ht_size,
                d_out_count, d_out_col0, d_out_col1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            return 0;
        }
        int n_batches = (num_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
        for (int b = 0; b < n_batches; ++b) {
            int b_off = b * FASCAL_GPU_BATCH;
            int b_cnt = std::min(FASCAL_GPU_BATCH, num_tuples - b_off);
            int b_blk = (b_cnt + TILE_SIZE - 1) / TILE_SIZE;
            cudaStream_t st = 0;
            tpch_q20_cpu_prefilter_supplier(h_data, b_cnt, h_seed_bitmap, h_tile_summary, bf_array, 2, b_off, num_tuples);
            tpch_q20_kernel_supplier<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blk, BLOCK_THREADS, 0, st>>>(
                num_tuples, b_off,
                d_s_name, d_s_address, d_s_nationkey, d_s_suppkey,
                d_seed_bitmap, d_tile_summary,
                gpu_bf_nation.d_bits, (bloom_off ? 0u : gpu_bf_nation.size_bits),
                ht_nation.d_entries, ht_nation.ht_size,
                gpu_bf_qual.d_bits, (bloom_off ? 0u : gpu_bf_qual.size_bits),
                ht_qual.d_entries, ht_qual.ht_size,
                d_out_count, d_out_col0, d_out_col1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        return 0;
    };

    cudaEventRecord(t0);

    // Build nation HT
    tpch_q20_cpu_prefilter_nation(h_n_nationkey, h_n_name, num_nation, h_nation_pf, h_nation_ts);
    tpch_q20_kernel_nation<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<(num_nation + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
        num_nation, d_n_nk, d_n_nm,
        ht_nation.d_entries, ht_nation.ht_size,
        gpu_bf_nation.d_bits, gpu_bf_nation.size_bits, 3,
        d_nation_pf, d_nation_ts);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    if (bf_nation_host.bits) gpu_bf_nation.copy_to_host(bf_nation_host.bits, ((size_t)(bf_nation_host.size_bits + 63) / 64) * sizeof(uint64_t));

    // Build qualifying-suppkey HT
    if (num_qual > 0) {
        tpch_q20_kernel_qual_suppkey<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<(num_qual + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
            num_qual, d_qual_suppkeys,
            ht_qual.d_entries, ht_qual.ht_size,
            gpu_bf_qual.d_bits, gpu_bf_qual.size_bits, 3);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
    }
    if (bf_qual_host.bits) gpu_bf_qual.copy_to_host(bf_qual_host.bits, ((size_t)(bf_qual_host.size_bits + 63) / 64) * sizeof(uint64_t));

    // === Result pipeline ===
    if (run_result_kernel() != 0) return 1;
    cudaEventRecord(t1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t_query));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("Kernel: %.3f ms\n", ms);

    int in_memory_repeats = 1;
    if (const char *repeat_env = getenv("FASCAL_IN_MEMORY_REPEATS")) {
        in_memory_repeats = atoi(repeat_env);
        if (in_memory_repeats < 1) in_memory_repeats = 1;
    }
    if (in_memory_repeats > 1) {
        float hot_total_ms = 0.0f, hot_best_ms = 0.0f;
        for (int r = 0; r < in_memory_repeats; ++r) {
            std::memset(h_seed_bitmap, 0, ((size_t)(num_tuples + 31) / 32) * sizeof(uint32_t));
            std::memset(h_tile_summary, 0, ((size_t)(num_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE));
            CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));
            CUDA_CHECK(cudaEventRecord(t0));
            if (run_result_kernel() != 0) return 1;
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaDeviceSynchronize());
            float hms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&hms, t0, t1));
            hot_total_ms += hms;
            if (r == 0 || hms < hot_best_ms) hot_best_ms = hms;
        }
        printf("In-memory result kernel: best %.3f ms avg %.3f ms (%d runs)\n",
               hot_best_ms, hot_total_ms / (float)in_memory_repeats, in_memory_repeats);
    }

    // === Output ===
    int h_out_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_out_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
    int *h_out_col0 = (int*)malloc((size_t)num_tuples * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out_col0, d_out_col0, (size_t)num_tuples * sizeof(int), cudaMemcpyDeviceToHost));
    int *h_out_col1 = (int*)malloc((size_t)num_tuples * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out_col1, d_out_col1, (size_t)num_tuples * sizeof(int), cudaMemcpyDeviceToHost));

    // Sort by s_name (dict index)
    std::vector<int> row_order((size_t)h_out_count);
    for (int i = 0; i < h_out_count; i++) row_order[i] = i;
    std::sort(row_order.begin(), row_order.end(), [&](int a, int b) {
        return h_out_col0[a] < h_out_col0[b];
    });

    // Load s_address dictionary from binary _dict_strings.bin / _dict_offsets[64].bin
    // (high-cardinality SF=100 supplier dict that is too large for the legacy
    // .dict text format, which is no longer produced by the data prep).
    static std::unordered_map<int, std::string> addr_dict;
    {
        char ps[1024], po64[1024], po32[1024];
        snprintf(ps,   sizeof(ps),   "%s/supplier_s_address_dict_strings.bin", data_dir);
        snprintf(po64, sizeof(po64), "%s/supplier_s_address_dict_offsets64.bin", data_dir);
        snprintf(po32, sizeof(po32), "%s/supplier_s_address_dict_offsets.bin", data_dir);
        FILE* fs = fopen(ps, "rb");
        if (fs) {
            fseek(fs, 0, SEEK_END); long sb = ftell(fs); rewind(fs);
            std::vector<char> blob((size_t)sb);
            fread(blob.data(), 1, (size_t)sb, fs); fclose(fs);
            FILE* fo = fopen(po64, "rb");
            if (fo) {
                fseek(fo, 0, SEEK_END); long ob = ftell(fo); rewind(fo);
                size_t n = (size_t)ob / sizeof(int64_t);
                std::vector<int64_t> off(n);
                fread(off.data(), 1, n * sizeof(int64_t), fo); fclose(fo);
                addr_dict.reserve(n);
                for (size_t i = 0; i < n; ++i) addr_dict[(int)i] = std::string(blob.data() + off[i]);
            } else if ((fo = fopen(po32, "rb"))) {
                fseek(fo, 0, SEEK_END); long ob = ftell(fo); rewind(fo);
                size_t n = (size_t)ob / sizeof(int32_t);
                std::vector<int32_t> off(n);
                fread(off.data(), 1, n * sizeof(int32_t), fo); fclose(fo);
                addr_dict.reserve(n);
                for (size_t i = 0; i < n; ++i) addr_dict[(int)i] = std::string(blob.data() + off[i]);
            }
        }
    }

    for (int i = 0; i < h_out_count; i++) {
        int ri = row_order[i];
        int name_idx = h_out_col0[ri];
        int addr_idx = h_out_col1[ri];
        // s_name: Supplier#XXXXXXXXX (1-indexed)
        int suppnum = 1 + name_idx;
        printf("ROW: Supplier#%09d", suppnum);
        auto it = addr_dict.find(addr_idx);
        printf("|%s", it != addr_dict.end() ? it->second.c_str() : "UNKNOWN");
        printf("\n");
    }

    printf("Total output rows: %d\n", h_out_count);

    free(h_out_col0);
    free(h_out_col1);
    CUDA_CHECK(cudaFree(d_out_col0));
    CUDA_CHECK(cudaFree(d_out_col1));
    CUDA_CHECK(cudaFree(d_out_count));
    CUDA_CHECK(cudaEventRecord(t_result));
    CUDA_CHECK(cudaDeviceSynchronize());

    // === Timing Report ===
    {
        float ms_load = 0, ms_query = 0, ms_result = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_load, t_start, t_load));
        CUDA_CHECK(cudaEventElapsedTime(&ms_query, t_load, t_query));
        CUDA_CHECK(cudaEventElapsedTime(&ms_result, t_query, t_result));
        printf("Timing: load=%.1fms query=%.1fms result=%.1fms total=%.1fms\n",
               ms_load, ms_query, ms_result,
               ms_load + ms_query + ms_result);
    }

    // === Cleanup ===
    odzc_mgr->unregister_column(mc_s_name);
    odzc_mgr->unregister_column(mc_s_address);
    odzc_mgr->unregister_column(mc_s_nationkey);
    odzc_mgr->unregister_column(mc_s_suppkey);
    odzc_mgr->unregister_column(mc_n_nk);
    odzc_mgr->unregister_column(mc_n_nm);

    CUDA_CHECK(cudaFreeHost(h_nation_ts));
    CUDA_CHECK(cudaFreeHost(h_nation_pf));
    gpu_bf_nation.free_filter();
    ht_nation.free_table();
    bf_nation_host.free_filter();
    gpu_bf_qual.free_filter();
    ht_qual.free_table();
    bf_qual_host.free_filter();
    if (d_qual_suppkeys) CUDA_CHECK(cudaFree(d_qual_suppkeys));

    fascal_free_seed_bitmap(h_seed_bitmap, h_tile_summary);
    fascal_free_column(h_s_name);
    fascal_free_column(h_s_address);
    fascal_free_column(h_s_nationkey);
    fascal_free_column(h_s_suppkey);
    fascal_free_column(h_n_nationkey);
    fascal_free_column(h_n_name);
    for (int s = 0; s < FASCAL_NUM_STREAMS; ++s) cudaStreamDestroy(fascal_streams[s]);
    delete[] fascal_streams;
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_load);
        cudaEventDestroy(t_query);
    cudaEventDestroy(t_result);

    return 0;
}
