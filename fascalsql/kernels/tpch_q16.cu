// TPC-H Q16 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Source: fascalsql/kernels/tpch_q16.cu

#define CUB_STDERR

// FASCAL_TILE_SIZE must be defined BEFORE including fascal_generated_common.cuh
// and fascal_query_runtime.cuh so that the runtime uses the correct tile size.
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
#define MAX_PREDICATES 18
#define NUM_AGGREGATES 1
__device__ __constant__ unsigned char d_pred_flags[MAX_PREDICATES];
static unsigned char h_pred_flags[MAX_PREDICATES] = {0};

static inline void normalize_or_group_pred_flags(unsigned char *pred_flags) {
  {
    // Mixed OR-group placements fall back to GPU because the AFP seed bitmap is boolean.
    const unsigned char normalized = ((pred_flags[6] == 1) && (pred_flags[7] == 1) && (pred_flags[8] == 1) && (pred_flags[9] == 1) && (pred_flags[10] == 1) && (pred_flags[11] == 1) && (pred_flags[12] == 1) && (pred_flags[13] == 1)) ? 1 : 0;
    pred_flags[6] = normalized;
    pred_flags[7] = normalized;
    pred_flags[8] = normalized;
    pred_flags[9] = normalized;
    pred_flags[10] = normalized;
    pred_flags[11] = normalized;
    pred_flags[12] = normalized;
    pred_flags[13] = normalized;
  }
  {
    // Mixed OR-group placements fall back to GPU because the AFP seed bitmap is boolean.
    const unsigned char normalized = ((pred_flags[14] == 1) && (pred_flags[15] == 1) && (pred_flags[16] == 1) && (pred_flags[17] == 1)) ? 1 : 0;
    pred_flags[14] = normalized;
    pred_flags[15] = normalized;
    pred_flags[16] = normalized;
    pred_flags[17] = normalized;
  }
}

// Runtime bitset for s_comment LIKE '%Customer%Complaints%'
// Built from supplier_s_comment dictionary at startup (not hardcoded SF=1 IDs).
// 1 = matches (included in correlated-subquery HT for antijoin).
__device__ __constant__ unsigned char* d_sc_comment_bitset;
unsigned char* d_sc_comment_bitset_ptr = nullptr;   // GPU pointer (alias into d_sc_comment_bitset)
int h_sc_comment_bitset_size = 0;                   // number of entries (dict size)
static unsigned char* g_sc_comment_bitset_host = nullptr;  // host-side copy for CPU prefilter

// ================= STAGE 1: tpch_q16 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q16_kernel_partsupp(
    int num_tuples,
    int batch_offset,
    int *d_partsupp_ps_partkey,
    int *d_partsupp_ps_suppkey,
    int *d_part_p_brand_enc,
    int *d_part_p_type_enc,
    int *d_part_p_size_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_part,
    uint32_t bloom_size_bits_part,
    HtEntry *ht_part,
    uint32_t ht_size_part,
    const uint64_t *__restrict__ d_bloom___correlated_sq_1,
    uint32_t bloom_size_bits___correlated_sq_1,
    HtEntry *ht___correlated_sq_1,
    uint32_t ht_size___correlated_sq_1,
    int *d_cd_keys,
    int *d_cd_vals,
    int *d_cd_count,
    int d_cd_capacity)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  int loc_partsupp_ps_partkey[ITEMS_PER_THREAD_T];
  int loc_partsupp_ps_suppkey[ITEMS_PER_THREAD_T];
  int loc_part_p_brand_enc[ITEMS_PER_THREAD_T];
  int loc_part_p_type_enc[ITEMS_PER_THREAD_T];
  int loc_part_p_size_enc[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // Phase 1: Semi-joins (Bloom filter); each FK loaded immediately before its probe
  // ODZC lazy load (join FK, before part probe): ps_partkey
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_partsupp_ps_partkey + tile_offset, loc_partsupp_ps_partkey,
      selection_flags, num_tile_items);

  BlockBloomProbe<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_partsupp_ps_partkey, selection_flags, num_tile_items,
      bloom_size_bits_part, d_bloom_part, 3);

  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  // Phase 2: All hash table joins
  // Join 0 (inner): ps_partkey -> part
  int loc_part_rid[ITEMS_PER_THREAD_T];
  BlockJoinProbePayloadDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_partsupp_ps_partkey, selection_flags, num_tile_items,
      ht_part, ht_size_part, 1, loc_part_rid);
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ITEM++) {
    if (selection_flags[ITEM]) {
      int rid = loc_part_rid[ITEM];
      loc_part_p_brand_enc[ITEM] = d_part_p_brand_enc[rid];
      loc_part_p_type_enc[ITEM] = d_part_p_type_enc[rid];
      loc_part_p_size_enc[ITEM] = d_part_p_size_enc[rid];
    }
  }

  // Join 1 (antijoin): ps_suppkey -> __correlated_sq_1
  // ODZC lazy load (join FK, before __correlated_sq_1 hash probe): ps_suppkey
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_partsupp_ps_suppkey + tile_offset, loc_partsupp_ps_suppkey,
      selection_flags, num_tile_items);

  BlockJoinProbeAnti<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_partsupp_ps_suppkey, selection_flags, num_tile_items,
      ht___correlated_sq_1, ht_size___correlated_sq_1);

  // COUNT_DISTINCT: materialize (group_key, value) pairs
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int gk = loc_part_p_brand_enc[ITEM] + (loc_part_p_type_enc[ITEM] * 25) + (loc_part_p_size_enc[ITEM] * 3750);
        int dv = loc_partsupp_ps_suppkey[ITEM];
        int pos = atomicAdd(d_cd_count, 1);
        if (pos < d_cd_capacity) {
          d_cd_keys[pos] = gk;
          d_cd_vals[pos] = dv;
        }
      }
  }
  } while(0);
}

static void tpch_q16_cpu_predicate_partsupp(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bloom_filters[], int num_bloom_filters) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  if (bloom_filters && 0 < num_bloom_filters && bloom_filters[0])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, h_data[0], offset, cnt, bloom_filters[0]);
}

static void tpch_q16_cpu_prefilter_partsupp(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
  fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
    [&](int off, int cnt) { tpch_q16_cpu_predicate_partsupp(h_data, off, cnt, bitmap, bloom_filters, num_bloom_filters); },
    batch_offset, total_tuples);
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q16_kernel_part(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_part_p_partkey,
    const int *__restrict__ d_part_p_brand,
    const int *__restrict__ d_part_p_type,
    const int *__restrict__ d_part_p_size,
    const int *__restrict__ d_part_p_brand_enc,
    const int *__restrict__ d_part_p_type_enc,
    const int *__restrict__ d_part_p_size_enc,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_part_p_partkey[ITEMS_PER_THREAD_T];
  int loc_part_p_brand[ITEMS_PER_THREAD_T];
  int loc_part_p_type[ITEMS_PER_THREAD_T];
  int loc_part_p_size[ITEMS_PER_THREAD_T];
  int loc_part_p_brand_enc[ITEMS_PER_THREAD_T];
  int loc_part_p_type_enc[ITEMS_PER_THREAD_T];
  int loc_part_p_size_enc[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_partkey + tile_offset), loc_part_p_partkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_brand + tile_offset), loc_part_p_brand, selection_flags, num_tile_items);

  BlockPredAndNEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_part_p_brand, 19, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_type + tile_offset), loc_part_p_type, selection_flags, num_tile_items);

  BlockPredAndNEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_part_p_type, 70, selection_flags, num_tile_items);

  BlockPredAndNEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_part_p_type, 71, selection_flags, num_tile_items);

  BlockPredAndNEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_part_p_type, 72, selection_flags, num_tile_items);

  BlockPredAndNEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_part_p_type, 73, selection_flags, num_tile_items);

  BlockPredAndNEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_part_p_type, 74, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_size + tile_offset), loc_part_p_size, selection_flags, num_tile_items);

  // OR predicate group 1 in sub-pipeline build kernel
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM])
        selection_flags[ITEM] = (((loc_part_p_size[ITEM] == 14)) || ((loc_part_p_size[ITEM] == 23)) || ((loc_part_p_size[ITEM] == 45)) || ((loc_part_p_size[ITEM] == 19)) || ((loc_part_p_size[ITEM] == 3)) || ((loc_part_p_size[ITEM] == 36)) || ((loc_part_p_size[ITEM] == 9)) || ((loc_part_p_size[ITEM] == 49))) ? 1 : 0;
  }

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_brand_enc + tile_offset), loc_part_p_brand_enc, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_type_enc + tile_offset), loc_part_p_type_enc, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_size_enc + tile_offset), loc_part_p_size_enc, selection_flags, num_tile_items);

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int payload = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one_direct(loc_part_p_partkey[ITEM], payload, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_part_p_partkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'part' (morsel-batched) ----
// Fills prefilter bitmap for [batch_offset, batch_offset+batch_count) then GPU kernel
// launches one morsel at a time on a CUDA stream (no doorbell).
static void tpch_q16_cpu_prefilter_part(
    int *h_part_p_partkey,
    int *h_part_p_brand,
    int *h_part_p_type,
    int *h_part_p_size,
    int *h_part_p_brand_enc,
    int *h_part_p_type_enc,
    int *h_part_p_size_enc,
    int num_tuples,
    int batch_offset,
    int batch_count,
    uint32_t *bitmap,
    uint8_t *tile_summary) {
    fascal_cpu_prefilter_run_batch(num_tuples, batch_offset, batch_count, bitmap, tile_summary,
        [&](int offset, int cnt) {
            // 0. Zero bitmap words for this morsel.
            uint32_t word_start = (uint32_t)offset >> 5;
            uint32_t word_end   = ((uint32_t)(offset + cnt) + 31) >> 5;
            memset(&bitmap[word_start], 0, (word_end - word_start) * sizeof(uint32_t));
            // 1. Fill packed bitmap with scalar predicate result for this morsel.
            for (int i = 0; i < cnt; ++i) {
                int idx = offset + i;
                if ((h_part_p_brand[idx] != 19) && (h_part_p_type[idx] != 70) && (h_part_p_type[idx] != 71) && (h_part_p_type[idx] != 72) && (h_part_p_type[idx] != 73) && (h_part_p_type[idx] != 74) && ((h_part_p_size[idx] == 14) || (h_part_p_size[idx] == 23) || (h_part_p_size[idx] == 45) || (h_part_p_size[idx] == 19) || (h_part_p_size[idx] == 3) || (h_part_p_size[idx] == 36) || (h_part_p_size[idx] == 9) || (h_part_p_size[idx] == 49))) {
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
                }
            }
            // 2. Apply upstream BF probes (AND-mask with FK column probes).

        });
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q16_kernel___correlated_sq_1(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d___correlated_sq_1_s_suppkey,
    const int *__restrict__ d___correlated_sq_1_s_comment,
    int sc_bitset_size,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc___correlated_sq_1_s_suppkey[ITEMS_PER_THREAD_T];
  int loc___correlated_sq_1_s_comment[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d___correlated_sq_1_s_suppkey + tile_offset), loc___correlated_sq_1_s_suppkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d___correlated_sq_1_s_comment + tile_offset), loc___correlated_sq_1_s_comment, selection_flags, num_tile_items);

  // Runtime bitset for s_comment LIKE '%Customer%Complaints%' (built from dictionary at startup).
  // Replaces hardcoded SF=1 dict IDs which are wrong at higher scale factors.
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int sc_id = loc___correlated_sq_1_s_comment[ITEM];
        // sc_id must be within the bitset range; entries outside dict are not matching.
        selection_flags[ITEM] = (sc_id >= 0 && sc_id < sc_bitset_size && d_sc_comment_bitset[sc_id]) ? 1 : 0;
      }
  }

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    unsigned long long key = static_cast<unsigned int>(loc___correlated_sq_1_s_suppkey[ITEM]);
    int payload = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one(key, payload, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc___correlated_sq_1_s_suppkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline '__correlated_sq_1' (morsel-batched) ----
// Fills prefilter bitmap for [batch_offset, batch_offset+batch_count) then GPU kernel
// launches one morsel at a time on a CUDA stream (no doorbell).
static void tpch_q16_cpu_prefilter___correlated_sq_1(
    int *h___correlated_sq_1_s_suppkey,
    int *h___correlated_sq_1_s_comment,
    int num_tuples,
    int batch_offset,
    int batch_count,
    uint32_t *bitmap,
    uint8_t *tile_summary) {
    fascal_cpu_prefilter_run_batch(num_tuples, batch_offset, batch_count, bitmap, tile_summary,
        [&](int offset, int cnt) {
            // 0. Zero bitmap words for this morsel.
            uint32_t word_start = (uint32_t)offset >> 5;
            uint32_t word_end   = ((uint32_t)(offset + cnt) + 31) >> 5;
            memset(&bitmap[word_start], 0, (word_end - word_start) * sizeof(uint32_t));
            // 1. Fill packed bitmap with scalar predicate result for this morsel.
            for (int i = 0; i < cnt; ++i) {
                int idx = offset + i;
                int sc_id = h___correlated_sq_1_s_comment[idx];
                // Use runtime-built host bitset for '%Customer%Complaints%' (SF-agnostic).
                bool is_match = (sc_id >= 0 && sc_id < h_sc_comment_bitset_size
                                 && g_sc_comment_bitset_host && g_sc_comment_bitset_host[sc_id]);
                if (is_match) {
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
                }
            }
            // 2. Apply upstream BF probes (AND-mask with FK column probes).

        });
}


int main(int argc, char** argv) {

    const char *data_dir = getenv("FASCALSQL_DATA_DIR");
    if (!data_dir) {
        data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
    }
    if (argc >= 3) {
        data_dir = argv[2];
    }
    if (!data_dir) {
        fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n");
        return 1;
    }

    auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));

    // ---------------- STAGE 1 EXECUTION: tpch_q16 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q16
  // ============================================================
      printf("=== FaScalSQL: tpch_q16 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/ps_partkey.bin", data_dir);
          int _auto_count = 0;
          int *_tmp = load_binary_column_auto(_auto_path, &_auto_count);
          if (_tmp) fascal_free_column(_tmp);
          num_tuples = _auto_count;
          if (num_tuples <= 0) {
              fprintf(stderr, "Cannot auto-detect num_tuples from %s\n", _auto_path);
              return 1;
          }
      }
      printf("num_tuples=%d data_dir=%s\n", num_tuples, data_dir);
  
      // === Timing Instrumentation ===
      cudaEvent_t t_start, t_load, t_query, t_result;
      CUDA_CHECK(cudaEventCreate(&t_start));
      CUDA_CHECK(cudaEventCreate(&t_load));
      CUDA_CHECK(cudaEventCreate(&t_query));
      CUDA_CHECK(cudaEventCreate(&t_result));
      CUDA_CHECK(cudaEventRecord(t_start));

      // === Pinned Arena Pre-allocation ===
      {
          size_t arena_cols = 2;
          size_t arena_bytes = (size_t)num_tuples * sizeof(int) * arena_cols;
          arena_bytes = arena_bytes + arena_bytes / 8;
          fascal_arena_init(arena_bytes);
      }

      // === Data Loading ===
      char path[512];
      // ----- Load result-pipeline (fact) columns -----
  
      snprintf(path, sizeof(path), "%s/ps_partkey.bin", data_dir);
      int *h_ps_partkey = load_binary_column(path, num_tuples);
      if (!h_ps_partkey) { fprintf(stderr, "Failed to load ps_partkey\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/ps_suppkey.bin", data_dir);
      int *h_ps_suppkey = load_binary_column(path, num_tuples);
      if (!h_ps_suppkey) { fprintf(stderr, "Failed to load ps_suppkey\n"); return 1; }
  
      // ----- Load dimension: part -----
      int num_part_tuples = 0;
  
      snprintf(path, sizeof(path), "%s/p_partkey.bin", data_dir);
      int *h_part_p_partkey = load_binary_column_auto(path, &num_part_tuples);
  
      if (!h_part_p_partkey || num_part_tuples == 0) {
          fprintf(stderr, "Failed to load table column p_partkey\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/p_brand.bin", data_dir);
      int *h_part_p_brand = load_binary_column_auto(path, &num_part_tuples);
  
      if (!h_part_p_brand || num_part_tuples == 0) {
          fprintf(stderr, "Failed to load table column p_brand\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/p_type.bin", data_dir);
      int *h_part_p_type = load_binary_column_auto(path, &num_part_tuples);
  
      if (!h_part_p_type || num_part_tuples == 0) {
          fprintf(stderr, "Failed to load table column p_type\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/p_size.bin", data_dir);
      int *h_part_p_size = load_binary_column_auto(path, &num_part_tuples);
  
      if (!h_part_p_size || num_part_tuples == 0) {
          fprintf(stderr, "Failed to load table column p_size\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/p_brand_enc.bin", data_dir);
      int *h_part_p_brand_enc = load_binary_column_auto(path, &num_part_tuples);
  
      if (!h_part_p_brand_enc || num_part_tuples == 0) {
          fprintf(stderr, "Failed to load table column p_brand_enc\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/p_type_enc.bin", data_dir);
      int *h_part_p_type_enc = load_binary_column_auto(path, &num_part_tuples);
  
      if (!h_part_p_type_enc || num_part_tuples == 0) {
          fprintf(stderr, "Failed to load table column p_type_enc\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/p_size_enc.bin", data_dir);
      int *h_part_p_size_enc = load_binary_column_auto(path, &num_part_tuples);
  
      if (!h_part_p_size_enc || num_part_tuples == 0) {
          fprintf(stderr, "Failed to load table column p_size_enc\n"); return 1;
      }
      printf("Loaded part: %d tuples\n", num_part_tuples);
  
      // ----- Load dimension: __correlated_sq_1 -----
      int num___correlated_sq_1_tuples = 0;
  
      snprintf(path, sizeof(path), "%s/s_suppkey.bin", data_dir);
      int *h___correlated_sq_1_s_suppkey = load_binary_column_auto(path, &num___correlated_sq_1_tuples);
  
      if (!h___correlated_sq_1_s_suppkey || num___correlated_sq_1_tuples == 0) {
          fprintf(stderr, "Failed to load table column s_suppkey\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/s_comment.bin", data_dir);
      int *h___correlated_sq_1_s_comment = load_binary_column_auto(path, &num___correlated_sq_1_tuples);
  
      if (!h___correlated_sq_1_s_comment || num___correlated_sq_1_tuples == 0) {
          fprintf(stderr, "Failed to load table column s_comment\n"); return 1;
      }
      printf("Loaded __correlated_sq_1: %d tuples\n", num___correlated_sq_1_tuples);
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      cqo_in.predicate_selectivities.push_back(0.100000);
      cqo_in.predicate_selectivities.push_back(0.100000);
      cqo_in.predicate_selectivities.push_back(0.100000);
      cqo_in.predicate_selectivities.push_back(0.100000);
      cqo_in.predicate_selectivities.push_back(0.100000);
      cqo_in.predicate_selectivities.push_back(0.100000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      // Dynamic BF selectivity: part(filtered by brand/type/size), supplier(anti-join subquery -- keep estimate)
      cqo_in.join_bf_selectivities.push_back(fascal_estimate_join_bf_sel(num_part_tuples, num_part_tuples));  // part predicates complex, ~all pass partkey join
      cqo_in.join_bf_selectivities.push_back(0.05);  // supplier anti-join subquery
      cqo_in.num_gpu_only_ops = 0;
      auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
      fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);
      normalize_or_group_pred_flags(h_pred_flags);
      int in_memory_repeats = 1;
      if (const char *repeat_env = getenv("FASCAL_IN_MEMORY_REPEATS")) {
          in_memory_repeats = atoi(repeat_env);
          if (in_memory_repeats < 1) in_memory_repeats = 1;
      }
  
      int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr) || cqo_out.bloom_off;
  
      // === AFP Manager & Predicate Filters ===
      int *h_data[] = {h_ps_partkey, h_ps_suppkey};
      fascal::runtime::AFPManager afp_manager(0);
      int afp_filter_id = 0;
      (void)afp_manager;
  
      // === Bloom Filters (Build Pipelines) ===
      fascal::runtime::BloomFilter bf_storage[2];
      fascal::runtime::BloomFilter *bf_array[2];
  
  
      bf_array[0] = (!bloom_off && (int)0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[0] : nullptr;
      bf_array[1] = (!bloom_off && (int)1 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[1]) ? &bf_storage[1] : nullptr;
  
      // === ODZC & Device Buffers ===
      auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(
          new fascal::runtime::ODZCManager(0));
  
      // === Seed Bitmap + Tile Summary (CPU-first, no doorbell) ===
      uint32_t *h_seed_bitmap_storage = nullptr, *d_seed_bitmap = nullptr;
      uint8_t *h_tile_summary = nullptr, *d_tile_summary = nullptr;
      fascal_alloc_seed_bitmap(num_tuples, &h_seed_bitmap_storage, &d_seed_bitmap, &h_tile_summary, &d_tile_summary);
  
      fascal::runtime::ODZCManager::MappedColumn mc_partsupp_ps_partkey;
      int *d_partsupp_ps_partkey = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_ps_partkey, num_tuples, "ps_partkey", &mc_partsupp_ps_partkey, &d_partsupp_ps_partkey);
  
      fascal::runtime::ODZCManager::MappedColumn mc_partsupp_ps_suppkey;
      int *d_partsupp_ps_suppkey = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_ps_suppkey, num_tuples, "ps_suppkey", &mc_partsupp_ps_suppkey, &d_partsupp_ps_suppkey);
  
      fascal::runtime::ODZCManager::MappedColumn mc_part_p_partkey;
      mc_part_p_partkey = odzc_mgr->register_column(h_part_p_partkey, num_part_tuples * sizeof(int));
      int *d_part_p_partkey = static_cast<int*>(mc_part_p_partkey.device_ptr);
      if (!d_part_p_partkey) {
          fprintf(stderr, "ODZC failed for part.p_partkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_part_p_partkey, num_part_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_part_p_partkey, h_part_p_partkey, num_part_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_part_p_brand;
      mc_part_p_brand = odzc_mgr->register_column(h_part_p_brand, num_part_tuples * sizeof(int));
      int *d_part_p_brand = static_cast<int*>(mc_part_p_brand.device_ptr);
      if (!d_part_p_brand) {
          fprintf(stderr, "ODZC failed for part.p_brand, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_part_p_brand, num_part_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_part_p_brand, h_part_p_brand, num_part_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_part_p_type;
      mc_part_p_type = odzc_mgr->register_column(h_part_p_type, num_part_tuples * sizeof(int));
      int *d_part_p_type = static_cast<int*>(mc_part_p_type.device_ptr);
      if (!d_part_p_type) {
          fprintf(stderr, "ODZC failed for part.p_type, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_part_p_type, num_part_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_part_p_type, h_part_p_type, num_part_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_part_p_size;
      mc_part_p_size = odzc_mgr->register_column(h_part_p_size, num_part_tuples * sizeof(int));
      int *d_part_p_size = static_cast<int*>(mc_part_p_size.device_ptr);
      if (!d_part_p_size) {
          fprintf(stderr, "ODZC failed for part.p_size, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_part_p_size, num_part_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_part_p_size, h_part_p_size, num_part_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_part_p_brand_enc;
      mc_part_p_brand_enc = odzc_mgr->register_column(h_part_p_brand_enc, num_part_tuples * sizeof(int));
      int *d_part_p_brand_enc = static_cast<int*>(mc_part_p_brand_enc.device_ptr);
      if (!d_part_p_brand_enc) {
          fprintf(stderr, "ODZC failed for part.p_brand_enc, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_part_p_brand_enc, num_part_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_part_p_brand_enc, h_part_p_brand_enc, num_part_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_part_p_type_enc;
      mc_part_p_type_enc = odzc_mgr->register_column(h_part_p_type_enc, num_part_tuples * sizeof(int));
      int *d_part_p_type_enc = static_cast<int*>(mc_part_p_type_enc.device_ptr);
      if (!d_part_p_type_enc) {
          fprintf(stderr, "ODZC failed for part.p_type_enc, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_part_p_type_enc, num_part_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_part_p_type_enc, h_part_p_type_enc, num_part_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_part_p_size_enc;
      mc_part_p_size_enc = odzc_mgr->register_column(h_part_p_size_enc, num_part_tuples * sizeof(int));
      int *d_part_p_size_enc = static_cast<int*>(mc_part_p_size_enc.device_ptr);
      if (!d_part_p_size_enc) {
          fprintf(stderr, "ODZC failed for part.p_size_enc, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_part_p_size_enc, num_part_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_part_p_size_enc, h_part_p_size_enc, num_part_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc___correlated_sq_1_s_suppkey;
      mc___correlated_sq_1_s_suppkey = odzc_mgr->register_column(h___correlated_sq_1_s_suppkey, num___correlated_sq_1_tuples * sizeof(int));
      int *d___correlated_sq_1_s_suppkey = static_cast<int*>(mc___correlated_sq_1_s_suppkey.device_ptr);
      if (!d___correlated_sq_1_s_suppkey) {
          fprintf(stderr, "ODZC failed for __correlated_sq_1.s_suppkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d___correlated_sq_1_s_suppkey, num___correlated_sq_1_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d___correlated_sq_1_s_suppkey, h___correlated_sq_1_s_suppkey, num___correlated_sq_1_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc___correlated_sq_1_s_comment;
      mc___correlated_sq_1_s_comment = odzc_mgr->register_column(h___correlated_sq_1_s_comment, num___correlated_sq_1_tuples * sizeof(int));
      int *d___correlated_sq_1_s_comment = static_cast<int*>(mc___correlated_sq_1_s_comment.device_ptr);
      if (!d___correlated_sq_1_s_comment) {
          fprintf(stderr, "ODZC failed for __correlated_sq_1.s_comment, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d___correlated_sq_1_s_comment, num___correlated_sq_1_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d___correlated_sq_1_s_comment, h___correlated_sq_1_s_comment, num___correlated_sq_1_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

      // === Build runtime s_comment bitset for LIKE '%Customer%Complaints%' ===
      // Hardcoded SF=1 IDs {1521,1522,1930,2012} are wrong at SF>=10 — use runtime LIKE scan.
      if (!d_sc_comment_bitset_ptr) {
          std::string sc_str_path  = std::string(data_dir) + "/supplier_s_comment_dict_strings.bin";
          std::string sc_off32_path = std::string(data_dir) + "/supplier_s_comment_dict_offsets.bin";
          std::string sc_off64_path = std::string(data_dir) + "/supplier_s_comment_dict_offsets64.bin";

          FILE* fp_sc_s = fopen(sc_str_path.c_str(), "rb");
          if (!fp_sc_s) { fprintf(stderr, "Error: supplier_s_comment_dict_strings.bin not found\n"); exit(1); }
          fseek(fp_sc_s, 0, SEEK_END); long sc_ssize = ftell(fp_sc_s); rewind(fp_sc_s);

          int sc_dict_count = 0;
          bool sc_use64 = false;
          int32_t* sc_off32 = nullptr; int64_t* sc_off64 = nullptr;
          FILE* fp_sc32 = fopen(sc_off32_path.c_str(), "rb");
          FILE* fp_sc64 = fopen(sc_off64_path.c_str(), "rb");
          if (fp_sc32) {
              fseek(fp_sc32, 0, SEEK_END); long ob = ftell(fp_sc32); rewind(fp_sc32);
              sc_dict_count = (int)(ob / 4); sc_off32 = (int32_t*)malloc(ob);
              size_t _rr = fread(sc_off32, 1, ob, fp_sc32); (void)_rr; fclose(fp_sc32);
              if (fp_sc64) fclose(fp_sc64);
          } else if (fp_sc64) {
              fseek(fp_sc64, 0, SEEK_END); long ob = ftell(fp_sc64); rewind(fp_sc64);
              sc_dict_count = (int)(ob / 8); sc_off64 = (int64_t*)malloc(ob); sc_use64 = true;
              size_t _rr = fread(sc_off64, 1, ob, fp_sc64); (void)_rr; fclose(fp_sc64);
          } else { fprintf(stderr, "Error: supplier s_comment dict offsets not found\n"); exit(1); }

          char* sc_strings = (char*)malloc(sc_ssize);
          size_t _rr = fread(sc_strings, 1, sc_ssize, fp_sc_s); (void)_rr; fclose(fp_sc_s);

          g_sc_comment_bitset_host = (unsigned char*)calloc(sc_dict_count, 1);
          int sc_matches = 0;
          for (int i = 0; i < sc_dict_count; i++) {
              long s2 = sc_use64 ? (long)sc_off64[i] : (long)sc_off32[i];
              long e2 = (i+1 < sc_dict_count)
                  ? (sc_use64 ? (long)sc_off64[i+1] : (long)sc_off32[i+1])
                  : sc_ssize;
              const char* sp = sc_strings + s2;
              int ln = 0; while (s2 + ln < e2 && sp[ln]) ln++;
              for (int j = 0; j <= ln - 8 && !g_sc_comment_bitset_host[i]; j++) {
                  if (!strncmp(sp+j, "Customer", 8)) {
                      for (int k = j+8; k <= ln - 10; k++) {
                          if (!strncmp(sp+k, "Complaints", 10)) {
                              g_sc_comment_bitset_host[i] = 1; sc_matches++; break;
                          }
                      }
                  }
              }
          }

          unsigned char* d_sc_bs;
          CUDA_CHECK(cudaMalloc(&d_sc_bs, sc_dict_count));
          CUDA_CHECK(cudaMemcpy(d_sc_bs, g_sc_comment_bitset_host, sc_dict_count, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpyToSymbol(d_sc_comment_bitset, &d_sc_bs, sizeof(unsigned char*)));
          d_sc_comment_bitset_ptr = d_sc_bs;
          h_sc_comment_bitset_size = sc_dict_count;

          if (sc_off32) free(sc_off32); if (sc_off64) free(sc_off64);
          free(sc_strings);
          printf("Q16: s_comment LIKE bitset: %d/%d match '%%Customer%%Complaints%%'\n",
                 sc_matches, sc_dict_count);
      }

      // === Aggregation Table ===
      size_t num_groups = 1ULL;
      num_groups *= 25ULL;
      num_groups *= 150ULL;
      num_groups *= 50ULL;
      // COUNT_DISTINCT: materialization buffers
      int cd_capacity = num_tuples * 2;  // generous capacity
      int *d_cd_keys, *d_cd_vals, *d_cd_count;
      CUDA_CHECK(cudaMalloc(&d_cd_keys, (size_t)cd_capacity * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_cd_vals, (size_t)cd_capacity * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_cd_count, sizeof(int)));
      CUDA_CHECK(cudaMemset(d_cd_count, 0, sizeof(int)));
  
      // === Hash Table Build ===
  
  
      GpuHashTable ht_part;
      ht_part = GpuHashTable::allocate_direct(num_part_tuples);
      GpuBloomFilter gpu_bf_part = GpuBloomFilter::allocate(num_part_tuples, 3);
      bf_storage[0] = fascal::runtime::BloomFilter::allocate(num_part_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_part.d_entries, 0xff, ht_part.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_part.d_bits, 0, ((size_t)gpu_bf_part.size_bits + 63) / 64 * sizeof(uint64_t)));
  
      // AFP all-pipeline: pinned mapped memory for part packed prefilter + tile summary (no doorbell).
      size_t part_bitmap_words = ((size_t)num_part_tuples + 31) / 32;
      size_t part_num_tiles = ((size_t)num_part_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_part_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_part_prefilter_pinned, part_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_part_prefilter_pinned, 0, part_bitmap_words * sizeof(uint32_t));
      uint32_t *d_part_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_part_prefilter_mapped, h_part_prefilter_pinned, 0));
      uint8_t *h_part_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_part_tile_summary, part_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_part_tile_summary, 0, part_num_tiles);
      uint8_t *d_part_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_part_tile_summary, h_part_tile_summary, 0));
  
      int *h_part_composite = (int*)malloc(num_part_tuples * sizeof(int));
      for (int i = 0; i < num_part_tuples; ++i) h_part_composite[i] = (h_part_p_brand_enc[i] * 1) + (h_part_p_type_enc[i] * 25) + (h_part_p_size_enc[i] * 3750);
  
  
      GpuHashTable ht___correlated_sq_1;
      ht___correlated_sq_1 = GpuHashTable::allocate(num___correlated_sq_1_tuples);
      GpuBloomFilter gpu_bf___correlated_sq_1 = GpuBloomFilter::allocate(num___correlated_sq_1_tuples, 3);
      bf_storage[1] = fascal::runtime::BloomFilter::allocate(num___correlated_sq_1_tuples, 3);
      CUDA_CHECK(cudaMemset(ht___correlated_sq_1.d_entries, 0xff, ht___correlated_sq_1.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf___correlated_sq_1.d_bits, 0, ((size_t)gpu_bf___correlated_sq_1.size_bits + 63) / 64 * sizeof(uint64_t)));
  
      // AFP all-pipeline: pinned mapped memory for __correlated_sq_1 packed prefilter + tile summary (no doorbell).
      size_t __correlated_sq_1_bitmap_words = ((size_t)num___correlated_sq_1_tuples + 31) / 32;
      size_t __correlated_sq_1_num_tiles = ((size_t)num___correlated_sq_1_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h___correlated_sq_1_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h___correlated_sq_1_prefilter_pinned, __correlated_sq_1_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h___correlated_sq_1_prefilter_pinned, 0, __correlated_sq_1_bitmap_words * sizeof(uint32_t));
      uint32_t *d___correlated_sq_1_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d___correlated_sq_1_prefilter_mapped, h___correlated_sq_1_prefilter_pinned, 0));
      uint8_t *h___correlated_sq_1_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h___correlated_sq_1_tile_summary, __correlated_sq_1_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h___correlated_sq_1_tile_summary, 0, __correlated_sq_1_num_tiles);
      uint8_t *d___correlated_sq_1_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d___correlated_sq_1_tile_summary, h___correlated_sq_1_tile_summary, 0));
  
  
      cudaEvent_t t0, t1;
      CUDA_CHECK(cudaEventCreate(&t0));
      CUDA_CHECK(cudaEventCreate(&t1));
  
      int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
      printf("Launch: %d blocks x %d threads (CPU-first prefilter)\n", num_blocks, BLOCK_THREADS);
      // --- Pipelined CPU-GPU execution (Section V-E) ---
      // GPU_BATCH_SIZE tuples per batch; CPU prefilters one batch, then async GPU launch on a stream.
      // Overlap: while GPU processes batch N, CPU prefilters batch N+1.
      const int FASCAL_GPU_BATCH = []() -> int {
          if (const char *e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; }
          return 4194304;  // 4M tuples default (~8K tiles, fills all SMs with minimal launch overhead)
      }();
      const int FASCAL_NUM_STREAMS = []() -> int {
          if (const char *e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; }
          return 4;
      }();
      const bool pipeline_off = (getenv("FASCAL_PIPELINE_OFF") != nullptr);
      cudaStream_t *fascal_streams = new cudaStream_t[FASCAL_NUM_STREAMS];
      for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamCreate(&fascal_streams[_s]));
      if (!pipeline_off) {
          int _n_batches = (num_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
          printf("Pipeline: %d batches x %d tuples, %d streams\n", _n_batches, FASCAL_GPU_BATCH, FASCAL_NUM_STREAMS);
      } else {
          printf("Pipeline: OFF (sequential CPU-then-GPU)\n");
      }
  
      auto run_result_kernel = [&]() -> int {
          if (pipeline_off) {
              // Sequential fallback: CPU prefilters ALL rows, then single GPU launch
              tpch_q16_cpu_prefilter_partsupp(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 2);
              tpch_q16_kernel_partsupp<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_partsupp_ps_partkey,
          d_partsupp_ps_suppkey,
          d_part_p_brand_enc,
          d_part_p_type_enc,
          d_part_p_size_enc,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_part.d_bits, (bloom_off ? 0u : gpu_bf_part.size_bits),
          ht_part.d_entries, ht_part.ht_size,
          gpu_bf___correlated_sq_1.d_bits, (bloom_off ? 0u : gpu_bf___correlated_sq_1.size_bits),
          ht___correlated_sq_1.d_entries, ht___correlated_sq_1.ht_size,
              d_cd_keys, d_cd_vals, d_cd_count, cd_capacity);
              CUDA_CHECK(cudaDeviceSynchronize());
              CUDA_CHECK(cudaGetLastError());
              return 0;
          }
          // Pipelined: process GPU_BATCH_SIZE tuples at a time
          int num_batches = (num_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
          for (int _b = 0; _b < num_batches; ++_b) {
              int b_offset = _b * FASCAL_GPU_BATCH;
              int b_count  = std::min(FASCAL_GPU_BATCH, num_tuples - b_offset);
              int b_blocks = (b_count + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _stream = 0;
              // CPU prefilter this batch (fills bitmap[b_offset..b_offset+b_count-1])
              tpch_q16_cpu_prefilter_partsupp(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 2, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q16_kernel_partsupp<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_partsupp_ps_partkey,
          d_partsupp_ps_suppkey,
          d_part_p_brand_enc,
          d_part_p_type_enc,
          d_part_p_size_enc,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_part.d_bits, (bloom_off ? 0u : gpu_bf_part.size_bits),
          ht_part.d_entries, ht_part.ht_size,
          gpu_bf___correlated_sq_1.d_bits, (bloom_off ? 0u : gpu_bf___correlated_sq_1.size_bits),
          ht___correlated_sq_1.d_entries, ht___correlated_sq_1.ht_size,
              d_cd_keys, d_cd_vals, d_cd_count, cd_capacity);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };
  
      cudaEventRecord(t0);
  
      // AFP all-pipeline for 'part': morsel-batched CPU prefilter + GPU build per stream.
      {
          int _n_build_batches = (num_part_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
          for (int _b = 0; _b < _n_build_batches; ++_b) {
              int _b_off = _b * FASCAL_GPU_BATCH;
              int _b_cnt = std::min(FASCAL_GPU_BATCH, num_part_tuples - _b_off);
              int _b_blks = (_b_cnt + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _bs = 0;
              tpch_q16_cpu_prefilter_part(h_part_p_partkey, h_part_p_brand, h_part_p_type, h_part_p_size,
                  h_part_p_brand_enc, h_part_p_type_enc, h_part_p_size_enc,
                  num_part_tuples, _b_off, _b_cnt, h_part_prefilter_pinned, h_part_tile_summary);
              tpch_q16_kernel_part<BLOCK_THREADS, ITEMS_PER_THREAD><<<_b_blks, BLOCK_THREADS, 0, _bs>>>(
                  num_part_tuples, _b_off,
                  d_part_p_partkey, d_part_p_brand, d_part_p_type, d_part_p_size,
                  d_part_p_brand_enc, d_part_p_type_enc, d_part_p_size_enc,
                  ht_part.d_entries, ht_part.ht_size,
                  gpu_bf_part.d_bits, gpu_bf_part.size_bits, 3,
                  d_part_prefilter_mapped, d_part_tile_summary);
          }
          for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(fascal_streams[_s]));
          CUDA_CHECK(cudaGetLastError());
      }
      if (bf_storage[0].bits) gpu_bf_part.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));
      // AFP all-pipeline for '__correlated_sq_1': morsel-batched CPU prefilter + GPU build per stream.
      {
          int _n_build_batches = (num___correlated_sq_1_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
          for (int _b = 0; _b < _n_build_batches; ++_b) {
              int _b_off = _b * FASCAL_GPU_BATCH;
              int _b_cnt = std::min(FASCAL_GPU_BATCH, num___correlated_sq_1_tuples - _b_off);
              int _b_blks = (_b_cnt + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _bs = 0;
              tpch_q16_cpu_prefilter___correlated_sq_1(h___correlated_sq_1_s_suppkey, h___correlated_sq_1_s_comment,
                  num___correlated_sq_1_tuples, _b_off, _b_cnt,
                  h___correlated_sq_1_prefilter_pinned, h___correlated_sq_1_tile_summary);
              tpch_q16_kernel___correlated_sq_1<BLOCK_THREADS, ITEMS_PER_THREAD><<<_b_blks, BLOCK_THREADS, 0, _bs>>>(
                  num___correlated_sq_1_tuples, _b_off,
                  d___correlated_sq_1_s_suppkey, d___correlated_sq_1_s_comment,
                  h_sc_comment_bitset_size,
                  ht___correlated_sq_1.d_entries, ht___correlated_sq_1.ht_size,
                  gpu_bf___correlated_sq_1.d_bits, gpu_bf___correlated_sq_1.size_bits, 3,
                  d___correlated_sq_1_prefilter_mapped, d___correlated_sq_1_tile_summary);
          }
          for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(fascal_streams[_s]));
          CUDA_CHECK(cudaGetLastError());
      }
      if (bf_storage[1].bits) gpu_bf___correlated_sq_1.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));
  
      // === Result Pipeline ===
      if (run_result_kernel() != 0) {
          return 1;
      }
      cudaEventRecord(t1);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(t_query));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
      printf("Kernel: %.3f ms\n", ms);
      if (in_memory_repeats > 1) {
          float hot_total_ms = 0.0f;
          float hot_best_ms = 0.0f;
          for (int _repeat = 0; _repeat < in_memory_repeats; ++_repeat) {
              std::memset(h_seed_bitmap_storage, 0, ((size_t)(num_tuples + 31) / 32) * sizeof(uint32_t));
              std::memset(h_tile_summary, 0, ((size_t)(num_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE));
      CUDA_CHECK(cudaMemset(d_cd_count, 0, sizeof(int)));
              CUDA_CHECK(cudaEventRecord(t0));
              if (run_result_kernel() != 0) {
                  return 1;
              }
              CUDA_CHECK(cudaEventRecord(t1));
              CUDA_CHECK(cudaDeviceSynchronize());
              float _hot_ms = 0.0f;
              CUDA_CHECK(cudaEventElapsedTime(&_hot_ms, t0, t1));
              hot_total_ms += _hot_ms;
              if (_repeat == 0 || _hot_ms < hot_best_ms) hot_best_ms = _hot_ms;
          }
          printf("In-memory result kernel: best %.3f ms avg %.3f ms (%d runs)\n",
                 hot_best_ms, hot_total_ms / (float)in_memory_repeats, in_memory_repeats);
      }
  
      // === Output ===
      // Read back COUNT_DISTINCT materialized pairs
      int h_cd_total = 0;
      CUDA_CHECK(cudaMemcpy(&h_cd_total, d_cd_count, sizeof(int), cudaMemcpyDeviceToHost));
      if (h_cd_total > cd_capacity) h_cd_total = cd_capacity;
      int *h_cd_keys = (int*)malloc((size_t)h_cd_total * sizeof(int));
      int *h_cd_vals = (int*)malloc((size_t)h_cd_total * sizeof(int));
      CUDA_CHECK(cudaMemcpy(h_cd_keys, d_cd_keys, (size_t)h_cd_total * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_cd_vals, d_cd_vals, (size_t)h_cd_total * sizeof(int), cudaMemcpyDeviceToHost));
      // CPU-side distinct counting per group
      std::unordered_map<int, std::unordered_set<int>> distinct_map;
      for (int i = 0; i < h_cd_total; i++) {
        distinct_map[h_cd_keys[i]].insert(h_cd_vals[i]);
      }
      // Output results
      for (auto& kv : distinct_map) {
        size_t gk = kv.first;
        auto& dset = kv.second;
        int key0 = (int)(gk % 25ULL);
        gk /= 25;
        int key1 = (int)(gk % 150ULL);
        gk /= 150;
        int key2 = (int)(gk % 50ULL);
        gk /= 50;
        printf("ROW: ");
        printf("%s", [&]() -> const char* {
    static const int nums[] = { 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45, 51, 52, 53, 54, 55 };
    static thread_local char buf[64];
    int idx = (int)key0;
    if (idx < 0 || idx >= 25) return "UNKNOWN";
    snprintf(buf, sizeof(buf), "Brand#%d", nums[idx]);
    return buf;
  }());
        printf("|%s", [&]() -> const char* {
      static std::unordered_map<int, std::string> dict;
      static bool loaded = false;
      if (!loaded) {
          char ps[1024], po64[1024], po32[1024];
          snprintf(ps,   sizeof(ps),   "%s/part_p_type_dict_strings.bin", data_dir);
          snprintf(po64, sizeof(po64), "%s/part_p_type_dict_offsets64.bin", data_dir);
          snprintf(po32, sizeof(po32), "%s/part_p_type_dict_offsets.bin", data_dir);
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
                  dict.reserve(n);
                  for (size_t i = 0; i < n; ++i) dict[(int)i] = std::string(blob.data() + off[i]);
              } else if ((fo = fopen(po32, "rb"))) {
                  fseek(fo, 0, SEEK_END); long ob = ftell(fo); rewind(fo);
                  size_t n = (size_t)ob / sizeof(int32_t);
                  std::vector<int32_t> off(n);
                  fread(off.data(), 1, n * sizeof(int32_t), fo); fclose(fo);
                  dict.reserve(n);
                  for (size_t i = 0; i < n; ++i) dict[(int)i] = std::string(blob.data() + off[i]);
              }
          }
          loaded = true;
      }
      int target_key = (int)key1;
      auto it = dict.find(target_key);
      return it != dict.end() ? it->second.c_str() : "UNKNOWN";
  }());
        // key2 is p_size_enc (0-indexed). SQL output needs actual p_size = p_size_enc + 1.
        printf("|%d", (int)key2 + 1);
        printf("|%d", (int)dset.size());
        printf("\n");
      }
      free(h_cd_keys);
      free(h_cd_vals);
      CUDA_CHECK(cudaFree(d_cd_keys));
      CUDA_CHECK(cudaFree(d_cd_vals));
      CUDA_CHECK(cudaFree(d_cd_count));
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
      odzc_mgr->unregister_column(mc_partsupp_ps_partkey);
      odzc_mgr->unregister_column(mc_partsupp_ps_suppkey);
  
      odzc_mgr->unregister_column(mc_part_p_partkey);
      odzc_mgr->unregister_column(mc_part_p_brand);
      odzc_mgr->unregister_column(mc_part_p_type);
      odzc_mgr->unregister_column(mc_part_p_size);
      odzc_mgr->unregister_column(mc_part_p_brand_enc);
      odzc_mgr->unregister_column(mc_part_p_type_enc);
      odzc_mgr->unregister_column(mc_part_p_size_enc);
      odzc_mgr->unregister_column(mc___correlated_sq_1_s_suppkey);
      odzc_mgr->unregister_column(mc___correlated_sq_1_s_comment);
      CUDA_CHECK(cudaFreeHost(h_part_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_part_prefilter_pinned));
      gpu_bf_part.free_filter();
      ht_part.free_table();
      bf_storage[0].free_filter();
      free(h_part_composite);
      fascal_free_column(h_part_p_partkey);
      fascal_free_column(h_part_p_brand);
      fascal_free_column(h_part_p_type);
      fascal_free_column(h_part_p_size);
      fascal_free_column(h_part_p_brand_enc);
      fascal_free_column(h_part_p_type_enc);
      fascal_free_column(h_part_p_size_enc);
      CUDA_CHECK(cudaFreeHost(h___correlated_sq_1_tile_summary));
      CUDA_CHECK(cudaFreeHost(h___correlated_sq_1_prefilter_pinned));
      gpu_bf___correlated_sq_1.free_filter();
      ht___correlated_sq_1.free_table();
      bf_storage[1].free_filter();
      fascal_free_column(h___correlated_sq_1_s_suppkey);
      fascal_free_column(h___correlated_sq_1_s_comment);
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_ps_partkey);
      fascal_free_column(h_ps_suppkey);
      fascal_arena_destroy();
      for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) cudaStreamDestroy(fascal_streams[_s]);
      delete[] fascal_streams;
      cudaEventDestroy(t0);
      cudaEventDestroy(t1);
      cudaEventDestroy(t_start);
      cudaEventDestroy(t_load);
            cudaEventDestroy(t_query);
      cudaEventDestroy(t_result);
      return 0;
  
    }
}
