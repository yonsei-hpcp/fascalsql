// TPC-H Q18 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Source: fascalsql/kernels/tpch_q18.cu

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
#define MAX_PREDICATES 16
#define NUM_AGGREGATES 1

__device__ __constant__ unsigned char d_pred_flags[MAX_PREDICATES];
static unsigned char h_pred_flags[MAX_PREDICATES] = {0};

static inline void normalize_or_group_pred_flags(unsigned char *pred_flags) {
  (void)pred_flags;
}
// ================= STAGE 1: tpch_q18___correlated_sq_1 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q18___correlated_sq_1_kernel_lineitem(
    int num_tuples,
    int batch_offset,
    float *d_lineitem_l_quantity,
    int *d_lineitem_l_orderkey_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    unsigned long long *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  float loc_lineitem_l_quantity[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_orderkey_enc[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // ODZC lazy load (fact GROUP BY, before aggregation): l_orderkey_enc
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_orderkey_enc + tile_offset, loc_lineitem_l_orderkey_enc,
      selection_flags, num_tile_items);

  // ODZC lazy load (aggregation only): l_quantity
  BlockLoadSelect<float, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_quantity + tile_offset, loc_lineitem_l_quantity,
      selection_flags, num_tile_items);
  // GROUP BY direct-index aggregation (multi-aggregate)
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int gk = loc_lineitem_l_orderkey_enc[ITEM];
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], (((long long)loc_lineitem_l_quantity[ITEM])));
      }
  }
  } while(0);
}

static void tpch_q18___correlated_sq_1_cpu_predicate_lineitem(int *h_data[], int offset, int cnt, uint32_t *bitmap) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
}

static void tpch_q18___correlated_sq_1_cpu_prefilter_lineitem(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
  fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
    [&](int off, int cnt) { tpch_q18___correlated_sq_1_cpu_predicate_lineitem(h_data, off, cnt, bitmap); },
    batch_offset, total_tuples);
}


// ================= STAGE 2: tpch_q18 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q18_kernel_lineitem(
    int num_tuples,
    int batch_offset,
    int *d_lineitem_l_orderkey_enc,
    float *d_lineitem_l_quantity,
    int *d_orders_o_orderkey,
    int *d_orders_o_custkey,
    int *d_orders_o_orderdate,
    int *d_orders_o_totalprice,
    // d_customer_c_custkey_enc removed: now stored as HT payload (payload optimization)
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_orders,
    uint32_t bloom_size_bits_orders,
    HtEntry *ht_orders,
    uint32_t ht_size_orders,
    const uint64_t *__restrict__ d_bloom___correlated_sq_1,
    uint32_t bloom_size_bits___correlated_sq_1,
    HtEntry *ht___correlated_sq_1,
    uint32_t ht_size___correlated_sq_1,
    const uint64_t *__restrict__ d_bloom_customer,
    uint32_t bloom_size_bits_customer,
    HtEntry *ht_customer,
    uint32_t ht_size_customer,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  int loc_lineitem_l_orderkey_enc[ITEMS_PER_THREAD_T];
  float loc_lineitem_l_quantity[ITEMS_PER_THREAD_T];
  int loc_orders_o_orderkey[ITEMS_PER_THREAD_T];
  int loc_orders_o_custkey[ITEMS_PER_THREAD_T];
  int loc_orders_o_orderdate[ITEMS_PER_THREAD_T];
  int loc_orders_o_totalprice[ITEMS_PER_THREAD_T];
  int loc_customer_c_custkey_enc[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // Phase 1: Semi-joins (Bloom filter); each FK loaded immediately before its probe
  // ODZC lazy load (join FK, before orders probe): l_orderkey_enc (encoded FK = orders row ID)
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_orderkey_enc + tile_offset, loc_lineitem_l_orderkey_enc,
      selection_flags, num_tile_items);

  BlockBloomProbe<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_lineitem_l_orderkey_enc, selection_flags, num_tile_items,
      bloom_size_bits_orders, d_bloom_orders, 3);

  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  // Phase 2: All hash table joins
  // Join 0 (inner): l_orderkey_enc -> orders (direct-address, O(1))
  int loc_orders_rid[ITEMS_PER_THREAD_T];
  BlockJoinProbePayloadDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_lineitem_l_orderkey_enc, selection_flags, num_tile_items,
      ht_orders, ht_size_orders, 0, loc_orders_rid);
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ITEM++) {
    if (selection_flags[ITEM]) {
      int rid = loc_orders_rid[ITEM];
      loc_orders_o_orderkey[ITEM] = d_orders_o_orderkey[rid];
      loc_orders_o_custkey[ITEM] = d_orders_o_custkey[rid];
      loc_orders_o_orderdate[ITEM] = d_orders_o_orderdate[rid];
      loc_orders_o_totalprice[ITEM] = d_orders_o_totalprice[rid];
    }
  }

  // Join 1 (semijoin): o_orderkey -> __correlated_sq_1
  BlockJoinProbeSemi<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_orders_o_orderkey, selection_flags, num_tile_items,
      ht___correlated_sq_1, ht_size___correlated_sq_1);

  // Join 2 (inner): o_custkey -> customer
  // Payload optimization: c_custkey_enc stored as HT payload (no rid lookup needed)
  BlockJoinProbePayloadDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_orders_o_custkey, selection_flags, num_tile_items,
      ht_customer, ht_size_customer, 1, loc_customer_c_custkey_enc);

  // ODZC lazy load (aggregation only): l_quantity
  BlockLoadSelect<float, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_quantity + tile_offset, loc_lineitem_l_quantity,
      selection_flags, num_tile_items);
  // Row-ID GROUP BY aggregation (high-cardinality dim table row ID as key)
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int gk = loc_orders_rid[ITEM];
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], (((double)loc_lineitem_l_quantity[ITEM])));
      }
  }
  } while(0);
}

static void tpch_q18_cpu_predicate_lineitem(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bloom_filters[], int num_bloom_filters) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  if (bloom_filters && 0 < num_bloom_filters && bloom_filters[0])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, h_data[0], offset, cnt, bloom_filters[0]);
}

static void tpch_q18_cpu_prefilter_lineitem(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
  fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
    [&](int off, int cnt) { tpch_q18_cpu_predicate_lineitem(h_data, off, cnt, bitmap, bloom_filters, num_bloom_filters); },
    batch_offset, total_tuples);
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q18_kernel_orders(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_orders_o_orderkey,
    const int *__restrict__ d_orders_o_custkey,
    const int *__restrict__ d_orders_o_orderdate,
    const int *__restrict__ d_orders_o_totalprice,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht___correlated_sq_1, uint32_t ht_size_f___correlated_sq_1,
    HtEntry* __restrict__ d_filter_ht_customer, uint32_t ht_size_f_customer)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_orders_o_orderkey[ITEMS_PER_THREAD_T];
  int loc_orders_o_custkey[ITEMS_PER_THREAD_T];
  int loc_orders_o_orderdate[ITEMS_PER_THREAD_T];
  int loc_orders_o_totalprice[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_orders_o_orderkey + tile_offset), loc_orders_o_orderkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_orders_o_custkey + tile_offset), loc_orders_o_custkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_orders_o_orderdate + tile_offset), loc_orders_o_orderdate, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_orders_o_totalprice + tile_offset), loc_orders_o_totalprice, selection_flags, num_tile_items);

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe(loc_orders_o_orderkey[ITEM], d_filter_ht___correlated_sq_1, ht_size_f___correlated_sq_1, nullptr)) continue;
    if (!gpu_ht_probe_direct(loc_orders_o_custkey[ITEM], d_filter_ht_customer, ht_size_f_customer, 1, nullptr)) continue;
    int row_idx = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one_direct(row_idx, row_idx, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(row_idx, 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'orders' (morsel-batched) ----
// Fills prefilter bitmap for [batch_offset, batch_offset+batch_count) then GPU kernel
// launches one morsel at a time on a CUDA stream (no doorbell).
static void tpch_q18_cpu_prefilter_orders(
    int *h_orders_o_orderkey,
    int *h_orders_o_custkey,
    int *h_orders_o_orderdate,
    int *h_orders_o_totalprice,
    int num_tuples,
    int batch_offset,
    int batch_count,
    uint32_t *bitmap,
    uint8_t *tile_summary,
    fascal::runtime::BloomFilter *upstream_bf___correlated_sq_1, fascal::runtime::BloomFilter *upstream_bf_customer) {
    fascal_cpu_prefilter_run_batch(num_tuples, batch_offset, batch_count, bitmap, tile_summary,
        [&](int offset, int cnt) {
            // 0. Zero bitmap words for this morsel.
            uint32_t word_start = (uint32_t)offset >> 5;
            uint32_t word_end   = ((uint32_t)(offset + cnt) + 31) >> 5;
            memset(&bitmap[word_start], 0, (word_end - word_start) * sizeof(uint32_t));
            // 1. Fill packed bitmap with scalar predicate result for this morsel.
            for (int i = 0; i < cnt; ++i) {
                int idx = offset + i;
                if (1) {
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
                }
            }
            // 2. Apply upstream BF probes (AND-mask with FK column probes).
                if (upstream_bf___correlated_sq_1 && upstream_bf___correlated_sq_1->bits) {
                    upstream_bf___correlated_sq_1->probe_and_mask_packed(h_orders_o_orderkey + offset, bitmap, offset, cnt);
                }
                if (upstream_bf_customer && upstream_bf_customer->bits) {
                    upstream_bf_customer->probe_and_mask_packed(h_orders_o_custkey + offset, bitmap, offset, cnt);
                }
        });
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q18_kernel___correlated_sq_1(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d___correlated_sq_1_l_orderkey,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc___correlated_sq_1_l_orderkey[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d___correlated_sq_1_l_orderkey + tile_offset), loc___correlated_sq_1_l_orderkey, selection_flags, num_tile_items);

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    unsigned long long key = static_cast<unsigned int>(loc___correlated_sq_1_l_orderkey[ITEM]);
    int payload = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one(key, payload, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc___correlated_sq_1_l_orderkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline '__correlated_sq_1' (morsel-batched) ----
// Fills prefilter bitmap for [batch_offset, batch_offset+batch_count) then GPU kernel
// launches one morsel at a time on a CUDA stream (no doorbell).
static void tpch_q18_cpu_prefilter___correlated_sq_1(
    int *h___correlated_sq_1_l_orderkey,
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
                if (1) {
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
                }
            }
            // 2. Apply upstream BF probes (AND-mask with FK column probes).

        });
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q18_kernel_customer(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_customer_c_custkey,
    const int *__restrict__ d_customer_c_custkey_enc,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_customer_c_custkey[ITEMS_PER_THREAD_T];
  int loc_customer_c_custkey_enc[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_customer_c_custkey + tile_offset), loc_customer_c_custkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_customer_c_custkey_enc + tile_offset), loc_customer_c_custkey_enc, selection_flags, num_tile_items);

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  // Payload optimization: store c_custkey_enc as payload (avoids rid lookup in main kernel)
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    gpu_ht_insert_one_direct(loc_customer_c_custkey[ITEM], loc_customer_c_custkey_enc[ITEM], 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_customer_c_custkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'customer' (morsel-batched) ----
// Fills prefilter bitmap for [batch_offset, batch_offset+batch_count) then GPU kernel
// launches one morsel at a time on a CUDA stream (no doorbell).
static void tpch_q18_cpu_prefilter_customer(
    int *h_customer_c_custkey,
    int *h_customer_c_custkey_enc,
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
                if (1) {
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

    // Shared pinned subquery materializations
    int _mat_rows___correlated_sq_1 = 0;
    int *_mat___correlated_sq_1_l_orderkey = nullptr;
    unsigned long long *_mat___correlated_sq_1_sum = nullptr;
    auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));

    // ---------------- STAGE 1 EXECUTION: tpch_q18___correlated_sq_1 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q18___correlated_sq_1
  // ============================================================
      printf("=== FaScalSQL: tpch_q18___correlated_sq_1 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/l_quantity.bin", data_dir);
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
          size_t arena_cols = 5;
          size_t arena_bytes = (size_t)num_tuples * sizeof(int) * arena_cols;
          arena_bytes = arena_bytes + arena_bytes / 8;
          fascal_arena_init(arena_bytes);
      }

      // === Data Loading ===
      char path[512];
      // ----- Load result-pipeline (fact) columns -----
  
      snprintf(path, sizeof(path), "%s/l_quantity.bin", data_dir);
      float *h_l_quantity = (float*)load_binary_column(path, num_tuples);
      if (!h_l_quantity) { fprintf(stderr, "Failed to load l_quantity\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_orderkey_enc.bin", data_dir);
      int *h_l_orderkey_enc = load_binary_column(path, num_tuples);
      if (!h_l_orderkey_enc) { fprintf(stderr, "Failed to load l_orderkey_enc\n"); return 1; }

      // Load raw l_orderkey for decode map (enc -> actual orderkey)
      snprintf(path, sizeof(path), "%s/l_orderkey.bin", data_dir);
      int *h_l_orderkey_raw = load_binary_column(path, num_tuples);
      if (!h_l_orderkey_raw) { fprintf(stderr, "Failed to load l_orderkey\n"); return 1; }

      CUDA_CHECK(cudaEventRecord(t_load));

      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      (void)cqo_in;

      cqo_in.num_gpu_only_ops = 1;
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
  
  
  
      // === AFP Manager & Predicate Filters ===
      int *h_data[] = {(int*)h_l_quantity};
      fascal::runtime::AFPManager afp_manager(0);
      int afp_filter_id = 0;
      (void)afp_manager;
  
      // === Bloom Filters (Build Pipelines) ===
  
  
  
  
      // === ODZC & Device Buffers ===
      auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(
          new fascal::runtime::ODZCManager(0));
  
      // === Seed Bitmap + Tile Summary (CPU-first, no doorbell) ===
      uint32_t *h_seed_bitmap_storage = nullptr, *d_seed_bitmap = nullptr;
      uint8_t *h_tile_summary = nullptr, *d_tile_summary = nullptr;
      fascal_alloc_seed_bitmap(num_tuples, &h_seed_bitmap_storage, &d_seed_bitmap, &h_tile_summary, &d_tile_summary);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_quantity;
      float *d_lineitem_l_quantity = nullptr;
      { int *_tmp_d = nullptr; fascal_odzc_map(odzc_mgr.get(), (int*)h_l_quantity, num_tuples, "l_quantity", &mc_lineitem_l_quantity, &_tmp_d); d_lineitem_l_quantity = (float*)_tmp_d; }
  
      int *d_lineitem_l_orderkey_enc = nullptr;
      CUDA_CHECK(cudaMalloc(&d_lineitem_l_orderkey_enc, num_tuples * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(d_lineitem_l_orderkey_enc, h_l_orderkey_enc, num_tuples * sizeof(int), cudaMemcpyHostToDevice));
  
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      size_t num_groups = 1ULL;
      num_groups *= 150000001ULL;
      unsigned long long *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, num_groups * NUM_AGGREGATES * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(unsigned long long)));
  
      // === Hash Table Build ===
  
  
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
              tpch_q18___correlated_sq_1_cpu_prefilter_lineitem(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0);
              tpch_q18___correlated_sq_1_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_lineitem_l_quantity,
          d_lineitem_l_orderkey_enc,
                  d_seed_bitmap,
                  d_tile_summary,
              d_aggtable);
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
              tpch_q18___correlated_sq_1_cpu_prefilter_lineitem(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q18___correlated_sq_1_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_lineitem_l_quantity,
          d_lineitem_l_orderkey_enc,
                  d_seed_bitmap,
                  d_tile_summary,
              d_aggtable);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };
  
      cudaEventRecord(t0);
  
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
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(unsigned long long)));
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
      unsigned long long *h_agg_table = (unsigned long long*)malloc(num_groups * NUM_AGGREGATES * sizeof(unsigned long long));
      CUDA_CHECK(cudaMemcpy(h_agg_table, d_aggtable, num_groups * NUM_AGGREGATES * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      for (size_t k = 0; k < num_groups; k++) {
        int any = ((h_agg_table[k * NUM_AGGREGATES + 0] > 300)) ? 1 : 0;
        if (any) {
          size_t gk = k;
        int key0 = (int)(gk % 150000001ULL);
        gk /= 150000001;
        }
      }
      // Materializing aggregation results to __correlated_sq_1_*.bin
      // Build decode map: enc (orders row_id) -> raw o_orderkey value
      std::vector<int> h_gb_decode0(150000001, std::numeric_limits<int>::min());
      for (int _row = 0; _row < num_tuples; ++_row) {
        int _enc = h_l_orderkey_enc[_row];
        if (_enc >= 0 && _enc < 150000001 && h_gb_decode0[(size_t)_enc] == std::numeric_limits<int>::min())
          h_gb_decode0[(size_t)_enc] = h_l_orderkey_raw[_row];
      }
      int _h_count = 0;
      for (size_t k = 0; k < num_groups; k++) {
        bool _any = ((h_agg_table[k * NUM_AGGREGATES + 0] > 300));
        if (_any) {
          _h_count++;
        }
      }
      int *h_out_col0 = (int*)malloc(_h_count * sizeof(int));
      unsigned long long *h_out_col1 = (unsigned long long*)malloc(_h_count * sizeof(unsigned long long));
      int _h_idx = 0;
      for (size_t k = 0; k < num_groups; k++) {
        bool _any = ((h_agg_table[k * NUM_AGGREGATES + 0] > 300));
        if (_any) {
          int key0 = (int)k;
          int _decoded_key0 = key0;
          if (key0 >= 0 && key0 < (int)h_gb_decode0.size() && h_gb_decode0[(size_t)key0] != std::numeric_limits<int>::min())
            _decoded_key0 = h_gb_decode0[(size_t)key0];
          h_out_col0[_h_idx] = _decoded_key0;
          h_out_col1[_h_idx] = h_agg_table[k * NUM_AGGREGATES + 0];
          _h_idx++;
        }
      }
      char path_mat0[512];
      snprintf(path_mat0, sizeof(path_mat0), "%s/__correlated_sq_1_l_orderkey.bin", data_dir);
      FILE *fp_mat0 = fopen(path_mat0, "wb");
      if (fp_mat0) {
        fwrite(h_out_col0, sizeof(int), _h_count, fp_mat0);
        fclose(fp_mat0);
        printf("Materialized %d rows to %s\n", _h_count, path_mat0);
      }
      char path_mat1[512];
      snprintf(path_mat1, sizeof(path_mat1), "%s/__correlated_sq_1_sum.bin", data_dir);
      FILE *fp_mat1 = fopen(path_mat1, "wb");
      if (fp_mat1) {
        fwrite(h_out_col1, sizeof(unsigned long long), _h_count, fp_mat1);
        fclose(fp_mat1);
        printf("Materialized %d rows to %s\n", _h_count, path_mat1);
      }
      _mat_rows___correlated_sq_1 = _h_count;
      if (_mat___correlated_sq_1_l_orderkey) CUDA_CHECK(cudaFreeHost(_mat___correlated_sq_1_l_orderkey));
      if (_mat_rows___correlated_sq_1 > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat___correlated_sq_1_l_orderkey), (size_t)_mat_rows___correlated_sq_1 * sizeof(int), cudaHostAllocDefault));
        std::memcpy(_mat___correlated_sq_1_l_orderkey, h_out_col0, (size_t)_mat_rows___correlated_sq_1 * sizeof(int));
      } else {
        _mat___correlated_sq_1_l_orderkey = nullptr;
      }
      if (_mat___correlated_sq_1_sum) CUDA_CHECK(cudaFreeHost(_mat___correlated_sq_1_sum));
      if (_mat_rows___correlated_sq_1 > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat___correlated_sq_1_sum), (size_t)_mat_rows___correlated_sq_1 * sizeof(unsigned long long), cudaHostAllocDefault));
        std::memcpy(_mat___correlated_sq_1_sum, h_out_col1, (size_t)_mat_rows___correlated_sq_1 * sizeof(unsigned long long));
      } else {
        _mat___correlated_sq_1_sum = nullptr;
      }
      free(h_out_col0);
      free(h_out_col1);
      printf("Stage materialization complete: %d rows\n", _h_count);
      free(h_agg_table);
      CUDA_CHECK(cudaFree(d_aggtable));
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
      odzc_mgr->unregister_column(mc_lineitem_l_quantity);
      CUDA_CHECK(cudaFree(d_lineitem_l_orderkey_enc));
  
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_l_quantity);
      fascal_free_column(h_l_orderkey_enc);
      fascal_free_column(h_l_orderkey_raw);
      fascal_arena_destroy();
      for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) cudaStreamDestroy(fascal_streams[_s]);
      delete[] fascal_streams;
      cudaEventDestroy(t0);
      cudaEventDestroy(t1);
      cudaEventDestroy(t_start);
      cudaEventDestroy(t_load);
            cudaEventDestroy(t_query);
      cudaEventDestroy(t_result);
  
    }

    // ---------------- STAGE 2 EXECUTION: tpch_q18 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q18
  // ============================================================
      printf("=== FaScalSQL: tpch_q18 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/l_orderkey_enc.bin", data_dir);
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
  
      // === Data Loading ===
      char path[512];
      // ----- Load result-pipeline (fact) columns -----
  
      snprintf(path, sizeof(path), "%s/l_orderkey_enc.bin", data_dir);
      int *h_l_orderkey_enc = load_binary_column(path, num_tuples);
      if (!h_l_orderkey_enc) { fprintf(stderr, "Failed to load l_orderkey_enc\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_quantity.bin", data_dir);
      float *h_l_quantity = (float*)load_binary_column(path, num_tuples);
      if (!h_l_quantity) { fprintf(stderr, "Failed to load l_quantity\n"); return 1; }
  
      // ----- Load dimension: __correlated_sq_1 -----
      int num___correlated_sq_1_tuples = 0;
  
      int *h___correlated_sq_1_l_orderkey = nullptr;
      if (_mat_rows___correlated_sq_1 > 0 && _mat___correlated_sq_1_l_orderkey != nullptr) {
          num___correlated_sq_1_tuples = _mat_rows___correlated_sq_1;
          h___correlated_sq_1_l_orderkey = _mat___correlated_sq_1_l_orderkey;
      } else {
  
      snprintf(path, sizeof(path), "%s/__correlated_sq_1_l_orderkey.bin", data_dir);
      h___correlated_sq_1_l_orderkey = load_binary_column_auto(path, &num___correlated_sq_1_tuples);
      }
  
      if (!h___correlated_sq_1_l_orderkey || num___correlated_sq_1_tuples == 0) {
          fprintf(stderr, "Failed to load table column l_orderkey\n"); return 1;
      }
      printf("Loaded __correlated_sq_1: %d tuples\n", num___correlated_sq_1_tuples);
  
      // ----- Load dimension: customer -----
      int num_customer_tuples = 0;
  
      snprintf(path, sizeof(path), "%s/c_custkey.bin", data_dir);
      int *h_customer_c_custkey = load_binary_column_auto(path, &num_customer_tuples);
  
      if (!h_customer_c_custkey || num_customer_tuples == 0) {
          fprintf(stderr, "Failed to load table column c_custkey\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/c_custkey_enc.bin", data_dir);
      int *h_customer_c_custkey_enc = load_binary_column_auto(path, &num_customer_tuples);
  
      if (!h_customer_c_custkey_enc || num_customer_tuples == 0) {
          fprintf(stderr, "Failed to load table column c_custkey_enc\n"); return 1;
      }
      printf("Loaded customer: %d tuples\n", num_customer_tuples);
  
      // ----- Load dimension: orders -----
      int num_orders_tuples = 0;
  
      snprintf(path, sizeof(path), "%s/o_orderkey.bin", data_dir);
      int *h_orders_o_orderkey = load_binary_column_auto(path, &num_orders_tuples);
  
      if (!h_orders_o_orderkey || num_orders_tuples == 0) {
          fprintf(stderr, "Failed to load table column o_orderkey\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/o_custkey.bin", data_dir);
      int *h_orders_o_custkey = load_binary_column_auto(path, &num_orders_tuples);
  
      if (!h_orders_o_custkey || num_orders_tuples == 0) {
          fprintf(stderr, "Failed to load table column o_custkey\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/o_orderdate.bin", data_dir);
      int *h_orders_o_orderdate = load_binary_column_auto(path, &num_orders_tuples);
  
      if (!h_orders_o_orderdate || num_orders_tuples == 0) {
          fprintf(stderr, "Failed to load table column o_orderdate\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/o_totalprice.bin", data_dir);
      int *h_orders_o_totalprice = load_binary_column_auto(path, &num_orders_tuples);
  
      if (!h_orders_o_totalprice || num_orders_tuples == 0) {
          fprintf(stderr, "Failed to load table column o_totalprice\n"); return 1;
      }
      printf("Loaded orders: %d tuples\n", num_orders_tuples);
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      (void)cqo_in;
      cqo_in.join_bf_selectivities.push_back(0.05);
      cqo_in.join_bf_selectivities.push_back(0.05);
      cqo_in.join_bf_selectivities.push_back(0.05);
      cqo_in.num_gpu_only_ops = 2;
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
      int *h_data[] = {h_l_orderkey_enc, (int*)h_l_quantity};
      fascal::runtime::AFPManager afp_manager(0);
      int afp_filter_id = 0;
      (void)afp_manager;
  
      // === Bloom Filters (Build Pipelines) ===
      fascal::runtime::BloomFilter bf_storage[3];
      fascal::runtime::BloomFilter *bf_array[3];
  
  
      bf_array[0] = (!bloom_off && (int)0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[2] : nullptr;
      bf_array[1] = nullptr;
      bf_array[2] = nullptr;
  
      // === ODZC & Device Buffers ===
      auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(
          new fascal::runtime::ODZCManager(0));
  
      // === Seed Bitmap + Tile Summary (CPU-first, no doorbell) ===
      uint32_t *h_seed_bitmap_storage = nullptr, *d_seed_bitmap = nullptr;
      uint8_t *h_tile_summary = nullptr, *d_tile_summary = nullptr;
      fascal_alloc_seed_bitmap(num_tuples, &h_seed_bitmap_storage, &d_seed_bitmap, &h_tile_summary, &d_tile_summary);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_orderkey_enc;
      int *d_lineitem_l_orderkey_enc = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_orderkey_enc, num_tuples, "l_orderkey_enc", &mc_lineitem_l_orderkey_enc, &d_lineitem_l_orderkey_enc);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_quantity;
      float *d_lineitem_l_quantity = nullptr;
      { int *_tmp_d = nullptr; fascal_odzc_map(odzc_mgr.get(), (int*)h_l_quantity, num_tuples, "l_quantity", &mc_lineitem_l_quantity, &_tmp_d); d_lineitem_l_quantity = (float*)_tmp_d; }
  
      fascal::runtime::ODZCManager::MappedColumn mc___correlated_sq_1_l_orderkey;
      mc___correlated_sq_1_l_orderkey = odzc_mgr->register_column(h___correlated_sq_1_l_orderkey, num___correlated_sq_1_tuples * sizeof(int));
      int *d___correlated_sq_1_l_orderkey = static_cast<int*>(mc___correlated_sq_1_l_orderkey.device_ptr);
      if (!d___correlated_sq_1_l_orderkey) {
          fprintf(stderr, "ODZC failed for __correlated_sq_1.l_orderkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d___correlated_sq_1_l_orderkey, num___correlated_sq_1_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d___correlated_sq_1_l_orderkey, h___correlated_sq_1_l_orderkey, num___correlated_sq_1_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_customer_c_custkey;
      mc_customer_c_custkey = odzc_mgr->register_column(h_customer_c_custkey, num_customer_tuples * sizeof(int));
      int *d_customer_c_custkey = static_cast<int*>(mc_customer_c_custkey.device_ptr);
      if (!d_customer_c_custkey) {
          fprintf(stderr, "ODZC failed for customer.c_custkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_customer_c_custkey, num_customer_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_customer_c_custkey, h_customer_c_custkey, num_customer_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_customer_c_custkey_enc;
      mc_customer_c_custkey_enc = odzc_mgr->register_column(h_customer_c_custkey_enc, num_customer_tuples * sizeof(int));
      int *d_customer_c_custkey_enc = static_cast<int*>(mc_customer_c_custkey_enc.device_ptr);
      if (!d_customer_c_custkey_enc) {
          fprintf(stderr, "ODZC failed for customer.c_custkey_enc, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_customer_c_custkey_enc, num_customer_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_customer_c_custkey_enc, h_customer_c_custkey_enc, num_customer_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_orders_o_orderkey;
      mc_orders_o_orderkey = odzc_mgr->register_column(h_orders_o_orderkey, num_orders_tuples * sizeof(int));
      int *d_orders_o_orderkey = static_cast<int*>(mc_orders_o_orderkey.device_ptr);
      if (!d_orders_o_orderkey) {
          fprintf(stderr, "ODZC failed for orders.o_orderkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_orders_o_orderkey, num_orders_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_orders_o_orderkey, h_orders_o_orderkey, num_orders_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_orders_o_custkey;
      mc_orders_o_custkey = odzc_mgr->register_column(h_orders_o_custkey, num_orders_tuples * sizeof(int));
      int *d_orders_o_custkey = static_cast<int*>(mc_orders_o_custkey.device_ptr);
      if (!d_orders_o_custkey) {
          fprintf(stderr, "ODZC failed for orders.o_custkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_orders_o_custkey, num_orders_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_orders_o_custkey, h_orders_o_custkey, num_orders_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_orders_o_orderdate;
      mc_orders_o_orderdate = odzc_mgr->register_column(h_orders_o_orderdate, num_orders_tuples * sizeof(int));
      int *d_orders_o_orderdate = static_cast<int*>(mc_orders_o_orderdate.device_ptr);
      if (!d_orders_o_orderdate) {
          fprintf(stderr, "ODZC failed for orders.o_orderdate, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_orders_o_orderdate, num_orders_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_orders_o_orderdate, h_orders_o_orderdate, num_orders_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_orders_o_totalprice;
      mc_orders_o_totalprice = odzc_mgr->register_column(h_orders_o_totalprice, num_orders_tuples * sizeof(int));
      int *d_orders_o_totalprice = static_cast<int*>(mc_orders_o_totalprice.device_ptr);
      if (!d_orders_o_totalprice) {
          fprintf(stderr, "ODZC failed for orders.o_totalprice, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_orders_o_totalprice, num_orders_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_orders_o_totalprice, h_orders_o_totalprice, num_orders_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      size_t num_groups = (size_t)num_orders_tuples;
      double *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, num_groups * NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(double)));
  
      // === Hash Table Build ===
  
  
      GpuHashTable ht___correlated_sq_1;
      ht___correlated_sq_1 = GpuHashTable::allocate(num___correlated_sq_1_tuples);
      GpuBloomFilter gpu_bf___correlated_sq_1 = GpuBloomFilter::allocate(num___correlated_sq_1_tuples, 3);
      bf_storage[0] = fascal::runtime::BloomFilter::allocate(num___correlated_sq_1_tuples, 3);
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
  
  
      GpuHashTable ht_customer;
      ht_customer = GpuHashTable::allocate_direct(num_customer_tuples);
      GpuBloomFilter gpu_bf_customer = GpuBloomFilter::allocate(num_customer_tuples, 3);
      bf_storage[1] = fascal::runtime::BloomFilter::allocate(num_customer_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_customer.d_entries, 0xff, ht_customer.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_customer.d_bits, 0, ((size_t)gpu_bf_customer.size_bits + 63) / 64 * sizeof(uint64_t)));
  
      // AFP all-pipeline: pinned mapped memory for customer packed prefilter + tile summary (no doorbell).
      size_t customer_bitmap_words = ((size_t)num_customer_tuples + 31) / 32;
      size_t customer_num_tiles = ((size_t)num_customer_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_customer_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_customer_prefilter_pinned, customer_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_customer_prefilter_pinned, 0, customer_bitmap_words * sizeof(uint32_t));
      uint32_t *d_customer_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_customer_prefilter_mapped, h_customer_prefilter_pinned, 0));
      uint8_t *h_customer_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_customer_tile_summary, customer_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_customer_tile_summary, 0, customer_num_tiles);
      uint8_t *d_customer_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_customer_tile_summary, h_customer_tile_summary, 0));
  
  
      GpuHashTable ht_orders;
      ht_orders = GpuHashTable::allocate_direct(num_orders_tuples);
      GpuBloomFilter gpu_bf_orders = GpuBloomFilter::allocate(num_orders_tuples, 3);
      bf_storage[2] = fascal::runtime::BloomFilter::allocate(num_orders_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_orders.d_entries, 0xff, ht_orders.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_orders.d_bits, 0, ((size_t)gpu_bf_orders.size_bits + 63) / 64 * sizeof(uint64_t)));
  
      // AFP all-pipeline: pinned mapped memory for orders packed prefilter + tile summary (no doorbell).
      size_t orders_bitmap_words = ((size_t)num_orders_tuples + 31) / 32;
      size_t orders_num_tiles = ((size_t)num_orders_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_orders_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_orders_prefilter_pinned, orders_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_orders_prefilter_pinned, 0, orders_bitmap_words * sizeof(uint32_t));
      uint32_t *d_orders_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_orders_prefilter_mapped, h_orders_prefilter_pinned, 0));
      uint8_t *h_orders_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_orders_tile_summary, orders_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_orders_tile_summary, 0, orders_num_tiles);
      uint8_t *d_orders_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_orders_tile_summary, h_orders_tile_summary, 0));
  
  
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
              tpch_q18_cpu_prefilter_lineitem(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 3);
              tpch_q18_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_lineitem_l_orderkey_enc,
          d_lineitem_l_quantity,
          d_orders_o_orderkey,
          d_orders_o_custkey,
          d_orders_o_orderdate,
          d_orders_o_totalprice,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_orders.d_bits, (bloom_off ? 0u : gpu_bf_orders.size_bits),
          ht_orders.d_entries, ht_orders.ht_size,
          gpu_bf___correlated_sq_1.d_bits, (bloom_off ? 0u : gpu_bf___correlated_sq_1.size_bits),
          ht___correlated_sq_1.d_entries, ht___correlated_sq_1.ht_size,
          gpu_bf_customer.d_bits, (bloom_off ? 0u : gpu_bf_customer.size_bits),
          ht_customer.d_entries, ht_customer.ht_size,
              d_aggtable);
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
              tpch_q18_cpu_prefilter_lineitem(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 3, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q18_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_lineitem_l_orderkey_enc,
          d_lineitem_l_quantity,
          d_orders_o_orderkey,
          d_orders_o_custkey,
          d_orders_o_orderdate,
          d_orders_o_totalprice,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_orders.d_bits, (bloom_off ? 0u : gpu_bf_orders.size_bits),
          ht_orders.d_entries, ht_orders.ht_size,
          gpu_bf___correlated_sq_1.d_bits, (bloom_off ? 0u : gpu_bf___correlated_sq_1.size_bits),
          ht___correlated_sq_1.d_entries, ht___correlated_sq_1.ht_size,
          gpu_bf_customer.d_bits, (bloom_off ? 0u : gpu_bf_customer.size_bits),
          ht_customer.d_entries, ht_customer.ht_size,
              d_aggtable);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };
  
      cudaEventRecord(t0);
  
      // AFP all-pipeline for '__correlated_sq_1': morsel-batched CPU prefilter + GPU build per stream.
      {
          int _n_build_batches = (num___correlated_sq_1_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
          for (int _b = 0; _b < _n_build_batches; ++_b) {
              int _b_off = _b * FASCAL_GPU_BATCH;
              int _b_cnt = std::min(FASCAL_GPU_BATCH, num___correlated_sq_1_tuples - _b_off);
              int _b_blks = (_b_cnt + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _bs = 0;
              tpch_q18_cpu_prefilter___correlated_sq_1(h___correlated_sq_1_l_orderkey,
                  num___correlated_sq_1_tuples, _b_off, _b_cnt,
                  h___correlated_sq_1_prefilter_pinned, h___correlated_sq_1_tile_summary);
              tpch_q18_kernel___correlated_sq_1<BLOCK_THREADS, ITEMS_PER_THREAD><<<_b_blks, BLOCK_THREADS, 0, _bs>>>(
                  num___correlated_sq_1_tuples, _b_off,
                  d___correlated_sq_1_l_orderkey,
                  ht___correlated_sq_1.d_entries, ht___correlated_sq_1.ht_size,
                  gpu_bf___correlated_sq_1.d_bits, gpu_bf___correlated_sq_1.size_bits, 3,
                  d___correlated_sq_1_prefilter_mapped, d___correlated_sq_1_tile_summary);
          }
          for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(fascal_streams[_s]));
          CUDA_CHECK(cudaGetLastError());
      }
      if (bf_storage[0].bits) gpu_bf___correlated_sq_1.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));
      // AFP all-pipeline for 'customer': morsel-batched CPU prefilter + GPU build per stream.
      {
          int _n_build_batches = (num_customer_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
          for (int _b = 0; _b < _n_build_batches; ++_b) {
              int _b_off = _b * FASCAL_GPU_BATCH;
              int _b_cnt = std::min(FASCAL_GPU_BATCH, num_customer_tuples - _b_off);
              int _b_blks = (_b_cnt + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _bs = 0;
              tpch_q18_cpu_prefilter_customer(h_customer_c_custkey, h_customer_c_custkey_enc,
                  num_customer_tuples, _b_off, _b_cnt,
                  h_customer_prefilter_pinned, h_customer_tile_summary);
              tpch_q18_kernel_customer<BLOCK_THREADS, ITEMS_PER_THREAD><<<_b_blks, BLOCK_THREADS, 0, _bs>>>(
                  num_customer_tuples, _b_off,
                  d_customer_c_custkey, d_customer_c_custkey_enc,
                  ht_customer.d_entries, ht_customer.ht_size,
                  gpu_bf_customer.d_bits, gpu_bf_customer.size_bits, 3,
                  d_customer_prefilter_mapped, d_customer_tile_summary);
          }
          for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(fascal_streams[_s]));
          CUDA_CHECK(cudaGetLastError());
      }
      if (bf_storage[1].bits) gpu_bf_customer.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));
      // AFP all-pipeline for 'orders': morsel-batched CPU prefilter + GPU build per stream.
      {
          int _n_build_batches = (num_orders_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
          for (int _b = 0; _b < _n_build_batches; ++_b) {
              int _b_off = _b * FASCAL_GPU_BATCH;
              int _b_cnt = std::min(FASCAL_GPU_BATCH, num_orders_tuples - _b_off);
              int _b_blks = (_b_cnt + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _bs = 0;
              tpch_q18_cpu_prefilter_orders(h_orders_o_orderkey, h_orders_o_custkey,
                  h_orders_o_orderdate, h_orders_o_totalprice,
                  num_orders_tuples, _b_off, _b_cnt,
                  h_orders_prefilter_pinned, h_orders_tile_summary,
                  (bf_storage[0].bits ? &bf_storage[0] : nullptr),
                  (bf_storage[1].bits ? &bf_storage[1] : nullptr));
              tpch_q18_kernel_orders<BLOCK_THREADS, ITEMS_PER_THREAD><<<_b_blks, BLOCK_THREADS, 0, _bs>>>(
                  num_orders_tuples, _b_off,
                  d_orders_o_orderkey, d_orders_o_custkey, d_orders_o_orderdate, d_orders_o_totalprice,
                  ht_orders.d_entries, ht_orders.ht_size,
                  gpu_bf_orders.d_bits, gpu_bf_orders.size_bits, 3,
                  d_orders_prefilter_mapped, d_orders_tile_summary,
                  ht___correlated_sq_1.d_entries, ht___correlated_sq_1.ht_size,
                  ht_customer.d_entries, ht_customer.ht_size);
          }
          for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(fascal_streams[_s]));
          CUDA_CHECK(cudaGetLastError());
      }
      if (bf_storage[2].bits) gpu_bf_orders.copy_to_host(bf_storage[2].bits, ((size_t)(bf_storage[2].size_bits + 63) / 64) * sizeof(uint64_t));
  
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
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(double)));
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
      double *h_agg_table = (double*)malloc(num_groups * NUM_AGGREGATES * sizeof(double));
      CUDA_CHECK(cudaMemcpy(h_agg_table, d_aggtable, num_groups * NUM_AGGREGATES * sizeof(double), cudaMemcpyDeviceToHost));
      std::vector<std::pair<std::tuple<long long, long long>, size_t>> _result_rows;
      for (size_t k = 0; k < num_groups; k++) {
        if (!((h_agg_table[k * NUM_AGGREGATES + 0] != 0))) continue;
        _result_rows.push_back({std::tuple<long long, long long>{(long long)(h_orders_o_totalprice[(int)k]), (long long)(h_orders_o_orderdate[(int)k])}, k});
      }
      std::sort(_result_rows.begin(), _result_rows.end(), [](const auto &_a, const auto &_b) {
        if (std::get<0>(_a.first) != std::get<0>(_b.first)) return std::get<0>(_a.first) > std::get<0>(_b.first);
        if (std::get<1>(_a.first) != std::get<1>(_b.first)) return std::get<1>(_a.first) < std::get<1>(_b.first);
        return false;
      });
      int _print_limit = std::min((int)_result_rows.size(), (int)100);
      for (int _ri = 0; _ri < _print_limit; _ri++) {
        size_t k = _result_rows[_ri].second;
        printf("ROW: ");
        printf("%s", [&]() -> const char* {
    static thread_local char buf[64];
    int idx = (int)(h_orders_o_custkey[(int)k] - 1);
    int val = 1 + idx;
    if (val < 1) return "UNKNOWN";
    snprintf(buf, sizeof(buf), "Customer#%09d", val);
    return buf;
  }());
        printf("|%d", (int)h_orders_o_custkey[(int)k]);
        printf("|%d", (int)h_orders_o_orderkey[(int)k]);
        printf("|%04d-%02d-%02d", (int)(h_orders_o_orderdate[(int)k] / 10000), (int)((h_orders_o_orderdate[(int)k] / 100) % 100), (int)(h_orders_o_orderdate[(int)k] % 100));
        printf("|%.2f", ((double)(long long)h_orders_o_totalprice[(int)k]) / 100.0);
        printf("|%.2f", (double)h_agg_table[k * NUM_AGGREGATES + 0]);
        printf("\n");
      }
      free(h_agg_table);
      CUDA_CHECK(cudaFree(d_aggtable));
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
      odzc_mgr->unregister_column(mc_lineitem_l_orderkey_enc);
      odzc_mgr->unregister_column(mc_lineitem_l_quantity);
  
      odzc_mgr->unregister_column(mc___correlated_sq_1_l_orderkey);
      odzc_mgr->unregister_column(mc_customer_c_custkey);
      odzc_mgr->unregister_column(mc_customer_c_custkey_enc);
      odzc_mgr->unregister_column(mc_orders_o_orderkey);
      odzc_mgr->unregister_column(mc_orders_o_custkey);
      odzc_mgr->unregister_column(mc_orders_o_orderdate);
      odzc_mgr->unregister_column(mc_orders_o_totalprice);
      CUDA_CHECK(cudaFreeHost(h___correlated_sq_1_tile_summary));
      CUDA_CHECK(cudaFreeHost(h___correlated_sq_1_prefilter_pinned));
      gpu_bf___correlated_sq_1.free_filter();
      ht___correlated_sq_1.free_table();
      bf_storage[0].free_filter();
      if (h___correlated_sq_1_l_orderkey && (void*)h___correlated_sq_1_l_orderkey != (void*)_mat___correlated_sq_1_l_orderkey) fascal_free_column(h___correlated_sq_1_l_orderkey);
      CUDA_CHECK(cudaFreeHost(h_customer_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_customer_prefilter_pinned));
      gpu_bf_customer.free_filter();
      ht_customer.free_table();
      bf_storage[1].free_filter();
      fascal_free_column(h_customer_c_custkey);
      fascal_free_column(h_customer_c_custkey_enc);
      CUDA_CHECK(cudaFreeHost(h_orders_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_orders_prefilter_pinned));
      gpu_bf_orders.free_filter();
      ht_orders.free_table();
      bf_storage[2].free_filter();
      fascal_free_column(h_orders_o_orderkey);
      fascal_free_column(h_orders_o_custkey);
      fascal_free_column(h_orders_o_orderdate);
      fascal_free_column(h_orders_o_totalprice);
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_l_orderkey_enc);
      fascal_free_column(h_l_quantity);
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
    // Cleanup shared pinned subquery materializations
    if (_mat___correlated_sq_1_l_orderkey) CUDA_CHECK(cudaFreeHost(_mat___correlated_sq_1_l_orderkey));
    if (_mat___correlated_sq_1_sum) CUDA_CHECK(cudaFreeHost(_mat___correlated_sq_1_sum));
}
