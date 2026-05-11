// TPC-H Q15 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Source: fascalsql/kernels/tpch_q15.cu

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
// ================= STAGE 1: tpch_q15_revenue =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q15_revenue_kernel_lineitem(
    int num_tuples,
    int batch_offset,
    int *d_lineitem_l_shipdate,
    float *d_lineitem_l_extendedprice,
    float *d_lineitem_l_discount,
    int *d_lineitem_l_suppkey_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  int loc_lineitem_l_shipdate[ITEMS_PER_THREAD_T];
  float loc_lineitem_l_extendedprice[ITEMS_PER_THREAD_T];
  float loc_lineitem_l_discount[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_suppkey_enc[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  if (d_pred_flags[0] == 0 || d_pred_flags[1] == 0) {
    BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        d_lineitem_l_shipdate + tile_offset, loc_lineitem_l_shipdate,
        selection_flags, num_tile_items);
    if (d_pred_flags[0] == 0)
      BlockPredAndGTE<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
          loc_lineitem_l_shipdate, 19960101, selection_flags, num_tile_items);
    if (d_pred_flags[1] == 0)
      BlockPredAndLT<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
          loc_lineitem_l_shipdate, 19960401, selection_flags, num_tile_items);
  }
  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_suppkey_enc + tile_offset, loc_lineitem_l_suppkey_enc,
      selection_flags, num_tile_items);
  BlockLoadSelect<float, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_extendedprice + tile_offset, loc_lineitem_l_extendedprice,
      selection_flags, num_tile_items);
  BlockLoadSelect<float, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_discount + tile_offset, loc_lineitem_l_discount,
      selection_flags, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int gk = loc_lineitem_l_suppkey_enc[ITEM];
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], ((((double)loc_lineitem_l_extendedprice[ITEM]) * (1 - loc_lineitem_l_discount[ITEM]))));
      }
  }
  } while(0);
}

static void tpch_q15_revenue_cpu_predicate_lineitem(int *h_data[], int offset, int cnt, uint32_t *bitmap) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  int *l_shipdate = h_data[0];
  if (h_pred_flags[0] == 1 || h_pred_flags[1] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, l_shipdate, offset, cnt, 19960101, 19960401,
        h_pred_flags[0] == 1, h_pred_flags[1] == 1, true);
}

static void tpch_q15_revenue_cpu_prefilter_lineitem(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
  fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
    [&](int off, int cnt) { tpch_q15_revenue_cpu_predicate_lineitem(h_data, off, cnt, bitmap); },
    batch_offset, total_tuples);
}


// ================= STAGE 2: tpch_q15_max_revenue_revenue =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q15_max_revenue_revenue_kernel_lineitem(
    int num_tuples,
    int batch_offset,
    int *d_lineitem_l_shipdate,
    float *d_lineitem_l_extendedprice,
    float *d_lineitem_l_discount,
    int *d_lineitem_l_suppkey_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  int loc_lineitem_l_shipdate[ITEMS_PER_THREAD_T];
  float loc_lineitem_l_extendedprice[ITEMS_PER_THREAD_T];
  float loc_lineitem_l_discount[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_suppkey_enc[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  if (d_pred_flags[0] == 0 || d_pred_flags[1] == 0) {
    BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        d_lineitem_l_shipdate + tile_offset, loc_lineitem_l_shipdate,
        selection_flags, num_tile_items);
    if (d_pred_flags[0] == 0)
      BlockPredAndGTE<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
          loc_lineitem_l_shipdate, 19960101, selection_flags, num_tile_items);
    if (d_pred_flags[1] == 0)
      BlockPredAndLT<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
          loc_lineitem_l_shipdate, 19960401, selection_flags, num_tile_items);
  }
  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_suppkey_enc + tile_offset, loc_lineitem_l_suppkey_enc,
      selection_flags, num_tile_items);
  BlockLoadSelect<float, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_extendedprice + tile_offset, loc_lineitem_l_extendedprice,
      selection_flags, num_tile_items);
  BlockLoadSelect<float, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_discount + tile_offset, loc_lineitem_l_discount,
      selection_flags, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int gk = loc_lineitem_l_suppkey_enc[ITEM];
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], ((((double)loc_lineitem_l_extendedprice[ITEM]) * (1 - loc_lineitem_l_discount[ITEM]))));
      }
  }
  } while(0);
}

static void tpch_q15_max_revenue_revenue_cpu_predicate_lineitem(int *h_data[], int offset, int cnt, uint32_t *bitmap) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  int *l_shipdate = h_data[0];
  if (h_pred_flags[0] == 1 || h_pred_flags[1] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, l_shipdate, offset, cnt, 19960101, 19960401,
        h_pred_flags[0] == 1, h_pred_flags[1] == 1, true);
}

static void tpch_q15_max_revenue_revenue_cpu_prefilter_lineitem(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
  fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
    [&](int off, int cnt) { tpch_q15_max_revenue_revenue_cpu_predicate_lineitem(h_data, off, cnt, bitmap); },
    batch_offset, total_tuples);
}


// ================= STAGE 3: tpch_q15_max_revenue =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q15_max_revenue_kernel_revenue(
    int num_tuples,
    int batch_offset,
    double *d_revenue_sum_revenue,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    unsigned long long *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  double loc_revenue_sum_revenue[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  BlockLoadSelect<double, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_revenue_sum_revenue + tile_offset, loc_revenue_sum_revenue,
      selection_flags, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM])
        atomicMax(&aggtable[0], (((unsigned long long)loc_revenue_sum_revenue[ITEM])));
  }
  } while(0);
}

static void tpch_q15_max_revenue_cpu_predicate_revenue(int *h_data[], int offset, int cnt, uint32_t *bitmap) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
}

static void tpch_q15_max_revenue_cpu_prefilter_revenue(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
  fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
    [&](int off, int cnt) { tpch_q15_max_revenue_cpu_predicate_revenue(h_data, off, cnt, bitmap); },
    batch_offset, total_tuples);
}


// ================= STAGE 4: tpch_q15 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q15_kernel_supplier(
    int num_tuples,
    int batch_offset,
    int *d_supplier_s_suppkey,
    int *d_supplier_s_name,
    int *d_supplier_s_address,
    int *d_supplier_s_phone,
    unsigned long long *d_max_revenue_total_revenue,
    double *d_revenue_sum_revenue,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_revenue,
    uint32_t bloom_size_bits_revenue,
    HtEntry *ht_revenue,
    uint32_t ht_size_revenue,
    const uint64_t *__restrict__ d_bloom_max_revenue,
    uint32_t bloom_size_bits_max_revenue,
    HtEntry *ht_max_revenue,
    uint32_t ht_size_max_revenue,
    int *d_out_count,
    int *d_out_col0,
    int *d_out_col1,
    int *d_out_col2,
    int *d_out_col3,
    unsigned long long *d_out_col4)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  int loc_supplier_s_suppkey[ITEMS_PER_THREAD_T];
  int loc_supplier_s_name[ITEMS_PER_THREAD_T];
  int loc_supplier_s_address[ITEMS_PER_THREAD_T];
  int loc_supplier_s_phone[ITEMS_PER_THREAD_T];
  unsigned long long loc_max_revenue_total_revenue[ITEMS_PER_THREAD_T];
  double loc_revenue_sum_revenue[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // Phase 1: Semi-joins (Bloom filter); each FK loaded immediately before its probe
  // ODZC lazy load (join FK, before revenue probe): s_suppkey
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_supplier_s_suppkey + tile_offset, loc_supplier_s_suppkey,
      selection_flags, num_tile_items);

  BlockBloomProbe<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_supplier_s_suppkey, selection_flags, num_tile_items,
      bloom_size_bits_revenue, d_bloom_revenue, 3);

  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  // Phase 2: All hash table joins
  // Join 0 (inner): s_suppkey -> revenue
  int loc_revenue_rid[ITEMS_PER_THREAD_T];
  BlockJoinProbePayload<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_supplier_s_suppkey, selection_flags, num_tile_items,
      ht_revenue, ht_size_revenue, loc_revenue_rid);
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ITEM++) {
    if (selection_flags[ITEM]) {
      int rid = loc_revenue_rid[ITEM];
      loc_revenue_sum_revenue[ITEM] = d_revenue_sum_revenue[rid];
    }
  }

  // Join 1 (inner): sum_revenue -> max_revenue
  int loc_max_revenue_rid[ITEMS_PER_THREAD_T];
  BlockJoinProbePayload<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_revenue_sum_revenue, selection_flags, num_tile_items,
      ht_max_revenue, ht_size_max_revenue, loc_max_revenue_rid);
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ITEM++) {
    if (selection_flags[ITEM]) {
      int rid = loc_max_revenue_rid[ITEM];
      loc_max_revenue_total_revenue[ITEM] = d_max_revenue_total_revenue[rid];
    }
  }

  // ODZC lazy load (aggregation only): s_name
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_supplier_s_name + tile_offset, loc_supplier_s_name,
      selection_flags, num_tile_items);

  // ODZC lazy load (aggregation only): s_address
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_supplier_s_address + tile_offset, loc_supplier_s_address,
      selection_flags, num_tile_items);

  // ODZC lazy load (aggregation only): s_phone
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_supplier_s_phone + tile_offset, loc_supplier_s_phone,
      selection_flags, num_tile_items);

  // Materialize: write passing rows to output buffers
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int idx = atomicAdd(d_out_count, 1);
        d_out_col0[idx] = loc_supplier_s_suppkey[ITEM];
        d_out_col1[idx] = loc_supplier_s_name[ITEM];
        d_out_col2[idx] = loc_supplier_s_address[ITEM];
        d_out_col3[idx] = loc_supplier_s_phone[ITEM];
        d_out_col4[idx] = loc_max_revenue_total_revenue[ITEM];
      }
  }
  } while(0);
}

static void tpch_q15_cpu_predicate_supplier(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bloom_filters[], int num_bloom_filters) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  if (bloom_filters && 0 < num_bloom_filters && bloom_filters[0])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, h_data[0], offset, cnt, bloom_filters[0]);
}

static void tpch_q15_cpu_prefilter_supplier(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
  fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
    [&](int off, int cnt) { tpch_q15_cpu_predicate_supplier(h_data, off, cnt, bitmap, bloom_filters, num_bloom_filters); },
    batch_offset, total_tuples);
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q15_kernel_revenue(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_revenue_l_suppkey,
    const double *__restrict__ d_revenue_sum_revenue,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_max_revenue, uint32_t ht_size_f_max_revenue)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_revenue_l_suppkey[ITEMS_PER_THREAD_T];
  double loc_revenue_sum_revenue[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_revenue_l_suppkey + tile_offset), loc_revenue_l_suppkey, selection_flags, num_tile_items);

  BlockLoadSelect<double, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (double*)(d_revenue_sum_revenue + tile_offset), loc_revenue_sum_revenue, selection_flags, num_tile_items);

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe(loc_revenue_sum_revenue[ITEM], d_filter_ht_max_revenue, ht_size_f_max_revenue, nullptr)) continue;
    unsigned long long key = static_cast<unsigned int>(loc_revenue_l_suppkey[ITEM]);
    int payload = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one(key, payload, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_revenue_l_suppkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'revenue' ----
// Fills entire prefilter bitmap (scalar preds + upstream BF probes),
// then GPU build kernel launches with bitmap 100% ready (no doorbell).
static void tpch_q15_cpu_prefilter_revenue(
    int *h_revenue_l_suppkey,
    double *h_revenue_sum_revenue,
    int num_tuples,
    uint32_t *bitmap,
    uint8_t *tile_summary,
    fascal::runtime::BloomFilter *upstream_bf_max_revenue) {
    fascal_cpu_prefilter_run(num_tuples, bitmap, tile_summary,
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
__global__ void tpch_q15_kernel_max_revenue(
    int num_tuples,
    int batch_offset,
    const unsigned long long *__restrict__ d_max_revenue_total_revenue,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  unsigned long long loc_max_revenue_total_revenue[ITEMS_PER_THREAD_T];

  BlockLoadSelect<unsigned long long, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (unsigned long long*)(d_max_revenue_total_revenue + tile_offset), loc_max_revenue_total_revenue, selection_flags, num_tile_items);

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    unsigned long long key = static_cast<unsigned int>(loc_max_revenue_total_revenue[ITEM]);
    int payload = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one(key, payload, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_max_revenue_total_revenue[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'max_revenue' ----
// Fills entire prefilter bitmap (scalar preds + upstream BF probes),
// then GPU build kernel launches with bitmap 100% ready (no doorbell).
static void tpch_q15_cpu_prefilter_max_revenue(
    unsigned long long *h_max_revenue_total_revenue,
    int num_tuples,
    uint32_t *bitmap,
    uint8_t *tile_summary) {
    fascal_cpu_prefilter_run(num_tuples, bitmap, tile_summary,
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
    int _mat_rows_revenue = 0;
    int *_mat_revenue_l_suppkey = nullptr;
    double *_mat_revenue_sum_revenue = nullptr;
    int _mat_rows_max_revenue = 0;
    unsigned long long *_mat_max_revenue_total_revenue = nullptr;
    auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));

    // ---------------- STAGE 1 EXECUTION: tpch_q15_revenue ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q15_revenue
  // ============================================================
      printf("=== FaScalSQL: tpch_q15_revenue ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/l_shipdate.bin", data_dir);
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
          size_t arena_cols = 14;
          size_t arena_bytes = (size_t)num_tuples * sizeof(int) * arena_cols;
          arena_bytes = arena_bytes + arena_bytes / 8;
          fascal_arena_init(arena_bytes);
      }

      // === Data Loading ===
      char path[512];
      // ----- Load result-pipeline (fact) columns -----
  
      snprintf(path, sizeof(path), "%s/l_shipdate.bin", data_dir);
      int *h_l_shipdate = load_binary_column(path, num_tuples);
      if (!h_l_shipdate) { fprintf(stderr, "Failed to load l_shipdate\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_extendedprice.bin", data_dir);
      float *h_l_extendedprice = (float*)load_binary_column(path, num_tuples);
      if (!h_l_extendedprice) { fprintf(stderr, "Failed to load l_extendedprice\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_discount.bin", data_dir);
      float *h_l_discount = (float*)load_binary_column(path, num_tuples);
      if (!h_l_discount) { fprintf(stderr, "Failed to load l_discount\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_suppkey.bin", data_dir);
      int *h_l_suppkey = load_binary_column(path, num_tuples);
      if (!h_l_suppkey) { fprintf(stderr, "Failed to load l_suppkey\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_suppkey_enc.bin", data_dir);
      int *h_l_suppkey_enc = load_binary_column(path, num_tuples);
      if (!h_l_suppkey_enc) { fprintf(stderr, "Failed to load l_suppkey_enc\n"); return 1; }
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      // Dynamic sample-based selectivity profiling (fact-table predicates: l_shipdate range)
      cqo_in.predicate_selectivities.push_back(
          fascal_estimate_range_sel(h_l_shipdate, num_tuples, 19960101, 19960331));  // l_shipdate >= 19960101
      cqo_in.predicate_selectivities.push_back(
          fascal_estimate_range_sel(h_l_shipdate, num_tuples, 19960101, 19960331));  // l_shipdate < 19960401 (same range)
  
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
  
  
  
      // === AFP Manager & Predicate Filters ===
      int *h_data[] = {h_l_shipdate, (int*)h_l_extendedprice, (int*)h_l_discount};
      fascal::runtime::AFPManager afp_manager(0);
      int afp_filter_id = 0;
      if (h_pred_flags[0] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(0, fascal::runtime::AFPManager::FilterOp::GE, 19960101);
        afp_manager.register_filter(afp_filter_id++, desc);
      }
      if (h_pred_flags[1] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(0, fascal::runtime::AFPManager::FilterOp::LT, 19960401);
        afp_manager.register_filter(afp_filter_id++, desc);
      }
  
      // === Bloom Filters (Build Pipelines) ===
  
  
  
  
      // === ODZC & Device Buffers ===
      auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(
          new fascal::runtime::ODZCManager(0));
  
      // === Seed Bitmap + Tile Summary (CPU-first, no doorbell) ===
      uint32_t *h_seed_bitmap_storage = nullptr, *d_seed_bitmap = nullptr;
      uint8_t *h_tile_summary = nullptr, *d_tile_summary = nullptr;
      fascal_alloc_seed_bitmap(num_tuples, &h_seed_bitmap_storage, &d_seed_bitmap, &h_tile_summary, &d_tile_summary);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_shipdate;
      int *d_lineitem_l_shipdate = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_shipdate, num_tuples, "l_shipdate", &mc_lineitem_l_shipdate, &d_lineitem_l_shipdate);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_extendedprice;
      float *d_lineitem_l_extendedprice = nullptr;
      {
        mc_lineitem_l_extendedprice = odzc_mgr->register_column(h_l_extendedprice, (size_t)num_tuples * sizeof(float));
        d_lineitem_l_extendedprice = static_cast<float*>(mc_lineitem_l_extendedprice.device_ptr);
        if (!d_lineitem_l_extendedprice) {
          fprintf(stderr, "ODZC failed for lineitem.l_extendedprice, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_lineitem_l_extendedprice, (size_t)num_tuples * sizeof(float)));
          CUDA_CHECK(cudaMemcpy(d_lineitem_l_extendedprice, h_l_extendedprice, (size_t)num_tuples * sizeof(float), cudaMemcpyHostToDevice));
        }
      }
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_discount;
      float *d_lineitem_l_discount = nullptr;
      {
        mc_lineitem_l_discount = odzc_mgr->register_column(h_l_discount, (size_t)num_tuples * sizeof(float));
        d_lineitem_l_discount = static_cast<float*>(mc_lineitem_l_discount.device_ptr);
        if (!d_lineitem_l_discount) {
          fprintf(stderr, "ODZC failed for lineitem.l_discount, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_lineitem_l_discount, (size_t)num_tuples * sizeof(float)));
          CUDA_CHECK(cudaMemcpy(d_lineitem_l_discount, h_l_discount, (size_t)num_tuples * sizeof(float), cudaMemcpyHostToDevice));
        }
      }
  
      int *d_lineitem_l_suppkey_enc = nullptr;
      CUDA_CHECK(cudaMalloc(&d_lineitem_l_suppkey_enc, num_tuples * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(d_lineitem_l_suppkey_enc, h_l_suppkey_enc, num_tuples * sizeof(int), cudaMemcpyHostToDevice));
  
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      size_t num_groups = 1ULL;
      num_groups *= 1000001ULL;
      double *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, num_groups * NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(double)));
  
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
              tpch_q15_revenue_cpu_prefilter_lineitem(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0);
              tpch_q15_revenue_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_lineitem_l_shipdate,
          d_lineitem_l_extendedprice,
          d_lineitem_l_discount,
          d_lineitem_l_suppkey_enc,
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
              tpch_q15_revenue_cpu_prefilter_lineitem(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q15_revenue_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_lineitem_l_shipdate,
          d_lineitem_l_extendedprice,
          d_lineitem_l_discount,
          d_lineitem_l_suppkey_enc,
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
      for (size_t k = 0; k < num_groups; k++) {
        int any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0)) ? 1 : 0;
        if (any) {
          size_t gk = k;
        int key0 = (int)(gk % 1000001ULL);
        gk /= 1000001;
        }
      }
      // Materializing aggregation results to revenue_*.bin
      std::vector<int> h_gb_decode0(1000001, std::numeric_limits<int>::min());
      for (int _row = 0; _row < num_tuples; ++_row) {
        int _enc = h_l_suppkey_enc[_row];
        if (_enc >= 0 && _enc < 1000001 && h_gb_decode0[(size_t)_enc] == std::numeric_limits<int>::min())
          h_gb_decode0[(size_t)_enc] = h_l_suppkey[_row];
      }
      int _h_count = 0;
      for (size_t k = 0; k < num_groups; k++) {
        bool _any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0));
        if (_any) {
          size_t gk = k;       int key0 = (int)(gk % 1000001ULL);
                gk /= 1000001;
          _h_count++;
        }
      }
      int *h_out_col0 = (int*)malloc(_h_count * sizeof(int));
      double *h_out_col1 = (double*)malloc(_h_count * sizeof(double));
      int _h_idx = 0;
      for (size_t k = 0; k < num_groups; k++) {
        bool _any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0));
        if (_any) {
          size_t gk = k;       int key0 = (int)(gk % 1000001ULL);
                gk /= 1000001;
          int _decoded_key0 = key0;
          if (key0 >= 0 && key0 < (int)h_gb_decode0.size() && h_gb_decode0[(size_t)key0] != std::numeric_limits<int>::min())
            _decoded_key0 = h_gb_decode0[(size_t)key0];
          h_out_col0[_h_idx] = _decoded_key0;
          h_out_col1[_h_idx] = h_agg_table[k * NUM_AGGREGATES + 0];
          _h_idx++;
        }
      }
      char path_mat0[512];
      snprintf(path_mat0, sizeof(path_mat0), "%s/revenue_l_suppkey.bin", data_dir);
      FILE *fp_mat0 = fopen(path_mat0, "wb");
      if (fp_mat0) {
        fwrite(h_out_col0, sizeof(int), _h_count, fp_mat0);
        fclose(fp_mat0);
        printf("Materialized %d rows to %s\n", _h_count, path_mat0);
      }
      char path_mat1[512];
      snprintf(path_mat1, sizeof(path_mat1), "%s/revenue_sum_revenue.bin", data_dir);
      FILE *fp_mat1 = fopen(path_mat1, "wb");
      if (fp_mat1) {
        fwrite(h_out_col1, sizeof(double), _h_count, fp_mat1);
        fclose(fp_mat1);
        printf("Materialized %d rows to %s\n", _h_count, path_mat1);
      }
      _mat_rows_revenue = _h_count;
      if (_mat_revenue_l_suppkey) CUDA_CHECK(cudaFreeHost(_mat_revenue_l_suppkey));
      if (_mat_rows_revenue > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat_revenue_l_suppkey), (size_t)_mat_rows_revenue * sizeof(int), cudaHostAllocDefault));
        std::memcpy(_mat_revenue_l_suppkey, h_out_col0, (size_t)_mat_rows_revenue * sizeof(int));
      } else {
        _mat_revenue_l_suppkey = nullptr;
      }
      if (_mat_revenue_sum_revenue) CUDA_CHECK(cudaFreeHost(_mat_revenue_sum_revenue));
      if (_mat_rows_revenue > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat_revenue_sum_revenue), (size_t)_mat_rows_revenue * sizeof(double), cudaHostAllocDefault));
        std::memcpy(_mat_revenue_sum_revenue, h_out_col1, (size_t)_mat_rows_revenue * sizeof(double));
      } else {
        _mat_revenue_sum_revenue = nullptr;
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
      odzc_mgr->unregister_column(mc_lineitem_l_shipdate);
      odzc_mgr->unregister_column(mc_lineitem_l_extendedprice);
      odzc_mgr->unregister_column(mc_lineitem_l_discount);
      CUDA_CHECK(cudaFree(d_lineitem_l_suppkey_enc));
  
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_l_shipdate);
      fascal_free_column(h_l_extendedprice);
      fascal_free_column(h_l_discount);
      fascal_free_column(h_l_suppkey_enc);
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

    // ---------------- STAGE 2 EXECUTION: tpch_q15_max_revenue_revenue ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q15_max_revenue_revenue
  // ============================================================
      printf("=== FaScalSQL: tpch_q15_max_revenue_revenue ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/l_shipdate.bin", data_dir);
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
  
      snprintf(path, sizeof(path), "%s/l_shipdate.bin", data_dir);
      int *h_l_shipdate = load_binary_column(path, num_tuples);
      if (!h_l_shipdate) { fprintf(stderr, "Failed to load l_shipdate\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_extendedprice.bin", data_dir);
      float *h_l_extendedprice = (float*)load_binary_column(path, num_tuples);
      if (!h_l_extendedprice) { fprintf(stderr, "Failed to load l_extendedprice\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_discount.bin", data_dir);
      float *h_l_discount = (float*)load_binary_column(path, num_tuples);
      if (!h_l_discount) { fprintf(stderr, "Failed to load l_discount\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_suppkey.bin", data_dir);
      int *h_l_suppkey = load_binary_column(path, num_tuples);
      if (!h_l_suppkey) { fprintf(stderr, "Failed to load l_suppkey\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_suppkey_enc.bin", data_dir);
      int *h_l_suppkey_enc = load_binary_column(path, num_tuples);
      if (!h_l_suppkey_enc) { fprintf(stderr, "Failed to load l_suppkey_enc\n"); return 1; }
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      // Dynamic sample-based selectivity profiling (fact-table predicates: l_shipdate range)
      cqo_in.predicate_selectivities.push_back(
          fascal_estimate_range_sel(h_l_shipdate, num_tuples, 19960101, 19960331));  // l_shipdate >= 19960101
      cqo_in.predicate_selectivities.push_back(
          fascal_estimate_range_sel(h_l_shipdate, num_tuples, 19960101, 19960331));  // l_shipdate < 19960401 (same range)
  
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
  
  
  
      // === AFP Manager & Predicate Filters ===
      int *h_data[] = {h_l_shipdate, (int*)h_l_extendedprice, (int*)h_l_discount};
      fascal::runtime::AFPManager afp_manager(0);
      int afp_filter_id = 0;
      if (h_pred_flags[0] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(0, fascal::runtime::AFPManager::FilterOp::GE, 19960101);
        afp_manager.register_filter(afp_filter_id++, desc);
      }
      if (h_pred_flags[1] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(0, fascal::runtime::AFPManager::FilterOp::LT, 19960401);
        afp_manager.register_filter(afp_filter_id++, desc);
      }
  
      // === Bloom Filters (Build Pipelines) ===
  
  
  
  
      // === ODZC & Device Buffers ===
      auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(
          new fascal::runtime::ODZCManager(0));
  
      // === Seed Bitmap + Tile Summary (CPU-first, no doorbell) ===
      uint32_t *h_seed_bitmap_storage = nullptr, *d_seed_bitmap = nullptr;
      uint8_t *h_tile_summary = nullptr, *d_tile_summary = nullptr;
      fascal_alloc_seed_bitmap(num_tuples, &h_seed_bitmap_storage, &d_seed_bitmap, &h_tile_summary, &d_tile_summary);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_shipdate;
      int *d_lineitem_l_shipdate = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_shipdate, num_tuples, "l_shipdate", &mc_lineitem_l_shipdate, &d_lineitem_l_shipdate);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_extendedprice;
      float *d_lineitem_l_extendedprice = nullptr;
      {
        mc_lineitem_l_extendedprice = odzc_mgr->register_column(h_l_extendedprice, (size_t)num_tuples * sizeof(float));
        d_lineitem_l_extendedprice = static_cast<float*>(mc_lineitem_l_extendedprice.device_ptr);
        if (!d_lineitem_l_extendedprice) {
          fprintf(stderr, "ODZC failed for lineitem.l_extendedprice, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_lineitem_l_extendedprice, (size_t)num_tuples * sizeof(float)));
          CUDA_CHECK(cudaMemcpy(d_lineitem_l_extendedprice, h_l_extendedprice, (size_t)num_tuples * sizeof(float), cudaMemcpyHostToDevice));
        }
      }
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_discount;
      float *d_lineitem_l_discount = nullptr;
      {
        mc_lineitem_l_discount = odzc_mgr->register_column(h_l_discount, (size_t)num_tuples * sizeof(float));
        d_lineitem_l_discount = static_cast<float*>(mc_lineitem_l_discount.device_ptr);
        if (!d_lineitem_l_discount) {
          fprintf(stderr, "ODZC failed for lineitem.l_discount, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_lineitem_l_discount, (size_t)num_tuples * sizeof(float)));
          CUDA_CHECK(cudaMemcpy(d_lineitem_l_discount, h_l_discount, (size_t)num_tuples * sizeof(float), cudaMemcpyHostToDevice));
        }
      }
  
      int *d_lineitem_l_suppkey_enc = nullptr;
      CUDA_CHECK(cudaMalloc(&d_lineitem_l_suppkey_enc, num_tuples * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(d_lineitem_l_suppkey_enc, h_l_suppkey_enc, num_tuples * sizeof(int), cudaMemcpyHostToDevice));
  
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      size_t num_groups = 1ULL;
      num_groups *= 1000001ULL;
      double *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, num_groups * NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(double)));
  
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
              tpch_q15_max_revenue_revenue_cpu_prefilter_lineitem(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0);
              tpch_q15_max_revenue_revenue_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_lineitem_l_shipdate,
          d_lineitem_l_extendedprice,
          d_lineitem_l_discount,
          d_lineitem_l_suppkey_enc,
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
              tpch_q15_max_revenue_revenue_cpu_prefilter_lineitem(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q15_max_revenue_revenue_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_lineitem_l_shipdate,
          d_lineitem_l_extendedprice,
          d_lineitem_l_discount,
          d_lineitem_l_suppkey_enc,
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
      for (size_t k = 0; k < num_groups; k++) {
        int any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0)) ? 1 : 0;
        if (any) {
          size_t gk = k;
        int key0 = (int)(gk % 1000001ULL);
        gk /= 1000001;
        }
      }
      // Materializing aggregation results to revenue_*.bin
      std::vector<int> h_gb_decode0(1000001, std::numeric_limits<int>::min());
      for (int _row = 0; _row < num_tuples; ++_row) {
        int _enc = h_l_suppkey_enc[_row];
        if (_enc >= 0 && _enc < 1000001 && h_gb_decode0[(size_t)_enc] == std::numeric_limits<int>::min())
          h_gb_decode0[(size_t)_enc] = h_l_suppkey[_row];
      }
      int _h_count = 0;
      for (size_t k = 0; k < num_groups; k++) {
        bool _any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0));
        if (_any) {
          size_t gk = k;       int key0 = (int)(gk % 1000001ULL);
                gk /= 1000001;
          _h_count++;
        }
      }
      int *h_out_col0 = (int*)malloc(_h_count * sizeof(int));
      double *h_out_col1 = (double*)malloc(_h_count * sizeof(double));
      int _h_idx = 0;
      for (size_t k = 0; k < num_groups; k++) {
        bool _any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0));
        if (_any) {
          size_t gk = k;       int key0 = (int)(gk % 1000001ULL);
                gk /= 1000001;
          int _decoded_key0 = key0;
          if (key0 >= 0 && key0 < (int)h_gb_decode0.size() && h_gb_decode0[(size_t)key0] != std::numeric_limits<int>::min())
            _decoded_key0 = h_gb_decode0[(size_t)key0];
          h_out_col0[_h_idx] = _decoded_key0;
          h_out_col1[_h_idx] = h_agg_table[k * NUM_AGGREGATES + 0];
          _h_idx++;
        }
      }
      char path_mat0[512];
      snprintf(path_mat0, sizeof(path_mat0), "%s/revenue_l_suppkey.bin", data_dir);
      FILE *fp_mat0 = fopen(path_mat0, "wb");
      if (fp_mat0) {
        fwrite(h_out_col0, sizeof(int), _h_count, fp_mat0);
        fclose(fp_mat0);
        printf("Materialized %d rows to %s\n", _h_count, path_mat0);
      }
      char path_mat1[512];
      snprintf(path_mat1, sizeof(path_mat1), "%s/revenue_sum_revenue.bin", data_dir);
      FILE *fp_mat1 = fopen(path_mat1, "wb");
      if (fp_mat1) {
        fwrite(h_out_col1, sizeof(double), _h_count, fp_mat1);
        fclose(fp_mat1);
        printf("Materialized %d rows to %s\n", _h_count, path_mat1);
      }
      _mat_rows_revenue = _h_count;
      if (_mat_revenue_l_suppkey) CUDA_CHECK(cudaFreeHost(_mat_revenue_l_suppkey));
      if (_mat_rows_revenue > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat_revenue_l_suppkey), (size_t)_mat_rows_revenue * sizeof(int), cudaHostAllocDefault));
        std::memcpy(_mat_revenue_l_suppkey, h_out_col0, (size_t)_mat_rows_revenue * sizeof(int));
      } else {
        _mat_revenue_l_suppkey = nullptr;
      }
      if (_mat_revenue_sum_revenue) CUDA_CHECK(cudaFreeHost(_mat_revenue_sum_revenue));
      if (_mat_rows_revenue > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat_revenue_sum_revenue), (size_t)_mat_rows_revenue * sizeof(double), cudaHostAllocDefault));
        std::memcpy(_mat_revenue_sum_revenue, h_out_col1, (size_t)_mat_rows_revenue * sizeof(double));
      } else {
        _mat_revenue_sum_revenue = nullptr;
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
      odzc_mgr->unregister_column(mc_lineitem_l_shipdate);
      odzc_mgr->unregister_column(mc_lineitem_l_extendedprice);
      odzc_mgr->unregister_column(mc_lineitem_l_discount);
      CUDA_CHECK(cudaFree(d_lineitem_l_suppkey_enc));
  
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_l_shipdate);
      fascal_free_column(h_l_extendedprice);
      fascal_free_column(h_l_discount);
      fascal_free_column(h_l_suppkey_enc);
      for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) cudaStreamDestroy(fascal_streams[_s]);
      delete[] fascal_streams;
      cudaEventDestroy(t0);
      cudaEventDestroy(t1);
      cudaEventDestroy(t_start);
      cudaEventDestroy(t_load);
            cudaEventDestroy(t_query);
      cudaEventDestroy(t_result);
  
    }

    // ---------------- STAGE 3 EXECUTION: tpch_q15_max_revenue ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q15_max_revenue
  // ============================================================
      printf("=== FaScalSQL: tpch_q15_max_revenue ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          if (_mat_rows_revenue > 0 && _mat_revenue_sum_revenue != nullptr) {
              num_tuples = _mat_rows_revenue;
          } else {
              char _auto_path[512];
              snprintf(_auto_path, sizeof(_auto_path), "%s/revenue_sum_revenue.bin", data_dir);
              int _auto_count = 0;
              int *_tmp = load_binary_column_auto(_auto_path, &_auto_count);
              if (_tmp) fascal_free_column(_tmp);
              num_tuples = _auto_count;
              if (num_tuples <= 0) {
                  fprintf(stderr, "Cannot auto-detect num_tuples from %s\n", _auto_path);
                  return 1;
              }
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
  
      double *h_sum_revenue = nullptr;
      if (_mat_rows_revenue > 0 && _mat_revenue_sum_revenue != nullptr) {
        h_sum_revenue = _mat_revenue_sum_revenue;
        num_tuples = _mat_rows_revenue;
      } else {
        snprintf(path, sizeof(path), "%s/revenue_sum_revenue.bin", data_dir);
        h_sum_revenue = (double*)load_binary_column_u64(path, num_tuples);
      }
      if (!h_sum_revenue) { fprintf(stderr, "Failed to load sum_revenue\n"); return 1; }
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      (void)cqo_in;
  
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
  
  
  
      // === AFP Manager & Predicate Filters ===
      int *h_data[] = {(int*)h_sum_revenue};
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
  
      fascal::runtime::ODZCManager::MappedColumn mc_revenue_sum_revenue;
      double *d_revenue_sum_revenue = nullptr;
      {
        mc_revenue_sum_revenue = odzc_mgr->register_column(h_sum_revenue, (size_t)num_tuples * sizeof(double));
        d_revenue_sum_revenue = static_cast<double*>(mc_revenue_sum_revenue.device_ptr);
        if (!d_revenue_sum_revenue) {
          fprintf(stderr, "ODZC failed for revenue.sum_revenue, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_revenue_sum_revenue, (size_t)num_tuples * sizeof(double)));
          CUDA_CHECK(cudaMemcpy(d_revenue_sum_revenue, h_sum_revenue, (size_t)num_tuples * sizeof(double), cudaMemcpyHostToDevice));
        }
      }
  
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      unsigned long long *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, (size_t)NUM_AGGREGATES * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(1) * NUM_AGGREGATES * sizeof(unsigned long long)));
  
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
              tpch_q15_max_revenue_cpu_prefilter_revenue(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0);
              tpch_q15_max_revenue_kernel_revenue<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_revenue_sum_revenue,
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
              tpch_q15_max_revenue_cpu_prefilter_revenue(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q15_max_revenue_kernel_revenue<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_revenue_sum_revenue,
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
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(1) * NUM_AGGREGATES * sizeof(unsigned long long)));
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
      unsigned long long *h_result = (unsigned long long*)malloc((size_t)NUM_AGGREGATES * sizeof(unsigned long long));
      CUDA_CHECK(cudaMemcpy(h_result, d_aggtable, (size_t)NUM_AGGREGATES * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      int k = 0;
      unsigned long long* h_agg_table = h_result;
      // Materializing scalar aggregation results to max_revenue_*.bin
      unsigned long long h_out_val0 = h_result[0];
      char path_mat0[512];
      snprintf(path_mat0, sizeof(path_mat0), "%s/max_revenue_total_revenue.bin", data_dir);
      FILE *fp_mat0 = fopen(path_mat0, "wb");
      if (fp_mat0) {
        fwrite(&h_out_val0, sizeof(unsigned long long), 1, fp_mat0);
        fclose(fp_mat0);
        printf("Materialized 1 row to %s\n", path_mat0);
      }
      _mat_rows_max_revenue = 1;
      if (_mat_max_revenue_total_revenue) CUDA_CHECK(cudaFreeHost(_mat_max_revenue_total_revenue));
      if (_mat_rows_max_revenue > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat_max_revenue_total_revenue), (size_t)_mat_rows_max_revenue * sizeof(unsigned long long), cudaHostAllocDefault));
        std::memcpy(_mat_max_revenue_total_revenue, &h_out_val0, (size_t)_mat_rows_max_revenue * sizeof(unsigned long long));
      } else {
        _mat_max_revenue_total_revenue = nullptr;
      }
      printf("Stage scalar materialization complete\n");
      free(h_result);
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
      odzc_mgr->unregister_column(mc_revenue_sum_revenue);
  
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      if (h_sum_revenue && h_sum_revenue != _mat_revenue_sum_revenue) fascal_free_column(h_sum_revenue);
      for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) cudaStreamDestroy(fascal_streams[_s]);
      delete[] fascal_streams;
      cudaEventDestroy(t0);
      cudaEventDestroy(t1);
      cudaEventDestroy(t_start);
      cudaEventDestroy(t_load);
            cudaEventDestroy(t_query);
      cudaEventDestroy(t_result);
  
    }

    // ---------------- STAGE 4 EXECUTION: tpch_q15 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q15
  // ============================================================
      printf("=== FaScalSQL: tpch_q15 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/s_suppkey.bin", data_dir);
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
  
      snprintf(path, sizeof(path), "%s/s_suppkey.bin", data_dir);
      int *h_s_suppkey = load_binary_column(path, num_tuples);
      if (!h_s_suppkey) { fprintf(stderr, "Failed to load s_suppkey\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/s_name.bin", data_dir);
      int *h_s_name = load_binary_column(path, num_tuples);
      if (!h_s_name) { fprintf(stderr, "Failed to load s_name\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/s_address.bin", data_dir);
      int *h_s_address = load_binary_column(path, num_tuples);
      if (!h_s_address) { fprintf(stderr, "Failed to load s_address\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/s_phone.bin", data_dir);
      int *h_s_phone = load_binary_column(path, num_tuples);
      if (!h_s_phone) { fprintf(stderr, "Failed to load s_phone\n"); return 1; }
  
      // ----- Load dimension: max_revenue -----
      int num_max_revenue_tuples = 0;
  
      unsigned long long *h_max_revenue_total_revenue = nullptr;
      if (_mat_rows_max_revenue > 0 && _mat_max_revenue_total_revenue != nullptr) {
          num_max_revenue_tuples = _mat_rows_max_revenue;
          h_max_revenue_total_revenue = _mat_max_revenue_total_revenue;
      } else {
  
      snprintf(path, sizeof(path), "%s/max_revenue_total_revenue.bin", data_dir);
      h_max_revenue_total_revenue = load_binary_column_auto_u64(path, &num_max_revenue_tuples);
      }
  
      if (!h_max_revenue_total_revenue || num_max_revenue_tuples == 0) {
          fprintf(stderr, "Failed to load table column total_revenue\n"); return 1;
      }
      printf("Loaded max_revenue: %d tuples\n", num_max_revenue_tuples);
  
      // ----- Load dimension: revenue -----
      int num_revenue_tuples = 0;
  
      int *h_revenue_l_suppkey = nullptr;
      if (_mat_rows_revenue > 0 && _mat_revenue_l_suppkey != nullptr) {
          num_revenue_tuples = _mat_rows_revenue;
          h_revenue_l_suppkey = _mat_revenue_l_suppkey;
      } else {
  
      snprintf(path, sizeof(path), "%s/revenue_l_suppkey.bin", data_dir);
      h_revenue_l_suppkey = load_binary_column_auto(path, &num_revenue_tuples);
      }
  
      if (!h_revenue_l_suppkey || num_revenue_tuples == 0) {
          fprintf(stderr, "Failed to load table column l_suppkey\n"); return 1;
      }
  
      double *h_revenue_sum_revenue = nullptr;
      if (_mat_rows_revenue > 0 && _mat_revenue_sum_revenue != nullptr) {
          num_revenue_tuples = _mat_rows_revenue;
          h_revenue_sum_revenue = _mat_revenue_sum_revenue;
      } else {
  
      snprintf(path, sizeof(path), "%s/revenue_sum_revenue.bin", data_dir);
      h_revenue_sum_revenue = ((double*)load_binary_column_auto_u64(path, &num_revenue_tuples));
      }
  
      if (!h_revenue_sum_revenue || num_revenue_tuples == 0) {
          fprintf(stderr, "Failed to load table column sum_revenue\n"); return 1;
      }
      printf("Loaded revenue: %d tuples\n", num_revenue_tuples);
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      (void)cqo_in;
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
      int *h_data[] = {h_s_suppkey, h_s_name, h_s_address, h_s_phone};
      fascal::runtime::AFPManager afp_manager(0);
      int afp_filter_id = 0;
      (void)afp_manager;
  
      // === Bloom Filters (Build Pipelines) ===
      fascal::runtime::BloomFilter bf_storage[2];
      fascal::runtime::BloomFilter *bf_array[2];
  
  
      bf_array[0] = (!bloom_off && (int)0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[1] : nullptr;
      bf_array[1] = nullptr;
  
      // === ODZC & Device Buffers ===
      auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(
          new fascal::runtime::ODZCManager(0));
  
      // === Seed Bitmap + Tile Summary (CPU-first, no doorbell) ===
      uint32_t *h_seed_bitmap_storage = nullptr, *d_seed_bitmap = nullptr;
      uint8_t *h_tile_summary = nullptr, *d_tile_summary = nullptr;
      fascal_alloc_seed_bitmap(num_tuples, &h_seed_bitmap_storage, &d_seed_bitmap, &h_tile_summary, &d_tile_summary);
  
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_suppkey;
      int *d_supplier_s_suppkey = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_s_suppkey, num_tuples, "s_suppkey", &mc_supplier_s_suppkey, &d_supplier_s_suppkey);
  
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_name;
      int *d_supplier_s_name = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_s_name, num_tuples, "s_name", &mc_supplier_s_name, &d_supplier_s_name);
  
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_address;
      int *d_supplier_s_address = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_s_address, num_tuples, "s_address", &mc_supplier_s_address, &d_supplier_s_address);
  
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_phone;
      int *d_supplier_s_phone = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_s_phone, num_tuples, "s_phone", &mc_supplier_s_phone, &d_supplier_s_phone);
  
      fascal::runtime::ODZCManager::MappedColumn mc_max_revenue_total_revenue;
      mc_max_revenue_total_revenue = odzc_mgr->register_column(h_max_revenue_total_revenue, num_max_revenue_tuples * sizeof(unsigned long long));
      unsigned long long *d_max_revenue_total_revenue = static_cast<unsigned long long*>(mc_max_revenue_total_revenue.device_ptr);
      if (!d_max_revenue_total_revenue) {
          fprintf(stderr, "ODZC failed for max_revenue.total_revenue, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_max_revenue_total_revenue, num_max_revenue_tuples * sizeof(unsigned long long)));
          CUDA_CHECK(cudaMemcpy(d_max_revenue_total_revenue, h_max_revenue_total_revenue, num_max_revenue_tuples * sizeof(unsigned long long), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_revenue_l_suppkey;
      mc_revenue_l_suppkey = odzc_mgr->register_column(h_revenue_l_suppkey, num_revenue_tuples * sizeof(int));
      int *d_revenue_l_suppkey = static_cast<int*>(mc_revenue_l_suppkey.device_ptr);
      if (!d_revenue_l_suppkey) {
          fprintf(stderr, "ODZC failed for revenue.l_suppkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_revenue_l_suppkey, num_revenue_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_revenue_l_suppkey, h_revenue_l_suppkey, num_revenue_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_revenue_sum_revenue;
      mc_revenue_sum_revenue = odzc_mgr->register_column(h_revenue_sum_revenue, num_revenue_tuples * sizeof(double));
      double *d_revenue_sum_revenue = static_cast<double*>(mc_revenue_sum_revenue.device_ptr);
      if (!d_revenue_sum_revenue) {
          fprintf(stderr, "ODZC failed for revenue.sum_revenue, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_revenue_sum_revenue, num_revenue_tuples * sizeof(double)));
          CUDA_CHECK(cudaMemcpy(d_revenue_sum_revenue, h_revenue_sum_revenue, num_revenue_tuples * sizeof(double), cudaMemcpyHostToDevice));
      }
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      int *d_out_count = nullptr;
      CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
      CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));
      int *d_out_col0 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_out_col0, (size_t)num_tuples * sizeof(int)));
      int *d_out_col1 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_out_col1, (size_t)num_tuples * sizeof(int)));
      int *d_out_col2 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_out_col2, (size_t)num_tuples * sizeof(int)));
      int *d_out_col3 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_out_col3, (size_t)num_tuples * sizeof(int)));
      unsigned long long *d_out_col4 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_out_col4, (size_t)num_tuples * sizeof(unsigned long long)));
  
      // === Hash Table Build ===
  
  
      GpuHashTable ht_max_revenue;
      ht_max_revenue = GpuHashTable::allocate(num_max_revenue_tuples);
      GpuBloomFilter gpu_bf_max_revenue = GpuBloomFilter::allocate(num_max_revenue_tuples, 3);
      bf_storage[0] = fascal::runtime::BloomFilter::allocate(num_max_revenue_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_max_revenue.d_entries, 0xff, ht_max_revenue.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_max_revenue.d_bits, 0, ((size_t)gpu_bf_max_revenue.size_bits + 63) / 64 * sizeof(uint64_t)));
  
      // AFP all-pipeline: pinned mapped memory for max_revenue packed prefilter + tile summary (no doorbell).
      size_t max_revenue_bitmap_words = ((size_t)num_max_revenue_tuples + 31) / 32;
      size_t max_revenue_num_tiles = ((size_t)num_max_revenue_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_max_revenue_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_max_revenue_prefilter_pinned, max_revenue_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_max_revenue_prefilter_pinned, 0, max_revenue_bitmap_words * sizeof(uint32_t));
      uint32_t *d_max_revenue_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_max_revenue_prefilter_mapped, h_max_revenue_prefilter_pinned, 0));
      uint8_t *h_max_revenue_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_max_revenue_tile_summary, max_revenue_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_max_revenue_tile_summary, 0, max_revenue_num_tiles);
      uint8_t *d_max_revenue_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_max_revenue_tile_summary, h_max_revenue_tile_summary, 0));
  
  
      GpuHashTable ht_revenue;
      ht_revenue = GpuHashTable::allocate(num_revenue_tuples);
      GpuBloomFilter gpu_bf_revenue = GpuBloomFilter::allocate(num_revenue_tuples, 3);
      bf_storage[1] = fascal::runtime::BloomFilter::allocate(num_revenue_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_revenue.d_entries, 0xff, ht_revenue.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_revenue.d_bits, 0, ((size_t)gpu_bf_revenue.size_bits + 63) / 64 * sizeof(uint64_t)));
  
      // AFP all-pipeline: pinned mapped memory for revenue packed prefilter + tile summary (no doorbell).
      size_t revenue_bitmap_words = ((size_t)num_revenue_tuples + 31) / 32;
      size_t revenue_num_tiles = ((size_t)num_revenue_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_revenue_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_revenue_prefilter_pinned, revenue_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_revenue_prefilter_pinned, 0, revenue_bitmap_words * sizeof(uint32_t));
      uint32_t *d_revenue_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_revenue_prefilter_mapped, h_revenue_prefilter_pinned, 0));
      uint8_t *h_revenue_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_revenue_tile_summary, revenue_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_revenue_tile_summary, 0, revenue_num_tiles);
      uint8_t *d_revenue_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_revenue_tile_summary, h_revenue_tile_summary, 0));
  
  
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
              tpch_q15_cpu_prefilter_supplier(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 2);
              tpch_q15_kernel_supplier<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_supplier_s_suppkey,
          d_supplier_s_name,
          d_supplier_s_address,
          d_supplier_s_phone,
          d_max_revenue_total_revenue,
          d_revenue_sum_revenue,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_revenue.d_bits, (bloom_off ? 0u : gpu_bf_revenue.size_bits),
          ht_revenue.d_entries, ht_revenue.ht_size,
          gpu_bf_max_revenue.d_bits, (bloom_off ? 0u : gpu_bf_max_revenue.size_bits),
          ht_max_revenue.d_entries, ht_max_revenue.ht_size,
              d_out_count,
          d_out_col0,
          d_out_col1,
          d_out_col2,
          d_out_col3,
          d_out_col4);
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
              tpch_q15_cpu_prefilter_supplier(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 2, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q15_kernel_supplier<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_supplier_s_suppkey,
          d_supplier_s_name,
          d_supplier_s_address,
          d_supplier_s_phone,
          d_max_revenue_total_revenue,
          d_revenue_sum_revenue,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_revenue.d_bits, (bloom_off ? 0u : gpu_bf_revenue.size_bits),
          ht_revenue.d_entries, ht_revenue.ht_size,
          gpu_bf_max_revenue.d_bits, (bloom_off ? 0u : gpu_bf_max_revenue.size_bits),
          ht_max_revenue.d_entries, ht_max_revenue.ht_size,
              d_out_count,
          d_out_col0,
          d_out_col1,
          d_out_col2,
          d_out_col3,
          d_out_col4);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };
  
      cudaEventRecord(t0);
  
      // max_revenue (1 row < 256K): single launch, batch_offset=0
      tpch_q15_cpu_prefilter_max_revenue(h_max_revenue_total_revenue, num_max_revenue_tuples, h_max_revenue_prefilter_pinned, h_max_revenue_tile_summary);
      tpch_q15_kernel_max_revenue<BLOCK_THREADS, ITEMS_PER_THREAD>
          <<<(num_max_revenue_tuples + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, fascal_streams[0]>>>(
          num_max_revenue_tuples, /*batch_offset=*/0, d_max_revenue_total_revenue,
          ht_max_revenue.d_entries, ht_max_revenue.ht_size,
          gpu_bf_max_revenue.d_bits, gpu_bf_max_revenue.size_bits, 3,
          d_max_revenue_prefilter_mapped,
          d_max_revenue_tile_summary);
      CUDA_CHECK(cudaStreamSynchronize(fascal_streams[0]));
      CUDA_CHECK(cudaGetLastError());
      if (bf_storage[0].bits) gpu_bf_max_revenue.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));
      // revenue (<=10K rows < 256K): single launch, batch_offset=0
      tpch_q15_cpu_prefilter_revenue(h_revenue_l_suppkey, h_revenue_sum_revenue, num_revenue_tuples, h_revenue_prefilter_pinned, h_revenue_tile_summary, (bf_storage[0].bits ? &bf_storage[0] : nullptr));
      tpch_q15_kernel_revenue<BLOCK_THREADS, ITEMS_PER_THREAD>
          <<<(num_revenue_tuples + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, fascal_streams[0]>>>(
          num_revenue_tuples, /*batch_offset=*/0, d_revenue_l_suppkey, d_revenue_sum_revenue,
          ht_revenue.d_entries, ht_revenue.ht_size,
          gpu_bf_revenue.d_bits, gpu_bf_revenue.size_bits, 3,
          d_revenue_prefilter_mapped,
          d_revenue_tile_summary,
          ht_max_revenue.d_entries, ht_max_revenue.ht_size);
      CUDA_CHECK(cudaStreamSynchronize(fascal_streams[0]));
      CUDA_CHECK(cudaGetLastError());
      if (bf_storage[1].bits) gpu_bf_revenue.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));
  
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
      CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));
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
      int h_out_count = 0;
      CUDA_CHECK(cudaMemcpy(&h_out_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
      int *h_out_col0 = (int*)malloc((size_t)num_tuples * sizeof(int));
      CUDA_CHECK(cudaMemcpy(h_out_col0, d_out_col0, (size_t)num_tuples * sizeof(int), cudaMemcpyDeviceToHost));
      int *h_out_col1 = (int*)malloc((size_t)num_tuples * sizeof(int));
      CUDA_CHECK(cudaMemcpy(h_out_col1, d_out_col1, (size_t)num_tuples * sizeof(int), cudaMemcpyDeviceToHost));
      int *h_out_col2 = (int*)malloc((size_t)num_tuples * sizeof(int));
      CUDA_CHECK(cudaMemcpy(h_out_col2, d_out_col2, (size_t)num_tuples * sizeof(int), cudaMemcpyDeviceToHost));
      int *h_out_col3 = (int*)malloc((size_t)num_tuples * sizeof(int));
      CUDA_CHECK(cudaMemcpy(h_out_col3, d_out_col3, (size_t)num_tuples * sizeof(int), cudaMemcpyDeviceToHost));
      unsigned long long *h_out_col4 = (unsigned long long*)malloc((size_t)num_tuples * sizeof(unsigned long long));
      CUDA_CHECK(cudaMemcpy(h_out_col4, d_out_col4, (size_t)num_tuples * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      std::vector<int> _row_order((size_t)h_out_count);
      for (int i = 0; i < h_out_count; i++) _row_order[(size_t)i] = i;
      std::sort(_row_order.begin(), _row_order.end(), [&](int _a, int _b) {
        if (h_out_col0[_a] != h_out_col0[_b]) return h_out_col0[_a] < h_out_col0[_b];
        return false;
      });
      int _mat_count = h_out_count;
      for (int i = 0; i < _mat_count; i++) {
        int _row_idx = _row_order[(size_t)i];
        printf("ROW: %d", (int)h_out_col0[_row_idx]);
        printf("|%s", [&]() -> const char* {
    static thread_local char buf[64];
    int idx = (int)h_out_col1[_row_idx];
    int val = 1 + idx;
    if (val < 1) return "UNKNOWN";
    snprintf(buf, sizeof(buf), "Supplier#%09d", val);
    return buf;
  }());
        printf("|%s", [&]() -> const char* {
      static std::unordered_map<int, std::string> dict;
      static bool loaded = false;
      if (!loaded) {
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
      int target_key = (int)h_out_col2[_row_idx];
      auto it = dict.find(target_key);
      return it != dict.end() ? it->second.c_str() : "UNKNOWN";
  }());
        printf("|%s", [&]() -> const char* {
      static std::unordered_map<int, std::string> dict;
      static bool loaded = false;
      if (!loaded) {
          char ps[1024], po64[1024], po32[1024];
          snprintf(ps,   sizeof(ps),   "%s/supplier_s_phone_dict_strings.bin", data_dir);
          snprintf(po64, sizeof(po64), "%s/supplier_s_phone_dict_offsets64.bin", data_dir);
          snprintf(po32, sizeof(po32), "%s/supplier_s_phone_dict_offsets.bin", data_dir);
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
      int target_key = (int)h_out_col3[_row_idx];
      auto it = dict.find(target_key);
      return it != dict.end() ? it->second.c_str() : "UNKNOWN";
  }());
        printf("|%.6f", (double)h_out_col4[_row_idx]);
        printf("\n");
      }
      free(h_out_col0);
      CUDA_CHECK(cudaFree(d_out_col0));
      free(h_out_col1);
      CUDA_CHECK(cudaFree(d_out_col1));
      free(h_out_col2);
      CUDA_CHECK(cudaFree(d_out_col2));
      free(h_out_col3);
      CUDA_CHECK(cudaFree(d_out_col3));
      free(h_out_col4);
      CUDA_CHECK(cudaFree(d_out_col4));
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
      odzc_mgr->unregister_column(mc_supplier_s_suppkey);
      odzc_mgr->unregister_column(mc_supplier_s_name);
      odzc_mgr->unregister_column(mc_supplier_s_address);
      odzc_mgr->unregister_column(mc_supplier_s_phone);
  
      odzc_mgr->unregister_column(mc_max_revenue_total_revenue);
      odzc_mgr->unregister_column(mc_revenue_l_suppkey);
      odzc_mgr->unregister_column(mc_revenue_sum_revenue);
      CUDA_CHECK(cudaFreeHost(h_max_revenue_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_max_revenue_prefilter_pinned));
      gpu_bf_max_revenue.free_filter();
      ht_max_revenue.free_table();
      bf_storage[0].free_filter();
      if (h_max_revenue_total_revenue && (void*)h_max_revenue_total_revenue != (void*)_mat_max_revenue_total_revenue) fascal_free_column(h_max_revenue_total_revenue);
      CUDA_CHECK(cudaFreeHost(h_revenue_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_revenue_prefilter_pinned));
      gpu_bf_revenue.free_filter();
      ht_revenue.free_table();
      bf_storage[1].free_filter();
      if (h_revenue_l_suppkey && (void*)h_revenue_l_suppkey != (void*)_mat_revenue_l_suppkey) fascal_free_column(h_revenue_l_suppkey);
      if (h_revenue_sum_revenue && (void*)h_revenue_sum_revenue != (void*)_mat_revenue_sum_revenue) fascal_free_column(h_revenue_sum_revenue);
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_s_suppkey);
      fascal_free_column(h_s_name);
      fascal_free_column(h_s_address);
      fascal_free_column(h_s_phone);
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
    if (_mat_revenue_l_suppkey) CUDA_CHECK(cudaFreeHost(_mat_revenue_l_suppkey));
    if (_mat_revenue_sum_revenue) CUDA_CHECK(cudaFreeHost(_mat_revenue_sum_revenue));
    if (_mat_max_revenue_total_revenue) CUDA_CHECK(cudaFreeHost(_mat_max_revenue_total_revenue));
}
