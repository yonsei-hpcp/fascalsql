// TPC-H Q11 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Source: fascalsql/kernels/tpch_q11.cu

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
#include "fascal_query_runtime.cuh"
#include "fascal_gpu_macros.cuh"
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

// ================= STAGE 1: tpch_q11_stage1 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q11_stage1_kernel_partsupp(
    int num_tuples,
    int batch_offset,
    int *d_partsupp_ps_suppkey,
    int *d_partsupp_ps_supplycost,
    int *d_partsupp_ps_availqty,
    int *d_partsupp_ps_partkey_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_supplier,
    uint32_t bloom_size_bits_supplier,
    HtEntry *ht_supplier,
    uint32_t ht_size_supplier,
    const uint64_t *__restrict__ d_bloom_nation,
    uint32_t bloom_size_bits_nation,
    HtEntry *ht_nation,
    uint32_t ht_size_nation,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  int loc_partsupp_ps_suppkey[ITEMS_PER_THREAD_T];
  int loc_partsupp_ps_supplycost[ITEMS_PER_THREAD_T];
  int loc_partsupp_ps_availqty[ITEMS_PER_THREAD_T];
  int loc_supplier_s_nationkey[ITEMS_PER_THREAD_T];
  int loc_partsupp_ps_partkey_enc[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // Phase 1: Semi-joins (Bloom filter); each FK loaded immediately before its probe
  // ODZC lazy load (join FK, before supplier probe): ps_suppkey
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_partsupp_ps_suppkey + tile_offset, loc_partsupp_ps_suppkey,
      selection_flags, num_tile_items);

  BlockBloomProbe<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_partsupp_ps_suppkey, selection_flags, num_tile_items,
      bloom_size_bits_supplier, d_bloom_supplier, 3);

  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  // Phase 2: All hash table joins
  BlockJoinProbePayloadDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_partsupp_ps_suppkey, selection_flags, num_tile_items,
      ht_supplier, ht_size_supplier, 1, loc_supplier_s_nationkey);
  BlockJoinProbeDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_supplier_s_nationkey, selection_flags, num_tile_items,
      ht_nation, ht_size_nation, 0);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_partsupp_ps_partkey_enc + tile_offset, loc_partsupp_ps_partkey_enc,
      selection_flags, num_tile_items);
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_partsupp_ps_supplycost + tile_offset, loc_partsupp_ps_supplycost,
      selection_flags, num_tile_items);
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_partsupp_ps_availqty + tile_offset, loc_partsupp_ps_availqty,
      selection_flags, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int gk = loc_partsupp_ps_partkey_enc[ITEM];
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], (((((double)((double)loc_partsupp_ps_supplycost[ITEM])) / 100.0) * loc_partsupp_ps_availqty[ITEM])));
      }
  }
  } while(0);
}
static void tpch_q11_stage1_cpu_predicate_partsupp(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bloom_filters[], int num_bloom_filters) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  if (bloom_filters && 0 < num_bloom_filters && bloom_filters[0])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, h_data[0], offset, cnt, bloom_filters[0]);
}

static void tpch_q11_stage1_cpu_prefilter_partsupp(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
    fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
        [&](int off, int cnt) { tpch_q11_stage1_cpu_predicate_partsupp(h_data, off, cnt, bitmap, bloom_filters, num_bloom_filters); },
        batch_offset, total_tuples);
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q11_stage1_kernel_supplier(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_supplier_s_suppkey,
    const int *__restrict__ d_supplier_s_nationkey,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_nation, uint32_t ht_size_f_nation)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_supplier_s_suppkey[ITEMS_PER_THREAD_T];
  int loc_supplier_s_nationkey[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_supplier_s_suppkey + tile_offset), loc_supplier_s_suppkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_supplier_s_nationkey + tile_offset), loc_supplier_s_nationkey, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe_direct(loc_supplier_s_nationkey[ITEM], d_filter_ht_nation, ht_size_f_nation, 0, nullptr)) continue;
    int payload = loc_supplier_s_nationkey[ITEM];  // value-as-payload: store s_nationkey instead of row_id
    gpu_ht_insert_one_direct(loc_supplier_s_suppkey[ITEM], payload, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_supplier_s_suppkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'supplier' ----
// Fills entire prefilter bitmap (scalar preds + upstream BF probes),
// then GPU build kernel launches with bitmap 100% ready (no doorbell).
static void tpch_q11_stage1_cpu_prefilter_supplier(
    int *h_supplier_s_suppkey,
    int *h_supplier_s_nationkey,
    int num_tuples,
    uint32_t *bitmap,
    uint8_t *tile_summary,
    fascal::runtime::BloomFilter *upstream_bf_nation) {
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
                if (upstream_bf_nation && upstream_bf_nation->bits) {
                    upstream_bf_nation->probe_and_mask_packed(h_supplier_s_nationkey + offset, bitmap, offset, cnt);
                }
        });
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q11_stage1_kernel_nation(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_nation_n_nationkey,
    const int *__restrict__ d_nation_n_name,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_nation_n_nationkey[ITEMS_PER_THREAD_T];
  int loc_nation_n_name[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_nation_n_nationkey + tile_offset), loc_nation_n_nationkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_nation_n_name + tile_offset), loc_nation_n_name, selection_flags, num_tile_items);

  BlockPredAndEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_nation_n_name, 1, selection_flags, num_tile_items);

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int payload = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one_direct(loc_nation_n_nationkey[ITEM], payload, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(loc_nation_n_nationkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'nation' ----
// Fills entire prefilter bitmap (scalar preds + upstream BF probes),
// then GPU build kernel launches with bitmap 100% ready (no doorbell).
static void tpch_q11_stage1_cpu_prefilter_nation(
    int *h_nation_n_nationkey,
    int *h_nation_n_name,
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
                if ((h_nation_n_name[idx] == 1)) {
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
                }
            }
            // 2. Apply upstream BF probes (AND-mask with FK column probes).

        });
}

// ================= STAGE 2: tpch_q11_stage2___scalar_sq_1 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q11_stage2___scalar_sq_1_kernel_partsupp(
    int num_tuples,
    int batch_offset,
    int *d_partsupp_ps_suppkey,
    int *d_partsupp_ps_supplycost,
    int *d_partsupp_ps_availqty,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_supplier,
    uint32_t bloom_size_bits_supplier,
    HtEntry *ht_supplier,
    uint32_t ht_size_supplier,
    const uint64_t *__restrict__ d_bloom_nation,
    uint32_t bloom_size_bits_nation,
    HtEntry *ht_nation,
    uint32_t ht_size_nation,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  int loc_partsupp_ps_suppkey[ITEMS_PER_THREAD_T];
  int loc_partsupp_ps_supplycost[ITEMS_PER_THREAD_T];
  int loc_partsupp_ps_availqty[ITEMS_PER_THREAD_T];
  int loc_supplier_s_nationkey[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_partsupp_ps_suppkey + tile_offset, loc_partsupp_ps_suppkey,
      selection_flags, num_tile_items);
  BlockBloomProbe<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_partsupp_ps_suppkey, selection_flags, num_tile_items,
      bloom_size_bits_supplier, d_bloom_supplier, 3);
  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  BlockJoinProbePayloadDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_partsupp_ps_suppkey, selection_flags, num_tile_items,
      ht_supplier, ht_size_supplier, 1, loc_supplier_s_nationkey);
  BlockJoinProbeDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_supplier_s_nationkey, selection_flags, num_tile_items,
      ht_nation, ht_size_nation, 0);

  // ODZC lazy load (aggregation only): ps_supplycost
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_partsupp_ps_supplycost + tile_offset, loc_partsupp_ps_supplycost,
      selection_flags, num_tile_items);

  // ODZC lazy load (aggregation only): ps_availqty
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_partsupp_ps_availqty + tile_offset, loc_partsupp_ps_availqty,
      selection_flags, num_tile_items);

  } while(0); // thread_any_alive
  // Scalar aggregation with block-level reduction (one atomic per block)
  double _acc0 = 0;
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        _acc0 += (((((double)((double)loc_partsupp_ps_supplycost[ITEM])) / 100.0) * loc_partsupp_ps_availqty[ITEM]));
      }
  }
  _acc0 = blockReduceSumDouble<BLOCK_THREADS_T>(_acc0);
  if (threadIdx.x == 0) atomicAdd(&aggtable[0], (double)_acc0);
}
static void tpch_q11_stage2___scalar_sq_1_cpu_predicate_partsupp(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bloom_filters[], int num_bloom_filters) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  if (bloom_filters && 0 < num_bloom_filters && bloom_filters[0])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, h_data[0], offset, cnt, bloom_filters[0]);
}

static void tpch_q11_stage2___scalar_sq_1_cpu_prefilter_partsupp(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
    fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
        [&](int off, int cnt) { tpch_q11_stage2___scalar_sq_1_cpu_predicate_partsupp(h_data, off, cnt, bitmap, bloom_filters, num_bloom_filters); },
        batch_offset, total_tuples);
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q11_stage2___scalar_sq_1_kernel_supplier(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_supplier_s_suppkey,
    const int *__restrict__ d_supplier_s_nationkey,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_nation, uint32_t ht_size_f_nation)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_supplier_s_suppkey[ITEMS_PER_THREAD_T];
  int loc_supplier_s_nationkey[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_supplier_s_suppkey + tile_offset), loc_supplier_s_suppkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_supplier_s_nationkey + tile_offset), loc_supplier_s_nationkey, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe_direct(loc_supplier_s_nationkey[ITEM], d_filter_ht_nation, ht_size_f_nation, 0, nullptr)) continue;
    int payload = loc_supplier_s_nationkey[ITEM];  // value-as-payload: store s_nationkey instead of row_id
    gpu_ht_insert_one_direct(loc_supplier_s_suppkey[ITEM], payload, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_supplier_s_suppkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'supplier' ----
// Fills entire prefilter bitmap (scalar preds + upstream BF probes),
// then GPU build kernel launches with bitmap 100% ready (no doorbell).
static void tpch_q11_stage2___scalar_sq_1_cpu_prefilter_supplier(
    int *h_supplier_s_suppkey,
    int *h_supplier_s_nationkey,
    int num_tuples,
    uint32_t *bitmap,
    uint8_t *tile_summary,
    fascal::runtime::BloomFilter *upstream_bf_nation) {
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
                if (upstream_bf_nation && upstream_bf_nation->bits) {
                    upstream_bf_nation->probe_and_mask_packed(h_supplier_s_nationkey + offset, bitmap, offset, cnt);
                }
        });
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q11_stage2___scalar_sq_1_kernel_nation(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_nation_n_nationkey,
    const int *__restrict__ d_nation_n_name,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_nation_n_nationkey[ITEMS_PER_THREAD_T];
  int loc_nation_n_name[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_nation_n_nationkey + tile_offset), loc_nation_n_nationkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_nation_n_name + tile_offset), loc_nation_n_name, selection_flags, num_tile_items);

  BlockPredAndEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_nation_n_name, 1, selection_flags, num_tile_items);

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int payload = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one_direct(loc_nation_n_nationkey[ITEM], payload, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(loc_nation_n_nationkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'nation' ----
// Fills entire prefilter bitmap (scalar preds + upstream BF probes),
// then GPU build kernel launches with bitmap 100% ready (no doorbell).
static void tpch_q11_stage2___scalar_sq_1_cpu_prefilter_nation(
    int *h_nation_n_nationkey,
    int *h_nation_n_name,
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
                if ((h_nation_n_name[idx] == 1)) {
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
                }
            }
            // 2. Apply upstream BF probes (AND-mask with FK column probes).

        });
}

// ================= STAGE 3: tpch_q11_stage2 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q11_stage2_kernel___intermediate_agg_tpch_q11(
    int num_tuples,
    int batch_offset,
    double *d___scalar_sq_1_sum,
    int *d___intermediate_agg_tpch_q11_ps_partkey,
    double *d___intermediate_agg_tpch_q11_sum,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    int *d_out_count,
    int *d_out_col0,
    double *d_out_col1)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  double loc___scalar_sq_1_sum[ITEMS_PER_THREAD_T];
  int loc___intermediate_agg_tpch_q11_ps_partkey[ITEMS_PER_THREAD_T];
  double loc___intermediate_agg_tpch_q11_sum[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // Phase 1: Semi-joins (Bloom filter); each FK loaded immediately before its probe
  // Phase 2: All hash table joins
  // Broadcast load (scalar auxiliary table): __scalar_sq_1.sum
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM])
        loc___scalar_sq_1_sum[ITEM] = d___scalar_sq_1_sum[0];
  }

  // ODZC lazy load (aggregation only): sum
  BlockLoadSelect<double, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d___intermediate_agg_tpch_q11_sum + tile_offset, loc___intermediate_agg_tpch_q11_sum,
      selection_flags, num_tile_items);

  // ODZC lazy load (aggregation only): ps_partkey
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d___intermediate_agg_tpch_q11_ps_partkey + tile_offset, loc___intermediate_agg_tpch_q11_ps_partkey,
      selection_flags, num_tile_items);

  // ODZC lazy load (aggregation only): sum
  BlockLoadSelect<double, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d___intermediate_agg_tpch_q11_sum + tile_offset, loc___intermediate_agg_tpch_q11_sum,
      selection_flags, num_tile_items);

  // Post-join filter 0
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if (selection_flags[ITEM]) {
      if (!((loc___intermediate_agg_tpch_q11_sum[ITEM] > (loc___scalar_sq_1_sum[ITEM] * 1e-04))))
        selection_flags[ITEM] = 0;
    }
  }

  // Materialize: write passing rows to output buffers
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int idx = atomicAdd(d_out_count, 1);
        d_out_col0[idx] = loc___intermediate_agg_tpch_q11_ps_partkey[ITEM];
        d_out_col1[idx] = loc___intermediate_agg_tpch_q11_sum[ITEM];
      }
  }
  } while(0); // thread_any_alive
}
static void tpch_q11_stage2_cpu_predicate___intermediate_agg_tpch_q11(int *h_data[], int offset, int cnt, uint32_t *bitmap) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
}

static void tpch_q11_stage2_cpu_prefilter___intermediate_agg_tpch_q11(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
    fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
        [&](int off, int cnt) { tpch_q11_stage2_cpu_predicate___intermediate_agg_tpch_q11(h_data, off, cnt, bitmap); },
        batch_offset, total_tuples);
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
    int _mat_rows___intermediate_agg_tpch_q11 = 0;
    int *_mat___intermediate_agg_tpch_q11_ps_partkey = nullptr;
    double *_mat___intermediate_agg_tpch_q11_sum = nullptr;
    int _mat_rows___scalar_sq_1 = 0;
    double *_mat___scalar_sq_1_sum = nullptr;
    auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));

    // ---------------- STAGE 1 EXECUTION: tpch_q11_stage1 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q11_stage1
  // ============================================================
      printf("=== FaScalSQL: tpch_q11_stage1 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/ps_suppkey.bin", data_dir);
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
          size_t arena_cols = 9;
          size_t arena_bytes = (size_t)num_tuples * sizeof(int) * arena_cols;
          arena_bytes = arena_bytes + arena_bytes / 8;
          fascal_arena_init(arena_bytes);
      }

      // === Data Loading ===
      char path[512];
      // ----- Load result-pipeline (fact) columns -----
  
      snprintf(path, sizeof(path), "%s/ps_suppkey.bin", data_dir);
      int *h_ps_suppkey = load_binary_column(path, num_tuples);
      if (!h_ps_suppkey) { fprintf(stderr, "Failed to load ps_suppkey\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/ps_supplycost.bin", data_dir);
      int *h_ps_supplycost = load_binary_column(path, num_tuples);
      if (!h_ps_supplycost) { fprintf(stderr, "Failed to load ps_supplycost\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/ps_availqty.bin", data_dir);
      int *h_ps_availqty = load_binary_column(path, num_tuples);
      if (!h_ps_availqty) { fprintf(stderr, "Failed to load ps_availqty\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/ps_partkey.bin", data_dir);
      int *h_ps_partkey = load_binary_column(path, num_tuples);
      if (!h_ps_partkey) { fprintf(stderr, "Failed to load ps_partkey\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/ps_partkey_enc.bin", data_dir);
      int *h_ps_partkey_enc = load_binary_column(path, num_tuples);
      if (!h_ps_partkey_enc) { fprintf(stderr, "Failed to load ps_partkey_enc\n"); return 1; }
  
      // ----- Load dimension: nation -----
      int num_nation_tuples = 0;
  
      snprintf(path, sizeof(path), "%s/n_nationkey.bin", data_dir);
      int *h_nation_n_nationkey = load_binary_column_auto(path, &num_nation_tuples);
  
      if (!h_nation_n_nationkey || num_nation_tuples == 0) {
          fprintf(stderr, "Failed to load table column n_nationkey\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/n_name.bin", data_dir);
      int *h_nation_n_name = load_binary_column_auto(path, &num_nation_tuples);
  
      if (!h_nation_n_name || num_nation_tuples == 0) {
          fprintf(stderr, "Failed to load table column n_name\n"); return 1;
      }
      printf("Loaded nation: %d tuples\n", num_nation_tuples);
  
      // ----- Load dimension: supplier -----
      int num_supplier_tuples = 0;
  
      snprintf(path, sizeof(path), "%s/s_suppkey.bin", data_dir);
      int *h_supplier_s_suppkey = load_binary_column_auto(path, &num_supplier_tuples);
  
      if (!h_supplier_s_suppkey || num_supplier_tuples == 0) {
          fprintf(stderr, "Failed to load table column s_suppkey\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/s_nationkey.bin", data_dir);
      int *h_supplier_s_nationkey = load_binary_column_auto(path, &num_supplier_tuples);
  
      if (!h_supplier_s_nationkey || num_supplier_tuples == 0) {
          fprintf(stderr, "Failed to load table column s_nationkey\n"); return 1;
      }
      printf("Loaded supplier: %d tuples\n", num_supplier_tuples);
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      cqo_in.predicate_selectivities.push_back(0.100000);
      // Dynamic BF selectivity: nation(n_name=GERMANY=7), supplier(all)
      { int nq = (int)(fascal_estimate_eq_sel(h_nation_n_name, num_nation_tuples, 7) * num_nation_tuples);
        cqo_in.join_bf_selectivities.push_back(fascal_estimate_join_bf_sel(nq, num_nation_tuples));
        cqo_in.join_bf_selectivities.push_back(fascal_estimate_join_bf_sel(num_supplier_tuples, num_supplier_tuples)); }
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
      int *h_data[] = {h_ps_suppkey, h_ps_supplycost, h_ps_availqty};
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
  
      fascal::runtime::ODZCManager::MappedColumn mc_partsupp_ps_suppkey;
      int *d_partsupp_ps_suppkey = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_ps_suppkey, num_tuples, "ps_suppkey", &mc_partsupp_ps_suppkey, &d_partsupp_ps_suppkey);
  
      fascal::runtime::ODZCManager::MappedColumn mc_partsupp_ps_supplycost;
      int *d_partsupp_ps_supplycost = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_ps_supplycost, num_tuples, "ps_supplycost", &mc_partsupp_ps_supplycost, &d_partsupp_ps_supplycost);
  
      fascal::runtime::ODZCManager::MappedColumn mc_partsupp_ps_availqty;
      int *d_partsupp_ps_availqty = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_ps_availqty, num_tuples, "ps_availqty", &mc_partsupp_ps_availqty, &d_partsupp_ps_availqty);
  
      int *d_partsupp_ps_partkey_enc = nullptr;
      CUDA_CHECK(cudaMalloc(&d_partsupp_ps_partkey_enc, num_tuples * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(d_partsupp_ps_partkey_enc, h_ps_partkey_enc, num_tuples * sizeof(int), cudaMemcpyHostToDevice));
  
      fascal::runtime::ODZCManager::MappedColumn mc_nation_n_nationkey;
      mc_nation_n_nationkey = odzc_mgr->register_column(h_nation_n_nationkey, num_nation_tuples * sizeof(int));
      int *d_nation_n_nationkey = static_cast<int*>(mc_nation_n_nationkey.device_ptr);
      if (!d_nation_n_nationkey) {
          fprintf(stderr, "ODZC failed for nation.n_nationkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_nation_n_nationkey, num_nation_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_nation_n_nationkey, h_nation_n_nationkey, num_nation_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_nation_n_name;
      mc_nation_n_name = odzc_mgr->register_column(h_nation_n_name, num_nation_tuples * sizeof(int));
      int *d_nation_n_name = static_cast<int*>(mc_nation_n_name.device_ptr);
      if (!d_nation_n_name) {
          fprintf(stderr, "ODZC failed for nation.n_name, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_nation_n_name, num_nation_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_nation_n_name, h_nation_n_name, num_nation_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_suppkey;
      mc_supplier_s_suppkey = odzc_mgr->register_column(h_supplier_s_suppkey, num_supplier_tuples * sizeof(int));
      int *d_supplier_s_suppkey = static_cast<int*>(mc_supplier_s_suppkey.device_ptr);
      if (!d_supplier_s_suppkey) {
          fprintf(stderr, "ODZC failed for supplier.s_suppkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_supplier_s_suppkey, num_supplier_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_supplier_s_suppkey, h_supplier_s_suppkey, num_supplier_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_nationkey;
      mc_supplier_s_nationkey = odzc_mgr->register_column(h_supplier_s_nationkey, num_supplier_tuples * sizeof(int));
      int *d_supplier_s_nationkey = static_cast<int*>(mc_supplier_s_nationkey.device_ptr);
      if (!d_supplier_s_nationkey) {
          fprintf(stderr, "ODZC failed for supplier.s_nationkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_supplier_s_nationkey, num_supplier_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_supplier_s_nationkey, h_supplier_s_nationkey, num_supplier_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      size_t num_groups = 1ULL;
      num_groups *= 20000001ULL;
      double *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, num_groups * NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(double)));
  
      // === Hash Table Build ===
  
  
      GpuHashTable ht_nation;
      ht_nation = GpuHashTable::allocate_direct(num_nation_tuples);
      GpuBloomFilter gpu_bf_nation = GpuBloomFilter::allocate(num_nation_tuples, 3);
      bf_storage[0] = fascal::runtime::BloomFilter::allocate(num_nation_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_nation.d_entries, 0xff, ht_nation.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_nation.d_bits, 0, ((size_t)gpu_bf_nation.size_bits + 63) / 64 * sizeof(uint64_t)));
  
      // AFP all-pipeline: pinned mapped memory for nation packed prefilter + tile summary (no doorbell).
      size_t nation_bitmap_words = ((size_t)num_nation_tuples + 31) / 32;
      size_t nation_num_tiles = ((size_t)num_nation_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_nation_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_nation_prefilter_pinned, nation_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_nation_prefilter_pinned, 0, nation_bitmap_words * sizeof(uint32_t));
      uint32_t *d_nation_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_nation_prefilter_mapped, h_nation_prefilter_pinned, 0));
      uint8_t *h_nation_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_nation_tile_summary, nation_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_nation_tile_summary, 0, nation_num_tiles);
      uint8_t *d_nation_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_nation_tile_summary, h_nation_tile_summary, 0));
  
  
      GpuHashTable ht_supplier;
      ht_supplier = GpuHashTable::allocate_direct(num_supplier_tuples);
      GpuBloomFilter gpu_bf_supplier = GpuBloomFilter::allocate(num_supplier_tuples, 3);
      bf_storage[1] = fascal::runtime::BloomFilter::allocate(num_supplier_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_supplier.d_entries, 0xff, ht_supplier.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_supplier.d_bits, 0, ((size_t)gpu_bf_supplier.size_bits + 63) / 64 * sizeof(uint64_t)));
  
      // AFP all-pipeline: pinned mapped memory for supplier packed prefilter + tile summary (no doorbell).
      size_t supplier_bitmap_words = ((size_t)num_supplier_tuples + 31) / 32;
      size_t supplier_num_tiles = ((size_t)num_supplier_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_supplier_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_supplier_prefilter_pinned, supplier_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_supplier_prefilter_pinned, 0, supplier_bitmap_words * sizeof(uint32_t));
      uint32_t *d_supplier_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_supplier_prefilter_mapped, h_supplier_prefilter_pinned, 0));
      uint8_t *h_supplier_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_supplier_tile_summary, supplier_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_supplier_tile_summary, 0, supplier_num_tiles);
      uint8_t *d_supplier_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_supplier_tile_summary, h_supplier_tile_summary, 0));
  
  
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
              tpch_q11_stage1_cpu_prefilter_partsupp(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 2);
              tpch_q11_stage1_kernel_partsupp<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_partsupp_ps_suppkey,
          d_partsupp_ps_supplycost,
          d_partsupp_ps_availqty,
          d_partsupp_ps_partkey_enc,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_supplier.d_bits, (bloom_off ? 0u : gpu_bf_supplier.size_bits),
          ht_supplier.d_entries, ht_supplier.ht_size,
          gpu_bf_nation.d_bits, (bloom_off ? 0u : gpu_bf_nation.size_bits),
          ht_nation.d_entries, ht_nation.ht_size,
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
              tpch_q11_stage1_cpu_prefilter_partsupp(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 2, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q11_stage1_kernel_partsupp<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_partsupp_ps_suppkey,
          d_partsupp_ps_supplycost,
          d_partsupp_ps_availqty,
          d_partsupp_ps_partkey_enc,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_supplier.d_bits, (bloom_off ? 0u : gpu_bf_supplier.size_bits),
          ht_supplier.d_entries, ht_supplier.ht_size,
          gpu_bf_nation.d_bits, (bloom_off ? 0u : gpu_bf_nation.size_bits),
          ht_nation.d_entries, ht_nation.ht_size,
              d_aggtable);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };
  
      cudaEventRecord(t0);
  
      // nation (25 rows < 256K): single launch, batch_offset=0
      tpch_q11_stage1_cpu_prefilter_nation(h_nation_n_nationkey, h_nation_n_name, num_nation_tuples, h_nation_prefilter_pinned, h_nation_tile_summary);
      tpch_q11_stage1_kernel_nation<BLOCK_THREADS, ITEMS_PER_THREAD>
          <<<(num_nation_tuples + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, fascal_streams[0]>>>(
          num_nation_tuples, /*batch_offset=*/0, d_nation_n_nationkey, d_nation_n_name,
          ht_nation.d_entries, ht_nation.ht_size,
          gpu_bf_nation.d_bits, gpu_bf_nation.size_bits, 3,
          d_nation_prefilter_mapped,
          d_nation_tile_summary);
      CUDA_CHECK(cudaStreamSynchronize(fascal_streams[0]));
      CUDA_CHECK(cudaGetLastError());
      if (bf_storage[0].bits) gpu_bf_nation.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));
      // supplier (~10K rows < 256K): single launch, batch_offset=0
      tpch_q11_stage1_cpu_prefilter_supplier(h_supplier_s_suppkey, h_supplier_s_nationkey, num_supplier_tuples, h_supplier_prefilter_pinned, h_supplier_tile_summary, (bf_storage[0].bits ? &bf_storage[0] : nullptr));
      tpch_q11_stage1_kernel_supplier<BLOCK_THREADS, ITEMS_PER_THREAD>
          <<<(num_supplier_tuples + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, fascal_streams[0]>>>(
          num_supplier_tuples, /*batch_offset=*/0, d_supplier_s_suppkey, d_supplier_s_nationkey,
          ht_supplier.d_entries, ht_supplier.ht_size,
          gpu_bf_supplier.d_bits, gpu_bf_supplier.size_bits, 3,
          d_supplier_prefilter_mapped,
          d_supplier_tile_summary,
          ht_nation.d_entries, ht_nation.ht_size);
      CUDA_CHECK(cudaStreamSynchronize(fascal_streams[0]));
      CUDA_CHECK(cudaGetLastError());
      if (bf_storage[1].bits) gpu_bf_supplier.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));
  
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
        int key0 = (int)(gk % 20000001ULL);
        gk /= 20000001;
        }
      }
      // Materializing aggregation results to __intermediate_agg_tpch_q11_*.bin
      std::vector<int> h_gb_decode0(20000001, std::numeric_limits<int>::min());
      for (int _row = 0; _row < num_tuples; ++_row) {
        int _enc = h_ps_partkey_enc[_row];
        if (_enc >= 0 && _enc < 20000001 && h_gb_decode0[(size_t)_enc] == std::numeric_limits<int>::min())
          h_gb_decode0[(size_t)_enc] = h_ps_partkey[_row];
      }
      int _h_count = 0;
      for (size_t k = 0; k < num_groups; k++) {
        bool _any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0));
        if (_any) {
          size_t gk = k;       int key0 = (int)(gk % 20000001ULL);
                gk /= 20000001;
          _h_count++;
        }
      }
      int *h_out_col0 = (int*)malloc(_h_count * sizeof(int));
      double *h_out_col1 = (double*)malloc(_h_count * sizeof(double));
      int _h_idx = 0;
      for (size_t k = 0; k < num_groups; k++) {
        bool _any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0));
        if (_any) {
          size_t gk = k;       int key0 = (int)(gk % 20000001ULL);
                gk /= 20000001;
          int _decoded_key0 = key0;
          if (key0 >= 0 && key0 < (int)h_gb_decode0.size() && h_gb_decode0[(size_t)key0] != std::numeric_limits<int>::min())
            _decoded_key0 = h_gb_decode0[(size_t)key0];
          h_out_col0[_h_idx] = _decoded_key0;
          h_out_col1[_h_idx] = h_agg_table[k * NUM_AGGREGATES + 0];
          _h_idx++;
        }
      }
      char path_mat0[512];
      snprintf(path_mat0, sizeof(path_mat0), "%s/__intermediate_agg_tpch_q11_ps_partkey.bin", data_dir);
      FILE *fp_mat0 = fopen(path_mat0, "wb");
      if (fp_mat0) {
        fwrite(h_out_col0, sizeof(int), _h_count, fp_mat0);
        fclose(fp_mat0);
        printf("Materialized %d rows to %s\n", _h_count, path_mat0);
      }
      char path_mat1[512];
      snprintf(path_mat1, sizeof(path_mat1), "%s/__intermediate_agg_tpch_q11_sum.bin", data_dir);
      FILE *fp_mat1 = fopen(path_mat1, "wb");
      if (fp_mat1) {
        fwrite(h_out_col1, sizeof(double), _h_count, fp_mat1);
        fclose(fp_mat1);
        printf("Materialized %d rows to %s\n", _h_count, path_mat1);
      }
      _mat_rows___intermediate_agg_tpch_q11 = _h_count;
      if (_mat___intermediate_agg_tpch_q11_ps_partkey) CUDA_CHECK(cudaFreeHost(_mat___intermediate_agg_tpch_q11_ps_partkey));
      if (_mat_rows___intermediate_agg_tpch_q11 > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat___intermediate_agg_tpch_q11_ps_partkey), (size_t)_mat_rows___intermediate_agg_tpch_q11 * sizeof(int), cudaHostAllocDefault));
        std::memcpy(_mat___intermediate_agg_tpch_q11_ps_partkey, h_out_col0, (size_t)_mat_rows___intermediate_agg_tpch_q11 * sizeof(int));
      } else {
        _mat___intermediate_agg_tpch_q11_ps_partkey = nullptr;
      }
      if (_mat___intermediate_agg_tpch_q11_sum) CUDA_CHECK(cudaFreeHost(_mat___intermediate_agg_tpch_q11_sum));
      if (_mat_rows___intermediate_agg_tpch_q11 > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat___intermediate_agg_tpch_q11_sum), (size_t)_mat_rows___intermediate_agg_tpch_q11 * sizeof(double), cudaHostAllocDefault));
        std::memcpy(_mat___intermediate_agg_tpch_q11_sum, h_out_col1, (size_t)_mat_rows___intermediate_agg_tpch_q11 * sizeof(double));
      } else {
        _mat___intermediate_agg_tpch_q11_sum = nullptr;
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
      odzc_mgr->unregister_column(mc_partsupp_ps_suppkey);
      odzc_mgr->unregister_column(mc_partsupp_ps_supplycost);
      odzc_mgr->unregister_column(mc_partsupp_ps_availqty);
      CUDA_CHECK(cudaFree(d_partsupp_ps_partkey_enc));
  
      odzc_mgr->unregister_column(mc_nation_n_nationkey);
      odzc_mgr->unregister_column(mc_nation_n_name);
      odzc_mgr->unregister_column(mc_supplier_s_suppkey);
      odzc_mgr->unregister_column(mc_supplier_s_nationkey);
      CUDA_CHECK(cudaFreeHost(h_nation_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_nation_prefilter_pinned));
      gpu_bf_nation.free_filter();
      ht_nation.free_table();
      bf_storage[0].free_filter();
      fascal_free_column(h_nation_n_nationkey);
      fascal_free_column(h_nation_n_name);
      CUDA_CHECK(cudaFreeHost(h_supplier_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_supplier_prefilter_pinned));
      gpu_bf_supplier.free_filter();
      ht_supplier.free_table();
      bf_storage[1].free_filter();
      fascal_free_column(h_supplier_s_suppkey);
      fascal_free_column(h_supplier_s_nationkey);
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_ps_suppkey);
      fascal_free_column(h_ps_supplycost);
      fascal_free_column(h_ps_availqty);
      fascal_free_column(h_ps_partkey_enc);
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

    // ---------------- STAGE 2 EXECUTION: tpch_q11_stage2___scalar_sq_1 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q11_stage2___scalar_sq_1
  // ============================================================
      printf("=== FaScalSQL: tpch_q11_stage2___scalar_sq_1 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/ps_suppkey.bin", data_dir);
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
  
      snprintf(path, sizeof(path), "%s/ps_suppkey.bin", data_dir);
      int *h_ps_suppkey = load_binary_column(path, num_tuples);
      if (!h_ps_suppkey) { fprintf(stderr, "Failed to load ps_suppkey\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/ps_supplycost.bin", data_dir);
      int *h_ps_supplycost = load_binary_column(path, num_tuples);
      if (!h_ps_supplycost) { fprintf(stderr, "Failed to load ps_supplycost\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/ps_availqty.bin", data_dir);
      int *h_ps_availqty = load_binary_column(path, num_tuples);
      if (!h_ps_availqty) { fprintf(stderr, "Failed to load ps_availqty\n"); return 1; }
  
      // ----- Load dimension: nation -----
      int num_nation_tuples = 0;
  
      snprintf(path, sizeof(path), "%s/n_nationkey.bin", data_dir);
      int *h_nation_n_nationkey = load_binary_column_auto(path, &num_nation_tuples);
  
      if (!h_nation_n_nationkey || num_nation_tuples == 0) {
          fprintf(stderr, "Failed to load table column n_nationkey\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/n_name.bin", data_dir);
      int *h_nation_n_name = load_binary_column_auto(path, &num_nation_tuples);
  
      if (!h_nation_n_name || num_nation_tuples == 0) {
          fprintf(stderr, "Failed to load table column n_name\n"); return 1;
      }
      printf("Loaded nation: %d tuples\n", num_nation_tuples);
  
      // ----- Load dimension: supplier -----
      int num_supplier_tuples = 0;
  
      snprintf(path, sizeof(path), "%s/s_suppkey.bin", data_dir);
      int *h_supplier_s_suppkey = load_binary_column_auto(path, &num_supplier_tuples);
  
      if (!h_supplier_s_suppkey || num_supplier_tuples == 0) {
          fprintf(stderr, "Failed to load table column s_suppkey\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/s_nationkey.bin", data_dir);
      int *h_supplier_s_nationkey = load_binary_column_auto(path, &num_supplier_tuples);
  
      if (!h_supplier_s_nationkey || num_supplier_tuples == 0) {
          fprintf(stderr, "Failed to load table column s_nationkey\n"); return 1;
      }
      printf("Loaded supplier: %d tuples\n", num_supplier_tuples);
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      cqo_in.predicate_selectivities.push_back(0.100000);
      // Dynamic BF selectivity: nation(n_name=GERMANY=7), supplier(all)
      { int nq = (int)(fascal_estimate_eq_sel(h_nation_n_name, num_nation_tuples, 7) * num_nation_tuples);
        cqo_in.join_bf_selectivities.push_back(fascal_estimate_join_bf_sel(nq, num_nation_tuples));
        cqo_in.join_bf_selectivities.push_back(fascal_estimate_join_bf_sel(num_supplier_tuples, num_supplier_tuples)); }
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
      int *h_data[] = {h_ps_suppkey, h_ps_supplycost, h_ps_availqty};
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
  
      fascal::runtime::ODZCManager::MappedColumn mc_partsupp_ps_suppkey;
      int *d_partsupp_ps_suppkey = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_ps_suppkey, num_tuples, "ps_suppkey", &mc_partsupp_ps_suppkey, &d_partsupp_ps_suppkey);
  
      fascal::runtime::ODZCManager::MappedColumn mc_partsupp_ps_supplycost;
      int *d_partsupp_ps_supplycost = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_ps_supplycost, num_tuples, "ps_supplycost", &mc_partsupp_ps_supplycost, &d_partsupp_ps_supplycost);
  
      fascal::runtime::ODZCManager::MappedColumn mc_partsupp_ps_availqty;
      int *d_partsupp_ps_availqty = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_ps_availqty, num_tuples, "ps_availqty", &mc_partsupp_ps_availqty, &d_partsupp_ps_availqty);
  
      fascal::runtime::ODZCManager::MappedColumn mc_nation_n_nationkey;
      mc_nation_n_nationkey = odzc_mgr->register_column(h_nation_n_nationkey, num_nation_tuples * sizeof(int));
      int *d_nation_n_nationkey = static_cast<int*>(mc_nation_n_nationkey.device_ptr);
      if (!d_nation_n_nationkey) {
          fprintf(stderr, "ODZC failed for nation.n_nationkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_nation_n_nationkey, num_nation_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_nation_n_nationkey, h_nation_n_nationkey, num_nation_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_nation_n_name;
      mc_nation_n_name = odzc_mgr->register_column(h_nation_n_name, num_nation_tuples * sizeof(int));
      int *d_nation_n_name = static_cast<int*>(mc_nation_n_name.device_ptr);
      if (!d_nation_n_name) {
          fprintf(stderr, "ODZC failed for nation.n_name, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_nation_n_name, num_nation_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_nation_n_name, h_nation_n_name, num_nation_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_suppkey;
      mc_supplier_s_suppkey = odzc_mgr->register_column(h_supplier_s_suppkey, num_supplier_tuples * sizeof(int));
      int *d_supplier_s_suppkey = static_cast<int*>(mc_supplier_s_suppkey.device_ptr);
      if (!d_supplier_s_suppkey) {
          fprintf(stderr, "ODZC failed for supplier.s_suppkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_supplier_s_suppkey, num_supplier_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_supplier_s_suppkey, h_supplier_s_suppkey, num_supplier_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_nationkey;
      mc_supplier_s_nationkey = odzc_mgr->register_column(h_supplier_s_nationkey, num_supplier_tuples * sizeof(int));
      int *d_supplier_s_nationkey = static_cast<int*>(mc_supplier_s_nationkey.device_ptr);
      if (!d_supplier_s_nationkey) {
          fprintf(stderr, "ODZC failed for supplier.s_nationkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_supplier_s_nationkey, num_supplier_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_supplier_s_nationkey, h_supplier_s_nationkey, num_supplier_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      double *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, (size_t)NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(1) * NUM_AGGREGATES * sizeof(double)));
  
      // === Hash Table Build ===
  
  
      GpuHashTable ht_nation;
      ht_nation = GpuHashTable::allocate_direct(num_nation_tuples);
      GpuBloomFilter gpu_bf_nation = GpuBloomFilter::allocate(num_nation_tuples, 3);
      bf_storage[0] = fascal::runtime::BloomFilter::allocate(num_nation_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_nation.d_entries, 0xff, ht_nation.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_nation.d_bits, 0, ((size_t)gpu_bf_nation.size_bits + 63) / 64 * sizeof(uint64_t)));
  
      // AFP all-pipeline: pinned mapped memory for nation packed prefilter + tile summary (no doorbell).
      size_t nation_bitmap_words = ((size_t)num_nation_tuples + 31) / 32;
      size_t nation_num_tiles = ((size_t)num_nation_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_nation_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_nation_prefilter_pinned, nation_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_nation_prefilter_pinned, 0, nation_bitmap_words * sizeof(uint32_t));
      uint32_t *d_nation_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_nation_prefilter_mapped, h_nation_prefilter_pinned, 0));
      uint8_t *h_nation_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_nation_tile_summary, nation_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_nation_tile_summary, 0, nation_num_tiles);
      uint8_t *d_nation_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_nation_tile_summary, h_nation_tile_summary, 0));
  
  
      GpuHashTable ht_supplier;
      ht_supplier = GpuHashTable::allocate_direct(num_supplier_tuples);
      GpuBloomFilter gpu_bf_supplier = GpuBloomFilter::allocate(num_supplier_tuples, 3);
      bf_storage[1] = fascal::runtime::BloomFilter::allocate(num_supplier_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_supplier.d_entries, 0xff, ht_supplier.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_supplier.d_bits, 0, ((size_t)gpu_bf_supplier.size_bits + 63) / 64 * sizeof(uint64_t)));
  
      // AFP all-pipeline: pinned mapped memory for supplier packed prefilter + tile summary (no doorbell).
      size_t supplier_bitmap_words = ((size_t)num_supplier_tuples + 31) / 32;
      size_t supplier_num_tiles = ((size_t)num_supplier_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_supplier_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_supplier_prefilter_pinned, supplier_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_supplier_prefilter_pinned, 0, supplier_bitmap_words * sizeof(uint32_t));
      uint32_t *d_supplier_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_supplier_prefilter_mapped, h_supplier_prefilter_pinned, 0));
      uint8_t *h_supplier_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_supplier_tile_summary, supplier_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_supplier_tile_summary, 0, supplier_num_tiles);
      uint8_t *d_supplier_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_supplier_tile_summary, h_supplier_tile_summary, 0));
  
  
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
              tpch_q11_stage2___scalar_sq_1_cpu_prefilter_partsupp(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 2);
              tpch_q11_stage2___scalar_sq_1_kernel_partsupp<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_partsupp_ps_suppkey,
          d_partsupp_ps_supplycost,
          d_partsupp_ps_availqty,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_supplier.d_bits, (bloom_off ? 0u : gpu_bf_supplier.size_bits),
          ht_supplier.d_entries, ht_supplier.ht_size,
          gpu_bf_nation.d_bits, (bloom_off ? 0u : gpu_bf_nation.size_bits),
          ht_nation.d_entries, ht_nation.ht_size,
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
              tpch_q11_stage2___scalar_sq_1_cpu_prefilter_partsupp(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 2, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q11_stage2___scalar_sq_1_kernel_partsupp<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_partsupp_ps_suppkey,
          d_partsupp_ps_supplycost,
          d_partsupp_ps_availqty,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_supplier.d_bits, (bloom_off ? 0u : gpu_bf_supplier.size_bits),
          ht_supplier.d_entries, ht_supplier.ht_size,
          gpu_bf_nation.d_bits, (bloom_off ? 0u : gpu_bf_nation.size_bits),
          ht_nation.d_entries, ht_nation.ht_size,
              d_aggtable);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };
  
      cudaEventRecord(t0);
  
      // nation (25 rows < 256K): single launch, batch_offset=0
      tpch_q11_stage2___scalar_sq_1_cpu_prefilter_nation(h_nation_n_nationkey, h_nation_n_name, num_nation_tuples, h_nation_prefilter_pinned, h_nation_tile_summary);
      tpch_q11_stage2___scalar_sq_1_kernel_nation<BLOCK_THREADS, ITEMS_PER_THREAD>
          <<<(num_nation_tuples + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, fascal_streams[0]>>>(
          num_nation_tuples, /*batch_offset=*/0, d_nation_n_nationkey, d_nation_n_name,
          ht_nation.d_entries, ht_nation.ht_size,
          gpu_bf_nation.d_bits, gpu_bf_nation.size_bits, 3,
          d_nation_prefilter_mapped,
          d_nation_tile_summary);
      CUDA_CHECK(cudaStreamSynchronize(fascal_streams[0]));
      CUDA_CHECK(cudaGetLastError());
      if (bf_storage[0].bits) gpu_bf_nation.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));
      // supplier (~10K rows < 256K): single launch, batch_offset=0
      tpch_q11_stage2___scalar_sq_1_cpu_prefilter_supplier(h_supplier_s_suppkey, h_supplier_s_nationkey, num_supplier_tuples, h_supplier_prefilter_pinned, h_supplier_tile_summary, (bf_storage[0].bits ? &bf_storage[0] : nullptr));
      tpch_q11_stage2___scalar_sq_1_kernel_supplier<BLOCK_THREADS, ITEMS_PER_THREAD>
          <<<(num_supplier_tuples + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, fascal_streams[0]>>>(
          num_supplier_tuples, /*batch_offset=*/0, d_supplier_s_suppkey, d_supplier_s_nationkey,
          ht_supplier.d_entries, ht_supplier.ht_size,
          gpu_bf_supplier.d_bits, gpu_bf_supplier.size_bits, 3,
          d_supplier_prefilter_mapped,
          d_supplier_tile_summary,
          ht_nation.d_entries, ht_nation.ht_size);
      CUDA_CHECK(cudaStreamSynchronize(fascal_streams[0]));
      CUDA_CHECK(cudaGetLastError());
      if (bf_storage[1].bits) gpu_bf_supplier.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));
  
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
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(1) * NUM_AGGREGATES * sizeof(double)));
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
      double *h_result = (double*)malloc((size_t)NUM_AGGREGATES * sizeof(double));
      CUDA_CHECK(cudaMemcpy(h_result, d_aggtable, (size_t)NUM_AGGREGATES * sizeof(double), cudaMemcpyDeviceToHost));
      int k = 0;
      double* h_agg_table = h_result;
      // Materializing scalar aggregation results to __scalar_sq_1_*.bin
      double h_out_val0 = h_result[0];
      char path_mat0[512];
      snprintf(path_mat0, sizeof(path_mat0), "%s/__scalar_sq_1_sum.bin", data_dir);
      FILE *fp_mat0 = fopen(path_mat0, "wb");
      if (fp_mat0) {
        fwrite(&h_out_val0, sizeof(double), 1, fp_mat0);
        fclose(fp_mat0);
        printf("Materialized 1 row to %s\n", path_mat0);
      }
      _mat_rows___scalar_sq_1 = 1;
      if (_mat___scalar_sq_1_sum) CUDA_CHECK(cudaFreeHost(_mat___scalar_sq_1_sum));
      if (_mat_rows___scalar_sq_1 > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat___scalar_sq_1_sum), (size_t)_mat_rows___scalar_sq_1 * sizeof(double), cudaHostAllocDefault));
        std::memcpy(_mat___scalar_sq_1_sum, &h_out_val0, (size_t)_mat_rows___scalar_sq_1 * sizeof(double));
      } else {
        _mat___scalar_sq_1_sum = nullptr;
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
      odzc_mgr->unregister_column(mc_partsupp_ps_suppkey);
      odzc_mgr->unregister_column(mc_partsupp_ps_supplycost);
      odzc_mgr->unregister_column(mc_partsupp_ps_availqty);
  
      odzc_mgr->unregister_column(mc_nation_n_nationkey);
      odzc_mgr->unregister_column(mc_nation_n_name);
      odzc_mgr->unregister_column(mc_supplier_s_suppkey);
      odzc_mgr->unregister_column(mc_supplier_s_nationkey);
      CUDA_CHECK(cudaFreeHost(h_nation_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_nation_prefilter_pinned));
      gpu_bf_nation.free_filter();
      ht_nation.free_table();
      bf_storage[0].free_filter();
      fascal_free_column(h_nation_n_nationkey);
      fascal_free_column(h_nation_n_name);
      CUDA_CHECK(cudaFreeHost(h_supplier_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_supplier_prefilter_pinned));
      gpu_bf_supplier.free_filter();
      ht_supplier.free_table();
      bf_storage[1].free_filter();
      fascal_free_column(h_supplier_s_suppkey);
      fascal_free_column(h_supplier_s_nationkey);
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_ps_suppkey);
      fascal_free_column(h_ps_supplycost);
      fascal_free_column(h_ps_availqty);
      for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) cudaStreamDestroy(fascal_streams[_s]);
      delete[] fascal_streams;
      cudaEventDestroy(t0);
      cudaEventDestroy(t1);
      cudaEventDestroy(t_start);
      cudaEventDestroy(t_load);
            cudaEventDestroy(t_query);
      cudaEventDestroy(t_result);
  
    }

    // ---------------- STAGE 3 EXECUTION: tpch_q11_stage2 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q11_stage2
  // ============================================================
      printf("=== FaScalSQL: tpch_q11_stage2 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          if (_mat_rows___intermediate_agg_tpch_q11 > 0 && _mat___intermediate_agg_tpch_q11_ps_partkey != nullptr) {
              num_tuples = _mat_rows___intermediate_agg_tpch_q11;
          } else {
              char _auto_path[512];
              snprintf(_auto_path, sizeof(_auto_path), "%s/__intermediate_agg_tpch_q11_ps_partkey.bin", data_dir);
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
  
      int *h_ps_partkey = nullptr;
      if (_mat_rows___intermediate_agg_tpch_q11 > 0 && _mat___intermediate_agg_tpch_q11_ps_partkey != nullptr) {
        h_ps_partkey = _mat___intermediate_agg_tpch_q11_ps_partkey;
        num_tuples = _mat_rows___intermediate_agg_tpch_q11;
      } else {
        snprintf(path, sizeof(path), "%s/__intermediate_agg_tpch_q11_ps_partkey.bin", data_dir);
        h_ps_partkey = load_binary_column(path, num_tuples);
      }
      if (!h_ps_partkey) { fprintf(stderr, "Failed to load ps_partkey\n"); return 1; }
  
      double *h_sum = nullptr;
      if (_mat_rows___intermediate_agg_tpch_q11 > 0 && _mat___intermediate_agg_tpch_q11_sum != nullptr) {
        h_sum = _mat___intermediate_agg_tpch_q11_sum;
        num_tuples = _mat_rows___intermediate_agg_tpch_q11;
      } else {
        snprintf(path, sizeof(path), "%s/__intermediate_agg_tpch_q11_sum.bin", data_dir);
        h_sum = (double*)load_binary_column_u64(path, num_tuples);
      }
      if (!h_sum) { fprintf(stderr, "Failed to load sum\n"); return 1; }
  
      // ----- Load scalar auxiliary materialized tables -----
  
      int num___scalar_sq_1_tuples = 0;
      double *h___scalar_sq_1_sum = nullptr;
      if (_mat_rows___scalar_sq_1 > 0 && _mat___scalar_sq_1_sum != nullptr) {
        num___scalar_sq_1_tuples = _mat_rows___scalar_sq_1;
        h___scalar_sq_1_sum = _mat___scalar_sq_1_sum;
      } else {
        snprintf(path, sizeof(path), "%s/__scalar_sq_1_sum.bin", data_dir);
        h___scalar_sq_1_sum = (double*)load_binary_column_auto_u64(path, &num___scalar_sq_1_tuples);
      }
      if (!h___scalar_sq_1_sum || num___scalar_sq_1_tuples <= 0) { fprintf(stderr, "Failed to load __scalar_sq_1.sum\n"); return 1; }
  
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
      int *h_data[] = {h_ps_partkey, (int*)h_sum};
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
  
      fascal::runtime::ODZCManager::MappedColumn mc___scalar_sq_1_sum;
      double *d___scalar_sq_1_sum = nullptr;
      {
        mc___scalar_sq_1_sum = odzc_mgr->register_column(h___scalar_sq_1_sum, (size_t)num___scalar_sq_1_tuples * sizeof(double));
        d___scalar_sq_1_sum = static_cast<double*>(mc___scalar_sq_1_sum.device_ptr);
        if (!d___scalar_sq_1_sum) {
          fprintf(stderr, "ODZC failed for __scalar_sq_1.sum, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d___scalar_sq_1_sum, (size_t)num___scalar_sq_1_tuples * sizeof(double)));
          CUDA_CHECK(cudaMemcpy(d___scalar_sq_1_sum, h___scalar_sq_1_sum, (size_t)num___scalar_sq_1_tuples * sizeof(double), cudaMemcpyHostToDevice));
        }
      }
  
      fascal::runtime::ODZCManager::MappedColumn mc___intermediate_agg_tpch_q11_ps_partkey;
      int *d___intermediate_agg_tpch_q11_ps_partkey = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_ps_partkey, num_tuples, "ps_partkey", &mc___intermediate_agg_tpch_q11_ps_partkey, &d___intermediate_agg_tpch_q11_ps_partkey);
  
      fascal::runtime::ODZCManager::MappedColumn mc___intermediate_agg_tpch_q11_sum;
      double *d___intermediate_agg_tpch_q11_sum = nullptr;
      {
        mc___intermediate_agg_tpch_q11_sum = odzc_mgr->register_column(h_sum, (size_t)num_tuples * sizeof(double));
        d___intermediate_agg_tpch_q11_sum = static_cast<double*>(mc___intermediate_agg_tpch_q11_sum.device_ptr);
        if (!d___intermediate_agg_tpch_q11_sum) {
          fprintf(stderr, "ODZC failed for __intermediate_agg_tpch_q11.sum, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d___intermediate_agg_tpch_q11_sum, (size_t)num_tuples * sizeof(double)));
          CUDA_CHECK(cudaMemcpy(d___intermediate_agg_tpch_q11_sum, h_sum, (size_t)num_tuples * sizeof(double), cudaMemcpyHostToDevice));
        }
      }
  
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      int *d_out_count = nullptr;
      CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
      CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));
      int *d_out_col0 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_out_col0, (size_t)num_tuples * sizeof(int)));
      double *d_out_col1 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_out_col1, (size_t)num_tuples * sizeof(double)));
  
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
              tpch_q11_stage2_cpu_prefilter___intermediate_agg_tpch_q11(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0);
              tpch_q11_stage2_kernel___intermediate_agg_tpch_q11<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d___scalar_sq_1_sum,
          d___intermediate_agg_tpch_q11_ps_partkey,
          d___intermediate_agg_tpch_q11_sum,
                  d_seed_bitmap,
                  d_tile_summary,
              d_out_count,
          d_out_col0,
          d_out_col1);
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
              tpch_q11_stage2_cpu_prefilter___intermediate_agg_tpch_q11(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q11_stage2_kernel___intermediate_agg_tpch_q11<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d___scalar_sq_1_sum,
          d___intermediate_agg_tpch_q11_ps_partkey,
          d___intermediate_agg_tpch_q11_sum,
                  d_seed_bitmap,
                  d_tile_summary,
              d_out_count,
          d_out_col0,
          d_out_col1);
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
      double *h_out_col1 = (double*)malloc((size_t)num_tuples * sizeof(double));
      CUDA_CHECK(cudaMemcpy(h_out_col1, d_out_col1, (size_t)num_tuples * sizeof(double), cudaMemcpyDeviceToHost));
      std::vector<int> _row_order((size_t)h_out_count);
      for (int i = 0; i < h_out_count; i++) _row_order[(size_t)i] = i;
      std::sort(_row_order.begin(), _row_order.end(), [&](int _a, int _b) {
        if (h_out_col1[_a] != h_out_col1[_b]) return h_out_col1[_a] > h_out_col1[_b];
        return false;
      });
      int _mat_count = h_out_count;
      for (int i = 0; i < _mat_count; i++) {
        int _row_idx = _row_order[(size_t)i];
        printf("ROW: %d", h_out_col0[_row_idx]);
        printf("|%.12g", h_out_col1[_row_idx]);
        printf("\n");
      }
      free(h_out_col0);
      CUDA_CHECK(cudaFree(d_out_col0));
      free(h_out_col1);
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
      odzc_mgr->unregister_column(mc___scalar_sq_1_sum);
      odzc_mgr->unregister_column(mc___intermediate_agg_tpch_q11_ps_partkey);
      odzc_mgr->unregister_column(mc___intermediate_agg_tpch_q11_sum);
  
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      if (h_ps_partkey && h_ps_partkey != _mat___intermediate_agg_tpch_q11_ps_partkey) fascal_free_column(h_ps_partkey);
      if (h_sum && h_sum != _mat___intermediate_agg_tpch_q11_sum) fascal_free_column(h_sum);
      if (h___scalar_sq_1_sum && h___scalar_sq_1_sum != _mat___scalar_sq_1_sum) fascal_free_column(h___scalar_sq_1_sum);
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
    if (_mat___intermediate_agg_tpch_q11_ps_partkey) CUDA_CHECK(cudaFreeHost(_mat___intermediate_agg_tpch_q11_ps_partkey));
    if (_mat___intermediate_agg_tpch_q11_sum) CUDA_CHECK(cudaFreeHost(_mat___intermediate_agg_tpch_q11_sum));
    if (_mat___scalar_sq_1_sum) CUDA_CHECK(cudaFreeHost(_mat___scalar_sq_1_sum));
}
