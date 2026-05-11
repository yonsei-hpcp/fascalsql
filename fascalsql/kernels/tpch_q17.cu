// TPC-H Q17 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Source: fascalsql/kernels/tpch_q17.cu

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
#define NUM_AGGREGATES 2

__device__ __constant__ unsigned char d_pred_flags[MAX_PREDICATES];
static unsigned char h_pred_flags[MAX_PREDICATES] = {0};

static inline void normalize_or_group_pred_flags(unsigned char *pred_flags) {
  (void)pred_flags;
}
// ================= STAGE 1: tpch_q17___scalar_sq_1 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q17___scalar_sq_1_kernel_lineitem(
    int num_tuples,
    int batch_offset,
    float *d_lineitem_l_quantity,
    int *d_lineitem_l_partkey_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  float loc_lineitem_l_quantity[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_partkey_enc[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // ODZC lazy load (fact GROUP BY, before aggregation): l_partkey_enc
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_partkey_enc + tile_offset, loc_lineitem_l_partkey_enc,
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
        int gk = loc_lineitem_l_partkey_enc[ITEM];
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], (((double)loc_lineitem_l_quantity[ITEM])));
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 1], 1LL);
      }
  }
  } while(0);
}

static void tpch_q17___scalar_sq_1_cpu_predicate_lineitem(int *h_data[], int offset, int cnt, uint32_t *bitmap) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
}

static void tpch_q17___scalar_sq_1_cpu_prefilter_lineitem(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
  fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
    [&](int off, int cnt) { tpch_q17___scalar_sq_1_cpu_predicate_lineitem(h_data, off, cnt, bitmap); },
    batch_offset, total_tuples);
}


// ================= STAGE 2: tpch_q17 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q17_kernel_lineitem(
    int num_tuples,
    int batch_offset,
    double *d___scalar_sq_1_avg,
    int *d_lineitem_l_partkey,
    float *d_lineitem_l_quantity,
    float *d_lineitem_l_extendedprice,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_part,
    uint32_t bloom_size_bits_part,
    HtEntry *ht_part,
    uint32_t ht_size_part,
    const uint64_t *__restrict__ d_bloom___scalar_sq_1,
    uint32_t bloom_size_bits___scalar_sq_1,
    HtEntry *ht___scalar_sq_1,
    uint32_t ht_size___scalar_sq_1,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  double loc___scalar_sq_1_avg[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_partkey[ITEMS_PER_THREAD_T];
  float loc_lineitem_l_quantity[ITEMS_PER_THREAD_T];
  float loc_lineitem_l_extendedprice[ITEMS_PER_THREAD_T];
  int loc_part_p_partkey[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // Phase 1: Semi-joins (Bloom filter); each FK loaded immediately before its probe
  // ODZC lazy load (join FK, before part probe): l_partkey
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_partkey + tile_offset, loc_lineitem_l_partkey,
      selection_flags, num_tile_items);

  BlockBloomProbe<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_lineitem_l_partkey, selection_flags, num_tile_items,
      bloom_size_bits_part, d_bloom_part, 3);

  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  // Phase 2: All hash table joins
  // Join 0 (inner): l_partkey -> part (value-as-payload: p_partkey stored in HT)
  BlockJoinProbePayloadDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_lineitem_l_partkey, selection_flags, num_tile_items,
      ht_part, ht_size_part, 1, loc_part_p_partkey);

  // Join 1 (inner): p_partkey -> __scalar_sq_1
  int loc___scalar_sq_1_rid[ITEMS_PER_THREAD_T];
  BlockJoinProbePayload<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_part_p_partkey, selection_flags, num_tile_items,
      ht___scalar_sq_1, ht_size___scalar_sq_1, loc___scalar_sq_1_rid);
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ITEM++) {
    if (selection_flags[ITEM]) {
      int rid = loc___scalar_sq_1_rid[ITEM];
      loc___scalar_sq_1_avg[ITEM] = d___scalar_sq_1_avg[rid];
    }
  }

  // ODZC lazy load (aggregation only): l_quantity
  BlockLoadSelect<float, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_quantity + tile_offset, loc_lineitem_l_quantity,
      selection_flags, num_tile_items);

  // ODZC lazy load (aggregation only): l_extendedprice
  BlockLoadSelect<float, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_extendedprice + tile_offset, loc_lineitem_l_extendedprice,
      selection_flags, num_tile_items);

  // Post-join filter 0
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if (selection_flags[ITEM]) {
      if (!((loc_lineitem_l_quantity[ITEM] < (0.2 * loc___scalar_sq_1_avg[ITEM]))))
        selection_flags[ITEM] = 0;
    }
  }

  } while(0);
  // Scalar aggregation with block-level reduction (one atomic per block)
  double _acc0 = 0;
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM])
        _acc0 += (((double)loc_lineitem_l_extendedprice[ITEM]));
  }
  _acc0 = blockReduceSumDouble<BLOCK_THREADS_T>(_acc0);
  if (threadIdx.x == 0) atomicAdd(&aggtable[0], (double)_acc0);
}

static void tpch_q17_cpu_predicate_lineitem(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bloom_filters[], int num_bloom_filters) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  if (bloom_filters && 0 < num_bloom_filters && bloom_filters[0])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, h_data[0], offset, cnt, bloom_filters[0]);
}

static void tpch_q17_cpu_prefilter_lineitem(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
  fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
    [&](int off, int cnt) { tpch_q17_cpu_predicate_lineitem(h_data, off, cnt, bitmap, bloom_filters, num_bloom_filters); },
    batch_offset, total_tuples);
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q17_kernel_part(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_part_p_partkey,
    const int *__restrict__ d_part_p_brand,
    const int *__restrict__ d_part_p_container,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht___scalar_sq_1, uint32_t ht_size_f___scalar_sq_1)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_part_p_partkey[ITEMS_PER_THREAD_T];
  int loc_part_p_brand[ITEMS_PER_THREAD_T];
  int loc_part_p_container[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_partkey + tile_offset), loc_part_p_partkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_brand + tile_offset), loc_part_p_brand, selection_flags, num_tile_items);

  BlockPredAndEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_part_p_brand, 21, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_container + tile_offset), loc_part_p_container, selection_flags, num_tile_items);

  BlockPredAndEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_part_p_container, 10, selection_flags, num_tile_items);

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe(loc_part_p_partkey[ITEM], d_filter_ht___scalar_sq_1, ht_size_f___scalar_sq_1, nullptr)) continue;
    int payload = loc_part_p_partkey[ITEM];  // value-as-payload: store p_partkey instead of row_id
    gpu_ht_insert_one_direct(loc_part_p_partkey[ITEM], payload, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_part_p_partkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'part' (morsel-batched) ----
// Fills prefilter bitmap for [batch_offset, batch_offset+batch_count) then GPU kernel
// launches one morsel at a time on a CUDA stream (no doorbell).
static void tpch_q17_cpu_prefilter_part(
    int *h_part_p_partkey,
    int *h_part_p_brand,
    int *h_part_p_container,
    int num_tuples,
    int batch_offset,
    int batch_count,
    uint32_t *bitmap,
    uint8_t *tile_summary,
    fascal::runtime::BloomFilter *upstream_bf___scalar_sq_1) {
    fascal_cpu_prefilter_run_batch(num_tuples, batch_offset, batch_count, bitmap, tile_summary,
        [&](int offset, int cnt) {
            // 0. Zero bitmap words for this morsel.
            uint32_t word_start = (uint32_t)offset >> 5;
            uint32_t word_end   = ((uint32_t)(offset + cnt) + 31) >> 5;
            memset(&bitmap[word_start], 0, (word_end - word_start) * sizeof(uint32_t));
            // 1. Fill packed bitmap with scalar predicate result for this morsel.
            for (int i = 0; i < cnt; ++i) {
                int idx = offset + i;
                if ((h_part_p_brand[idx] == 21) && (h_part_p_container[idx] == 10)) {
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
                }
            }
            // 2. Apply upstream BF probes (AND-mask with FK column probes).
                if (upstream_bf___scalar_sq_1 && upstream_bf___scalar_sq_1->bits) {
                    upstream_bf___scalar_sq_1->probe_and_mask_packed(h_part_p_partkey + offset, bitmap, offset, cnt);
                }
        });
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q17_kernel___scalar_sq_1(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d___scalar_sq_1_l_partkey,
    const double *__restrict__ d___scalar_sq_1_avg,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc___scalar_sq_1_l_partkey[ITEMS_PER_THREAD_T];
  double loc___scalar_sq_1_avg[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d___scalar_sq_1_l_partkey + tile_offset), loc___scalar_sq_1_l_partkey, selection_flags, num_tile_items);

  BlockLoadSelect<double, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (double*)(d___scalar_sq_1_avg + tile_offset), loc___scalar_sq_1_avg, selection_flags, num_tile_items);

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    unsigned long long key = static_cast<unsigned int>(loc___scalar_sq_1_l_partkey[ITEM]);
    int payload = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one(key, payload, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc___scalar_sq_1_l_partkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline '__scalar_sq_1' (morsel-batched) ----
// Fills prefilter bitmap for [batch_offset, batch_offset+batch_count) then GPU kernel
// launches one morsel at a time on a CUDA stream (no doorbell).
static void tpch_q17_cpu_prefilter___scalar_sq_1(
    int *h___scalar_sq_1_l_partkey,
    double *h___scalar_sq_1_avg,
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
    int _mat_rows___scalar_sq_1 = 0;
    int *_mat___scalar_sq_1_l_partkey = nullptr;
    double *_mat___scalar_sq_1_avg = nullptr;
    auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));

    // ---------------- STAGE 1 EXECUTION: tpch_q17___scalar_sq_1 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q17___scalar_sq_1
  // ============================================================
      printf("=== FaScalSQL: tpch_q17___scalar_sq_1 ===\n");
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
          size_t arena_cols = 6;
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
  
      snprintf(path, sizeof(path), "%s/l_partkey.bin", data_dir);
      int *h_l_partkey = load_binary_column(path, num_tuples);
      if (!h_l_partkey) { fprintf(stderr, "Failed to load l_partkey\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_partkey_enc.bin", data_dir);
      int *h_l_partkey_enc = load_binary_column(path, num_tuples);
      if (!h_l_partkey_enc) { fprintf(stderr, "Failed to load l_partkey_enc\n"); return 1; }
  
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
      {
        mc_lineitem_l_quantity = odzc_mgr->register_column(h_l_quantity, (size_t)num_tuples * sizeof(float));
        d_lineitem_l_quantity = static_cast<float*>(mc_lineitem_l_quantity.device_ptr);
        if (!d_lineitem_l_quantity) {
          fprintf(stderr, "ODZC failed for lineitem.l_quantity, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_lineitem_l_quantity, (size_t)num_tuples * sizeof(float)));
          CUDA_CHECK(cudaMemcpy(d_lineitem_l_quantity, h_l_quantity, (size_t)num_tuples * sizeof(float), cudaMemcpyHostToDevice));
        }
      }
  
      int *d_lineitem_l_partkey_enc = nullptr;
      CUDA_CHECK(cudaMalloc(&d_lineitem_l_partkey_enc, num_tuples * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(d_lineitem_l_partkey_enc, h_l_partkey_enc, num_tuples * sizeof(int), cudaMemcpyHostToDevice));
  
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      size_t num_groups = 1ULL;
      num_groups *= 20000001ULL;
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
              tpch_q17___scalar_sq_1_cpu_prefilter_lineitem(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0);
              tpch_q17___scalar_sq_1_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_lineitem_l_quantity,
          d_lineitem_l_partkey_enc,
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
              tpch_q17___scalar_sq_1_cpu_prefilter_lineitem(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q17___scalar_sq_1_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_lineitem_l_quantity,
          d_lineitem_l_partkey_enc,
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
        int any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0) || (h_agg_table[k * NUM_AGGREGATES + 1] != 0)) ? 1 : 0;
        if (any) {
          size_t gk = k;
        int key0 = (int)(gk % 20000001ULL);
        gk /= 20000001;
        }
      }
      // Materializing aggregation results to __scalar_sq_1_*.bin
      std::vector<int> h_gb_decode0(20000001, std::numeric_limits<int>::min());
      for (int _row = 0; _row < num_tuples; ++_row) {
        int _enc = h_l_partkey_enc[_row];
        if (_enc >= 0 && _enc < 20000001 && h_gb_decode0[(size_t)_enc] == std::numeric_limits<int>::min())
          h_gb_decode0[(size_t)_enc] = h_l_partkey[_row];
      }
      int _h_count = 0;
      for (size_t k = 0; k < num_groups; k++) {
        bool _any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0) || (h_agg_table[k * NUM_AGGREGATES + 1] != 0));
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
        bool _any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0) || (h_agg_table[k * NUM_AGGREGATES + 1] != 0));
        if (_any) {
          size_t gk = k;       int key0 = (int)(gk % 20000001ULL);
                gk /= 20000001;
          int _decoded_key0 = key0;
          if (key0 >= 0 && key0 < (int)h_gb_decode0.size() && h_gb_decode0[(size_t)key0] != std::numeric_limits<int>::min())
            _decoded_key0 = h_gb_decode0[(size_t)key0];
          h_out_col0[_h_idx] = _decoded_key0;
          h_out_col1[_h_idx] = ((h_agg_table[k * NUM_AGGREGATES + 1]) != 0 ? ((double)(h_agg_table[k * NUM_AGGREGATES + 0]) / (double)(h_agg_table[k * NUM_AGGREGATES + 1])) : 0.0);
          _h_idx++;
        }
      }
      char path_mat0[512];
      snprintf(path_mat0, sizeof(path_mat0), "%s/__scalar_sq_1_l_partkey.bin", data_dir);
      FILE *fp_mat0 = fopen(path_mat0, "wb");
      if (fp_mat0) {
        fwrite(h_out_col0, sizeof(int), _h_count, fp_mat0);
        fclose(fp_mat0);
        printf("Materialized %d rows to %s\n", _h_count, path_mat0);
      }
      char path_mat1[512];
      snprintf(path_mat1, sizeof(path_mat1), "%s/__scalar_sq_1_avg.bin", data_dir);
      FILE *fp_mat1 = fopen(path_mat1, "wb");
      if (fp_mat1) {
        fwrite(h_out_col1, sizeof(double), _h_count, fp_mat1);
        fclose(fp_mat1);
        printf("Materialized %d rows to %s\n", _h_count, path_mat1);
      }
      _mat_rows___scalar_sq_1 = _h_count;
      if (_mat___scalar_sq_1_l_partkey) CUDA_CHECK(cudaFreeHost(_mat___scalar_sq_1_l_partkey));
      if (_mat_rows___scalar_sq_1 > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat___scalar_sq_1_l_partkey), (size_t)_mat_rows___scalar_sq_1 * sizeof(int), cudaHostAllocDefault));
        std::memcpy(_mat___scalar_sq_1_l_partkey, h_out_col0, (size_t)_mat_rows___scalar_sq_1 * sizeof(int));
      } else {
        _mat___scalar_sq_1_l_partkey = nullptr;
      }
      if (_mat___scalar_sq_1_avg) CUDA_CHECK(cudaFreeHost(_mat___scalar_sq_1_avg));
      if (_mat_rows___scalar_sq_1 > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat___scalar_sq_1_avg), (size_t)_mat_rows___scalar_sq_1 * sizeof(double), cudaHostAllocDefault));
        std::memcpy(_mat___scalar_sq_1_avg, h_out_col1, (size_t)_mat_rows___scalar_sq_1 * sizeof(double));
      } else {
        _mat___scalar_sq_1_avg = nullptr;
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
      CUDA_CHECK(cudaFree(d_lineitem_l_partkey_enc));
  
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_l_quantity);
      fascal_free_column(h_l_partkey_enc);
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

    // ---------------- STAGE 2 EXECUTION: tpch_q17 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q17
  // ============================================================
      printf("=== FaScalSQL: tpch_q17 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/l_partkey.bin", data_dir);
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
  
      snprintf(path, sizeof(path), "%s/l_partkey.bin", data_dir);
      int *h_l_partkey = load_binary_column(path, num_tuples);
      if (!h_l_partkey) { fprintf(stderr, "Failed to load l_partkey\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_quantity.bin", data_dir);
      float *h_l_quantity = (float*)load_binary_column(path, num_tuples);
      if (!h_l_quantity) { fprintf(stderr, "Failed to load l_quantity\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_extendedprice.bin", data_dir);
      float *h_l_extendedprice = (float*)load_binary_column(path, num_tuples);
      if (!h_l_extendedprice) { fprintf(stderr, "Failed to load l_extendedprice\n"); return 1; }
  
      // ----- Load dimension: __scalar_sq_1 -----
      int num___scalar_sq_1_tuples = 0;
  
      int *h___scalar_sq_1_l_partkey = nullptr;
      if (_mat_rows___scalar_sq_1 > 0 && _mat___scalar_sq_1_l_partkey != nullptr) {
          num___scalar_sq_1_tuples = _mat_rows___scalar_sq_1;
          h___scalar_sq_1_l_partkey = _mat___scalar_sq_1_l_partkey;
      } else {
  
      snprintf(path, sizeof(path), "%s/__scalar_sq_1_l_partkey.bin", data_dir);
      h___scalar_sq_1_l_partkey = load_binary_column_auto(path, &num___scalar_sq_1_tuples);
      }
  
      if (!h___scalar_sq_1_l_partkey || num___scalar_sq_1_tuples == 0) {
          fprintf(stderr, "Failed to load table column l_partkey\n"); return 1;
      }
  
      double *h___scalar_sq_1_avg = nullptr;
      if (_mat_rows___scalar_sq_1 > 0 && _mat___scalar_sq_1_avg != nullptr) {
          num___scalar_sq_1_tuples = _mat_rows___scalar_sq_1;
          h___scalar_sq_1_avg = _mat___scalar_sq_1_avg;
      } else {
  
      snprintf(path, sizeof(path), "%s/__scalar_sq_1_avg.bin", data_dir);
      h___scalar_sq_1_avg = ((double*)load_binary_column_auto_u64(path, &num___scalar_sq_1_tuples));
      }
  
      if (!h___scalar_sq_1_avg || num___scalar_sq_1_tuples == 0) {
          fprintf(stderr, "Failed to load table column avg\n"); return 1;
      }
      printf("Loaded __scalar_sq_1: %d tuples\n", num___scalar_sq_1_tuples);
  
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
  
      snprintf(path, sizeof(path), "%s/p_container.bin", data_dir);
      int *h_part_p_container = load_binary_column_auto(path, &num_part_tuples);
  
      if (!h_part_p_container || num_part_tuples == 0) {
          fprintf(stderr, "Failed to load table column p_container\n"); return 1;
      }
      printf("Loaded part: %d tuples\n", num_part_tuples);
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      cqo_in.predicate_selectivities.push_back(0.100000);
      cqo_in.predicate_selectivities.push_back(0.100000);
      cqo_in.join_bf_selectivities.push_back(0.05);
      cqo_in.join_bf_selectivities.push_back(0.05);
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
  
      int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr) || cqo_out.bloom_off;
  
      // === AFP Manager & Predicate Filters ===
      int *h_data[] = {h_l_partkey, (int*)h_l_quantity, (int*)h_l_extendedprice};
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
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_partkey;
      int *d_lineitem_l_partkey = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_partkey, num_tuples, "l_partkey", &mc_lineitem_l_partkey, &d_lineitem_l_partkey);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_quantity;
      float *d_lineitem_l_quantity = nullptr;
      {
        mc_lineitem_l_quantity = odzc_mgr->register_column(h_l_quantity, (size_t)num_tuples * sizeof(float));
        d_lineitem_l_quantity = static_cast<float*>(mc_lineitem_l_quantity.device_ptr);
        if (!d_lineitem_l_quantity) {
          fprintf(stderr, "ODZC failed for lineitem.l_quantity, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_lineitem_l_quantity, (size_t)num_tuples * sizeof(float)));
          CUDA_CHECK(cudaMemcpy(d_lineitem_l_quantity, h_l_quantity, (size_t)num_tuples * sizeof(float), cudaMemcpyHostToDevice));
        }
      }
  
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
  
      fascal::runtime::ODZCManager::MappedColumn mc___scalar_sq_1_l_partkey;
      mc___scalar_sq_1_l_partkey = odzc_mgr->register_column(h___scalar_sq_1_l_partkey, num___scalar_sq_1_tuples * sizeof(int));
      int *d___scalar_sq_1_l_partkey = static_cast<int*>(mc___scalar_sq_1_l_partkey.device_ptr);
      if (!d___scalar_sq_1_l_partkey) {
          fprintf(stderr, "ODZC failed for __scalar_sq_1.l_partkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d___scalar_sq_1_l_partkey, num___scalar_sq_1_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d___scalar_sq_1_l_partkey, h___scalar_sq_1_l_partkey, num___scalar_sq_1_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc___scalar_sq_1_avg;
      mc___scalar_sq_1_avg = odzc_mgr->register_column(h___scalar_sq_1_avg, num___scalar_sq_1_tuples * sizeof(double));
      double *d___scalar_sq_1_avg = static_cast<double*>(mc___scalar_sq_1_avg.device_ptr);
      if (!d___scalar_sq_1_avg) {
          fprintf(stderr, "ODZC failed for __scalar_sq_1.avg, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d___scalar_sq_1_avg, num___scalar_sq_1_tuples * sizeof(double)));
          CUDA_CHECK(cudaMemcpy(d___scalar_sq_1_avg, h___scalar_sq_1_avg, num___scalar_sq_1_tuples * sizeof(double), cudaMemcpyHostToDevice));
      }
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
      fascal::runtime::ODZCManager::MappedColumn mc_part_p_container;
      mc_part_p_container = odzc_mgr->register_column(h_part_p_container, num_part_tuples * sizeof(int));
      int *d_part_p_container = static_cast<int*>(mc_part_p_container.device_ptr);
      if (!d_part_p_container) {
          fprintf(stderr, "ODZC failed for part.p_container, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_part_p_container, num_part_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_part_p_container, h_part_p_container, num_part_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      double *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, (size_t)NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(1) * NUM_AGGREGATES * sizeof(double)));
  
      // === Hash Table Build ===
  
  
      GpuHashTable ht___scalar_sq_1;
      ht___scalar_sq_1 = GpuHashTable::allocate(num___scalar_sq_1_tuples);
      GpuBloomFilter gpu_bf___scalar_sq_1 = GpuBloomFilter::allocate(num___scalar_sq_1_tuples, 3);
      bf_storage[0] = fascal::runtime::BloomFilter::allocate(num___scalar_sq_1_tuples, 3);
      CUDA_CHECK(cudaMemset(ht___scalar_sq_1.d_entries, 0xff, ht___scalar_sq_1.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf___scalar_sq_1.d_bits, 0, ((size_t)gpu_bf___scalar_sq_1.size_bits + 63) / 64 * sizeof(uint64_t)));
  
      // AFP all-pipeline: pinned mapped memory for __scalar_sq_1 packed prefilter + tile summary (no doorbell).
      size_t __scalar_sq_1_bitmap_words = ((size_t)num___scalar_sq_1_tuples + 31) / 32;
      size_t __scalar_sq_1_num_tiles = ((size_t)num___scalar_sq_1_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h___scalar_sq_1_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h___scalar_sq_1_prefilter_pinned, __scalar_sq_1_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h___scalar_sq_1_prefilter_pinned, 0, __scalar_sq_1_bitmap_words * sizeof(uint32_t));
      uint32_t *d___scalar_sq_1_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d___scalar_sq_1_prefilter_mapped, h___scalar_sq_1_prefilter_pinned, 0));
      uint8_t *h___scalar_sq_1_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h___scalar_sq_1_tile_summary, __scalar_sq_1_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h___scalar_sq_1_tile_summary, 0, __scalar_sq_1_num_tiles);
      uint8_t *d___scalar_sq_1_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d___scalar_sq_1_tile_summary, h___scalar_sq_1_tile_summary, 0));
  
  
      GpuHashTable ht_part;
      ht_part = GpuHashTable::allocate_direct(num_part_tuples);
      GpuBloomFilter gpu_bf_part = GpuBloomFilter::allocate(num_part_tuples, 3);
      bf_storage[1] = fascal::runtime::BloomFilter::allocate(num_part_tuples, 3);
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
              tpch_q17_cpu_prefilter_lineitem(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 2);
              tpch_q17_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d___scalar_sq_1_avg,
          d_lineitem_l_partkey,
          d_lineitem_l_quantity,
          d_lineitem_l_extendedprice,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_part.d_bits, (bloom_off ? 0u : gpu_bf_part.size_bits),
          ht_part.d_entries, ht_part.ht_size,
          gpu_bf___scalar_sq_1.d_bits, (bloom_off ? 0u : gpu_bf___scalar_sq_1.size_bits),
          ht___scalar_sq_1.d_entries, ht___scalar_sq_1.ht_size,
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
              tpch_q17_cpu_prefilter_lineitem(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 2, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q17_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d___scalar_sq_1_avg,
          d_lineitem_l_partkey,
          d_lineitem_l_quantity,
          d_lineitem_l_extendedprice,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_part.d_bits, (bloom_off ? 0u : gpu_bf_part.size_bits),
          ht_part.d_entries, ht_part.ht_size,
          gpu_bf___scalar_sq_1.d_bits, (bloom_off ? 0u : gpu_bf___scalar_sq_1.size_bits),
          ht___scalar_sq_1.d_entries, ht___scalar_sq_1.ht_size,
              d_aggtable);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };
  
      cudaEventRecord(t0);
  
      // AFP all-pipeline for '__scalar_sq_1': morsel-batched CPU prefilter + GPU build per stream.
      {
          int _n_build_batches = (num___scalar_sq_1_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
          for (int _b = 0; _b < _n_build_batches; ++_b) {
              int _b_off = _b * FASCAL_GPU_BATCH;
              int _b_cnt = std::min(FASCAL_GPU_BATCH, num___scalar_sq_1_tuples - _b_off);
              int _b_blks = (_b_cnt + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _bs = 0;
              tpch_q17_cpu_prefilter___scalar_sq_1(h___scalar_sq_1_l_partkey, h___scalar_sq_1_avg,
                  num___scalar_sq_1_tuples, _b_off, _b_cnt,
                  h___scalar_sq_1_prefilter_pinned, h___scalar_sq_1_tile_summary);
              tpch_q17_kernel___scalar_sq_1<BLOCK_THREADS, ITEMS_PER_THREAD><<<_b_blks, BLOCK_THREADS, 0, _bs>>>(
                  num___scalar_sq_1_tuples, _b_off,
                  d___scalar_sq_1_l_partkey, d___scalar_sq_1_avg,
                  ht___scalar_sq_1.d_entries, ht___scalar_sq_1.ht_size,
                  gpu_bf___scalar_sq_1.d_bits, gpu_bf___scalar_sq_1.size_bits, 3,
                  d___scalar_sq_1_prefilter_mapped, d___scalar_sq_1_tile_summary);
          }
          for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(fascal_streams[_s]));
          CUDA_CHECK(cudaGetLastError());
      }
      if (bf_storage[0].bits) gpu_bf___scalar_sq_1.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));
      // AFP all-pipeline for 'part': morsel-batched CPU prefilter + GPU build per stream.
      {
          int _n_build_batches = (num_part_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
          for (int _b = 0; _b < _n_build_batches; ++_b) {
              int _b_off = _b * FASCAL_GPU_BATCH;
              int _b_cnt = std::min(FASCAL_GPU_BATCH, num_part_tuples - _b_off);
              int _b_blks = (_b_cnt + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _bs = 0;
              tpch_q17_cpu_prefilter_part(h_part_p_partkey, h_part_p_brand, h_part_p_container,
                  num_part_tuples, _b_off, _b_cnt,
                  h_part_prefilter_pinned, h_part_tile_summary,
                  (bf_storage[0].bits ? &bf_storage[0] : nullptr));
              tpch_q17_kernel_part<BLOCK_THREADS, ITEMS_PER_THREAD><<<_b_blks, BLOCK_THREADS, 0, _bs>>>(
                  num_part_tuples, _b_off,
                  d_part_p_partkey, d_part_p_brand, d_part_p_container,
                  ht_part.d_entries, ht_part.ht_size,
                  gpu_bf_part.d_bits, gpu_bf_part.size_bits, 3,
                  d_part_prefilter_mapped, d_part_tile_summary,
                  ht___scalar_sq_1.d_entries, ht___scalar_sq_1.ht_size);
          }
          for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(fascal_streams[_s]));
          CUDA_CHECK(cudaGetLastError());
      }
      if (bf_storage[1].bits) gpu_bf_part.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));
  
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
      printf("ROW: ");
      printf("%.4f", (double)(h_result[0] / 7));
      printf("\n");
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
      odzc_mgr->unregister_column(mc_lineitem_l_partkey);
      odzc_mgr->unregister_column(mc_lineitem_l_quantity);
      odzc_mgr->unregister_column(mc_lineitem_l_extendedprice);
  
      odzc_mgr->unregister_column(mc___scalar_sq_1_l_partkey);
      odzc_mgr->unregister_column(mc___scalar_sq_1_avg);
      odzc_mgr->unregister_column(mc_part_p_partkey);
      odzc_mgr->unregister_column(mc_part_p_brand);
      odzc_mgr->unregister_column(mc_part_p_container);
      CUDA_CHECK(cudaFreeHost(h___scalar_sq_1_tile_summary));
      CUDA_CHECK(cudaFreeHost(h___scalar_sq_1_prefilter_pinned));
      gpu_bf___scalar_sq_1.free_filter();
      ht___scalar_sq_1.free_table();
      bf_storage[0].free_filter();
      if (h___scalar_sq_1_l_partkey && (void*)h___scalar_sq_1_l_partkey != (void*)_mat___scalar_sq_1_l_partkey) fascal_free_column(h___scalar_sq_1_l_partkey);
      if (h___scalar_sq_1_avg && (void*)h___scalar_sq_1_avg != (void*)_mat___scalar_sq_1_avg) fascal_free_column(h___scalar_sq_1_avg);
      CUDA_CHECK(cudaFreeHost(h_part_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_part_prefilter_pinned));
      gpu_bf_part.free_filter();
      ht_part.free_table();
      bf_storage[1].free_filter();
      fascal_free_column(h_part_p_partkey);
      fascal_free_column(h_part_p_brand);
      fascal_free_column(h_part_p_container);
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_l_partkey);
      fascal_free_column(h_l_quantity);
      fascal_free_column(h_l_extendedprice);
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
    if (_mat___scalar_sq_1_l_partkey) CUDA_CHECK(cudaFreeHost(_mat___scalar_sq_1_l_partkey));
    if (_mat___scalar_sq_1_avg) CUDA_CHECK(cudaFreeHost(_mat___scalar_sq_1_avg));
}
