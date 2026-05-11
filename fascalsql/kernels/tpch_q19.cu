// TPC-H Q19 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Source: fascalsql/kernels/tpch_q19.cu

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

// --- Ablation flags (0=GPU, 1=CPU) ---
__device__ __constant__ unsigned char d_pred_flags[MAX_PREDICATES];
static unsigned char h_pred_flags[MAX_PREDICATES] = {0};

static inline void normalize_or_group_pred_flags(unsigned char *pred_flags) {
  {
    // Mixed OR-group placements fall back to GPU because the AFP seed bitmap is boolean.
    const unsigned char normalized = ((pred_flags[0] == 1) && (pred_flags[1] == 1)) ? 1 : 0;
    pred_flags[0] = normalized;
    pred_flags[1] = normalized;
  }
}

// GPU-side Bitset pointers for string predicates (LIKE/NOT_LIKE)
// Pre-filled by regex_bitset_kernel on unique dictionary entries.
// ================= STAGE 1: tpch_q19 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q19_kernel_lineitem(
    int num_tuples,
    int batch_offset,
    int *d_part_p_brand,
    int *d_part_p_container,
    int *d_part_p_size,
    int *d_lineitem_l_partkey,
    float *d_lineitem_l_quantity,
    int *d_lineitem_l_shipmode,
    int *d_lineitem_l_shipinstruct,
    float *d_lineitem_l_extendedprice,
    float *d_lineitem_l_discount,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_part,
    uint32_t bloom_size_bits_part,
    HtEntry *ht_part,
    uint32_t ht_size_part,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  int loc_part_p_brand[ITEMS_PER_THREAD_T];
  int loc_part_p_container[ITEMS_PER_THREAD_T];
  int loc_part_p_size[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_partkey[ITEMS_PER_THREAD_T];
  float loc_lineitem_l_quantity[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_shipmode[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_shipinstruct[ITEMS_PER_THREAD_T];
  float loc_lineitem_l_extendedprice[ITEMS_PER_THREAD_T];
  float loc_lineitem_l_discount[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // Always load l_shipinstruct (needed by post-join filter regardless of pred_flags)
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_shipinstruct + tile_offset, loc_lineitem_l_shipinstruct,
      selection_flags, num_tile_items);

  if (d_pred_flags[2] == 0) {  // GPU handles predicate(s) on l_shipinstruct → evaluate
    // Predicate 2: lineitem.l_shipinstruct (unified with sub-pipeline selection)
    BlockPredAndEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        loc_lineitem_l_shipinstruct, 0, selection_flags, num_tile_items);

  }  // If CPU handled this predicate, bitmap already reflects the filter.
  // ODZC lazy load (OR group): l_shipmode
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_shipmode + tile_offset, loc_lineitem_l_shipmode,
      selection_flags, num_tile_items);

  // OR predicate group (GPU): l_shipmode IN ('AIR'=0, 'REG AIR'=4) AND with selection
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM])
        selection_flags[ITEM] = (((loc_lineitem_l_shipmode[ITEM] == 1)) || ((loc_lineitem_l_shipmode[ITEM] == 0))) ? 1 : 0;
  }

  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

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
  // Join 0 (inner): l_partkey -> part
  int loc_part_rid[ITEMS_PER_THREAD_T];
  BlockJoinProbePayloadDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_lineitem_l_partkey, selection_flags, num_tile_items,
      ht_part, ht_size_part, 1, loc_part_rid);
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ITEM++) {
    if (selection_flags[ITEM]) {
      int rid = loc_part_rid[ITEM];
      loc_part_p_brand[ITEM] = d_part_p_brand[rid];
      loc_part_p_container[ITEM] = d_part_p_container[rid];
      loc_part_p_size[ITEM] = d_part_p_size[rid];
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

  // ODZC lazy load (aggregation only): l_discount
  BlockLoadSelect<float, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_discount + tile_offset, loc_lineitem_l_discount,
      selection_flags, num_tile_items);

  // Post-join filter: Q19 three-branch OR predicate
  // Branch 1: Brand#12(1), SM CASE(27)/SM BOX(25)/SM PACK(30)/SM PKG(31), qty 1..11, size 1..5
  // Branch 2: Brand#23(7), MED BAG(16)/MED BOX(17)/MED PKG(23)/MED PACK(22), qty 10..20, size 1..10
  // Branch 3: Brand#34(13), LG CASE(11)/LG BOX(9)/LG PACK(14)/LG PKG(15), qty 20..30, size 1..15
  // Plus: l_shipinstruct = 'DELIVER IN PERSON'(0), l_shipmode IN ('AIR'=1, 'REG AIR'=0), p_size >= 1
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if (selection_flags[ITEM]) {
      int brand = loc_part_p_brand[ITEM];
      int container = loc_part_p_container[ITEM];
      float qty = loc_lineitem_l_quantity[ITEM];
      int psize = loc_part_p_size[ITEM];
      int shipmode = loc_lineitem_l_shipmode[ITEM];
      int shipinstruct = loc_lineitem_l_shipinstruct[ITEM];
      // Seed-specific Q19 (Brand#22, Brand#23, Brand#12 with qty 8-18, 10-20, 24-34)
      bool branch1 = (brand == 6)   // Brand#22
        && (container == 27 || container == 25 || container == 30 || container == 31)
        && (qty >= 8.0f) && (qty <= 18.0f)
        && (psize >= 1) && (psize <= 5);
      bool branch2 = (brand == 7)   // Brand#23
        && (container == 16 || container == 17 || container == 23 || container == 22)
        && (qty >= 10.0f) && (qty <= 20.0f)
        && (psize >= 1) && (psize <= 10);
      bool branch3 = (brand == 1)   // Brand#12
        && (container == 11 || container == 9 || container == 14 || container == 15)
        && (qty >= 24.0f) && (qty <= 34.0f)
        && (psize >= 1) && (psize <= 15);
      bool common = (shipinstruct == 0) && (shipmode == 1 || shipmode == 0);
      if (!((branch1 || branch2 || branch3) && common))
        selection_flags[ITEM] = 0;
    }
  }

  } while(0); // thread_any_alive
  // Scalar aggregation with block-level reduction (one atomic per block)
  double _acc0 = 0; double _acc1 = 0;
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        _acc0 += ((((double)loc_lineitem_l_extendedprice[ITEM]) * (1 - loc_lineitem_l_discount[ITEM])));
        _acc1 += 1.0;
      }
  }
  _acc0 = blockReduceSumDouble<BLOCK_THREADS_T>(_acc0);
  if (threadIdx.x == 0) atomicAdd(&aggtable[0], (double)_acc0);
  _acc1 = blockReduceSumDouble<BLOCK_THREADS_T>(_acc1);
  if (threadIdx.x == 0) atomicAdd(&aggtable[1], (double)_acc1);
}
// CPU predicate evaluation for tpch_q19 (fact table: lineitem)
// Multi-pass: one pass per column group + BF probe; bitmap skip between passes
static void tpch_q19_cpu_predicate_lineitem(
    int *h_data[],
    int morsel_offset,
    int elem_cnt,
    uint32_t *bitmap,
    int selection_step,
    int ablation_indicator,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0)
{
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = morsel_offset >> 3, nb = (elem_cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, elem_cnt);

  int *l_partkey = h_data[0], *l_shipmode = h_data[2], *l_shipinstruct = h_data[3];

  if (h_pred_flags[2] == 1)
    fascal::cpu_pred::avx2_eq_pass(bm, bs, l_shipinstruct, morsel_offset, elem_cnt, 0);

  if (h_pred_flags[0] == 1 || h_pred_flags[1] == 1)
    fascal::cpu_pred::avx2_or_eq2_pass(bm, bs, l_shipmode, morsel_offset, elem_cnt, 1, 0);

  if (bloom_filters && 0 < num_bloom_filters && bloom_filters[0])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, l_partkey, morsel_offset, elem_cnt, bloom_filters[0]);
}

// ---- AFP: multi-threaded CPU prefilter for result pipeline 'lineitem' ----
// CPU fills bitmap, then GPU launches (no doorbell overlap).
// total_tuples: total fact-table size (for bitmap bounds).
// batch_offset + batch_count define the sub-range to prefilter.
// Sequential mode: batch_offset=0, batch_count=total_tuples (default).
// Pipelined mode: called per-batch with specific offset/count.
static void tpch_q19_cpu_prefilter_lineitem(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
    fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
        [&](int off, int cnt) { tpch_q19_cpu_predicate_lineitem(h_data, off, cnt, bitmap, 0, 0, bloom_filters, num_bloom_filters); },
        batch_offset, total_tuples);
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q19_kernel_part(
    int num_tuples, int batch_offset,
    const int *__restrict__ d_part_p_partkey,
    const int *__restrict__ d_part_p_size,
    const int *__restrict__ d_part_p_brand,
    const int *__restrict__ d_part_p_container,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_part_p_partkey[ITEMS_PER_THREAD_T];
  int loc_part_p_size[ITEMS_PER_THREAD_T];
  int loc_part_p_brand[ITEMS_PER_THREAD_T];
  int loc_part_p_container[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_partkey + tile_offset), loc_part_p_partkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_size + tile_offset), loc_part_p_size, selection_flags, num_tile_items);

  BlockPredAndGTE<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_part_p_size, 1, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_brand + tile_offset), loc_part_p_brand, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_part_p_container + tile_offset), loc_part_p_container, selection_flags, num_tile_items);

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

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'part' ----
// Fills entire prefilter bitmap (scalar preds + upstream BF probes),
// then GPU build kernel launches with bitmap 100% ready (no doorbell).
static void tpch_q19_cpu_prefilter_part(
    int *h_part_p_partkey,
    int *h_part_p_size,
    int *h_part_p_brand,
    int *h_part_p_container,
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
                if ((h_part_p_size[idx] >= 1)) {
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

    // ---------------- STAGE 1 EXECUTION: tpch_q19 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q19
  // ============================================================
      printf("=== FaScalSQL: tpch_q19 ===\n");
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
  
      snprintf(path, sizeof(path), "%s/l_partkey.bin", data_dir);
      int *h_l_partkey = load_binary_column(path, num_tuples);
      if (!h_l_partkey) { fprintf(stderr, "Failed to load l_partkey\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_quantity.bin", data_dir);
      float *h_l_quantity = (float*)load_binary_column(path, num_tuples);
      if (!h_l_quantity) { fprintf(stderr, "Failed to load l_quantity\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_shipmode.bin", data_dir);
      int *h_l_shipmode = load_binary_column(path, num_tuples);
      if (!h_l_shipmode) { fprintf(stderr, "Failed to load l_shipmode\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_shipinstruct.bin", data_dir);
      int *h_l_shipinstruct = load_binary_column(path, num_tuples);
      if (!h_l_shipinstruct) { fprintf(stderr, "Failed to load l_shipinstruct\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_extendedprice.bin", data_dir);
      float *h_l_extendedprice = (float*)load_binary_column(path, num_tuples);
      if (!h_l_extendedprice) { fprintf(stderr, "Failed to load l_extendedprice\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_discount.bin", data_dir);
      float *h_l_discount = (float*)load_binary_column(path, num_tuples);
      if (!h_l_discount) { fprintf(stderr, "Failed to load l_discount\n"); return 1; }
  
      // ----- Load dimension: part -----
      int num_part_tuples = 0;
  
      snprintf(path, sizeof(path), "%s/p_partkey.bin", data_dir);
      int *h_part_p_partkey = load_binary_column_auto(path, &num_part_tuples);
  
      if (!h_part_p_partkey || num_part_tuples == 0) {
          fprintf(stderr, "Failed to load table column p_partkey\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/p_size.bin", data_dir);
      int *h_part_p_size = load_binary_column_auto(path, &num_part_tuples);
  
      if (!h_part_p_size || num_part_tuples == 0) {
          fprintf(stderr, "Failed to load table column p_size\n"); return 1;
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
      // Dynamic sample-based selectivity profiling (fact-table predicates)
      cqo_in.predicate_selectivities.push_back(
          fascal_estimate_selectivity(h_l_shipmode, num_tuples, [](int v){ return v == 1 || v == 0; }));  // OR: AIR(1) | REG AIR(0)
      cqo_in.predicate_selectivities.push_back(
          fascal_estimate_selectivity(h_l_shipmode, num_tuples, [](int v){ return v == 1 || v == 0; }));  // same OR group (normalized)
      cqo_in.predicate_selectivities.push_back(
          fascal_estimate_eq_sel(h_l_shipinstruct, num_tuples, 0));  // l_shipinstruct == DELIVER IN PERSON(0)
      cqo_in.predicate_selectivities.push_back(0.300000);  // post-join compound OR-of-AND (dim-side, keep hardcoded)
      // Dynamic BF selectivity: part(all rows qualify -- predicates are post-join)
      cqo_in.join_bf_selectivities.push_back(fascal_estimate_join_bf_sel(num_part_tuples, num_part_tuples));
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
      int *h_data[] = {h_l_partkey, (int*)h_l_quantity, h_l_shipmode, h_l_shipinstruct, (int*)h_l_extendedprice, (int*)h_l_discount};
      fascal::runtime::AFPManager afp_manager(0);
      int afp_filter_id = 0;
      if (h_pred_flags[0] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(2, fascal::runtime::AFPManager::FilterOp::EQ, 0);
        afp_manager.register_filter(afp_filter_id++, desc);
      }
      if (h_pred_flags[1] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(2, fascal::runtime::AFPManager::FilterOp::EQ, -1);
        afp_manager.register_filter(afp_filter_id++, desc);
      }
      if (h_pred_flags[2] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(3, fascal::runtime::AFPManager::FilterOp::EQ, 0);
        afp_manager.register_filter(afp_filter_id++, desc);
      }
  
      // === Bloom Filters (Build Pipelines) ===
      fascal::runtime::BloomFilter bf_storage[1];
      fascal::runtime::BloomFilter *bf_array[1];
  
  
      bf_array[0] = (!bloom_off && (int)0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[0] : nullptr;
  
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
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_shipmode;
      int *d_lineitem_l_shipmode = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_shipmode, num_tuples, "l_shipmode", &mc_lineitem_l_shipmode, &d_lineitem_l_shipmode);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_shipinstruct;
      int *d_lineitem_l_shipinstruct = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_shipinstruct, num_tuples, "l_shipinstruct", &mc_lineitem_l_shipinstruct, &d_lineitem_l_shipinstruct);
  
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
  
      fascal::runtime::ODZCManager::MappedColumn mc_part_p_partkey;
      mc_part_p_partkey = odzc_mgr->register_column(h_part_p_partkey, num_part_tuples * sizeof(int));
      int *d_part_p_partkey = static_cast<int*>(mc_part_p_partkey.device_ptr);
      if (!d_part_p_partkey) {
          fprintf(stderr, "ODZC failed for part.p_partkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_part_p_partkey, num_part_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_part_p_partkey, h_part_p_partkey, num_part_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_part_p_size;
      mc_part_p_size = odzc_mgr->register_column(h_part_p_size, num_part_tuples * sizeof(int));
      int *d_part_p_size = static_cast<int*>(mc_part_p_size.device_ptr);
      if (!d_part_p_size) {
          fprintf(stderr, "ODZC failed for part.p_size, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_part_p_size, num_part_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_part_p_size, h_part_p_size, num_part_tuples * sizeof(int), cudaMemcpyHostToDevice));
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
              tpch_q19_cpu_prefilter_lineitem(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 1);
              tpch_q19_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_part_p_brand,
          d_part_p_container,
          d_part_p_size,
          d_lineitem_l_partkey,
          d_lineitem_l_quantity,
          d_lineitem_l_shipmode,
          d_lineitem_l_shipinstruct,
          d_lineitem_l_extendedprice,
          d_lineitem_l_discount,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_part.d_bits, (bloom_off ? 0u : gpu_bf_part.size_bits),
          ht_part.d_entries, ht_part.ht_size,
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
              tpch_q19_cpu_prefilter_lineitem(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 1, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q19_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_part_p_brand,
          d_part_p_container,
          d_part_p_size,
          d_lineitem_l_partkey,
          d_lineitem_l_quantity,
          d_lineitem_l_shipmode,
          d_lineitem_l_shipinstruct,
          d_lineitem_l_extendedprice,
          d_lineitem_l_discount,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_part.d_bits, (bloom_off ? 0u : gpu_bf_part.size_bits),
          ht_part.d_entries, ht_part.ht_size,
              d_aggtable);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };
  
      cudaEventRecord(t0);
  
      // AFP all-pipeline for 'part': morsel-batched build on fascal_streams[b % N].
      tpch_q19_cpu_prefilter_part(h_part_p_partkey, h_part_p_size, h_part_p_brand, h_part_p_container, num_part_tuples, h_part_prefilter_pinned, h_part_tile_summary);
      {
        int _build_nb = (num_part_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
        for (int _b = 0; _b < _build_nb; ++_b) {
          int _bo = _b * FASCAL_GPU_BATCH;
          int _bc = std::min(FASCAL_GPU_BATCH, num_part_tuples - _bo);
          int _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
          cudaStream_t _bst = 0;
          tpch_q19_kernel_part<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, _bst>>>(
              num_part_tuples, _bo, d_part_p_partkey, d_part_p_size, d_part_p_brand, d_part_p_container,
              ht_part.d_entries, ht_part.ht_size,
              gpu_bf_part.d_bits, gpu_bf_part.size_bits, 3,
              d_part_prefilter_mapped, d_part_tile_summary);
        }
        for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(fascal_streams[_s]));
      }
      CUDA_CHECK(cudaGetLastError());
      if (bf_storage[0].bits) gpu_bf_part.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));
  
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
      printf("%.4f", (double)h_result[0]);
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
      odzc_mgr->unregister_column(mc_lineitem_l_shipmode);
      odzc_mgr->unregister_column(mc_lineitem_l_shipinstruct);
      odzc_mgr->unregister_column(mc_lineitem_l_extendedprice);
      odzc_mgr->unregister_column(mc_lineitem_l_discount);
  
      odzc_mgr->unregister_column(mc_part_p_partkey);
      odzc_mgr->unregister_column(mc_part_p_size);
      odzc_mgr->unregister_column(mc_part_p_brand);
      odzc_mgr->unregister_column(mc_part_p_container);
      CUDA_CHECK(cudaFreeHost(h_part_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_part_prefilter_pinned));
      gpu_bf_part.free_filter();
      ht_part.free_table();
      bf_storage[0].free_filter();
      fascal_free_column(h_part_p_partkey);
      fascal_free_column(h_part_p_size);
      fascal_free_column(h_part_p_brand);
      fascal_free_column(h_part_p_container);
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_l_partkey);
      fascal_free_column(h_l_quantity);
      fascal_free_column(h_l_shipmode);
      fascal_free_column(h_l_shipinstruct);
      fascal_free_column(h_l_extendedprice);
      fascal_free_column(h_l_discount);
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
