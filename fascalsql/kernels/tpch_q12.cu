// TPC-H Q12 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Source: fascalsql/kernels/tpch_q12.cu

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
#define NUM_AGGREGATES 2

// -----------------------------------------------------------------------------
// Ablation flags (0=GPU, 1=CPU)
// -----------------------------------------------------------------------------
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
// ================= STAGE 1: tpch_q12 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q12_kernel_lineitem(
    int num_tuples,
    int batch_offset,
    int *d_lineitem_l_receiptdate,
    int *d_lineitem_l_commitdate,
    int *d_lineitem_l_shipdate,
    int *d_lineitem_l_orderkey_enc,
    int *d_lineitem_l_shipmode,
    int *d_lineitem_l_shipmode_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_orders,
    uint32_t bloom_size_bits_orders,
    HtEntry *ht_orders,
    uint32_t ht_size_orders,
    unsigned long long *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  int loc_lineitem_l_receiptdate[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_commitdate[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_shipdate[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_orderkey_enc[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_shipmode[ITEMS_PER_THREAD_T];
  int loc_lineitem_l_shipmode_enc[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // ODZC lazy load: l_receiptdate (survivors of prior predicates)
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_receiptdate + tile_offset, loc_lineitem_l_receiptdate,
      selection_flags, num_tile_items);

  // Predicate 2: lineitem.l_receiptdate (unified with sub-pipeline selection)
  if (d_pred_flags[2] == 0) {  // GPU
    BlockPredAndGTE<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        loc_lineitem_l_receiptdate, 19940101, selection_flags, num_tile_items);
  }

  // Predicate 3: lineitem.l_receiptdate (unified with sub-pipeline selection)
  if (d_pred_flags[3] == 0) {  // GPU
    BlockPredAndLT<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        loc_lineitem_l_receiptdate, 19950101, selection_flags, num_tile_items);
  }

  // ODZC lazy load (OR group, also used in aggregation): l_shipmode
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_shipmode + tile_offset, loc_lineitem_l_shipmode,
      selection_flags, num_tile_items);

  // OR predicate group (GPU): (p1 OR p2 OR ...) AND with selection
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM])
        selection_flags[ITEM] = (((loc_lineitem_l_shipmode[ITEM] == 3)) || ((loc_lineitem_l_shipmode[ITEM] == 5))) ? 1 : 0;
  }

  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

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
  // ODZC lazy load (post-join check source): lineitem.l_commitdate
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_commitdate + tile_offset, loc_lineitem_l_commitdate,
      selection_flags, num_tile_items);

  // ODZC lazy load (post-join check source): lineitem.l_shipdate
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_shipdate + tile_offset, loc_lineitem_l_shipdate,
      selection_flags, num_tile_items);

  // Join (payload = o_orderpriority, no separate column lookup)
  int loc_orders_o_orderpriority[ITEMS_PER_THREAD_T];
  BlockJoinProbePayloadDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_lineitem_l_orderkey_enc, selection_flags, num_tile_items,
      ht_orders, ht_size_orders, 0, loc_orders_o_orderpriority);

  // ODZC lazy load (fact GROUP BY, before aggregation): l_shipmode_enc
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_lineitem_l_shipmode_enc + tile_offset, loc_lineitem_l_shipmode_enc,
      selection_flags, num_tile_items);

  // Post-join check: lineitem.l_receiptdate > lineitem.l_commitdate
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if (selection_flags[ITEM]) {
      if (!(loc_lineitem_l_receiptdate[ITEM] > loc_lineitem_l_commitdate[ITEM]))
        selection_flags[ITEM] = 0;
    }
  }

  // Post-join check: lineitem.l_shipdate < lineitem.l_commitdate
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if (selection_flags[ITEM]) {
      if (!(loc_lineitem_l_shipdate[ITEM] < loc_lineitem_l_commitdate[ITEM]))
        selection_flags[ITEM] = 0;
    }
  }

  // GROUP BY direct-index aggregation (multi-aggregate)
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int gk = loc_lineitem_l_shipmode_enc[ITEM];
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], (((((((long long)loc_orders_o_orderpriority[ITEM]) == 0) || (loc_orders_o_orderpriority[ITEM] == 1))) ? (1) : (0))));
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 1], (((((((long long)loc_orders_o_orderpriority[ITEM]) != 0) && (loc_orders_o_orderpriority[ITEM] != 1))) ? (1) : (0))));
      }
  }
  } while(0);
}
static void tpch_q12_cpu_predicate_lineitem(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bloom_filters[], int num_bloom_filters) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  int *l_receiptdate = h_data[0], *l_orderkey_enc = h_data[3], *l_shipmode = h_data[4];
  // Pass 0: range predicates on l_receiptdate
  if (h_pred_flags[2] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, l_receiptdate, offset, cnt, 19940101, 19950101, true, true, true);
  if (h_pred_flags[3] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, l_receiptdate, offset, cnt, 19940101, 19950101, true, true, true);
  // Pass 1: OR group (l_shipmode == 2 OR l_shipmode == 5) -- kept as normalize_or_group_pred_flags ensures both flags are in sync
  if (h_pred_flags[0] == 1 || h_pred_flags[1] == 1)
    fascal::cpu_pred::avx2_or_eq2_pass(bm, bs, l_shipmode, offset, cnt, 3, 5);
  // Pass 2: Bloom probe
  if (bloom_filters && 0 < num_bloom_filters && bloom_filters[0])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, l_orderkey_enc, offset, cnt, bloom_filters[0]);
}

static void tpch_q12_cpu_prefilter_lineitem(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
    fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
        [&](int off, int cnt) { tpch_q12_cpu_predicate_lineitem(h_data, off, cnt, bitmap, bloom_filters, num_bloom_filters); },
        batch_offset, total_tuples);
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q12_kernel_orders(
    int num_tuples, int batch_offset,
    const int *__restrict__ d_orders_o_orderkey,
    const int *__restrict__ d_orders_o_orderpriority,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_orders_o_orderkey[ITEMS_PER_THREAD_T];
  int loc_orders_o_orderpriority[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_orders_o_orderkey + tile_offset), loc_orders_o_orderkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_orders_o_orderpriority + tile_offset), loc_orders_o_orderpriority, selection_flags, num_tile_items);

  // Optional: sub-stage aggregation (e.g. atomicAdd to d_aggtable) can be added here for TPC-H.
  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int row_idx = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one_direct(row_idx, loc_orders_o_orderpriority[ITEM], 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(row_idx, 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'orders' ----
// Fills entire prefilter bitmap (scalar preds + upstream BF probes),
// then GPU build kernel launches with bitmap 100% ready (no doorbell).
static void tpch_q12_cpu_prefilter_orders(
    int *h_orders_o_orderkey,
    int *h_orders_o_orderpriority,
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

    auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));

    // ---------------- STAGE 1 EXECUTION: tpch_q12 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q12
  // ============================================================
      printf("=== FaScalSQL: tpch_q12 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/l_receiptdate.bin", data_dir);
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
  
      snprintf(path, sizeof(path), "%s/l_receiptdate.bin", data_dir);
      int *h_l_receiptdate = load_binary_column(path, num_tuples);
      if (!h_l_receiptdate) { fprintf(stderr, "Failed to load l_receiptdate\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_commitdate.bin", data_dir);
      int *h_l_commitdate = load_binary_column(path, num_tuples);
      if (!h_l_commitdate) { fprintf(stderr, "Failed to load l_commitdate\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_shipdate.bin", data_dir);
      int *h_l_shipdate = load_binary_column(path, num_tuples);
      if (!h_l_shipdate) { fprintf(stderr, "Failed to load l_shipdate\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_orderkey_enc.bin", data_dir);
      int *h_l_orderkey_enc = load_binary_column(path, num_tuples);
      if (!h_l_orderkey_enc) { fprintf(stderr, "Failed to load l_orderkey_enc\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_shipmode.bin", data_dir);
      int *h_l_shipmode = load_binary_column(path, num_tuples);
      if (!h_l_shipmode) { fprintf(stderr, "Failed to load l_shipmode\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/l_shipmode_enc.bin", data_dir);
      int *h_l_shipmode_enc = load_binary_column(path, num_tuples);
      if (!h_l_shipmode_enc) { fprintf(stderr, "Failed to load l_shipmode_enc\n"); return 1; }
  
      // ----- Load dimension: orders -----
      int num_orders_tuples = 0;
  
      snprintf(path, sizeof(path), "%s/o_orderkey.bin", data_dir);
      int *h_orders_o_orderkey = load_binary_column_auto(path, &num_orders_tuples);
  
      if (!h_orders_o_orderkey || num_orders_tuples == 0) {
          fprintf(stderr, "Failed to load table column o_orderkey\n"); return 1;
      }
  
      snprintf(path, sizeof(path), "%s/o_orderpriority.bin", data_dir);
      int *h_orders_o_orderpriority = load_binary_column_auto(path, &num_orders_tuples);
  
      if (!h_orders_o_orderpriority || num_orders_tuples == 0) {
          fprintf(stderr, "Failed to load table column o_orderpriority\n"); return 1;
      }
      printf("Loaded orders: %d tuples\n", num_orders_tuples);
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      // Dynamic sample-based selectivity profiling (fact-table predicates)
      cqo_in.predicate_selectivities.push_back(
          fascal_estimate_selectivity(h_l_shipmode, num_tuples, [](int v){ return v == 3 || v == 5; }));  // OR: SHIP(3) | MAIL(5)
      cqo_in.predicate_selectivities.push_back(
          fascal_estimate_selectivity(h_l_shipmode, num_tuples, [](int v){ return v == 3 || v == 5; }));  // same OR group (normalized)
      cqo_in.predicate_selectivities.push_back(
          fascal_estimate_range_sel(h_l_receiptdate, num_tuples, 19940101, 19941231));  // l_receiptdate >= 19940101
      cqo_in.predicate_selectivities.push_back(
          fascal_estimate_range_sel(h_l_receiptdate, num_tuples, 19940101, 19941231));  // l_receiptdate < 19950101 (same range)
      // Dynamic BF selectivity: orders(all rows qualify)
      cqo_in.join_bf_selectivities.push_back(fascal_estimate_join_bf_sel(num_orders_tuples, num_orders_tuples));
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
      int *h_data[] = {h_l_receiptdate, h_l_commitdate, h_l_shipdate, h_l_orderkey_enc, h_l_shipmode};
      fascal::runtime::AFPManager afp_manager(0);
      int afp_filter_id = 0;
      if (h_pred_flags[0] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(4, fascal::runtime::AFPManager::FilterOp::EQ, 3);
        afp_manager.register_filter(afp_filter_id++, desc);
      }
      if (h_pred_flags[1] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(4, fascal::runtime::AFPManager::FilterOp::EQ, 5);
        afp_manager.register_filter(afp_filter_id++, desc);
      }
      if (h_pred_flags[2] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(0, fascal::runtime::AFPManager::FilterOp::GE, 19940101);
        afp_manager.register_filter(afp_filter_id++, desc);
      }
      if (h_pred_flags[3] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(0, fascal::runtime::AFPManager::FilterOp::LT, 19950101);
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
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_receiptdate;
      int *d_lineitem_l_receiptdate = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_receiptdate, num_tuples, "l_receiptdate", &mc_lineitem_l_receiptdate, &d_lineitem_l_receiptdate);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_commitdate;
      int *d_lineitem_l_commitdate = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_commitdate, num_tuples, "l_commitdate", &mc_lineitem_l_commitdate, &d_lineitem_l_commitdate);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_shipdate;
      int *d_lineitem_l_shipdate = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_shipdate, num_tuples, "l_shipdate", &mc_lineitem_l_shipdate, &d_lineitem_l_shipdate);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_orderkey_enc;
      int *d_lineitem_l_orderkey_enc = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_orderkey_enc, num_tuples, "l_orderkey_enc", &mc_lineitem_l_orderkey_enc, &d_lineitem_l_orderkey_enc);
  
      fascal::runtime::ODZCManager::MappedColumn mc_lineitem_l_shipmode;
      int *d_lineitem_l_shipmode = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_shipmode, num_tuples, "l_shipmode", &mc_lineitem_l_shipmode, &d_lineitem_l_shipmode);
  
      int *d_lineitem_l_shipmode_enc = nullptr;
      CUDA_CHECK(cudaMalloc(&d_lineitem_l_shipmode_enc, num_tuples * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(d_lineitem_l_shipmode_enc, h_l_shipmode_enc, num_tuples * sizeof(int), cudaMemcpyHostToDevice));
  
      fascal::runtime::ODZCManager::MappedColumn mc_orders_o_orderkey;
      mc_orders_o_orderkey = odzc_mgr->register_column(h_orders_o_orderkey, num_orders_tuples * sizeof(int));
      int *d_orders_o_orderkey = static_cast<int*>(mc_orders_o_orderkey.device_ptr);
      if (!d_orders_o_orderkey) {
          fprintf(stderr, "ODZC failed for orders.o_orderkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_orders_o_orderkey, num_orders_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_orders_o_orderkey, h_orders_o_orderkey, num_orders_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_orders_o_orderpriority;
      mc_orders_o_orderpriority = odzc_mgr->register_column(h_orders_o_orderpriority, num_orders_tuples * sizeof(int));
      int *d_orders_o_orderpriority = static_cast<int*>(mc_orders_o_orderpriority.device_ptr);
      if (!d_orders_o_orderpriority) {
          fprintf(stderr, "ODZC failed for orders.o_orderpriority, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_orders_o_orderpriority, num_orders_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_orders_o_orderpriority, h_orders_o_orderpriority, num_orders_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
  
  
      // === Aggregation Table ===
      size_t num_groups = 1ULL;
      num_groups *= 7ULL;
      unsigned long long *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, num_groups * NUM_AGGREGATES * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(unsigned long long)));
  
      // === Hash Table Build ===
  
  
      GpuHashTable ht_orders;
      ht_orders = GpuHashTable::allocate_direct(num_orders_tuples);
      GpuBloomFilter gpu_bf_orders = GpuBloomFilter::allocate(num_orders_tuples, 3);
      bf_storage[0] = fascal::runtime::BloomFilter::allocate(num_orders_tuples, 3);
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
              tpch_q12_cpu_prefilter_lineitem(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 1);
              tpch_q12_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_lineitem_l_receiptdate,
          d_lineitem_l_commitdate,
          d_lineitem_l_shipdate,
          d_lineitem_l_orderkey_enc,
          d_lineitem_l_shipmode,
          d_lineitem_l_shipmode_enc,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_orders.d_bits, (bloom_off ? 0u : gpu_bf_orders.size_bits),
          ht_orders.d_entries, ht_orders.ht_size,
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
              tpch_q12_cpu_prefilter_lineitem(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 1, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q12_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_lineitem_l_receiptdate,
          d_lineitem_l_commitdate,
          d_lineitem_l_shipdate,
          d_lineitem_l_orderkey_enc,
          d_lineitem_l_shipmode,
          d_lineitem_l_shipmode_enc,
                  d_seed_bitmap,
                  d_tile_summary,
          gpu_bf_orders.d_bits, (bloom_off ? 0u : gpu_bf_orders.size_bits),
          ht_orders.d_entries, ht_orders.ht_size,
              d_aggtable);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };
  
      cudaEventRecord(t0);
  
      // AFP all-pipeline for 'orders': morsel-batched build on fascal_streams[b % N].
      tpch_q12_cpu_prefilter_orders(h_orders_o_orderkey, h_orders_o_orderpriority, num_orders_tuples, h_orders_prefilter_pinned, h_orders_tile_summary);
      {
        int _build_nb = (num_orders_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
        for (int _b = 0; _b < _build_nb; ++_b) {
          int _bo = _b * FASCAL_GPU_BATCH;
          int _bc = std::min(FASCAL_GPU_BATCH, num_orders_tuples - _bo);
          int _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
          cudaStream_t _bst = 0;
          tpch_q12_kernel_orders<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, _bst>>>(
              num_orders_tuples, _bo, d_orders_o_orderkey, d_orders_o_orderpriority,
              ht_orders.d_entries, ht_orders.ht_size,
              gpu_bf_orders.d_bits, gpu_bf_orders.size_bits, 3,
              d_orders_prefilter_mapped, d_orders_tile_summary);
        }
        for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(fascal_streams[_s]));
      }
      CUDA_CHECK(cudaGetLastError());
      // Build CPU BF from host data using the CPU hash function.
      // Q12 has no filter on orders: all row indices 0..num_orders_tuples-1 qualify.
      if (bf_storage[0].bits) {
          for (int _r = 0; _r < num_orders_tuples; ++_r)
              bf_storage[0].insert(_r);
      }
  
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
        int any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0) || (h_agg_table[k * NUM_AGGREGATES + 1] != 0)) ? 1 : 0;
        if (any) {
          size_t gk = k;
        int key0 = (int)(gk % 7ULL);
        gk /= 7;
          printf("ROW: ");
          printf("%s", [&]() -> const char* {
    static const int keys[] = { 0, 1, 2, 3, 4, 5, 6 };
    static const char* vals[] = { "REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB" };
    int l = 0, r = 6;
    int target = (int)key0;
    while (l <= r) {
      int m = l + (r - l) / 2;
      if (keys[m] == target) return vals[m];
      if (keys[m] < target) l = m + 1;
      else r = m - 1;
    }
    return "UNKNOWN";
  }());
          printf("|%lld", (long long)h_agg_table[k * NUM_AGGREGATES + 0]);
          printf("|%lld", (long long)h_agg_table[k * NUM_AGGREGATES + 1]);
          printf("\n");
        }
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
      odzc_mgr->unregister_column(mc_lineitem_l_receiptdate);
      odzc_mgr->unregister_column(mc_lineitem_l_commitdate);
      odzc_mgr->unregister_column(mc_lineitem_l_shipdate);
      odzc_mgr->unregister_column(mc_lineitem_l_orderkey_enc);
      odzc_mgr->unregister_column(mc_lineitem_l_shipmode);
      CUDA_CHECK(cudaFree(d_lineitem_l_shipmode_enc));
  
      odzc_mgr->unregister_column(mc_orders_o_orderkey);
      odzc_mgr->unregister_column(mc_orders_o_orderpriority);
      CUDA_CHECK(cudaFreeHost(h_orders_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_orders_prefilter_pinned));
      gpu_bf_orders.free_filter();
      ht_orders.free_table();
      bf_storage[0].free_filter();
      fascal_free_column(h_orders_o_orderkey);
      fascal_free_column(h_orders_o_orderpriority);
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_l_receiptdate);
      fascal_free_column(h_l_commitdate);
      fascal_free_column(h_l_shipdate);
      fascal_free_column(h_l_orderkey_enc);
      fascal_free_column(h_l_shipmode);
      fascal_free_column(h_l_shipmode_enc);
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
