// TPC-H Q4 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Refactored: uses shared macros and CPU predicate helpers.

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

// =============================================================================
// GPU Kernel: scan orders, apply predicates, semijoin, GROUP BY aggregate
// =============================================================================
template <int BT, int IPT>
__global__ void tpch_q4_kernel(
    int num_tuples, int batch_offset,
    int *d_o_orderkey, int *d_o_orderdate, int *d_o_orderpriority_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const unsigned int *__restrict__ d_late_bitmap_words,
    int max_orderkey,
    unsigned long long *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int o_orderkey[IPT], o_orderdate[IPT], o_orderpriority_enc[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // Pred 0,1: o_orderdate >= 19970701 AND o_orderdate < 19971001
    if (d_pred_flags[0] == 0 || d_pred_flags[1] == 0) {
      BlockLoadSelect<int, BT, IPT>(d_o_orderdate + tile_offset, o_orderdate, selection_flags, num_tile_items);
      if (d_pred_flags[0] == 0)
        BlockPredAndGTE<int, BT, IPT>(o_orderdate, 19970701, selection_flags, num_tile_items);
      if (d_pred_flags[1] == 0)
        BlockPredAndLT<int, BT, IPT>(o_orderdate, 19971001, selection_flags, num_tile_items);
    }

    FASCAL_CHECK_ALIVE(IPT);

    // Dense bitmap EXISTS-probe on o_orderkey: 1 if any lineitem with same
    // orderkey has l_receiptdate > l_commitdate. Replaces the previous
    // generic HT + Bloom (which fell back to ~9.6 GB zero-copy at SF=100).
    BlockLoadSelect<int, BT, IPT>(d_o_orderkey + tile_offset, o_orderkey, selection_flags, num_tile_items);
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if (selection_flags[i]) {
        unsigned int ok = static_cast<unsigned int>(o_orderkey[i]);
        bool found = (ok <= static_cast<unsigned int>(max_orderkey)) &&
                     ((d_late_bitmap_words[ok >> 5] >> (ok & 31u)) & 1u);
        selection_flags[i] = found ? 1 : 0;
      }
    }

    FASCAL_CHECK_ALIVE(IPT);

    // Load GROUP BY key
    BlockLoadSelect<int, BT, IPT>(d_o_orderpriority_enc + tile_offset, o_orderpriority_enc, selection_flags, num_tile_items);

    // GROUP BY direct-index aggregation
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i]) {
        int gk = o_orderpriority_enc[i];
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], 1LL);
      }
    }
  } while(0);
}

// =============================================================================
// Build kernel: correlated subquery (lineitem l_receiptdate > l_commitdate)
// =============================================================================
template <int BT, int IPT>
__global__ void tpch_q4_build_sq1(
    int num_tuples, int batch_offset,
    const int *__restrict__ d_l_orderkey,
    const int *__restrict__ d_l_receiptdate,
    const int *__restrict__ d_l_commitdate,
    unsigned int *__restrict__ d_late_bitmap_words,
    int max_orderkey,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);

  int l_orderkey[IPT], l_receiptdate[IPT], l_commitdate[IPT];

  BlockLoadSelect<int, BT, IPT>((int*)(d_l_orderkey + tile_offset), l_orderkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_l_receiptdate + tile_offset), l_receiptdate, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_l_commitdate + tile_offset), l_commitdate, selection_flags, num_tile_items);

  // Build-side filter: l_receiptdate > l_commitdate
  #pragma unroll
  for (int i = 0; i < IPT; ++i)
    if (selection_flags[i] && !(l_receiptdate[i] > l_commitdate[i]))
      selection_flags[i] = 0;

  // Set the late-orderkey bit for every survivor — replaces the prior
  // HT + Bloom build that became ~9.6 GB zero-copy at SF=100.
  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    unsigned int ok = static_cast<unsigned int>(l_orderkey[i]);
    if (ok > static_cast<unsigned int>(max_orderkey)) continue;
    atomicOr(&d_late_bitmap_words[ok >> 5], 1u << (ok & 31u));
  }
}

// =============================================================================
// CPU Predicate: AVX2 bitmap filtering (orders fact table)
// =============================================================================
static void cpu_predicate(int *h_data[], int offset, int cnt, uint32_t *bitmap,
                          fascal::runtime::BloomFilter **bloom_filters, int num_bloom_filters) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);

  int *o_orderkey = h_data[0], *o_orderdate = h_data[1];

  // Pass 0: o_orderdate >= 19970701 AND o_orderdate < 19971001
  if (h_pred_flags[0] == 1 || h_pred_flags[1] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, o_orderdate, offset, cnt,
        19970701, 19971001, h_pred_flags[0] == 1, h_pred_flags[1] == 1, true);

  // Pass 1: Bloom probe on o_orderkey
  if (bloom_filters && 0 < num_bloom_filters && bloom_filters[0])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, o_orderkey, offset, cnt, bloom_filters[0]);
}

// =============================================================================
// CPU Prefilter: correlated subquery build pipeline
// =============================================================================
static void cpu_prefilter_sq1(
    int *h_l_orderkey, int *h_l_receiptdate, int *h_l_commitdate,
    int num_tuples, uint32_t *bitmap, uint8_t *tile_summary) {
  fascal_cpu_prefilter_run(num_tuples, bitmap, tile_summary,
      [&](int offset, int cnt) {
        uint32_t word_start = (uint32_t)offset >> 5;
        uint32_t word_end   = ((uint32_t)(offset + cnt) + 31) >> 5;
        memset(&bitmap[word_start], 0, (word_end - word_start) * sizeof(uint32_t));
        for (int i = 0; i < cnt; ++i) {
          int idx = offset + i;
          bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
        }
      });
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
  const char *data_dir = getenv("FASCALSQL_DATA_DIR");
  if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
  if (argc >= 3) data_dir = argv[2];
  if (!data_dir) { fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n"); return 1; }

  printf("=== FaScalSQL: tpch_q4 ===\n");
  if (argc < 3) { printf("Usage: %s <num_tuples> <data_dir>\n", argv[0]); return 1; }
  data_dir = argv[2];

  // Auto-detect num_tuples (orders table)
  int num_tuples = 0;
  { char p[512]; snprintf(p, sizeof(p), "%s/o_orderkey.bin", data_dir);
    int c = 0; int *t = load_binary_column_auto(p, &c); if (t) fascal_free_column(t);
    num_tuples = c; }
  if (num_tuples <= 0) { fprintf(stderr, "Cannot auto-detect num_tuples\n"); return 1; }
  printf("num_tuples=%d data_dir=%s\n", num_tuples, data_dir);

  // Timing
  cudaEvent_t t_start, t_load, t_query, t_result;
  CUDA_CHECK(cudaEventCreate(&t_start)); CUDA_CHECK(cudaEventCreate(&t_load));
  CUDA_CHECK(cudaEventCreate(&t_query));
  CUDA_CHECK(cudaEventCreate(&t_result)); CUDA_CHECK(cudaEventRecord(t_start));

  // Arena
  { size_t arena_bytes = (size_t)num_tuples * sizeof(int) * 4;
    arena_bytes += arena_bytes / 8;
    fascal_arena_init(arena_bytes); }

  // Load fact columns (orders)
  char path[512];
  #define LOAD_COL(var, name) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    int *var = load_binary_column(path, num_tuples); \
    if (!var) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

  LOAD_COL(h_o_orderkey, "o_orderkey")
  LOAD_COL(h_o_orderdate, "o_orderdate")
  LOAD_COL(h_o_orderpriority_enc, "o_orderpriority_enc")
  #undef LOAD_COL

  // Load correlated subquery columns (lineitem)
  int num_sq1_tuples = 0;
  snprintf(path, sizeof(path), "%s/l_orderkey.bin", data_dir);
  int *h_sq1_l_orderkey = load_binary_column_auto(path, &num_sq1_tuples);
  if (!h_sq1_l_orderkey || num_sq1_tuples == 0) { fprintf(stderr, "Failed to load l_orderkey\n"); return 1; }

  snprintf(path, sizeof(path), "%s/l_receiptdate.bin", data_dir);
  int *h_sq1_l_receiptdate = load_binary_column_auto(path, &num_sq1_tuples);
  if (!h_sq1_l_receiptdate) { fprintf(stderr, "Failed to load l_receiptdate\n"); return 1; }

  snprintf(path, sizeof(path), "%s/l_commitdate.bin", data_dir);
  int *h_sq1_l_commitdate = load_binary_column_auto(path, &num_sq1_tuples);
  if (!h_sq1_l_commitdate) { fprintf(stderr, "Failed to load l_commitdate\n"); return 1; }
  printf("Loaded __correlated_sq_1: %d tuples\n", num_sq1_tuples);

  CUDA_CHECK(cudaEventRecord(t_load));

  // CQO
  auto cqo_in = fascal_cqo_init(num_tuples);
  cqo_in.predicate_selectivities = {
    fascal_estimate_range_sel(h_o_orderdate, num_tuples, 19970701, 19971001),
    0.3  // subquery shipdate-commitdate check
  };
  cqo_in.join_bf_selectivities.push_back(0.05);
  cqo_in.num_gpu_only_ops = 1;
  auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
  fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);

  int in_memory_repeats = 1;
  if (const char *e = getenv("FASCAL_IN_MEMORY_REPEATS")) { in_memory_repeats = atoi(e); if (in_memory_repeats < 1) in_memory_repeats = 1; }
  int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr) || cqo_out.bloom_off;

  // Bloom filter setup
  fascal::runtime::BloomFilter bf_storage[1];
  fascal::runtime::BloomFilter *bf_array[1];
  bf_array[0] = (!bloom_off && (int)0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[0] : nullptr;

  // ODZC mapping
  auto odzc = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));
  uint32_t *h_bm = nullptr, *d_bm = nullptr; uint8_t *h_ts = nullptr, *d_ts = nullptr;
  fascal_alloc_seed_bitmap(num_tuples, &h_bm, &d_bm, &h_ts, &d_ts);

  #define ODZC_MAP(var, host, cnt, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; int *d_##var = nullptr; \
    fascal_odzc_map(odzc.get(), host, cnt, name, &mc_##var, &d_##var);

  ODZC_MAP(o_orderkey, h_o_orderkey, num_tuples, "o_orderkey")
  ODZC_MAP(o_orderdate, h_o_orderdate, num_tuples, "o_orderdate")

  int *d_o_orderpriority_enc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_o_orderpriority_enc, num_tuples * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_o_orderpriority_enc, h_o_orderpriority_enc, num_tuples * sizeof(int), cudaMemcpyHostToDevice));

  // ODZC map subquery columns
  fascal::runtime::ODZCManager::MappedColumn mc_sq1_l_orderkey;
  mc_sq1_l_orderkey = odzc->register_column(h_sq1_l_orderkey, num_sq1_tuples * sizeof(int));
  int *d_sq1_l_orderkey = static_cast<int*>(mc_sq1_l_orderkey.device_ptr);
  if (!d_sq1_l_orderkey) {
    fprintf(stderr, "ODZC failed for sq1.l_orderkey, fallback to cudaMalloc\n");
    CUDA_CHECK(cudaMalloc(&d_sq1_l_orderkey, num_sq1_tuples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sq1_l_orderkey, h_sq1_l_orderkey, num_sq1_tuples * sizeof(int), cudaMemcpyHostToDevice));
  }
  fascal::runtime::ODZCManager::MappedColumn mc_sq1_l_receiptdate;
  mc_sq1_l_receiptdate = odzc->register_column(h_sq1_l_receiptdate, num_sq1_tuples * sizeof(int));
  int *d_sq1_l_receiptdate = static_cast<int*>(mc_sq1_l_receiptdate.device_ptr);
  if (!d_sq1_l_receiptdate) {
    fprintf(stderr, "ODZC failed for sq1.l_receiptdate, fallback to cudaMalloc\n");
    CUDA_CHECK(cudaMalloc(&d_sq1_l_receiptdate, num_sq1_tuples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sq1_l_receiptdate, h_sq1_l_receiptdate, num_sq1_tuples * sizeof(int), cudaMemcpyHostToDevice));
  }
  fascal::runtime::ODZCManager::MappedColumn mc_sq1_l_commitdate;
  mc_sq1_l_commitdate = odzc->register_column(h_sq1_l_commitdate, num_sq1_tuples * sizeof(int));
  int *d_sq1_l_commitdate = static_cast<int*>(mc_sq1_l_commitdate.device_ptr);
  if (!d_sq1_l_commitdate) {
    fprintf(stderr, "ODZC failed for sq1.l_commitdate, fallback to cudaMalloc\n");
    CUDA_CHECK(cudaMalloc(&d_sq1_l_commitdate, num_sq1_tuples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sq1_l_commitdate, h_sq1_l_commitdate, num_sq1_tuples * sizeof(int), cudaMemcpyHostToDevice));
  }
  #undef ODZC_MAP

  CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

  // Aggregation table (5 groups * 1 aggregate)
  size_t num_groups = 5ULL;
  unsigned long long *d_agg;
  CUDA_CHECK(cudaMalloc(&d_agg, num_groups * NUM_AGGREGATES * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * NUM_AGGREGATES * sizeof(unsigned long long)));

  // Dense bitmap: one bit per orderkey in [0, max_orderkey].
  // Set during the build kernel by lineitem rows where l_receiptdate >
  // l_commitdate. Probed in the result kernel via a single word load + bit
  // test. At SF=100 max orderkey ≈ 600M → bitmap ≈ 75 MB on device, vs the
  // previous HT (~9.6 GB) that fell back to zero-copy mapped host memory.
  int max_orderkey = 0;
  for (int i = 0; i < num_sq1_tuples; ++i) {
    if (h_sq1_l_orderkey[i] > max_orderkey) max_orderkey = h_sq1_l_orderkey[i];
  }
  for (int i = 0; i < num_tuples; ++i) {
    if (h_o_orderkey[i] > max_orderkey) max_orderkey = h_o_orderkey[i];
  }
  size_t late_bitmap_words_count = (static_cast<size_t>(max_orderkey) + 31u) / 32u + 1u;
  unsigned int *d_late_bitmap_words = nullptr;
  CUDA_CHECK(cudaMalloc(&d_late_bitmap_words, late_bitmap_words_count * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(d_late_bitmap_words, 0, late_bitmap_words_count * sizeof(unsigned int)));

  // Subquery AFP prefilter bitmap
  size_t sq1_bm_words = ((size_t)num_sq1_tuples + 31) / 32;
  size_t sq1_num_tiles = ((size_t)num_sq1_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
  uint32_t *h_sq1_pf = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_sq1_pf, sq1_bm_words * sizeof(uint32_t), cudaHostAllocMapped));
  memset(h_sq1_pf, 0, sq1_bm_words * sizeof(uint32_t));
  uint32_t *d_sq1_pf = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&d_sq1_pf, h_sq1_pf, 0));
  uint8_t *h_sq1_ts = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_sq1_ts, sq1_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
  memset(h_sq1_ts, 0, sq1_num_tiles);
  uint8_t *d_sq1_ts = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&d_sq1_ts, h_sq1_ts, 0));

  // Pipeline setup
  int *h_data[] = {h_o_orderkey, h_o_orderdate};
  const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
  const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
  cudaStream_t *streams = new cudaStream_t[N_STREAMS];
  for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
  int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

  auto run_kernel = [&]() {
    auto prefilter = [&](int bc, int bo, int tt) {
      fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
        [&](int off, int cnt) { cpu_predicate(h_data, off, cnt, h_bm, bf_array, 1); }, bo, tt);
    };
    auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
      tpch_q4_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
        nt, bo, d_o_orderkey, d_o_orderdate, d_o_orderpriority_enc,
        d_bm, d_ts,
        d_late_bitmap_words, max_orderkey,
        d_agg);
    };
      int nb = (num_tuples + GPU_BATCH - 1) / GPU_BATCH;
      for (int b = 0; b < nb; ++b) {
        int bo = b * GPU_BATCH, bc = std::min(GPU_BATCH, num_tuples - bo);
        prefilter(bc, bo, num_tuples);
        launch(num_tuples, bo, (bc + TILE_SIZE - 1) / TILE_SIZE, 0);
      }
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  };

  // Execute: build subquery HT+BF first
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
  cudaEventRecord(t0);

  cpu_prefilter_sq1(h_sq1_l_orderkey, h_sq1_l_receiptdate, h_sq1_l_commitdate,
                    num_sq1_tuples, h_sq1_pf, h_sq1_ts);
  // Build late-orderkey bitmap (morsel-batched; all morsels write into same shared bitmap)
  {
    int _build_nb = (num_sq1_tuples + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_sq1_tuples - _bo);
      int _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      tpch_q4_build_sq1<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
          num_sq1_tuples, _bo, d_sq1_l_orderkey, d_sq1_l_receiptdate, d_sq1_l_commitdate,
          d_late_bitmap_words, max_orderkey,
          d_sq1_pf, d_sq1_ts);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }

  // Result pipeline
  run_kernel();
  cudaEventRecord(t1);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaEventRecord(t_query));
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  printf("Kernel: %.3f ms\n", ms);

  // In-memory repeats
  if (in_memory_repeats > 1) {
    float best = 1e9f, total = 0;
    for (int r = 0; r < in_memory_repeats; ++r) {
      memset(h_bm, 0, ((size_t)(num_tuples + 31) / 32) * 4);
      memset(h_ts, 0, (num_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE);
      CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * NUM_AGGREGATES * sizeof(unsigned long long)));
      CUDA_CHECK(cudaEventRecord(t0)); run_kernel(); CUDA_CHECK(cudaEventRecord(t1));
      CUDA_CHECK(cudaDeviceSynchronize());
      float m; CUDA_CHECK(cudaEventElapsedTime(&m, t0, t1));
      total += m; if (m < best) best = m;
    }
    printf("In-memory result kernel: best %.3f ms avg %.3f ms (%d runs)\n", best, total / in_memory_repeats, in_memory_repeats);
  }

  // Output: decode GROUP BY keys and print results
  unsigned long long *h_agg = (unsigned long long*)malloc(num_groups * NUM_AGGREGATES * sizeof(unsigned long long));
  CUDA_CHECK(cudaMemcpy(h_agg, d_agg, num_groups * NUM_AGGREGATES * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  for (size_t k = 0; k < num_groups; k++) {
    if (h_agg[k * NUM_AGGREGATES + 0] == 0) continue;
    static const char* priority_vals[] = { "1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW" };
    printf("ROW: %s|%lld\n", priority_vals[k], (long long)h_agg[k * NUM_AGGREGATES + 0]);
  }
  free(h_agg);

  // Timing report
  CUDA_CHECK(cudaFree(d_agg));
  CUDA_CHECK(cudaEventRecord(t_result)); CUDA_CHECK(cudaDeviceSynchronize());
  { float ml, mb, mr;
    CUDA_CHECK(cudaEventElapsedTime(&ml, t_start, t_load));
    CUDA_CHECK(cudaEventElapsedTime(&mb, t_load, t_query));
    /* merged into query */
    CUDA_CHECK(cudaEventElapsedTime(&mr, t_query, t_result));
    printf("Timing: load=%.1fms query=%.1fms result=%.1fms total=%.1fms\n", ml, mb, mr, ml+mb+mr); }

  // Cleanup
  odzc->unregister_column(mc_o_orderkey);
  odzc->unregister_column(mc_o_orderdate);
  CUDA_CHECK(cudaFree(d_o_orderpriority_enc));
  odzc->unregister_column(mc_sq1_l_orderkey);
  odzc->unregister_column(mc_sq1_l_receiptdate);
  odzc->unregister_column(mc_sq1_l_commitdate);
  CUDA_CHECK(cudaFreeHost(h_sq1_ts));
  CUDA_CHECK(cudaFreeHost(h_sq1_pf));
  CUDA_CHECK(cudaFree(d_late_bitmap_words));
  fascal_free_column(h_sq1_l_orderkey);
  fascal_free_column(h_sq1_l_receiptdate);
  fascal_free_column(h_sq1_l_commitdate);
  fascal_free_seed_bitmap(h_bm, h_ts);
  fascal_free_column(h_o_orderkey);
  fascal_free_column(h_o_orderdate);
  fascal_free_column(h_o_orderpriority_enc);
  fascal_arena_destroy();
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  delete[] streams;
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(t_start); cudaEventDestroy(t_load);   cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  return 0;
}
