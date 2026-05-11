// SSB Q1.3 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Refactored: uses shared macros and CPU predicate helpers.

#define CUB_STDERR
#define FASCAL_TILE_SIZE 1024

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
#define ITEMS_PER_THREAD 8
#define TILE_SIZE 1024
#define MAX_PREDICATES 16
#define NUM_AGGREGATES 1

__device__ __constant__ unsigned char d_pred_flags[MAX_PREDICATES];
static unsigned char h_pred_flags[MAX_PREDICATES] = {0};

// =============================================================================
// GPU Kernel: scan lineorder, apply predicates, aggregate
// =============================================================================
template <int BT, int IPT>
__global__ void ssb_q13_kernel(
    int num_tuples, int batch_offset,
    int *d_lo_orderdate, int *d_lo_discount, int *d_lo_quantity, int *d_lo_extendedprice,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    unsigned long long *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int lo_discount[IPT], lo_quantity[IPT], lo_orderdate[IPT], lo_extendedprice[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // Pred 0: lo_discount BETWEEN 5 AND 7
    BlockLoadSelect<int, BT, IPT>(d_lo_discount + tile_offset, lo_discount, selection_flags, num_tile_items);
    if (d_pred_flags[0] == 0) {
      BlockPredAndGTE<int, BT, IPT>(lo_discount, 5, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BT, IPT>(lo_discount, 7, selection_flags, num_tile_items);
    }

    // Pred 1: lo_quantity BETWEEN 26 AND 35
    if (d_pred_flags[1] == 0) {
      BlockLoadSelect<int, BT, IPT>(d_lo_quantity + tile_offset, lo_quantity, selection_flags, num_tile_items);
      BlockPredAndGTE<int, BT, IPT>(lo_quantity, 26, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BT, IPT>(lo_quantity, 35, selection_flags, num_tile_items);
    }

    // Pred 2: lo_orderdate BETWEEN 19940204 AND 19940210 (replaces date dim join)
    if (d_pred_flags[2] == 0) {
      BlockLoadSelect<int, BT, IPT>(d_lo_orderdate + tile_offset, lo_orderdate, selection_flags, num_tile_items);
      BlockPredAndGTE<int, BT, IPT>(lo_orderdate, 19940204, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BT, IPT>(lo_orderdate, 19940210, selection_flags, num_tile_items);
    }

    FASCAL_CHECK_ALIVE(IPT);

    // Load aggregation column
    BlockLoadSelect<int, BT, IPT>(d_lo_extendedprice + tile_offset, lo_extendedprice, selection_flags, num_tile_items);
  } while(0);

  // Block-level reduction → 1 atomicAdd per block
  long long acc = 0;
  #pragma unroll
  for (int i = 0; i < IPT; ++i)
    if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i])
      acc += (long long)lo_extendedprice[i] * lo_discount[i];
  unsigned long long uacc = blockReduceSum<BT>((unsigned long long)acc);
  if (threadIdx.x == 0) atomicAdd(&aggtable[0], uacc);
}

// =============================================================================
// CPU Predicate: AVX2 multi-pass bitmap filtering
// =============================================================================
static void cpu_predicate(int *h_data[], int offset, int cnt, uint32_t *bitmap) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);

  int *lo_orderdate = h_data[0], *lo_discount = h_data[1], *lo_quantity = h_data[2];

  if (h_pred_flags[0] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, lo_discount, offset, cnt, 5, 7, true, true);
  if (h_pred_flags[1] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, lo_quantity, offset, cnt, 26, 35, true, true);
  if (h_pred_flags[2] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, lo_orderdate, offset, cnt, 19940204, 19940210, true, true);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
  const char *data_dir = getenv("FASCALSQL_DATA_DIR");
  if (!data_dir) data_dir = getenv("FASCALSQL_SSB_DATA_DIR");
  if (argc >= 3) data_dir = argv[2];
  if (!data_dir) { fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n"); return 1; }

  printf("=== FaScalSQL: ssb_q13 ===\n");
  if (argc < 3) { printf("Usage: %s <num_tuples> <data_dir>\n", argv[0]); return 1; }
  data_dir = argv[2];

  // Auto-detect num_tuples
  int num_tuples = 0;
  { char p[512]; snprintf(p, sizeof(p), "%s/lo_orderdate.bin", data_dir);
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
  fascal_arena_init((size_t)num_tuples * sizeof(int) * 5);

  // Load fact columns
  char path[512];
  #define LOAD_COL(var, name) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    int *var = load_binary_column(path, num_tuples); \
    if (!var) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

  LOAD_COL(h_lo_orderdate, "lo_orderdate")
  LOAD_COL(h_lo_discount, "lo_discount")
  LOAD_COL(h_lo_quantity, "lo_quantity")
  LOAD_COL(h_lo_extendedprice, "lo_extendedprice")
  #undef LOAD_COL
  CUDA_CHECK(cudaEventRecord(t_load));

  // CQO
  auto cqo_in = fascal_cqo_init(num_tuples);
  cqo_in.predicate_selectivities = {
    fascal_estimate_range_sel(h_lo_discount, num_tuples, 5, 7),
    fascal_estimate_range_sel(h_lo_quantity, num_tuples, 26, 35),
    fascal_estimate_range_sel(h_lo_orderdate, num_tuples, 19940204, 19940210)
  };
  cqo_in.num_gpu_only_ops = 1;
  auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
  fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);

  int in_memory_repeats = 1;
  if (const char *e = getenv("FASCAL_IN_MEMORY_REPEATS")) { in_memory_repeats = atoi(e); if (in_memory_repeats < 1) in_memory_repeats = 1; }

  // ODZC mapping
  auto odzc = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));
  uint32_t *h_bm = nullptr, *d_bm = nullptr; uint8_t *h_ts = nullptr, *d_ts = nullptr;
  fascal_alloc_seed_bitmap(num_tuples, &h_bm, &d_bm, &h_ts, &d_ts);

  #define ODZC_MAP(var, host, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; int *d_##var = nullptr; \
    fascal_odzc_map(odzc.get(), host, num_tuples, name, &mc_##var, &d_##var);

  ODZC_MAP(lo_orderdate, h_lo_orderdate, "lo_orderdate")
  ODZC_MAP(lo_discount, h_lo_discount, "lo_discount")
  ODZC_MAP(lo_quantity, h_lo_quantity, "lo_quantity")
  ODZC_MAP(lo_extendedprice, h_lo_extendedprice, "lo_extendedprice")
  #undef ODZC_MAP

  CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

  // Aggregation table
  unsigned long long *d_agg;
  CUDA_CHECK(cudaMalloc(&d_agg, NUM_AGGREGATES * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_agg, 0, NUM_AGGREGATES * sizeof(unsigned long long)));

  // Pipeline setup
  int *h_data[] = {h_lo_orderdate, h_lo_discount, h_lo_quantity, h_lo_extendedprice};
  const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
  const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
  cudaStream_t *streams = new cudaStream_t[N_STREAMS];
  for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
  int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

  auto run_kernel = [&]() {
    auto prefilter = [&](int bc, int bo, int tt) {
      fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
        [&](int off, int cnt) { cpu_predicate(h_data, off, cnt, h_bm); }, bo, tt);
    };
    auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
      ssb_q13_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
        nt, bo, d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, d_bm, d_ts, d_agg);
    };
      int nb = (num_tuples + GPU_BATCH - 1) / GPU_BATCH;
      for (int b = 0; b < nb; ++b) {
        int bo = b * GPU_BATCH, bc = std::min(GPU_BATCH, num_tuples - bo);
        prefilter(bc, bo, num_tuples);
        launch(num_tuples, bo, (bc + TILE_SIZE - 1) / TILE_SIZE, 0);
      }
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  };

  // Execute
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
  cudaEventRecord(t0);
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
      CUDA_CHECK(cudaMemset(d_agg, 0, NUM_AGGREGATES * sizeof(unsigned long long)));
      CUDA_CHECK(cudaEventRecord(t0)); run_kernel(); CUDA_CHECK(cudaEventRecord(t1));
      CUDA_CHECK(cudaDeviceSynchronize());
      float m; CUDA_CHECK(cudaEventElapsedTime(&m, t0, t1));
      total += m; if (m < best) best = m;
    }
    printf("In-memory result kernel: best %.3f ms avg %.3f ms (%d runs)\n", best, total / in_memory_repeats, in_memory_repeats);
  }

  // Output
  unsigned long long h_res[NUM_AGGREGATES];
  CUDA_CHECK(cudaMemcpy(h_res, d_agg, sizeof(h_res), cudaMemcpyDeviceToHost));
  printf("ROW: %llu\n", h_res[0]);

  // Timing report
  CUDA_CHECK(cudaEventRecord(t_result)); CUDA_CHECK(cudaDeviceSynchronize());
  { float ml, mb, mr;
    CUDA_CHECK(cudaEventElapsedTime(&ml, t_start, t_load));
    CUDA_CHECK(cudaEventElapsedTime(&mb, t_load, t_query));
    /* merged into query */
    CUDA_CHECK(cudaEventElapsedTime(&mr, t_query, t_result));
    printf("Timing: load=%.1fms query=%.1fms result=%.1fms total=%.1fms\n", ml, mb, mr, ml+mb+mr); }

  // Cleanup
  odzc->unregister_column(mc_lo_orderdate); odzc->unregister_column(mc_lo_discount);
  odzc->unregister_column(mc_lo_quantity);  odzc->unregister_column(mc_lo_extendedprice);
  fascal_free_seed_bitmap(h_bm, h_ts);
  fascal_free_column(h_lo_orderdate); fascal_free_column(h_lo_discount);
  fascal_free_column(h_lo_quantity);  fascal_free_column(h_lo_extendedprice);
  CUDA_CHECK(cudaFree(d_agg));
  fascal_arena_destroy();
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  delete[] streams;
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(t_start); cudaEventDestroy(t_load);   cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  return 0;
}
