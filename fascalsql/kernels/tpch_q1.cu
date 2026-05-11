// TPC-H Q1 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
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
#define NUM_AGGREGATES 8

__device__ __constant__ unsigned char d_pred_flags[MAX_PREDICATES];
static unsigned char h_pred_flags[MAX_PREDICATES] = {0};

// =============================================================================
// GPU Kernel: scan lineitem, apply predicates, GROUP BY aggregate
// =============================================================================
template <int BT, int IPT>
__global__ void tpch_q1_kernel(
    int num_tuples, int batch_offset,
    int *d_l_shipdate,
    float *d_l_extendedprice, float *d_l_discount, float *d_l_tax, float *d_l_quantity,
    int *d_l_returnflag_enc, int *d_l_linestatus_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int l_shipdate[IPT];
  float l_extendedprice[IPT], l_discount[IPT], l_tax[IPT], l_quantity[IPT];
  int l_returnflag_enc[IPT], l_linestatus_enc[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // Pred 0: l_shipdate <= 19980921
    if (d_pred_flags[0] == 0) {
      BlockLoadSelect<int, BT, IPT>(d_l_shipdate + tile_offset, l_shipdate, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BT, IPT>(l_shipdate, 19980921, selection_flags, num_tile_items);
    }

    FASCAL_CHECK_ALIVE(IPT);

    // Load GROUP BY keys
    BlockLoadSelect<int, BT, IPT>(d_l_returnflag_enc + tile_offset, l_returnflag_enc, selection_flags, num_tile_items);
    BlockLoadSelect<int, BT, IPT>(d_l_linestatus_enc + tile_offset, l_linestatus_enc, selection_flags, num_tile_items);

    // Load aggregation columns
    BlockLoadSelect<float, BT, IPT>(d_l_extendedprice + tile_offset, l_extendedprice, selection_flags, num_tile_items);
    BlockLoadSelect<float, BT, IPT>(d_l_discount + tile_offset, l_discount, selection_flags, num_tile_items);
    BlockLoadSelect<float, BT, IPT>(d_l_tax + tile_offset, l_tax, selection_flags, num_tile_items);
    BlockLoadSelect<float, BT, IPT>(d_l_quantity + tile_offset, l_quantity, selection_flags, num_tile_items);

    // GROUP BY direct-index aggregation
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i]) {
        int gk = l_returnflag_enc[i] + (l_linestatus_enc[i] * 128);
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], (double)l_quantity[i]);
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 1], (double)l_extendedprice[i]);
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 2], (double)l_extendedprice[i] * (1 - l_discount[i]));
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 3], (double)l_extendedprice[i] * (1 - l_discount[i]) * (1 + l_tax[i]));
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 4], (double)l_quantity[i]);
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 5], (double)l_extendedprice[i]);
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 6], (double)l_discount[i]);
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 7], 1LL);
      }
    }
  } while(0);
}

// =============================================================================
// CPU Predicate: AVX2 bitmap filtering
// =============================================================================
static void cpu_predicate(int *h_data[], int offset, int cnt, uint32_t *bitmap) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);

  int *l_shipdate = h_data[0];

  // Pass 0: l_shipdate <= 19980921
  if (h_pred_flags[0] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, l_shipdate, offset, cnt, 0, 19980921, false, true);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
  const char *data_dir = getenv("FASCALSQL_DATA_DIR");
  if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
  if (argc >= 3) data_dir = argv[2];
  if (!data_dir) { fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n"); return 1; }

  printf("=== FaScalSQL: tpch_q1 ===\n");
  if (argc < 3) { printf("Usage: %s <num_tuples> <data_dir>\n", argv[0]); return 1; }
  data_dir = argv[2];

  // Auto-detect num_tuples
  int num_tuples = 0;
  { char p[512]; snprintf(p, sizeof(p), "%s/l_shipdate.bin", data_dir);
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
  fascal_arena_init((size_t)num_tuples * sizeof(int) * 10);

  // Load fact columns
  char path[512];
  #define LOAD_COL(var, name) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    int *var = load_binary_column(path, num_tuples); \
    if (!var) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

  LOAD_COL(h_l_shipdate, "l_shipdate")
  LOAD_COL(h_l_extendedprice, "l_extendedprice")
  LOAD_COL(h_l_discount, "l_discount")
  LOAD_COL(h_l_tax, "l_tax")
  LOAD_COL(h_l_quantity, "l_quantity")
  LOAD_COL(h_l_returnflag_enc, "l_returnflag_enc")
  LOAD_COL(h_l_linestatus_enc, "l_linestatus_enc")
  #undef LOAD_COL
  CUDA_CHECK(cudaEventRecord(t_load));

  // CQO
  auto cqo_in = fascal_cqo_init(num_tuples);
  cqo_in.predicate_selectivities = {
    fascal_estimate_selectivity(h_l_shipdate, num_tuples, [](int v){ return v <= 19980921; })
  };
  cqo_in.num_gpu_only_ops = 6;
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

  ODZC_MAP(l_shipdate, h_l_shipdate, "l_shipdate")

  // Float columns: manual ODZC mapping
  fascal::runtime::ODZCManager::MappedColumn mc_l_extendedprice;
  float *d_l_extendedprice = nullptr;
  { mc_l_extendedprice = odzc->register_column(h_l_extendedprice, (size_t)num_tuples * sizeof(float));
    d_l_extendedprice = static_cast<float*>(mc_l_extendedprice.device_ptr);
    if (!d_l_extendedprice) {
      fprintf(stderr, "ODZC failed for l_extendedprice, fallback to cudaMalloc\n");
      CUDA_CHECK(cudaMalloc(&d_l_extendedprice, (size_t)num_tuples * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_l_extendedprice, h_l_extendedprice, (size_t)num_tuples * sizeof(float), cudaMemcpyHostToDevice));
    }
  }
  fascal::runtime::ODZCManager::MappedColumn mc_l_discount;
  float *d_l_discount = nullptr;
  { mc_l_discount = odzc->register_column(h_l_discount, (size_t)num_tuples * sizeof(float));
    d_l_discount = static_cast<float*>(mc_l_discount.device_ptr);
    if (!d_l_discount) {
      fprintf(stderr, "ODZC failed for l_discount, fallback to cudaMalloc\n");
      CUDA_CHECK(cudaMalloc(&d_l_discount, (size_t)num_tuples * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_l_discount, h_l_discount, (size_t)num_tuples * sizeof(float), cudaMemcpyHostToDevice));
    }
  }
  fascal::runtime::ODZCManager::MappedColumn mc_l_tax;
  float *d_l_tax = nullptr;
  { mc_l_tax = odzc->register_column(h_l_tax, (size_t)num_tuples * sizeof(float));
    d_l_tax = static_cast<float*>(mc_l_tax.device_ptr);
    if (!d_l_tax) {
      fprintf(stderr, "ODZC failed for l_tax, fallback to cudaMalloc\n");
      CUDA_CHECK(cudaMalloc(&d_l_tax, (size_t)num_tuples * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_l_tax, h_l_tax, (size_t)num_tuples * sizeof(float), cudaMemcpyHostToDevice));
    }
  }
  fascal::runtime::ODZCManager::MappedColumn mc_l_quantity;
  float *d_l_quantity = nullptr;
  { mc_l_quantity = odzc->register_column(h_l_quantity, (size_t)num_tuples * sizeof(float));
    d_l_quantity = static_cast<float*>(mc_l_quantity.device_ptr);
    if (!d_l_quantity) {
      fprintf(stderr, "ODZC failed for l_quantity, fallback to cudaMalloc\n");
      CUDA_CHECK(cudaMalloc(&d_l_quantity, (size_t)num_tuples * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_l_quantity, h_l_quantity, (size_t)num_tuples * sizeof(float), cudaMemcpyHostToDevice));
    }
  }

  // Enc columns: direct cudaMalloc (small cardinality, always loaded)
  int *d_l_returnflag_enc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_l_returnflag_enc, num_tuples * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_l_returnflag_enc, h_l_returnflag_enc, num_tuples * sizeof(int), cudaMemcpyHostToDevice));

  int *d_l_linestatus_enc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_l_linestatus_enc, num_tuples * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_l_linestatus_enc, h_l_linestatus_enc, num_tuples * sizeof(int), cudaMemcpyHostToDevice));

  #undef ODZC_MAP

  CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

  // Aggregation table (128 * 128 groups * 8 aggregates)
  size_t num_groups = 128ULL * 128ULL;
  double *d_agg;
  CUDA_CHECK(cudaMalloc(&d_agg, num_groups * NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * NUM_AGGREGATES * sizeof(double)));

  // Pipeline setup
  int *h_data[] = {h_l_shipdate, (int*)h_l_extendedprice, (int*)h_l_discount, (int*)h_l_tax, (int*)h_l_quantity};
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
      tpch_q1_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
        nt, bo, d_l_shipdate,
        d_l_extendedprice, d_l_discount, d_l_tax, d_l_quantity,
        d_l_returnflag_enc, d_l_linestatus_enc,
        d_bm, d_ts, d_agg);
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
      CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * NUM_AGGREGATES * sizeof(double)));
      CUDA_CHECK(cudaEventRecord(t0)); run_kernel(); CUDA_CHECK(cudaEventRecord(t1));
      CUDA_CHECK(cudaDeviceSynchronize());
      float m; CUDA_CHECK(cudaEventElapsedTime(&m, t0, t1));
      total += m; if (m < best) best = m;
    }
    printf("In-memory result kernel: best %.3f ms avg %.3f ms (%d runs)\n", best, total / in_memory_repeats, in_memory_repeats);
  }

  // Output: decode GROUP BY keys and print results
  double *h_agg = (double*)malloc(num_groups * NUM_AGGREGATES * sizeof(double));
  CUDA_CHECK(cudaMemcpy(h_agg, d_agg, num_groups * NUM_AGGREGATES * sizeof(double), cudaMemcpyDeviceToHost));
  for (size_t k = 0; k < num_groups; k++) {
    int any = 0;
    for (int a = 0; a < NUM_AGGREGATES; ++a) if (h_agg[k * NUM_AGGREGATES + a] != 0) { any = 1; break; }
    if (!any) continue;
    size_t gk = k;
    int key0 = (int)(gk % 128ULL); gk /= 128;
    int key1 = (int)(gk % 128ULL);
    // Decode l_returnflag
    static const int rf_keys[] = { 0, 1, 2 };
    static const char* rf_vals[] = { "A", "N", "R" };
    const char *rf = "UNKNOWN";
    for (int i = 0; i < 3; ++i) if (rf_keys[i] == key0) { rf = rf_vals[i]; break; }
    // Decode l_linestatus
    static const int ls_keys[] = { 0, 1 };
    static const char* ls_vals[] = { "F", "O" };
    const char *ls = "UNKNOWN";
    for (int i = 0; i < 2; ++i) if (ls_keys[i] == key1) { ls = ls_vals[i]; break; }

    printf("ROW: %s|%s", rf, ls);
    printf("|%.6f", h_agg[k * NUM_AGGREGATES + 0]);  // sum_qty
    printf("|%.6f", h_agg[k * NUM_AGGREGATES + 1]);  // sum_base_price
    printf("|%.4f",  h_agg[k * NUM_AGGREGATES + 2]);  // sum_disc_price
    printf("|%.6f", h_agg[k * NUM_AGGREGATES + 3]);  // sum_charge
    double cnt = h_agg[k * NUM_AGGREGATES + 7];
    printf("|%.6f", cnt == 0 ? 0 : h_agg[k * NUM_AGGREGATES + 4] / cnt);  // avg_qty
    printf("|%.6f", cnt == 0 ? 0 : h_agg[k * NUM_AGGREGATES + 5] / cnt);  // avg_price
    printf("|%.6f", cnt == 0 ? 0 : h_agg[k * NUM_AGGREGATES + 6] / cnt);  // avg_disc
    printf("|%lld\n", (long long)cnt);  // count_order
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
  odzc->unregister_column(mc_l_shipdate);
  odzc->unregister_column(mc_l_extendedprice);
  odzc->unregister_column(mc_l_discount);
  odzc->unregister_column(mc_l_tax);
  odzc->unregister_column(mc_l_quantity);
  CUDA_CHECK(cudaFree(d_l_returnflag_enc));
  CUDA_CHECK(cudaFree(d_l_linestatus_enc));
  fascal_free_seed_bitmap(h_bm, h_ts);
  fascal_free_column(h_l_shipdate);       fascal_free_column(h_l_extendedprice);
  fascal_free_column(h_l_discount);       fascal_free_column(h_l_tax);
  fascal_free_column(h_l_quantity);       fascal_free_column(h_l_returnflag_enc);
  fascal_free_column(h_l_linestatus_enc);
  fascal_arena_destroy();
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  delete[] streams;
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(t_start); cudaEventDestroy(t_load);   cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  return 0;
}
