// SSB Q3.2 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Refactored: uses shared macros and CPU predicate helpers.
// Dimensions: customer (c_nation=23/US -> c_city_enc), supplier (s_nation=23/US -> s_city_enc)
// Date filter: lo_orderdate BETWEEN 19920101 AND 19971231 (replaces date dim join)
// GROUP BY: c_city_enc, s_city_enc, year (inline lo_orderdate/10000-1992)

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
// GPU Kernel: scan lineorder, BF probes, date pred, HT joins, GROUP BY agg
// =============================================================================
template <int BT, int IPT>
__global__ void ssb_q32_kernel(
    int num_tuples, int batch_offset,
    int *d_lo_custkey, int *d_lo_suppkey, int *d_lo_orderdate, int *d_lo_revenue,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_customer, uint32_t bloom_bits_customer,
    HtEntry *ht_customer, uint32_t ht_size_customer,
    const uint64_t *__restrict__ d_bloom_supplier, uint32_t bloom_bits_supplier,
    HtEntry *ht_supplier, uint32_t ht_size_supplier,
    unsigned long long *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int lo_custkey[IPT], lo_suppkey[IPT], lo_orderdate[IPT], lo_revenue[IPT];
  int c_city_enc[IPT], s_city_enc[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // BF probe: customer
    BlockLoadSelect<int, BT, IPT>(d_lo_custkey + tile_offset, lo_custkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(lo_custkey, selection_flags, num_tile_items,
        bloom_bits_customer, d_bloom_customer, 3);

    // BF probe: supplier
    BlockLoadSelect<int, BT, IPT>(d_lo_suppkey + tile_offset, lo_suppkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(lo_suppkey, selection_flags, num_tile_items,
        bloom_bits_supplier, d_bloom_supplier, 3);

    // Pred 0: lo_orderdate BETWEEN 19920101 AND 19971231
    if (d_pred_flags[0] == 0) {
      BlockLoadSelect<int, BT, IPT>(d_lo_orderdate + tile_offset, lo_orderdate, selection_flags, num_tile_items);
      BlockPredAndGTE<int, BT, IPT>(lo_orderdate, 19920101, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BT, IPT>(lo_orderdate, 19971231, selection_flags, num_tile_items);
    }

    FASCAL_CHECK_ALIVE(IPT);

    // HT join: customer -> c_city_enc
    BlockJoinProbePayloadDirect<BT, IPT>(lo_custkey, selection_flags, num_tile_items,
        ht_customer, ht_size_customer, 1, c_city_enc);

    // HT join: supplier -> s_city_enc
    BlockJoinProbePayloadDirect<BT, IPT>(lo_suppkey, selection_flags, num_tile_items,
        ht_supplier, ht_size_supplier, 1, s_city_enc);

    // Ensure lo_orderdate loaded (if CPU handled date pred)
    if (d_pred_flags[0] != 0)
      BlockLoadSelect<int, BT, IPT>(d_lo_orderdate + tile_offset, lo_orderdate, selection_flags, num_tile_items);

    // Load revenue
    BlockLoadSelect<int, BT, IPT>(d_lo_revenue + tile_offset, lo_revenue, selection_flags, num_tile_items);

    // GROUP BY: c_city_enc + s_city_enc*250 + year*62500
    #pragma unroll
    for (int i = 0; i < IPT; ++i)
      if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i]) {
        int gk = c_city_enc[i] + s_city_enc[i] * 250 + (lo_orderdate[i] / 10000 - 1992) * 62500;
        atomicAdd(&aggtable[gk], (unsigned long long)(long long)lo_revenue[i]);
      }
  } while(0);
}

// =============================================================================
// Build kernel: customer (c_nation == 23 -> UNITED STATES)
// =============================================================================
template <int BT, int IPT>
__global__ void build_customer(
    int num_tuples,
    const int *__restrict__ d_c_custkey, const int *__restrict__ d_c_nation,
    const int *__restrict__ d_c_city_enc,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int key[IPT], nation[IPT], payload[IPT];

  BlockLoadSelect<int, BT, IPT>((int*)(d_c_custkey + tile_offset), key, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_c_nation + tile_offset), nation, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(nation, 23, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_c_city_enc + tile_offset), payload, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    gpu_ht_insert_one_direct(key[i], payload[i], 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(key[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// Build kernel: supplier (s_nation == 23 -> UNITED STATES)
// =============================================================================
template <int BT, int IPT>
__global__ void build_supplier(
    int num_tuples,
    const int *__restrict__ d_s_suppkey, const int *__restrict__ d_s_nation,
    const int *__restrict__ d_s_city_enc,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int key[IPT], nation[IPT], payload[IPT];

  BlockLoadSelect<int, BT, IPT>((int*)(d_s_suppkey + tile_offset), key, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_nation + tile_offset), nation, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(nation, 23, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_city_enc + tile_offset), payload, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    gpu_ht_insert_one_direct(key[i], payload[i], 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(key[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// CPU Predicate: date range + 2 BF probes
// =============================================================================
static void cpu_predicate(int *h_data[], int offset, int cnt, uint32_t *bitmap,
                          fascal::runtime::BloomFilter *bf_cust, fascal::runtime::BloomFilter *bf_supp) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);

  int *lo_custkey = h_data[0], *lo_suppkey = h_data[1], *lo_orderdate = h_data[2];

  if (h_pred_flags[0] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, lo_orderdate, offset, cnt, 19920101, 19971231, true, true);
  fascal::cpu_pred::bloom_probe_pass(bm, bs, lo_custkey, offset, cnt, bf_cust);
  fascal::cpu_pred::bloom_probe_pass(bm, bs, lo_suppkey, offset, cnt, bf_supp);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
  const char *data_dir = getenv("FASCALSQL_DATA_DIR");
  if (!data_dir) data_dir = getenv("FASCALSQL_SSB_DATA_DIR");
  if (argc >= 3) data_dir = argv[2];
  if (!data_dir) { fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n"); return 1; }

  printf("=== FaScalSQL: ssb_q32 ===\n");
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

  LOAD_COL(h_lo_custkey, "lo_custkey")
  LOAD_COL(h_lo_suppkey, "lo_suppkey")
  LOAD_COL(h_lo_orderdate, "lo_orderdate")
  LOAD_COL(h_lo_revenue, "lo_revenue")
  #undef LOAD_COL

  // Load dimension: customer
  int num_cust = 0;
  #define LOAD_DIM(var, name, cnt) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    int *var = load_binary_column_auto(path, &cnt); \
    if (!var || cnt == 0) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

  LOAD_DIM(h_c_custkey, "c_custkey", num_cust)
  LOAD_DIM(h_c_nation, "c_nation", num_cust)
  LOAD_DIM(h_c_city_enc, "c_city_enc", num_cust)
  printf("Loaded customer: %d tuples\n", num_cust);

  // Load dimension: supplier
  int num_supp = 0;
  LOAD_DIM(h_s_suppkey, "s_suppkey", num_supp)
  LOAD_DIM(h_s_nation, "s_nation", num_supp)
  LOAD_DIM(h_s_city_enc, "s_city_enc", num_supp)
  printf("Loaded supplier: %d tuples\n", num_supp);
  #undef LOAD_DIM

  CUDA_CHECK(cudaEventRecord(t_load));

  // CQO
  auto cqo_in = fascal_cqo_init(num_tuples);
  cqo_in.predicate_selectivities = {
    fascal_estimate_range_sel(h_lo_orderdate, num_tuples, 19920101, 19971231)
  };
  int cust_qual = (int)(fascal_estimate_eq_sel(h_c_nation, num_cust, 23) * num_cust);
  int supp_qual = (int)(fascal_estimate_eq_sel(h_s_nation, num_supp, 23) * num_supp);
  cqo_in.join_bf_selectivities = {
    fascal_estimate_join_bf_sel(cust_qual, num_cust),
    fascal_estimate_join_bf_sel(supp_qual, num_supp)
  };
  cqo_in.num_gpu_only_ops = 1;
  auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
  fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);

  int in_memory_repeats = 1;
  if (const char *e = getenv("FASCAL_IN_MEMORY_REPEATS")) { in_memory_repeats = atoi(e); if (in_memory_repeats < 1) in_memory_repeats = 1; }
  int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr);

  // ODZC mapping
  auto odzc = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));
  uint32_t *h_bm = nullptr, *d_bm = nullptr; uint8_t *h_ts = nullptr, *d_ts = nullptr;
  fascal_alloc_seed_bitmap(num_tuples, &h_bm, &d_bm, &h_ts, &d_ts);

  #define ODZC_MAP(var, host, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; int *d_##var = nullptr; \
    fascal_odzc_map(odzc.get(), host, num_tuples, name, &mc_##var, &d_##var);

  ODZC_MAP(lo_custkey, h_lo_custkey, "lo_custkey")
  ODZC_MAP(lo_suppkey, h_lo_suppkey, "lo_suppkey")
  ODZC_MAP(lo_orderdate, h_lo_orderdate, "lo_orderdate")
  ODZC_MAP(lo_revenue, h_lo_revenue, "lo_revenue")
  #undef ODZC_MAP

  #define ODZC_DIM(var, host, n, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; \
    mc_##var = odzc->register_column(host, (size_t)(n) * sizeof(int)); \
    int *d_##var = static_cast<int*>(mc_##var.device_ptr); \
    if (!d_##var) { CUDA_CHECK(cudaMalloc(&d_##var, (size_t)(n)*sizeof(int))); \
                    CUDA_CHECK(cudaMemcpy(d_##var, host, (size_t)(n)*sizeof(int), cudaMemcpyHostToDevice)); }

  ODZC_DIM(c_custkey, h_c_custkey, num_cust, "c_custkey")
  ODZC_DIM(c_nation, h_c_nation, num_cust, "c_nation")
  ODZC_DIM(c_city_enc, h_c_city_enc, num_cust, "c_city_enc")
  ODZC_DIM(s_suppkey, h_s_suppkey, num_supp, "s_suppkey")
  ODZC_DIM(s_nation, h_s_nation, num_supp, "s_nation")
  ODZC_DIM(s_city_enc, h_s_city_enc, num_supp, "s_city_enc")
  #undef ODZC_DIM

  CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

  // Aggregation table: 250 cities * 250 cities * 10 years = 625000 groups
  const size_t num_groups = 250ULL * 250 * 10;
  unsigned long long *d_agg;
  CUDA_CHECK(cudaMalloc(&d_agg, num_groups * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * sizeof(unsigned long long)));

  // Hash tables + Bloom filters
  GpuHashTable ht_cust = GpuHashTable::allocate_direct(num_cust);
  GpuBloomFilter bf_cust = GpuBloomFilter::allocate(num_cust, 3);
  CUDA_CHECK(cudaMemset(ht_cust.d_entries, 0xff, ht_cust.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(bf_cust.d_bits, 0, ((size_t)bf_cust.size_bits + 63) / 64 * sizeof(uint64_t)));

  GpuHashTable ht_supp = GpuHashTable::allocate_direct(num_supp);
  GpuBloomFilter bf_supp = GpuBloomFilter::allocate(num_supp, 3);
  CUDA_CHECK(cudaMemset(ht_supp.d_entries, 0xff, ht_supp.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(bf_supp.d_bits, 0, ((size_t)bf_supp.size_bits + 63) / 64 * sizeof(uint64_t)));

  // AFP prefilter bitmaps for build kernels
  #define BUILD_PREFILTER(prefix, n) \
    size_t prefix##_bm_words = ((size_t)(n) + 31) / 32; \
    size_t prefix##_ntiles = ((size_t)(n) + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE; \
    uint32_t *h_##prefix##_pf = nullptr; \
    CUDA_CHECK(cudaHostAlloc(&h_##prefix##_pf, prefix##_bm_words * sizeof(uint32_t), cudaHostAllocMapped)); \
    memset(h_##prefix##_pf, 0, prefix##_bm_words * sizeof(uint32_t)); \
    uint32_t *d_##prefix##_pf = nullptr; \
    CUDA_CHECK(cudaHostGetDevicePointer(&d_##prefix##_pf, h_##prefix##_pf, 0)); \
    uint8_t *h_##prefix##_ts = nullptr; \
    CUDA_CHECK(cudaHostAlloc(&h_##prefix##_ts, prefix##_ntiles, cudaHostAllocMapped)); \
    memset(h_##prefix##_ts, 0, prefix##_ntiles); \
    uint8_t *d_##prefix##_ts = nullptr; \
    CUDA_CHECK(cudaHostGetDevicePointer(&d_##prefix##_ts, h_##prefix##_ts, 0));

  BUILD_PREFILTER(cust, num_cust)
  BUILD_PREFILTER(supp, num_supp)
  #undef BUILD_PREFILTER

  // CPU prefilter for build: customer (c_nation == 23)
  fascal_cpu_prefilter_run(num_cust, h_cust_pf, h_cust_ts,
      [&](int off, int cnt) {
          uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
          memset(&h_cust_pf[ws], 0, (we - ws) * sizeof(uint32_t));
          for (int i = 0; i < cnt; ++i) { int idx = off + i;
            if (h_c_nation[idx] == 23) h_cust_pf[idx >> 5] |= (1u << (idx & 31)); }
      });

  // CPU prefilter for build: supplier (s_nation == 23)
  fascal_cpu_prefilter_run(num_supp, h_supp_pf, h_supp_ts,
      [&](int off, int cnt) {
          uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
          memset(&h_supp_pf[ws], 0, (we - ws) * sizeof(uint32_t));
          for (int i = 0; i < cnt; ++i) { int idx = off + i;
            if (h_s_nation[idx] == 23) h_supp_pf[idx >> 5] |= (1u << (idx & 31)); }
      });

  // Pipeline setup
  int *h_data[] = {h_lo_custkey, h_lo_suppkey, h_lo_orderdate, h_lo_revenue};
  fascal::runtime::BloomFilter h_bf_cust, h_bf_supp;
  h_bf_cust.bits = nullptr; h_bf_supp.bits = nullptr;

  const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
  const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
  cudaStream_t *streams = new cudaStream_t[N_STREAMS];
  for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
  int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
  cudaEventRecord(t0);

  // Build customer
  build_customer<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_cust + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_cust, d_c_custkey, d_c_nation, d_c_city_enc,
      ht_cust.d_entries, ht_cust.ht_size, bf_cust.d_bits, bf_cust.size_bits, 3,
      d_cust_pf, d_cust_ts);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (!bloom_off && 0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) {
      h_bf_cust = fascal::runtime::BloomFilter::allocate(num_cust, 3, llc_bytes);
      bf_cust.copy_to_host(h_bf_cust.bits, ((h_bf_cust.size_bits + 63) / 64) * sizeof(uint64_t));
  }

  // Build supplier
  build_supplier<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_supp + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_supp, d_s_suppkey, d_s_nation, d_s_city_enc,
      ht_supp.d_entries, ht_supp.ht_size, bf_supp.d_bits, bf_supp.size_bits, 3,
      d_supp_pf, d_supp_ts);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (!bloom_off && 1 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[1]) {
      h_bf_supp = fascal::runtime::BloomFilter::allocate(num_supp, 3, llc_bytes);
      bf_supp.copy_to_host(h_bf_supp.bits, ((h_bf_supp.size_bits + 63) / 64) * sizeof(uint64_t));
  }

  // Result pipeline
  auto run_kernel = [&]() {
    auto prefilter = [&](int bc, int bo, int tt) {
      fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
        [&](int off, int cnt) { cpu_predicate(h_data, off, cnt, h_bm, &h_bf_cust, &h_bf_supp); }, bo, tt);
    };
    auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
      ssb_q32_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
        nt, bo, d_lo_custkey, d_lo_suppkey, d_lo_orderdate, d_lo_revenue,
        d_bm, d_ts,
        bf_cust.d_bits, (bloom_off ? 0u : bf_cust.size_bits), ht_cust.d_entries, ht_cust.ht_size,
        bf_supp.d_bits, (bloom_off ? 0u : bf_supp.size_bits), ht_supp.d_entries, ht_supp.ht_size,
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
      CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * sizeof(unsigned long long)));
      CUDA_CHECK(cudaEventRecord(t0)); run_kernel(); CUDA_CHECK(cudaEventRecord(t1));
      CUDA_CHECK(cudaDeviceSynchronize());
      float m; CUDA_CHECK(cudaEventElapsedTime(&m, t0, t1));
      total += m; if (m < best) best = m;
    }
    printf("In-memory result kernel: best %.3f ms avg %.3f ms (%d runs)\n", best, total / in_memory_repeats, in_memory_repeats);
  }

  // Output
  unsigned long long *h_agg = (unsigned long long*)malloc(num_groups * sizeof(unsigned long long));
  CUDA_CHECK(cudaMemcpy(h_agg, d_agg, num_groups * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  for (size_t k = 0; k < num_groups; ++k) {
    if (h_agg[k] == 0) continue;
    size_t gk = k;
    int key0 = (int)(gk % 250); gk /= 250;
    int key1 = (int)(gk % 250); gk /= 250;
    int key2 = (int)(gk % 10);
    printf("ROW: %d|%d|%d|%llu\n", key0, key1, key2, h_agg[k]);
  }
  free(h_agg);

  // Timing report
  CUDA_CHECK(cudaEventRecord(t_result)); CUDA_CHECK(cudaDeviceSynchronize());
  { float ml, mb, mr;
    CUDA_CHECK(cudaEventElapsedTime(&ml, t_start, t_load));
    CUDA_CHECK(cudaEventElapsedTime(&mb, t_load, t_query));
    /* merged into query */
    CUDA_CHECK(cudaEventElapsedTime(&mr, t_query, t_result));
    printf("Timing: load=%.1fms query=%.1fms result=%.1fms total=%.1fms\n", ml, mb, mr, ml+mb+mr); }

  // Cleanup
  odzc->unregister_column(mc_lo_custkey); odzc->unregister_column(mc_lo_suppkey);
  odzc->unregister_column(mc_lo_orderdate); odzc->unregister_column(mc_lo_revenue);
  odzc->unregister_column(mc_c_custkey); odzc->unregister_column(mc_c_nation);
  odzc->unregister_column(mc_c_city_enc);
  odzc->unregister_column(mc_s_suppkey); odzc->unregister_column(mc_s_nation);
  odzc->unregister_column(mc_s_city_enc);
  fascal_free_seed_bitmap(h_bm, h_ts);
  fascal_free_column(h_lo_custkey); fascal_free_column(h_lo_suppkey);
  fascal_free_column(h_lo_orderdate); fascal_free_column(h_lo_revenue);
  CUDA_CHECK(cudaFree(d_agg));
  CUDA_CHECK(cudaFreeHost(h_cust_pf)); CUDA_CHECK(cudaFreeHost(h_cust_ts));
  CUDA_CHECK(cudaFreeHost(h_supp_pf)); CUDA_CHECK(cudaFreeHost(h_supp_ts));
  if (h_bf_cust.bits) h_bf_cust.free_filter();
  if (h_bf_supp.bits) h_bf_supp.free_filter();
  fascal_arena_destroy();
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  delete[] streams;
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(t_start); cudaEventDestroy(t_load);   cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  return 0;
}
