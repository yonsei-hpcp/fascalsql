// SSB Q4.1 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Refactored: uses shared macros and CPU predicate helpers.
// Dims: customer(c_region=AMERICA, payload=c_nation_enc), supplier(s_region=AMERICA), part(p_mfgr IN (MFGR#1,MFGR#2))
// GROUP BY d_year(inline) x c_nation_enc, AGG SUM(lo_revenue - lo_supplycost)

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
// GPU Kernel: scan lineorder, bloom probes, HT joins, aggregate
// =============================================================================
template <int BT, int IPT>
__global__ void ssb_q41_kernel(
    int num_tuples, int batch_offset,
    int *d_lo_orderdate, int *d_lo_custkey, int *d_lo_suppkey, int *d_lo_partkey,
    int *d_lo_revenue, int *d_lo_supplycost,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_customer, uint32_t bloom_size_bits_customer,
    HtEntry *ht_customer, uint32_t ht_size_customer,
    const uint64_t *__restrict__ d_bloom_supplier, uint32_t bloom_size_bits_supplier,
    HtEntry *ht_supplier, uint32_t ht_size_supplier,
    const uint64_t *__restrict__ d_bloom_part, uint32_t bloom_size_bits_part,
    HtEntry *ht_part, uint32_t ht_size_part,
    unsigned long long *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int lo_orderdate[IPT], lo_custkey[IPT], lo_suppkey[IPT], lo_partkey[IPT];
  int lo_revenue[IPT], lo_supplycost[IPT], c_nation_enc[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // Pred 0: lo_orderdate BETWEEN 19920101 AND 19981231 (all SSB years)
    if (d_pred_flags[0] == 0) {
      BlockLoadSelect<int, BT, IPT>(d_lo_orderdate + tile_offset, lo_orderdate, selection_flags, num_tile_items);
      BlockPredAndGTE<int, BT, IPT>(lo_orderdate, 19920101, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BT, IPT>(lo_orderdate, 19981231, selection_flags, num_tile_items);
    }

    // Bloom probe: customer
    BlockLoadSelect<int, BT, IPT>(d_lo_custkey + tile_offset, lo_custkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(lo_custkey, selection_flags, num_tile_items, bloom_size_bits_customer, d_bloom_customer, 3);

    // Bloom probe: supplier
    BlockLoadSelect<int, BT, IPT>(d_lo_suppkey + tile_offset, lo_suppkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(lo_suppkey, selection_flags, num_tile_items, bloom_size_bits_supplier, d_bloom_supplier, 3);

    // Bloom probe: part
    BlockLoadSelect<int, BT, IPT>(d_lo_partkey + tile_offset, lo_partkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(lo_partkey, selection_flags, num_tile_items, bloom_size_bits_part, d_bloom_part, 3);

    FASCAL_CHECK_ALIVE(IPT);

    // Load orderdate if CPU handled pred 0
    if (d_pred_flags[0] != 0)
      BlockLoadSelect<int, BT, IPT>(d_lo_orderdate + tile_offset, lo_orderdate, selection_flags, num_tile_items);

    // HT joins
    BlockJoinProbePayloadDirect<BT, IPT>(lo_custkey, selection_flags, num_tile_items, ht_customer, ht_size_customer, 1, c_nation_enc);
    BlockJoinProbeDirect<BT, IPT>(lo_suppkey, selection_flags, num_tile_items, ht_supplier, ht_size_supplier, 1);
    BlockJoinProbeDirect<BT, IPT>(lo_partkey, selection_flags, num_tile_items, ht_part, ht_size_part, 1);

    // Load aggregation columns
    BlockLoadSelect<int, BT, IPT>(d_lo_revenue + tile_offset, lo_revenue, selection_flags, num_tile_items);
    BlockLoadSelect<int, BT, IPT>(d_lo_supplycost + tile_offset, lo_supplycost, selection_flags, num_tile_items);

    // GROUP BY year(10) x nation(25) -> profit
    #pragma unroll
    for (int i = 0; i < IPT; ++i)
      if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i]) {
        int gk = (lo_orderdate[i] / 10000 - 1992) + c_nation_enc[i] * 10;
        atomicAdd(&aggtable[gk], (unsigned long long)((long long)lo_revenue[i] - lo_supplycost[i]));
      }
  } while(0);
}

// =============================================================================
// CPU Predicate
// =============================================================================
static void cpu_predicate(int *h_data[], int offset, int cnt, uint32_t *bitmap) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);

  int *lo_orderdate = h_data[0];

  if (h_pred_flags[0] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, lo_orderdate, offset, cnt, 19920101, 19981231, true, true);
}

// =============================================================================
// Build kernels (dimension sub-pipelines)
// =============================================================================

// customer: c_region == 1 (AMERICA), payload = c_nation_enc
template <int BT, int IPT>
__global__ void build_customer(
    int num_tuples,
    const int *__restrict__ d_c_custkey, const int *__restrict__ d_c_region, const int *__restrict__ d_c_nation_enc,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int k[IPT], r[IPT], n[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_c_custkey + tile_offset), k, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_c_region + tile_offset), r, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(r, 1, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_c_nation_enc + tile_offset), n, selection_flags, num_tile_items);
  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    gpu_ht_insert_one_direct(k[i], n[i], 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(k[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// supplier: s_region == 1 (AMERICA), no payload
template <int BT, int IPT>
__global__ void build_supplier(
    int num_tuples,
    const int *__restrict__ d_s_suppkey, const int *__restrict__ d_s_region,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int k[IPT], r[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_suppkey + tile_offset), k, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_region + tile_offset), r, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(r, 1, selection_flags, num_tile_items);
  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    gpu_ht_insert_one_direct(k[i], tile_offset + threadIdx.x + i * BT, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(k[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// part: p_mfgr == 0 OR p_mfgr == 1 (MFGR#1 or MFGR#2), no payload
template <int BT, int IPT>
__global__ void build_part(
    int num_tuples,
    const int *__restrict__ d_p_partkey, const int *__restrict__ d_p_mfgr,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int k[IPT], m[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_p_partkey + tile_offset), k, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_p_mfgr + tile_offset), m, selection_flags, num_tile_items);
  #pragma unroll
  for (int i = 0; i < IPT; ++i)
    if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i])
      selection_flags[i] = (m[i] == 0 || m[i] == 1) ? 1 : 0;
  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    gpu_ht_insert_one_direct(k[i], tile_offset + threadIdx.x + i * BT, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(k[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
  const char *data_dir = getenv("FASCALSQL_DATA_DIR");
  if (!data_dir) data_dir = getenv("FASCALSQL_SSB_DATA_DIR");
  if (argc >= 3) data_dir = argv[2];
  if (!data_dir) { fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n"); return 1; }

  printf("=== FaScalSQL: ssb_q41 ===\n");
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
  fascal_arena_init((size_t)num_tuples * sizeof(int) * 7);

  // Load fact columns
  char path[512];
  #define LOAD_COL(var, name) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    int *var = load_binary_column(path, num_tuples); \
    if (!var) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

  LOAD_COL(h_lo_orderdate, "lo_orderdate")
  LOAD_COL(h_lo_custkey, "lo_custkey")
  LOAD_COL(h_lo_suppkey, "lo_suppkey")
  LOAD_COL(h_lo_partkey, "lo_partkey")
  LOAD_COL(h_lo_revenue, "lo_revenue")
  LOAD_COL(h_lo_supplycost, "lo_supplycost")
  #undef LOAD_COL

  // Load dimension columns
  #define LOAD_DIM(var, name, cnt) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    int *var = load_binary_column_auto(path, &cnt); \
    if (!var || cnt == 0) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

  int num_cust = 0, num_supp = 0, num_part = 0;
  LOAD_DIM(h_c_custkey, "c_custkey", num_cust)
  LOAD_DIM(h_c_region, "c_region", num_cust)
  LOAD_DIM(h_c_nation_enc, "c_nation_enc", num_cust)
  printf("Loaded customer: %d tuples\n", num_cust);

  LOAD_DIM(h_s_suppkey, "s_suppkey", num_supp)
  LOAD_DIM(h_s_region, "s_region", num_supp)
  printf("Loaded supplier: %d tuples\n", num_supp);

  LOAD_DIM(h_p_partkey, "p_partkey", num_part)
  LOAD_DIM(h_p_mfgr, "p_mfgr", num_part)
  printf("Loaded part: %d tuples\n", num_part);
  #undef LOAD_DIM
  CUDA_CHECK(cudaEventRecord(t_load));

  // CQO
  auto cqo_in = fascal_cqo_init(num_tuples);
  cqo_in.predicate_selectivities = {1.0};
  int cust_qual = (int)(fascal_estimate_eq_sel(h_c_region, num_cust, 1) * num_cust);
  int supp_qual = (int)(fascal_estimate_eq_sel(h_s_region, num_supp, 1) * num_supp);
  int part_qual = (int)(fascal_estimate_selectivity(h_p_mfgr, num_part, [](int v){ return v == 0 || v == 1; }) * num_part);
  cqo_in.join_bf_selectivities = {
    fascal_estimate_join_bf_sel(cust_qual, num_cust),
    fascal_estimate_join_bf_sel(supp_qual, num_supp),
    fascal_estimate_join_bf_sel(part_qual, num_part)
  };
  cqo_in.num_gpu_only_ops = 2;
  auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
  fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);

  int in_memory_repeats = 1;
  if (const char *e = getenv("FASCAL_IN_MEMORY_REPEATS")) { in_memory_repeats = atoi(e); if (in_memory_repeats < 1) in_memory_repeats = 1; }
  int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr);

  // ODZC
  auto odzc = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));
  uint32_t *h_bm = nullptr, *d_bm = nullptr; uint8_t *h_ts = nullptr, *d_ts = nullptr;
  fascal_alloc_seed_bitmap(num_tuples, &h_bm, &d_bm, &h_ts, &d_ts);

  #define ODZC_MAP(var, host, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; int *d_##var = nullptr; \
    fascal_odzc_map(odzc.get(), host, num_tuples, name, &mc_##var, &d_##var);

  ODZC_MAP(lo_orderdate, h_lo_orderdate, "lo_orderdate")
  ODZC_MAP(lo_custkey, h_lo_custkey, "lo_custkey")
  ODZC_MAP(lo_suppkey, h_lo_suppkey, "lo_suppkey")
  ODZC_MAP(lo_partkey, h_lo_partkey, "lo_partkey")
  ODZC_MAP(lo_revenue, h_lo_revenue, "lo_revenue")
  ODZC_MAP(lo_supplycost, h_lo_supplycost, "lo_supplycost")
  #undef ODZC_MAP

  // Dim ODZC (register_column for small dims)
  #define DIM_ODZC(var, host, nt, name) \
    auto mc_##var = odzc->register_column(host, (size_t)(nt) * sizeof(int)); \
    int *d_##var = static_cast<int*>(mc_##var.device_ptr); \
    if (!d_##var) { fprintf(stderr, "ODZC failed for " name ", fallback\n"); \
      CUDA_CHECK(cudaMalloc(&d_##var, (size_t)(nt)*sizeof(int))); \
      CUDA_CHECK(cudaMemcpy(d_##var, host, (size_t)(nt)*sizeof(int), cudaMemcpyHostToDevice)); }

  DIM_ODZC(c_custkey, h_c_custkey, num_cust, "c_custkey")
  DIM_ODZC(c_region, h_c_region, num_cust, "c_region")
  DIM_ODZC(c_nation_enc, h_c_nation_enc, num_cust, "c_nation_enc")
  DIM_ODZC(s_suppkey, h_s_suppkey, num_supp, "s_suppkey")
  DIM_ODZC(s_region, h_s_region, num_supp, "s_region")
  DIM_ODZC(p_partkey, h_p_partkey, num_part, "p_partkey")
  DIM_ODZC(p_mfgr, h_p_mfgr, num_part, "p_mfgr")
  #undef DIM_ODZC

  CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

  // Aggregation table: year(7+pad=10) x nation(25)
  size_t num_groups = 10ULL * 25ULL;
  unsigned long long *d_agg;
  CUDA_CHECK(cudaMalloc(&d_agg, num_groups * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * sizeof(unsigned long long)));

  // HT + BF allocation
  fascal::runtime::BloomFilter bf_storage[3];
  GpuHashTable ht_cust = GpuHashTable::allocate_direct(num_cust);
  GpuBloomFilter gpu_bf_cust = GpuBloomFilter::allocate(num_cust, 3);
  if (!bloom_off && 0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0])
      bf_storage[0] = fascal::runtime::BloomFilter::allocate(num_cust, 3, llc_bytes);
  CUDA_CHECK(cudaMemset(ht_cust.d_entries, 0xff, ht_cust.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(gpu_bf_cust.d_bits, 0, ((size_t)gpu_bf_cust.size_bits + 63) / 64 * sizeof(uint64_t)));

  GpuHashTable ht_supp = GpuHashTable::allocate_direct(num_supp);
  GpuBloomFilter gpu_bf_supp = GpuBloomFilter::allocate(num_supp, 3);
  if (!bloom_off && 1 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[1])
      bf_storage[1] = fascal::runtime::BloomFilter::allocate(num_supp, 3, llc_bytes);
  CUDA_CHECK(cudaMemset(ht_supp.d_entries, 0xff, ht_supp.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(gpu_bf_supp.d_bits, 0, ((size_t)gpu_bf_supp.size_bits + 63) / 64 * sizeof(uint64_t)));

  GpuHashTable ht_part = GpuHashTable::allocate_direct(num_part);
  GpuBloomFilter gpu_bf_part = GpuBloomFilter::allocate(num_part, 3);
  if (!bloom_off && 2 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[2])
      bf_storage[2] = fascal::runtime::BloomFilter::allocate(num_part, 3, llc_bytes);
  CUDA_CHECK(cudaMemset(ht_part.d_entries, 0xff, ht_part.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(gpu_bf_part.d_bits, 0, ((size_t)gpu_bf_part.size_bits + 63) / 64 * sizeof(uint64_t)));

  // AFP prefilter bitmaps for dim build pipelines
  #define ALLOC_DIM_BM(prefix, nt) \
    size_t prefix##_bm_words = ((size_t)(nt) + 31) / 32; \
    size_t prefix##_ntiles = ((size_t)(nt) + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE; \
    uint32_t *h_##prefix##_pf = nullptr; \
    CUDA_CHECK(cudaHostAlloc(&h_##prefix##_pf, prefix##_bm_words * sizeof(uint32_t), cudaHostAllocMapped)); \
    std::memset(h_##prefix##_pf, 0, prefix##_bm_words * sizeof(uint32_t)); \
    uint32_t *d_##prefix##_pf = nullptr; \
    CUDA_CHECK(cudaHostGetDevicePointer(&d_##prefix##_pf, h_##prefix##_pf, 0)); \
    uint8_t *h_##prefix##_ts = nullptr; \
    CUDA_CHECK(cudaHostAlloc(&h_##prefix##_ts, prefix##_ntiles, cudaHostAllocMapped)); \
    std::memset(h_##prefix##_ts, 0, prefix##_ntiles); \
    uint8_t *d_##prefix##_ts = nullptr; \
    CUDA_CHECK(cudaHostGetDevicePointer(&d_##prefix##_ts, h_##prefix##_ts, 0));

  ALLOC_DIM_BM(cust, num_cust)
  ALLOC_DIM_BM(supp, num_supp)
  ALLOC_DIM_BM(part, num_part)
  #undef ALLOC_DIM_BM

  // Pipeline setup
  int *h_data[] = {h_lo_orderdate, h_lo_custkey, h_lo_suppkey, h_lo_partkey, h_lo_revenue, h_lo_supplycost};
  const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
  const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
  cudaStream_t *streams = new cudaStream_t[N_STREAMS];
  for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
  int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

  // Build dim pipelines
  cudaEventRecord(t0);

  // Customer build
  fascal_cpu_prefilter_run(num_cust, h_cust_pf, h_cust_ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&h_cust_pf[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      if (h_c_region[idx] == 1) h_cust_pf[idx >> 5] |= (1u << (idx & 31)); }
  });
  build_customer<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_cust + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
    num_cust, d_c_custkey, d_c_region, d_c_nation_enc,
    ht_cust.d_entries, ht_cust.ht_size, gpu_bf_cust.d_bits, gpu_bf_cust.size_bits, 3,
    d_cust_pf, d_cust_ts);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (bf_storage[0].bits) gpu_bf_cust.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));

  // Supplier build
  fascal_cpu_prefilter_run(num_supp, h_supp_pf, h_supp_ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&h_supp_pf[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      if (h_s_region[idx] == 1) h_supp_pf[idx >> 5] |= (1u << (idx & 31)); }
  });
  build_supplier<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_supp + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
    num_supp, d_s_suppkey, d_s_region,
    ht_supp.d_entries, ht_supp.ht_size, gpu_bf_supp.d_bits, gpu_bf_supp.size_bits, 3,
    d_supp_pf, d_supp_ts);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (bf_storage[1].bits) gpu_bf_supp.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));

  // Part build
  fascal_cpu_prefilter_run(num_part, h_part_pf, h_part_ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&h_part_pf[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      if (h_p_mfgr[idx] == 0 || h_p_mfgr[idx] == 1) h_part_pf[idx >> 5] |= (1u << (idx & 31)); }
  });
  build_part<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_part + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
    num_part, d_p_partkey, d_p_mfgr,
    ht_part.d_entries, ht_part.ht_size, gpu_bf_part.d_bits, gpu_bf_part.size_bits, 3,
    d_part_pf, d_part_ts);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (bf_storage[2].bits) gpu_bf_part.copy_to_host(bf_storage[2].bits, ((size_t)(bf_storage[2].size_bits + 63) / 64) * sizeof(uint64_t));

  // Result pipeline
  auto run_kernel = [&]() {
    auto prefilter = [&](int bc, int bo, int tt) {
      fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
        [&](int off, int cnt) { cpu_predicate(h_data, off, cnt, h_bm); }, bo, tt);
    };
    auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
      ssb_q41_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
        nt, bo, d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_partkey, d_lo_revenue, d_lo_supplycost,
        d_bm, d_ts,
        gpu_bf_cust.d_bits, (bloom_off ? 0u : gpu_bf_cust.size_bits), ht_cust.d_entries, ht_cust.ht_size,
        gpu_bf_supp.d_bits, (bloom_off ? 0u : gpu_bf_supp.size_bits), ht_supp.d_entries, ht_supp.ht_size,
        gpu_bf_part.d_bits, (bloom_off ? 0u : gpu_bf_part.size_bits), ht_part.d_entries, ht_part.ht_size,
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
  for (size_t k = 0; k < num_groups; k++) {
    if (h_agg[k] != 0) {
      size_t gk = k;
      int year_off = (int)(gk % 10ULL); gk /= 10;
      int nation = (int)(gk % 25ULL);
      printf("ROW: %d|%d|%llu\n", year_off, nation, (unsigned long long)h_agg[k]);
    }
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
  odzc->unregister_column(mc_lo_orderdate); odzc->unregister_column(mc_lo_custkey);
  odzc->unregister_column(mc_lo_suppkey);   odzc->unregister_column(mc_lo_partkey);
  odzc->unregister_column(mc_lo_revenue);   odzc->unregister_column(mc_lo_supplycost);
  odzc->unregister_column(mc_c_custkey);    odzc->unregister_column(mc_c_region);
  odzc->unregister_column(mc_c_nation_enc); odzc->unregister_column(mc_s_suppkey);
  odzc->unregister_column(mc_s_region);     odzc->unregister_column(mc_p_partkey);
  odzc->unregister_column(mc_p_mfgr);
  CUDA_CHECK(cudaFreeHost(h_cust_ts)); CUDA_CHECK(cudaFreeHost(h_cust_pf));
  CUDA_CHECK(cudaFreeHost(h_supp_ts)); CUDA_CHECK(cudaFreeHost(h_supp_pf));
  CUDA_CHECK(cudaFreeHost(h_part_ts)); CUDA_CHECK(cudaFreeHost(h_part_pf));
  gpu_bf_cust.free_filter(); ht_cust.free_table();
  gpu_bf_supp.free_filter(); ht_supp.free_table();
  gpu_bf_part.free_filter(); ht_part.free_table();
  for (int i = 0; i < 3; ++i) if (bf_storage[i].bits) bf_storage[i].free_filter();
  fascal_free_column(h_c_custkey); fascal_free_column(h_c_region); fascal_free_column(h_c_nation_enc);
  fascal_free_column(h_s_suppkey); fascal_free_column(h_s_region);
  fascal_free_column(h_p_partkey); fascal_free_column(h_p_mfgr);
  fascal_free_seed_bitmap(h_bm, h_ts);
  fascal_free_column(h_lo_orderdate); fascal_free_column(h_lo_custkey);
  fascal_free_column(h_lo_suppkey);   fascal_free_column(h_lo_partkey);
  fascal_free_column(h_lo_revenue);   fascal_free_column(h_lo_supplycost);
  CUDA_CHECK(cudaFree(d_agg));
  fascal_arena_destroy();
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  delete[] streams;
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(t_start); cudaEventDestroy(t_load);   cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  return 0;
}
