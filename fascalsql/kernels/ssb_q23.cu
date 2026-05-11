// SSB Q2.3 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Refactored: uses shared macros and CPU predicate helpers.
// Part dim: p_brand1=272 (MFGR#2239), payload=p_brand1. Supplier dim: s_region=3 (EUROPE).
// Date join eliminated: inline year = lo_orderdate / 10000 - 1992

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

#define BLOCK_THREADS 128
#define ITEMS_PER_THREAD 8
#define TILE_SIZE 1024
#define MAX_PREDICATES 16
#define NUM_AGGREGATES 1

__device__ __constant__ unsigned char d_pred_flags[MAX_PREDICATES];
static unsigned char h_pred_flags[MAX_PREDICATES] = {0};

// =============================================================================
// GPU Kernel: scan lineorder, BF probes, HT joins, GROUP BY year x brand
// =============================================================================
template <int BT, int IPT>
__global__ void ssb_q23_kernel(
    int num_tuples, int batch_offset,
    int *d_lo_orderdate, int *d_lo_partkey, int *d_lo_suppkey, int *d_lo_revenue,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bf_part, uint32_t bf_bits_part,
    HtEntry *ht_part, uint32_t ht_sz_part,
    const uint64_t *__restrict__ d_bf_supp, uint32_t bf_bits_supp,
    HtEntry *ht_supp, uint32_t ht_sz_supp,
    unsigned long long *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int lo_partkey[IPT], lo_suppkey[IPT], lo_orderdate[IPT], lo_revenue[IPT];
  int p_brand1[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // BF probe: part
    BlockLoadSelect<int, BT, IPT>(d_lo_partkey + tile_offset, lo_partkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(lo_partkey, selection_flags, num_tile_items, bf_bits_part, d_bf_part, 3);

    // BF probe: supplier
    BlockLoadSelect<int, BT, IPT>(d_lo_suppkey + tile_offset, lo_suppkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(lo_suppkey, selection_flags, num_tile_items, bf_bits_supp, d_bf_supp, 3);

    // Pred 0: lo_orderdate BETWEEN 19920101 AND 19981231
    if (d_pred_flags[0] == 0) {
      BlockLoadSelect<int, BT, IPT>(d_lo_orderdate + tile_offset, lo_orderdate, selection_flags, num_tile_items);
      BlockPredAndGTE<int, BT, IPT>(lo_orderdate, 19920101, selection_flags, num_tile_items);
      BlockPredAndLTE<int, BT, IPT>(lo_orderdate, 19981231, selection_flags, num_tile_items);
    }

    FASCAL_CHECK_ALIVE(IPT);

    // HT join: part (payload = p_brand1)
    BlockJoinProbePayloadDirect<BT, IPT>(lo_partkey, selection_flags, num_tile_items,
        ht_part, ht_sz_part, 1, p_brand1);
    // HT join: supplier (existence only)
    BlockJoinProbeDirect<BT, IPT>(lo_suppkey, selection_flags, num_tile_items,
        ht_supp, ht_sz_supp, 1);

    // Load lo_orderdate if CPU handled pred 0
    if (d_pred_flags[0] != 0)
      BlockLoadSelect<int, BT, IPT>(d_lo_orderdate + tile_offset, lo_orderdate, selection_flags, num_tile_items);

    BlockLoadSelect<int, BT, IPT>(d_lo_revenue + tile_offset, lo_revenue, selection_flags, num_tile_items);

    // GROUP BY: year x brand -> direct-index atomicAdd
    #pragma unroll
    for (int i = 0; i < IPT; ++i)
      if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i]) {
        int gk = (lo_orderdate[i] / 10000 - 1992) + (p_brand1[i] * 10);
        atomicAdd(&aggtable[gk], (long long)lo_revenue[i]);
      }
  } while(0);
}

// =============================================================================
// Build kernel: part dimension (p_brand1=272, payload=p_brand1)
// =============================================================================
template <int BT, int IPT>
__global__ void build_part_kernel(
    int num_tuples,
    const int *__restrict__ d_p_partkey, const int *__restrict__ d_p_brand1,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);

  int pk[IPT], br[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_p_partkey + tile_offset), pk, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_p_brand1 + tile_offset), br, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(br, 272, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    gpu_ht_insert_one_direct(pk[i], br[i], 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(pk[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// Build kernel: supplier dimension (s_region=3, existence HT)
// =============================================================================
template <int BT, int IPT>
__global__ void build_supp_kernel(
    int num_tuples,
    const int *__restrict__ d_s_suppkey, const int *__restrict__ d_s_region,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);

  int sk[IPT], sr[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_suppkey + tile_offset), sk, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_region + tile_offset), sr, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(sr, 3, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    int payload = tile_offset + threadIdx.x + i * BT;
    gpu_ht_insert_one_direct(sk[i], payload, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(sk[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// CPU Predicate: AVX2 multi-pass bitmap filtering
// =============================================================================
static void cpu_predicate(int *h_data[], int offset, int cnt, uint32_t *bitmap,
                          fascal::runtime::BloomFilter **bfs, int nbf) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);

  int *lo_orderdate = h_data[0], *lo_partkey = h_data[1], *lo_suppkey = h_data[2];

  if (h_pred_flags[0] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, lo_orderdate, offset, cnt, 19920101, 19981231, true, true);
  if (bfs && nbf > 0 && bfs[0])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, lo_partkey, offset, cnt, bfs[0]);
  if (bfs && nbf > 1 && bfs[1])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, lo_suppkey, offset, cnt, bfs[1]);
}

static void cpu_prefilter_part(int *h_pk, int *h_br, int nt, uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(nt, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      if (h_br[idx] == 272) bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
    }
  });
}

static void cpu_prefilter_supp(int *h_sk, int *h_sr, int nt, uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(nt, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      if (h_sr[idx] == 3) bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
    }
  });
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
  const char *data_dir = getenv("FASCALSQL_DATA_DIR");
  if (!data_dir) data_dir = getenv("FASCALSQL_SSB_DATA_DIR");
  if (argc >= 3) data_dir = argv[2];
  if (!data_dir) { fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n"); return 1; }

  printf("=== FaScalSQL: ssb_q23 ===\n");
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
  LOAD_COL(h_lo_partkey, "lo_partkey")
  LOAD_COL(h_lo_suppkey, "lo_suppkey")
  LOAD_COL(h_lo_revenue, "lo_revenue")

  // Load part dimension
  int num_part = 0;
  snprintf(path, sizeof(path), "%s/p_partkey.bin", data_dir);
  int *h_p_partkey = load_binary_column_auto(path, &num_part);
  if (!h_p_partkey || !num_part) { fprintf(stderr, "Failed to load p_partkey\n"); return 1; }
  snprintf(path, sizeof(path), "%s/p_brand1.bin", data_dir);
  int *h_p_brand1 = load_binary_column_auto(path, &num_part);
  if (!h_p_brand1) { fprintf(stderr, "Failed to load p_brand1\n"); return 1; }
  printf("Loaded part: %d tuples\n", num_part);

  // Load supplier dimension
  int num_supp = 0;
  snprintf(path, sizeof(path), "%s/s_suppkey.bin", data_dir);
  int *h_s_suppkey = load_binary_column_auto(path, &num_supp);
  if (!h_s_suppkey || !num_supp) { fprintf(stderr, "Failed to load s_suppkey\n"); return 1; }
  snprintf(path, sizeof(path), "%s/s_region.bin", data_dir);
  int *h_s_region = load_binary_column_auto(path, &num_supp);
  if (!h_s_region) { fprintf(stderr, "Failed to load s_region\n"); return 1; }
  printf("Loaded supplier: %d tuples\n", num_supp);
  #undef LOAD_COL
  CUDA_CHECK(cudaEventRecord(t_load));

  // CQO
  auto cqo_in = fascal_cqo_init(num_tuples);
  cqo_in.predicate_selectivities = {1.0};
  int part_qual = (int)(fascal_estimate_eq_sel(h_p_brand1, num_part, 272) * num_part);
  int supp_qual = (int)(fascal_estimate_eq_sel(h_s_region, num_supp, 3) * num_supp);
  cqo_in.join_bf_selectivities = {
    fascal_estimate_join_bf_sel(part_qual, num_part),
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

  // Bloom filters
  fascal::runtime::BloomFilter bf_storage[2];
  fascal::runtime::BloomFilter *bf_array[2];
  bf_array[0] = (!bloom_off && 0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[0] : nullptr;
  bf_array[1] = (!bloom_off && 1 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[1]) ? &bf_storage[1] : nullptr;

  // ODZC mapping
  auto odzc = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));
  uint32_t *h_bm = nullptr, *d_bm = nullptr; uint8_t *h_ts = nullptr, *d_ts = nullptr;
  fascal_alloc_seed_bitmap(num_tuples, &h_bm, &d_bm, &h_ts, &d_ts);

  #define ODZC_MAP(var, host, n, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; int *d_##var = nullptr; \
    fascal_odzc_map(odzc.get(), host, n, name, &mc_##var, &d_##var);

  #define ODZC_DIM(var, host, n, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; \
    mc_##var = odzc->register_column(host, (size_t)(n) * sizeof(int)); \
    int *d_##var = static_cast<int*>(mc_##var.device_ptr); \
    if (!d_##var) { fprintf(stderr, "ODZC fallback: " name "\n"); \
      CUDA_CHECK(cudaMalloc(&d_##var, (size_t)(n)*sizeof(int))); \
      CUDA_CHECK(cudaMemcpy(d_##var, host, (size_t)(n)*sizeof(int), cudaMemcpyHostToDevice)); }

  ODZC_MAP(lo_orderdate, h_lo_orderdate, num_tuples, "lo_orderdate")
  ODZC_MAP(lo_partkey, h_lo_partkey, num_tuples, "lo_partkey")
  ODZC_MAP(lo_suppkey, h_lo_suppkey, num_tuples, "lo_suppkey")
  ODZC_MAP(lo_revenue, h_lo_revenue, num_tuples, "lo_revenue")
  ODZC_DIM(p_partkey, h_p_partkey, num_part, "p_partkey")
  ODZC_DIM(p_brand1, h_p_brand1, num_part, "p_brand1")
  ODZC_DIM(s_suppkey, h_s_suppkey, num_supp, "s_suppkey")
  ODZC_DIM(s_region, h_s_region, num_supp, "s_region")
  #undef ODZC_MAP
  #undef ODZC_DIM

  CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

  // Aggregation table: 10 years x 1000 brands
  const size_t num_groups = 10000ULL;
  unsigned long long *d_agg;
  CUDA_CHECK(cudaMalloc(&d_agg, num_groups * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * sizeof(unsigned long long)));

  // Build HT + BF: part
  GpuHashTable ht_part = GpuHashTable::allocate_direct(num_part);
  GpuBloomFilter gpu_bf_part = GpuBloomFilter::allocate(num_part, 3);
  bf_storage[0] = fascal::runtime::BloomFilter::allocate(num_part, 3, llc_bytes);
  CUDA_CHECK(cudaMemset(ht_part.d_entries, 0xff, ht_part.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(gpu_bf_part.d_bits, 0, ((size_t)gpu_bf_part.size_bits + 63) / 64 * sizeof(uint64_t)));

  // Part prefilter bitmap
  size_t part_bm_w = ((size_t)num_part + 31) / 32, part_nt = ((size_t)num_part + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
  uint32_t *h_part_pf = nullptr; CUDA_CHECK(cudaHostAlloc(&h_part_pf, part_bm_w * 4, cudaHostAllocMapped));
  memset(h_part_pf, 0, part_bm_w * 4);
  uint32_t *d_part_pf = nullptr; CUDA_CHECK(cudaHostGetDevicePointer(&d_part_pf, h_part_pf, 0));
  uint8_t *h_part_ts = nullptr; CUDA_CHECK(cudaHostAlloc(&h_part_ts, part_nt, cudaHostAllocMapped));
  memset(h_part_ts, 0, part_nt);
  uint8_t *d_part_ts = nullptr; CUDA_CHECK(cudaHostGetDevicePointer(&d_part_ts, h_part_ts, 0));

  // Build HT + BF: supplier
  GpuHashTable ht_supp = GpuHashTable::allocate_direct(num_supp);
  GpuBloomFilter gpu_bf_supp = GpuBloomFilter::allocate(num_supp, 3);
  bf_storage[1] = fascal::runtime::BloomFilter::allocate(num_supp, 3, llc_bytes);
  CUDA_CHECK(cudaMemset(ht_supp.d_entries, 0xff, ht_supp.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(gpu_bf_supp.d_bits, 0, ((size_t)gpu_bf_supp.size_bits + 63) / 64 * sizeof(uint64_t)));

  // Supplier prefilter bitmap
  size_t supp_bm_w = ((size_t)num_supp + 31) / 32, supp_nt = ((size_t)num_supp + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
  uint32_t *h_supp_pf = nullptr; CUDA_CHECK(cudaHostAlloc(&h_supp_pf, supp_bm_w * 4, cudaHostAllocMapped));
  memset(h_supp_pf, 0, supp_bm_w * 4);
  uint32_t *d_supp_pf = nullptr; CUDA_CHECK(cudaHostGetDevicePointer(&d_supp_pf, h_supp_pf, 0));
  uint8_t *h_supp_ts = nullptr; CUDA_CHECK(cudaHostAlloc(&h_supp_ts, supp_nt, cudaHostAllocMapped));
  memset(h_supp_ts, 0, supp_nt);
  uint8_t *d_supp_ts = nullptr; CUDA_CHECK(cudaHostGetDevicePointer(&d_supp_ts, h_supp_ts, 0));

  // Pipeline setup
  int *h_data[] = {h_lo_orderdate, h_lo_partkey, h_lo_suppkey, h_lo_revenue};
  const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
  const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
  cudaStream_t *streams = new cudaStream_t[N_STREAMS];
  for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
  int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

  auto run_kernel = [&]() {
    auto prefilter = [&](int bc, int bo, int tt) {
      fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
        [&](int off, int cnt) { cpu_predicate(h_data, off, cnt, h_bm, bf_array, 2); }, bo, tt);
    };
    auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
      ssb_q23_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
        nt, bo, d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue, d_bm, d_ts,
        gpu_bf_part.d_bits, (bloom_off ? 0u : gpu_bf_part.size_bits), ht_part.d_entries, ht_part.ht_size,
        gpu_bf_supp.d_bits, (bloom_off ? 0u : gpu_bf_supp.size_bits), ht_supp.d_entries, ht_supp.ht_size,
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

  // Execute: build dims, then result kernel
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
  cudaEventRecord(t0);

  // Build part
  cpu_prefilter_part(h_p_partkey, h_p_brand1, num_part, h_part_pf, h_part_ts);
  build_part_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_part + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_part, d_p_partkey, d_p_brand1,
      ht_part.d_entries, ht_part.ht_size, gpu_bf_part.d_bits, gpu_bf_part.size_bits, 3,
      d_part_pf, d_part_ts);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (bf_storage[0].bits) gpu_bf_part.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));

  // Build supplier
  cpu_prefilter_supp(h_s_suppkey, h_s_region, num_supp, h_supp_pf, h_supp_ts);
  build_supp_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_supp + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_supp, d_s_suppkey, d_s_region,
      ht_supp.d_entries, ht_supp.ht_size, gpu_bf_supp.d_bits, gpu_bf_supp.size_bits, 3,
      d_supp_pf, d_supp_ts);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (bf_storage[1].bits) gpu_bf_supp.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));

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

  // Output: year x brand
  unsigned long long *h_agg = (unsigned long long*)malloc(num_groups * sizeof(unsigned long long));
  CUDA_CHECK(cudaMemcpy(h_agg, d_agg, num_groups * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  for (size_t k = 0; k < num_groups; k++) {
    if (h_agg[k] != 0) {
      size_t gk = k;
      int year = (int)(gk % 10); gk /= 10;
      int brand = (int)(gk % 1000);
      printf("ROW: %d|%d|%llu\n", year, brand, (unsigned long long)h_agg[k]);
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
  odzc->unregister_column(mc_lo_orderdate); odzc->unregister_column(mc_lo_partkey);
  odzc->unregister_column(mc_lo_suppkey);   odzc->unregister_column(mc_lo_revenue);
  odzc->unregister_column(mc_p_partkey);    odzc->unregister_column(mc_p_brand1);
  odzc->unregister_column(mc_s_suppkey);    odzc->unregister_column(mc_s_region);
  CUDA_CHECK(cudaFreeHost(h_part_ts)); CUDA_CHECK(cudaFreeHost(h_part_pf));
  gpu_bf_part.free_filter(); ht_part.free_table(); bf_storage[0].free_filter();
  fascal_free_column(h_p_partkey); fascal_free_column(h_p_brand1);
  CUDA_CHECK(cudaFreeHost(h_supp_ts)); CUDA_CHECK(cudaFreeHost(h_supp_pf));
  gpu_bf_supp.free_filter(); ht_supp.free_table(); bf_storage[1].free_filter();
  fascal_free_column(h_s_suppkey); fascal_free_column(h_s_region);
  fascal_free_seed_bitmap(h_bm, h_ts);
  fascal_free_column(h_lo_orderdate); fascal_free_column(h_lo_partkey);
  fascal_free_column(h_lo_suppkey);   fascal_free_column(h_lo_revenue);
  CUDA_CHECK(cudaFree(d_agg));
  fascal_arena_destroy();
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  delete[] streams;
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(t_start); cudaEventDestroy(t_load);   cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  return 0;
}
