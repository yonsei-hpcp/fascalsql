// TPC-H Q3 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Refactored: uses shared macros and CPU predicate helpers.
// Dimensions: orders (o_orderdate < 19950329 -> rid multi-col), customer (c_mktsegment = 'BUILDING'=1)
// Fact filter: l_shipdate > 19950329
// GROUP BY: orders row ID (high-cardinality), aggregation = SUM(l_extendedprice * (1 - l_discount))
// Result: top 10 by revenue DESC, then by o_orderdate ASC

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
#include <algorithm>

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
// GPU Kernel: scan lineitem, pred on l_shipdate, BF + HT joins, row-ID agg
// =============================================================================
template <int BT, int IPT>
__global__ void tpch_q3_kernel(
    int num_tuples, int batch_offset,
    int *d_l_orderkey_enc, int *d_l_shipdate,
    float *d_l_extendedprice, float *d_l_discount,
    int *d_o_custkey, int *d_o_orderkey, int *d_o_orderdate, int *d_o_shippriority,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_orders, uint32_t bloom_bits_orders,
    HtEntry *ht_orders, uint32_t ht_size_orders,
    const uint64_t *__restrict__ d_bloom_customer, uint32_t bloom_bits_customer,
    HtEntry *ht_customer, uint32_t ht_size_customer,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int l_orderkey_enc[IPT], l_shipdate[IPT];
  float l_extendedprice[IPT], l_discount[IPT];
  int o_custkey[IPT], o_orderkey[IPT], o_orderdate[IPT], o_shippriority[IPT];
  int orders_rid[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // Pred 0: l_shipdate > 19950329
    if (d_pred_flags[2] == 0) {
      BlockLoadSelect<int, BT, IPT>(d_l_shipdate + tile_offset, l_shipdate, selection_flags, num_tile_items);
      BlockPredAndGT<int, BT, IPT>(l_shipdate, 19950329, selection_flags, num_tile_items);
    }

    FASCAL_CHECK_ALIVE(IPT);

    // BF probe: orders
    BlockLoadSelect<int, BT, IPT>(d_l_orderkey_enc + tile_offset, l_orderkey_enc, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(l_orderkey_enc, selection_flags, num_tile_items,
        bloom_bits_orders, d_bloom_orders, 3);

    FASCAL_CHECK_ALIVE(IPT);

    // HT join: l_orderkey_enc -> orders (direct-address, multi-col payload via rid)
    BlockJoinProbePayloadDirect<BT, IPT>(
        l_orderkey_enc, selection_flags, num_tile_items,
        ht_orders, ht_size_orders, 0, orders_rid);
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if (selection_flags[i]) {
        int rid = orders_rid[i];
        o_custkey[i] = d_o_custkey[rid];
        o_orderkey[i] = d_o_orderkey[rid];
        o_orderdate[i] = d_o_orderdate[rid];
        o_shippriority[i] = d_o_shippriority[rid];
      }
    }

    // HT join: o_custkey -> customer (existence-only)
    BlockJoinProbeDirect<BT, IPT>(
        o_custkey, selection_flags, num_tile_items,
        ht_customer, ht_size_customer, 1);

    // Load aggregation columns
    BlockLoadSelect<float, BT, IPT>(d_l_extendedprice + tile_offset, l_extendedprice, selection_flags, num_tile_items);
    BlockLoadSelect<float, BT, IPT>(d_l_discount + tile_offset, l_discount, selection_flags, num_tile_items);

    // Row-ID GROUP BY: orders row ID as key
    #pragma unroll
    for (int i = 0; i < IPT; ++i)
      if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i])
        atomicAdd(&aggtable[orders_rid[i] * NUM_AGGREGATES], (double)l_extendedprice[i] * (1 - l_discount[i]));
  } while(0);
}

// =============================================================================
// Build kernel: orders (o_orderdate < 19950329, with customer BF semi-join)
// =============================================================================
template <int BT, int IPT>
__global__ void build_orders(
    int num_tuples, int batch_offset,
    const int *__restrict__ d_o_orderkey,
    const int *__restrict__ d_o_orderdate,
    const int *__restrict__ d_o_custkey,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry *__restrict__ d_filter_ht_customer, uint32_t ht_size_f_customer)
{
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int key[IPT], orderdate[IPT], custkey[IPT];

  BlockLoadSelect<int, BT, IPT>((int*)(d_o_orderkey + tile_offset), key, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_o_orderdate + tile_offset), orderdate, selection_flags, num_tile_items);
  BlockPredAndLT<int, BT, IPT>(orderdate, 19950329, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_o_custkey + tile_offset), custkey, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    if (!gpu_ht_probe_direct(custkey[i], d_filter_ht_customer, ht_size_f_customer, 1, nullptr)) continue;
    int row_idx = tile_offset + threadIdx.x + i * BT;
    gpu_ht_insert_one_direct(row_idx, row_idx, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(row_idx, 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// Build kernel: customer (c_mktsegment == 1 -> BUILDING)
// =============================================================================
template <int BT, int IPT>
__global__ void build_customer(
    int num_tuples, int batch_offset,
    const int *__restrict__ d_c_custkey,
    const int *__restrict__ d_c_mktsegment,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int key[IPT], mktseg[IPT];

  BlockLoadSelect<int, BT, IPT>((int*)(d_c_custkey + tile_offset), key, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_c_mktsegment + tile_offset), mktseg, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(mktseg, 1, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    gpu_ht_insert_one_direct(key[i], 0, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(key[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// CPU Predicate: l_shipdate > 19950329 + BF probe
// =============================================================================
static void cpu_predicate(int *h_data[], int offset, int cnt, uint32_t *bitmap,
                          fascal::runtime::BloomFilter *bf_orders) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);

  int *l_orderkey_enc = h_data[0], *l_shipdate = h_data[1];

  if (h_pred_flags[2] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, l_shipdate, offset, cnt, 19950330, INT_MAX, true, false);
  fascal::cpu_pred::bloom_probe_pass(bm, bs, l_orderkey_enc, offset, cnt, bf_orders);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
  const char *data_dir = getenv("FASCALSQL_DATA_DIR");
  if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
  if (argc >= 3) data_dir = argv[2];
  if (!data_dir) { fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n"); return 1; }

  printf("=== FaScalSQL: tpch_q3 ===\n");
  if (argc < 3) { printf("Usage: %s <num_tuples> <data_dir>\n", argv[0]); return 1; }
  data_dir = argv[2];

  // Auto-detect num_tuples
  int num_tuples = 0;
  { char p[512]; snprintf(p, sizeof(p), "%s/l_orderkey_enc.bin", data_dir);
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
  #define LOAD_COL_F(var, name) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    float *var = (float*)load_binary_column(path, num_tuples); \
    if (!var) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

  LOAD_COL(h_l_orderkey_enc, "l_orderkey_enc")
  LOAD_COL(h_l_shipdate, "l_shipdate")
  LOAD_COL_F(h_l_extendedprice, "l_extendedprice")
  LOAD_COL_F(h_l_discount, "l_discount")
  #undef LOAD_COL
  #undef LOAD_COL_F

  // Load dimension: customer
  int num_cust = 0;
  #define LOAD_DIM(var, name, cnt) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    int *var = load_binary_column_auto(path, &cnt); \
    if (!var || cnt == 0) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

  LOAD_DIM(h_c_custkey, "c_custkey", num_cust)
  LOAD_DIM(h_c_mktsegment, "c_mktsegment", num_cust)
  printf("Loaded customer: %d tuples\n", num_cust);

  // Load dimension: orders
  int num_orders = 0;
  LOAD_DIM(h_o_orderkey, "o_orderkey", num_orders)
  LOAD_DIM(h_o_orderdate, "o_orderdate", num_orders)
  LOAD_DIM(h_o_custkey, "o_custkey", num_orders)
  LOAD_DIM(h_o_shippriority, "o_shippriority", num_orders)
  printf("Loaded orders: %d tuples\n", num_orders);
  #undef LOAD_DIM

  CUDA_CHECK(cudaEventRecord(t_load));

  // CQO
  auto cqo_in = fascal_cqo_init(num_tuples);
  cqo_in.predicate_selectivities = {
    fascal_estimate_selectivity(h_l_shipdate, num_tuples, [](int v){ return v > 19950315; }),
    0.40, 0.50  // dimension-side (customer mktsegment, orders orderdate)
  };
  int cust_qual = (int)(fascal_estimate_eq_sel(h_c_mktsegment, num_cust, 0) * num_cust);  // BUILDING=0
  int orders_qual = (int)(fascal_estimate_selectivity(h_o_orderdate, num_orders, [](int v){ return v < 19950315; }) * num_orders);
  cqo_in.join_bf_selectivities = {
    fascal_estimate_join_bf_sel(cust_qual, num_cust),
    fascal_estimate_join_bf_sel(orders_qual, num_orders)
  };
  cqo_in.num_gpu_only_ops = 2;
  auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
  fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);

  int in_memory_repeats = 1;
  if (const char *e = getenv("FASCAL_IN_MEMORY_REPEATS")) { in_memory_repeats = atoi(e); if (in_memory_repeats < 1) in_memory_repeats = 1; }
  int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr) || cqo_out.bloom_off;

  // ODZC mapping
  auto odzc = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));
  uint32_t *h_bm = nullptr, *d_bm = nullptr; uint8_t *h_ts = nullptr, *d_ts = nullptr;
  fascal_alloc_seed_bitmap(num_tuples, &h_bm, &d_bm, &h_ts, &d_ts);

  #define ODZC_MAP(var, host, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; int *d_##var = nullptr; \
    fascal_odzc_map(odzc.get(), host, num_tuples, name, &mc_##var, &d_##var);
  #define ODZC_MAP_F(var, host, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; float *d_##var = nullptr; \
    { mc_##var = odzc->register_column(host, (size_t)num_tuples * sizeof(float)); \
      d_##var = static_cast<float*>(mc_##var.device_ptr); \
      if (!d_##var) { CUDA_CHECK(cudaMalloc(&d_##var, (size_t)num_tuples * sizeof(float))); \
                      CUDA_CHECK(cudaMemcpy(d_##var, host, (size_t)num_tuples * sizeof(float), cudaMemcpyHostToDevice)); } }

  ODZC_MAP(l_orderkey_enc, h_l_orderkey_enc, "l_orderkey_enc")
  ODZC_MAP(l_shipdate, h_l_shipdate, "l_shipdate")
  ODZC_MAP_F(l_extendedprice, h_l_extendedprice, "l_extendedprice")
  ODZC_MAP_F(l_discount, h_l_discount, "l_discount")
  #undef ODZC_MAP
  #undef ODZC_MAP_F

  // ODZC for dimension columns
  #define ODZC_DIM(var, host, n, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; \
    mc_##var = odzc->register_column(host, (size_t)(n) * sizeof(int)); \
    int *d_##var = static_cast<int*>(mc_##var.device_ptr); \
    if (!d_##var) { CUDA_CHECK(cudaMalloc(&d_##var, (size_t)(n)*sizeof(int))); \
                    CUDA_CHECK(cudaMemcpy(d_##var, host, (size_t)(n)*sizeof(int), cudaMemcpyHostToDevice)); }

  ODZC_DIM(c_custkey, h_c_custkey, num_cust, "c_custkey")
  ODZC_DIM(c_mktsegment, h_c_mktsegment, num_cust, "c_mktsegment")
  ODZC_DIM(o_orderkey, h_o_orderkey, num_orders, "o_orderkey")
  ODZC_DIM(o_orderdate, h_o_orderdate, num_orders, "o_orderdate")
  ODZC_DIM(o_custkey, h_o_custkey, num_orders, "o_custkey")
  ODZC_DIM(o_shippriority, h_o_shippriority, num_orders, "o_shippriority")
  #undef ODZC_DIM

  CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

  // Aggregation table: orders row ID -> revenue (row-ID GROUP BY)
  const size_t num_groups = (size_t)num_orders;
  double *d_agg;
  CUDA_CHECK(cudaMalloc(&d_agg, num_groups * NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * NUM_AGGREGATES * sizeof(double)));

  // Hash tables + Bloom filters
  GpuHashTable ht_cust = GpuHashTable::allocate_direct(num_cust);
  GpuBloomFilter bf_cust = GpuBloomFilter::allocate(num_cust, 3);
  CUDA_CHECK(cudaMemset(ht_cust.d_entries, 0xff, ht_cust.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(bf_cust.d_bits, 0, ((size_t)bf_cust.size_bits + 63) / 64 * sizeof(uint64_t)));

  GpuHashTable ht_orders_tbl = GpuHashTable::allocate_direct(num_orders);
  GpuBloomFilter bf_orders_gpu = GpuBloomFilter::allocate(num_orders, 3);
  fascal::runtime::BloomFilter bf_orders_cpu;
  // CPU BF must be built from host data (CPU hash ≠ GPU hash; copy_to_host is invalid).
  if (!bloom_off && 0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0])
      bf_orders_cpu = fascal::runtime::BloomFilter::allocate(num_orders, 3);
  CUDA_CHECK(cudaMemset(ht_orders_tbl.d_entries, 0xff, ht_orders_tbl.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(bf_orders_gpu.d_bits, 0, ((size_t)bf_orders_gpu.size_bits + 63) / 64 * sizeof(uint64_t)));

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
  BUILD_PREFILTER(ord, num_orders)
  #undef BUILD_PREFILTER

  // CPU prefilter: customer (c_mktsegment == 1 -> BUILDING)
  fascal_cpu_prefilter_run(num_cust, h_cust_pf, h_cust_ts,
      [&](int off, int cnt) {
          uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
          memset(&h_cust_pf[ws], 0, (we - ws) * sizeof(uint32_t));
          for (int i = 0; i < cnt; ++i) { int idx = off + i;
            if (h_c_mktsegment[idx] == 1) h_cust_pf[idx >> 5] |= (1u << (idx & 31)); }
      });

  // CPU prefilter: orders (o_orderdate < 19950329 + customer BF semi-join)
  fascal::runtime::BloomFilter bf_cust_cpu;
  // CPU BF must be built from host data (CPU hash ≠ GPU hash; copy_to_host is invalid).
  if (!bloom_off && 1 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[1])
      bf_cust_cpu = fascal::runtime::BloomFilter::allocate(num_cust, 3);
  // Will populate after customer build

  // Pipeline setup
  int *h_data[] = {h_l_orderkey_enc, h_l_shipdate, (int*)h_l_extendedprice, (int*)h_l_discount};
  fascal::runtime::BloomFilter h_bf_orders;
  h_bf_orders.bits = nullptr;

  const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
  const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
  cudaStream_t *streams = new cudaStream_t[N_STREAMS];
  for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
  int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
  cudaEventRecord(t0);

  // Build customer (morsel-batched)
  {
    int _build_nb = (num_cust + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_cust - _bo);
      int _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      build_customer<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
          num_cust, _bo, d_c_custkey, d_c_mktsegment,
          ht_cust.d_entries, ht_cust.ht_size, bf_cust.d_bits, bf_cust.size_bits, 3,
          d_cust_pf, d_cust_ts);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }
  // Build CPU customer BF from host data (CPU hash != GPU hash -- copy_to_host is invalid).
  // Insert c_custkey for BUILDING customers (c_mktsegment == 1) using CPU hash function.
  if (bf_cust_cpu.bits) {
      for (int _i = 0; _i < num_cust; ++_i)
          if (h_c_mktsegment[_i] == 1) bf_cust_cpu.insert(h_c_custkey[_i]);
  }

  // CPU prefilter: orders (o_orderdate < 19950329 + customer BF)
  fascal_cpu_prefilter_run(num_orders, h_ord_pf, h_ord_ts,
      [&](int off, int cnt) {
          uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
          memset(&h_ord_pf[ws], 0, (we - ws) * sizeof(uint32_t));
          for (int i = 0; i < cnt; ++i) { int idx = off + i;
            if (h_o_orderdate[idx] < 19950329) h_ord_pf[idx >> 5] |= (1u << (idx & 31)); }
          if (bf_cust_cpu.bits)
            bf_cust_cpu.probe_and_mask_packed(h_o_custkey + off, h_ord_pf, off, cnt);
      });

  // Build orders (morsel-batched)
  {
    int _build_nb = (num_orders + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_orders - _bo);
      int _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      build_orders<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
          num_orders, _bo, d_o_orderkey, d_o_orderdate, d_o_custkey,
          ht_orders_tbl.d_entries, ht_orders_tbl.ht_size, bf_orders_gpu.d_bits, bf_orders_gpu.size_bits, 3,
          d_ord_pf, d_ord_ts,
          ht_cust.d_entries, ht_cust.ht_size);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }
  // Build CPU orders BF from host data (CPU hash != GPU hash -- copy_to_host is invalid).
  // Insert row indices of orders passing date < 19950329 AND BUILDING customer semi-join.
  if (bf_orders_cpu.bits) {
      for (int _i = 0; _i < num_orders; ++_i) {
          if (h_o_orderdate[_i] < 19950329) {
              if (!bf_cust_cpu.bits || bf_cust_cpu.query(h_o_custkey[_i]))
                  bf_orders_cpu.insert(_i);
          }
      }
  }
  h_bf_orders = bf_orders_cpu;

  // Result pipeline
  auto run_kernel = [&]() {
    auto prefilter = [&](int bc, int bo, int tt) {
      fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
        [&](int off, int cnt) { cpu_predicate(h_data, off, cnt, h_bm, &h_bf_orders); }, bo, tt);
    };
    auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
      tpch_q3_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
        nt, bo, d_l_orderkey_enc, d_l_shipdate, d_l_extendedprice, d_l_discount,
        d_o_custkey, d_o_orderkey, d_o_orderdate, d_o_shippriority,
        d_bm, d_ts,
        bf_orders_gpu.d_bits, (bloom_off ? 0u : bf_orders_gpu.size_bits), ht_orders_tbl.d_entries, ht_orders_tbl.ht_size,
        bf_cust.d_bits, (bloom_off ? 0u : bf_cust.size_bits), ht_cust.d_entries, ht_cust.ht_size,
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
      CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * NUM_AGGREGATES * sizeof(double)));
      CUDA_CHECK(cudaEventRecord(t0)); run_kernel(); CUDA_CHECK(cudaEventRecord(t1));
      CUDA_CHECK(cudaDeviceSynchronize());
      float m; CUDA_CHECK(cudaEventElapsedTime(&m, t0, t1));
      total += m; if (m < best) best = m;
    }
    printf("In-memory result kernel: best %.3f ms avg %.3f ms (%d runs)\n", best, total / in_memory_repeats, in_memory_repeats);
  }

  // Output: collect non-zero groups, sort, top-10
  double *h_agg = (double*)malloc(num_groups * NUM_AGGREGATES * sizeof(double));
  CUDA_CHECK(cudaMemcpy(h_agg, d_agg, num_groups * NUM_AGGREGATES * sizeof(double), cudaMemcpyDeviceToHost));

  struct Q3Row { int orderkey; double revenue; int orderdate; int shippriority; };
  std::vector<Q3Row> rows;
  for (size_t k = 0; k < num_groups; ++k) {
    if (h_agg[k] == 0.0) continue;
    rows.push_back({h_o_orderkey[k], h_agg[k], h_o_orderdate[k], h_o_shippriority[k]});
  }
  std::sort(rows.begin(), rows.end(), [](const Q3Row &a, const Q3Row &b) {
    if (a.revenue != b.revenue) return a.revenue > b.revenue;
    return a.orderdate < b.orderdate;
  });
  int limit = std::min((int)rows.size(), 10);
  for (int i = 0; i < limit; ++i)
    { int d = rows[i].orderdate; printf("ROW: %d|%.4f|%d-%02d-%02d|%d\n", rows[i].orderkey, rows[i].revenue, d/10000, (d/100)%100, d%100, rows[i].shippriority); }
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
  odzc->unregister_column(mc_l_orderkey_enc); odzc->unregister_column(mc_l_shipdate);
  odzc->unregister_column(mc_l_extendedprice); odzc->unregister_column(mc_l_discount);
  odzc->unregister_column(mc_c_custkey); odzc->unregister_column(mc_c_mktsegment);
  odzc->unregister_column(mc_o_orderkey); odzc->unregister_column(mc_o_orderdate);
  odzc->unregister_column(mc_o_custkey); odzc->unregister_column(mc_o_shippriority);
  fascal_free_seed_bitmap(h_bm, h_ts);
  fascal_free_column(h_l_orderkey_enc); fascal_free_column(h_l_shipdate);
  fascal_free_column(h_l_extendedprice); fascal_free_column(h_l_discount);
  CUDA_CHECK(cudaFree(d_agg));
  CUDA_CHECK(cudaFreeHost(h_cust_pf)); CUDA_CHECK(cudaFreeHost(h_cust_ts));
  CUDA_CHECK(cudaFreeHost(h_ord_pf)); CUDA_CHECK(cudaFreeHost(h_ord_ts));
  bf_orders_cpu.free_filter(); bf_cust_cpu.free_filter();
  fascal_arena_destroy();
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  delete[] streams;
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(t_start); cudaEventDestroy(t_load);   cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  return 0;
}
