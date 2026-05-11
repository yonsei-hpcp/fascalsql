// TPC-H Q10 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
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
#include <unordered_map>
#include <string>
#include <cstring>
#include <limits>
#include <algorithm>
#include <tuple>

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
// GPU Kernel: scan lineitem, predicate on returnflag, 3 joins, Row-ID GROUP BY
// =============================================================================
template <int BT, int IPT>
__global__ void tpch_q10_kernel_lineitem(
    int num_tuples, int batch_offset,
    int *d_lineitem_l_orderkey_enc, int *d_lineitem_l_returnflag_enc,
    float *d_lineitem_l_extendedprice, float *d_lineitem_l_discount,
    int *d_customer_c_nationkey, int *d_customer_c_custkey,
    int *d_customer_c_name, int *d_customer_c_acctbal,
    int *d_nation_n_name,
    int *d_customer_c_address, int *d_customer_c_phone, int *d_customer_c_comment,
    int *d_customer_c_custkey_enc, int *d_nation_n_name_enc,
    const uint32_t *__restrict__ seed_bitmap, const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_orders, uint32_t bloom_size_bits_orders,
    HtEntry *ht_orders, uint32_t ht_size_orders,
    const uint64_t *__restrict__ d_bloom_customer, uint32_t bloom_size_bits_customer,
    HtEntry *ht_customer, uint32_t ht_size_customer,
    const uint64_t *__restrict__ d_bloom_nation, uint32_t bloom_size_bits_nation,
    HtEntry *ht_nation, uint32_t ht_size_nation,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int loc_l_orderkey_enc[IPT], loc_l_returnflag_enc[IPT];
  float loc_l_extendedprice[IPT], loc_l_discount[IPT];
  int loc_orders_o_custkey[IPT], loc_customer_rid[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // Predicate: l_returnflag_enc == 2 ('R')
    if (d_pred_flags[2] == 0) {
      BlockLoadSelect<int, BT, IPT>(d_lineitem_l_returnflag_enc + tile_offset, loc_l_returnflag_enc, selection_flags, num_tile_items);
      BlockPredAndEQ<int, BT, IPT>(loc_l_returnflag_enc, 2, selection_flags, num_tile_items);
    }
    FASCAL_CHECK_ALIVE(IPT);

    // Bloom probe: l_orderkey_enc -> orders
    BlockLoadSelect<int, BT, IPT>(d_lineitem_l_orderkey_enc + tile_offset, loc_l_orderkey_enc, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(loc_l_orderkey_enc, selection_flags, num_tile_items, bloom_size_bits_orders, d_bloom_orders, 3);

    FASCAL_CHECK_ALIVE(IPT);

    // Join 0: l_orderkey_enc -> orders [payload=o_custkey]
    BlockJoinProbePayloadDirect<BT, IPT>(loc_l_orderkey_enc, selection_flags, num_tile_items, ht_orders, ht_size_orders, 0, loc_orders_o_custkey);

    // Join 1: o_custkey -> customer [payload=rid]
    BlockJoinProbePayloadDirect<BT, IPT>(loc_orders_o_custkey, selection_flags, num_tile_items, ht_customer, ht_size_customer, 1, loc_customer_rid);

    // Load aggregation columns
    BlockLoadSelect<float, BT, IPT>(d_lineitem_l_extendedprice + tile_offset, loc_l_extendedprice, selection_flags, num_tile_items);
    BlockLoadSelect<float, BT, IPT>(d_lineitem_l_discount + tile_offset, loc_l_discount, selection_flags, num_tile_items);

    // Row-ID GROUP BY (customer rid as key)
    #pragma unroll
    for (int ITEM = 0; ITEM < IPT; ++ITEM) {
      if ((threadIdx.x + (BT * ITEM)) < num_tile_items && selection_flags[ITEM]) {
        int gk = loc_customer_rid[ITEM];
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], (double)loc_l_extendedprice[ITEM] * (1.0 - loc_l_discount[ITEM]));
      }
    }
  } while(0);
}

// =============================================================================
// CPU Predicate
// =============================================================================
static void cpu_predicate_lineitem(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bloom_filters[], int num_bloom_filters) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  int *l_orderkey_enc = h_data[0], *l_returnflag_enc = h_data[1];
  // Predicate on l_returnflag_enc == 2
  if (h_pred_flags[2] == 1)
    fascal::cpu_pred::avx2_eq_pass(bm, bs, l_returnflag_enc, offset, cnt, 2);
  // Bloom probe on l_orderkey_enc
  if (bloom_filters && 0 < num_bloom_filters)
    fascal::cpu_pred::bloom_probe_pass(bm, bs, l_orderkey_enc, offset, cnt, bloom_filters[0]);
}

// =============================================================================
// Build kernels
// =============================================================================
template <int BT, int IPT>
__global__ void tpch_q10_kernel_orders(
    int num_tuples, int batch_offset, const int *__restrict__ d_ok, const int *__restrict__ d_od, const int *__restrict__ d_ck,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_cust, uint32_t ht_size_f_cust) {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int loc_ok[IPT], loc_od[IPT], loc_ck[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_ok + tile_offset), loc_ok, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_od + tile_offset), loc_od, selection_flags, num_tile_items);
  BlockPredAndGTE<int, BT, IPT>(loc_od, 19940101, selection_flags, num_tile_items);
  BlockPredAndLT<int, BT, IPT>(loc_od, 19940401, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_ck + tile_offset), loc_ck, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe_direct(loc_ck[ITEM], d_filter_ht_cust, ht_size_f_cust, 1, nullptr)) continue;
    int row_idx = tile_offset + threadIdx.x + ITEM * BT;
    gpu_ht_insert_one_direct(row_idx, loc_ck[ITEM], 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(row_idx, 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_orders(int *h_ok, int *h_od, int *h_ck, int n, uint32_t *bm, uint8_t *ts,
    fascal::runtime::BloomFilter *bf_cust) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; if (h_od[idx] >= 19940101 && h_od[idx] < 19940401) bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (bf_cust && bf_cust->bits) bf_cust->probe_and_mask_packed(h_ck + off, bm, off, cnt);
  });
}

template <int BT, int IPT>
__global__ void tpch_q10_kernel_customer(
    int num_tuples, int batch_offset,
    const int *__restrict__ d_ck, const int *__restrict__ d_ck_enc, const int *__restrict__ d_cnk,
    const int *__restrict__ d_cname, const int *__restrict__ d_cacctbal,
    const int *__restrict__ d_caddr, const int *__restrict__ d_cphone, const int *__restrict__ d_ccomment,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_nation, uint32_t ht_size_f_nation) {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int loc_ck[IPT], loc_cnk[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_ck + tile_offset), loc_ck, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_cnk + tile_offset), loc_cnk, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe_direct(loc_cnk[ITEM], d_filter_ht_nation, ht_size_f_nation, 0, nullptr)) continue;
    int payload = tile_offset + threadIdx.x + ITEM * BT;
    gpu_ht_insert_one_direct(loc_ck[ITEM], payload, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_ck[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_customer(int *h_ck, int *h_cnk, int n, uint32_t *bm, uint8_t *ts,
    fascal::runtime::BloomFilter *bf_nation) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (bf_nation && bf_nation->bits) bf_nation->probe_and_mask_packed(h_cnk + off, bm, off, cnt);
  });
}

template <int BT, int IPT>
__global__ void tpch_q10_kernel_nation(
    int num_tuples, const int *__restrict__ d_nk, const int *__restrict__ d_name_enc, const int *__restrict__ d_name,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary) {
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_nk[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_nk + tile_offset), loc_nk, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int payload = tile_offset + threadIdx.x + ITEM * BT;
    gpu_ht_insert_one_direct(loc_nk[ITEM], payload, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(loc_nk[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_nation(int n, uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
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

  printf("=== FaScalSQL: tpch_q10 ===\n");
  if (argc < 3) { printf("Usage: %s <num_tuples> <data_dir>\n", argv[0]); return 1; }
  data_dir = argv[2];

  int num_tuples = 0;
  { char p[512]; snprintf(p, sizeof(p), "%s/l_orderkey_enc.bin", data_dir);
    int c = 0; int *t = load_binary_column_auto(p, &c); if (t) fascal_free_column(t);
    num_tuples = c; }
  if (num_tuples <= 0) { fprintf(stderr, "Cannot auto-detect num_tuples\n"); return 1; }
  printf("num_tuples=%d data_dir=%s\n", num_tuples, data_dir);

  cudaEvent_t t_start, t_load, t_query, t_result;
  CUDA_CHECK(cudaEventCreate(&t_start)); CUDA_CHECK(cudaEventCreate(&t_load));
  CUDA_CHECK(cudaEventCreate(&t_query));
  CUDA_CHECK(cudaEventCreate(&t_result)); CUDA_CHECK(cudaEventRecord(t_start));

  fascal_arena_init((size_t)num_tuples * sizeof(int) * 5);

  char path[512];
  #define LOAD_COL(var, name, type) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    type *var = (type*)load_binary_column(path, num_tuples); \
    if (!var) { fprintf(stderr, "Failed to load " name "\n"); return 1; }
  #define LOAD_DIM(var, name, cnt) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    int *var = load_binary_column_auto(path, &cnt); \
    if (!var || cnt == 0) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

  LOAD_COL(h_l_orderkey_enc, "l_orderkey_enc", int)
  LOAD_COL(h_l_returnflag_enc, "l_returnflag_enc", int)
  LOAD_COL(h_l_extendedprice, "l_extendedprice", float)
  LOAD_COL(h_l_discount, "l_discount", float)

  int num_nation = 0;
  LOAD_DIM(h_n_nk, "n_nationkey", num_nation) LOAD_DIM(h_n_name_enc, "n_name_enc", num_nation) LOAD_DIM(h_n_name, "n_name", num_nation)
  printf("Loaded nation: %d\n", num_nation);

  int num_customer = 0;
  LOAD_DIM(h_c_ck, "c_custkey", num_customer) LOAD_DIM(h_c_ck_enc, "c_custkey_enc", num_customer)
  LOAD_DIM(h_c_nk, "c_nationkey", num_customer) LOAD_DIM(h_c_name, "c_name", num_customer)
  LOAD_DIM(h_c_acctbal, "c_acctbal", num_customer) LOAD_DIM(h_c_addr, "c_address", num_customer)
  LOAD_DIM(h_c_phone, "c_phone", num_customer) LOAD_DIM(h_c_comment, "c_comment", num_customer)
  printf("Loaded customer: %d\n", num_customer);

  int num_orders = 0;
  LOAD_DIM(h_o_ok, "o_orderkey", num_orders) LOAD_DIM(h_o_od, "o_orderdate", num_orders) LOAD_DIM(h_o_ck, "o_custkey", num_orders)
  printf("Loaded orders: %d\n", num_orders);
  #undef LOAD_COL
  #undef LOAD_DIM
  CUDA_CHECK(cudaEventRecord(t_load));

  // CQO
  auto cqo_in = fascal_cqo_init(num_tuples);
  cqo_in.predicate_selectivities = {
    0.3, 0.3,  // dimension-side (orders orderdate, nation)
    fascal_estimate_eq_sel(h_l_returnflag_enc, num_tuples, 2)
  };
  // Dynamic BF selectivity: nation(all), customer(all), orders(o_orderdate filtered)
  int orders_qual = (int)(fascal_estimate_range_sel(h_o_od, num_orders, 19931001, 19940101) * num_orders);
  cqo_in.join_bf_selectivities = {
    fascal_estimate_join_bf_sel(num_nation, num_nation),
    fascal_estimate_join_bf_sel(num_customer, num_customer),
    fascal_estimate_join_bf_sel(orders_qual, num_orders)
  };
  cqo_in.num_gpu_only_ops = 3;
  auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
  fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);
  int in_memory_repeats = 1;
  if (const char *e = getenv("FASCAL_IN_MEMORY_REPEATS")) { in_memory_repeats = atoi(e); if (in_memory_repeats < 1) in_memory_repeats = 1; }
  int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr) || cqo_out.bloom_off;

  int *h_data[] = {h_l_orderkey_enc, h_l_returnflag_enc, (int*)h_l_extendedprice, (int*)h_l_discount};
  fascal::runtime::AFPManager afp_manager(0);
  int afp_filter_id = 0;
  if (h_pred_flags[2] == 1) {
    auto desc = fascal::runtime::afp::make_filter<int>(1, fascal::runtime::AFPManager::FilterOp::EQ, 2);
    afp_manager.register_filter(afp_filter_id++, desc);
  }

  fascal::runtime::BloomFilter bf_storage[3];
  fascal::runtime::BloomFilter *bf_array[3];
  bf_array[0] = (!bloom_off && 0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[2] : nullptr;
  bf_array[1] = nullptr; bf_array[2] = nullptr;

  // ODZC
  auto odzc = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));
  uint32_t *h_bm = nullptr, *d_bm = nullptr; uint8_t *h_ts = nullptr, *d_ts = nullptr;
  fascal_alloc_seed_bitmap(num_tuples, &h_bm, &d_bm, &h_ts, &d_ts);

  #define ODZC_INT(var, host, n, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; int *d_##var = nullptr; \
    fascal_odzc_map(odzc.get(), host, n, name, &mc_##var, &d_##var);
  #define ODZC_FLOAT(var, host, n, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; float *d_##var = nullptr; \
    { mc_##var = odzc->register_column(host, (size_t)(n) * sizeof(float)); \
      d_##var = static_cast<float*>(mc_##var.device_ptr); \
      if (!d_##var) { CUDA_CHECK(cudaMalloc(&d_##var, (size_t)(n)*sizeof(float))); CUDA_CHECK(cudaMemcpy(d_##var, host, (size_t)(n)*sizeof(float), cudaMemcpyHostToDevice)); } }
  #define ODZC_DIM(var, host, n, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; \
    mc_##var = odzc->register_column(host, (size_t)(n) * sizeof(int)); \
    int *d_##var = static_cast<int*>(mc_##var.device_ptr); \
    if (!d_##var) { CUDA_CHECK(cudaMalloc(&d_##var, (size_t)(n)*sizeof(int))); CUDA_CHECK(cudaMemcpy(d_##var, host, (size_t)(n)*sizeof(int), cudaMemcpyHostToDevice)); }

  ODZC_INT(l_ok, h_l_orderkey_enc, num_tuples, "l_orderkey_enc")
  ODZC_INT(l_rf, h_l_returnflag_enc, num_tuples, "l_returnflag_enc")
  ODZC_FLOAT(l_ep, h_l_extendedprice, num_tuples, "l_extendedprice")
  ODZC_FLOAT(l_disc, h_l_discount, num_tuples, "l_discount")
  ODZC_DIM(n_nk, h_n_nk, num_nation, "n_nk") ODZC_DIM(n_name_enc, h_n_name_enc, num_nation, "n_name_enc") ODZC_DIM(n_name, h_n_name, num_nation, "n_name")
  ODZC_DIM(c_ck, h_c_ck, num_customer, "c_ck") ODZC_DIM(c_ck_enc, h_c_ck_enc, num_customer, "c_ck_enc")
  ODZC_DIM(c_nk, h_c_nk, num_customer, "c_nk") ODZC_DIM(c_name, h_c_name, num_customer, "c_name")
  ODZC_DIM(c_acctbal, h_c_acctbal, num_customer, "c_acctbal") ODZC_DIM(c_addr, h_c_addr, num_customer, "c_addr")
  ODZC_DIM(c_phone, h_c_phone, num_customer, "c_phone") ODZC_DIM(c_comment, h_c_comment, num_customer, "c_comment")
  ODZC_DIM(o_ok, h_o_ok, num_orders, "o_ok") ODZC_DIM(o_od, h_o_od, num_orders, "o_od") ODZC_DIM(o_ck, h_o_ck, num_orders, "o_ck")
  #undef ODZC_INT
  #undef ODZC_FLOAT
  #undef ODZC_DIM

  CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

  // Aggregation: one group per customer row
  size_t num_groups = (size_t)num_customer;
  double *d_agg;
  CUDA_CHECK(cudaMalloc(&d_agg, num_groups * NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * NUM_AGGREGATES * sizeof(double)));

  // Build HTs
  // CPU BFs must be built from host data -- GPU and CPU use different hash functions,
  // so copy_to_host from GpuBloomFilter produces incorrect CPU BF probes (wrong bit positions).
  #define ALLOC_BUILD(name, n, alloc_fn, idx) \
    GpuHashTable ht_##name = GpuHashTable::alloc_fn(n); \
    GpuBloomFilter gpu_bf_##name = GpuBloomFilter::allocate(n, 3); \
    bf_storage[idx] = fascal::runtime::BloomFilter::allocate(n, 3); \
    CUDA_CHECK(cudaMemset(ht_##name.d_entries, 0xff, ht_##name.ht_size * sizeof(HtEntry))); \
    CUDA_CHECK(cudaMemset(gpu_bf_##name.d_bits, 0, ((size_t)gpu_bf_##name.size_bits + 63) / 64 * sizeof(uint64_t))); \
    uint32_t *h_pf_##name = nullptr, *d_pf_##name = nullptr; uint8_t *h_pts_##name = nullptr, *d_pts_##name = nullptr; \
    { size_t bw = ((size_t)(n) + 31) / 32, nt = ((size_t)(n) + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE; \
      CUDA_CHECK(cudaHostAlloc(&h_pf_##name, bw * 4, cudaHostAllocMapped)); memset(h_pf_##name, 0, bw * 4); \
      CUDA_CHECK(cudaHostGetDevicePointer(&d_pf_##name, h_pf_##name, 0)); \
      CUDA_CHECK(cudaHostAlloc(&h_pts_##name, nt, cudaHostAllocMapped)); memset(h_pts_##name, 0, nt); \
      CUDA_CHECK(cudaHostGetDevicePointer(&d_pts_##name, h_pts_##name, 0)); }

  ALLOC_BUILD(nation, num_nation, allocate_direct, 0)
  ALLOC_BUILD(customer, num_customer, allocate_direct, 1)
  ALLOC_BUILD(orders, num_orders, allocate_direct, 2)
  #undef ALLOC_BUILD

  // Pipeline
  const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
  const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
  cudaStream_t *streams = new cudaStream_t[N_STREAMS];
  for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
  int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

  auto run_kernel = [&]() {
    auto prefilter = [&](int bc, int bo, int tt) {
      fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
        [&](int off, int cnt) { cpu_predicate_lineitem(h_data, off, cnt, h_bm, bf_array, 3); }, bo, tt);
    };
    auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
      tpch_q10_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
        nt, bo, d_l_ok, d_l_rf, d_l_ep, d_l_disc,
        d_c_nk, d_c_ck, d_c_name, d_c_acctbal, d_n_name, d_c_addr, d_c_phone, d_c_comment, d_c_ck_enc, d_n_name_enc,
        d_bm, d_ts,
        gpu_bf_orders.d_bits, (bloom_off ? 0u : gpu_bf_orders.size_bits), ht_orders.d_entries, ht_orders.ht_size,
        gpu_bf_customer.d_bits, (bloom_off ? 0u : gpu_bf_customer.size_bits), ht_customer.d_entries, ht_customer.ht_size,
        gpu_bf_nation.d_bits, (bloom_off ? 0u : gpu_bf_nation.size_bits), ht_nation.d_entries, ht_nation.ht_size,
        d_agg);
    };
    { int nb = (num_tuples + GPU_BATCH - 1) / GPU_BATCH;
      for (int b = 0; b < nb; ++b) { int bo = b * GPU_BATCH, bc = std::min(GPU_BATCH, num_tuples - bo);
        prefilter(bc, bo, num_tuples); launch(num_tuples, bo, (bc + TILE_SIZE - 1) / TILE_SIZE, 0); } }
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  };

  // Build pipelines
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
  cudaEventRecord(t0);

  cpu_prefilter_nation(num_nation, h_pf_nation, h_pts_nation);
  // nation -- single launch (25 rows)
  tpch_q10_kernel_nation<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_nation + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
    num_nation, d_n_nk, d_n_name_enc, d_n_name, ht_nation.d_entries, ht_nation.ht_size, gpu_bf_nation.d_bits, gpu_bf_nation.size_bits, 3, d_pf_nation, d_pts_nation);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  // Build CPU nation BF from host data using CPU hash (all nations qualify).
  if (bf_storage[0].bits) {
      for (int _i = 0; _i < num_nation; ++_i)
          bf_storage[0].insert(h_n_nk[_i]);
  }

  // CPU customer prefilter uses the nation BF (now correctly built with CPU hash).
  cpu_prefilter_customer(h_c_ck, h_c_nk, num_customer, h_pf_customer, h_pts_customer, (bf_storage[0].bits ? &bf_storage[0] : nullptr));
  // customer -- morsel-batched
  {
    int _build_nb = (num_customer + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_customer - _bo), _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      tpch_q10_kernel_customer<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
        num_customer, _bo, d_c_ck, d_c_ck_enc, d_c_nk, d_c_name, d_c_acctbal, d_c_addr, d_c_phone, d_c_comment,
        ht_customer.d_entries, ht_customer.ht_size, gpu_bf_customer.d_bits, gpu_bf_customer.size_bits, 3,
        d_pf_customer, d_pts_customer, ht_nation.d_entries, ht_nation.ht_size);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }
  // Build CPU customer BF from host data using CPU hash.
  // Customers qualify if their c_nationkey is in the nation HT (= all customers since all nations qualify).
  if (bf_storage[1].bits) {
      for (int _i = 0; _i < num_customer; ++_i)
          if (bf_storage[0].query(h_c_nk[_i])) bf_storage[1].insert(h_c_ck[_i]);
  }

  // CPU orders prefilter uses the customer BF (now correctly built with CPU hash).
  cpu_prefilter_orders(h_o_ok, h_o_od, h_o_ck, num_orders, h_pf_orders, h_pts_orders, (bf_storage[1].bits ? &bf_storage[1] : nullptr));
  // orders -- morsel-batched
  {
    int _build_nb = (num_orders + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_orders - _bo), _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      tpch_q10_kernel_orders<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
        num_orders, _bo, d_o_ok, d_o_od, d_o_ck, ht_orders.d_entries, ht_orders.ht_size, gpu_bf_orders.d_bits, gpu_bf_orders.size_bits, 3,
        d_pf_orders, d_pts_orders, ht_customer.d_entries, ht_customer.ht_size);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }
  // Build CPU orders BF from host data using CPU hash.
  // Orders qualify if o_orderdate in [19940101, 19940401) AND o_custkey in customer BF.
  if (bf_storage[2].bits) {
      for (int _i = 0; _i < num_orders; ++_i) {
          if (h_o_od[_i] >= 19940101 && h_o_od[_i] < 19940401) {
              if (!bf_storage[1].bits || bf_storage[1].query(h_o_ck[_i]))
                  bf_storage[2].insert(_i);
          }
      }
  }

  // Result pipeline
  run_kernel();
  cudaEventRecord(t1);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaEventRecord(t_query));
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  printf("Kernel: %.3f ms\n", ms);

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

  // Output (top 20 by revenue DESC)
  static const char* nation_names[] = { "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "CHINA", "EGYPT", "ETHIOPIA", "FRANCE", "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN", "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU", "ROMANIA", "RUSSIA", "SAUDI ARABIA", "UNITED KINGDOM", "UNITED STATES", "VIETNAM" };
  double *h_agg = (double*)malloc(num_groups * NUM_AGGREGATES * sizeof(double));
  CUDA_CHECK(cudaMemcpy(h_agg, d_agg, num_groups * NUM_AGGREGATES * sizeof(double), cudaMemcpyDeviceToHost));
  vector<pair<tuple<long long>, size_t>> rows;
  for (size_t k = 0; k < num_groups; k++)
    if (h_agg[k * NUM_AGGREGATES + 0] != 0)
      rows.push_back({tuple<long long>{(long long)(h_agg[k * NUM_AGGREGATES + 0])}, k});
  sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return get<0>(a.first) > get<0>(b.first); });
  int limit = min((int)rows.size(), 20);
  for (int ri = 0; ri < limit; ri++) {
    size_t k = rows[ri].second;
    // Decode customer name
    static thread_local char cname_buf[64];
    int cname_idx = h_c_name[(int)k]; int cname_val = 1 + cname_idx;
    if (cname_val >= 1) snprintf(cname_buf, sizeof(cname_buf), "Customer#%09d", cname_val);
    else snprintf(cname_buf, sizeof(cname_buf), "UNKNOWN");
    // Nation name
    int nk = h_c_nk[(int)k];
    const char* nname = (nk >= 0 && nk < 25) ? nation_names[h_n_name[nk]] : "UNKNOWN";
    // Dict lookups for address, phone, comment
    auto load_dict = [&](const char* col_name) -> std::unordered_map<int, std::string>& {
      static std::unordered_map<std::string, std::unordered_map<int, std::string>> cache;
      auto &dict = cache[col_name];
      if (dict.empty()) {
        char ps[1024], po64[1024], po32[1024];
        snprintf(ps,   sizeof(ps),   "%s/customer_%s_dict_strings.bin", data_dir, col_name);
        snprintf(po64, sizeof(po64), "%s/customer_%s_dict_offsets64.bin", data_dir, col_name);
        snprintf(po32, sizeof(po32), "%s/customer_%s_dict_offsets.bin", data_dir, col_name);
        FILE* fs = fopen(ps, "rb");
        if (fs) {
          fseek(fs, 0, SEEK_END); long sb = ftell(fs); rewind(fs);
          std::vector<char> blob((size_t)sb);
          fread(blob.data(), 1, (size_t)sb, fs); fclose(fs);
          FILE* fo = fopen(po64, "rb");
          if (fo) {
            fseek(fo, 0, SEEK_END); long ob = ftell(fo); rewind(fo);
            size_t n = (size_t)ob / sizeof(int64_t);
            std::vector<int64_t> off(n);
            fread(off.data(), 1, n * sizeof(int64_t), fo); fclose(fo);
            dict.reserve(n);
            for (size_t i = 0; i < n; ++i) dict[(int)i] = std::string(blob.data() + off[i]);
          } else if ((fo = fopen(po32, "rb"))) {
            fseek(fo, 0, SEEK_END); long ob = ftell(fo); rewind(fo);
            size_t n = (size_t)ob / sizeof(int32_t);
            std::vector<int32_t> off(n);
            fread(off.data(), 1, n * sizeof(int32_t), fo); fclose(fo);
            dict.reserve(n);
            for (size_t i = 0; i < n; ++i) dict[(int)i] = std::string(blob.data() + off[i]);
          }
        }
      }
      return dict;
    };
    auto &addr_dict = load_dict("c_address");
    auto &phone_dict = load_dict("c_phone");
    auto &comment_dict = load_dict("c_comment");
    auto dict_get = [](std::unordered_map<int, std::string>& d, int key) -> const char* {
      auto it = d.find(key); return it != d.end() ? it->second.c_str() : "UNKNOWN"; };
    printf("ROW: %d|%s|%.4f|%.2f|%s|%s|%s|%s\n",
      h_c_ck[(int)k], cname_buf, h_agg[k * NUM_AGGREGATES + 0],
      ((double)(long long)h_c_acctbal[(int)k]) / 100.0, nname,
      dict_get(addr_dict, h_c_addr[(int)k]),
      dict_get(phone_dict, h_c_phone[(int)k]),
      dict_get(comment_dict, h_c_comment[(int)k]));
  }
  free(h_agg);

  CUDA_CHECK(cudaEventRecord(t_result)); CUDA_CHECK(cudaDeviceSynchronize());
  { float ml, mb, mr;
    CUDA_CHECK(cudaEventElapsedTime(&ml, t_start, t_load));
    CUDA_CHECK(cudaEventElapsedTime(&mb, t_load, t_query));
    /* merged into query */
    CUDA_CHECK(cudaEventElapsedTime(&mr, t_query, t_result));
    printf("Timing: load=%.1fms query=%.1fms result=%.1fms total=%.1fms\n", ml, mb, mr, ml+mb+mr); }

  // Cleanup
  odzc->unregister_column(mc_l_ok); odzc->unregister_column(mc_l_rf);
  odzc->unregister_column(mc_l_ep); odzc->unregister_column(mc_l_disc);
  odzc->unregister_column(mc_n_nk); odzc->unregister_column(mc_n_name_enc); odzc->unregister_column(mc_n_name);
  odzc->unregister_column(mc_c_ck); odzc->unregister_column(mc_c_ck_enc); odzc->unregister_column(mc_c_nk);
  odzc->unregister_column(mc_c_name); odzc->unregister_column(mc_c_acctbal);
  odzc->unregister_column(mc_c_addr); odzc->unregister_column(mc_c_phone); odzc->unregister_column(mc_c_comment);
  odzc->unregister_column(mc_o_ok); odzc->unregister_column(mc_o_od); odzc->unregister_column(mc_o_ck);
  fascal_free_seed_bitmap(h_bm, h_ts);
  CUDA_CHECK(cudaFree(d_agg));
  fascal_free_column(h_l_orderkey_enc); fascal_free_column(h_l_returnflag_enc);
  fascal_free_column(h_l_extendedprice); fascal_free_column(h_l_discount);
  fascal_free_column(h_n_nk); fascal_free_column(h_n_name_enc); fascal_free_column(h_n_name);
  fascal_free_column(h_c_ck); fascal_free_column(h_c_ck_enc); fascal_free_column(h_c_nk);
  fascal_free_column(h_c_name); fascal_free_column(h_c_acctbal);
  fascal_free_column(h_c_addr); fascal_free_column(h_c_phone); fascal_free_column(h_c_comment);
  fascal_free_column(h_o_ok); fascal_free_column(h_o_od); fascal_free_column(h_o_ck);
  CUDA_CHECK(cudaFreeHost(h_pf_nation)); CUDA_CHECK(cudaFreeHost(h_pts_nation)); gpu_bf_nation.free_filter(); ht_nation.free_table(); bf_storage[0].free_filter();
  CUDA_CHECK(cudaFreeHost(h_pf_customer)); CUDA_CHECK(cudaFreeHost(h_pts_customer)); gpu_bf_customer.free_filter(); ht_customer.free_table(); bf_storage[1].free_filter();
  CUDA_CHECK(cudaFreeHost(h_pf_orders)); CUDA_CHECK(cudaFreeHost(h_pts_orders)); gpu_bf_orders.free_filter(); ht_orders.free_table(); bf_storage[2].free_filter();
  fascal_arena_destroy();
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  delete[] streams;
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(t_start); cudaEventDestroy(t_load);   cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  return 0;
}
