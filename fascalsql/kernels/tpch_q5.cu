// TPC-H Q5 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Refactored: uses shared macros and CPU predicate helpers.
// Dimensions: region (r_name=4=MIDDLE EAST) -> nation -> supplier, customer -> orders (date filter)
// Join chain: lineitem -> orders (date+custkey), supplier (nationkey), nation (multi-col), region
// Post-join: customer.c_nationkey == supplier.s_nationkey
// GROUP BY: nation row ID -> SUM(l_extendedprice * (1 - l_discount))

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
// GPU Kernel: scan lineitem, BF probes, HT joins, post-join check, agg
// =============================================================================
template <int BT, int IPT>
__global__ void tpch_q5_kernel(
    int num_tuples, int batch_offset,
    int *d_l_orderkey_enc, int *d_l_suppkey,
    float *d_l_extendedprice, float *d_l_discount,
    int *d_n_regionkey, int *d_n_name, int *d_n_name_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_orders, uint32_t bloom_bits_orders,
    HtEntry *ht_orders, uint32_t ht_size_orders,
    const uint64_t *__restrict__ d_bloom_supplier, uint32_t bloom_bits_supplier,
    HtEntry *ht_supplier, uint32_t ht_size_supplier,
    const uint64_t *__restrict__ d_bloom_nation, uint32_t bloom_bits_nation,
    HtEntry *ht_nation, uint32_t ht_size_nation,
    const uint64_t *__restrict__ d_bloom_region, uint32_t bloom_bits_region,
    HtEntry *ht_region, uint32_t ht_size_region,
    const uint64_t *__restrict__ d_bloom_customer, uint32_t bloom_bits_customer,
    HtEntry *ht_customer, uint32_t ht_size_customer,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int l_orderkey_enc[IPT], l_suppkey[IPT];
  float l_extendedprice[IPT], l_discount[IPT];
  int o_custkey[IPT], s_nationkey[IPT], n_regionkey[IPT], n_name[IPT], n_name_enc[IPT];
  int nation_rid[IPT], c_nationkey[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // BF probe: orders
    BlockLoadSelect<int, BT, IPT>(d_l_orderkey_enc + tile_offset, l_orderkey_enc, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(l_orderkey_enc, selection_flags, num_tile_items,
        bloom_bits_orders, d_bloom_orders, 3);

    // BF probe: supplier
    BlockLoadSelect<int, BT, IPT>(d_l_suppkey + tile_offset, l_suppkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(l_suppkey, selection_flags, num_tile_items,
        bloom_bits_supplier, d_bloom_supplier, 3);

    FASCAL_CHECK_ALIVE(IPT);

    // Join 0: l_orderkey_enc -> orders (direct-address) [payload=o_custkey]
    BlockJoinProbePayloadDirect<BT, IPT>(
        l_orderkey_enc, selection_flags, num_tile_items,
        ht_orders, ht_size_orders, 0, o_custkey);

    // Join 1: l_suppkey -> supplier [payload=s_nationkey]
    BlockJoinProbePayloadDirect<BT, IPT>(
        l_suppkey, selection_flags, num_tile_items,
        ht_supplier, ht_size_supplier, 1, s_nationkey);

    // Join 2: s_nationkey -> nation [multi-col: keep rid]
    BlockJoinProbePayloadDirect<BT, IPT>(
        s_nationkey, selection_flags, num_tile_items,
        ht_nation, ht_size_nation, 0, nation_rid);
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if (selection_flags[i]) {
        int rid = nation_rid[i];
        n_regionkey[i] = d_n_regionkey[rid];
        n_name[i] = d_n_name[rid];
        n_name_enc[i] = d_n_name_enc[rid];
      }
    }

    // Join 3: n_regionkey -> region (existence-only)
    BlockJoinProbeDirect<BT, IPT>(
        n_regionkey, selection_flags, num_tile_items,
        ht_region, ht_size_region, 0);

    // Join 4: o_custkey -> customer [payload=c_nationkey]
    BlockJoinProbePayloadDirect<BT, IPT>(
        o_custkey, selection_flags, num_tile_items,
        ht_customer, ht_size_customer, 1, c_nationkey);

    // Load aggregation columns
    BlockLoadSelect<float, BT, IPT>(d_l_extendedprice + tile_offset, l_extendedprice, selection_flags, num_tile_items);
    BlockLoadSelect<float, BT, IPT>(d_l_discount + tile_offset, l_discount, selection_flags, num_tile_items);

    // Post-join check: customer.c_nationkey == supplier.s_nationkey
    #pragma unroll
    for (int i = 0; i < IPT; ++i)
      if (selection_flags[i] && c_nationkey[i] != s_nationkey[i])
        selection_flags[i] = 0;

    // Row-ID GROUP BY: nation row ID
    #pragma unroll
    for (int i = 0; i < IPT; ++i)
      if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i])
        atomicAdd(&aggtable[nation_rid[i] * NUM_AGGREGATES], (double)l_extendedprice[i] * (1 - l_discount[i]));
  } while(0);
}

// =============================================================================
// Build kernel: region (r_name == 4 -> MIDDLE EAST)
// =============================================================================
template <int BT, int IPT>
__global__ void build_region(
    int num_tuples,
    const int *__restrict__ d_r_regionkey, const int *__restrict__ d_r_name,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int key[IPT], name[IPT];

  BlockLoadSelect<int, BT, IPT>((int*)(d_r_regionkey + tile_offset), key, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_r_name + tile_offset), name, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(name, 4, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    int payload = tile_offset + threadIdx.x + i * BT;
    gpu_ht_insert_one_direct(key[i], payload, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(key[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// Build kernel: nation (upstream BF probe on n_regionkey -> region)
// =============================================================================
template <int BT, int IPT>
__global__ void build_nation(
    int num_tuples,
    const int *__restrict__ d_n_nationkey, const int *__restrict__ d_n_name_enc,
    const int *__restrict__ d_n_regionkey, const int *__restrict__ d_n_name,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry *__restrict__ d_filter_ht_region, uint32_t ht_size_f_region)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int key[IPT], name_enc[IPT], regionkey[IPT], name[IPT];

  BlockLoadSelect<int, BT, IPT>((int*)(d_n_nationkey + tile_offset), key, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_n_name_enc + tile_offset), name_enc, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_n_regionkey + tile_offset), regionkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_n_name + tile_offset), name, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    if (!gpu_ht_probe_direct(regionkey[i], d_filter_ht_region, ht_size_f_region, 0, nullptr)) continue;
    int payload = tile_offset + threadIdx.x + i * BT;
    gpu_ht_insert_one_direct(key[i], payload, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(key[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// Build kernel: supplier (upstream BF probe on s_nationkey -> nation)
// =============================================================================
template <int BT, int IPT>
__global__ void build_supplier(
    int num_tuples, int batch_offset,
    const int *__restrict__ d_s_suppkey, const int *__restrict__ d_s_nationkey,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry *__restrict__ d_filter_ht_nation, uint32_t ht_size_f_nation)
{
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int key[IPT], nationkey[IPT];

  BlockLoadSelect<int, BT, IPT>((int*)(d_s_suppkey + tile_offset), key, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_nationkey + tile_offset), nationkey, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    if (!gpu_ht_probe_direct(nationkey[i], d_filter_ht_nation, ht_size_f_nation, 0, nullptr)) continue;
    gpu_ht_insert_one_direct(key[i], nationkey[i], 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(key[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// Build kernel: customer (no pred, just load)
// =============================================================================
template <int BT, int IPT>
__global__ void build_customer(
    int num_tuples, int batch_offset,
    const int *__restrict__ d_c_custkey, const int *__restrict__ d_c_nationkey,
    HtEntry *d_ht, uint32_t ht_size,
    uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int key[IPT], nationkey[IPT];

  BlockLoadSelect<int, BT, IPT>((int*)(d_c_custkey + tile_offset), key, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_c_nationkey + tile_offset), nationkey, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    gpu_ht_insert_one_direct(key[i], nationkey[i], 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(key[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// Build kernel: orders (o_orderdate BETWEEN 19940101 AND 19941231, customer BF)
// =============================================================================
template <int BT, int IPT>
__global__ void build_orders(
    int num_tuples, int batch_offset,
    const int *__restrict__ d_o_orderkey, const int *__restrict__ d_o_orderdate,
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
  BlockPredAndGTE<int, BT, IPT>(orderdate, 19940101, selection_flags, num_tile_items);
  BlockPredAndLT<int, BT, IPT>(orderdate, 19950101, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_o_custkey + tile_offset), custkey, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    if (!gpu_ht_probe_direct(custkey[i], d_filter_ht_customer, ht_size_f_customer, 1, nullptr)) continue;
    int row_idx = tile_offset + threadIdx.x + i * BT;
    gpu_ht_insert_one_direct(row_idx, custkey[i], 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(row_idx, 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// =============================================================================
// CPU Predicate: BF probes on l_orderkey_enc and l_suppkey
// =============================================================================
static void cpu_predicate(int *h_data[], int offset, int cnt, uint32_t *bitmap,
                          fascal::runtime::BloomFilter *bf_orders,
                          fascal::runtime::BloomFilter *bf_supplier) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);

  int *l_orderkey_enc = h_data[0], *l_suppkey = h_data[1];

  fascal::cpu_pred::bloom_probe_pass(bm, bs, l_orderkey_enc, offset, cnt, bf_orders);
  fascal::cpu_pred::bloom_probe_pass(bm, bs, l_suppkey, offset, cnt, bf_supplier);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
  const char *data_dir = getenv("FASCALSQL_DATA_DIR");
  if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
  if (argc >= 3) data_dir = argv[2];
  if (!data_dir) { fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n"); return 1; }

  printf("=== FaScalSQL: tpch_q5 ===\n");
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
  fascal_arena_init((size_t)num_tuples * sizeof(int) * 6);

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
  LOAD_COL(h_l_suppkey, "l_suppkey")
  LOAD_COL_F(h_l_extendedprice, "l_extendedprice")
  LOAD_COL_F(h_l_discount, "l_discount")
  #undef LOAD_COL
  #undef LOAD_COL_F

  // Load dimensions
  #define LOAD_DIM(var, name, cnt) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    int *var = load_binary_column_auto(path, &cnt); \
    if (!var || cnt == 0) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

  int num_region = 0;
  LOAD_DIM(h_r_regionkey, "r_regionkey", num_region)
  LOAD_DIM(h_r_name, "r_name", num_region)
  printf("Loaded region: %d tuples\n", num_region);

  int num_nation = 0;
  LOAD_DIM(h_n_nationkey, "n_nationkey", num_nation)
  LOAD_DIM(h_n_name_enc, "n_name_enc", num_nation)
  LOAD_DIM(h_n_regionkey, "n_regionkey", num_nation)
  LOAD_DIM(h_n_name, "n_name", num_nation)
  printf("Loaded nation: %d tuples\n", num_nation);

  int num_supplier = 0;
  LOAD_DIM(h_s_suppkey, "s_suppkey", num_supplier)
  LOAD_DIM(h_s_nationkey, "s_nationkey", num_supplier)
  printf("Loaded supplier: %d tuples\n", num_supplier);

  int num_customer = 0;
  LOAD_DIM(h_c_custkey, "c_custkey", num_customer)
  LOAD_DIM(h_c_nationkey, "c_nationkey", num_customer)
  printf("Loaded customer: %d tuples\n", num_customer);

  int num_orders = 0;
  LOAD_DIM(h_o_orderkey, "o_orderkey", num_orders)
  LOAD_DIM(h_o_orderdate, "o_orderdate", num_orders)
  LOAD_DIM(h_o_custkey, "o_custkey", num_orders)
  printf("Loaded orders: %d tuples\n", num_orders);
  #undef LOAD_DIM

  CUDA_CHECK(cudaEventRecord(t_load));

  // CQO
  auto cqo_in = fascal_cqo_init(num_tuples);
  // Dim joins: region(r_name=ASIA=2), nation(filtered by region), supplier(all), customer(all), orders(date filter)
  int region_qual = (int)(fascal_estimate_eq_sel(h_r_name, num_region, 2) * num_region);
  int nation_qual = (int)(fascal_estimate_eq_sel(h_n_regionkey, num_nation, 2) * num_nation);  // nations in ASIA
  int orders_qual = (int)(fascal_estimate_range_sel(h_o_orderdate, num_orders, 19940101, 19941231) * num_orders);
  cqo_in.join_bf_selectivities = {
    fascal_estimate_join_bf_sel(region_qual, num_region),
    fascal_estimate_join_bf_sel(nation_qual, num_nation),
    fascal_estimate_join_bf_sel(num_supplier, num_supplier),
    fascal_estimate_join_bf_sel(num_customer, num_customer),
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
  ODZC_MAP(l_suppkey, h_l_suppkey, "l_suppkey")
  ODZC_MAP_F(l_extendedprice, h_l_extendedprice, "l_extendedprice")
  ODZC_MAP_F(l_discount, h_l_discount, "l_discount")
  #undef ODZC_MAP
  #undef ODZC_MAP_F

  #define ODZC_DIM(var, host, n, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; \
    mc_##var = odzc->register_column(host, (size_t)(n) * sizeof(int)); \
    int *d_##var = static_cast<int*>(mc_##var.device_ptr); \
    if (!d_##var) { CUDA_CHECK(cudaMalloc(&d_##var, (size_t)(n)*sizeof(int))); \
                    CUDA_CHECK(cudaMemcpy(d_##var, host, (size_t)(n)*sizeof(int), cudaMemcpyHostToDevice)); }

  ODZC_DIM(r_regionkey, h_r_regionkey, num_region, "r_regionkey")
  ODZC_DIM(r_name, h_r_name, num_region, "r_name")
  ODZC_DIM(n_nationkey, h_n_nationkey, num_nation, "n_nationkey")
  ODZC_DIM(n_name_enc, h_n_name_enc, num_nation, "n_name_enc")
  ODZC_DIM(n_regionkey, h_n_regionkey, num_nation, "n_regionkey")
  ODZC_DIM(n_name, h_n_name, num_nation, "n_name")
  ODZC_DIM(s_suppkey, h_s_suppkey, num_supplier, "s_suppkey")
  ODZC_DIM(s_nationkey, h_s_nationkey, num_supplier, "s_nationkey")
  ODZC_DIM(c_custkey, h_c_custkey, num_customer, "c_custkey")
  ODZC_DIM(c_nationkey, h_c_nationkey, num_customer, "c_nationkey")
  ODZC_DIM(o_orderkey, h_o_orderkey, num_orders, "o_orderkey")
  ODZC_DIM(o_orderdate, h_o_orderdate, num_orders, "o_orderdate")
  ODZC_DIM(o_custkey, h_o_custkey, num_orders, "o_custkey")
  #undef ODZC_DIM

  CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

  // Aggregation table: nation row ID (25 nations)
  const size_t num_groups = (size_t)num_nation;
  double *d_agg;
  CUDA_CHECK(cudaMalloc(&d_agg, num_groups * NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * NUM_AGGREGATES * sizeof(double)));

  // Hash tables + Bloom filters (5 dimensions)
  // bf_storage[0]=region, [1]=nation, [2]=supplier, [3]=customer, [4]=orders
  fascal::runtime::BloomFilter bf_cpu[5];

  GpuHashTable ht_region_tbl = GpuHashTable::allocate_direct(num_region);
  GpuBloomFilter bf_region_gpu = GpuBloomFilter::allocate(num_region, 3);
  if (!bloom_off && 0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0])
      bf_cpu[0] = fascal::runtime::BloomFilter::allocate(num_region, 3);
  CUDA_CHECK(cudaMemset(ht_region_tbl.d_entries, 0xff, ht_region_tbl.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(bf_region_gpu.d_bits, 0, ((size_t)bf_region_gpu.size_bits + 63) / 64 * sizeof(uint64_t)));

  GpuHashTable ht_nation_tbl = GpuHashTable::allocate_direct(num_nation);
  GpuBloomFilter bf_nation_gpu = GpuBloomFilter::allocate(num_nation, 3);
  if (!bloom_off && 1 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[1])
      bf_cpu[1] = fascal::runtime::BloomFilter::allocate(num_nation, 3);
  CUDA_CHECK(cudaMemset(ht_nation_tbl.d_entries, 0xff, ht_nation_tbl.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(bf_nation_gpu.d_bits, 0, ((size_t)bf_nation_gpu.size_bits + 63) / 64 * sizeof(uint64_t)));

  GpuHashTable ht_supplier_tbl = GpuHashTable::allocate_direct(num_supplier);
  GpuBloomFilter bf_supplier_gpu = GpuBloomFilter::allocate(num_supplier, 3);
  if (!bloom_off && 2 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[2])
      bf_cpu[2] = fascal::runtime::BloomFilter::allocate(num_supplier, 3);
  CUDA_CHECK(cudaMemset(ht_supplier_tbl.d_entries, 0xff, ht_supplier_tbl.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(bf_supplier_gpu.d_bits, 0, ((size_t)bf_supplier_gpu.size_bits + 63) / 64 * sizeof(uint64_t)));

  GpuHashTable ht_customer_tbl = GpuHashTable::allocate_direct(num_customer);
  GpuBloomFilter bf_customer_gpu = GpuBloomFilter::allocate(num_customer, 3);
  if (!bloom_off && 3 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[3])
      bf_cpu[3] = fascal::runtime::BloomFilter::allocate(num_customer, 3);
  CUDA_CHECK(cudaMemset(ht_customer_tbl.d_entries, 0xff, ht_customer_tbl.ht_size * sizeof(HtEntry)));
  CUDA_CHECK(cudaMemset(bf_customer_gpu.d_bits, 0, ((size_t)bf_customer_gpu.size_bits + 63) / 64 * sizeof(uint64_t)));

  GpuHashTable ht_orders_tbl = GpuHashTable::allocate_direct(num_orders);
  GpuBloomFilter bf_orders_gpu = GpuBloomFilter::allocate(num_orders, 3);
  if (!bloom_off && 4 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[4])
      bf_cpu[4] = fascal::runtime::BloomFilter::allocate(num_orders, 3);
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

  BUILD_PREFILTER(reg, num_region)
  BUILD_PREFILTER(nat, num_nation)
  BUILD_PREFILTER(supp, num_supplier)
  BUILD_PREFILTER(cust, num_customer)
  BUILD_PREFILTER(ord, num_orders)
  #undef BUILD_PREFILTER

  // CPU prefilter: region (r_name == 4)
  fascal_cpu_prefilter_run(num_region, h_reg_pf, h_reg_ts,
      [&](int off, int cnt) {
          uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
          memset(&h_reg_pf[ws], 0, (we - ws) * sizeof(uint32_t));
          for (int i = 0; i < cnt; ++i) { int idx = off + i;
            if (h_r_name[idx] == 4) h_reg_pf[idx >> 5] |= (1u << (idx & 31)); }
      });

  // CPU prefilter: nation (upstream BF on n_regionkey)
  auto prefilter_all_with_bf = [](int n, uint32_t *pf, uint8_t *ts, int *fk_col, fascal::runtime::BloomFilter *bf) {
    fascal_cpu_prefilter_run(n, pf, ts,
        [&](int off, int cnt) {
            uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
            memset(&pf[ws], 0, (we - ws) * sizeof(uint32_t));
            for (int i = 0; i < cnt; ++i) { int idx = off + i;
              pf[idx >> 5] |= (1u << (idx & 31)); }
            if (bf && bf->bits) bf->probe_and_mask_packed(fk_col + off, pf, off, cnt);
        });
  };

  // CPU prefilter: supplier (upstream BF on s_nationkey)
  // CPU prefilter: customer (all pass)
  auto prefilter_all = [](int n, uint32_t *pf, uint8_t *ts) {
    fascal_cpu_prefilter_run(n, pf, ts,
        [&](int off, int cnt) {
            uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
            memset(&pf[ws], 0, (we - ws) * sizeof(uint32_t));
            for (int i = 0; i < cnt; ++i) { int idx = off + i;
              pf[idx >> 5] |= (1u << (idx & 31)); }
        });
  };

  // CPU prefilter: orders (o_orderdate BETWEEN, customer BF)
  auto prefilter_orders = [&]() {
    fascal_cpu_prefilter_run(num_orders, h_ord_pf, h_ord_ts,
        [&](int off, int cnt) {
            uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
            memset(&h_ord_pf[ws], 0, (we - ws) * sizeof(uint32_t));
            for (int i = 0; i < cnt; ++i) { int idx = off + i;
              if (h_o_orderdate[idx] >= 19940101 && h_o_orderdate[idx] < 19950101)
                h_ord_pf[idx >> 5] |= (1u << (idx & 31)); }
            if (bf_cpu[3].bits) bf_cpu[3].probe_and_mask_packed(h_o_custkey + off, h_ord_pf, off, cnt);
        });
  };

  // Pipeline setup
  int *h_data[] = {h_l_orderkey_enc, h_l_suppkey, (int*)h_l_extendedprice, (int*)h_l_discount};

  const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
  const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
  cudaStream_t *streams = new cudaStream_t[N_STREAMS];
  for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
  int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
  cudaEventRecord(t0);

  // Build region
  build_region<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_region + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_region, d_r_regionkey, d_r_name,
      ht_region_tbl.d_entries, ht_region_tbl.ht_size, bf_region_gpu.d_bits, bf_region_gpu.size_bits, 3,
      d_reg_pf, d_reg_ts);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (bf_cpu[0].bits) bf_region_gpu.copy_to_host(bf_cpu[0].bits, ((size_t)(bf_cpu[0].size_bits + 63) / 64) * sizeof(uint64_t));

  // Build nation (upstream BF: region)
  prefilter_all_with_bf(num_nation, h_nat_pf, h_nat_ts, h_n_regionkey, (bf_cpu[0].bits ? &bf_cpu[0] : nullptr));
  build_nation<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_nation + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_nation, d_n_nationkey, d_n_name_enc, d_n_regionkey, d_n_name,
      ht_nation_tbl.d_entries, ht_nation_tbl.ht_size, bf_nation_gpu.d_bits, bf_nation_gpu.size_bits, 3,
      d_nat_pf, d_nat_ts,
      ht_region_tbl.d_entries, ht_region_tbl.ht_size);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (bf_cpu[1].bits) bf_nation_gpu.copy_to_host(bf_cpu[1].bits, ((size_t)(bf_cpu[1].size_bits + 63) / 64) * sizeof(uint64_t));

  // Build supplier (upstream BF: nation) -- morsel-batched
  prefilter_all_with_bf(num_supplier, h_supp_pf, h_supp_ts, h_s_nationkey, (bf_cpu[1].bits ? &bf_cpu[1] : nullptr));
  {
    int _build_nb = (num_supplier + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_supplier - _bo);
      int _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      build_supplier<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
          num_supplier, _bo, d_s_suppkey, d_s_nationkey,
          ht_supplier_tbl.d_entries, ht_supplier_tbl.ht_size, bf_supplier_gpu.d_bits, bf_supplier_gpu.size_bits, 3,
          d_supp_pf, d_supp_ts,
          ht_nation_tbl.d_entries, ht_nation_tbl.ht_size);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }
  if (bf_cpu[2].bits) bf_supplier_gpu.copy_to_host(bf_cpu[2].bits, ((size_t)(bf_cpu[2].size_bits + 63) / 64) * sizeof(uint64_t));

  // Build customer -- morsel-batched
  prefilter_all(num_customer, h_cust_pf, h_cust_ts);
  {
    int _build_nb = (num_customer + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_customer - _bo);
      int _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      build_customer<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
          num_customer, _bo, d_c_custkey, d_c_nationkey,
          ht_customer_tbl.d_entries, ht_customer_tbl.ht_size, bf_customer_gpu.d_bits, bf_customer_gpu.size_bits, 3,
          d_cust_pf, d_cust_ts);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }
  if (bf_cpu[3].bits) bf_customer_gpu.copy_to_host(bf_cpu[3].bits, ((size_t)(bf_cpu[3].size_bits + 63) / 64) * sizeof(uint64_t));

  // Build orders (upstream BF: customer) -- morsel-batched
  prefilter_orders();
  {
    int _build_nb = (num_orders + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_orders - _bo);
      int _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      build_orders<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
          num_orders, _bo, d_o_orderkey, d_o_orderdate, d_o_custkey,
          ht_orders_tbl.d_entries, ht_orders_tbl.ht_size, bf_orders_gpu.d_bits, bf_orders_gpu.size_bits, 3,
          d_ord_pf, d_ord_ts,
          ht_customer_tbl.d_entries, ht_customer_tbl.ht_size);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }
  if (bf_cpu[4].bits) bf_orders_gpu.copy_to_host(bf_cpu[4].bits, ((size_t)(bf_cpu[4].size_bits + 63) / 64) * sizeof(uint64_t));

  // Result pipeline
  fascal::runtime::BloomFilter *bf_ord_ptr = (bf_cpu[4].bits ? &bf_cpu[4] : nullptr);
  fascal::runtime::BloomFilter *bf_supp_ptr = (bf_cpu[2].bits ? &bf_cpu[2] : nullptr);

  auto run_kernel = [&]() {
    auto prefilter = [&](int bc, int bo, int tt) {
      fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
        [&](int off, int cnt) { cpu_predicate(h_data, off, cnt, h_bm, bf_ord_ptr, bf_supp_ptr); }, bo, tt);
    };
    auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
      tpch_q5_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
        nt, bo, d_l_orderkey_enc, d_l_suppkey, d_l_extendedprice, d_l_discount,
        d_n_regionkey, d_n_name, d_n_name_enc,
        d_bm, d_ts,
        bf_orders_gpu.d_bits, (bloom_off ? 0u : bf_orders_gpu.size_bits), ht_orders_tbl.d_entries, ht_orders_tbl.ht_size,
        bf_supplier_gpu.d_bits, (bloom_off ? 0u : bf_supplier_gpu.size_bits), ht_supplier_tbl.d_entries, ht_supplier_tbl.ht_size,
        bf_nation_gpu.d_bits, (bloom_off ? 0u : bf_nation_gpu.size_bits), ht_nation_tbl.d_entries, ht_nation_tbl.ht_size,
        bf_region_gpu.d_bits, (bloom_off ? 0u : bf_region_gpu.size_bits), ht_region_tbl.d_entries, ht_region_tbl.ht_size,
        bf_customer_gpu.d_bits, (bloom_off ? 0u : bf_customer_gpu.size_bits), ht_customer_tbl.d_entries, ht_customer_tbl.ht_size,
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

  // Output: nation row ID -> decode name
  double *h_agg = (double*)malloc(num_groups * NUM_AGGREGATES * sizeof(double));
  CUDA_CHECK(cudaMemcpy(h_agg, d_agg, num_groups * NUM_AGGREGATES * sizeof(double), cudaMemcpyDeviceToHost));

  static const char* nation_names[] = {
    "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "CHINA",
    "EGYPT", "ETHIOPIA", "FRANCE", "GERMANY", "INDIA",
    "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN",
    "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU", "ROMANIA",
    "RUSSIA", "SAUDI ARABIA", "UNITED KINGDOM", "UNITED STATES", "VIETNAM"
  };

  for (size_t k = 0; k < num_groups; ++k) {
    if (h_agg[k] == 0.0) continue;
    int name_enc = h_n_name[k];
    const char *name_str = (name_enc >= 0 && name_enc < 25) ? nation_names[name_enc] : "UNKNOWN";
    printf("ROW: %s|%.4f\n", name_str, h_agg[k]);
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
  odzc->unregister_column(mc_l_orderkey_enc); odzc->unregister_column(mc_l_suppkey);
  odzc->unregister_column(mc_l_extendedprice); odzc->unregister_column(mc_l_discount);
  odzc->unregister_column(mc_r_regionkey); odzc->unregister_column(mc_r_name);
  odzc->unregister_column(mc_n_nationkey); odzc->unregister_column(mc_n_name_enc);
  odzc->unregister_column(mc_n_regionkey); odzc->unregister_column(mc_n_name);
  odzc->unregister_column(mc_s_suppkey); odzc->unregister_column(mc_s_nationkey);
  odzc->unregister_column(mc_c_custkey); odzc->unregister_column(mc_c_nationkey);
  odzc->unregister_column(mc_o_orderkey); odzc->unregister_column(mc_o_orderdate);
  odzc->unregister_column(mc_o_custkey);
  fascal_free_seed_bitmap(h_bm, h_ts);
  fascal_free_column(h_l_orderkey_enc); fascal_free_column(h_l_suppkey);
  fascal_free_column(h_l_extendedprice); fascal_free_column(h_l_discount);
  CUDA_CHECK(cudaFree(d_agg));
  CUDA_CHECK(cudaFreeHost(h_reg_pf)); CUDA_CHECK(cudaFreeHost(h_reg_ts));
  CUDA_CHECK(cudaFreeHost(h_nat_pf)); CUDA_CHECK(cudaFreeHost(h_nat_ts));
  CUDA_CHECK(cudaFreeHost(h_supp_pf)); CUDA_CHECK(cudaFreeHost(h_supp_ts));
  CUDA_CHECK(cudaFreeHost(h_cust_pf)); CUDA_CHECK(cudaFreeHost(h_cust_ts));
  CUDA_CHECK(cudaFreeHost(h_ord_pf)); CUDA_CHECK(cudaFreeHost(h_ord_ts));
  for (int i = 0; i < 5; ++i) bf_cpu[i].free_filter();
  fascal_arena_destroy();
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  delete[] streams;
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(t_start); cudaEventDestroy(t_load);   cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  return 0;
}
