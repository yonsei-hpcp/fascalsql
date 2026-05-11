// TPC-H Q8 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
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
#define NUM_AGGREGATES 2

__device__ __constant__ unsigned char d_pred_flags[MAX_PREDICATES];
static unsigned char h_pred_flags[MAX_PREDICATES] = {0};

// =============================================================================
// GPU Kernel: scan lineitem, 7 joins, GROUP BY year, aggregate
// =============================================================================
template <int BT, int IPT>
__global__ void tpch_q8_kernel_lineitem(
    int num_tuples, int batch_offset,
    int *d_orders_o_orderdate,
    int *d_lineitem_l_partkey, int *d_lineitem_l_suppkey, int *d_lineitem_l_orderkey_enc,
    float *d_lineitem_l_extendedprice, float *d_lineitem_l_discount,
    int *d_orders_o_custkey,
    const uint32_t *__restrict__ seed_bitmap, const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_part, uint32_t bloom_size_bits_part,
    HtEntry *ht_part, uint32_t ht_size_part,
    const uint64_t *__restrict__ d_bloom_supplier, uint32_t bloom_size_bits_supplier,
    HtEntry *ht_supplier, uint32_t ht_size_supplier,
    const uint64_t *__restrict__ d_bloom_orders, uint32_t bloom_size_bits_orders,
    HtEntry *ht_orders, uint32_t ht_size_orders,
    const uint64_t *__restrict__ d_bloom_customer, uint32_t bloom_size_bits_customer,
    HtEntry *ht_customer, uint32_t ht_size_customer,
    const uint64_t *__restrict__ d_bloom_n1, uint32_t bloom_size_bits_n1,
    HtEntry *ht_n1, uint32_t ht_size_n1,
    const uint64_t *__restrict__ d_bloom_n2, uint32_t bloom_size_bits_n2,
    HtEntry *ht_n2, uint32_t ht_size_n2,
    const uint64_t *__restrict__ d_bloom_region, uint32_t bloom_size_bits_region,
    HtEntry *ht_region, uint32_t ht_size_region,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int loc_l_partkey[IPT], loc_l_suppkey[IPT], loc_l_orderkey_enc[IPT];
  float loc_l_extendedprice[IPT], loc_l_discount[IPT];
  int loc_orders_o_orderdate[IPT], loc_orders_o_custkey[IPT];
  int loc_n2_n_name[IPT], loc_supplier_s_nationkey[IPT];
  int loc_customer_c_nationkey[IPT], loc_n1_n_regionkey[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // Phase 1: Bloom probes
    BlockLoadSelect<int, BT, IPT>(d_lineitem_l_partkey + tile_offset, loc_l_partkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(loc_l_partkey, selection_flags, num_tile_items, bloom_size_bits_part, d_bloom_part, 3);

    BlockLoadSelect<int, BT, IPT>(d_lineitem_l_suppkey + tile_offset, loc_l_suppkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(loc_l_suppkey, selection_flags, num_tile_items, bloom_size_bits_supplier, d_bloom_supplier, 3);

    BlockLoadSelect<int, BT, IPT>(d_lineitem_l_orderkey_enc + tile_offset, loc_l_orderkey_enc, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(loc_l_orderkey_enc, selection_flags, num_tile_items, bloom_size_bits_orders, d_bloom_orders, 3);

    FASCAL_CHECK_ALIVE(IPT);

    // Phase 2: Hash table joins
    // Join 0: l_partkey -> part (existence only)
    BlockJoinProbeDirect<BT, IPT>(loc_l_partkey, selection_flags, num_tile_items, ht_part, ht_size_part, 1);

    // Join 1: l_suppkey -> supplier [payload=s_nationkey]
    BlockJoinProbePayloadDirect<BT, IPT>(loc_l_suppkey, selection_flags, num_tile_items, ht_supplier, ht_size_supplier, 1, loc_supplier_s_nationkey);

    // Join 2: l_orderkey_enc -> orders [payload=rid -> o_orderdate, o_custkey]
    int loc_orders_rid[IPT];
    BlockJoinProbePayloadDirect<BT, IPT>(loc_l_orderkey_enc, selection_flags, num_tile_items, ht_orders, ht_size_orders, 0, loc_orders_rid);
    #pragma unroll
    for (int ITEM = 0; ITEM < IPT; ITEM++) {
      if (selection_flags[ITEM]) {
        int rid = loc_orders_rid[ITEM];
        loc_orders_o_orderdate[ITEM] = d_orders_o_orderdate[rid];
        loc_orders_o_custkey[ITEM] = d_orders_o_custkey[rid];
      }
    }

    // Join 3: o_custkey -> customer [payload=c_nationkey]
    BlockJoinProbePayloadDirect<BT, IPT>(loc_orders_o_custkey, selection_flags, num_tile_items, ht_customer, ht_size_customer, 1, loc_customer_c_nationkey);

    // Join 4: c_nationkey -> n1 [payload=n_regionkey]
    BlockJoinProbePayload<BT, IPT>(loc_customer_c_nationkey, selection_flags, num_tile_items, ht_n1, ht_size_n1, loc_n1_n_regionkey);

    // Join 5: s_nationkey -> n2 [payload=n_name]
    BlockJoinProbePayload<BT, IPT>(loc_supplier_s_nationkey, selection_flags, num_tile_items, ht_n2, ht_size_n2, loc_n2_n_name);

    // Join 6: n_regionkey -> region (existence only)
    BlockJoinProbeDirect<BT, IPT>(loc_n1_n_regionkey, selection_flags, num_tile_items, ht_region, ht_size_region, 0);

    // Load aggregation columns
    BlockLoadSelect<float, BT, IPT>(d_lineitem_l_extendedprice + tile_offset, loc_l_extendedprice, selection_flags, num_tile_items);
    BlockLoadSelect<float, BT, IPT>(d_lineitem_l_discount + tile_offset, loc_l_discount, selection_flags, num_tile_items);

    // GROUP BY year, aggregate
    #pragma unroll
    for (int ITEM = 0; ITEM < IPT; ++ITEM) {
      if ((threadIdx.x + (BT * ITEM)) < num_tile_items && selection_flags[ITEM]) {
        int gk = (((loc_orders_o_orderdate[ITEM] / 10000)) - 1992);
        double vol = (double)loc_l_extendedprice[ITEM] * (1.0 - loc_l_discount[ITEM]);
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], (((double)loc_n2_n_name[ITEM] == 2.0) ? vol : 0.0));
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 1], vol);
      }
    }
  } while(0);
}

// =============================================================================
// CPU Predicate: Bloom filter probes only (no scalar predicates on lineitem)
// =============================================================================
static void cpu_predicate_lineitem(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bloom_filters[], int num_bloom_filters) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  int *l_partkey = h_data[0], *l_suppkey = h_data[1], *l_orderkey_enc = h_data[2];
  if (bloom_filters && 0 < num_bloom_filters)
    fascal::cpu_pred::bloom_probe_pass(bm, bs, l_partkey, offset, cnt, bloom_filters[0]);
  if (bloom_filters && 1 < num_bloom_filters)
    fascal::cpu_pred::bloom_probe_pass(bm, bs, l_suppkey, offset, cnt, bloom_filters[1]);
  if (bloom_filters && 2 < num_bloom_filters)
    fascal::cpu_pred::bloom_probe_pass(bm, bs, l_orderkey_enc, offset, cnt, bloom_filters[2]);
}

// =============================================================================
// Build kernels (dimension sub-pipelines)
// =============================================================================
template <int BT, int IPT>
__global__ void tpch_q8_kernel_part(
    int num_tuples, int batch_offset, const int *__restrict__ d_part_p_partkey, const int *__restrict__ d_part_p_type,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary) {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int loc_pk[IPT], loc_type[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_part_p_partkey + tile_offset), loc_pk, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_part_p_type + tile_offset), loc_type, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(loc_type, 3, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int payload = tile_offset + threadIdx.x + ITEM * BT;
    gpu_ht_insert_one_direct(loc_pk[ITEM], payload, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_pk[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_part(int *h_pk, int *h_type, int n, uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; if (h_type[idx] == 3) bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
  });
}

template <int BT, int IPT>
__global__ void tpch_q8_kernel_supplier(
    int num_tuples, int batch_offset, const int *__restrict__ d_sk, const int *__restrict__ d_nk,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_n2, uint32_t ht_size_f_n2) {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int loc_sk[IPT], loc_nk[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_sk + tile_offset), loc_sk, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_nk + tile_offset), loc_nk, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe(loc_nk[ITEM], d_filter_ht_n2, ht_size_f_n2, nullptr)) continue;
    gpu_ht_insert_one_direct(loc_sk[ITEM], loc_nk[ITEM], 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_sk[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_supplier(int *h_sk, int *h_nk, int n, uint32_t *bm, uint8_t *ts,
    fascal::runtime::BloomFilter *bf_n2) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (bf_n2 && bf_n2->bits) bf_n2->probe_and_mask_packed(h_nk + off, bm, off, cnt);
  });
}

template <int BT, int IPT>
__global__ void tpch_q8_kernel_orders(
    int num_tuples, int batch_offset, const int *__restrict__ d_ok, const int *__restrict__ d_od, const int *__restrict__ d_ck,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_cust, uint32_t ht_size_f_cust) {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int loc_ok[IPT], loc_od[IPT], loc_ck[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_ok + tile_offset), loc_ok, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_od + tile_offset), loc_od, selection_flags, num_tile_items);
  BlockPredAndGTE<int, BT, IPT>(loc_od, 19950101, selection_flags, num_tile_items);
  BlockPredAndLTE<int, BT, IPT>(loc_od, 19961231, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_ck + tile_offset), loc_ck, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe_direct(loc_ck[ITEM], d_filter_ht_cust, ht_size_f_cust, 1, nullptr)) continue;
    int row_idx = tile_offset + threadIdx.x + ITEM * BT;
    gpu_ht_insert_one_direct(row_idx, row_idx, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(row_idx, 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_orders(int *h_ok, int *h_od, int *h_ck, int n, uint32_t *bm, uint8_t *ts,
    fascal::runtime::BloomFilter *bf_cust) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; if (h_od[idx] >= 19950101 && h_od[idx] <= 19961231) bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (bf_cust && bf_cust->bits) bf_cust->probe_and_mask_packed(h_ck + off, bm, off, cnt);
  });
}

template <int BT, int IPT>
__global__ void tpch_q8_kernel_customer(
    int num_tuples, int batch_offset, const int *__restrict__ d_ck, const int *__restrict__ d_nk,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_n1, uint32_t ht_size_f_n1) {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int loc_ck[IPT], loc_nk[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_ck + tile_offset), loc_ck, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_nk + tile_offset), loc_nk, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe(loc_nk[ITEM], d_filter_ht_n1, ht_size_f_n1, nullptr)) continue;
    gpu_ht_insert_one_direct(loc_ck[ITEM], loc_nk[ITEM], 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_ck[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_customer(int *h_ck, int *h_nk, int n, uint32_t *bm, uint8_t *ts,
    fascal::runtime::BloomFilter *bf_n1) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (bf_n1 && bf_n1->bits) bf_n1->probe_and_mask_packed(h_nk + off, bm, off, cnt);
  });
}

template <int BT, int IPT>
__global__ void tpch_q8_kernel_n1(
    int num_tuples, int batch_offset, const int *__restrict__ d_nk, const int *__restrict__ d_rk,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_region, uint32_t ht_size_f_region) {
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_nk[IPT], loc_rk[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_nk + tile_offset), loc_nk, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_rk + tile_offset), loc_rk, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe_direct(loc_rk[ITEM], d_filter_ht_region, ht_size_f_region, 0, nullptr)) continue;
    unsigned long long key = static_cast<unsigned int>(loc_nk[ITEM]);
    gpu_ht_insert_one(key, loc_rk[ITEM], 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_nk[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_n1(int *h_nk, int *h_rk, int n, uint32_t *bm, uint8_t *ts,
    fascal::runtime::BloomFilter *bf_region) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (bf_region && bf_region->bits) bf_region->probe_and_mask_packed(h_rk + off, bm, off, cnt);
  });
}

template <int BT, int IPT>
__global__ void tpch_q8_kernel_n2(
    int num_tuples, int batch_offset, const int *__restrict__ d_nk, const int *__restrict__ d_name,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary) {
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_nk[IPT], loc_name[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_nk + tile_offset), loc_nk, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_name + tile_offset), loc_name, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    unsigned long long key = static_cast<unsigned int>(loc_nk[ITEM]);
    gpu_ht_insert_one(key, loc_name[ITEM], 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_nk[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_n2(int *h_nk, int *h_name, int n, uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
  });
}

template <int BT, int IPT>
__global__ void tpch_q8_kernel_region(
    int num_tuples, int batch_offset, const int *__restrict__ d_rk, const int *__restrict__ d_rname,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary) {
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_rk[IPT], loc_rname[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_rk + tile_offset), loc_rk, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_rname + tile_offset), loc_rname, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(loc_rname, 1, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int payload = tile_offset + threadIdx.x + ITEM * BT;
    gpu_ht_insert_one_direct(loc_rk[ITEM], payload, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(loc_rk[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_region(int *h_rk, int *h_rname, int n, uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; if (h_rname[idx] == 1) bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
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

  printf("=== FaScalSQL: tpch_q8 ===\n");
  if (argc < 3) { printf("Usage: %s <num_tuples> <data_dir>\n", argv[0]); return 1; }
  data_dir = argv[2];

  // Auto-detect num_tuples
  int num_tuples = 0;
  { char p[512]; snprintf(p, sizeof(p), "%s/l_partkey.bin", data_dir);
    int c = 0; int *t = load_binary_column_auto(p, &c); if (t) fascal_free_column(t);
    num_tuples = c; }
  if (num_tuples <= 0) { fprintf(stderr, "Cannot auto-detect num_tuples\n"); return 1; }
  printf("num_tuples=%d data_dir=%s\n", num_tuples, data_dir);

  cudaEvent_t t_start, t_load, t_query, t_result;
  CUDA_CHECK(cudaEventCreate(&t_start)); CUDA_CHECK(cudaEventCreate(&t_load));
  CUDA_CHECK(cudaEventCreate(&t_query));
  CUDA_CHECK(cudaEventCreate(&t_result)); CUDA_CHECK(cudaEventRecord(t_start));

  fascal_arena_init((size_t)num_tuples * sizeof(int) * 6);

  char path[512];
  #define LOAD_COL(var, name, type) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    type *var = (type*)load_binary_column(path, num_tuples); \
    if (!var) { fprintf(stderr, "Failed to load " name "\n"); return 1; }
  #define LOAD_DIM(var, name, cnt) \
    snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
    int *var = load_binary_column_auto(path, &cnt); \
    if (!var || cnt == 0) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

  LOAD_COL(h_l_partkey, "l_partkey", int)
  LOAD_COL(h_l_suppkey, "l_suppkey", int)
  LOAD_COL(h_l_orderkey_enc, "l_orderkey_enc", int)
  LOAD_COL(h_l_extendedprice, "l_extendedprice", float)
  LOAD_COL(h_l_discount, "l_discount", float)

  int num_part = 0; LOAD_DIM(h_p_partkey, "p_partkey", num_part) LOAD_DIM(h_p_type, "p_type", num_part) printf("Loaded part: %d\n", num_part);
  int num_n2 = 0; LOAD_DIM(h_n2_nk, "n_nationkey", num_n2) LOAD_DIM(h_n2_name, "n_name", num_n2) printf("Loaded n2: %d\n", num_n2);
  int num_region = 0; LOAD_DIM(h_r_rk, "r_regionkey", num_region) LOAD_DIM(h_r_name, "r_name", num_region) printf("Loaded region: %d\n", num_region);
  int num_supplier = 0; LOAD_DIM(h_s_sk, "s_suppkey", num_supplier) LOAD_DIM(h_s_nk, "s_nationkey", num_supplier) printf("Loaded supplier: %d\n", num_supplier);
  int num_n1 = 0; LOAD_DIM(h_n1_nk, "n_nationkey", num_n1) LOAD_DIM(h_n1_rk, "n_regionkey", num_n1) printf("Loaded n1: %d\n", num_n1);
  int num_customer = 0; LOAD_DIM(h_c_ck, "c_custkey", num_customer) LOAD_DIM(h_c_nk, "c_nationkey", num_customer) printf("Loaded customer: %d\n", num_customer);
  int num_orders = 0; LOAD_DIM(h_o_ok, "o_orderkey", num_orders) LOAD_DIM(h_o_od, "o_orderdate", num_orders) LOAD_DIM(h_o_ck, "o_custkey", num_orders) printf("Loaded orders: %d\n", num_orders);
  #undef LOAD_COL
  #undef LOAD_DIM
  CUDA_CHECK(cudaEventRecord(t_load));

  // CQO
  auto cqo_in = fascal_cqo_init(num_tuples);
  // Dynamic BF selectivity: part(p_type filtered), n2(all), region(r_name=AMERICA=1), supplier(all), n1(region-filtered), customer(all), orders(date)
  int part_qual = (int)(fascal_estimate_eq_sel(h_p_type, num_part, 43) * num_part);  // ECONOMY ANODIZED STEEL=43
  int region_qual = (int)(fascal_estimate_eq_sel(h_r_name, num_region, 1) * num_region);
  int n1_qual = (int)(fascal_estimate_eq_sel(h_n1_rk, num_n1, 1) * num_n1);  // nations in AMERICA region
  int orders_qual = (int)(fascal_estimate_range_sel(h_o_od, num_orders, 19950101, 19961231) * num_orders);
  cqo_in.join_bf_selectivities = {
    fascal_estimate_join_bf_sel(part_qual, num_part),
    fascal_estimate_join_bf_sel(num_n2, num_n2),
    fascal_estimate_join_bf_sel(region_qual, num_region),
    fascal_estimate_join_bf_sel(num_supplier, num_supplier),
    fascal_estimate_join_bf_sel(n1_qual, num_n1),
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

  int *h_data[] = {h_l_partkey, h_l_suppkey, h_l_orderkey_enc, (int*)h_l_extendedprice, (int*)h_l_discount};
  fascal::runtime::BloomFilter bf_storage[7];
  fascal::runtime::BloomFilter *bf_array[7];
  bf_array[0] = (!bloom_off && 0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[0] : nullptr;
  bf_array[1] = (!bloom_off && 1 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[1]) ? &bf_storage[3] : nullptr;
  bf_array[2] = (!bloom_off && 2 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[2]) ? &bf_storage[6] : nullptr;
  for (int i = 3; i < 7; ++i) bf_array[i] = nullptr;

  // ODZC
  auto odzc = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));
  uint32_t *h_bm = nullptr, *d_bm = nullptr; uint8_t *h_ts = nullptr, *d_ts = nullptr;
  fascal_alloc_seed_bitmap(num_tuples, &h_bm, &d_bm, &h_ts, &d_ts);

  #define ODZC_MAP_INT(var, host, n, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; int *d_##var = nullptr; \
    fascal_odzc_map(odzc.get(), host, n, name, &mc_##var, &d_##var);
  #define ODZC_MAP_FLOAT(var, host, n, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; float *d_##var = nullptr; \
    { mc_##var = odzc->register_column(host, (size_t)(n) * sizeof(float)); \
      d_##var = static_cast<float*>(mc_##var.device_ptr); \
      if (!d_##var) { CUDA_CHECK(cudaMalloc(&d_##var, (size_t)(n)*sizeof(float))); CUDA_CHECK(cudaMemcpy(d_##var, host, (size_t)(n)*sizeof(float), cudaMemcpyHostToDevice)); } }
  #define ODZC_MAP_DIM(var, host, n, name) \
    fascal::runtime::ODZCManager::MappedColumn mc_##var; \
    mc_##var = odzc->register_column(host, (size_t)(n) * sizeof(int)); \
    int *d_##var = static_cast<int*>(mc_##var.device_ptr); \
    if (!d_##var) { CUDA_CHECK(cudaMalloc(&d_##var, (size_t)(n)*sizeof(int))); CUDA_CHECK(cudaMemcpy(d_##var, host, (size_t)(n)*sizeof(int), cudaMemcpyHostToDevice)); }

  ODZC_MAP_INT(l_pk, h_l_partkey, num_tuples, "l_partkey")
  ODZC_MAP_INT(l_sk, h_l_suppkey, num_tuples, "l_suppkey")
  ODZC_MAP_INT(l_ok, h_l_orderkey_enc, num_tuples, "l_orderkey_enc")
  ODZC_MAP_FLOAT(l_ep, h_l_extendedprice, num_tuples, "l_extendedprice")
  ODZC_MAP_FLOAT(l_disc, h_l_discount, num_tuples, "l_discount")
  ODZC_MAP_DIM(p_pk, h_p_partkey, num_part, "p_partkey") ODZC_MAP_DIM(p_type, h_p_type, num_part, "p_type")
  ODZC_MAP_DIM(n2_nk, h_n2_nk, num_n2, "n2_nk") ODZC_MAP_DIM(n2_name, h_n2_name, num_n2, "n2_name")
  ODZC_MAP_DIM(r_rk, h_r_rk, num_region, "r_rk") ODZC_MAP_DIM(r_name, h_r_name, num_region, "r_name")
  ODZC_MAP_DIM(s_sk, h_s_sk, num_supplier, "s_sk") ODZC_MAP_DIM(s_nk, h_s_nk, num_supplier, "s_nk")
  ODZC_MAP_DIM(n1_nk, h_n1_nk, num_n1, "n1_nk") ODZC_MAP_DIM(n1_rk, h_n1_rk, num_n1, "n1_rk")
  ODZC_MAP_DIM(c_ck, h_c_ck, num_customer, "c_ck") ODZC_MAP_DIM(c_nk, h_c_nk, num_customer, "c_nk")
  ODZC_MAP_DIM(o_ok, h_o_ok, num_orders, "o_ok") ODZC_MAP_DIM(o_od, h_o_od, num_orders, "o_od") ODZC_MAP_DIM(o_ck, h_o_ck, num_orders, "o_ck")
  #undef ODZC_MAP_INT
  #undef ODZC_MAP_FLOAT
  #undef ODZC_MAP_DIM

  CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

  // Aggregation
  size_t num_groups = 7;
  double *d_agg;
  CUDA_CHECK(cudaMalloc(&d_agg, num_groups * NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * NUM_AGGREGATES * sizeof(double)));

  // Build HTs + BFs
  #define ALLOC_BUILD(name, n, alloc_fn) \
    GpuHashTable ht_##name = GpuHashTable::alloc_fn(n); \
    GpuBloomFilter gpu_bf_##name = GpuBloomFilter::allocate(n, 3); \
    CUDA_CHECK(cudaMemset(ht_##name.d_entries, 0xff, ht_##name.ht_size * sizeof(HtEntry))); \
    CUDA_CHECK(cudaMemset(gpu_bf_##name.d_bits, 0, ((size_t)gpu_bf_##name.size_bits + 63) / 64 * sizeof(uint64_t)));
  #define ALLOC_PREFILTER(name, n, idx) \
    bf_storage[idx] = fascal::runtime::BloomFilter::allocate(n, 3); \
    uint32_t *h_pf_##name = nullptr, *d_pf_##name = nullptr; uint8_t *h_pts_##name = nullptr, *d_pts_##name = nullptr; \
    { size_t bw = ((size_t)(n) + 31) / 32, nt = ((size_t)(n) + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE; \
      CUDA_CHECK(cudaHostAlloc(&h_pf_##name, bw * 4, cudaHostAllocMapped)); memset(h_pf_##name, 0, bw * 4); \
      CUDA_CHECK(cudaHostGetDevicePointer(&d_pf_##name, h_pf_##name, 0)); \
      CUDA_CHECK(cudaHostAlloc(&h_pts_##name, nt, cudaHostAllocMapped)); memset(h_pts_##name, 0, nt); \
      CUDA_CHECK(cudaHostGetDevicePointer(&d_pts_##name, h_pts_##name, 0)); }

  ALLOC_BUILD(part, num_part, allocate_direct) ALLOC_PREFILTER(part, num_part, 0)
  ALLOC_BUILD(n2, num_n2, allocate) ALLOC_PREFILTER(n2, num_n2, 1)
  ALLOC_BUILD(region, num_region, allocate_direct) ALLOC_PREFILTER(region, num_region, 2)
  ALLOC_BUILD(supplier, num_supplier, allocate_direct) ALLOC_PREFILTER(supplier, num_supplier, 3)
  ALLOC_BUILD(n1, num_n1, allocate) ALLOC_PREFILTER(n1, num_n1, 4)
  ALLOC_BUILD(customer, num_customer, allocate_direct) ALLOC_PREFILTER(customer, num_customer, 5)
  ALLOC_BUILD(orders, num_orders, allocate_direct) ALLOC_PREFILTER(orders, num_orders, 6)
  #undef ALLOC_BUILD
  #undef ALLOC_PREFILTER

  // Pipeline setup
  const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
  const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
  cudaStream_t *streams = new cudaStream_t[N_STREAMS];
  for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
  int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

  auto run_kernel = [&]() {
    auto prefilter = [&](int bc, int bo, int tt) {
      fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
        [&](int off, int cnt) { cpu_predicate_lineitem(h_data, off, cnt, h_bm, bf_array, 7); }, bo, tt);
    };
    auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
      tpch_q8_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
        nt, bo, d_o_od, d_l_pk, d_l_sk, d_l_ok, d_l_ep, d_l_disc, d_o_ck,
        d_bm, d_ts,
        gpu_bf_part.d_bits, (bloom_off ? 0u : gpu_bf_part.size_bits), ht_part.d_entries, ht_part.ht_size,
        gpu_bf_supplier.d_bits, (bloom_off ? 0u : gpu_bf_supplier.size_bits), ht_supplier.d_entries, ht_supplier.ht_size,
        gpu_bf_orders.d_bits, (bloom_off ? 0u : gpu_bf_orders.size_bits), ht_orders.d_entries, ht_orders.ht_size,
        gpu_bf_customer.d_bits, (bloom_off ? 0u : gpu_bf_customer.size_bits), ht_customer.d_entries, ht_customer.ht_size,
        gpu_bf_n1.d_bits, (bloom_off ? 0u : gpu_bf_n1.size_bits), ht_n1.d_entries, ht_n1.ht_size,
        gpu_bf_n2.d_bits, (bloom_off ? 0u : gpu_bf_n2.size_bits), ht_n2.d_entries, ht_n2.ht_size,
        gpu_bf_region.d_bits, (bloom_off ? 0u : gpu_bf_region.size_bits), ht_region.d_entries, ht_region.ht_size,
        d_agg);
    };
    { int nb = (num_tuples + GPU_BATCH - 1) / GPU_BATCH;
      for (int b = 0; b < nb; ++b) { int bo = b * GPU_BATCH, bc = std::min(GPU_BATCH, num_tuples - bo);
        prefilter(bc, bo, num_tuples); launch(num_tuples, bo, (bc + TILE_SIZE - 1) / TILE_SIZE, 0); } }
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  };

  // Execute build pipelines
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
  cudaEventRecord(t0);

  // part -- morsel-batched
  cpu_prefilter_part(h_p_partkey, h_p_type, num_part, h_pf_part, h_pts_part);
  {
    int _build_nb = (num_part + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_part - _bo), _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      tpch_q8_kernel_part<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
        num_part, _bo, d_p_pk, d_p_type, ht_part.d_entries, ht_part.ht_size, gpu_bf_part.d_bits, gpu_bf_part.size_bits, 3, d_pf_part, d_pts_part);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }
  if (bf_storage[0].bits) gpu_bf_part.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));

  // n2 -- single launch (25 rows)
  cpu_prefilter_n2(h_n2_nk, h_n2_name, num_n2, h_pf_n2, h_pts_n2);
  tpch_q8_kernel_n2<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_n2 + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
    num_n2, 0, d_n2_nk, d_n2_name, ht_n2.d_entries, ht_n2.ht_size, gpu_bf_n2.d_bits, gpu_bf_n2.size_bits, 3, d_pf_n2, d_pts_n2);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (bf_storage[1].bits) gpu_bf_n2.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));

  // region -- single launch (5 rows)
  cpu_prefilter_region(h_r_rk, h_r_name, num_region, h_pf_region, h_pts_region);
  tpch_q8_kernel_region<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_region + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
    num_region, 0, d_r_rk, d_r_name, ht_region.d_entries, ht_region.ht_size, gpu_bf_region.d_bits, gpu_bf_region.size_bits, 3, d_pf_region, d_pts_region);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (bf_storage[2].bits) gpu_bf_region.copy_to_host(bf_storage[2].bits, ((size_t)(bf_storage[2].size_bits + 63) / 64) * sizeof(uint64_t));

  // supplier (filtered by n2 BF) -- morsel-batched
  cpu_prefilter_supplier(h_s_sk, h_s_nk, num_supplier, h_pf_supplier, h_pts_supplier, (bf_storage[1].bits ? &bf_storage[1] : nullptr));
  {
    int _build_nb = (num_supplier + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_supplier - _bo), _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      tpch_q8_kernel_supplier<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
        num_supplier, _bo, d_s_sk, d_s_nk, ht_supplier.d_entries, ht_supplier.ht_size, gpu_bf_supplier.d_bits, gpu_bf_supplier.size_bits, 3,
        d_pf_supplier, d_pts_supplier, ht_n2.d_entries, ht_n2.ht_size);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }
  if (bf_storage[3].bits) gpu_bf_supplier.copy_to_host(bf_storage[3].bits, ((size_t)(bf_storage[3].size_bits + 63) / 64) * sizeof(uint64_t));

  // n1 (filtered by region BF) -- single launch (25 rows)
  cpu_prefilter_n1(h_n1_nk, h_n1_rk, num_n1, h_pf_n1, h_pts_n1, (bf_storage[2].bits ? &bf_storage[2] : nullptr));
  tpch_q8_kernel_n1<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_n1 + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
    num_n1, 0, d_n1_nk, d_n1_rk, ht_n1.d_entries, ht_n1.ht_size, gpu_bf_n1.d_bits, gpu_bf_n1.size_bits, 3,
    d_pf_n1, d_pts_n1, ht_region.d_entries, ht_region.ht_size);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  if (bf_storage[4].bits) gpu_bf_n1.copy_to_host(bf_storage[4].bits, ((size_t)(bf_storage[4].size_bits + 63) / 64) * sizeof(uint64_t));

  // customer (filtered by n1 BF) -- morsel-batched
  cpu_prefilter_customer(h_c_ck, h_c_nk, num_customer, h_pf_customer, h_pts_customer, (bf_storage[4].bits ? &bf_storage[4] : nullptr));
  {
    int _build_nb = (num_customer + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_customer - _bo), _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      tpch_q8_kernel_customer<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
        num_customer, _bo, d_c_ck, d_c_nk, ht_customer.d_entries, ht_customer.ht_size, gpu_bf_customer.d_bits, gpu_bf_customer.size_bits, 3,
        d_pf_customer, d_pts_customer, ht_n1.d_entries, ht_n1.ht_size);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }
  if (bf_storage[5].bits) gpu_bf_customer.copy_to_host(bf_storage[5].bits, ((size_t)(bf_storage[5].size_bits + 63) / 64) * sizeof(uint64_t));

  // orders (filtered by customer BF) -- morsel-batched
  cpu_prefilter_orders(h_o_ok, h_o_od, h_o_ck, num_orders, h_pf_orders, h_pts_orders, (bf_storage[5].bits ? &bf_storage[5] : nullptr));
  {
    int _build_nb = (num_orders + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _build_nb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_orders - _bo), _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
      tpch_q8_kernel_orders<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, 0>>>(
        num_orders, _bo, d_o_ok, d_o_od, d_o_ck, ht_orders.d_entries, ht_orders.ht_size, gpu_bf_orders.d_bits, gpu_bf_orders.size_bits, 3,
        d_pf_orders, d_pts_orders, ht_customer.d_entries, ht_customer.ht_size);
    }
    for (int _s = 0; _s < N_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(streams[_s]));
    CUDA_CHECK(cudaGetLastError());
  }
  if (bf_storage[6].bits) gpu_bf_orders.copy_to_host(bf_storage[6].bits, ((size_t)(bf_storage[6].size_bits + 63) / 64) * sizeof(uint64_t));

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

  // Output
  double *h_agg = (double*)malloc(num_groups * NUM_AGGREGATES * sizeof(double));
  CUDA_CHECK(cudaMemcpy(h_agg, d_agg, num_groups * NUM_AGGREGATES * sizeof(double), cudaMemcpyDeviceToHost));
  for (size_t k = 0; k < num_groups; k++) {
    if (h_agg[k * NUM_AGGREGATES + 0] != 0 || h_agg[k * NUM_AGGREGATES + 1] != 0) {
      printf("ROW: %d|%.6f|%.6f\n", (int)(k % 7) + 1992, h_agg[k * NUM_AGGREGATES + 0], h_agg[k * NUM_AGGREGATES + 1]);
    }
  }
  free(h_agg);

  // Timing
  CUDA_CHECK(cudaEventRecord(t_result)); CUDA_CHECK(cudaDeviceSynchronize());
  { float ml, mb, mr;
    CUDA_CHECK(cudaEventElapsedTime(&ml, t_start, t_load));
    CUDA_CHECK(cudaEventElapsedTime(&mb, t_load, t_query));
    /* merged into query */
    CUDA_CHECK(cudaEventElapsedTime(&mr, t_query, t_result));
    printf("Timing: load=%.1fms query=%.1fms result=%.1fms total=%.1fms\n", ml, mb, mr, ml+mb+mr); }

  // Cleanup
  odzc->unregister_column(mc_l_pk); odzc->unregister_column(mc_l_sk); odzc->unregister_column(mc_l_ok);
  odzc->unregister_column(mc_l_ep); odzc->unregister_column(mc_l_disc);
  odzc->unregister_column(mc_p_pk); odzc->unregister_column(mc_p_type);
  odzc->unregister_column(mc_n2_nk); odzc->unregister_column(mc_n2_name);
  odzc->unregister_column(mc_r_rk); odzc->unregister_column(mc_r_name);
  odzc->unregister_column(mc_s_sk); odzc->unregister_column(mc_s_nk);
  odzc->unregister_column(mc_n1_nk); odzc->unregister_column(mc_n1_rk);
  odzc->unregister_column(mc_c_ck); odzc->unregister_column(mc_c_nk);
  odzc->unregister_column(mc_o_ok); odzc->unregister_column(mc_o_od); odzc->unregister_column(mc_o_ck);
  fascal_free_seed_bitmap(h_bm, h_ts);
  CUDA_CHECK(cudaFree(d_agg));
  fascal_free_column(h_l_partkey); fascal_free_column(h_l_suppkey); fascal_free_column(h_l_orderkey_enc);
  fascal_free_column(h_l_extendedprice); fascal_free_column(h_l_discount);
  fascal_free_column(h_p_partkey); fascal_free_column(h_p_type);
  fascal_free_column(h_n2_nk); fascal_free_column(h_n2_name);
  fascal_free_column(h_r_rk); fascal_free_column(h_r_name);
  fascal_free_column(h_s_sk); fascal_free_column(h_s_nk);
  fascal_free_column(h_n1_nk); fascal_free_column(h_n1_rk);
  fascal_free_column(h_c_ck); fascal_free_column(h_c_nk);
  fascal_free_column(h_o_ok); fascal_free_column(h_o_od); fascal_free_column(h_o_ck);
  CUDA_CHECK(cudaFreeHost(h_pf_part)); CUDA_CHECK(cudaFreeHost(h_pts_part)); gpu_bf_part.free_filter(); ht_part.free_table(); bf_storage[0].free_filter();
  CUDA_CHECK(cudaFreeHost(h_pf_n2)); CUDA_CHECK(cudaFreeHost(h_pts_n2)); gpu_bf_n2.free_filter(); ht_n2.free_table(); bf_storage[1].free_filter();
  CUDA_CHECK(cudaFreeHost(h_pf_region)); CUDA_CHECK(cudaFreeHost(h_pts_region)); gpu_bf_region.free_filter(); ht_region.free_table(); bf_storage[2].free_filter();
  CUDA_CHECK(cudaFreeHost(h_pf_supplier)); CUDA_CHECK(cudaFreeHost(h_pts_supplier)); gpu_bf_supplier.free_filter(); ht_supplier.free_table(); bf_storage[3].free_filter();
  CUDA_CHECK(cudaFreeHost(h_pf_n1)); CUDA_CHECK(cudaFreeHost(h_pts_n1)); gpu_bf_n1.free_filter(); ht_n1.free_table(); bf_storage[4].free_filter();
  CUDA_CHECK(cudaFreeHost(h_pf_customer)); CUDA_CHECK(cudaFreeHost(h_pts_customer)); gpu_bf_customer.free_filter(); ht_customer.free_table(); bf_storage[5].free_filter();
  CUDA_CHECK(cudaFreeHost(h_pf_orders)); CUDA_CHECK(cudaFreeHost(h_pts_orders)); gpu_bf_orders.free_filter(); ht_orders.free_table(); bf_storage[6].free_filter();
  fascal_arena_destroy();
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  delete[] streams;
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(t_start); cudaEventDestroy(t_load);   cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  return 0;
}
