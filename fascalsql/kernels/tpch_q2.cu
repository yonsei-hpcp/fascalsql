// TPC-H Q2 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Refactored: uses shared macros and CPU predicate helpers.
// 2 stages: scalar_sq_1 (MIN supplycost per part) + main query (top-100 output).

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
#include <unordered_set>
#include <unordered_map>
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
// STAGE 1: tpch_q2___scalar_sq_1 -- MIN(ps_supplycost) per ps_partkey
// =============================================================================

// --- Stage 1 result-pipeline kernel: partsupp scan with joins ---
template <int BT, int IPT>
__global__ void tpch_q2___scalar_sq_1_kernel_partsupp(
    int num_tuples, int batch_offset,
    int *d_partsupp_ps_suppkey, int *d_partsupp_ps_supplycost,
    int *d_partsupp_ps_partkey_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_supplier, uint32_t bloom_size_bits_supplier,
    HtEntry *ht_supplier, uint32_t ht_size_supplier,
    const uint64_t *__restrict__ d_bloom_nation, uint32_t bloom_size_bits_nation,
    HtEntry *ht_nation, uint32_t ht_size_nation,
    const uint64_t *__restrict__ d_bloom_region, uint32_t bloom_size_bits_region,
    HtEntry *ht_region, uint32_t ht_size_region,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int loc_ps_suppkey[IPT], loc_ps_supplycost[IPT];
  int loc_s_nationkey[IPT], loc_n_regionkey[IPT], loc_ps_partkey_enc[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // Bloom probe: ps_suppkey -> supplier
    BlockLoadSelect<int, BT, IPT>(d_partsupp_ps_suppkey + tile_offset, loc_ps_suppkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(loc_ps_suppkey, selection_flags, num_tile_items, bloom_size_bits_supplier, d_bloom_supplier, 3);
    FASCAL_CHECK_ALIVE(IPT);

    // Join 0: ps_suppkey -> supplier (payload: s_nationkey)
    BlockJoinProbePayloadDirect<BT, IPT>(loc_ps_suppkey, selection_flags, num_tile_items, ht_supplier, ht_size_supplier, 1, loc_s_nationkey);
    // Join 1: s_nationkey -> nation (payload: n_regionkey)
    BlockJoinProbePayloadDirect<BT, IPT>(loc_s_nationkey, selection_flags, num_tile_items, ht_nation, ht_size_nation, 0, loc_n_regionkey);
    // Join 2: n_regionkey -> region
    BlockJoinProbeDirect<BT, IPT>(loc_n_regionkey, selection_flags, num_tile_items, ht_region, ht_size_region, 0);

    // Load GROUP BY key + aggregation column
    BlockLoadSelect<int, BT, IPT>(d_partsupp_ps_partkey_enc + tile_offset, loc_ps_partkey_enc, selection_flags, num_tile_items);
    BlockLoadSelect<int, BT, IPT>(d_partsupp_ps_supplycost + tile_offset, loc_ps_supplycost, selection_flags, num_tile_items);

    // GROUP BY: atomicMin(supplycost / 100.0) per partkey_enc
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i]) {
        int gk = loc_ps_partkey_enc[i];
        atomicMin(&aggtable[gk * NUM_AGGREGATES + 0], ((double)((unsigned long long)loc_ps_supplycost[i])) / 100.0);
      }
    }
  } while(0);
}

// --- Stage 1 CPU predicate: Bloom probe on ps_suppkey ---
static void sq1_cpu_predicate_partsupp(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bf_array[], int num_bf) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  if (num_bf > 0) fascal::cpu_pred::bloom_probe_pass(bm, bs, h_data[0], offset, cnt, bf_array[0]);
}

// --- Stage 1 build kernels ---
template <int BT, int IPT>
__global__ void sq1_kernel_supplier(
    int num_tuples,
    const int *__restrict__ d_s_suppkey, const int *__restrict__ d_s_nationkey,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_nation, uint32_t ht_size_f_nation)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_suppkey[IPT], loc_nationkey[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_suppkey + tile_offset), loc_suppkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_nationkey + tile_offset), loc_nationkey, selection_flags, num_tile_items);
  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    if (!gpu_ht_probe_direct(loc_nationkey[i], d_filter_ht_nation, ht_size_f_nation, 0, nullptr)) continue;
    gpu_ht_insert_one_direct(loc_suppkey[i], loc_nationkey[i], 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_suppkey[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

template <int BT, int IPT>
__global__ void sq1_kernel_nation(
    int num_tuples,
    const int *__restrict__ d_n_nationkey, const int *__restrict__ d_n_regionkey,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_region, uint32_t ht_size_f_region)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_nationkey[IPT], loc_regionkey[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_n_nationkey + tile_offset), loc_nationkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_n_regionkey + tile_offset), loc_regionkey, selection_flags, num_tile_items);
  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    if (!gpu_ht_probe_direct(loc_regionkey[i], d_filter_ht_region, ht_size_f_region, 0, nullptr)) continue;
    gpu_ht_insert_one_direct(loc_nationkey[i], loc_regionkey[i], 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(loc_nationkey[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

template <int BT, int IPT>
__global__ void sq1_kernel_region(
    int num_tuples,
    const int *__restrict__ d_r_regionkey, const int *__restrict__ d_r_name,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_regionkey[IPT], loc_name[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_r_regionkey + tile_offset), loc_regionkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_r_name + tile_offset), loc_name, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(loc_name, 4, selection_flags, num_tile_items);
  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    int payload = tile_offset + threadIdx.x + i * BT;
    gpu_ht_insert_one_direct(loc_regionkey[i], payload, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(loc_regionkey[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// --- Stage 1 build CPU prefilters ---
static void sq1_cpu_prefilter_region(int *h_r_regionkey, int *h_r_name, int n,
    uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      if (h_r_name[idx] == 4) bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
  });
}

static void sq1_cpu_prefilter_nation(int *h_n_nationkey, int *h_n_regionkey, int n,
    uint32_t *bm, uint8_t *ts, fascal::runtime::BloomFilter *upstream_bf) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (upstream_bf && upstream_bf->bits)
      upstream_bf->probe_and_mask_packed(h_n_regionkey + off, bm, off, cnt);
  });
}

static void sq1_cpu_prefilter_supplier(int *h_s_suppkey, int *h_s_nationkey, int n,
    uint32_t *bm, uint8_t *ts, fascal::runtime::BloomFilter *upstream_bf) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (upstream_bf && upstream_bf->bits)
      upstream_bf->probe_and_mask_packed(h_s_nationkey + off, bm, off, cnt);
  });
}

// =============================================================================
// STAGE 2: tpch_q2 -- main query (materialize + top-100 sort)
// =============================================================================

// --- Stage 2 result-pipeline kernel: partsupp scan with 5 joins + post-join check ---
template <int BT, int IPT>
__global__ void tpch_q2_kernel_partsupp(
    int num_tuples, int batch_offset,
    int *d_supplier_s_acctbal, int *d_supplier_s_name, int *d_nation_n_name,
    int *d_part_p_partkey, int *d_part_p_mfgr,
    int *d_supplier_s_address, int *d_supplier_s_phone, int *d_supplier_s_comment,
    int *d_partsupp_ps_partkey, int *d_partsupp_ps_suppkey,
    int *d_supplier_s_nationkey, int *d_nation_n_regionkey,
    int *d_partsupp_ps_supplycost,
    double *d___scalar_sq_1_min_ps_supplycost,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_part, uint32_t bloom_size_bits_part,
    HtEntry *ht_part, uint32_t ht_size_part,
    const uint64_t *__restrict__ d_bloom_supplier, uint32_t bloom_size_bits_supplier,
    HtEntry *ht_supplier, uint32_t ht_size_supplier,
    const uint64_t *__restrict__ d_bloom_nation, uint32_t bloom_size_bits_nation,
    HtEntry *ht_nation, uint32_t ht_size_nation,
    const uint64_t *__restrict__ d_bloom_region, uint32_t bloom_size_bits_region,
    HtEntry *ht_region, uint32_t ht_size_region,
    const uint64_t *__restrict__ d_bloom_sq1, uint32_t bloom_size_bits_sq1,
    HtEntry *ht_sq1, uint32_t ht_size_sq1,
    int *d_out_count,
    int *d_out_col0, int *d_out_col1, int *d_out_col2, int *d_out_col3,
    int *d_out_col4, int *d_out_col5, int *d_out_col6, int *d_out_col7)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int loc_ps_partkey[IPT], loc_ps_suppkey[IPT], loc_ps_supplycost[IPT];
  int loc_s_acctbal[IPT], loc_s_name[IPT], loc_s_address[IPT], loc_s_phone[IPT], loc_s_comment[IPT];
  int loc_s_nationkey[IPT], loc_n_name[IPT], loc_n_regionkey[IPT];
  int loc_p_partkey[IPT], loc_p_mfgr[IPT];
  double loc_sq1_min_cost[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // Phase 1: Bloom probes
    BlockLoadSelect<int, BT, IPT>(d_partsupp_ps_partkey + tile_offset, loc_ps_partkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(loc_ps_partkey, selection_flags, num_tile_items, bloom_size_bits_part, d_bloom_part, 3);
    BlockLoadSelect<int, BT, IPT>(d_partsupp_ps_suppkey + tile_offset, loc_ps_suppkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(loc_ps_suppkey, selection_flags, num_tile_items, bloom_size_bits_supplier, d_bloom_supplier, 3);
    FASCAL_CHECK_ALIVE(IPT);

    // Phase 2: Hash table joins
    // Load supplycost before join (for post-join check)
    BlockLoadSelect<int, BT, IPT>(d_partsupp_ps_supplycost + tile_offset, loc_ps_supplycost, selection_flags, num_tile_items);

    // Join 0: ps_partkey -> part (rid-based, fetch p_partkey + p_mfgr)
    int loc_part_rid[IPT];
    BlockJoinProbePayloadDirect<BT, IPT>(loc_ps_partkey, selection_flags, num_tile_items, ht_part, ht_size_part, 1, loc_part_rid);
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
      if (selection_flags[i]) {
        int rid = loc_part_rid[i];
        loc_p_partkey[i] = d_part_p_partkey[rid];
        loc_p_mfgr[i] = d_part_p_mfgr[rid];
      }
    }

    // Join 1: ps_suppkey -> supplier (rid-based, fetch many columns)
    int loc_supplier_rid[IPT];
    BlockJoinProbePayloadDirect<BT, IPT>(loc_ps_suppkey, selection_flags, num_tile_items, ht_supplier, ht_size_supplier, 1, loc_supplier_rid);
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
      if (selection_flags[i]) {
        int rid = loc_supplier_rid[i];
        loc_s_acctbal[i] = d_supplier_s_acctbal[rid];
        loc_s_name[i] = d_supplier_s_name[rid];
        loc_s_address[i] = d_supplier_s_address[rid];
        loc_s_phone[i] = d_supplier_s_phone[rid];
        loc_s_comment[i] = d_supplier_s_comment[rid];
        loc_s_nationkey[i] = d_supplier_s_nationkey[rid];
      }
    }

    // Join 2: s_nationkey -> nation (rid-based, fetch n_name + n_regionkey)
    int loc_nation_rid[IPT];
    BlockJoinProbePayloadDirect<BT, IPT>(loc_s_nationkey, selection_flags, num_tile_items, ht_nation, ht_size_nation, 0, loc_nation_rid);
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
      if (selection_flags[i]) {
        int rid = loc_nation_rid[i];
        loc_n_name[i] = d_nation_n_name[rid];
        loc_n_regionkey[i] = d_nation_n_regionkey[rid];
      }
    }

    // Join 3: n_regionkey -> region
    BlockJoinProbeDirect<BT, IPT>(loc_n_regionkey, selection_flags, num_tile_items, ht_region, ht_size_region, 0);

    // Join 4: p_partkey -> __scalar_sq_1 (rid-based, fetch min_supplycost)
    int loc_sq1_rid[IPT];
    BlockJoinProbePayload<BT, IPT>(loc_p_partkey, selection_flags, num_tile_items, ht_sq1, ht_size_sq1, loc_sq1_rid);
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
      if (selection_flags[i])
        loc_sq1_min_cost[i] = d___scalar_sq_1_min_ps_supplycost[loc_sq1_rid[i]];
    }

    // Post-join check: ps_supplycost == min_supplycost
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if (selection_flags[i]) {
        if (!(((double)loc_ps_supplycost[i]) / 100.0 == loc_sq1_min_cost[i]))
          selection_flags[i] = 0;
      }
    }

    // Materialize passing rows
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i]) {
        int idx = atomicAdd(d_out_count, 1);
        d_out_col0[idx] = loc_s_acctbal[i];
        d_out_col1[idx] = loc_s_name[i];
        d_out_col2[idx] = loc_n_name[i];
        d_out_col3[idx] = loc_p_partkey[i];
        d_out_col4[idx] = loc_p_mfgr[i];
        d_out_col5[idx] = loc_s_address[i];
        d_out_col6[idx] = loc_s_phone[i];
        d_out_col7[idx] = loc_s_comment[i];
      }
    }
  } while(0);
}

// --- Stage 2 CPU predicate: Bloom probes on ps_partkey + ps_suppkey ---
static void s2_cpu_predicate_partsupp(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bf_array[], int num_bf) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  if (num_bf > 0) fascal::cpu_pred::bloom_probe_pass(bm, bs, h_data[0], offset, cnt, bf_array[0]);
  if (num_bf > 1) fascal::cpu_pred::bloom_probe_pass(bm, bs, h_data[1], offset, cnt, bf_array[1]);
}

// --- Stage 2 build kernels ---
template <int BT, int IPT>
__global__ void s2_kernel_part(
    int num_tuples,
    const int *__restrict__ d_p_partkey, const int *__restrict__ d_p_size,
    const int *__restrict__ d_p_type, const int *__restrict__ d_p_mfgr,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_sq1, uint32_t ht_size_f_sq1)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_partkey[IPT], loc_size[IPT], loc_type[IPT], loc_mfgr[IPT];

  BlockLoadSelect<int, BT, IPT>((int*)(d_p_partkey + tile_offset), loc_partkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_p_size + tile_offset), loc_size, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(loc_size, 38, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_p_type + tile_offset), loc_type, selection_flags, num_tile_items);

  // OR predicate: p_type ends with BRASS (encoded values % 5 == 4)
  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) < num_tile_items && selection_flags[i])
      selection_flags[i] = (((loc_type[i] == 4)) || ((loc_type[i] == 9)) || ((loc_type[i] == 14)) || ((loc_type[i] == 19)) || ((loc_type[i] == 24)) || ((loc_type[i] == 29)) || ((loc_type[i] == 34)) || ((loc_type[i] == 39)) || ((loc_type[i] == 44)) || ((loc_type[i] == 49)) || ((loc_type[i] == 54)) || ((loc_type[i] == 59)) || ((loc_type[i] == 64)) || ((loc_type[i] == 69)) || ((loc_type[i] == 74)) || ((loc_type[i] == 79)) || ((loc_type[i] == 84)) || ((loc_type[i] == 89)) || ((loc_type[i] == 94)) || ((loc_type[i] == 99)) || ((loc_type[i] == 104)) || ((loc_type[i] == 109)) || ((loc_type[i] == 114)) || ((loc_type[i] == 119)) || ((loc_type[i] == 124)) || ((loc_type[i] == 129)) || ((loc_type[i] == 134)) || ((loc_type[i] == 139)) || ((loc_type[i] == 144)) || ((loc_type[i] == 149))) ? 1 : 0;
  }

  BlockLoadSelect<int, BT, IPT>((int*)(d_p_mfgr + tile_offset), loc_mfgr, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    if (!gpu_ht_probe(loc_partkey[i], d_filter_ht_sq1, ht_size_f_sq1, nullptr)) continue;
    int payload = tile_offset + threadIdx.x + i * BT;
    gpu_ht_insert_one_direct(loc_partkey[i], payload, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_partkey[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

template <int BT, int IPT>
__global__ void s2_kernel_supplier(
    int num_tuples,
    const int *__restrict__ d_s_suppkey, const int *__restrict__ d_s_nationkey,
    const int *__restrict__ d_s_acctbal, const int *__restrict__ d_s_name,
    const int *__restrict__ d_s_address, const int *__restrict__ d_s_phone,
    const int *__restrict__ d_s_comment,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_nation, uint32_t ht_size_f_nation)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_suppkey[IPT], loc_nationkey[IPT];
  int loc_acctbal[IPT], loc_name[IPT], loc_address[IPT], loc_phone[IPT], loc_comment[IPT];

  BlockLoadSelect<int, BT, IPT>((int*)(d_s_suppkey + tile_offset), loc_suppkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_nationkey + tile_offset), loc_nationkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_acctbal + tile_offset), loc_acctbal, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_name + tile_offset), loc_name, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_address + tile_offset), loc_address, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_phone + tile_offset), loc_phone, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_s_comment + tile_offset), loc_comment, selection_flags, num_tile_items);

  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    if (!gpu_ht_probe_direct(loc_nationkey[i], d_filter_ht_nation, ht_size_f_nation, 0, nullptr)) continue;
    int payload = tile_offset + threadIdx.x + i * BT;
    gpu_ht_insert_one_direct(loc_suppkey[i], payload, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_suppkey[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

template <int BT, int IPT>
__global__ void s2_kernel_nation(
    int num_tuples,
    const int *__restrict__ d_n_nationkey, const int *__restrict__ d_n_regionkey,
    const int *__restrict__ d_n_name,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_region, uint32_t ht_size_f_region)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_nationkey[IPT], loc_regionkey[IPT], loc_name[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_n_nationkey + tile_offset), loc_nationkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_n_regionkey + tile_offset), loc_regionkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_n_name + tile_offset), loc_name, selection_flags, num_tile_items);
  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    if (!gpu_ht_probe_direct(loc_regionkey[i], d_filter_ht_region, ht_size_f_region, 0, nullptr)) continue;
    int payload = tile_offset + threadIdx.x + i * BT;
    gpu_ht_insert_one_direct(loc_nationkey[i], payload, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(loc_nationkey[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

template <int BT, int IPT>
__global__ void s2_kernel_region(
    int num_tuples,
    const int *__restrict__ d_r_regionkey, const int *__restrict__ d_r_name,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_regionkey[IPT], loc_name[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_r_regionkey + tile_offset), loc_regionkey, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_r_name + tile_offset), loc_name, selection_flags, num_tile_items);
  BlockPredAndEQ<int, BT, IPT>(loc_name, 4, selection_flags, num_tile_items);
  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    int payload = tile_offset + threadIdx.x + i * BT;
    gpu_ht_insert_one_direct(loc_regionkey[i], payload, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(loc_regionkey[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

template <int BT, int IPT>
__global__ void s2_kernel_sq1(
    int num_tuples,
    const int *__restrict__ d_sq1_partkey,
    const double *__restrict__ d_sq1_min_cost,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary)
{
  FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT);
  int loc_partkey[IPT];
  double loc_min_cost[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_sq1_partkey + tile_offset), loc_partkey, selection_flags, num_tile_items);
  BlockLoadSelect<double, BT, IPT>((double*)(d_sq1_min_cost + tile_offset), loc_min_cost, selection_flags, num_tile_items);
  #pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((threadIdx.x + BT * i) >= num_tile_items) break;
    if (!selection_flags[i]) continue;
    unsigned long long key = static_cast<unsigned int>(loc_partkey[i]);
    int payload = tile_offset + threadIdx.x + i * BT;
    gpu_ht_insert_one(key, payload, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_partkey[i], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// --- Stage 2 build CPU prefilters ---
static void s2_cpu_prefilter_region(int *h_r_regionkey, int *h_r_name, int n,
    uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      if (h_r_name[idx] == 4) bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
  });
}

static void s2_cpu_prefilter_sq1(int *h_partkey, double *h_min_cost, int n,
    uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
  });
}

static void s2_cpu_prefilter_nation(int *h_n_nationkey, int *h_n_regionkey, int *h_n_name,
    int n, uint32_t *bm, uint8_t *ts, fascal::runtime::BloomFilter *upstream_bf) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (upstream_bf && upstream_bf->bits)
      upstream_bf->probe_and_mask_packed(h_n_regionkey + off, bm, off, cnt);
  });
}

static void s2_cpu_prefilter_part(int *h_p_partkey, int *h_p_size, int *h_p_type, int *h_p_mfgr,
    int n, uint32_t *bm, uint8_t *ts, fascal::runtime::BloomFilter *upstream_bf) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      if ((h_p_size[idx] == 38) && ((h_p_type[idx] == 4) || (h_p_type[idx] == 9) || (h_p_type[idx] == 14) || (h_p_type[idx] == 19) || (h_p_type[idx] == 24) || (h_p_type[idx] == 29) || (h_p_type[idx] == 34) || (h_p_type[idx] == 39) || (h_p_type[idx] == 44) || (h_p_type[idx] == 49) || (h_p_type[idx] == 54) || (h_p_type[idx] == 59) || (h_p_type[idx] == 64) || (h_p_type[idx] == 69) || (h_p_type[idx] == 74) || (h_p_type[idx] == 79) || (h_p_type[idx] == 84) || (h_p_type[idx] == 89) || (h_p_type[idx] == 94) || (h_p_type[idx] == 99) || (h_p_type[idx] == 104) || (h_p_type[idx] == 109) || (h_p_type[idx] == 114) || (h_p_type[idx] == 119) || (h_p_type[idx] == 124) || (h_p_type[idx] == 129) || (h_p_type[idx] == 134) || (h_p_type[idx] == 139) || (h_p_type[idx] == 144) || (h_p_type[idx] == 149)))
        bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (upstream_bf && upstream_bf->bits)
      upstream_bf->probe_and_mask_packed(h_p_partkey + off, bm, off, cnt);
  });
}

static void s2_cpu_prefilter_supplier(int *h_s_suppkey, int *h_s_nationkey,
    int *h_s_acctbal, int *h_s_name, int *h_s_address, int *h_s_phone, int *h_s_comment,
    int n, uint32_t *bm, uint8_t *ts, fascal::runtime::BloomFilter *upstream_bf) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i;
      bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (upstream_bf && upstream_bf->bits)
      upstream_bf->probe_and_mask_packed(h_s_nationkey + off, bm, off, cnt);
  });
}

// =============================================================================
// Helper: allocate pinned mapped AFP prefilter for a dim table
// =============================================================================
struct DimPrefilter {
  uint32_t *h_pinned; uint32_t *d_mapped;
  uint8_t *h_tile_summary; uint8_t *d_tile_summary;
};
static DimPrefilter alloc_dim_prefilter(int n) {
  DimPrefilter pf = {};
  size_t bm_words = ((size_t)n + 31) / 32;
  size_t n_tiles = ((size_t)n + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
  CUDA_CHECK(cudaHostAlloc(&pf.h_pinned, bm_words * sizeof(uint32_t), cudaHostAllocMapped));
  std::memset(pf.h_pinned, 0, bm_words * sizeof(uint32_t));
  CUDA_CHECK(cudaHostGetDevicePointer(&pf.d_mapped, pf.h_pinned, 0));
  CUDA_CHECK(cudaHostAlloc(&pf.h_tile_summary, n_tiles * sizeof(uint8_t), cudaHostAllocMapped));
  std::memset(pf.h_tile_summary, 0, n_tiles);
  CUDA_CHECK(cudaHostGetDevicePointer(&pf.d_tile_summary, pf.h_tile_summary, 0));
  return pf;
}
static void free_dim_prefilter(DimPrefilter &pf) {
  CUDA_CHECK(cudaFreeHost(pf.h_tile_summary));
  CUDA_CHECK(cudaFreeHost(pf.h_pinned));
}

// =============================================================================
// Helper: ODZC map a dim column with fallback
// =============================================================================
struct OdzcDimCol {
  fascal::runtime::ODZCManager::MappedColumn mc;
  void *d_ptr;
};
template<typename T>
static OdzcDimCol odzc_dim_map(fascal::runtime::ODZCManager *mgr, T *h_ptr, int n, const char *name) {
  OdzcDimCol c;
  c.mc = mgr->register_column(h_ptr, (size_t)n * sizeof(T));
  c.d_ptr = c.mc.device_ptr;
  if (!c.d_ptr) {
    fprintf(stderr, "ODZC failed for %s, fallback to cudaMalloc\n", name);
    CUDA_CHECK(cudaMalloc(&c.d_ptr, (size_t)n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(c.d_ptr, h_ptr, (size_t)n * sizeof(T), cudaMemcpyHostToDevice));
  }
  return c;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
  const char *data_dir = getenv("FASCALSQL_DATA_DIR");
  if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
  if (argc >= 3) data_dir = argv[2];
  if (!data_dir) { fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n"); return 1; }

  printf("=== FaScalSQL: tpch_q2 (2-stage) ===\n");
  if (argc < 3) { printf("Usage: %s <num_tuples> <data_dir>\n", argv[0]); return 1; }
  data_dir = argv[2];

  // Shared subquery materializations
  int _mat_rows_sq1 = 0;
  int *_mat_sq1_partkey = nullptr;
  double *_mat_sq1_min_cost = nullptr;
  auto odzc_mgr_outer = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));

  // ================================================================
  // STAGE 1: tpch_q2___scalar_sq_1
  // ================================================================
  {
    printf("=== FaScalSQL: tpch_q2___scalar_sq_1 ===\n");

    // Auto-detect num_tuples
    int num_tuples = 0;
    { char p[512]; snprintf(p, sizeof(p), "%s/ps_suppkey.bin", data_dir);
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
    fascal_arena_init((size_t)num_tuples * sizeof(int) * 8);

    // Load fact columns
    char path[512];
    #define LOAD_COL(var, name) \
      snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
      int *var = load_binary_column(path, num_tuples); \
      if (!var) { fprintf(stderr, "Failed to load " name "\n"); return 1; }
    LOAD_COL(h_ps_suppkey, "ps_suppkey")
    LOAD_COL(h_ps_supplycost, "ps_supplycost")
    LOAD_COL(h_ps_partkey, "ps_partkey")
    LOAD_COL(h_ps_partkey_enc, "ps_partkey_enc")
    #undef LOAD_COL

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
    LOAD_DIM(h_n_regionkey, "n_regionkey", num_nation)
    printf("Loaded nation: %d tuples\n", num_nation);

    int num_supplier = 0;
    LOAD_DIM(h_s_suppkey, "s_suppkey", num_supplier)
    LOAD_DIM(h_s_nationkey, "s_nationkey", num_supplier)
    printf("Loaded supplier: %d tuples\n", num_supplier);
    #undef LOAD_DIM

    CUDA_CHECK(cudaEventRecord(t_load));

    // CQO
    auto cqo_in = fascal_cqo_init(num_tuples);
    // Dynamic BF selectivity: region(r_name=EUROPE=2), nation(region-filtered), supplier(all)
    int region_qual = (int)(fascal_estimate_eq_sel(h_r_name, num_region, 2) * num_region);
    int nation_qual = (int)(fascal_estimate_eq_sel(h_n_regionkey, num_nation, 2) * num_nation);
    cqo_in.join_bf_selectivities = {
      fascal_estimate_join_bf_sel(region_qual, num_region),
      fascal_estimate_join_bf_sel(nation_qual, num_nation),
      fascal_estimate_join_bf_sel(num_supplier, num_supplier)
    };
    cqo_in.num_gpu_only_ops = 0;
    auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
    fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);

    int in_memory_repeats = 1;
    if (const char *e = getenv("FASCAL_IN_MEMORY_REPEATS")) { in_memory_repeats = atoi(e); if (in_memory_repeats < 1) in_memory_repeats = 1; }
    int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr) || cqo_out.bloom_off;

    // AFP
    int *h_data[] = {h_ps_suppkey, h_ps_supplycost};
    fascal::runtime::BloomFilter bf_storage[3];
    fascal::runtime::BloomFilter *bf_array[3];
    bf_array[0] = (!bloom_off && 0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[2] : nullptr;
    bf_array[1] = nullptr; bf_array[2] = nullptr;

    // ODZC + seed bitmap
    auto odzc = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));
    uint32_t *h_bm = nullptr, *d_bm = nullptr; uint8_t *h_ts = nullptr, *d_ts = nullptr;
    fascal_alloc_seed_bitmap(num_tuples, &h_bm, &d_bm, &h_ts, &d_ts);

    #define ODZC_MAP(var, host, name) \
      fascal::runtime::ODZCManager::MappedColumn mc_##var; int *d_##var = nullptr; \
      fascal_odzc_map(odzc.get(), host, num_tuples, name, &mc_##var, &d_##var);
    ODZC_MAP(ps_suppkey, h_ps_suppkey, "ps_suppkey")
    ODZC_MAP(ps_supplycost, h_ps_supplycost, "ps_supplycost")
    #undef ODZC_MAP

    int *d_ps_partkey_enc = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_partkey_enc, num_tuples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ps_partkey_enc, h_ps_partkey_enc, num_tuples * sizeof(int), cudaMemcpyHostToDevice));

    // Dim ODZC
    auto oc_r_regionkey = odzc_dim_map(odzc.get(), h_r_regionkey, num_region, "region.r_regionkey");
    auto oc_r_name = odzc_dim_map(odzc.get(), h_r_name, num_region, "region.r_name");
    auto oc_n_nationkey = odzc_dim_map(odzc.get(), h_n_nationkey, num_nation, "nation.n_nationkey");
    auto oc_n_regionkey = odzc_dim_map(odzc.get(), h_n_regionkey, num_nation, "nation.n_regionkey");
    auto oc_s_suppkey = odzc_dim_map(odzc.get(), h_s_suppkey, num_supplier, "supplier.s_suppkey");
    auto oc_s_nationkey = odzc_dim_map(odzc.get(), h_s_nationkey, num_supplier, "supplier.s_nationkey");

    CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

    // Aggregation table (MIN per partkey_enc group)
    size_t num_groups = 20000001ULL;
    double *d_aggtable;
    CUDA_CHECK(cudaMalloc(&d_aggtable, num_groups * NUM_AGGREGATES * sizeof(double)));
    std::vector<double> h_agg_init(num_groups * NUM_AGGREGATES, 0.0);
    for (size_t i = 0; i < num_groups * NUM_AGGREGATES; i += NUM_AGGREGATES)
      h_agg_init[i] = std::numeric_limits<double>::infinity();
    CUDA_CHECK(cudaMemcpy(d_aggtable, h_agg_init.data(), num_groups * NUM_AGGREGATES * sizeof(double), cudaMemcpyHostToDevice));

    // Hash tables + bloom filters
    GpuHashTable ht_region = GpuHashTable::allocate_direct(num_region);
    GpuBloomFilter gpu_bf_region = GpuBloomFilter::allocate(num_region, 3);
    bf_storage[0] = fascal::runtime::BloomFilter::allocate(num_region, 3);
    CUDA_CHECK(cudaMemset(ht_region.d_entries, 0xff, ht_region.ht_size * sizeof(HtEntry)));
    CUDA_CHECK(cudaMemset(gpu_bf_region.d_bits, 0, ((size_t)gpu_bf_region.size_bits + 63) / 64 * sizeof(uint64_t)));
    auto pf_region = alloc_dim_prefilter(num_region);

    GpuHashTable ht_nation = GpuHashTable::allocate_direct(num_nation);
    GpuBloomFilter gpu_bf_nation = GpuBloomFilter::allocate(num_nation, 3);
    bf_storage[1] = fascal::runtime::BloomFilter::allocate(num_nation, 3);
    CUDA_CHECK(cudaMemset(ht_nation.d_entries, 0xff, ht_nation.ht_size * sizeof(HtEntry)));
    CUDA_CHECK(cudaMemset(gpu_bf_nation.d_bits, 0, ((size_t)gpu_bf_nation.size_bits + 63) / 64 * sizeof(uint64_t)));
    auto pf_nation = alloc_dim_prefilter(num_nation);

    GpuHashTable ht_supplier = GpuHashTable::allocate_direct(num_supplier);
    GpuBloomFilter gpu_bf_supplier = GpuBloomFilter::allocate(num_supplier, 3);
    bf_storage[2] = fascal::runtime::BloomFilter::allocate(num_supplier, 3);
    CUDA_CHECK(cudaMemset(ht_supplier.d_entries, 0xff, ht_supplier.ht_size * sizeof(HtEntry)));
    CUDA_CHECK(cudaMemset(gpu_bf_supplier.d_bits, 0, ((size_t)gpu_bf_supplier.size_bits + 63) / 64 * sizeof(uint64_t)));
    auto pf_supplier = alloc_dim_prefilter(num_supplier);

    // Pipeline setup
    const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
    const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
    cudaStream_t *streams = new cudaStream_t[N_STREAMS];
    for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
    int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

    auto run_kernel = [&]() {
      auto prefilter = [&](int bc, int bo, int tt) {
        fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
          [&](int off, int cnt) { sq1_cpu_predicate_partsupp(h_data, off, cnt, h_bm, bf_array, 3); }, bo, tt);
      };
      auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
        tpch_q2___scalar_sq_1_kernel_partsupp<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
          nt, bo, d_ps_suppkey, d_ps_supplycost, d_ps_partkey_enc, d_bm, d_ts,
          gpu_bf_supplier.d_bits, (bloom_off ? 0u : gpu_bf_supplier.size_bits), ht_supplier.d_entries, ht_supplier.ht_size,
          gpu_bf_nation.d_bits, (bloom_off ? 0u : gpu_bf_nation.size_bits), ht_nation.d_entries, ht_nation.ht_size,
          gpu_bf_region.d_bits, (bloom_off ? 0u : gpu_bf_region.size_bits), ht_region.d_entries, ht_region.ht_size,
          d_aggtable);
      };
        int nb = (num_tuples + GPU_BATCH - 1) / GPU_BATCH;
        for (int b = 0; b < nb; ++b) {
          int bo = b * GPU_BATCH, bc = std::min(GPU_BATCH, num_tuples - bo);
          prefilter(bc, bo, num_tuples);
          launch(num_tuples, bo, (bc + TILE_SIZE - 1) / TILE_SIZE, 0);
      }
      CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
    };

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    cudaEventRecord(t0);

    // Build pipelines: region -> nation -> supplier
    sq1_cpu_prefilter_region(h_r_regionkey, h_r_name, num_region, pf_region.h_pinned, pf_region.h_tile_summary);
    sq1_kernel_region<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_region + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_region, (int*)oc_r_regionkey.d_ptr, (int*)oc_r_name.d_ptr,
      ht_region.d_entries, ht_region.ht_size, gpu_bf_region.d_bits, gpu_bf_region.size_bits, 3,
      pf_region.d_mapped, pf_region.d_tile_summary);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
    if (bf_storage[0].bits) gpu_bf_region.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));

    sq1_cpu_prefilter_nation(h_n_nationkey, h_n_regionkey, num_nation, pf_nation.h_pinned, pf_nation.h_tile_summary, (bf_storage[0].bits ? &bf_storage[0] : nullptr));
    sq1_kernel_nation<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_nation + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_nation, (int*)oc_n_nationkey.d_ptr, (int*)oc_n_regionkey.d_ptr,
      ht_nation.d_entries, ht_nation.ht_size, gpu_bf_nation.d_bits, gpu_bf_nation.size_bits, 3,
      pf_nation.d_mapped, pf_nation.d_tile_summary, ht_region.d_entries, ht_region.ht_size);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
    if (bf_storage[1].bits) gpu_bf_nation.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));

    sq1_cpu_prefilter_supplier(h_s_suppkey, h_s_nationkey, num_supplier, pf_supplier.h_pinned, pf_supplier.h_tile_summary, (bf_storage[1].bits ? &bf_storage[1] : nullptr));
    sq1_kernel_supplier<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_supplier + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_supplier, (int*)oc_s_suppkey.d_ptr, (int*)oc_s_nationkey.d_ptr,
      ht_supplier.d_entries, ht_supplier.ht_size, gpu_bf_supplier.d_bits, gpu_bf_supplier.size_bits, 3,
      pf_supplier.d_mapped, pf_supplier.d_tile_summary, ht_nation.d_entries, ht_nation.ht_size);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
    if (bf_storage[2].bits) gpu_bf_supplier.copy_to_host(bf_storage[2].bits, ((size_t)(bf_storage[2].size_bits + 63) / 64) * sizeof(uint64_t));

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
        std::vector<double> reinit(num_groups * NUM_AGGREGATES, 0.0);
        for (size_t i = 0; i < num_groups * NUM_AGGREGATES; i += NUM_AGGREGATES)
          reinit[i] = std::numeric_limits<double>::infinity();
        CUDA_CHECK(cudaMemcpy(d_aggtable, reinit.data(), num_groups * NUM_AGGREGATES * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t0)); run_kernel(); CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaDeviceSynchronize());
        float m; CUDA_CHECK(cudaEventElapsedTime(&m, t0, t1));
        total += m; if (m < best) best = m;
      }
      printf("In-memory result kernel: best %.3f ms avg %.3f ms (%d runs)\n", best, total / in_memory_repeats, in_memory_repeats);
    }

    // Materialize aggregation results to __scalar_sq_1
    double *h_agg = (double*)malloc(num_groups * NUM_AGGREGATES * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_agg, d_aggtable, num_groups * NUM_AGGREGATES * sizeof(double), cudaMemcpyDeviceToHost));

    // Build decode map: enc -> raw partkey
    std::vector<int> h_gb_decode(20000001, std::numeric_limits<int>::min());
    for (int i = 0; i < num_tuples; ++i) {
      int enc = h_ps_partkey_enc[i];
      if (enc >= 0 && enc < 20000001 && h_gb_decode[enc] == std::numeric_limits<int>::min())
        h_gb_decode[enc] = h_ps_partkey[i];
    }

    // Count active groups
    int mat_count = 0;
    for (size_t k = 0; k < num_groups; k++)
      if (h_agg[k * NUM_AGGREGATES] != std::numeric_limits<double>::infinity()) mat_count++;

    int *h_out_pk = (int*)malloc(mat_count * sizeof(int));
    double *h_out_cost = (double*)malloc(mat_count * sizeof(double));
    int idx = 0;
    for (size_t k = 0; k < num_groups; k++) {
      if (h_agg[k * NUM_AGGREGATES] != std::numeric_limits<double>::infinity()) {
        int key0 = (int)(k % 20000001ULL);
        int decoded = key0;
        if (key0 >= 0 && key0 < (int)h_gb_decode.size() && h_gb_decode[key0] != std::numeric_limits<int>::min())
          decoded = h_gb_decode[key0];
        h_out_pk[idx] = decoded;
        h_out_cost[idx] = h_agg[k * NUM_AGGREGATES];
        idx++;
      }
    }

    // Write materialized files
    char path_mat[512];
    snprintf(path_mat, sizeof(path_mat), "%s/__scalar_sq_1_ps_partkey.bin", data_dir);
    FILE *fp = fopen(path_mat, "wb");
    if (fp) { fwrite(h_out_pk, sizeof(int), mat_count, fp); fclose(fp); printf("Materialized %d rows to %s\n", mat_count, path_mat); }

    snprintf(path_mat, sizeof(path_mat), "%s/__scalar_sq_1_min_partsupp_ps_supplycost_.bin", data_dir);
    fp = fopen(path_mat, "wb");
    if (fp) { fwrite(h_out_cost, sizeof(double), mat_count, fp); fclose(fp); printf("Materialized %d rows to %s\n", mat_count, path_mat); }

    // Pin materialized data for stage 2
    _mat_rows_sq1 = mat_count;
    if (_mat_sq1_partkey) CUDA_CHECK(cudaFreeHost(_mat_sq1_partkey));
    if (mat_count > 0) {
      CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat_sq1_partkey), (size_t)mat_count * sizeof(int), cudaHostAllocDefault));
      std::memcpy(_mat_sq1_partkey, h_out_pk, (size_t)mat_count * sizeof(int));
    }
    if (_mat_sq1_min_cost) CUDA_CHECK(cudaFreeHost(_mat_sq1_min_cost));
    if (mat_count > 0) {
      CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat_sq1_min_cost), (size_t)mat_count * sizeof(double), cudaHostAllocDefault));
      std::memcpy(_mat_sq1_min_cost, h_out_cost, (size_t)mat_count * sizeof(double));
    }
    free(h_out_pk); free(h_out_cost);
    printf("Stage materialization complete: %d rows\n", mat_count);
    free(h_agg);

    // Timing
    CUDA_CHECK(cudaFree(d_aggtable));
    CUDA_CHECK(cudaEventRecord(t_result)); CUDA_CHECK(cudaDeviceSynchronize());
    { float ml, mb, mr;
      CUDA_CHECK(cudaEventElapsedTime(&ml, t_start, t_load));
      CUDA_CHECK(cudaEventElapsedTime(&mb, t_load, t_query));
      /* merged into query */
      CUDA_CHECK(cudaEventElapsedTime(&mr, t_query, t_result));
      printf("Timing: load=%.1fms query=%.1fms result=%.1fms total=%.1fms\n", ml, mb, mr, ml+mb+mr); }

    // Cleanup stage 1
    odzc->unregister_column(mc_ps_suppkey); odzc->unregister_column(mc_ps_supplycost);
    CUDA_CHECK(cudaFree(d_ps_partkey_enc));
    odzc->unregister_column(oc_r_regionkey.mc); odzc->unregister_column(oc_r_name.mc);
    odzc->unregister_column(oc_n_nationkey.mc); odzc->unregister_column(oc_n_regionkey.mc);
    odzc->unregister_column(oc_s_suppkey.mc); odzc->unregister_column(oc_s_nationkey.mc);
    free_dim_prefilter(pf_region); gpu_bf_region.free_filter(); ht_region.free_table(); bf_storage[0].free_filter();
    free_dim_prefilter(pf_nation); gpu_bf_nation.free_filter(); ht_nation.free_table(); bf_storage[1].free_filter();
    free_dim_prefilter(pf_supplier); gpu_bf_supplier.free_filter(); ht_supplier.free_table(); bf_storage[2].free_filter();
    fascal_free_column(h_r_regionkey); fascal_free_column(h_r_name);
    fascal_free_column(h_n_nationkey); fascal_free_column(h_n_regionkey);
    fascal_free_column(h_s_suppkey); fascal_free_column(h_s_nationkey);
    fascal_free_seed_bitmap(h_bm, h_ts);
    fascal_free_column(h_ps_suppkey); fascal_free_column(h_ps_supplycost); fascal_free_column(h_ps_partkey_enc);
    fascal_arena_destroy();
    for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
    delete[] streams;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaEventDestroy(t_start); cudaEventDestroy(t_load);     cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  }

  // ================================================================
  // STAGE 2: tpch_q2 (main query with top-100 output)
  // ================================================================
  {
    printf("=== FaScalSQL: tpch_q2 ===\n");

    // Auto-detect num_tuples
    int num_tuples = 0;
    { char p[512]; snprintf(p, sizeof(p), "%s/ps_partkey.bin", data_dir);
      int c = 0; int *t = load_binary_column_auto(p, &c); if (t) fascal_free_column(t);
      num_tuples = c; }
    if (num_tuples <= 0) { fprintf(stderr, "Cannot auto-detect num_tuples\n"); return 1; }
    printf("num_tuples=%d data_dir=%s\n", num_tuples, data_dir);

    // Timing
    cudaEvent_t t_start, t_load, t_query, t_result;
    CUDA_CHECK(cudaEventCreate(&t_start)); CUDA_CHECK(cudaEventCreate(&t_load));
    CUDA_CHECK(cudaEventCreate(&t_query));
    CUDA_CHECK(cudaEventCreate(&t_result)); CUDA_CHECK(cudaEventRecord(t_start));

    // Load fact columns
    char path[512];
    #define LOAD_COL(var, name) \
      snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
      int *var = load_binary_column(path, num_tuples); \
      if (!var) { fprintf(stderr, "Failed to load " name "\n"); return 1; }
    LOAD_COL(h_ps_partkey, "ps_partkey")
    LOAD_COL(h_ps_suppkey, "ps_suppkey")
    LOAD_COL(h_ps_supplycost, "ps_supplycost")
    #undef LOAD_COL

    // Load dimensions
    #define LOAD_DIM(var, name, cnt) \
      snprintf(path, sizeof(path), "%s/" name ".bin", data_dir); \
      int *var = load_binary_column_auto(path, &cnt); \
      if (!var || cnt == 0) { fprintf(stderr, "Failed to load " name "\n"); return 1; }

    int num_region = 0;
    LOAD_DIM(h_r_regionkey, "r_regionkey", num_region)
    LOAD_DIM(h_r_name, "r_name", num_region)
    printf("Loaded region: %d tuples\n", num_region);

    // __scalar_sq_1
    int num_sq1 = 0;
    int *h_sq1_partkey = nullptr;
    if (_mat_rows_sq1 > 0 && _mat_sq1_partkey) {
      num_sq1 = _mat_rows_sq1; h_sq1_partkey = _mat_sq1_partkey;
    } else {
      snprintf(path, sizeof(path), "%s/__scalar_sq_1_ps_partkey.bin", data_dir);
      h_sq1_partkey = load_binary_column_auto(path, &num_sq1);
    }
    if (!h_sq1_partkey || num_sq1 == 0) { fprintf(stderr, "Failed to load __scalar_sq_1 ps_partkey\n"); return 1; }

    double *h_sq1_min_cost = nullptr;
    if (_mat_rows_sq1 > 0 && _mat_sq1_min_cost) {
      num_sq1 = _mat_rows_sq1; h_sq1_min_cost = _mat_sq1_min_cost;
    } else {
      snprintf(path, sizeof(path), "%s/__scalar_sq_1_min_partsupp_ps_supplycost_.bin", data_dir);
      h_sq1_min_cost = (double*)load_binary_column_auto_u64(path, &num_sq1);
    }
    if (!h_sq1_min_cost || num_sq1 == 0) { fprintf(stderr, "Failed to load __scalar_sq_1 min_cost\n"); return 1; }
    printf("Loaded __scalar_sq_1: %d tuples\n", num_sq1);

    int num_nation = 0;
    LOAD_DIM(h_n_nationkey, "n_nationkey", num_nation)
    LOAD_DIM(h_n_regionkey, "n_regionkey", num_nation)
    LOAD_DIM(h_n_name, "n_name", num_nation)
    printf("Loaded nation: %d tuples\n", num_nation);

    int num_part = 0;
    LOAD_DIM(h_p_partkey, "p_partkey", num_part)
    LOAD_DIM(h_p_size, "p_size", num_part)
    LOAD_DIM(h_p_type, "p_type", num_part)
    LOAD_DIM(h_p_mfgr, "p_mfgr", num_part)
    printf("Loaded part: %d tuples\n", num_part);

    int num_supplier = 0;
    LOAD_DIM(h_s_suppkey, "s_suppkey", num_supplier)
    LOAD_DIM(h_s_nationkey, "s_nationkey", num_supplier)
    LOAD_DIM(h_s_acctbal, "s_acctbal", num_supplier)
    LOAD_DIM(h_s_name, "s_name", num_supplier)
    LOAD_DIM(h_s_address, "s_address", num_supplier)
    LOAD_DIM(h_s_phone, "s_phone", num_supplier)
    LOAD_DIM(h_s_comment, "s_comment", num_supplier)
    printf("Loaded supplier: %d tuples\n", num_supplier);
    #undef LOAD_DIM

    CUDA_CHECK(cudaEventRecord(t_load));

    // CQO
    auto cqo_in = fascal_cqo_init(num_tuples);
    // Dynamic BF selectivity: region(r_name=EUROPE=2), sq1(subquery), nation(region-filtered), part(p_size=15), supplier(all)
    { int rq = (int)(fascal_estimate_eq_sel(h_r_name, num_region, 2) * num_region);
      int nq = (int)(fascal_estimate_eq_sel(h_n_regionkey, num_nation, 2) * num_nation);
      int pq = (int)(fascal_estimate_eq_sel(h_p_size, num_part, 15) * num_part);
      cqo_in.join_bf_selectivities = {
        fascal_estimate_join_bf_sel(rq, num_region),
        0.05,  // subquery result
        fascal_estimate_join_bf_sel(nq, num_nation),
        fascal_estimate_join_bf_sel(pq, num_part),
        fascal_estimate_join_bf_sel(num_supplier, num_supplier)
      }; }
    cqo_in.num_gpu_only_ops = 3;
    auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
    fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);

    int in_memory_repeats = 1;
    if (const char *e = getenv("FASCAL_IN_MEMORY_REPEATS")) { in_memory_repeats = atoi(e); if (in_memory_repeats < 1) in_memory_repeats = 1; }
    int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr) || cqo_out.bloom_off;

    // AFP
    int *h_data[] = {h_ps_partkey, h_ps_suppkey, h_ps_supplycost};
    fascal::runtime::BloomFilter bf_storage[5];
    fascal::runtime::BloomFilter *bf_array[5];
    bf_array[0] = (!bloom_off && 0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[3] : nullptr;
    bf_array[1] = (!bloom_off && 1 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[1]) ? &bf_storage[4] : nullptr;
    bf_array[2] = nullptr; bf_array[3] = nullptr; bf_array[4] = nullptr;

    // ODZC + seed bitmap
    auto odzc = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));
    uint32_t *h_bm = nullptr, *d_bm = nullptr; uint8_t *h_ts = nullptr, *d_ts = nullptr;
    fascal_alloc_seed_bitmap(num_tuples, &h_bm, &d_bm, &h_ts, &d_ts);

    #define ODZC_MAP(var, host, name) \
      fascal::runtime::ODZCManager::MappedColumn mc_##var; int *d_##var = nullptr; \
      fascal_odzc_map(odzc.get(), host, num_tuples, name, &mc_##var, &d_##var);
    ODZC_MAP(ps_partkey, h_ps_partkey, "ps_partkey")
    ODZC_MAP(ps_suppkey, h_ps_suppkey, "ps_suppkey")
    ODZC_MAP(ps_supplycost, h_ps_supplycost, "ps_supplycost")
    #undef ODZC_MAP

    // Dim ODZC
    auto oc_r_regionkey = odzc_dim_map(odzc.get(), h_r_regionkey, num_region, "region.r_regionkey");
    auto oc_r_name = odzc_dim_map(odzc.get(), h_r_name, num_region, "region.r_name");
    auto oc_sq1_partkey = odzc_dim_map(odzc.get(), h_sq1_partkey, num_sq1, "sq1.partkey");
    auto oc_sq1_min_cost = odzc_dim_map(odzc.get(), h_sq1_min_cost, num_sq1, "sq1.min_cost");
    auto oc_n_nationkey = odzc_dim_map(odzc.get(), h_n_nationkey, num_nation, "nation.n_nationkey");
    auto oc_n_regionkey = odzc_dim_map(odzc.get(), h_n_regionkey, num_nation, "nation.n_regionkey");
    auto oc_n_name = odzc_dim_map(odzc.get(), h_n_name, num_nation, "nation.n_name");
    auto oc_p_partkey = odzc_dim_map(odzc.get(), h_p_partkey, num_part, "part.p_partkey");
    auto oc_p_size = odzc_dim_map(odzc.get(), h_p_size, num_part, "part.p_size");
    auto oc_p_type = odzc_dim_map(odzc.get(), h_p_type, num_part, "part.p_type");
    auto oc_p_mfgr = odzc_dim_map(odzc.get(), h_p_mfgr, num_part, "part.p_mfgr");
    auto oc_s_suppkey = odzc_dim_map(odzc.get(), h_s_suppkey, num_supplier, "supplier.s_suppkey");
    auto oc_s_nationkey = odzc_dim_map(odzc.get(), h_s_nationkey, num_supplier, "supplier.s_nationkey");
    auto oc_s_acctbal = odzc_dim_map(odzc.get(), h_s_acctbal, num_supplier, "supplier.s_acctbal");
    auto oc_s_name = odzc_dim_map(odzc.get(), h_s_name, num_supplier, "supplier.s_name");
    auto oc_s_address = odzc_dim_map(odzc.get(), h_s_address, num_supplier, "supplier.s_address");
    auto oc_s_phone = odzc_dim_map(odzc.get(), h_s_phone, num_supplier, "supplier.s_phone");
    auto oc_s_comment = odzc_dim_map(odzc.get(), h_s_comment, num_supplier, "supplier.s_comment");

    CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

    // Output buffers
    int *d_out_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));
    int *d_out_cols[8];
    for (int i = 0; i < 8; i++) { CUDA_CHECK(cudaMalloc(&d_out_cols[i], (size_t)num_tuples * sizeof(int))); }

    // Hash tables + bloom filters
    GpuHashTable ht_region = GpuHashTable::allocate_direct(num_region);
    GpuBloomFilter gpu_bf_region = GpuBloomFilter::allocate(num_region, 3);
    bf_storage[0] = fascal::runtime::BloomFilter::allocate(num_region, 3);
    CUDA_CHECK(cudaMemset(ht_region.d_entries, 0xff, ht_region.ht_size * sizeof(HtEntry)));
    CUDA_CHECK(cudaMemset(gpu_bf_region.d_bits, 0, ((size_t)gpu_bf_region.size_bits + 63) / 64 * sizeof(uint64_t)));
    auto pf_region = alloc_dim_prefilter(num_region);

    GpuHashTable ht_sq1 = GpuHashTable::allocate(num_sq1);
    GpuBloomFilter gpu_bf_sq1 = GpuBloomFilter::allocate(num_sq1, 3);
    bf_storage[1] = fascal::runtime::BloomFilter::allocate(num_sq1, 3);
    CUDA_CHECK(cudaMemset(ht_sq1.d_entries, 0xff, ht_sq1.ht_size * sizeof(HtEntry)));
    CUDA_CHECK(cudaMemset(gpu_bf_sq1.d_bits, 0, ((size_t)gpu_bf_sq1.size_bits + 63) / 64 * sizeof(uint64_t)));
    auto pf_sq1 = alloc_dim_prefilter(num_sq1);

    GpuHashTable ht_nation = GpuHashTable::allocate_direct(num_nation);
    GpuBloomFilter gpu_bf_nation = GpuBloomFilter::allocate(num_nation, 3);
    bf_storage[2] = fascal::runtime::BloomFilter::allocate(num_nation, 3);
    CUDA_CHECK(cudaMemset(ht_nation.d_entries, 0xff, ht_nation.ht_size * sizeof(HtEntry)));
    CUDA_CHECK(cudaMemset(gpu_bf_nation.d_bits, 0, ((size_t)gpu_bf_nation.size_bits + 63) / 64 * sizeof(uint64_t)));
    auto pf_nation = alloc_dim_prefilter(num_nation);

    GpuHashTable ht_part = GpuHashTable::allocate_direct(num_part);
    GpuBloomFilter gpu_bf_part = GpuBloomFilter::allocate(num_part, 3);
    bf_storage[3] = fascal::runtime::BloomFilter::allocate(num_part, 3);
    CUDA_CHECK(cudaMemset(ht_part.d_entries, 0xff, ht_part.ht_size * sizeof(HtEntry)));
    CUDA_CHECK(cudaMemset(gpu_bf_part.d_bits, 0, ((size_t)gpu_bf_part.size_bits + 63) / 64 * sizeof(uint64_t)));
    auto pf_part = alloc_dim_prefilter(num_part);

    GpuHashTable ht_supplier = GpuHashTable::allocate_direct(num_supplier);
    GpuBloomFilter gpu_bf_supplier = GpuBloomFilter::allocate(num_supplier, 3);
    bf_storage[4] = fascal::runtime::BloomFilter::allocate(num_supplier, 3);
    CUDA_CHECK(cudaMemset(ht_supplier.d_entries, 0xff, ht_supplier.ht_size * sizeof(HtEntry)));
    CUDA_CHECK(cudaMemset(gpu_bf_supplier.d_bits, 0, ((size_t)gpu_bf_supplier.size_bits + 63) / 64 * sizeof(uint64_t)));
    auto pf_supplier = alloc_dim_prefilter(num_supplier);

    // Pipeline setup
    const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
    const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
    cudaStream_t *streams = new cudaStream_t[N_STREAMS];
    for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
    int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

    auto run_kernel = [&]() {
      auto prefilter = [&](int bc, int bo, int tt) {
        fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
          [&](int off, int cnt) { s2_cpu_predicate_partsupp(h_data, off, cnt, h_bm, bf_array, 5); }, bo, tt);
      };
      auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
        tpch_q2_kernel_partsupp<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
          nt, bo,
          (int*)oc_s_acctbal.d_ptr, (int*)oc_s_name.d_ptr, (int*)oc_n_name.d_ptr,
          (int*)oc_p_partkey.d_ptr, (int*)oc_p_mfgr.d_ptr,
          (int*)oc_s_address.d_ptr, (int*)oc_s_phone.d_ptr, (int*)oc_s_comment.d_ptr,
          d_ps_partkey, d_ps_suppkey, (int*)oc_s_nationkey.d_ptr, (int*)oc_n_regionkey.d_ptr,
          d_ps_supplycost, (double*)oc_sq1_min_cost.d_ptr,
          d_bm, d_ts,
          gpu_bf_part.d_bits, (bloom_off ? 0u : gpu_bf_part.size_bits), ht_part.d_entries, ht_part.ht_size,
          gpu_bf_supplier.d_bits, (bloom_off ? 0u : gpu_bf_supplier.size_bits), ht_supplier.d_entries, ht_supplier.ht_size,
          gpu_bf_nation.d_bits, (bloom_off ? 0u : gpu_bf_nation.size_bits), ht_nation.d_entries, ht_nation.ht_size,
          gpu_bf_region.d_bits, (bloom_off ? 0u : gpu_bf_region.size_bits), ht_region.d_entries, ht_region.ht_size,
          gpu_bf_sq1.d_bits, (bloom_off ? 0u : gpu_bf_sq1.size_bits), ht_sq1.d_entries, ht_sq1.ht_size,
          d_out_count, d_out_cols[0], d_out_cols[1], d_out_cols[2], d_out_cols[3],
          d_out_cols[4], d_out_cols[5], d_out_cols[6], d_out_cols[7]);
      };
        int nb = (num_tuples + GPU_BATCH - 1) / GPU_BATCH;
        for (int b = 0; b < nb; ++b) {
          int bo = b * GPU_BATCH, bc = std::min(GPU_BATCH, num_tuples - bo);
          prefilter(bc, bo, num_tuples);
          launch(num_tuples, bo, (bc + TILE_SIZE - 1) / TILE_SIZE, 0);
      }
      CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
    };

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    cudaEventRecord(t0);

    // Build pipelines: region -> sq1 -> nation -> part -> supplier
    s2_cpu_prefilter_region(h_r_regionkey, h_r_name, num_region, pf_region.h_pinned, pf_region.h_tile_summary);
    s2_kernel_region<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_region + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_region, (int*)oc_r_regionkey.d_ptr, (int*)oc_r_name.d_ptr,
      ht_region.d_entries, ht_region.ht_size, gpu_bf_region.d_bits, gpu_bf_region.size_bits, 3,
      pf_region.d_mapped, pf_region.d_tile_summary);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
    if (bf_storage[0].bits) gpu_bf_region.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));

    s2_cpu_prefilter_sq1(h_sq1_partkey, h_sq1_min_cost, num_sq1, pf_sq1.h_pinned, pf_sq1.h_tile_summary);
    s2_kernel_sq1<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_sq1 + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_sq1, (int*)oc_sq1_partkey.d_ptr, (double*)oc_sq1_min_cost.d_ptr,
      ht_sq1.d_entries, ht_sq1.ht_size, gpu_bf_sq1.d_bits, gpu_bf_sq1.size_bits, 3,
      pf_sq1.d_mapped, pf_sq1.d_tile_summary);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
    if (bf_storage[1].bits) gpu_bf_sq1.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));

    s2_cpu_prefilter_nation(h_n_nationkey, h_n_regionkey, h_n_name, num_nation, pf_nation.h_pinned, pf_nation.h_tile_summary, (bf_storage[0].bits ? &bf_storage[0] : nullptr));
    s2_kernel_nation<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_nation + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_nation, (int*)oc_n_nationkey.d_ptr, (int*)oc_n_regionkey.d_ptr, (int*)oc_n_name.d_ptr,
      ht_nation.d_entries, ht_nation.ht_size, gpu_bf_nation.d_bits, gpu_bf_nation.size_bits, 3,
      pf_nation.d_mapped, pf_nation.d_tile_summary, ht_region.d_entries, ht_region.ht_size);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
    if (bf_storage[2].bits) gpu_bf_nation.copy_to_host(bf_storage[2].bits, ((size_t)(bf_storage[2].size_bits + 63) / 64) * sizeof(uint64_t));

    s2_cpu_prefilter_part(h_p_partkey, h_p_size, h_p_type, h_p_mfgr, num_part, pf_part.h_pinned, pf_part.h_tile_summary, (bf_storage[1].bits ? &bf_storage[1] : nullptr));
    s2_kernel_part<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_part + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_part, (int*)oc_p_partkey.d_ptr, (int*)oc_p_size.d_ptr, (int*)oc_p_type.d_ptr, (int*)oc_p_mfgr.d_ptr,
      ht_part.d_entries, ht_part.ht_size, gpu_bf_part.d_bits, gpu_bf_part.size_bits, 3,
      pf_part.d_mapped, pf_part.d_tile_summary, ht_sq1.d_entries, ht_sq1.ht_size);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
    if (bf_storage[3].bits) gpu_bf_part.copy_to_host(bf_storage[3].bits, ((size_t)(bf_storage[3].size_bits + 63) / 64) * sizeof(uint64_t));

    s2_cpu_prefilter_supplier(h_s_suppkey, h_s_nationkey, h_s_acctbal, h_s_name, h_s_address, h_s_phone, h_s_comment,
      num_supplier, pf_supplier.h_pinned, pf_supplier.h_tile_summary, (bf_storage[2].bits ? &bf_storage[2] : nullptr));
    s2_kernel_supplier<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_supplier + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS>>>(
      num_supplier, (int*)oc_s_suppkey.d_ptr, (int*)oc_s_nationkey.d_ptr,
      (int*)oc_s_acctbal.d_ptr, (int*)oc_s_name.d_ptr, (int*)oc_s_address.d_ptr, (int*)oc_s_phone.d_ptr, (int*)oc_s_comment.d_ptr,
      ht_supplier.d_entries, ht_supplier.ht_size, gpu_bf_supplier.d_bits, gpu_bf_supplier.size_bits, 3,
      pf_supplier.d_mapped, pf_supplier.d_tile_summary, ht_nation.d_entries, ht_nation.ht_size);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
    if (bf_storage[4].bits) gpu_bf_supplier.copy_to_host(bf_storage[4].bits, ((size_t)(bf_storage[4].size_bits + 63) / 64) * sizeof(uint64_t));

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
        CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));
        CUDA_CHECK(cudaEventRecord(t0)); run_kernel(); CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaDeviceSynchronize());
        float m; CUDA_CHECK(cudaEventElapsedTime(&m, t0, t1));
        total += m; if (m < best) best = m;
      }
      printf("In-memory result kernel: best %.3f ms avg %.3f ms (%d runs)\n", best, total / in_memory_repeats, in_memory_repeats);
    }

    // === Output: top-100 with sorting ===
    int h_out_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_out_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
    int *h_out[8];
    for (int i = 0; i < 8; i++) {
      h_out[i] = (int*)malloc((size_t)num_tuples * sizeof(int));
      CUDA_CHECK(cudaMemcpy(h_out[i], d_out_cols[i], (size_t)num_tuples * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // Sort: s_acctbal DESC, n_name ASC, s_name ASC, p_partkey ASC
    std::vector<int> row_order(h_out_count);
    for (int i = 0; i < h_out_count; i++) row_order[i] = i;
    std::sort(row_order.begin(), row_order.end(), [&](int a, int b) {
      if (h_out[0][a] != h_out[0][b]) return h_out[0][a] > h_out[0][b];
      if (h_out[2][a] != h_out[2][b]) return h_out[2][a] < h_out[2][b];
      if (h_out[1][a] != h_out[1][b]) return h_out[1][a] < h_out[1][b];
      if (h_out[3][a] != h_out[3][b]) return h_out[3][a] < h_out[3][b];
      return false;
    });

    int mat_count = std::min(h_out_count, 100);
    for (int i = 0; i < mat_count; i++) {
      int r = row_order[i];
      printf("ROW: %.2f", ((double)(long long)h_out[0][r]) / 100.0);
      // s_name (Supplier#NNNNNNNNN, supports any SF — supplier domain is
      // [1..num_supplier] which scales with SF=1/10/100/...).
      printf("|%s", [&]() -> const char* {
        static thread_local char buf[64];
        int val = 1 + (int)h_out[1][r];
        if (val < 1) return "UNKNOWN";
        snprintf(buf, sizeof(buf), "Supplier#%09d", val);
        return buf;
      }());
      // n_name
      printf("|%s", [&]() -> const char* {
        static const char* vals[] = { "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "CHINA", "EGYPT", "ETHIOPIA", "FRANCE", "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN", "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU", "ROMANIA", "RUSSIA", "SAUDI ARABIA", "UNITED KINGDOM", "UNITED STATES", "VIETNAM" };
        int t = (int)h_out[2][r];
        return (t >= 0 && t <= 24) ? vals[t] : "UNKNOWN";
      }());
      // p_partkey
      printf("|%d", (int)h_out[3][r]);
      // p_mfgr
      printf("|%s", [&]() -> const char* {
        static thread_local char buf[64];
        static const int nums[] = { 1, 2, 3, 4, 5 };
        int idx = (int)h_out[4][r];
        if (idx < 0 || idx >= 5) return "UNKNOWN";
        snprintf(buf, sizeof(buf), "Manufacturer#%d", nums[idx]);
        return buf;
      }());
      // s_address (dict lookup)
      printf("|%s", [&]() -> const char* {
        static std::unordered_map<int, std::string> dict;
        static bool loaded = false;
        if (!loaded) {
          // Try binary _dict_strings.bin + _dict_offsets[64].bin (SF>=10 high-cardinality)
          char ps[1024], po64[1024], po32[1024];
          snprintf(ps,   sizeof(ps),   "%s/supplier_s_address_dict_strings.bin", data_dir);
          snprintf(po64, sizeof(po64), "%s/supplier_s_address_dict_offsets64.bin", data_dir);
          snprintf(po32, sizeof(po32), "%s/supplier_s_address_dict_offsets.bin", data_dir);
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
          loaded = true;
        }
        auto it = dict.find((int)h_out[5][r]);
        return it != dict.end() ? it->second.c_str() : "UNKNOWN";
      }());
      // s_phone (dict lookup)
      printf("|%s", [&]() -> const char* {
        static std::unordered_map<int, std::string> dict;
        static bool loaded = false;
        if (!loaded) {
          char ps[1024], po64[1024], po32[1024];
          snprintf(ps,   sizeof(ps),   "%s/supplier_s_phone_dict_strings.bin", data_dir);
          snprintf(po64, sizeof(po64), "%s/supplier_s_phone_dict_offsets64.bin", data_dir);
          snprintf(po32, sizeof(po32), "%s/supplier_s_phone_dict_offsets.bin", data_dir);
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
          loaded = true;
        }
        auto it = dict.find((int)h_out[6][r]);
        return it != dict.end() ? it->second.c_str() : "UNKNOWN";
      }());
      // s_comment (dict lookup)
      printf("|%s", [&]() -> const char* {
        static std::unordered_map<int, std::string> dict;
        static bool loaded = false;
        if (!loaded) {
          char ps[1024], po64[1024], po32[1024];
          snprintf(ps,   sizeof(ps),   "%s/supplier_s_comment_dict_strings.bin", data_dir);
          snprintf(po64, sizeof(po64), "%s/supplier_s_comment_dict_offsets64.bin", data_dir);
          snprintf(po32, sizeof(po32), "%s/supplier_s_comment_dict_offsets.bin", data_dir);
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
          loaded = true;
        }
        auto it = dict.find((int)h_out[7][r]);
        return it != dict.end() ? it->second.c_str() : "UNKNOWN";
      }());
      printf("\n");
    }

    // Cleanup
    for (int i = 0; i < 8; i++) { free(h_out[i]); CUDA_CHECK(cudaFree(d_out_cols[i])); }
    CUDA_CHECK(cudaFree(d_out_count));
    CUDA_CHECK(cudaEventRecord(t_result)); CUDA_CHECK(cudaDeviceSynchronize());
    { float ml, mb, mr;
      CUDA_CHECK(cudaEventElapsedTime(&ml, t_start, t_load));
      CUDA_CHECK(cudaEventElapsedTime(&mb, t_load, t_query));
      /* merged into query */
      CUDA_CHECK(cudaEventElapsedTime(&mr, t_query, t_result));
      printf("Timing: load=%.1fms query=%.1fms result=%.1fms total=%.1fms\n", ml, mb, mr, ml+mb+mr); }

    odzc->unregister_column(mc_ps_partkey); odzc->unregister_column(mc_ps_suppkey); odzc->unregister_column(mc_ps_supplycost);
    odzc->unregister_column(oc_r_regionkey.mc); odzc->unregister_column(oc_r_name.mc);
    odzc->unregister_column(oc_sq1_partkey.mc); odzc->unregister_column(oc_sq1_min_cost.mc);
    odzc->unregister_column(oc_n_nationkey.mc); odzc->unregister_column(oc_n_regionkey.mc); odzc->unregister_column(oc_n_name.mc);
    odzc->unregister_column(oc_p_partkey.mc); odzc->unregister_column(oc_p_size.mc); odzc->unregister_column(oc_p_type.mc); odzc->unregister_column(oc_p_mfgr.mc);
    odzc->unregister_column(oc_s_suppkey.mc); odzc->unregister_column(oc_s_nationkey.mc); odzc->unregister_column(oc_s_acctbal.mc);
    odzc->unregister_column(oc_s_name.mc); odzc->unregister_column(oc_s_address.mc); odzc->unregister_column(oc_s_phone.mc); odzc->unregister_column(oc_s_comment.mc);
    free_dim_prefilter(pf_region); gpu_bf_region.free_filter(); ht_region.free_table(); bf_storage[0].free_filter();
    free_dim_prefilter(pf_sq1); gpu_bf_sq1.free_filter(); ht_sq1.free_table(); bf_storage[1].free_filter();
    if (h_sq1_partkey && (void*)h_sq1_partkey != (void*)_mat_sq1_partkey) fascal_free_column(h_sq1_partkey);
    if (h_sq1_min_cost && (void*)h_sq1_min_cost != (void*)_mat_sq1_min_cost) fascal_free_column(h_sq1_min_cost);
    free_dim_prefilter(pf_nation); gpu_bf_nation.free_filter(); ht_nation.free_table(); bf_storage[2].free_filter();
    free_dim_prefilter(pf_part); gpu_bf_part.free_filter(); ht_part.free_table(); bf_storage[3].free_filter();
    free_dim_prefilter(pf_supplier); gpu_bf_supplier.free_filter(); ht_supplier.free_table(); bf_storage[4].free_filter();
    fascal_free_column(h_r_regionkey); fascal_free_column(h_r_name);
    fascal_free_column(h_n_nationkey); fascal_free_column(h_n_regionkey); fascal_free_column(h_n_name);
    fascal_free_column(h_p_partkey); fascal_free_column(h_p_size); fascal_free_column(h_p_type); fascal_free_column(h_p_mfgr);
    fascal_free_column(h_s_suppkey); fascal_free_column(h_s_nationkey); fascal_free_column(h_s_acctbal);
    fascal_free_column(h_s_name); fascal_free_column(h_s_address); fascal_free_column(h_s_phone); fascal_free_column(h_s_comment);
    fascal_free_seed_bitmap(h_bm, h_ts);
    fascal_free_column(h_ps_partkey); fascal_free_column(h_ps_suppkey); fascal_free_column(h_ps_supplycost);
    for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
    delete[] streams;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaEventDestroy(t_start); cudaEventDestroy(t_load);     cudaEventDestroy(t_query); cudaEventDestroy(t_result);
    return 0;
  }

  // Cleanup shared pinned subquery materializations
  if (_mat_sq1_partkey) CUDA_CHECK(cudaFreeHost(_mat_sq1_partkey));
  if (_mat_sq1_min_cost) CUDA_CHECK(cudaFreeHost(_mat_sq1_min_cost));
}
