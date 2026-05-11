// TPC-H Q9 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Optimizations (Session 16):
//   (b) Probe reorder: part BF+HT checked first (most selective ~33% pass rate)
//   (c) partsupp HT -> direct sorted array: sort by (pk,sk), [num_parts x 4] layout,
//       eliminates 1.92 GB HT and hash collision overhead
//   (d) Parallel HT builds: orders/nation/supplier builds overlap via CUDA streams

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

// ---- GPU SQL LIKE Matching ----
__device__ bool sql_like_match(const char* s, const char* p) {
    const char* cp = nullptr; const char* cs = nullptr;
    while (*s && *p != '*') {
        if (*p == '%') { while (*p == '%') p++; if (!*p) return true; cp = p; cs = s + 1; }
        else if (*p == '_' || *p == *s) { p++; s++; }
        else if (cp) { p = cp; s = cs++; }
        else { return false; }
    }
    while (*p == '%') p++;
    return !*p && !*s;
}

// Bitset predicate for dictionary-encoded LIKE
template<typename T, int BT, int IPT>
__device__ void BlockPredAndBitset(const T* items, int* selection_flags, int num_items, const unsigned char* bitset) {
    #pragma unroll
    for (int i = 0; i < IPT; i++)
        if ((threadIdx.x + (BT * i)) < num_items && selection_flags[i])
            if (!bitset[items[i]]) selection_flags[i] = 0;
}

__global__ void regex_bitset_kernel(const char* dict_strings, const int* dict_offsets, int num_entries, const char* pattern, unsigned char* bitset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_entries) bitset[idx] = sql_like_match(dict_strings + dict_offsets[idx], pattern) ? 1 : 0;
}

// GPU-side bitset pointers for LIKE predicates
__constant__ unsigned char* d_pred_bitset_0;
unsigned char* h_pred_bitset_0 = nullptr;
unsigned char* d_pred_bitset_ptr_0 = nullptr;

// =============================================================================
// GPU Kernel: scan lineitem, 5 joins, GROUP BY nation+year, aggregate
//
// OPT (b): probe order rewritten for selectivity:
//   1. part BF probe (l_partkey) -- kills ~67% of rows via LIKE '%green%' filter
//   2. supplier BF probe (l_suppkey) -- after part prune
//   3. orders BF probe (l_orderkey_enc)
//   4. FASCAL_CHECK_ALIVE early exit
//   5. HT joins: part (verify+no payload), supplier (payload=s_nationkey)
//   6. direct partsupp lookup: ps_sk_direct[l_partkey-1][0..3] scan (4 entries)
//   7. orders HT (payload=o_orderdate), nation HT (payload=n_name)
//
// OPT (c): partsupp HT replaced by ps_sk_direct (int4 array) + ps_sc_direct (int array)
//   Both sized [num_parts x 4], stored as flat arrays.
//   Probe: unroll 4 comparisons on suppkeys, grab cost at matching slot.
// =============================================================================
template <int BT, int IPT>
__global__ void tpch_q9_kernel_lineitem(
    int num_tuples, int batch_offset,
    // OPT(c): direct partsupp arrays instead of HT
    const int4 *__restrict__ d_ps_sk_direct,   // [num_parts] x (sk0,sk1,sk2,sk3)
    const int  *__restrict__ d_ps_sc_direct,   // [num_parts x 4] flattened supplycost
    int *d_lineitem_l_partkey, int *d_lineitem_l_suppkey, int *d_lineitem_l_orderkey_enc,
    float *d_lineitem_l_extendedprice, float *d_lineitem_l_discount, float *d_lineitem_l_quantity,
    const uint32_t *__restrict__ seed_bitmap, const uint8_t *__restrict__ tile_summary,
    // part BF + HT (most selective, probe first)
    const uint64_t *__restrict__ d_bloom_part, uint32_t bloom_size_bits_part,
    HtEntry *ht_part, uint32_t ht_size_part,
    // supplier BF + HT
    const uint64_t *__restrict__ d_bloom_supplier, uint32_t bloom_size_bits_supplier,
    HtEntry *ht_supplier, uint32_t ht_size_supplier,
    // orders BF + HT
    const uint64_t *__restrict__ d_bloom_orders, uint32_t bloom_size_bits_orders,
    HtEntry *ht_orders, uint32_t ht_size_orders,
    // nation HT (no BF needed, keyed by s_nationkey which is tiny)
    HtEntry *ht_nation, uint32_t ht_size_nation,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BT, IPT);

  int loc_l_partkey[IPT], loc_l_suppkey[IPT], loc_l_orderkey_enc[IPT];
  float loc_l_extendedprice[IPT], loc_l_discount[IPT], loc_l_quantity[IPT];
  int loc_nation_n_name[IPT], loc_orders_o_orderdate[IPT];
  int loc_partsupp_ps_supplycost[IPT], loc_supplier_s_nationkey[IPT];

  if (FASCAL_ANY_ALIVE(IPT)) do {
    // OPT(b) Phase 1: Load keys and apply most-selective BF first (part, then supplier, then orders)
    BlockLoadSelect<int, BT, IPT>(d_lineitem_l_partkey + tile_offset, loc_l_partkey, selection_flags, num_tile_items);
    // part BF: kills ~67% of rows (only ~33% parts have 'green' in name)
    BlockBloomProbe<BT, IPT>(loc_l_partkey, selection_flags, num_tile_items, bloom_size_bits_part, d_bloom_part, 3);

    BlockLoadSelect<int, BT, IPT>(d_lineitem_l_suppkey + tile_offset, loc_l_suppkey, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(loc_l_suppkey, selection_flags, num_tile_items, bloom_size_bits_supplier, d_bloom_supplier, 3);

    BlockLoadSelect<int, BT, IPT>(d_lineitem_l_orderkey_enc + tile_offset, loc_l_orderkey_enc, selection_flags, num_tile_items);
    BlockBloomProbe<BT, IPT>(loc_l_orderkey_enc, selection_flags, num_tile_items, bloom_size_bits_orders, d_bloom_orders, 3);

    FASCAL_CHECK_ALIVE(IPT);

    // Phase 2: HT joins
    // Join 0: l_partkey -> part (existence + LIKE '%green%' filter already in HT)
    BlockJoinProbeDirect<BT, IPT>(loc_l_partkey, selection_flags, num_tile_items, ht_part, ht_size_part, 1);

    // Join 1: l_suppkey -> supplier [payload=s_nationkey]
    BlockJoinProbePayloadDirect<BT, IPT>(loc_l_suppkey, selection_flags, num_tile_items, ht_supplier, ht_size_supplier, 1, loc_supplier_s_nationkey);

    // OPT(c): Direct partsupp lookup -- no HT, no hash, no collision chain.
    // For each lineitem row: scan up to 4 suppkeys in ps_sk_direct[l_partkey-1],
    // find matching suppkey, load supplycost from ps_sc_direct[(l_partkey-1)*4 + slot].
    // If no match found (should not happen for valid data), kill the row.
    #pragma unroll
    for (int ITEM = 0; ITEM < IPT; ++ITEM) {
      if ((threadIdx.x + (BT * ITEM)) < num_tile_items && selection_flags[ITEM]) {
        int pk = loc_l_partkey[ITEM];
        int sk = loc_l_suppkey[ITEM];
        int base = (pk - 1) * 4;
        int4 sks = d_ps_sk_direct[pk - 1];
        int cost = -1;
        if (sks.x == sk) cost = d_ps_sc_direct[base + 0];
        else if (sks.y == sk) cost = d_ps_sc_direct[base + 1];
        else if (sks.z == sk) cost = d_ps_sc_direct[base + 2];
        else if (sks.w == sk) cost = d_ps_sc_direct[base + 3];
        if (cost < 0) { selection_flags[ITEM] = 0; continue; }
        loc_partsupp_ps_supplycost[ITEM] = cost;
      }
    }

    // Join 3: l_orderkey_enc -> orders [payload=o_orderdate]
    BlockJoinProbePayloadDirect<BT, IPT>(loc_l_orderkey_enc, selection_flags, num_tile_items, ht_orders, ht_size_orders, 0, loc_orders_o_orderdate);

    // Join 4: s_nationkey -> nation [payload=n_name]
    BlockJoinProbePayloadDirect<BT, IPT>(loc_supplier_s_nationkey, selection_flags, num_tile_items, ht_nation, ht_size_nation, 0, loc_nation_n_name);

    // Load aggregation columns
    BlockLoadSelect<float, BT, IPT>(d_lineitem_l_extendedprice + tile_offset, loc_l_extendedprice, selection_flags, num_tile_items);
    BlockLoadSelect<float, BT, IPT>(d_lineitem_l_discount + tile_offset, loc_l_discount, selection_flags, num_tile_items);
    BlockLoadSelect<float, BT, IPT>(d_lineitem_l_quantity + tile_offset, loc_l_quantity, selection_flags, num_tile_items);

    // GROUP BY nation+year, aggregate profit
    #pragma unroll
    for (int ITEM = 0; ITEM < IPT; ++ITEM) {
      if ((threadIdx.x + (BT * ITEM)) < num_tile_items && selection_flags[ITEM]) {
        int gk = loc_nation_n_name[ITEM] + ((((loc_orders_o_orderdate[ITEM] / 10000)) - 1992) * 25);
        double profit = ((double)loc_l_extendedprice[ITEM] * (1.0 - loc_l_discount[ITEM])) -
                        (((double)loc_partsupp_ps_supplycost[ITEM] / 100.0) * loc_l_quantity[ITEM]);
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], profit);
      }
    }
  } while(0);
}

// =============================================================================
// CPU Predicate: Bloom filter probes only (no partsupp BF since HT eliminated)
// =============================================================================
static void cpu_predicate_lineitem(int *h_data[], int offset, int cnt, uint32_t *bitmap,
    fascal::runtime::BloomFilter *bloom_filters[], int num_bloom_filters) {
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = offset >> 3, nb = (cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, cnt);
  int *l_partkey = h_data[0], *l_suppkey = h_data[1], *l_orderkey_enc = h_data[2];
  // OPT(b): part BF first (most selective)
  if (bloom_filters && 0 < num_bloom_filters) fascal::cpu_pred::bloom_probe_pass(bm, bs, l_partkey, offset, cnt, bloom_filters[0]);
  if (bloom_filters && 1 < num_bloom_filters) fascal::cpu_pred::bloom_probe_pass(bm, bs, l_suppkey, offset, cnt, bloom_filters[1]);
  if (bloom_filters && 2 < num_bloom_filters) fascal::cpu_pred::bloom_probe_pass(bm, bs, l_orderkey_enc, offset, cnt, bloom_filters[2]);
}

// =============================================================================
// Build kernels
// =============================================================================
template <int BT, int IPT>
__global__ void tpch_q9_kernel_part(
    int num_tuples, int batch_offset, const int *__restrict__ d_pk, const int *__restrict__ d_pname,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary) {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int loc_pk[IPT], loc_pname[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_pk + tile_offset), loc_pk, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_pname + tile_offset), loc_pname, selection_flags, num_tile_items);
  BlockPredAndBitset<int, BT, IPT>(loc_pname, selection_flags, num_tile_items, d_pred_bitset_0);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int payload = tile_offset + threadIdx.x + ITEM * BT;
    gpu_ht_insert_one_direct(loc_pk[ITEM], payload, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_pk[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_part(int *h_pk, int *h_pname, int n, uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; if (h_pred_bitset_0[h_pname[idx]]) bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
  });
}

template <int BT, int IPT>
__global__ void tpch_q9_kernel_supplier(
    int num_tuples, int batch_offset, const int *__restrict__ d_sk, const int *__restrict__ d_nk,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht, uint32_t ht_size_f) {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int loc_sk[IPT], loc_nk[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_sk + tile_offset), loc_sk, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_nk + tile_offset), loc_nk, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe_direct(loc_nk[ITEM], d_filter_ht, ht_size_f, 0, nullptr)) continue;
    gpu_ht_insert_one_direct(loc_sk[ITEM], loc_nk[ITEM], 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_sk[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_supplier(int *h_sk, int *h_nk, int n, uint32_t *bm, uint8_t *ts,
    fascal::runtime::BloomFilter *bf_nation) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
    if (bf_nation && bf_nation->bits) bf_nation->probe_and_mask_packed(h_nk + off, bm, off, cnt);
  });
}

template <int BT, int IPT>
__global__ void tpch_q9_kernel_orders(
    int num_tuples, int batch_offset, const int *__restrict__ d_ok, const int *__restrict__ d_od,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary) {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int loc_ok[IPT], loc_od[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_ok + tile_offset), loc_ok, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_od + tile_offset), loc_od, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int row_idx = tile_offset + threadIdx.x + ITEM * BT;
    gpu_ht_insert_one_direct(row_idx, loc_od[ITEM], 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(row_idx, 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_orders(int *h_ok, int *h_od, int n, uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
  });
}

template <int BT, int IPT>
__global__ void tpch_q9_kernel_nation(
    int num_tuples, int batch_offset, const int *__restrict__ d_nk, const int *__restrict__ d_name,
    HtEntry* d_ht, uint32_t ht_size, uint64_t *__restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter, const uint8_t *d_tile_summary) {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT);
  int loc_nk[IPT], loc_name[IPT];
  BlockLoadSelect<int, BT, IPT>((int*)(d_nk + tile_offset), loc_nk, selection_flags, num_tile_items);
  BlockLoadSelect<int, BT, IPT>((int*)(d_name + tile_offset), loc_name, selection_flags, num_tile_items);
  #pragma unroll
  for (int ITEM = 0; ITEM < IPT; ++ITEM) {
    if ((threadIdx.x + (BT * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    gpu_ht_insert_one_direct(loc_nk[ITEM], loc_name[ITEM], 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(loc_nk[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

static void cpu_prefilter_nation(int *h_nk, int *h_name, int n, uint32_t *bm, uint8_t *ts) {
  fascal_cpu_prefilter_run(n, bm, ts, [&](int off, int cnt) {
    uint32_t ws = (uint32_t)off >> 5, we = ((uint32_t)(off + cnt) + 31) >> 5;
    memset(&bm[ws], 0, (we - ws) * sizeof(uint32_t));
    for (int i = 0; i < cnt; ++i) { int idx = off + i; bm[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31)); }
  });
}

// =============================================================================
// OPT(c): CPU helper to build direct partsupp arrays from raw (pk,sk,sc) columns.
// Sorts by (pk,sk), then reshapes to [num_parts x 4] layout.
// Returns: ps_sk4 (int4*) and ps_sc_flat (int*) allocated via malloc.
// =============================================================================
static void build_partsupp_direct(
    const int *h_ps_pk, const int *h_ps_sk, const int *h_ps_sc,
    int num_partsupp, int num_parts,
    int4 **out_sk4, int **out_sc_flat) {

  // Build index array and sort by (pk, sk)
  std::vector<int> idx(num_partsupp);
  for (int i = 0; i < num_partsupp; i++) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    if (h_ps_pk[a] != h_ps_pk[b]) return h_ps_pk[a] < h_ps_pk[b];
    return h_ps_sk[a] < h_ps_sk[b];
  });

  int4 *sk4 = (int4*)malloc((size_t)num_parts * sizeof(int4));
  int  *sc_flat = (int*)malloc((size_t)num_parts * 4 * sizeof(int));
  if (!sk4 || !sc_flat) { fprintf(stderr, "OOM for partsupp direct arrays\n"); exit(1); }

  // Each part has exactly 4 entries in sorted order
  for (int i = 0; i < num_partsupp; i++) {
    int r = idx[i];
    int pk = h_ps_pk[r];
    int slot = i % 4;  // relies on exactly 4 per partkey
    int base = (pk - 1) * 4;
    int *sks_arr = (int*)&sk4[pk - 1];
    sks_arr[slot] = h_ps_sk[r];
    sc_flat[base + slot] = h_ps_sc[r];
  }

  *out_sk4 = sk4;
  *out_sc_flat = sc_flat;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
  const char *data_dir = getenv("FASCALSQL_DATA_DIR");
  if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
  if (argc >= 3) data_dir = argv[2];
  if (!data_dir) { fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n"); return 1; }

  // Regex bitset for part.p_name LIKE '%green%'
  {
    int dict_entry_count = 0;
    int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/part_p_name_dict_offsets.bin").c_str(), &dict_entry_count);
    if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for part.p_name\n"); exit(1); }
    long strings_size_bytes = 0;
    FILE* fp_s = fopen(std::string(std::string(data_dir) + "/part_p_name_dict_strings.bin").c_str(), "rb");
    if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for part.p_name\n"); exit(1); }
    fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
    char* h_dict_strings = (char*)malloc(strings_size_bytes);
    fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
    char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
    CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
    CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count));
    CUDA_CHECK(cudaMalloc(&d_pattern, 8));
    CUDA_CHECK(cudaMemcpy(d_pattern, "%green%", 8, cudaMemcpyHostToDevice));
    regex_bitset_kernel<<<(dict_entry_count + 255) / 256, 256>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
    CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_0, &d_bitset, sizeof(unsigned char*)));
    d_pred_bitset_ptr_0 = d_bitset;
    h_pred_bitset_0 = (unsigned char*)malloc(dict_entry_count);
    CUDA_CHECK(cudaMemcpy(h_pred_bitset_0, d_bitset, dict_entry_count, cudaMemcpyDeviceToHost));
    fascal_free_column(h_dict_offsets); free(h_dict_strings);
  }

  printf("=== FaScalSQL: tpch_q9 ===\n");
  if (argc < 3) { printf("Usage: %s <num_tuples> <data_dir>\n", argv[0]); return 1; }
  data_dir = argv[2];

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

  fascal_arena_init((size_t)num_tuples * sizeof(int) * 7);

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
  LOAD_COL(h_l_quantity, "l_quantity", float)

  int num_part = 0; LOAD_DIM(h_p_pk, "p_partkey", num_part) LOAD_DIM(h_p_name, "p_name", num_part) printf("Loaded part: %d\n", num_part);
  int num_partsupp = 0; LOAD_DIM(h_ps_sk, "ps_suppkey", num_partsupp) LOAD_DIM(h_ps_sc, "ps_supplycost", num_partsupp) LOAD_DIM(h_ps_pk, "ps_partkey", num_partsupp) printf("Loaded partsupp: %d\n", num_partsupp);
  int num_orders = 0; LOAD_DIM(h_o_ok, "o_orderkey", num_orders) LOAD_DIM(h_o_od, "o_orderdate", num_orders) printf("Loaded orders: %d\n", num_orders);
  int num_nation = 0; LOAD_DIM(h_n_nk, "n_nationkey", num_nation) LOAD_DIM(h_n_name, "n_name", num_nation) printf("Loaded nation: %d\n", num_nation);
  int num_supplier = 0; LOAD_DIM(h_s_sk, "s_suppkey", num_supplier) LOAD_DIM(h_s_nk, "s_nationkey", num_supplier) printf("Loaded supplier: %d\n", num_supplier);
  #undef LOAD_COL
  #undef LOAD_DIM
  CUDA_CHECK(cudaEventRecord(t_load));

  // CQO -- partsupp BF eliminated (replaced by direct array).
  // BF slots correspond to CPU prefilter probe order:
  //   slot 0: part     (probes l_partkey)
  //   slot 1: supplier (probes l_suppkey)
  //   slot 2: orders   (probes l_orderkey_enc)
  // nation BF is only used to filter supplier table at build time, not in lineitem prefilter.
  auto cqo_in = fascal_cqo_init(num_tuples);
  int part_qual = (int)(0.33 * num_part);
  cqo_in.join_bf_selectivities = {
    fascal_estimate_join_bf_sel(part_qual, num_part),        // slot 0: part     -> l_partkey
    fascal_estimate_join_bf_sel(num_supplier, num_supplier), // slot 1: supplier -> l_suppkey
    fascal_estimate_join_bf_sel(num_orders, num_orders),     // slot 2: orders   -> l_orderkey_enc
  };
  cqo_in.num_gpu_only_ops = 4;
  auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
  fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);
  int in_memory_repeats = 1;
  if (const char *e = getenv("FASCAL_IN_MEMORY_REPEATS")) { in_memory_repeats = atoi(e); if (in_memory_repeats < 1) in_memory_repeats = 1; }
  int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr) || cqo_out.bloom_off;

  int *h_data[] = {h_l_partkey, h_l_suppkey, h_l_orderkey_enc, (int*)h_l_extendedprice, (int*)h_l_discount, (int*)h_l_quantity};
  // OPT(c): only 3 BFs now (part, supplier, orders). partsupp BF removed.
  // bf_storage[0]=part, bf_storage[1]=supplier, bf_storage[2]=orders, bf_storage[3]=nation
  fascal::runtime::BloomFilter bf_storage[4];
  // CPU prefilter BF pointers -- indexed same as cqo_in.join_bf_selectivities
  fascal::runtime::BloomFilter *bf_part_cpu     = (!bloom_off && 0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[0] : nullptr;
  fascal::runtime::BloomFilter *bf_supplier_cpu = (!bloom_off && 1 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[1]) ? &bf_storage[1] : nullptr;
  fascal::runtime::BloomFilter *bf_orders_cpu   = (!bloom_off && 2 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[2]) ? &bf_storage[2] : nullptr;

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

  ODZC_INT(l_pk, h_l_partkey, num_tuples, "l_partkey")
  ODZC_INT(l_sk, h_l_suppkey, num_tuples, "l_suppkey")
  ODZC_INT(l_ok, h_l_orderkey_enc, num_tuples, "l_orderkey_enc")
  ODZC_FLOAT(l_ep, h_l_extendedprice, num_tuples, "l_extendedprice")
  ODZC_FLOAT(l_disc, h_l_discount, num_tuples, "l_discount")
  ODZC_FLOAT(l_qty, h_l_quantity, num_tuples, "l_quantity")
  ODZC_DIM(p_pk, h_p_pk, num_part, "p_pk") ODZC_DIM(p_name, h_p_name, num_part, "p_name")
  ODZC_DIM(o_ok, h_o_ok, num_orders, "o_ok") ODZC_DIM(o_od, h_o_od, num_orders, "o_od")
  ODZC_DIM(n_nk, h_n_nk, num_nation, "n_nk") ODZC_DIM(n_name, h_n_name, num_nation, "n_name")
  ODZC_DIM(s_sk, h_s_sk, num_supplier, "s_sk") ODZC_DIM(s_nk, h_s_nk, num_supplier, "s_nk")
  #undef ODZC_INT
  #undef ODZC_FLOAT
  #undef ODZC_DIM

  CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));

  // Aggregation: 25 nations x 7 years
  size_t num_groups = 25ULL * 7ULL;
  double *d_agg;
  CUDA_CHECK(cudaMalloc(&d_agg, num_groups * NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_agg, 0, num_groups * NUM_AGGREGATES * sizeof(double)));

  // Build HTs + prefilters (no partsupp HT)
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

  ALLOC_BUILD(part, num_part, allocate_direct, 0)
  ALLOC_BUILD(orders, num_orders, allocate_direct, 2)
  ALLOC_BUILD(nation, num_nation, allocate_direct, 3)
  ALLOC_BUILD(supplier, num_supplier, allocate_direct, 1)
  #undef ALLOC_BUILD

  // Pipeline
  const int GPU_BATCH = []() -> int { if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; } return 4194304; }();
  const int N_STREAMS = []() -> int { if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; } return 4; }();
  cudaStream_t *streams = new cudaStream_t[N_STREAMS];
  for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
  int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;

  // OPT(c): CPU build of partsupp direct arrays (overlapped with GPU builds on separate thread)
  int4 *h_ps_sk4 = nullptr;
  int  *h_ps_sc_flat = nullptr;
  int4 *d_ps_sk4 = nullptr;
  int  *d_ps_sc_flat = nullptr;

  // OPT(d): start partsupp sort on a background CPU thread
  std::atomic<bool> partsupp_ready{false};
  std::thread ps_sort_thread([&]() {
    build_partsupp_direct(h_ps_pk, h_ps_sk, h_ps_sc, num_partsupp, num_part, &h_ps_sk4, &h_ps_sc_flat);
    partsupp_ready.store(true, std::memory_order_release);
  });

  // OPT(d): Build orders, nation, supplier, part concurrently using separate streams.
  // All HTs are independent of each other (no dependency chain except supplier needs nation HT).
  // Build order: [stream0] part, [stream1] orders, [stream2] nation, then supplier.
  cudaStream_t s0, s1, s2;
  CUDA_CHECK(cudaStreamCreate(&s0));
  CUDA_CHECK(cudaStreamCreate(&s1));
  CUDA_CHECK(cudaStreamCreate(&s2));

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
  cudaEventRecord(t0);

  // Stream 0: part build -- morsel-batched on s0
  cpu_prefilter_part(h_p_pk, h_p_name, num_part, h_pf_part, h_pts_part);
  {
    int _pnb = (num_part + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _pnb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_part - _bo);
      tpch_q9_kernel_part<BLOCK_THREADS, ITEMS_PER_THREAD><<<(_bc + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, s0>>>(
        num_part, _bo, d_p_pk, d_p_name, ht_part.d_entries, ht_part.ht_size, gpu_bf_part.d_bits, gpu_bf_part.size_bits, 3, d_pf_part, d_pts_part);
    }
  }

  // Stream 1: orders build -- morsel-batched on s1 (independent of part)
  cpu_prefilter_orders(h_o_ok, h_o_od, num_orders, h_pf_orders, h_pts_orders);
  {
    int _onb = (num_orders + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _onb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_orders - _bo);
      tpch_q9_kernel_orders<BLOCK_THREADS, ITEMS_PER_THREAD><<<(_bc + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, s1>>>(
        num_orders, _bo, d_o_ok, d_o_od, ht_orders.d_entries, ht_orders.ht_size, gpu_bf_orders.d_bits, gpu_bf_orders.size_bits, 3, d_pf_orders, d_pts_orders);
    }
  }

  // Stream 2: nation build -- single launch (25 rows, trivially small)
  cpu_prefilter_nation(h_n_nk, h_n_name, num_nation, h_pf_nation, h_pts_nation);
  tpch_q9_kernel_nation<BLOCK_THREADS, ITEMS_PER_THREAD><<<(num_nation + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, s2>>>(
    num_nation, 0, d_n_nk, d_n_name, ht_nation.d_entries, ht_nation.ht_size, gpu_bf_nation.d_bits, gpu_bf_nation.size_bits, 3, d_pf_nation, d_pts_nation);

  // Supplier needs nation HT, so sync stream2 first, then morsel-batch supplier on s2
  CUDA_CHECK(cudaStreamSynchronize(s2));
  if (bf_storage[3].bits) gpu_bf_nation.copy_to_host(bf_storage[3].bits, ((size_t)(bf_storage[3].size_bits + 63) / 64) * sizeof(uint64_t));
  cpu_prefilter_supplier(h_s_sk, h_s_nk, num_supplier, h_pf_supplier, h_pts_supplier, (bf_storage[3].bits ? &bf_storage[3] : nullptr));
  {
    int _snb = (num_supplier + GPU_BATCH - 1) / GPU_BATCH;
    for (int _b = 0; _b < _snb; ++_b) {
      int _bo = _b * GPU_BATCH, _bc = std::min(GPU_BATCH, num_supplier - _bo);
      tpch_q9_kernel_supplier<BLOCK_THREADS, ITEMS_PER_THREAD><<<(_bc + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, s2>>>(
        num_supplier, _bo, d_s_sk, d_s_nk, ht_supplier.d_entries, ht_supplier.ht_size, gpu_bf_supplier.d_bits, gpu_bf_supplier.size_bits, 3,
        d_pf_supplier, d_pts_supplier, ht_nation.d_entries, ht_nation.ht_size);
    }
  }

  // Sync all build streams
  CUDA_CHECK(cudaStreamSynchronize(s0));
  CUDA_CHECK(cudaStreamSynchronize(s1));
  CUDA_CHECK(cudaStreamSynchronize(s2));
  CUDA_CHECK(cudaGetLastError());

  // Copy BFs to host for CPU prefilter
  if (bf_storage[0].bits) gpu_bf_part.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));
  if (bf_storage[2].bits) gpu_bf_orders.copy_to_host(bf_storage[2].bits, ((size_t)(bf_storage[2].size_bits + 63) / 64) * sizeof(uint64_t));
  if (bf_storage[1].bits) gpu_bf_supplier.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));

  // Wait for partsupp CPU sort to finish, then upload to GPU
  ps_sort_thread.join();
  printf("partsupp direct arrays built: %d parts x 4 slots\n", num_part);
  {
    cudaError_t err = cudaMalloc(&d_ps_sk4, (size_t)num_part * sizeof(int4));
    if (err != cudaSuccess) {
      // Fallback: zero-copy mapped memory
      CUDA_CHECK(cudaHostRegister(h_ps_sk4, (size_t)num_part * sizeof(int4), cudaHostRegisterMapped));
      CUDA_CHECK(cudaHostGetDevicePointer(&d_ps_sk4, h_ps_sk4, 0));
    } else {
      CUDA_CHECK(cudaMemcpy(d_ps_sk4, h_ps_sk4, (size_t)num_part * sizeof(int4), cudaMemcpyHostToDevice));
    }
  }
  {
    cudaError_t err = cudaMalloc(&d_ps_sc_flat, (size_t)num_part * 4 * sizeof(int));
    if (err != cudaSuccess) {
      CUDA_CHECK(cudaHostRegister(h_ps_sc_flat, (size_t)num_part * 4 * sizeof(int), cudaHostRegisterMapped));
      CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_ps_sc_flat, h_ps_sc_flat, 0));
    } else {
      CUDA_CHECK(cudaMemcpy(d_ps_sc_flat, h_ps_sc_flat, (size_t)num_part * 4 * sizeof(int), cudaMemcpyHostToDevice));
    }
  }
  printf("partsupp direct arrays on device: sk4=%p sc_flat=%p\n", (void*)d_ps_sk4, (void*)d_ps_sc_flat);

  auto run_kernel = [&]() {
    // OPT(b): CPU prefilter BF array aligns with cpu_predicate_lineitem probe order:
    //   [0] = part BF    -> probes l_partkey (most selective: LIKE '%green%' ~33%)
    //   [1] = supplier BF -> probes l_suppkey
    //   [2] = orders BF  -> probes l_orderkey_enc
    fascal::runtime::BloomFilter *bf_cpu[3] = {
      bf_part_cpu,
      bf_supplier_cpu,
      bf_orders_cpu
    };

    auto prefilter = [&](int bc, int bo, int tt) {
      fascal::cpu_pred::run_prefilter(bc, h_bm, h_ts,
        [&](int off, int cnt) { cpu_predicate_lineitem(h_data, off, cnt, h_bm, bf_cpu, 3); }, bo, tt);
    };
    auto launch = [&](int nt, int bo, int nb, cudaStream_t st) {
      tpch_q9_kernel_lineitem<BLOCK_THREADS, ITEMS_PER_THREAD><<<nb, BLOCK_THREADS, 0, st>>>(
        nt, bo,
        d_ps_sk4, d_ps_sc_flat,
        d_l_pk, d_l_sk, d_l_ok, d_l_ep, d_l_disc, d_l_qty,
        d_bm, d_ts,
        gpu_bf_part.d_bits, (bloom_off ? 0u : gpu_bf_part.size_bits), ht_part.d_entries, ht_part.ht_size,
        gpu_bf_supplier.d_bits, (bloom_off ? 0u : gpu_bf_supplier.size_bits), ht_supplier.d_entries, ht_supplier.ht_size,
        gpu_bf_orders.d_bits, (bloom_off ? 0u : gpu_bf_orders.size_bits), ht_orders.d_entries, ht_orders.ht_size,
        ht_nation.d_entries, ht_nation.ht_size,
        d_agg);
    };
    { int nb = (num_tuples + GPU_BATCH - 1) / GPU_BATCH;
      for (int b = 0; b < nb; ++b) { int bo = b * GPU_BATCH, bc = std::min(GPU_BATCH, num_tuples - bo);
        prefilter(bc, bo, num_tuples); launch(num_tuples, bo, (bc + TILE_SIZE - 1) / TILE_SIZE, 0); } }
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  };

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
  static const char* nation_names[] = { "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "CHINA", "EGYPT", "ETHIOPIA", "FRANCE", "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN", "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU", "ROMANIA", "RUSSIA", "SAUDI ARABIA", "UNITED KINGDOM", "UNITED STATES", "VIETNAM" };
  double *h_agg = (double*)malloc(num_groups * NUM_AGGREGATES * sizeof(double));
  CUDA_CHECK(cudaMemcpy(h_agg, d_agg, num_groups * NUM_AGGREGATES * sizeof(double), cudaMemcpyDeviceToHost));
  for (size_t k = 0; k < num_groups; k++) {
    if (h_agg[k * NUM_AGGREGATES + 0] != 0) {
      int key0 = (int)(k % 25);
      int key1 = (int)(k / 25);
      printf("ROW: %s|%d|%.4f\n", (key0 >= 0 && key0 < 25) ? nation_names[key0] : "UNKNOWN", key1 + 1992, h_agg[k * NUM_AGGREGATES + 0]);
    }
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
  odzc->unregister_column(mc_l_pk); odzc->unregister_column(mc_l_sk); odzc->unregister_column(mc_l_ok);
  odzc->unregister_column(mc_l_ep); odzc->unregister_column(mc_l_disc); odzc->unregister_column(mc_l_qty);
  odzc->unregister_column(mc_p_pk); odzc->unregister_column(mc_p_name);
  odzc->unregister_column(mc_o_ok); odzc->unregister_column(mc_o_od);
  odzc->unregister_column(mc_n_nk); odzc->unregister_column(mc_n_name);
  odzc->unregister_column(mc_s_sk); odzc->unregister_column(mc_s_nk);
  fascal_free_seed_bitmap(h_bm, h_ts);
  CUDA_CHECK(cudaFree(d_agg));
  fascal_free_column(h_l_partkey); fascal_free_column(h_l_suppkey); fascal_free_column(h_l_orderkey_enc);
  fascal_free_column(h_l_extendedprice); fascal_free_column(h_l_discount); fascal_free_column(h_l_quantity);
  fascal_free_column(h_p_pk); fascal_free_column(h_p_name);
  fascal_free_column(h_ps_sk); fascal_free_column(h_ps_sc); fascal_free_column(h_ps_pk);
  fascal_free_column(h_o_ok); fascal_free_column(h_o_od);
  fascal_free_column(h_n_nk); fascal_free_column(h_n_name);
  fascal_free_column(h_s_sk); fascal_free_column(h_s_nk);
  CUDA_CHECK(cudaFreeHost(h_pf_part)); CUDA_CHECK(cudaFreeHost(h_pts_part)); gpu_bf_part.free_filter(); ht_part.free_table(); bf_storage[0].free_filter();
  CUDA_CHECK(cudaFreeHost(h_pf_orders)); CUDA_CHECK(cudaFreeHost(h_pts_orders)); gpu_bf_orders.free_filter(); ht_orders.free_table(); bf_storage[2].free_filter();
  CUDA_CHECK(cudaFreeHost(h_pf_nation)); CUDA_CHECK(cudaFreeHost(h_pts_nation)); gpu_bf_nation.free_filter(); ht_nation.free_table(); bf_storage[3].free_filter();
  CUDA_CHECK(cudaFreeHost(h_pf_supplier)); CUDA_CHECK(cudaFreeHost(h_pts_supplier)); gpu_bf_supplier.free_filter(); ht_supplier.free_table(); bf_storage[1].free_filter();
  if (h_ps_sk4) free(h_ps_sk4);
  if (h_ps_sc_flat) free(h_ps_sc_flat);
  fascal_arena_destroy();
  cudaStreamDestroy(s0); cudaStreamDestroy(s1); cudaStreamDestroy(s2);
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  delete[] streams;
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(t_start); cudaEventDestroy(t_load);   cudaEventDestroy(t_query); cudaEventDestroy(t_result);
  return 0;
}
