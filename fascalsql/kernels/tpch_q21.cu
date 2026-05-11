// TPC-H Q21 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Source: fascalsql/kernels/tpch_q21.cu
//
// SF=100 FIX (Stage A+B): Replace l2/l3 GpuHashTable (zero-copy, 32 GB each) with
// compact per-orderkey uint32_t state arrays (600 MB each, device memory).
//
// Root cause of wrong results: gpu_ht_insert_exists_ne spin-waits on ht[].val in
// zero-copy host memory. The GPU's L2 cache may serve stale -1 values, causing the
// spin to never terminate for some warps, leaving HT entries uncommitted. The EXISTS
// probe then misses those entries and spuriously returns false, killing valid l1 rows
// and producing catastrophically wrong (too-low) counts.
//
// Fix: state[orderkey_enc] is a single uint32_t per order:
//   state == 0                  → no rows seen
//   state > 0, bit31 == 0       → exactly one distinct suppkey = state value
//   bit31 == 1                  → two or more distinct suppkeys
// Build uses atomicCAS + atomicOr (no spin-wait). Probe is a single device-memory read.
// Memory: 2 x 150M x 4B = 1.2 GB (vs 2 x 32 GB zero-copy HTs).

#define CUB_STDERR

// FASCAL_TILE_SIZE must be defined BEFORE including fascal_generated_common.cuh
// and fascal_query_runtime.cuh so that the runtime uses the correct tile size.
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
#include <chrono>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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

// --- Ablation flags (0=GPU, 1=CPU) ---
__device__ __constant__ unsigned char d_pred_flags[MAX_PREDICATES];
static unsigned char h_pred_flags[MAX_PREDICATES] = {0};

static inline void normalize_or_group_pred_flags(unsigned char *pred_flags) {
  (void)pred_flags;
}

// GPU-side Bitset pointers for string predicates (LIKE/NOT_LIKE)
// Pre-filled by regex_bitset_kernel on unique dictionary entries.
// ================= STAGE 1: tpch_q21 =================

// -----------------------------------------------------------------------------
// mmap-based column loader: avoids cudaMallocManaged for huge columns.
// Returns a host pointer (mmap'd read-only). Caller must munmap.
// Falls back to fread+malloc if mmap fails.
// -----------------------------------------------------------------------------
struct MmapColumn {
    int *ptr;
    size_t bytes;
    bool is_mmap;  // true = munmap, false = free
};

static MmapColumn __attribute__((unused)) load_column_mmap(const char *path, int expected_count) {
    MmapColumn mc = {nullptr, 0, false};
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open %s\n", path);
        return mc;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return mc;
    }
    size_t file_bytes = (size_t)st.st_size;
    int count = (int)(file_bytes / sizeof(int));
    if (expected_count > 0 && count < expected_count) {
        fprintf(stderr, "Warning: %s has %d ints, expected %d\n", path, count, expected_count);
    }

    void *mapped = mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        // Fallback to malloc+fread
        FILE *fp = fopen(path, "rb");
        if (!fp) return mc;
        int *buf = (int*)malloc(file_bytes);
        if (!buf) { fclose(fp); return mc; }
        size_t nread = fread(buf, sizeof(int), count, fp);
        fclose(fp);
        if ((int)nread != count) {
            fprintf(stderr, "Warning: read %zu of %d from %s\n", nread, count, path);
        }
        mc.ptr = buf;
        mc.bytes = file_bytes;
        mc.is_mmap = false;
        return mc;
    }
    // Advise sequential access for readahead
    madvise(mapped, file_bytes, MADV_SEQUENTIAL);
    madvise(mapped, file_bytes, MADV_WILLNEED);
    mc.ptr = (int*)mapped;
    mc.bytes = file_bytes;
    mc.is_mmap = true;
    return mc;
}

static void __attribute__((unused)) free_mmap_column(MmapColumn &mc) {
    if (!mc.ptr) return;
    if (mc.is_mmap) {
        munmap(mc.ptr, mc.bytes);
    } else {
        free(mc.ptr);
    }
    mc.ptr = nullptr;
}

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// All lineitem aliases (l1, l2, l3) share the SAME device pointers.
// -----------------------------------------------------------------------------

// ============================================================================
// Per-orderkey state array helpers (Stage B: compact EXISTS/NOT EXISTS probes)
// ============================================================================
// State encoding: uint32_t per orderkey_enc slot
//   0                     → no rows seen (empty)
//   v > 0, bit31 == 0     → exactly one distinct suppkey = v (1..10^6 < 2^30)
//   bit31 == 1            → two or more distinct suppkeys exist
#define Q21_STATE_MULTI_BIT 0x80000000u

// Update a state slot atomically (no spin-wait):
//   - First insertion: CAS 0 → suppkey
//   - Same suppkey again: no-op (via CAS result check)
//   - Different suppkey: set multi-bit via atomicOr
__device__ __forceinline__ void q21_state_insert(uint32_t *state, uint32_t k, uint32_t supp)
{
    uint32_t old = atomicCAS(&state[k], 0u, supp);
    if (old == 0u) return;                          // claimed slot with first suppkey
    if ((old & ~Q21_STATE_MULTI_BIT) == supp) return; // same suppkey, no change needed
    atomicOr(&state[k], Q21_STATE_MULTI_BIT);       // mark: multiple distinct suppkeys
}

// EXISTS probe: is there any row with this orderkey where suppkey != l1_supp?
__device__ __forceinline__ bool q21_l2_exists(const uint32_t *state, uint32_t k, uint32_t l1_supp)
{
    uint32_t s = state[k];
    if (s == 0u) return false;                      // no rows for this order
    if (s & Q21_STATE_MULTI_BIT) return true;       // multiple suppkeys → at least one != l1
    return s != l1_supp;                            // single suppkey, different from l1
}

// NOT EXISTS probe: there is NO row with this orderkey where suppkey != l1_supp AND late receipt
__device__ __forceinline__ bool q21_l3_not_exists(const uint32_t *state, uint32_t k, uint32_t l1_supp)
{
    uint32_t s = state[k];
    if (s == 0u) return true;                       // no late-receipt rows for this order
    if (s & Q21_STATE_MULTI_BIT) return false;      // multiple late suppkeys → some != l1
    return s == l1_supp;                            // single late suppkey = l1's own supplier
}

// ============================================================================
// Build kernel: update l2 or l3 state array (template for reuse)
// ============================================================================
template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T, bool DO_DATE_FILTER>
__global__ void tpch_q21_kernel_state_build(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_l_orderkey,   // l_orderkey_enc (0-based)
    const int *__restrict__ d_l_suppkey,
    const int *__restrict__ d_l_receiptdate,  // only used when DO_DATE_FILTER=true
    const int *__restrict__ d_l_commitdate,   // only used when DO_DATE_FILTER=true
    uint32_t *__restrict__ d_state,           // output: per-orderkey state array
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
{
    FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

    int loc_l_orderkey[ITEMS_PER_THREAD_T];
    int loc_l_suppkey[ITEMS_PER_THREAD_T];

    BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        (int*)(d_l_orderkey + tile_offset), loc_l_orderkey, selection_flags, num_tile_items);
    BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        (int*)(d_l_suppkey + tile_offset), loc_l_suppkey, selection_flags, num_tile_items);

    if (DO_DATE_FILTER) {
        int loc_receiptdate[ITEMS_PER_THREAD_T];
        int loc_commitdate[ITEMS_PER_THREAD_T];
        BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
            (int*)(d_l_receiptdate + tile_offset), loc_receiptdate, selection_flags, num_tile_items);
        BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
            (int*)(d_l_commitdate + tile_offset), loc_commitdate, selection_flags, num_tile_items);
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
            if (selection_flags[ITEM]) {
                if (!(loc_receiptdate[ITEM] > loc_commitdate[ITEM]))
                    selection_flags[ITEM] = 0;
            }
        }
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
            if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
            if (!selection_flags[ITEM]) continue;
            q21_state_insert(d_state,
                             (uint32_t)(unsigned int)loc_l_orderkey[ITEM],
                             (uint32_t)loc_l_suppkey[ITEM]);
        }
    } else {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
            if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
            if (!selection_flags[ITEM]) continue;
            q21_state_insert(d_state,
                             (uint32_t)(unsigned int)loc_l_orderkey[ITEM],
                             (uint32_t)loc_l_suppkey[ITEM]);
        }
    }
}

// ============================================================================
// Result kernel: l1 scan with state-array EXISTS/NOT EXISTS probes
// ============================================================================
template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q21_kernel_l1(
    int num_tuples,
    int batch_offset,
    int *d_l_suppkey,        // shared lineitem column
    int *d_l_receiptdate,    // shared lineitem column
    int *d_l_commitdate,     // shared lineitem column
    int *d_l_orderkey,       // shared lineitem column (l_orderkey_enc)
    int *d_supplier_s_nationkey,
    int *d_supplier_s_name,
    int *d_supplier_s_name_enc,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const uint64_t *__restrict__ d_bloom_supplier,
    uint32_t bloom_size_bits_supplier,
    HtEntry *ht_supplier,
    uint32_t ht_size_supplier,
    const uint64_t *__restrict__ d_bloom_orders,
    uint32_t bloom_size_bits_orders,
    HtEntry *ht_orders,
    uint32_t ht_size_orders,
    const uint64_t *__restrict__ d_bloom_nation,
    uint32_t bloom_size_bits_nation,
    HtEntry *ht_nation,
    uint32_t ht_size_nation,
    const uint32_t *__restrict__ d_l2_state,   // compact l2 EXISTS state array
    const uint32_t *__restrict__ d_l3_state,   // compact l3 NOT EXISTS state array
    unsigned long long *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  int loc_l_suppkey[ITEMS_PER_THREAD_T];
  int loc_l_receiptdate[ITEMS_PER_THREAD_T];
  int loc_l_commitdate[ITEMS_PER_THREAD_T];
  int loc_l_orderkey[ITEMS_PER_THREAD_T];
  int loc_supplier_s_nationkey[ITEMS_PER_THREAD_T];
  int loc_supplier_s_name[ITEMS_PER_THREAD_T];
  int loc_supplier_s_name_enc[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // Phase 1: Semi-joins (Bloom filter); each FK loaded immediately before its probe
  // ODZC lazy load (join FK, before supplier probe): l_suppkey
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_l_suppkey + tile_offset, loc_l_suppkey,
      selection_flags, num_tile_items);

  BlockBloomProbe<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_l_suppkey, selection_flags, num_tile_items,
      bloom_size_bits_supplier, d_bloom_supplier, 3);

  // ODZC lazy load (join FK, before orders probe): l_orderkey_enc (encoded FK = orders row ID)
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_l_orderkey + tile_offset, loc_l_orderkey,
      selection_flags, num_tile_items);

  BlockBloomProbe<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_l_orderkey, selection_flags, num_tile_items,
      bloom_size_bits_orders, d_bloom_orders, 3);

  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  // Phase 2: All hash table joins
  // Join 0 (inner): l_suppkey -> supplier
  // ODZC lazy load (post-join check source): l1.l_receiptdate
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_l_receiptdate + tile_offset, loc_l_receiptdate,
      selection_flags, num_tile_items);

  // ODZC lazy load (post-join check source): l1.l_commitdate
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_l_commitdate + tile_offset, loc_l_commitdate,
      selection_flags, num_tile_items);

  int loc_supplier_rid[ITEMS_PER_THREAD_T];
  BlockJoinProbePayloadDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_l_suppkey, selection_flags, num_tile_items,
      ht_supplier, ht_size_supplier, 1, loc_supplier_rid);
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ITEM++) {
    if (selection_flags[ITEM]) {
      int rid = loc_supplier_rid[ITEM];
      loc_supplier_s_nationkey[ITEM] = d_supplier_s_nationkey[rid];
      loc_supplier_s_name[ITEM] = d_supplier_s_name[rid];
      loc_supplier_s_name_enc[ITEM] = d_supplier_s_name_enc[rid];
    }
  }

  // Join 1 (inner): l_orderkey_enc -> orders (direct-address, O(1))
  BlockJoinProbeDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_l_orderkey, selection_flags, num_tile_items,
      ht_orders, ht_size_orders, 0);

  // Join 2 (inner): s_nationkey -> nation
  // NOTE: BlockJoinProbeDirectShmem available but not used here because the
  // thread_any_alive divergent block prevents uniform __syncthreads.
  BlockJoinProbeDirect<BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      loc_supplier_s_nationkey, selection_flags, num_tile_items,
      ht_nation, ht_size_nation, 0);

  // Post-join check: l1.l_receiptdate > l1.l_commitdate (applied before EXISTS checks)
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if (selection_flags[ITEM]) {
      if (!(loc_l_receiptdate[ITEM] > loc_l_commitdate[ITEM]))
        selection_flags[ITEM] = 0;
    }
  }

  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  // Join 3 (semijoin): EXISTS(l2.orderkey = l1.orderkey AND l2.suppkey != l1.suppkey)
  // Probe compact state array (device memory, O(1) lookup, no hash collision).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items) {
      if (selection_flags[ITEM]) {
        if (!q21_l2_exists(d_l2_state,
                           (uint32_t)(unsigned int)loc_l_orderkey[ITEM],
                           (uint32_t)loc_l_suppkey[ITEM]))
          selection_flags[ITEM] = 0;
      }
    }
  }

  // Join 4 (antijoin): NOT EXISTS(l3.orderkey = l1.orderkey AND l3.suppkey != l1.suppkey
  //                               AND l3.receiptdate > l3.commitdate)
  // Probe compact state array for late-receipt suppkeys.
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items) {
      if (selection_flags[ITEM]) {
        if (!q21_l3_not_exists(d_l3_state,
                               (uint32_t)(unsigned int)loc_l_orderkey[ITEM],
                               (uint32_t)loc_l_suppkey[ITEM]))
          selection_flags[ITEM] = 0;
      }
    }
  }

  // Row-ID GROUP BY aggregation (high-cardinality dim table row ID as key)
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int gk = loc_supplier_rid[ITEM];
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], 1LL);
      }
  }
  } while(0); // thread_any_alive
}
// CPU predicate evaluation for tpch_q21 (fact table: l1)
// Multi-pass: one pass per column group + BF probe; bitmap skip between passes
// All aliases share the same host pointers.
static void tpch_q21_cpu_predicate_l1(
    int *h_data[],
    int morsel_offset,
    int elem_cnt,
    uint32_t *bitmap,
    int selection_step,
    int ablation_indicator,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0)
{
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = morsel_offset >> 3, nb = (elem_cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, elem_cnt);

  int *l_suppkey = h_data[0], *l_orderkey_enc = h_data[3];

  // Bloom probe on l_suppkey (supplier)
  if (bloom_filters && 0 < num_bloom_filters && bloom_filters[0])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, l_suppkey, morsel_offset, elem_cnt, bloom_filters[0]);
  // Bloom probe on l_orderkey_enc (orders)
  if (bloom_filters && 1 < num_bloom_filters && bloom_filters[1])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, l_orderkey_enc, morsel_offset, elem_cnt, bloom_filters[1]);
  // Bloom probe on l_orderkey_enc (l2 semi-join)
  if (bloom_filters && 3 < num_bloom_filters && bloom_filters[3])
    fascal::cpu_pred::bloom_probe_pass(bm, bs, l_orderkey_enc, morsel_offset, elem_cnt, bloom_filters[3]);
}

// ---- AFP: multi-threaded CPU prefilter for result pipeline 'l1' ----
// CPU fills bitmap, then GPU launches (no doorbell overlap).
// total_tuples: total fact-table size (for bitmap bounds).
// batch_offset + batch_count define the sub-range to prefilter.
// Sequential mode: batch_offset=0, batch_count=total_tuples (default).
// Pipelined mode: called per-batch with specific offset/count.
static void tpch_q21_cpu_prefilter_l1(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
    fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
        [&](int off, int cnt) { tpch_q21_cpu_predicate_l1(h_data, off, cnt, bitmap, 0, 0, bloom_filters, num_bloom_filters); },
        batch_offset, total_tuples);
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q21_kernel_supplier(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_supplier_s_suppkey,
    const int *__restrict__ d_supplier_s_name_enc,
    const int *__restrict__ d_supplier_s_nationkey,
    const int *__restrict__ d_supplier_s_name,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary,
    HtEntry* __restrict__ d_filter_ht_nation, uint32_t ht_size_f_nation)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_supplier_s_suppkey[ITEMS_PER_THREAD_T];
  int loc_supplier_s_name_enc[ITEMS_PER_THREAD_T];
  int loc_supplier_s_nationkey[ITEMS_PER_THREAD_T];
  int loc_supplier_s_name[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_supplier_s_suppkey + tile_offset), loc_supplier_s_suppkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_supplier_s_name_enc + tile_offset), loc_supplier_s_name_enc, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_supplier_s_nationkey + tile_offset), loc_supplier_s_nationkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_supplier_s_name + tile_offset), loc_supplier_s_name, selection_flags, num_tile_items);

  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    if (!gpu_ht_probe_direct(loc_supplier_s_nationkey[ITEM], d_filter_ht_nation, ht_size_f_nation, 0, nullptr)) continue;
    int payload = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one_direct(loc_supplier_s_suppkey[ITEM], payload, 1, 1, d_ht, ht_size);
    gpu_bloom_set_one(loc_supplier_s_suppkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'supplier' ----
static void tpch_q21_cpu_prefilter_supplier(
    int *h_supplier_s_suppkey,
    int *h_supplier_s_name_enc,
    int *h_supplier_s_nationkey,
    int *h_supplier_s_name,
    int num_tuples,
    uint32_t *bitmap,
    uint8_t *tile_summary,
    fascal::runtime::BloomFilter *upstream_bf_nation) {
    fascal_cpu_prefilter_run(num_tuples, bitmap, tile_summary,
        [&](int offset, int cnt) {
            // 0. Zero bitmap words for this morsel.
            uint32_t word_start = (uint32_t)offset >> 5;
            uint32_t word_end   = ((uint32_t)(offset + cnt) + 31) >> 5;
            memset(&bitmap[word_start], 0, (word_end - word_start) * sizeof(uint32_t));
            // 1. Fill packed bitmap with scalar predicate result for this morsel.
            for (int i = 0; i < cnt; ++i) {
                int idx = offset + i;
                if (1) {
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
                }
            }
            // 2. Apply upstream BF probes (AND-mask with FK column probes).
                if (upstream_bf_nation && upstream_bf_nation->bits) {
                    upstream_bf_nation->probe_and_mask_packed(h_supplier_s_nationkey + offset, bitmap, offset, cnt);
                }
        });
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q21_kernel_orders(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_orders_o_orderkey,
    const int *__restrict__ d_orders_o_orderstatus_enc,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_orders_o_orderkey[ITEMS_PER_THREAD_T];
  int loc_orders_o_orderstatus_enc[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_orders_o_orderkey + tile_offset), loc_orders_o_orderkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_orders_o_orderstatus_enc + tile_offset), loc_orders_o_orderstatus_enc, selection_flags, num_tile_items);

  BlockPredAndEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_orders_o_orderstatus_enc, 0, selection_flags, num_tile_items);

  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int row_idx = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one_direct(row_idx, row_idx, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(row_idx, 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'orders' ----
static void tpch_q21_cpu_prefilter_orders(
    int *h_orders_o_orderkey,
    int *h_orders_o_orderstatus_enc,
    int num_tuples,
    uint32_t *bitmap,
    uint8_t *tile_summary) {
    fascal_cpu_prefilter_run(num_tuples, bitmap, tile_summary,
        [&](int offset, int cnt) {
            // 0. Zero bitmap words for this morsel.
            uint32_t word_start = (uint32_t)offset >> 5;
            uint32_t word_end   = ((uint32_t)(offset + cnt) + 31) >> 5;
            memset(&bitmap[word_start], 0, (word_end - word_start) * sizeof(uint32_t));
            // 1. Fill packed bitmap with scalar predicate result for this morsel.
            for (int i = 0; i < cnt; ++i) {
                int idx = offset + i;
                if ((h_orders_o_orderstatus_enc[idx] == 0)) {
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
                }
            }
            // 2. Apply upstream BF probes (AND-mask with FK column probes).

        });
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q21_kernel_nation(
    int num_tuples,
    int batch_offset,
    const int *__restrict__ d_nation_n_nationkey,
    const int *__restrict__ d_nation_n_name,
    HtEntry* d_ht, uint32_t ht_size,
    uint64_t* __restrict__ d_bf_bits, uint32_t bf_size_bits, uint8_t num_hashes,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc_nation_n_nationkey[ITEMS_PER_THREAD_T];
  int loc_nation_n_name[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_nation_n_nationkey + tile_offset), loc_nation_n_nationkey, selection_flags, num_tile_items);

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d_nation_n_name + tile_offset), loc_nation_n_name, selection_flags, num_tile_items);

  BlockPredAndEQ<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(loc_nation_n_name, 21, selection_flags, num_tile_items);

  // Insert selected rows into HT and Bloom filter (after all probes).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int payload = tile_offset + threadIdx.x + ITEM * BLOCK_THREADS_T;
    gpu_ht_insert_one_direct(loc_nation_n_nationkey[ITEM], payload, 1, 0, d_ht, ht_size);
    gpu_bloom_set_one(loc_nation_n_nationkey[ITEM], 1, d_bf_bits, bf_size_bits, num_hashes);
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline 'nation' ----
static void tpch_q21_cpu_prefilter_nation(
    int *h_nation_n_nationkey,
    int *h_nation_n_name,
    int num_tuples,
    uint32_t *bitmap,
    uint8_t *tile_summary) {
    fascal_cpu_prefilter_run(num_tuples, bitmap, tile_summary,
        [&](int offset, int cnt) {
            // 0. Zero bitmap words for this morsel.
            uint32_t word_start = (uint32_t)offset >> 5;
            uint32_t word_end   = ((uint32_t)(offset + cnt) + 31) >> 5;
            memset(&bitmap[word_start], 0, (word_end - word_start) * sizeof(uint32_t));
            // 1. Fill packed bitmap with scalar predicate result for this morsel.
            for (int i = 0; i < cnt; ++i) {
                int idx = offset + i;
                if ((h_nation_n_name[idx] == 21)) {
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
                }
            }
            // 2. Apply upstream BF probes (AND-mask with FK column probes).

        });
}

// l2 build: instantiate the shared template (no date filter)
// Note: d_l2_state is allocated in main and passed at launch time.
// This function wrapper exists for readability; the actual kernel is
// tpch_q21_kernel_state_build<BT, IPT, false>.

// CPU prefilter for l2/l3 build: passes all rows (no CPU-side predicates for l2;
// l3 date filter is applied on GPU inside the templated kernel).
static void tpch_q21_cpu_prefilter_l23(
    int num_tuples,
    uint32_t *bitmap,
    uint8_t *tile_summary) {
    // Set all bits: the GPU kernel applies any row-level filters.
    size_t word_count = ((size_t)num_tuples + 31) / 32;
    memset(bitmap, 0xFF, word_count * sizeof(uint32_t));
    // Mark all tiles as active.
    size_t tile_count = ((size_t)num_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
    memset(tile_summary, 1, tile_count);
}

// l3 build: instantiate the shared template (with date filter).
// See tpch_q21_kernel_state_build<BT, IPT, true>.


int main(int argc, char** argv) {

    const char *data_dir = getenv("FASCALSQL_DATA_DIR");
    if (!data_dir) {
        data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
    }
    if (argc >= 3) {
        data_dir = argv[2];
    }
    if (!data_dir) {
        fprintf(stderr, "FASCALSQL_DATA_DIR not set.\n");
        return 1;
    }

    auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));

    // ---------------- STAGE 1 EXECUTION: tpch_q21 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body -- FaScalSQL Query: tpch_q21
  // ============================================================
      printf("=== FaScalSQL: tpch_q21 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/l_suppkey.bin", data_dir);
          int _auto_count = 0;
          int *_tmp = load_binary_column_auto(_auto_path, &_auto_count);
          if (_tmp) fascal_free_column(_tmp);
          num_tuples = _auto_count;
          if (num_tuples <= 0) {
              fprintf(stderr, "Cannot auto-detect num_tuples from %s\n", _auto_path);
              return 1;
          }
      }
      printf("num_tuples=%d data_dir=%s\n", num_tuples, data_dir);

      // === Timing Instrumentation ===
      cudaEvent_t t_start, t_load, t_query, t_result;
      CUDA_CHECK(cudaEventCreate(&t_start));
      CUDA_CHECK(cudaEventCreate(&t_load));
      CUDA_CHECK(cudaEventCreate(&t_query));
      CUDA_CHECK(cudaEventCreate(&t_result));
      CUDA_CHECK(cudaEventRecord(t_start));

      // === Data Loading ===
      // SF=100 FIX: Load lineitem columns ONCE (shared across l1/l2/l3).
      // At SF=100 this saves ~24GB of cudaMallocManaged allocations.
      char path[512];
      auto t_load_start = std::chrono::high_resolution_clock::now();

      // ----- Load SHARED lineitem columns (one copy for l1/l2/l3) -----
      snprintf(path, sizeof(path), "%s/l_suppkey.bin", data_dir);
      int *h_l_suppkey = load_binary_column(path, num_tuples);
      if (!h_l_suppkey) { fprintf(stderr, "Failed to load l_suppkey\n"); return 1; }

      snprintf(path, sizeof(path), "%s/l_receiptdate.bin", data_dir);
      int *h_l_receiptdate = load_binary_column(path, num_tuples);
      if (!h_l_receiptdate) { fprintf(stderr, "Failed to load l_receiptdate\n"); return 1; }

      snprintf(path, sizeof(path), "%s/l_commitdate.bin", data_dir);
      int *h_l_commitdate = load_binary_column(path, num_tuples);
      if (!h_l_commitdate) { fprintf(stderr, "Failed to load l_commitdate\n"); return 1; }

      snprintf(path, sizeof(path), "%s/l_orderkey_enc.bin", data_dir);
      int *h_l_orderkey_enc = load_binary_column(path, num_tuples);
      if (!h_l_orderkey_enc) { fprintf(stderr, "Failed to load l_orderkey_enc\n"); return 1; }

      printf("Loaded lineitem: %d tuples (shared across l1/l2/l3 -- 4 columns x 1 copy)\n", num_tuples);

      auto t_lineitem_done = std::chrono::high_resolution_clock::now();
      double lineitem_ms = std::chrono::duration<double, std::milli>(t_lineitem_done - t_load_start).count();
      printf("Lineitem load time: %.1f ms\n", lineitem_ms);

      // ----- Load dimension: orders -----
      int num_orders_tuples = 0;

      snprintf(path, sizeof(path), "%s/o_orderkey.bin", data_dir);
      int *h_orders_o_orderkey = load_binary_column_auto(path, &num_orders_tuples);

      if (!h_orders_o_orderkey || num_orders_tuples == 0) {
          fprintf(stderr, "Failed to load table column o_orderkey\n"); return 1;
      }

      snprintf(path, sizeof(path), "%s/o_orderstatus_enc.bin", data_dir);
      int *h_orders_o_orderstatus_enc = load_binary_column_auto(path, &num_orders_tuples);

      if (!h_orders_o_orderstatus_enc || num_orders_tuples == 0) {
          fprintf(stderr, "Failed to load table column o_orderstatus_enc\n"); return 1;
      }
      printf("Loaded orders: %d tuples\n", num_orders_tuples);

      // ----- Load dimension: nation -----
      int num_nation_tuples = 0;

      snprintf(path, sizeof(path), "%s/n_nationkey.bin", data_dir);
      int *h_nation_n_nationkey = load_binary_column_auto(path, &num_nation_tuples);

      if (!h_nation_n_nationkey || num_nation_tuples == 0) {
          fprintf(stderr, "Failed to load table column n_nationkey\n"); return 1;
      }

      snprintf(path, sizeof(path), "%s/n_name.bin", data_dir);
      int *h_nation_n_name = load_binary_column_auto(path, &num_nation_tuples);

      if (!h_nation_n_name || num_nation_tuples == 0) {
          fprintf(stderr, "Failed to load table column n_name\n"); return 1;
      }
      printf("Loaded nation: %d tuples\n", num_nation_tuples);

      // NOTE: l2 and l3 share the same physical lineitem columns.
      // No separate load needed. State arrays keyed by l_orderkey_enc (0..num_orders_tuples-1).
      printf("l2/l3 alias lineitem: %d tuples (shared, zero additional memory)\n", num_tuples);

      // ----- Load dimension: supplier -----
      int num_supplier_tuples = 0;

      snprintf(path, sizeof(path), "%s/s_suppkey.bin", data_dir);
      int *h_supplier_s_suppkey = load_binary_column_auto(path, &num_supplier_tuples);

      if (!h_supplier_s_suppkey || num_supplier_tuples == 0) {
          fprintf(stderr, "Failed to load table column s_suppkey\n"); return 1;
      }

      snprintf(path, sizeof(path), "%s/s_name_enc.bin", data_dir);
      int *h_supplier_s_name_enc = load_binary_column_auto(path, &num_supplier_tuples);

      if (!h_supplier_s_name_enc || num_supplier_tuples == 0) {
          fprintf(stderr, "Failed to load table column s_name_enc\n"); return 1;
      }

      snprintf(path, sizeof(path), "%s/s_nationkey.bin", data_dir);
      int *h_supplier_s_nationkey = load_binary_column_auto(path, &num_supplier_tuples);

      if (!h_supplier_s_nationkey || num_supplier_tuples == 0) {
          fprintf(stderr, "Failed to load table column s_nationkey\n"); return 1;
      }

      snprintf(path, sizeof(path), "%s/s_name.bin", data_dir);
      int *h_supplier_s_name = load_binary_column_auto(path, &num_supplier_tuples);

      if (!h_supplier_s_name || num_supplier_tuples == 0) {
          fprintf(stderr, "Failed to load table column s_name\n"); return 1;
      }
      printf("Loaded supplier: %d tuples\n", num_supplier_tuples);

      CUDA_CHECK(cudaEventRecord(t_load));

      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      cqo_in.predicate_selectivities.push_back(0.100000);
      cqo_in.predicate_selectivities.push_back(0.100000);
      cqo_in.join_bf_selectivities.push_back(0.05);
      cqo_in.join_bf_selectivities.push_back(0.05);
      cqo_in.join_bf_selectivities.push_back(0.05);
      cqo_in.join_bf_selectivities.push_back(0.05);
      cqo_in.join_bf_selectivities.push_back(0.05);
      cqo_in.num_gpu_only_ops = 0;
      auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
  size_t llc_bytes = cqo_in.llc_size_bytes > 0 ? cqo_in.llc_size_bytes : fascal_get_llc_size();
      fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);
      normalize_or_group_pred_flags(h_pred_flags);
      int in_memory_repeats = 1;
      if (const char *repeat_env = getenv("FASCAL_IN_MEMORY_REPEATS")) {
          in_memory_repeats = atoi(repeat_env);
          if (in_memory_repeats < 1) in_memory_repeats = 1;
      }

      int bloom_off = (getenv("FASCAL_BLOOM_OFF") != nullptr) || cqo_out.bloom_off;

      // === AFP Manager & Predicate Filters ===
      // h_data[] uses shared lineitem pointers: [l_suppkey, l_receiptdate, l_commitdate, l_orderkey]
      int *h_data[] = {h_l_suppkey, h_l_receiptdate, h_l_commitdate, h_l_orderkey_enc};
      fascal::runtime::AFPManager afp_manager(0);
      (void)afp_manager;

      // === Bloom Filters (Build Pipelines) ===
      // Only supplier (bf[0]) and orders (bf[1]) BFs remain.
      // l2/l3 BFs removed: replaced by compact state arrays.
      fascal::runtime::BloomFilter bf_storage[3];  // [0]=orders, [1]=nation, [2]=supplier
      fascal::runtime::BloomFilter *bf_array[5];

      bf_array[0] = (!bloom_off && (int)0 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[0]) ? &bf_storage[2] : nullptr;
      bf_array[1] = (!bloom_off && (int)1 < (int)cqo_out.join_bf_active.size() && cqo_out.join_bf_active[1]) ? &bf_storage[0] : nullptr;
      bf_array[2] = nullptr;
      bf_array[3] = nullptr;  // l2 EXISTS: no BF (state array used instead)
      bf_array[4] = nullptr;  // l3 NOT EXISTS: no BF

      // === ODZC & Device Buffers ===
      // SF=100 FIX: Register lineitem columns ONCE with ODZC, share across l1/l2/l3
      auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(
          new fascal::runtime::ODZCManager(0));

      // === Seed Bitmap + Tile Summary (CPU-first, no doorbell) ===
      uint32_t *h_seed_bitmap_storage = nullptr, *d_seed_bitmap = nullptr;
      uint8_t *h_tile_summary = nullptr, *d_tile_summary = nullptr;
      fascal_alloc_seed_bitmap(num_tuples, &h_seed_bitmap_storage, &d_seed_bitmap, &h_tile_summary, &d_tile_summary);

      // Shared lineitem ODZC mappings (one registration per physical column)
      fascal::runtime::ODZCManager::MappedColumn mc_l_suppkey;
      int *d_l_suppkey = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_suppkey, num_tuples, "l_suppkey", &mc_l_suppkey, &d_l_suppkey);

      fascal::runtime::ODZCManager::MappedColumn mc_l_receiptdate;
      int *d_l_receiptdate = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_receiptdate, num_tuples, "l_receiptdate", &mc_l_receiptdate, &d_l_receiptdate);

      fascal::runtime::ODZCManager::MappedColumn mc_l_commitdate;
      int *d_l_commitdate = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_commitdate, num_tuples, "l_commitdate", &mc_l_commitdate, &d_l_commitdate);

      fascal::runtime::ODZCManager::MappedColumn mc_l_orderkey;
      int *d_l_orderkey = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_l_orderkey_enc, num_tuples, "l_orderkey_enc", &mc_l_orderkey, &d_l_orderkey);

      // Orders ODZC
      fascal::runtime::ODZCManager::MappedColumn mc_orders_o_orderkey;
      mc_orders_o_orderkey = odzc_mgr->register_column(h_orders_o_orderkey, num_orders_tuples * sizeof(int));
      int *d_orders_o_orderkey = static_cast<int*>(mc_orders_o_orderkey.device_ptr);
      if (!d_orders_o_orderkey) {
          fprintf(stderr, "ODZC failed for orders.o_orderkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_orders_o_orderkey, num_orders_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_orders_o_orderkey, h_orders_o_orderkey, num_orders_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_orders_o_orderstatus_enc;
      mc_orders_o_orderstatus_enc = odzc_mgr->register_column(h_orders_o_orderstatus_enc, num_orders_tuples * sizeof(int));
      int *d_orders_o_orderstatus_enc = static_cast<int*>(mc_orders_o_orderstatus_enc.device_ptr);
      if (!d_orders_o_orderstatus_enc) {
          fprintf(stderr, "ODZC failed for orders.o_orderstatus_enc, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_orders_o_orderstatus_enc, num_orders_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_orders_o_orderstatus_enc, h_orders_o_orderstatus_enc, num_orders_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      // Nation ODZC
      fascal::runtime::ODZCManager::MappedColumn mc_nation_n_nationkey;
      mc_nation_n_nationkey = odzc_mgr->register_column(h_nation_n_nationkey, num_nation_tuples * sizeof(int));
      int *d_nation_n_nationkey = static_cast<int*>(mc_nation_n_nationkey.device_ptr);
      if (!d_nation_n_nationkey) {
          fprintf(stderr, "ODZC failed for nation.n_nationkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_nation_n_nationkey, num_nation_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_nation_n_nationkey, h_nation_n_nationkey, num_nation_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_nation_n_name;
      mc_nation_n_name = odzc_mgr->register_column(h_nation_n_name, num_nation_tuples * sizeof(int));
      int *d_nation_n_name = static_cast<int*>(mc_nation_n_name.device_ptr);
      if (!d_nation_n_name) {
          fprintf(stderr, "ODZC failed for nation.n_name, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_nation_n_name, num_nation_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_nation_n_name, h_nation_n_name, num_nation_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      // l2 and l3 share d_l_orderkey and d_l_suppkey -- no separate ODZC needed

      // Supplier ODZC
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_suppkey;
      mc_supplier_s_suppkey = odzc_mgr->register_column(h_supplier_s_suppkey, num_supplier_tuples * sizeof(int));
      int *d_supplier_s_suppkey = static_cast<int*>(mc_supplier_s_suppkey.device_ptr);
      if (!d_supplier_s_suppkey) {
          fprintf(stderr, "ODZC failed for supplier.s_suppkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_supplier_s_suppkey, num_supplier_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_supplier_s_suppkey, h_supplier_s_suppkey, num_supplier_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_name_enc;
      mc_supplier_s_name_enc = odzc_mgr->register_column(h_supplier_s_name_enc, num_supplier_tuples * sizeof(int));
      int *d_supplier_s_name_enc = static_cast<int*>(mc_supplier_s_name_enc.device_ptr);
      if (!d_supplier_s_name_enc) {
          fprintf(stderr, "ODZC failed for supplier.s_name_enc, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_supplier_s_name_enc, num_supplier_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_supplier_s_name_enc, h_supplier_s_name_enc, num_supplier_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_nationkey;
      mc_supplier_s_nationkey = odzc_mgr->register_column(h_supplier_s_nationkey, num_supplier_tuples * sizeof(int));
      int *d_supplier_s_nationkey = static_cast<int*>(mc_supplier_s_nationkey.device_ptr);
      if (!d_supplier_s_nationkey) {
          fprintf(stderr, "ODZC failed for supplier.s_nationkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_supplier_s_nationkey, num_supplier_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_supplier_s_nationkey, h_supplier_s_nationkey, num_supplier_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
      fascal::runtime::ODZCManager::MappedColumn mc_supplier_s_name;
      mc_supplier_s_name = odzc_mgr->register_column(h_supplier_s_name, num_supplier_tuples * sizeof(int));
      int *d_supplier_s_name = static_cast<int*>(mc_supplier_s_name.device_ptr);
      if (!d_supplier_s_name) {
          fprintf(stderr, "ODZC failed for supplier.s_name, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d_supplier_s_name, num_supplier_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_supplier_s_name, h_supplier_s_name, num_supplier_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }

      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));



      // === Aggregation Table ===
      size_t num_groups = (size_t)num_supplier_tuples;
      unsigned long long *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, num_groups * NUM_AGGREGATES * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(unsigned long long)));

      // === Hash Table Build ===

      GpuHashTable ht_orders;
      ht_orders = GpuHashTable::allocate_direct(num_orders_tuples);
      GpuBloomFilter gpu_bf_orders = GpuBloomFilter::allocate(num_orders_tuples, 3);
      bf_storage[0] = fascal::runtime::BloomFilter::allocate(num_orders_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_orders.d_entries, 0xff, ht_orders.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_orders.d_bits, 0, ((size_t)gpu_bf_orders.size_bits + 63) / 64 * sizeof(uint64_t)));

      // AFP all-pipeline: pinned mapped memory for orders packed prefilter + tile summary (no doorbell).
      size_t orders_bitmap_words = ((size_t)num_orders_tuples + 31) / 32;
      size_t orders_num_tiles = ((size_t)num_orders_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_orders_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_orders_prefilter_pinned, orders_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_orders_prefilter_pinned, 0, orders_bitmap_words * sizeof(uint32_t));
      uint32_t *d_orders_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_orders_prefilter_mapped, h_orders_prefilter_pinned, 0));
      uint8_t *h_orders_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_orders_tile_summary, orders_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_orders_tile_summary, 0, orders_num_tiles);
      uint8_t *d_orders_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_orders_tile_summary, h_orders_tile_summary, 0));

      GpuHashTable ht_nation;
      ht_nation = GpuHashTable::allocate_direct(num_nation_tuples);
      GpuBloomFilter gpu_bf_nation = GpuBloomFilter::allocate(num_nation_tuples, 3);
      bf_storage[1] = fascal::runtime::BloomFilter::allocate(num_nation_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_nation.d_entries, 0xff, ht_nation.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_nation.d_bits, 0, ((size_t)gpu_bf_nation.size_bits + 63) / 64 * sizeof(uint64_t)));

      // AFP all-pipeline: pinned mapped memory for nation packed prefilter + tile summary (no doorbell).
      size_t nation_bitmap_words = ((size_t)num_nation_tuples + 31) / 32;
      size_t nation_num_tiles = ((size_t)num_nation_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_nation_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_nation_prefilter_pinned, nation_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_nation_prefilter_pinned, 0, nation_bitmap_words * sizeof(uint32_t));
      uint32_t *d_nation_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_nation_prefilter_mapped, h_nation_prefilter_pinned, 0));
      uint8_t *h_nation_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_nation_tile_summary, nation_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_nation_tile_summary, 0, nation_num_tiles);
      uint8_t *d_nation_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_nation_tile_summary, h_nation_tile_summary, 0));

      GpuHashTable ht_supplier;
      ht_supplier = GpuHashTable::allocate_direct(num_supplier_tuples);
      GpuBloomFilter gpu_bf_supplier = GpuBloomFilter::allocate(num_supplier_tuples, 3);
      bf_storage[2] = fascal::runtime::BloomFilter::allocate(num_supplier_tuples, 3);
      CUDA_CHECK(cudaMemset(ht_supplier.d_entries, 0xff, ht_supplier.ht_size * sizeof(HtEntry)));
      CUDA_CHECK(cudaMemset(gpu_bf_supplier.d_bits, 0, ((size_t)gpu_bf_supplier.size_bits + 63) / 64 * sizeof(uint64_t)));

      // AFP all-pipeline: pinned mapped memory for supplier packed prefilter + tile summary (no doorbell).
      size_t supplier_bitmap_words = ((size_t)num_supplier_tuples + 31) / 32;
      size_t supplier_num_tiles = ((size_t)num_supplier_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_supplier_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_supplier_prefilter_pinned, supplier_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h_supplier_prefilter_pinned, 0, supplier_bitmap_words * sizeof(uint32_t));
      uint32_t *d_supplier_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_supplier_prefilter_mapped, h_supplier_prefilter_pinned, 0));
      uint8_t *h_supplier_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_supplier_tile_summary, supplier_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h_supplier_tile_summary, 0, supplier_num_tiles);
      uint8_t *d_supplier_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_supplier_tile_summary, h_supplier_tile_summary, 0));

      // === Compact per-orderkey state arrays for l2/l3 (Stage B) ===
      // Size: num_orders_tuples entries (one per possible l_orderkey_enc value).
      // Memory: 2 x 150M x 4B = 1.2 GB (vs 2 x 32 GB zero-copy HTs).
      // Pinned prefilter for l2/l3 build kernels (all-ones: GPU applies filters).
      size_t l23_bitmap_words = ((size_t)num_tuples + 31) / 32;
      size_t l23_num_tiles    = ((size_t)num_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h_l23_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_l23_prefilter_pinned, l23_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      uint32_t *d_l23_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_l23_prefilter_mapped, h_l23_prefilter_pinned, 0));
      uint8_t *h_l23_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h_l23_tile_summary, l23_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      uint8_t *d_l23_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d_l23_tile_summary, h_l23_tile_summary, 0));
      // Fill all-ones (all rows pass the CPU prefilter; GPU applies date filter for l3).
      tpch_q21_cpu_prefilter_l23(num_tuples, h_l23_prefilter_pinned, h_l23_tile_summary);

      // Allocate state arrays in device memory.
      uint32_t *d_l2_state = nullptr, *d_l3_state = nullptr;
      {
          size_t state_bytes = (size_t)num_orders_tuples * sizeof(uint32_t);
          CUDA_CHECK(cudaMalloc(&d_l2_state, state_bytes));
          CUDA_CHECK(cudaMemset(d_l2_state, 0, state_bytes));  // 0 = "no rows seen"
          CUDA_CHECK(cudaMalloc(&d_l3_state, state_bytes));
          CUDA_CHECK(cudaMemset(d_l3_state, 0, state_bytes));
          printf("State arrays: 2 x %zu MB = %zu MB (device)\n",
                 state_bytes / (1024*1024), 2 * state_bytes / (1024*1024));
      }


      cudaEvent_t t0, t1;
      CUDA_CHECK(cudaEventCreate(&t0));
      CUDA_CHECK(cudaEventCreate(&t1));

      int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
      printf("Launch: %d blocks x %d threads (CPU-first prefilter)\n", num_blocks, BLOCK_THREADS);
      // --- Pipelined CPU-GPU execution (Section V-E) ---
      const int FASCAL_GPU_BATCH = []() -> int {
          if (const char *e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; }
          return 4194304;  // 4M tuples default
      }();
      const int FASCAL_NUM_STREAMS = []() -> int {
          if (const char *e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; }
          return 4;
      }();
      const bool pipeline_off = (getenv("FASCAL_PIPELINE_OFF") != nullptr);
      cudaStream_t *fascal_streams = new cudaStream_t[FASCAL_NUM_STREAMS];
      for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamCreate(&fascal_streams[_s]));
      if (!pipeline_off) {
          int _n_batches = (num_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
          printf("Pipeline: %d batches x %d tuples, %d streams\n", _n_batches, FASCAL_GPU_BATCH, FASCAL_NUM_STREAMS);
      } else {
          printf("Pipeline: OFF (sequential CPU-then-GPU)\n");
      }

      auto run_result_kernel = [&]() -> int {
          if (pipeline_off) {
              // Sequential fallback: CPU prefilters ALL rows, then single GPU launch
              auto cpu_start = std::chrono::high_resolution_clock::now();
              tpch_q21_cpu_prefilter_l1(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 5);
              auto cpu_end = std::chrono::high_resolution_clock::now();
              double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
              printf("CPU prefilter (l1): %.3f ms\n", cpu_ms);
              tpch_q21_kernel_l1<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_l_suppkey,
                  d_l_receiptdate,
                  d_l_commitdate,
                  d_l_orderkey,
                  d_supplier_s_nationkey,
                  d_supplier_s_name,
                  d_supplier_s_name_enc,
                  d_seed_bitmap,
                  d_tile_summary,
                  gpu_bf_supplier.d_bits, (bloom_off ? 0u : gpu_bf_supplier.size_bits),
                  ht_supplier.d_entries, ht_supplier.ht_size,
                  gpu_bf_orders.d_bits, (bloom_off ? 0u : gpu_bf_orders.size_bits),
                  ht_orders.d_entries, ht_orders.ht_size,
                  gpu_bf_nation.d_bits, (bloom_off ? 0u : gpu_bf_nation.size_bits),
                  ht_nation.d_entries, ht_nation.ht_size,
                  d_l2_state,
                  d_l3_state,
                  d_aggtable);
              CUDA_CHECK(cudaDeviceSynchronize());
              CUDA_CHECK(cudaGetLastError());
              return 0;
          }
          // Pipelined: process GPU_BATCH_SIZE tuples at a time
          int num_batches = (num_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
          for (int _b = 0; _b < num_batches; ++_b) {
              int b_offset = _b * FASCAL_GPU_BATCH;
              int b_count  = std::min(FASCAL_GPU_BATCH, num_tuples - b_offset);
              int b_blocks = (b_count + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _stream = 0;
              // CPU prefilter this batch (fills bitmap[b_offset..b_offset+b_count-1])
              tpch_q21_cpu_prefilter_l1(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, bf_array, 5, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q21_kernel_l1<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_l_suppkey,
                  d_l_receiptdate,
                  d_l_commitdate,
                  d_l_orderkey,
                  d_supplier_s_nationkey,
                  d_supplier_s_name,
                  d_supplier_s_name_enc,
                  d_seed_bitmap,
                  d_tile_summary,
                  gpu_bf_supplier.d_bits, (bloom_off ? 0u : gpu_bf_supplier.size_bits),
                  ht_supplier.d_entries, ht_supplier.ht_size,
                  gpu_bf_orders.d_bits, (bloom_off ? 0u : gpu_bf_orders.size_bits),
                  ht_orders.d_entries, ht_orders.ht_size,
                  gpu_bf_nation.d_bits, (bloom_off ? 0u : gpu_bf_nation.size_bits),
                  ht_nation.d_entries, ht_nation.ht_size,
                  d_l2_state,
                  d_l3_state,
                  d_aggtable);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };

      cudaEventRecord(t0);

      // orders (~1.5M rows at SF=100): morsel-batched build loop
      {
          const int FASCAL_GPU_BATCH_BUILD = []() -> int {
              if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; }
              return 4194304;
          }();
          const int FASCAL_NUM_STREAMS_BUILD = []() -> int {
              if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; }
              return 4;
          }();
          tpch_q21_cpu_prefilter_orders(h_orders_o_orderkey, h_orders_o_orderstatus_enc, num_orders_tuples, h_orders_prefilter_pinned, h_orders_tile_summary);
          int n_b = (num_orders_tuples + FASCAL_GPU_BATCH_BUILD - 1) / FASCAL_GPU_BATCH_BUILD;
          for (int _b = 0; _b < n_b; ++_b) {
              int b_off = _b * FASCAL_GPU_BATCH_BUILD;
              int b_cnt = std::min(FASCAL_GPU_BATCH_BUILD, num_orders_tuples - b_off);
              int b_blk = (b_cnt + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _stream = 0;
              tpch_q21_kernel_orders<BLOCK_THREADS, ITEMS_PER_THREAD>
                  <<<b_blk, BLOCK_THREADS, 0, _stream>>>(
                  num_orders_tuples, b_off, d_orders_o_orderkey, d_orders_o_orderstatus_enc,
                  ht_orders.d_entries, ht_orders.ht_size,
                  gpu_bf_orders.d_bits, gpu_bf_orders.size_bits, 3,
                  d_orders_prefilter_mapped,
                  d_orders_tile_summary);
          }
          for (int _s = 0; _s < FASCAL_NUM_STREAMS_BUILD; ++_s)
              cudaStreamSynchronize(fascal_streams[_s]);
          CUDA_CHECK(cudaGetLastError());
      }
      if (bf_storage[0].bits) gpu_bf_orders.copy_to_host(bf_storage[0].bits, ((size_t)(bf_storage[0].size_bits + 63) / 64) * sizeof(uint64_t));
      // nation (25 rows < 256K): single launch, batch_offset=0
      tpch_q21_cpu_prefilter_nation(h_nation_n_nationkey, h_nation_n_name, num_nation_tuples, h_nation_prefilter_pinned, h_nation_tile_summary);
      tpch_q21_kernel_nation<BLOCK_THREADS, ITEMS_PER_THREAD>
          <<<(num_nation_tuples + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, fascal_streams[0]>>>(
          num_nation_tuples, /*batch_offset=*/0, d_nation_n_nationkey, d_nation_n_name,
          ht_nation.d_entries, ht_nation.ht_size,
          gpu_bf_nation.d_bits, gpu_bf_nation.size_bits, 3,
          d_nation_prefilter_mapped,
          d_nation_tile_summary);
      CUDA_CHECK(cudaStreamSynchronize(fascal_streams[0]));
      CUDA_CHECK(cudaGetLastError());
      if (bf_storage[1].bits) gpu_bf_nation.copy_to_host(bf_storage[1].bits, ((size_t)(bf_storage[1].size_bits + 63) / 64) * sizeof(uint64_t));

      // Build l2 state array: morsel-batched loop over all lineitem rows.
      // Prefilter bitmap is all-ones (already set by tpch_q21_cpu_prefilter_l23).
      {
          const int FASCAL_GPU_BATCH_BUILD = []() -> int {
              if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; }
              return 4194304;
          }();
          const int FASCAL_NUM_STREAMS_BUILD = []() -> int {
              if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; }
              return 4;
          }();
          int n_b = (num_tuples + FASCAL_GPU_BATCH_BUILD - 1) / FASCAL_GPU_BATCH_BUILD;
          for (int _b = 0; _b < n_b; ++_b) {
              int b_off = _b * FASCAL_GPU_BATCH_BUILD;
              int b_cnt = std::min(FASCAL_GPU_BATCH_BUILD, num_tuples - b_off);
              int b_blk = (b_cnt + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _stream = 0;
              tpch_q21_kernel_state_build<BLOCK_THREADS, ITEMS_PER_THREAD, false>
                  <<<b_blk, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples, b_off, d_l_orderkey, d_l_suppkey,
                  nullptr, nullptr,
                  d_l2_state,
                  d_l23_prefilter_mapped, d_l23_tile_summary);
          }
          for (int _s = 0; _s < FASCAL_NUM_STREAMS_BUILD; ++_s)
              cudaStreamSynchronize(fascal_streams[_s]);
          CUDA_CHECK(cudaGetLastError());
      }
      printf("l2 state build done\n");

      // Build l3 state array: morsel-batched loop with l_receiptdate > l_commitdate filter on GPU.
      {
          const int FASCAL_GPU_BATCH_BUILD = []() -> int {
              if (auto e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; }
              return 4194304;
          }();
          const int FASCAL_NUM_STREAMS_BUILD = []() -> int {
              if (auto e = getenv("FASCAL_NUM_STREAMS")) { int v = atoi(e); if (v > 0) return v; }
              return 4;
          }();
          int n_b = (num_tuples + FASCAL_GPU_BATCH_BUILD - 1) / FASCAL_GPU_BATCH_BUILD;
          for (int _b = 0; _b < n_b; ++_b) {
              int b_off = _b * FASCAL_GPU_BATCH_BUILD;
              int b_cnt = std::min(FASCAL_GPU_BATCH_BUILD, num_tuples - b_off);
              int b_blk = (b_cnt + TILE_SIZE - 1) / TILE_SIZE;
              cudaStream_t _stream = 0;
              tpch_q21_kernel_state_build<BLOCK_THREADS, ITEMS_PER_THREAD, true>
                  <<<b_blk, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples, b_off, d_l_orderkey, d_l_suppkey,
                  d_l_receiptdate, d_l_commitdate,
                  d_l3_state,
                  d_l23_prefilter_mapped, d_l23_tile_summary);
          }
          for (int _s = 0; _s < FASCAL_NUM_STREAMS_BUILD; ++_s)
              cudaStreamSynchronize(fascal_streams[_s]);
          CUDA_CHECK(cudaGetLastError());
      }
      printf("l3 state build done\n");

      // supplier (~10K rows < 256K): single launch, batch_offset=0
      tpch_q21_cpu_prefilter_supplier(h_supplier_s_suppkey, h_supplier_s_name_enc, h_supplier_s_nationkey, h_supplier_s_name, num_supplier_tuples, h_supplier_prefilter_pinned, h_supplier_tile_summary, (bf_storage[1].bits ? &bf_storage[1] : nullptr));
      tpch_q21_kernel_supplier<BLOCK_THREADS, ITEMS_PER_THREAD>
          <<<(num_supplier_tuples + TILE_SIZE - 1) / TILE_SIZE, BLOCK_THREADS, 0, fascal_streams[0]>>>(
          num_supplier_tuples, /*batch_offset=*/0, d_supplier_s_suppkey, d_supplier_s_name_enc, d_supplier_s_nationkey, d_supplier_s_name,
          ht_supplier.d_entries, ht_supplier.ht_size,
          gpu_bf_supplier.d_bits, gpu_bf_supplier.size_bits, 3,
          d_supplier_prefilter_mapped,
          d_supplier_tile_summary,
          ht_nation.d_entries, ht_nation.ht_size);
      CUDA_CHECK(cudaStreamSynchronize(fascal_streams[0]));
      CUDA_CHECK(cudaGetLastError());
      if (bf_storage[2].bits) gpu_bf_supplier.copy_to_host(bf_storage[2].bits, ((size_t)(bf_storage[2].size_bits + 63) / 64) * sizeof(uint64_t));

      // === Result Pipeline ===
      if (run_result_kernel() != 0) {
          return 1;
      }
      cudaEventRecord(t1);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(t_query));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
      printf("Kernel: %.3f ms\n", ms);
      if (in_memory_repeats > 1) {
          float hot_total_ms = 0.0f;
          float hot_best_ms = 0.0f;
          for (int _repeat = 0; _repeat < in_memory_repeats; ++_repeat) {
              std::memset(h_seed_bitmap_storage, 0, ((size_t)(num_tuples + 31) / 32) * sizeof(uint32_t));
              std::memset(h_tile_summary, 0, ((size_t)(num_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(unsigned long long)));
              CUDA_CHECK(cudaEventRecord(t0));
              if (run_result_kernel() != 0) {
                  return 1;
              }
              CUDA_CHECK(cudaEventRecord(t1));
              CUDA_CHECK(cudaDeviceSynchronize());
              float _hot_ms = 0.0f;
              CUDA_CHECK(cudaEventElapsedTime(&_hot_ms, t0, t1));
              hot_total_ms += _hot_ms;
              if (_repeat == 0 || _hot_ms < hot_best_ms) hot_best_ms = _hot_ms;
          }
          printf("In-memory result kernel: best %.3f ms avg %.3f ms (%d runs)\n",
                 hot_best_ms, hot_total_ms / (float)in_memory_repeats, in_memory_repeats);
      }

      // === Output ===
      unsigned long long *h_agg_table = (unsigned long long*)malloc(num_groups * NUM_AGGREGATES * sizeof(unsigned long long));
      CUDA_CHECK(cudaMemcpy(h_agg_table, d_aggtable, num_groups * NUM_AGGREGATES * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      std::vector<std::pair<std::tuple<long long, long long>, size_t>> _result_rows;
      for (size_t k = 0; k < num_groups; k++) {
        if (!((h_agg_table[k * NUM_AGGREGATES + 0] != 0))) continue;
        _result_rows.push_back({std::tuple<long long, long long>{(long long)(h_agg_table[k * NUM_AGGREGATES + 0]), (long long)(h_supplier_s_name[(int)k])}, k});
      }
      std::sort(_result_rows.begin(), _result_rows.end(), [](const auto &_a, const auto &_b) {
        if (std::get<0>(_a.first) != std::get<0>(_b.first)) return std::get<0>(_a.first) > std::get<0>(_b.first);
        if (std::get<1>(_a.first) != std::get<1>(_b.first)) return std::get<1>(_a.first) < std::get<1>(_b.first);
        return false;
      });
      int _print_limit = std::min((int)_result_rows.size(), (int)100);
      for (int _ri = 0; _ri < _print_limit; _ri++) {
        size_t k = _result_rows[_ri].second;
        printf("ROW: ");
        printf("%s", [&]() -> const char* {
    static thread_local char buf[64];
    int idx = (int)h_supplier_s_name[(int)k];
    int val = 1 + idx;
    if (val < 1) return "UNKNOWN";
    snprintf(buf, sizeof(buf), "Supplier#%09d", val);
    return buf;
  }());
        printf("|%lld", (long long)h_agg_table[k * NUM_AGGREGATES + 0]);
        printf("\n");
      }
      free(h_agg_table);
      CUDA_CHECK(cudaFree(d_aggtable));
      CUDA_CHECK(cudaEventRecord(t_result));
      CUDA_CHECK(cudaDeviceSynchronize());

      // === Timing Report ===
      {
          float ms_load = 0, ms_query = 0, ms_result = 0;
          CUDA_CHECK(cudaEventElapsedTime(&ms_load, t_start, t_load));
          CUDA_CHECK(cudaEventElapsedTime(&ms_query, t_load, t_query));
          CUDA_CHECK(cudaEventElapsedTime(&ms_result, t_query, t_result));
          printf("Timing: load=%.1fms query=%.1fms result=%.1fms total=%.1fms\n",
                 ms_load, ms_query, ms_result,
                 ms_load + ms_query + ms_result);
      }

      // === Cleanup ===
      // Shared lineitem columns: unregister ONCE per physical column
      odzc_mgr->unregister_column(mc_l_suppkey);
      odzc_mgr->unregister_column(mc_l_receiptdate);
      odzc_mgr->unregister_column(mc_l_commitdate);
      odzc_mgr->unregister_column(mc_l_orderkey);

      odzc_mgr->unregister_column(mc_orders_o_orderkey);
      odzc_mgr->unregister_column(mc_orders_o_orderstatus_enc);
      odzc_mgr->unregister_column(mc_nation_n_nationkey);
      odzc_mgr->unregister_column(mc_nation_n_name);
      // l2/l3 share lineitem columns -- no separate unregister
      odzc_mgr->unregister_column(mc_supplier_s_suppkey);
      odzc_mgr->unregister_column(mc_supplier_s_name_enc);
      odzc_mgr->unregister_column(mc_supplier_s_nationkey);
      odzc_mgr->unregister_column(mc_supplier_s_name);
      CUDA_CHECK(cudaFreeHost(h_orders_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_orders_prefilter_pinned));
      gpu_bf_orders.free_filter();
      ht_orders.free_table();
      bf_storage[0].free_filter();
      fascal_free_column(h_orders_o_orderkey);
      fascal_free_column(h_orders_o_orderstatus_enc);
      CUDA_CHECK(cudaFreeHost(h_nation_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_nation_prefilter_pinned));
      gpu_bf_nation.free_filter();
      ht_nation.free_table();
      bf_storage[1].free_filter();
      fascal_free_column(h_nation_n_nationkey);
      fascal_free_column(h_nation_n_name);
      // l2/l3 compact state arrays (device memory)
      CUDA_CHECK(cudaFree(d_l2_state));
      CUDA_CHECK(cudaFree(d_l3_state));
      // Shared l2/l3 prefilter (all-ones, pinned)
      CUDA_CHECK(cudaFreeHost(h_l23_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_l23_prefilter_pinned));
      CUDA_CHECK(cudaFreeHost(h_supplier_tile_summary));
      CUDA_CHECK(cudaFreeHost(h_supplier_prefilter_pinned));
      gpu_bf_supplier.free_filter();
      ht_supplier.free_table();
      bf_storage[2].free_filter();
      fascal_free_column(h_supplier_s_suppkey);
      fascal_free_column(h_supplier_s_name_enc);
      fascal_free_column(h_supplier_s_nationkey);
      fascal_free_column(h_supplier_s_name);

      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      // Free shared lineitem columns (one cudaFree per physical column)
      fascal_free_column(h_l_suppkey);
      fascal_free_column(h_l_receiptdate);
      fascal_free_column(h_l_commitdate);
      fascal_free_column(h_l_orderkey_enc);
      fascal_arena_destroy();
      for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) cudaStreamDestroy(fascal_streams[_s]);
      delete[] fascal_streams;
      cudaEventDestroy(t0);
      cudaEventDestroy(t1);
      cudaEventDestroy(t_start);
      cudaEventDestroy(t_load);
            cudaEventDestroy(t_query);
      cudaEventDestroy(t_result);
      return 0;

    }
}
