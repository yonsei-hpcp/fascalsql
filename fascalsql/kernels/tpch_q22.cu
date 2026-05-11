// TPC-H Q22 -- FaScalSQL optimized kernel (ODZC + AFP + CQO)
// Source: fascalsql/kernels/tpch_q22.cu

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


// ---- GPU SQL LIKE Matching ----
__device__ bool sql_like_match(const char* s, const char* p) {
    const char* cp = nullptr;
    const char* cs = nullptr;
    while (*s && *p != '*') {
        if (*p == '%') {
            while (*p == '%') p++;
            if (!*p) return true;
            cp = p;
            cs = s + 1;
        } else if (*p == '_' || *p == *s) {
            p++;
            s++;
        } else if (cp) {
            p = cp;
            s = cs++;
        } else {
            return false;
        }
    }
    while (*p == '%') p++;
    return !*p && !*s;
}

// Bitset primitives for main kernel
template<typename T, int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__device__ void BlockPredAndBitset(const T* items, int* selection_flags, int num_items, const unsigned char* bitset) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD_T; i++) {
        if ((threadIdx.x + (BLOCK_THREADS_T * i)) < num_items) {
            if (selection_flags[i]) {
                if (!bitset[items[i]]) selection_flags[i] = 0;
            }
        }
    }
}

template<typename T, int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__device__ void BlockPredAndBitsetInv(const T* items, int* selection_flags, int num_items, const unsigned char* bitset) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD_T; i++) {
        if ((threadIdx.x + (BLOCK_THREADS_T * i)) < num_items) {
            if (selection_flags[i]) {
                if (bitset[items[i]]) selection_flags[i] = 0;
            }
        }
    }
}

__global__ void regex_bitset_kernel(const char* dict_strings, const int* dict_offsets, int num_entries, const char* pattern, unsigned char* bitset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_entries) {
        const char* s = dict_strings + dict_offsets[idx];
        bitset[idx] = sql_like_match(s, pattern) ? 1 : 0;
    }
}

// -----------------------------------------------------------------------------
// Ablation flags (0=GPU, 1=CPU)
// -----------------------------------------------------------------------------
__device__ __constant__ unsigned char d_pred_flags[MAX_PREDICATES];
static unsigned char h_pred_flags[MAX_PREDICATES] = {0};

static inline void normalize_or_group_pred_flags(unsigned char *pred_flags) {
  {
    const unsigned char normalized = 0;
    pred_flags[1] = normalized;
    pred_flags[2] = normalized;
    pred_flags[3] = normalized;
    pred_flags[4] = normalized;
    pred_flags[5] = normalized;
    pred_flags[6] = normalized;
    pred_flags[7] = normalized;
  }
}

// GPU-side Bitset pointers for string predicates (LIKE/NOT_LIKE)
// Pre-filled by regex_bitset_kernel on unique dictionary entries.
__constant__ unsigned char* d_pred_bitset_1;
unsigned char* h_pred_bitset_1 = nullptr;
unsigned char* d_pred_bitset_ptr_1 = nullptr;
__constant__ unsigned char* d_pred_bitset_2;
unsigned char* h_pred_bitset_2 = nullptr;
unsigned char* d_pred_bitset_ptr_2 = nullptr;
__constant__ unsigned char* d_pred_bitset_3;
unsigned char* h_pred_bitset_3 = nullptr;
unsigned char* d_pred_bitset_ptr_3 = nullptr;
__constant__ unsigned char* d_pred_bitset_4;
unsigned char* h_pred_bitset_4 = nullptr;
unsigned char* d_pred_bitset_ptr_4 = nullptr;
__constant__ unsigned char* d_pred_bitset_5;
unsigned char* h_pred_bitset_5 = nullptr;
unsigned char* d_pred_bitset_ptr_5 = nullptr;
__constant__ unsigned char* d_pred_bitset_6;
unsigned char* h_pred_bitset_6 = nullptr;
unsigned char* d_pred_bitset_ptr_6 = nullptr;
__constant__ unsigned char* d_pred_bitset_7;
unsigned char* h_pred_bitset_7 = nullptr;
unsigned char* d_pred_bitset_ptr_7 = nullptr;
__constant__ unsigned char* d_pred_bitset_0;
unsigned char* h_pred_bitset_0 = nullptr;
unsigned char* d_pred_bitset_ptr_0 = nullptr;
__constant__ int* d_dict_transform__customer__c_phone__substr_1_2_int;
int* d_dict_transform__customer__c_phone__substr_1_2_int_ptr = nullptr;
// ================= STAGE 1: tpch_q22___scalar_sq_2 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q22___scalar_sq_2_kernel_customer(
    int num_tuples,
    int batch_offset,
    int *d_customer_c_acctbal,
    int *d_customer_c_phone,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  int loc_customer_c_acctbal[ITEMS_PER_THREAD_T];
  int loc_customer_c_phone[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // ODZC lazy load: c_acctbal (survivors of prior predicates)
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_customer_c_acctbal + tile_offset, loc_customer_c_acctbal,
      selection_flags, num_tile_items);

  // Predicate 0: customer.c_acctbal (unified with sub-pipeline selection)
  if (d_pred_flags[0] == 0) {  // GPU
    BlockPredAndGT<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
        loc_customer_c_acctbal, 0, selection_flags, num_tile_items);
  }

  // ODZC lazy load (OR group): c_phone
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_customer_c_phone + tile_offset, loc_customer_c_phone,
      selection_flags, num_tile_items);

  // OR predicate group (GPU): (p1 OR p2 OR ...) AND with selection
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM])
        selection_flags[ITEM] = (((d_pred_bitset_1[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_2[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_3[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_4[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_5[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_6[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_7[loc_customer_c_phone[ITEM]]))) ? 1 : 0;
  }

  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  } while(0);
  // Scalar aggregation with block-level reduction (one atomic per block)
  double _acc0 = 0, _acc1 = 0;
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        _acc0 += ((((double)((double)loc_customer_c_acctbal[ITEM])) / 100.0));
        _acc1 += 1LL;
      }
  }
  _acc0 = blockReduceSumDouble<BLOCK_THREADS_T>(_acc0);
  if (threadIdx.x == 0) atomicAdd(&aggtable[0], (double)_acc0);
  _acc1 = blockReduceSumDouble<BLOCK_THREADS_T>(_acc1);
  if (threadIdx.x == 0) atomicAdd(&aggtable[1], (double)_acc1);
}
// CPU predicate evaluation for tpch_q22___scalar_sq_2 (fact table: customer)
// Multi-pass: one pass per column group + BF probe; bitmap skip between passes
static void tpch_q22___scalar_sq_2_cpu_predicate_customer(
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

  // c_acctbal > 0 (range pass with lo=1, no upper bound)
  if (h_pred_flags[0] == 1)
    fascal::cpu_pred::avx2_range_pass(bm, bs, h_data[0], morsel_offset, elem_cnt, 1, INT32_MAX, true, false);

  // OR group 1: predicates 1..7 stay on GPU
}

// ---- AFP: multi-threaded CPU prefilter for result pipeline 'customer' ----
// CPU fills bitmap, then GPU launches (no doorbell overlap).
// total_tuples: total fact-table size (for bitmap bounds).
// batch_offset + batch_count define the sub-range to prefilter.
// Sequential mode: batch_offset=0, batch_count=total_tuples (default).
// Pipelined mode: called per-batch with specific offset/count.
static void tpch_q22___scalar_sq_2_cpu_prefilter_customer(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    fascal::runtime::BloomFilter *bloom_filters[] = nullptr,
    int num_bloom_filters = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
    fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
        [&](int off, int cnt) { tpch_q22___scalar_sq_2_cpu_predicate_customer(h_data, off, cnt, bitmap, 0, 0, bloom_filters, num_bloom_filters); },
        batch_offset, total_tuples);
}


// ================= STAGE 2: tpch_q22 =================

// -----------------------------------------------------------------------------
// Result-pipeline kernel (CPU-first: seed bitmap fully ready, predicates, joins, aggregation)
// -----------------------------------------------------------------------------

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q22_kernel_customer(
    int num_tuples,
    int batch_offset,
    double *d___scalar_sq_2_avg,
    int *d_customer_c_acctbal,
    int *d_customer_c_custkey,
    int *d_customer_c_phone,
    const uint32_t *__restrict__ seed_bitmap,
    const uint8_t *__restrict__ tile_summary,
    const unsigned int *__restrict__ d_has_order_bitmap,
    int max_custkey,
    double *aggtable)
{
  FASCAL_KERNEL_PROLOGUE(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);

  double loc___scalar_sq_2_avg[ITEMS_PER_THREAD_T];
  int loc_customer_c_acctbal[ITEMS_PER_THREAD_T];
  int loc_customer_c_custkey[ITEMS_PER_THREAD_T];
  int loc_customer_c_phone[ITEMS_PER_THREAD_T];

  if (FASCAL_ANY_ALIVE(ITEMS_PER_THREAD_T)) do {
  // ODZC lazy load (OR group, also used in aggregation): c_phone
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_customer_c_phone + tile_offset, loc_customer_c_phone,
      selection_flags, num_tile_items);

  // OR predicate group (GPU): (p1 OR p2 OR ...) AND with selection
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM])
        selection_flags[ITEM] = (((d_pred_bitset_0[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_1[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_2[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_3[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_4[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_5[loc_customer_c_phone[ITEM]])) || ((d_pred_bitset_6[loc_customer_c_phone[ITEM]]))) ? 1 : 0;
  }

  FASCAL_CHECK_ALIVE(ITEMS_PER_THREAD_T);

  // Join 0 (antijoin): c_custkey -> __correlated_sq_1
  // ODZC lazy load (join FK, before __correlated_sq_1 hash probe): c_custkey
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_customer_c_custkey + tile_offset, loc_customer_c_custkey,
      selection_flags, num_tile_items);

  // ODZC lazy load (post-join check source): customer.c_acctbal
  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      d_customer_c_acctbal + tile_offset, loc_customer_c_acctbal,
      selection_flags, num_tile_items);

  // Antijoin via dense bitmap: keep row only if c_custkey bit NOT set (NOT IN orders).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items && selection_flags[ITEM]) {
      int ck = loc_customer_c_custkey[ITEM];
      bool in_orders = (ck >= 0 && ck < max_custkey)
                       ? ((d_has_order_bitmap[(unsigned int)ck >> 5] >> ((unsigned int)ck & 31u)) & 1u)
                       : 0;
      if (in_orders) selection_flags[ITEM] = 0;
    }
  }

  // Broadcast load (scalar auxiliary table): __scalar_sq_2.avg
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM])
        loc___scalar_sq_2_avg[ITEM] = d___scalar_sq_2_avg[0];
  }

  // Post-join check: customer.c_acctbal > __scalar_sq_2.avg
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if (selection_flags[ITEM]) {
      if (!((((double)loc_customer_c_acctbal[ITEM]) / 100.0) > loc___scalar_sq_2_avg[ITEM]))
        selection_flags[ITEM] = 0;
    }
  }

  // GROUP BY direct-index aggregation (multi-aggregate)
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) < num_tile_items)
      if (selection_flags[ITEM]) {
        int gk = d_dict_transform__customer__c_phone__substr_1_2_int[loc_customer_c_phone[ITEM]];
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 0], 1LL);
        atomicAdd(&aggtable[gk * NUM_AGGREGATES + 1], ((((double)((double)loc_customer_c_acctbal[ITEM])) / 100.0)));
      }
  }
  } while(0); // thread_any_alive
}
// CPU predicate evaluation for tpch_q22 (fact table: customer)
// h_has_order_bitmap: host copy of the dense antijoin bitmap (uint32 words).
static void tpch_q22_cpu_predicate_customer(
    int *h_data[],
    int morsel_offset,
    int elem_cnt,
    uint32_t *bitmap,
    int selection_step,
    int ablation_indicator,
    const unsigned int *h_has_order_bitmap = nullptr,
    int max_custkey = 0)
{
  uint8_t *bm = (uint8_t*)bitmap;
  int bs = morsel_offset >> 3, nb = (elem_cnt + 7) >> 3;
  fascal::cpu_pred::init_bitmap(bm, bs, nb, elem_cnt);

  // Antijoin probe: exclude rows whose c_custkey appears in orders bitmap (col idx 1 = c_custkey).
  if (h_has_order_bitmap && max_custkey > 0 && h_data[1]) {
    int *h_c_custkey = h_data[1];
    for (int i = 0; i < elem_cnt; ++i) {
      int idx = morsel_offset + i;
      // Skip if already filtered out.
      if (!((bm[idx >> 3] >> (idx & 7)) & 1u)) continue;
      int ck = h_c_custkey[idx];
      bool in_orders = (ck >= 0 && ck < max_custkey)
                       ? ((h_has_order_bitmap[(unsigned int)ck >> 5] >> ((unsigned int)ck & 31u)) & 1u)
                       : false;
      if (in_orders) bm[idx >> 3] &= ~(uint8_t)(1u << (idx & 7));
    }
  }
  // OR group: c_phone prefix predicates stay on GPU
}

// ---- AFP: multi-threaded CPU prefilter for result pipeline 'customer' ----
// CPU fills bitmap (including antijoin), then GPU launches.
static void tpch_q22_cpu_prefilter_customer(
    int *h_data[], int batch_count, uint32_t *bitmap, uint8_t *tile_summary,
    const unsigned int *h_has_order_bitmap = nullptr,
    int max_custkey = 0,
    int batch_offset = 0,
    int total_tuples = 0) {
    fascal::cpu_pred::run_prefilter(batch_count, bitmap, tile_summary,
        [&](int off, int cnt) { tpch_q22_cpu_predicate_customer(h_data, off, cnt, bitmap, 0, 0, h_has_order_bitmap, max_custkey); },
        batch_offset, total_tuples);
}

template <int BLOCK_THREADS_T, int ITEMS_PER_THREAD_T>
__global__ void tpch_q22_kernel___correlated_sq_1(
    int num_tuples, int batch_offset,
    const int *__restrict__ d___correlated_sq_1_o_custkey,
    unsigned int *d_has_order_bitmap,
    int max_custkey,
    const uint32_t *d_prefilter,
    const uint8_t *d_tile_summary)
 {
  FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BLOCK_THREADS_T, ITEMS_PER_THREAD_T);
  int loc___correlated_sq_1_o_custkey[ITEMS_PER_THREAD_T];

  BlockLoadSelect<int, BLOCK_THREADS_T, ITEMS_PER_THREAD_T>(
      (int*)(d___correlated_sq_1_o_custkey + tile_offset), loc___correlated_sq_1_o_custkey, selection_flags, num_tile_items);

  // Dense bitmap: set bit for each o_custkey seen in orders (replaces HT + Bloom filter).
  // Use 32-bit word atomicOr (CUDA has no byte-granularity atomicOr).
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD_T; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS_T * ITEM)) >= num_tile_items) break;
    if (!selection_flags[ITEM]) continue;
    int ck = loc___correlated_sq_1_o_custkey[ITEM];
    if (ck >= 0 && ck < max_custkey) {
      atomicOr(&d_has_order_bitmap[(unsigned int)ck >> 5], 1u << ((unsigned int)ck & 31u));
    }
  }
}

// ---- AFP: multi-threaded CPU prefilter for build pipeline '__correlated_sq_1' ----
// Fills entire prefilter bitmap (scalar preds + upstream BF probes),
// then GPU build kernel launches with bitmap 100% ready (no doorbell).
static void tpch_q22_cpu_prefilter___correlated_sq_1(
    int *h___correlated_sq_1_o_custkey,
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
                if (1) {
                    bitmap[(uint32_t)idx >> 5] |= (1u << ((uint32_t)idx & 31));
                }
            }
            // 2. Apply upstream BF probes (AND-mask with FK column probes).

        });
}


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

    // Shared pinned subquery materializations
    int _mat_rows___scalar_sq_2 = 0;
    double *_mat___scalar_sq_2_avg = nullptr;
    // Stage 1 Regex Bitsets
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "13%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_1, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_1 = d_bitset;
        h_pred_bitset_1 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_1, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "31%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_2, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_2 = d_bitset;
        h_pred_bitset_2 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_2, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "23%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_3, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_3 = d_bitset;
        h_pred_bitset_3 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_3, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "29%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_4, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_4 = d_bitset;
        h_pred_bitset_4 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_4, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "30%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_5, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_5 = d_bitset;
        h_pred_bitset_5 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_5, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "18%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_6, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_6 = d_bitset;
        h_pred_bitset_6 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_6, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "17%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_7, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_7 = d_bitset;
        h_pred_bitset_7 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_7, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    // Stage 2 Regex Bitsets
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "13%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_0, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_0 = d_bitset;
        h_pred_bitset_0 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_0, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "31%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_1, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_1 = d_bitset;
        h_pred_bitset_1 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_1, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "23%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_2, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_2 = d_bitset;
        h_pred_bitset_2 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_2, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "29%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_3, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_3 = d_bitset;
        h_pred_bitset_3 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_3, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "30%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_4, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_4 = d_bitset;
        h_pred_bitset_4 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_4, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "18%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_5, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_5 = d_bitset;
        h_pred_bitset_5 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_5, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    {
        // Prepare regex bitset for customer.c_phone (dict files under 'customer')
        int dict_entry_count = 0;
        int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
        if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
        long strings_size_bytes = 0;
        FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
        if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
        fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
        char* h_dict_strings = (char*)malloc(strings_size_bytes);
        fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
        
        char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
        CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
        CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_pattern, 4));
        CUDA_CHECK(cudaMemcpy(d_pattern, "17%", 4, cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (dict_entry_count + threads - 1) / threads;
        regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
        CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_6, &d_bitset, sizeof(unsigned char*)));
        d_pred_bitset_ptr_6 = d_bitset;
        h_pred_bitset_6 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
        CUDA_CHECK(cudaMemcpy(h_pred_bitset_6, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        fascal_free_column(h_dict_offsets); free(h_dict_strings);
    }
    auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(new fascal::runtime::ODZCManager(0));

    // ---------------- STAGE 1 EXECUTION: tpch_q22___scalar_sq_2 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q22___scalar_sq_2
  // ============================================================
      printf("=== FaScalSQL: tpch_q22___scalar_sq_2 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/c_acctbal.bin", data_dir);
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

      // === Pinned Arena Pre-allocation ===
      {
          size_t arena_cols = 5;
          size_t arena_bytes = (size_t)num_tuples * sizeof(int) * arena_cols;
          arena_bytes = arena_bytes + arena_bytes / 8;
          fascal_arena_init(arena_bytes);
      }

      // === Data Loading ===
      char path[512];
      // ----- Load result-pipeline (fact) columns -----
  
      snprintf(path, sizeof(path), "%s/c_acctbal.bin", data_dir);
      int *h_c_acctbal = load_binary_column(path, num_tuples);
      if (!h_c_acctbal) { fprintf(stderr, "Failed to load c_acctbal\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/c_phone.bin", data_dir);
      int *h_c_phone = load_binary_column(path, num_tuples);
      if (!h_c_phone) { fprintf(stderr, "Failed to load c_phone\n"); return 1; }
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      cqo_in.predicate_selectivities.push_back(0.300000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
  
      cqo_in.num_gpu_only_ops = 1;
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
  
  
  
      // === AFP Manager & Predicate Filters ===
      int *h_data[] = {h_c_acctbal, h_c_phone};
      fascal::runtime::AFPManager afp_manager(0);
      int afp_filter_id = 0;
      if (h_pred_flags[0] == 1) {
        auto desc = fascal::runtime::afp::make_filter<int>(0, fascal::runtime::AFPManager::FilterOp::GT, 0);
        afp_manager.register_filter(afp_filter_id++, desc);
      }
  
      // === Bloom Filters (Build Pipelines) ===
  
  
  
  
      // === ODZC & Device Buffers ===
      auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(
          new fascal::runtime::ODZCManager(0));
  
      // === Seed Bitmap + Tile Summary (CPU-first, no doorbell) ===
      uint32_t *h_seed_bitmap_storage = nullptr, *d_seed_bitmap = nullptr;
      uint8_t *h_tile_summary = nullptr, *d_tile_summary = nullptr;
      fascal_alloc_seed_bitmap(num_tuples, &h_seed_bitmap_storage, &d_seed_bitmap, &h_tile_summary, &d_tile_summary);
  
      fascal::runtime::ODZCManager::MappedColumn mc_customer_c_acctbal;
      int *d_customer_c_acctbal = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_c_acctbal, num_tuples, "c_acctbal", &mc_customer_c_acctbal, &d_customer_c_acctbal);
  
      fascal::runtime::ODZCManager::MappedColumn mc_customer_c_phone;
      int *d_customer_c_phone = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_c_phone, num_tuples, "c_phone", &mc_customer_c_phone, &d_customer_c_phone);
  
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
  
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "13%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_1, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_1 = d_bitset;
          h_pred_bitset_1 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_1, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "31%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_2, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_2 = d_bitset;
          h_pred_bitset_2 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_2, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "23%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_3, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_3 = d_bitset;
          h_pred_bitset_3 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_3, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "29%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_4, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_4 = d_bitset;
          h_pred_bitset_4 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_4, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "30%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_5, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_5 = d_bitset;
          h_pred_bitset_5 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_5, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "18%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_6, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_6 = d_bitset;
          h_pred_bitset_6 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_6, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "17%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_7, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_7 = d_bitset;
          h_pred_bitset_7 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_7, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
  
      // === Aggregation Table ===
      double *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, (size_t)NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(1) * NUM_AGGREGATES * sizeof(double)));
  
      // === Hash Table Build ===
  
  
      cudaEvent_t t0, t1;
      CUDA_CHECK(cudaEventCreate(&t0));
      CUDA_CHECK(cudaEventCreate(&t1));
  
      int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
      printf("Launch: %d blocks x %d threads (CPU-first prefilter)\n", num_blocks, BLOCK_THREADS);
      // --- Pipelined CPU-GPU execution (Section V-E) ---
      // GPU_BATCH_SIZE tuples per batch; CPU prefilters one batch, then async GPU launch on a stream.
      // Overlap: while GPU processes batch N, CPU prefilters batch N+1.
      const int FASCAL_GPU_BATCH = []() -> int {
          if (const char *e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; }
          return 4194304;  // 4M tuples default (~8K tiles, fills all SMs with minimal launch overhead)
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
              tpch_q22___scalar_sq_2_cpu_prefilter_customer(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0);
              tpch_q22___scalar_sq_2_kernel_customer<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d_customer_c_acctbal,
          d_customer_c_phone,
                  d_seed_bitmap,
                  d_tile_summary,
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
              tpch_q22___scalar_sq_2_cpu_prefilter_customer(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, nullptr, 0, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q22___scalar_sq_2_kernel_customer<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d_customer_c_acctbal,
          d_customer_c_phone,
                  d_seed_bitmap,
                  d_tile_summary,
              d_aggtable);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };
  
      cudaEventRecord(t0);
  
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
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(1) * NUM_AGGREGATES * sizeof(double)));
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
      double *h_result = (double*)malloc((size_t)NUM_AGGREGATES * sizeof(double));
      CUDA_CHECK(cudaMemcpy(h_result, d_aggtable, (size_t)NUM_AGGREGATES * sizeof(double), cudaMemcpyDeviceToHost));
      int k = 0;
      double* h_agg_table = h_result;
      // Materializing scalar aggregation results to __scalar_sq_2_*.bin
      double h_out_val0 = ((h_result[1]) != 0 ? ((double)(h_result[0]) / (double)(h_result[1])) : 0.0);
      char path_mat0[512];
      snprintf(path_mat0, sizeof(path_mat0), "%s/__scalar_sq_2_avg.bin", data_dir);
      FILE *fp_mat0 = fopen(path_mat0, "wb");
      if (fp_mat0) {
        fwrite(&h_out_val0, sizeof(double), 1, fp_mat0);
        fclose(fp_mat0);
        printf("Materialized 1 row to %s\n", path_mat0);
      }
      _mat_rows___scalar_sq_2 = 1;
      if (_mat___scalar_sq_2_avg) CUDA_CHECK(cudaFreeHost(_mat___scalar_sq_2_avg));
      if (_mat_rows___scalar_sq_2 > 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&_mat___scalar_sq_2_avg), (size_t)_mat_rows___scalar_sq_2 * sizeof(double), cudaHostAllocDefault));
        std::memcpy(_mat___scalar_sq_2_avg, &h_out_val0, (size_t)_mat_rows___scalar_sq_2 * sizeof(double));
      } else {
        _mat___scalar_sq_2_avg = nullptr;
      }
      printf("Stage scalar materialization complete\n");
      free(h_result);
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
      odzc_mgr->unregister_column(mc_customer_c_acctbal);
      odzc_mgr->unregister_column(mc_customer_c_phone);
  
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_c_acctbal);
      fascal_free_column(h_c_phone);
      fascal_arena_destroy();
      for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) cudaStreamDestroy(fascal_streams[_s]);
      delete[] fascal_streams;
      cudaEventDestroy(t0);
      cudaEventDestroy(t1);
      cudaEventDestroy(t_start);
      cudaEventDestroy(t_load);
            cudaEventDestroy(t_query);
      cudaEventDestroy(t_result);
  
    }

    // ---------------- STAGE 2 EXECUTION: tpch_q22 ----------------
    {
      if (!data_dir) data_dir = getenv("FASCALSQL_TPCH_DATA_DIR");
      if (!data_dir) { fprintf(stderr, "Missing default data dir!\n"); return 1; }
  // ============================================================
  // Main Stage Body — FaScalSQL Query: tpch_q22
  // ============================================================
      printf("=== FaScalSQL: tpch_q22 ===\n");
      if (argc < 3) {
          printf("Usage: %s <num_tuples> <data_dir> [pred_flags...]\n", argv[0]);
          return 1;
      }
      int num_tuples = 0;  // auto-detect in multi-stage mode
      data_dir = argv[2];
      if (num_tuples <= 0) {
          char _auto_path[512];
          snprintf(_auto_path, sizeof(_auto_path), "%s/c_acctbal.bin", data_dir);
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
      char path[512];
      // ----- Load result-pipeline (fact) columns -----
  
      snprintf(path, sizeof(path), "%s/c_acctbal.bin", data_dir);
      int *h_c_acctbal = load_binary_column(path, num_tuples);
      if (!h_c_acctbal) { fprintf(stderr, "Failed to load c_acctbal\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/c_custkey.bin", data_dir);
      int *h_c_custkey = load_binary_column(path, num_tuples);
      if (!h_c_custkey) { fprintf(stderr, "Failed to load c_custkey\n"); return 1; }
  
      snprintf(path, sizeof(path), "%s/c_phone.bin", data_dir);
      int *h_c_phone = load_binary_column(path, num_tuples);
      if (!h_c_phone) { fprintf(stderr, "Failed to load c_phone\n"); return 1; }
  
      // ----- Load scalar auxiliary materialized tables -----
  
      int num___scalar_sq_2_tuples = 0;
      double *h___scalar_sq_2_avg = nullptr;
      if (_mat_rows___scalar_sq_2 > 0 && _mat___scalar_sq_2_avg != nullptr) {
        num___scalar_sq_2_tuples = _mat_rows___scalar_sq_2;
        h___scalar_sq_2_avg = _mat___scalar_sq_2_avg;
      } else {
        snprintf(path, sizeof(path), "%s/__scalar_sq_2_avg.bin", data_dir);
        h___scalar_sq_2_avg = (double*)load_binary_column_auto_u64(path, &num___scalar_sq_2_tuples);
      }
      if (!h___scalar_sq_2_avg || num___scalar_sq_2_tuples <= 0) { fprintf(stderr, "Failed to load __scalar_sq_2.avg\n"); return 1; }
  
      // ----- Load dimension: __correlated_sq_1 -----
      int num___correlated_sq_1_tuples = 0;
  
      snprintf(path, sizeof(path), "%s/o_custkey.bin", data_dir);
      int *h___correlated_sq_1_o_custkey = load_binary_column_auto(path, &num___correlated_sq_1_tuples);
  
      if (!h___correlated_sq_1_o_custkey || num___correlated_sq_1_tuples == 0) {
          fprintf(stderr, "Failed to load table column o_custkey\n"); return 1;
      }
      printf("Loaded __correlated_sq_1: %d tuples\n", num___correlated_sq_1_tuples);
  
      CUDA_CHECK(cudaEventRecord(t_load));
  
      // === CQO: Optimal AFP Placement ===
      auto cqo_in = fascal_cqo_init(num_tuples);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.predicate_selectivities.push_back(0.400000);
      cqo_in.num_gpu_only_ops = 1;
      auto cqo_out = fascal::optimizer::cqo_evaluate(cqo_in);
      fascal_cqo_apply(cqo_out, h_pred_flags, MAX_PREDICATES, argc, argv);
  fascal_cqo_print_decisions(cqo_in, cqo_out, h_pred_flags, MAX_PREDICATES);
      normalize_or_group_pred_flags(h_pred_flags);
      int in_memory_repeats = 1;
      if (const char *repeat_env = getenv("FASCAL_IN_MEMORY_REPEATS")) {
          in_memory_repeats = atoi(repeat_env);
          if (in_memory_repeats < 1) in_memory_repeats = 1;
      }
  
      // === AFP Manager & Predicate Filters ===
      int *h_data[] = {h_c_acctbal, h_c_custkey, h_c_phone};
      fascal::runtime::AFPManager afp_manager(0);
      int afp_filter_id = 0;
      (void)afp_manager;
  
      // === Bloom Filters (Build Pipelines) ===
      // __correlated_sq_1 antijoin now uses dense bitmap; no Bloom filter needed.
      fascal::runtime::BloomFilter *bf_array[1] = { nullptr };
  
      // === ODZC & Device Buffers ===
      auto odzc_mgr = std::unique_ptr<fascal::runtime::ODZCManager>(
          new fascal::runtime::ODZCManager(0));
  
      // === Seed Bitmap + Tile Summary (CPU-first, no doorbell) ===
      uint32_t *h_seed_bitmap_storage = nullptr, *d_seed_bitmap = nullptr;
      uint8_t *h_tile_summary = nullptr, *d_tile_summary = nullptr;
      fascal_alloc_seed_bitmap(num_tuples, &h_seed_bitmap_storage, &d_seed_bitmap, &h_tile_summary, &d_tile_summary);
  
      fascal::runtime::ODZCManager::MappedColumn mc___scalar_sq_2_avg;
      double *d___scalar_sq_2_avg = nullptr;
      {
        mc___scalar_sq_2_avg = odzc_mgr->register_column(h___scalar_sq_2_avg, (size_t)num___scalar_sq_2_tuples * sizeof(double));
        d___scalar_sq_2_avg = static_cast<double*>(mc___scalar_sq_2_avg.device_ptr);
        if (!d___scalar_sq_2_avg) {
          fprintf(stderr, "ODZC failed for __scalar_sq_2.avg, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d___scalar_sq_2_avg, (size_t)num___scalar_sq_2_tuples * sizeof(double)));
          CUDA_CHECK(cudaMemcpy(d___scalar_sq_2_avg, h___scalar_sq_2_avg, (size_t)num___scalar_sq_2_tuples * sizeof(double), cudaMemcpyHostToDevice));
        }
      }
  
      fascal::runtime::ODZCManager::MappedColumn mc_customer_c_acctbal;
      int *d_customer_c_acctbal = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_c_acctbal, num_tuples, "c_acctbal", &mc_customer_c_acctbal, &d_customer_c_acctbal);
  
      fascal::runtime::ODZCManager::MappedColumn mc_customer_c_custkey;
      int *d_customer_c_custkey = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_c_custkey, num_tuples, "c_custkey", &mc_customer_c_custkey, &d_customer_c_custkey);
  
      fascal::runtime::ODZCManager::MappedColumn mc_customer_c_phone;
      int *d_customer_c_phone = nullptr;
      fascal_odzc_map(odzc_mgr.get(), h_c_phone, num_tuples, "c_phone", &mc_customer_c_phone, &d_customer_c_phone);
  
      fascal::runtime::ODZCManager::MappedColumn mc___correlated_sq_1_o_custkey;
      mc___correlated_sq_1_o_custkey = odzc_mgr->register_column(h___correlated_sq_1_o_custkey, num___correlated_sq_1_tuples * sizeof(int));
      int *d___correlated_sq_1_o_custkey = static_cast<int*>(mc___correlated_sq_1_o_custkey.device_ptr);
      if (!d___correlated_sq_1_o_custkey) {
          fprintf(stderr, "ODZC failed for __correlated_sq_1.o_custkey, fallback to cudaMalloc\n");
          CUDA_CHECK(cudaMalloc(&d___correlated_sq_1_o_custkey, num___correlated_sq_1_tuples * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d___correlated_sq_1_o_custkey, h___correlated_sq_1_o_custkey, num___correlated_sq_1_tuples * sizeof(int), cudaMemcpyHostToDevice));
      }
  
      CUDA_CHECK(cudaMemcpyToSymbol(d_pred_flags, h_pred_flags, sizeof(h_pred_flags)));
      int *d_dict_transform__customer__c_phone__substr_1_2_int_ptr = nullptr;
      {
          // Runtime dict transform for high-cardinality customer.c_phone
          int dict_entry_count = 0;
          int* h_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_offsets) { fprintf(stderr, "Error: Could not load dict offsets for runtime transform customer.c_phone\n"); exit(1); }
          FILE* fp_st = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_st) { fprintf(stderr, "Error: Could not load dict strings for runtime transform customer.c_phone\n"); exit(1); }
          fseek(fp_st, 0, SEEK_END); long st_bytes = ftell(fp_st); rewind(fp_st);
          char* h_strings = (char*)malloc(st_bytes);
          fread(h_strings, 1, st_bytes, fp_st); fclose(fp_st);
          int* h_transform = (int*)calloc(dict_entry_count, sizeof(int));
          for (int i = 0; i < dict_entry_count; ++i) {
              const char* entry = h_strings + h_offsets[i];
              int slen = (int)strlen(entry);
              int s0 = 0;  // 0-based start
              if (s0 >= 0 && s0 < slen) {
                  int end = s0 + 2 < slen ? s0 + 2 : slen;
                  char buf[32]; memset(buf, 0, sizeof(buf));
                  memcpy(buf, entry + s0, end - s0);
                  h_transform[i] = atoi(buf);
              }
          }
          CUDA_CHECK(cudaMalloc(&d_dict_transform__customer__c_phone__substr_1_2_int_ptr, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_transform__customer__c_phone__substr_1_2_int_ptr, h_transform, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpyToSymbol(d_dict_transform__customer__c_phone__substr_1_2_int, &d_dict_transform__customer__c_phone__substr_1_2_int_ptr, sizeof(int*)));
          fascal_free_column(h_offsets); free(h_strings); free(h_transform);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "13%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_0, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_0 = d_bitset;
          h_pred_bitset_0 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_0, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "31%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_1, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_1 = d_bitset;
          h_pred_bitset_1 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_1, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "23%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_2, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_2 = d_bitset;
          h_pred_bitset_2 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_2, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "29%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_3, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_3 = d_bitset;
          h_pred_bitset_3 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_3, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "30%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_4, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_4 = d_bitset;
          h_pred_bitset_4 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_4, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "18%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_5, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_5 = d_bitset;
          h_pred_bitset_5 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_5, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
      {
          // Prepare regex bitset for customer.c_phone (dict files under 'customer')
          int dict_entry_count = 0;
          int* h_dict_offsets = load_binary_column_auto(std::string(std::string(data_dir) + "/customer_c_phone_dict_offsets.bin").c_str(), &dict_entry_count);
          if (!h_dict_offsets) { fprintf(stderr, "Error: Could not load dictionary offsets for customer.c_phone (path: %s/customer_c_phone_dict_offsets.bin)\n", data_dir); exit(1); }
          long strings_size_bytes = 0;
          FILE* fp_s = fopen(std::string(std::string(data_dir) + "/customer_c_phone_dict_strings.bin").c_str(), "rb");
          if (!fp_s) { fprintf(stderr, "Error: Could not load dictionary strings for customer.c_phone\n"); exit(1); }
          fseek(fp_s, 0, SEEK_END); strings_size_bytes = ftell(fp_s); rewind(fp_s);
          char* h_dict_strings = (char*)malloc(strings_size_bytes);
          fread(h_dict_strings, 1, strings_size_bytes, fp_s); fclose(fp_s);
          
          char *d_dict_strings; int *d_dict_offsets; unsigned char *d_bitset; char *d_pattern;
          CUDA_CHECK(cudaMalloc(&d_dict_strings, strings_size_bytes));
          CUDA_CHECK(cudaMemcpy(d_dict_strings, h_dict_strings, strings_size_bytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_dict_offsets, dict_entry_count * sizeof(int)));
          CUDA_CHECK(cudaMemcpy(d_dict_offsets, h_dict_offsets, dict_entry_count * sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&d_bitset, dict_entry_count * sizeof(unsigned char)));
          CUDA_CHECK(cudaMalloc(&d_pattern, 4));
          CUDA_CHECK(cudaMemcpy(d_pattern, "17%", 4, cudaMemcpyHostToDevice));
          
          int threads = 256;
          int blocks = (dict_entry_count + threads - 1) / threads;
          regex_bitset_kernel<<<blocks, threads>>>(d_dict_strings, d_dict_offsets, dict_entry_count, d_pattern, d_bitset);
          CUDA_CHECK(cudaMemcpyToSymbol(d_pred_bitset_6, &d_bitset, sizeof(unsigned char*)));
          d_pred_bitset_ptr_6 = d_bitset;
          h_pred_bitset_6 = (unsigned char *)malloc(dict_entry_count * sizeof(unsigned char));
          CUDA_CHECK(cudaMemcpy(h_pred_bitset_6, d_bitset, dict_entry_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
          
          fascal_free_column(h_dict_offsets); free(h_dict_strings);
      }
  
      // === Aggregation Table ===
      size_t num_groups = 1ULL;
      num_groups *= 100ULL;
      double *d_aggtable;
      CUDA_CHECK(cudaMalloc(&d_aggtable, num_groups * NUM_AGGREGATES * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(double)));
  
      // === Dense Bitmap (replaces HT+BF for __correlated_sq_1 antijoin) ===
      // max_custkey = customer row count = upper bound on c_custkey domain (~15M at SF=100).
      // Bitmap = 1 word per 32 keys; ~1.9 MB device memory vs ~2.4 GB for HT.
      int max_custkey = num_tuples;  // customer table size
      size_t bitmap_words = ((size_t)max_custkey + 31) / 32;
      unsigned int *d_has_order_bitmap = nullptr;
      CUDA_CHECK(cudaMalloc(&d_has_order_bitmap, bitmap_words * sizeof(unsigned int)));
      CUDA_CHECK(cudaMemset(d_has_order_bitmap, 0, bitmap_words * sizeof(unsigned int)));
      // Host mirror: filled after build (D->H copy) for CPU AFP antijoin probe.
      unsigned int *h_has_order_bitmap = (unsigned int*)calloc(bitmap_words, sizeof(unsigned int));
  
      // AFP all-pipeline: pinned mapped memory for __correlated_sq_1 packed prefilter + tile summary (no doorbell).
      size_t __correlated_sq_1_bitmap_words = ((size_t)num___correlated_sq_1_tuples + 31) / 32;
      size_t __correlated_sq_1_num_tiles = ((size_t)num___correlated_sq_1_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
      uint32_t *h___correlated_sq_1_prefilter_pinned = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h___correlated_sq_1_prefilter_pinned, __correlated_sq_1_bitmap_words * sizeof(uint32_t), cudaHostAllocMapped));
      std::memset(h___correlated_sq_1_prefilter_pinned, 0, __correlated_sq_1_bitmap_words * sizeof(uint32_t));
      uint32_t *d___correlated_sq_1_prefilter_mapped = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d___correlated_sq_1_prefilter_mapped, h___correlated_sq_1_prefilter_pinned, 0));
      uint8_t *h___correlated_sq_1_tile_summary = nullptr;
      CUDA_CHECK(cudaHostAlloc(&h___correlated_sq_1_tile_summary, __correlated_sq_1_num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
      std::memset(h___correlated_sq_1_tile_summary, 0, __correlated_sq_1_num_tiles);
      uint8_t *d___correlated_sq_1_tile_summary = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&d___correlated_sq_1_tile_summary, h___correlated_sq_1_tile_summary, 0));
  
  
      cudaEvent_t t0, t1;
      CUDA_CHECK(cudaEventCreate(&t0));
      CUDA_CHECK(cudaEventCreate(&t1));
  
      int num_blocks = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
      printf("Launch: %d blocks x %d threads (CPU-first prefilter)\n", num_blocks, BLOCK_THREADS);
      // --- Pipelined CPU-GPU execution (Section V-E) ---
      // GPU_BATCH_SIZE tuples per batch; CPU prefilters one batch, then async GPU launch on a stream.
      // Overlap: while GPU processes batch N, CPU prefilters batch N+1.
      const int FASCAL_GPU_BATCH = []() -> int {
          if (const char *e = getenv("FASCAL_GPU_BATCH")) { int v = atoi(e); if (v > 0) return v; }
          return 4194304;  // 4M tuples default (~8K tiles, fills all SMs with minimal launch overhead)
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
              tpch_q22_cpu_prefilter_customer(h_data, num_tuples, h_seed_bitmap_storage,
                  h_tile_summary, h_has_order_bitmap, max_custkey);
              tpch_q22_kernel_customer<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
                  num_tuples,
                  0,  // batch_offset = 0 (whole table)
                  d___scalar_sq_2_avg,
          d_customer_c_acctbal,
          d_customer_c_custkey,
          d_customer_c_phone,
                  d_seed_bitmap,
                  d_tile_summary,
                  d_has_order_bitmap, max_custkey,
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
              tpch_q22_cpu_prefilter_customer(h_data, b_count, h_seed_bitmap_storage,
                  h_tile_summary, h_has_order_bitmap, max_custkey, b_offset, num_tuples);
              // Async GPU kernel launch on stream
              tpch_q22_kernel_customer<BLOCK_THREADS, ITEMS_PER_THREAD><<<b_blocks, BLOCK_THREADS, 0, _stream>>>(
                  num_tuples,
                  b_offset,
                  d___scalar_sq_2_avg,
          d_customer_c_acctbal,
          d_customer_c_custkey,
          d_customer_c_phone,
                  d_seed_bitmap,
                  d_tile_summary,
                  d_has_order_bitmap, max_custkey,
              d_aggtable);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaGetLastError());
          return 0;
      };
  
      cudaEventRecord(t0);
  
      // Build dense bitmap from orders.o_custkey: morsel-batched on fascal_streams[b % N].
      // atomicOr is commutative + idempotent so morsel ordering does not affect bitmap contents.
      tpch_q22_cpu_prefilter___correlated_sq_1(h___correlated_sq_1_o_custkey, num___correlated_sq_1_tuples, h___correlated_sq_1_prefilter_pinned, h___correlated_sq_1_tile_summary);
      {
        int _build_nb = (num___correlated_sq_1_tuples + FASCAL_GPU_BATCH - 1) / FASCAL_GPU_BATCH;
        for (int _b = 0; _b < _build_nb; ++_b) {
          int _bo = _b * FASCAL_GPU_BATCH;
          int _bc = std::min(FASCAL_GPU_BATCH, num___correlated_sq_1_tuples - _bo);
          int _bblks = (_bc + TILE_SIZE - 1) / TILE_SIZE;
          cudaStream_t _bst = 0;
          tpch_q22_kernel___correlated_sq_1<BLOCK_THREADS, ITEMS_PER_THREAD><<<_bblks, BLOCK_THREADS, 0, _bst>>>(
              num___correlated_sq_1_tuples, _bo, d___correlated_sq_1_o_custkey,
              d_has_order_bitmap, max_custkey,
              d___correlated_sq_1_prefilter_mapped, d___correlated_sq_1_tile_summary);
        }
        for (int _s = 0; _s < FASCAL_NUM_STREAMS; ++_s) CUDA_CHECK(cudaStreamSynchronize(fascal_streams[_s]));
      }
      CUDA_CHECK(cudaGetLastError());
      // Copy bitmap to host for CPU AFP antijoin probe in result pipeline.
      CUDA_CHECK(cudaMemcpy(h_has_order_bitmap, d_has_order_bitmap, bitmap_words * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  
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
  CUDA_CHECK(cudaMemset(d_aggtable, 0, (size_t)(num_groups) * NUM_AGGREGATES * sizeof(double)));
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
      double *h_agg_table = (double*)malloc(num_groups * NUM_AGGREGATES * sizeof(double));
      CUDA_CHECK(cudaMemcpy(h_agg_table, d_aggtable, num_groups * NUM_AGGREGATES * sizeof(double), cudaMemcpyDeviceToHost));
      for (size_t k = 0; k < num_groups; k++) {
        int any = ((h_agg_table[k * NUM_AGGREGATES + 0] != 0) || (h_agg_table[k * NUM_AGGREGATES + 1] != 0)) ? 1 : 0;
        if (any) {
          size_t gk = k;
        int key0 = (int)(gk % 100ULL);
        gk /= 100;
          printf("ROW: ");
          printf("%d", key0);
          printf("|%lld", (long long)h_agg_table[k * NUM_AGGREGATES + 0]);
          printf("|%.2f", (double)h_agg_table[k * NUM_AGGREGATES + 1]);
          printf("\n");
        }
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
      odzc_mgr->unregister_column(mc___scalar_sq_2_avg);
      odzc_mgr->unregister_column(mc_customer_c_acctbal);
      odzc_mgr->unregister_column(mc_customer_c_custkey);
      odzc_mgr->unregister_column(mc_customer_c_phone);
  
      odzc_mgr->unregister_column(mc___correlated_sq_1_o_custkey);
      CUDA_CHECK(cudaFreeHost(h___correlated_sq_1_tile_summary));
      CUDA_CHECK(cudaFreeHost(h___correlated_sq_1_prefilter_pinned));
      CUDA_CHECK(cudaFree(d_has_order_bitmap));
      free(h_has_order_bitmap);
      fascal_free_column(h___correlated_sq_1_o_custkey);
  
      fascal_free_seed_bitmap(h_seed_bitmap_storage, h_tile_summary);
      fascal_free_column(h_c_acctbal);
      fascal_free_column(h_c_custkey);
      fascal_free_column(h_c_phone);
      if (h___scalar_sq_2_avg && h___scalar_sq_2_avg != _mat___scalar_sq_2_avg) fascal_free_column(h___scalar_sq_2_avg);
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
    // Cleanup shared pinned subquery materializations
    if (_mat___scalar_sq_2_avg) CUDA_CHECK(cudaFreeHost(_mat___scalar_sq_2_avg));
}
