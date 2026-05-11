/**
 * Shared definitions for FaScalSQL-generated query .cu files.
 * Include this once; link with fascal_generated_common.o for host functions.
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Forward declarations for gpu_hashtable.cuh device functions used here.
// The full definitions are in gpu_hashtable.cuh (included by each generated .cu).
__device__ __forceinline__ uint32_t gpu_bloom_h1(uint32_t k, uint32_t m);
__device__ __forceinline__ uint32_t gpu_bloom_h2(uint32_t k, uint32_t m);
__device__ __forceinline__ uint32_t gpu_bloom_hash_i(int key, uint8_t i, uint32_t m);
__device__ __forceinline__ bool gpu_bloom_query(int key, const uint64_t * __restrict__ bits, uint32_t size_bits, uint8_t num_hashes);

struct HtEntry;
__device__ __forceinline__ bool gpu_ht_probe(uint64_t key, const HtEntry* __restrict__ ht, uint32_t ht_size, int* out_val);
__device__ __forceinline__ bool gpu_ht_probe_direct(int key, const HtEntry* __restrict__ ht, uint32_t ht_size, int key_min, int* out_val);
__device__ __forceinline__ bool gpu_ht_probe_semi(uint64_t key, HtEntry* __restrict__ ht, uint32_t ht_size, int* out_val);
__device__ __forceinline__ bool gpu_ht_probe_exists_ne(uint64_t key, int probe_payload, const HtEntry* __restrict__ ht, uint32_t ht_size);

#define FASCAL_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

/* Alias for generated code compatibility */
#ifndef CUDA_CHECK
#define CUDA_CHECK FASCAL_CUDA_CHECK
#endif

/* AFPDoorbell removed — pipelining uses batch-based CPU prefilter instead of doorbell signaling. */

template<int BLOCK_THREADS>
__device__ __forceinline__ unsigned long long blockReduceSum(unsigned long long val) {
    __shared__ unsigned long long shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    if (wid == 0) {
        val = (lane < (BLOCK_THREADS / 32)) ? shared[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

template<int BLOCK_THREADS>
__device__ __forceinline__ double blockReduceSumDouble(double val) {
    __shared__ double shared_d[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) {
        shared_d[wid] = val;
    }
    __syncthreads();
    if (wid == 0) {
        val = (lane < (BLOCK_THREADS / 32)) ? shared_d[lane] : 0.0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ T BlockSum(T val, T* buffer) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        buffer[threadIdx.x >> 5] = val;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        val = (threadIdx.x < (_BLOCK_THREADS >> 5)) ? buffer[threadIdx.x] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

// Per-thread early-out helper: returns true if ANY flag is set
template<int _ITEMS_PER_THREAD>
__device__ __forceinline__ bool _any_flag_set(const int* flags) {
    int any = 0;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) any |= flags[ITEM];
    return any != 0;
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadSelect(T* src, T* dst, int* flags, int num_items) {
    if (!_any_flag_set<_ITEMS_PER_THREAD>(flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        int local_idx = threadIdx.x + ITEM * _BLOCK_THREADS;
        if (local_idx < num_items && flags[ITEM]) {
            dst[ITEM] = __ldg(&src[local_idx]);
        }
    }
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndGTE(T* vals, T threshold, int* flags, int num_items) {
    if (!_any_flag_set<_ITEMS_PER_THREAD>(flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if (threadIdx.x + ITEM * _BLOCK_THREADS < num_items) {
            flags[ITEM] = flags[ITEM] && (vals[ITEM] >= threshold);
        }
    }
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndLTE(T* vals, T threshold, int* flags, int num_items) {
    if (!_any_flag_set<_ITEMS_PER_THREAD>(flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if (threadIdx.x + ITEM * _BLOCK_THREADS < num_items) {
            flags[ITEM] = flags[ITEM] && (vals[ITEM] <= threshold);
        }
    }
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndLT(T* vals, T threshold, int* flags, int num_items) {
    if (!_any_flag_set<_ITEMS_PER_THREAD>(flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if (threadIdx.x + ITEM * _BLOCK_THREADS < num_items) {
            flags[ITEM] = flags[ITEM] && (vals[ITEM] < threshold);
        }
    }
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndGT(T* vals, T threshold, int* flags, int num_items) {
    if (!_any_flag_set<_ITEMS_PER_THREAD>(flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if (threadIdx.x + ITEM * _BLOCK_THREADS < num_items) {
            flags[ITEM] = flags[ITEM] && (vals[ITEM] > threshold);
        }
    }
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndEQ(T* vals, T value, int* flags, int num_items) {
    if (!_any_flag_set<_ITEMS_PER_THREAD>(flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if (threadIdx.x + ITEM * _BLOCK_THREADS < num_items) {
            flags[ITEM] = flags[ITEM] && (vals[ITEM] == value);
        }
    }
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndNEQ(T* vals, T value, int* flags, int num_items) {
    if (!_any_flag_set<_ITEMS_PER_THREAD>(flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if (threadIdx.x + ITEM * _BLOCK_THREADS < num_items) {
            flags[ITEM] = flags[ITEM] && (vals[ITEM] != value);
        }
    }
}

/** Block-level Bloom filter probe: for each ITEM, if selection_flags[ITEM], probe loc_fk[ITEM];
 *  set selection_flags[ITEM] = 0 if bloom says no. Requires gpu_bloom_query (gpu_hashtable.cuh) in scope. */
template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBloomProbe(
    int* loc_fk,
    int* selection_flags,
    int num_tile_items,
    uint32_t bloom_size_bits,
    const uint64_t* __restrict__ d_bloom,
    int num_hashes = 3)
{
    if (!_any_flag_set<_ITEMS_PER_THREAD>(selection_flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                int fk_key = loc_fk[ITEM];
                if (bloom_size_bits > 0 && !gpu_bloom_query(fk_key, d_bloom, bloom_size_bits, (uint8_t)num_hashes))
                    selection_flags[ITEM] = 0;
            }
        }
    }
}

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD, typename KeyT = int>
__device__ __forceinline__ void BlockJoinProbe(
    KeyT* loc_fk, int* selection_flags, int num_tile_items,
    const HtEntry* ht, uint32_t ht_size)
{
    if (!_any_flag_set<_ITEMS_PER_THREAD>(selection_flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                if (!gpu_ht_probe(loc_fk[ITEM], ht, ht_size, nullptr))
                    selection_flags[ITEM] = 0;
            }
        }
    }
}

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD, typename KeyT = int>
__device__ __forceinline__ void BlockJoinProbeAnti(
    KeyT* loc_fk, int* selection_flags, int num_tile_items,
    const HtEntry* ht, uint32_t ht_size)
{
    if (!_any_flag_set<_ITEMS_PER_THREAD>(selection_flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                if (gpu_ht_probe(loc_fk[ITEM], ht, ht_size, nullptr))
                    selection_flags[ITEM] = 0;
            }
        }
    }
}

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD, typename KeyT = int>
__device__ __forceinline__ void BlockJoinProbePayload(
    KeyT* loc_fk, int* selection_flags, int num_tile_items,
    const HtEntry* ht, uint32_t ht_size, int* out_payload)
{
    if (!_any_flag_set<_ITEMS_PER_THREAD>(selection_flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                int payload;
                if (gpu_ht_probe(loc_fk[ITEM], ht, ht_size, &payload)) {
                    out_payload[ITEM] = payload;
                } else {
                    selection_flags[ITEM] = 0;
                }
            }
        }
    }
}

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD, typename KeyT = int>
__device__ __forceinline__ void BlockJoinProbeSemi(
    KeyT* loc_fk, int* selection_flags, int num_tile_items,
    HtEntry* ht, uint32_t ht_size)
{
    if (!_any_flag_set<_ITEMS_PER_THREAD>(selection_flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                if (!gpu_ht_probe(loc_fk[ITEM], ht, ht_size, nullptr))
                    selection_flags[ITEM] = 0;
            }
        }
    }
}

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD, typename KeyT = int>
__device__ __forceinline__ void BlockJoinProbeSemiPayload(
    KeyT* loc_fk, int* selection_flags, int num_tile_items,
    HtEntry* ht, uint32_t ht_size, int* out_payload)
{
    if (!_any_flag_set<_ITEMS_PER_THREAD>(selection_flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                int payload;
                if (gpu_ht_probe(loc_fk[ITEM], ht, ht_size, &payload)) {
                    out_payload[ITEM] = payload;
                } else {
                    selection_flags[ITEM] = 0;
                }
            }
        }
    }
}

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD, typename KeyT = int>
__device__ __forceinline__ void BlockJoinProbeExistsNE(
    KeyT* loc_fk, int* loc_probe_payload, int* selection_flags, int num_tile_items,
    const HtEntry* ht, uint32_t ht_size)
{
    if (!_any_flag_set<_ITEMS_PER_THREAD>(selection_flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                if (!gpu_ht_probe_exists_ne(loc_fk[ITEM], loc_probe_payload[ITEM], ht, ht_size))
                    selection_flags[ITEM] = 0;
            }
        }
    }
}

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD, typename KeyT = int>
__device__ __forceinline__ void BlockJoinProbeAntiExistsNE(
    KeyT* loc_fk, int* loc_probe_payload, int* selection_flags, int num_tile_items,
    const HtEntry* ht, uint32_t ht_size)
{
    if (!_any_flag_set<_ITEMS_PER_THREAD>(selection_flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                if (gpu_ht_probe_exists_ne(loc_fk[ITEM], loc_probe_payload[ITEM], ht, ht_size))
                    selection_flags[ITEM] = 0;
            }
        }
    }
}

// ---- Direct-address join probe variants (no hashing, slot = fk - key_min) ----

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockJoinProbeDirect(
    int* loc_fk, int* selection_flags, int num_tile_items,
    const HtEntry* ht, uint32_t ht_size, int key_min)
{
    if (!_any_flag_set<_ITEMS_PER_THREAD>(selection_flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                if (!gpu_ht_probe_direct(loc_fk[ITEM], ht, ht_size, key_min, nullptr))
                    selection_flags[ITEM] = 0;
            }
        }
    }
}

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockJoinProbePayloadDirect(
    int* loc_fk, int* selection_flags, int num_tile_items,
    const HtEntry* ht, uint32_t ht_size, int key_min, int* out_payload)
{
    if (!_any_flag_set<_ITEMS_PER_THREAD>(selection_flags)) return;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                int payload;
                if (gpu_ht_probe_direct(loc_fk[ITEM], ht, ht_size, key_min, &payload)) {
                    out_payload[ITEM] = payload;
                } else {
                    selection_flags[ITEM] = 0;
                }
            }
        }
    }
}

/** atomicMin for double — CUDA has no native overload; emulate via atomicCAS. */
__device__ __forceinline__ double atomicMin(double* addr, double val) {
    unsigned long long int* addr_as_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old_ull = *addr_as_ull, assumed;
    do {
        assumed = old_ull;
        double old_val = __longlong_as_double(assumed);
        if (old_val <= val) break;  // current value is already smaller
        old_ull = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old_ull);
    return __longlong_as_double(old_ull);
}

/** atomicMax for double — CUDA has no native overload; emulate via atomicCAS. */
__device__ __forceinline__ double atomicMax(double* addr, double val) {
    unsigned long long int* addr_as_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old_ull = *addr_as_ull, assumed;
    do {
        assumed = old_ull;
        double old_val = __longlong_as_double(assumed);
        if (old_val >= val) break;  // current value is already larger
        old_ull = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old_ull);
    return __longlong_as_double(old_ull);
}

/** Pack two int32 keys into a single uint64 for composite-key hash table lookup. */
__device__ __forceinline__ uint64_t gpu_join_key_pack_2(int key0, int key1) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(key0)) << 32) |
           static_cast<uint32_t>(key1);
}

// ---- Packed bitmap helpers (1 bit/row → 32× less PCIe traffic) ----

/// TILE_SIZE constant for tile-level early exit.
/// Default: BLOCK_THREADS * ITEMS_PER_THREAD = 128 * 4 = 512.
/// Generated code defines FASCAL_TILE_SIZE before including this header.
#ifndef FASCAL_TILE_SIZE
#define FASCAL_TILE_SIZE 512
#endif

/**
 * BlockLoadBitmap: unpack packed uint32_t bitmap → per-thread int flags[].
 * Each thread reads _ITEMS_PER_THREAD bits from the packed bitmap.
 */
template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadBitmap(
    const uint32_t* __restrict__ bitmap, int* flags,
    int tile_offset, int num_tuples) {
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        int idx = tile_offset + threadIdx.x + ITEM * _BLOCK_THREADS;
        flags[ITEM] = (idx < num_tuples) ? ((__ldg(&bitmap[idx >> 5]) >> (idx & 31)) & 1) : 0;
    }
}

/* Host functions: implemented in fascal_generated_common.cu */
int* load_binary_column(const char* filename, int num_tuples);
int* load_binary_column_auto(const char* filename, int* out_count);
unsigned long long* load_binary_column_u64(const char* filename, int num_tuples);
unsigned long long* load_binary_column_auto_u64(const char* filename, int* out_count);
void bitmap_set(uint32_t* bitmap, int index, bool value);
bool bitmap_get(const uint32_t* bitmap, int index);
void SetBitmap(uint32_t* bitmap, int count);

/* Pinned arena for batch column allocation (reduces per-column cudaHostAlloc overhead).
 * Call fascal_arena_init() before loading columns, fascal_arena_destroy() at cleanup.
 * If not initialized, load_binary_column falls back to individual allocations. */
void fascal_arena_init(size_t total_bytes);
void fascal_arena_destroy();

/* Safe column deallocation — handles arena, pinned, managed, and malloc'd memory. */
void fascal_free_column(void* ptr);
