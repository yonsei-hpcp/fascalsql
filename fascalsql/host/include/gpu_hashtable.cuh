/**
 * FaScalSQL GPU Hash Table
 * Adapted from DogQC hashing.cuh (original_dogqc/dogqc/include/hashing.cuh).
 * Provides build/probe kernels for SSB/TPC-H joins.
 *
 * Key design choices:
 * - Linear probing with atomicCAS for lock-free concurrent build
 * - Table size = next_power_of_2(n * 2) for <50% load factor
 * - For SF=1: allocate in GPU device memory (fast)
 * - For large SF: use cudaHostAllocMapped zero-copy path (see zero_copy_ht)
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define GPU_HT_EMPTY  (-1)   // sentinel for empty payload slot
#define GPU_HT_EMPTY_KEY (~0ULL)  // sentinel for empty key slot
#define GPU_HT_SEMI_USED (-2) // sentinel for already-matched semi-join slot

// ============================================================
// Hash Function (Knuth multiplicative hash)
// ============================================================
__device__ __host__ __forceinline__
uint32_t gpu_ht_hash(uint64_t key, uint32_t table_size) {
    uint64_t mixed = key * 11400714819323198485ull;
    mixed ^= (mixed >> 32);
    return (uint32_t)mixed & (table_size - 1u);
}

// ============================================================
// Hash Table Entry
// key = join key (up to packed 2x int32); val = payload (int32, -1 if no payload)
// ============================================================
struct HtEntry {
    uint64_t key;   // GPU_HT_EMPTY_KEY if slot is unoccupied
    int val;   // payload associated with this key
};

// ============================================================
// next_power_of_2: smallest power of 2 >= x
// ============================================================
__host__ __device__ __forceinline__
uint32_t next_power_of_2(uint32_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16;
    return x + 1;
}

// ============================================================
// Build Kernel: insert (key, val) pairs into hash table
// Skips keys where selection_flags[i] == 0 (dimension-side predicate filter)
// ============================================================
__global__ void gpu_ht_build_kernel(
    const int* __restrict__ keys,
    const int* __restrict__ vals,   // may be NULL if no payload needed
    const int* __restrict__ selection_flags,  // may be NULL (all selected)
    int n,
    HtEntry* ht,
    uint32_t ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Apply dimension-side filter
    if (selection_flags && selection_flags[idx] == 0) return;

    uint64_t key = static_cast<uint32_t>(keys[idx]);
    if (key == GPU_HT_EMPTY_KEY) return;  // skip sentinel keys

    int payload = vals ? vals[idx] : idx;

    uint32_t slot = gpu_ht_hash(key, ht_size);
    for (uint32_t i = 0; i < ht_size; i++) {
        uint32_t curr = (slot + i) & (ht_size - 1u);
        unsigned long long existing =
            atomicCAS(reinterpret_cast<unsigned long long*>(&ht[curr].key), GPU_HT_EMPTY_KEY, key);
        if (existing == GPU_HT_EMPTY_KEY || existing == key) {
            ht[curr].val = payload;
            return;
        }
    }
    // Table full — should not happen with 50% load factor
}

// ============================================================
// GPU Bloom Filter (same hash as CPU bloom_filter.hpp for bit-compatibility)
// Built alongside hash table; used for semi-join before HT probe (CPU + GPU).
// ============================================================
__device__ __forceinline__ uint32_t gpu_bloom_h1(uint32_t k, uint32_t m) {
    return (uint32_t)((uint64_t)(k * 2654435761u) % (uint64_t)m);
}
__device__ __forceinline__ uint32_t gpu_bloom_h2(uint32_t k, uint32_t m) {
    return (uint32_t)((uint64_t)((k ^ (k >> 16)) * 2246822519u) % (uint64_t)m);
}
__device__ __forceinline__ uint32_t gpu_bloom_hash_i(int key, uint8_t i, uint32_t m) {
    uint32_t k = (uint32_t)key;
    return (uint32_t)((uint64_t)gpu_bloom_h1(k, m) + (uint64_t)i * gpu_bloom_h2(k, m)) % (uint64_t)m;
}

// ============================================================
// Device helpers: single-key HT insert and BF set (for morsel build kernel)
// ============================================================
__device__ __forceinline__ void gpu_ht_insert_one(
    uint64_t key, int payload, int selected,
    HtEntry* ht, uint32_t ht_size)
{
    if (!selected || key == GPU_HT_EMPTY_KEY) return;
    uint32_t slot = gpu_ht_hash(key, ht_size);
    for (uint32_t i = 0; i < ht_size; i++) {
        uint32_t curr = (slot + i) & (ht_size - 1u);
        unsigned long long existing =
            atomicCAS(reinterpret_cast<unsigned long long*>(&ht[curr].key), GPU_HT_EMPTY_KEY, key);
        if (existing == GPU_HT_EMPTY_KEY || existing == key) {
            ht[curr].val = payload;
            return;
        }
    }
}

// Duplicate-aware existence index for predicates of the form:
//   EXISTS/NOT EXISTS rows with same key and payload != probe_payload
// Encoding in HtEntry.val:
//   val == -1  -> key claimed, first payload not published yet
//   val >= 0   -> first payload observed for the key
//   val <= -2  -> first payload is (-val - 2) and at least one different payload exists
__device__ __forceinline__ void gpu_ht_insert_exists_ne(
    uint64_t key, int payload, int selected,
    HtEntry* ht, uint32_t ht_size)
{
    if (!selected || key == GPU_HT_EMPTY_KEY) return;
    uint32_t slot = gpu_ht_hash(key, ht_size);
    for (uint32_t i = 0; i < ht_size; i++) {
        uint32_t curr = (slot + i) & (ht_size - 1u);
        unsigned long long existing =
            atomicCAS(reinterpret_cast<unsigned long long*>(&ht[curr].key), GPU_HT_EMPTY_KEY, key);
        if (existing == GPU_HT_EMPTY_KEY) {
            atomicExch(&ht[curr].val, payload);
            return;
        }
        if (existing == key) {
            while (true) {
                int old_val = ht[curr].val;
                if (old_val == GPU_HT_EMPTY) continue;
                if (old_val <= -2) return;
                int first_payload = old_val;
                if (first_payload == payload) return;
                int new_val = -first_payload - 2;
                if (atomicCAS(&ht[curr].val, old_val, new_val) == old_val) return;
            }
        }
    }
}

// Direct-index insert: slot = key - key_min; one writer per slot (dimension PK), no atomic.
__device__ __forceinline__ void gpu_ht_insert_one_direct(
    int key, int payload, int selected,
    int key_min,
    HtEntry* ht, uint32_t ht_size)
{
    if (!selected || key == GPU_HT_EMPTY) return;
    int slot = key - key_min;
    if (slot < 0 || (uint32_t)slot >= ht_size) return;
    ht[slot].key = static_cast<uint32_t>(key);
    ht[slot].val = payload;
}

__device__ __forceinline__ void gpu_bloom_set_one(
    int key, int selected,
    uint64_t* __restrict__ bits, uint32_t size_bits, uint8_t num_hashes)
{
    if (!selected) return;
    for (uint8_t i = 0; i < num_hashes; i++) {
        uint32_t p = gpu_bloom_hash_i(key, i, size_bits);
        uint64_t mask = 1ULL << (p % 64);
        atomicOr((unsigned long long*)(bits + p / 64), mask);
    }
}

__global__ void gpu_bloom_build_kernel(
    const int* __restrict__ keys,
    const int* __restrict__ selection_flags,  // may be NULL
    int n,
    uint64_t* __restrict__ bits,
    uint32_t size_bits,
    uint8_t num_hashes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (selection_flags && selection_flags[idx] == 0) return;
    int key = keys[idx];
    for (uint8_t i = 0; i < num_hashes; i++) {
        uint32_t p = gpu_bloom_hash_i(key, i, size_bits);
        uint64_t mask = 1ULL << (p % 64);
        atomicOr((unsigned long long*)(bits + p / 64), mask);
    }
}

__device__ __forceinline__ bool gpu_bloom_query(
    int key,
    const uint64_t* __restrict__ bits,
    uint32_t size_bits,
    uint8_t num_hashes)
{
    for (uint8_t i = 0; i < num_hashes; i++) {
        uint32_t p = gpu_bloom_hash_i(key, i, size_bits);
        if (!((bits[p / 64] >> (p % 64)) & 1ULL)) return false;
    }
    return true;
}

// ============================================================
// Probe Kernel: for each probe key, lookup in hash table
// Sets selection_flags[i] = 0 if key not found (inner join semantics)
// ============================================================
// ============================================================
// Single-key probe (device function for use inside kernels)
// Returns true if key is found; when out_val != nullptr, writes payload to *out_val.
// ============================================================
__device__ __forceinline__ bool gpu_ht_probe(
    uint64_t key,
    const HtEntry* __restrict__ ht,
    uint32_t ht_size,
    int* out_val)
{
    uint32_t slot = gpu_ht_hash(key, ht_size);
    for (uint32_t i = 0; i < ht_size; i++) {
        uint32_t curr = (slot + i) & (ht_size - 1u);
        uint64_t existing = ht[curr].key;
        if (existing == GPU_HT_EMPTY_KEY) return false;
        if (existing == key) {
            if (out_val) *out_val = ht[curr].val;
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool gpu_ht_probe_exists_ne(
    uint64_t key,
    int probe_payload,
    const HtEntry* __restrict__ ht,
    uint32_t ht_size)
{
    uint32_t slot = gpu_ht_hash(key, ht_size);
    for (uint32_t i = 0; i < ht_size; i++) {
        uint32_t curr = (slot + i) & (ht_size - 1u);
        uint64_t existing = ht[curr].key;
        if (existing == GPU_HT_EMPTY_KEY) return false;
        if (existing == key) {
            int stored = ht[curr].val;
            if (stored == GPU_HT_EMPTY) continue;
            int first_payload = stored >= 0 ? stored : (-stored - 2);
            return stored <= -2 || first_payload != probe_payload;
        }
    }
    return false;
}

// ============================================================
// Semi-join probe (device function for use inside kernels)
// Returns true ONLY the FIRST time a key is matched. Uses atomicExch
// to mark the slot as used.
// ============================================================
__device__ __forceinline__ bool gpu_ht_probe_semi(
    uint64_t key,
    HtEntry* __restrict__ ht,
    uint32_t ht_size,
    int* out_val)
{
    uint32_t slot = gpu_ht_hash(key, ht_size);
    for (uint32_t i = 0; i < ht_size; i++) {
        uint32_t curr = (slot + i) & (ht_size - 1u);
        uint64_t existing = ht[curr].key;
        if (existing == GPU_HT_EMPTY_KEY) return false;
        if (existing == key) {
            // Atomically swap payload to GPU_HT_SEMI_USED
            int old_val = atomicExch(&ht[curr].val, GPU_HT_SEMI_USED);
            if (old_val != GPU_HT_SEMI_USED) {
                if (out_val) *out_val = old_val;
                return true;
            }
            return false;
        }
    }
    return false;
}

// ============================================================
// Direct-index probe: dimension size = HT size, slot = key - key_min.
// No hash collisions; use when dimension row count is fixed and PK is dense (e.g. 1..N).
// ============================================================
__device__ __forceinline__ bool gpu_ht_probe_direct(
    int key,
    const HtEntry* __restrict__ ht,
    uint32_t ht_size,
    int key_min,
    int* out_val)
{
    int slot = key - key_min;
    if (slot < 0 || (uint32_t)slot >= ht_size) return false;
    uint64_t existing = ht[slot].key;
    if (existing == GPU_HT_EMPTY_KEY || existing != static_cast<uint32_t>(key)) return false;
    if (out_val) *out_val = ht[slot].val;
    return true;
}

__global__ void gpu_ht_probe_kernel(
    const int* __restrict__ probe_keys,
    int n,
    const HtEntry* __restrict__ ht,
    uint32_t ht_size,
    int* selection_flags)   // in/out: AND with existing flags
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (selection_flags[idx] == 0) return;  // already filtered out

    int key = probe_keys[idx];
    if (!gpu_ht_probe(key, ht, ht_size, nullptr))
        selection_flags[idx] = 0;
}

// ============================================================
// Probe Kernel (semi-join): set flag=0 if key not found OR already matched
// ============================================================
__global__ void gpu_ht_semijoin_probe_kernel(
    const int* __restrict__ probe_keys,
    int n,
    HtEntry* __restrict__ ht,
    uint32_t ht_size,
    int* selection_flags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (selection_flags[idx] == 0) return;

    int key = probe_keys[idx];
    if (!gpu_ht_probe_semi(key, ht, ht_size, nullptr))
        selection_flags[idx] = 0;
}

// ============================================================
// Probe Kernel (anti-join): set flag=0 if key IS found
// ============================================================
__global__ void gpu_ht_antijoin_probe_kernel(
    const int* __restrict__ probe_keys,
    int n,
    const HtEntry* __restrict__ ht,
    uint32_t ht_size,
    int* selection_flags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (selection_flags[idx] == 0) return;

    int key = probe_keys[idx];
    if (gpu_ht_probe(key, ht, ht_size, nullptr))
        selection_flags[idx] = 0;
}

// ============================================================
// Host-side GPU hash table manager
// ============================================================
struct GpuHashTable {
    HtEntry*  d_entries = nullptr;   // device memory
    HtEntry*  h_entries = nullptr;   // host memory when zero_copy=true
    uint32_t  ht_size   = 0;         // number of slots (power of 2)
    bool      zero_copy = false;     // true if entries in mapped host memory

    /// Allocate hash table for n key-value pairs (power-of-2, ~50% load factor).
    /// Uses device memory for small tables; zero-copy for large ones.
    static GpuHashTable allocate(int n, bool force_zero_copy = false) {
        GpuHashTable ht;
        ht.ht_size = next_power_of_2((uint32_t)(n * 2));
        size_t bytes = (size_t)ht.ht_size * sizeof(HtEntry);

        // Threshold for "keep on device vs zero-copy fallback".  Small dim
        // HTs always go on device; very large ones fall back to mapped host
        // memory.  Override via FASCAL_HT_DEVICE_MAX_MB (default 8 GB so that
        // SF=100 part / orders / partsupp HTs (each ≲ 1 GB) stay on device).
        size_t fascal_ht_device_max = []() -> size_t {
            const char *e = getenv("FASCAL_HT_DEVICE_MAX_MB");
            if (e) { long v = atol(e); if (v > 0) return (size_t)v * 1024ull * 1024ull; }
            return 8ull * 1024ull * 1024ull * 1024ull;
        }();
        if (!force_zero_copy && bytes <= fascal_ht_device_max) {
            // Small table: device memory (faster access)
            cudaError_t err = cudaMalloc(&ht.d_entries, bytes);
            if (err != cudaSuccess) {
                fprintf(stderr, "GpuHashTable::allocate: cudaMalloc failed (%s), falling back to zero-copy\n",
                        cudaGetErrorString(err));
                force_zero_copy = true;
            } else {
                ht.zero_copy = false;
            }
        }

        if (force_zero_copy || ht.d_entries == nullptr) {
            // Large table: zero-copy mapped host memory
            cudaError_t err = cudaHostAlloc((void**)&ht.h_entries, bytes, cudaHostAllocMapped);
            if (err != cudaSuccess) {
                fprintf(stderr, "GpuHashTable::allocate: cudaHostAlloc failed (%s)\n",
                        cudaGetErrorString(err));
                return ht;
            }
            err = cudaHostGetDevicePointer((void**)&ht.d_entries, ht.h_entries, 0);
            if (err != cudaSuccess) {
                fprintf(stderr, "GpuHashTable::allocate: cudaHostGetDevicePointer failed (%s)\n",
                        cudaGetErrorString(err));
                cudaFreeHost(ht.h_entries);
                ht.h_entries = nullptr;
                ht.d_entries = nullptr;
                return ht;
            }
            ht.zero_copy = true;
        }

        // Initialize all slots to empty (key = GPU_HT_EMPTY = -1)
        if (ht.d_entries) {
            if (ht.zero_copy && ht.h_entries) {
                memset(ht.h_entries, 0xFF, bytes);
            } else {
                cudaMemset(ht.d_entries, 0xFF, bytes);  // 0xFF fills int to -1
            }
        }
        return ht;
    }

    /// Allocate hash table with exactly n slots (direct-index mode: no hash, slot = key - key_min).
    /// Use with gpu_ht_probe_direct / gpu_ht_insert_one_direct to avoid collisions when dimension size is fixed.
    static GpuHashTable allocate_direct(int n, bool force_zero_copy = false) {
        GpuHashTable ht;
        ht.ht_size = (n <= 0) ? 1u : (uint32_t)n;
        size_t bytes = (size_t)ht.ht_size * sizeof(HtEntry);

        // Threshold for "keep on device vs zero-copy fallback".  Small dim
        // HTs always go on device; very large ones fall back to mapped host
        // memory.  Override via FASCAL_HT_DEVICE_MAX_MB (default 8 GB so that
        // SF=100 part / orders / partsupp HTs (each ≲ 1 GB) stay on device).
        size_t fascal_ht_device_max = []() -> size_t {
            const char *e = getenv("FASCAL_HT_DEVICE_MAX_MB");
            if (e) { long v = atol(e); if (v > 0) return (size_t)v * 1024ull * 1024ull; }
            return 8ull * 1024ull * 1024ull * 1024ull;
        }();
        if (!force_zero_copy && bytes <= fascal_ht_device_max) {
            cudaError_t err = cudaMalloc(&ht.d_entries, bytes);
            if (err != cudaSuccess) {
                fprintf(stderr, "GpuHashTable::allocate_direct: cudaMalloc failed (%s), falling back to zero-copy\n",
                        cudaGetErrorString(err));
                force_zero_copy = true;
            } else {
                ht.zero_copy = false;
            }
        }

        if (force_zero_copy || ht.d_entries == nullptr) {
            cudaError_t err = cudaHostAlloc((void**)&ht.h_entries, bytes, cudaHostAllocMapped);
            if (err != cudaSuccess) {
                fprintf(stderr, "GpuHashTable::allocate_direct: cudaHostAlloc failed (%s)\n",
                        cudaGetErrorString(err));
                return ht;
            }
            err = cudaHostGetDevicePointer((void**)&ht.d_entries, ht.h_entries, 0);
            if (err != cudaSuccess) {
                fprintf(stderr, "GpuHashTable::allocate_direct: cudaHostGetDevicePointer failed (%s)\n",
                        cudaGetErrorString(err));
                cudaFreeHost(ht.h_entries);
                ht.h_entries = nullptr;
                ht.d_entries = nullptr;
                return ht;
            }
            ht.zero_copy = true;
        }

        if (ht.d_entries) {
            if (ht.zero_copy && ht.h_entries) {
                memset(ht.h_entries, 0xFF, bytes);
            } else {
                cudaMemset(ht.d_entries, 0xFF, bytes);
            }
        }
        return ht;
    }

    void build(const int* d_keys, const int* d_vals, const int* d_flags, int n, cudaStream_t s = 0) {
        if (!d_entries || n == 0) return;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        gpu_ht_build_kernel<<<blocks, threads, 0, s>>>(d_keys, d_vals, d_flags, n, d_entries, ht_size);
    }

    void probe(const int* d_probe_keys, int n, int* d_sel_flags, cudaStream_t s = 0) const {
        if (!d_entries || n == 0) return;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        gpu_ht_probe_kernel<<<blocks, threads, 0, s>>>(d_probe_keys, n, d_entries, ht_size, d_sel_flags);
    }

    void antijoin_probe(const int* d_probe_keys, int n, int* d_sel_flags, cudaStream_t s = 0) const {
        if (!d_entries || n == 0) return;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        gpu_ht_antijoin_probe_kernel<<<blocks, threads, 0, s>>>(d_probe_keys, n, d_entries, ht_size, d_sel_flags);
    }

    void semijoin_probe(const int* d_probe_keys, int n, int* d_sel_flags, cudaStream_t s = 0) {
        if (!d_entries || n == 0) return;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        // Non-const pointer required for atomicExch in semijoin
        gpu_ht_semijoin_probe_kernel<<<blocks, threads, 0, s>>>(d_probe_keys, n, d_entries, ht_size, d_sel_flags);
    }

    void free_table() {
        if (!d_entries) return;
        if (zero_copy) {
            if (h_entries) {
                cudaFreeHost(h_entries);
            }
        } else {
            cudaFree(d_entries);
        }
        d_entries = nullptr;
        h_entries = nullptr;
    }
};

// ============================================================
// GPU Bloom Filter (built with hash table; copy to host for CPU semi-join)
// ============================================================
struct GpuBloomFilter {
    uint64_t* d_bits    = nullptr;
    uint32_t  size_bits = 0;
    uint8_t   num_hashes = 3;

    static GpuBloomFilter allocate(int n, uint8_t k = 3) {
        GpuBloomFilter bf;
        bf.num_hashes = k;
        bf.size_bits = (uint32_t)n * 8u;  // 8 bits/key → ~2% FPR with k=3
        if (bf.size_bits < 64) bf.size_bits = 64;
        if (bf.size_bits == 0) bf.size_bits = 64;
        size_t n_uint64 = (bf.size_bits + 63) / 64;
        size_t bytes = n_uint64 * sizeof(uint64_t);
        cudaError_t err = cudaMalloc(&bf.d_bits, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "GpuBloomFilter::allocate failed: %s\n", cudaGetErrorString(err));
            return bf;
        }
        cudaMemset(bf.d_bits, 0, bytes);
        return bf;
    }

    void build(const int* d_keys, const int* d_flags, int n, cudaStream_t s = 0) {
        if (!d_bits || n == 0) return;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        gpu_bloom_build_kernel<<<blocks, threads, 0, s>>>(d_keys, d_flags, n, d_bits, size_bits, num_hashes);
    }

    /** Copy device bits to host BloomFilter (host bf must have same size_bits/num_hashes and pre-allocated bits). */
    void copy_to_host(void* h_bits, size_t bytes) const {
        if (!d_bits || !h_bits) return;
        size_t n_uint64 = (size_bits + 63) / 64;
        size_t sz = n_uint64 * sizeof(uint64_t);
        if (bytes < sz) sz = bytes;
        cudaMemcpy(h_bits, d_bits, sz, cudaMemcpyDeviceToHost);
    }

    void free_filter() {
        if (d_bits) { cudaFree(d_bits); d_bits = nullptr; }
        size_bits = 0;
    }
};

// ============================================================
// DEPRECATED: run_build_sub_pipeline — single pattern for all sub-pipelines.
// Generated code uses per-stage kernels ({query}_kernel_{table}) with the same ops as main
// (selection, multiple HT probes, optional aggregation, then HT+BF build). Use those instead.
// If out_bloom != NULL, builds Bloom on GPU from same keys; caller can copy_to_host for CPU semi-join.
// ============================================================
inline void run_build_sub_pipeline(
    GpuHashTable& ht,
    const int* d_keys,
    const int* d_vals,         // may be NULL
    const int* d_filter_flags, // may be NULL (all selected)
    int n_dim_tuples,
    cudaStream_t stream = 0,
    GpuBloomFilter* out_bloom = nullptr)
{
    ht.build(d_keys, d_vals, d_filter_flags, n_dim_tuples, stream);
    if (out_bloom && out_bloom->d_bits) {
        out_bloom->build(d_keys, d_filter_flags, n_dim_tuples, stream);
    }
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "run_build_sub_pipeline launch failed: %s\n", cudaGetErrorString(launch_err));
        exit(1);
    }
    cudaError_t sync_err = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "run_build_sub_pipeline sync failed: %s\n", cudaGetErrorString(sync_err));
        exit(1);
    }
}

// ============================================================================
// Shared-memory HT probe for small dimension tables
// ============================================================================
// For HTs that fit in shared memory (~48KB per SM = 4096 HtEntry slots),
// loading the HT into shared memory once per block avoids repeated L2 cache
// reads. Use for dimension tables with < 2048 entries (direct-index or hashed).
// SMEM_HT_SLOTS must be a compile-time constant.

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD, int SMEM_HT_SLOTS>
__device__ __forceinline__ void BlockJoinProbeDirectShmem(
    int* loc_fk, int* selection_flags, int num_tile_items,
    const HtEntry* __restrict__ ht_global, uint32_t ht_size, int key_min)
{
    // Each HtEntry is 12 bytes. 4096 entries = 48KB = typical smem limit.
    __shared__ HtEntry smem_ht[SMEM_HT_SLOTS];

    // Cooperatively load the HT into shared memory
    for (int i = threadIdx.x; i < SMEM_HT_SLOTS && i < (int)ht_size; i += _BLOCK_THREADS) {
        smem_ht[i] = ht_global[i];
    }
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                int slot = loc_fk[ITEM] - key_min;
                if (slot < 0 || (uint32_t)slot >= ht_size || (uint32_t)slot >= SMEM_HT_SLOTS) {
                    selection_flags[ITEM] = 0;
                } else {
                    uint64_t existing = smem_ht[slot].key;
                    if (existing == GPU_HT_EMPTY_KEY ||
                        existing != static_cast<uint64_t>(static_cast<uint32_t>(loc_fk[ITEM]))) {
                        selection_flags[ITEM] = 0;
                    }
                }
            }
        }
    }
}

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD, int SMEM_HT_SLOTS>
__device__ __forceinline__ void BlockJoinProbePayloadDirectShmem(
    int* loc_fk, int* selection_flags, int num_tile_items,
    const HtEntry* __restrict__ ht_global, uint32_t ht_size, int key_min,
    int* out_payload)
{
    __shared__ HtEntry smem_ht[SMEM_HT_SLOTS];

    for (int i = threadIdx.x; i < SMEM_HT_SLOTS && i < (int)ht_size; i += _BLOCK_THREADS) {
        smem_ht[i] = ht_global[i];
    }
    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (_BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                int slot = loc_fk[ITEM] - key_min;
                if (slot < 0 || (uint32_t)slot >= ht_size || (uint32_t)slot >= SMEM_HT_SLOTS) {
                    selection_flags[ITEM] = 0;
                } else {
                    uint64_t existing = smem_ht[slot].key;
                    if (existing == GPU_HT_EMPTY_KEY ||
                        existing != static_cast<uint64_t>(static_cast<uint32_t>(loc_fk[ITEM]))) {
                        selection_flags[ITEM] = 0;
                    } else {
                        out_payload[ITEM] = smem_ht[slot].val;
                    }
                }
            }
        }
    }
}
