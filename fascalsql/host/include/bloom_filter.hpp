/**
 * FaScalSQL AFP Bloom Filter
 * Cache-line aligned blocked Bloom filter for CPU-side join key filtering.
 * Reduces GPU data touch by eliminating probe-side FKs that can't match
 * any dimension row — before GPU sees them.
 *
 * Parameters (default): size_bits = 8*n, num_hashes = 3 → ~5% false-positive rate
 * Formula: fp_rate = (1 - e^{-k*n/m})^k  where k=num_hashes, m=size_bits, n=keys
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace fascal {
namespace runtime {

struct BloomFilter {
    uint64_t* bits     = nullptr;  // bit array (cache-line aligned)
    uint32_t  size_bits = 0;       // total bits in the filter
    uint8_t   num_hashes = 3;      // number of hash functions
    bool      owns_bits = false;   // whether we own the bits allocation

    // ── Hash functions (double hashing: h_i(k) = h1(k) + i*h2(k)) ──
    static uint32_t _h1(uint32_t key, uint32_t m) {
        return (key * 2654435761u) % m;
    }
    static uint32_t _h2(uint32_t key, uint32_t m) {
        return ((key ^ (key >> 16)) * 2246822519u) % m;
    }
    static uint32_t hash_i(int key, uint8_t i, uint32_t m) {
        uint32_t k = (uint32_t)key;
        return (_h1(k, m) + (uint32_t)i * _h2(k, m)) % m;
    }

    /// Allocate a new bloom filter sized for n keys.
    /// If max_bytes > 0, cap the BF at max_bytes (e.g., LLC size) to avoid cache thrashing.
    static BloomFilter allocate(int n, uint8_t k = 3, size_t max_bytes = 0) {
        BloomFilter bf;
        bf.num_hashes = k;
        bf.size_bits = (uint32_t)n * 8u;  // 8 bits/key → ~2% FPR with k=3
        // Cap at max_bytes to ensure BF fits in L3 cache
        if (max_bytes > 0) {
            uint32_t max_bits = (uint32_t)std::min((size_t)UINT32_MAX, max_bytes * 8);
            if (bf.size_bits > max_bits) bf.size_bits = max_bits;
        }
        if (bf.size_bits < 64) bf.size_bits = 64;
        if (bf.size_bits == 0) bf.size_bits = 64;
        size_t n_uint64 = (bf.size_bits + 63) / 64;
        // Allocate aligned to 64 bytes (cache line)
        bf.bits = (uint64_t*)aligned_alloc(64, n_uint64 * sizeof(uint64_t));
        if (!bf.bits) {
            bf.bits = (uint64_t*)malloc(n_uint64 * sizeof(uint64_t));
        }
        memset(bf.bits, 0, n_uint64 * sizeof(uint64_t));
        bf.owns_bits = true;
        return bf;
    }

    /// Free filter memory.
    void free_filter() {
        if (owns_bits && bits) {
            free(bits);
            bits = nullptr;
        }
    }

    /// Set a bit at position p.
    inline void set_bit(uint32_t p) {
        bits[p / 64] |= (1ULL << (p % 64));
    }

    /// Test a bit at position p.
    inline bool test_bit(uint32_t p) const {
        return (bits[p / 64] >> (p % 64)) & 1ULL;
    }

    /// Insert key into the filter.
    inline void insert(int key) {
        for (uint8_t i = 0; i < num_hashes; i++) {
            set_bit(hash_i(key, i, size_bits));
        }
    }

    /// Query whether key is possibly in the filter (may have false positives).
    inline bool query(int key) const {
        for (uint8_t i = 0; i < num_hashes; i++) {
            if (!test_bit(hash_i(key, i, size_bits))) return false;
        }
        return true;
    }

    /// Build filter from array of keys.
    void build(const int* keys, int n) {
        for (int i = 0; i < n; i++) {
            insert(keys[i]);
        }
    }

    /// Probe an array of keys, writing 1/0 into bitmap (AND with existing).
    /// bitmap[i] &= query(probe_keys[i])
    void probe_and_mask(const int* probe_keys, int* bitmap, int n) const {
        for (int i = 0; i < n; i++) {
            if (bitmap[i]) {
                bitmap[i] = query(probe_keys[i]) ? 1 : 0;
            }
        }
    }

    /// Packed-bitmap variant: AND bloom probe results into packed uint32_t bitmap.
    /// offset is the global row index of probe_keys[0].
    void probe_and_mask_packed(const int* probe_keys, uint32_t* bitmap, int offset, int n) const {
        for (int i = 0; i < n; i++) {
            uint32_t gidx = (uint32_t)(offset + i);
            uint32_t word = gidx >> 5;
            uint32_t bit  = gidx & 31;
            if ((bitmap[word] >> bit) & 1) {
                if (!query(probe_keys[i])) {
                    bitmap[word] &= ~(1u << bit);
                }
            }
        }
    }

    /// Compute theoretical false positive rate for n keys inserted.
    double false_positive_rate(int n) const {
        if (size_bits == 0 || n == 0) return 1.0;
        double k = num_hashes;
        double m = size_bits;
        double exponent = -k * n / m;
        return std::pow(1.0 - std::exp(exponent), k);
    }
};

// ============================================================
// Helper functions
// ============================================================
namespace bloom {

/// Build a Bloom filter from integer keys.
inline BloomFilter build_filter(const int* keys, int n, uint8_t num_hashes = 3) {
    BloomFilter bf = BloomFilter::allocate(n, num_hashes);
    bf.build(keys, n);
    return bf;
}

/// Probe a set of keys, updating selection bitmap.
inline void probe_filter(const BloomFilter& bf, const int* probe_keys,
                          int* bitmap, int n) {
    bf.probe_and_mask(probe_keys, bitmap, n);
}

} // namespace bloom

} // namespace runtime
} // namespace fascal
