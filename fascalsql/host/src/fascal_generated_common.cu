/**
 * Shared host implementations for FaScalSQL-generated query .cu files.
 * Link this object with generated ssb_*.cu / tpch_*.cu.
 *
 * Memory model: true zero-copy via cudaHostAlloc(cudaHostAllocMapped).
 * GPU accesses host pinned memory directly over PCIe — no UVM page migration,
 * no explicit cudaMemcpy.  ODZC register_column obtains the device pointer
 * via cudaHostGetDevicePointer, which succeeds immediately because the
 * memory was allocated with cudaHostAllocMapped.
 *
 * A batch arena allocator further reduces overhead by making a single
 * cudaHostAlloc call for all columns instead of one per column.
 */
#include "fascal_generated_common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>

namespace {

FILE* open_with_fallbacks(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (fp) {
        return fp;
    }

    const std::string full_path(filename ? filename : "");
    const std::size_t slash = full_path.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string() : full_path.substr(0, slash + 1);
    const std::string base = (slash == std::string::npos) ? full_path : full_path.substr(slash + 1);

    if (base.empty()) {
        return nullptr;
    }

    std::vector<std::string> candidates;
    auto add_candidate = [&](const std::string& candidate) {
        if (candidate.empty() || candidate == base) {
            return;
        }
        for (const auto& seen : candidates) {
            if (seen == candidate) {
                return;
            }
        }
        candidates.push_back(candidate);
    };

    if (base.size() > 8 && base.compare(base.size() - 8, 8, "_enc.bin") == 0) {
        add_candidate(base.substr(0, base.size() - 8) + ".bin");
    }

    static const char* kPhysicalPrefixes[] = {
        "ps_", "lo_", "l_", "o_", "c_", "p_", "s_", "n_", "r_", "d_",
    };
    for (const char* prefix : kPhysicalPrefixes) {
        const std::string needle = std::string("_") + prefix;
        const std::size_t pos = base.find(needle);
        if (pos != std::string::npos) {
            const std::string canonical = base.substr(pos + 1);
            add_candidate(canonical);
            if (canonical.size() > 8 && canonical.compare(canonical.size() - 8, 8, "_enc.bin") == 0) {
                add_candidate(canonical.substr(0, canonical.size() - 8) + ".bin");
            }
        }
    }

    for (const auto& candidate : candidates) {
        const std::string path = dir + candidate;
        fp = fopen(path.c_str(), "rb");
        if (fp) {
            std::fprintf(stderr, "Falling back from %s to %s\n", filename, path.c_str());
            return fp;
        }
    }

    return nullptr;
}

// ============================================================================
// Pinned Arena Allocator — single cudaHostAlloc for all columns
// ============================================================================
// Eliminates per-column cudaHostAlloc overhead (~200ms each).
// Thread-safe: multiple threads can allocate from the arena concurrently.

struct PinnedArena {
    void* base = nullptr;
    size_t capacity = 0;
    size_t offset = 0;
    std::mutex mu;

    // Initialize the arena with cudaHostAlloc(cudaHostAllocMapped).
    // True zero-copy: GPU accesses host pinned memory via PCIe, no page migration.
    bool init(size_t total_bytes) {
        std::lock_guard<std::mutex> lk(mu);
        if (base) return true;
        cudaError_t err = cudaHostAlloc(&base, total_bytes, cudaHostAllocMapped);
        if (err != cudaSuccess) {
            base = nullptr;
            capacity = 0;
            (void)cudaGetLastError();
            return false;
        }
        capacity = total_bytes;
        offset = 0;
        return true;
    }

    // Allocate a slice from the arena, 256-byte aligned.
    // Returns nullptr if the arena is exhausted or not initialized.
    void* alloc(size_t bytes) {
        std::lock_guard<std::mutex> lk(mu);
        if (!base) return nullptr;
        size_t aligned = (bytes + 255) & ~(size_t)255;
        if (offset + aligned > capacity) return nullptr;
        void* ptr = (char*)base + offset;
        offset += aligned;
        return ptr;
    }

    // Check if a pointer was allocated from this arena.
    bool owns(const void* ptr) const {
        if (!base) return false;
        return ptr >= base && ptr < (const char*)base + capacity;
    }

    // Destroy the arena (frees all memory at once via cudaFreeHost).
    void destroy() {
        std::lock_guard<std::mutex> lk(mu);
        if (base) {
            cudaFreeHost(base);
            base = nullptr;
            capacity = 0;
            offset = 0;
        }
    }
};

static PinnedArena g_pinned_arena;

// Allocate zero-copy pinned mapped memory.
// Uses cudaHostAlloc(cudaHostAllocMapped) so the GPU can read host memory
// directly via PCIe without page migration (true zero-copy, not UVM).
// Returns nullptr on failure. Sets *from_arena to true if allocated from arena.
static void* alloc_pinned_mapped(size_t bytes, bool* from_arena) {
    *from_arena = false;
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
    if (err == cudaSuccess) {
        return ptr;
    }
    (void)cudaGetLastError();
    return nullptr;
}

}  // namespace

// ============================================================================
// Public API: Arena management
// ============================================================================

void fascal_arena_init(size_t total_bytes) {
    if (!g_pinned_arena.init(total_bytes)) {
        fprintf(stderr, "Warning: pinned arena init failed for %zu bytes, "
                "falling back to per-column allocation\n", total_bytes);
    }
}

void fascal_arena_destroy() {
    g_pinned_arena.destroy();
}

// ============================================================================
// Column loading — pinned mapped memory (ODZC-compatible, zero redundant copy)
// ============================================================================

int* load_binary_column(const char* filename, int num_tuples) {
    FILE* fp = open_with_fallbacks(filename);
    if (!fp) {
        fprintf(stderr, "Failed to load %s\n", filename);
        return NULL;
    }
    size_t bytes = (size_t)num_tuples * sizeof(int);

    // Strategy: allocate pinned mapped memory so ODZC register_column
    // can get the device pointer directly (fast path) without an extra copy.
    bool from_arena = false;
    int* data = (int*)alloc_pinned_mapped(bytes, &from_arena);
    if (data) {
        size_t nread = fread(data, sizeof(int), (size_t)num_tuples, fp);
        fclose(fp);
        if ((int)nread != num_tuples)
            fprintf(stderr, "Warning: read %zu of %d from %s\n", nread, num_tuples, filename);
        return data;
    }

    // Fallback: cudaHostAlloc(cudaHostAllocMapped) — true zero-copy
    cudaError_t err = cudaHostAlloc((void**)&data, bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        data = (int*)malloc(bytes);
        if (!data) { fclose(fp); return NULL; }
        size_t nread = fread(data, sizeof(int), (size_t)num_tuples, fp);
        fclose(fp);
        if ((int)nread != num_tuples)
            fprintf(stderr, "Warning: read %zu of %d from %s\n", nread, num_tuples, filename);
        return data;
    }
    size_t nread = fread(data, sizeof(int), (size_t)num_tuples, fp);
    fclose(fp);
    if ((int)nread != num_tuples)
        fprintf(stderr, "Warning: read %zu of %d from %s\n", nread, num_tuples, filename);
    return data;
}

int* load_binary_column_auto(const char* filename, int* out_count) {
    FILE* fp = open_with_fallbacks(filename);
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", filename);
        *out_count = 0;
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);
    int count = (int)(fsize / (long)sizeof(int));
    size_t bytes = (size_t)count * sizeof(int);

    bool from_arena = false;
    int* data = (int*)alloc_pinned_mapped(bytes, &from_arena);
    if (data) {
        size_t nread = fread(data, sizeof(int), (size_t)count, fp);
        fclose(fp);
        if ((size_t)nread != (size_t)count) {
            if (!from_arena) cudaFree(data);
            *out_count = 0;
            return NULL;
        }
        *out_count = count;
        return data;
    }

    // Fallback: cudaHostAlloc(cudaHostAllocMapped)
    cudaError_t err = cudaHostAlloc((void**)&data, bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        data = (int*)malloc(bytes);
    }
    if (!data) {
        fclose(fp);
        *out_count = 0;
        return NULL;
    }
    size_t nread = fread(data, sizeof(int), (size_t)count, fp);
    fclose(fp);
    if ((size_t)nread != (size_t)count) {
        cudaFreeHost(data);
        *out_count = 0;
        return NULL;
    }
    *out_count = count;
    return data;
}

unsigned long long* load_binary_column_u64(const char* filename, int num_tuples) {
    FILE* fp = open_with_fallbacks(filename);
    if (!fp) {
        fprintf(stderr, "Failed to load %s\n", filename);
        return NULL;
    }
    size_t bytes = (size_t)num_tuples * sizeof(unsigned long long);

    bool from_arena = false;
    unsigned long long* data = (unsigned long long*)alloc_pinned_mapped(bytes, &from_arena);
    if (data) {
        size_t nread = fread(data, sizeof(unsigned long long), (size_t)num_tuples, fp);
        fclose(fp);
        if ((int)nread != num_tuples)
            fprintf(stderr, "Warning: read %zu of %d from %s\n", nread, num_tuples, filename);
        return data;
    }

    // Fallback: cudaHostAlloc(cudaHostAllocMapped)
    cudaError_t err = cudaHostAlloc((void**)&data, bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        data = (unsigned long long*)malloc(bytes);
    }
    if (!data) { fclose(fp); return NULL; }
    size_t nread = fread(data, sizeof(unsigned long long), (size_t)num_tuples, fp);
    fclose(fp);
    if ((int)nread != num_tuples)
        fprintf(stderr, "Warning: read %zu of %d from %s\n", nread, num_tuples, filename);
    return data;
}

unsigned long long* load_binary_column_auto_u64(const char* filename, int* out_count) {
    FILE* fp = open_with_fallbacks(filename);
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", filename);
        *out_count = 0;
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);
    int count = (int)(fsize / (long)sizeof(unsigned long long));
    size_t bytes = (size_t)count * sizeof(unsigned long long);

    bool from_arena = false;
    unsigned long long* data = (unsigned long long*)alloc_pinned_mapped(bytes, &from_arena);
    if (data) {
        size_t nread = fread(data, sizeof(unsigned long long), (size_t)count, fp);
        fclose(fp);
        if ((size_t)nread != (size_t)count) {
            if (!from_arena) cudaFree(data);
            *out_count = 0;
            return NULL;
        }
        *out_count = count;
        return data;
    }

    // Fallback: cudaHostAlloc(cudaHostAllocMapped)
    cudaError_t err = cudaHostAlloc((void**)&data, bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        data = (unsigned long long*)malloc(bytes);
    }
    if (!data) { fclose(fp); *out_count = 0; return NULL; }
    size_t nread = fread(data, sizeof(unsigned long long), (size_t)count, fp);
    fclose(fp);
    if ((size_t)nread != (size_t)count) {
        cudaFreeHost(data);
        *out_count = 0;
        return NULL;
    }
    *out_count = count;
    return data;
}

// ============================================================================
// Safe free — handles pinned, managed, and malloc'd memory
// ============================================================================

void fascal_free_column(void* ptr) {
    if (!ptr) return;
    // Arena memory: do not free (arena owns the memory, freed by fascal_arena_destroy)
    if (g_pinned_arena.owns(ptr)) return;
    // Try cudaFreeHost first (for individually cudaHostAlloc'd memory)
    cudaError_t err = cudaFreeHost(ptr);
    if (err == cudaSuccess) return;
    (void)cudaGetLastError();  // clear sticky error
    // Try cudaFree (for cudaMallocManaged memory)
    err = cudaFree(ptr);
    if (err == cudaSuccess) return;
    (void)cudaGetLastError();  // clear sticky error
    // Last resort: plain free (for malloc'd memory)
    free(ptr);
}

void bitmap_set(uint32_t* bitmap, int index, bool value) {
    uint32_t word_idx = (uint32_t)index >> 5;
    uint32_t bit_pos  = (uint32_t)index & 31;
    if (value) {
        bitmap[word_idx] |= (1u << bit_pos);
    } else {
        bitmap[word_idx] &= ~(1u << bit_pos);
    }
}

bool bitmap_get(const uint32_t* bitmap, int index) {
    uint32_t word_idx = (uint32_t)index >> 5;
    uint32_t bit_pos  = (uint32_t)index & 31;
    return (bitmap[word_idx] >> bit_pos) & 1;
}

void SetBitmap(uint32_t* bitmap, int count) {
    int full_words = count >> 5;
    int rem = count & 31;
    for (int w = 0; w < full_words; ++w) {
        bitmap[w] = 0xFFFFFFFFu;
    }
    if (rem > 0) {
        bitmap[full_words] = (1u << rem) - 1u;
    }
}
