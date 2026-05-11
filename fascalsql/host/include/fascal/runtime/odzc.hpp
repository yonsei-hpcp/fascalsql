#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <functional>
#include <mutex>

using namespace std;

namespace fascal {
namespace runtime {

/// On-Demand Zero-Copy (ODZC) Memory Manager.
/// Provides zero-copy memory access with on-demand page migration.
/// This allows GPU kernels to access host memory without explicit transfers,
/// with pages being migrated on first access (demand paging).
class ODZCManager {
 public:
  /// Memory access hint for optimization.
  enum class AccessHint {
    READ_ONLY,      // Host memory, GPU reads only
    WRITE_ONLY,     // GPU writes, host reads
    READ_WRITE,     // Bidirectional access
    PREFER_HOST,    // Prefer keeping data on host
    PREFER_DEVICE   // Prefer keeping data on device
  };

  /// Memory region descriptor.
  struct MemoryRegion {
    void* host_ptr;        // Host memory pointer
    void* device_ptr;      // Device memory pointer (if unified)
    size_t size;           // Size in bytes
    AccessHint hint;       // Access pattern hint
    bool is_pinned;        // Whether host memory is pinned
    bool is_mapped;        // Whether memory is mapped to GPU
    int device_id;         // GPU device ID
  };

  /// Statistics for ODZC operations.
  struct Stats {
    size_t total_mapped_bytes;
    size_t total_page_faults;
    size_t total_prefetch_bytes;
    size_t total_evict_bytes;
    double avg_migration_time_ms;
  };

  /// Column buffer for zero-copy GPU access.
  /// GPU reads directly from host pinned memory; NO cudaMemPrefetchAsync.
  struct MappedColumn {
    void* host_ptr = nullptr;    // Host pointer (caller's buffer or our cudaHostAlloc)
    void* device_ptr = nullptr;  // Mapped device pointer (same physical memory if zero-copy)
    size_t size = 0;             // Size in bytes
    bool is_zero_copy = true;    // true if GPU reads via PCIe zero-copy
    bool owned_host_alloc = false;  // true if host_ptr was allocated by us (cudaHostAlloc) → free with cudaFreeHost
    bool is_pinned_alloc = false;   // true if host_ptr is cudaHostAlloc'd (pinned, not registered) → no cudaHostUnregister
  };

  ODZCManager();
  explicit ODZCManager(int device_id);
  ~ODZCManager();

  /// Register host memory for ODZC access.
  /// @param host_ptr Host memory pointer
  /// @param size Size in bytes
  /// @param hint Access pattern hint
  /// @return Memory region descriptor
  MemoryRegion register_memory(void* host_ptr, size_t size, 
                               AccessHint hint = AccessHint::READ_ONLY);

  /// Unregister memory region.
  /// @param region Memory region to unregister
  void unregister_memory(MemoryRegion& region);

  /// Pin host memory for faster GPU access.
  /// @param host_ptr Host memory pointer
  /// @param size Size in bytes
  /// @return true if successful
  bool pin_memory(void* host_ptr, size_t size);

  /// Unpin host memory.
  /// @param host_ptr Host memory pointer
  /// @param size Size in bytes
  void unpin_memory(void* host_ptr, size_t size);

  /// Map memory region to GPU address space.
  /// @param region Memory region to map
  /// @return Device pointer (may be same as host for unified memory)
  void* map_to_gpu(MemoryRegion& region);

  /// Unmap memory region from GPU.
  /// @param region Memory region to unmap
  void unmap_from_gpu(MemoryRegion& region);

  /// Prefetch memory to GPU.
  /// @param region Memory region to prefetch
  /// @param stream CUDA stream for async operation
  void prefetch_to_device(MemoryRegion& region, cudaStream_t stream = 0);

  /// Prefetch memory to host.
  /// @param region Memory region to prefetch
  /// @param stream CUDA stream for async operation
  void prefetch_to_host(MemoryRegion& region, cudaStream_t stream = 0);

  /// Advise on memory usage pattern.
  /// @param region Memory region
  /// @param advice Memory access advice
  void advise(MemoryRegion& region, cudaMemoryAdvise advice);

  /// Get device pointer for a registered region.
  /// @param host_ptr Host pointer
  /// @return Device pointer or nullptr if not registered
  void* get_device_ptr(void* host_ptr);

  /// Check if pointer is in registered region.
  /// @param ptr Pointer to check
  /// @return true if pointer is in a registered region
  bool is_registered(void* ptr) const;

  /// Get statistics.
  /// @return Current statistics
  Stats get_stats() const;

  /// Reset statistics.
  void reset_stats();

  /// Set the GPU device ID.
  /// @param device_id CUDA device ID
  void set_device(int device_id);

  /// Get current GPU device ID.
  /// @return CUDA device ID
  int get_device() const;

  /// Check if unified memory is supported.
  /// @return true if unified memory is available
  bool is_unified_memory_supported() const;

  /// Register a column for zero-copy GPU access.
  /// Does NOT prefetch — GPU reads directly from pinned host memory.
  /// @param host_col Host memory pointer to column data
  /// @param bytes Size in bytes
  /// @return MappedColumn descriptor with device pointer ready for GPU use
  MappedColumn register_column(void* host_col, size_t bytes);

  /// Unregister a column, releasing pinned memory registration.
  void unregister_column(MappedColumn& mc);

 private:
  int device_id_;
  bool unified_memory_supported_;
  mutable mutex mutex_;
  unordered_map<void*, MemoryRegion> regions_;
  Stats stats_;

  /// Initialize CUDA context and check capabilities.
  void initialize();

  /// Find region containing pointer.
  /// @param ptr Pointer to find
  /// @return Iterator to region or end()
  unordered_map<void*, MemoryRegion>::iterator find_region(void* ptr);
};

/// RAII wrapper for ODZC memory registration.
class ODZCScope {
 public:
  ODZCScope(ODZCManager& manager, void* host_ptr, size_t size,
            ODZCManager::AccessHint hint = ODZCManager::AccessHint::READ_ONLY);
  ~ODZCScope();

  /// Get device pointer.
  /// @return Device pointer for GPU access
  void* device_ptr() const { return region_.device_ptr; }

  /// Get host pointer.
  /// @return Original host pointer
  void* host_ptr() const { return region_.host_ptr; }

  /// Get size.
  /// @return Size in bytes
  size_t size() const { return region_.size; }

  /// Get memory region.
  /// @return Reference to memory region descriptor
  const ODZCManager::MemoryRegion& region() const { return region_; }

  /// Prefetch to device.
  /// @param stream CUDA stream
  void prefetch_to_device(cudaStream_t stream = 0);

  /// Prefetch to host.
  /// @param stream CUDA stream
  void prefetch_to_host(cudaStream_t stream = 0);

  // Non-copyable
  ODZCScope(const ODZCScope&) = delete;
  ODZCScope& operator=(const ODZCScope&) = delete;

 private:
  ODZCManager& manager_;
  ODZCManager::MemoryRegion region_;
};

/// ODZC-enabled column accessor for query execution.
template<typename T>
class ODZCColumn {
 public:
  ODZCColumn(ODZCManager& manager, T* host_data, size_t num_elements,
             ODZCManager::AccessHint hint = ODZCManager::AccessHint::READ_ONLY);
  ~ODZCColumn() = default;

  /// Get device pointer for GPU kernel access.
  /// @return Device pointer
  T* device_ptr() { return device_ptr_; }

  /// Get const device pointer.
  /// @return Const device pointer
  const T* device_ptr() const { return device_ptr_; }

  /// Get host pointer.
  /// @return Host pointer
  T* host_ptr() { return host_ptr_; }

  /// Get number of elements.
  /// @return Number of elements
  size_t size() const { return num_elements_; }

  /// Prefetch to device.
  /// @param stream CUDA stream
  void prefetch_to_device(cudaStream_t stream = 0);

 private:
  ODZCManager* manager_;
  T* host_ptr_;
  T* device_ptr_;
  size_t num_elements_;
  ODZCManager::MemoryRegion region_;
};

/// Helper functions for ODZC usage.
namespace odzc {

/// Create ODZC column with automatic type deduction.
template<typename T>
ODZCColumn<T> make_column(ODZCManager& manager, T* host_data, size_t num_elements,
                          ODZCManager::AccessHint hint = ODZCManager::AccessHint::READ_ONLY) {
  return ODZCColumn<T>(manager, host_data, num_elements, hint);
}

/// Batch prefetch multiple regions to device.
/// @param regions Vector of memory regions
/// @param stream CUDA stream
void batch_prefetch_to_device(const vector<ODZCManager::MemoryRegion>& regions,
                               cudaStream_t stream = 0);

/// Batch prefetch multiple regions to host.
/// @param regions Vector of memory regions
/// @param stream CUDA stream
void batch_prefetch_to_host(const vector<ODZCManager::MemoryRegion>& regions,
                             cudaStream_t stream = 0);

}  // namespace odzc

}  // namespace runtime
}  // namespace fascal
