/**
 * FaScalSQL ODZC (On-Demand Zero-Copy) Manager Implementation.
 *
 * Provides zero-copy memory access: GPU reads host-pinned memory
 * directly via PCIe without explicit cudaMemcpy.  Falls back to
 * cudaHostAlloc+memcpy when cudaHostRegister fails (e.g. the buffer
 * was already registered, or the OS limit on pinned pages is reached).
 */

#include "fascal/runtime/odzc.hpp"
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace fascal {
namespace runtime {

// --------------------------------------------------------------------------
// ODZCManager
// --------------------------------------------------------------------------

ODZCManager::ODZCManager()
    : device_id_(0), unified_memory_supported_(false), stats_{}
{
    initialize();
}

ODZCManager::ODZCManager(int device_id)
    : device_id_(device_id), unified_memory_supported_(false), stats_{}
{
    initialize();
}

ODZCManager::~ODZCManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : regions_) {
        auto& r = kv.second;
        if (r.is_mapped && r.is_pinned) {
            cudaHostUnregister(r.host_ptr);
        }
    }
    regions_.clear();
}

void ODZCManager::initialize() {
    cudaSetDevice(device_id_);

    // Check for unified memory support (managed memory).
    int attr = 0;
    cudaError_t err = cudaDeviceGetAttribute(&attr, cudaDevAttrManagedMemory, device_id_);
    unified_memory_supported_ = (err == cudaSuccess && attr != 0);

    // Reset stats.
    stats_ = Stats{};
}

// ------------- register_memory / unregister_memory -----------------------

ODZCManager::MemoryRegion ODZCManager::register_memory(
    void* host_ptr, size_t size, AccessHint hint)
{
    std::lock_guard<std::mutex> lock(mutex_);
    MemoryRegion region;
    region.host_ptr   = host_ptr;
    region.device_ptr = nullptr;
    region.size       = size;
    region.hint       = hint;
    region.is_pinned  = false;
    region.is_mapped  = false;
    region.device_id  = device_id_;

    // Try to pin + map.
    cudaError_t err = cudaHostRegister(host_ptr, size, cudaHostRegisterMapped);
    if (err == cudaSuccess) {
        region.is_pinned = true;
        void* dptr = nullptr;
        err = cudaHostGetDevicePointer(&dptr, host_ptr, 0);
        if (err == cudaSuccess) {
            region.device_ptr = dptr;
            region.is_mapped  = true;
        }
    }

    regions_[host_ptr] = region;
    stats_.total_mapped_bytes += size;
    return region;
}

void ODZCManager::unregister_memory(MemoryRegion& region) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (region.is_pinned && region.host_ptr) {
        cudaHostUnregister(region.host_ptr);
    }
    regions_.erase(region.host_ptr);
    region.is_pinned = false;
    region.is_mapped = false;
    region.device_ptr = nullptr;
}

// ------------- pin / unpin -----------------------------------------------

bool ODZCManager::pin_memory(void* host_ptr, size_t size) {
    cudaError_t err = cudaHostRegister(host_ptr, size, cudaHostRegisterDefault);
    return err == cudaSuccess;
}

void ODZCManager::unpin_memory(void* host_ptr, size_t /*size*/) {
    cudaHostUnregister(host_ptr);
}

// ------------- map / unmap -----------------------------------------------

void* ODZCManager::map_to_gpu(MemoryRegion& region) {
    if (region.is_mapped) return region.device_ptr;
    if (!region.is_pinned) {
        cudaError_t err = cudaHostRegister(region.host_ptr, region.size, cudaHostRegisterMapped);
        if (err != cudaSuccess) return nullptr;
        region.is_pinned = true;
    }
    void* dptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&dptr, region.host_ptr, 0);
    if (err == cudaSuccess) {
        region.device_ptr = dptr;
        region.is_mapped  = true;
    }
    return region.device_ptr;
}

void ODZCManager::unmap_from_gpu(MemoryRegion& region) {
    if (region.is_pinned && region.host_ptr) {
        cudaHostUnregister(region.host_ptr);
    }
    region.is_pinned  = false;
    region.is_mapped  = false;
    region.device_ptr = nullptr;
}

// ------------- prefetch / advise -----------------------------------------

void ODZCManager::prefetch_to_device(MemoryRegion& /*region*/, cudaStream_t /*stream*/) {
    // Zero-copy model: no prefetch needed; GPU reads via PCIe on demand.
}

void ODZCManager::prefetch_to_host(MemoryRegion& /*region*/, cudaStream_t /*stream*/) {
    // No-op for zero-copy.
}

void ODZCManager::advise(MemoryRegion& /*region*/, cudaMemoryAdvise /*advice*/) {
    // Advise only applicable to managed memory; zero-copy ignores.
}

// ------------- get_device_ptr / is_registered ----------------------------

void* ODZCManager::get_device_ptr(void* host_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = regions_.find(host_ptr);
    if (it != regions_.end()) return it->second.device_ptr;
    return nullptr;
}

bool ODZCManager::is_registered(void* ptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return regions_.count(const_cast<void*>(ptr)) > 0;
}

// ------------- stats -----------------------------------------------------

ODZCManager::Stats ODZCManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void ODZCManager::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = Stats{};
}

// ------------- device accessors ------------------------------------------

void ODZCManager::set_device(int device_id) {
    device_id_ = device_id;
    cudaSetDevice(device_id_);
}

int ODZCManager::get_device() const {
    return device_id_;
}

bool ODZCManager::is_unified_memory_supported() const {
    return unified_memory_supported_;
}

// ------------- register_column / unregister_column -----------------------
// These are the KEY methods used by generated .cu code via fascal_odzc_map().

ODZCManager::MappedColumn ODZCManager::register_column(void* host_col, size_t bytes) {
    MappedColumn mc;
    mc.host_ptr = host_col;
    mc.size     = bytes;

    // Fast path 1: memory already has a valid GPU mapping.
    {
        cudaPointerAttributes attr;
        cudaError_t chk = cudaPointerGetAttributes(&attr, host_col);
        if (chk == cudaSuccess &&
            (attr.type == cudaMemoryTypeManaged ||
             attr.type == cudaMemoryTypeDevice ||
             (attr.type == cudaMemoryTypeHost && attr.devicePointer != nullptr))) {
            mc.device_ptr       = (attr.devicePointer ? attr.devicePointer : host_col);
            mc.is_zero_copy     = true;
            mc.owned_host_alloc = false;
            mc.is_pinned_alloc  = (attr.type == cudaMemoryTypeHost && attr.devicePointer != nullptr);
            return mc;
        }
        (void)cudaGetLastError();
    }

    // Fast path 2: already registered (e.g. by cudaHostRegister earlier).
    {
        void* dptr = nullptr;
        cudaError_t chk = cudaHostGetDevicePointer(&dptr, host_col, 0);
        if (chk == cudaSuccess && dptr) {
            mc.device_ptr       = dptr;
            mc.is_zero_copy     = true;
            mc.owned_host_alloc = false;
            return mc;
        }
        (void)cudaGetLastError();
    }

    // Try to pin the caller's buffer in-place.
    cudaError_t err = cudaHostRegister(host_col, bytes, cudaHostRegisterMapped);
    if (err == cudaSuccess) {
        void* dptr = nullptr;
        err = cudaHostGetDevicePointer(&dptr, host_col, 0);
        if (err == cudaSuccess) {
            mc.device_ptr      = dptr;
            mc.is_zero_copy    = true;
            mc.owned_host_alloc = false;
            return mc;
        }
        // GetDevicePointer failed — unregister and fall through.
        cudaHostUnregister(host_col);
        // Clear sticky CUDA error state so downstream cudaGetLastError() is clean.
        (void)cudaGetLastError();
    } else {
        // Clear sticky CUDA error state from failed cudaHostRegister.
        (void)cudaGetLastError();
    }

    // Fallback: allocate pinned mapped memory and copy.
    void* pinned = nullptr;
    err = cudaHostAlloc(&pinned, bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "ODZCManager::register_column: cudaHostAlloc fallback failed (%s)\n",
                cudaGetErrorString(err));
        // Clear sticky error state.
        (void)cudaGetLastError();
        mc.device_ptr = nullptr;
        return mc;
    }
    std::memcpy(pinned, host_col, bytes);

    void* dptr = nullptr;
    err = cudaHostGetDevicePointer(&dptr, pinned, 0);
    if (err != cudaSuccess) {
        cudaFreeHost(pinned);
        (void)cudaGetLastError();  // Clear sticky error state.
        mc.device_ptr = nullptr;
        return mc;
    }

    mc.host_ptr        = pinned;
    mc.device_ptr      = dptr;
    mc.is_zero_copy    = true;
    mc.owned_host_alloc = true;
    return mc;
}

void ODZCManager::unregister_column(MappedColumn& mc) {
    if (!mc.host_ptr) return;
    if (mc.owned_host_alloc) {
        cudaFreeHost(mc.host_ptr);
    } else if (mc.is_pinned_alloc) {
        // cudaHostAlloc'd memory: already pinned, no cudaHostUnregister needed.
        // The caller owns the allocation and will free it via cudaFreeHost.
    } else {
        cudaHostUnregister(mc.host_ptr);
    }
    mc.host_ptr   = nullptr;
    mc.device_ptr = nullptr;
}

// ------------- find_region -----------------------------------------------

std::unordered_map<void*, ODZCManager::MemoryRegion>::iterator
ODZCManager::find_region(void* ptr) {
    return regions_.find(ptr);
}

// --------------------------------------------------------------------------
// ODZCScope
// --------------------------------------------------------------------------

ODZCScope::ODZCScope(ODZCManager& manager, void* host_ptr, size_t size,
                     ODZCManager::AccessHint hint)
    : manager_(manager)
{
    region_ = manager_.register_memory(host_ptr, size, hint);
}

ODZCScope::~ODZCScope() {
    manager_.unregister_memory(region_);
}

void ODZCScope::prefetch_to_device(cudaStream_t stream) {
    manager_.prefetch_to_device(region_, stream);
}

void ODZCScope::prefetch_to_host(cudaStream_t stream) {
    manager_.prefetch_to_host(region_, stream);
}

// --------------------------------------------------------------------------
// ODZCColumn<T>
// --------------------------------------------------------------------------

template<typename T>
ODZCColumn<T>::ODZCColumn(ODZCManager& manager, T* host_data, size_t num_elements,
                           ODZCManager::AccessHint hint)
    : manager_(&manager), host_ptr_(host_data), device_ptr_(nullptr),
      num_elements_(num_elements)
{
    region_ = manager_->register_memory(host_data, num_elements * sizeof(T), hint);
    device_ptr_ = static_cast<T*>(region_.device_ptr);
}

template<typename T>
void ODZCColumn<T>::prefetch_to_device(cudaStream_t stream) {
    manager_->prefetch_to_device(region_, stream);
}

// Explicit template instantiations for common types.
template class ODZCColumn<int>;
template class ODZCColumn<float>;
template class ODZCColumn<double>;
template class ODZCColumn<unsigned int>;
template class ODZCColumn<unsigned long long>;

// --------------------------------------------------------------------------
// Namespace-level helpers
// --------------------------------------------------------------------------

namespace odzc {

void batch_prefetch_to_device(const std::vector<ODZCManager::MemoryRegion>& /*regions*/,
                               cudaStream_t /*stream*/) {
    // Zero-copy: no prefetch needed.
}

void batch_prefetch_to_host(const std::vector<ODZCManager::MemoryRegion>& /*regions*/,
                             cudaStream_t /*stream*/) {
    // Zero-copy: no prefetch needed.
}

}  // namespace odzc

}  // namespace runtime
}  // namespace fascal
