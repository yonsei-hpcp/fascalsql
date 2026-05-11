/**
 * FaScalSQL Query Runtime Utilities
 *
 * Shared inline functions / macros that every generated query .cu uses.
 * Keeps generated code short → better readability.
 *
 * Include after fascal_generated_common.cuh.
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <immintrin.h>

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "fascal/optimizer/cqo.hpp"
#include "fascal/runtime/odzc.hpp"
#include "fascal/runtime/afp.hpp"
#include "fascal_generated_common.cuh"

// ============================================================================
// 0. Runtime Hardware Profiling + LLC Detection
// ============================================================================

/**
 * Detect L3 (LLC) cache size from Linux sysfs.
 * Falls back to 30MB if sysfs unavailable.
 */
inline size_t fascal_detect_llc_size() {
    // Try sysfs (Linux)
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
    if (f) {
        char buf[64];
        if (fgets(buf, sizeof(buf), f)) {
            fclose(f);
            size_t val = 0;
            char unit = 0;
            if (sscanf(buf, "%zu%c", &val, &unit) >= 1) {
                if (unit == 'K' || unit == 'k') return val * 1024;
                if (unit == 'M' || unit == 'm') return val * 1024 * 1024;
                if (unit == 'G' || unit == 'g') return val * 1024 * 1024 * 1024;
                return val;  // assume bytes
            }
        }
        fclose(f);
    }
    return 30ULL * 1024 * 1024;  // default 30MB
}

/**
 * One-time hardware profiling. Measures GPU bandwidth, CPU AVX2 throughput,
 * ODZC sector latency, and detects LLC size.
 *
 * Results are cached in hw_profile.json. Skip profiling if file has "profiled": true.
 * Force re-profile with FASCAL_REPROFILE=1.
 */
struct FascalHwProfile {
    double tp_gpu_gbs;
    double tp_pcie_gbs;
    double tp_afp_gips;
    double lat_sector_ns;
    size_t sector_size_bytes;
    size_t col_val_bytes;
    int    available_cpu_threads;
    double warm_lat_factor;
    double kernel_launch_us;
    double tile_scan_ns;
    size_t llc_size_bytes;
    bool   profiled;
};

// ── GPU profiling kernels ──────────────────────────────────────────────
// 0) Empty kernel for launch overhead measurement
__global__ void empty_launch_helper() {}

// 0b) Simple int scan for ODZC throughput measurement
__global__ void dev_scan_kernel(const int* __restrict__ data, int N, long long* __restrict__ out) {
    long long sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        sum += data[i];
    if (sum != 0) atomicAdd(reinterpret_cast<unsigned long long*>(out), (unsigned long long)sum);
}

// 1) GPU memory bandwidth: vectorized scan reading 4 ints per thread per iteration
//    to maximize memory throughput and saturate GPU memory bus.
__global__ void fascal_profile_gpu_bw_kernel(const int4* __restrict__ data, int n4,
                                              unsigned long long* __restrict__ out) {
    long long sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n4; i += stride) {
        int4 v = data[i];
        sum += v.x + v.y + v.z + v.w;
    }
    // Single atomicAdd to prevent dead-code elimination
    if (sum != 0) atomicAdd(out, (unsigned long long)sum);
}

// 2) ODZC sector latency: GPU kernel reads from host-mapped memory via zero-copy
//    Single-thread pointer-chase to measure true PCIe sector access latency
__global__ void fascal_profile_sector_lat_kernel(
    const volatile int* __restrict__ mapped, int n_sectors,
    unsigned long long* __restrict__ out_avg_cycles, int num_reads)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned long long total = 0;
    int dummy = 0;
    for (int i = 0; i < num_reads; ++i) {
        int sector = (i * 97 + 31) % n_sectors;  // pseudo-random sector
        int word_offset = sector * 32;            // 128 bytes = 32 ints per sector
        __threadfence_system();
        unsigned long long t0 = clock64();
        dummy += mapped[word_offset];  // trigger PCIe sector fetch
        __threadfence_system();
        unsigned long long t1 = clock64();
        total += (t1 - t0);
    }
    *out_avg_cycles = total / (unsigned long long)num_reads;
    // prevent dead-code elimination
    if (dummy == -999999) *out_avg_cycles = 0;
}

inline FascalHwProfile fascal_runtime_profile() {
    FascalHwProfile prof;
    prof.tp_gpu_gbs = 100.0;
    prof.tp_pcie_gbs = 26.7;
    prof.tp_afp_gips = 8.0;
    prof.lat_sector_ns = 300.0;
    prof.sector_size_bytes = 128;
    prof.col_val_bytes = 4;
    prof.available_cpu_threads = (int)std::thread::hardware_concurrency();
    if (prof.available_cpu_threads <= 0) prof.available_cpu_threads = 8;
    prof.warm_lat_factor = 0.002;
    prof.kernel_launch_us = 2.5;
    prof.tile_scan_ns = 12.0;
    prof.llc_size_bytes = fascal_detect_llc_size();
    prof.profiled = false;

    // ════════════════════════════════════════════════════════════════════
    // 1. TP_GPU: GPU processing throughput (device memory bandwidth)
    //    Paper §IV-D: TP_j^GPU — how fast GPU processes tuples from its
    //    own memory, NOT PCIe bandwidth.
    //    Method: vectorized scan kernel (int4) over 256MB device memory,
    //    large grid to saturate all SMs. Measures effective GB/s.
    // ════════════════════════════════════════════════════════════════════
    {
        const int N = 64 * 1024 * 1024;  // 64M ints = 256 MB
        const int N4 = N / 4;            // int4 elements
        const size_t SZ = (size_t)N * sizeof(int);
        int *d_data = nullptr;
        unsigned long long *d_out = nullptr;
        cudaError_t e1 = cudaMalloc(&d_data, SZ);
        cudaError_t e2 = cudaMalloc(&d_out, sizeof(unsigned long long));
        if (e1 == cudaSuccess && e2 == cudaSuccess) {
            cudaMemset(d_data, 0x42, SZ);
            cudaMemset(d_out, 0, sizeof(unsigned long long));
            // Query SM count for optimal grid sizing
            int sm_count = 0;
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
            if (sm_count <= 0) sm_count = 48;
            int grid = sm_count * 8;  // ~8 blocks per SM for high occupancy
            int block = 256;

            // Warm up (3 iterations)
            for (int i = 0; i < 3; ++i)
                fascal_profile_gpu_bw_kernel<<<grid, block>>>((const int4*)d_data, N4, d_out);
            cudaDeviceSynchronize();

            cudaEvent_t t0, t1;
            cudaEventCreate(&t0); cudaEventCreate(&t1);
            const int ITERS = 20;
            cudaMemset(d_out, 0, sizeof(unsigned long long));
            cudaEventRecord(t0);
            for (int i = 0; i < ITERS; ++i)
                fascal_profile_gpu_bw_kernel<<<grid, block>>>((const int4*)d_data, N4, d_out);
            cudaEventRecord(t1);
            cudaDeviceSynchronize();
            float ms = 0;
            cudaEventElapsedTime(&ms, t0, t1);
            if (ms > 0)
                prof.tp_gpu_gbs = (double)SZ * ITERS / ((double)ms * 1e-3) / 1e9;
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }
        if (d_data) cudaFree(d_data);
        if (d_out) cudaFree(d_out);
    }

    // ════════════════════════════════════════════════════════════════════
    // 1b. TP_PCIE: ODZC zero-copy column scan throughput (GB/s)
    //    Measures actual PCIe BW when GPU reads host-pinned memory.
    //    This is the effective BW for ODZC column loads in the kernel.
    // ════════════════════════════════════════════════════════════════════
    {
        const size_t N_ODZC = 64ULL * 1024 * 1024;  // 64M ints = 256 MB
        const size_t SZ_ODZC = N_ODZC * sizeof(int);
        int *h_odzc = nullptr;
        cudaError_t e = cudaHostAlloc((void**)&h_odzc, SZ_ODZC, cudaHostAllocMapped);
        if (e == cudaSuccess) {
            for (size_t i = 0; i < N_ODZC; ++i) h_odzc[i] = (int)(i & 0x7FFF);
            int *d_odzc = nullptr;
            cudaHostGetDevicePointer((void**)&d_odzc, h_odzc, 0);
            unsigned long long *d_odzc_out = nullptr;
            cudaMalloc(&d_odzc_out, sizeof(unsigned long long));
            int sm_cnt = 0;
            cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, 0);
            int grid = (sm_cnt > 0 ? sm_cnt : 48) * 8;
            int block = 256;
            // Warm up
            for (int i = 0; i < 3; ++i) {
                cudaMemset(d_odzc_out, 0, sizeof(unsigned long long));
                dev_scan_kernel<<<grid, block>>>(d_odzc, (int)N_ODZC, (long long*)d_odzc_out);
            }
            cudaDeviceSynchronize();
            // Measure
            cudaEvent_t to0, to1;
            cudaEventCreate(&to0); cudaEventCreate(&to1);
            const int OITERS = 10;
            cudaEventRecord(to0);
            for (int i = 0; i < OITERS; ++i) {
                cudaMemset(d_odzc_out, 0, sizeof(unsigned long long));
                dev_scan_kernel<<<grid, block>>>(d_odzc, (int)N_ODZC, (long long*)d_odzc_out);
            }
            cudaEventRecord(to1);
            cudaDeviceSynchronize();
            float oms = 0;
            cudaEventElapsedTime(&oms, to0, to1);
            if (oms > 0)
                prof.tp_pcie_gbs = (double)SZ_ODZC * OITERS / ((double)oms * 1e-3) / 1e9;
            cudaEventDestroy(to0); cudaEventDestroy(to1);
            cudaFree(d_odzc_out);
            cudaFreeHost(h_odzc);
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // 1c. Kernel launch overhead (µs)
    //    Measures bare launch latency for a full-grid empty kernel.
    // ════════════════════════════════════════════════════════════════════
    {
        int sm_cnt = 0;
        cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, 0);
        int grid = (sm_cnt > 0 ? sm_cnt : 48) * 8;
        int block = 128;
        // warm up
        for (int i = 0; i < 10; ++i) empty_launch_helper<<<grid, block>>>();
        cudaDeviceSynchronize();
        cudaEvent_t tl0, tl1;
        cudaEventCreate(&tl0); cudaEventCreate(&tl1);
        const int LITERS = 1000;
        cudaEventRecord(tl0);
        for (int i = 0; i < LITERS; ++i) empty_launch_helper<<<grid, block>>>();
        cudaEventRecord(tl1);
        cudaDeviceSynchronize();
        float lms = 0;
        cudaEventElapsedTime(&lms, tl0, tl1);
        prof.kernel_launch_us = lms * 1000.0 / (double)LITERS;
        cudaEventDestroy(tl0); cudaEventDestroy(tl1);
    }

    // ════════════════════════════════════════════════════════════════════
    // 2. TP_AFP: CPU AVX2 predicate scan AGGREGATE throughput (GIPS)
    //    Paper §IV-D: TP_j^AFP — actual AVX2-vectorized range predicate
    //    throughput **across all available CPU threads**, mirroring how
    //    AFP runs in production (one worker thread per morsel batch).
    //
    //    Critical: this number is the **aggregate** throughput (sum across
    //    threads), NOT per-thread × N_CPU.  AVX2 predicates are memory-bound,
    //    so per-thread throughput collapses past a few threads (DRAM ~50 GB/s
    //    saturates with 3-4 threads at 18 GB/s/thread for 4-byte ints).
    //    Multiplying single-thread throughput by N_CPU overestimates by ~10×,
    //    causing CQO to wrongly favour AFP-heavy plans.
    //
    //    Method: 256M int32 array (large enough to defeat L3 cache),
    //    spawn N_CPU pthread workers each running the same avx2_range_pass
    //    pattern over its slab.  Wall-clock the full pass.
    // ════════════════════════════════════════════════════════════════════
    {
        const size_t N = 256ULL * 1024 * 1024;  // 256M ints = 1 GB
        int *arr = (int*)aligned_alloc(64, N * sizeof(int));
        const size_t bm_bytes = (N + 7) / 8;
        uint8_t *bm8 = (uint8_t*)aligned_alloc(64, bm_bytes);
        if (arr && bm8) {
            for (size_t i = 0; i < N; ++i) arr[i] = (int)(i & 0x7FFF);

            const int n_threads = std::max(1, prof.available_cpu_threads);

            // Per-thread worker: range predicate over [start, end).  Mirrors
            // fascal::cpu_pred::avx2_range_pass exactly so the measured cost
            // matches what runs in production AFP.
            auto worker = [&](size_t start, size_t end) {
                const __m256i v_lo = _mm256_set1_epi32(1000);
                const __m256i v_hi = _mm256_set1_epi32(30000);
                const __m256i ones = _mm256_set1_epi32(-1);
                memset(bm8 + (start >> 3), 0xFF, ((end - start) + 7) / 8);
                for (size_t i = start; i + 7 < end; i += 8) {
                    size_t bidx = i >> 3;
                    if (!bm8[bidx]) continue;
                    __m256i v = _mm256_loadu_si256((const __m256i*)(arr + i));
                    __m256i ge = _mm256_or_si256(_mm256_cmpgt_epi32(v, v_lo),
                                                  _mm256_cmpeq_epi32(v, v_lo));
                    __m256i le = _mm256_xor_si256(_mm256_cmpgt_epi32(v, v_hi), ones);
                    bm8[bidx] &= (uint8_t)_mm256_movemask_ps(
                        _mm256_castsi256_ps(_mm256_and_si256(ge, le)));
                }
            };

            auto run_pass = [&]() {
                std::vector<std::thread> ths;
                ths.reserve(n_threads);
                size_t per = N / n_threads;
                per = (per + 7) & ~7ULL;  // 8-int aligned slabs
                for (int t = 0; t < n_threads; ++t) {
                    size_t s = (size_t)t * per;
                    size_t e = (t + 1 == n_threads) ? N : std::min(N, s + per);
                    if (s < e) ths.emplace_back(worker, s, e);
                }
                for (auto &th : ths) th.join();
            };

            // Warm up (fault pages, prime caches, JIT-warm OS threadpool)
            run_pass();
            run_pass();

            // Timed run: 5 iterations, take wall-clock
            const int ITERS = 5;
            auto start = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < ITERS; ++r) run_pass();
            auto end = std::chrono::high_resolution_clock::now();
            double secs = std::chrono::duration<double>(end - start).count();
            if (secs > 0)
                prof.tp_afp_gips = (double)N * ITERS / secs / 1e9;
        }
        if (arr) free(arr);
        if (bm8) free(bm8);
    }

    // ════════════════════════════════════════════════════════════════════
    // 3. Lat_Sector: ODZC zero-copy sector access latency (ns)
    //    Paper §IV-D: Lat_Sector — latency for GPU kernel to read a
    //    128-byte sector from host-pinned mapped memory via PCIe.
    //    Method: GPU kernel with clock64() reads random sectors from
    //    cudaHostAllocMapped memory, converts cycles→ns via GPU clock.
    // ════════════════════════════════════════════════════════════════════
    {
        const size_t SZ = 4ULL * 1024 * 1024;  // 4 MB mapped buffer
        const int N_SECTORS = (int)(SZ / 128);
        const int NUM_READS = 512;
        int *h_buf = nullptr;
        cudaError_t e = cudaHostAlloc((void**)&h_buf, SZ, cudaHostAllocMapped);
        if (e == cudaSuccess) {
            int *d_buf = nullptr;
            cudaHostGetDevicePointer((void**)&d_buf, h_buf, 0);
            // Fill with data (ensure pages are faulted in)
            for (size_t i = 0; i < SZ / sizeof(int); ++i)
                h_buf[i] = (int)i;

            unsigned long long *d_cycles = nullptr;
            cudaMalloc(&d_cycles, sizeof(unsigned long long));

            // Warm up: first access pulls pages into host TLB
            cudaMemset(d_cycles, 0, sizeof(unsigned long long));
            fascal_profile_sector_lat_kernel<<<1, 1>>>(
                (const volatile int*)d_buf, N_SECTORS, d_cycles, 32);
            cudaDeviceSynchronize();

            // Timed run
            cudaMemset(d_cycles, 0, sizeof(unsigned long long));
            fascal_profile_sector_lat_kernel<<<1, 1>>>(
                (const volatile int*)d_buf, N_SECTORS, d_cycles, NUM_READS);
            cudaDeviceSynchronize();

            unsigned long long avg_cycles = 0;
            cudaMemcpy(&avg_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

            // Convert GPU cycles to nanoseconds using SM clock rate
            int clock_khz = 0;
            cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
            if (clock_khz > 0 && avg_cycles > 0) {
                double gpu_ghz = (double)clock_khz / 1e6;  // kHz → GHz
                prof.lat_sector_ns = (double)avg_cycles / gpu_ghz;
            }

            if (d_cycles) cudaFree(d_cycles);
        }
        if (h_buf) cudaFreeHost(h_buf);
    }

    prof.profiled = true;
    return prof;
}

/**
 * Write profile to hw_profile.json.
 */
inline void fascal_write_hw_profile(const FascalHwProfile &prof, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "{\n");
    fprintf(f, "  \"tp_gpu_gbs\": %.2f,\n", prof.tp_gpu_gbs);
    fprintf(f, "  \"tp_pcie_gbs\": %.2f,\n", prof.tp_pcie_gbs);
    fprintf(f, "  \"tp_afp_gips\": %.2f,\n", prof.tp_afp_gips);
    fprintf(f, "  \"lat_sector_ns\": %.2f,\n", prof.lat_sector_ns);
    fprintf(f, "  \"sector_size_bytes\": %zu,\n", prof.sector_size_bytes);
    fprintf(f, "  \"col_val_bytes\": %zu,\n", prof.col_val_bytes);
    fprintf(f, "  \"available_cpu_threads\": %d,\n", prof.available_cpu_threads);
    fprintf(f, "  \"warm_lat_factor\": %.4f,\n", prof.warm_lat_factor);
    fprintf(f, "  \"kernel_launch_us\": %.1f,\n", prof.kernel_launch_us);
    fprintf(f, "  \"tile_scan_ns\": %.1f,\n", prof.tile_scan_ns);
    fprintf(f, "  \"llc_size_bytes\": %zu,\n", prof.llc_size_bytes);
    fprintf(f, "  \"profiled\": true\n");
    fprintf(f, "}\n");
    fclose(f);
}

/**
 * Get LLC size (cached from profile or sysfs).
 */
inline size_t fascal_get_llc_size() {
    static size_t cached = 0;
    if (cached == 0) cached = fascal_detect_llc_size();
    return cached;
}

/**
 * Detect GPU L2 cache size via cudaDeviceProp.
 * Falls back to 4MB if detection fails (typical for consumer GPUs).
 * Used to cap Bloom filter sizes so BF data stays in GPU L2 during probe.
 */
inline size_t fascal_detect_gpu_l2_size() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.l2CacheSize > 0) {
        return (size_t)prop.l2CacheSize;
    }
    return 4ULL * 1024 * 1024;  // default 4MB
}

/**
 * Get GPU L2 cache size (cached after first call).
 */
inline size_t fascal_get_gpu_l2_size() {
    static size_t cached = 0;
    if (cached == 0) cached = fascal_detect_gpu_l2_size();
    return cached;
}

// ============================================================================
// 1. CQO Initialization
// ============================================================================

/**
 * Initialize CQO input with default HW parameters, then override from
 * hw_profile.json (with runtime profiling if needed). Returns a fully-populated CqoInput.
 *
 * Per FaScalSQL §IV-D, the throughput / latency fields below are populated
 * from a one-time profiling pass (the values committed to hw_profile.json
 * were captured with the FASCAL_REPROFILE microbench). The defaults below
 * are only used when no profile is available.
 *
 * @param num_tuples   Fact-table tuple count.
 */
inline fascal::optimizer::CqoInput fascal_cqo_init(int num_tuples) {
    fascal::optimizer::CqoInput cqo_in;
    cqo_in.num_tuples    = num_tuples;
    cqo_in.tp_gpu_gbs    = 100.0;
    cqo_in.tp_afp_gips   = 8.0;
    cqo_in.lat_sector_ns = 300.0;
    cqo_in.sector_size_bytes   = 128;
    cqo_in.col_val_bytes       = 4;
    cqo_in.llc_size_bytes = 0;

    cqo_in.cpu_contention = 0.0;  // default: no contention (overridden after hw_profile)
    cqo_in.tp_pcie_gbs = 26.7;   // default PCIe ODZC BW (overridden by profile)
    cqo_in.kernel_launch_us = 2.5; // default kernel launch overhead
    cqo_in.tile_scan_ns = 12.0;   // default tile scan overhead
    cqo_in.gpu_batch_size = 4194304; // default GPU batch size

    // Try multiple paths for hw_profile.json:
    //  1. FASCAL_HW_PROFILE env var (explicit override)
    //  2. Relative paths for common invocation directories
    FILE *hw_fp = nullptr;
    const char *hw_path_used = nullptr;
    static const char *hw_paths[] = {
        nullptr,  // slot 0: env var
        "../configs/hw_profile.json",
        "fascalsql/configs/hw_profile.json",
        "configs/hw_profile.json",
        "../../fascalsql/configs/hw_profile.json",
    };
    const char *hw_env = getenv("FASCAL_HW_PROFILE");
    hw_paths[0] = hw_env;
    for (int i = 0; i < 5 && !hw_fp; ++i) {
        if (!hw_paths[i]) continue;
        hw_fp = fopen(hw_paths[i], "r");
        if (hw_fp) hw_path_used = hw_paths[i];
    }

    bool need_profile = (getenv("FASCAL_REPROFILE") != nullptr);
    bool have_profile = false;

    if (hw_fp) {
        char hw_buf[8192];
        size_t hw_n = fread(hw_buf, 1, sizeof(hw_buf) - 1, hw_fp);
        hw_buf[hw_n] = '\0';
        fclose(hw_fp);
        char *p = nullptr;
        if ((p = strstr(hw_buf, "\"tp_gpu_gbs\"")))
            sscanf(p, "\"tp_gpu_gbs\"%*[^0-9.-]%lf", &cqo_in.tp_gpu_gbs);
        if ((p = strstr(hw_buf, "\"tp_afp_gips\"")))
            sscanf(p, "\"tp_afp_gips\"%*[^0-9.-]%lf", &cqo_in.tp_afp_gips);
        if ((p = strstr(hw_buf, "\"lat_sector_ns\"")))
            sscanf(p, "\"lat_sector_ns\"%*[^0-9.-]%lf", &cqo_in.lat_sector_ns);
        if ((p = strstr(hw_buf, "\"sector_size_bytes\"")))
            sscanf(p, "\"sector_size_bytes\"%*[^0-9]%zu", &cqo_in.sector_size_bytes);
        if ((p = strstr(hw_buf, "\"col_val_bytes\"")))
            sscanf(p, "\"col_val_bytes\"%*[^0-9]%zu", &cqo_in.col_val_bytes);
        if ((p = strstr(hw_buf, "\"available_cpu_threads\"")))
            sscanf(p, "\"available_cpu_threads\"%*[^0-9]%d", &cqo_in.available_cpu_threads);
        if ((p = strstr(hw_buf, "\"warm_lat_factor\"")))
            sscanf(p, "\"warm_lat_factor\"%*[^0-9.-]%lf", &cqo_in.warm_lat_factor);
        if ((p = strstr(hw_buf, "\"tp_pcie_gbs\"")))
            sscanf(p, "\"tp_pcie_gbs\"%*[^0-9.-]%lf", &cqo_in.tp_pcie_gbs);
        if ((p = strstr(hw_buf, "\"kernel_launch_us\"")))
            sscanf(p, "\"kernel_launch_us\"%*[^0-9.-]%lf", &cqo_in.kernel_launch_us);
        if ((p = strstr(hw_buf, "\"tile_scan_ns\"")))
            sscanf(p, "\"tile_scan_ns\"%*[^0-9.-]%lf", &cqo_in.tile_scan_ns);
        if ((p = strstr(hw_buf, "\"llc_size_bytes\""))) {
            sscanf(p, "\"llc_size_bytes\"%*[^0-9]%zu", &cqo_in.llc_size_bytes);
        }
        if ((p = strstr(hw_buf, "\"profiled\""))) {
            have_profile = (strstr(p, "true") != nullptr);
        }
    }

    // Run profiling if no valid profile or FASCAL_REPROFILE is set
    if (!have_profile || need_profile) {
        printf("FaScal: running one-time hardware profiling...\n");
        FascalHwProfile prof = fascal_runtime_profile();
        cqo_in.tp_gpu_gbs = prof.tp_gpu_gbs;
        cqo_in.tp_pcie_gbs = prof.tp_pcie_gbs;
        cqo_in.tp_afp_gips = prof.tp_afp_gips;
        cqo_in.lat_sector_ns = prof.lat_sector_ns;
        cqo_in.sector_size_bytes = prof.sector_size_bytes;
        cqo_in.col_val_bytes = prof.col_val_bytes;
        cqo_in.available_cpu_threads = prof.available_cpu_threads;
        cqo_in.warm_lat_factor = prof.warm_lat_factor;
        cqo_in.kernel_launch_us = prof.kernel_launch_us;
        cqo_in.tile_scan_ns = prof.tile_scan_ns;
        cqo_in.llc_size_bytes = prof.llc_size_bytes;
        printf("FaScal: profile: gpu=%.1f GB/s, pcie=%.1f GB/s, afp=%.2f GIPS, lat=%.0f ns, launch=%.1f µs, LLC=%zuMB\n",
               prof.tp_gpu_gbs, prof.tp_pcie_gbs, prof.tp_afp_gips, prof.lat_sector_ns,
               prof.kernel_launch_us, prof.llc_size_bytes / (1024 * 1024));
        // Write to file: prefer FASCAL_HW_PROFILE env var, then read path, then cwd
        const char *write_env = getenv("FASCAL_HW_PROFILE");
        const char *write_path = write_env ? write_env
                               : (hw_path_used ? hw_path_used : "hw_profile.json");
        fascal_write_hw_profile(prof, write_path);
        printf("FaScal: profile written to %s\n", write_path);
    }

    // Detect LLC if not in profile
    if (cqo_in.llc_size_bytes == 0)
        cqo_in.llc_size_bytes = fascal_detect_llc_size();

    // FASCAL_WARM_LAT env var overrides hw_profile
    if (const char *wlf = getenv("FASCAL_WARM_LAT")) {
        double v = atof(wlf);
        if (v > 0.0 && v <= 1.0) cqo_in.warm_lat_factor = v;
    }

    // FASCAL_CPU_THREADS override (AFTER hw_profile, so it takes precedence)
    // Derives cpu_contention from ratio of available vs hardware threads.
    {
        int hw_threads = (int)std::thread::hardware_concurrency();
        if (hw_threads <= 0) hw_threads = 16;
        if (const char *env = getenv("FASCAL_CPU_THREADS")) {
            int v = atoi(env);
            if (v > 0) {
                cqo_in.available_cpu_threads = v;
                cqo_in.cpu_contention = 1.0 - (double)v / (double)hw_threads;
                cqo_in.cpu_contention = std::max(0.0, std::min(0.95, cqo_in.cpu_contention));
            }
        }
        // Explicit contention override
        if (const char *cc = getenv("CQO_CPU_CONTENTION")) {
            double v = atof(cc);
            if (v >= 0.0 && v <= 1.0) cqo_in.cpu_contention = v;
        }
    }

    return cqo_in;
}

/**
 * Apply CQO output to h_pred_flags with optional argv overrides.
 */
inline void fascal_cqo_apply(
    const fascal::optimizer::CqoOutput &cqo_out,
    unsigned char *h_pred_flags,
    int max_predicates,
    int argc, char **argv)
{
    for (size_t i = 0; i < cqo_out.predicate_on_cpu.size() && i < (size_t)max_predicates; ++i)
        h_pred_flags[i] = (argc > 3 + (int)i)
            ? atoi(argv[3 + (int)i])
            : (cqo_out.predicate_on_cpu[i] ? 1 : 0);
}

/**
 * Print CQO decision summary: selectivities, placement, BF activation.
 * Useful for debugging and artifact evaluation.
 */
inline void fascal_cqo_print_decisions(
    const fascal::optimizer::CqoInput &in,
    const fascal::optimizer::CqoOutput &out,
    const unsigned char *h_pred_flags, int n_flags)
{
    if (getenv("FASCAL_QUIET")) return;  // suppress in batch mode
    printf("CQO: cost_gpu=%.6e cost_cpu=%.6e ratio=%.3f speedup=%.2f\n",
           out.cost_gpu, out.cost_cpu, out.workload_dist_ratio, out.estimated_speedup);
    printf("CQO: cpu_threads=%d contention=%.0f%% llc=%zuMB\n",
           in.available_cpu_threads,
           in.cpu_contention * 100.0,
           in.llc_size_bytes / (1024 * 1024));
    // Per-predicate: selectivity + placement
    for (size_t i = 0; i < in.predicate_selectivities.size(); i++) {
        const char *where = (i < (size_t)n_flags && h_pred_flags[i]) ? "CPU" : "GPU";
        printf("CQO:   pred[%zu] sel=%.4f → %s\n", i, in.predicate_selectivities[i], where);
    }
    // Per-join BF
    for (size_t j = 0; j < in.join_bf_selectivities.size(); j++) {
        const char *active = (j < out.join_bf_active.size() && out.join_bf_active[j]) ? "ON" : "OFF";
        printf("CQO:   bf[%zu]   fpr=%.4f → CPU %s\n", j, in.join_bf_selectivities[j], active);
    }
}

// ============================================================================
// 1b. Selectivity Sample Profiling
// ============================================================================

/**
 * Estimate predicate selectivity by sampling a fraction of rows.
 * Paper §IV-D: "selectivities derived from one-time profiling."
 *
 * Samples up to SAMPLE_SIZE rows uniformly, applies the predicate,
 * and returns the fraction of rows that pass.
 *
 * @param col        Column data pointer.
 * @param num_tuples Total rows.
 * @param pred_fn    bool(int value) — returns true if row passes.
 * @param sample_size Number of rows to sample (default 65536).
 * @return Estimated selectivity in [0, 1].
 */
template <typename PredFn>
inline double fascal_estimate_selectivity(
    const int *col, int num_tuples, PredFn pred_fn, int sample_size = 65536)
{
    if (num_tuples <= 0 || !col) return 0.5;
    int n = std::min(sample_size, num_tuples);
    int stride = std::max(1, num_tuples / n);
    int pass = 0;
    for (int i = 0; i < n; ++i) {
        int idx = (int)((long long)i * stride % num_tuples);
        if (pred_fn(col[idx])) ++pass;
    }
    double sel = (double)pass / (double)n;
    return std::max(0.001, std::min(1.0, sel));  // clamp to avoid 0/1 extremes
}

/**
 * Estimate range predicate selectivity: col BETWEEN lo AND hi.
 */
inline double fascal_estimate_range_sel(
    const int *col, int num_tuples, int lo, int hi, int sample_size = 65536)
{
    return fascal_estimate_selectivity(col, num_tuples,
        [lo, hi](int v) { return v >= lo && v <= hi; }, sample_size);
}

/**
 * Estimate equality predicate selectivity: col == value.
 */
inline double fascal_estimate_eq_sel(
    const int *col, int num_tuples, int value, int sample_size = 65536)
{
    return fascal_estimate_selectivity(col, num_tuples,
        [value](int v) { return v == value; }, sample_size);
}

/**
 * Estimate join BF selectivity from dim table cardinality ratio.
 * BF pass rate ≈ (dim_qualifying / dim_total) + BF_FPR.
 * Use when BF is not yet built (before CQO).
 *
 * @param dim_qualifying  Number of dimension rows passing build-side predicates.
 * @param dim_total       Total dimension table rows.
 * @param bf_fpr          Bloom filter false-positive rate (default 0.02 for 8bits/key k=3).
 */
inline double fascal_estimate_join_bf_sel(int dim_qualifying, int dim_total, double bf_fpr = 0.02) {
    if (dim_total <= 0) return 0.05;
    double match_rate = (double)dim_qualifying / (double)dim_total;
    return std::min(1.0, match_rate + bf_fpr);
}

/**
 * Estimate BF selectivity by probing a sample of FK values against a BF.
 * Returns the fraction of probes that pass (true positives + false positives).
 */
inline double fascal_estimate_bf_sel(
    const int *fk_col, int num_tuples,
    const fascal::runtime::BloomFilter *bf, int sample_size = 65536)
{
    if (!bf || !bf->bits || num_tuples <= 0) return 0.05;
    int n = std::min(sample_size, num_tuples);
    int stride = std::max(1, num_tuples / n);
    int pass = 0;
    for (int i = 0; i < n; ++i) {
        int idx = (int)((long long)i * stride % num_tuples);
        if (bf->query(fk_col[idx])) ++pass;
    }
    return std::max(0.001, std::min(1.0, (double)pass / n));
}

// ============================================================================
// 2. Seed Bitmap Allocation
// ============================================================================

// ============================================================================
// 3. ODZC Helpers
// ============================================================================

/**
 * Register a host column with ODZC and get device pointer.
 * Falls back to cudaMalloc+cudaMemcpy if ODZC mapping fails.
 *
 * @param mgr       ODZCManager to register with.
 * @param h_ptr     Host pointer to map.
 * @param n         Number of ints.
 * @param col_name  Column name (for error messages).
 * @param mc_out    [out] MappedColumn handle (for later unregister).
 * @param d_ptr_out [out] Device pointer.
 * @param used_malloc_out [out] True if malloc fallback was used.
 */
inline void fascal_odzc_map(
    fascal::runtime::ODZCManager *mgr,
    int *h_ptr, int n, const char *col_name,
    fascal::runtime::ODZCManager::MappedColumn *mc_out,
    int **d_ptr_out,
    bool *used_malloc_out = nullptr)
{
    *mc_out = mgr->register_column(h_ptr, (size_t)n * sizeof(int));
    *d_ptr_out = static_cast<int*>(mc_out->device_ptr);
    if (used_malloc_out) *used_malloc_out = false;
    if (!*d_ptr_out) {
        fprintf(stderr, "ODZC failed for %s, fallback to cudaMalloc\n", col_name);
        CUDA_CHECK(cudaMalloc(d_ptr_out, (size_t)n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(*d_ptr_out, h_ptr, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
        if (used_malloc_out) *used_malloc_out = true;
    }
}

inline void fascal_odzc_map_u64(
    fascal::runtime::ODZCManager *mgr,
    unsigned long long *h_ptr, int n, const char *col_name,
    fascal::runtime::ODZCManager::MappedColumn *mc_out,
    unsigned long long **d_ptr_out,
    bool *used_malloc_out = nullptr)
{
    *mc_out = mgr->register_column(h_ptr, (size_t)n * sizeof(unsigned long long));
    *d_ptr_out = static_cast<unsigned long long*>(mc_out->device_ptr);
    if (used_malloc_out) *used_malloc_out = false;
    if (!*d_ptr_out) {
        fprintf(stderr, "ODZC failed for %s, fallback to cudaMalloc\n", col_name);
        CUDA_CHECK(cudaMalloc(d_ptr_out, (size_t)n * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpy(*d_ptr_out, h_ptr, (size_t)n * sizeof(unsigned long long), cudaMemcpyHostToDevice));
        if (used_malloc_out) *used_malloc_out = true;
    }
}

// Template ODZC mapping for arbitrary column types (float, double, etc.)
template <typename T>
inline void fascal_odzc_map_typed(
    fascal::runtime::ODZCManager *mgr,
    T *h_ptr, int n, const char *col_name,
    fascal::runtime::ODZCManager::MappedColumn *mc_out,
    T **d_ptr_out,
    bool *used_malloc_out = nullptr)
{
    *mc_out = mgr->register_column(h_ptr, (size_t)n * sizeof(T));
    *d_ptr_out = static_cast<T*>(mc_out->device_ptr);
    if (used_malloc_out) *used_malloc_out = false;
    if (!*d_ptr_out) {
        fprintf(stderr, "ODZC failed for %s, fallback to cudaMalloc\n", col_name);
        CUDA_CHECK(cudaMalloc(d_ptr_out, (size_t)n * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(*d_ptr_out, h_ptr, (size_t)n * sizeof(T), cudaMemcpyHostToDevice));
        if (used_malloc_out) *used_malloc_out = true;
    }
}

// ============================================================================
// 4. CPU Multi-threaded Prefilter
// ============================================================================

// ============================================================================
// Persistent Thread Pool — avoids thread spawn overhead per pipeline invocation
// ============================================================================

class FascalThreadPool {
public:
    static FascalThreadPool& instance() {
        static FascalThreadPool pool;
        return pool;
    }

    void run_work_stealing(int total_units, std::function<void(int)> unit_fn) {
        if (total_units == 0) return;
        next_unit_.store(0, std::memory_order_relaxed);
        total_units_ = total_units;
        unit_fn_ = &unit_fn;
        done_count_.store(0, std::memory_order_relaxed);

        // Wake all workers
        generation_.fetch_add(1, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mu_);
        }
        cv_.notify_all();

        // Spin-wait for all workers to finish (low latency)
        const int n = (int)workers_.size();
        while (done_count_.load(std::memory_order_acquire) < n)
            _mm_pause();
    }

    /**
     * Template version: avoids std::function virtual dispatch overhead.
     * The lambda/functor is type-erased via a thin wrapper that stores a
     * pointer, eliminating heap allocation and indirect calls.
     */
    template <typename F>
    void run_work_stealing_t(int total_units, F &&fn) {
        if (total_units == 0) return;
        // Wrap the lambda in a std::function (unavoidable since workers
        // are pre-spawned and need a type-erased pointer). But since we
        // store a pointer to a stack-local std::function, the lambda
        // itself is not heap-allocated if it fits in small-buffer.
        std::function<void(int)> fn_wrapper = std::forward<F>(fn);
        run_work_stealing(total_units, fn_wrapper);
    }

    int num_threads() const { return (int)workers_.size(); }

    ~FascalThreadPool() {
        shutdown_.store(true, std::memory_order_release);
        generation_.fetch_add(1, std::memory_order_release);
        cv_.notify_all();
        for (auto &w : workers_) w.join();
    }

private:
    FascalThreadPool() {
        int n = (int)std::thread::hardware_concurrency();
        if (n <= 0) n = 8;
        // Respect FASCAL_CPU_THREADS env for limiting thread count
        if (const char *env = getenv("FASCAL_CPU_THREADS")) {
            int v = atoi(env);
            if (v > 0) n = v;
        }
        workers_.reserve(n);
        for (int i = 0; i < n; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    void worker_loop() {
        int my_gen = 0;
        while (true) {
            // Wait for new task
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]() {
                    return generation_.load(std::memory_order_acquire) != my_gen;
                });
                my_gen = generation_.load(std::memory_order_acquire);
            }
            if (shutdown_.load(std::memory_order_acquire)) return;

            // Work-stealing loop
            auto& fn = *unit_fn_;
            while (true) {
                int uid = next_unit_.fetch_add(1, std::memory_order_relaxed);
                if (uid >= total_units_) break;
                fn(uid);
            }
            done_count_.fetch_add(1, std::memory_order_release);
        }
    }

    FascalThreadPool(const FascalThreadPool&) = delete;
    FascalThreadPool& operator=(const FascalThreadPool&) = delete;

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<int> generation_{0};
    std::atomic<bool> shutdown_{false};
    std::atomic<int> next_unit_{0};
    std::atomic<int> done_count_{0};
    int total_units_ = 0;
    std::function<void(int)> *unit_fn_ = nullptr;
};

/**
 * CPU prefilter using persistent thread pool (zero thread-spawn overhead).
 * Workers grab 4K-tuple morsels via work-stealing, fill bitmap + tile summary.
 */
inline void fascal_cpu_prefilter_run(
    int num_tuples,
    uint32_t *bitmap,
    uint8_t *tile_summary,
    std::function<void(int offset, int count)> morsel_fn)
{
    static constexpr int CPU_WORK_UNIT = 4096;
    const int TILE_SZ = FASCAL_TILE_SIZE;
    const int bitmap_words = (num_tuples + 31) / 32;
    const int total_units = (num_tuples + CPU_WORK_UNIT - 1) / CPU_WORK_UNIT;

    if (total_units == 0) return;

    std::function<void(int)> unit_fn = [&](int uid) {
        int offset = uid * CPU_WORK_UNIT;
        int cnt = std::min(CPU_WORK_UNIT, num_tuples - offset);

        morsel_fn(offset, cnt);

        // Compute tile summaries
        int first_tile = offset / TILE_SZ;
        int last_tile  = (offset + cnt - 1) / TILE_SZ;
        for (int tile = first_tile; tile <= last_tile; ++tile) {
            uint32_t any_set = 0;
            int base_word = (tile * TILE_SZ) >> 5;
            int words_per_tile = TILE_SZ / 32;
            for (int w = 0; w < words_per_tile && (base_word + w) < bitmap_words; ++w)
                any_set |= bitmap[base_word + w];
            tile_summary[tile] = (any_set != 0) ? 1 : 0;
        }
    };

    FascalThreadPool::instance().run_work_stealing(total_units, unit_fn);
}

/**
 * Batch-aware CPU prefilter: processes tuples in [batch_offset, batch_offset + batch_count).
 * Bitmap and tile_summary are indexed globally (same buffers as full-table version).
 * The morsel_fn receives GLOBAL offsets so that predicate code uses correct array indices.
 */
inline void fascal_cpu_prefilter_run_batch(
    int total_tuples,
    int batch_offset,
    int batch_count,
    uint32_t *bitmap,
    uint8_t *tile_summary,
    std::function<void(int offset, int count)> morsel_fn)
{
    static constexpr int CPU_WORK_UNIT = 4096;
    const int TILE_SZ = FASCAL_TILE_SIZE;
    const int bitmap_words = (total_tuples + 31) / 32;
    const int total_units = (batch_count + CPU_WORK_UNIT - 1) / CPU_WORK_UNIT;

    if (total_units == 0) return;

    std::function<void(int)> unit_fn = [&](int uid) {
        int local_offset = uid * CPU_WORK_UNIT;
        int global_offset = batch_offset + local_offset;
        int cnt = std::min(CPU_WORK_UNIT, batch_count - local_offset);

        morsel_fn(global_offset, cnt);

        // Compute tile summaries (global tile IDs)
        int first_tile = global_offset / TILE_SZ;
        int last_tile  = (global_offset + cnt - 1) / TILE_SZ;
        for (int tile = first_tile; tile <= last_tile; ++tile) {
            uint32_t any_set = 0;
            int base_word = (tile * TILE_SZ) >> 5;
            int words_per_tile = TILE_SZ / 32;
            for (int w = 0; w < words_per_tile && (base_word + w) < bitmap_words; ++w)
                any_set |= bitmap[base_word + w];
            tile_summary[tile] = (any_set != 0) ? 1 : 0;
        }
    };

    FascalThreadPool::instance().run_work_stealing(total_units, unit_fn);
}

/**
 * Allocate packed seed bitmap + tile summary (no doorbell) for CPU-first model.
 * Allocates BOTH pinned host memory (for CPU prefilter writes) AND GPU VRAM
 * (for GPU kernel reads via cudaMemcpy, avoiding zero-copy PCIe overhead).
 */
inline void fascal_alloc_seed_bitmap(
    int num_tuples,
    uint32_t **h_bitmap_out, uint32_t **d_bitmap_out,
    uint8_t **h_tile_summary_out, uint8_t **d_tile_summary_out,
    uint32_t **d_bitmap_vram_out = nullptr,
    uint8_t **d_tile_summary_vram_out = nullptr)
{
    size_t bitmap_words = ((size_t)num_tuples + 31) / 32;
    size_t bitmap_bytes = bitmap_words * sizeof(uint32_t);
    CUDA_CHECK(cudaHostAlloc(h_bitmap_out, bitmap_bytes, cudaHostAllocMapped));
    memset(*h_bitmap_out, 0, bitmap_bytes);
    CUDA_CHECK(cudaHostGetDevicePointer((void**)d_bitmap_out, *h_bitmap_out, 0));

    size_t num_tiles = ((size_t)num_tuples + FASCAL_TILE_SIZE - 1) / FASCAL_TILE_SIZE;
    CUDA_CHECK(cudaHostAlloc(h_tile_summary_out, num_tiles * sizeof(uint8_t), cudaHostAllocMapped));
    memset(*h_tile_summary_out, 0, num_tiles);
    CUDA_CHECK(cudaHostGetDevicePointer((void**)d_tile_summary_out, *h_tile_summary_out, 0));

    if (d_bitmap_vram_out) {
        CUDA_CHECK(cudaMalloc(d_bitmap_vram_out, bitmap_bytes));
        CUDA_CHECK(cudaMemset(*d_bitmap_vram_out, 0, bitmap_bytes));
    }
    if (d_tile_summary_vram_out) {
        CUDA_CHECK(cudaMalloc(d_tile_summary_vram_out, num_tiles * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemset(*d_tile_summary_vram_out, 0, num_tiles));
    }
}

/**
 * Copy a batch region of the bitmap + tile summary from pinned host memory
 * to GPU VRAM.  GPU kernels should use the VRAM pointers.
 */
inline void fascal_copy_bitmap_batch_to_vram(
    int num_tuples,
    int batch_offset, int batch_count,
    const uint32_t *h_bitmap, uint32_t *d_bitmap_vram,
    const uint8_t *h_tile_summary, uint8_t *d_tile_summary_vram,
    cudaStream_t stream = 0)
{
    size_t bm_words = ((size_t)num_tuples + 31) / 32;
    size_t start_word = (size_t)batch_offset >> 5;
    size_t end_word   = ((size_t)(batch_offset + batch_count) + 31) >> 5;
    if (end_word > bm_words) end_word = bm_words;
    size_t copy_words = end_word - start_word;
    if (copy_words > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_bitmap_vram + start_word,
                                   h_bitmap + start_word,
                                   copy_words * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));
    }

    int TILE_SZ = FASCAL_TILE_SIZE;
    size_t num_tiles = ((size_t)num_tuples + TILE_SZ - 1) / TILE_SZ;
    size_t start_tile = (size_t)batch_offset / TILE_SZ;
    size_t end_tile   = ((size_t)(batch_offset + batch_count) + TILE_SZ - 1) / TILE_SZ;
    if (end_tile > num_tiles) end_tile = num_tiles;
    size_t copy_tiles = end_tile - start_tile;
    if (copy_tiles > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_tile_summary_vram + start_tile,
                                   h_tile_summary + start_tile,
                                   copy_tiles * sizeof(uint8_t),
                                   cudaMemcpyHostToDevice, stream));
    }
}

/**
 * Free seed bitmap and tile summary (no doorbell variant).
 */
inline void fascal_free_seed_bitmap(
    uint32_t *h_bitmap, uint8_t *h_tile_summary,
    uint32_t *d_bitmap_vram = nullptr, uint8_t *d_tile_summary_vram = nullptr)
{
    if (d_tile_summary_vram) cudaFree(d_tile_summary_vram);
    if (d_bitmap_vram)       cudaFree(d_bitmap_vram);
    if (h_tile_summary)      cudaFreeHost(h_tile_summary);
    if (h_bitmap)            cudaFreeHost(h_bitmap);
}

// ============================================================================
// 5. Auto-detect num_tuples
// ============================================================================

/**
 * Auto-detect num_tuples from the first binary column file if user passed 0.
 * Returns the detected count, or -1 on failure.
 */
inline int fascal_autodetect_tuples(const char *data_dir, const char *first_col_name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", data_dir, first_col_name);
    int count = 0;
    int *tmp = load_binary_column_auto(path, &count);
    if (tmp) fascal_free_column(tmp);
    return count > 0 ? count : -1;
}

// ============================================================================
// 6. Dimension table column loading helper
// ============================================================================

/**
 * Load a dimension table column and get the tuple count.
 * @return Host pointer, or nullptr on failure. *out_count set.
 */
inline int* fascal_load_dim_column(const char *data_dir, const char *col_name, int *out_count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", data_dir, col_name);
    return load_binary_column_auto(path, out_count);
}

// ============================================================================
// 7. cudaMemPrefetchAsync helper for ODZC columns
// ============================================================================

/**
 * Prefetch a range of a managed-memory column to the GPU before kernel launch.
 * This overlaps PCIe transfer with CPU prefilter work (paper Section IV-B).
 * Safe to call on non-managed pointers (no-op if cudaMemPrefetchAsync fails).
 *
 * @param d_ptr      Device pointer (from cudaMallocManaged).
 * @param offset     Start offset in elements.
 * @param count      Number of elements to prefetch.
 * @param stream     CUDA stream for async prefetch.
 */
inline void fascal_prefetch_column(int *d_ptr, size_t offset, size_t count, cudaStream_t stream = 0) {
    if (!d_ptr || count == 0) return;
    int device = 0;
    cudaGetDevice(&device);
    // cudaMemPrefetchAsync is a hint; ignore errors (pointer may not be managed)
    cudaMemPrefetchAsync(d_ptr + offset, count * sizeof(int), device, stream);
}

inline void fascal_prefetch_column_u64(unsigned long long *d_ptr, size_t offset, size_t count, cudaStream_t stream = 0) {
    if (!d_ptr || count == 0) return;
    int device = 0;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(d_ptr + offset, count * sizeof(unsigned long long), device, stream);
}

// ============================================================================
// 8. CPU Prefilter Timing (optional instrumentation)
// ============================================================================

#include <chrono>

/**
 * Simple RAII timer for measuring CPU prefilter duration.
 * Usage: { FascalCpuTimer timer("l1 prefilter"); ... code ... }
 * Prints elapsed time on destruction.
 */
struct FascalCpuTimer {
    const char *label;
    std::chrono::high_resolution_clock::time_point start;
    double *out_ms;  // optional: store result here

    FascalCpuTimer(const char *l, double *out = nullptr)
        : label(l), start(std::chrono::high_resolution_clock::now()), out_ms(out) {}

    ~FascalCpuTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (out_ms) *out_ms = ms;
        printf("CPU timer [%s]: %.3f ms\n", label, ms);
    }
};
