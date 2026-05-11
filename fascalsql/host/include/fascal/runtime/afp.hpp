#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <type_traits>
#include <string>
#include "bloom_filter.hpp"

using namespace std;

namespace fascal {
namespace runtime {

/// Asynchronous Filter Pushdown (AFP) System.
/// Enables early filtering of data before full processing,
/// reducing memory bandwidth and improving cache utilization.
class AFPManager {
 public:
  /// Filter operation types.
  enum class FilterOp {
    EQ,           // == value
    NE,           // != value
    LT,           // < value
    LE,           // <= value
    GT,           // > value
    GE,           // >= value
    BETWEEN,      // value1 <= x <= value2
    IN_LIST,      // x IN (values...)
    LIKE,         // string pattern match
    IS_NULL,      // x IS NULL
    IS_NOT_NULL   // x IS NOT NULL
  };

  /// Filter descriptor for a single predicate.
  struct FilterDescriptor {
    int column_id;          // Column to filter
    FilterOp op;            // Filter operation
    int64_t int_value;      // Integer comparison value
    double float_value;     // Float comparison value
    string str_value;       // String comparison value
    vector<int64_t> int_list;  // For IN_LIST
    int64_t int_value2;     // For BETWEEN (upper bound)
    double float_value2;    // For BETWEEN (upper bound)
    bool negated;           // NOT filter
  };

  /// Filter result containing bitmap and statistics.
  struct FilterResult {
    vector<uint8_t> bitmap;     // Selection bitmap (1 = pass)
    size_t num_passing;         // Number of tuples passing filter
    size_t total_tuples;        // Total tuples processed
    double filter_time_ms;      // Time to execute filter
    double selectivity;         // num_passing / total_tuples
  };

  /// Pushdown statistics.
  struct Stats {
    size_t total_filters_executed;
    size_t total_tuples_filtered;
    size_t total_bytes_saved;   // Bytes not transferred due to filtering
    double total_filter_time_ms;
    double avg_selectivity;
  };

  AFPManager();
  explicit AFPManager(int device_id);
  ~AFPManager();

  /// Register a filter for pushdown execution.
  /// @param filter_id Unique filter identifier
  /// @param descriptor Filter descriptor
  void register_filter(int filter_id, const FilterDescriptor& descriptor);

  /// Unregister a filter.
  /// @param filter_id Filter identifier
  void unregister_filter(int filter_id);

  /// Execute filter on CPU (for early filtering).
  /// @param filter_id Filter identifier
  /// @param data Column data pointer
  /// @param num_tuples Number of tuples
  /// @param type Data type (0=int, 1=float, 2=string)
  /// @return Filter result with bitmap
  FilterResult execute_filter_cpu(int filter_id, const void* data,
                                   size_t num_tuples, int type);

  /// Execute filter on GPU (asynchronous).
  /// @param filter_id Filter identifier
  /// @param d_data Device column data pointer
  /// @param num_tuples Number of tuples
  /// @param type Data type (0=int, 1=float, 2=string)
  /// @param stream CUDA stream for async execution
  /// @return Filter result with device bitmap
  FilterResult execute_filter_gpu(int filter_id, const void* d_data,
                                   size_t num_tuples, int type,
                                   cudaStream_t stream = 0);

  /// Execute multiple filters in batch (conjunction).
  /// @param filter_ids Vector of filter identifiers
  /// @param data_ptrs Vector of column data pointers
  /// @param num_tuples Number of tuples
  /// @param types Vector of data types
  /// @param stream CUDA stream
  /// @return Combined filter result
  FilterResult execute_filter_batch(const vector<int>& filter_ids,
                                     const vector<const void*>& data_ptrs,
                                     size_t num_tuples,
                                     const vector<int>& types,
                                     cudaStream_t stream = 0);

  /// Compact data based on filter bitmap.
  /// @param src Source data
  /// @param dst Destination data
  /// @param bitmap Filter bitmap
  /// @param num_tuples Number of tuples
  /// @param elem_size Size of each element
  /// @param stream CUDA stream
  void compact_data(const void* src, void* dst, const vector<uint8_t>& bitmap,
                    size_t num_tuples, size_t elem_size, cudaStream_t stream = 0);

  /// Estimate selectivity for a filter.
  /// @param filter_id Filter identifier
  /// @return Estimated selectivity (0.0 - 1.0)
  double estimate_selectivity(int filter_id) const;

  /// Check if filter pushdown is beneficial.
  /// @param filter_id Filter identifier
  /// @param num_tuples Number of tuples to filter
  /// @param tuple_size Size of each tuple
  /// @return true if pushdown is estimated to be beneficial
  bool is_pushdown_beneficial(int filter_id, size_t num_tuples,
                               size_t tuple_size) const;

  /// Get statistics.
  /// @return Current statistics
  Stats get_stats() const;

  /// Reset statistics.
  void reset_stats();

  /// Set GPU device.
  /// @param device_id CUDA device ID
  void set_device(int device_id);

  /// Get current GPU device.
  /// @return CUDA device ID
  int get_device() const;

  /// Build a Bloom filter from dimension join keys.
  /// Must be called after GPU hash table build (P2).
  /// @param bloom_id  Unique identifier for this filter (typically = join index)
  /// @param dim_keys  Dimension join key column (host pointer)
  /// @param n         Number of dimension rows
  void build_bloom_filter(int bloom_id, const int* dim_keys, int n);

  /// Execute filter with Bloom filter probe on join key column.
  /// ANDs Bloom filter results into existing bitmap.
  /// @param filter_id  Scalar predicate filter identifier
  /// @param bloom_id   Bloom filter identifier (from build_bloom_filter)
  /// @param probe_col  Probe-side FK column (host pointer)
  /// @param bitmap     Existing selection bitmap (int*, 1=pass, 0=fail)
  /// @param n          Number of tuples
  void execute_filter_with_bloom(int filter_id, int bloom_id,
                                 const int* probe_col, int* bitmap, int n);

  /// Get false positive rate for a built Bloom filter.
  /// @param bloom_id  Bloom filter identifier
  /// @param n_keys    Number of keys that were inserted
  /// @return Estimated false positive rate in [0, 1]
  double bloom_false_positive_rate(int bloom_id, int n_keys) const;

 private:
  int device_id_;
  mutable mutex mutex_;
  unordered_map<int, FilterDescriptor> filters_;
  Stats stats_;

  // Bloom filter storage (one per join, indexed by bloom_id)
  unordered_map<int, fascal::runtime::BloomFilter> bloom_filters_;

  /// Apply single filter to integer data.
  void apply_filter_int(const int* data, uint8_t* bitmap,
                         size_t num_tuples, const FilterDescriptor& desc);

  /// Apply single filter to float data.
  void apply_filter_float(const double* data, uint8_t* bitmap,
                           size_t num_tuples, const FilterDescriptor& desc);

  /// Calculate selectivity for filter operation.
  double calc_selectivity(const FilterDescriptor& desc) const;
};

/// RAII wrapper for filter registration.
class AFPFilterScope {
 public:
  AFPFilterScope(AFPManager& manager, int filter_id,
                 const AFPManager::FilterDescriptor& descriptor);
  ~AFPFilterScope();

  /// Get filter ID.
  /// @return Filter identifier
  int filter_id() const { return filter_id_; }

  // Non-copyable
  AFPFilterScope(const AFPFilterScope&) = delete;
  AFPFilterScope& operator=(const AFPFilterScope&) = delete;

 private:
  AFPManager& manager_;
  int filter_id_;
};

/// AFP-enabled scan operator for query execution.
class AFPScanOperator {
 public:
  /// Scan configuration.
  struct Config {
    bool enable_cpu_pushdown;     // Enable CPU-side filtering
    bool enable_gpu_pushdown;     // Enable GPU-side filtering
    bool enable_compaction;       // Compact data after filtering
    double selectivity_threshold; // Minimum selectivity for pushdown
    size_t min_tuples_for_pushdown; // Minimum tuples to consider pushdown
  };

  AFPScanOperator(AFPManager& manager, const Config& config = default_config());
  ~AFPScanOperator() = default;

  /// Add filter to scan.
  /// @param column_id Column to filter
  /// @param op Filter operation
  /// @param value Filter value
  template<typename T>
  void add_filter(int column_id, AFPManager::FilterOp op, T value);

  /// Execute scan with filter pushdown.
  /// @param columns Vector of column data pointers
  /// @param num_tuples Number of tuples
  /// @param column_types Vector of column types
  /// @param stream CUDA stream
  /// @return Filter result
  AFPManager::FilterResult execute(const vector<const void*>& columns,
                                    size_t num_tuples,
                                    const vector<int>& column_types,
                                    cudaStream_t stream = 0);

  /// Clear all filters.
  void clear_filters();

  /// Get configuration.
  /// @return Current configuration
  const Config& config() const { return config_; }

  /// Set configuration.
  /// @param config New configuration
  void set_config(const Config& config) { config_ = config; }

  /// Get default configuration.
  /// @return Default configuration
  static Config default_config() {
    return Config{
      true,   // enable_cpu_pushdown
      true,   // enable_gpu_pushdown
      true,   // enable_compaction
      0.5,    // selectivity_threshold
      1000    // min_tuples_for_pushdown
    };
  }

 private:
  AFPManager& manager_;
  Config config_;
  vector<pair<int, AFPManager::FilterDescriptor>> filters_;
  int next_filter_id_;
};

/// Helper functions for AFP usage.
namespace afp {

/// Create filter descriptor for comparison operation.
template<typename T>
AFPManager::FilterDescriptor make_filter(int column_id, 
                                          AFPManager::FilterOp op, 
                                          T value) {
  AFPManager::FilterDescriptor desc;
  desc.column_id = column_id;
  desc.op = op;
  desc.negated = false;
  
  if (is_integral<T>::value) {
    desc.int_value = static_cast<int64_t>(value);
  } else if (is_floating_point<T>::value) {
    desc.float_value = static_cast<double>(value);
  }
  
  return desc;
}

/// Create filter descriptor for BETWEEN operation.
template<typename T>
AFPManager::FilterDescriptor make_between_filter(int column_id,
                                                  T low, T high) {
  AFPManager::FilterDescriptor desc;
  desc.column_id = column_id;
  desc.op = AFPManager::FilterOp::BETWEEN;
  desc.negated = false;
  
  if (is_integral<T>::value) {
    desc.int_value = static_cast<int64_t>(low);
    desc.int_value2 = static_cast<int64_t>(high);
  } else if (is_floating_point<T>::value) {
    desc.float_value = static_cast<double>(low);
    desc.float_value2 = static_cast<double>(high);
  }
  
  return desc;
}

/// Create filter descriptor for IN list operation.
AFPManager::FilterDescriptor make_in_list_filter(int column_id,
                                                  const vector<int64_t>& values);

/// Combine multiple filter results (AND).
AFPManager::FilterResult combine_results_and(const vector<AFPManager::FilterResult>& results);

/// Combine multiple filter results (OR).
AFPManager::FilterResult combine_results_or(const vector<AFPManager::FilterResult>& results);

}  // namespace afp

}  // namespace runtime
}  // namespace fascal
