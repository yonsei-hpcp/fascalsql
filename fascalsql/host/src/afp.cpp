/**
 * FaScalSQL AFP (Asynchronous Filter Pushdown) Manager Implementation.
 *
 * The generated .cu code calls AFPManager::register_filter() and the
 * constructor/destructor.  All actual CPU filtering is performed by
 * codegen-emitted inline functions (AVX2 + scalar), NOT by AFPManager
 * methods.  Therefore most methods here are thin stubs that satisfy
 * the linker.
 */

#include "fascal/runtime/afp.hpp"
#include <cstdio>
#include <cstring>
#include <numeric>
#include <algorithm>

namespace fascal {
namespace runtime {

// --------------------------------------------------------------------------
// AFPManager
// --------------------------------------------------------------------------

AFPManager::AFPManager()
    : device_id_(0), stats_{}
{
}

AFPManager::AFPManager(int device_id)
    : device_id_(device_id), stats_{}
{
}

AFPManager::~AFPManager() {
    // Free any built Bloom filters.
    for (auto& kv : bloom_filters_) {
        kv.second.free_filter();
    }
    bloom_filters_.clear();
}

// ------------- register / unregister filters -----------------------------

void AFPManager::register_filter(int filter_id, const FilterDescriptor& descriptor) {
    std::lock_guard<std::mutex> lock(mutex_);
    filters_[filter_id] = descriptor;
}

void AFPManager::unregister_filter(int filter_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    filters_.erase(filter_id);
}

// ------------- execute_filter_cpu ----------------------------------------

AFPManager::FilterResult AFPManager::execute_filter_cpu(
    int filter_id, const void* data, size_t num_tuples, int type)
{
    FilterResult result;
    result.bitmap.resize(num_tuples, 1);
    result.total_tuples = num_tuples;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = filters_.find(filter_id);
    if (it == filters_.end()) {
        result.num_passing = num_tuples;
        result.selectivity = 1.0;
        return result;
    }

    const FilterDescriptor& desc = it->second;

    if (type == 0) {
        apply_filter_int(static_cast<const int*>(data), result.bitmap.data(), num_tuples, desc);
    } else if (type == 1) {
        apply_filter_float(static_cast<const double*>(data), result.bitmap.data(), num_tuples, desc);
    }

    result.num_passing = 0;
    for (size_t i = 0; i < num_tuples; ++i) {
        if (result.bitmap[i]) ++result.num_passing;
    }
    result.selectivity = num_tuples > 0
        ? (double)result.num_passing / (double)num_tuples : 1.0;

    stats_.total_filters_executed++;
    stats_.total_tuples_filtered += num_tuples;
    return result;
}

// ------------- execute_filter_gpu (stub) ---------------------------------

AFPManager::FilterResult AFPManager::execute_filter_gpu(
    int /*filter_id*/, const void* /*d_data*/,
    size_t num_tuples, int /*type*/, cudaStream_t /*stream*/)
{
    // GPU filtering is done inline by the generated kernel; this is a no-op stub.
    FilterResult result;
    result.bitmap.resize(num_tuples, 1);
    result.num_passing  = num_tuples;
    result.total_tuples = num_tuples;
    result.selectivity  = 1.0;
    return result;
}

// ------------- execute_filter_batch (stub) -------------------------------

AFPManager::FilterResult AFPManager::execute_filter_batch(
    const std::vector<int>& /*filter_ids*/,
    const std::vector<const void*>& /*data_ptrs*/,
    size_t num_tuples,
    const std::vector<int>& /*types*/,
    cudaStream_t /*stream*/)
{
    FilterResult result;
    result.bitmap.resize(num_tuples, 1);
    result.num_passing  = num_tuples;
    result.total_tuples = num_tuples;
    result.selectivity  = 1.0;
    return result;
}

// ------------- compact_data (stub) ---------------------------------------

void AFPManager::compact_data(const void* /*src*/, void* /*dst*/,
                               const std::vector<uint8_t>& /*bitmap*/,
                               size_t /*num_tuples*/, size_t /*elem_size*/,
                               cudaStream_t /*stream*/)
{
    // Compaction is not used by generated code; stub.
}

// ------------- selectivity estimation ------------------------------------

double AFPManager::estimate_selectivity(int filter_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = filters_.find(filter_id);
    if (it == filters_.end()) return 1.0;
    return calc_selectivity(it->second);
}

bool AFPManager::is_pushdown_beneficial(int /*filter_id*/, size_t /*num_tuples*/,
                                         size_t /*tuple_size*/) const {
    // Always beneficial in our model; CQO decides placement.
    return true;
}

// ------------- stats -----------------------------------------------------

AFPManager::Stats AFPManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void AFPManager::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = Stats{};
}

// ------------- device accessors ------------------------------------------

void AFPManager::set_device(int device_id) {
    device_id_ = device_id;
}

int AFPManager::get_device() const {
    return device_id_;
}

// ------------- Bloom filter methods --------------------------------------

void AFPManager::build_bloom_filter(int bloom_id, const int* dim_keys, int n) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Free existing one if any.
    auto it = bloom_filters_.find(bloom_id);
    if (it != bloom_filters_.end()) {
        it->second.free_filter();
    }
    BloomFilter bf = BloomFilter::allocate(n, 3);
    bf.build(dim_keys, n);
    bloom_filters_[bloom_id] = bf;
}

void AFPManager::execute_filter_with_bloom(
    int /*filter_id*/, int bloom_id,
    const int* probe_col, int* bitmap, int n)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = bloom_filters_.find(bloom_id);
    if (it == bloom_filters_.end()) return;
    it->second.probe_and_mask(probe_col, bitmap, n);
}

double AFPManager::bloom_false_positive_rate(int bloom_id, int n_keys) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = bloom_filters_.find(bloom_id);
    if (it == bloom_filters_.end()) return 1.0;
    return it->second.false_positive_rate(n_keys);
}

// ------------- private: apply filters ------------------------------------

void AFPManager::apply_filter_int(const int* data, uint8_t* bitmap,
                                   size_t num_tuples, const FilterDescriptor& desc)
{
    int64_t val  = desc.int_value;
    int64_t val2 = desc.int_value2;

    for (size_t i = 0; i < num_tuples; ++i) {
        if (!bitmap[i]) continue;
        int v = data[i];
        bool pass = false;
        switch (desc.op) {
            case FilterOp::EQ:      pass = (v == val); break;
            case FilterOp::NE:      pass = (v != val); break;
            case FilterOp::LT:      pass = (v <  val); break;
            case FilterOp::LE:      pass = (v <= val); break;
            case FilterOp::GT:      pass = (v >  val); break;
            case FilterOp::GE:      pass = (v >= val); break;
            case FilterOp::BETWEEN: pass = (v >= val && v <= val2); break;
            case FilterOp::IN_LIST: {
                pass = false;
                for (auto iv : desc.int_list) {
                    if (v == iv) { pass = true; break; }
                }
                break;
            }
            default: pass = true; break;
        }
        if (desc.negated) pass = !pass;
        bitmap[i] = pass ? 1 : 0;
    }
}

void AFPManager::apply_filter_float(const double* data, uint8_t* bitmap,
                                     size_t num_tuples, const FilterDescriptor& desc)
{
    double val  = desc.float_value;
    double val2 = desc.float_value2;

    for (size_t i = 0; i < num_tuples; ++i) {
        if (!bitmap[i]) continue;
        double v = data[i];
        bool pass = false;
        switch (desc.op) {
            case FilterOp::EQ:      pass = (v == val); break;
            case FilterOp::NE:      pass = (v != val); break;
            case FilterOp::LT:      pass = (v <  val); break;
            case FilterOp::LE:      pass = (v <= val); break;
            case FilterOp::GT:      pass = (v >  val); break;
            case FilterOp::GE:      pass = (v >= val); break;
            case FilterOp::BETWEEN: pass = (v >= val && v <= val2); break;
            default: pass = true; break;
        }
        if (desc.negated) pass = !pass;
        bitmap[i] = pass ? 1 : 0;
    }
}

double AFPManager::calc_selectivity(const FilterDescriptor& desc) const {
    // Default heuristic; CQO uses plan-level selectivities.
    switch (desc.op) {
        case FilterOp::EQ:      return 0.1;
        case FilterOp::NE:      return 0.9;
        case FilterOp::LT:
        case FilterOp::LE:
        case FilterOp::GT:
        case FilterOp::GE:      return 0.33;
        case FilterOp::BETWEEN: return 0.25;
        case FilterOp::IN_LIST: return 0.1 * (double)desc.int_list.size();
        default:                return 0.5;
    }
}

// --------------------------------------------------------------------------
// AFPFilterScope
// --------------------------------------------------------------------------

AFPFilterScope::AFPFilterScope(AFPManager& manager, int filter_id,
                               const AFPManager::FilterDescriptor& descriptor)
    : manager_(manager), filter_id_(filter_id)
{
    manager_.register_filter(filter_id_, descriptor);
}

AFPFilterScope::~AFPFilterScope() {
    manager_.unregister_filter(filter_id_);
}

// --------------------------------------------------------------------------
// AFPScanOperator
// --------------------------------------------------------------------------

AFPScanOperator::AFPScanOperator(AFPManager& manager, const Config& config)
    : manager_(manager), config_(config), next_filter_id_(0)
{
}

void AFPScanOperator::clear_filters() {
    for (auto& p : filters_) {
        manager_.unregister_filter(p.first);
    }
    filters_.clear();
    next_filter_id_ = 0;
}

AFPManager::FilterResult AFPScanOperator::execute(
    const std::vector<const void*>& /*columns*/,
    size_t num_tuples,
    const std::vector<int>& /*column_types*/,
    cudaStream_t /*stream*/)
{
    AFPManager::FilterResult result;
    result.bitmap.resize(num_tuples, 1);
    result.num_passing  = num_tuples;
    result.total_tuples = num_tuples;
    result.selectivity  = 1.0;
    return result;
}

// --------------------------------------------------------------------------
// Namespace-level helper functions
// --------------------------------------------------------------------------

namespace afp {

AFPManager::FilterDescriptor make_in_list_filter(int column_id,
                                                  const std::vector<int64_t>& values)
{
    AFPManager::FilterDescriptor desc;
    desc.column_id = column_id;
    desc.op        = AFPManager::FilterOp::IN_LIST;
    desc.int_list  = values;
    desc.negated   = false;
    return desc;
}

AFPManager::FilterResult combine_results_and(
    const std::vector<AFPManager::FilterResult>& results)
{
    if (results.empty()) return AFPManager::FilterResult{};
    size_t n = results[0].total_tuples;
    AFPManager::FilterResult combined;
    combined.bitmap.resize(n, 1);
    combined.total_tuples = n;
    for (const auto& r : results) {
        for (size_t i = 0; i < n && i < r.bitmap.size(); ++i) {
            combined.bitmap[i] = combined.bitmap[i] && r.bitmap[i];
        }
    }
    combined.num_passing = 0;
    for (size_t i = 0; i < n; ++i) {
        if (combined.bitmap[i]) ++combined.num_passing;
    }
    combined.selectivity = n > 0 ? (double)combined.num_passing / (double)n : 1.0;
    return combined;
}

AFPManager::FilterResult combine_results_or(
    const std::vector<AFPManager::FilterResult>& results)
{
    if (results.empty()) return AFPManager::FilterResult{};
    size_t n = results[0].total_tuples;
    AFPManager::FilterResult combined;
    combined.bitmap.resize(n, 0);
    combined.total_tuples = n;
    for (const auto& r : results) {
        for (size_t i = 0; i < n && i < r.bitmap.size(); ++i) {
            combined.bitmap[i] = combined.bitmap[i] || r.bitmap[i];
        }
    }
    combined.num_passing = 0;
    for (size_t i = 0; i < n; ++i) {
        if (combined.bitmap[i]) ++combined.num_passing;
    }
    combined.selectivity = n > 0 ? (double)combined.num_passing / (double)n : 1.0;
    return combined;
}

}  // namespace afp

}  // namespace runtime
}  // namespace fascal
