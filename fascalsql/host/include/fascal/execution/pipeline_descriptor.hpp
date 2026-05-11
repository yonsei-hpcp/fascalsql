#pragma once

#include <cstdint>
#include <string>

namespace fascal {

/// Query-independent pipeline description for runtime.
/// Used to configure a single pipeline (one table scan + selection + kernel).
struct PipelineDescriptor {
  int64_t num_tuples{0};
  std::string name;

  PipelineDescriptor() = default;
  PipelineDescriptor(int64_t n, const char* pipeline_name = "")
      : num_tuples(n), name(pipeline_name ? pipeline_name : "") {}
};

/// Describes the valid range for one GPU iteration (ODZC: offset + count).
struct IterationRange {
  int offset{0};
  int count{0};
};

}  // namespace fascal
