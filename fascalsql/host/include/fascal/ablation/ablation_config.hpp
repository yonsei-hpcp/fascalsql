#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

#include "fascal/execution/selection_spec.hpp"

namespace fascal {
namespace ablation {

/// Predicate-level ablation: which predicates run on CPU (AFP pushdown) vs GPU.
/// Replaces numeric ablation_level with explicit per-predicate decisions.
class AblationConfig {
 public:
  static constexpr int kMaxPredicateId = 128;

  AblationConfig() = default;

  /// Set whether a predicate runs on CPU (true) or GPU (false).
  void set_predicate_cpu(int predicate_id, bool run_on_cpu);
  bool run_on_cpu(int predicate_id) const;

  /// Apply a named preset (for backward compatibility and experiments).
  /// Presets: "main" (0), "phm" (1), "cvd" (2), "all_cpu", "all_gpu".
  void set_preset(const std::string& name);
  void set_preset_from_legacy_level(int legacy_ablation_level);

  /// Build a SelectionSpec from this config for the given predicate ids.
  void fill_selection_spec(fascal::SelectionSpec* out,
                           const std::vector<int>& predicate_ids) const;

  /// Number of predicates currently configured.
  int num_predicates() const { return static_cast<int>(cpu_mask_.size()); }

 private:
  std::unordered_map<int, bool> cpu_mask_;
};

}  // namespace ablation
}  // namespace fascal
