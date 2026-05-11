#pragma once

#include <vector>
#include <string>
#include <cmath>
#include "fascal/execution/extended_selection_spec.hpp"

using namespace std;

namespace fascal {
namespace ablation {

/// Ablation configuration with name and SelectionSpec.
struct AblationConfig {
  string name;
  ExtendedSelectionSpec spec;

  AblationConfig() = default;
  AblationConfig(const string& n, const ExtendedSelectionSpec& s)
      : name(n), spec(s) {}
};

/// AblationCombinator: Generates all 2^N combinations of CPU/GPU pushdown.
/// For N predicates, generates 2^N configurations where each predicate can be
/// either on CPU or GPU.
class AblationCombinator {
 public:
  /// Generate all 2^N combinations for N predicates.
  static vector<AblationConfig> generate_all_combinations(
      const ExtendedSelectionSpec& base_spec) {
    vector<AblationConfig> combinations;
    int num_predicates = base_spec.num_selections();
    int num_joins = base_spec.num_joins();
    int total_predicates = num_predicates + num_joins;

    // 2^total_predicates combinations
    int total_combinations = 1 << total_predicates;  // 2^N

    for (int i = 0; i < total_combinations; ++i) {
      ExtendedSelectionSpec spec;
      spec.predicates = base_spec.predicates;
      spec.joins = base_spec.joins;

      // Set CPU/GPU decisions based on binary representation of i
      for (int j = 0; j < total_predicates; ++j) {
        bool run_on_cpu = (i & (1 << j)) != 0;

        if (j < num_predicates) {
          // Selection predicate
          spec.predicates[j].run_on_cpu = run_on_cpu;
        } else {
          // Join predicate
          int join_idx = j - num_predicates;
          spec.joins[join_idx].build_on_cpu = run_on_cpu;
        }
      }

      // Generate descriptive name
      char name[128];
      snprintf(name, sizeof(name), "ablation_%d", i);
      combinations.push_back(AblationConfig(name, spec));
    }

    return combinations;
  }

  /// Generate combinations with constraints (e.g., max CPU predicates).
  static vector<AblationConfig> generate_constrained_combinations(
      const ExtendedSelectionSpec& base_spec,
      int max_cpu_predicates) {
    vector<AblationConfig> all_combinations =
        generate_all_combinations(base_spec);

    // Filter by max CPU predicates constraint
    vector<AblationConfig> filtered;
    for (const auto& config : all_combinations) {
      int cpu_count = 0;
      for (const auto& p : config.spec.predicates) {
        if (p.run_on_cpu) cpu_count++;
      }
      for (const auto& j : config.spec.joins) {
        if (j.build_on_cpu) cpu_count++;
      }

      if (cpu_count <= max_cpu_predicates) {
        filtered.push_back(config);
      }
    }

    return filtered;
  }

  /// Generate specific combinations from config (e.g., all CPU, all GPU).
  static vector<AblationConfig> generate_from_config(
      const ExtendedSelectionSpec& base_spec,
      const string& mode) {
    vector<AblationConfig> combinations;

    if (mode == "all_cpu") {
      ExtendedSelectionSpec spec = base_spec;
      for (auto& p : spec.predicates) p.run_on_cpu = true;
      for (auto& j : spec.joins) j.build_on_cpu = true;
      combinations.push_back(AblationConfig("all_cpu", spec));
    } else if (mode == "all_gpu") {
      ExtendedSelectionSpec spec = base_spec;
      for (auto& p : spec.predicates) p.run_on_cpu = false;
      for (auto& j : spec.joins) j.build_on_cpu = false;
      combinations.push_back(AblationConfig("all_gpu", spec));
    } else if (mode == "selective_cpu") {
      // Push selective predicates to CPU
      ExtendedSelectionSpec spec = base_spec;
      // Simple heuristic: push first half of predicates to CPU
      for (size_t i = 0; i < spec.predicates.size(); ++i) {
        spec.predicates[i].run_on_cpu = (i < spec.predicates.size() / 2);
      }
      combinations.push_back(AblationConfig("selective_cpu", spec));
    }

    return combinations;
  }

  /// Generate binary representation of ablation configuration.
  static string to_binary_string(const ExtendedSelectionSpec& spec) {
    string result;
    for (const auto& p : spec.predicates) {
      result += p.run_on_cpu ? "1" : "0";
    }
    for (const auto& j : spec.joins) {
      result += j.build_on_cpu ? "1" : "0";
    }
    return result;
  }
};

}  // namespace ablation
}  // namespace fascal
