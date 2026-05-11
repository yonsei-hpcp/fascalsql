#pragma once

#include <vector>
#include "fascal/execution/selection_spec.hpp"

using namespace std;

namespace fascal {

/// Join predicate slot: join ID and whether to build hash table on CPU.
struct JoinSlot {
  int join_id{0};
  bool build_on_cpu{false};
};

/// Extended SelectionSpec that includes join predicate support.
/// This allows ablation control over both selection predicates and join build side.
struct ExtendedSelectionSpec {
  /// Selection predicates (from base SelectionSpec).
  vector<PredicateSlot> predicates;

  /// Join predicates for this query.
  vector<JoinSlot> joins;

  /// Add a selection predicate with CPU/GPU decision.
  void add(int predicate_id, bool run_on_cpu) {
    predicates.push_back({predicate_id, run_on_cpu});
  }

  /// Add a join predicate with CPU/GPU decision.
  void add_join(int join_id, bool build_on_cpu) {
    joins.push_back({join_id, build_on_cpu});
  }

  /// Check if a selection predicate should run on CPU.
  bool run_on_cpu(int predicate_id) const {
    for (const auto& p : predicates)
      if (p.predicate_id == predicate_id) return p.run_on_cpu;
    return false;
  }

  /// Check if a join should be built on CPU.
  bool build_on_cpu(int join_id) const {
    for (const auto& j : joins) {
      if (j.join_id == join_id) {
        return j.build_on_cpu;
      }
    }
    return false;
  }

  /// Clear all predicates and joins.
  void clear() {
    predicates.clear();
    joins.clear();
  }

  /// Get total number of predicates (selection + join).
  int total_predicates() const {
    return static_cast<int>(predicates.size()) + static_cast<int>(joins.size());
  }

  /// Get number of selection predicates.
  int num_selections() const {
    return static_cast<int>(predicates.size());
  }

  /// Get number of join predicates.
  int num_joins() const {
    return static_cast<int>(joins.size());
  }
};

}  // namespace fascal
