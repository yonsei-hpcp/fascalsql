#pragma once

#include <cstdint>
#include <vector>

namespace fascal {

/// Per-predicate slot: id and whether it runs on CPU (AFP pushdown).
struct PredicateSlot {
  int predicate_id{0};
  bool run_on_cpu{false};
};

/// Specifies which predicates exist and which run on CPU vs GPU.
/// Used by runtime and ablation; supports arbitrary combinations for ablation.
struct SelectionSpec {
  static constexpr int kMaxPredicates = 64;

  std::vector<PredicateSlot> predicates;

  void clear() { predicates.clear(); }
  void add(int predicate_id, bool run_on_cpu) {
    predicates.push_back({predicate_id, run_on_cpu});
  }
  bool run_on_cpu(int predicate_id) const {
    for (const auto& p : predicates)
      if (p.predicate_id == predicate_id) return p.run_on_cpu;
    return false;
  }
};

}  // namespace fascal
