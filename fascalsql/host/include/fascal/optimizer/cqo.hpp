#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

namespace fascal {
namespace optimizer {

/// Contention-aware Query Optimizer (CQO): decides which predicates/BFs
/// run on CPU (AFP pushdown) vs GPU, minimizing max(Cost_GPU, Cost_CPU).
/// Implements the analytical cost model from FaScalSQL §IV-D.
struct CqoInput {
  /// Estimated selectivity per predicate (0..1). Ordered by pipeline position.
  std::vector<double> predicate_selectivities;
  /// CPU contention factor in [0, 1]: 0 = idle, 1 = fully contended.
  double cpu_contention{0.0};

  // ── Hardware parameters (from runtime profiling) ──
  size_t num_tuples{0};
  double tp_gpu_gbs{100.0};       // GPU HBM throughput (device memory BW)
  double tp_pcie_gbs{26.7};       // ODZC zero-copy (PCIe) throughput, measured warm
  double tp_afp_gips{8.0};        // CPU AVX2 filter throughput (aggregate)
  double lat_sector_ns{300.0};    // ODZC zero-copy sector latency (cold)
  double warm_lat_factor{0.002};  // warm-run discount for lat_sector
  size_t sector_size_bytes{128};
  size_t col_val_bytes{4};
  int available_cpu_threads{8};
  double kernel_launch_us{2.5};   // per-batch kernel launch overhead (µs)
  int gpu_batch_size{4194304};    // tuples per batch (FASCAL_GPU_BATCH)
  double tile_scan_ns{12.0};      // fixed overhead per tile (bitmap decode, scheduling)

  /// Per-join BF false-positive rate (0..1)
  std::vector<double> join_bf_selectivities;
  /// L3 cache size in bytes (0 = detect at runtime, fallback 30MB)
  size_t llc_size_bytes{0};

  /// Number of GPU-only operations (columns loaded for join/agg, not pred/BF).
  /// These are always on GPU and see the final selectivity.
  /// Default 1 = single aggregation pass over surviving tuples.
  int num_gpu_only_ops{1};
};

struct CqoOutput {
  double workload_dist_ratio{0.5};
  std::vector<bool> predicate_on_cpu;
  double cost_gpu{0.0};
  double cost_cpu{0.0};
  double estimated_speedup{1.0};
  std::vector<bool> join_bf_active;
  /// CQO decided no Bloom filter (CPU or GPU) is profitable — runtime should
  /// disable all BF probes for this query.  Equivalent to the user setting
  /// FASCAL_BLOOM_OFF=1 but driven by the cost model.
  bool bloom_off{false};
  std::string rationale;
};

/// FaScalSQL §IV-D: P* = arg min_{P∈P} max(Cost_GPU, Cost_CPU)
///
/// Cost_GPU = Σ_{j=1}^{|GPU_Ops|} N · (R_j^GPU/TP_GPU + P_j^ODZC · Lat/(Sector/ColVal))
/// Cost_CPU = Σ_{j=1}^{|AFP_Ops|} N · R_j^AFP / (TP_AFP · Available_CPU)
///
/// R_j^GPU tracks progressive filtering: each GPU operation sees fewer tuples
/// than the previous one, reflecting the actual data movement through the pipeline.
inline CqoOutput cqo_evaluate(const CqoInput& in) {
  CqoOutput best;
  best.predicate_on_cpu.assign(in.predicate_selectivities.size(), false);
  best.join_bf_active.assign(in.join_bf_selectivities.size(), false);
  best.cost_gpu = 1e18;
  best.cost_cpu = 1e18;

  const double N = (double)std::max((size_t)1, in.num_tuples);
  const double TP_GPU = std::max(1.0, in.tp_gpu_gbs);  // GB/s (GPU HBM)
  const double TP_PCIE = std::max(1.0, in.tp_pcie_gbs);  // GB/s (PCIe/ODZC)
  const double TP_AFP = std::max(1.0, in.tp_afp_gips);  // GIPS
  const double LAT = std::max(1.0, in.lat_sector_ns * std::max(0.001, in.warm_lat_factor));
  const double SECTOR = (double)in.sector_size_bytes;
  const double COL_VAL = (double)in.col_val_bytes;
  const double N_CPU = std::max(1, in.available_cpu_threads);
  const double vals_per_sector = SECTOR / COL_VAL;
  const double LAUNCH_US = std::max(0.1, in.kernel_launch_us);
  const int GPU_BATCH = std::max(1, in.gpu_batch_size);
  const int N_BATCHES = std::max(1, (int)((N + GPU_BATCH - 1) / GPU_BATCH));
  const double TILE_SIZE = (double)in.sector_size_bytes / (double)in.col_val_bytes;  // typically 32
  const double TILE_SCAN_NS = std::max(0.1, in.tile_scan_ns);
  const double n_tiles = N / TILE_SIZE;
  int n_preds = (int)in.predicate_selectivities.size();
  int n_joins = (int)in.join_bf_selectivities.size();
  int n_pred_combos = 1 << std::min(n_preds, 8);
  int n_bf_combos   = 1 << std::min(n_joins, 4);

  double best_cost = 1e18;

  // Try one extra "BF entirely OFF" combination per pred_mask: when a Bloom
  // filter adds overhead without enough selectivity to pay for itself, the
  // best plan is to skip the BF probe on both CPU and GPU and let the HT
  // join filter directly. The original two-mask sweep only chooses between
  // CPU-side BF and GPU-side BF; this third option captures the case where
  // CQO should match the ODZC-only mode the user can force via FASCAL_BLOOM_OFF.
  // Encode it as bf_off = (-1) and skip the BF cost contribution entirely.
  for (int pred_mask = 0; pred_mask < n_pred_combos; pred_mask++) {
    for (int bf_mask = -1; bf_mask < n_bf_combos; bf_mask++) {
      const bool bf_off = (bf_mask < 0);

      // ── Cost_CPU: Σ N · R_j^AFP / (TP_AFP · (1 − contention)) ──
      // TP_AFP is now the **aggregate** AVX2 predicate throughput across
      // all available CPU threads (measured by the multi-threaded
      // microbench in fascal_query_runtime.cuh::fascal_runtime_profile).
      // Do NOT multiply by N_CPU here — single-threaded throughput × N_CPU
      // is unrealistic for memory-bound predicates (DRAM bandwidth caps
      // scaling well below thread count).  See FaScalSQL §IV-D.
      double available_cpu = std::max(0.01, 1.0 - in.cpu_contention);
      double eff_cpu_tp = TP_AFP * 1e9 * available_cpu;
      (void)N_CPU;  // retained for documentation; see microbench note above
      double cost_cpu = 0.0;
      double R_afp_cumul = 1.0;  // R_j^AFP: fraction before j-th AFP op

      for (int i = 0; i < n_preds; i++) {
        if (pred_mask & (1 << i)) {
          // This predicate is on CPU: costs N * R_afp_cumul / throughput
          cost_cpu += N * R_afp_cumul / eff_cpu_tp;
          double sel = in.predicate_selectivities[i];
          R_afp_cumul *= std::max(0.0, std::min(1.0, sel));
        }
      }
      // BF probes on CPU: run after all CPU predicates, on surviving tuples.
      // Per FaScalSQL §IV-D: Cost_CPU = Σ N · R_j / TP_AFP.
      // A BF probe is more expensive than a simple predicate (hash + LLC read
      // on top of the DRAM column read), modeled as a constant multiplier.
      static constexpr double BF_COST_FACTOR = 50.0;
      if (!bf_off) {
        for (int j = 0; j < n_joins; j++) {
          if (bf_mask & (1 << j)) {
            cost_cpu += N * R_afp_cumul * BF_COST_FACTOR / eff_cpu_tp;
            double fp = in.join_bf_selectivities[j];
            R_afp_cumul *= std::max(0.0, std::min(1.0, fp));
          }
        }
      }

      // ── Cost_GPU: Σ_{j} N · (R_j^GPU · ColVal/TP_GPU + P_j^ODZC · Lat/(Sector/ColVal)) ──
      // Paper §IV-D: per-operation R_j^GPU tracks progressive filtering.
      //
      // GPU pipeline operations (in order):
      //   1. Bitmap load (if AFP active): 1 bit/row
      //   2. For each predicate NOT on CPU: GPU evaluates it, R decreases
      //   3. For each BF NOT on CPU: GPU probes BF, R decreases
      //   4. For each join: GPU probes HT (R may decrease)
      //   5. GPU-only ops (agg columns, output): sees final R

      double cost_gpu = 0.0;

      // Per-batch kernel launch overhead
      cost_gpu += (double)N_BATCHES * LAUNCH_US * 1e-6;  // µs → s

      // Fixed tile scan overhead: every tile is visited regardless of selectivity
      // (bitmap decode, thread scheduling, tile_summary check)
      cost_gpu += n_tiles * TILE_SCAN_NS * 1e-9;  // ns → s

      // Bitmap read cost (when any AFP is active)
      if (pred_mask != 0 || bf_mask != 0) {
        cost_gpu += N / 8.0 / (TP_GPU * 1e9);
      }

      // R_1^GPU = R_AFP (fraction surviving all CPU operations)
      double R_gpu = R_afp_cumul;

      // GPU predicate operations: predicates NOT on CPU
      // Column is in host memory → ODZC access (PCIe BW + page fault)
      for (int i = 0; i < n_preds; i++) {
        if (!(pred_mask & (1 << i))) {
          double p_odzc = 1.0 - std::pow(1.0 - R_gpu, vals_per_sector);
          cost_gpu += N * (R_gpu * COL_VAL / (TP_PCIE * 1e9)
                          + p_odzc * LAT * 1e-9 / vals_per_sector);
          double sel = in.predicate_selectivities[i];
          R_gpu *= std::max(0.0, std::min(1.0, sel));
        }
      }

      // GPU BF probe operations
      for (int j = 0; j < n_joins; j++) {
        if (!bf_off && !(bf_mask & (1 << j))) {
          // FK column in host memory → ODZC access
          double p_odzc = 1.0 - std::pow(1.0 - R_gpu, vals_per_sector);
          cost_gpu += N * (R_gpu * COL_VAL / (TP_PCIE * 1e9)
                          + p_odzc * LAT * 1e-9 / vals_per_sector);
          double fp = in.join_bf_selectivities[j];
          R_gpu *= std::max(0.0, std::min(1.0, fp));
        }
        // HT join probe: FK column ODZC + HT access in GPU memory
        {
          // FK column load (host → GPU via ODZC)
          double p_odzc = 1.0 - std::pow(1.0 - R_gpu, vals_per_sector);
          cost_gpu += N * (R_gpu * COL_VAL / (TP_PCIE * 1e9)
                          + p_odzc * LAT * 1e-9 / vals_per_sector);
          // HT access (GPU HBM, no ODZC)
          cost_gpu += N * R_gpu * COL_VAL / (TP_GPU * 1e9);
          // Join doesn't further filter (inner join selectivity ≈ 1 for FK-PK)
        }
      }

      // GPU-only operations (aggregation columns, output materialization)
      // These columns are in host memory → ODZC access
      for (int k = 0; k < in.num_gpu_only_ops; k++) {
        double p_odzc = 1.0 - std::pow(1.0 - R_gpu, vals_per_sector);
        cost_gpu += N * (R_gpu * COL_VAL / (TP_PCIE * 1e9)
                        + p_odzc * LAT * 1e-9 / vals_per_sector);
      }

      // ── Objective: P* = arg min max(Cost_GPU, Cost_CPU) ──
      // Per FaScalSQL §IV-D, costs are derived from per-predicate-type
      // microbench profiling captured into the CqoInput throughput fields
      // (tp_gpu_gbs, tp_afp_gips, lat_sector_ns).  The runtime is responsible
      // for populating those fields from a startup microbench — see
      // `fascal_cqo_microbench_*` in fascal_query_runtime.cuh.
      double combo_cost = std::max(cost_gpu, cost_cpu);

      if (combo_cost < best_cost) {
        best_cost = combo_cost;
        best.workload_dist_ratio = (cost_cpu > 1e-10)
            ? std::max(0.2, std::min(0.95, cost_cpu / (cost_cpu + cost_gpu)))
            : 0.5;
        best.predicate_on_cpu.assign(n_preds, false);
        for (int i = 0; i < n_preds; i++)
          best.predicate_on_cpu[i] = (bool)(pred_mask & (1 << i));
        best.join_bf_active.assign(n_joins, false);
        if (!bf_off) {
          for (int j = 0; j < n_joins; j++)
            best.join_bf_active[j] = (bool)(bf_mask & (1 << j));
        }
        best.bloom_off = bf_off;
        best.cost_gpu = cost_gpu;
        best.cost_cpu = cost_cpu;
        double cost_gpu_only = N * COL_VAL * (n_preds + 2 * n_joins + in.num_gpu_only_ops)
                               / (TP_GPU * 1e9);
        best.estimated_speedup = combo_cost > 0 ? cost_gpu_only / combo_cost : 1.0;
      }
    }
  }

  if (n_preds == 0 && n_joins == 0) {
    const double c = std::max(0.0, std::min(1.0, in.cpu_contention));
    best.workload_dist_ratio = std::max(0.2, std::min(0.95, 0.5 + 0.3 * c));
    best.rationale = "no predicates: contention-based routing";
  } else {
    best.rationale = "P* minimizes max(Cost_GPU, Cost_CPU) with per-op R_j^GPU";
  }

  return best;
}

}  // namespace optimizer
}  // namespace fascal
