/**
 * FaScalSQL GPU Kernel Boilerplate Macros
 * Reduces per-kernel boilerplate for tile setup, bitmap loading, and early exit.
 * Include AFTER fascal_generated_common.cuh (needs BlockLoadBitmap).
 */
#pragma once

// ---- Tile setup + bitmap load + early exit ----
// Replaces ~25 lines at the start of every result-pipeline GPU kernel.
// Declares: global_tile_id, tile_offset, num_tile_items, selection_flags[]
// Requires in scope: num_tuples, batch_offset, seed_bitmap, tile_summary
#define FASCAL_KERNEL_PROLOGUE(BT, IPT) \
    int global_tile_id = batch_offset / (BT * IPT) + blockIdx.x; \
    if (tile_summary[global_tile_id] == 0) return; \
    int tile_offset = batch_offset + blockIdx.x * (BT * IPT); \
    int num_tile_items = BT * IPT; \
    if ((tile_offset + BT * IPT) > num_tuples) \
        num_tile_items = num_tuples - tile_offset; \
    int selection_flags[IPT]; \
    BlockLoadBitmap<BT, IPT>(seed_bitmap, selection_flags, tile_offset, num_tuples); \
    __syncthreads()

// ---- Check if any thread has alive items, break if none ----
// Use inside do { ... } while(0) blocks between expensive operations.
#define FASCAL_CHECK_ALIVE(IPT) do { \
    int _fascal_alive = 0; \
    _Pragma("unroll") \
    for (int _I = 0; _I < (IPT); ++_I) _fascal_alive |= selection_flags[_I]; \
    if (!_fascal_alive) break; \
} while(0)

// ---- Initial alive check (for the if(alive) do { ... } while(0) pattern) ----
// Returns true if any item survived AFP bitmap filtering.
#define FASCAL_ANY_ALIVE(IPT) [&]() -> bool { \
    int _a = 0; \
    _Pragma("unroll") \
    for (int _I = 0; _I < (IPT); ++_I) _a |= selection_flags[_I]; \
    return _a != 0; \
}()

// ---- Build kernel prologue (dimension sub-pipeline, whole-table) ----
// Declares: tile_offset, num_tile_items, selection_flags[]
// Handles optional AFP prefilter bitmap.
// Requires in scope: num_tuples, d_prefilter, d_tile_summary
#define FASCAL_BUILD_KERNEL_PROLOGUE(BT, IPT) \
    if (d_tile_summary && d_tile_summary[blockIdx.x] == 0) return; \
    int tile_offset = blockIdx.x * (BT * IPT); \
    int num_tile_items = BT * IPT; \
    if ((tile_offset + BT * IPT) > num_tuples) \
        num_tile_items = num_tuples - tile_offset; \
    int selection_flags[IPT]; \
    if (d_prefilter) { \
        BlockLoadBitmap<BT, IPT>(d_prefilter, selection_flags, tile_offset, num_tuples); \
    } else { \
        _Pragma("unroll") \
        for (int _I = 0; _I < IPT; ++_I) \
            selection_flags[_I] = ((threadIdx.x + (BT * _I)) < num_tile_items) ? 1 : 0; \
    } \
    { int _a = 0; \
      _Pragma("unroll") \
      for (int _I = 0; _I < IPT; ++_I) _a |= selection_flags[_I]; \
      if (!_a) return; }

// ---- Build kernel prologue (morsel-batched, SSB-style AFP overlap) ----
// Same as above but uses batch_offset to compute the global tile id.
// Required for build kernels launched one morsel at a time on a CUDA stream.
// Requires in scope: num_tuples, d_prefilter, d_tile_summary, batch_offset.
#define FASCAL_BUILD_KERNEL_PROLOGUE_BATCHED(BT, IPT) \
    int global_tile_id = batch_offset / (BT * IPT) + blockIdx.x; \
    if (d_tile_summary && d_tile_summary[global_tile_id] == 0) return; \
    int tile_offset = batch_offset + blockIdx.x * (BT * IPT); \
    int num_tile_items = BT * IPT; \
    if ((tile_offset + BT * IPT) > num_tuples) \
        num_tile_items = num_tuples - tile_offset; \
    int selection_flags[IPT]; \
    if (d_prefilter) { \
        BlockLoadBitmap<BT, IPT>(d_prefilter, selection_flags, tile_offset, num_tuples); \
    } else { \
        _Pragma("unroll") \
        for (int _I = 0; _I < IPT; ++_I) \
            selection_flags[_I] = ((threadIdx.x + (BT * _I)) < num_tile_items) ? 1 : 0; \
    } \
    { int _a = 0; \
      _Pragma("unroll") \
      for (int _I = 0; _I < IPT; ++_I) _a |= selection_flags[_I]; \
      if (!_a) return; }
