/**
 * FaScalSQL CPU Predicate AVX2 Helpers
 * Reusable building blocks for CPU-side AFP predicate evaluation.
 * Each function implements one "pass" over a bitmap, applying a predicate to 8 rows at a time.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace fascal {
namespace cpu_pred {

// ---- Initialize bitmap to all-ones for a morsel ----
static inline void init_bitmap(uint8_t* bitmap8, int byte_start, int num_bytes, int elem_cnt) {
    std::memset(&bitmap8[byte_start], 0xFF, num_bytes);
    int tail = elem_cnt & 7;
    if (tail) bitmap8[byte_start + num_bytes - 1] &= (uint8_t)((1u << tail) - 1);
}

// ---- Range predicate pass: col BETWEEN lo AND hi ----
// Supports partial range: check_ge=false means no lower bound, check_lt_strict means < instead of <=
static inline void avx2_range_pass(
    uint8_t* bitmap8, int byte_start, const int* col, int morsel_offset, int elem_cnt,
    int lo, int hi, bool check_ge, bool check_le, bool hi_strict = false)
{
    const __m256i avx_all_ones = _mm256_set1_epi32(-1);
    const __m256i v_lo = _mm256_set1_epi32(lo);
    const __m256i v_hi = _mm256_set1_epi32(hi);

    for (int i = 0; i + 7 < elem_cnt; i += 8) {
        int bidx = byte_start + (i >> 3);
        if (!bitmap8[bidx]) continue;
        int idx = morsel_offset + i;
        __m256i v = _mm256_loadu_si256((const __m256i*)(col + idx));
        uint8_t pass8 = 0xFF;
        if (check_ge) {
            __m256i ge = _mm256_or_si256(_mm256_cmpgt_epi32(v, v_lo), _mm256_cmpeq_epi32(v, v_lo));
            pass8 &= (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(ge));
        }
        if (check_le) {
            __m256i le;
            if (hi_strict)
                le = _mm256_cmpgt_epi32(v_hi, v);  // v < hi
            else
                le = _mm256_xor_si256(_mm256_cmpgt_epi32(v, v_hi), avx_all_ones);  // v <= hi
            pass8 &= (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(le));
        }
        bitmap8[bidx] &= pass8;
    }
    // Scalar tail
    int tail_start = elem_cnt & ~7;
    if (tail_start < elem_cnt) {
        int bidx = byte_start + (tail_start >> 3);
        uint8_t bv = bitmap8[bidx];
        if (bv) {
            for (int i = tail_start; i < elem_cnt; ++i) {
                int bit = i & 7;
                if (!(bv & (1u << bit))) continue;
                int val = col[morsel_offset + i];
                bool pass = true;
                if (check_ge) pass = pass && (val >= lo);
                if (check_le) pass = pass && (hi_strict ? (val < hi) : (val <= hi));
                if (!pass) bv &= ~(1u << bit);
            }
            bitmap8[bidx] = bv;
        }
    }
}

// ---- Float range predicate pass: float col BETWEEN lo AND hi ----
// Mirrors avx2_range_pass but operates on float columns with float comparisons.
static inline void avx2_range_pass_float(
    uint8_t* bitmap8, int byte_start, const float* col, int morsel_offset, int elem_cnt,
    float lo, float hi, bool check_ge, bool check_le, bool hi_strict = false)
{
    const __m256 v_lo = _mm256_set1_ps(lo);
    const __m256 v_hi = _mm256_set1_ps(hi);

    for (int i = 0; i + 7 < elem_cnt; i += 8) {
        int bidx = byte_start + (i >> 3);
        if (!bitmap8[bidx]) continue;
        int idx = morsel_offset + i;
        __m256 v = _mm256_loadu_ps(col + idx);
        uint8_t pass8 = 0xFF;
        if (check_ge) {
            __m256 ge = _mm256_cmp_ps(v, v_lo, _CMP_GE_OS);
            pass8 &= (uint8_t)_mm256_movemask_ps(ge);
        }
        if (check_le) {
            __m256 le;
            if (hi_strict)
                le = _mm256_cmp_ps(v, v_hi, _CMP_LT_OS);
            else
                le = _mm256_cmp_ps(v, v_hi, _CMP_LE_OS);
            pass8 &= (uint8_t)_mm256_movemask_ps(le);
        }
        bitmap8[bidx] &= pass8;
    }
    // Scalar tail
    int tail_start = elem_cnt & ~7;
    if (tail_start < elem_cnt) {
        int bidx = byte_start + (tail_start >> 3);
        uint8_t bv = bitmap8[bidx];
        if (bv) {
            for (int i = tail_start; i < elem_cnt; ++i) {
                int bit = i & 7;
                if (!(bv & (1u << bit))) continue;
                float val = col[morsel_offset + i];
                bool pass = true;
                if (check_ge) pass = pass && (val >= lo);
                if (check_le) pass = pass && (hi_strict ? (val < hi) : (val <= hi));
                if (!pass) bv &= ~(1u << bit);
            }
            bitmap8[bidx] = bv;
        }
    }
}

// ---- Equality predicate pass: col == value ----
static inline void avx2_eq_pass(
    uint8_t* bitmap8, int byte_start, const int* col, int morsel_offset, int elem_cnt, int value)
{
    const __m256i v_val = _mm256_set1_epi32(value);
    for (int i = 0; i + 7 < elem_cnt; i += 8) {
        int bidx = byte_start + (i >> 3);
        if (!bitmap8[bidx]) continue;
        int idx = morsel_offset + i;
        __m256i v = _mm256_loadu_si256((const __m256i*)(col + idx));
        __m256i eq = _mm256_cmpeq_epi32(v, v_val);
        bitmap8[bidx] &= (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(eq));
    }
    // Scalar tail
    int tail_start = elem_cnt & ~7;
    if (tail_start < elem_cnt) {
        int bidx = byte_start + (tail_start >> 3);
        uint8_t bv = bitmap8[bidx];
        if (bv) {
            for (int i = tail_start; i < elem_cnt; ++i) {
                int bit = i & 7;
                if (!(bv & (1u << bit))) continue;
                if (col[morsel_offset + i] != value) bv &= ~(1u << bit);
            }
            bitmap8[bidx] = bv;
        }
    }
}

// ---- OR equality pass: col == v1 OR col == v2 ----
static inline void avx2_or_eq2_pass(
    uint8_t* bitmap8, int byte_start, const int* col, int morsel_offset, int elem_cnt,
    int v1, int v2)
{
    const __m256i vv1 = _mm256_set1_epi32(v1);
    const __m256i vv2 = _mm256_set1_epi32(v2);
    for (int i = 0; i + 7 < elem_cnt; i += 8) {
        int bidx = byte_start + (i >> 3);
        if (!bitmap8[bidx]) continue;
        int idx = morsel_offset + i;
        __m256i v = _mm256_loadu_si256((const __m256i*)(col + idx));
        __m256i m = _mm256_or_si256(_mm256_cmpeq_epi32(v, vv1), _mm256_cmpeq_epi32(v, vv2));
        bitmap8[bidx] &= (uint8_t)_mm256_movemask_ps(_mm256_castsi256_ps(m));
    }
    // Scalar tail
    int tail_start = elem_cnt & ~7;
    if (tail_start < elem_cnt) {
        int bidx = byte_start + (tail_start >> 3);
        uint8_t bv = bitmap8[bidx];
        if (bv) {
            for (int i = tail_start; i < elem_cnt; ++i) {
                int bit = i & 7;
                if (!(bv & (1u << bit))) continue;
                int val = col[morsel_offset + i];
                if (val != v1 && val != v2) bv &= ~(1u << bit);
            }
            bitmap8[bidx] = bv;
        }
    }
}

// ---- Bloom filter probe pass ----
static inline void bloom_probe_pass(
    uint8_t* bitmap8, int byte_start, const int* fk_col, int morsel_offset, int elem_cnt,
    fascal::runtime::BloomFilter* bf)
{
    if (!bf || !bf->bits) return;
    for (int i = 0; i + 7 < elem_cnt; i += 8) {
        int bidx = byte_start + (i >> 3);
        if (!bitmap8[bidx]) continue;
        int idx = morsel_offset + i;
        __m256i v = _mm256_loadu_si256((const __m256i*)(fk_col + idx));
        int keys[8];
        _mm256_storeu_si256((__m256i*)keys, v);
        uint8_t pass8 = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            pass8 |= (bf->query(keys[j]) ? 1 : 0) << j;
        bitmap8[bidx] &= pass8;
    }
    // Scalar tail
    int tail_start = elem_cnt & ~7;
    if (tail_start < elem_cnt) {
        int bidx = byte_start + (tail_start >> 3);
        uint8_t bv = bitmap8[bidx];
        if (bv) {
            for (int i = tail_start; i < elem_cnt; ++i) {
                int bit = i & 7;
                if (!(bv & (1u << bit))) continue;
                if (!bf->query(fk_col[morsel_offset + i])) bv &= ~(1u << bit);
            }
            bitmap8[bidx] = bv;
        }
    }
}

// ---- Generic CPU prefilter wrapper (sequential + pipelined) ----
template <typename PredFn>
static inline void run_prefilter(
    int batch_count, uint32_t* bitmap, uint8_t* tile_summary,
    PredFn pred_fn, int batch_offset = 0, int total_tuples = 0)
{
    if (total_tuples == 0) total_tuples = batch_count;
    if (batch_offset == 0 && batch_count == total_tuples) {
        fascal_cpu_prefilter_run(batch_count, bitmap, tile_summary,
            [&](int offset, int cnt) { pred_fn(offset, cnt); });
    } else {
        fascal_cpu_prefilter_run_batch(total_tuples, batch_offset, batch_count,
            bitmap, tile_summary,
            [&](int offset, int cnt) { pred_fn(offset, cnt); });
    }
}

} // namespace cpu_pred
} // namespace fascal
