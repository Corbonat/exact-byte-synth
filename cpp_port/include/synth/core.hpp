#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ankerl/unordered_dense.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace synth {

inline constexpr int WIDTH = 8;
inline constexpr std::uint8_t MASK = 0xFF;
inline constexpr int DOMAIN_SIZE = 256;
inline constexpr std::size_t SIGNATURE_SIZE = 24;

using Signature24 = std::array<std::uint8_t, SIGNATURE_SIZE>;
using Semantics256 = std::array<std::uint8_t, DOMAIN_SIZE>;
using Bytes = std::vector<std::uint8_t>;
using OracleFn = std::function<std::uint8_t(std::uint8_t)>;

template <std::size_t N>
struct FixedBytesHash {
    using is_avalanching = void;
    std::size_t operator()(const std::array<std::uint8_t, N>& bytes) const noexcept;
};

struct VectorBytesHash {
    using is_avalanching = void;
    std::size_t operator()(const Bytes& bytes) const noexcept;
};

enum class ExprKind {
    Var,
    Const,
    MulConst,
    Add,
    Sub,
    Xor,
    ShlConst,
    ShrConst,
    Square,
    Mul,
    Sin8,
    Cos8,
};

struct Expr;
using ExprHandle = const Expr*;
using ExprId = std::uint32_t;

struct Expr {
    ExprKind kind{};
    ExprId id{0};
    ExprHandle left{};
    ExprHandle right{};
    std::uint16_t value{0};
    int cost{0};
    std::string shape_key;
    std::bitset<256> literal_const_values;
    int square_count{0};
    int nonlinear_count{0};

    [[nodiscard]] std::uint8_t eval(std::uint8_t x) const;
    [[nodiscard]] std::string to_source() const;
    [[nodiscard]] std::string to_verilog() const;
    [[nodiscard]] std::string to_verilog8() const;
};

struct BenchmarkCase {
    std::string name;
    std::string category;
    std::string description;
    OracleFn oracle;
    bool expected_found{false};
};

struct NonlinearConfig {
    std::string mode;
    int max_nonlinear_count{1};
    bool allow_mul_of_nonlinear{false};
};

struct Candidate {
    ExprHandle expr{};
    Signature24 sig{};
    Semantics256 sem{};
};

struct LevelStats {
    std::uint64_t generated{0};
    std::uint64_t rule_pruned{0};
    std::uint64_t shape_pruned{0};
    std::uint64_t sig_pruned{0};
    std::uint64_t sig_bucket_kept{0};
    std::uint64_t sig_bucket_replaced{0};
    std::uint64_t full_eval{0};
    std::uint64_t sem_pruned{0};
    std::uint64_t accepted{0};
    std::uint64_t generated_mul_candidates{0};
    std::uint64_t accepted_mul_candidates{0};
    std::uint64_t shape_pruned_mul{0};
    std::uint64_t sem_pruned_mul{0};
    std::uint64_t max_nonlinear_depth_seen{0};
};

struct BucketSummary {
    std::uint64_t max_bucket_count{0};
    std::uint64_t heavy_bucket_observations{0};
    std::uint64_t heavy_bucket_size_total{0};

    [[nodiscard]] double avg_bucket_count_on_heavy_levels() const;
};

struct PromotionRound {
    int round_index{0};
    int const_set_size_before{0};
    int const_set_size_after{0};
    std::vector<int> derived_consts_discovered;
    std::vector<int> derived_consts_promoted;
    std::optional<std::string> found_expr;
};

struct SampleSet {
    std::vector<std::uint8_t> points;
    Bytes values;

    static SampleSet from_oracle(const OracleFn& oracle, const std::vector<std::uint8_t>& points);
    [[nodiscard]] SampleSet with_point(const OracleFn& oracle, std::uint8_t point) const;
};

struct CegisRound {
    int round_index{0};
    int sample_size{0};
    std::vector<std::uint8_t> sample_points;
    std::optional<std::string> backend;
    std::optional<std::string> expression;
    std::optional<int> counterexample;
    bool full_verify_passed{false};
};

using LevelStatsMap = std::map<int, LevelStats>;

struct SearchResult {
    std::string backend;
    ExprHandle expr{};
    LevelStatsMap stats;
    std::vector<PromotionRound> promotion_rounds;
    std::optional<BucketSummary> bucket_summary;
    bool fallback_triggered{false};
};

struct SearchAttempt {
    std::optional<SearchResult> result;
    std::vector<std::string> attempted_backends;
    std::vector<std::pair<std::string, double>> backend_timings_s;
    std::optional<int> solved_by_backend_index;
    LevelStatsMap signature_stats;
    std::vector<PromotionRound> signature_promotion_rounds;
    std::optional<BucketSummary> signature_bucket_summary;
    LevelStatsMap fallback_stats;
    std::vector<PromotionRound> fallback_promotion_rounds;
    std::optional<BucketSummary> fallback_bucket_summary;
    std::vector<CegisRound> cegis_rounds;

    [[nodiscard]] bool fallback_triggered() const;
    [[nodiscard]] bool solved_by_first_backend() const;
};

using SigBucketLimitFn = std::function<int(int)>;

extern const int MAX_COST;
extern const int SIGNATURE_PRUNE_COST;
extern const int MAX_PROMOTION_ROUNDS;
extern const char* const DEFAULT_NONLINEAR_MODE;

[[nodiscard]] const std::vector<int>& base_const_set();
[[nodiscard]] const std::vector<int>& shift_set();
[[nodiscard]] const std::array<int, SIGNATURE_SIZE>& sig_points();
[[nodiscard]] const std::vector<std::uint8_t>& initial_cegis_points();
[[nodiscard]] const std::map<std::string, NonlinearConfig>& nonlinear_configs();
[[nodiscard]] const NonlinearConfig& nonlinear_config(std::string_view mode);

[[nodiscard]] ExprHandle var_expr();
[[nodiscard]] ExprHandle const_expr(int value);
[[nodiscard]] bool expr_equal(const ExprHandle& left, const ExprHandle& right);

[[nodiscard]] ExprHandle make_mul_const(const ExprHandle& expr, int constant);
[[nodiscard]] ExprHandle make_shl_const(const ExprHandle& expr, int shift);
[[nodiscard]] ExprHandle make_shr_const(const ExprHandle& expr, int shift);
[[nodiscard]] ExprHandle make_square(const ExprHandle& expr, int max_nonlinear_count = 1);
[[nodiscard]] ExprHandle make_mul(
    const ExprHandle& left,
    const ExprHandle& right,
    bool allow_nonlinear_operands = false,
    int max_nonlinear_count = 1
);
[[nodiscard]] ExprHandle make_sin8(const ExprHandle& expr);
[[nodiscard]] ExprHandle make_cos8(const ExprHandle& expr);
[[nodiscard]] ExprHandle make_add(const ExprHandle& left, const ExprHandle& right);
[[nodiscard]] ExprHandle make_sub(const ExprHandle& left, const ExprHandle& right);
[[nodiscard]] ExprHandle make_xor(const ExprHandle& left, const ExprHandle& right);

[[nodiscard]] std::uint8_t sin8_value(std::uint8_t value);
[[nodiscard]] std::uint8_t cos8_value(std::uint8_t value);
[[nodiscard]] Semantics256 collect_target();
[[nodiscard]] Semantics256 build_target(const OracleFn& oracle);
[[nodiscard]] Semantics256 evaluate_expr(const ExprHandle& expr);
[[nodiscard]] std::optional<int> find_counterexample(const ExprHandle& expr, const OracleFn& oracle);
[[nodiscard]] bool better(const ExprHandle& newer, const ExprHandle& older);

template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> mul_const_values(const std::array<std::uint8_t, N>& values, int constant);
template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> shl_values(const std::array<std::uint8_t, N>& values, int shift);
template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> shr_values(const std::array<std::uint8_t, N>& values, int shift);
template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> add_values(const std::array<std::uint8_t, N>& left, const std::array<std::uint8_t, N>& right);
template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> sub_values(const std::array<std::uint8_t, N>& left, const std::array<std::uint8_t, N>& right);
template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> xor_values(const std::array<std::uint8_t, N>& left, const std::array<std::uint8_t, N>& right);
template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> square_values(const std::array<std::uint8_t, N>& values);
template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> mul_values(const std::array<std::uint8_t, N>& left, const std::array<std::uint8_t, N>& right);
template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> sin8_values(const std::array<std::uint8_t, N>& values);
template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> cos8_values(const std::array<std::uint8_t, N>& values);
[[nodiscard]] Bytes mul_const_values(const Bytes& values, int constant);
[[nodiscard]] Bytes shl_values(const Bytes& values, int shift);
[[nodiscard]] Bytes shr_values(const Bytes& values, int shift);
[[nodiscard]] Bytes add_values(const Bytes& left, const Bytes& right);
[[nodiscard]] Bytes sub_values(const Bytes& left, const Bytes& right);
[[nodiscard]] Bytes xor_values(const Bytes& left, const Bytes& right);
[[nodiscard]] Bytes square_values(const Bytes& values);
[[nodiscard]] Bytes mul_values(const Bytes& left, const Bytes& right);
[[nodiscard]] Bytes sin8_values(const Bytes& values);
[[nodiscard]] Bytes cos8_values(const Bytes& values);
[[nodiscard]] Signature24 target_signature(const Semantics256& target);

[[nodiscard]] int top1_sig_bucket_limit(int cost);
[[nodiscard]] int selective_sig_bucket_limit(int cost);

void merge_level_stats(LevelStats& dst, const LevelStats& src);
[[nodiscard]] BucketSummary merge_bucket_summaries(const BucketSummary& dst, const BucketSummary& src);

[[nodiscard]] SearchAttempt run_search_attempt(
    const std::optional<Semantics256>& target = std::nullopt,
    int max_cost = 6,
    bool log_progress = true,
    SigBucketLimitFn sig_bucket_limit_fn = top1_sig_bucket_limit,
    std::string_view nonlinear_mode = DEFAULT_NONLINEAR_MODE,
    bool enable_periodic = false
);

[[nodiscard]] SearchAttempt run_sample_search_attempt(
    const SampleSet& sample_set,
    int max_cost = 6,
    bool log_progress = true,
    std::string_view nonlinear_mode = DEFAULT_NONLINEAR_MODE,
    bool enable_periodic = false
);

[[nodiscard]] SearchAttempt synthesize_cegis(
    const OracleFn& oracle,
    const std::vector<std::uint8_t>& initial_points = initial_cegis_points(),
    int max_rounds = 32,
    int max_cost = 6,
    bool log_progress = true,
    std::string_view nonlinear_mode = DEFAULT_NONLINEAR_MODE,
    bool enable_periodic = false
);

[[nodiscard]] std::optional<SearchResult> synthesize(
    const Semantics256& target,
    int max_cost = 6,
    bool log_progress = true,
    SigBucketLimitFn sig_bucket_limit_fn = top1_sig_bucket_limit,
    std::string_view nonlinear_mode = DEFAULT_NONLINEAR_MODE,
    bool enable_periodic = false
);

[[nodiscard]] std::pair<bool, std::optional<int>> verify_expression(const ExprHandle& expr, const Semantics256& target);
[[nodiscard]] std::string emit_verilog(const ExprHandle& expr, std::string_view module_name = "generated_function");

[[nodiscard]] const std::vector<BenchmarkCase>& benchmarks();
[[nodiscard]] const BenchmarkCase* find_benchmark(std::string_view name);
[[nodiscard]] std::vector<std::string> recommended_demo_cases();

[[nodiscard]] std::uint8_t current_oracle(std::uint8_t x);

[[nodiscard]] std::string json_escape(std::string_view value);
void write_text_file(const std::filesystem::path& path, std::string_view content);
void ensure_directory(const std::filesystem::path& path);

template <std::size_t N>
inline std::size_t FixedBytesHash<N>::operator()(const std::array<std::uint8_t, N>& bytes) const noexcept {
    return static_cast<std::size_t>(
        ankerl::unordered_dense::detail::wyhash::hash(bytes.data(), N));
}

template <std::size_t N>
inline std::array<std::uint8_t, N> mul_const_values(const std::array<std::uint8_t, N>& values, int constant) {
    std::array<std::uint8_t, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = static_cast<std::uint8_t>((values[index] * (constant & MASK)) & MASK);
    }
    return result;
}

template <std::size_t N>
inline std::array<std::uint8_t, N> shl_values(const std::array<std::uint8_t, N>& values, int shift) {
    std::array<std::uint8_t, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = static_cast<std::uint8_t>((values[index] << shift) & MASK);
    }
    return result;
}

template <std::size_t N>
inline std::array<std::uint8_t, N> shr_values(const std::array<std::uint8_t, N>& values, int shift) {
    std::array<std::uint8_t, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = static_cast<std::uint8_t>((values[index] >> shift) & MASK);
    }
    return result;
}

template <std::size_t N>
inline std::array<std::uint8_t, N> add_values(const std::array<std::uint8_t, N>& left, const std::array<std::uint8_t, N>& right) {
    std::array<std::uint8_t, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = static_cast<std::uint8_t>((left[index] + right[index]) & MASK);
    }
    return result;
}

template <std::size_t N>
inline std::array<std::uint8_t, N> sub_values(const std::array<std::uint8_t, N>& left, const std::array<std::uint8_t, N>& right) {
    std::array<std::uint8_t, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = static_cast<std::uint8_t>((left[index] - right[index]) & MASK);
    }
    return result;
}

template <std::size_t N>
inline std::array<std::uint8_t, N> xor_values(const std::array<std::uint8_t, N>& left, const std::array<std::uint8_t, N>& right) {
    std::array<std::uint8_t, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = static_cast<std::uint8_t>((left[index] ^ right[index]) & MASK);
    }
    return result;
}

template <std::size_t N>
inline std::array<std::uint8_t, N> square_values(const std::array<std::uint8_t, N>& values) {
    std::array<std::uint8_t, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = static_cast<std::uint8_t>((values[index] * values[index]) & MASK);
    }
    return result;
}

template <std::size_t N>
inline std::array<std::uint8_t, N> mul_values(const std::array<std::uint8_t, N>& left, const std::array<std::uint8_t, N>& right) {
    std::array<std::uint8_t, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = static_cast<std::uint8_t>((left[index] * right[index]) & MASK);
    }
    return result;
}

template <std::size_t N>
inline std::array<std::uint8_t, N> sin8_values(const std::array<std::uint8_t, N>& values) {
    std::array<std::uint8_t, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = sin8_value(values[index]);
    }
    return result;
}

template <std::size_t N>
inline std::array<std::uint8_t, N> cos8_values(const std::array<std::uint8_t, N>& values) {
    std::array<std::uint8_t, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = cos8_value(values[index]);
    }
    return result;
}

#if defined(__AVX2__)

// AVX2 specializations for Semantics256 (256 unsigned bytes = 8 * 32 byte ymm registers).
// Uses unaligned load/store because std::array<uint8_t, 256> has 1-byte alignment.

namespace simd_detail {

inline __m256i mul_u8_epi8(__m256i a, __m256i b) noexcept {
    const __m256i lo_mask = _mm256_set1_epi16(0x00FF);
    const __m256i a_lo = _mm256_and_si256(a, lo_mask);
    const __m256i b_lo = _mm256_and_si256(b, lo_mask);
    const __m256i a_hi = _mm256_srli_epi16(a, 8);
    const __m256i b_hi = _mm256_srli_epi16(b, 8);
    const __m256i r_lo = _mm256_mullo_epi16(a_lo, b_lo);
    const __m256i r_hi = _mm256_mullo_epi16(a_hi, b_hi);
    return _mm256_or_si256(
        _mm256_and_si256(r_lo, lo_mask),
        _mm256_slli_epi16(_mm256_and_si256(r_hi, lo_mask), 8));
}

inline __m256i shl_u8_epi8(__m256i a, int shift) noexcept {
    const __m256i mask = _mm256_set1_epi8(static_cast<char>(0xFFu << shift));
    return _mm256_and_si256(_mm256_slli_epi16(a, shift), mask);
}

inline __m256i shr_u8_epi8(__m256i a, int shift) noexcept {
    const __m256i mask = _mm256_set1_epi8(static_cast<char>(0xFFu >> shift));
    return _mm256_and_si256(_mm256_srli_epi16(a, shift), mask);
}

}  // namespace simd_detail

template <>
inline std::array<std::uint8_t, 256> add_values<256>(
    const std::array<std::uint8_t, 256>& left,
    const std::array<std::uint8_t, 256>& right) {
    std::array<std::uint8_t, 256> result;
    for (std::size_t i = 0; i < 256; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left.data() + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right.data() + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), _mm256_add_epi8(a, b));
    }
    return result;
}

template <>
inline std::array<std::uint8_t, 256> sub_values<256>(
    const std::array<std::uint8_t, 256>& left,
    const std::array<std::uint8_t, 256>& right) {
    std::array<std::uint8_t, 256> result;
    for (std::size_t i = 0; i < 256; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left.data() + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right.data() + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), _mm256_sub_epi8(a, b));
    }
    return result;
}

template <>
inline std::array<std::uint8_t, 256> xor_values<256>(
    const std::array<std::uint8_t, 256>& left,
    const std::array<std::uint8_t, 256>& right) {
    std::array<std::uint8_t, 256> result;
    for (std::size_t i = 0; i < 256; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left.data() + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right.data() + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), _mm256_xor_si256(a, b));
    }
    return result;
}

template <>
inline std::array<std::uint8_t, 256> mul_values<256>(
    const std::array<std::uint8_t, 256>& left,
    const std::array<std::uint8_t, 256>& right) {
    std::array<std::uint8_t, 256> result;
    for (std::size_t i = 0; i < 256; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left.data() + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right.data() + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), simd_detail::mul_u8_epi8(a, b));
    }
    return result;
}

template <>
inline std::array<std::uint8_t, 256> square_values<256>(
    const std::array<std::uint8_t, 256>& values) {
    std::array<std::uint8_t, 256> result;
    for (std::size_t i = 0; i < 256; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values.data() + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), simd_detail::mul_u8_epi8(a, a));
    }
    return result;
}

template <>
inline std::array<std::uint8_t, 256> mul_const_values<256>(
    const std::array<std::uint8_t, 256>& values, int constant) {
    std::array<std::uint8_t, 256> result;
    const __m256i b = _mm256_set1_epi8(static_cast<char>(constant & MASK));
    for (std::size_t i = 0; i < 256; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values.data() + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), simd_detail::mul_u8_epi8(a, b));
    }
    return result;
}

template <>
inline std::array<std::uint8_t, 256> shl_values<256>(
    const std::array<std::uint8_t, 256>& values, int shift) {
    std::array<std::uint8_t, 256> result;
    for (std::size_t i = 0; i < 256; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values.data() + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), simd_detail::shl_u8_epi8(a, shift));
    }
    return result;
}

template <>
inline std::array<std::uint8_t, 256> shr_values<256>(
    const std::array<std::uint8_t, 256>& values, int shift) {
    std::array<std::uint8_t, 256> result;
    for (std::size_t i = 0; i < 256; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values.data() + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), simd_detail::shr_u8_epi8(a, shift));
    }
    return result;
}

#endif  // __AVX2__

}  // namespace synth
