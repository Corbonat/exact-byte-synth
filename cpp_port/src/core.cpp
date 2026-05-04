#include "synth/core.hpp"

#include "ankerl/unordered_dense.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace synth {

namespace {

template <typename K, typename V, typename H = ankerl::unordered_dense::hash<K>>
using FlatMap = ankerl::unordered_dense::map<K, V, H>;
template <typename K, typename H = ankerl::unordered_dense::hash<K>>
using FlatSet = ankerl::unordered_dense::set<K, H>;

using SignatureBucket = std::vector<ExprHandle>;

struct SampleCandidate {
    ExprHandle expr{};
    Bytes sem;
};

std::optional<int> const_value(const ExprHandle& expr) {
    if (expr && expr->kind == ExprKind::Const) {
        return static_cast<int>(expr->value & MASK);
    }
    return std::nullopt;
}

bool is_zero(const ExprHandle& expr) {
    const auto value = const_value(expr);
    return value.has_value() && value.value() == 0;
}

std::uint8_t rounded_wave_value(std::uint8_t value, bool cosine) {
    constexpr double kPi = 3.1415926535897932384626433832795;
    const double phase = (2.0 * kPi * static_cast<double>(value)) / static_cast<double>(DOMAIN_SIZE);
    const double wave = cosine ? std::cos(phase) : std::sin(phase);
    const int rounded = static_cast<int>(std::lround(127.5 + (127.5 * wave)));
    return static_cast<std::uint8_t>(std::clamp(rounded, 0, static_cast<int>(MASK)));
}

const std::array<std::uint8_t, DOMAIN_SIZE>& sin8_table() {
    static const std::array<std::uint8_t, DOMAIN_SIZE> values = [] {
        std::array<std::uint8_t, DOMAIN_SIZE> table{};
        for (int value = 0; value < DOMAIN_SIZE; ++value) {
            table[static_cast<std::size_t>(value)] = rounded_wave_value(static_cast<std::uint8_t>(value), false);
        }
        return table;
    }();
    return values;
}

const std::array<std::uint8_t, DOMAIN_SIZE>& cos8_table() {
    static const std::array<std::uint8_t, DOMAIN_SIZE> values = [] {
        std::array<std::uint8_t, DOMAIN_SIZE> table{};
        for (int value = 0; value < DOMAIN_SIZE; ++value) {
            table[static_cast<std::size_t>(value)] = rounded_wave_value(static_cast<std::uint8_t>(value), true);
        }
        return table;
    }();
    return values;
}

std::tuple<std::string, int> canonical_expr_sort_key(const ExprHandle& expr) {
    return {expr->shape_key, expr->cost};
}

std::tuple<int, int, std::string> expr_rank_key(const ExprHandle& expr) {
    return {expr->cost, static_cast<int>(expr->shape_key.size()), expr->shape_key};
}

struct InternKey {
    ExprKind kind;
    ExprId left_id;
    ExprId right_id;
    std::uint16_t value;

    bool operator==(const InternKey& other) const noexcept {
        return kind == other.kind
            && left_id == other.left_id
            && right_id == other.right_id
            && value == other.value;
    }
};

struct InternKeyHash {
    using is_avalanching = void;
    std::size_t operator()(const InternKey& key) const noexcept {
        std::array<std::uint64_t, 2> buf{};
        buf[0] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.kind)) << 32)
               | static_cast<std::uint64_t>(key.value);
        buf[1] = (static_cast<std::uint64_t>(key.left_id) << 32)
               | static_cast<std::uint64_t>(key.right_id);
        return static_cast<std::size_t>(
            ankerl::unordered_dense::detail::wyhash::hash(buf.data(), sizeof(buf)));
    }
};

std::deque<Expr>& expr_arena() {
    static std::deque<Expr> arena;
    return arena;
}

FlatMap<InternKey, ExprHandle, InternKeyHash>& intern_pool() {
    static FlatMap<InternKey, ExprHandle, InternKeyHash> pool = [] {
        FlatMap<InternKey, ExprHandle, InternKeyHash> p;
        p.reserve(1u << 20);
        return p;
    }();
    return pool;
}

// Reserve hint table: typical unique-semantics counts per cost level for 8-bit, max_cost<=6.
// Sized to minimize rehashes without exploding memory for small cases.
constexpr std::size_t level_reserve_hint(int cost) {
    switch (cost) {
    case 0: return 0;
    case 1: return 64;
    case 2: return 512;
    case 3: return 8192;
    case 4: return 131072;
    case 5: return 524288;
    default: return 1048576;
    }
}

ExprId next_expr_id() {
    static ExprId counter{0};
    return ++counter;
}

ExprHandle make_node(
    ExprKind kind,
    const ExprHandle& left,
    const ExprHandle& right,
    std::uint16_t value,
    int cost,
    std::string shape_key,
    const std::bitset<256>& literal_consts,
    int square_count,
    int nonlinear_count
) {
    const InternKey key{
        kind,
        left ? left->id : 0u,
        right ? right->id : 0u,
        value,
    };
    auto& pool = intern_pool();
    if (const auto it = pool.find(key); it != pool.end()) {
        return it->second;
    }

    auto& arena = expr_arena();
    arena.emplace_back();
    Expr* expr = &arena.back();
    expr->kind = kind;
    expr->id = next_expr_id();
    expr->left = left;
    expr->right = right;
    expr->value = value;
    expr->cost = cost;
    expr->shape_key = std::move(shape_key);
    expr->literal_const_values = literal_consts;
    expr->square_count = square_count;
    expr->nonlinear_count = nonlinear_count;
    pool.emplace(key, expr);
    return expr;
}

ExprHandle make_raw_add(const ExprHandle& left, const ExprHandle& right) {
    return make_node(
        ExprKind::Add,
        left,
        right,
        0,
        left->cost + right->cost + 1,
        "(" + left->to_source() + " + " + right->to_source() + ")",
        left->literal_const_values | right->literal_const_values,
        left->square_count + right->square_count,
        left->nonlinear_count + right->nonlinear_count
    );
}

ExprHandle make_raw_sub(const ExprHandle& left, const ExprHandle& right) {
    return make_node(
        ExprKind::Sub,
        left,
        right,
        0,
        left->cost + right->cost + 1,
        "(" + left->to_source() + " - " + right->to_source() + ")",
        left->literal_const_values | right->literal_const_values,
        left->square_count + right->square_count,
        left->nonlinear_count + right->nonlinear_count
    );
}

ExprHandle make_raw_xor(const ExprHandle& left, const ExprHandle& right) {
    return make_node(
        ExprKind::Xor,
        left,
        right,
        0,
        left->cost + right->cost + 1,
        "(" + left->to_source() + " ^ " + right->to_source() + ")",
        left->literal_const_values | right->literal_const_values,
        left->square_count + right->square_count,
        left->nonlinear_count + right->nonlinear_count
    );
}

ExprHandle make_raw_mulconst(const ExprHandle& expr, int constant) {
    auto literal_consts = expr->literal_const_values;
    literal_consts.set(static_cast<std::size_t>(constant & MASK));
    return make_node(
        ExprKind::MulConst,
        expr,
        nullptr,
        static_cast<std::uint16_t>(constant & MASK),
        expr->cost + 1,
        "(" + std::to_string(constant & MASK) + " * " + expr->to_source() + ")",
        literal_consts,
        expr->square_count,
        expr->nonlinear_count
    );
}

ExprHandle make_raw_shift(const ExprHandle& expr, int shift, ExprKind kind) {
    const auto op = (kind == ExprKind::ShlConst) ? " << " : " >> ";
    return make_node(
        kind,
        expr,
        nullptr,
        static_cast<std::uint16_t>(shift),
        expr->cost + 1,
        "(" + expr->to_source() + op + std::to_string(shift) + ")",
        expr->literal_const_values,
        expr->square_count,
        expr->nonlinear_count
    );
}

ExprHandle make_raw_square(const ExprHandle& expr) {
    return make_node(
        ExprKind::Square,
        expr,
        nullptr,
        0,
        expr->cost + 1,
        "Square(" + expr->to_source() + ")",
        expr->literal_const_values,
        expr->square_count + 1,
        expr->nonlinear_count + 1
    );
}

ExprHandle make_raw_mul(const ExprHandle& left, const ExprHandle& right) {
    return make_node(
        ExprKind::Mul,
        left,
        right,
        0,
        left->cost + right->cost + 1,
        "(" + left->to_source() + " * " + right->to_source() + ")",
        left->literal_const_values | right->literal_const_values,
        left->square_count + right->square_count,
        left->nonlinear_count + right->nonlinear_count + 1
    );
}

ExprHandle make_raw_periodic(const ExprHandle& expr, ExprKind kind) {
    const std::string name = (kind == ExprKind::Sin8) ? "Sin8" : "Cos8";
    return make_node(
        kind,
        expr,
        nullptr,
        0,
        expr->cost + 1,
        name + "(" + expr->to_source() + ")",
        expr->literal_const_values,
        expr->square_count,
        expr->nonlinear_count
    );
}

std::pair<std::vector<ExprHandle>, int> split_add(const ExprHandle& expr) {
    if (expr->kind == ExprKind::Add) {
        auto [left_terms, left_const] = split_add(expr->left);
        auto [right_terms, right_const] = split_add(expr->right);
        left_terms.insert(left_terms.end(), right_terms.begin(), right_terms.end());
        return {left_terms, (left_const + right_const) & MASK};
    }

    if (const auto value = const_value(expr); value.has_value()) {
        return {{}, value.value()};
    }

    return {{expr}, 0};
}

ExprHandle build_add_chain(const std::vector<ExprHandle>& terms) {
    if (terms.empty()) {
        return const_expr(0);
    }
    auto expr = terms.front();
    for (std::size_t index = 1; index < terms.size(); ++index) {
        expr = make_raw_add(expr, terms[index]);
    }
    return expr;
}

std::pair<std::vector<ExprHandle>, int> split_xor(const ExprHandle& expr) {
    if (expr->kind == ExprKind::Xor) {
        auto [left_terms, left_const] = split_xor(expr->left);
        auto [right_terms, right_const] = split_xor(expr->right);
        left_terms.insert(left_terms.end(), right_terms.begin(), right_terms.end());
        return {left_terms, left_const ^ right_const};
    }

    if (const auto value = const_value(expr); value.has_value()) {
        return {{}, value.value()};
    }

    return {{expr}, 0};
}

ExprHandle build_xor_chain(const std::vector<ExprHandle>& terms) {
    if (terms.empty()) {
        return const_expr(0);
    }
    auto expr = terms.front();
    for (std::size_t index = 1; index < terms.size(); ++index) {
        expr = make_raw_xor(expr, terms[index]);
    }
    return expr;
}

std::bitset<256> values_to_bitset(const std::vector<int>& values) {
    std::bitset<256> bits;
    for (const int value : values) {
        bits.set(static_cast<std::size_t>(value & MASK));
    }
    return bits;
}

std::vector<int> bitset_to_sorted_values(const std::bitset<256>& bits) {
    std::vector<int> values;
    values.reserve(bits.count());
    for (int value = 0; value < DOMAIN_SIZE; ++value) {
        if (bits.test(static_cast<std::size_t>(value))) {
            values.push_back(value);
        }
    }
    return values;
}

void log_promotion_round(const PromotionRound& round) {
    std::cout
        << "const_round=" << round.round_index
        << " const_set_size_before=" << round.const_set_size_before
        << " derived_consts_discovered=" << round.derived_consts_discovered.size()
        << " derived_consts_promoted=" << round.derived_consts_promoted.size()
        << " const_set_size_after=" << round.const_set_size_after;
    if (round.found_expr.has_value()) {
        std::cout << " found_expr=" << round.found_expr.value();
    }
    std::cout << '\n';
}

void log_level(int cost, const LevelStats& stats, std::size_t sem_unique) {
    std::cout
        << "cost=" << cost
        << " generated=" << stats.generated
        << " rule_pruned=" << stats.rule_pruned
        << " simplified_to_existing_shape=" << stats.shape_pruned
        << " sig_bucket_pruned=" << stats.sig_pruned
        << " sig_bucket_kept=" << stats.sig_bucket_kept
        << " sig_bucket_replaced=" << stats.sig_bucket_replaced
        << " full_eval=" << stats.full_eval
        << " sem_pruned=" << stats.sem_pruned
        << " sem_unique=" << sem_unique
        << '\n';
}

template <typename T>
std::map<int, T> vector_to_level_map(const std::vector<T>& levels) {
    std::map<int, T> result;
    for (std::size_t index = 0; index < levels.size(); ++index) {
        if (index == 0) {
            continue;
        }
        result.emplace(static_cast<int>(index), levels[index]);
    }
    return result;
}

void discover_derived_consts(
    const ExprHandle& expr,
    const std::bitset<256>& known_consts,
    std::bitset<256>& discovered_consts
) {
    discovered_consts |= (expr->literal_const_values & ~known_consts);
}

bool update_signature_bucket(
    FlatMap<Signature24, SignatureBucket, FixedBytesHash<SIGNATURE_SIZE>>& bucket_by_sig,
    const Signature24& signature,
    const ExprHandle& expr,
    int bucket_capacity
) {
    auto& bucket = bucket_by_sig[signature];
    const auto rank = expr_rank_key(expr);

    for (const auto& existing : bucket) {
        if (existing == expr) {
            return false;
        }
    }

    if (static_cast<int>(bucket.size()) < bucket_capacity) {
        bucket.push_back(expr);
        std::sort(bucket.begin(), bucket.end(), [](const ExprHandle& left, const ExprHandle& right) {
            return expr_rank_key(left) < expr_rank_key(right);
        });
        return false;
    }

    auto worst_it = std::max_element(bucket.begin(), bucket.end(), [](const ExprHandle& left, const ExprHandle& right) {
        return expr_rank_key(left) < expr_rank_key(right);
    });
    if (worst_it == bucket.end() || !(rank < expr_rank_key(*worst_it))) {
        return false;
    }

    *worst_it = expr;
    std::sort(bucket.begin(), bucket.end(), [](const ExprHandle& left, const ExprHandle& right) {
        return expr_rank_key(left) < expr_rank_key(right);
    });
    return true;
}

bool should_signature_prune(
    const ExprHandle& expr,
    const Signature24& signature,
    const Signature24&,
    const FlatMap<Signature24, SignatureBucket, FixedBytesHash<SIGNATURE_SIZE>>& bucket_by_sig,
    bool use_signature_filter,
    const SigBucketLimitFn& sig_bucket_limit_fn,
    BucketSummary& bucket_summary
) {
    if (!use_signature_filter) {
        return false;
    }

    const auto found = bucket_by_sig.find(signature);
    if (found == bucket_by_sig.end()) {
        return false;
    }

    const int bucket_limit = std::max(1, sig_bucket_limit_fn(expr->cost));
    const auto& bucket = found->second;
    bucket_summary.max_bucket_count = std::max<std::uint64_t>(
        bucket_summary.max_bucket_count,
        static_cast<std::uint64_t>(bucket.size())
    );
    if (expr->cost >= SIGNATURE_PRUNE_COST) {
        bucket_summary.heavy_bucket_observations += 1;
        bucket_summary.heavy_bucket_size_total += static_cast<std::uint64_t>(bucket.size());
    }

    if (static_cast<int>(bucket.size()) < bucket_limit) {
        return false;
    }

    const auto worst_it = std::max_element(bucket.begin(), bucket.end(), [](const ExprHandle& left, const ExprHandle& right) {
        return expr_rank_key(left) < expr_rank_key(right);
    });
    if (worst_it == bucket.end()) {
        return false;
    }
    return !(expr_rank_key(expr) < expr_rank_key(*worst_it));
}

const Signature24& var_sig() {
    static const Signature24 signature = [] {
        Signature24 values{};
        const auto& points = sig_points();
        for (std::size_t index = 0; index < points.size(); ++index) {
            values[index] = static_cast<std::uint8_t>(points[index] & MASK);
        }
        return values;
    }();
    return signature;
}

const Semantics256& var_sem() {
    static const Semantics256 semantics = [] {
        Semantics256 values{};
        for (int index = 0; index < DOMAIN_SIZE; ++index) {
            values[static_cast<std::size_t>(index)] = static_cast<std::uint8_t>(index & MASK);
        }
        return values;
    }();
    return semantics;
}

const std::array<Signature24, DOMAIN_SIZE>& const_sig_cache() {
    static const std::array<Signature24, DOMAIN_SIZE> cache = [] {
        std::array<Signature24, DOMAIN_SIZE> values{};
        for (int constant = 0; constant < DOMAIN_SIZE; ++constant) {
            values[static_cast<std::size_t>(constant)].fill(static_cast<std::uint8_t>(constant));
        }
        return values;
    }();
    return cache;
}

const std::array<Semantics256, DOMAIN_SIZE>& const_sem_cache() {
    static const std::array<Semantics256, DOMAIN_SIZE> cache = [] {
        std::array<Semantics256, DOMAIN_SIZE> values{};
        for (int constant = 0; constant < DOMAIN_SIZE; ++constant) {
            values[static_cast<std::size_t>(constant)].fill(static_cast<std::uint8_t>(constant));
        }
        return values;
    }();
    return cache;
}

struct EnumerativeRoundResult {
    ExprHandle expr{};
    LevelStatsMap stats;
    std::bitset<256> discovered_consts;
    BucketSummary bucket_summary;
};

EnumerativeRoundResult find_enumerative_round(
    const Semantics256& target,
    const std::vector<int>& const_values,
    int max_cost,
    bool log_progress,
    bool use_signature_filter,
    const SigBucketLimitFn& sig_bucket_limit_fn,
    const NonlinearConfig& nonlinear,
    bool enable_periodic
);

EnumerativeRoundResult find_enumerative_sample_round(
    const SampleSet& sample_set,
    const std::vector<int>& const_values,
    int max_cost,
    bool log_progress,
    const NonlinearConfig& nonlinear,
    bool enable_periodic
);

struct EnumerativePromotionResult {
    ExprHandle expr{};
    LevelStatsMap stats;
    std::vector<PromotionRound> promotion_rounds;
    BucketSummary bucket_summary;
};

std::vector<int> bitset_to_sorted_values(const std::bitset<256>& bits);
std::bitset<256> values_to_bitset(const std::vector<int>& values);
void log_promotion_round(const PromotionRound& round);

EnumerativePromotionResult find_enumerative_with_promotion(
    const Semantics256& target,
    int max_cost,
    bool log_progress,
    bool use_signature_filter,
    int max_promotion_rounds,
    const SigBucketLimitFn& sig_bucket_limit_fn,
    const NonlinearConfig& nonlinear,
    bool enable_periodic
) {
    EnumerativePromotionResult aggregate;
    std::bitset<256> const_set_bits = values_to_bitset(base_const_set());

    for (int round_index = 0; round_index <= max_promotion_rounds; ++round_index) {
        const auto const_values = bitset_to_sorted_values(const_set_bits);
        auto round = find_enumerative_round(
            target,
            const_values,
            max_cost,
            log_progress,
            use_signature_filter,
            sig_bucket_limit_fn,
            nonlinear,
            enable_periodic
        );

        for (const auto& [cost, level] : round.stats) {
            merge_level_stats(aggregate.stats[cost], level);
        }
        aggregate.bucket_summary = merge_bucket_summaries(aggregate.bucket_summary, round.bucket_summary);
        if (round.expr && better(round.expr, aggregate.expr)) {
            aggregate.expr = round.expr;
        }

        const std::bitset<256> promoted_bits = round.discovered_consts & ~const_set_bits;
        const int const_set_size_before = static_cast<int>(const_set_bits.count());
        PromotionRound summary{
            round_index,
            const_set_size_before,
            const_set_size_before + static_cast<int>(promoted_bits.count()),
            bitset_to_sorted_values(round.discovered_consts),
            bitset_to_sorted_values(promoted_bits),
            round.expr ? std::optional<std::string>(round.expr->to_source()) : std::nullopt,
        };
        aggregate.promotion_rounds.push_back(summary);
        if (log_progress) {
            log_promotion_round(aggregate.promotion_rounds.back());
        }

        if (promoted_bits.none()) {
            break;
        }
        const_set_bits |= promoted_bits;
    }

    return aggregate;
}

EnumerativePromotionResult find_enumerative_sample_with_promotion(
    const SampleSet& sample_set,
    int max_cost,
    bool log_progress,
    int max_promotion_rounds,
    const NonlinearConfig& nonlinear,
    bool enable_periodic
) {
    EnumerativePromotionResult aggregate;
    std::bitset<256> const_set_bits = values_to_bitset(base_const_set());

    for (int round_index = 0; round_index <= max_promotion_rounds; ++round_index) {
        const auto const_values = bitset_to_sorted_values(const_set_bits);
        auto round = find_enumerative_sample_round(
            sample_set,
            const_values,
            max_cost,
            log_progress,
            nonlinear,
            enable_periodic
        );

        for (const auto& [cost, level] : round.stats) {
            merge_level_stats(aggregate.stats[cost], level);
        }
        aggregate.bucket_summary = merge_bucket_summaries(aggregate.bucket_summary, round.bucket_summary);
        if (round.expr && better(round.expr, aggregate.expr)) {
            aggregate.expr = round.expr;
        }

        const std::bitset<256> promoted_bits = round.discovered_consts & ~const_set_bits;
        const int const_set_size_before = static_cast<int>(const_set_bits.count());
        PromotionRound summary{
            round_index,
            const_set_size_before,
            const_set_size_before + static_cast<int>(promoted_bits.count()),
            bitset_to_sorted_values(round.discovered_consts),
            bitset_to_sorted_values(promoted_bits),
            round.expr ? std::optional<std::string>(round.expr->to_source()) : std::nullopt,
        };
        aggregate.promotion_rounds.push_back(summary);
        if (log_progress) {
            log_promotion_round(aggregate.promotion_rounds.back());
        }

        if (promoted_bits.none()) {
            break;
        }
        const_set_bits |= promoted_bits;
    }

    return aggregate;
}

}  // namespace

SearchAttempt run_search_attempt(
    const std::optional<Semantics256>& target,
    int max_cost,
    bool log_progress,
    SigBucketLimitFn sig_bucket_limit_fn,
    std::string_view nonlinear_mode,
    bool enable_periodic
) {
    const auto target_values = target.value_or(collect_target());
    const auto& nonlinear = nonlinear_config(nonlinear_mode);
    const auto started = std::chrono::steady_clock::now();

    auto signature = find_enumerative_with_promotion(
        target_values,
        max_cost,
        log_progress,
        true,
        MAX_PROMOTION_ROUNDS,
        sig_bucket_limit_fn,
        nonlinear,
        enable_periodic
    );

    const auto elapsed = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    };

    SearchAttempt attempt;
    attempt.attempted_backends = {"enumerative"};
    attempt.signature_stats = signature.stats;
    attempt.signature_bucket_summary = signature.bucket_summary;
    attempt.signature_promotion_rounds = signature.promotion_rounds;

    if (signature.expr) {
        attempt.backend_timings_s.push_back({"enumerative", elapsed()});
        attempt.solved_by_backend_index = 0;
        attempt.result = SearchResult{
            "enumerative",
            signature.expr,
            signature.stats,
            signature.promotion_rounds,
            signature.bucket_summary,
            false,
        };
        return attempt;
    }

    if (log_progress) {
        std::cout << "retry=exact_enumerative_without_signature\n";
    }

    auto fallback = find_enumerative_with_promotion(
        target_values,
        max_cost,
        log_progress,
        false,
        MAX_PROMOTION_ROUNDS,
        sig_bucket_limit_fn,
        nonlinear,
        enable_periodic
    );

    attempt.fallback_stats = fallback.stats;
    attempt.fallback_bucket_summary = fallback.bucket_summary;
    attempt.fallback_promotion_rounds = fallback.promotion_rounds;

    attempt.backend_timings_s.push_back({"enumerative", elapsed()});
    if (fallback.expr) {
        attempt.solved_by_backend_index = 0;
        attempt.result = SearchResult{
            "enumerative",
            fallback.expr,
            fallback.stats,
            fallback.promotion_rounds,
            fallback.bucket_summary,
            true,
        };
    }
    return attempt;
}

SearchAttempt run_sample_search_attempt(
    const SampleSet& sample_set,
    int max_cost,
    bool log_progress,
    std::string_view nonlinear_mode,
    bool enable_periodic
) {
    const auto& nonlinear = nonlinear_config(nonlinear_mode);
    const auto started = std::chrono::steady_clock::now();
    auto round = find_enumerative_sample_with_promotion(
        sample_set,
        max_cost,
        log_progress,
        MAX_PROMOTION_ROUNDS,
        nonlinear,
        enable_periodic
    );
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    SearchAttempt attempt;
    attempt.attempted_backends = {"enumerative"};
    attempt.backend_timings_s.push_back({"enumerative", elapsed});
    attempt.signature_stats = round.stats;
    attempt.signature_bucket_summary = round.bucket_summary;
    attempt.signature_promotion_rounds = round.promotion_rounds;

    if (round.expr) {
        attempt.solved_by_backend_index = 0;
        attempt.result = SearchResult{
            "enumerative",
            round.expr,
            round.stats,
            round.promotion_rounds,
            round.bucket_summary,
            false,
        };
    }
    return attempt;
}

SearchAttempt synthesize_cegis(
    const OracleFn& oracle,
    const std::vector<std::uint8_t>& initial_points,
    int max_rounds,
    int max_cost,
    bool log_progress,
    std::string_view nonlinear_mode,
    bool enable_periodic
) {
    auto sample_set = SampleSet::from_oracle(oracle, initial_points);
    std::vector<CegisRound> cegis_rounds;
    LevelStatsMap aggregate_stats;
    BucketSummary aggregate_bucket_summary;
    std::map<std::string, double> aggregate_timings;
    SearchAttempt last_attempt;
    bool have_last = false;

    for (int round_index = 0; round_index < max_rounds; ++round_index) {
        auto attempt = run_sample_search_attempt(sample_set, max_cost, log_progress, nonlinear_mode, enable_periodic);
        last_attempt = attempt;
        have_last = true;

        for (const auto& [name, value] : attempt.backend_timings_s) {
            aggregate_timings[name] += value;
        }
        for (const auto& [cost, level] : attempt.signature_stats) {
            merge_level_stats(aggregate_stats[cost], level);
        }
        if (attempt.signature_bucket_summary.has_value()) {
            aggregate_bucket_summary = merge_bucket_summaries(aggregate_bucket_summary, attempt.signature_bucket_summary.value());
        }

        if (!attempt.result.has_value()) {
            cegis_rounds.push_back(CegisRound{
                round_index,
                static_cast<int>(sample_set.points.size()),
                sample_set.points,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                false,
            });
            SearchAttempt result;
            result.attempted_backends = attempt.attempted_backends;
            result.signature_stats = aggregate_stats;
            result.signature_bucket_summary = aggregate_bucket_summary;
            result.cegis_rounds = cegis_rounds;
            for (const auto& [name, value] : aggregate_timings) {
                result.backend_timings_s.push_back({name, value});
            }
            return result;
        }

        const auto counterexample = find_counterexample(attempt.result->expr, oracle);
        const bool full_verify_passed = !counterexample.has_value();
        cegis_rounds.push_back(CegisRound{
            round_index,
            static_cast<int>(sample_set.points.size()),
            sample_set.points,
            attempt.result->backend,
            attempt.result->expr->to_source(),
            counterexample,
            full_verify_passed,
        });

        if (full_verify_passed) {
            SearchAttempt result;
            result.result = SearchResult{
                attempt.result->backend,
                attempt.result->expr,
                aggregate_stats.empty() ? attempt.result->stats : aggregate_stats,
                attempt.signature_promotion_rounds,
                aggregate_bucket_summary,
                false,
            };
            result.attempted_backends = attempt.attempted_backends;
            result.solved_by_backend_index = attempt.solved_by_backend_index;
            result.signature_stats = aggregate_stats;
            result.signature_promotion_rounds = attempt.signature_promotion_rounds;
            result.signature_bucket_summary = aggregate_bucket_summary;
            result.cegis_rounds = cegis_rounds;
            for (const auto& [name, value] : aggregate_timings) {
                result.backend_timings_s.push_back({name, value});
            }
            return result;
        }

        sample_set = sample_set.with_point(oracle, static_cast<std::uint8_t>(counterexample.value()));
    }

    SearchAttempt result;
    if (have_last) {
        result.attempted_backends = last_attempt.attempted_backends;
        result.solved_by_backend_index = last_attempt.solved_by_backend_index;
    }
    result.signature_stats = aggregate_stats;
    result.signature_bucket_summary = aggregate_bucket_summary;
    result.cegis_rounds = cegis_rounds;
    for (const auto& [name, value] : aggregate_timings) {
        result.backend_timings_s.push_back({name, value});
    }
    return result;
}

std::optional<SearchResult> synthesize(
    const Semantics256& target,
    int max_cost,
    bool log_progress,
    SigBucketLimitFn sig_bucket_limit_fn,
    std::string_view nonlinear_mode,
    bool enable_periodic
) {
    auto attempt = run_search_attempt(target, max_cost, log_progress, std::move(sig_bucket_limit_fn), nonlinear_mode, enable_periodic);
    return attempt.result;
}

std::pair<bool, std::optional<int>> verify_expression(const ExprHandle& expr, const Semantics256& target) {
    for (int x = 0; x < DOMAIN_SIZE; ++x) {
        if (expr->eval(static_cast<std::uint8_t>(x)) != target[static_cast<std::size_t>(x)]) {
            return {false, x};
        }
    }
    return {true, std::nullopt};
}

namespace {

bool expr_contains_kind(const ExprHandle& expr, ExprKind kind) {
    if (!expr) {
        return false;
    }
    if (expr->kind == kind) {
        return true;
    }
    return expr_contains_kind(expr->left, kind) || expr_contains_kind(expr->right, kind);
}

std::string emit_periodic_lut_function(std::string_view name, const std::array<std::uint8_t, DOMAIN_SIZE>& values) {
    std::ostringstream out;
    out
        << "    function [7:0] " << name << ";\n"
        << "        input [7:0] v;\n"
        << "        begin\n"
        << "            case (v)\n";
    for (int value = 0; value < DOMAIN_SIZE; ++value) {
        out
            << "                8'd" << value << ": " << name << " = 8'd"
            << static_cast<int>(values[static_cast<std::size_t>(value)]) << ";\n";
    }
    out
        << "                default: " << name << " = 8'd0;\n"
        << "            endcase\n"
        << "        end\n"
        << "    endfunction\n";
    return out.str();
}

}  // namespace

std::string emit_verilog(const ExprHandle& expr, std::string_view module_name) {
    std::ostringstream out;
    out
        << "module " << module_name << " (\n"
        << "    input  wire [7:0] x,\n"
        << "    output wire [7:0] y\n"
        << ");\n";
    if (expr_contains_kind(expr, ExprKind::Sin8)) {
        out << emit_periodic_lut_function("sin8", sin8_table()) << "\n";
    }
    if (expr_contains_kind(expr, ExprKind::Cos8)) {
        out << emit_periodic_lut_function("cos8", cos8_table()) << "\n";
    }
    out
        << "    assign y = " << expr->to_verilog8() << ";\n"
        << "endmodule\n";
    return out.str();
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << ch; break;
        }
    }
    return out.str();
}

void write_text_file(const std::filesystem::path& path, std::string_view content) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }
    out << content;
}

void ensure_directory(const std::filesystem::path& path) {
    std::filesystem::create_directories(path);
}

const int MAX_COST = 6;
const int SIGNATURE_PRUNE_COST = 5;
const int MAX_PROMOTION_ROUNDS = 8;
const char* const DEFAULT_NONLINEAR_MODE = "restricted";

std::uint8_t sin8_value(std::uint8_t value) {
    return sin8_table()[static_cast<std::size_t>(value & MASK)];
}

std::uint8_t cos8_value(std::uint8_t value) {
    return cos8_table()[static_cast<std::size_t>(value & MASK)];
}

std::size_t VectorBytesHash::operator()(const Bytes& bytes) const noexcept {
    return static_cast<std::size_t>(
        ankerl::unordered_dense::detail::wyhash::hash(bytes.data(), bytes.size()));
}

std::uint8_t Expr::eval(std::uint8_t x) const {
    switch (kind) {
    case ExprKind::Var:
        return static_cast<std::uint8_t>(x & MASK);
    case ExprKind::Const:
        return static_cast<std::uint8_t>(value & MASK);
    case ExprKind::MulConst:
        return static_cast<std::uint8_t>((left->eval(x) * (value & MASK)) & MASK);
    case ExprKind::Add:
        return static_cast<std::uint8_t>((left->eval(x) + right->eval(x)) & MASK);
    case ExprKind::Sub:
        return static_cast<std::uint8_t>((left->eval(x) - right->eval(x)) & MASK);
    case ExprKind::Xor:
        return static_cast<std::uint8_t>((left->eval(x) ^ right->eval(x)) & MASK);
    case ExprKind::ShlConst:
        return static_cast<std::uint8_t>((left->eval(x) << value) & MASK);
    case ExprKind::ShrConst:
        return static_cast<std::uint8_t>((left->eval(x) >> value) & MASK);
    case ExprKind::Square: {
        const auto inner = left->eval(x);
        return static_cast<std::uint8_t>((inner * inner) & MASK);
    }
    case ExprKind::Mul:
        return static_cast<std::uint8_t>((left->eval(x) * right->eval(x)) & MASK);
    case ExprKind::Sin8:
        return sin8_value(left->eval(x));
    case ExprKind::Cos8:
        return cos8_value(left->eval(x));
    }
    throw std::runtime_error("Unsupported ExprKind");
}

std::string Expr::to_source() const {
    return shape_key;
}

std::string Expr::to_verilog() const {
    switch (kind) {
    case ExprKind::Var:
        return "x";
    case ExprKind::Const:
        return "8'd" + std::to_string(value & MASK);
    case ExprKind::MulConst:
        return "(" + left->to_verilog8() + " * 8'd" + std::to_string(value & MASK) + ")";
    case ExprKind::Add:
        return "(" + left->to_verilog8() + " + " + right->to_verilog8() + ")";
    case ExprKind::Sub:
        return "(" + left->to_verilog8() + " - " + right->to_verilog8() + ")";
    case ExprKind::Xor:
        return "(" + left->to_verilog8() + " ^ " + right->to_verilog8() + ")";
    case ExprKind::ShlConst:
        return "(" + left->to_verilog8() + " << " + std::to_string(value) + ")";
    case ExprKind::ShrConst:
        return "(" + left->to_verilog8() + " >> " + std::to_string(value) + ")";
    case ExprKind::Square: {
        const auto inner = left->to_verilog8();
        return "(" + inner + " * " + inner + ")";
    }
    case ExprKind::Mul:
        return "(" + left->to_verilog8() + " * " + right->to_verilog8() + ")";
    case ExprKind::Sin8:
        return "sin8(" + left->to_verilog8() + ")";
    case ExprKind::Cos8:
        return "cos8(" + left->to_verilog8() + ")";
    }
    throw std::runtime_error("Unsupported ExprKind");
}

std::string Expr::to_verilog8() const {
    return "((" + to_verilog() + ") & 8'hFF)";
}

double BucketSummary::avg_bucket_count_on_heavy_levels() const {
    if (heavy_bucket_observations == 0) {
        return 0.0;
    }
    return static_cast<double>(heavy_bucket_size_total) / static_cast<double>(heavy_bucket_observations);
}

bool SearchAttempt::fallback_triggered() const {
    return !fallback_stats.empty();
}

bool SearchAttempt::solved_by_first_backend() const {
    return solved_by_backend_index.has_value() && solved_by_backend_index.value() == 0;
}

SampleSet SampleSet::from_oracle(const OracleFn& oracle, const std::vector<std::uint8_t>& points) {
    std::bitset<256> seen;
    for (const auto point : points) {
        seen.set(static_cast<std::size_t>(point & MASK));
    }

    SampleSet result;
    result.points.reserve(points.size());
    result.values.reserve(points.size());
    for (int point = 0; point < DOMAIN_SIZE; ++point) {
        if (seen.test(static_cast<std::size_t>(point))) {
            result.points.push_back(static_cast<std::uint8_t>(point));
            result.values.push_back(oracle(static_cast<std::uint8_t>(point)) & MASK);
        }
    }
    return result;
}

SampleSet SampleSet::with_point(const OracleFn& oracle, std::uint8_t point) const {
    auto combined = points;
    combined.push_back(static_cast<std::uint8_t>(point & MASK));
    return from_oracle(oracle, combined);
}

const std::vector<int>& base_const_set() {
    static const std::vector<int> values = {0, 1, 2, 3, 5, 7, 8, 10, 13, 15, 16, 17, 20, 31, 32, 63, 64, 127, 128, 255};
    return values;
}

const std::vector<int>& shift_set() {
    static const std::vector<int> values = {1, 2, 3, 4, 5, 6, 7};
    return values;
}

const std::array<int, SIGNATURE_SIZE>& sig_points() {
    static const std::array<int, SIGNATURE_SIZE> values = {
        0, 1, 2, 3, 4, 5, 7, 8,
        15, 16, 31, 32, 63, 64, 127, 128,
        129, 170, 192, 223, 240, 247, 251, 255,
    };
    return values;
}

const std::vector<std::uint8_t>& initial_cegis_points() {
    static const std::vector<std::uint8_t> values = {
        0, 1, 2, 3, 4, 5, 7, 8,
        15, 16, 31, 32, 63, 64, 127, 128, 255,
    };
    return values;
}

const std::map<std::string, NonlinearConfig>& nonlinear_configs() {
    static const std::map<std::string, NonlinearConfig> values = {
        {"restricted", {"restricted", 1, false}},
        {"relaxed", {"relaxed", 2, true}},
        {"open", {"open", 3, true}},
    };
    return values;
}

const NonlinearConfig& nonlinear_config(std::string_view mode) {
    const auto found = nonlinear_configs().find(std::string(mode));
    if (found == nonlinear_configs().end()) {
        throw std::runtime_error("Unsupported nonlinear mode: " + std::string(mode));
    }
    return found->second;
}

ExprHandle var_expr() {
    static const ExprHandle expr = make_node(ExprKind::Var, nullptr, nullptr, 0, 1, "x", {}, 0, 0);
    return expr;
}

ExprHandle const_expr(int value) {
    static const std::array<ExprHandle, DOMAIN_SIZE> cache = [] {
        std::array<ExprHandle, DOMAIN_SIZE> values{};
        for (int constant = 0; constant < DOMAIN_SIZE; ++constant) {
            std::bitset<256> literal_consts;
            literal_consts.set(static_cast<std::size_t>(constant));
            values[static_cast<std::size_t>(constant)] = make_node(
                ExprKind::Const,
                nullptr,
                nullptr,
                static_cast<std::uint16_t>(constant),
                1,
                std::to_string(constant),
                literal_consts,
                0,
                0
            );
        }
        return values;
    }();
    return cache[static_cast<std::size_t>(value & MASK)];
}

bool expr_equal(const ExprHandle& left, const ExprHandle& right) {
    return left == right;
}

ExprHandle make_mul_const(const ExprHandle& expr, int constant) {
    constant &= MASK;
    if (constant == 0) {
        return const_expr(0);
    }
    if (constant == 1) {
        return expr;
    }

    if (const auto value = const_value(expr); value.has_value()) {
        return const_expr((value.value() * constant) & MASK);
    }

    if (expr->kind == ExprKind::MulConst) {
        return make_mul_const(expr->left, (static_cast<int>(expr->value) * constant) & MASK);
    }

    return make_raw_mulconst(expr, constant);
}

ExprHandle make_shl_const(const ExprHandle& expr, int shift) {
    if (shift == 0) {
        return expr;
    }

    if (const auto value = const_value(expr); value.has_value()) {
        return const_expr((value.value() << shift) & MASK);
    }

    if (expr->kind == ExprKind::ShlConst) {
        const int total_shift = static_cast<int>(expr->value) + shift;
        if (total_shift >= WIDTH) {
            return const_expr(0);
        }
        return make_raw_shift(expr->left, total_shift, ExprKind::ShlConst);
    }

    return make_raw_shift(expr, shift, ExprKind::ShlConst);
}

ExprHandle make_shr_const(const ExprHandle& expr, int shift) {
    if (shift == 0) {
        return expr;
    }

    if (const auto value = const_value(expr); value.has_value()) {
        return const_expr(value.value() >> shift);
    }

    if (expr->kind == ExprKind::ShrConst) {
        const int total_shift = static_cast<int>(expr->value) + shift;
        if (total_shift >= WIDTH) {
            return const_expr(0);
        }
        return make_raw_shift(expr->left, total_shift, ExprKind::ShrConst);
    }

    return make_raw_shift(expr, shift, ExprKind::ShrConst);
}

ExprHandle make_square(const ExprHandle& expr, int max_nonlinear_count) {
    if (expr->nonlinear_count + 1 > max_nonlinear_count) {
        throw std::runtime_error("Square exceeds the current nonlinear-count limit");
    }

    if (const auto value = const_value(expr); value.has_value()) {
        return const_expr((value.value() * value.value()) & MASK);
    }

    return make_raw_square(expr);
}

ExprHandle make_mul(const ExprHandle& left, const ExprHandle& right, bool allow_nonlinear_operands, int max_nonlinear_count) {
    const auto left_const = const_value(left);
    const auto right_const = const_value(right);

    if ((left_const.has_value() && left_const.value() == 0) || (right_const.has_value() && right_const.value() == 0)) {
        return const_expr(0);
    }
    if (left_const.has_value() && left_const.value() == 1) {
        return right;
    }
    if (right_const.has_value() && right_const.value() == 1) {
        return left;
    }
    if (left_const.has_value() && right_const.has_value()) {
        return const_expr((left_const.value() * right_const.value()) & MASK);
    }
    if (left_const.has_value()) {
        return make_mul_const(right, left_const.value());
    }
    if (right_const.has_value()) {
        return make_mul_const(left, right_const.value());
    }
    if (expr_equal(left, right)) {
        return make_square(left, max_nonlinear_count);
    }
    if (!allow_nonlinear_operands && (left->nonlinear_count > 0 || right->nonlinear_count > 0)) {
        throw std::runtime_error("Mul operands must be linear under the current DSL limit");
    }
    if (left->nonlinear_count + right->nonlinear_count + 1 > max_nonlinear_count) {
        throw std::runtime_error("Mul exceeds the current nonlinear-count limit");
    }

    auto ordered_left = left;
    auto ordered_right = right;
    if (canonical_expr_sort_key(ordered_right) < canonical_expr_sort_key(ordered_left)) {
        std::swap(ordered_left, ordered_right);
    }
    return make_raw_mul(ordered_left, ordered_right);
}

ExprHandle make_sin8(const ExprHandle& expr) {
    if (const auto value = const_value(expr); value.has_value()) {
        return const_expr(sin8_value(static_cast<std::uint8_t>(value.value() & MASK)));
    }

    return make_raw_periodic(expr, ExprKind::Sin8);
}

ExprHandle make_cos8(const ExprHandle& expr) {
    if (const auto value = const_value(expr); value.has_value()) {
        return const_expr(cos8_value(static_cast<std::uint8_t>(value.value() & MASK)));
    }

    return make_raw_periodic(expr, ExprKind::Cos8);
}

ExprHandle make_add(const ExprHandle& left, const ExprHandle& right) {
    auto [left_terms, left_const] = split_add(left);
    auto [right_terms, right_const] = split_add(right);
    left_terms.insert(left_terms.end(), right_terms.begin(), right_terms.end());

    const int const_sum = (left_const + right_const) & MASK;
    std::sort(left_terms.begin(), left_terms.end(), [](const ExprHandle& a, const ExprHandle& b) {
        return canonical_expr_sort_key(a) < canonical_expr_sort_key(b);
    });
    if (const_sum != 0) {
        left_terms.push_back(const_expr(const_sum));
    }
    if (left_terms.empty()) {
        return const_expr(0);
    }
    if (left_terms.size() == 1) {
        return left_terms.front();
    }
    return build_add_chain(left_terms);
}

ExprHandle make_sub(const ExprHandle& left, const ExprHandle& right) {
    if (is_zero(right)) {
        return left;
    }
    if (expr_equal(left, right)) {
        return const_expr(0);
    }

    const auto left_const = const_value(left);
    const auto right_const = const_value(right);
    if (left_const.has_value() && right_const.has_value()) {
        return const_expr((left_const.value() - right_const.value()) & MASK);
    }

    if (right_const.has_value()) {
        auto [add_terms, add_const] = split_add(left);
        if (!add_terms.empty() || add_const != 0) {
            const int const_term = (add_const - right_const.value()) & MASK;
            std::sort(add_terms.begin(), add_terms.end(), [](const ExprHandle& a, const ExprHandle& b) {
                return canonical_expr_sort_key(a) < canonical_expr_sort_key(b);
            });
            if (const_term != 0) {
                add_terms.push_back(const_expr(const_term));
            }
            return build_add_chain(add_terms);
        }
        if (left->kind == ExprKind::Sub) {
            const auto nested_right_const = const_value(left->right);
            if (nested_right_const.has_value()) {
                return make_sub(left->left, const_expr((nested_right_const.value() + right_const.value()) & MASK));
            }
        }
    }

    return make_raw_sub(left, right);
}

ExprHandle make_xor(const ExprHandle& left, const ExprHandle& right) {
    auto [left_terms, left_const] = split_xor(left);
    auto [right_terms, right_const] = split_xor(right);

    FlatMap<ExprId, std::pair<ExprHandle, int>> parity;
    for (const auto& term : left_terms) {
        auto& entry = parity[term->id];
        entry.first = term;
        entry.second ^= 1;
    }
    for (const auto& term : right_terms) {
        auto& entry = parity[term->id];
        entry.first = term;
        entry.second ^= 1;
    }

    std::vector<ExprHandle> terms;
    for (const auto& [_, entry] : parity) {
        if (entry.second != 0) {
            terms.push_back(entry.first);
        }
    }
    std::sort(terms.begin(), terms.end(), [](const ExprHandle& a, const ExprHandle& b) {
        return canonical_expr_sort_key(a) < canonical_expr_sort_key(b);
    });

    const int const_term = left_const ^ right_const;
    if (const_term != 0) {
        terms.push_back(const_expr(const_term & MASK));
    }
    if (terms.empty()) {
        return const_expr(0);
    }
    if (terms.size() == 1) {
        return terms.front();
    }
    return build_xor_chain(terms);
}

Semantics256 build_target(const OracleFn& oracle) {
    Semantics256 target{};
    for (int x = 0; x < DOMAIN_SIZE; ++x) {
        target[static_cast<std::size_t>(x)] = oracle(static_cast<std::uint8_t>(x)) & MASK;
    }
    return target;
}

Semantics256 collect_target() {
    return build_target(current_oracle);
}

Semantics256 evaluate_expr(const ExprHandle& expr) {
    Semantics256 values{};
    for (int x = 0; x < DOMAIN_SIZE; ++x) {
        values[static_cast<std::size_t>(x)] = expr->eval(static_cast<std::uint8_t>(x));
    }
    return values;
}

std::optional<int> find_counterexample(const ExprHandle& expr, const OracleFn& oracle) {
    for (int x = 0; x < DOMAIN_SIZE; ++x) {
        if (expr->eval(static_cast<std::uint8_t>(x)) != (oracle(static_cast<std::uint8_t>(x)) & MASK)) {
            return x;
        }
    }
    return std::nullopt;
}

bool better(const ExprHandle& newer, const ExprHandle& older) {
    if (!older) {
        return true;
    }
    if (newer->cost != older->cost) {
        return newer->cost < older->cost;
    }
    if (newer->shape_key.size() != older->shape_key.size()) {
        return newer->shape_key.size() < older->shape_key.size();
    }
    return newer->shape_key < older->shape_key;
}

Bytes mul_const_values(const Bytes& values, int constant) {
    Bytes result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = static_cast<std::uint8_t>((values[index] * (constant & MASK)) & MASK);
    }
    return result;
}

Bytes shl_values(const Bytes& values, int shift) {
    Bytes result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = static_cast<std::uint8_t>((values[index] << shift) & MASK);
    }
    return result;
}

Bytes shr_values(const Bytes& values, int shift) {
    Bytes result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = static_cast<std::uint8_t>((values[index] >> shift) & MASK);
    }
    return result;
}

Bytes add_values(const Bytes& left, const Bytes& right) {
    Bytes result(left.size());
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = static_cast<std::uint8_t>((left[index] + right[index]) & MASK);
    }
    return result;
}

Bytes sub_values(const Bytes& left, const Bytes& right) {
    Bytes result(left.size());
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = static_cast<std::uint8_t>((left[index] - right[index]) & MASK);
    }
    return result;
}

Bytes xor_values(const Bytes& left, const Bytes& right) {
    Bytes result(left.size());
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = static_cast<std::uint8_t>((left[index] ^ right[index]) & MASK);
    }
    return result;
}

Bytes square_values(const Bytes& values) {
    Bytes result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = static_cast<std::uint8_t>((values[index] * values[index]) & MASK);
    }
    return result;
}

Bytes mul_values(const Bytes& left, const Bytes& right) {
    Bytes result(left.size());
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = static_cast<std::uint8_t>((left[index] * right[index]) & MASK);
    }
    return result;
}

Bytes sin8_values(const Bytes& values) {
    Bytes result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = sin8_value(values[index]);
    }
    return result;
}

Bytes cos8_values(const Bytes& values) {
    Bytes result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = cos8_value(values[index]);
    }
    return result;
}

Signature24 target_signature(const Semantics256& target) {
    Signature24 signature{};
    const auto& points = sig_points();
    for (std::size_t index = 0; index < points.size(); ++index) {
        signature[index] = target[static_cast<std::size_t>(points[index])];
    }
    return signature;
}

int top1_sig_bucket_limit(int) {
    return 1;
}

int selective_sig_bucket_limit(int cost) {
    return (cost < 6) ? 1 : 3;
}

void merge_level_stats(LevelStats& dst, const LevelStats& src) {
    dst.generated += src.generated;
    dst.rule_pruned += src.rule_pruned;
    dst.shape_pruned += src.shape_pruned;
    dst.sig_pruned += src.sig_pruned;
    dst.sig_bucket_kept += src.sig_bucket_kept;
    dst.sig_bucket_replaced += src.sig_bucket_replaced;
    dst.full_eval += src.full_eval;
    dst.sem_pruned += src.sem_pruned;
    dst.accepted += src.accepted;
    dst.generated_mul_candidates += src.generated_mul_candidates;
    dst.accepted_mul_candidates += src.accepted_mul_candidates;
    dst.shape_pruned_mul += src.shape_pruned_mul;
    dst.sem_pruned_mul += src.sem_pruned_mul;
    dst.max_nonlinear_depth_seen = std::max(dst.max_nonlinear_depth_seen, src.max_nonlinear_depth_seen);
}

BucketSummary merge_bucket_summaries(const BucketSummary& dst, const BucketSummary& src) {
    return BucketSummary{
        std::max(dst.max_bucket_count, src.max_bucket_count),
        dst.heavy_bucket_observations + src.heavy_bucket_observations,
        dst.heavy_bucket_size_total + src.heavy_bucket_size_total,
    };
}

namespace {

EnumerativeRoundResult find_enumerative_round(
    const Semantics256& target,
    const std::vector<int>& const_values,
    int max_cost,
    bool log_progress,
    bool use_signature_filter,
    const SigBucketLimitFn& sig_bucket_limit_fn,
    const NonlinearConfig& nonlinear,
    bool enable_periodic
) {
    std::vector<FlatMap<Semantics256, Candidate, FixedBytesHash<DOMAIN_SIZE>>> levels(static_cast<std::size_t>(max_cost + 1));
    FlatMap<Semantics256, Candidate, FixedBytesHash<DOMAIN_SIZE>> best_by_sem;
    FlatMap<Signature24, SignatureBucket, FixedBytesHash<SIGNATURE_SIZE>> bucket_by_sig;
    std::vector<FlatSet<ExprId>> seen_ids(static_cast<std::size_t>(max_cost + 1));
    FlatSet<ExprId> reduced_ids_seen;
    std::vector<LevelStats> stats(static_cast<std::size_t>(max_cost + 1));

    std::size_t best_reserve = 0;
    for (int cost = 1; cost <= max_cost; ++cost) {
        const auto hint = level_reserve_hint(cost);
        levels[static_cast<std::size_t>(cost)].reserve(hint);
        seen_ids[static_cast<std::size_t>(cost)].reserve(hint);
        best_reserve += hint;
    }
    best_by_sem.reserve(best_reserve);
    bucket_by_sig.reserve(best_reserve / 2 + 1);

    std::bitset<256> discovered_consts;
    const auto known_consts = values_to_bitset(const_values);
    const auto target_sig = target_signature(target);
    ExprHandle best_target_expr{};
    int bucket_capacity = 1;
    for (int cost = 1; cost <= max_cost; ++cost) {
        bucket_capacity = std::max(bucket_capacity, std::max(1, sig_bucket_limit_fn(cost)));
    }
    BucketSummary bucket_summary;

    auto accept = [&](const Candidate& candidate, LevelStats& cost_stats, const char* op = nullptr) -> bool {
        const auto current_best = best_by_sem.find(candidate.sem);
        if (current_best != best_by_sem.end() && !better(candidate.expr, current_best->second.expr)) {
            cost_stats.sem_pruned += 1;
            if (op && std::string_view(op) == "mul") {
                cost_stats.sem_pruned_mul += 1;
            }
            return false;
        }

        if (current_best != best_by_sem.end()) {
            levels[static_cast<std::size_t>(current_best->second.expr->cost)].erase(candidate.sem);
        }

        best_by_sem[candidate.sem] = candidate;
        levels[static_cast<std::size_t>(candidate.expr->cost)][candidate.sem] = candidate;
        cost_stats.accepted += 1;
        if (op && std::string_view(op) == "mul") {
            cost_stats.accepted_mul_candidates += 1;
        }
        const bool replaced = update_signature_bucket(bucket_by_sig, candidate.sig, candidate.expr, bucket_capacity);
        if (replaced) {
            cost_stats.sig_bucket_replaced += 1;
        }
        if (candidate.sem == target && better(candidate.expr, best_target_expr)) {
            best_target_expr = candidate.expr;
        }
        return true;
    };

    auto try_candidate = [&](const ExprHandle& expr, const Signature24& sig, const auto& sem_builder, int target_cost, const char* op = nullptr) {
        auto& cost_stats = stats[static_cast<std::size_t>(target_cost)];
        cost_stats.generated += 1;
        cost_stats.max_nonlinear_depth_seen = std::max<std::uint64_t>(
            cost_stats.max_nonlinear_depth_seen,
            static_cast<std::uint64_t>(expr->nonlinear_count)
        );
        if (op && std::string_view(op) == "mul") {
            cost_stats.generated_mul_candidates += 1;
        }

        if (expr->cost != target_cost) {
            if (expr->cost < target_cost && sig == target_sig && (!best_target_expr || expr->cost < best_target_expr->cost)) {
                if (reduced_ids_seen.insert(expr->id).second) {
                    discover_derived_consts(expr, known_consts, discovered_consts);
                }
            }
            cost_stats.rule_pruned += 1;
            return;
        }

        if (!seen_ids[static_cast<std::size_t>(target_cost)].insert(expr->id).second) {
            cost_stats.shape_pruned += 1;
            if (op && std::string_view(op) == "mul") {
                cost_stats.shape_pruned_mul += 1;
            }
            return;
        }

        if (should_signature_prune(expr, sig, target_sig, bucket_by_sig, use_signature_filter, sig_bucket_limit_fn, bucket_summary)) {
            cost_stats.sig_pruned += 1;
            return;
        }
        if (use_signature_filter) {
            cost_stats.sig_bucket_kept += 1;
        }

        cost_stats.full_eval += 1;
        const auto sem = sem_builder();
        accept(Candidate{expr, sig, sem}, cost_stats, op);
    };

    auto& base_stats = stats[1];
    Candidate var_candidate{var_expr(), var_sig(), var_sem()};
    base_stats.generated += 1;
    base_stats.full_eval += 1;
    seen_ids[1].insert(var_candidate.expr->id);
    accept(var_candidate, base_stats);

    for (const int constant : const_values) {
        const auto node = const_expr(constant);
        base_stats.generated += 1;
        base_stats.full_eval += 1;
        seen_ids[1].insert(node->id);
        accept(Candidate{node, const_sig_cache()[static_cast<std::size_t>(constant & MASK)], const_sem_cache()[static_cast<std::size_t>(constant & MASK)]}, base_stats);
    }

    for (int cost = 2; cost <= max_cost; ++cost) {
        std::vector<Candidate> prev_items;
        prev_items.reserve(levels[static_cast<std::size_t>(cost - 1)].size());
        for (const auto& [_, candidate] : levels[static_cast<std::size_t>(cost - 1)]) {
            prev_items.push_back(candidate);
        }

        for (const auto& prev : prev_items) {
            if (prev.expr->nonlinear_count + 1 <= nonlinear.max_nonlinear_count) {
                const auto expr = make_square(prev.expr, nonlinear.max_nonlinear_count);
                const auto sig = square_values(prev.sig);
                try_candidate(expr, sig, [&prev] { return square_values(prev.sem); }, cost);
            }

            if (enable_periodic) {
                const auto expr = make_sin8(prev.expr);
                const auto sig = sin8_values(prev.sig);
                try_candidate(expr, sig, [&prev] { return sin8_values(prev.sem); }, cost);
            }
            if (enable_periodic) {
                const auto expr = make_cos8(prev.expr);
                const auto sig = cos8_values(prev.sig);
                try_candidate(expr, sig, [&prev] { return cos8_values(prev.sem); }, cost);
            }

            for (const int constant : const_values) {
                const auto expr = make_mul_const(prev.expr, constant);
                const auto sig = mul_const_values(prev.sig, constant);
                try_candidate(expr, sig, [&prev, constant] { return mul_const_values(prev.sem, constant); }, cost);
            }

            for (const int shift : shift_set()) {
                auto expr = make_shl_const(prev.expr, shift);
                auto sig = shl_values(prev.sig, shift);
                try_candidate(expr, sig, [&prev, shift] { return shl_values(prev.sem, shift); }, cost);

                expr = make_shr_const(prev.expr, shift);
                sig = shr_values(prev.sig, shift);
                try_candidate(expr, sig, [&prev, shift] { return shr_values(prev.sem, shift); }, cost);
            }
        }

        for (int left_cost = 1; left_cost < cost; ++left_cost) {
            const int right_cost = cost - left_cost - 1;
            if (right_cost < 1) {
                continue;
            }

            std::vector<Candidate> left_items;
            std::vector<Candidate> right_items;
            left_items.reserve(levels[static_cast<std::size_t>(left_cost)].size());
            right_items.reserve(levels[static_cast<std::size_t>(right_cost)].size());
            for (const auto& [_, candidate] : levels[static_cast<std::size_t>(left_cost)]) {
                left_items.push_back(candidate);
            }
            for (const auto& [_, candidate] : levels[static_cast<std::size_t>(right_cost)]) {
                right_items.push_back(candidate);
            }

            if (left_cost <= right_cost) {
                for (const auto& left : left_items) {
                    for (const auto& right : right_items) {
                        if (left_cost == right_cost && left.expr->shape_key > right.expr->shape_key) {
                            continue;
                        }

                        const int combined_nonlinear = left.expr->nonlinear_count + right.expr->nonlinear_count;
                        if (combined_nonlinear <= nonlinear.max_nonlinear_count) {
                            auto expr = make_add(left.expr, right.expr);
                            auto sig = add_values(left.sig, right.sig);
                            try_candidate(expr, sig, [&left, &right] { return add_values(left.sem, right.sem); }, cost);

                            expr = make_xor(left.expr, right.expr);
                            sig = xor_values(left.sig, right.sig);
                            try_candidate(expr, sig, [&left, &right] { return xor_values(left.sem, right.sem); }, cost);
                        }

                        const bool mul_is_allowed = (
                            expr_equal(left.expr, right.expr) &&
                            left.expr->nonlinear_count + 1 <= nonlinear.max_nonlinear_count
                        ) || (
                            (nonlinear.allow_mul_of_nonlinear || (left.expr->nonlinear_count == 0 && right.expr->nonlinear_count == 0)) &&
                            left.expr->nonlinear_count + right.expr->nonlinear_count + 1 <= nonlinear.max_nonlinear_count
                        );
                        if (mul_is_allowed) {
                            const auto expr = make_mul(
                                left.expr,
                                right.expr,
                                nonlinear.allow_mul_of_nonlinear,
                                nonlinear.max_nonlinear_count
                            );
                            const auto sig = mul_values(left.sig, right.sig);
                            try_candidate(
                                expr,
                                sig,
                                [&left, &right] { return mul_values(left.sem, right.sem); },
                                cost,
                                "mul"
                            );
                        }
                    }
                }
            }

            for (const auto& left : left_items) {
                for (const auto& right : right_items) {
                    if (left.expr->nonlinear_count + right.expr->nonlinear_count > nonlinear.max_nonlinear_count) {
                        continue;
                    }
                    const auto expr = make_sub(left.expr, right.expr);
                    const auto sig = sub_values(left.sig, right.sig);
                    try_candidate(expr, sig, [&left, &right] { return sub_values(left.sem, right.sem); }, cost);
                }
            }
        }

        if (log_progress) {
            log_level(cost, stats[static_cast<std::size_t>(cost)], levels[static_cast<std::size_t>(cost)].size());
        }
        if (best_target_expr) {
            return {best_target_expr, vector_to_level_map(stats), discovered_consts, bucket_summary};
        }
    }

    return {best_target_expr, vector_to_level_map(stats), discovered_consts, bucket_summary};
}

EnumerativeRoundResult find_enumerative_sample_round(
    const SampleSet& sample_set,
    const std::vector<int>& const_values,
    int max_cost,
    bool log_progress,
    const NonlinearConfig& nonlinear,
    bool enable_periodic
) {
    std::vector<FlatMap<Bytes, SampleCandidate, VectorBytesHash>> levels(static_cast<std::size_t>(max_cost + 1));
    FlatMap<Bytes, SampleCandidate, VectorBytesHash> best_by_sem;
    std::vector<FlatSet<ExprId>> seen_ids(static_cast<std::size_t>(max_cost + 1));
    FlatSet<ExprId> reduced_ids_seen;
    std::vector<LevelStats> stats(static_cast<std::size_t>(max_cost + 1));

    std::size_t best_reserve = 0;
    for (int cost = 1; cost <= max_cost; ++cost) {
        const auto hint = level_reserve_hint(cost);
        levels[static_cast<std::size_t>(cost)].reserve(hint);
        seen_ids[static_cast<std::size_t>(cost)].reserve(hint);
        best_reserve += hint;
    }
    best_by_sem.reserve(best_reserve);

    std::bitset<256> discovered_consts;
    const auto known_consts = values_to_bitset(const_values);
    ExprHandle best_target_expr{};
    BucketSummary bucket_summary;

    auto accept = [&](const SampleCandidate& candidate, LevelStats& cost_stats, const char* op = nullptr) -> bool {
        const auto current_best = best_by_sem.find(candidate.sem);
        if (current_best != best_by_sem.end() && !better(candidate.expr, current_best->second.expr)) {
            cost_stats.sem_pruned += 1;
            if (op && std::string_view(op) == "mul") {
                cost_stats.sem_pruned_mul += 1;
            }
            return false;
        }

        if (current_best != best_by_sem.end()) {
            levels[static_cast<std::size_t>(current_best->second.expr->cost)].erase(candidate.sem);
        }

        best_by_sem[candidate.sem] = candidate;
        levels[static_cast<std::size_t>(candidate.expr->cost)][candidate.sem] = candidate;
        cost_stats.accepted += 1;
        if (op && std::string_view(op) == "mul") {
            cost_stats.accepted_mul_candidates += 1;
        }
        if (candidate.sem == sample_set.values && better(candidate.expr, best_target_expr)) {
            best_target_expr = candidate.expr;
        }
        return true;
    };

    auto try_candidate = [&](const ExprHandle& expr, const Bytes& sem, int target_cost, const char* op = nullptr) {
        auto& cost_stats = stats[static_cast<std::size_t>(target_cost)];
        cost_stats.generated += 1;
        cost_stats.max_nonlinear_depth_seen = std::max<std::uint64_t>(
            cost_stats.max_nonlinear_depth_seen,
            static_cast<std::uint64_t>(expr->nonlinear_count)
        );
        if (op && std::string_view(op) == "mul") {
            cost_stats.generated_mul_candidates += 1;
        }

        if (expr->cost != target_cost) {
            if (expr->cost < target_cost && sem == sample_set.values && (!best_target_expr || expr->cost < best_target_expr->cost)) {
                if (reduced_ids_seen.insert(expr->id).second) {
                    discover_derived_consts(expr, known_consts, discovered_consts);
                }
            }
            cost_stats.rule_pruned += 1;
            return;
        }

        if (!seen_ids[static_cast<std::size_t>(target_cost)].insert(expr->id).second) {
            cost_stats.shape_pruned += 1;
            if (op && std::string_view(op) == "mul") {
                cost_stats.shape_pruned_mul += 1;
            }
            return;
        }

        cost_stats.full_eval += 1;
        accept(SampleCandidate{expr, sem}, cost_stats, op);
    };

    auto& base_stats = stats[1];
    Bytes var_sem = sample_set.points;
    SampleCandidate var_candidate{var_expr(), var_sem};
    base_stats.generated += 1;
    base_stats.full_eval += 1;
    seen_ids[1].insert(var_candidate.expr->id);
    accept(var_candidate, base_stats);

    for (const int constant : const_values) {
        const auto node = const_expr(constant);
        Bytes const_sem(sample_set.points.size(), static_cast<std::uint8_t>(constant & MASK));
        base_stats.generated += 1;
        base_stats.full_eval += 1;
        seen_ids[1].insert(node->id);
        accept(SampleCandidate{node, const_sem}, base_stats);
    }

    if (best_target_expr) {
        return {best_target_expr, vector_to_level_map(stats), discovered_consts, bucket_summary};
    }

    for (int cost = 2; cost <= max_cost; ++cost) {
        std::vector<SampleCandidate> prev_items;
        prev_items.reserve(levels[static_cast<std::size_t>(cost - 1)].size());
        for (const auto& [_, candidate] : levels[static_cast<std::size_t>(cost - 1)]) {
            prev_items.push_back(candidate);
        }

        for (const auto& prev : prev_items) {
            if (prev.expr->nonlinear_count + 1 <= nonlinear.max_nonlinear_count) {
                const auto expr = make_square(prev.expr, nonlinear.max_nonlinear_count);
                try_candidate(expr, square_values(prev.sem), cost);
            }
            if (enable_periodic) {
                const auto expr = make_sin8(prev.expr);
                try_candidate(expr, sin8_values(prev.sem), cost);
            }
            if (enable_periodic) {
                const auto expr = make_cos8(prev.expr);
                try_candidate(expr, cos8_values(prev.sem), cost);
            }
            for (const int constant : const_values) {
                const auto expr = make_mul_const(prev.expr, constant);
                try_candidate(expr, mul_const_values(prev.sem, constant), cost);
            }
            for (const int shift : shift_set()) {
                auto expr = make_shl_const(prev.expr, shift);
                try_candidate(expr, shl_values(prev.sem, shift), cost);
                expr = make_shr_const(prev.expr, shift);
                try_candidate(expr, shr_values(prev.sem, shift), cost);
            }
        }

        for (int left_cost = 1; left_cost < cost; ++left_cost) {
            const int right_cost = cost - left_cost - 1;
            if (right_cost < 1) {
                continue;
            }

            std::vector<SampleCandidate> left_items;
            std::vector<SampleCandidate> right_items;
            left_items.reserve(levels[static_cast<std::size_t>(left_cost)].size());
            right_items.reserve(levels[static_cast<std::size_t>(right_cost)].size());
            for (const auto& [_, candidate] : levels[static_cast<std::size_t>(left_cost)]) {
                left_items.push_back(candidate);
            }
            for (const auto& [_, candidate] : levels[static_cast<std::size_t>(right_cost)]) {
                right_items.push_back(candidate);
            }

            if (left_cost <= right_cost) {
                for (const auto& left : left_items) {
                    for (const auto& right : right_items) {
                        if (left_cost == right_cost && left.expr->shape_key > right.expr->shape_key) {
                            continue;
                        }

                        const int combined_nonlinear = left.expr->nonlinear_count + right.expr->nonlinear_count;
                        if (combined_nonlinear <= nonlinear.max_nonlinear_count) {
                            auto expr = make_add(left.expr, right.expr);
                            try_candidate(expr, add_values(left.sem, right.sem), cost);
                            expr = make_xor(left.expr, right.expr);
                            try_candidate(expr, xor_values(left.sem, right.sem), cost);
                        }

                        const bool mul_is_allowed = (
                            expr_equal(left.expr, right.expr) &&
                            left.expr->nonlinear_count + 1 <= nonlinear.max_nonlinear_count
                        ) || (
                            (nonlinear.allow_mul_of_nonlinear || (left.expr->nonlinear_count == 0 && right.expr->nonlinear_count == 0)) &&
                            left.expr->nonlinear_count + right.expr->nonlinear_count + 1 <= nonlinear.max_nonlinear_count
                        );
                        if (mul_is_allowed) {
                            const auto expr = make_mul(
                                left.expr,
                                right.expr,
                                nonlinear.allow_mul_of_nonlinear,
                                nonlinear.max_nonlinear_count
                            );
                            try_candidate(expr, mul_values(left.sem, right.sem), cost, "mul");
                        }
                    }
                }
            }

            for (const auto& left : left_items) {
                for (const auto& right : right_items) {
                    if (left.expr->nonlinear_count + right.expr->nonlinear_count > nonlinear.max_nonlinear_count) {
                        continue;
                    }
                    const auto expr = make_sub(left.expr, right.expr);
                    try_candidate(expr, sub_values(left.sem, right.sem), cost);
                }
            }
        }

        if (log_progress) {
            log_level(cost, stats[static_cast<std::size_t>(cost)], levels[static_cast<std::size_t>(cost)].size());
        }
        if (best_target_expr) {
            return {best_target_expr, vector_to_level_map(stats), discovered_consts, bucket_summary};
        }
    }

    return {best_target_expr, vector_to_level_map(stats), discovered_consts, bucket_summary};
}

}  // namespace

}  // namespace synth
