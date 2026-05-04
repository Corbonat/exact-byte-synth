#include "synth/core.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct BenchmarkOptions {
    std::vector<std::string> names;
    int max_cost = synth::MAX_COST;
    std::string sig_mode = "top1";
    std::string mode = "full_domain";
    std::string nonlinear_mode = synth::DEFAULT_NONLINEAR_MODE;
    bool log_progress = false;
    std::string json_out;
};

struct BenchmarkResult {
    std::string name;
    std::string category;
    std::string description;
    bool expected_found = false;
    bool found = false;
    bool verified = false;
    std::string backend = "none";
    std::optional<int> cost;
    double elapsed_s = 0.0;
    std::uint64_t full_eval = 0;
    std::uint64_t sig_pruned = 0;
    std::uint64_t shape_pruned = 0;
    bool fallback_triggered = false;
    std::vector<std::string> attempted_backends;
    std::map<std::string, double> backend_timings_s;
    bool solved_by_first_backend = false;
    std::optional<int> solved_by_backend_index;
    int cegis_rounds = 0;
    std::optional<int> cegis_sample_size;
    std::vector<int> counterexamples;
    int const_promotion_rounds = 0;
    int promoted_consts_total = 0;
    std::uint64_t max_bucket_count = 0;
    double avg_bucket_count_on_heavy_levels = 0.0;
    std::optional<std::string> expression;
    std::string nonlinear_mode;
    std::uint64_t generated_mul_candidates = 0;
    std::uint64_t accepted_mul_candidates = 0;
    std::uint64_t shape_pruned_mul = 0;
    std::uint64_t sem_pruned_mul = 0;
    std::uint64_t max_nonlinear_depth_seen = 0;
};

BenchmarkOptions parse_args(int argc, char** argv) {
    BenchmarkOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--names") {
            while (index + 1 < argc) {
                const std::string next = argv[index + 1];
                if (next.rfind("--", 0) == 0) {
                    break;
                }
                options.names.push_back(argv[++index]);
            }
        } else if (arg == "--max-cost" && index + 1 < argc) {
            options.max_cost = std::stoi(argv[++index]);
        } else if (arg == "--sig-mode" && index + 1 < argc) {
            options.sig_mode = argv[++index];
        } else if (arg == "--mode" && index + 1 < argc) {
            options.mode = argv[++index];
        } else if (arg == "--nonlinear-mode" && index + 1 < argc) {
            options.nonlinear_mode = argv[++index];
        } else if (arg == "--json-out" && index + 1 < argc) {
            options.json_out = argv[++index];
        } else if (arg == "--log-progress") {
            options.log_progress = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return options;
}

synth::SigBucketLimitFn sig_strategy(const std::string& mode) {
    if (mode == "top1") {
        return synth::top1_sig_bucket_limit;
    }
    if (mode == "selective") {
        return synth::selective_sig_bucket_limit;
    }
    throw std::runtime_error("Unsupported signature strategy: " + mode);
}

std::vector<const synth::BenchmarkCase*> select_cases(const std::vector<std::string>& names) {
    std::vector<const synth::BenchmarkCase*> cases;
    if (names.empty()) {
        for (const auto& benchmark : synth::benchmarks()) {
            cases.push_back(&benchmark);
        }
        return cases;
    }

    for (const auto& name : names) {
        const auto* benchmark = synth::find_benchmark(name);
        if (benchmark == nullptr) {
            throw std::runtime_error("Unknown benchmark: " + name);
        }
        cases.push_back(benchmark);
    }
    return cases;
}

std::map<int, synth::LevelStats> combine_attempt_stats(const synth::SearchAttempt& attempt) {
    std::map<int, synth::LevelStats> combined;
    for (const auto& [cost, stats] : attempt.signature_stats) {
        synth::merge_level_stats(combined[cost], stats);
    }
    for (const auto& [cost, stats] : attempt.fallback_stats) {
        synth::merge_level_stats(combined[cost], stats);
    }
    return combined;
}

synth::BucketSummary combine_attempt_bucket_summary(const synth::SearchAttempt& attempt) {
    synth::BucketSummary summary;
    if (attempt.signature_bucket_summary.has_value()) {
        summary = synth::merge_bucket_summaries(summary, attempt.signature_bucket_summary.value());
    }
    if (attempt.fallback_bucket_summary.has_value()) {
        summary = synth::merge_bucket_summaries(summary, attempt.fallback_bucket_summary.value());
    }
    return summary;
}

std::uint64_t aggregate_metric(const std::map<int, synth::LevelStats>& stats_map, std::uint64_t synth::LevelStats::*field) {
    std::uint64_t total = 0;
    for (const auto& [_, stats] : stats_map) {
        total += stats.*field;
    }
    return total;
}

BenchmarkResult run_case(
    const synth::BenchmarkCase& benchmark,
    int max_cost,
    const std::string& sig_mode,
    bool log_progress,
    const std::string& mode,
    const std::string& nonlinear_mode
) {
    const auto target = synth::build_target(benchmark.oracle);
    const auto strategy = sig_strategy(sig_mode);
    const bool enable_periodic = benchmark.category == "periodic";
    const auto started = std::chrono::steady_clock::now();

    synth::SearchAttempt attempt;
    if (mode == "full_domain" || mode == "core_only" || mode == "full_system") {
        attempt = synth::run_search_attempt(target, max_cost, log_progress, strategy, nonlinear_mode, enable_periodic);
    } else if (mode == "cegis") {
        attempt = synth::synthesize_cegis(benchmark.oracle, synth::initial_cegis_points(), 32, max_cost, log_progress, nonlinear_mode, enable_periodic);
    } else {
        throw std::runtime_error("Unsupported benchmark mode: " + mode);
    }

    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto combined_stats = combine_attempt_stats(attempt);
    const auto bucket_summary = combine_attempt_bucket_summary(attempt);

    BenchmarkResult result;
    result.name = benchmark.name;
    result.category = benchmark.category;
    result.description = benchmark.description;
    result.expected_found = benchmark.expected_found;
    result.elapsed_s = elapsed;
    result.fallback_triggered = attempt.fallback_triggered();
    result.attempted_backends = attempt.attempted_backends;
    result.solved_by_first_backend = attempt.solved_by_first_backend();
    result.solved_by_backend_index = attempt.solved_by_backend_index;
    result.cegis_rounds = static_cast<int>(attempt.cegis_rounds.size());
    result.nonlinear_mode = nonlinear_mode;
    result.full_eval = aggregate_metric(combined_stats, &synth::LevelStats::full_eval);
    result.sig_pruned = aggregate_metric(combined_stats, &synth::LevelStats::sig_pruned);
    result.shape_pruned = aggregate_metric(combined_stats, &synth::LevelStats::shape_pruned);
    result.generated_mul_candidates = aggregate_metric(combined_stats, &synth::LevelStats::generated_mul_candidates);
    result.accepted_mul_candidates = aggregate_metric(combined_stats, &synth::LevelStats::accepted_mul_candidates);
    result.shape_pruned_mul = aggregate_metric(combined_stats, &synth::LevelStats::shape_pruned_mul);
    result.sem_pruned_mul = aggregate_metric(combined_stats, &synth::LevelStats::sem_pruned_mul);
    result.max_bucket_count = bucket_summary.max_bucket_count;
    result.avg_bucket_count_on_heavy_levels = bucket_summary.avg_bucket_count_on_heavy_levels();

    for (const auto& [name, seconds] : attempt.backend_timings_s) {
        result.backend_timings_s[name] = seconds;
    }
    for (const auto& round : attempt.cegis_rounds) {
        if (round.counterexample.has_value()) {
            result.counterexamples.push_back(round.counterexample.value());
        }
    }
    if (!attempt.cegis_rounds.empty()) {
        result.cegis_sample_size = attempt.cegis_rounds.back().sample_size;
    }

    const auto promotion_rounds = static_cast<int>(
        attempt.signature_promotion_rounds.size() + attempt.fallback_promotion_rounds.size()
    );
    result.const_promotion_rounds = promotion_rounds;
    for (const auto& round : attempt.signature_promotion_rounds) {
        result.promoted_consts_total += static_cast<int>(round.derived_consts_promoted.size());
    }
    for (const auto& round : attempt.fallback_promotion_rounds) {
        result.promoted_consts_total += static_cast<int>(round.derived_consts_promoted.size());
    }

    for (const auto& [_, stats] : combined_stats) {
        result.max_nonlinear_depth_seen = std::max(result.max_nonlinear_depth_seen, stats.max_nonlinear_depth_seen);
    }

    if (attempt.result.has_value()) {
        result.found = true;
        result.backend = attempt.result->backend;
        result.cost = attempt.result->expr->cost;
        result.expression = attempt.result->expr->to_source();
        result.verified = synth::verify_expression(attempt.result->expr, target).first;
    }

    return result;
}

void print_table(const std::vector<BenchmarkResult>& rows) {
    std::cout
        << std::left
        << std::setw(18) << "name"
        << std::setw(16) << "category"
        << std::setw(7) << "found"
        << std::setw(24) << "backend"
        << std::setw(8) << "dispatch"
        << std::setw(6) << "cost"
        << std::setw(10) << "elapsed_s"
        << std::setw(10) << "full_eval"
        << std::setw(11) << "sig_pruned"
        << std::setw(13) << "shape_pruned"
        << std::setw(9) << "fallback"
        << std::setw(7) << "cegis"
        << std::setw(8) << "promote"
        << '\n';

    for (const auto& row : rows) {
        std::cout
            << std::left
            << std::setw(18) << row.name
            << std::setw(16) << row.category
            << std::setw(7) << (row.found ? "True" : "False")
            << std::setw(24) << row.backend
            << std::setw(8) << (row.solved_by_backend_index.has_value() ? std::to_string(row.solved_by_backend_index.value()) : "-")
            << std::setw(6) << (row.cost.has_value() ? std::to_string(row.cost.value()) : "-")
            << std::setw(10) << std::fixed << std::setprecision(3) << row.elapsed_s
            << std::setw(10) << row.full_eval
            << std::setw(11) << row.sig_pruned
            << std::setw(13) << row.shape_pruned
            << std::setw(9) << (row.fallback_triggered ? "True" : "False")
            << std::setw(7) << (row.cegis_rounds > 0 ? std::to_string(row.cegis_rounds) : "-")
            << std::setw(8) << row.promoted_consts_total
            << '\n';
    }
}

void print_summary(const std::vector<BenchmarkResult>& rows, const BenchmarkOptions& options) {
    int found = 0;
    int verified = 0;
    int fallback_count = 0;
    int promotion_hits = 0;
    int solved_by_first = 0;
    int cegis_rounds = 0;
    double total_elapsed = 0.0;
    std::uint64_t total_full_eval = 0;
    std::uint64_t total_sig_pruned = 0;
    std::uint64_t generated_mul = 0;
    std::uint64_t accepted_mul = 0;
    std::uint64_t shape_pruned_mul = 0;
    std::uint64_t sem_pruned_mul = 0;
    std::uint64_t max_nonlinear = 0;

    for (const auto& row : rows) {
        found += row.found ? 1 : 0;
        verified += row.verified ? 1 : 0;
        fallback_count += row.fallback_triggered ? 1 : 0;
        promotion_hits += row.promoted_consts_total > 0 ? 1 : 0;
        solved_by_first += row.solved_by_first_backend ? 1 : 0;
        cegis_rounds += row.cegis_rounds;
        total_elapsed += row.elapsed_s;
        total_full_eval += row.full_eval;
        total_sig_pruned += row.sig_pruned;
        generated_mul += row.generated_mul_candidates;
        accepted_mul += row.accepted_mul_candidates;
        shape_pruned_mul += row.shape_pruned_mul;
        sem_pruned_mul += row.sem_pruned_mul;
        max_nonlinear = std::max(max_nonlinear, row.max_nonlinear_depth_seen);
    }

    std::cout << '\n';
    std::cout
        << "suite_cases=" << rows.size()
        << " mode=" << options.mode
        << " nonlinear_mode=" << options.nonlinear_mode
        << " sig_mode=" << options.sig_mode
        << " max_cost=" << options.max_cost
        << '\n';
    std::cout << "found=" << found << " verified=" << verified << " total_elapsed_s=" << std::fixed << std::setprecision(3) << total_elapsed << '\n';
    std::cout << "total_full_eval=" << total_full_eval << " total_sig_pruned=" << total_sig_pruned << '\n';
    std::cout
        << "fallback_triggered=" << fallback_count
        << " promotion_hits=" << promotion_hits
        << " solved_by_first_backend=" << solved_by_first
        << " cegis_rounds=" << cegis_rounds
        << '\n';
    std::cout
        << "generated_mul_candidates=" << generated_mul
        << " accepted_mul_candidates=" << accepted_mul
        << " shape_pruned_mul=" << shape_pruned_mul
        << " sem_pruned_mul=" << sem_pruned_mul
        << " max_nonlinear_depth_seen=" << max_nonlinear
        << '\n';
}

std::string json_array_of_strings(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << "\"" << synth::json_escape(values[index]) << "\"";
    }
    out << "]";
    return out.str();
}

std::string json_array_of_ints(const std::vector<int>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << values[index];
    }
    out << "]";
    return out.str();
}

std::string json_object_of_timings(const std::map<std::string, double>& timings) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    for (const auto& [name, seconds] : timings) {
        if (!first) {
            out << ", ";
        }
        first = false;
        out << "\"" << synth::json_escape(name) << "\": " << std::fixed << std::setprecision(6) << seconds;
    }
    out << "}";
    return out.str();
}

void write_json(const std::vector<BenchmarkResult>& rows, const std::filesystem::path& path) {
    if (path.has_parent_path()) {
        synth::ensure_directory(path.parent_path());
    }
    std::ostringstream out;
    out << "[\n";
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto& row = rows[index];
        out
            << "  {\n"
            << "    \"name\": \"" << synth::json_escape(row.name) << "\",\n"
            << "    \"category\": \"" << synth::json_escape(row.category) << "\",\n"
            << "    \"description\": \"" << synth::json_escape(row.description) << "\",\n"
            << "    \"expected_found\": " << (row.expected_found ? "true" : "false") << ",\n"
            << "    \"found\": " << (row.found ? "true" : "false") << ",\n"
            << "    \"verified\": " << (row.verified ? "true" : "false") << ",\n"
            << "    \"backend\": \"" << synth::json_escape(row.backend) << "\",\n"
            << "    \"cost\": " << (row.cost.has_value() ? std::to_string(row.cost.value()) : "null") << ",\n"
            << "    \"elapsed_s\": " << std::fixed << std::setprecision(6) << row.elapsed_s << ",\n"
            << "    \"full_eval\": " << row.full_eval << ",\n"
            << "    \"sig_pruned\": " << row.sig_pruned << ",\n"
            << "    \"shape_pruned\": " << row.shape_pruned << ",\n"
            << "    \"fallback_triggered\": " << (row.fallback_triggered ? "true" : "false") << ",\n"
            << "    \"attempted_backends\": " << json_array_of_strings(row.attempted_backends) << ",\n"
            << "    \"backend_timings_s\": " << json_object_of_timings(row.backend_timings_s) << ",\n"
            << "    \"solved_by_first_backend\": " << (row.solved_by_first_backend ? "true" : "false") << ",\n"
            << "    \"solved_by_backend_index\": " << (row.solved_by_backend_index.has_value() ? std::to_string(row.solved_by_backend_index.value()) : "null") << ",\n"
            << "    \"cegis_rounds\": " << row.cegis_rounds << ",\n"
            << "    \"cegis_sample_size\": " << (row.cegis_sample_size.has_value() ? std::to_string(row.cegis_sample_size.value()) : "null") << ",\n"
            << "    \"counterexamples\": " << json_array_of_ints(row.counterexamples) << ",\n"
            << "    \"const_promotion_rounds\": " << row.const_promotion_rounds << ",\n"
            << "    \"promoted_consts_total\": " << row.promoted_consts_total << ",\n"
            << "    \"max_bucket_count\": " << row.max_bucket_count << ",\n"
            << "    \"avg_bucket_count_on_heavy_levels\": " << std::fixed << std::setprecision(6) << row.avg_bucket_count_on_heavy_levels << ",\n"
            << "    \"expression\": " << (row.expression.has_value() ? ("\"" + synth::json_escape(row.expression.value()) + "\"") : "null") << ",\n"
            << "    \"nonlinear_mode\": \"" << synth::json_escape(row.nonlinear_mode) << "\",\n"
            << "    \"generated_mul_candidates\": " << row.generated_mul_candidates << ",\n"
            << "    \"accepted_mul_candidates\": " << row.accepted_mul_candidates << ",\n"
            << "    \"shape_pruned_mul\": " << row.shape_pruned_mul << ",\n"
            << "    \"sem_pruned_mul\": " << row.sem_pruned_mul << ",\n"
            << "    \"max_nonlinear_depth_seen\": " << row.max_nonlinear_depth_seen << "\n"
            << "  }";
        if (index + 1 != rows.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "]\n";
    synth::write_text_file(path, out.str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        const auto cases = select_cases(options.names);

        std::vector<BenchmarkResult> results;
        results.reserve(cases.size());
        for (std::size_t index = 0; index < cases.size(); ++index) {
            const auto case_started = std::chrono::steady_clock::now();
            auto result = run_case(*cases[index], options.max_cost, options.sig_mode, options.log_progress, options.mode, options.nonlinear_mode);
            const auto case_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - case_started).count();
            std::cout
                << "[" << (index + 1) << "/" << cases.size() << "] " << result.name
                << ": found=" << (result.found ? "true" : "false")
                << " cost=" << (result.cost.has_value() ? std::to_string(result.cost.value()) : "-")
                << " elapsed_s=" << std::fixed << std::setprecision(3) << case_elapsed
                << '\n';
            results.push_back(std::move(result));
        }

        std::cout << '\n';
        print_table(results);
        print_summary(results, options);
        if (!options.json_out.empty()) {
            write_json(results, options.json_out);
            std::cout << "wrote_json=" << options.json_out << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
