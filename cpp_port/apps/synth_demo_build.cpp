#include "synth/core.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::filesystem::path kDemoDir = "fpga_demo";
const std::filesystem::path kActiveVerilog = kDemoDir / "generated_function.v";
const std::filesystem::path kReportsDir = kDemoDir / "reports";
const std::filesystem::path kArchiveDir = kDemoDir / "generated_cases";

struct DemoOptions {
    std::optional<std::string> benchmark;
    bool list_benchmarks = false;
    std::optional<int> max_cost;
    std::string nonlinear_mode = synth::DEFAULT_NONLINEAR_MODE;
};

DemoOptions parse_args(int argc, char** argv) {
    DemoOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--benchmark" && index + 1 < argc) {
            options.benchmark = argv[++index];
        } else if (arg == "--list-benchmarks") {
            options.list_benchmarks = true;
        } else if (arg == "--max-cost" && index + 1 < argc) {
            options.max_cost = std::stoi(argv[++index]);
        } else if (arg == "--nonlinear-mode" && index + 1 < argc) {
            options.nonlinear_mode = argv[++index];
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return options;
}

struct DemoTarget {
    std::string name;
    std::string description;
    std::optional<bool> expected_found;
    synth::Semantics256 target;
    bool enable_periodic = false;
};

DemoTarget resolve_target(const std::optional<std::string>& benchmark_name) {
    if (!benchmark_name.has_value()) {
        return {"oracle", "Current oracle from current_oracle.cpp", std::nullopt, synth::collect_target(), true};
    }

    const auto* benchmark = synth::find_benchmark(benchmark_name.value());
    if (benchmark == nullptr) {
        throw std::runtime_error("Unknown benchmark: " + benchmark_name.value());
    }
    return {
        benchmark->name,
        benchmark->description,
        benchmark->expected_found,
        synth::build_target(benchmark->oracle),
        benchmark->category == "periodic"
    };
}

void ensure_demo_dirs() {
    synth::ensure_directory(kDemoDir);
    synth::ensure_directory(kReportsDir);
    synth::ensure_directory(kArchiveDir);
}

std::string utc_now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
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

std::string json_object_of_timings(const std::vector<std::pair<std::string, double>>& timings) {
    std::ostringstream out;
    out << "{";
    for (std::size_t index = 0; index < timings.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << "\"" << synth::json_escape(timings[index].first) << "\": " << std::fixed << std::setprecision(6) << timings[index].second;
    }
    out << "}";
    return out.str();
}

void print_benchmark_list() {
    std::cout << "recommended_demo_cases:\n";
    for (const auto& name : synth::recommended_demo_cases()) {
        const auto* benchmark = synth::find_benchmark(name);
        if (benchmark != nullptr) {
            std::cout << "  " << benchmark->name << ": " << benchmark->description << "\n";
        }
    }
}

std::filesystem::path write_report(
    const DemoTarget& target,
    const synth::SearchAttempt& attempt,
    const std::optional<synth::SearchResult>& result,
    bool verified,
    const std::optional<std::filesystem::path>& active_verilog,
    const std::optional<std::filesystem::path>& archived_verilog
) {
    const auto path = kReportsDir / (target.name + ".json");
    std::ostringstream out;
    out
        << "{\n"
        << "  \"benchmark\": \"" << synth::json_escape(target.name) << "\",\n"
        << "  \"description\": \"" << synth::json_escape(target.description) << "\",\n"
        << "  \"expected_found\": ";
    if (target.expected_found.has_value()) {
        out << (target.expected_found.value() ? "true" : "false");
    } else {
        out << "null";
    }
    out
        << ",\n"
        << "  \"found\": " << (result.has_value() ? "true" : "false") << ",\n"
        << "  \"verified\": " << (verified ? "true" : "false") << ",\n"
        << "  \"backend\": " << (result.has_value() ? ("\"" + synth::json_escape(result->backend) + "\"") : "null") << ",\n"
        << "  \"cost\": " << (result.has_value() ? std::to_string(result->expr->cost) : "null") << ",\n"
        << "  \"expression\": " << (result.has_value() ? ("\"" + synth::json_escape(result->expr->to_source()) + "\"") : "null") << ",\n"
        << "  \"fallback_triggered\": " << (attempt.fallback_triggered() ? "true" : "false") << ",\n"
        << "  \"attempted_backends\": " << json_array_of_strings(attempt.attempted_backends) << ",\n"
        << "  \"backend_timings_s\": " << json_object_of_timings(attempt.backend_timings_s) << ",\n"
        << "  \"solved_by_backend_index\": " << (attempt.solved_by_backend_index.has_value() ? std::to_string(attempt.solved_by_backend_index.value()) : "null") << ",\n"
        << "  \"active_verilog\": " << (active_verilog.has_value() ? ("\"" + synth::json_escape(active_verilog->string()) + "\"") : "null") << ",\n"
        << "  \"archived_verilog\": " << (archived_verilog.has_value() ? ("\"" + synth::json_escape(archived_verilog->string()) + "\"") : "null") << ",\n"
        << "  \"report_path\": \"" << synth::json_escape(path.string()) << "\",\n"
        << "  \"built_at_utc\": \"" << utc_now_iso8601() << "\"\n"
        << "}\n";
    synth::write_text_file(path, out.str());
    return path;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        if (options.list_benchmarks) {
            print_benchmark_list();
            return 0;
        }

        ensure_demo_dirs();
        const auto target = resolve_target(options.benchmark);
        const auto attempt = synth::run_search_attempt(
            target.target,
            options.max_cost.value_or(synth::MAX_COST),
            false,
            synth::top1_sig_bucket_limit,
            options.nonlinear_mode,
            target.enable_periodic
        );

        if (!attempt.result.has_value()) {
            const auto report_path = write_report(target, attempt, std::nullopt, false, std::nullopt, std::nullopt);
            std::cout << "build_status=FAILED benchmark=" << target.name << " reason=no_exact_match report=" << report_path.string() << "\n";
            return 1;
        }

        const auto [verified, failing_x] = synth::verify_expression(attempt.result->expr, target.target);
        if (!verified) {
            const auto report_path = write_report(target, attempt, attempt.result, false, std::nullopt, std::nullopt);
            std::cout
                << "build_status=FAILED benchmark=" << target.name
                << " reason=verify_failed x=" << (failing_x.has_value() ? std::to_string(failing_x.value()) : "-")
                << " report=" << report_path.string() << "\n";
            return 1;
        }

        const auto verilog = synth::emit_verilog(attempt.result->expr, "generated_function");
        synth::write_text_file(kActiveVerilog, verilog);
        const auto archived_path = kArchiveDir / (target.name + ".v");
        synth::write_text_file(archived_path, verilog);
        const auto report_path = write_report(target, attempt, attempt.result, true, kActiveVerilog, archived_path);

        std::cout << "benchmark=" << target.name << "\n";
        std::cout << "description=" << target.description << "\n";
        std::cout << "backend=" << attempt.result->backend << "\n";
        std::cout << "cost=" << attempt.result->expr->cost << "\n";
        std::cout << "expression=" << attempt.result->expr->to_source() << "\n";
        std::cout << "verify=PASS on " << synth::DOMAIN_SIZE << "/" << synth::DOMAIN_SIZE << "\n";
        std::cout << "active_verilog=" << kActiveVerilog.string() << "\n";
        std::cout << "archived_verilog=" << archived_path.string() << "\n";
        std::cout << "report=" << report_path.string() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
