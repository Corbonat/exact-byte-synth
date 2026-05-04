#include "synth/core.hpp"

#include <iostream>
#include <optional>
#include <string>

namespace {

struct CodegenOptions {
    std::optional<std::string> benchmark;
    std::string output = "generated_function.v";
    std::string module_name = "generated_function";
    std::optional<int> max_cost;
    std::string nonlinear_mode = synth::DEFAULT_NONLINEAR_MODE;
};

CodegenOptions parse_args(int argc, char** argv) {
    CodegenOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--benchmark" && index + 1 < argc) {
            options.benchmark = argv[++index];
        } else if (arg == "--output" && index + 1 < argc) {
            options.output = argv[++index];
        } else if (arg == "--module-name" && index + 1 < argc) {
            options.module_name = argv[++index];
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        synth::Semantics256 target{};
        std::string target_name = "oracle";
        bool enable_periodic = true;
        if (options.benchmark.has_value()) {
            const auto* benchmark = synth::find_benchmark(options.benchmark.value());
            if (benchmark == nullptr) {
                std::cout << "Unknown benchmark: " << options.benchmark.value() << '\n';
                return 1;
            }
            target = synth::build_target(benchmark->oracle);
            target_name = benchmark->name;
            enable_periodic = benchmark->category == "periodic";
        } else {
            target = synth::collect_target();
        }

        const int max_cost = options.max_cost.value_or(synth::MAX_COST);
        const auto result = synth::synthesize(target, max_cost, true, synth::top1_sig_bucket_limit, options.nonlinear_mode, enable_periodic);
        if (!result.has_value()) {
            std::cout << "Code generation aborted: no matching expression was found.\n";
            return 1;
        }

        const auto [ok, failing_x] = synth::verify_expression(result->expr, target);
        if (!ok) {
            std::cout << "Code generation aborted: verification failed at x=" << failing_x.value() << ".\n";
            return 1;
        }

        synth::write_text_file(options.output, synth::emit_verilog(result->expr, options.module_name));
        std::cout << "Wrote " << options.output << " for " << target_name << " using " << result->backend << " backend\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
