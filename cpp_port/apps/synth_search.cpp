#include "synth/core.hpp"

#include <iostream>
#include <string>

namespace {

struct SearchOptions {
    int max_cost = synth::MAX_COST;
    std::string nonlinear_mode = synth::DEFAULT_NONLINEAR_MODE;
};

SearchOptions parse_args(int argc, char** argv) {
    SearchOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--max-cost" && index + 1 < argc) {
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
        const auto target = synth::collect_target();
        const auto result = synth::synthesize(target, options.max_cost, true, synth::top1_sig_bucket_limit, options.nonlinear_mode, true);
        if (!result.has_value()) {
            std::cout << "No expression matched the oracle within the configured search bounds.\n";
            return 1;
        }

        std::cout
            << "Recovered expression with " << result->backend << " backend: "
            << result->expr->to_source() << " mod 256 (cost=" << result->expr->cost << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
