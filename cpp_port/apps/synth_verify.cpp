#include "synth/core.hpp"

#include <iostream>
#include <optional>
#include <string>

namespace {

struct VerifyOptions {
    std::optional<int> max_cost;
    std::string nonlinear_mode = synth::DEFAULT_NONLINEAR_MODE;
};

VerifyOptions parse_args(int argc, char** argv) {
    VerifyOptions options;
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
        const int max_cost = options.max_cost.value_or(synth::MAX_COST);
        const auto result = synth::synthesize(target, max_cost, true, synth::top1_sig_bucket_limit, options.nonlinear_mode, true);
        if (!result.has_value()) {
            std::cout << "Verification aborted: no matching expression was found.\n";
            return 1;
        }

        const auto [ok, failing_x] = synth::verify_expression(result->expr, target);
        if (!ok) {
            std::cout << "Verification failed at x=" << failing_x.value() << ".\n";
            return 1;
        }

        std::cout
            << "Verified expression on all " << synth::DOMAIN_SIZE << " inputs: "
            << result->expr->to_source() << " mod 256 ("
            << result->backend << ", cost=" << result->expr->cost << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
