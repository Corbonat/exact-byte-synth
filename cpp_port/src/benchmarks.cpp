#include "synth/core.hpp"

namespace synth {

namespace {

std::uint8_t mask_value(int value) {
    return static_cast<std::uint8_t>(value & MASK);
}

}  // namespace

const std::vector<BenchmarkCase>& benchmarks() {
    static const std::vector<BenchmarkCase> cases = {
        {"affine_linear", "affine", "(3*x + 5) mod 256", [](std::uint8_t x) { return mask_value((3 * x) + 5); }, true},
        {"affine_mulsub", "affine", "(7*x - 13) mod 256", [](std::uint8_t x) { return mask_value((7 * x) - 13); }, true},
        {"xor_shift_add", "xor_shift_add", "(x ^ 17) + (x >> 2)", [](std::uint8_t x) { return mask_value((x ^ 17) + (x >> 2)); }, true},
        {"nested_xor", "promotion", "(x ^ 17) ^ 5", [](std::uint8_t x) { return mask_value((x ^ 17) ^ 5); }, true},
        {"add_tail", "promotion", "(x + 7) + 13", [](std::uint8_t x) { return mask_value((x + 7) + 13); }, true},
        {"shift_add_const", "shift", "(x << 3) + 5", [](std::uint8_t x) { return mask_value((x << 3) + 5); }, true},
        {"shift_mix", "shift", "(x << 3) + (x >> 1)", [](std::uint8_t x) { return mask_value((x << 3) + (x >> 1)); }, true},
        {"mul_xor_const", "mulconst", "(3*x) ^ 5", [](std::uint8_t x) { return mask_value((3 * x) ^ 5); }, true},
        {"mul_shift_add", "mulconst", "(7*x) + (x >> 1)", [](std::uint8_t x) { return mask_value((7 * x) + (x >> 1)); }, true},
        {"sin8", "periodic", "round(127.5 + 127.5*sin(2*pi*x/256))", [](std::uint8_t x) { return sin8_value(x); }, true},
        {"cos8", "periodic", "round(127.5 + 127.5*cos(2*pi*x/256))", [](std::uint8_t x) { return cos8_value(x); }, true},
        {"sin8_plus_5", "periodic", "sin8(x) + 5", [](std::uint8_t x) { return mask_value(sin8_value(x) + 5); }, true},
        {"cos8_xor_17", "periodic", "cos8(x) ^ 17", [](std::uint8_t x) { return mask_value(cos8_value(x) ^ 17); }, true},
        {"sin8_shr_1", "periodic", "sin8(x >> 1)", [](std::uint8_t x) { return sin8_value(static_cast<std::uint8_t>(x >> 1)); }, true},
        {"cos8_3x", "periodic", "cos8(3*x)", [](std::uint8_t x) { return cos8_value(mask_value(3 * x)); }, true},
        {"sin8_x_plus_20", "periodic", "sin8(x + 20)", [](std::uint8_t x) { return sin8_value(mask_value(x + 20)); }, true},
        {"sin8_plus_cos8", "periodic", "sin8(x) + cos8(x)", [](std::uint8_t x) { return mask_value(sin8_value(x) + cos8_value(x)); }, true},
        {"shift_chain", "canonicalization", "(x << 2) << 3", [](std::uint8_t x) { return mask_value((x << 2) << 3); }, true},
        {"mul_chain", "canonicalization", "(3*x) * 5", [](std::uint8_t x) { return mask_value((3 * x) * 5); }, true},
        {"square", "nonlinear", "x*x", [](std::uint8_t x) { return mask_value(x * x); }, true},
        {"square_plus_5", "nonlinear", "(x*x) + 5", [](std::uint8_t x) { return mask_value((x * x) + 5); }, true},
        {"square_xor_17", "nonlinear", "(x*x) ^ 17", [](std::uint8_t x) { return mask_value((x * x) ^ 17); }, true},
        {"shifted_square_10", "nonlinear", "(x - 10)^2", [](std::uint8_t x) { return mask_value((x - 10) * (x - 10)); }, true},
        {"shifted_square_plus_const", "nonlinear", "(x - 10)^2 + 5", [](std::uint8_t x) { return mask_value(((x - 10) * (x - 10)) + 5); }, true},
        {"x_times_x_plus_1", "nonlinear_mul", "x * (x + 1)", [](std::uint8_t x) { return mask_value(x * (x + 1)); }, true},
        {"x_times_x_minus_7", "nonlinear_mul", "x * (x - 7)", [](std::uint8_t x) { return mask_value(x * (x - 7)); }, true},
        {"x_times_x_plus_1_plus_5", "nonlinear_mul", "x * (x + 1) + 5", [](std::uint8_t x) { return mask_value((x * (x + 1)) + 5); }, true},
        {"x_times_x_shr_1", "nonlinear_mul", "x * (x >> 1)", [](std::uint8_t x) { return mask_value(x * (x >> 1)); }, true},
        {"x_plus_1_times_x_shr_1", "nonlinear_mul", "(x + 1) * (x >> 1)", [](std::uint8_t x) { return mask_value((x + 1) * (x >> 1)); }, true},
        {"x_cubed", "nonlinear_cubic", "x^3", [](std::uint8_t x) { return mask_value(x * x * x); }, false},
        {"x_cubed_plus_5", "nonlinear_cubic", "x^3 + 5", [](std::uint8_t x) { return mask_value((x * x * x) + 5); }, false},
        {"square_plus_affine", "nonlinear", "x^2 + 17*x + 5", [](std::uint8_t x) { return mask_value((x * x) + (17 * x) + 5); }, false},
        {"mixed_not_found", "mixed", "(x << 2) ^ ((3*x) + 9)", [](std::uint8_t x) { return mask_value((x << 2) ^ ((3 * x) + 9)); }, false},
        {"awkward_not_found", "mixed", "((x ^ 91) + (x << 1)) - 37", [](std::uint8_t x) { return mask_value(((x ^ 91) + (x << 1)) - 37); }, false},
    };
    return cases;
}

const BenchmarkCase* find_benchmark(std::string_view name) {
    for (const auto& benchmark : benchmarks()) {
        if (benchmark.name == name) {
            return &benchmark;
        }
    }
    return nullptr;
}

std::vector<std::string> recommended_demo_cases() {
    return {
        "affine_linear",
        "nested_xor",
        "xor_shift_add",
        "mul_xor_const",
        "mul_shift_add",
        "sin8",
        "cos8",
        "square",
    };
}

}  // namespace synth
