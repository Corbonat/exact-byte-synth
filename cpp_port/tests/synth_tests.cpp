#include "synth/core.hpp"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct TestFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void expect_true(bool condition, const std::string& message) {
    if (!condition) {
        throw TestFailure(message);
    }
}

void expect_eq(const std::string& actual, const std::string& expected, const std::string& label) {
    if (actual != expected) {
        throw TestFailure(label + ": expected '" + expected + "' got '" + actual + "'");
    }
}

void test_add_canonicalization() {
    const auto expr = synth::make_add(synth::make_add(synth::var_expr(), synth::const_expr(5)), synth::const_expr(3));
    expect_eq(expr->to_source(), "(x + 8)", "Add canonicalization");

    const auto zero = synth::make_add(synth::var_expr(), synth::const_expr(0));
    expect_eq(zero->to_source(), "x", "Add with zero");

    const auto both_const = synth::make_add(synth::const_expr(100), synth::const_expr(200));
    expect_eq(both_const->to_source(), "44", "Add two constants folds mod 256");
}

void test_sub_canonicalization() {
    const auto equal = synth::make_sub(synth::var_expr(), synth::var_expr());
    expect_eq(equal->to_source(), "0", "Sub of equals is 0");

    const auto zero_right = synth::make_sub(synth::var_expr(), synth::const_expr(0));
    expect_eq(zero_right->to_source(), "x", "Sub of zero is identity");

    const auto both_const = synth::make_sub(synth::const_expr(5), synth::const_expr(13));
    expect_eq(both_const->to_source(), "248", "Sub of two constants wraps mod 256");
}

void test_xor_canonicalization() {
    const auto expr = synth::make_xor(synth::make_xor(synth::var_expr(), synth::const_expr(17)), synth::const_expr(5));
    expect_eq(expr->to_source(), "(x ^ 20)", "Xor canonicalization");

    const auto duplicates = synth::make_xor(synth::var_expr(), synth::var_expr());
    expect_eq(duplicates->to_source(), "0", "Xor of duplicates is 0");

    const auto zero = synth::make_xor(synth::var_expr(), synth::const_expr(0));
    expect_eq(zero->to_source(), "x", "Xor with zero is identity");
}

void test_shift_mul_merge() {
    const auto shifted = synth::make_shl_const(synth::make_shl_const(synth::var_expr(), 2), 3);
    expect_eq(shifted->to_source(), "(x << 5)", "Shl merge");

    const auto shr_chain = synth::make_shr_const(synth::make_shr_const(synth::var_expr(), 1), 2);
    expect_eq(shr_chain->to_source(), "(x >> 3)", "Shr merge");

    const auto shl_too_wide = synth::make_shl_const(synth::make_shl_const(synth::var_expr(), 4), 4);
    expect_eq(shl_too_wide->to_source(), "0", "Chained Shl beyond width collapses to 0");

    const auto multiplied = synth::make_mul_const(synth::make_mul_const(synth::var_expr(), 3), 5);
    expect_eq(multiplied->to_source(), "(15 * x)", "MulConst merge");

    const auto mul_zero = synth::make_mul_const(synth::var_expr(), 0);
    expect_eq(mul_zero->to_source(), "0", "MulConst * 0 is 0");

    const auto mul_one = synth::make_mul_const(synth::var_expr(), 1);
    expect_eq(mul_one->to_source(), "x", "MulConst * 1 is identity");
}

void test_mul_square_rule() {
    const auto expr = synth::make_mul(synth::var_expr(), synth::var_expr(), false, 1);
    expect_true(expr->kind == synth::ExprKind::Square, "Mul(e,e) should canonicalize to Square(e)");

    const auto var = synth::var_expr();
    const auto const_mul = synth::make_mul(synth::const_expr(5), var, false, 1);
    expect_eq(const_mul->to_source(), "(5 * x)", "Mul(Const, Var) becomes MulConst");

    const auto zero_mul = synth::make_mul(synth::const_expr(0), var, false, 1);
    expect_eq(zero_mul->to_source(), "0", "Mul by 0 is 0");

    const auto one_mul = synth::make_mul(synth::const_expr(1), var, false, 1);
    expect_eq(one_mul->to_source(), "x", "Mul by 1 is identity");
}

void test_square_folding() {
    const auto con = synth::make_square(synth::const_expr(7), 1);
    expect_eq(con->to_source(), "49", "Square folds constants");
    const auto var_square = synth::make_square(synth::var_expr(), 1);
    expect_true(var_square->kind == synth::ExprKind::Square, "Square(var) remains Square");
    expect_true(var_square->nonlinear_count == 1, "Square increases nonlinear count");
}

void test_periodic_ops() {
    expect_true(synth::sin8_value(0) == 128, "Sin8(0) should be centered");
    expect_true(synth::sin8_value(64) == 255, "Sin8(64) should be the positive peak");
    expect_true(synth::sin8_value(192) == 0, "Sin8(192) should be the negative peak");
    expect_true(synth::cos8_value(0) == 255, "Cos8(0) should be the positive peak");
    expect_true(synth::cos8_value(128) == 0, "Cos8(128) should be the negative peak");

    const auto sin_var = synth::make_sin8(synth::var_expr());
    expect_eq(sin_var->to_source(), "Sin8(x)", "Sin8(var) source");
    const auto cos_var = synth::make_cos8(synth::var_expr());
    expect_eq(cos_var->to_source(), "Cos8(x)", "Cos8(var) source");
    const auto folded = synth::make_sin8(synth::const_expr(64));
    expect_eq(folded->to_source(), "255", "Sin8 folds constants");
}

void test_nonlinear_limits() {
    const auto square = synth::make_square(synth::var_expr(), 1);
    bool threw = false;
    try {
        static_cast<void>(synth::make_mul(square, synth::var_expr(), false, 1));
    } catch (const std::exception&) {
        threw = true;
    }
    expect_true(threw, "Restricted nonlinear limit should reject Square(x) * x");

    const auto cubic = synth::make_mul(square, synth::var_expr(), true, 2);
    expect_eq(cubic->to_source(), "(Square(x) * x)", "Relaxed nonlinear mode should allow x^3");
}

struct GoldenCase {
    std::string_view name;
    bool expected_found;
    std::optional<int> expected_cost;
    std::string_view expected_expression;
    std::string_view nonlinear_mode;
};

const std::vector<GoldenCase>& golden_cases() {
    static const std::vector<GoldenCase> cases = {
        {"affine_linear",             true,  4,            "((3 * x) + 5)",                  "restricted"},
        {"affine_mulsub",             true,  4,            "((7 * x) + 243)",                "restricted"},
        {"xor_shift_add",             true,  6,            "((x >> 2) + (x ^ 17))",          "restricted"},
        {"nested_xor",                true,  3,            "(x ^ 20)",                       "restricted"},
        {"add_tail",                  true,  3,            "(x + 20)",                       "restricted"},
        {"shift_add_const",           true,  4,            "((8 * x) + 5)",                  "restricted"},
        {"shift_mix",                 true,  5,            "((8 * x) + (x >> 1))",           "restricted"},
        {"mul_xor_const",             true,  4,            "((3 * x) ^ 5)",                  "restricted"},
        {"mul_shift_add",             true,  5,            "((7 * x) + (x >> 1))",           "restricted"},
        {"sin8",                      true,  2,            "Sin8(x)",                        "restricted"},
        {"cos8",                      true,  2,            "Cos8(x)",                        "restricted"},
        {"sin8_plus_5",               true,  4,            "(Sin8(x) + 5)",                  "restricted"},
        {"cos8_xor_17",               true,  4,            "(Cos8(x) ^ 17)",                 "restricted"},
        {"sin8_shr_1",                true,  3,            "Sin8((x >> 1))",                 "restricted"},
        {"cos8_3x",                   true,  3,            "Cos8((3 * x))",                  "restricted"},
        {"sin8_x_plus_20",            true,  4,            "Sin8((x + 20))",                 "restricted"},
        {"sin8_plus_cos8",            true,  5,            "(Cos8(x) + Sin8(x))",            "restricted"},
        {"shift_chain",               true,  2,            "(32 * x)",                       "restricted"},
        {"mul_chain",                 true,  2,            "(15 * x)",                       "restricted"},
        {"square",                    true,  2,            "Square(x)",                      "restricted"},
        {"square_plus_5",             true,  4,            "(Square(x) + 5)",                "restricted"},
        {"square_xor_17",             true,  4,            "(Square(x) ^ 17)",               "restricted"},
        {"shifted_square_10",         true,  4,            "Square((10 - x))",               "restricted"},
        {"shifted_square_plus_const", true,  6,            "(Square((10 - x)) + 5)",         "restricted"},
        {"x_times_x_plus_1",          true,  4,            "(Square(x) + x)",                "restricted"},
        {"x_times_x_minus_7",         true,  5,            "((x + 249) * x)",                "restricted"},
        {"x_times_x_plus_1_plus_5",   true,  6,            "((Square(x) + x) + 5)",          "restricted"},
        {"x_times_x_shr_1",           true,  4,            "((x >> 1) * x)",                 "restricted"},
        {"x_plus_1_times_x_shr_1",    true,  6,            "((x + 1) * (x >> 1))",           "restricted"},
        {"x_cubed",                   false, std::nullopt, "",                               "restricted"},
        {"x_cubed_plus_5",            false, std::nullopt, "",                               "restricted"},
        {"square_plus_affine",        false, std::nullopt, "",                               "restricted"},
        {"mixed_not_found",           false, std::nullopt, "",                               "restricted"},
        {"awkward_not_found",         false, std::nullopt, "",                               "restricted"},
    };
    return cases;
}

void test_search_cases_golden() {
    const auto& cases = golden_cases();
    if (cases.size() != synth::benchmarks().size()) {
        throw TestFailure(
            "golden_cases().size() (" + std::to_string(cases.size()) +
            ") != benchmarks().size() (" + std::to_string(synth::benchmarks().size()) + ")"
        );
    }
    for (const auto& gold : cases) {
        const auto* benchmark = synth::find_benchmark(std::string(gold.name));
        if (benchmark == nullptr) {
            throw TestFailure("missing benchmark: " + std::string(gold.name));
        }
        const auto target = synth::build_target(benchmark->oracle);
        const bool enable_periodic = benchmark->category == "periodic";
        const auto attempt = synth::run_search_attempt(
            target,
            synth::MAX_COST,
            false,
            synth::top1_sig_bucket_limit,
            gold.nonlinear_mode,
            enable_periodic
        );

        const bool found = attempt.result.has_value();
        if (found != gold.expected_found) {
            throw TestFailure(
                std::string(gold.name) + ": found=" + (found ? "true" : "false")
                + " but expected " + (gold.expected_found ? "true" : "false")
            );
        }

        if (!gold.expected_found) {
            continue;
        }

        const auto& result = attempt.result.value();
        if (gold.expected_cost.has_value() && result.expr->cost != gold.expected_cost.value()) {
            throw TestFailure(
                std::string(gold.name) + ": cost=" + std::to_string(result.expr->cost)
                + " but expected " + std::to_string(gold.expected_cost.value())
            );
        }

        const auto actual = result.expr->to_source();
        if (actual != gold.expected_expression) {
            throw TestFailure(
                std::string(gold.name) + ": expression='" + actual
                + "' but expected '" + std::string(gold.expected_expression) + "'"
            );
        }

        const auto [verified, failing_x] = synth::verify_expression(result.expr, target);
        if (!verified) {
            throw TestFailure(
                std::string(gold.name) + ": verification failed at x="
                + (failing_x.has_value() ? std::to_string(failing_x.value()) : "-")
            );
        }
    }
}

void test_codegen() {
    const auto expr = synth::make_square(synth::var_expr(), 1);
    const auto verilog = synth::emit_verilog(expr, "generated_function");
    expect_true(verilog.find("module generated_function") != std::string::npos, "Verilog should contain module name");
    expect_true(verilog.find("assign y =") != std::string::npos, "Verilog should contain assign y");

    const auto periodic = synth::make_sin8(synth::var_expr());
    const auto periodic_verilog = synth::emit_verilog(periodic, "periodic_function");
    expect_true(periodic_verilog.find("function [7:0] sin8") != std::string::npos, "Periodic Verilog should contain sin8 LUT");
    expect_true(periodic_verilog.find("sin8(") != std::string::npos, "Periodic Verilog should call sin8 LUT");
}

}  // namespace

int main() {
    try {
        test_add_canonicalization();
        test_sub_canonicalization();
        test_xor_canonicalization();
        test_shift_mul_merge();
        test_mul_square_rule();
        test_square_folding();
        test_periodic_ops();
        test_nonlinear_limits();
        test_search_cases_golden();
        test_codegen();
        std::cout << "synth_tests: PASS (" << golden_cases().size() << " golden cases)\n";
        return 0;
    } catch (const TestFailure& error) {
        std::cerr << "synth_tests: FAIL: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "synth_tests: ERROR: " << error.what() << '\n';
        return 1;
    }
}
