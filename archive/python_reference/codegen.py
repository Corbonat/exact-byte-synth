"""Generate Verilog from a recovered expression."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks import BENCHMARK_BY_NAME
from dsl import Expr
from oracle import DOMAIN_SIZE
from search import DEFAULT_NONLINEAR_MODE, NONLINEAR_CONFIGS, collect_target, synthesize
from verify import verify_expression


def emit_verilog(expr: Expr, module_name: str = "generated_function") -> str:
    """Return a simple combinational Verilog module for the expression."""
    return f"""module {module_name} (
    input  wire [7:0] x,
    output wire [7:0] y
);
    assign y = {expr.to_verilog8()};
endmodule
"""


def write_verilog(expr: Expr, output_path: str = "generated_function.v") -> Path:
    output = Path(output_path)
    output.write_text(emit_verilog(expr), encoding="ascii")
    return output


def build_target(oracle_fn) -> bytes:
    """Evaluate an oracle on the full 8-bit domain."""
    return bytes(oracle_fn(x) for x in range(DOMAIN_SIZE))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        help="Optional benchmark name from benchmarks.py. Defaults to the current oracle.",
    )
    parser.add_argument(
        "--output",
        default="generated_function.v",
        help="Output Verilog path.",
    )
    parser.add_argument(
        "--module-name",
        default="generated_function",
        help="Generated Verilog module name.",
    )
    parser.add_argument("--max-cost", type=int, default=None, help="Override maximum synthesis cost.")
    parser.add_argument(
        "--nonlinear-mode",
        choices=tuple(sorted(NONLINEAR_CONFIGS)),
        default=DEFAULT_NONLINEAR_MODE,
        help="Nonlinear grammar limits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.benchmark:
        if args.benchmark not in BENCHMARK_BY_NAME:
            print(f"Unknown benchmark: {args.benchmark}")
            return 1
        benchmark = BENCHMARK_BY_NAME[args.benchmark]
        target = build_target(benchmark.oracle)
    else:
        benchmark = None
        target = collect_target()

    synthesize_kwargs = {"nonlinear_mode": args.nonlinear_mode}
    if args.max_cost is not None:
        synthesize_kwargs["max_cost"] = args.max_cost
    result = synthesize(target, **synthesize_kwargs)
    if result is None:
        print("Code generation aborted: no matching expression was found.")
        return 1

    ok, failing_x = verify_expression(result.expr, list(target))
    if not ok:
        print(f"Code generation aborted: verification failed at x={failing_x}.")
        return 1

    output = Path(args.output)
    output.write_text(emit_verilog(result.expr, module_name=args.module_name), encoding="ascii")
    target_name = benchmark.name if benchmark is not None else "oracle"
    print(f"Wrote {output} for {target_name} using {result.backend} backend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
