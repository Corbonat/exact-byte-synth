"""Full-domain verification for recovered expressions."""

from __future__ import annotations

import argparse

from dsl import Expr
from oracle import DOMAIN_SIZE
from search import DEFAULT_NONLINEAR_MODE, NONLINEAR_CONFIGS, collect_target, synthesize


def verify_expression(expr: Expr, target: list[int]) -> tuple[bool, int | None]:
    """Compare the candidate against the target on every 8-bit input."""
    for x in range(DOMAIN_SIZE):
        actual = expr.eval(x)
        expected = target[x]
        if actual != expected:
            return False, x
    return True, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    target = collect_target()
    synthesize_kwargs = {"nonlinear_mode": args.nonlinear_mode}
    if args.max_cost is not None:
        synthesize_kwargs["max_cost"] = args.max_cost
    result = synthesize(target, **synthesize_kwargs)
    if result is None:
        print("Verification aborted: no matching expression was found.")
        return 1

    ok, failing_x = verify_expression(result.expr, list(target))
    if not ok:
        print(f"Verification failed at x={failing_x}.")
        return 1

    print(
        f"Verified expression on all {DOMAIN_SIZE} inputs: "
        f"{result.expr.to_source()} mod 256 ({result.backend}, cost={result.expr.cost})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
