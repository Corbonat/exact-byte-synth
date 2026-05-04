"""Compare two baseline runs produced by ``run_baseline.py``.

Writes a compact parity table to stdout (optionally JSON) and exits
with a non-zero status if the two runs disagree on the recovered
expression, recovery success, or cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" not in payload:
        raise SystemExit(f"{path} is not a baseline JSON (missing 'results').")
    return payload


def _index(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in results}


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "     - "
    return f"{seconds:7.3f}"


def _fmt_speedup(baseline_s: float | None, current_s: float | None) -> str:
    if baseline_s is None or current_s is None:
        return "     -"
    if current_s <= 0:
        return "   inf"
    return f"{baseline_s / current_s:6.2f}x"


def _fmt_status(ok: bool) -> str:
    return "OK" if ok else "MISMATCH"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Reference baseline JSON (e.g. python.json).")
    parser.add_argument("current", help="Current run JSON (e.g. cpp_naive.json).")
    parser.add_argument("--json-out", help="Optional JSON summary output path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on any expression/cost/found mismatch.",
    )
    args = parser.parse_args(argv)

    baseline = _load(Path(args.baseline))
    current = _load(Path(args.current))

    base_by_name = _index(baseline["results"])
    curr_by_name = _index(current["results"])

    names = sorted(set(base_by_name) | set(curr_by_name))

    header = (
        f"{'name':22} {'found':12} {'cost':10} {'expr':10} "
        f"{'base_s':>8} {'curr_s':>8} {'speedup':>8}"
    )
    sep = "-" * len(header)
    print(f"baseline: {args.baseline} ({baseline.get('engine')}, {baseline.get('label') or '-'})")
    print(f"current:  {args.current} ({current.get('engine')}, {current.get('label') or '-'})")
    print(sep)
    print(header)
    print(sep)

    rows: list[dict[str, Any]] = []
    total_base = 0.0
    total_curr = 0.0
    mismatches = 0

    for name in names:
        base = base_by_name.get(name, {})
        curr = curr_by_name.get(name, {})
        found_match = bool(base.get("found")) == bool(curr.get("found"))
        cost_match = base.get("cost") == curr.get("cost")
        expr_match = base.get("expression") == curr.get("expression")

        base_s = base.get("elapsed_s_median")
        curr_s = curr.get("elapsed_s_median")
        if isinstance(base_s, (int, float)):
            total_base += float(base_s)
        if isinstance(curr_s, (int, float)):
            total_curr += float(curr_s)

        row = {
            "name": name,
            "found_match": found_match,
            "cost_match": cost_match,
            "expr_match": expr_match,
            "baseline_found": base.get("found"),
            "current_found": curr.get("found"),
            "baseline_cost": base.get("cost"),
            "current_cost": curr.get("cost"),
            "baseline_expression": base.get("expression"),
            "current_expression": curr.get("expression"),
            "baseline_elapsed_s_median": base_s,
            "current_elapsed_s_median": curr_s,
            "speedup": (
                (base_s / curr_s) if isinstance(base_s, (int, float))
                and isinstance(curr_s, (int, float))
                and curr_s
                else None
            ),
        }
        rows.append(row)

        if not (found_match and cost_match and expr_match):
            mismatches += 1

        print(
            f"{name:22} "
            f"{_fmt_status(found_match):12} "
            f"{_fmt_status(cost_match):10} "
            f"{_fmt_status(expr_match):10} "
            f"{_fmt_time(base_s):>8} {_fmt_time(curr_s):>8} "
            f"{_fmt_speedup(base_s, curr_s):>8}"
        )

    print(sep)
    print(
        f"totals: base={total_base:.3f}s curr={total_curr:.3f}s "
        f"speedup={(total_base / total_curr) if total_curr else float('inf'):.2f}x "
        f"mismatches={mismatches}/{len(names)}"
    )

    summary = {
        "baseline_path": str(args.baseline),
        "current_path": str(args.current),
        "baseline_header": {
            key: baseline.get(key)
            for key in ("engine", "label", "mode", "nonlinear_mode", "sig_mode", "max_cost")
        },
        "current_header": {
            key: current.get(key)
            for key in ("engine", "label", "mode", "nonlinear_mode", "sig_mode", "max_cost")
        },
        "cases": len(names),
        "mismatches": mismatches,
        "total_baseline_s": total_base,
        "total_current_s": total_curr,
        "overall_speedup": (total_base / total_curr) if total_curr else None,
        "rows": rows,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote_json={out}")

    if args.strict and mismatches:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
