"""Baseline benchmark harness.

Runs the full synthesis suite through either the Python reference
implementation or one of the compiled C++ binaries, repeating each
case several times and picking the median elapsed time. Produces a
single JSON document that is directly consumable by ``compare.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
_PY_REFERENCE_PATHS = (
    REPO_ROOT,
    REPO_ROOT / "archive" / "python_reference",
)
for _path in _PY_REFERENCE_PATHS:
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


SCHEMA_VERSION = 1
DEFAULT_REPEATS = 3
DEFAULT_MAX_COST = 6
DEFAULT_NONLINEAR_MODE = "restricted"
DEFAULT_SIG_MODE = "top1"
DEFAULT_MODE = "full_domain"
DEFAULT_CPP_BINARY = REPO_ROOT / "cpp_port" / "build" / "synth_benchmark.exe"


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _select_names(names: list[str] | None) -> list[str]:
    from benchmarks import BENCHMARKS, BENCHMARK_BY_NAME

    all_names = [case.name for case in BENCHMARKS]
    if not names:
        return all_names
    missing = [name for name in names if name not in BENCHMARK_BY_NAME]
    if missing:
        raise SystemExit(f"Unknown benchmark names: {', '.join(missing)}")
    return list(names)


def _python_run_case(name: str, args: argparse.Namespace) -> dict[str, Any]:
    from benchmark_runner import run_case
    from benchmarks import BENCHMARK_BY_NAME

    case = BENCHMARK_BY_NAME[name]
    result = run_case(
        case,
        max_cost=args.max_cost,
        sig_mode=args.sig_mode,
        log_progress=False,
        mode=args.mode,
        nonlinear_mode=args.nonlinear_mode,
    )
    return asdict(result)


def _cpp_run_case(name: str, args: argparse.Namespace) -> dict[str, Any]:
    binary = Path(args.cpp_binary).resolve()
    if not binary.exists():
        raise SystemExit(f"C++ binary not found: {binary}. Build with cpp_port/build.ps1.")

    with tempfile.TemporaryDirectory() as tmp:
        json_out = Path(tmp) / "case.json"
        cmd = [
            str(binary),
            "--names", name,
            "--max-cost", str(args.max_cost),
            "--sig-mode", args.sig_mode,
            "--mode", args.mode,
            "--nonlinear-mode", args.nonlinear_mode,
            "--json-out", str(json_out),
        ]
        env = os.environ.copy()
        env["PATH"] = r"C:\msys64\ucrt64\bin;" + env.get("PATH", "")
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(REPO_ROOT),
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"C++ binary failed for {name}: exit={proc.returncode}\n"
                f"stderr={proc.stderr.decode('utf-8', errors='replace')}"
            )
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        if len(payload) != 1:
            raise SystemExit(f"Expected 1 record for {name}, got {len(payload)}")
        return payload[0]


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Drop engine-specific shape of numeric fields to ease comparison."""
    normalized = dict(record)
    for key in ("attempted_backends", "counterexamples"):
        if key in normalized and isinstance(normalized[key], (list, tuple)):
            normalized[key] = list(normalized[key])
    return normalized


def _merge_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine N run records for the same case into one summary."""
    elapsed = [float(run.get("elapsed_s", 0.0)) for run in runs]
    base = _normalize_record(runs[0])
    base["elapsed_s_median"] = _median(elapsed)
    base["elapsed_s_min"] = min(elapsed)
    base["elapsed_s_max"] = max(elapsed)
    base["elapsed_s_samples"] = elapsed
    base.pop("elapsed_s", None)
    base.pop("backend_timings_s", None)
    return base


def _run_engine(names: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    runner = _python_run_case if args.engine == "python" else _cpp_run_case

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names, 1):
        runs: list[dict[str, Any]] = []
        started = time.perf_counter()
        for repeat in range(args.repeats):
            runs.append(runner(name, args))
        elapsed = time.perf_counter() - started
        merged = _merge_runs(runs)
        rows.append(merged)
        print(
            f"[{index}/{len(names)}] {name}: "
            f"median={merged['elapsed_s_median']:.3f}s "
            f"found={merged.get('found')} cost={merged.get('cost')} "
            f"wall={elapsed:.2f}s",
            flush=True,
        )
    return rows


def _collect_build_info(engine: str, args: argparse.Namespace) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    if engine == "cpp":
        binary = Path(args.cpp_binary).resolve()
        info["cpp_binary"] = str(binary)
        info["cpp_binary_size"] = binary.stat().st_size if binary.exists() else None
        info["cpp_binary_mtime"] = (
            int(binary.stat().st_mtime) if binary.exists() else None
        )
        try:
            out = subprocess.run(
                [str(binary), "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(REPO_ROOT),
                check=False,
                timeout=5,
            )
            info["cpp_binary_banner"] = out.stderr.decode("utf-8", errors="replace")[:200]
        except Exception:
            info["cpp_binary_banner"] = None
    return info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=("python", "cpp"),
        required=True,
        help="Which implementation to benchmark.",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--names", nargs="*", help="Subset of benchmark names to run.")
    parser.add_argument("--max-cost", type=int, default=DEFAULT_MAX_COST)
    parser.add_argument("--sig-mode", default=DEFAULT_SIG_MODE, choices=("top1", "selective"))
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=("full_domain", "cegis"))
    parser.add_argument("--nonlinear-mode", default=DEFAULT_NONLINEAR_MODE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--cpp-binary",
        default=str(DEFAULT_CPP_BINARY),
        help="Path to the C++ benchmark binary (cpp engine only).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label stored in the baseline header (e.g. 'cpp_naive').",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = _select_names(args.names)
    started = time.perf_counter()
    rows = _run_engine(names, args)
    elapsed = time.perf_counter() - started

    payload = {
        "schema_version": SCHEMA_VERSION,
        "engine": args.engine,
        "label": args.label,
        "mode": args.mode,
        "nonlinear_mode": args.nonlinear_mode,
        "sig_mode": args.sig_mode,
        "max_cost": args.max_cost,
        "repeats": args.repeats,
        "cases": len(rows),
        "wall_elapsed_s": elapsed,
        "build_info": _collect_build_info(args.engine, args),
        "results": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    total_median = sum(row["elapsed_s_median"] for row in rows)
    found = sum(1 for row in rows if row.get("found"))
    verified = sum(1 for row in rows if row.get("verified"))
    print()
    print(
        f"engine={args.engine} cases={len(rows)} found={found} verified={verified} "
        f"median_sum={total_median:.3f}s wall={elapsed:.2f}s"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
