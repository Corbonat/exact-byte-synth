"""Build a demo-ready FPGA artifact from a benchmark or the current oracle."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmarks import BENCHMARK_BY_NAME
from codegen import emit_verilog
from oracle import DOMAIN_SIZE
from search import DEFAULT_NONLINEAR_MODE, NONLINEAR_CONFIGS, collect_target, run_search_attempt
from verify import verify_expression


FPGA_DEMO_DIR = Path("fpga_demo")
ACTIVE_VERILOG = FPGA_DEMO_DIR / "generated_function.v"
REPORTS_DIR = FPGA_DEMO_DIR / "reports"
ARCHIVE_DIR = FPGA_DEMO_DIR / "generated_cases"
DEFAULT_DEMO_CASES = (
    "affine_linear",
    "nested_xor",
    "xor_shift_add",
    "mul_xor_const",
    "mul_shift_add",
    "square",
)


@dataclass(frozen=True)
class DemoBuildReport:
    """One demo build result written to disk for later FPGA packaging."""

    benchmark: str
    description: str
    expected_found: bool | None
    found: bool
    verified: bool
    backend: str | None
    cost: int | None
    expression: str | None
    fallback_triggered: bool
    attempted_backends: tuple[str, ...]
    backend_timings_s: dict[str, float]
    solved_by_backend_index: int | None
    active_verilog: str | None
    archived_verilog: str | None
    report_path: str
    built_at_utc: str


def build_target(oracle_fn) -> bytes:
    """Evaluate one oracle on the full 8-bit domain."""
    return bytes(oracle_fn(x) for x in range(DOMAIN_SIZE))


def ensure_demo_dirs() -> None:
    """Create output directories used by the demo flow."""
    FPGA_DEMO_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        help="Benchmark case to synthesize. Defaults to the current oracle if omitted.",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="Print the recommended demo cases and exit.",
    )
    parser.add_argument("--max-cost", type=int, default=None, help="Override maximum synthesis cost.")
    parser.add_argument(
        "--nonlinear-mode",
        choices=tuple(sorted(NONLINEAR_CONFIGS)),
        default=DEFAULT_NONLINEAR_MODE,
        help="Nonlinear grammar limits.",
    )
    return parser.parse_args()


def resolve_target(benchmark_name: str | None) -> tuple[str, str, bool | None, bytes]:
    """Resolve either one benchmark or the current oracle into a full-domain target."""
    if benchmark_name is None:
        return "oracle", "Current oracle from oracle.py", None, collect_target()

    benchmark = BENCHMARK_BY_NAME.get(benchmark_name)
    if benchmark is None:
        known = ", ".join(sorted(BENCHMARK_BY_NAME))
        raise SystemExit(f"Unknown benchmark: {benchmark_name}. Known benchmarks: {known}")
    return benchmark.name, benchmark.description, benchmark.expected_found, build_target(benchmark.oracle)


def write_demo_outputs(name: str, verilog_text: str) -> tuple[Path, Path]:
    """Write the active FPGA file and a per-benchmark archived copy."""
    ACTIVE_VERILOG.write_text(verilog_text, encoding="ascii")
    archived = ARCHIVE_DIR / f"{name}.v"
    archived.write_text(verilog_text, encoding="ascii")
    return ACTIVE_VERILOG, archived


def write_report(report: DemoBuildReport) -> Path:
    """Write one JSON report for the generated demo artifact."""
    path = REPORTS_DIR / f"{report.benchmark}.json"
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


def print_benchmark_list() -> None:
    """Print the recommended demo cases."""
    print("recommended_demo_cases:")
    for name in DEFAULT_DEMO_CASES:
        benchmark = BENCHMARK_BY_NAME[name]
        print(f"  {benchmark.name}: {benchmark.description}")


def main() -> int:
    args = parse_args()
    if args.list_benchmarks:
        print_benchmark_list()
        return 0

    ensure_demo_dirs()
    name, description, expected_found, target = resolve_target(args.benchmark)
    search_kwargs = {
        "target": target,
        "log_progress": False,
        "nonlinear_mode": args.nonlinear_mode,
    }
    if args.max_cost is not None:
        search_kwargs["max_cost"] = args.max_cost
    attempt = run_search_attempt(**search_kwargs)
    result = attempt.result

    if result is None:
        report = DemoBuildReport(
            benchmark=name,
            description=description,
            expected_found=expected_found,
            found=False,
            verified=False,
            backend=None,
            cost=None,
            expression=None,
            fallback_triggered=attempt.fallback_triggered,
            attempted_backends=attempt.attempted_backends,
            backend_timings_s={backend: elapsed for backend, elapsed in attempt.backend_timings_s},
            solved_by_backend_index=None,
            active_verilog=None,
            archived_verilog=None,
            report_path=str((REPORTS_DIR / f"{name}.json").resolve()),
            built_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        path = write_report(report)
        print(f"build_status=FAILED benchmark={name} reason=no_exact_match report={path}")
        return 1

    verified, failing_x = verify_expression(result.expr, list(target))
    if not verified:
        report = DemoBuildReport(
            benchmark=name,
            description=description,
            expected_found=expected_found,
            found=True,
            verified=False,
            backend=result.backend,
            cost=result.expr.cost,
            expression=result.expr.to_source(),
            fallback_triggered=attempt.fallback_triggered,
            attempted_backends=attempt.attempted_backends,
            backend_timings_s={backend: elapsed for backend, elapsed in attempt.backend_timings_s},
            solved_by_backend_index=attempt.solved_by_backend_index,
            active_verilog=None,
            archived_verilog=None,
            report_path=str((REPORTS_DIR / f"{name}.json").resolve()),
            built_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        path = write_report(report)
        print(f"build_status=FAILED benchmark={name} reason=verify_failed x={failing_x} report={path}")
        return 1

    verilog = emit_verilog(result.expr, module_name="generated_function")
    active_path, archived_path = write_demo_outputs(name, verilog)
    report = DemoBuildReport(
        benchmark=name,
        description=description,
        expected_found=expected_found,
        found=True,
        verified=True,
        backend=result.backend,
        cost=result.expr.cost,
        expression=result.expr.to_source(),
        fallback_triggered=attempt.fallback_triggered,
        attempted_backends=attempt.attempted_backends,
        backend_timings_s={backend: elapsed for backend, elapsed in attempt.backend_timings_s},
        solved_by_backend_index=attempt.solved_by_backend_index,
        active_verilog=str(active_path.resolve()),
        archived_verilog=str(archived_path.resolve()),
        report_path=str((REPORTS_DIR / f"{name}.json").resolve()),
        built_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    report_path = write_report(report)

    print(f"benchmark={name}")
    print(f"description={description}")
    print(f"backend={result.backend}")
    print(f"cost={result.expr.cost}")
    print(f"expression={result.expr.to_source()}")
    print(f"verify=PASS on {DOMAIN_SIZE}/{DOMAIN_SIZE}")
    print(f"active_verilog={active_path}")
    print(f"archived_verilog={archived_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
