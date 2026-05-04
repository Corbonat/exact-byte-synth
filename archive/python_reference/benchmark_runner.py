"""Sequential benchmark runner for the 8-bit exact synthesizer."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from benchmarks import BENCHMARK_BY_NAME, BENCHMARKS, BenchmarkCase
from oracle import DOMAIN_SIZE
from search import (
    BucketSummary,
    LevelStats,
    MAX_COST,
    DEFAULT_NONLINEAR_MODE,
    NONLINEAR_CONFIGS,
    SearchAttempt,
    merge_bucket_summaries,
    merge_level_stats,
    run_search_attempt,
    selective_sig_bucket_limit,
    synthesize_cegis,
    top1_sig_bucket_limit,
)
from verify import verify_expression


@dataclass(frozen=True)
class BenchmarkResult:
    """Collected metrics for one benchmark case."""

    name: str
    category: str
    description: str
    expected_found: bool
    found: bool
    verified: bool
    backend: str
    cost: int | None
    elapsed_s: float
    full_eval: int
    sig_pruned: int
    shape_pruned: int
    fallback_triggered: bool
    attempted_backends: tuple[str, ...]
    backend_timings_s: dict[str, float]
    solved_by_first_backend: bool
    solved_by_backend_index: int | None
    cegis_rounds: int
    cegis_sample_size: int | None
    counterexamples: tuple[int, ...]
    const_promotion_rounds: int
    promoted_consts_total: int
    max_bucket_count: int
    avg_bucket_count_on_heavy_levels: float
    expression: str | None
    nonlinear_mode: str
    generated_mul_candidates: int
    accepted_mul_candidates: int
    shape_pruned_mul: int
    sem_pruned_mul: int
    max_nonlinear_depth_seen: int


def build_target(oracle_fn) -> bytes:
    """Evaluate one oracle on the full 8-bit domain."""
    return bytes(oracle_fn(x) for x in range(DOMAIN_SIZE))


def combine_attempt_stats(attempt: SearchAttempt) -> dict[int, LevelStats]:
    """Merge signature and fallback stats into one view of spent work."""
    combined: dict[int, LevelStats] = defaultdict(LevelStats)
    for stats_map in (attempt.signature_stats, attempt.fallback_stats):
        if stats_map is None:
            continue
        for cost, level in stats_map.items():
            merge_level_stats(combined[cost], level)
    return dict(combined)


def combine_attempt_bucket_summary(attempt: SearchAttempt) -> BucketSummary:
    """Merge signature and fallback bucket summaries."""
    summary = BucketSummary()
    for item in (attempt.signature_bucket_summary, attempt.fallback_bucket_summary):
        if item is None:
            continue
        summary = merge_bucket_summaries(summary, item)
    return summary


def aggregate_metric(stats_map: dict[int, LevelStats], field: str) -> int:
    """Sum one per-level metric across the whole search."""
    return sum(getattr(level, field) for level in stats_map.values())


def sig_strategy(name: str):
    """Return the signature bucket policy by CLI name."""
    if name == "top1":
        return top1_sig_bucket_limit
    if name == "selective":
        return selective_sig_bucket_limit
    raise ValueError(f"Unsupported signature strategy: {name}")


def run_case(
    case: BenchmarkCase,
    max_cost: int,
    sig_mode: str,
    log_progress: bool,
    mode: str = "full_domain",
    nonlinear_mode: str = DEFAULT_NONLINEAR_MODE,
) -> BenchmarkResult:
    """Run one benchmark case end-to-end."""
    target = build_target(case.oracle)
    strategy = sig_strategy(sig_mode)
    started = time.perf_counter()
    if mode in ("full_domain", "core_only", "full_system"):
        attempt = run_search_attempt(
            target=target,
            max_cost=max_cost,
            log_progress=log_progress,
            sig_bucket_limit_fn=strategy,
            nonlinear_mode=nonlinear_mode,
        )
    elif mode == "cegis":
        attempt = synthesize_cegis(
            oracle_fn=case.oracle,
            max_cost=max_cost,
            log_progress=log_progress,
            nonlinear_mode=nonlinear_mode,
        )
    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")
    elapsed = time.perf_counter() - started
    result = attempt.result

    verified = False
    if result is not None:
        verified, _ = verify_expression(result.expr, list(target))

    combined_stats = combine_attempt_stats(attempt)
    bucket_summary = combine_attempt_bucket_summary(attempt)
    promotion_rounds = attempt.signature_promotion_rounds + attempt.fallback_promotion_rounds
    promoted_consts_total = sum(len(round_info.derived_consts_promoted) for round_info in promotion_rounds)
    counterexamples = tuple(
        round_info.counterexample
        for round_info in attempt.cegis_rounds
        if round_info.counterexample is not None
    )
    sample_size = attempt.cegis_rounds[-1].sample_size if attempt.cegis_rounds else None

    return BenchmarkResult(
        name=case.name,
        category=case.category,
        description=case.description,
        expected_found=case.expected_found,
        found=result is not None,
        verified=verified,
        backend=result.backend if result is not None else "none",
        cost=result.expr.cost if result is not None else None,
        elapsed_s=elapsed,
        full_eval=aggregate_metric(combined_stats, "full_eval"),
        sig_pruned=aggregate_metric(combined_stats, "sig_pruned"),
        shape_pruned=aggregate_metric(combined_stats, "shape_pruned"),
        fallback_triggered=attempt.fallback_triggered,
        attempted_backends=attempt.attempted_backends,
        backend_timings_s={name: elapsed for name, elapsed in attempt.backend_timings_s},
        solved_by_first_backend=attempt.solved_by_first_backend if result is not None else False,
        solved_by_backend_index=attempt.solved_by_backend_index,
        cegis_rounds=len(attempt.cegis_rounds),
        cegis_sample_size=sample_size,
        counterexamples=counterexamples,
        const_promotion_rounds=len(promotion_rounds),
        promoted_consts_total=promoted_consts_total,
        max_bucket_count=bucket_summary.max_bucket_count,
        avg_bucket_count_on_heavy_levels=bucket_summary.avg_bucket_count_on_heavy_levels,
        expression=result.expr.to_source() if result is not None else None,
        nonlinear_mode=nonlinear_mode,
        generated_mul_candidates=aggregate_metric(combined_stats, "generated_mul_candidates"),
        accepted_mul_candidates=aggregate_metric(combined_stats, "accepted_mul_candidates"),
        shape_pruned_mul=aggregate_metric(combined_stats, "shape_pruned_mul"),
        sem_pruned_mul=aggregate_metric(combined_stats, "sem_pruned_mul"),
        max_nonlinear_depth_seen=max(
            (level.max_nonlinear_depth_seen for level in combined_stats.values()),
            default=0,
        ),
    )


def print_table(results: Iterable[BenchmarkResult]) -> None:
    """Print a compact fixed-width table."""
    rows = list(results)
    headers = (
        "name",
        "category",
        "found",
        "backend",
        "dispatch",
        "cost",
        "elapsed_s",
        "full_eval",
        "sig_pruned",
        "shape_pruned",
        "fallback",
        "cegis",
        "promote",
    )
    widths = {
        "name": 18,
        "category": 16,
        "found": 7,
        "backend": 24,
        "dispatch": 8,
        "cost": 6,
        "elapsed_s": 10,
        "full_eval": 10,
        "sig_pruned": 11,
        "shape_pruned": 13,
        "fallback": 9,
        "cegis": 7,
        "promote": 8,
    }

    print(" ".join(header.ljust(widths[header]) for header in headers))
    print(" ".join("-" * widths[header] for header in headers))
    for row in rows:
        print(
            " ".join(
                (
                    row.name.ljust(widths["name"]),
                    row.category.ljust(widths["category"]),
                    str(row.found).ljust(widths["found"]),
                    row.backend.ljust(widths["backend"]),
                    str(row.solved_by_backend_index if row.solved_by_backend_index is not None else "-").ljust(widths["dispatch"]),
                    str(row.cost if row.cost is not None else "-").ljust(widths["cost"]),
                    f"{row.elapsed_s:.3f}".ljust(widths["elapsed_s"]),
                    str(row.full_eval).ljust(widths["full_eval"]),
                    str(row.sig_pruned).ljust(widths["sig_pruned"]),
                    str(row.shape_pruned).ljust(widths["shape_pruned"]),
                    str(row.fallback_triggered).ljust(widths["fallback"]),
                    str(row.cegis_rounds if row.cegis_rounds else "-").ljust(widths["cegis"]),
                    str(row.promoted_consts_total).ljust(widths["promote"]),
                )
            )
        )


def print_summary(results: Iterable[BenchmarkResult], sig_mode: str, max_cost: int, mode: str) -> None:
    """Print aggregate suite-level metrics."""
    rows = list(results)
    found = sum(1 for row in rows if row.found)
    verified = sum(1 for row in rows if row.verified)
    total_elapsed = sum(row.elapsed_s for row in rows)
    total_full_eval = sum(row.full_eval for row in rows)
    total_sig_pruned = sum(row.sig_pruned for row in rows)
    fallback_count = sum(1 for row in rows if row.fallback_triggered)
    promotion_hits = sum(1 for row in rows if row.promoted_consts_total > 0)
    solved_by_first = sum(1 for row in rows if row.solved_by_first_backend)
    cegis_rounds = sum(row.cegis_rounds for row in rows)
    generated_mul = sum(row.generated_mul_candidates for row in rows)
    accepted_mul = sum(row.accepted_mul_candidates for row in rows)
    shape_pruned_mul = sum(row.shape_pruned_mul for row in rows)
    sem_pruned_mul = sum(row.sem_pruned_mul for row in rows)
    max_nonlinear = max((row.max_nonlinear_depth_seen for row in rows), default=0)
    nonlinear_modes = ",".join(sorted({row.nonlinear_mode for row in rows}))

    print()
    print(
        f"suite_cases={len(rows)} mode={mode} nonlinear_mode={nonlinear_modes} "
        f"sig_mode={sig_mode} max_cost={max_cost}"
    )
    print(f"found={found} verified={verified} total_elapsed_s={total_elapsed:.3f}")
    print(f"total_full_eval={total_full_eval} total_sig_pruned={total_sig_pruned}")
    print(
        f"fallback_triggered={fallback_count} promotion_hits={promotion_hits} "
        f"solved_by_first_backend={solved_by_first} cegis_rounds={cegis_rounds}"
    )
    print(
        f"generated_mul_candidates={generated_mul} accepted_mul_candidates={accepted_mul} "
        f"shape_pruned_mul={shape_pruned_mul} sem_pruned_mul={sem_pruned_mul} "
        f"max_nonlinear_depth_seen={max_nonlinear}"
    )


def write_json(results: Iterable[BenchmarkResult], output_path: str) -> None:
    """Write benchmark results as JSON."""
    payload = [asdict(result) for result in results]
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="*", help="Run only selected benchmark names.")
    parser.add_argument("--max-cost", type=int, default=MAX_COST, help="Maximum synthesis cost.")
    parser.add_argument(
        "--sig-mode",
        choices=("top1", "selective"),
        default="top1",
        help="Signature bucket strategy.",
    )
    parser.add_argument(
        "--mode",
        choices=("full_domain", "cegis", "core_only", "full_system"),
        default="full_domain",
        help="Synthesis mode used by each benchmark.",
    )
    parser.add_argument(
        "--nonlinear-mode",
        choices=tuple(sorted(NONLINEAR_CONFIGS)),
        default=DEFAULT_NONLINEAR_MODE,
        help="Nonlinear grammar limits.",
    )
    parser.add_argument(
        "--log-progress",
        action="store_true",
        help="Show per-cost search logs during each case.",
    )
    parser.add_argument("--json-out", help="Optional JSON output path.")
    parser.add_argument(
        "--profile-case",
        help="Profile one benchmark case with cProfile instead of running the whole suite.",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=20,
        help="How many profiling rows to print.",
    )
    return parser.parse_args()


def select_cases(names: list[str] | None) -> tuple[BenchmarkCase, ...]:
    """Resolve the user-selected benchmark subset."""
    if not names:
        return BENCHMARKS
    missing = [name for name in names if name not in BENCHMARK_BY_NAME]
    if missing:
        raise SystemExit(f"Unknown benchmark names: {', '.join(missing)}")
    return tuple(BENCHMARK_BY_NAME[name] for name in names)


def run_profile(
    case: BenchmarkCase,
    max_cost: int,
    sig_mode: str,
    log_progress: bool,
    profile_top: int,
    mode: str,
    nonlinear_mode: str,
) -> None:
    """Profile one benchmark case with cProfile."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = run_case(
        case,
        max_cost=max_cost,
        sig_mode=sig_mode,
        log_progress=log_progress,
        mode=mode,
        nonlinear_mode=nonlinear_mode,
    )
    profiler.disable()

    print_table([result])
    print_summary([result], sig_mode=sig_mode, max_cost=max_cost, mode=mode)
    print()
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(profile_top)


def main() -> int:
    args = parse_args()
    cases = select_cases(args.names)

    if args.profile_case:
        if args.profile_case not in BENCHMARK_BY_NAME:
            raise SystemExit(f"Unknown benchmark for profiling: {args.profile_case}")
        run_profile(
            BENCHMARK_BY_NAME[args.profile_case],
            max_cost=args.max_cost,
            sig_mode=args.sig_mode,
            log_progress=args.log_progress,
            profile_top=args.profile_top,
            mode=args.mode,
            nonlinear_mode=args.nonlinear_mode,
        )
        return 0

    results: list[BenchmarkResult] = []
    for case in cases:
        started = time.perf_counter()
        result = run_case(
            case,
            max_cost=args.max_cost,
            sig_mode=args.sig_mode,
            log_progress=args.log_progress,
            mode=args.mode,
            nonlinear_mode=args.nonlinear_mode,
        )
        results.append(result)
        case_elapsed = time.perf_counter() - started
        print(
            f"[{len(results)}/{len(cases)}] {case.name}: "
            f"found={result.found} cost={result.cost if result.cost is not None else '-'} "
            f"elapsed_s={case_elapsed:.3f}"
        )

    print()
    print_table(results)
    print_summary(results, sig_mode=args.sig_mode, max_cost=args.max_cost, mode=args.mode)

    if args.json_out:
        write_json(results, args.json_out)
        print(f"wrote_json={args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
