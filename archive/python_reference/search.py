"""Search backends for exact recovery from a black-box oracle."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from dsl import (
    Expr,
    const_expr,
    make_add,
    make_mul,
    make_mul_const,
    make_shl_const,
    make_shr_const,
    make_square,
    make_sub,
    make_xor,
    var_expr,
)
from oracle import DOMAIN_SIZE, MASK, oracle


BASE_CONST_SET = [0, 1, 2, 3, 5, 7, 8, 10, 13, 15, 16, 17, 31, 32, 63, 64, 127, 128, 255]
SHIFT_SET = [1, 2, 3, 4, 5, 6, 7]
SIG_POINTS = [
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    15,
    16,
    31,
    32,
    63,
    64,
    127,
    128,
    129,
    170,
    192,
    223,
    240,
    247,
    251,
    255,
]
MAX_COST = 6
SIGNATURE_PRUNE_COST = 5
MAX_PROMOTION_ROUNDS = 8
DEFAULT_NONLINEAR_MODE = "restricted"
INITIAL_CEGIS_POINTS = (
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    15,
    16,
    31,
    32,
    63,
    64,
    127,
    128,
    255,
)


@dataclass(frozen=True)
class NonlinearConfig:
    """Grammar limits for nonlinear candidate generation."""

    mode: str
    max_nonlinear_count: int
    allow_mul_of_nonlinear: bool


NONLINEAR_CONFIGS = {
    "restricted": NonlinearConfig(
        mode="restricted",
        max_nonlinear_count=1,
        allow_mul_of_nonlinear=False,
    ),
    "relaxed": NonlinearConfig(
        mode="relaxed",
        max_nonlinear_count=2,
        allow_mul_of_nonlinear=True,
    ),
    "open": NonlinearConfig(
        mode="open",
        max_nonlinear_count=3,
        allow_mul_of_nonlinear=True,
    ),
}


def nonlinear_config(mode: str = DEFAULT_NONLINEAR_MODE) -> NonlinearConfig:
    """Return the configured nonlinear grammar limits."""
    try:
        return NONLINEAR_CONFIGS[mode]
    except KeyError as exc:
        known = ", ".join(sorted(NONLINEAR_CONFIGS))
        raise ValueError(f"Unsupported nonlinear mode: {mode}. Known modes: {known}") from exc

VAR_SIG = bytes(SIG_POINTS)
VAR_SEM = bytes(range(DOMAIN_SIZE))
CONST_SIG_CACHE = tuple(bytes([value]) * len(SIG_POINTS) for value in range(DOMAIN_SIZE))
CONST_SEM_CACHE = tuple(bytes([value]) * DOMAIN_SIZE for value in range(DOMAIN_SIZE))
MUL_TABLES = tuple(
    bytes((value * const) & MASK for value in range(DOMAIN_SIZE))
    for const in range(DOMAIN_SIZE)
)
SHL_TABLES = {
    shift: bytes((value << shift) & MASK for value in range(DOMAIN_SIZE))
    for shift in SHIFT_SET
}
SHR_TABLES = {
    shift: bytes((value >> shift) & MASK for value in range(DOMAIN_SIZE))
    for shift in SHIFT_SET
}


@dataclass(frozen=True)
class Candidate:
    """Accepted expression with cached signature and full semantics."""

    expr: Expr
    sig: bytes
    sem: bytes


@dataclass
class LevelStats:
    """Per-cost search counters."""

    generated: int = 0
    rule_pruned: int = 0
    shape_pruned: int = 0
    sig_pruned: int = 0
    sig_bucket_kept: int = 0
    sig_bucket_replaced: int = 0
    full_eval: int = 0
    sem_pruned: int = 0
    accepted: int = 0
    generated_mul_candidates: int = 0
    accepted_mul_candidates: int = 0
    shape_pruned_mul: int = 0
    sem_pruned_mul: int = 0
    max_nonlinear_depth_seen: int = 0


@dataclass
class BucketSummary:
    """Aggregate statistics for signature buckets."""

    max_bucket_count: int = 0
    heavy_bucket_observations: int = 0
    heavy_bucket_size_total: int = 0

    @property
    def avg_bucket_count_on_heavy_levels(self) -> float:
        if self.heavy_bucket_observations == 0:
            return 0.0
        return self.heavy_bucket_size_total / self.heavy_bucket_observations


@dataclass(frozen=True)
class PromotionRound:
    """Summary of one constant-promotion round."""

    round_index: int
    const_set_size_before: int
    const_set_size_after: int
    derived_consts_discovered: tuple[int, ...]
    derived_consts_promoted: tuple[int, ...]
    found_expr: str | None


@dataclass(frozen=True)
class SampleSet:
    """Current CEGIS sample points and oracle values."""

    points: tuple[int, ...]
    values: bytes

    @classmethod
    def from_oracle(cls, oracle_fn: Callable[[int], int], points: tuple[int, ...]) -> "SampleSet":
        normalized_points = tuple(sorted({point & MASK for point in points}))
        values = bytes(oracle_fn(point) & MASK for point in normalized_points)
        return cls(points=normalized_points, values=values)

    def with_point(self, oracle_fn: Callable[[int], int], point: int) -> "SampleSet":
        return SampleSet.from_oracle(oracle_fn, self.points + (point & MASK,))


@dataclass(frozen=True)
class CegisRound:
    """One CEGIS search/verify round."""

    round_index: int
    sample_size: int
    sample_points: tuple[int, ...]
    backend: str | None
    expression: str | None
    counterexample: int | None
    full_verify_passed: bool


@dataclass(frozen=True)
class SearchResult:
    """Recovered exact expression together with the backend that found it."""

    backend: str
    expr: Expr
    stats: dict[int, LevelStats] | None = None
    promotion_rounds: tuple[PromotionRound, ...] = ()
    bucket_summary: BucketSummary | None = None
    fallback_triggered: bool = False


@dataclass(frozen=True)
class SearchAttempt:
    """Full search trace, including failed runs and fallback work."""

    result: SearchResult | None
    attempted_backends: tuple[str, ...] = ()
    backend_timings_s: tuple[tuple[str, float], ...] = ()
    solved_by_backend_index: int | None = None
    signature_stats: dict[int, LevelStats] | None = None
    signature_promotion_rounds: tuple[PromotionRound, ...] = ()
    signature_bucket_summary: BucketSummary | None = None
    fallback_stats: dict[int, LevelStats] | None = None
    fallback_promotion_rounds: tuple[PromotionRound, ...] = ()
    fallback_bucket_summary: BucketSummary | None = None
    cegis_rounds: tuple[CegisRound, ...] = ()

    @property
    def fallback_triggered(self) -> bool:
        return self.fallback_stats is not None

    @property
    def solved_by_first_backend(self) -> bool:
        return self.solved_by_backend_index == 0


def collect_target() -> bytes:
    """Evaluate the oracle on the full 8-bit domain."""
    return bytes(oracle(x) for x in range(DOMAIN_SIZE))


def evaluate_expr(expr: Expr) -> bytes:
    """Evaluate a candidate expression on the full 8-bit domain."""
    return bytes(expr.eval(x) for x in range(DOMAIN_SIZE))


def find_counterexample(expr: Expr, oracle_fn: Callable[[int], int]) -> int | None:
    """Return the first full-domain input where expr disagrees with the oracle."""
    for x in range(DOMAIN_SIZE):
        if expr.eval(x) != (oracle_fn(x) & MASK):
            return x
    return None


def better(new_expr: Expr, old_expr: Expr | None) -> bool:
    """Choose the cheaper expression, then the shorter textual form."""
    if old_expr is None:
        return True
    if new_expr.cost != old_expr.cost:
        return new_expr.cost < old_expr.cost

    new_source = new_expr.shape_key
    old_source = old_expr.shape_key
    if len(new_source) != len(old_source):
        return len(new_source) < len(old_source)
    return new_source < old_source


def expr_sort_key(expr: Expr) -> tuple[int, int, str]:
    source = expr.shape_key
    return (expr.cost, len(source), source)


def mul_const_values(values: bytes, const: int) -> bytes:
    return values.translate(MUL_TABLES[const & MASK])


def shl_values(values: bytes, shift: int) -> bytes:
    return values.translate(SHL_TABLES[shift])


def shr_values(values: bytes, shift: int) -> bytes:
    return values.translate(SHR_TABLES[shift])


def add_values(left: bytes, right: bytes) -> bytes:
    return bytes((a + b) & MASK for a, b in zip(left, right))


def sub_values(left: bytes, right: bytes) -> bytes:
    return bytes((a - b) & MASK for a, b in zip(left, right))


def xor_values(left: bytes, right: bytes) -> bytes:
    return bytes((a ^ b) & MASK for a, b in zip(left, right))


def square_values(values: bytes) -> bytes:
    return bytes((value * value) & MASK for value in values)


def mul_values(left: bytes, right: bytes) -> bytes:
    return bytes((a * b) & MASK for a, b in zip(left, right))


def target_signature(target: bytes) -> bytes:
    return bytes(target[x] for x in SIG_POINTS)


def log_level(cost: int, stats: LevelStats, sem_unique: int) -> None:
    """Print a compact per-cost summary."""
    line = (
        f"cost={cost} "
        f"generated={stats.generated} "
        f"rule_pruned={stats.rule_pruned} "
        f"simplified_to_existing_shape={stats.shape_pruned} "
        f"sig_bucket_pruned={stats.sig_pruned} "
        f"sig_bucket_kept={stats.sig_bucket_kept} "
        f"sig_bucket_replaced={stats.sig_bucket_replaced} "
        f"full_eval={stats.full_eval} "
        f"sem_pruned={stats.sem_pruned} "
        f"sem_unique={sem_unique}"
    )
    if (
        stats.generated_mul_candidates
        or stats.accepted_mul_candidates
        or stats.shape_pruned_mul
        or stats.sem_pruned_mul
        or stats.max_nonlinear_depth_seen
    ):
        line += (
            f" generated_mul_candidates={stats.generated_mul_candidates} "
            f"accepted_mul_candidates={stats.accepted_mul_candidates} "
            f"shape_pruned_mul={stats.shape_pruned_mul} "
            f"sem_pruned_mul={stats.sem_pruned_mul} "
            f"max_nonlinear_depth_seen={stats.max_nonlinear_depth_seen}"
        )
    print(line)


def format_consts(values: tuple[int, ...], limit: int = 12) -> str:
    """Return a compact printable representation of constant lists."""
    if not values:
        return "-"
    head = ", ".join(str(value) for value in values[:limit])
    if len(values) > limit:
        return f"{head}, ..."
    return head


def log_promotion_round(summary: PromotionRound) -> None:
    """Print a compact summary for one promotion round."""
    print(
        f"const_round={summary.round_index} "
        f"const_set_size_before={summary.const_set_size_before} "
        f"derived_consts_discovered={len(summary.derived_consts_discovered)} "
        f"derived_consts_promoted={len(summary.derived_consts_promoted)} "
        f"const_set_size_after={summary.const_set_size_after} "
        f"found_expr={summary.found_expr or '-'}"
    )
    if summary.derived_consts_promoted:
        print(f"promoted_consts={format_consts(summary.derived_consts_promoted)}")


def selective_sig_bucket_limit(cost: int) -> int:
    """Use top-1 on light levels and top-3 on heavy levels."""
    if cost < 6:
        return 1
    return 3


def top1_sig_bucket_limit(cost: int) -> int:
    """Keep a single best candidate per signature on every level."""
    return 1


def fixed_sig_bucket_limit(limit: int) -> Callable[[int], int]:
    """Return a constant signature-bucket limit function."""
    normalized = max(1, limit)
    return lambda cost: normalized


def max_sig_bucket_limit(max_cost: int, sig_bucket_limit_fn: Callable[[int], int]) -> int:
    """Return the maximum bucket size needed across all explored costs."""
    return max(max(1, sig_bucket_limit_fn(cost)) for cost in range(1, max_cost + 1))


def update_signature_bucket(
    bucket_by_sig: dict[bytes, list[Expr]],
    sig: bytes,
    expr: Expr,
    bucket_capacity: int,
) -> bool:
    """Keep only the best few expressions for each signature bucket."""
    bucket = bucket_by_sig[sig]
    shape_key = expr.shape_key
    if any(existing.shape_key == shape_key for existing in bucket):
        return False
    was_full = len(bucket) >= bucket_capacity
    bucket.append(expr)
    bucket.sort(key=expr_sort_key)
    del bucket[bucket_capacity:]
    return was_full and any(existing.shape_key == shape_key for existing in bucket)


def should_signature_prune(
    expr: Expr,
    sig: bytes,
    target_sig: bytes,
    bucket_by_sig: dict[bytes, list[Expr]],
    use_signature_filter: bool,
    sig_bucket_limit_fn: Callable[[int], int],
    bucket_summary: BucketSummary,
) -> bool:
    """Cheap pre-filter before full-domain semantics are computed."""
    if not use_signature_filter:
        return False

    bucket = bucket_by_sig.get(sig, [])
    bucket_size = len(bucket)
    bucket_summary.max_bucket_count = max(bucket_summary.max_bucket_count, bucket_size)
    if expr.cost >= 6:
        bucket_summary.heavy_bucket_observations += 1
        bucket_summary.heavy_bucket_size_total += bucket_size

    if expr.cost < SIGNATURE_PRUNE_COST:
        return False
    if sig == target_sig:
        return False

    bucket_limit = max(1, sig_bucket_limit_fn(expr.cost))
    if bucket_size < bucket_limit:
        return False
    return not better(expr, bucket[bucket_limit - 1])


def discover_derived_consts(expr: Expr, known_consts: set[int], discovered_consts: set[int]) -> None:
    """Record new literal constants that appear in canonicalized expressions."""
    for value in expr.literal_const_values:
        if value not in known_consts:
            discovered_consts.add(value)


def find_enumerative_round(
    target: bytes,
    const_values: tuple[int, ...],
    max_cost: int = MAX_COST,
    log_progress: bool = True,
    use_signature_filter: bool = True,
    sig_bucket_limit_fn: Callable[[int], int] = top1_sig_bucket_limit,
    nonlinear: NonlinearConfig | None = None,
) -> tuple[Expr | None, dict[int, LevelStats], set[int], BucketSummary]:
    """Recover an expression by bottom-up enumeration with semantic pruning."""
    nonlinear = nonlinear or nonlinear_config()
    levels: dict[int, dict[bytes, Candidate]] = defaultdict(dict)
    best_by_sem: dict[bytes, Candidate] = {}
    bucket_by_sig: dict[bytes, list[Expr]] = defaultdict(list)
    seen_shapes: dict[int, set[str]] = defaultdict(set)
    reduced_shapes_seen: set[str] = set()
    stats: dict[int, LevelStats] = defaultdict(LevelStats)
    discovered_consts: set[int] = set()
    known_consts = set(const_values)
    target_sig = target_signature(target)
    best_target_expr: Expr | None = None
    bucket_capacity = max_sig_bucket_limit(max_cost, sig_bucket_limit_fn)
    bucket_summary = BucketSummary()

    def accept(candidate: Candidate, cost_stats: LevelStats, op: str | None = None) -> bool:
        nonlocal best_target_expr
        current_best = best_by_sem.get(candidate.sem)
        if current_best is not None and not better(candidate.expr, current_best.expr):
            cost_stats.sem_pruned += 1
            if op == "mul":
                cost_stats.sem_pruned_mul += 1
            return False

        if current_best is not None:
            levels[current_best.expr.cost].pop(candidate.sem, None)

        best_by_sem[candidate.sem] = candidate
        levels[candidate.expr.cost][candidate.sem] = candidate
        cost_stats.accepted += 1
        if op == "mul":
            cost_stats.accepted_mul_candidates += 1
        replaced = update_signature_bucket(
            bucket_by_sig,
            candidate.sig,
            candidate.expr,
            bucket_capacity=bucket_capacity,
        )
        if replaced:
            cost_stats.sig_bucket_replaced += 1

        if candidate.sem == target and better(candidate.expr, best_target_expr):
            best_target_expr = candidate.expr

        return True

    def try_candidate(expr: Expr, sig: bytes, sem_builder, target_cost: int, op: str | None = None) -> None:
        cost_stats = stats[target_cost]
        cost_stats.generated += 1
        cost_stats.max_nonlinear_depth_seen = max(
            cost_stats.max_nonlinear_depth_seen,
            expr.nonlinear_count,
        )
        if op == "mul":
            cost_stats.generated_mul_candidates += 1

        if expr.cost != target_cost:
            if (
                expr.cost < target_cost
                and sig == target_sig
                and (best_target_expr is None or expr.cost < best_target_expr.cost)
            ):
                reduced_shape = expr.shape_key
                if reduced_shape not in reduced_shapes_seen:
                    reduced_shapes_seen.add(reduced_shape)
                    discover_derived_consts(expr, known_consts, discovered_consts)
            cost_stats.rule_pruned += 1
            return

        shape_key = expr.shape_key
        if shape_key in seen_shapes[target_cost]:
            cost_stats.shape_pruned += 1
            if op == "mul":
                cost_stats.shape_pruned_mul += 1
            return
        seen_shapes[target_cost].add(shape_key)

        if should_signature_prune(
            expr,
            sig,
            target_sig,
            bucket_by_sig,
            use_signature_filter,
            sig_bucket_limit_fn,
            bucket_summary,
        ):
            cost_stats.sig_pruned += 1
            return
        if use_signature_filter:
            cost_stats.sig_bucket_kept += 1

        cost_stats.full_eval += 1
        sem = sem_builder()
        accept(Candidate(expr=expr, sig=sig, sem=sem), cost_stats, op=op)

    base_stats = stats[1]
    var_candidate = Candidate(expr=var_expr(), sig=VAR_SIG, sem=VAR_SEM)
    base_stats.generated += 1
    base_stats.full_eval += 1
    seen_shapes[1].add(var_candidate.expr.shape_key)
    accept(var_candidate, base_stats)

    for const in const_values:
        const_node = const_expr(const)
        base_stats.generated += 1
        base_stats.full_eval += 1
        seen_shapes[1].add(const_node.shape_key)
        accept(
            Candidate(
                expr=const_node,
                sig=CONST_SIG_CACHE[const & MASK],
                sem=CONST_SEM_CACHE[const & MASK],
            ),
            base_stats,
        )

    for cost in range(2, max_cost + 1):
        prev_items = tuple(levels[cost - 1].values())
        for prev in prev_items:
            if prev.expr.nonlinear_count + 1 <= nonlinear.max_nonlinear_count:
                expr = make_square(prev.expr, max_nonlinear_count=nonlinear.max_nonlinear_count)
                sig = square_values(prev.sig)
                try_candidate(expr, sig, lambda prev_sem=prev.sem: square_values(prev_sem), cost)

            for const in const_values:
                expr = make_mul_const(prev.expr, const)
                sig = mul_const_values(prev.sig, const)
                try_candidate(expr, sig, lambda prev_sem=prev.sem, c=const: mul_const_values(prev_sem, c), cost)

            for shift in SHIFT_SET:
                expr = make_shl_const(prev.expr, shift)
                sig = shl_values(prev.sig, shift)
                try_candidate(expr, sig, lambda prev_sem=prev.sem, k=shift: shl_values(prev_sem, k), cost)

                expr = make_shr_const(prev.expr, shift)
                sig = shr_values(prev.sig, shift)
                try_candidate(expr, sig, lambda prev_sem=prev.sem, k=shift: shr_values(prev_sem, k), cost)

        for left_cost in range(1, cost):
            right_cost = cost - left_cost - 1
            if right_cost < 1:
                continue

            left_items = tuple(levels[left_cost].values())
            right_items = tuple(levels[right_cost].values())

            if left_cost <= right_cost:
                for left in left_items:
                    for right in right_items:
                        if left_cost == right_cost and left.expr.shape_key > right.expr.shape_key:
                            continue

                        combined_nonlinear = left.expr.nonlinear_count + right.expr.nonlinear_count
                        if combined_nonlinear <= nonlinear.max_nonlinear_count:
                            expr = make_add(left.expr, right.expr)
                            sig = add_values(left.sig, right.sig)
                            try_candidate(
                                expr,
                                sig,
                                lambda left_sem=left.sem, right_sem=right.sem: add_values(left_sem, right_sem),
                                cost,
                            )

                            expr = make_xor(left.expr, right.expr)
                            sig = xor_values(left.sig, right.sig)
                            try_candidate(
                                expr,
                                sig,
                                lambda left_sem=left.sem, right_sem=right.sem: xor_values(left_sem, right_sem),
                                cost,
                            )

                        mul_is_allowed = (
                            left.expr == right.expr
                            and left.expr.nonlinear_count + 1 <= nonlinear.max_nonlinear_count
                        ) or (
                            (
                                nonlinear.allow_mul_of_nonlinear
                                or (left.expr.nonlinear_count == 0 and right.expr.nonlinear_count == 0)
                            )
                            and left.expr.nonlinear_count
                            + right.expr.nonlinear_count
                            + 1
                            <= nonlinear.max_nonlinear_count
                        )
                        if mul_is_allowed:
                            expr = make_mul(
                                left.expr,
                                right.expr,
                                allow_nonlinear_operands=nonlinear.allow_mul_of_nonlinear,
                                max_nonlinear_count=nonlinear.max_nonlinear_count,
                            )
                            sig = mul_values(left.sig, right.sig)
                            try_candidate(
                                expr,
                                sig,
                                lambda left_sem=left.sem, right_sem=right.sem: mul_values(left_sem, right_sem),
                                cost,
                                op="mul",
                            )

            for left in left_items:
                for right in right_items:
                    if left.expr.nonlinear_count + right.expr.nonlinear_count > nonlinear.max_nonlinear_count:
                        continue

                    expr = make_sub(left.expr, right.expr)
                    sig = sub_values(left.sig, right.sig)
                    try_candidate(
                        expr,
                        sig,
                        lambda left_sem=left.sem, right_sem=right.sem: sub_values(left_sem, right_sem),
                        cost,
                    )

        if log_progress:
            log_level(cost, stats[cost], len(levels[cost]))

        if best_target_expr is not None:
            return best_target_expr, stats, discovered_consts, bucket_summary

    return best_target_expr, stats, discovered_consts, bucket_summary


def find_enumerative_sample_round(
    sample_set: SampleSet,
    const_values: tuple[int, ...],
    max_cost: int = MAX_COST,
    log_progress: bool = True,
    nonlinear: NonlinearConfig | None = None,
) -> tuple[Expr | None, dict[int, LevelStats], set[int], BucketSummary]:
    """Recover an expression by bottom-up enumeration on a CEGIS sample set."""
    nonlinear = nonlinear or nonlinear_config()
    levels: dict[int, dict[bytes, Candidate]] = defaultdict(dict)
    best_by_sem: dict[bytes, Candidate] = {}
    seen_shapes: dict[int, set[str]] = defaultdict(set)
    reduced_shapes_seen: set[str] = set()
    stats: dict[int, LevelStats] = defaultdict(LevelStats)
    discovered_consts: set[int] = set()
    known_consts = set(const_values)
    target = sample_set.values
    best_target_expr: Expr | None = None
    bucket_summary = BucketSummary()

    def accept(candidate: Candidate, cost_stats: LevelStats, op: str | None = None) -> bool:
        nonlocal best_target_expr
        current_best = best_by_sem.get(candidate.sem)
        if current_best is not None and not better(candidate.expr, current_best.expr):
            cost_stats.sem_pruned += 1
            if op == "mul":
                cost_stats.sem_pruned_mul += 1
            return False

        if current_best is not None:
            levels[current_best.expr.cost].pop(candidate.sem, None)

        best_by_sem[candidate.sem] = candidate
        levels[candidate.expr.cost][candidate.sem] = candidate
        cost_stats.accepted += 1
        if op == "mul":
            cost_stats.accepted_mul_candidates += 1

        if candidate.sem == target and better(candidate.expr, best_target_expr):
            best_target_expr = candidate.expr

        return True

    def try_candidate(expr: Expr, sem: bytes, target_cost: int, op: str | None = None) -> None:
        cost_stats = stats[target_cost]
        cost_stats.generated += 1
        cost_stats.max_nonlinear_depth_seen = max(
            cost_stats.max_nonlinear_depth_seen,
            expr.nonlinear_count,
        )
        if op == "mul":
            cost_stats.generated_mul_candidates += 1

        if expr.cost != target_cost:
            if (
                expr.cost < target_cost
                and sem == target
                and (best_target_expr is None or expr.cost < best_target_expr.cost)
            ):
                reduced_shape = expr.shape_key
                if reduced_shape not in reduced_shapes_seen:
                    reduced_shapes_seen.add(reduced_shape)
                    discover_derived_consts(expr, known_consts, discovered_consts)
            cost_stats.rule_pruned += 1
            return

        shape_key = expr.shape_key
        if shape_key in seen_shapes[target_cost]:
            cost_stats.shape_pruned += 1
            if op == "mul":
                cost_stats.shape_pruned_mul += 1
            return
        seen_shapes[target_cost].add(shape_key)

        cost_stats.full_eval += 1
        accept(Candidate(expr=expr, sig=sem, sem=sem), cost_stats, op=op)

    sample_size = len(sample_set.points)
    base_stats = stats[1]
    var_sem = bytes(sample_set.points)
    var_candidate = Candidate(expr=var_expr(), sig=var_sem, sem=var_sem)
    base_stats.generated += 1
    base_stats.full_eval += 1
    seen_shapes[1].add(var_candidate.expr.shape_key)
    accept(var_candidate, base_stats)

    for const in const_values:
        const_node = const_expr(const)
        const_sem = bytes([const & MASK]) * sample_size
        base_stats.generated += 1
        base_stats.full_eval += 1
        seen_shapes[1].add(const_node.shape_key)
        accept(Candidate(expr=const_node, sig=const_sem, sem=const_sem), base_stats)

    if best_target_expr is not None:
        return best_target_expr, stats, discovered_consts, bucket_summary

    for cost in range(2, max_cost + 1):
        prev_items = tuple(levels[cost - 1].values())
        for prev in prev_items:
            if prev.expr.nonlinear_count + 1 <= nonlinear.max_nonlinear_count:
                expr = make_square(prev.expr, max_nonlinear_count=nonlinear.max_nonlinear_count)
                sem = square_values(prev.sem)
                try_candidate(expr, sem, cost)

            for const in const_values:
                expr = make_mul_const(prev.expr, const)
                sem = mul_const_values(prev.sem, const)
                try_candidate(expr, sem, cost)

            for shift in SHIFT_SET:
                expr = make_shl_const(prev.expr, shift)
                sem = shl_values(prev.sem, shift)
                try_candidate(expr, sem, cost)

                expr = make_shr_const(prev.expr, shift)
                sem = shr_values(prev.sem, shift)
                try_candidate(expr, sem, cost)

        for left_cost in range(1, cost):
            right_cost = cost - left_cost - 1
            if right_cost < 1:
                continue

            left_items = tuple(levels[left_cost].values())
            right_items = tuple(levels[right_cost].values())

            if left_cost <= right_cost:
                for left in left_items:
                    for right in right_items:
                        if left_cost == right_cost and left.expr.shape_key > right.expr.shape_key:
                            continue

                        combined_nonlinear = left.expr.nonlinear_count + right.expr.nonlinear_count
                        if combined_nonlinear <= nonlinear.max_nonlinear_count:
                            expr = make_add(left.expr, right.expr)
                            sem = add_values(left.sem, right.sem)
                            try_candidate(expr, sem, cost)

                            expr = make_xor(left.expr, right.expr)
                            sem = xor_values(left.sem, right.sem)
                            try_candidate(expr, sem, cost)

                        mul_is_allowed = (
                            left.expr == right.expr
                            and left.expr.nonlinear_count + 1 <= nonlinear.max_nonlinear_count
                        ) or (
                            (
                                nonlinear.allow_mul_of_nonlinear
                                or (left.expr.nonlinear_count == 0 and right.expr.nonlinear_count == 0)
                            )
                            and left.expr.nonlinear_count
                            + right.expr.nonlinear_count
                            + 1
                            <= nonlinear.max_nonlinear_count
                        )
                        if mul_is_allowed:
                            expr = make_mul(
                                left.expr,
                                right.expr,
                                allow_nonlinear_operands=nonlinear.allow_mul_of_nonlinear,
                                max_nonlinear_count=nonlinear.max_nonlinear_count,
                            )
                            sem = mul_values(left.sem, right.sem)
                            try_candidate(expr, sem, cost, op="mul")

            for left in left_items:
                for right in right_items:
                    if left.expr.nonlinear_count + right.expr.nonlinear_count > nonlinear.max_nonlinear_count:
                        continue

                    expr = make_sub(left.expr, right.expr)
                    sem = sub_values(left.sem, right.sem)
                    try_candidate(expr, sem, cost)

        if log_progress:
            log_level(cost, stats[cost], len(levels[cost]))

        if best_target_expr is not None:
            return best_target_expr, stats, discovered_consts, bucket_summary

    return best_target_expr, stats, discovered_consts, bucket_summary


def merge_level_stats(dst: LevelStats, src: LevelStats) -> None:
    """Add one level-stat block into another."""
    dst.generated += src.generated
    dst.rule_pruned += src.rule_pruned
    dst.shape_pruned += src.shape_pruned
    dst.sig_pruned += src.sig_pruned
    dst.sig_bucket_kept += src.sig_bucket_kept
    dst.sig_bucket_replaced += src.sig_bucket_replaced
    dst.full_eval += src.full_eval
    dst.sem_pruned += src.sem_pruned
    dst.accepted += src.accepted
    dst.generated_mul_candidates += src.generated_mul_candidates
    dst.accepted_mul_candidates += src.accepted_mul_candidates
    dst.shape_pruned_mul += src.shape_pruned_mul
    dst.sem_pruned_mul += src.sem_pruned_mul
    dst.max_nonlinear_depth_seen = max(dst.max_nonlinear_depth_seen, src.max_nonlinear_depth_seen)


def merge_bucket_summaries(dst: BucketSummary, src: BucketSummary) -> BucketSummary:
    """Combine two bucket summaries."""
    return BucketSummary(
        max_bucket_count=max(dst.max_bucket_count, src.max_bucket_count),
        heavy_bucket_observations=dst.heavy_bucket_observations + src.heavy_bucket_observations,
        heavy_bucket_size_total=dst.heavy_bucket_size_total + src.heavy_bucket_size_total,
    )


def find_enumerative_with_promotion(
    target: bytes,
    max_cost: int = MAX_COST,
    log_progress: bool = True,
    use_signature_filter: bool = True,
    max_promotion_rounds: int = MAX_PROMOTION_ROUNDS,
    sig_bucket_limit_fn: Callable[[int], int] = top1_sig_bucket_limit,
    nonlinear: NonlinearConfig | None = None,
) -> tuple[Expr | None, dict[int, LevelStats], tuple[PromotionRound, ...], BucketSummary]:
    """Run enumerative search with iterative promotion of derived constants."""
    nonlinear = nonlinear or nonlinear_config()
    const_set = set(BASE_CONST_SET)
    promotion_rounds: list[PromotionRound] = []
    aggregate_stats: dict[int, LevelStats] = defaultdict(LevelStats)
    aggregate_bucket_summary = BucketSummary()
    best_expr: Expr | None = None

    for round_index in range(max_promotion_rounds + 1):
        const_values = tuple(sorted(const_set))
        expr, stats, discovered_consts, bucket_summary = find_enumerative_round(
            target,
            const_values=const_values,
            max_cost=max_cost,
            log_progress=log_progress,
            use_signature_filter=use_signature_filter,
            sig_bucket_limit_fn=sig_bucket_limit_fn,
            nonlinear=nonlinear,
        )
        for cost, level_stats in stats.items():
            merge_level_stats(aggregate_stats[cost], level_stats)
        aggregate_bucket_summary = merge_bucket_summaries(aggregate_bucket_summary, bucket_summary)
        if expr is not None and better(expr, best_expr):
            best_expr = expr

        promoted_consts = tuple(sorted(value for value in discovered_consts if value not in const_set))
        summary = PromotionRound(
            round_index=round_index,
            const_set_size_before=len(const_set),
            const_set_size_after=len(const_set) + len(promoted_consts),
            derived_consts_discovered=tuple(sorted(discovered_consts)),
            derived_consts_promoted=promoted_consts,
            found_expr=expr.to_source() if expr is not None else None,
        )
        promotion_rounds.append(summary)
        if log_progress:
            log_promotion_round(summary)

        if not promoted_consts:
            break
        const_set.update(promoted_consts)

    return best_expr, dict(aggregate_stats), tuple(promotion_rounds), aggregate_bucket_summary


def find_enumerative_sample_with_promotion(
    sample_set: SampleSet,
    max_cost: int = MAX_COST,
    log_progress: bool = True,
    max_promotion_rounds: int = MAX_PROMOTION_ROUNDS,
    nonlinear: NonlinearConfig | None = None,
) -> tuple[Expr | None, dict[int, LevelStats], tuple[PromotionRound, ...], BucketSummary]:
    """Run sample-driven enumerative search with iterative derived-constant promotion."""
    nonlinear = nonlinear or nonlinear_config()
    const_set = set(BASE_CONST_SET)
    promotion_rounds: list[PromotionRound] = []
    aggregate_stats: dict[int, LevelStats] = defaultdict(LevelStats)
    aggregate_bucket_summary = BucketSummary()
    best_expr: Expr | None = None

    for round_index in range(max_promotion_rounds + 1):
        const_values = tuple(sorted(const_set))
        expr, stats, discovered_consts, bucket_summary = find_enumerative_sample_round(
            sample_set,
            const_values=const_values,
            max_cost=max_cost,
            log_progress=log_progress,
            nonlinear=nonlinear,
        )
        for cost, level_stats in stats.items():
            merge_level_stats(aggregate_stats[cost], level_stats)
        aggregate_bucket_summary = merge_bucket_summaries(aggregate_bucket_summary, bucket_summary)
        if expr is not None and better(expr, best_expr):
            best_expr = expr

        promoted_consts = tuple(sorted(value for value in discovered_consts if value not in const_set))
        summary = PromotionRound(
            round_index=round_index,
            const_set_size_before=len(const_set),
            const_set_size_after=len(const_set) + len(promoted_consts),
            derived_consts_discovered=tuple(sorted(discovered_consts)),
            derived_consts_promoted=promoted_consts,
            found_expr=expr.to_source() if expr is not None else None,
        )
        promotion_rounds.append(summary)
        if log_progress:
            log_promotion_round(summary)

        if not promoted_consts:
            break
        const_set.update(promoted_consts)

    return best_expr, dict(aggregate_stats), tuple(promotion_rounds), aggregate_bucket_summary


def find_enumerative(
    target: bytes,
    max_cost: int = MAX_COST,
    log_progress: bool = True,
    use_signature_filter: bool = True,
    sig_bucket_limit_fn: Callable[[int], int] = top1_sig_bucket_limit,
    nonlinear_mode: str = DEFAULT_NONLINEAR_MODE,
) -> tuple[Expr | None, dict[int, LevelStats]]:
    """Backward-compatible wrapper for enumerative search."""
    expr, stats, _, _ = find_enumerative_with_promotion(
        target,
        max_cost=max_cost,
        log_progress=log_progress,
        use_signature_filter=use_signature_filter,
        sig_bucket_limit_fn=sig_bucket_limit_fn,
        nonlinear=nonlinear_config(nonlinear_mode),
    )
    return expr, stats


def run_search_attempt(
    target: bytes | None = None,
    max_cost: int = MAX_COST,
    log_progress: bool = True,
    sig_bucket_limit_fn: Callable[[int], int] = top1_sig_bucket_limit,
    nonlinear_mode: str = DEFAULT_NONLINEAR_MODE,
) -> SearchAttempt:
    """Run the enumerative search pipeline and keep stats even on failure."""
    target_values = collect_target() if target is None else target
    nonlinear = nonlinear_config(nonlinear_mode)
    enumerative_backend_name = "enumerative"
    enumerative_started = perf_counter()
    enumerative_expr, signature_stats, signature_promotion_rounds, signature_bucket_summary = find_enumerative_with_promotion(
        target_values,
        max_cost=max_cost,
        log_progress=log_progress,
        use_signature_filter=True,
        sig_bucket_limit_fn=sig_bucket_limit_fn,
        nonlinear=nonlinear,
    )
    if enumerative_expr is not None:
        backend_timings_s = ((enumerative_backend_name, perf_counter() - enumerative_started),)
        return SearchAttempt(
            result=SearchResult(
                backend=enumerative_backend_name,
                expr=enumerative_expr,
                stats=signature_stats,
                promotion_rounds=signature_promotion_rounds,
                bucket_summary=signature_bucket_summary,
            ),
            attempted_backends=(enumerative_backend_name,),
            backend_timings_s=backend_timings_s,
            solved_by_backend_index=0,
            signature_stats=signature_stats,
            signature_promotion_rounds=signature_promotion_rounds,
            signature_bucket_summary=signature_bucket_summary,
        )

    if log_progress:
        print("retry=exact_enumerative_without_signature")

    enumerative_expr, fallback_stats, fallback_promotion_rounds, fallback_bucket_summary = find_enumerative_with_promotion(
        target_values,
        max_cost=max_cost,
        log_progress=log_progress,
        use_signature_filter=False,
        sig_bucket_limit_fn=sig_bucket_limit_fn,
        nonlinear=nonlinear,
    )
    backend_timings_s = ((enumerative_backend_name, perf_counter() - enumerative_started),)
    if enumerative_expr is not None:
        return SearchAttempt(
            result=SearchResult(
                backend=enumerative_backend_name,
                expr=enumerative_expr,
                stats=fallback_stats,
                promotion_rounds=fallback_promotion_rounds,
                bucket_summary=fallback_bucket_summary,
                fallback_triggered=True,
            ),
            attempted_backends=(enumerative_backend_name,),
            backend_timings_s=backend_timings_s,
            solved_by_backend_index=0,
            signature_stats=signature_stats,
            signature_promotion_rounds=signature_promotion_rounds,
            signature_bucket_summary=signature_bucket_summary,
            fallback_stats=fallback_stats,
            fallback_promotion_rounds=fallback_promotion_rounds,
            fallback_bucket_summary=fallback_bucket_summary,
        )

    return SearchAttempt(
        result=None,
        attempted_backends=(enumerative_backend_name,),
        backend_timings_s=backend_timings_s,
        solved_by_backend_index=None,
        signature_stats=signature_stats,
        signature_promotion_rounds=signature_promotion_rounds,
        signature_bucket_summary=signature_bucket_summary,
        fallback_stats=fallback_stats,
        fallback_promotion_rounds=fallback_promotion_rounds,
        fallback_bucket_summary=fallback_bucket_summary,
    )


def run_sample_search_attempt(
    sample_set: SampleSet,
    max_cost: int = MAX_COST,
    log_progress: bool = True,
    nonlinear_mode: str = DEFAULT_NONLINEAR_MODE,
) -> SearchAttempt:
    """Run the sample-set dispatcher used by CEGIS."""
    enumerative_backend_name = "enumerative"
    enumerative_started = perf_counter()
    expr, stats, promotion_rounds, bucket_summary = find_enumerative_sample_with_promotion(
        sample_set,
        max_cost=max_cost,
        log_progress=log_progress,
        nonlinear=nonlinear_config(nonlinear_mode),
    )
    enumerative_timing = ((enumerative_backend_name, perf_counter() - enumerative_started),)
    attempted_backends = (enumerative_backend_name,)
    backend_timings_s = enumerative_timing
    if expr is not None:
        return SearchAttempt(
            result=SearchResult(
                backend=enumerative_backend_name,
                expr=expr,
                stats=stats,
                promotion_rounds=promotion_rounds,
                bucket_summary=bucket_summary,
            ),
            attempted_backends=attempted_backends,
            backend_timings_s=backend_timings_s,
            solved_by_backend_index=0,
            signature_stats=stats,
            signature_promotion_rounds=promotion_rounds,
            signature_bucket_summary=bucket_summary,
        )

    return SearchAttempt(
        result=None,
        attempted_backends=attempted_backends,
        backend_timings_s=backend_timings_s,
        solved_by_backend_index=None,
        signature_stats=stats,
        signature_promotion_rounds=promotion_rounds,
        signature_bucket_summary=bucket_summary,
    )


def synthesize_cegis(
    oracle_fn: Callable[[int], int] = oracle,
    initial_points: tuple[int, ...] = INITIAL_CEGIS_POINTS,
    max_rounds: int = 32,
    max_cost: int = MAX_COST,
    log_progress: bool = True,
    nonlinear_mode: str = DEFAULT_NONLINEAR_MODE,
) -> SearchAttempt:
    """Counterexample-guided synthesis over a growing sample set."""
    sample_set = SampleSet.from_oracle(oracle_fn, initial_points)
    cegis_rounds: list[CegisRound] = []
    aggregate_stats: dict[int, LevelStats] = defaultdict(LevelStats)
    aggregate_bucket_summary = BucketSummary()
    aggregate_timings: dict[str, float] = defaultdict(float)
    last_attempt: SearchAttempt | None = None

    for round_index in range(max_rounds):
        attempt = run_sample_search_attempt(
            sample_set,
            max_cost=max_cost,
            log_progress=log_progress,
            nonlinear_mode=nonlinear_mode,
        )
        last_attempt = attempt
        for backend_name, elapsed in attempt.backend_timings_s:
            aggregate_timings[backend_name] += elapsed
        if attempt.signature_stats is not None:
            for cost, level_stats in attempt.signature_stats.items():
                merge_level_stats(aggregate_stats[cost], level_stats)
        if attempt.signature_bucket_summary is not None:
            aggregate_bucket_summary = merge_bucket_summaries(
                aggregate_bucket_summary,
                attempt.signature_bucket_summary,
            )

        result = attempt.result
        if result is None:
            cegis_rounds.append(
                CegisRound(
                    round_index=round_index,
                    sample_size=len(sample_set.points),
                    sample_points=sample_set.points,
                    backend=None,
                    expression=None,
                    counterexample=None,
                    full_verify_passed=False,
                )
            )
            return SearchAttempt(
                result=None,
                attempted_backends=attempt.attempted_backends,
                backend_timings_s=tuple(aggregate_timings.items()),
                solved_by_backend_index=None,
                signature_stats=dict(aggregate_stats),
                signature_bucket_summary=aggregate_bucket_summary,
                cegis_rounds=tuple(cegis_rounds),
            )

        counterexample = find_counterexample(result.expr, oracle_fn)
        full_verify_passed = counterexample is None
        cegis_rounds.append(
            CegisRound(
                round_index=round_index,
                sample_size=len(sample_set.points),
                sample_points=sample_set.points,
                backend=result.backend,
                expression=result.expr.to_source(),
                counterexample=counterexample,
                full_verify_passed=full_verify_passed,
            )
        )

        if full_verify_passed:
            return SearchAttempt(
                result=SearchResult(
                    backend=result.backend,
                    expr=result.expr,
                    stats=dict(aggregate_stats) if aggregate_stats else result.stats,
                    promotion_rounds=attempt.signature_promotion_rounds,
                    bucket_summary=aggregate_bucket_summary,
                    fallback_triggered=False,
                ),
                attempted_backends=attempt.attempted_backends,
                backend_timings_s=tuple(aggregate_timings.items()),
                solved_by_backend_index=attempt.solved_by_backend_index,
                signature_stats=dict(aggregate_stats),
                signature_promotion_rounds=attempt.signature_promotion_rounds,
                signature_bucket_summary=aggregate_bucket_summary,
                cegis_rounds=tuple(cegis_rounds),
            )

        sample_set = sample_set.with_point(oracle_fn, counterexample)

    attempted_backends = last_attempt.attempted_backends if last_attempt is not None else ()
    solved_index = last_attempt.solved_by_backend_index if last_attempt is not None else None
    return SearchAttempt(
        result=None,
        attempted_backends=attempted_backends,
        backend_timings_s=tuple(aggregate_timings.items()),
        solved_by_backend_index=solved_index,
        signature_stats=dict(aggregate_stats),
        signature_bucket_summary=aggregate_bucket_summary,
        cegis_rounds=tuple(cegis_rounds),
    )


def synthesize(
    target: bytes | None = None,
    max_cost: int = MAX_COST,
    log_progress: bool = True,
    sig_bucket_limit_fn: Callable[[int], int] = top1_sig_bucket_limit,
    nonlinear_mode: str = DEFAULT_NONLINEAR_MODE,
) -> SearchResult | None:
    """Run the general enumerative search backend."""
    return run_search_attempt(
        target=target,
        max_cost=max_cost,
        log_progress=log_progress,
        sig_bucket_limit_fn=sig_bucket_limit_fn,
        nonlinear_mode=nonlinear_mode,
    ).result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cost", type=int, default=MAX_COST, help="Maximum synthesis cost.")
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
    result = synthesize(
        target,
        max_cost=args.max_cost,
        nonlinear_mode=args.nonlinear_mode,
    )
    if result is None:
        print("No expression matched the oracle within the configured search bounds.")
        return 1

    print(
        f"Recovered expression with {result.backend} backend: "
        f"{result.expr.to_source()} mod 256 (cost={result.expr.cost})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
