"""Benchmark targets for the 8-bit exact synthesizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


OracleFn = Callable[[int], int]


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark oracle together with lightweight metadata."""

    name: str
    category: str
    description: str
    oracle: OracleFn
    expected_found: bool


def _mask(value: int) -> int:
    return value & 0xFF


BENCHMARKS: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="affine_linear",
        category="affine",
        description="(3*x + 5) mod 256",
        oracle=lambda x: _mask((3 * x) + 5),
        expected_found=True,
    ),
    BenchmarkCase(
        name="affine_mulsub",
        category="affine",
        description="(7*x - 13) mod 256",
        oracle=lambda x: _mask((7 * x) - 13),
        expected_found=True,
    ),
    BenchmarkCase(
        name="xor_shift_add",
        category="xor_shift_add",
        description="(x ^ 17) + (x >> 2)",
        oracle=lambda x: _mask((x ^ 17) + (x >> 2)),
        expected_found=True,
    ),
    BenchmarkCase(
        name="nested_xor",
        category="promotion",
        description="(x ^ 17) ^ 5",
        oracle=lambda x: _mask((x ^ 17) ^ 5),
        expected_found=True,
    ),
    BenchmarkCase(
        name="add_tail",
        category="promotion",
        description="(x + 7) + 13",
        oracle=lambda x: _mask((x + 7) + 13),
        expected_found=True,
    ),
    BenchmarkCase(
        name="shift_add_const",
        category="shift",
        description="(x << 3) + 5",
        oracle=lambda x: _mask((x << 3) + 5),
        expected_found=True,
    ),
    BenchmarkCase(
        name="shift_mix",
        category="shift",
        description="(x << 3) + (x >> 1)",
        oracle=lambda x: _mask((x << 3) + (x >> 1)),
        expected_found=True,
    ),
    BenchmarkCase(
        name="mul_xor_const",
        category="mulconst",
        description="(3*x) ^ 5",
        oracle=lambda x: _mask((3 * x) ^ 5),
        expected_found=True,
    ),
    BenchmarkCase(
        name="mul_shift_add",
        category="mulconst",
        description="(7*x) + (x >> 1)",
        oracle=lambda x: _mask((7 * x) + (x >> 1)),
        expected_found=True,
    ),
    BenchmarkCase(
        name="shift_chain",
        category="canonicalization",
        description="(x << 2) << 3",
        oracle=lambda x: _mask((x << 2) << 3),
        expected_found=True,
    ),
    BenchmarkCase(
        name="mul_chain",
        category="canonicalization",
        description="(3*x) * 5",
        oracle=lambda x: _mask((3 * x) * 5),
        expected_found=True,
    ),
    BenchmarkCase(
        name="square",
        category="nonlinear",
        description="x*x",
        oracle=lambda x: _mask(x * x),
        expected_found=True,
    ),
    BenchmarkCase(
        name="square_plus_5",
        category="nonlinear",
        description="(x*x) + 5",
        oracle=lambda x: _mask((x * x) + 5),
        expected_found=True,
    ),
    BenchmarkCase(
        name="square_xor_17",
        category="nonlinear",
        description="(x*x) ^ 17",
        oracle=lambda x: _mask((x * x) ^ 17),
        expected_found=True,
    ),
    BenchmarkCase(
        name="shifted_square_10",
        category="nonlinear",
        description="(x - 10)^2",
        oracle=lambda x: _mask((x - 10) * (x - 10)),
        expected_found=True,
    ),
    BenchmarkCase(
        name="shifted_square_plus_const",
        category="nonlinear",
        description="(x - 10)^2 + 5",
        oracle=lambda x: _mask(((x - 10) * (x - 10)) + 5),
        expected_found=True,
    ),
    BenchmarkCase(
        name="x_times_x_plus_1",
        category="nonlinear_mul",
        description="x * (x + 1)",
        oracle=lambda x: _mask(x * (x + 1)),
        expected_found=True,
    ),
    BenchmarkCase(
        name="x_times_x_minus_7",
        category="nonlinear_mul",
        description="x * (x - 7)",
        oracle=lambda x: _mask(x * (x - 7)),
        expected_found=True,
    ),
    BenchmarkCase(
        name="x_times_x_plus_1_plus_5",
        category="nonlinear_mul",
        description="x * (x + 1) + 5",
        oracle=lambda x: _mask((x * (x + 1)) + 5),
        expected_found=True,
    ),
    BenchmarkCase(
        name="x_times_x_shr_1",
        category="nonlinear_mul",
        description="x * (x >> 1)",
        oracle=lambda x: _mask(x * (x >> 1)),
        expected_found=True,
    ),
    BenchmarkCase(
        name="x_plus_1_times_x_shr_1",
        category="nonlinear_mul",
        description="(x + 1) * (x >> 1)",
        oracle=lambda x: _mask((x + 1) * (x >> 1)),
        expected_found=True,
    ),
    BenchmarkCase(
        name="x_cubed",
        category="nonlinear_cubic",
        description="x^3",
        oracle=lambda x: _mask(x * x * x),
        expected_found=False,
    ),
    BenchmarkCase(
        name="x_cubed_plus_5",
        category="nonlinear_cubic",
        description="x^3 + 5",
        oracle=lambda x: _mask((x * x * x) + 5),
        expected_found=False,
    ),
    BenchmarkCase(
        name="square_plus_affine",
        category="nonlinear",
        description="x^2 + 17*x + 5",
        oracle=lambda x: _mask((x * x) + (17 * x) + 5),
        expected_found=False,
    ),
    BenchmarkCase(
        name="mixed_not_found",
        category="mixed",
        description="(x << 2) ^ ((3*x) + 9)",
        oracle=lambda x: _mask((x << 2) ^ ((3 * x) + 9)),
        expected_found=False,
    ),
    BenchmarkCase(
        name="awkward_not_found",
        category="mixed",
        description="((x ^ 91) + (x << 1)) - 37",
        oracle=lambda x: _mask(((x ^ 91) + (x << 1)) - 37),
        expected_found=False,
    ),
)


BENCHMARK_BY_NAME = {case.name: case for case in BENCHMARKS}
