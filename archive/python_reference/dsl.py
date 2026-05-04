"""DSL nodes for exact expression recovery."""

from __future__ import annotations

from dataclasses import dataclass, field


WIDTH = 8
MASK = (1 << WIDTH) - 1


class Expr:
    """Base class for synthesizable 8-bit expressions."""

    cost: int
    shape_key: str
    literal_const_values: frozenset[int]
    square_count: int
    nonlinear_count: int

    def eval(self, x: int) -> int:
        raise NotImplementedError

    def to_verilog(self) -> str:
        raise NotImplementedError

    def to_source(self) -> str:
        return self.shape_key

    def repr_key(self) -> str:
        return self.shape_key

    def to_verilog8(self) -> str:
        return f"(({self.to_verilog()}) & 8'hFF)"


def _mask(value: int) -> int:
    return value & MASK


def _const_value(expr: Expr) -> int | None:
    if isinstance(expr, Const):
        return expr.value & MASK
    return None


def _is_zero(expr: Expr) -> bool:
    value = _const_value(expr)
    return value == 0


def _expr_sort_key(expr: Expr) -> tuple[str, int]:
    return (expr.shape_key, expr.cost)


def _split_add(expr: Expr) -> tuple[list[Expr], int]:
    if isinstance(expr, Add):
        left_terms, left_const = _split_add(expr.left)
        right_terms, right_const = _split_add(expr.right)
        return left_terms + right_terms, _mask(left_const + right_const)

    const_value = _const_value(expr)
    if const_value is not None:
        return [], const_value

    return [expr], 0


def _build_add_chain(terms: list[Expr]) -> Expr:
    if not terms:
        return const_expr(0)
    expr = terms[0]
    for term in terms[1:]:
        expr = Add(expr, term)
    return expr


def _split_xor(expr: Expr) -> tuple[list[Expr], int]:
    if isinstance(expr, Xor):
        left_terms, left_const = _split_xor(expr.left)
        right_terms, right_const = _split_xor(expr.right)
        return left_terms + right_terms, left_const ^ right_const

    const_value = _const_value(expr)
    if const_value is not None:
        return [], const_value

    return [expr], 0


def _build_xor_chain(terms: list[Expr]) -> Expr:
    if not terms:
        return const_expr(0)
    expr = terms[0]
    for term in terms[1:]:
        expr = Xor(expr, term)
    return expr


@dataclass(frozen=True, slots=True)
class Var(Expr):
    """Input variable."""

    cost: int = field(init=False, default=1, compare=False, repr=False)
    shape_key: str = field(init=False, default="x", compare=False, repr=False)
    square_count: int = field(init=False, default=0, compare=False, repr=False)
    nonlinear_count: int = field(init=False, default=0, compare=False, repr=False)
    literal_const_values: frozenset[int] = field(
        init=False,
        default_factory=frozenset,
        compare=False,
        repr=False,
    )

    def eval(self, x: int) -> int:
        return x & MASK

    def to_verilog(self) -> str:
        return "x"


@dataclass(frozen=True, slots=True)
class Const(Expr):
    """8-bit literal constant."""

    value: int
    cost: int = field(init=False, default=1, compare=False, repr=False)
    shape_key: str = field(init=False, default="", compare=False, repr=False)
    square_count: int = field(init=False, default=0, compare=False, repr=False)
    nonlinear_count: int = field(init=False, default=0, compare=False, repr=False)
    literal_const_values: frozenset[int] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        value = _mask(self.value)
        object.__setattr__(self, "value", value)
        shape_key = str(value)
        object.__setattr__(self, "shape_key", shape_key)
        object.__setattr__(self, "literal_const_values", frozenset({value}))


    def eval(self, x: int) -> int:
        return self.value & MASK

    def to_verilog(self) -> str:
        return f"8'd{self.value & MASK}"


@dataclass(frozen=True, slots=True)
class MulConst(Expr):
    """Multiply an expression by an 8-bit constant."""

    expr: Expr
    const: int
    cost: int = field(init=False, default=0, compare=False, repr=False)
    shape_key: str = field(init=False, default="", compare=False, repr=False)
    square_count: int = field(init=False, default=0, compare=False, repr=False)
    nonlinear_count: int = field(init=False, default=0, compare=False, repr=False)
    literal_const_values: frozenset[int] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        const = _mask(self.const)
        object.__setattr__(self, "const", const)
        cost = self.expr.cost + 1
        shape_key = f"({const} * {self.expr.to_source()})"
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "shape_key", shape_key)
        object.__setattr__(self, "square_count", self.expr.square_count)
        object.__setattr__(self, "nonlinear_count", self.expr.nonlinear_count)
        object.__setattr__(
            self,
            "literal_const_values",
            self.expr.literal_const_values | frozenset({const}),
        )


    def eval(self, x: int) -> int:
        return (self.expr.eval(x) * (self.const & MASK)) & MASK

    def to_verilog(self) -> str:
        return f"({self.expr.to_verilog8()} * 8'd{self.const & MASK})"


@dataclass(frozen=True, slots=True)
class Add(Expr):
    """Add two 8-bit expressions modulo 256."""

    left: Expr
    right: Expr
    cost: int = field(init=False, default=0, compare=False, repr=False)
    shape_key: str = field(init=False, default="", compare=False, repr=False)
    square_count: int = field(init=False, default=0, compare=False, repr=False)
    nonlinear_count: int = field(init=False, default=0, compare=False, repr=False)
    literal_const_values: frozenset[int] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        cost = self.left.cost + self.right.cost + 1
        shape_key = f"({self.left.to_source()} + {self.right.to_source()})"
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "shape_key", shape_key)
        object.__setattr__(self, "square_count", self.left.square_count + self.right.square_count)
        object.__setattr__(
            self,
            "nonlinear_count",
            self.left.nonlinear_count + self.right.nonlinear_count,
        )
        object.__setattr__(
            self,
            "literal_const_values",
            self.left.literal_const_values | self.right.literal_const_values,
        )


    def eval(self, x: int) -> int:
        return (self.left.eval(x) + self.right.eval(x)) & MASK

    def to_verilog(self) -> str:
        return f"({self.left.to_verilog8()} + {self.right.to_verilog8()})"


@dataclass(frozen=True, slots=True)
class Sub(Expr):
    """Subtract two 8-bit expressions modulo 256."""

    left: Expr
    right: Expr
    cost: int = field(init=False, default=0, compare=False, repr=False)
    shape_key: str = field(init=False, default="", compare=False, repr=False)
    square_count: int = field(init=False, default=0, compare=False, repr=False)
    nonlinear_count: int = field(init=False, default=0, compare=False, repr=False)
    literal_const_values: frozenset[int] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        cost = self.left.cost + self.right.cost + 1
        shape_key = f"({self.left.to_source()} - {self.right.to_source()})"
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "shape_key", shape_key)
        object.__setattr__(self, "square_count", self.left.square_count + self.right.square_count)
        object.__setattr__(
            self,
            "nonlinear_count",
            self.left.nonlinear_count + self.right.nonlinear_count,
        )
        object.__setattr__(
            self,
            "literal_const_values",
            self.left.literal_const_values | self.right.literal_const_values,
        )


    def eval(self, x: int) -> int:
        return (self.left.eval(x) - self.right.eval(x)) & MASK

    def to_verilog(self) -> str:
        return f"({self.left.to_verilog8()} - {self.right.to_verilog8()})"


@dataclass(frozen=True, slots=True)
class Xor(Expr):
    """Bitwise XOR between two 8-bit expressions."""

    left: Expr
    right: Expr
    cost: int = field(init=False, default=0, compare=False, repr=False)
    shape_key: str = field(init=False, default="", compare=False, repr=False)
    square_count: int = field(init=False, default=0, compare=False, repr=False)
    nonlinear_count: int = field(init=False, default=0, compare=False, repr=False)
    literal_const_values: frozenset[int] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        cost = self.left.cost + self.right.cost + 1
        shape_key = f"({self.left.to_source()} ^ {self.right.to_source()})"
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "shape_key", shape_key)
        object.__setattr__(self, "square_count", self.left.square_count + self.right.square_count)
        object.__setattr__(
            self,
            "nonlinear_count",
            self.left.nonlinear_count + self.right.nonlinear_count,
        )
        object.__setattr__(
            self,
            "literal_const_values",
            self.left.literal_const_values | self.right.literal_const_values,
        )


    def eval(self, x: int) -> int:
        return (self.left.eval(x) ^ self.right.eval(x)) & MASK

    def to_verilog(self) -> str:
        return f"({self.left.to_verilog8()} ^ {self.right.to_verilog8()})"


@dataclass(frozen=True, slots=True)
class ShlConst(Expr):
    """Shift an 8-bit expression left by a constant amount."""

    expr: Expr
    shift: int
    cost: int = field(init=False, default=0, compare=False, repr=False)
    shape_key: str = field(init=False, default="", compare=False, repr=False)
    square_count: int = field(init=False, default=0, compare=False, repr=False)
    nonlinear_count: int = field(init=False, default=0, compare=False, repr=False)
    literal_const_values: frozenset[int] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        cost = self.expr.cost + 1
        shape_key = f"({self.expr.to_source()} << {self.shift})"
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "shape_key", shape_key)
        object.__setattr__(self, "square_count", self.expr.square_count)
        object.__setattr__(self, "nonlinear_count", self.expr.nonlinear_count)
        object.__setattr__(self, "literal_const_values", self.expr.literal_const_values)


    def eval(self, x: int) -> int:
        return (self.expr.eval(x) << self.shift) & MASK

    def to_verilog(self) -> str:
        return f"({self.expr.to_verilog8()} << {self.shift})"


@dataclass(frozen=True, slots=True)
class ShrConst(Expr):
    """Shift an 8-bit expression right by a constant amount."""

    expr: Expr
    shift: int
    cost: int = field(init=False, default=0, compare=False, repr=False)
    shape_key: str = field(init=False, default="", compare=False, repr=False)
    square_count: int = field(init=False, default=0, compare=False, repr=False)
    nonlinear_count: int = field(init=False, default=0, compare=False, repr=False)
    literal_const_values: frozenset[int] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        cost = self.expr.cost + 1
        shape_key = f"({self.expr.to_source()} >> {self.shift})"
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "shape_key", shape_key)
        object.__setattr__(self, "square_count", self.expr.square_count)
        object.__setattr__(self, "nonlinear_count", self.expr.nonlinear_count)
        object.__setattr__(self, "literal_const_values", self.expr.literal_const_values)


    def eval(self, x: int) -> int:
        return (self.expr.eval(x) >> self.shift) & MASK

    def to_verilog(self) -> str:
        return f"({self.expr.to_verilog8()} >> {self.shift})"


@dataclass(frozen=True, slots=True)
class Square(Expr):
    """Square an expression modulo 256."""

    expr: Expr
    cost: int = field(init=False, default=0, compare=False, repr=False)
    shape_key: str = field(init=False, default="", compare=False, repr=False)
    square_count: int = field(init=False, default=0, compare=False, repr=False)
    nonlinear_count: int = field(init=False, default=0, compare=False, repr=False)
    literal_const_values: frozenset[int] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        cost = self.expr.cost + 1
        shape_key = f"Square({self.expr.to_source()})"
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "shape_key", shape_key)
        object.__setattr__(self, "square_count", self.expr.square_count + 1)
        object.__setattr__(self, "nonlinear_count", self.expr.nonlinear_count + 1)
        object.__setattr__(self, "literal_const_values", self.expr.literal_const_values)

    def eval(self, x: int) -> int:
        value = self.expr.eval(x)
        return (value * value) & MASK

    def to_verilog(self) -> str:
        value = self.expr.to_verilog8()
        return f"({value} * {value})"


@dataclass(frozen=True, slots=True)
class Mul(Expr):
    """Multiply two expressions modulo 256."""

    left: Expr
    right: Expr
    cost: int = field(init=False, default=0, compare=False, repr=False)
    shape_key: str = field(init=False, default="", compare=False, repr=False)
    square_count: int = field(init=False, default=0, compare=False, repr=False)
    nonlinear_count: int = field(init=False, default=0, compare=False, repr=False)
    literal_const_values: frozenset[int] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        cost = self.left.cost + self.right.cost + 1
        shape_key = f"({self.left.to_source()} * {self.right.to_source()})"
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "shape_key", shape_key)
        object.__setattr__(self, "square_count", self.left.square_count + self.right.square_count)
        object.__setattr__(
            self,
            "nonlinear_count",
            self.left.nonlinear_count + self.right.nonlinear_count + 1,
        )
        object.__setattr__(
            self,
            "literal_const_values",
            self.left.literal_const_values | self.right.literal_const_values,
        )

    def eval(self, x: int) -> int:
        return (self.left.eval(x) * self.right.eval(x)) & MASK

    def to_verilog(self) -> str:
        return f"({self.left.to_verilog8()} * {self.right.to_verilog8()})"


_VAR_EXPR = Var()
_CONST_CACHE = tuple(Const(value) for value in range(MASK + 1))


def var_expr() -> Var:
    """Return the shared input-variable node."""
    return _VAR_EXPR


def const_expr(value: int) -> Const:
    """Return a shared 8-bit literal node."""
    return _CONST_CACHE[_mask(value)]


def literal_consts(expr: Expr) -> frozenset[int]:
    """Return all literal constants that appear inside the expression tree."""
    return expr.literal_const_values


def make_mul_const(expr: Expr, const: int) -> Expr:
    const = _mask(const)
    if const == 0:
        return const_expr(0)
    if const == 1:
        return expr

    const_value = _const_value(expr)
    if const_value is not None:
        return const_expr(_mask(const_value * const))

    if isinstance(expr, MulConst):
        return make_mul_const(expr.expr, _mask(expr.const * const))

    return MulConst(expr, const)


def make_shl_const(expr: Expr, shift: int) -> Expr:
    if shift == 0:
        return expr

    const_value = _const_value(expr)
    if const_value is not None:
        return const_expr(_mask(const_value << shift))

    if isinstance(expr, ShlConst):
        total_shift = expr.shift + shift
        if total_shift >= WIDTH:
            return const_expr(0)
        return ShlConst(expr.expr, total_shift)

    return ShlConst(expr, shift)


def make_shr_const(expr: Expr, shift: int) -> Expr:
    if shift == 0:
        return expr

    const_value = _const_value(expr)
    if const_value is not None:
        return const_expr(_mask(const_value >> shift))

    if isinstance(expr, ShrConst):
        total_shift = expr.shift + shift
        if total_shift >= WIDTH:
            return const_expr(0)
        return ShrConst(expr.expr, total_shift)

    return ShrConst(expr, shift)


def make_square(expr: Expr, max_nonlinear_count: int = 1) -> Expr:
    if expr.nonlinear_count + 1 > max_nonlinear_count:
        raise ValueError("Square exceeds the current nonlinear-count limit")

    const_value = _const_value(expr)
    if const_value is not None:
        return const_expr(_mask(const_value * const_value))

    return Square(expr)


def make_mul(
    left: Expr,
    right: Expr,
    allow_nonlinear_operands: bool = False,
    max_nonlinear_count: int = 1,
) -> Expr:
    left_const = _const_value(left)
    right_const = _const_value(right)

    if left_const == 0 or right_const == 0:
        return const_expr(0)
    if left_const == 1:
        return right
    if right_const == 1:
        return left
    if left_const is not None and right_const is not None:
        return const_expr(_mask(left_const * right_const))
    if left_const is not None:
        return make_mul_const(right, left_const)
    if right_const is not None:
        return make_mul_const(left, right_const)
    if left == right:
        return make_square(left, max_nonlinear_count=max_nonlinear_count)

    if not allow_nonlinear_operands and (left.nonlinear_count > 0 or right.nonlinear_count > 0):
        raise ValueError("Mul operands must be linear under the current DSL limit")
    if left.nonlinear_count + right.nonlinear_count + 1 > max_nonlinear_count:
        raise ValueError("Mul exceeds the current nonlinear-count limit")

    if _expr_sort_key(right) < _expr_sort_key(left):
        left, right = right, left

    return Mul(left, right)


def make_add(left: Expr, right: Expr) -> Expr:
    terms: list[Expr] = []
    left_terms, left_const = _split_add(left)
    right_terms, right_const = _split_add(right)
    terms.extend(left_terms)
    terms.extend(right_terms)

    const_sum = _mask(left_const + right_const)
    terms.sort(key=_expr_sort_key)
    if const_sum != 0:
        terms.append(const_expr(const_sum))

    if not terms:
        return const_expr(0)
    if len(terms) == 1:
        return terms[0]
    return _build_add_chain(terms)


def make_sub(left: Expr, right: Expr) -> Expr:
    if _is_zero(right):
        return left
    if left == right:
        return const_expr(0)

    left_const = _const_value(left)
    right_const = _const_value(right)
    if left_const is not None and right_const is not None:
        return const_expr(_mask(left_const - right_const))

    if right_const is not None:
        add_terms, add_const = _split_add(left)
        if add_terms or add_const != 0:
            const_term = _mask(add_const - right_const)
            ordered_terms = sorted(add_terms, key=_expr_sort_key)
            if const_term != 0:
                ordered_terms.append(const_expr(const_term))
            return _build_add_chain(ordered_terms)

        if isinstance(left, Sub):
            nested_right_const = _const_value(left.right)
            if nested_right_const is not None:
                return make_sub(left.left, const_expr(_mask(nested_right_const + right_const)))

    return Sub(left, right)


def make_xor(left: Expr, right: Expr) -> Expr:
    left_terms, left_const = _split_xor(left)
    right_terms, right_const = _split_xor(right)

    parity: dict[Expr, int] = {}
    for term in left_terms + right_terms:
        parity[term] = parity.get(term, 0) ^ 1

    terms = sorted((term for term, bit in parity.items() if bit), key=_expr_sort_key)
    const_term = left_const ^ right_const
    if const_term != 0:
        terms.append(const_expr(_mask(const_term)))

    if not terms:
        return const_expr(0)
    if len(terms) == 1:
        return terms[0]
    return _build_xor_chain(terms)
