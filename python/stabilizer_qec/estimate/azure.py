# Ported from the Azure Quantum Resource Estimator (microsoft/qdk, source/resource_estimator/src, the `estimates`
# and `system` modules; Copyright (c) Microsoft Corporation, MIT License; see LICENSE-qdk in this directory),
# modified for stabilizer_qec: Python dataclasses in place of the Rust structs and traits, fasteval 0.2.4's
# parser, constant folding and evaluator written out in place of the compiled expressions, the binomial quantile
# of the `probability` crate (0.20) written out, the profiling output left out; with the QDK's Python
# `LogicalCounts.estimate` and `EstimatorResult` (qdk/estimator/_estimator.py) as data, their HTML left out.
# The numbers are the QDK's bit for bit on its own test cases (tests/estimate_port/qdk_estimates.json), and the
# whole result, the report's text included, on each of them (tests/estimate_port/qdk_reports.json.gz).
"""Azure's Resource Estimator: from logical counts (qubits, T, rotations, CCZ, measurements) to physical
qubits and runtime on a chosen qubit model and QEC scheme, with the T factories found by the QDK's exhaustive
search over distillation rounds and code distances, and the QDK's report: the counts formatted as it formats
them and its groups of labelled, explained entries.

Two differences from the QDK are accepted. The QDK's counts are u64 and wrap silently when a product
overflows (a runtime past 1.8e19 ns, a count past 2^64), where the integers here keep growing. And a float
reaches the QDK through its JSON round trip, which keeps 15 significant digits exactly, so an input written with
16 or 17 may land one ulp away from the value used here.

>>> from stabilizer_qec.estimate import LogicalCounts, azure_estimate
>>> r = azure_estimate(LogicalCounts(num_qubits=100, t_count=100_000, ccz_count=10_000, measurement_count=10_000))
>>> r["physicalCounts"]["physicalQubits"], r["physicalCounts"]["runtime"], r["logicalQubit"]["codeDistance"]
(400060, 1064000000, 19)
>>> r["tfactory"]["numUnitsPerRound"], r["tfactory"]["codeDistancePerRound"]
([18, 1], [5, 15])
>>> r["physicalCountsFormatted"]["runtime"], r["physicalCountsFormatted"]["physicalQubits"]
('1 secs', '400.06k')
>>> res = LogicalCounts(num_qubits=100, t_count=100_000).estimate({"qubitParams": {"name": "qubit_gate_ns_e4"}})
>>> res["reportData"]["groups"][0]["title"], res.summary.splitlines()[1].split()
('Physical resource estimates', ['Runtime', '360', 'millisecs', 'Total', 'runtime'])
"""

from __future__ import annotations

import decimal
import functools
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

MAX_CODE_DISTANCE = 50
MAX_DISTILLATION_ROUNDS = 3
MAX_EXTRA_DISTILLATION_ROUNDS = 4
NUM_MEASUREMENTS_PER_R = 1
NUM_MEASUREMENTS_PER_TOF = 3
NUM_TS_PER_ROTATION_A_COEFFICIENT = 0.53
NUM_TS_PER_ROTATION_B_COEFFICIENT = 4.86

# ---------------------------------------------------------------------------------------------
# Arithmetic the way the QDK does it
# ---------------------------------------------------------------------------------------------


def _powi(a: float, b: int) -> float:
    """`f64::powi`: repeated squaring (compiler-rt's `__powidf2`), which is not `a ** b` in the last bit."""
    recip = b < 0
    b = -b if recip else b
    r = 1.0
    while True:
        if b & 1:
            r *= a
        b //= 2
        if b == 0:
            break
        a *= a
    return 1.0 / r if recip else r


def _round_half_away(x: float) -> float:
    """`f64::round`: half away from zero, exact (not `floor(x + 0.5)`, which rounds 0.49999999999999994 up); a
    non-finite value is returned as it is."""
    if not math.isfinite(x):
        return x
    a = abs(x)
    r = math.floor(a)
    if a - r >= 0.5:
        r += 1
    return math.copysign(float(r), x)


def _as_u64(x: float) -> int:
    """`x as u64` (and `as usize`): saturating, NaN to 0, the fraction dropped."""
    if x != x or x <= 0.0:
        return 0
    if x >= 18446744073709551616.0:
        return 2**64 - 1
    return int(x)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ---------------------------------------------------------------------------------------------
# Formulas: fasteval 0.2.4 as the QDK drives it (`Parser::new`, `Slab::new`, `compile`, `eval` on a
# `BTreeMap<String, f64>`), its grammar, its constant folding and its f64 arithmetic written out
# ---------------------------------------------------------------------------------------------

_EXPR_LEN_LIMIT, _EXPR_DEPTH_LIMIT, _SLAB_CAPACITY = 4096, 32, 64
_EPS8 = 8.0 * 2.220446049250313e-16
_PRECEDENCE = {"||": 1, "&&": 2, "!=": 3, "==": 4, ">=": 5, "<=": 6, ">": 7, "<": 8, "+": 9, "-": 10, "*": 11, "/": 12, "%": 13, "^": 14}
_COMPARISONS = ("==", "!=", "<", ">", "<=", ">=")
_SUFFIXES = {"k": 3, "K": 3, "M": 6, "G": 9, "T": 12, "m": -3, "u": -6, "µ": -6, "n": -9, "p": -12}
_ARITY = {"int": (1, 1), "ceil": (1, 1), "floor": (1, 1), "abs": (1, 1), "sign": (1, 1), "log": (1, 2), "round": (1, 2), "min": (1, 10**9), "max": (1, 10**9), "e": (0, 0), "pi": (0, 0), "sin": (1, 1), "cos": (1, 1), "tan": (1, 1), "asin": (1, 1), "acos": (1, 1), "atan": (1, 1), "sinh": (1, 1), "cosh": (1, 1), "tanh": (1, 1), "asinh": (1, 1), "acosh": (1, 1), "atanh": (1, 1)}


def _eq(a: float, b: float) -> bool:
    """fasteval's `f64_eq!`: within 8 ε (so two infinities are not equal, their difference being NaN)."""
    return abs(a - b) <= _EPS8


def _ne(a: float, b: float) -> bool:
    return abs(a - b) > _EPS8


def _div(a: float, b: float) -> float:
    """f64 division: by zero, ±infinity by the signs, NaN for 0/0."""
    if b == 0.0:
        return math.nan if a == 0.0 or a != a else math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def _fmod(a: float, b: float) -> float:
    """f64 `%`: the sign of the dividend; NaN for an infinite dividend or a zero divisor."""
    try:
        return math.fmod(a, b)
    except ValueError:
        return math.nan


def _powf(a: float, b: float) -> float:
    """`f64::powf` (C `pow`): a negative base to a fractional power is NaN, zero to a negative power infinite
    with the sign of the zero for an odd integer power, an overflow infinite with the sign of the result."""
    try:
        out = a**b
    except ZeroDivisionError:
        odd = math.isfinite(b) and b == math.floor(b) and math.fmod(b, 2.0) != 0.0
        return math.copysign(math.inf, a) if odd else math.inf
    except OverflowError:
        odd = math.isfinite(b) and b == math.floor(b) and math.fmod(b, 2.0) != 0.0
        return -math.inf if a < 0.0 and odd else math.inf
    return out if isinstance(out, float) else math.nan


def _ln(x: float, fn=math.log) -> float:
    if x != x or x < 0.0:
        return math.nan
    if x == 0.0:
        return -math.inf
    return x if x == math.inf else fn(x)


def _log(base: float, x: float) -> float:
    """fasteval's `log`: base 2 and base 10 (within 8 ε) by the dedicated functions, else `ln x / ln base`."""
    if _eq(base, 2.0):
        return _ln(x, math.log2)
    if _eq(base, 10.0):
        return _ln(x, math.log10)
    return _div(_ln(x), _ln(base))


def _guarded(fn, overflow=None):
    """A `math` function with f64 semantics: NaN outside its domain, ±infinity (by `overflow`) when it overflows."""

    def f(x):
        try:
            return fn(x)
        except ValueError:
            return math.nan
        except OverflowError:
            return overflow(x) if overflow else math.inf

    return f


def _atanh(x: float) -> float:
    return math.copysign(math.inf, x) if abs(x) == 1.0 else _guarded(math.atanh)(x)


_UNARY_FUNCTIONS = {
    "int": lambda x: x if not math.isfinite(x) else math.copysign(float(math.trunc(x)), x),
    "ceil": lambda x: x if not math.isfinite(x) else math.copysign(float(math.ceil(x)), x),
    "floor": lambda x: x if not math.isfinite(x) else math.copysign(float(math.floor(x)), x),
    "abs": abs,
    "sign": lambda x: x if x != x else math.copysign(1.0, x),
    "sin": _guarded(math.sin), "cos": _guarded(math.cos), "tan": _guarded(math.tan), "asin": _guarded(math.asin), "acos": _guarded(math.acos), "atan": math.atan,
    "sinh": _guarded(math.sinh, lambda x: math.copysign(math.inf, x)), "cosh": _guarded(math.cosh), "tanh": math.tanh, "asinh": math.asinh, "acosh": _guarded(math.acosh), "atanh": _atanh,
}


def _compare(op: str, l: float, r: float) -> float:
    if op == "==":
        return 1.0 if _eq(l, r) else 0.0
    if op == "!=":
        return 1.0 if _ne(l, r) else 0.0
    return 1.0 if {"<": l < r, ">": l > r, "<=": l <= r, ">=": l >= r}[op] else 0.0


class _Formula:
    """fasteval's parser (parser.rs): `Expression: Value (BinaryOp Value)*`; a value is a constant (a sign may
    be part of it, so `-2^2` is 4), a unary `+ - !`, a bracketed expression, a variable or a function call; a
    constant may carry one of the SI suffixes `p n u µ m k K M G T` or an exponent, and `inf`/`NaN` are
    constants. Errors are fasteval's: TooLong (over 4096 bytes), TooDeep (nesting over 32), SlabOverflow (over
    64 sub-expressions or unary values), InvalidValue, UnparsedTokensRemaining, ParseF64, Expected, WrongArgs."""

    __slots__ = ("text", "pos", "exprs", "vals")

    def __init__(self, text: str):
        self.text, self.pos, self.exprs, self.vals = text, 0, 0, 0

    def fail(self, what: str):
        raise ValueError(f"cannot evaluate expression {self.text!r}: {what}")

    def peek(self, k: int = 0):
        i = self.pos + k
        return self.text[i] if i < len(self.text) else None

    def spaces(self) -> None:
        while self.peek() in (" ", "\t", "\n", "\r"):
            self.pos += 1

    def expression(self, depth: int, expect_eof: bool):
        if depth > _EXPR_DEPTH_LIMIT:
            self.fail("TooDeep")
        first = self.value(depth)
        pairs = []
        while True:
            op = self.binary_op()
            if op is None:
                break
            pairs.append((op, self.value(depth)))
        self.spaces()
        if expect_eof and self.pos < len(self.text):
            self.fail(f"UnparsedTokensRemaining({self.text[self.pos:]!r})")
        self.exprs += 1
        if self.exprs > _SLAB_CAPACITY:
            self.fail("SlabOverflow")
        return first, pairs

    def value(self, depth: int):
        if depth > _EXPR_DEPTH_LIMIT:
            self.fail("TooDeep")
        c = self.constant()
        if c is not None:
            return ("const", c)
        for read in (self.unary, self.callable):
            v = read(depth)
            if v is not None:
                return v
        self.fail("EofWhileParsing(value)" if self.pos >= len(self.text) else "InvalidValue")

    def constant(self):
        self.spaces()
        n, sign_ok, specials_ok, suffix_ok, saw_val = 0, True, True, True, False
        while True:
            b = self.peek(n)
            if b is None:
                break
            if "0" <= b <= "9" or b == ".":
                saw_val, sign_ok, specials_ok, n = True, False, False, n + 1
            elif sign_ok and b in ("+", "-"):
                sign_ok, n = False, n + 1
            elif saw_val and b in ("e", "E"):
                suffix_ok, sign_ok, n = False, True, n + 1
            elif specials_ok and self.text[self.pos + n : self.pos + n + 3] in ("NaN", "inf"):
                saw_val, suffix_ok, n = True, False, n + 3
                break
            else:
                break
        if not saw_val:
            return None
        token = self.text[self.pos : self.pos + n]
        if suffix_ok and self.peek(n) in _SUFFIXES:
            token, n = token + "e" + str(_SUFFIXES[self.peek(n)]), n + 1
        try:
            value = float(token)
        except ValueError:
            self.fail(f"ParseF64({token!r})")
        self.pos += n
        return value

    def unary(self, depth: int):
        self.spaces()
        b = self.peek()
        if b in ("+", "-", "!"):
            self.pos += 1
            v = self.value(depth + 1)
            self.vals += 1
            if self.vals > _SLAB_CAPACITY:
                self.fail("SlabOverflow")
            return ({"+": "pos", "-": "neg", "!": "not"}[b], v)
        if b in ("(", "["):
            self.pos += 1
            e = self.expression(depth + 1, False)
            self.spaces()
            close = ")" if b == "(" else "]"
            if self.peek() is None:
                self.fail("EofWhileParsing(parentheses)")
            if self.peek() != close:
                self.fail(f"Expected({close!r})")
            self.pos += 1
            return ("paren", e)
        return None

    def binary_op(self):
        self.spaces()
        b, b1 = self.peek(), self.peek(1)
        if b in ("+", "-", "*", "/", "%", "^"):
            self.pos += 1
            return b
        if b in ("<", ">"):
            self.pos += 2 if b1 == "=" else 1
            return b + "=" if b1 == "=" else b
        two = {("=", "="): "==", ("!", "="): "!=", ("o", "r"): "||", ("|", "|"): "||", ("&", "&"): "&&"}.get((b, b1))
        if two is not None:
            self.pos += 2
            return two
        if (b, b1, self.peek(2)) == ("a", "n", "d"):
            self.pos += 3
            return "&&"
        return None

    def callable(self, depth: int):
        self.spaces()
        n = 0
        while True:
            b = self.peek(n)
            if b is None or not (b == "_" or "a" <= b <= "z" or "A" <= b <= "Z" or (n > 0 and "0" <= b <= "9")):
                break
            n += 1
        if n == 0:
            return None
        name = self.text[self.pos : self.pos + n]
        self.pos += n
        self.spaces()
        opener = self.peek()
        if opener not in ("(", "["):
            return ("var", name)
        self.pos += 1
        if name == "print":
            self.fail("print() is not supported")
        close, args = ")" if opener == "(" else "]", []
        while True:
            self.spaces()
            b = self.peek()
            if b is None:
                self.fail(f"EofWhileParsing({name})")
            if b == close:
                self.pos += 1
                break
            if args:
                self.pos += 1
                if b not in (",", ";"):
                    self.fail("Expected(',' or ';')")
            args.append(self.expression(depth + 1, False))
        if name in _ARITY and not _ARITY[name][0] <= len(args) <= _ARITY[name][1]:
            self.fail(f"WrongArgs({name})")
        return ("func", name, args)


def _split(first, pairs, ops):
    slices, used = [(first, [])], []
    for op, val in pairs:
        if op in ops:
            slices.append((val, []))
            used.append(op)
        else:
            slices[-1][1].append((op, val))
    return slices, used


def _neg(instr):
    return ("C", -instr[1]) if instr[0] == "C" else instr[1] if instr[0] == "neg" else ("neg", instr)


def _inv(instr):
    return ("C", _div(1.0, instr[1])) if instr[0] == "C" else instr[1] if instr[0] == "inv" else ("inv", instr)


def _not(instr):
    return ("C", 1.0 if _eq(instr[1], 0.0) else 0.0) if instr[0] == "C" else instr[1] if instr[0] == "not" else ("not", instr)


def _leaves(node, out) -> None:
    """compiler.rs's `push_add_leaves`/`push_mul_leaves`: a nested chain flattened, right operand first."""
    tag, items, const = node
    if const is not None:
        out.append(("C", const))
    for item in list(reversed(items[1:])) + [items[0]]:
        if item[0] == tag:
            _leaves(item, out)
        else:
            out.append(item)


def _fold(tag, instrs):
    """`compile_add`/`compile_mul`: the constants summed (multiplied) in order into one, dropped when within
    8 ε of the identity (so a NaN constant vanishes and `0/0` is 1), the rest chained left to right."""
    items, total = [], 0.0 if tag == "add" else 1.0
    for instr in instrs:
        if instr[0] == "C":
            total = total + instr[1] if tag == "add" else total * instr[1]
        else:
            items.append(instr)
    if _ne(total, 0.0 if tag == "add" else 1.0):
        return (tag, items, total) if items else ("C", total)
    if not items:
        return ("C", 0.0 if tag == "add" else 1.0)
    return items[0] if len(items) == 1 else (tag, items, None)


def _chain(tag, out, instr):
    return (tag, out[1] + [instr]) if out[0] == tag else (tag, [out, instr])


def _compile_slice(first, pairs):
    """compiler.rs: the lowest-precedence operator of the chain splits it; constants fold at each level."""
    if not pairs:
        return _compile_value(first)
    lowest = min(_PRECEDENCE[op] for op, _ in pairs)
    if 3 <= lowest <= 8:
        slices, ops = _split(first, pairs, _COMPARISONS)
        out = _compile_slice(*slices[0])
        for op, s in zip(ops, slices[1:]):
            instr = _compile_slice(*s)
            if out[0] == "C" and instr[0] == "C":
                out = ("C", _compare(op, out[1], instr[1]))
            elif out[0] == "cmp":
                out = ("cmp", out[1], out[2] + [(op, instr)])
            else:
                out = ("cmp", out, [(op, instr)])
        return out
    op = {1: "||", 2: "&&", 9: "+", 10: "-", 11: "*", 12: "/", 13: "%", 14: "^"}[lowest]
    slices = _split(first, pairs, (op,))[0]
    if op == "||":
        out, out_set = ("C", 0.0), False
        for s in slices:
            instr = _compile_slice(*s)
            if out_set:
                out = _chain("or", out, instr)
            elif instr[0] == "C":
                if _ne(instr[1], 0.0):
                    return instr
            else:
                out, out_set = instr, True
        return out
    if op == "&&":
        out, out_set = ("C", 1.0), False
        for s in slices:
            instr = _compile_slice(*s)
            if instr[0] == "C" and _eq(instr[1], 0.0):
                return instr
            if out_set:
                out = instr if out[0] == "C" else _chain("and", out, instr)
            else:
                out, out_set = instr, True
        return out
    if op in ("+", "*"):
        tag, instrs = "add" if op == "+" else "mul", []
        for s in slices:
            instr = _compile_slice(*s)
            if instr[0] == tag:
                _leaves(instr, instrs)
            else:
                instrs.append(instr)
        return _fold(tag, instrs)
    if op in ("-", "/"):
        wrap = _neg if op == "-" else _inv
        return _fold("add" if op == "-" else "mul", [_compile_slice(*slices[0])] + [wrap(_compile_slice(*s)) for s in slices[1:]])
    if op == "%":
        out = None
        for s in slices:
            instr = _compile_slice(*s)
            out = instr if out is None else ("C", _fmod(out[1], instr[1])) if out[0] == "C" == instr[0] else ("mod", out, instr)
        return out
    out = None
    for s in reversed(slices):  # `^` is right-associative
        instr = _compile_slice(*s)
        out = instr if out is None else ("C", _powf(instr[1], out[1])) if out[0] == "C" == instr[0] else ("exp", instr, out)
    return out


def _compile_value(v):
    tag = v[0]
    if tag == "const":
        return ("C", v[1])
    if tag == "var":
        return v
    if tag == "pos":
        return _compile_value(v[1])
    if tag in ("neg", "not"):
        return (_neg if tag == "neg" else _not)(_compile_value(v[1]))
    if tag == "paren":
        return _compile_slice(*v[1])
    name, args = v[1], [_compile_slice(*a) for a in v[2]]
    if name in _UNARY_FUNCTIONS:
        return ("C", _UNARY_FUNCTIONS[name](args[0][1])) if args[0][0] == "C" else ("fn", name, args[0])
    if name in ("log", "round"):
        default, x = (("C", 10.0) if name == "log" else ("C", 1.0)), args[-1]
        base = args[0] if len(args) == 2 else default
        if base[0] == "C" == x[0]:
            return ("C", _log(base[1], x[1]) if name == "log" else _round_half_away(_div(x[1], base[1])) * base[1])
        return (name, base, x)
    if name in ("min", "max"):
        better = (lambda a, b: a < b) if name == "min" else (lambda a, b: a > b)
        out, best = (None, args[0][1]) if args[0][0] == "C" else (args[0], None)
        for instr in args[1:]:
            if instr[0] == "C":
                best = instr[1] if best is None or better(instr[1], best) else best
            else:
                out = instr if out is None else (name, out, instr)
        if best is not None:
            out = ("C", best) if out is None else (name, out, ("C", best))
        return out
    if name in ("e", "pi"):
        return ("C", math.e if name == "e" else math.pi)
    return ("call", name, args)


def _eval(node, ctx: Dict[str, float]) -> float:
    """evaler.rs on the compiled form; the chains iterate so a long formula needs no deep recursion."""
    tag = node[0]
    if tag == "C":
        return node[1]
    if tag in ("var", "call"):
        if node[1] in ctx and (tag == "var" or not node[2]):
            return float(ctx[node[1]])
        raise ValueError(f"unknown variable {node[1]!r}")
    if tag in ("add", "mul"):
        acc = _eval(node[1][0], ctx)
        for item in node[1][1:]:
            acc = acc + _eval(item, ctx) if tag == "add" else acc * _eval(item, ctx)
        if node[2] is not None:
            acc = acc + node[2] if tag == "add" else acc * node[2]
        return acc
    if tag == "neg":
        return -_eval(node[1], ctx)
    if tag == "inv":
        return _div(1.0, _eval(node[1], ctx))
    if tag == "not":
        return 1.0 if _eq(_eval(node[1], ctx), 0.0) else 0.0
    if tag in ("mod", "exp"):
        outer = []
        while node[0] == tag:
            outer.append(node[2] if tag == "mod" else node[1])
            node = node[1] if tag == "mod" else node[2]
        acc = _eval(node, ctx)
        for other in reversed(outer):
            acc = _fmod(acc, _eval(other, ctx)) if tag == "mod" else _powf(_eval(other, ctx), acc)
        return acc
    if tag == "cmp":
        acc = _eval(node[1], ctx)
        for op, x in node[2]:
            acc = _compare(op, acc, _eval(x, ctx))
        return acc
    if tag in ("or", "and"):
        for item in node[1]:
            v = _eval(item, ctx)
            if (_ne if tag == "or" else _eq)(v, 0.0):
                return v
        return v
    if tag == "fn":
        return _UNARY_FUNCTIONS[node[1]](_eval(node[2], ctx))
    if tag == "log":
        return _log(_eval(node[1], ctx), _eval(node[2], ctx))
    if tag == "round":
        m = _eval(node[1], ctx)
        return _round_half_away(_div(_eval(node[2], ctx), m)) * m
    l, r = _eval(node[1], ctx), _eval(node[2], ctx)  # min, max
    if l != l or r != r:
        return math.nan
    return l if (l < r if tag == "min" else l > r) else r


@functools.lru_cache(maxsize=256)
def _compile_expression(expression: str):
    if not isinstance(expression, str):
        raise ValueError(f"a formula is a string, not {expression!r}")
    if len(expression.encode("utf-8")) > _EXPR_LEN_LIMIT:
        raise ValueError(f"cannot evaluate expression {expression[:40]!r}...: TooLong")
    return _compile_slice(*_Formula(expression).expression(0, True))


def evaluate_expression(expression: str, context: Dict[str, float]) -> float:
    """A QEC scheme's or distillation unit's formula evaluated as the QDK's `fasteval` evaluates it. The grammar
    is fasteval 0.2.4's: `+ - * / % ^`, each its own precedence level from lowest to highest and `^`
    right-associative (so `7 * 3 % 2` is 7 and `2^3^2` is 512); unary minus binds tighter than every binary
    operator (`-2^2` is 4, `-(codeDistance^2)` at distance 3 is −9); comparisons, `and`/`&&`, `or`/`||`, `!`;
    the functions min, max, abs, log, int, ceil, floor, round, sign, the trigonometric and hyperbolic ones,
    pi(), e(); `inf` and `NaN`. Constants fold at compile time as fasteval folds them (every constant of a
    `+`/`*` chain is combined first, so `codeDistance + 0.1 + 0.2` is 3.3 and a NaN constant vanishes).
    Division by zero is infinite, a negative base to a fractional power NaN. Formulas over 4096 bytes or nested
    over 32 deep are refused, as are Python's own spellings (`**`, `1_000`, `0x10`)."""
    try:
        return _eval(_compile_expression(expression), context)
    except ValueError as e:
        raise ValueError(f"{e} in {expression!r}") if str(e).startswith("unknown variable") else e


_TIME_RE = re.compile(r"^(\+?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)[ \t\n\r\f\v]*(s|ms|μs|µs|us|ns)$")


def parse_time(value: Union[int, float, str]) -> int:
    """A duration in nanoseconds: a string such as ``"100 ns"``, ``"1 µs"``, ``"2 ms"``, ``"1 s"`` (the QDK's
    form, no surrounding spaces, the value rounded and saturated to u64 as the QDK's), or, beyond the QDK, a
    number of nanoseconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _as_u64(_round_half_away(float(value)))
    m = _TIME_RE.match(value) if isinstance(value, str) else None
    if m is None:
        raise ValueError(f"cannot parse time {value!r}")
    multiplier = {"s": 1e9, "ms": 1e6, "μs": 1e3, "µs": 1e3, "us": 1e3, "ns": 1.0}[m.group(3)]
    return _as_u64(_round_half_away(float(m.group(1)) * multiplier))


# ---------------------------------------------------------------------------------------------
# Binomial quantile of the `probability` crate (distribution/binomial.rs, gaussian.rs) and the incomplete beta
# function of the `special` crate (beta.rs, Algorithm AS 63)
# ---------------------------------------------------------------------------------------------


def _ln_beta(p: float, q: float) -> float:
    return math.lgamma(p) + math.lgamma(q) - math.lgamma(p + q)


def _inc_beta(x: float, p: float, q: float, ln_beta: float) -> float:
    acu = 0.1e-14
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    psq = p + q
    flip = p < psq * x
    if flip:
        pbase, qbase, p, q = 1.0 - x, x, q, p
    else:
        pbase, qbase = x, 1.0 - x
    term = 1.0
    ai = 1.0
    ns = int(q + qbase * psq)
    rx = pbase if ns == 0 else pbase / qbase
    a = 1.0
    temp = q - ai
    while True:
        term = term * temp * rx / (p + ai)
        a += term
        temp = -term if term < 0.0 else term
        if temp <= acu and temp <= acu * a:
            break
        ai += 1.0
        ns -= 1
        if 0 < ns:
            temp = q - ai
        elif ns == 0:
            temp = q - ai
            rx = pbase
        else:
            temp = psq
            psq += 1.0
    a = a * math.exp(p * math.log(pbase) + (q - 1.0) * math.log(qbase) - ln_beta) / p
    return 1.0 - a if flip else a


def _gaussian_inverse(p: float) -> float:
    """Wichura's AS 241 normal quantile, as `probability::distribution::gaussian::inverse`."""
    A = (3.3871328727963666080e+00, 1.3314166789178437745e+02, 1.9715909503065514427e+03, 1.3731693765509461125e+04, 4.5921953931549871457e+04, 6.7265770927008700853e+04, 3.3430575583588128105e+04, 2.5090809287301226727e+03)
    B = (1.0, 4.2313330701600911252e+01, 6.8718700749205790830e+02, 5.3941960214247511077e+03, 2.1213794301586595867e+04, 3.9307895800092710610e+04, 2.8729085735721942674e+04, 5.2264952788528545610e+03)
    C = (1.42343711074968357734e+00, 4.63033784615654529590e+00, 5.76949722146069140550e+00, 3.64784832476320460504e+00, 1.27045825245236838258e+00, 2.41780725177450611770e-01, 2.27238449892691845833e-02, 7.74545014278341407640e-04)
    D = (1.0, 2.05319162663775882187e+00, 1.67638483018380384940e+00, 6.89767334985100004550e-01, 1.48103976427480074590e-01, 1.51986665636164571966e-02, 5.47593808499534494600e-04, 1.05075007164441684324e-09)
    E = (6.65790464350110377720e+00, 5.46378491116411436990e+00, 1.78482653991729133580e+00, 2.96560571828504891230e-01, 2.65321895265761230930e-02, 1.24266094738807843860e-03, 2.71155556874348757815e-05, 2.01033439929228813265e-07)
    F = (1.0, 5.99832206555887937690e-01, 1.36929880922735805310e-01, 1.48753612908506148525e-02, 7.86869131145613259100e-04, 1.84631831751005468180e-05, 1.42151175831644588870e-07, 2.04426310338993978564e-15)

    def poly(c, x):
        return c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * (c[6] + x * c[7]))))))

    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    q = p - 0.5
    if abs(q) <= 0.425:
        x = 0.180625 - q * q
        return q * poly(A, x) / poly(B, x)
    x = p if q < 0.0 else 1.0 - p
    x = math.sqrt(-math.log(x))
    if x <= 5.0:
        x -= 1.6
        x = poly(C, x) / poly(D, x)
    else:
        x -= 5.0
        x = poly(E, x) / poly(F, x)
    return -x if q < 0.0 else x


_SFE = (0.0, 8.106146679532725822e-02, 4.134069595540929409e-02, 2.767792568499833915e-02, 2.079067210376509311e-02, 1.664469118982119216e-02, 1.387612882307074800e-02, 1.189670994589177010e-02, 1.041126526197209650e-02, 9.255462182712732918e-03, 8.330563433362871256e-03, 7.757367548795184079e-03, 6.942840107209529866e-03, 6.408994188004207068e-03, 5.951370112758847736e-03, 5.554733551962801371e-03)


def _stirlerr(n: float) -> float:
    S0, S1, S2, S3, S4 = 1.0 / 12.0, 1.0 / 360.0, 1.0 / 1260.0, 1.0 / 1680.0, 1.0 / 1188.0
    if n < 16.0:
        return _SFE[int(n)]
    nn = n * n
    if n > 500.0:
        return (S0 - S1 / nn) / n
    if n > 80.0:
        return (S0 - (S1 - S2 / nn) / nn) / n
    if n > 35.0:
        return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n
    return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n


def _ln_d0(x: float, np_: float) -> float:
    if abs(x - np_) < 0.1 * (x + np_):
        s = (x - np_) ** 2 / (x + np_)
        v = (x - np_) / (x + np_)
        ej = 2.0 * x * v
        j = 1
        while True:
            ej *= v * v
            s1 = s + ej / (2 * j + 1)
            if s1 == s:
                return s1
            s = s1
            j += 1
    return x * math.log(x / np_) + np_ - x


class _Binomial:
    """`Binomial::with_failure(n, q)`: n trials, each failing with probability q."""

    def __init__(self, n: int, q: float):
        self.n = n
        self.q = q
        self.p = 1.0 - q
        self.np = n * self.p
        self.nq = n * q
        self.npq = self.np * q

    def mass(self, x: int) -> float:
        if self.p == 0.0:
            return 1.0 if x == 0 else 0.0
        if self.p == 1.0:
            return 1.0 if x == self.n else 0.0
        n = float(self.n)
        if x == 0:
            return math.exp(n * math.log(self.q))
        if x == self.n:
            return math.exp(n * math.log(self.p))
        xf = float(x)
        n_m_x = float((self.n - x) % 2**64)  # the crate's `n - x` in usize, which wraps past n (and the QDK asks past n)
        try:
            ln_c = _stirlerr(n) - _stirlerr(xf) - _stirlerr(n_m_x) - _ln_d0(xf, self.np) - _ln_d0(n_m_x, self.nq)
            return math.exp(ln_c) * math.sqrt(n / (2.0 * math.pi * xf * n_m_x))
        except (ValueError, OverflowError, ZeroDivisionError):  # not reached for 0 < x; f64 would give NaN or an infinity
            return math.nan

    def distribution(self, x: float) -> float:
        if x < 0.0:
            return 0.0
        x = _as_u64(x)  # `x as usize`: saturating, NaN to 0
        if x == 0:
            return _powi(self.q, self.n)
        if x >= self.n:
            return 1.0
        p, q = float(self.n - x), float(x + 1)
        return _inc_beta(self.q, p, q, _ln_beta(p, q))

    def modes(self) -> List[int]:
        r = self.p * (self.n + 1)
        if r == 0.0:
            return [0]
        if self.p == 1.0:
            return [self.n]
        if r != math.trunc(r):
            return [int(math.floor(r))]
        return [int(r) - 1, int(r)]

    def inverse(self, p: float) -> int:
        """The quantile: how many successes are guaranteed with probability 1 − p."""
        if p == 0.0:
            return 0
        if p == 1.0:
            return self.n
        n = self.n
        if n < 1000:
            if p <= self.distribution(float(n // 2)):
                k = 1
                a = _powi(self.q, n)
                s = a - p
                while s < 0.0:
                    a *= self.p / self.q * ((n - k + 1) / k)
                    s += a
                    k += 1
                return k - 1
            k = 1
            a = _powi(self.p, n)
            s = (1.0 - p) - a
            while s >= 0.0:
                a *= self.q / self.p * ((n - k + 1) / k)
                s -= a
                k += 1
            return n - k + 1
        if self.npq > 80.0:
            return int(math.floor(_inverse_normal(self.p, self.np, self.npq, p)))
        # The crate's damped Newton iteration from the mode, with its casts: `q as usize` saturates (so a step past
        # n reads the CDF as 1 and the mass with a wrapped n − x, and a step below 0 reads them at 0), and the
        # loop has no cap. It converges in a few steps even with the failure probability near 1 (q falls to 0); it
        # never returns once q is NaN or infinite (the mass is 0 there, so every step is infinite), which is where
        # the QDK hangs: that, or an unreasonable number of steps, is an error here.
        q = float(self.modes()[0])
        alpha = 1.0
        for _ in range(1_000_000):
            numerator = alpha * (p - self.distribution(q))
            mass = self.mass(_as_u64(q))
            delta = _div(numerator, mass)
            if abs(delta) < 0.5:
                return _as_u64(q)
            q += delta
            alpha *= 0.999
            if not math.isfinite(q):
                break
        raise ValueError(f"the QDK's binomial quantile does not converge for n={n}, failure probability {self.q!r}, p={p!r}")


def _inverse_normal(p: float, np_: float, v: float, u: float) -> float:
    """The Cornish–Fisher-style binomial quantile of the `probability` crate for large n·p·q."""
    w = _gaussian_inverse(u)
    w2 = w * w
    w3 = w2 * w
    w4 = w3 * w
    w5 = w4 * w
    w6 = w5 * w
    sd = math.sqrt(v)
    sd_em1 = 1.0 / sd
    sd_em2 = 1.0 / v
    sd_em3 = sd_em1 * sd_em2
    sd_em4 = sd_em2 * sd_em2
    p2 = p * p
    p3 = p2 * p
    p4 = p2 * p2
    return (
        np_
        + sd * w
        + (p + 1.0) / 3.0
        - (2.0 * p - 1.0) * w2 / 6.0
        + sd_em1 * w3 * (2.0 * p2 - 2.0 * p - 1.0) / 72.0
        - w * (7.0 * p2 - 7.0 * p + 1.0) / 36.0
        + sd_em2 * (2.0 * p - 1.0) * (p + 1.0) * (p - 2.0) * (3.0 * w4 + 7.0 * w2 - 16.0 / 1620.0)
        + sd_em3 * (w5 * (4.0 * p4 - 8.0 * p3 - 48.0 * p2 + 52.0 * p - 23.0) / 17280.0 + w3 * (256.0 * p4 - 512.0 * p3 - 147.0 * p2 + 403.0 * p - 137.0) / 38880.0 - w * (433.0 * p4 - 866.0 * p3 - 921.0 * p2 + 1354.0 * p - 671.0) / 38880.0)
        + sd_em4 * (w6 * (2.0 * p - 1.0) * (p2 - p + 1.0) * (p2 - p + 19.0) / 34020.0 + w4 * (2.0 * p - 1.0) * (9.0 * p4 - 18.0 * p3 - 35.0 * p2 + 44.0 * p - 25.0) / 15120.0 + w2 * (2.0 * p - 1.0) * (923.0 * p4 - 1846.0 * p3 + 5271.0 * p2 - 4348.0 * p + 5189.0) / 408240.0 - 4.0 * (2.0 * p - 1.0) * (p + 1.0) * (p - 2.0) * (23.0 * p2 - 23.0 * p + 2.0) / 25515.0)
    )


# ---------------------------------------------------------------------------------------------
# Physical qubits (system/modeling/physical_qubit.rs)
# ---------------------------------------------------------------------------------------------

_NAN = float("nan")


def _is_nan(x: float) -> bool:
    return x != x


@dataclass
class GateBasedQubit:
    """A gate-based qubit model: times in nanoseconds, error rates per operation. Missing times and rates are
    filled in as the QDK does: a known `name` supplies its preset, then the two-qubit gate and T gate inherit
    the one-qubit gate's time and error, and idling inherits the measurement error."""

    name: str = ""
    one_qubit_measurement_time: Optional[int] = None
    one_qubit_gate_time: Optional[int] = None
    two_qubit_gate_time: Optional[int] = None
    t_gate_time: Optional[int] = None
    one_qubit_measurement_error_rate: float = _NAN
    one_qubit_gate_error_rate: float = _NAN
    two_qubit_gate_error_rate: float = _NAN
    t_gate_error_rate: float = _NAN
    idle_error_rate: float = _NAN
    instruction_set: str = field(default="GateBased", init=False)

    def normalized(self) -> "GateBasedQubit":
        q = replace(self)
        base = _GATE_BASED_PRESETS.get(q.name)
        if base is not None:
            for f in ("one_qubit_measurement_time", "one_qubit_gate_time", "two_qubit_gate_time", "t_gate_time"):
                if getattr(q, f) is None:
                    setattr(q, f, getattr(base, f))
            for f in ("one_qubit_measurement_error_rate", "one_qubit_gate_error_rate", "two_qubit_gate_error_rate", "t_gate_error_rate", "idle_error_rate"):
                if _is_nan(getattr(q, f)):
                    setattr(q, f, getattr(base, f))
        missing = [n for ok, n in ((q.one_qubit_measurement_time is not None, "oneQubitMeasurementTime"), (not _is_nan(q.t_gate_error_rate), "tGateErrorRate"), (not _is_nan(q.one_qubit_measurement_error_rate), "oneQubitMeasurementErrorRate"), (q.one_qubit_gate_time is not None, "oneQubitGateTime"), (not _is_nan(q.one_qubit_gate_error_rate), "oneQubitGateErrorRate")) if not ok]
        if missing:
            raise ValueError("missing qubit parameters: " + ", ".join(f"`{m}`" for m in missing))
        if q.two_qubit_gate_time is None:
            q.two_qubit_gate_time = q.one_qubit_gate_time
        if q.t_gate_time is None:
            q.t_gate_time = q.one_qubit_gate_time
        if _is_nan(q.two_qubit_gate_error_rate):
            q.two_qubit_gate_error_rate = q.one_qubit_gate_error_rate
        if _is_nan(q.t_gate_error_rate):
            q.t_gate_error_rate = q.one_qubit_gate_error_rate
        if _is_nan(q.idle_error_rate):
            q.idle_error_rate = q.one_qubit_measurement_error_rate
        return q

    def clifford_error_rate(self) -> float:
        return max(self.one_qubit_gate_error_rate, self.two_qubit_gate_error_rate, self.idle_error_rate)

    def readout_error_rate(self) -> float:
        return self.one_qubit_measurement_error_rate

    def expression_context(self, code_distance: int) -> Dict[str, float]:
        return {"oneQubitMeasurementTime": self.one_qubit_measurement_time, "oneQubitGateTime": self.one_qubit_gate_time, "twoQubitGateTime": self.two_qubit_gate_time, "codeDistance": code_distance}

    def asdict(self) -> dict:
        return {"instructionSet": "GateBased", "name": self.name, "oneQubitMeasurementTime": f"{self.one_qubit_measurement_time} ns", "oneQubitGateTime": f"{self.one_qubit_gate_time} ns", "twoQubitGateTime": f"{self.two_qubit_gate_time} ns", "tGateTime": f"{self.t_gate_time} ns", "oneQubitMeasurementErrorRate": self.one_qubit_measurement_error_rate, "oneQubitGateErrorRate": self.one_qubit_gate_error_rate, "twoQubitGateErrorRate": self.two_qubit_gate_error_rate, "tGateErrorRate": self.t_gate_error_rate, "idleErrorRate": self.idle_error_rate}


@dataclass
class MajoranaQubit:
    """A measurement-based (Majorana) qubit model. Measurement error rates are a single number or a
    ``(process, readout)`` pair; the two-qubit joint measurement and idling inherit the one-qubit readout."""

    name: str = ""
    one_qubit_measurement_time: Optional[int] = None
    two_qubit_joint_measurement_time: Optional[int] = None
    t_gate_time: Optional[int] = None
    one_qubit_measurement_error_rate: Union[float, Tuple[float, float]] = _NAN
    two_qubit_joint_measurement_error_rate: Union[float, Tuple[float, float]] = _NAN
    t_gate_error_rate: float = _NAN
    idle_error_rate: float = _NAN
    instruction_set: str = field(default="Majorana", init=False)

    @staticmethod
    def _rate_is_nan(v) -> bool:
        return _is_nan(v) if isinstance(v, float) else (_is_nan(v[0]) and _is_nan(v[1]))

    def normalized(self) -> "MajoranaQubit":
        q = replace(self)
        base = _MAJORANA_PRESETS.get(q.name)
        if base is not None:
            for f in ("one_qubit_measurement_time", "two_qubit_joint_measurement_time", "t_gate_time"):
                if getattr(q, f) is None:
                    setattr(q, f, getattr(base, f))
            for f in ("one_qubit_measurement_error_rate", "two_qubit_joint_measurement_error_rate"):
                if self._rate_is_nan(getattr(q, f)):
                    setattr(q, f, getattr(base, f))
            for f in ("t_gate_error_rate", "idle_error_rate"):
                if _is_nan(getattr(q, f)):
                    setattr(q, f, getattr(base, f))
        missing = [n for ok, n in ((q.one_qubit_measurement_time is not None, "oneQubitMeasurementTime"), (not self._rate_is_nan(q.one_qubit_measurement_error_rate), "oneQubitMeasurementErrorRate"), (not _is_nan(q.t_gate_error_rate), "tGateErrorRate")) if not ok]
        if missing:
            raise ValueError("missing qubit parameters: " + ", ".join(f"`{m}`" for m in missing))
        one = q.one_qubit_measurement_error_rate
        one = (one, one) if isinstance(one, float) else tuple(one)
        two = q.two_qubit_joint_measurement_error_rate
        two = (two, two) if isinstance(two, float) else tuple(two)
        if q.two_qubit_joint_measurement_time is None:
            q.two_qubit_joint_measurement_time = q.one_qubit_measurement_time
        if q.t_gate_time is None:
            q.t_gate_time = q.one_qubit_measurement_time
        two = (one[1] if _is_nan(two[0]) else two[0], one[1] if _is_nan(two[1]) else two[1])
        if _is_nan(q.idle_error_rate):
            q.idle_error_rate = one[1]
        q.one_qubit_measurement_error_rate = one
        q.two_qubit_joint_measurement_error_rate = two
        return q

    def clifford_error_rate(self) -> float:
        return max(self.idle_error_rate, self.one_qubit_measurement_error_rate[0], self.two_qubit_joint_measurement_error_rate[0])

    def readout_error_rate(self) -> float:
        return max(self.one_qubit_measurement_error_rate[1], self.two_qubit_joint_measurement_error_rate[1])

    def expression_context(self, code_distance: int) -> Dict[str, float]:
        return {"oneQubitMeasurementTime": self.one_qubit_measurement_time, "twoQubitJointMeasurementTime": self.two_qubit_joint_measurement_time, "codeDistance": code_distance}

    def asdict(self) -> dict:
        one, two = self.one_qubit_measurement_error_rate, self.two_qubit_joint_measurement_error_rate
        return {"instructionSet": "Majorana", "name": self.name, "oneQubitMeasurementTime": f"{self.one_qubit_measurement_time} ns", "twoQubitJointMeasurementTime": f"{self.two_qubit_joint_measurement_time} ns", "tGateTime": f"{self.t_gate_time} ns", "oneQubitMeasurementErrorRate": {"process": one[0], "readout": one[1]}, "twoQubitJointMeasurementErrorRate": {"process": two[0], "readout": two[1]}, "tGateErrorRate": self.t_gate_error_rate, "idleErrorRate": self.idle_error_rate}


PhysicalQubit = Union[GateBasedQubit, MajoranaQubit]

_GATE_BASED_PRESETS = {
    "qubit_gate_ns_e3": GateBasedQubit("qubit_gate_ns_e3", 100, 50, 50, 50, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3),
    "qubit_gate_ns_e4": GateBasedQubit("qubit_gate_ns_e4", 100, 50, 50, 50, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4),
    "qubit_gate_us_e3": GateBasedQubit("qubit_gate_us_e3", 100_000, 100_000, 100_000, 100_000, 1e-3, 1e-3, 1e-3, 1e-6, 1e-3),
    "qubit_gate_us_e4": GateBasedQubit("qubit_gate_us_e4", 100_000, 100_000, 100_000, 100_000, 1e-4, 1e-4, 1e-4, 1e-6, 1e-4),
}
_MAJORANA_PRESETS = {
    "qubit_maj_ns_e4": MajoranaQubit("qubit_maj_ns_e4", 100, 100, 100, (1e-4, 1e-4), (1e-4, 1e-4), 0.05, 1e-4),
    "qubit_maj_ns_e6": MajoranaQubit("qubit_maj_ns_e6", 100, 100, 100, (1e-6, 1e-6), (1e-6, 1e-6), 0.01, 1e-6),
}
QUBIT_NAMES = tuple(_GATE_BASED_PRESETS) + tuple(_MAJORANA_PRESETS)

_SNAKE = {"oneQubitMeasurementTime": "one_qubit_measurement_time", "oneQubitGateTime": "one_qubit_gate_time", "twoQubitGateTime": "two_qubit_gate_time", "twoQubitJointMeasurementTime": "two_qubit_joint_measurement_time", "tGateTime": "t_gate_time", "oneQubitMeasurementErrorRate": "one_qubit_measurement_error_rate", "oneQubitGateErrorRate": "one_qubit_gate_error_rate", "twoQubitGateErrorRate": "two_qubit_gate_error_rate", "twoQubitJointMeasurementErrorRate": "two_qubit_joint_measurement_error_rate", "tGateErrorRate": "t_gate_error_rate", "idleErrorRate": "idle_error_rate"}


def _error_rate(key: str, value: Any) -> float:
    """An error rate as the QDK's `deserialize_error_rate` takes it: a number strictly between 0 and 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) < 1.0:
        raise ValueError(f"{key}: expected value between 0.0 and 1.0, got {value!r}")
    return float(value)


def _count(key: str, value: Any, bits: int = 64) -> int:
    """A count as the QDK's `u64` (or `u32`) takes it: a non-negative integer below 2^bits (not a bool, not a
    float)."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**bits:
        raise ValueError(f"{key}: expected a non-negative integer below 2^{bits}, got {value!r}")
    return value


def _float_field(key: str, value: Any) -> float:
    """A number as the QDK's `f64` takes it: an int or a float, not a bool, not a string."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key}: expected a number, got {value!r}")
    return float(value)


def physical_qubit(spec: Union[str, dict, GateBasedQubit, MajoranaQubit, None] = None) -> PhysicalQubit:
    """A qubit model from a preset name (`QUBIT_NAMES`), a QDK ``qubitParams`` dict (camelCase keys, times as
    ints in ns or strings with units, Majorana measurement errors as numbers or ``{process, readout}``), or a
    model object; normalized."""
    if spec is None:
        spec = "qubit_gate_ns_e3"
    if isinstance(spec, (GateBasedQubit, MajoranaQubit)):
        return spec.normalized()
    if isinstance(spec, str):
        spec = {"name": spec}
    spec = dict(spec)
    instruction_set = spec.pop("instructionSet", None)
    name = spec.pop("name", "")
    if not isinstance(name, str):
        raise ValueError(f"the qubit's name is a string, not {name!r}")
    if instruction_set is None:
        if name in _GATE_BASED_PRESETS:
            instruction_set = "GateBased"
        elif name in _MAJORANA_PRESETS:
            instruction_set = "Majorana"
        else:
            raise ValueError(f"unknown qubit {name!r}: give `instructionSet` (GateBased or Majorana) with the parameters")
    kwargs = {}
    for key, value in spec.items():
        if key not in _SNAKE:
            raise ValueError(f"unknown qubit parameter {key!r}")
        if key.endswith("Time"):
            value = parse_time(value)
        elif isinstance(value, dict):
            # The QDK's Majorana measurement error: both parts, each a rate (other keys are ignored, as serde ignores them).
            if not {"process", "readout"} <= set(value):
                raise ValueError(f"{key} as a dict takes the keys 'process' and 'readout', not {sorted(value)}")
            value = (_error_rate(key + ".process", value["process"]), _error_rate(key + ".readout", value["readout"]))
        else:
            value = _error_rate(key, value)
        kwargs[_SNAKE[key]] = value
    if instruction_set in ("gate_based", "gateBased", "gate-based", "GateBased"):
        return GateBasedQubit(name=name, **kwargs).normalized()
    if instruction_set in ("Majorana", "majorana"):
        return MajoranaQubit(name=name, **kwargs).normalized()
    raise ValueError('expected "GateBased" or "Majorana" as value for instructionSet')


# ---------------------------------------------------------------------------------------------
# QEC schemes (system/modeling/fault_tolerance.rs, estimates/error_correction)
# ---------------------------------------------------------------------------------------------


@dataclass
class QECScheme:
    """A code family with a threshold law: logical error rate
    ``crossing_prefactor · d^distance_coefficient_power · (p / threshold)^((d+1)/2)``, a cycle-time formula and
    a qubit-count formula in the code distance."""

    name: str
    error_correction_threshold: float
    crossing_prefactor: float
    distance_coefficient_power: int = 0
    logical_cycle_time: str = ""
    physical_qubits_per_logical_qubit: str = ""
    max_code_distance: int = MAX_CODE_DISTANCE

    @staticmethod
    def surface_code_gate_based() -> "QECScheme":
        return QECScheme("surface_code", 0.01, 0.03, 0, "(4 * twoQubitGateTime + 2 * oneQubitMeasurementTime) * codeDistance", "2 * codeDistance * codeDistance")

    @staticmethod
    def surface_code_measurement_based() -> "QECScheme":
        return QECScheme("surface_code", 0.0015, 0.08, 0, "20 * oneQubitMeasurementTime * codeDistance", "2 * codeDistance * codeDistance")

    @staticmethod
    def floquet_code() -> "QECScheme":
        return QECScheme("floquet_code", 0.01, 0.07, 0, "3 * oneQubitMeasurementTime * codeDistance", "4 * codeDistance * codeDistance + 8 * (codeDistance - 1)")

    def physical_error_rate(self, qubit: PhysicalQubit) -> float:
        return max(qubit.clifford_error_rate(), qubit.readout_error_rate())

    def physical_qubits(self, code_distance: int) -> int:
        """The formula's value `as u64`, as the QDK casts it: a NaN passes its non-positive check and becomes 0,
        an infinity 2^64 − 1."""
        value = evaluate_expression(self.physical_qubits_per_logical_qubit, {"codeDistance": code_distance})
        if value <= 0.0:
            raise ValueError(f"non-positive number of physical qubits per logical qubit at code distance {code_distance}")
        return _as_u64(value)

    def cycle_time(self, qubit: PhysicalQubit, code_distance: int) -> int:
        value = evaluate_expression(self.logical_cycle_time, qubit.expression_context(code_distance))
        if value <= 0.0:
            raise ValueError(f"non-positive logical cycle time at code distance {code_distance}")
        return _as_u64(_round_half_away(value))

    def logical_error_rate(self, qubit: PhysicalQubit, code_distance: int) -> float:
        """The QDK raises the distance to the coefficient power in u64 arithmetic, which wraps (50^12 already
        overflows); `pow(d, k, 2^64)` is that wrap, and cheap for any `k < 2^32`."""
        p = self.physical_error_rate(qubit)
        if p > self.error_correction_threshold:
            raise ValueError(f"invalid value for 'physical_error_rate', expected value between 0 and {self.error_correction_threshold}")
        return self.crossing_prefactor * float(pow(code_distance, self.distance_coefficient_power, 2**64)) * _powi(p / self.error_correction_threshold, (code_distance + 1) // 2)

    def code_distance_range(self, lower: Optional[int] = None) -> range:
        return range(lower if lower is not None else 1, self.max_code_distance + 1, 2)

    def compute_code_distance(self, qubit: PhysicalQubit, required_logical_error_rate: float) -> int:
        """The smallest odd distance reaching the required rate (the closed-form inverse when the law has no
        polynomial prefactor, as the QDK's)."""
        if self.distance_coefficient_power == 0:
            p = self.physical_error_rate(qubit)
            numerator = 2.0 * math.log(self.crossing_prefactor / required_logical_error_rate)
            denominator = math.log(self.error_correction_threshold / p)
            code_distance = max(0, math.ceil(numerator / denominator - 1.0)) | 1
            if code_distance > self.max_code_distance:
                raise ValueError(f"The computed code distance {code_distance} is too high; maximum allowed code distance is {self.max_code_distance}; try increasing the total logical error budget")
            return code_distance
        for d in self.code_distance_range():
            if self.logical_error_rate(qubit, d) <= required_logical_error_rate:
                return d
        raise ValueError(f"No code parameter achieves required logical error rate {required_logical_error_rate:.3e}")

    def asdict(self) -> dict:
        return {"name": self.name, "errorCorrectionThreshold": self.error_correction_threshold, "crossingPrefactor": self.crossing_prefactor, "distanceCoefficientPower": self.distance_coefficient_power, "logicalCycleTime": self.logical_cycle_time, "physicalQubitsPerLogicalQubit": self.physical_qubits_per_logical_qubit, "maxCodeDistance": self.max_code_distance}


_OPTIONAL_SCHEME_FIELDS = ("errorCorrectionThreshold", "crossingPrefactor", "distanceCoefficientPower", "logicalCycleTime", "physicalQubitsPerLogicalQubit")


def qec_scheme(spec: Union[str, dict, QECScheme, None], qubit: PhysicalQubit) -> QECScheme:
    """A scheme from a name (``surface_code``, ``floquet_code``; the surface code picks its gate-based or
    measurement-based variant from the qubit), a QDK ``qecScheme`` dict (a name with overrides, or a full custom
    scheme), or a `QECScheme`; validated against the qubit as the QDK does."""
    if spec is None:
        spec = {"name": "surface_code"}
    if isinstance(spec, str):
        spec = {"name": spec}
    if isinstance(spec, QECScheme):
        scheme = replace(spec)
    else:
        spec = {k: v for k, v in dict(spec).items() if not (v is None and k in _OPTIONAL_SCHEME_FIELDS)}  # null is absent for the QDK's `Option` fields (not for maxCodeDistance or name)
        name = spec.pop("name", "surface_code")
        if not isinstance(name, str):
            raise ValueError(f"the QEC scheme's name is a string, not {name!r}")
        for key in ("errorCorrectionThreshold", "crossingPrefactor"):
            if key in spec:
                spec[key] = _float_field(key, spec[key])
        for key in ("logicalCycleTime", "physicalQubitsPerLogicalQubit"):
            if key in spec and not isinstance(spec[key], str):
                raise ValueError(f"{key}: expected a formula string, got {spec[key]!r}")
        if "distanceCoefficientPower" in spec:
            spec["distanceCoefficientPower"] = _count("distanceCoefficientPower", spec["distanceCoefficientPower"], 32)
        if "maxCodeDistance" in spec:
            spec["maxCodeDistance"] = _count("maxCodeDistance", spec["maxCodeDistance"])
        majorana = isinstance(qubit, MajoranaQubit)
        if name in ("surface_code", "surfaceCode", "surface-code"):
            scheme = QECScheme.surface_code_measurement_based() if majorana else QECScheme.surface_code_gate_based()
        elif name in ("floquet_code", "floquetCode", "floquet-code"):
            if not majorana:
                scheme = None
            else:
                scheme = QECScheme.floquet_code()
        else:
            for required in ("errorCorrectionThreshold", "crossingPrefactor", "logicalCycleTime", "physicalQubitsPerLogicalQubit"):
                if required not in spec:
                    raise ValueError(f"missing field `{required}` of the QEC scheme {name!r}")
            scheme = QECScheme(name, spec["errorCorrectionThreshold"], spec["crossingPrefactor"], spec.get("distanceCoefficientPower", 0), spec["logicalCycleTime"], spec["physicalQubitsPerLogicalQubit"], spec.get("maxCodeDistance", MAX_CODE_DISTANCE))
        if scheme is None:
            raise ValueError("the Floquet code needs a Majorana (measurement-based) qubit")
        for key, attr in (("errorCorrectionThreshold", "error_correction_threshold"), ("crossingPrefactor", "crossing_prefactor"), ("distanceCoefficientPower", "distance_coefficient_power"), ("logicalCycleTime", "logical_cycle_time"), ("physicalQubitsPerLogicalQubit", "physical_qubits_per_logical_qubit"), ("maxCodeDistance", "max_code_distance")):
            if key in spec:
                setattr(scheme, attr, spec[key])
    if scheme.crossing_prefactor > 0.5:
        raise ValueError("invalid value for crossingPrefactor, expected value between 0 and 0.5")
    if qubit.clifford_error_rate() >= scheme.error_correction_threshold:
        raise ValueError(f"the qubit's Clifford error rate {qubit.clifford_error_rate()} is not below the scheme's threshold {scheme.error_correction_threshold}")
    for d in range(3, scheme.max_code_distance + 1):  # the QDK's `(1..=max).skip(2)`: even distances too
        scheme.cycle_time(qubit, d)
        scheme.physical_qubits(d)
    return scheme


@dataclass(frozen=True)
class LogicalPatch:
    """One logical qubit of a scheme at a code distance, on a qubit model."""

    scheme: QECScheme
    qubit: PhysicalQubit
    code_distance: int
    physical_qubits: int
    logical_cycle_time: int
    logical_error_rate: float
    logical_qubits: int = 1

    @staticmethod
    def new(scheme: QECScheme, qubit: PhysicalQubit, code_distance: int) -> "LogicalPatch":
        return LogicalPatch(scheme, qubit, code_distance, scheme.physical_qubits(code_distance), scheme.cycle_time(qubit, code_distance), scheme.logical_error_rate(qubit, code_distance))

    def logical_cycles_per_second(self) -> float:
        return 1e9 / self.logical_cycle_time

    def asdict(self) -> dict:
        return {"codeDistance": self.code_distance, "physicalQubits": self.physical_qubits, "logicalCycleTime": self.logical_cycle_time, "logicalErrorRate": self.logical_error_rate}


# ---------------------------------------------------------------------------------------------
# Logical counts and the error budget (system/data/logical_counts.rs, job_params.rs, estimates/error_budget.rs)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicalCounts:
    """The algorithm's logical resources, laid out with the QDK's PSSPC overhead: ``2Q + ⌈√(8Q)⌉ + 1`` logical
    qubits; one logical cycle per measurement, T and rotation, three per CCZ/CCiX, plus the rotation layers'
    synthesis T gates; a T state per T, four per CCZ/CCiX, ``⌈0.53·log₂(rotations/ε) + 4.86⌉`` per rotation."""

    num_qubits: int
    t_count: int = 0
    rotation_count: int = 0
    rotation_depth: int = 0
    ccz_count: int = 0
    ccix_count: int = 0
    measurement_count: int = 0
    # Counts the QDK takes and echoes back in `logicalCounts` but does not use in the estimate.
    num_compute_qubits: Optional[int] = None
    read_from_memory_count: Optional[int] = None
    write_to_memory_count: Optional[int] = None

    @staticmethod
    def fromdict(d: dict) -> "LogicalCounts":
        keys = {"numQubits": "num_qubits", "tCount": "t_count", "rotationCount": "rotation_count", "rotationDepth": "rotation_depth", "cczCount": "ccz_count", "ccixCount": "ccix_count", "measurementCount": "measurement_count", "numComputeQubits": "num_compute_qubits", "readFromMemoryCount": "read_from_memory_count", "writeToMemoryCount": "write_to_memory_count"}
        out = {}
        for k, v in d.items():
            if k in keys:
                out[keys[k]] = _count(k, v)
            elif k in keys.values():
                out[k] = _count(k, v)
            else:
                raise ValueError(f"unknown logical count {k!r}")
        return LogicalCounts(**out)

    def asdict(self) -> dict:
        d = {"numQubits": self.num_qubits, "tCount": self.t_count, "rotationCount": self.rotation_count, "rotationDepth": self.rotation_depth, "cczCount": self.ccz_count, "ccixCount": self.ccix_count, "measurementCount": self.measurement_count}
        for key, value in (("numComputeQubits", self.num_compute_qubits), ("readFromMemoryCount", self.read_from_memory_count), ("writeToMemoryCount", self.write_to_memory_count)):
            if value is not None:
                d[key] = value
        return d

    def logical_qubits(self) -> int:
        return 2 * self.num_qubits + math.ceil(math.sqrt(8 * self.num_qubits)) + 1

    def num_ts_per_rotation(self, eps_synthesis: float) -> Optional[int]:
        if self.rotation_count > 0:
            return math.ceil(NUM_TS_PER_ROTATION_A_COEFFICIENT * math.log2(self.rotation_count / eps_synthesis) + NUM_TS_PER_ROTATION_B_COEFFICIENT)
        return None

    def logical_depth(self, budget: "ErrorBudget") -> int:
        return (self.measurement_count + self.rotation_count + self.t_count) * NUM_MEASUREMENTS_PER_R + (self.ccz_count + self.ccix_count) * NUM_MEASUREMENTS_PER_TOF + (self.num_ts_per_rotation(budget.rotations) or 0) * self.rotation_depth * NUM_MEASUREMENTS_PER_R

    def num_magic_states(self, budget: "ErrorBudget") -> int:
        return 4 * (self.ccz_count + self.ccix_count) + self.t_count + (self.num_ts_per_rotation(budget.rotations) or 0) * self.rotation_count

    def has_tgates(self) -> bool:
        return self.t_count > 0 or self.ccz_count > 0 or self.ccix_count > 0 or self.rotation_count > 0

    def has_rotations(self) -> bool:
        return self.rotation_count > 0

    def estimate(self, params: Union[dict, Sequence[dict], Any, None] = None) -> "EstimatorResult":
        """The QDK's `LogicalCounts.estimate`: the estimate for a dict of job parameters (``qubitParams``,
        ``qecScheme``, ``errorBudget``, ``constraints``, ``distillationUnitSpecifications``, ``estimateType``), or
        a batch for a list of them (an item that fails holds its message; a single estimate that fails raises
        ValueError). The QDK's own `EstimatorParams` is taken too, through its ``as_dict``."""
        if params is None:
            items = [{}]
        elif isinstance(params, dict):
            items = [params]
        elif hasattr(params, "as_dict"):
            d = params.as_dict()
            items = d["items"] if "items" in d else [d]
        else:
            items = list(params)
        if len(items) == 1:
            return EstimatorResult(_estimate_item(self, items[0]))
        results = []
        for item in items:
            try:
                results.append(EstimatorResult(_estimate_item(self, item)))
            except ValueError as ex:
                results.append(EstimatorResult({"message": str(ex)}, error=ex))
        return EstimatorResult(results)


@dataclass
class ErrorBudget:
    """The total error budget split between logical (data) errors, T-state errors and rotation synthesis."""

    logical: float
    tstates: float
    rotations: float

    @staticmethod
    def partition(spec: Union[float, dict, "ErrorBudget"], counts: LogicalCounts) -> "ErrorBudget":
        """A total is split in thirds; in halves when there are no rotations; all to logical when there are
        no T gates. A dict ``{logical, tStates, rotations}`` is taken as is."""
        if isinstance(spec, ErrorBudget):
            return replace(spec)
        if isinstance(spec, dict):
            if "tstates" in spec and "tStates" not in spec:
                spec = {**spec, "tStates": spec["tstates"]}
            missing = [k for k in ("logical", "tStates", "rotations") if k not in spec]
            if missing:
                raise ValueError(f"an error budget dict names logical, tStates and rotations; missing {missing}")
            logical, tstates, rotations = (_float_field(k, spec[k]) for k in ("logical", "tStates", "rotations"))
            if logical + tstates + rotations >= 1.0 or logical <= 0.0 or tstates < 0.0 or (counts.has_tgates() and tstates == 0.0) or rotations < 0.0 or (counts.has_rotations() and rotations == 0.0):
                raise ValueError(f"invalid error budget {spec!r}")
            return ErrorBudget(logical, tstates, rotations)
        total = _float_field("errorBudget", spec)
        if total <= 0.0 or total >= 1.0:
            raise ValueError(f"invalid error budget {total}: expected a value strictly between 0 and 1")
        if counts.has_tgates() and counts.has_rotations():
            return ErrorBudget(total / 3.0, total / 3.0, total / 3.0)
        if counts.has_tgates():
            return ErrorBudget(total / 2.0, total / 2.0, 0.0)
        return ErrorBudget(total, 0.0, 0.0)

    def asdict(self) -> dict:
        return {"logical": self.logical, "tstates": self.tstates, "rotations": self.rotations}


# ---------------------------------------------------------------------------------------------
# Distillation units and T factories (system/modeling/tfactory.rs, estimates/factory/round_based.rs)
# ---------------------------------------------------------------------------------------------

Formula = Callable[[float, float, float], float]


def _formula(expression: str) -> Formula:
    _compile_expression(expression)  # refused now, as the QDK refuses it when it reads the specification

    def f(input_error_rate: float, clifford_error_rate: float, readout_error_rate: float) -> float:
        return evaluate_expression(expression, {"inputErrorRate": input_error_rate, "cliffordErrorRate": clifford_error_rate, "readoutErrorRate": readout_error_rate, "z": input_error_rate, "c": clifford_error_rate, "r": readout_error_rate})

    return f


def _fifteen_to_one_failure(z: float, c: float, r: float) -> float:
    return 15.0 * z + 356.0 * c


def _fifteen_to_one_output(z: float, c: float, r: float) -> float:
    return 35.0 * _powi(z, 3) + 7.1 * c


@dataclass(frozen=True)
class DistillationUnitTemplate:
    """A distillation protocol: its input and output T counts, failure and output-error formulas in the input
    error ``z``, the Clifford error ``c`` and the readout error ``r``, and its footprint and duration (in qubits
    and qubit cycles) on physical qubits and on logical qubits. ``unit_type`` is Physical, Logical or Combined
    (both specifications given)."""

    name: str
    num_input_ts: int
    num_output_ts: int
    failure_probability: Formula
    output_error_rate: Formula
    unit_type: str
    physical_qubit_specification: Optional[Tuple[int, int]] = None
    logical_qubit_specification: Optional[Tuple[int, int]] = None
    logical_qubit_specification_first_round_override: Optional[Tuple[int, int]] = None

    @staticmethod
    def fifteen_to_one_rm_prep() -> "DistillationUnitTemplate":
        return DistillationUnitTemplate("15-to-1 RM prep", 15, 1, _fifteen_to_one_failure, _fifteen_to_one_output, "Combined", (31, 24), (31, 11))

    @staticmethod
    def fifteen_to_one_space_efficient() -> "DistillationUnitTemplate":
        return DistillationUnitTemplate("15-to-1 space efficient", 15, 1, _fifteen_to_one_failure, _fifteen_to_one_output, "Combined", (12, 45), (20, 13))

    @staticmethod
    def trivial_one_to_one() -> "DistillationUnitTemplate":
        return DistillationUnitTemplate("trivial 1-to-1", 1, 1, lambda z, c, r: 0.0, lambda z, c, r: z, "Logical", None, (1, 1))

    @staticmethod
    def from_name(name: str) -> "DistillationUnitTemplate":
        if name in ("15-1 RM", "15-1 RM prep", "15-to-1 RM", "15-to-1 RM prep"):
            return DistillationUnitTemplate.fifteen_to_one_rm_prep()
        if name in ("15-1 space-efficient", "15-1 space efficient", "15-to-1 space-efficient", "15-to-1 space efficient"):
            return DistillationUnitTemplate.fifteen_to_one_space_efficient()
        raise ValueError(f"Invalid distillation unit specification name: {name}.")

    @staticmethod
    def defaults() -> List["DistillationUnitTemplate"]:
        return [DistillationUnitTemplate.fifteen_to_one_rm_prep(), DistillationUnitTemplate.fifteen_to_one_space_efficient()]

    @staticmethod
    def fromspec(spec: Union[str, dict, "DistillationUnitTemplate"]) -> "DistillationUnitTemplate":
        """A QDK ``distillationUnitSpecifications`` entry: ``{"name": ...}`` or a custom unit with
        ``displayName, numInputTs, numOutputTs, failureProbabilityFormula, outputErrorRateFormula`` and
        ``physicalQubitSpecification`` / ``logicalQubitSpecification`` ``{numUnitQubits, durationInQubitCycleTime}``."""
        if isinstance(spec, DistillationUnitTemplate):
            return spec
        if isinstance(spec, str):
            return DistillationUnitTemplate.from_name(spec)
        if not isinstance(spec, dict):
            raise ValueError(f"a distillation unit specification is a name or a dict, not {spec!r}")
        if isinstance(spec.get("name"), str):
            # The QDK's spec is an untagged union whose first arm is the named unit: a dict with
            # a name is that unit, whatever else it carries (a name that is not a string is ignored).
            return DistillationUnitTemplate.from_name(spec["name"])
        for key in ("displayName", "numInputTs", "numOutputTs", "failureProbabilityFormula", "outputErrorRateFormula"):
            if key not in spec:
                raise ValueError(f"missing field `{key}` of the distillation unit")
        if not isinstance(spec["displayName"], str):
            raise ValueError(f"displayName: expected a string, got {spec['displayName']!r}")
        num_input_ts, num_output_ts = _count("numInputTs", spec["numInputTs"]), _count("numOutputTs", spec["numOutputTs"])
        if num_input_ts == 0:
            raise ValueError("The number of input t states must be greater than 0.")
        if num_output_ts == 0:
            raise ValueError("The number of output t states must be greater than 0.")

        def resources(key):
            r = spec.get(key)
            if r is None:
                return None
            if not isinstance(r, dict) or not {"numUnitQubits", "durationInQubitCycleTime"} <= set(r):
                raise ValueError(f"{key} takes numUnitQubits and durationInQubitCycleTime, got {r!r}")
            num_unit_qubits, duration = _count(key + ".numUnitQubits", r["numUnitQubits"]), _count(key + ".durationInQubitCycleTime", r["durationInQubitCycleTime"])
            if num_unit_qubits == 0:
                raise ValueError("The number of unit qubits must be greater than 0.")
            return (num_unit_qubits, duration)

        physical, logical, override = resources("physicalQubitSpecification"), resources("logicalQubitSpecification"), resources("logicalQubitSpecificationFirstRoundOverride")
        if physical is None and logical is None:
            raise ValueError("At least one of the physical or the logical specification must be specified.")
        if physical is not None and logical is not None:
            unit_type = "Combined"
        elif physical is not None:
            if override is not None:
                raise ValueError("The logical specification can be overridden for the first round of distillation only if the logical specification is specified.")
            unit_type = "Physical"
        else:
            unit_type = "Logical"
        return DistillationUnitTemplate(spec["displayName"], num_input_ts, num_output_ts, _formula(spec["failureProbabilityFormula"]), _formula(spec["outputErrorRateFormula"]), unit_type, physical, logical, override)


def _unit_specification_echo(spec: Union[str, dict, DistillationUnitTemplate]) -> dict:
    """A distillation unit specification (one `DistillationUnitTemplate.fromspec` took) as the QDK echoes it in
    ``jobParams``, its reading written back: a named unit as its name, a custom unit's fields in the QDK's order
    without unknown keys or absent specifications. A template object is echoed by its name."""
    if isinstance(spec, str):
        return {"name": spec}
    if isinstance(spec, DistillationUnitTemplate):
        return {"name": spec.name}
    if isinstance(spec.get("name"), str):
        return {"name": spec["name"]}
    out = {key: spec[key] for key in ("displayName", "numInputTs", "numOutputTs", "failureProbabilityFormula", "outputErrorRateFormula")}
    for key in ("physicalQubitSpecification", "logicalQubitSpecification", "logicalQubitSpecificationFirstRoundOverride"):
        if spec.get(key) is not None:
            out[key] = {"numUnitQubits": spec[key]["numUnitQubits"], "durationInQubitCycleTime": spec[key]["durationInQubitCycleTime"]}
    return out


@dataclass(frozen=True)
class DistillationUnit:
    """A template placed on a qubit: a `LogicalPatch` (logical unit) or a physical qubit (``patch=None``)."""

    template: DistillationUnitTemplate
    code_distance: int
    physical_qubits_at_first_round: int
    physical_qubits_at_subsequent_rounds: int
    duration_at_first_round: int
    duration_at_subsequent_rounds: int
    clifford_error_rate: float
    readout_error_rate: float
    qubit_t_error_rate: float

    @staticmethod
    def by_template(template: DistillationUnitTemplate, qubit: PhysicalQubit, patch: Optional[LogicalPatch]) -> "DistillationUnit":
        if patch is None:
            code_distance, n_phys, cycle = 1, 1, qubit.t_gate_time
            clifford, readout = qubit.clifford_error_rate(), qubit.readout_error_rate()
        else:
            code_distance, n_phys, cycle = patch.code_distance, patch.physical_qubits, patch.logical_cycle_time
            clifford, readout = patch.logical_error_rate, 0.0
        if code_distance == 1 and template.unit_type in ("Combined", "Physical"):
            first = template.physical_qubit_specification
        elif template.unit_type == "Physical":
            first = None
        else:
            first = template.logical_qubit_specification_first_round_override or template.logical_qubit_specification
        subsequent = None if template.unit_type == "Physical" else template.logical_qubit_specification
        fq, fd = (first[0] * n_phys, first[1] * cycle) if first else (0, 0)
        sq, sd = (subsequent[0] * n_phys, subsequent[1] * cycle) if subsequent else (0, 0)
        return DistillationUnit(template, code_distance, fq, sq, fd, sd, clifford, readout, qubit.t_gate_error_rate)

    @property
    def name(self) -> str:
        return self.template.name

    def num_output_states(self) -> int:
        return self.template.num_output_ts

    def num_input_states(self) -> int:
        return self.template.num_input_ts

    def duration(self, position: int) -> int:
        return self.duration_at_first_round if position == 0 else self.duration_at_subsequent_rounds

    def physical_qubits(self, position: int) -> int:
        return self.physical_qubits_at_first_round if position == 0 else self.physical_qubits_at_subsequent_rounds

    def output_error_rate(self, input_error_rate: float) -> float:
        return self.template.output_error_rate(input_error_rate, self.clifford_error_rate, self.readout_error_rate)

    def failure_probability(self, input_error_rate: float) -> float:
        return self.template.failure_probability(input_error_rate, self.clifford_error_rate, self.readout_error_rate)

    def is_valid(self) -> bool:
        return self.clifford_error_rate <= 0.1 * self.qubit_t_error_rate


class FactoryBuildError(Exception):
    """One of the QDK's reasons a round sequence makes no factory."""

    LOW_FAILURE_PROBABILITY = "LowFailureProbability"
    HIGH_FAILURE_PROBABILITY = "HighFailureProbability"
    OUTPUT_ERROR_RATE_HIGHER_THAN_INPUT = "OutputErrorRateHigherThanInputErrorRate"
    UNREASONABLE_NUMBER_OF_UNITS = "UnreasonableHighNumberOfUnitsRequired"

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


@dataclass
class DistillationRound:
    """One round of a factory: `num_units` copies of a unit run in parallel."""

    num_units: int
    failure_probability_requirement: float
    num_output_states: int
    num_input_states: int
    duration: int
    physical_qubits_per_unit: int
    name: str
    code_distance: int

    @staticmethod
    def new(unit: DistillationUnit, failure_probability_requirement: float, position: int) -> "DistillationRound":
        return DistillationRound(1, failure_probability_requirement, unit.num_output_states(), unit.num_input_states(), unit.duration(position), unit.physical_qubits(position), unit.name, unit.code_distance)

    def physical_qubits(self) -> int:
        return self.num_units * self.physical_qubits_per_unit

    def compute_num_output_states(self, failure_probability: float) -> int:
        if failure_probability == 0.0 and self.failure_probability_requirement == 0.0:
            return self.num_units * self.num_output_states
        return _Binomial(self.num_units, failure_probability).inverse(self.failure_probability_requirement) * self.num_output_states

    def adjust_num_units_to(self, output_states_needed_next: int, failure_probability: float) -> None:
        """The fewest units whose guaranteed output (at the round's failure-probability requirement) covers the
        next round's input: doubling up, then bisection."""
        self.num_units = math.ceil(output_states_needed_next / (self.num_units * self.num_output_states))
        while True:
            if self.compute_num_output_states(failure_probability) < output_states_needed_next:
                self.num_units *= 2
                if self.num_units >= 1_000_000_000_000_000:
                    raise FactoryBuildError(FactoryBuildError.UNREASONABLE_NUMBER_OF_UNITS)
            else:
                break
        upper = self.num_units
        lower = self.num_units // 2
        while lower < upper:
            self.num_units = (lower + upper) // 2
            if self.compute_num_output_states(failure_probability) >= output_states_needed_next:
                upper = self.num_units
            else:
                lower = self.num_units + 1
        self.num_units = upper


@dataclass
class TFactory:
    """A round-based T factory: the QDK's `RoundBasedFactory`, physical qubits the maximum over rounds."""

    rounds: List[DistillationRound]
    failure_probability_requirement: float
    input_error_rate_before_each_round: List[float]
    failure_probability_after_each_round: List[float]

    @staticmethod
    def build(units: Sequence[DistillationUnit], initial_input_error_rate: float, failure_probability_requirement: float) -> "TFactory":
        """Units in order of the rounds; the error rate feeds forward, the unit counts are sized backwards from
        one output state. Raises `FactoryBuildError`."""
        f = TFactory([], failure_probability_requirement, [initial_input_error_rate], [1.0] * (len(units) + 1))
        f._compute_units_per_round(units, 1)
        return f

    def _compute_units_per_round(self, units: Sequence[DistillationUnit], multiplier: int) -> None:
        length = len(units)
        for unit in units:
            requirement = self.failure_probability_requirement / length
            input_error_rate = self.input_error_rate_before_each_round[-1]
            output_error_rate = unit.output_error_rate(input_error_rate)
            if output_error_rate > input_error_rate:
                raise FactoryBuildError(FactoryBuildError.OUTPUT_ERROR_RATE_HIGHER_THAN_INPUT)
            self.rounds.append(DistillationRound.new(unit, requirement, len(self.rounds)))
            self.input_error_rate_before_each_round.append(output_error_rate)
        if length > 0:
            states_needed_next = self.rounds[length - 1].num_output_states * multiplier
            for idx in range(length - 1, -1, -1):
                q = units[idx].failure_probability(self.input_error_rate_before_each_round[idx])
                if q <= 0.0:
                    raise FactoryBuildError(FactoryBuildError.LOW_FAILURE_PROBABILITY)
                if q >= 1.0:
                    raise FactoryBuildError(FactoryBuildError.HIGH_FAILURE_PROBABILITY)
                self.failure_probability_after_each_round[idx] = q
                self.rounds[idx].adjust_num_units_to(states_needed_next, q)
                states_needed_next = self.rounds[idx].num_input_states * self.rounds[idx].num_units

    @property
    def length(self) -> int:
        return len(self.rounds)

    def physical_qubits(self) -> int:
        return max((r.physical_qubits() for r in self.rounds), default=0)

    def duration(self) -> int:
        return sum(r.duration for r in self.rounds)

    def num_output_states(self) -> int:
        return self.rounds[-1].compute_num_output_states(self.failure_probability_after_each_round[self.length - 1])

    def num_input_states(self) -> int:
        return self.rounds[0].num_input_states * self.rounds[0].num_units if self.rounds else 0

    def input_error_rate(self) -> float:
        return self.input_error_rate_before_each_round[0]

    def output_error_rate(self) -> float:
        return self.input_error_rate_before_each_round[self.length]

    def normalized_qubits(self) -> float:
        return self.physical_qubits() / self.num_output_states()

    def normalized_volume(self) -> float:
        return (self.physical_qubits() * self.duration()) / self.num_output_states()

    def num_units_per_round(self) -> List[int]:
        return [r.num_units for r in self.rounds]

    def unit_names(self) -> List[str]:
        return [r.name for r in self.rounds]

    def code_distance_per_round(self) -> List[int]:
        return [r.code_distance for r in self.rounds]

    def physical_qubits_per_round(self) -> List[int]:
        return [r.physical_qubits() for r in self.rounds]

    def duration_per_round(self) -> List[int]:
        return [r.duration for r in self.rounds]

    def max_code_distance(self) -> int:
        return self.rounds[-1].code_distance

    def asdict(self) -> dict:
        return {"physicalQubits": self.physical_qubits(), "runtime": self.duration(), "numTstates": self.num_output_states(), "numInputTstates": self.num_input_states(), "numRounds": self.length, "numUnitsPerRound": self.num_units_per_round(), "unitNamePerRound": self.unit_names(), "codeDistancePerRound": self.code_distance_per_round(), "physicalQubitsPerRound": self.physical_qubits_per_round(), "runtimePerRound": self.duration_per_round(), "logicalErrorRate": self.output_error_rate()}


def default_t_factory(patch: LogicalPatch) -> TFactory:
    """The trivial factory when the physical T error already meets the requirement."""
    unit = DistillationUnit.by_template(DistillationUnitTemplate.trivial_one_to_one(), patch.qubit, patch)
    t_error = patch.qubit.t_gate_error_rate
    return TFactory([DistillationRound.new(unit, 0.0, 0)], 0.0, [t_error, t_error], [0.0, 0.0])


# ---------------------------------------------------------------------------------------------
# The exhaustive T-factory search (system/optimization/*, estimates/optimization/population.rs)
# ---------------------------------------------------------------------------------------------


class _Point2D:
    """An item with two costs to minimise; a factory's are its qubits per output state and its duration."""

    __slots__ = ("item", "value1", "value2")

    def __init__(self, item, value1: Optional[float] = None, value2: Optional[int] = None):
        self.item = item
        self.value1 = item.normalized_qubits() if value1 is None else value1
        self.value2 = item.duration() if value2 is None else value2

    def dominates(self, other: "_Point2D") -> bool:
        return (self.value1 < other.value1 and self.value2 <= other.value2) or (self.value1 <= other.value1 and self.value2 < other.value2)


class _Population:
    MAX_NONEXECUTED_ATTEMPTS = 1000

    def __init__(self):
        self.items: List[_Point2D] = []
        self.nonexecuted_attempts = 0

    def push(self, point: _Point2D) -> None:
        self.items.append(point)

    def sort_items(self) -> None:
        self.items.sort(key=lambda p: (p.value1, p.value2), reverse=True)

    def attempt_filter_out_dominated(self) -> None:
        if self.nonexecuted_attempts <= self.MAX_NONEXECUTED_ATTEMPTS:
            self.nonexecuted_attempts += 1
        else:
            self.filter_out_dominated()

    def filter_out_dominated(self) -> None:
        idx = 0
        length = len(self.items)
        while idx < length:
            if self.dominates(self.items[idx]):
                self.items[idx], self.items[length - 1] = self.items[length - 1], self.items[idx]
                length -= 1
            else:
                idx += 1
        self.nonexecuted_attempts = 0
        del self.items[length:]

    def dominates(self, other: _Point2D) -> bool:
        return any(t.dominates(other) for t in self.items)


class _DistillationUnitsMap:
    """Units by (round position, code distance, template), with the physical variants at distance 1 in the
    first round and the first valid distance of every template."""

    def __init__(self, qubit: PhysicalQubit, patches: List[Optional[LogicalPatch]], distances: List[int], templates: Sequence[DistillationUnitTemplate]):
        combined = [t for t in templates if t.unit_type == "Combined"]
        logical = [t for t in templates if t.unit_type == "Logical"]
        self.num_combined = len(combined)
        self.num_logical = len(logical)
        self.physical_units = [DistillationUnit.by_template(t, qubit, None) for t in combined]
        purely_physical = [u for u in (DistillationUnit.by_template(t, qubit, None) for t in templates if t.unit_type == "Physical") if u.is_valid()]
        self.num_physical = len(purely_physical)
        self.physical_units += purely_physical
        self.is_valid_physical = [True] * (self.num_combined + self.num_physical)
        for idx in range(self.num_combined):
            if not self.physical_units[idx].is_valid():
                self.is_valid_physical[idx] = False
        logical_templates = combined + logical
        self.logical_units: List[List[DistillationUnit]] = []
        for patch in patches:
            self.logical_units.append([DistillationUnit.by_template(t, qubit, patch) for t in logical_templates] if patch is not None else [])
        self.min_valid_code_distance_indexes = [len(patches)] * (self.num_logical + self.num_combined)
        nonempty = [units for units in self.logical_units if units]
        for idx in range(self.num_logical + self.num_combined):
            valid = [i for i, units in enumerate(nonempty) if units[idx].is_valid()]
            if valid:
                self.min_valid_code_distance_indexes[idx] = min(valid)
        self.num_code_distances = len(distances)
        self.distances = distances

    def get(self, position: int, distance: int, idx: int) -> DistillationUnit:
        if distance == 1 and position == 0:
            return self.physical_units[idx if idx < self.num_combined else idx - self.num_logical]
        return self.logical_units[distance][idx]

    def get_min_distance_indexes(self, indexes: Sequence[int]) -> List[int]:
        out, state = [], 0
        for position, idx in enumerate(indexes):
            state = max(state, self._min_distance_index(position, idx))
            out.append(state)
        return out

    def get_max_distance_indexes(self, indexes: Sequence[int]) -> List[int]:
        return [self._max_distance_index(position, idx) for position, idx in enumerate(indexes)]

    def _min_distance_index(self, position: int, idx: int) -> int:
        if position == 0:
            if idx >= self.num_logical + self.num_combined:
                return 0
            if idx < self.num_combined:
                return 0 if self.is_valid_physical[idx] else max(1, self.min_valid_code_distance_indexes[idx])
            return max(1, self.min_valid_code_distance_indexes[idx])
        if idx >= self.num_logical + self.num_combined:
            raise ValueError(f"Invalid position for physical unit: {position}")
        return self.min_valid_code_distance_indexes[idx]

    def _max_distance_index(self, position: int, idx: int) -> int:
        if idx >= self.num_logical + self.num_combined:
            if position == 0:
                return 0
            raise ValueError(f"Invalid position for physical unit: {position}")
        return self.num_code_distances - 1

    def get_many(self, distance_indexes: Sequence[int], indexes: Sequence[int]) -> List[DistillationUnit]:
        return [self.get(position, self.distances[di], idx) for position, (idx, di) in enumerate(zip(indexes, distance_indexes))]

    def iterate_for_all_distillation_units(self, num_rounds: int, action: Callable[[List[int]], None]) -> None:
        a = [0] * (num_rounds + 1)
        first_max = self.num_logical + self.num_combined + self.num_physical - 1
        other_max = self.num_logical + self.num_combined - 1
        while True:
            action(a[1:])
            j = num_rounds
            while a[j] == (1 if j == 0 else first_max if j == 1 else other_max):
                a[j] = 0
                j -= 1
            if j == 0:
                break
            a[j] += 1


def _search_for_code_distances(num_rounds: int, left: Sequence[int], right_in: Sequence[int], check: Callable[[List[int]], bool]) -> Optional[List[int]]:
    right = list(right_in)
    if check(right):
        return None
    if not check(list(left)):
        return None
    mid = list(right)
    for cur in range(num_rounds):
        lft = list(mid)
        lft[cur] = left[cur] if cur == 0 else mid[cur - 1]
        while lft[cur] + 1 < right[cur]:
            mid[cur] = (lft[cur] + right[cur]) // 2
            if check(mid):
                lft[cur] = mid[cur] + 1
            else:
                right[cur] = mid[cur]
        if lft[cur] + 1 == right[cur]:
            mid[cur] = lft[cur]
            if check(mid):
                mid[cur] = right[cur]
        else:
            mid[cur] = right[cur]
    return mid


def _increment_code_distance_indexes(current: List[int], left: Sequence[int], right: Sequence[int], num_rounds: int) -> bool:
    i = num_rounds - 1
    current[i] += 1
    while current[i] > right[i]:
        if i == 0:
            return False
        i -= 1
        current[i] += 1
    while i < num_rounds - 1:
        current[i + 1] = max(current[i], left[i + 1])
        i += 1
    return True


def _switch_to_non_comparable(current: List[int], left: Sequence[int], right: Sequence[int], num_rounds: int) -> bool:
    j = len(current) - 1
    while j > 0 and (current[j] <= current[j - 1] + 1 or current[j - 1] == right[j - 1]):
        j -= 1
    if j == 0:
        return False
    current[j - 1] += 1
    while j < num_rounds:
        current[j] = max(current[j - 1], left[j])
        j += 1
    return True


def _iterate_for_code_distances(num_rounds: int, left: Sequence[int], right: Sequence[int], initial: Sequence[int], check: Callable[[List[int]], bool]) -> None:
    current = [max(c, l) for c, l in zip(initial, left)]
    if not _switch_to_non_comparable(current, left, right, num_rounds):
        return
    while True:
        if check(current):
            if not _increment_code_distance_indexes(current, left, right, num_rounds):
                return
        elif not _switch_to_non_comparable(current, left, right, num_rounds):
            return


class _ExhaustiveSearch:
    def __init__(self, output_t_error_rate: float):
        self.output_t_error_rate = output_t_error_rate
        self.frontier = _Population()

    def check(self, units: Sequence[DistillationUnit]) -> bool:
        """Builds the factory of `units`; True when a larger code distance could still improve on it."""
        try:
            factory = TFactory.build(units, units[0].qubit_t_error_rate, 0.01)
        except FactoryBuildError as e:
            return e.kind != FactoryBuildError.LOW_FAILURE_PROBABILITY
        good = factory.output_error_rate() <= self.output_t_error_rate
        point = _Point2D(factory)
        not_dominated = not self.frontier.dominates(point)
        if not_dominated and good:
            self.frontier.push(point)
            self.frontier.attempt_filter_out_dominated()
        return not_dominated and not good


def find_nondominated_tfactories(scheme: QECScheme, qubit: PhysicalQubit, templates: Sequence[DistillationUnitTemplate], output_t_error_rate: float, max_code_distance: int, max_distillation_rounds: int = MAX_DISTILLATION_ROUNDS) -> List[TFactory]:
    """The Pareto frontier (qubits per output T state against duration) of factories reaching
    `output_t_error_rate`, over every sequence of up to `max_distillation_rounds` units (one more round at a
    time, up to four, when none is found) and non-decreasing odd code distances up to `max_code_distance`;
    sorted as the QDK sorts it (descending)."""
    distances = list(range(1, max_code_distance + 1, 2))
    if output_t_error_rate > qubit.t_gate_error_rate:
        population = _Population()
        try:
            population.push(_Point2D(default_t_factory(LogicalPatch.new(scheme, qubit, max_code_distance))))
        except ValueError:
            pass
        population.sort_items()
        return [p.item for p in population.items]
    patches: List[Optional[LogicalPatch]] = [None] * (max_code_distance + 1)
    for d in distances:
        try:
            patches[d] = LogicalPatch.new(scheme, qubit, d)
        except ValueError:
            patches[d] = None
    units_map = _DistillationUnitsMap(qubit, patches, distances, templates)
    searcher = _ExhaustiveSearch(output_t_error_rate)

    def process(unit_indexes: List[int]) -> None:
        left = units_map.get_min_distance_indexes(unit_indexes)
        right = units_map.get_max_distance_indexes(unit_indexes)
        n = len(unit_indexes)

        def check(distance_indexes: List[int]) -> bool:
            return searcher.check(units_map.get_many(distance_indexes, unit_indexes))

        result = _search_for_code_distances(n, left, right, check)
        if result is not None:
            _iterate_for_code_distances(n, left, right, result, check)

    for num_rounds in range(1, max_distillation_rounds + 1):
        units_map.iterate_for_all_distillation_units(num_rounds, process)
    if not searcher.frontier.items:
        for num_rounds in range(max_distillation_rounds + 1, MAX_EXTRA_DISTILLATION_ROUNDS + 1):
            units_map.iterate_for_all_distillation_units(num_rounds, process)
    searcher.frontier.filter_out_dominated()
    searcher.frontier.sort_items()
    return [p.item for p in searcher.frontier.items]


# ---------------------------------------------------------------------------------------------
# Physical estimation (estimates/physical_estimation*.rs, system.rs, system/data/result.rs)
# ---------------------------------------------------------------------------------------------


@dataclass
class Constraints:
    """The QDK's ``constraints``: a factor on the logical depth, at most so many T factories, a duration cap
    (ns) or a physical-qubit cap (not both), and the number of distillation rounds searched."""

    logical_depth_factor: Optional[float] = None
    max_t_factories: Optional[int] = None
    max_duration: Optional[int] = None
    max_physical_qubits: Optional[int] = None
    max_distillation_rounds: int = MAX_DISTILLATION_ROUNDS

    @staticmethod
    def fromspec(spec: Union[dict, "Constraints", None]) -> "Constraints":
        if spec is None:
            return Constraints()
        if isinstance(spec, Constraints):
            return replace(spec)
        if not isinstance(spec, dict):
            raise ValueError(f"constraints are a dict, not {spec!r}")
        keys = {"logicalDepthFactor": "logical_depth_factor", "maxTFactories": "max_t_factories", "maxDuration": "max_duration", "maxPhysicalQubits": "max_physical_qubits", "maxDistillationRounds": "max_distillation_rounds"}
        out = {}
        for k, v in spec.items():
            attr = keys.get(k, k if k in keys.values() else None)
            if attr is None:
                raise ValueError(f"unknown constraint {k!r}")
            if v is None and attr in ("logical_depth_factor", "max_t_factories", "max_physical_qubits"):
                continue  # the QDK's optional fields: null is absent (maxDuration and maxDistillationRounds refuse null)
            out[attr] = parse_time(v) if attr == "max_duration" else _float_field(k, v) if attr == "logical_depth_factor" else _count(k, v)
        return Constraints(**out)

    def asdict(self) -> dict:
        d = {}
        if self.logical_depth_factor is not None:
            d["logicalDepthFactor"] = self.logical_depth_factor
        if self.max_t_factories is not None:
            d["maxTFactories"] = self.max_t_factories
        if self.max_duration is not None:
            d["maxDuration"] = f"{self.max_duration} ns"
        if self.max_physical_qubits is not None:
            d["maxPhysicalQubits"] = self.max_physical_qubits
        d["maxDistillationRounds"] = self.max_distillation_rounds
        return d


@dataclass
class FactoryPart:
    factory: TFactory
    copies: int
    runs: int
    required_output_error_rate: float

    @staticmethod
    def new(factory: TFactory, copies: int, num_magic_states: int, required_output_error_rate: float) -> "FactoryPart":
        return FactoryPart(factory, copies, _ceil_div(num_magic_states, copies * factory.num_output_states()), required_output_error_rate)

    def physical_qubits(self) -> int:
        return self.factory.physical_qubits() * self.copies


# ---------------------------------------------------------------------------------------------
# The formatted counts and the report (system/data/report.rs, result.rs)
# ---------------------------------------------------------------------------------------------


def _nonfinite(x: float) -> Optional[str]:
    """How Rust writes a NaN or an infinity in any float format; None for a finite value."""
    if x != x:
        return "NaN"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return None


def _display(x: float) -> str:
    """Rust's `{}` for an f64: the shortest digits that read back as the value, never in exponent form
    (``1e-10`` is ``0.0000000001``, ``1.0`` is ``1``)."""
    return _nonfinite(x) or format(decimal.Decimal(repr(float(x))).normalize(), "f")


def _exp2(x: float) -> str:
    """Rust's `{:.2e}`: three significant digits, the exponent bare (``1.55e-11``, ``2.00e5``, ``0.00e0``)."""
    special = _nonfinite(x)
    if special:
        return special
    mantissa, exponent = f"{x:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _fixed2(x: float) -> str:
    """Rust's `{:.2}`."""
    return _nonfinite(x) or f"{x:.2f}"


def _thousands(value) -> str:
    """The QDK's `format_thousand_sep`: the digits in groups of three from the right, joined by commas."""
    s = str(value)
    head = len(s) % 3 or 3
    return ",".join([s[:head]] + [s[i:i + 3] for i in range(head, len(s), 3)])


def _thousands_f64(value: float) -> str:
    integer, fraction = f"{value:.2f}".split(".")
    return f"{_thousands(integer)}.{fraction}"


def format_metric_prefix(value: int) -> str:
    """The QDK's count with a metric prefix: as it is below 1000, else to two decimals with k, M, G, T, P, E, Z,
    Y, R or Q (``400.06k``, ``38.33M``)."""
    if value < 1000:
        return str(value)
    prefixes = "kMGTPEZYRQ"
    index, v = 0, float(value) / 1e3
    while True:
        following = v / 1e3
        if following < 1.0:
            break
        index += 1
        v = following
    return f"{v:.2f}{prefixes[index] if index < len(prefixes) else '?'}"


_DURATION_UNITS = (("nanosecs", 1), ("microsecs", 1000), ("millisecs", 1000), ("secs", 1000), ("mins", 60), ("hours", 60), ("days", 24), ("years", 365))


def format_duration(nanos: int) -> str:
    """The QDK's duration in words: the largest unit the value reaches, rounded half up by the remainder in the
    unit below (``1500`` ns is ``2 microsecs``, ``1064000000`` is ``1 secs``). As the QDK's, a value below a
    microsecond reads one nanosecond high (``500`` is ``501 nanosecs``)."""
    runtime, rem = int(nanos), 0
    for idx in range(1, len(_DURATION_UNITS)):
        if runtime // _DURATION_UNITS[idx][1] == 0:
            if rem >= _DURATION_UNITS[idx - 1][1] // 2:
                runtime += 1
            return f"{runtime} {_DURATION_UNITS[idx - 1][0]}"
        runtime, rem = runtime // _DURATION_UNITS[idx][1], runtime % _DURATION_UNITS[idx][1]
    return f"{runtime} {_DURATION_UNITS[-1][0]}"


def _budget_echo(spec) -> Union[float, dict]:
    """The error budget (one `ErrorBudget.partition` took) as the QDK echoes it in ``jobParams``: the total, or
    the parts as ``logical``, ``tStates``, ``rotations``."""
    if spec is None:
        return 1e-3
    if isinstance(spec, ErrorBudget):
        return {"logical": spec.logical, "tStates": spec.tstates, "rotations": spec.rotations}
    if isinstance(spec, dict):
        return {"logical": float(spec["logical"]), "tStates": float(spec["tStates"] if "tStates" in spec else spec["tstates"]), "rotations": float(spec["rotations"])}
    return float(spec)


def _budget_total(spec) -> float:
    """The QDK's `ErrorBudgetSpecification::total`: the total, or the parts' sum."""
    echo = _budget_echo(spec)
    return echo["logical"] + echo["tStates"] + echo["rotations"] if isinstance(echo, dict) else echo


def _formatted_counts(result: "EstimationResult") -> dict:
    est, counts, budget, patch, part = result.estimator, result.counts, result.error_budget, result.logical_patch, result.factory_part
    factory = part.factory if part else None
    no_tstates, unset = "No T states in algorithm", "constraint not set"
    M = format_metric_prefix
    c = est.constraints
    num_ts_per_rotation = counts.num_ts_per_rotation(budget.rotations)
    return {
        "runtime": format_duration(result.runtime()),
        "rqops": M(result.rqops()),
        "physicalQubits": M(result.physical_qubits()),
        "algorithmicLogicalQubits": M(counts.logical_qubits()),
        "algorithmicLogicalDepth": M(counts.logical_depth(budget)),
        "logicalDepth": M(result.num_cycles),
        "numTstates": M(counts.num_magic_states(budget)),
        "numTfactories": M(part.copies if part else 0),
        "numTfactoryRuns": M(part.runs if part else 0),
        "physicalQubitsForAlgorithm": M(result.physical_qubits_for_algorithm()),
        "physicalQubitsForTfactories": M(result.physical_qubits_for_factories()),
        "physicalQubitsForTfactoriesPercentage": _fixed2(_div(float(result.physical_qubits_for_factories() * 100), float(result.physical_qubits()))) + " %",
        "requiredLogicalQubitErrorRate": _exp2(result.required_logical_error_rate),
        "requiredLogicalTstateErrorRate": _exp2(part.required_output_error_rate) if part else no_tstates,
        "physicalQubitsPerLogicalQubit": M(patch.physical_qubits),
        "logicalCycleTime": format_duration(patch.logical_cycle_time),
        "clockFrequency": M(_as_u64(_round_half_away(patch.logical_cycles_per_second()))),
        "logicalErrorRate": _exp2(patch.logical_error_rate),
        "tfactoryPhysicalQubits": M(factory.physical_qubits()) if factory else no_tstates,
        "tfactoryRuntime": format_duration(factory.duration()) if factory else no_tstates,
        "numInputTstates": M(factory.num_input_states()) if factory else no_tstates,
        "numUnitsPerRound": ", ".join(str(n) for n in factory.num_units_per_round()) if factory else no_tstates,
        "unitNamePerRound": ", ".join(factory.unit_names()) if factory else no_tstates,
        "codeDistancePerRound": ", ".join(str(d) for d in factory.code_distance_per_round()) if factory else no_tstates,
        "physicalQubitsPerRound": ", ".join(M(q) for q in factory.physical_qubits_per_round()) if factory else no_tstates,
        "tfactoryRuntimePerRound": ", ".join(format_duration(t) for t in factory.duration_per_round()) if factory else no_tstates,
        "tstateLogicalErrorRate": _exp2(factory.output_error_rate()) if factory else no_tstates,
        "logicalCountsNumQubits": M(counts.num_qubits),
        "logicalCountsTCount": M(counts.t_count),
        "logicalCountsRotationCount": M(counts.rotation_count),
        "logicalCountsRotationDepth": M(counts.rotation_depth),
        "logicalCountsCczCount": M(counts.ccz_count),
        "logicalCountsCcixCount": M(counts.ccix_count),
        "logicalCountsMeasurementCount": M(counts.measurement_count),
        "errorBudget": _exp2(_budget_total(est.error_budget_spec)),
        "errorBudgetLogical": _exp2(budget.logical),
        "errorBudgetTstates": _exp2(budget.tstates),
        "errorBudgetRotations": _exp2(budget.rotations),
        "numTsPerRotation": M(num_ts_per_rotation) if num_ts_per_rotation is not None else "No rotations in algorithm",
        "logicalDepthFactor": _display(c.logical_depth_factor) if c.logical_depth_factor is not None else unset,
        "maxTFactories": str(c.max_t_factories) if c.max_t_factories is not None else unset,
        "maxDuration": str(c.max_duration) if c.max_duration is not None else unset,
        "maxPhysicalQubits": str(c.max_physical_qubits) if c.max_physical_qubits is not None else unset,
    }


_REPORT_ASSUMPTIONS = (
    "_More details on the following lists of assumptions can be found in the paper [Accessing requirements for scaling quantum computers and their applications](https://aka.ms/AQ/RE/Paper)._",
    "**Uniform independent physical noise.** We assume that the noise on physical qubits and physical qubit operations is the standard circuit noise model. In particular we assume error events at different space-time locations are independent and that error rates are uniform across the system in time and space.",
    "**Efficient classical computation.** We assume that classical overhead (compilation, control, feedback, readout, decoding, etc.) does not dominate the overall cost of implementing the full quantum algorithm.",
    "**Extraction circuits for planar quantum ISA.** We assume that stabilizer extraction circuits with similar depth and error correction performance to those for standard surface and Hastings-Haah code patches can be constructed to implement all operations of the planar quantum ISA (instruction set architecture).",
    "**Uniform independent logical noise.** We assume that the error rate of a logical operation is approximately equal to its space-time volume (the number of tiles multiplied by the number of logical time steps) multiplied by the error rate of a logical qubit in a standard one-tile patch in one logical time step.",
    "**Negligible Clifford costs for synthesis.** We assume that the space overhead for synthesis and space and time overhead for transport of magic states within magic state factories and to synthesis qubits are all negligible.",
    "**Smooth magic state consumption rate.** We assume that the rate of T state consumption throughout the compiled algorithm is almost constant, or can be made almost constant without significantly increasing the number of logical time steps for the algorithm.",
)


def _entry(path: str, label: str, description: str, explanation: str) -> dict:
    return {"path": path, "label": label, "description": description, "explanation": explanation}


def _report_data(result: "EstimationResult", fc: dict) -> dict:
    """The QDK's ``reportData``: its groups of entries (a path into the result, a label, a description, an
    explanation with this estimate's numbers in it), transcribed from report.rs (which the QDK generates from its
    docs/output_data.md), and its assumptions."""
    est, counts, budget, patch, part = result.estimator, result.counts, result.error_budget, result.logical_patch, result.factory_part
    factory = part.factory if part else None
    scheme, qubit = est.scheme, est.qubit
    majorana = isinstance(qubit, MajoranaQubit)
    logical_qubits, algorithmic_depth, num_tstates = counts.logical_qubits(), counts.logical_depth(budget), counts.num_magic_states(budget)
    T, D = _thousands, _display
    groups = []
    entries = []
    entries.append(_entry("physicalCountsFormatted/runtime", "Runtime", "Total runtime", "This is a runtime estimate for the execution time of the algorithm.  In general, the execution time corresponds to the duration of one logical cycle ({} nanosecs) multiplied by the {} logical cycles to run the algorithm.  If however the duration of a single T factory (here: {} nanosecs) is larger than the algorithm runtime, we extend the number of logical cycles artificially in order to exceed the runtime of a single T factory.".format(T(patch.logical_cycle_time), T(algorithmic_depth), T(factory.duration() if factory else 0))))
    entries.append(_entry("physicalCountsFormatted/rqops", "rQOPS", "Reliable quantum operations per second", "The value is computed as the number of logical qubits after layout ({}) (with a logical error rate of {}) multiplied by the clock frequency ({}), which is the number of logical cycles per second.".format(T(logical_qubits), fc["requiredLogicalQubitErrorRate"], _thousands_f64(patch.logical_cycles_per_second()))))
    entries.append(_entry("physicalCountsFormatted/physicalQubits", "Physical qubits", "Number of physical qubits", "This value represents the total number of physical qubits, which is the sum of {} physical qubits to implement the algorithm logic, and {} physical qubits to execute the T factories that are responsible to produce the T states that are consumed by the algorithm.".format(T(result.physical_qubits_for_algorithm()), T(result.physical_qubits_for_factories()))))
    groups.append({"title": "Physical resource estimates", "alwaysVisible": True, "entries": entries})
    entries = []
    entries.append(_entry("physicalCountsFormatted/algorithmicLogicalQubits", "Logical algorithmic qubits", "Number of logical qubits for the algorithm after layout", r"Laying out the logical qubits in the presence of nearest-neighbor constraints requires additional logical qubits.  In particular, to layout the $Q_{{\rm alg}} = {}$ logical qubits in the input algorithm, we require in total $2 \cdot Q_{{\rm alg}} + \lceil \sqrt{{8 \cdot Q_{{\rm alg}}}}\rceil + 1 = {}$ logical qubits.".format(T(counts.num_qubits), T(logical_qubits))))
    entries.append(_entry("physicalCountsFormatted/algorithmicLogicalDepth", "Algorithmic depth", "Number of logical cycles for the algorithm", "To execute the algorithm using _Parallel Synthesis Sequential Pauli Computation_ (PSSPC), operations are scheduled in terms of multi-qubit Pauli measurements, for which assume an execution time of one logical cycle.  Based on the input algorithm, we require one multi-qubit measurement for the {} single-qubit measurements, the {} arbitrary single-qubit rotations, and the {} T gates, three multi-qubit measurements for each of the {} CCZ and {} CCiX gates in the input program, as well as {} multi-qubit measurements for each of the {} non-Clifford layers in which there is at least one single-qubit rotation with an arbitrary angle rotation.".format(T(counts.measurement_count), T(counts.rotation_count), T(counts.t_count), T(counts.ccz_count), T(counts.ccix_count), fc["numTsPerRotation"], T(counts.rotation_depth))))
    entries.append(_entry("physicalCountsFormatted/logicalDepth", "Logical depth", "Number of logical cycles performed", "This number is usually equal to the logical depth of the algorithm, which is {}.  However, in the case in which a single T factory is slower than the execution time of the algorithm, we adjust the logical cycle depth to exceed the T factory's execution time.".format(T(algorithmic_depth))))
    entries.append(_entry("physicalCountsFormatted/clockFrequency", "Clock frequency", "Number of logical cycles per second", "This is the number of logical cycles that can be performed within one second.  The logical cycle time is {}.".format(fc["logicalCycleTime"])))
    entries.append(_entry("physicalCountsFormatted/numTstates", "Number of T states", "Number of T states consumed by the algorithm", "To execute the algorithm, we require one T state for each of the {} T gates, four T states for each of the {} CCZ and {} CCiX gates, as well as {} for each of the {} single-qubit rotation gates with arbitrary angle rotation.".format(T(counts.t_count), T(counts.ccz_count), T(counts.ccix_count), fc["numTsPerRotation"], T(counts.rotation_count))))
    entries.append(_entry("physicalCountsFormatted/numTfactories", "Number of T factories", "Number of T factories capable of producing the demanded {} T states during the algorithm's runtime".format(T(num_tstates)), r"The total number of T factories {} that are executed in parallel is computed as $\left\lceil\dfrac{{\text{{T states}}\cdot\text{{T factory duration}}}}{{\text{{T states per T factory}}\cdot\text{{algorithm runtime}}}}\right\rceil = \left\lceil\dfrac{{{} \cdot {}\;\text{{ns}}}}{{{} \cdot {}\;\text{{ns}}}}\right\rceil$".format(T(part.copies if part else 0), T(num_tstates), T(factory.duration() if factory else 0), T(factory.num_output_states() if factory else 0), T(result.runtime()))))
    entries.append(_entry("physicalCountsFormatted/numTfactoryRuns", "Number of T factory invocations", "Number of times all T factories are invoked", "In order to prepare the {} T states, the {} copies of the T factory are repeatedly invoked {} times.".format(T(num_tstates), T(part.copies if part else 0), T(part.runs if part else 0))))
    entries.append(_entry("physicalCountsFormatted/physicalQubitsForAlgorithm", "Physical algorithmic qubits", "Number of physical qubits for the algorithm after layout", "The {} are the product of the {} logical qubits after layout and the {} physical qubits that encode a single logical qubit.".format(T(result.physical_qubits_for_algorithm()), T(logical_qubits), T(patch.physical_qubits))))
    entries.append(_entry("physicalCountsFormatted/physicalQubitsForTfactories", "Physical T factory qubits", "Number of physical qubits for the T factories", r"Each T factory requires {} physical qubits and we run {} in parallel, therefore we need ${} = {} \cdot {}$ qubits.".format(T(factory.physical_qubits() if factory else 0), T(part.copies if part else 0), T(result.physical_qubits_for_factories()), T(factory.physical_qubits() if factory else 0), T(part.copies if part else 0))))
    entries.append(_entry("physicalCountsFormatted/requiredLogicalQubitErrorRate", "Required logical qubit error rate", "The minimum logical qubit error rate required to run the algorithm within the error budget", "The minimum logical qubit error rate is obtained by dividing the logical error probability {} by the product of {} logical qubits and the total cycle count {}.".format(fc["errorBudgetLogical"], T(logical_qubits), T(result.num_cycles))))
    entries.append(_entry("physicalCountsFormatted/requiredLogicalTstateErrorRate", "Required logical T state error rate", "The minimum T state error rate required for distilled T states", "The minimum T state error rate is obtained by dividing the T distillation error probability {} by the total number of T states {}.".format(fc["errorBudgetTstates"], T(num_tstates))))
    entries.append(_entry("physicalCountsFormatted/numTsPerRotation", "Number of T states per rotation", "Number of T states to implement a rotation with an arbitrary angle", r"The number of T states to implement a rotation with an arbitrary angle is $\lceil 0.53 \log_2({} / {}) + 4.86\rceil$ [[arXiv:2203.10064](https://arxiv.org/abs/2203.10064)].  For simplicity, we use this formula for all single-qubit arbitrary angle rotations, and do not distinguish between best, worst, and average cases.".format(T(counts.rotation_count), D(budget.rotations))))
    groups.append({"title": "Resource estimates breakdown", "alwaysVisible": False, "entries": entries})
    entries = []
    entries.append(_entry("jobParams/qecScheme/name", "QEC scheme", "Name of QEC scheme", "You can load pre-defined QEC schemes by using the name `surface_code` or `floquet_code`. The latter only works with Majorana qubits."))
    entries.append(_entry("logicalQubit/codeDistance", "Code distance", "Required code distance for error correction", r"The code distance is the smallest odd integer greater or equal to $\dfrac{{2\log({} / {})}}{{\log({}/{})}} - 1$".format(D(scheme.crossing_prefactor), D(result.required_logical_error_rate), D(scheme.error_correction_threshold), D(patch.qubit.clifford_error_rate()))))
    entries.append(_entry("physicalCountsFormatted/physicalQubitsPerLogicalQubit", "Physical qubits", "Number of physical qubits per logical qubit", "The number of physical qubits per logical qubit are evaluated using the formula {} that can be user-specified.".format(scheme.physical_qubits_per_logical_qubit)))
    entries.append(_entry("physicalCountsFormatted/logicalCycleTime", "Logical cycle time", "Duration of a logical cycle in nanoseconds", "The runtime of one logical cycle in nanoseconds is evaluated using the formula {} that can be user-specified.".format(scheme.logical_cycle_time)))
    entries.append(_entry("physicalCountsFormatted/logicalErrorRate", "Logical qubit error rate", "Logical qubit error rate", r"The logical qubit error rate is computed as ${} \cdot \left(\dfrac{{{}}}{{{}}}\right)^\frac{{{} + 1}}{{2}}$".format(D(scheme.crossing_prefactor), D(patch.qubit.clifford_error_rate()), D(scheme.error_correction_threshold), patch.code_distance)))
    entries.append(_entry("jobParams/qecScheme/crossingPrefactor", "Crossing prefactor", "Crossing prefactor used in QEC scheme", "The crossing prefactor is usually extracted numerically from simulations when fitting an exponential curve to model the relationship between logical and physical error rate."))
    entries.append(_entry("jobParams/qecScheme/errorCorrectionThreshold", "Error correction threshold", "Error correction threshold used in QEC scheme", "The error correction threshold is the physical error rate below which the error rate of the logical qubit is less than the error rate of the physical qubit that constitute it.  This value is usually extracted numerically from simulations of the logical error rate."))
    entries.append(_entry("jobParams/qecScheme/logicalCycleTime", "Logical cycle time formula", "QEC scheme formula used to compute logical cycle time", "This is the formula that is used to compute the logical cycle time {} ns.".format(T(patch.logical_cycle_time))))
    entries.append(_entry("jobParams/qecScheme/physicalQubitsPerLogicalQubit", "Physical qubits formula", "QEC scheme formula used to compute number of physical qubits per logical qubit", "This is the formula that is used to compute the number of physical qubits per logical qubits {}.".format(T(patch.physical_qubits))))
    groups.append({"title": "Logical qubit parameters", "alwaysVisible": False, "entries": entries})
    if part is not None:
        entries = []
        entries.append(_entry("physicalCountsFormatted/tfactoryPhysicalQubits", "Physical qubits", "Number of physical qubits for a single T factory", "This corresponds to the maximum number of physical qubits over all rounds of T distillation units in a T factory.  A round of distillation contains of multiple copies of distillation units to achieve the required success probability of producing a T state with the expected logical T state error rate."))
        entries.append(_entry("physicalCountsFormatted/tfactoryRuntime", "Runtime", "Runtime of a single T factory", "The runtime of a single T factory is the accumulated runtime of executing each round in a T factory."))
        entries.append(_entry("tfactory/numTstates", "Number of output T states per run", "Number of output T states produced in a single run of T factory", "The T factory takes as input {} noisy physical T states with an error rate of {} and produces {} T states with an error rate of {}.".format(T(factory.num_input_states()), D(qubit.t_gate_error_rate), T(factory.num_output_states()), fc["tstateLogicalErrorRate"])))
        entries.append(_entry("physicalCountsFormatted/numInputTstates", "Number of input T states per run", "Number of physical input T states consumed in a single run of a T factory", "This value includes the physical input T states of all copies of the distillation unit in the first round."))
        entries.append(_entry("tfactory/numRounds", "Distillation rounds", "The number of distillation rounds", "This is the number of distillation rounds.  In each round one or multiple copies of some distillation unit is executed."))
        entries.append(_entry("physicalCountsFormatted/numUnitsPerRound", "Distillation units per round", "The number of units in each round of distillation", "This is the number of copies for the distillation units per round."))
        entries.append(_entry("physicalCountsFormatted/unitNamePerRound", "Distillation units", "The types of distillation units", "These are the types of distillation units that are executed in each round.  The units can be either physical or logical, depending on what type of qubit they are operating.  Space-efficient units require fewer qubits for the cost of longer runtime compared to Reed-Muller preparation units."))
        entries.append(_entry("physicalCountsFormatted/codeDistancePerRound", "Distillation code distances", "The code distance in each round of distillation", "This is the code distance used for the units in each round.  If the code distance is 1, then the distillation unit operates on physical qubits instead of error-corrected logical qubits."))
        entries.append(_entry("physicalCountsFormatted/physicalQubitsPerRound", "Number of physical qubits per round", "The number of physical qubits used in each round of distillation", "The maximum number of physical qubits over all rounds is the number of physical qubits for the T factory, since qubits are reused by different rounds."))
        entries.append(_entry("physicalCountsFormatted/tfactoryRuntimePerRound", "Runtime per round", "The runtime of each distillation round", "The runtime of the T factory is the sum of the runtimes in all rounds."))
        entries.append(_entry("physicalCountsFormatted/tstateLogicalErrorRate", "Logical T state error rate", "Logical T state error rate", "This is the logical T state error rate achieved by the T factory which is equal or smaller than the required error rate {}.".format(fc["requiredLogicalTstateErrorRate"])))
        groups.append({"title": "T factory parameters", "alwaysVisible": False, "entries": entries})
    entries = []
    entries.append(_entry("physicalCountsFormatted/logicalCountsNumQubits", "Logical qubits (pre-layout)", "Number of logical qubits in the input quantum program", "We determine {} algorithmic logical qubits from this number by assuming to align them in a 2D grid.  Auxiliary qubits are added to allow for sufficient space to execute multi-qubit Pauli measurements on all or a subset of the logical qubits.".format(T(logical_qubits))))
    entries.append(_entry("physicalCountsFormatted/logicalCountsTCount", "T gates", "Number of T gates in the input quantum program", "This includes all T gates and adjoint T gates, but not T gates used to implement rotation gates with arbitrary angle, CCZ gates, or CCiX gates."))
    entries.append(_entry("physicalCountsFormatted/logicalCountsRotationCount", "Rotation gates", "Number of rotation gates in the input quantum program", "This is the number of all rotation gates. If an angle corresponds to a Pauli, Clifford, or T gate, it is not accounted for in this number."))
    entries.append(_entry("physicalCountsFormatted/logicalCountsRotationDepth", "Rotation depth", "Depth of rotation gates in the input quantum program", "This is the number of all non-Clifford layers that include at least one single-qubit rotation gate with an arbitrary angle."))
    entries.append(_entry("physicalCountsFormatted/logicalCountsCczCount", "CCZ gates", "Number of CCZ-gates in the input quantum program", "This is the number of CCZ gates."))
    entries.append(_entry("physicalCountsFormatted/logicalCountsCcixCount", "CCiX gates", "Number of CCiX-gates in the input quantum program", "This is the number of CCiX gates, which applies $-iX$ controlled on two control qubits [[1212.5069](https://arxiv.org/abs/1212.5069)]."))
    entries.append(_entry("physicalCountsFormatted/logicalCountsMeasurementCount", "Measurement operations", "Number of single qubit measurements in the input quantum program", "This is the number of single qubit measurements in Pauli basis that are used in the input program.  Note that all measurements are counted, however, the measurement result is is determined randomly (with a fixed seed) to be 0 or 1 with a probability of 50%."))
    groups.append({"title": "Pre-layout logical resources", "alwaysVisible": False, "entries": entries})
    entries = []
    entries.append(_entry("physicalCountsFormatted/errorBudget", "Total error budget", "Total error budget for the algorithm", r"The total error budget sets the overall allowed error for the algorithm, i.e., the number of times it is allowed to fail.  Its value must be between 0 and 1 and the default value is 0.001, which corresponds to 0.1%, and means that the algorithm is allowed to fail once in 1000 executions.  This parameter is highly application specific. For example, if one is running Shor's algorithm for factoring integers, a large value for the error budget may be tolerated as one can check that the output are indeed the prime factors of the input.  On the other hand, a much smaller error budget may be needed for an algorithm solving a problem with a solution which cannot be efficiently verified.  This budget $\epsilon = \epsilon_{\log} + \epsilon_{\rm dis} + \epsilon_{\rm syn}$ is uniformly distributed and applies to errors $\epsilon_{\log}$ to implement logical qubits, an error budget $\epsilon_{\rm dis}$ to produce T states through distillation, and an error budget $\epsilon_{\rm syn}$ to synthesize rotation gates with arbitrary angles.  Note that for distillation and rotation synthesis, the respective error budgets $\epsilon_{\rm dis}$ and $\epsilon_{\rm syn}$ are uniformly distributed among all T states and all rotation gates, respectively. If there are no rotation gates in the input algorithm, the error budget is uniformly distributed to logical errors and T state errors."))
    entries.append(_entry("physicalCountsFormatted/errorBudgetLogical", "Logical error probability", "Probability of at least one logical error", "This is one third of the total error budget {} if the input algorithm contains rotation with gates with arbitrary angles, or one half of it, otherwise.".format(fc["errorBudget"])))
    entries.append(_entry("physicalCountsFormatted/errorBudgetTstates", "T distillation error probability", "Probability of at least one faulty T distillation", "This is one third of the total error budget {} if the input algorithm contains rotation with gates with arbitrary angles, or one half of it, otherwise.".format(fc["errorBudget"])))
    entries.append(_entry("physicalCountsFormatted/errorBudgetRotations", "Rotation synthesis error probability", "Probability of at least one failed rotation synthesis", "This is one third of the total error budget {}.".format(fc["errorBudget"])))
    groups.append({"title": "Assumed error budget", "alwaysVisible": False, "entries": entries})
    entries = []
    entries.append(_entry("jobParams/qubitParams/name", "Qubit name", "Some descriptive name for the qubit model", "You can load pre-defined qubit parameters by using the names `qubit_gate_ns_e3`, `qubit_gate_ns_e4`, `qubit_gate_us_e3`, `qubit_gate_us_e4`, `qubit_maj_ns_e4`, or `qubit_maj_ns_e6`.  The names of these pre-defined qubit parameters indicate the instruction set (gate-based or Majorana), the operation speed (ns or µs regime), as well as the fidelity (e.g., e3 for $10^{-3}$ gate error rates)."))
    entries.append(_entry("jobParams/qubitParams/instructionSet", "Instruction set", "Underlying qubit technology (gate-based or Majorana)", "When modeling the physical qubit abstractions, we distinguish between two different physical instruction sets that are used to operate the qubits.  The physical instruction set can be either *gate-based* or *Majorana*.  A gate-based instruction set provides single-qubit measurement, single-qubit gates (incl. T gates), and two-qubit gates.  A Majorana instruction set provides a physical T gate, single-qubit measurement and two-qubit joint measurement operations."))
    entries.append(_entry("jobParams/qubitParams/oneQubitMeasurementTime", "Single-qubit measurement time", "Operation time for single-qubit measurement (t_meas) in ns", "This is the operation time in nanoseconds to perform a single-qubit measurement in the Pauli basis."))
    if majorana:
        entries.append(_entry("jobParams/qubitParams/twoQubitJointMeasurementTime", "Two-qubit measurement time", "Operation time for two-qubit measurement in ns", "This is the operation time in nanoseconds to perform a non-destructive two-qubit joint Pauli measurement."))
    if not majorana:
        entries.append(_entry("jobParams/qubitParams/oneQubitGateTime", "Single-qubit gate time", "Operation time for single-qubit gate (t_gate) in ns", "This is the operation time in nanoseconds to perform a single-qubit Clifford operation, e.g., Hadamard or Phase gates."))
    if not majorana:
        entries.append(_entry("jobParams/qubitParams/twoQubitGateTime", "Two-qubit gate time", "Operation time for two-qubit gate in ns", "This is the operation time in nanoseconds to perform a two-qubit Clifford operation, e.g., a CNOT or CZ gate."))
    entries.append(_entry("jobParams/qubitParams/tGateTime", "T gate time", "Operation time for a T gate", "This is the operation time in nanoseconds to execute a T gate."))
    entries.append(_entry("jobParams/qubitParams/oneQubitMeasurementErrorRate", "Single-qubit measurement error rate", "Error rate for single-qubit measurement", "This is the probability in which a single-qubit measurement in the Pauli basis may fail."))
    if majorana:
        entries.append(_entry("jobParams/qubitParams/twoQubitJointMeasurementErrorRate", "Two-qubit measurement error rate", "Error rate for two-qubit measurement", "This is the probability in which a non-destructive two-qubit joint Pauli measurement may fail."))
    if not majorana:
        entries.append(_entry("jobParams/qubitParams/oneQubitGateErrorRate", "Single-qubit error rate", "Error rate for single-qubit Clifford gate (p)", "This is the probability in which a single-qubit Clifford operation, e.g., Hadamard or Phase gates, may fail."))
    if not majorana:
        entries.append(_entry("jobParams/qubitParams/twoQubitGateErrorRate", "Two-qubit error rate", "Error rate for two-qubit Clifford gate", "This is the probability in which a two-qubit Clifford operation, e.g., CNOT or CZ gates, may fail."))
    entries.append(_entry("jobParams/qubitParams/tGateErrorRate", "T gate error rate", "Error rate to prepare single-qubit T state or apply a T gate (p_T)", "This is the probability in which executing a single T gate may fail."))
    groups.append({"title": "Physical qubit parameters", "alwaysVisible": False, "entries": entries})
    entries = []
    entries.append(_entry("physicalCountsFormatted/logicalDepthFactor", "Logical depth factor", "Factor the initial number of logical cycles is multiplied by", "This is the factor takes into account a potential overhead to the initial number of logical cycles."))
    entries.append(_entry("physicalCountsFormatted/maxTFactories", "Maximum number of T factories", "The maximum number of T factories can be utilized during the algorithm's runtime", "This is the maximum number of T factories used for producing the demanded T states, which can be created and executed by the algorithm in parallel."))
    entries.append(_entry("physicalCountsFormatted/maxDuration", "Maximum runtime duration", "The maximum runtime duration allowed for the algorithm runtime", "This is the maximum time allowed to the algorithm. If specified, the estimator targets to minimize the number of physical qubits consumed by the algorithm for runtimes under the maximum allowed."))
    entries.append(_entry("physicalCountsFormatted/maxPhysicalQubits", "Maximum number of physical qubits", "The maximum number of physical qubits allowed for utilization to the algorithm", "This is the maximum number of physical qubits available to the algorithm. If specified, the estimator targets to minimize the runtime of the algorithm with number of physical qubits consumed not exceeding this maximum."))
    groups.append({"title": "Constraints", "alwaysVisible": False, "entries": entries})
    return {"groups": groups, "assumptions": list(_REPORT_ASSUMPTIONS)}


@dataclass
class EstimationResult:
    """What the QDK reports for one estimate; `asdict` gives its JSON shape (``status``, ``jobParams``,
    ``physicalCounts``, ``physicalCountsFormatted``, ``logicalQubit``, ``tfactory``, ``errorBudget``,
    ``logicalCounts``, ``reportData``)."""

    estimator: "AzureEstimator"
    logical_patch: LogicalPatch
    num_cycles: int
    factory_part: Optional[FactoryPart]
    required_logical_error_rate: float
    error_budget: ErrorBudget

    @property
    def counts(self) -> LogicalCounts:
        return self.estimator.counts

    def physical_qubits_for_factories(self) -> int:
        return self.factory_part.physical_qubits() if self.factory_part else 0

    def physical_qubits_for_algorithm(self) -> int:
        return _ceil_div(self.counts.logical_qubits(), self.logical_patch.logical_qubits) * self.logical_patch.physical_qubits

    def physical_qubits(self) -> int:
        return self.physical_qubits_for_algorithm() + self.physical_qubits_for_factories()

    def runtime(self) -> int:
        return self.logical_patch.logical_cycle_time * self.num_cycles

    def rqops(self) -> int:
        return math.ceil(self.counts.logical_qubits() * self.logical_patch.logical_cycles_per_second())

    def physical_counts(self) -> dict:
        counts, budget, part = self.counts, self.error_budget, self.factory_part
        breakdown = {"algorithmicLogicalQubits": counts.logical_qubits(), "algorithmicLogicalDepth": counts.logical_depth(budget), "logicalDepth": self.num_cycles, "numTstates": counts.num_magic_states(budget), "clockFrequency": self.logical_patch.logical_cycles_per_second(), "numTfactories": part.copies if part else 0, "numTfactoryRuns": part.runs if part else 0, "physicalQubitsForTfactories": self.physical_qubits_for_factories(), "physicalQubitsForAlgorithm": self.physical_qubits_for_algorithm(), "requiredLogicalQubitErrorRate": self.required_logical_error_rate, "requiredLogicalTstateErrorRate": part.required_output_error_rate if part else None, "numTsPerRotation": counts.num_ts_per_rotation(budget.rotations), "cliffordErrorRate": self.logical_patch.qubit.clifford_error_rate()}
        return {"physicalQubits": self.physical_qubits(), "runtime": self.runtime(), "rqops": self.rqops(), "breakdown": breakdown}

    def formatted_counts(self) -> dict:
        """The QDK's ``physicalCountsFormatted``: the counts as its report shows them, as strings (``1 secs``,
        ``400.06k``, ``1.55e-11``, ``18, 1``, ``No T states in algorithm``)."""
        return _formatted_counts(self)

    def report_data(self, formatted: Optional[dict] = None) -> dict:
        """The QDK's ``reportData``: ``groups`` (a ``title``, ``alwaysVisible``, ``entries`` of ``path``,
        ``label``, ``description`` and ``explanation``) and ``assumptions``."""
        return _report_data(self, self.formatted_counts() if formatted is None else formatted)

    def asdict(self) -> dict:
        part = self.factory_part
        formatted = self.formatted_counts()
        out = {"status": "success", "jobParams": self.estimator.job_params(), "physicalCounts": self.physical_counts(), "physicalCountsFormatted": formatted, "logicalQubit": self.logical_patch.asdict()}
        if part is not None:
            out["tfactory"] = part.factory.asdict()
        out["errorBudget"] = self.error_budget.asdict()
        out["logicalCounts"] = self.counts.asdict()
        out["reportData"] = self.report_data(formatted)
        return out

    def frontier_entry(self) -> dict:
        """One of the QDK's ``frontierEntries`` (whose budget, unlike the single point's, names the T-state
        part ``magic_states``)."""
        part = self.factory_part
        budget = {"logical": self.error_budget.logical, "magic_states": self.error_budget.tstates, "rotations": self.error_budget.rotations}
        return {"logicalQubit": self.logical_patch.asdict(), "tfactory": part.factory.asdict() if part else None, "errorBudget": budget, "physicalCounts": self.physical_counts(), "physicalCountsFormatted": self.formatted_counts()}


class _NoFactory(Exception):
    """No factory fits; `any_factory` says whether any reached the error rate at all (then more
    cycles may fit one) or none did (then only a larger code distance can)."""

    def __init__(self, any_factory: bool) -> None:
        super().__init__()
        self.any_factory = any_factory


class AzureEstimator:
    """The QDK's `PhysicalResourceEstimation` for one algorithm: a qubit model, a QEC scheme, the distillation
    units, the logical counts and the constraints. `estimate(budget)` runs the single-point estimate."""

    def __init__(self, counts: LogicalCounts, qubit: PhysicalQubit, scheme: QECScheme, templates: Sequence[DistillationUnitTemplate], constraints: Constraints, error_budget_spec=None, estimate_type: str = "singlePoint", unit_specifications: Optional[Sequence[dict]] = None):
        self.counts = counts
        self.qubit = qubit
        self.scheme = scheme
        self.templates = list(templates)
        self.constraints = constraints
        self.error_budget_spec = error_budget_spec
        if estimate_type not in ("singlePoint", "frontier"):
            raise ValueError(f"estimateType must be 'singlePoint' or 'frontier', not {estimate_type!r}")
        self.estimate_type = estimate_type
        self.unit_specifications = list(unit_specifications or ())  # the QDK's echo of distillationUnitSpecifications

    def job_params(self) -> dict:
        out = {"qecScheme": self.scheme.asdict(), "errorBudget": _budget_echo(self.error_budget_spec), "qubitParams": self.qubit.asdict(), "constraints": self.constraints.asdict()}
        if self.unit_specifications:
            out["distillationUnitSpecifications"] = self.unit_specifications
        out["estimateType"] = self.estimate_type
        return out

    def result_dict(self, budget: ErrorBudget) -> dict:
        """The QDK's JSON: the single point's sections, or ``frontierEntries`` (and the report of the first, the
        shortest) for the frontier estimate."""
        if self.estimate_type == "frontier":
            frontier = self.build_frontier(budget)
            return {"status": "success", "jobParams": self.job_params(), "logicalCounts": self.counts.asdict(), "reportData": frontier[0].report_data(), "frontierEntries": [r.frontier_entry() for r in frontier]}
        return self.estimate(budget).asdict()

    # -- pieces --

    def find_factories(self, output_t_error_rate: float, max_code_distance: int) -> List[TFactory]:
        return find_nondominated_tfactories(self.scheme, self.qubit, self.templates, output_t_error_rate, max_code_distance, self.constraints.max_distillation_rounds)

    def patch(self, code_distance: int) -> LogicalPatch:
        return LogicalPatch.new(self.scheme, self.qubit, code_distance)

    def algorithmic_logical_depth(self, budget: ErrorBudget) -> int:
        """The layout's logical depth (`logicalDepthFactor` applied)."""
        num_cycles = self.counts.logical_depth(budget)
        factor = self.constraints.logical_depth_factor
        if factor is not None:
            if factor < 1.0:
                raise ValueError(f"logical depth scaling factor {factor} must be at least 1")
            scaled = num_cycles * factor
            num_cycles = 2**64 - 1 if not math.isfinite(scaled) else min(math.ceil(scaled), 2**64 - 1)  # the QDK's `.ceil() as u64` saturates
        return num_cycles

    def compute_num_cycles(self, budget: ErrorBudget) -> int:
        num_cycles = self.algorithmic_logical_depth(budget)
        if num_cycles == 0 and self.counts.num_magic_states(budget) == 0:
            raise ValueError("the algorithm has no logical depth and no magic states to estimate")
        return num_cycles

    def required_logical_error_rate(self, logical_budget: float, num_cycles: int) -> float:
        return logical_budget / (self.counts.logical_qubits() * num_cycles)

    def logical_cycles_for_code_distance(self, logical_budget: float, code_distance: int) -> int:
        return int(math.floor(logical_budget / (self.counts.logical_qubits() * self.scheme.logical_error_rate(self.qubit, code_distance))))

    def num_factories(self, patch: LogicalPatch, factory: TFactory, num_magic_states: int, num_cycles: int) -> int:
        total_duration = num_cycles * patch.logical_cycle_time
        if total_duration < 2**64:
            num_states_per_run = (total_duration // factory.duration()) * factory.num_output_states()
            return _ceil_div(num_magic_states, num_states_per_run)
        magic_states_per_cycle = num_magic_states / (factory.num_output_states() * num_cycles)
        return math.ceil(magic_states_per_cycle * (factory.duration() / patch.logical_cycle_time))

    def cycles_required_for_magic_states(self, num_factories: int, factory: TFactory, patch: LogicalPatch, num_magic_states: int) -> int:
        runs = _ceil_div(num_magic_states, num_factories * factory.num_output_states())
        return _ceil_div(runs * factory.duration(), patch.logical_cycle_time)

    def _max_factories_ok(self, patch: LogicalPatch, factory: TFactory, num_magic_states: int, num_cycles: int) -> bool:
        return self.constraints.max_t_factories is None or self.constraints.max_t_factories >= self.num_factories(patch, factory, num_magic_states, num_cycles)

    @staticmethod
    def _with_num_cycles(factories: Sequence[TFactory], patch: LogicalPatch, max_cycles: int) -> List[Tuple[TFactory, int]]:
        out = []
        for f in factories:
            num = _ceil_div(f.duration(), patch.logical_cycle_time)
            if num <= max_cycles:
                out.append((f, num))
        return out

    @staticmethod
    def _first_min(items, key):
        best, best_key = None, None
        for item in items:
            k = key(item)
            if best is None or k < best_key:
                best, best_key = item, k
        return best

    def find_factory(self, factories: Sequence[TFactory], patch: LogicalPatch, num_magic_states: int, min_cycles: int, max_cycles: int) -> Optional[Tuple[TFactory, int]]:
        """The factory of least normalized volume that runs within the algorithm; else the least-volume one that
        fits in `max_cycles` (the longest run the code distance allows), with the cycles it needs."""
        algorithm_duration = min_cycles * patch.logical_cycle_time
        fitting = [f for f in factories if self._max_factories_ok(patch, f, num_magic_states, min_cycles) and f.duration() <= algorithm_duration]
        best = self._first_min(fitting, lambda f: f.normalized_volume())
        if best is not None:
            return best, min_cycles
        if self.constraints.max_t_factories is None:
            candidates = self._with_num_cycles(factories, patch, max_cycles)
        else:
            candidates = []
            for f in factories:
                runs = _ceil_div(num_magic_states, self.constraints.max_t_factories * f.num_output_states())
                num = _ceil_div(runs * f.duration(), patch.logical_cycle_time)
                if num <= max_cycles:
                    candidates.append((f, num))
        return self._first_min(candidates, lambda fc: (fc[0].normalized_volume(), fc[1]))

    # -- estimates --

    def estimate(self, budget: ErrorBudget) -> EstimationResult:
        if self.constraints.max_duration is not None and self.constraints.max_physical_qubits is not None:
            raise ValueError("both maxDuration and maxPhysicalQubits provided; set at most one")
        if self.constraints.max_duration is not None:
            return self.estimate_with_max_duration(budget, self.constraints.max_duration)
        if self.constraints.max_physical_qubits is not None:
            return self.estimate_with_max_physical_qubits(budget, self.constraints.max_physical_qubits)
        return self.estimate_without_restrictions(budget)

    def estimate_without_restrictions(self, budget: ErrorBudget) -> EstimationResult:
        """The QDK's default: the code distance from the logical budget over the algorithm's depth, then the
        best factory; when none fits the depth, the depth grows until the distance and a factory agree."""
        num_cycles = self.compute_num_cycles(budget)
        last_distance_without_factories = None
        while True:
            budget = replace(budget)
            required = self.required_logical_error_rate(budget.logical, num_cycles)
            code_distance = self.scheme.compute_code_distance(self.qubit, required)
            max_allowed = self.logical_cycles_for_code_distance(budget.logical, code_distance)
            patch = self.patch(code_distance)
            num_magic_states = self.counts.num_magic_states(budget)
            if num_magic_states == 0:
                return EstimationResult(self, patch, num_cycles, None, required, budget)
            try:
                part, num_cycles_required = self._factory_part(patch, num_cycles, max_allowed, budget, num_magic_states)
            except _NoFactory as none:
                if none.any_factory or last_distance_without_factories != code_distance:
                    last_distance_without_factories = None if none.any_factory else code_distance
                    num_cycles = max(max_allowed + 1, num_cycles + 1)
                    continue
                # No factory at all at this distance, twice: the QDK adds one cycle at a time until the distance
                # moves (past 1e14 cycles the +1 no longer changes the required rate by an ulp, and it takes many
                # steps), then searches again. The cycle count where the distance moves is found directly here
                # (the distance is monotone in it: doubling, then bisection); when none below 2^64 moves it, or the
                # next distance is past the scheme's maximum, the QDK never returns and this is an error.
                no_solution = ValueError(f"no T factory reaches a T-state error rate of {budget.tstates / num_magic_states:.3g} at code distance {code_distance}: the estimate has no solution")

                def distance_at(n: int) -> int:
                    try:
                        return self.scheme.compute_code_distance(self.qubit, self.required_logical_error_rate(budget.logical, n))
                    except ValueError:
                        return self.scheme.max_code_distance + 2

                lo = hi = num_cycles
                while distance_at(hi) == code_distance:
                    lo, hi = hi, hi * 2
                    if hi >= 2**64:
                        raise no_solution from None
                while lo + 1 < hi:
                    mid = (lo + hi) // 2
                    if distance_at(mid) == code_distance:
                        lo = mid
                    else:
                        hi = mid
                if distance_at(hi) > self.scheme.max_code_distance:
                    raise no_solution from None
                num_cycles = hi
                continue
            return EstimationResult(self, patch, num_cycles_required, part, required, budget)

    def _factory_part(self, patch: LogicalPatch, min_cycles: int, max_cycles: int, budget: ErrorBudget, num_magic_states: int) -> Tuple[FactoryPart, int]:
        required_t_error = budget.tstates / num_magic_states
        factories = self.find_factories(required_t_error, patch.code_distance)
        if not factories:
            raise _NoFactory(False)
        found = self.find_factory(factories, patch, num_magic_states, min_cycles, max_cycles)
        if found is None:
            raise _NoFactory(True)
        factory, num_cycles = found
        copies = self.num_factories(patch, factory, num_magic_states, num_cycles)
        return FactoryPart.new(factory, copies, num_magic_states, required_t_error), num_cycles

    def _initial_values(self, budget: ErrorBudget):
        num_cycles = self.compute_num_cycles(budget)
        num_magic_states = self.counts.num_magic_states(budget)
        required_t_error = budget.tstates / num_magic_states if num_magic_states else math.inf
        required = self.required_logical_error_rate(budget.logical, num_cycles)
        return self.scheme.compute_code_distance(self.qubit, required), num_cycles, required, required_t_error, num_magic_states

    def estimate_with_max_duration(self, budget: ErrorBudget, max_duration: int) -> EstimationResult:
        """Fewest physical qubits among the code distances whose runtime fits `max_duration` nanoseconds."""
        min_d, num_cycles_layout, required, required_t_error, num_magic_states = self._initial_values(budget)
        if num_magic_states == 0:
            patch = self.patch(min_d)
            if num_cycles_layout * patch.logical_cycle_time <= max_duration:
                return EstimationResult(self, patch, num_cycles_layout, None, required, budget)
            raise ValueError("maxDuration is too small for this algorithm")
        best: Optional[EstimationResult] = None
        last_factories: List[TFactory] = []
        last_d: Optional[int] = None
        for d in reversed(self.scheme.code_distance_range(min_d)):
            patch = self.patch(d)
            by_duration = int(math.floor(max_duration / patch.logical_cycle_time))
            if by_duration < num_cycles_layout:
                continue
            by_error = self.logical_cycles_for_code_distance(budget.logical, d)
            if by_error < num_cycles_layout:
                continue
            max_cycles = min(by_duration, by_error)
            if last_d is None or last_d > d:
                last_factories = self.find_factories(required_t_error, d)
                last_d = max((f.max_code_distance() for f in last_factories), default=None)
            for factory, _ in self._with_num_cycles(last_factories, patch, max_cycles):
                copies = self.num_factories(patch, factory, num_magic_states, max_cycles)
                num_cycles = max(self.cycles_required_for_magic_states(copies, factory, patch, num_magic_states), num_cycles_layout)
                if self.constraints.max_t_factories is not None and copies > self.constraints.max_t_factories:
                    continue
                result = EstimationResult(self, patch, num_cycles, FactoryPart.new(factory, copies, num_magic_states, required_t_error), required, budget)
                if best is None or result.physical_qubits() < best.physical_qubits():
                    best = result
        if best is None:
            raise ValueError("maxDuration is too small for this algorithm")
        return best

    def build_frontier(self, budget: ErrorBudget) -> List[EstimationResult]:
        """The QDK's frontier estimate: for every code distance from the smallest that meets the logical
        budget up to the scheme's maximum, every factory of the frontier and every number of copies from the
        fewest that finish within the distance's allowed cycles, the Pareto frontier of (physical qubits,
        runtime); sorted by runtime, shortest first."""
        if self.constraints.max_duration is not None or self.constraints.max_physical_qubits is not None or self.constraints.max_t_factories is not None:
            raise ValueError("constraints maxDuration, maxPhysicalQubits and maxTFactories cannot be combined with the frontier estimate")
        min_d, min_cycles, required, required_t_error, num_magic_states = self._initial_values(budget)
        if num_magic_states == 0:
            return [EstimationResult(self, self.patch(min_d), min_cycles, None, required, budget)]
        population = _Population()
        last_factories: List[TFactory] = []
        last_d: Optional[int] = None
        for d in reversed(self.scheme.code_distance_range(min_d)):
            if last_d is None or last_d > d:
                last_factories = self.find_factories(required_t_error, d)
                last_d = max((f.max_code_distance() for f in last_factories), default=None)
            patch = self.patch(d)
            max_cycles = self.logical_cycles_for_code_distance(budget.logical, d)
            if max_cycles < min_cycles:
                continue
            for factory, _ in self._with_num_cycles(last_factories, patch, max_cycles):
                copies = self.num_factories(patch, factory, num_magic_states, max_cycles)
                while True:
                    for_magic_states = self.cycles_required_for_magic_states(copies, factory, patch, num_magic_states)
                    num_cycles = max(for_magic_states, min_cycles)
                    part = FactoryPart.new(factory, copies, num_magic_states, required_t_error)
                    result = EstimationResult(self, patch, num_cycles, part, required, budget)
                    population.push(_Point2D(result, float(result.physical_qubits()), result.runtime()))
                    if for_magic_states <= min_cycles or part.runs <= 1:
                        break
                    copies += 1
        population.filter_out_dominated()
        population.sort_items()
        if not population.items:
            raise ValueError(f"no T factory reaches a T-state error rate of {required_t_error:.3g} at any code distance: the frontier is empty (the QDK panics here)")
        return sorted((p.item for p in population.items), key=EstimationResult.runtime)

    def estimate_with_max_physical_qubits(self, budget: ErrorBudget, max_qubits: int) -> EstimationResult:
        """Shortest runtime among the code distances whose qubits (algorithm plus as many factories as fit)
        stay within `max_qubits`."""
        min_d, num_cycles_layout, required, required_t_error, num_magic_states = self._initial_values(budget)
        if num_magic_states == 0:
            patch = self.patch(min_d)
            if _ceil_div(self.counts.logical_qubits(), patch.logical_qubits) * patch.physical_qubits <= max_qubits:
                return EstimationResult(self, patch, num_cycles_layout, None, required, budget)
            raise ValueError("maxPhysicalQubits is too small for this algorithm")
        best: Optional[EstimationResult] = None
        last_factories: List[TFactory] = []
        last_d: Optional[int] = None
        for d in reversed(self.scheme.code_distance_range(min_d)):
            patch = self.patch(d)
            for_algorithm = _ceil_div(self.counts.logical_qubits(), patch.logical_qubits) * patch.physical_qubits
            if max_qubits <= for_algorithm:
                continue
            allowed = max_qubits - for_algorithm
            by_error = self.logical_cycles_for_code_distance(budget.logical, d)
            if by_error < num_cycles_layout:
                continue
            if last_d is None or last_d > d:
                last_factories = self.find_factories(required_t_error, d)
                last_d = max((f.max_code_distance() for f in last_factories), default=None)
            factory = self._first_min([f for f in last_factories if f.physical_qubits() <= allowed], lambda f: f.normalized_volume())
            if factory is None:
                continue
            copies = allowed // factory.physical_qubits()
            if copies == 0:
                continue
            num_cycles = max(self.cycles_required_for_magic_states(copies, factory, patch, num_magic_states), num_cycles_layout)
            if num_cycles > by_error:
                continue
            if self.constraints.max_t_factories is not None and copies > self.constraints.max_t_factories:
                continue
            result = EstimationResult(self, patch, num_cycles, FactoryPart.new(factory, copies, num_magic_states, required_t_error), required, budget)
            if best is None or result.runtime() < best.runtime():
                best = result
        if best is None:
            raise ValueError("maxPhysicalQubits is too small for this algorithm")
        return best


def azure_estimator(counts: Union[LogicalCounts, dict], qubit_params=None, qec_scheme_spec=None, constraints=None, distillation_unit_specifications=None, error_budget=1e-3, estimate_type: str = "singlePoint") -> AzureEstimator:
    """An `AzureEstimator` from QDK-style parameters (see `azure_estimate`)."""
    if not isinstance(counts, LogicalCounts):
        counts = LogicalCounts.fromdict(counts)
    qubit = physical_qubit(qubit_params)
    scheme = qec_scheme(qec_scheme_spec, qubit)
    if distillation_unit_specifications is not None and not isinstance(distillation_unit_specifications, (list, tuple)):
        raise ValueError(f"distillationUnitSpecifications is a list of unit specifications, not {distillation_unit_specifications!r}")
    templates = DistillationUnitTemplate.defaults() if not distillation_unit_specifications else [DistillationUnitTemplate.fromspec(s) for s in distillation_unit_specifications]
    echo = [_unit_specification_echo(s) for s in distillation_unit_specifications or ()]
    return AzureEstimator(counts, qubit, scheme, templates, Constraints.fromspec(constraints), error_budget, estimate_type, echo)


def azure_estimate(counts: Union[LogicalCounts, dict], qubit_params: Union[str, dict, GateBasedQubit, MajoranaQubit, None] = None, qec_scheme: Union[str, dict, QECScheme, None] = None, error_budget: Union[float, dict, ErrorBudget] = 1e-3, constraints: Union[dict, Constraints, None] = None, distillation_unit_specifications: Optional[Sequence[Union[str, dict, DistillationUnitTemplate]]] = None, estimate_type: str = "singlePoint") -> dict:
    """The QDK's physical resource estimate from logical counts, as its result dict: ``status``, ``jobParams``,
    ``physicalCounts`` (``physicalQubits``, ``runtime`` in ns, ``rqops``, a ``breakdown``),
    ``physicalCountsFormatted`` (the same as the report's strings), ``logicalQubit``, ``tfactory``,
    ``errorBudget``, ``logicalCounts`` and ``reportData`` (the report's groups of labelled, explained entries);
    with ``estimate_type="frontier"``, ``frontierEntries`` (each with those sections) along the qubits–runtime
    Pareto frontier instead, and the report of the first. Parameters are the QDK's: a
    qubit preset name (``qubit_gate_ns_e3``, ``qubit_gate_ns_e4``, ``qubit_gate_us_e3``, ``qubit_gate_us_e4``,
    ``qubit_maj_ns_e4``, ``qubit_maj_ns_e6``) or ``qubitParams`` dict; ``surface_code`` or ``floquet_code`` or a
    ``qecScheme`` dict; a total error budget or ``{logical, tStates, rotations}``; ``constraints``
    (``logicalDepthFactor``, ``maxTFactories``, ``maxDuration``, ``maxPhysicalQubits``,
    ``maxDistillationRounds``); ``distillationUnitSpecifications``."""
    estimator = azure_estimator(counts, qubit_params, qec_scheme, constraints, distillation_unit_specifications, error_budget, estimate_type)
    budget = ErrorBudget.partition(error_budget, estimator.counts)
    return estimator.result_dict(budget)


# ---------------------------------------------------------------------------------------------
# The QDK's Python entry point and result (qdk/estimator/_estimator.py: LogicalCounts.estimate, EstimatorResult)
# ---------------------------------------------------------------------------------------------


def _estimate_item(counts: LogicalCounts, params: dict) -> dict:
    """One item of the QDK's job parameters (``qubitParams``, ``qecScheme``, ``errorBudget``, ``constraints``,
    ``distillationUnitSpecifications``, ``estimateType``; other keys ignored, as the QDK's serde ignores them)."""
    if not isinstance(params, dict):
        raise ValueError(f"estimator parameters are a dict, not {params!r}")
    return azure_estimate(counts, params.get("qubitParams"), params.get("qecScheme"), params.get("errorBudget", 1e-3), params.get("constraints"), params.get("distillationUnitSpecifications"), params.get("estimateType", "singlePoint"))


def _lookup(result: dict, path: str):
    """A report entry's value: its path's keys followed into the result (into a frontier's first entry where the
    top level lacks one, as the QDK's tables do); ``N/A`` where the path ends."""
    value = result
    for key in path.split("/"):
        if isinstance(value, dict) and key not in value and "frontierEntries" in value:
            value = value["frontierEntries"][0]
        if not isinstance(value, dict) or key not in value:
            return "N/A"
        value = value[key]
    return value


def _text_table(header: Optional[List[str]], sections: List[Tuple[str, List[List[str]]]]) -> str:
    """Columns padded to their widest cell, each section under its title."""
    rows = [r for _, rs in sections for r in rs] + ([header] if header else [])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def line(r):
        return ("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths))).rstrip()

    out = [line(header)] if header else []
    for title, rs in sections:
        out.append(title)
        out.extend(line(r) for r in rs)
    return "\n".join(out)


class EstimatorResult(dict):
    """The QDK's `EstimatorResult`, its data: one estimate's result dict (as `azure_estimate` gives it), or a
    batch's, indexed 0, 1, … by the items' results (a failed item holds its ``message``, and the error in
    `error`). A key a frontier estimate's result lacks is looked up in its first (shortest) entry, as the QDK's
    is. `summary` is the QDK's summary table (and a batch's side-by-side table) as text; the HTML of
    ``_repr_html_``, the diagrams and the plot are not ported."""

    MAX_DEFAULT_ITEMS_IN_TABLE = 5

    def __init__(self, data: Union[dict, List[Union[dict, "EstimatorResult"]]], error: Optional[BaseException] = None):
        self._error = error
        if isinstance(data, list) and len(data) == 1 and not isinstance(data[0], EstimatorResult):
            data = data[0]
        if isinstance(data, dict):
            self._data = data
            self._is_simple = True
            super().__init__(data)
        else:
            items = [d if isinstance(d, EstimatorResult) else EstimatorResult(d) for d in data]
            self._data = [item.data() for item in items]
            self._is_simple = False
            super().__init__(enumerate(items))

    @staticmethod
    def _is_succeeded(data: dict) -> bool:
        return data.get("status") == "success"

    def data(self, idx: Optional[int] = None):
        """The raw result (a dict, or a batch's list of them); with `idx`, a batch's item."""
        if idx is None:
            return self._data
        if self._is_simple:
            raise ValueError("Cannot pass parameter 'idx' to 'data' for non-batching job")
        return self._data[idx]

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    @property
    def logical_counts(self) -> LogicalCounts:
        return LogicalCounts.fromdict(self.data()["logicalCounts"] if self._is_simple else self.data(0)["logicalCounts"])

    @property
    def json(self) -> str:
        return json.dumps(self._data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            if self._is_simple:
                raise ValueError("Cannot pass slice to '__getitem__' for non-batching job")
            return self._batch_table(range(len(self))[key])
        if super().__contains__(key):
            return super().__getitem__(key)
        if super().__contains__("frontierEntries"):
            return super().__getitem__("frontierEntries")[0][key]
        raise KeyError(key)

    @property
    def summary(self) -> str:
        """The summary table as text: each report group's entries as label, value and description; for a batch,
        the items side by side (the first five; slice the result, ``result[:]``, for all)."""
        if not self._is_simple:
            n = min(len(self), self.MAX_DEFAULT_ITEMS_IN_TABLE)
            note = f"(the first {n} of {len(self)} items; result[:] shows them all)\n" if len(self) > n else ""
            return note + self._batch_table(range(n))
        if self._error is not None:
            raise self._error
        groups = self["reportData"]["groups"]
        return _text_table(None, [(g["title"], [[e["label"], str(_lookup(self, e["path"])), e["description"]] for e in g["entries"]]) for g in groups])

    def _batch_table(self, indices) -> str:
        """The QDK's side-by-side table: the rows of the first succeeded item's groups, each label once, with
        every item's value (``N/A`` for a failed item or a path it lacks)."""
        items = {i: dict.__getitem__(self, i) for i in indices}
        succeeded = [i for i, item in items.items() if self._is_succeeded(item)]
        if not succeeded:
            return "None of the jobs succeeded"
        sections = []
        for group_index, group in enumerate(items[succeeded[0]]["reportData"]["groups"]):
            rows, seen = [], set()
            for i in succeeded:
                for entry in items[i]["reportData"]["groups"][group_index]["entries"]:
                    if entry["label"] in seen:
                        continue
                    seen.add(entry["label"])
                    rows.append([entry["label"]] + [str(_lookup(item, entry["path"])) if j in succeeded else "N/A" for j, item in items.items()])
            sections.append((group["title"], rows))
        return _text_table(["Item"] + [str(i) for i in items], sections)

    def summary_data_frame(self, labels: Optional[List] = None):
        """The QDK's batch overview as a pandas DataFrame (pandas needed), a row per item: logical qubits,
        logical depth, T states, code distance, T factories, T factory fraction, physical qubits, rQOPS and
        runtime, as the report writes them (a frontier item's first entry; a failed one "No solution found")."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Missing optional 'pandas' dependency. To install run: pip install pandas") from None
        items = [self] if self._is_simple else [dict.__getitem__(self, i) for i in range(len(self))]
        labels = list(labels or [])
        labels = (labels + list(range(len(labels), len(items))))[: len(items)]

        def row(result):
            if not self._is_succeeded(result):
                return ["No solution found"] * 9
            formatted = result["physicalCountsFormatted"]
            return [formatted["algorithmicLogicalQubits"], formatted["logicalDepth"], formatted["numTstates"], result["logicalQubit"]["codeDistance"], formatted["numTfactories"], formatted["physicalQubitsForTfactoriesPercentage"], formatted["physicalQubits"], formatted["rqops"], formatted["runtime"]]

        columns = ["Logical qubits", "Logical depth", "T states", "Code distance", "T factories", "T factory fraction", "Physical qubits", "rQOPS", "Physical runtime"]
        return pd.DataFrame([row(r) for r in items], columns=columns, index=labels)
