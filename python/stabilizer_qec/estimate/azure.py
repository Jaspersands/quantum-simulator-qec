# Ported from the Azure Quantum Resource Estimator (microsoft/qdk, source/resource_estimator/src, the `estimates`
# and `system` modules; Copyright (c) Microsoft Corporation, MIT License; see LICENSE-qdk in this directory),
# modified for stabilizer_qec: Python dataclasses in place of the Rust structs and traits, a restricted `ast`
# evaluator in place of the compiled expressions, the binomial quantile of the `probability` crate (0.20) written
# out, the frontier estimate type and the profiling output left out. The numbers are the QDK's bit for bit on
# its own test cases (tests/estimate_port/qdk_estimates.json).
"""Azure's Resource Estimator: from logical counts (qubits, T, rotations, CCZ, measurements) to physical
qubits and runtime on a chosen qubit model and QEC scheme, with the T factories found by the QDK's exhaustive
search over distillation rounds and code distances.

>>> from stabilizer_qec.estimate import LogicalCounts, azure_estimate
>>> r = azure_estimate(LogicalCounts(num_qubits=100, t_count=100_000, ccz_count=10_000, measurement_count=10_000))
>>> r["physicalCounts"]["physicalQubits"], r["physicalCounts"]["runtime"], r["logicalQubit"]["codeDistance"]
(400060, 1064000000, 19)
>>> r["tfactory"]["numUnitsPerRound"], r["tfactory"]["codeDistancePerRound"]
([18, 1], [5, 15])
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

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


def _round_half_away(x: float) -> int:
    """`f64::round`."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def evaluate_expression(expression: str, context: Dict[str, float]) -> float:
    """A QEC scheme's or distillation unit's formula (`+ - * / ^`, numbers, the named variables) evaluated in
    floating point."""
    tree = ast.parse(expression.replace("^", "**"), mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in context:
                return float(context[node.id])
            raise ValueError(f"unknown variable {node.id!r} in {expression!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported syntax in {expression!r}")

    return ev(tree)


_TIME_RE = re.compile(r"^(\+?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*(s|ms|μs|µs|us|ns)$")


def parse_time(value: Union[int, float, str, None]) -> Optional[int]:
    """A duration in nanoseconds: an int, or a string such as ``"100 ns"``, ``"1 µs"``, ``"2 ms"``, ``"1 s"``."""
    if value is None or isinstance(value, bool):
        return None if value is None else int(value)
    if isinstance(value, (int, float)):
        return int(value)
    m = _TIME_RE.match(value.strip())
    if m is None:
        raise ValueError(f"cannot parse time {value!r}")
    multiplier = {"s": 1e9, "ms": 1e6, "μs": 1e3, "µs": 1e3, "us": 1e3, "ns": 1.0}[m.group(3)]
    return _round_half_away(float(m.group(1)) * multiplier)


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
        n_m_x = n - xf
        ln_c = _stirlerr(n) - _stirlerr(xf) - _stirlerr(n_m_x) - _ln_d0(xf, self.np) - _ln_d0(n_m_x, self.nq)
        return math.exp(ln_c) * math.sqrt(n / (2.0 * math.pi * xf * n_m_x))

    def distribution(self, x: float) -> float:
        if x < 0.0:
            return 0.0
        x = int(x)
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
        q = float(self.modes()[0])
        alpha = 1.0
        while True:
            delta = alpha * (p - self.distribution(q)) / self.mass(int(q))
            if abs(delta) < 0.5:
                return int(q)
            q += delta
            alpha *= 0.999


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
            value = (float(value.get("process", _NAN)), float(value.get("readout", _NAN)))
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
        value = evaluate_expression(self.physical_qubits_per_logical_qubit, {"codeDistance": code_distance})
        if value <= 0.0:
            raise ValueError(f"non-positive number of physical qubits per logical qubit at code distance {code_distance}")
        return int(value)

    def cycle_time(self, qubit: PhysicalQubit, code_distance: int) -> int:
        value = evaluate_expression(self.logical_cycle_time, qubit.expression_context(code_distance))
        if value <= 0.0:
            raise ValueError(f"non-positive logical cycle time at code distance {code_distance}")
        return _round_half_away(value)

    def logical_error_rate(self, qubit: PhysicalQubit, code_distance: int) -> float:
        p = self.physical_error_rate(qubit)
        if p > self.error_correction_threshold:
            raise ValueError(f"invalid value for 'physical_error_rate', expected value between 0 and {self.error_correction_threshold}")
        return self.crossing_prefactor * float(code_distance**self.distance_coefficient_power) * _powi(p / self.error_correction_threshold, (code_distance + 1) // 2)

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
        spec = dict(spec)
        name = spec.pop("name", "surface_code")
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
    for d in scheme.code_distance_range(3):
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

    @staticmethod
    def fromdict(d: dict) -> "LogicalCounts":
        keys = {"numQubits": "num_qubits", "tCount": "t_count", "rotationCount": "rotation_count", "rotationDepth": "rotation_depth", "cczCount": "ccz_count", "ccixCount": "ccix_count", "measurementCount": "measurement_count"}
        out = {}
        for k, v in d.items():
            if k in keys:
                out[keys[k]] = int(v)
            elif k in keys.values():
                out[k] = int(v)
            elif k not in ("numComputeQubits", "readFromMemoryCount", "writeToMemoryCount"):
                raise ValueError(f"unknown logical count {k!r}")
        return LogicalCounts(**out)

    def asdict(self) -> dict:
        return {"numQubits": self.num_qubits, "tCount": self.t_count, "rotationCount": self.rotation_count, "rotationDepth": self.rotation_depth, "cczCount": self.ccz_count, "ccixCount": self.ccix_count, "measurementCount": self.measurement_count}

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
            logical, tstates, rotations = float(spec["logical"]), float(spec.get("tStates", spec.get("tstates", 0.0))), float(spec.get("rotations", 0.0))
            if logical + tstates + rotations >= 1.0 or logical <= 0.0 or tstates < 0.0 or (counts.has_tgates() and tstates == 0.0) or rotations < 0.0 or (counts.has_rotations() and rotations == 0.0):
                raise ValueError(f"invalid error budget {spec!r}")
            return ErrorBudget(logical, tstates, rotations)
        total = float(spec)
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
        if set(spec) == {"name"}:
            return DistillationUnitTemplate.from_name(spec["name"])
        if spec.get("numInputTs", 0) == 0:
            raise ValueError("The number of input t states must be greater than 0.")
        if spec.get("numOutputTs", 0) == 0:
            raise ValueError("The number of output t states must be greater than 0.")

        def resources(key):
            r = spec.get(key)
            if r is None:
                return None
            if r["numUnitQubits"] == 0:
                raise ValueError("The number of unit qubits must be greater than 0.")
            return (int(r["numUnitQubits"]), int(r["durationInQubitCycleTime"]))

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
        return DistillationUnitTemplate(spec["displayName"], int(spec["numInputTs"]), int(spec["numOutputTs"]), _formula(spec["failureProbabilityFormula"]), _formula(spec["outputErrorRateFormula"]), unit_type, physical, logical, override)


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
        keys = {"logicalDepthFactor": "logical_depth_factor", "maxTFactories": "max_t_factories", "maxDuration": "max_duration", "maxPhysicalQubits": "max_physical_qubits", "maxDistillationRounds": "max_distillation_rounds"}
        out = {}
        for k, v in spec.items():
            attr = keys.get(k, k if k in keys.values() else None)
            if attr is None:
                raise ValueError(f"unknown constraint {k!r}")
            out[attr] = parse_time(v) if attr == "max_duration" else v
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


@dataclass
class EstimationResult:
    """What the QDK reports for one estimate; `asdict` gives its JSON shape (``physicalCounts``,
    ``logicalQubit``, ``tfactory``, ``errorBudget``, ``logicalCounts``, ``jobParams``)."""

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

    def asdict(self) -> dict:
        part = self.factory_part
        out = {"status": "success", "jobParams": self.estimator.job_params(), "physicalCounts": self.physical_counts(), "logicalQubit": self.logical_patch.asdict()}
        if part is not None:
            out["tfactory"] = part.factory.asdict()
        out["errorBudget"] = self.error_budget.asdict()
        out["logicalCounts"] = self.counts.asdict()
        return out

    def frontier_entry(self) -> dict:
        """One of the QDK's ``frontierEntries`` (whose budget, unlike the single point's, names the T-state
        part ``magic_states``)."""
        part = self.factory_part
        budget = {"logical": self.error_budget.logical, "magic_states": self.error_budget.tstates, "rotations": self.error_budget.rotations}
        return {"logicalQubit": self.logical_patch.asdict(), "tfactory": part.factory.asdict() if part else None, "errorBudget": budget, "physicalCounts": self.physical_counts()}


class _NoFactory(Exception):
    pass


class AzureEstimator:
    """The QDK's `PhysicalResourceEstimation` for one algorithm: a qubit model, a QEC scheme, the distillation
    units, the logical counts and the constraints. `estimate(budget)` runs the single-point estimate."""

    def __init__(self, counts: LogicalCounts, qubit: PhysicalQubit, scheme: QECScheme, templates: Sequence[DistillationUnitTemplate], constraints: Constraints, error_budget_spec=None, estimate_type: str = "singlePoint"):
        self.counts = counts
        self.qubit = qubit
        self.scheme = scheme
        self.templates = list(templates)
        self.constraints = constraints
        self.error_budget_spec = error_budget_spec
        if estimate_type not in ("singlePoint", "frontier"):
            raise ValueError(f"estimateType must be 'singlePoint' or 'frontier', not {estimate_type!r}")
        self.estimate_type = estimate_type

    def job_params(self) -> dict:
        return {"qecScheme": self.scheme.asdict(), "errorBudget": self.error_budget_spec, "qubitParams": self.qubit.asdict(), "constraints": self.constraints.asdict(), "estimateType": self.estimate_type}

    def result_dict(self, budget: ErrorBudget) -> dict:
        """The QDK's JSON: the single point's sections, or ``frontierEntries`` for the frontier estimate."""
        if self.estimate_type == "frontier":
            return {"status": "success", "jobParams": self.job_params(), "logicalCounts": self.counts.asdict(), "frontierEntries": [r.frontier_entry() for r in self.build_frontier(budget)]}
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
            num_cycles = math.ceil(num_cycles * factor)
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
            except _NoFactory:
                num_cycles = max(max_allowed + 1, num_cycles + 1)
                continue
            return EstimationResult(self, patch, num_cycles_required, part, required, budget)

    def _factory_part(self, patch: LogicalPatch, min_cycles: int, max_cycles: int, budget: ErrorBudget, num_magic_states: int) -> Tuple[FactoryPart, int]:
        required_t_error = budget.tstates / num_magic_states
        factories = self.find_factories(required_t_error, patch.code_distance)
        if not factories:
            raise _NoFactory()
        found = self.find_factory(factories, patch, num_magic_states, min_cycles, max_cycles)
        if found is None:
            raise _NoFactory()
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
    templates = DistillationUnitTemplate.defaults() if not distillation_unit_specifications else [DistillationUnitTemplate.fromspec(s) for s in distillation_unit_specifications]
    return AzureEstimator(counts, qubit, scheme, templates, Constraints.fromspec(constraints), error_budget, estimate_type)


def azure_estimate(counts: Union[LogicalCounts, dict], qubit_params: Union[str, dict, GateBasedQubit, MajoranaQubit, None] = None, qec_scheme: Union[str, dict, QECScheme, None] = None, error_budget: Union[float, dict, ErrorBudget] = 1e-3, constraints: Union[dict, Constraints, None] = None, distillation_unit_specifications: Optional[Sequence[Union[str, dict, DistillationUnitTemplate]]] = None, estimate_type: str = "singlePoint") -> dict:
    """The QDK's physical resource estimate from logical counts, as its result dict: ``physicalCounts``
    (``physicalQubits``, ``runtime`` in ns, ``rqops``, a ``breakdown``), ``logicalQubit``, ``tfactory``,
    ``errorBudget``, ``logicalCounts``, ``jobParams``; with ``estimate_type="frontier"``, ``frontierEntries``
    (each with those sections) along the qubits–runtime Pareto frontier instead. Parameters are the QDK's: a
    qubit preset name (``qubit_gate_ns_e3``, ``qubit_gate_ns_e4``, ``qubit_gate_us_e3``, ``qubit_gate_us_e4``,
    ``qubit_maj_ns_e4``, ``qubit_maj_ns_e6``) or ``qubitParams`` dict; ``surface_code`` or ``floquet_code`` or a
    ``qecScheme`` dict; a total error budget or ``{logical, tStates, rotations}``; ``constraints``
    (``logicalDepthFactor``, ``maxTFactories``, ``maxDuration``, ``maxPhysicalQubits``,
    ``maxDistillationRounds``); ``distillationUnitSpecifications``."""
    estimator = azure_estimator(counts, qubit_params, qec_scheme, constraints, distillation_unit_specifications, error_budget, estimate_type)
    budget = ErrorBudget.partition(error_budget, estimator.counts)
    return estimator.result_dict(budget)
