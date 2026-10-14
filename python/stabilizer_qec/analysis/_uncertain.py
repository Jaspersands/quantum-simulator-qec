"""First-order (linear) propagation of uncertainty: the ``uncertainties`` package when it is
installed, otherwise a value that carries its derivatives with respect to independent standard
normal components, which is how ``uncertainties`` propagates them. ``ufloat``,
``correlated_values``, ``umath`` (``exp``, ``log``) and ``unumpy`` (``uarray``, ``log``,
``std_devs``, ``nominal_values``)."""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import numpy as np

_ids = itertools.count()


class UFloat:
    """A value and its linear dependence on independent unit-variance components."""

    __slots__ = ("nominal_value", "_d")

    def __init__(self, nominal_value: float, derivatives: Dict[int, float]) -> None:
        self.nominal_value = float(nominal_value)
        self._d = derivatives

    @property
    def std_dev(self) -> float:
        return math.sqrt(sum(c * c for c in self._d.values()))

    n = property(lambda self: self.nominal_value)
    s = property(lambda self: self.std_dev)

    def __repr__(self) -> str:
        return f"{self.nominal_value!r}+/-{self.std_dev!r}"

    def __float__(self) -> float:
        return self.nominal_value

    # Chain rule: f(x) has derivatives f'(x) * dx.
    def _apply(self, value: float, slope: float) -> "UFloat":
        return UFloat(value, {k: slope * c for k, c in self._d.items()})

    @staticmethod
    def _combine(value: float, a: "UFloat", da: float, b: Any, db: float) -> "UFloat":
        out = {k: da * c for k, c in a._d.items()}
        if isinstance(b, UFloat):
            for k, c in b._d.items():
                out[k] = out.get(k, 0.0) + db * c
        return UFloat(value, out)

    def __neg__(self) -> "UFloat":
        return self._apply(-self.nominal_value, -1.0)

    def __pos__(self) -> "UFloat":
        return self

    def __add__(self, other: Any) -> "UFloat":
        o = _nominal(other)
        return UFloat._combine(self.nominal_value + o, self, 1.0, other, 1.0)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "UFloat":
        o = _nominal(other)
        return UFloat._combine(self.nominal_value - o, self, 1.0, other, -1.0)

    def __rsub__(self, other: Any) -> "UFloat":
        return (-self) + other

    def __mul__(self, other: Any) -> "UFloat":
        o = _nominal(other)
        return UFloat._combine(self.nominal_value * o, self, o, other, self.nominal_value)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "UFloat":
        o = _nominal(other)
        return UFloat._combine(self.nominal_value / o, self, 1 / o, other, -self.nominal_value / (o * o))

    def __rtruediv__(self, other: Any) -> "UFloat":
        x = self.nominal_value
        return self._apply(other / x, -other / (x * x))

    def __pow__(self, other: Any) -> "UFloat":
        x, y = self.nominal_value, _nominal(other)
        value = x**y
        dx = y * x ** (y - 1) if y != 0 else 0.0
        dy = value * math.log(x) if isinstance(other, UFloat) and x > 0 else 0.0
        return UFloat._combine(value, self, dx, other, dy)

    def __rpow__(self, other: Any) -> "UFloat":
        value = other**self.nominal_value
        return self._apply(value, value * math.log(other))


def _nominal(x: Any) -> float:
    return x.nominal_value if isinstance(x, UFloat) else float(x)


def _ufloat(nominal_value: float, std_dev: float) -> UFloat:
    return UFloat(nominal_value, {next(_ids): float(std_dev)})


def _correlated_values(values: Sequence[float], covariance: Any) -> list:
    """Values with the given covariance, as ``uncertainties.correlated_values``."""
    cov = np.asarray(covariance, dtype=float)
    # cov = V diag(w) V^T: value i depends on independent component j as V[i, j] sqrt(w[j]).
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    ids = [next(_ids) for _ in range(len(w))]
    return [UFloat(values[i], {ids[j]: float(v[i, j] * math.sqrt(w[j])) for j in range(len(w))}) for i in range(len(values))]


def _exp(x: Any) -> Any:
    if isinstance(x, UFloat):
        value = math.exp(x.nominal_value)
        return x._apply(value, value)
    return math.exp(x)


def _log(x: Any) -> Any:
    if isinstance(x, UFloat):
        return x._apply(math.log(x.nominal_value), 1 / x.nominal_value)
    return math.log(x)


def _uarray(nominal: Any, std: Any) -> np.ndarray:
    return np.array([_ufloat(n, s) for n, s in zip(np.ravel(nominal), np.ravel(std))], dtype=object).reshape(np.shape(nominal))


def _ulog(a: Any) -> np.ndarray:
    return np.vectorize(_log, otypes=[object])(a)


def _std_devs(a: Any) -> np.ndarray:
    return np.vectorize(lambda x: x.std_dev if isinstance(x, UFloat) else 0.0, otypes=[float])(a)


def _nominal_values(a: Any) -> np.ndarray:
    return np.vectorize(_nominal, otypes=[float])(a)


try:  # pragma: no cover - depends on the environment
    from uncertainties import correlated_values, ufloat, umath, unumpy  # type: ignore[no-redef]
except ImportError:
    ufloat, correlated_values = _ufloat, _correlated_values
    umath = SimpleNamespace(exp=_exp, log=_log)
    unumpy = SimpleNamespace(uarray=_uarray, log=_ulog, std_devs=_std_devs, nominal_values=_nominal_values)
