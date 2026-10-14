"""lmfit's ``Model`` when lmfit is installed; otherwise a least-squares model with the part of
its interface that the fits here use (parameter hints, ``make_params``, ``fit`` and a result's
``params``), and ``ufloat`` from uncertainties or a stand-in."""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Dict, Optional

import numpy as np

from stabilizer_qec.analysis._lsq import _jacobian, _levenberg_marquardt

try:  # pragma: no cover - depends on the environment
    import lmfit as _lmfit
except ImportError:
    _lmfit = None

try:  # pragma: no cover - depends on the environment
    from uncertainties import ufloat
except ImportError:

    class UFloat:
        """A value and its standard deviation (``uncertainties.ufloat``'s ``n`` and ``s``)."""

        __slots__ = ("nominal_value", "std_dev")

        def __init__(self, nominal_value: float, std_dev: float) -> None:
            self.nominal_value = float(nominal_value)
            self.std_dev = float(std_dev)

        n = property(lambda self: self.nominal_value)
        s = property(lambda self: self.std_dev)

        def __repr__(self) -> str:
            return f"{self.nominal_value!r}+/-{self.std_dev!r}"

        def __float__(self) -> float:
            return self.nominal_value

    def ufloat(nominal_value: float, std_dev: float) -> "UFloat":
        return UFloat(nominal_value, std_dev)


class Parameter:
    """A fitted parameter: ``value``, ``stderr``, bounds and whether it varies."""

    def __init__(self, name: str, value: float = -math.inf, min: float = -math.inf, max: float = math.inf, vary: bool = True) -> None:
        self.name, self.min, self.max, self.vary = name, float(min), float(max), bool(vary)
        self.value = float(value)
        self.stderr: Optional[float] = None

    def __repr__(self) -> str:
        err = "" if self.stderr is None else f" +/- {self.stderr:.3g}"
        return f"<Parameter {self.name!r}, value={self.value!r}{err}, bounds=[{self.min}:{self.max}]>"

    # lmfit's transforms between a bounded value and an unbounded one (MINUIT's).
    def to_internal(self, x: float) -> float:
        lo, hi = self.min, self.max
        if math.isfinite(lo) and math.isfinite(hi):
            return math.asin(max(-1.0, min(1.0, 2 * (x - lo) / (hi - lo) - 1)))
        if math.isfinite(lo):
            return math.sqrt(max(0.0, (x - lo + 1) ** 2 - 1))
        if math.isfinite(hi):
            return math.sqrt(max(0.0, (hi - x + 1) ** 2 - 1))
        return x

    def from_internal(self, u: float) -> float:
        lo, hi = self.min, self.max
        if math.isfinite(lo) and math.isfinite(hi):
            return lo + (math.sin(u) + 1) * (hi - lo) / 2
        if math.isfinite(lo):
            return lo - 1 + math.sqrt(u * u + 1)
        if math.isfinite(hi):
            return hi + 1 - math.sqrt(u * u + 1)
        return u


def _inside(u: float, p: "Parameter") -> float:
    if math.isfinite(p.min) and math.isfinite(p.max):
        if abs(abs(u) - math.pi / 2) < 1e-6:
            return u - math.copysign(0.1, u)
    elif (math.isfinite(p.min) or math.isfinite(p.max)) and abs(u) < 1e-6:
        return 0.1
    return u


class Parameters(dict):
    """Parameters by name."""

    def valuesdict(self) -> Dict[str, float]:
        return {k: p.value for k, p in self.items()}


class ModelResult:
    """A fit's outcome: ``params``, ``best_fit``, ``residual``, ``chisqr``, ``redchi``,
    ``nfev`` and ``success``."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class _Model:
    """Least squares of a function of named parameters, with lmfit's bounds."""

    def __init__(self, func: Callable, prefix: str = "") -> None:
        self.func, self.prefix = func, prefix
        sig = inspect.signature(func).parameters
        names = list(sig)
        self.independent_vars = names[:1]
        self.param_names = names[1:]
        self.param_hints: Dict[str, Dict[str, Any]] = {}
        for n in self.param_names:
            if sig[n].default is not inspect.Parameter.empty:
                self.param_hints.setdefault(n, {})["value"] = sig[n].default

    def set_param_hint(self, name: str, **hints: Any) -> None:
        self.param_hints.setdefault(name, {}).update(hints)

    def make_params(self, **values: Any) -> Parameters:
        params = Parameters()
        for n in self.param_names:
            hint = dict(self.param_hints.get(n, {}))
            if n in values:
                hint["value"] = values[n]
            params[n] = Parameter(n, **{k: v for k, v in hint.items() if k in ("value", "min", "max", "vary")})
        for p in params.values():  # a value outside the bounds starts at the bound, as in lmfit
            p.value = min(max(p.value, p.min), p.max)
        return params

    def fit(self, data: Any, params: Optional[Parameters] = None, **kwargs: Any) -> ModelResult:
        data = np.asarray(data, dtype=float)
        x = np.asarray(kwargs.pop(self.independent_vars[0]))
        params = self.make_params() if params is None else params
        for n, v in list(kwargs.items()):
            if n in params:
                params[n].value = float(v)
        free = [p for p in params.values() if p.vary]
        fixed = {p.name: p.value for p in params.values() if not p.vary}

        def externals(u: np.ndarray) -> Dict[str, float]:
            out = dict(fixed)
            out.update({p.name: p.from_internal(ui) for p, ui in zip(free, u)})
            return out

        def residual(u: np.ndarray) -> np.ndarray:
            return np.asarray(self.func(x, **externals(u)), dtype=float) - data

        # A start on a bound sits where the bound's transform is flat: start just inside.
        u = np.array([_inside(p.to_internal(p.value), p) for p in free], dtype=float)
        u, nfev, success = _levenberg_marquardt(residual, u)
        best = externals(u)
        r = residual(u)
        chisqr = float(r @ r)
        nfree = max(1, len(data) - len(free))
        redchi = chisqr / nfree
        # Standard errors from the Jacobian in the parameters' own units, scaled by the
        # reduced chi-square, as lmfit reports them.
        if free:
            ext = np.array([best[p.name] for p in free])

            def residual_ext(v: np.ndarray) -> np.ndarray:
                vals = dict(fixed)
                vals.update({p.name: vi for p, vi in zip(free, v)})
                return np.asarray(self.func(x, **vals), dtype=float) - data

            jac = _jacobian(residual_ext, ext)
            try:
                cov = np.linalg.inv(jac.T @ jac) * redchi
                errs = np.sqrt(np.abs(np.diag(cov)))
            except np.linalg.LinAlgError:
                errs = [None] * len(free)
        out = Parameters()
        for p in params.values():
            q = Parameter(p.name, best[p.name], p.min, p.max, p.vary)
            out[p.name] = q
        for p, e in zip(free, errs if free else []):
            out[p.name].stderr = None if e is None else float(e)
        return ModelResult(params=out, best_fit=data + r, residual=r, chisqr=chisqr, redchi=redchi, nfev=nfev, success=success, ndata=len(data), nvarys=len(free))


if _lmfit is not None:  # pragma: no cover - depends on the environment
    Model = _lmfit.model.Model
    update_param_vals = _lmfit.models.update_param_vals
else:
    Model = _Model

    def update_param_vals(pars: Parameters, prefix: str, **kwargs: Any) -> Parameters:
        for name, value in kwargs.items():
            name = f"{prefix}{name}"
            if name in pars:
                pars[name].value = float(value)
        return pars
