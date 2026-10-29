"""lmfit's ``Model`` when lmfit is installed; otherwise a least-squares model with the part of
its interface that the fits here use (parameter hints, ``make_params``, ``fit`` with lmfit's
default 'leastsq' and its options, and a result's ``params``), fitting as lmfit does with
MINPACK's Levenberg-Marquardt, and ``ufloat`` from uncertainties or a stand-in."""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Any, Callable, Dict, Optional

import numpy as np

from stabilizer_qec.analysis._lsq import leastsq

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
    def setup_bounds(self) -> float:
        """The internal value the fit starts from (lmfit's ``setup_bounds``)."""
        lo, hi, v = self.min, self.max, self.value
        if lo == -math.inf and hi == math.inf:
            u = v
        elif hi == math.inf:
            u = math.sqrt((v - lo + 1.0) ** 2 - 1)
        elif lo == -math.inf:
            u = math.sqrt((hi - v + 1.0) ** 2 - 1)
        else:
            u = math.asin(2 * (v - lo) / (hi - lo) - 1)
        return 0.0 if abs(u) < 1.0e-15 else float(u)

    def from_internal(self, u: float) -> float:
        lo, hi = self.min, self.max
        if lo == -math.inf and hi == math.inf:
            return float(u)
        if hi == math.inf:
            return lo - 1.0 + math.sqrt(u * u + 1)
        if lo == -math.inf:
            return hi + 1 - math.sqrt(u * u + 1)
        return lo + (math.sin(u) + 1) * (hi - lo) / 2.0

    def scale_gradient(self, u: float) -> float:
        """d(value)/d(internal value) (lmfit's ``scale_gradient``)."""
        lo, hi = self.min, self.max
        if lo == -math.inf and hi == math.inf:
            return 1.0
        if hi == math.inf:
            return u / math.sqrt(u * u + 1)
        if lo == -math.inf:
            return -u / math.sqrt(u * u + 1)
        return math.cos(u) * (hi - lo) / 2.0


class Parameters(dict):
    """Parameters by name."""

    def valuesdict(self) -> Dict[str, float]:
        return {k: p.value for k, p in self.items()}


class ModelResult:
    """A fit's outcome, as lmfit's: ``params``, ``best_fit``, ``residual`` (weighted
    ``data - model``), ``chisqr``, ``redchi``, ``nfev``, ``success``, ``message``, ``ier``,
    ``covar``, ``ndata``, ``nvarys``, ``nfree``, ``data``, ``weights`` and ``var_names``."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class _Model:
    """lmfit's ``Model`` for least squares of a function of named parameters (its first argument
    the independent variable), with lmfit's bounds, options and errors."""

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

    def fit(self, data: Any, params: Optional[Parameters] = None, weights: Any = None, method: str = "leastsq", iter_cb: Any = None, scale_covar: bool = True, verbose: bool = False, fit_kws: Optional[Dict[str, Any]] = None, nan_policy: Optional[str] = None, calc_covar: bool = True, max_nfev: Optional[int] = None, coerce_farray: bool = True, **kwargs: Any) -> ModelResult:
        """lmfit's ``Model.fit`` with its default ``method='leastsq'``: the residual is
        ``(data - model) * weights``, minimised by MINPACK's Levenberg-Marquardt over the
        parameters' internal (bound-free) values with lmfit's options (``epsfcn=1e-10``, at most
        ``max_nfev`` evaluations, by default ``2000 * (nvarys + 1)``), and the standard errors are
        lmfit's from MINPACK's covariance. As in lmfit, keywords naming parameters set their
        starting values, other unknown keywords are ignored with a warning, NaN or inf in the
        data or the model raise ValueError (``nan_policy='raise'``; ``'omit'`` drops NaN data
        points, ``'propagate'`` checks nothing), and fewer data points than varying parameters
        raise TypeError. Other methods and ``iter_cb`` need lmfit."""
        if method != "leastsq":
            raise NotImplementedError(f"method={method!r} needs lmfit (pip install lmfit); without it, fits use 'leastsq'")
        if iter_cb is not None:
            raise NotImplementedError("iter_cb needs lmfit (pip install lmfit)")
        params = self.make_params() if params is None else Parameters({k: _copy(p) for k, p in params.items()})
        for name in [n for n in kwargs if n in self.param_names]:
            p = params[name]
            p.value = min(max(float(kwargs.pop(name)), p.min), p.max)
        for name in kwargs:
            if name not in self.independent_vars:
                warnings.warn(f"The keyword argument {name} does not match any arguments of the model function. It will be ignored.", UserWarning)
        nan_policy = "raise" if nan_policy is None else nan_policy
        if nan_policy not in ("propagate", "omit", "raise"):
            raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'.")
        var = self.independent_vars[0]
        if var not in kwargs:
            raise ValueError(f"'Missing independent variable '{var}'")
        x = kwargs[var]
        data = np.asarray(data, dtype=float)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
        if nan_policy == "omit":
            mask = ~np.isnan(data)
            data = data[mask]
            if weights is not None:
                weights = weights[mask]
            if not np.isscalar(x):
                x = np.asarray(x)[mask]
        x = np.asarray(x, dtype=float) if coerce_farray and not np.isscalar(x) else x
        lskws: Dict[str, Any] = dict(ftol=1.5e-8, xtol=1.5e-8, gtol=0.0, epsfcn=1.0e-10, factor=100, diag=None)
        fit_kws = dict(fit_kws or {})
        if "maxfev" in fit_kws:
            warnings.warn("ignoring `maxfev` argument to `leastsq()`. Use `max_nfev` instead.", RuntimeWarning)
            fit_kws.pop("maxfev")
        for k in fit_kws:
            if k not in lskws:
                raise TypeError(f"leastsq() got an unexpected keyword argument '{k}'")
        lskws.update(fit_kws)
        free = [p for p in params.values() if p.vary]
        fixed = {p.name: p.value for p in params.values() if not p.vary}
        max_nfev = 2000 * (len(free) + 1) if max_nfev in (None, np.inf) else int(max_nfev)
        # lmfit counts its residual's calls from -2, calls it once before the fit, and aborts
        # the fit at the call that takes the count past max_nfev, keeping the values it was given.
        state = {"nfev": -1, "last": None}

        def externals(u: np.ndarray) -> Dict[str, float]:
            out = dict(fixed)
            out.update({p.name: float(p.from_internal(ui)) for p, ui in zip(free, u)})
            return out

        def model(u: np.ndarray) -> np.ndarray:
            return np.asarray(self.func(x, **externals(u)), dtype=float)

        def residual(u: np.ndarray, count: bool = True) -> np.ndarray:
            if count:
                state["nfev"] += 1
                state["last"] = np.array(u, dtype=float)
                if state["nfev"] > max_nfev:
                    raise _Aborted
            m = model(u)
            if nan_policy == "raise" and not np.all(np.isfinite(m)):
                raise ValueError("The model function generated NaN values and the fit aborted! Please check your model function and/or set boundaries on parameters where applicable. In cases like this, using \"nan_policy='omit'\" will probably not work.")
            diff = data - m
            if weights is not None:
                diff = diff * weights
            diff = np.asarray(diff, dtype=float).ravel()
            if nan_policy == "raise" and not np.isfinite(np.sum(diff)):
                raise ValueError("NaN values detected in your input data or the output of your objective/model function - fitting algorithms cannot handle this! Please read https://lmfit.github.io/lmfit-py/faq.html#i-get-errors-from-nan-in-my-fit-what-can-i-do for more information.")
            if nan_policy == "omit":
                diff = diff[np.isfinite(diff)]
            return diff

        u0 = np.array([p.setup_bounds() for p in free], dtype=float)
        try:
            best, cov, infodict, _, ier = leastsq(residual, u0, maxfev=2 * max_nfev, **lskws)
            nfev = int(infodict["nfev"])
        except _Aborted:  # lmfit keeps the last values tried and reports no error bars
            best, cov, ier, nfev = state["last"], None, -1, max_nfev
        r = residual(best, count=False)
        values = externals(best)
        chisqr = float(r @ r)
        ndata, nvarys = len(r), len(free)
        nfree = ndata - nvarys
        redchi = chisqr / max(1, nfree)
        chisqr = max(chisqr, 1.0e-250 * ndata)
        out = Parameters()
        for p in params.values():
            out[p.name] = Parameter(p.name, values[p.name], p.min, p.max, p.vary)
        covar = None
        if cov is not None:
            # lmfit's covariance: MINPACK's in the internal values, scaled by the bounds'
            # gradients (MINUIT's recipe) and by the reduced chi-square.
            g = np.array([p.scale_gradient(u) for p, u in zip(free, best)])
            covar = cov * np.outer(g, g)
            if scale_covar:
                covar = covar * redchi
            for q in out.values():
                q.stderr = 0.0
            for p, e in zip(free, np.diag(covar)):
                out[p.name].stderr = float(np.sqrt(e))
        messages = {0: "Invalid Input Parameters. I.e. more variables than data points given, tolerance < 0.0, or no data provided.", 4: "One or more variable did not affect the fit.", 5: f"Too many function calls (max set to {2 * max_nfev})! Use: max_nfev=NNN to increase this maximum."}
        message = "Fit succeeded." if ier in (1, 2, 3) else messages.get(ier, "Tolerance seems to be too small.")
        if covar is None:
            message += " Could not estimate error-bars."
        best_fit = np.asarray(self.func(x, **values), dtype=float)
        return ModelResult(params=out, best_fit=best_fit, residual=r, chisqr=chisqr, redchi=redchi, nfev=nfev, success=ier in (1, 2, 3, 4), message=message, ier=ier, covar=covar, ndata=ndata, nvarys=nvarys, nfree=nfree, data=data, weights=weights, var_names=[p.name for p in free])


class _Aborted(Exception):
    """More model evaluations than ``max_nfev``: lmfit aborts the fit."""


def _copy(p: Parameter) -> Parameter:
    q = Parameter(p.name, p.value, p.min, p.max, p.vary)
    q.stderr = p.stderr
    return q


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
