"""Least squares without scipy: Levenberg-Marquardt, and ``curve_fit`` as scipy's (scipy's
own when it is installed)."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


def _jacobian(f: Callable, u: np.ndarray) -> np.ndarray:
    """Central differences, steps scaled to each parameter."""
    f0 = f(u)
    jac = np.empty((len(f0), len(u)))
    for k in range(len(u)):
        h = np.finfo(float).eps ** (1 / 3) * (abs(u[k]) or 1.0)
        up, down = u.copy(), u.copy()
        up[k] += h
        down[k] -= h
        jac[:, k] = (f(up) - f(down)) / (2 * h)
    return jac


def _levenberg_marquardt(f: Callable, u: np.ndarray, max_nfev: int = 2000) -> tuple:
    """Minimise |f(u)|^2 from ``u``."""
    if len(u) == 0:
        return u, 1, True
    lam, nfev = 1e-3, 0
    r = f(u)
    cost = float(r @ r)
    nfev += 1
    for _ in range(max_nfev):
        jac = _jacobian(f, u)
        nfev += len(u)
        g = jac.T @ r
        a = jac.T @ jac
        improved = False
        while lam < 1e16:
            step = np.linalg.lstsq(a + lam * np.diag(np.maximum(np.diag(a), 1e-300)), -g, rcond=None)[0]
            trial = u + step
            rt = f(trial)
            nfev += 1
            ct = float(rt @ rt)
            if np.isfinite(ct) and ct <= cost:
                done = cost - ct <= 1e-15 * max(cost, 1e-300) and np.all(np.abs(step) <= 1.5e-8 * (np.abs(u) + 1.5e-8))
                u, r, cost = trial, rt, ct
                lam = max(lam / 10, 1e-12)
                improved = True
                if done:
                    return u, nfev, True
                break
            lam *= 10
        if not improved:
            return u, nfev, True
        if nfev > max_nfev:
            break
    return u, nfev, False


def curve_fit(f: Callable, xdata: Any, ydata: Any, p0: Any, sigma: Optional[Any] = None, absolute_sigma: bool = False, maxfev: int = 100_000, **kwargs: Any) -> tuple:
    """``scipy.optimize.curve_fit`` (unbounded): the parameters minimising the squared,
    sigma-weighted residuals, and their covariance."""
    try:
        from scipy.optimize import curve_fit as scipy_curve_fit
    except ImportError:
        pass
    else:  # pragma: no cover - depends on the environment
        return scipy_curve_fit(f, xdata, ydata, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma, maxfev=maxfev, **kwargs)
    ydata = np.asarray(ydata, dtype=float)
    weights = 1.0 if sigma is None else 1.0 / np.asarray(sigma, dtype=float)

    def residual(params: np.ndarray) -> np.ndarray:
        return (np.asarray(f(xdata, *params), dtype=float) - ydata) * weights

    popt, _, _ = _levenberg_marquardt(residual, np.asarray(p0, dtype=float), max_nfev=maxfev)
    jac = _jacobian(residual, popt)
    try:
        pcov = np.linalg.pinv(jac.T @ jac)
    except np.linalg.LinAlgError:
        pcov = np.full((len(popt), len(popt)), np.inf)
    if not absolute_sigma:
        r = residual(popt)
        dof = len(ydata) - len(popt)
        pcov = pcov * (float(r @ r) / dof) if dof > 0 else np.full_like(pcov, np.inf)
    return popt, pcov


def beta_std(a: Any, b: Any) -> Any:
    """The standard deviation of a beta distribution."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
