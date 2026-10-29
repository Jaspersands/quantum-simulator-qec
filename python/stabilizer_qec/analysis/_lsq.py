"""Least squares without scipy: MINPACK's Levenberg-Marquardt as scipy's ``leastsq`` and
``curve_fit`` (scipy's own when it is installed), and a simpler Levenberg-Marquardt that the
bounded fits in ``_scipy`` still use."""

from __future__ import annotations

import inspect
import math
import warnings
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


# MINPACK's lmdif (Moré, Garbow and Hillstrom, Argonne 1980), which scipy.optimize.leastsq calls and
# so curve_fit (unbounded) and lmfit's 'leastsq': forward-difference Jacobians, a QR factorisation
# with column pivoting, and a trust region whose Levenberg-Marquardt parameter lmpar finds. Its steps,
# scaling and stopping rules are MINPACK's, so a fit stops where scipy's does, not merely where no
# step improves.

_EPSMCH = float(np.finfo(float).eps)
_DWARF = float(np.finfo(float).tiny)
_RDWARF, _RGIANT = 3.834e-20, 1.304e19


def _enorm(v: np.ndarray) -> float:
    """MINPACK's enorm: the Euclidean norm, its squares summed in three ranges so that none
    overflows or underflows."""
    v = np.abs(np.asarray(v, dtype=float))
    if v.size == 0:
        return 0.0
    agiant = _RGIANT / v.size
    big, small = v >= agiant, v <= _RDWARF
    mid = ~(big | small)
    s2 = float(np.sum(v[mid] ** 2))
    if big.any():
        x1max = float(v[big].max())
        s1 = float(np.sum((v[big] / x1max) ** 2))
        return x1max * math.sqrt(s1 + (s2 / x1max) / x1max)
    if s2 != 0.0:
        if small.any():
            x3max = float(v[small].max())
            s3 = float(np.sum((v[small] / x3max) ** 2)) if x3max > 0 else 0.0
            s3 *= x3max
            return math.sqrt(s2 * (1 + (x3max / s2) * s3))
        return math.sqrt(s2)
    x3max = float(v[small].max()) if small.any() else 0.0
    if x3max == 0.0:
        return 0.0
    return x3max * math.sqrt(float(np.sum((v[small] / x3max) ** 2)))


def _qrfac(a: np.ndarray) -> tuple:
    """MINPACK's qrfac with pivoting: Householder QR of ``a`` (in place), the columns ordered by
    their remaining norms. Returns the pivots, R's diagonal and the columns' norms; ``a`` holds R's
    strict upper triangle and the Householder vectors below it."""
    m, n = a.shape
    acnorm = np.array([_enorm(a[:, j]) for j in range(n)])
    rdiag, wa, ipvt = acnorm.copy(), acnorm.copy(), list(range(n))
    for j in range(min(m, n)):
        kmax = j
        for k in range(j, n):
            if rdiag[k] > rdiag[kmax]:
                kmax = k
        if kmax != j:
            a[:, [j, kmax]] = a[:, [kmax, j]]
            rdiag[kmax], wa[kmax] = rdiag[j], wa[j]
            ipvt[j], ipvt[kmax] = ipvt[kmax], ipvt[j]
        ajnorm = _enorm(a[j:, j])
        if ajnorm != 0.0:
            if a[j, j] < 0:
                ajnorm = -ajnorm
            a[j:, j] /= ajnorm
            a[j, j] += 1.0
            for k in range(j + 1, n):
                temp = float(a[j:, j] @ a[j:, k]) / a[j, j]
                a[j:, k] -= temp * a[j:, j]
                if rdiag[k] != 0.0:
                    temp = a[j, k] / rdiag[k]
                    rdiag[k] *= math.sqrt(max(0.0, 1.0 - temp * temp))
                    if 0.05 * (rdiag[k] / wa[k]) ** 2 <= _EPSMCH:
                        rdiag[k] = _enorm(a[j + 1 :, k])
                        wa[k] = rdiag[k]
        rdiag[j] = -ajnorm
    return ipvt, rdiag, acnorm


def _qrsolv(r: np.ndarray, ipvt: list, diag: np.ndarray, qtb: np.ndarray) -> tuple:
    """MINPACK's qrsolv: the least-squares solution of ``[A; D] x = [b; 0]`` from A's pivoted
    QR (R in ``r``'s upper triangle, Q^T b in ``qtb``), by Givens rotations; S's strict upper
    triangle is left transposed in ``r``'s strict lower one. Returns x and S's diagonal."""
    n = len(qtb)
    x, wa, sdiag = np.empty(n), np.array(qtb, dtype=float), np.zeros(n)
    for j in range(n):
        r[j:n, j] = r[j, j:n]
        x[j] = r[j, j]
    for j in range(n):
        l = ipvt[j]
        if diag[l] != 0.0:
            sdiag[j:] = 0.0
            sdiag[j] = diag[l]
            qtbpj = 0.0
            for k in range(j, n):
                if sdiag[k] == 0.0:
                    continue
                if abs(r[k, k]) < abs(sdiag[k]):
                    cotan = r[k, k] / sdiag[k]
                    sin = 0.5 / math.sqrt(0.25 + 0.25 * cotan * cotan)
                    cos = sin * cotan
                else:
                    tan = sdiag[k] / r[k, k]
                    cos = 0.5 / math.sqrt(0.25 + 0.25 * tan * tan)
                    sin = cos * tan
                r[k, k] = cos * r[k, k] + sin * sdiag[k]
                temp = cos * wa[k] + sin * qtbpj
                qtbpj = -sin * wa[k] + cos * qtbpj
                wa[k] = temp
                for i in range(k + 1, n):
                    temp = cos * r[i, k] + sin * sdiag[i]
                    sdiag[i] = -sin * r[i, k] + cos * sdiag[i]
                    r[i, k] = temp
        sdiag[j] = r[j, j]
        r[j, j] = x[j]
    nsing = n
    for j in range(n):
        if sdiag[j] == 0.0 and nsing == n:
            nsing = j
        if nsing < n:
            wa[j] = 0.0
    for j in range(nsing - 1, -1, -1):
        total = sum(r[i, j] * wa[i] for i in range(j + 1, nsing))
        wa[j] = (wa[j] - total) / sdiag[j]
    out = np.empty(n)
    for j in range(n):
        out[ipvt[j]] = wa[j]
    return out, sdiag


def _lmpar(r: np.ndarray, ipvt: list, diag: np.ndarray, qtb: np.ndarray, delta: float, par: float) -> tuple:
    """MINPACK's lmpar: the Levenberg-Marquardt parameter whose step has a scaled length within
    10% of ``delta`` (or the Gauss-Newton step, if that is shorter). Returns it and the step."""
    n = len(qtb)
    wa1 = np.array(qtb, dtype=float)
    nsing = n
    for j in range(n):
        if r[j, j] == 0.0 and nsing == n:
            nsing = j
        if nsing < n:
            wa1[j] = 0.0
    for j in range(nsing - 1, -1, -1):
        wa1[j] /= r[j, j]
        temp = wa1[j]
        for i in range(j):
            wa1[i] -= r[i, j] * temp
    x = np.empty(n)
    for j in range(n):
        x[ipvt[j]] = wa1[j]
    iters = 0
    wa2 = diag * x
    dxnorm = _enorm(wa2)
    fp = dxnorm - delta
    if fp <= 0.1 * delta:
        return 0.0, x
    parl = 0.0
    if nsing >= n:
        for j in range(n):
            l = ipvt[j]
            wa1[j] = diag[l] * (wa2[l] / dxnorm)
        for j in range(n):
            total = sum(r[i, j] * wa1[i] for i in range(j))
            wa1[j] = (wa1[j] - total) / r[j, j]
        temp = _enorm(wa1)
        parl = ((fp / delta) / temp) / temp
    for j in range(n):
        total = sum(r[i, j] * qtb[i] for i in range(j + 1))
        wa1[j] = total / diag[ipvt[j]]
    gnorm = _enorm(wa1)
    paru = gnorm / delta
    if paru == 0.0:
        paru = _DWARF / min(delta, 0.1)
    par = min(max(par, parl), paru)
    if par == 0.0:
        par = gnorm / dxnorm
    while True:
        iters += 1
        if par == 0.0:
            par = max(_DWARF, 0.001 * paru)
        x, sdiag = _qrsolv(r, ipvt, math.sqrt(par) * diag, qtb)
        wa2 = diag * x
        dxnorm = _enorm(wa2)
        temp = fp
        fp = dxnorm - delta
        if abs(fp) <= 0.1 * delta or (parl == 0.0 and fp <= temp and temp < 0) or iters == 10:
            return par, x
        for j in range(n):
            l = ipvt[j]
            wa1[j] = diag[l] * (wa2[l] / dxnorm)
        for j in range(n):
            wa1[j] /= sdiag[j]
            temp = wa1[j]
            for i in range(j + 1, n):
                wa1[i] -= r[i, j] * temp
        temp = _enorm(wa1)
        parc = ((fp / delta) / temp) / temp
        if fp > 0:
            parl = max(parl, par)
        if fp < 0:
            paru = min(paru, par)
        par = max(parl, par + parc)


def _lmdif(fcn: Callable, x: np.ndarray, ftol: float, xtol: float, gtol: float, maxfev: int, epsfcn: float, factor: float, diag: Optional[np.ndarray]) -> tuple:
    """MINPACK's lmdif from ``x``, whose residuals ``fcn(x)`` were already evaluated once (as
    scipy's leastsq does to check them). Returns x, its residuals, the factorised Jacobian (R in
    its upper triangle), the pivots, Q^T f, the evaluations and MINPACK's ``info``."""
    n = len(x)
    mode = 1 if diag is None else 2
    fvec = np.asarray(fcn(x), dtype=float)
    m = len(fvec)
    nfev, info = 1, 0
    fnorm = _enorm(fvec)
    par, it, delta, xnorm = 0.0, 1, 0.0, 0.0
    diag = np.zeros(n) if diag is None else np.array(diag, dtype=float)
    eps = math.sqrt(max(epsfcn, _EPSMCH))
    while True:
        # The Jacobian by forward differences.
        fjac = np.empty((m, n))
        for j in range(n):
            temp = x[j]
            h = eps * abs(temp)
            if h == 0.0:
                h = eps
            x[j] = temp + h
            wa = np.asarray(fcn(x), dtype=float)
            x[j] = temp
            fjac[:, j] = (wa - fvec) / h
        nfev += n
        ipvt, rdiag, acnorm = _qrfac(fjac)
        if it == 1:
            if mode == 1:
                diag = np.where(acnorm == 0.0, 1.0, acnorm)
            xnorm = _enorm(diag * x)
            delta = factor * xnorm
            if delta == 0.0:
                delta = factor
        wa4 = fvec.copy()
        qtf = np.empty(n)
        for j in range(n):
            if fjac[j, j] != 0.0:
                temp = -float(fjac[j:, j] @ wa4[j:]) / fjac[j, j]
                wa4[j:] += fjac[j:, j] * temp
            fjac[j, j] = rdiag[j]
            qtf[j] = wa4[j]
        gnorm = 0.0
        if fnorm != 0.0:
            for j in range(n):
                l = ipvt[j]
                if acnorm[l] != 0.0:
                    total = sum(fjac[i, j] * (qtf[i] / fnorm) for i in range(j + 1))
                    gnorm = max(gnorm, abs(total / acnorm[l]))
        if gnorm <= gtol:
            info = 4
        if info != 0:
            break
        if mode == 1:
            diag = np.maximum(diag, acnorm)
        while True:
            par, step = _lmpar(fjac, ipvt, diag, qtf, delta, par)
            wa1 = -step
            wa2 = x + wa1
            pnorm = _enorm(diag * wa1)
            if it == 1:
                delta = min(delta, pnorm)
            wa4 = np.asarray(fcn(wa2), dtype=float)
            nfev += 1
            fnorm1 = _enorm(wa4)
            actred = 1.0 - (fnorm1 / fnorm) ** 2 if 0.1 * fnorm1 < fnorm else -1.0
            wa3 = np.zeros(n)
            for j in range(n):
                temp = wa1[ipvt[j]]
                for i in range(j + 1):
                    wa3[i] += fjac[i, j] * temp
            temp1 = _enorm(wa3) / fnorm
            temp2 = (math.sqrt(par) * pnorm) / fnorm
            prered = temp1 * temp1 + temp2 * temp2 / 0.5
            dirder = -(temp1 * temp1 + temp2 * temp2)
            ratio = actred / prered if prered != 0.0 else 0.0
            if ratio <= 0.25:
                temp = 0.5 if actred >= 0 else 0.5 * dirder / (dirder + 0.5 * actred)
                if 0.1 * fnorm1 >= fnorm or temp < 0.1:
                    temp = 0.1
                delta = temp * min(delta, pnorm / 0.1)
                par /= temp
            elif par == 0.0 or ratio >= 0.75:
                delta = pnorm / 0.5
                par *= 0.5
            if ratio >= 1e-4:
                x = wa2
                fvec = wa4
                xnorm = _enorm(diag * x)
                fnorm = fnorm1
                it += 1
            if abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1:
                info = 1
            if delta <= xtol * xnorm:
                info = 3 if info == 1 else 2
            if info != 0:
                break
            if nfev >= maxfev:
                info = 5
            if abs(actred) <= _EPSMCH and prered <= _EPSMCH and 0.5 * ratio <= 1:
                info = 6
            if delta <= _EPSMCH * xnorm:
                info = 7
            if gnorm <= _EPSMCH:
                info = 8
            if info != 0:
                break
            if ratio >= 1e-4:
                break
        if info != 0:
            break
    return x, fvec, fjac, ipvt, qtf, nfev, info


def _leastsq_messages(ftol: float, xtol: float, gtol: float, maxfev: int) -> dict:
    return {
        0: "Improper input parameters.",
        1: "Both actual and predicted relative reductions in the sum of squares\n  are at most %f" % ftol,
        2: "The relative error between two consecutive iterates is at most %f" % xtol,
        3: "Both actual and predicted relative reductions in the sum of squares\n  are at most {:f} and the relative error between two consecutive iterates is at \n  most {:f}".format(ftol, xtol),
        4: "The cosine of the angle between func(x) and any column of the\n  Jacobian is at most %f in absolute value" % gtol,
        5: "Number of calls to function has reached maxfev = %d." % maxfev,
        6: "ftol=%f is too small, no further reduction in the sum of squares\n  is possible." % ftol,
        7: "xtol=%f is too small, no further improvement in the approximate\n  solution is possible." % xtol,
        8: "gtol=%f is too small, func(x) is orthogonal to the columns of\n  the Jacobian to machine precision." % gtol,
    }


def leastsq(func: Callable, x0: Any, args: tuple = (), ftol: float = 1.49012e-08, xtol: float = 1.49012e-08, gtol: float = 0.0, maxfev: int = 0, epsfcn: Optional[float] = None, factor: float = 100, diag: Optional[Any] = None) -> tuple:
    """``scipy.optimize.leastsq(func, x0, full_output=True)`` without scipy: MINPACK's lmdif
    (above). Returns the solution, the covariance ``(J^T J)^-1`` from its pivoted R (None where
    R is singular or the fit failed), ``{'fvec', 'nfev', 'fjac', 'ipvt', 'qtf'}``, the message
    and ``ier``."""
    x0 = np.asarray(x0, dtype=float).flatten()
    n = len(x0)

    def fcn(x: np.ndarray) -> np.ndarray:
        return np.atleast_1d(np.asarray(func(x, *args), dtype=float))

    m = len(fcn(x0.copy()))  # scipy evaluates once to check the shape
    if n > m:
        raise TypeError(f"Improper input: func input vector length N={n} must not exceed func output vector length M={m}")
    if epsfcn is None:
        epsfcn = _EPSMCH
    if maxfev == 0:
        maxfev = 200 * (n + 1)
    messages = _leastsq_messages(ftol, xtol, gtol, maxfev)
    if n <= 0 or ftol < 0 or xtol < 0 or gtol < 0 or maxfev <= 0 or factor <= 0 or (diag is not None and np.any(np.asarray(diag) <= 0)):
        return x0, None, {"fvec": fcn(x0.copy()), "nfev": 1}, messages[0], 0
    x, fvec, fjac, ipvt, qtf, nfev, info = _lmdif(fcn, x0.copy(), ftol, xtol, gtol, maxfev, epsfcn, factor, diag)
    cov_x = None
    if info in (1, 2, 3, 4):
        r = np.triu(fjac[:n, :])
        if np.all(np.diag(r) != 0.0):  # LAPACK's trtri refuses an exactly singular R
            inv_r = np.linalg.solve(r, np.eye(n)) if n else np.zeros((0, 0))
            inv_r[ipvt] = inv_r.copy()
            cov_x = inv_r @ inv_r.T
    infodict = {"fvec": fvec, "nfev": nfev, "fjac": fjac.T.copy(), "ipvt": np.array(ipvt, dtype=np.int32) + 1, "qtf": qtf}
    return x, cov_x, infodict, messages[info], info


class OptimizeWarning(UserWarning):
    """scipy.optimize's OptimizeWarning."""


def curve_fit(f: Callable, xdata: Any, ydata: Any, p0: Any = None, sigma: Optional[Any] = None, absolute_sigma: bool = False, check_finite: bool = True, **kwargs: Any) -> tuple:
    """``scipy.optimize.curve_fit`` (unbounded, Levenberg-Marquardt, a 1-D ``sigma``): the
    parameters minimising the squared, sigma-weighted residuals, and their covariance; scipy's
    own when it is installed. Without it, the fit is MINPACK's (``leastsq`` here), so it stops
    where scipy's does, and the covariance is scipy's too: infinite, with an OptimizeWarning,
    where R is singular or there are no more points than parameters."""
    try:
        from scipy.optimize import curve_fit as scipy_curve_fit
    except ImportError:
        pass
    else:  # pragma: no cover - depends on the environment
        return scipy_curve_fit(f, xdata, ydata, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma, check_finite=check_finite, **kwargs)
    if p0 is None:
        n = len(inspect.signature(f).parameters) - 1
        if n < 1:
            raise ValueError("Unable to determine number of fit parameters.")
        p0 = np.ones(n)
    p0 = np.atleast_1d(np.asarray(p0, dtype=float))
    n = p0.size
    if check_finite:
        ydata = np.asarray_chkfinite(ydata, float)
        if isinstance(xdata, (list, tuple, np.ndarray)):
            xdata = np.asarray_chkfinite(xdata, float)
    else:
        ydata = np.asarray(ydata, dtype=float)
    if ydata.size == 0:
        raise ValueError("`ydata` must not be empty!")
    transform = None if sigma is None else 1.0 / np.asarray(sigma, dtype=float)

    def residual(params: np.ndarray) -> np.ndarray:
        diff = np.asarray(f(xdata, *params), dtype=float) - ydata
        return diff if transform is None else transform * diff

    if ydata.size != 1 and n > ydata.size:
        raise TypeError(f"The number of func parameters={n} must not exceed the number of data points={ydata.size}")
    popt, pcov, infodict, errmsg, ier = leastsq(residual, p0, **kwargs)
    ysize = len(infodict["fvec"])
    cost = float(np.sum(infodict["fvec"] ** 2))
    if ier not in (1, 2, 3, 4):
        raise RuntimeError("Optimal parameters not found: " + errmsg)
    warn_cov = False
    if pcov is None or np.isnan(pcov).any():
        pcov = np.full((n, n), np.inf)
        warn_cov = True
    elif not absolute_sigma:
        if ysize > n:
            pcov = pcov * (cost / (ysize - n))
        else:
            pcov = np.full((n, n), np.inf)
            warn_cov = True
    if warn_cov:
        warnings.warn("Covariance of the parameters could not be estimated", category=OptimizeWarning, stacklevel=2)
    return popt, pcov


def beta_std(a: Any, b: Any) -> Any:
    """The standard deviation of a beta distribution."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
