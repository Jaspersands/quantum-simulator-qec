"""``optimize``: scipy's ``scipy.optimize`` when scipy is installed, otherwise numpy versions of
the few routines the analysis uses (``brentq``, ``minimize`` with Nelder-Mead,
``minimize_scalar`` (Brent, bounded or not) and ``curve_fit``), following scipy's algorithms
and default tolerances."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

import numpy as np

from stabilizer_qec.analysis._lsq import _jacobian, _levenberg_marquardt


class OptimizeResult(SimpleNamespace):
    """``x``, ``fun``, ``nfev``, ``nit``, ``success``, as scipy's."""


def brentq(f: Callable[[float], float], a: float, b: float, args: tuple = (), xtol: float = 2e-12, rtol: float = 8.881784197001252e-16, maxiter: int = 100, **_: Any) -> float:
    """A root of ``f`` in ``[a, b]``, where ``f(a)`` and ``f(b)`` differ in sign (Brent's
    method, as scipy's ``brentq``)."""
    xpre, xcur = float(a), float(b)
    fpre, fcur = f(xpre, *args), f(xcur, *args)
    if fpre * fcur > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur
    xblk, fblk, spre, scur = 0.0, 0.0, 0.0, 0.0
    for _ in range(maxiter):
        if fpre != 0 and fcur != 0 and (math.copysign(1, fpre) != math.copysign(1, fcur)):
            xblk, fblk = xpre, fpre
            spre = scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre, xcur, xblk = xcur, xblk, xcur
            fpre, fcur, fblk = fcur, fblk, fcur
        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        if fcur == 0 or abs(sbis) < delta:
            return xcur
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:  # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:  # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                spre, scur = scur, stry  # good short step
            else:
                spre, scur = sbis, sbis  # bisect
        else:
            spre, scur = sbis, sbis
        xpre, fpre = xcur, fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta
        fcur = f(xcur, *args)
    raise RuntimeError("brentq did not converge")


def _nelder_mead(fun: Callable, x0: Any, args: tuple = (), xatol: float = 1e-4, fatol: float = 1e-4, maxiter: Optional[int] = None) -> OptimizeResult:
    # scipy's Nelder-Mead (non-adaptive): its initial simplex, coefficients and stopping rule.
    x0 = np.asarray(x0, dtype=float).ravel()
    n = len(x0)
    maxiter = maxiter or 200 * n
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    sim = np.empty((n + 1, n))
    sim[0] = x0
    for k in range(n):
        y = x0.copy()
        y[k] = (1 + 0.05) * y[k] if y[k] != 0 else 0.00025
        sim[k + 1] = y
    fsim = np.array([fun(s, *args) for s in sim], dtype=float)
    nfev = n + 1
    order = np.argsort(fsim, kind="stable")
    sim, fsim = sim[order], fsim[order]
    it = 1
    while nfev < 200 * n and it < maxiter:
        if np.max(np.abs(sim[1:] - sim[0])) <= xatol and np.max(np.abs(fsim[0] - fsim[1:])) <= fatol:
            break
        xbar = np.add.reduce(sim[:-1], 0) / n
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = fun(xr, *args)
        nfev += 1
        shrink = False
        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = fun(xe, *args)
            nfev += 1
            sim[-1], fsim[-1] = (xe, fxe) if fxe < fxr else (xr, fxr)
        elif fxr < fsim[-2]:
            sim[-1], fsim[-1] = xr, fxr
        elif fxr < fsim[-1]:
            xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
            fxc = fun(xc, *args)
            nfev += 1
            if fxc <= fxr:
                sim[-1], fsim[-1] = xc, fxc
            else:
                shrink = True
        else:
            xcc = (1 - psi) * xbar + psi * sim[-1]
            fxcc = fun(xcc, *args)
            nfev += 1
            if fxcc < fsim[-1]:
                sim[-1], fsim[-1] = xcc, fxcc
            else:
                shrink = True
        if shrink:
            for j in range(1, n + 1):
                sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                fsim[j] = fun(sim[j], *args)
            nfev += n
        order = np.argsort(fsim, kind="stable")
        sim, fsim = sim[order], fsim[order]
        it += 1
    return OptimizeResult(x=sim[0], fun=float(fsim[0]), nfev=nfev, nit=it, success=True)


def minimize(fun: Callable, x0: Any, args: tuple = (), method: str = "Nelder-Mead", options: Optional[dict] = None, **_: Any) -> OptimizeResult:
    if method.lower() != "nelder-mead":
        raise ValueError(f"without scipy only the Nelder-Mead method is available, not {method!r}")
    return _nelder_mead(fun, x0, args, **(options or {}))


def _fminbound(func: Callable, x1: float, x2: float, args: tuple = (), xatol: float = 1e-5, maxiter: int = 500) -> OptimizeResult:
    # scipy's bounded Brent (fminbound).
    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")
    sqrt_eps = math.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - math.sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fu = math.inf
    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    while abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = 1
        if abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat
            if (abs(p) < abs(0.5 * q * r)) and (p > q * (a - xf)) and (p < q * (b - xf)):
                rat = (p + 0.0) / q
                x = xf + rat
                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:
                golden = 1
        if golden:
            e = (a - xf) if xf >= xm else (b - xf)
            rat = golden_mean * e
        si = np.sign(rat) + (rat == 0)
        x = xf + si * max(abs(rat), tol1)
        fu = func(x, *args)
        num += 1
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= maxiter:
            break
    return OptimizeResult(x=float(xf), fun=float(fx), nfev=num, nit=num, success=True)


def _bracket(func: Callable, args: tuple = (), xa: float = 0.0, xb: float = 1.0, grow_limit: float = 110.0, maxiter: int = 1000) -> tuple:
    # scipy's bracket: downhill by golden steps and parabolic extrapolation.
    gold, verysmall = 1.618034, 1e-21
    fa, fb = func(xa, *args), func(xb, *args)
    if fa < fb:
        xa, xb, fa, fb = xb, xa, fb, fa
    xc = xb + gold * (xb - xa)
    fc = func(xc, *args)
    it = 0
    while fc < fb:
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        denom = 2.0 * verysmall if abs(val) < verysmall else 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        if it > maxiter:
            raise RuntimeError("Too many iterations.")
        it += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = func(w, *args)
            if fw < fc:
                xa, xb, fa, fb = xb, w, fb, fw
                break
            elif fw > fb:
                xc, fc = w, fw
                break
            w = xc + gold * (xc - xb)
            fw = func(w, *args)
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw = func(w, *args)
        elif (w - wlim) * (xc - w) > 0.0:
            fw = func(w, *args)
            if fw < fc:
                xb, xc, w = xc, w, xc + gold * (xc - xb)
                fb, fc, fw = fc, fw, func(w, *args)
        else:
            w = xc + gold * (xc - xb)
            fw = func(w, *args)
        xa, xb, xc = xb, xc, w
        fa, fb, fc = fb, fc, fw
    return xa, xb, xc, fa, fb, fc


def _brent(func: Callable, args: tuple = (), tol: float = 1.48e-8, maxiter: int = 500) -> OptimizeResult:
    # Unbounded Brent minimisation from scipy's default bracket (0, 1).
    xa, xb, xc, fa, fb, fc = _bracket(func, args)
    a, b = min(xa, xc), max(xa, xc)
    x = w = v = xb
    fw = fv = fx = fb
    deltax = 0.0
    rat = 0.0
    cg = 0.3819660
    for it in range(maxiter):
        tol1 = tol * abs(x) + 1e-11
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        if abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break
        if abs(deltax) <= tol1:
            deltax = (a - x) if x >= xmid else (b - x)
            rat = cg * deltax
        else:
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                p = -p
            tmp2 = abs(tmp2)
            dx_temp = deltax
            deltax = rat
            if (p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and (abs(p) < abs(0.5 * tmp2 * dx_temp)):
                rat = p / tmp2
                u = x + rat
                if (u - a) < tol2 or (b - u) < tol2:
                    rat = tol1 if xmid - x >= 0 else -tol1
            else:
                deltax = (a - x) if x >= xmid else (b - x)
                rat = cg * deltax
        u = x + (tol1 if rat >= 0 else -tol1) if abs(rat) < tol1 else x + rat
        fu = func(u, *args)
        if fu > fx:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w, fv, fw = w, u, fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
        else:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
    return OptimizeResult(x=float(x), fun=float(fx), nit=it, success=True)


def minimize_scalar(fun: Callable, bracket: Any = None, bounds: Optional[Sequence[float]] = None, args: tuple = (), method: Optional[str] = None, options: Optional[dict] = None, **_: Any) -> OptimizeResult:
    method = (method or ("bounded" if bounds is not None else "brent")).lower()
    if method == "bounded":
        return _fminbound(fun, float(bounds[0]), float(bounds[1]), args, **(options or {}))
    return _brent(fun, args)


def curve_fit(f: Callable, xdata: Any, ydata: Any, p0: Any = None, sigma: Any = None, absolute_sigma: bool = False, bounds: Any = (-np.inf, np.inf), maxfev: Optional[int] = None, jac: Any = None, **_: Any) -> tuple:
    """Least squares of ``f(xdata, *params)`` to ``ydata``, as scipy's ``curve_fit``: the
    parameters and their covariance. Bounds are kept by lmfit's transforms."""
    ydata = np.asarray(ydata, dtype=float)
    if p0 is None:
        p0 = np.ones(len(__import__("inspect").signature(f).parameters) - 1)
    p0 = np.asarray(p0, dtype=float)
    lo, hi = np.broadcast_to(np.asarray(bounds[0], dtype=float), p0.shape), np.broadcast_to(np.asarray(bounds[1], dtype=float), p0.shape)
    weights = 1.0 if sigma is None else 1.0 / np.asarray(sigma, dtype=float)

    def to_ext(u: np.ndarray) -> np.ndarray:
        out = u.copy()
        for k in range(len(u)):
            if math.isfinite(lo[k]) and math.isfinite(hi[k]):
                out[k] = lo[k] + (math.sin(u[k]) + 1) * (hi[k] - lo[k]) / 2
            elif math.isfinite(lo[k]):
                out[k] = lo[k] - 1 + math.sqrt(u[k] * u[k] + 1)
            elif math.isfinite(hi[k]):
                out[k] = hi[k] + 1 - math.sqrt(u[k] * u[k] + 1)
        return out

    def to_int(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for k in range(len(x)):
            if math.isfinite(lo[k]) and math.isfinite(hi[k]):
                out[k] = math.asin(max(-1.0, min(1.0, 2 * (x[k] - lo[k]) / (hi[k] - lo[k]) - 1)))
            elif math.isfinite(lo[k]):
                out[k] = math.sqrt(max(0.0, (x[k] - lo[k] + 1) ** 2 - 1)) or 0.1
            elif math.isfinite(hi[k]):
                out[k] = math.sqrt(max(0.0, (hi[k] - x[k] + 1) ** 2 - 1)) or 0.1
        return out

    def residual_ext(x: np.ndarray) -> np.ndarray:
        return (np.asarray(f(xdata, *x), dtype=float) - ydata) * weights

    u, _, _ = _levenberg_marquardt(lambda u: residual_ext(to_ext(u)), to_int(p0), max_nfev=maxfev or 100_000)
    popt = to_ext(u)
    jac_ext = _jacobian(residual_ext, popt)
    pcov = np.linalg.pinv(jac_ext.T @ jac_ext)
    if not absolute_sigma:
        r = residual_ext(popt)
        dof = len(ydata) - len(popt)
        pcov = pcov * (float(r @ r) / dof) if dof > 0 else np.full_like(pcov, np.inf)
    return popt, pcov


try:  # pragma: no cover - depends on the environment
    import scipy.optimize as optimize  # type: ignore[no-redef]
except ImportError:
    optimize = SimpleNamespace(brentq=brentq, minimize=minimize, minimize_scalar=minimize_scalar, curve_fit=curve_fit, OptimizeResult=OptimizeResult)
