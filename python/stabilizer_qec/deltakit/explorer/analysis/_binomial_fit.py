# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Asymmetric binomial confidence intervals for error-rate estimates.

A logical error probability estimated from ``num_hits`` failures out of
``num_shots`` shots follows a binomial distribution. The symmetric Gaussian
interval ``p +/- sqrt(p (1 - p) / n)`` is a poor description of that
distribution when ``p`` is close to zero: it can extend below zero and it
underestimates the upper tail. This module computes the asymmetric interval
instead, following the approach used by Stim's ``sinter``.

``log_binomial`` and the ``fit_binomial`` search are adapted from
``sinter._probability_util`` (Apache-2.0, quantumlib/Stim). See
https://quantumcomputing.stackexchange.com/a/37268 for an interpretation of the
``max_likelihood_factor`` interval.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from scipy import optimize

#: Default Bayes factor used to define the confidence interval. Hypotheses whose
#: likelihood is more than this many times less likely than the best fit are
#: excluded. A factor of 1000 corresponds to roughly a 99% interval.
DEFAULT_MAX_LIKELIHOOD_FACTOR: float = 1000.0


@dataclasses.dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate together with its lower and upper bounds.

    Attributes:
        low: Smallest rate compatible with the data at the chosen confidence.
        best: Maximum-likelihood estimate (``num_hits / num_shots``).
        high: Largest rate compatible with the data at the chosen confidence.
    """

    low: float
    best: float
    high: float

    def __post_init__(self) -> None:
        if not (self.low <= self.best <= self.high):
            msg = f"Expected low <= best <= high, got {self.low}, {self.best}, {self.high}."
            raise ValueError(msg)


def log_binomial(*, p: float, num_trials: int, num_successes: int) -> float:
    """Return ``ln P(num_successes | Binomial(num_trials, p))``.

    The computation is done in log space so that the tiny probabilities involved
    in large experiments stay representable.

    Args:
        p: Hypothesis probability, between 0 and 1.
        num_trials: Number of shots.
        num_successes: Number of failures observed.

    Returns:
        The natural log of the binomial likelihood.
    """
    p = min(max(p, 0.0), 1.0)
    misses = num_trials - num_successes
    result = 0.0
    if num_successes != 0:
        if p == 0:
            return -math.inf
        result += math.log(p) * num_successes
    if misses != 0:
        if p == 1:
            return -math.inf
        result += math.log1p(-p) * misses
    result += (
        math.lgamma(num_trials + 1)
        - math.lgamma(misses + 1)
        - math.lgamma(num_successes + 1)
    )
    return result


def fit_binomial(
    num_shots: int,
    num_hits: int,
    *,
    max_likelihood_factor: float = DEFAULT_MAX_LIKELIHOOD_FACTOR,
) -> ConfidenceInterval:
    """Estimate an error rate and its asymmetric confidence interval.

    The interval contains every rate whose binomial likelihood is within
    ``max_likelihood_factor`` of the most likely rate ``num_hits / num_shots``.

    Args:
        num_shots: Number of shots.
        num_hits: Number of failures observed.
        max_likelihood_factor: How much less likely than the best fit a rate may
            be before it is excluded from the interval. Must be at least 1.

    Returns:
        A :class:`ConfidenceInterval` with the best estimate and its low and high bounds.

    Raises:
        ValueError: If the inputs are out of range.
    """
    if max_likelihood_factor < 1:
        msg = f"max_likelihood_factor={max_likelihood_factor} must be >= 1."
        raise ValueError(msg)
    if num_shots < 0 or num_hits < 0 or num_hits > num_shots:
        msg = f"Need 0 <= num_hits ({num_hits}) <= num_shots ({num_shots})."
        raise ValueError(msg)
    if num_shots == 0:
        return ConfidenceInterval(low=0.0, best=0.5, high=1.0)

    best = num_hits / num_shots
    target = log_binomial(
        p=best, num_trials=num_shots, num_successes=num_hits
    ) - math.log(max_likelihood_factor)

    def gap(p: float) -> float:
        return log_binomial(p=p, num_trials=num_shots, num_successes=num_hits) - target

    # The binomial likelihood is unimodal in ``p`` with its peak at ``best``, so
    # ``gap`` is positive at the peak and decreases monotonically towards each
    # end. Each side therefore crosses zero exactly once, and a bracketed root
    # finder (Brent's method) returns that single crossing robustly without
    # needing derivatives.
    low = 0.0 if num_hits == 0 else float(optimize.brentq(gap, 1e-18, best))
    high = (
        1.0 if num_hits == num_shots else float(optimize.brentq(gap, best, 1 - 1e-18))
    )
    return ConfidenceInterval(low=low, best=best, high=high)


def fit_binomial_batch(
    num_shots: npt.NDArray[np.int_] | Sequence[int],
    num_hits: npt.NDArray[np.int_] | Sequence[int],
    *,
    max_likelihood_factor: float = DEFAULT_MAX_LIKELIHOOD_FACTOR,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Apply :func:`fit_binomial` to each (shots, hits) pair.

    Args:
        num_shots: Number of shots per point.
        num_hits: Number of failures per point.
        max_likelihood_factor: Passed through to :func:`fit_binomial`.

    Returns:
        Three arrays ``(low, best, high)`` with the same shape as the inputs.

    Raises:
        ValueError: If ``num_shots`` and ``num_hits`` have different shapes.
    """
    shots = np.asarray(num_shots, dtype=np.int_)
    hits = np.asarray(num_hits, dtype=np.int_)
    if shots.shape != hits.shape:
        msg = "num_shots and num_hits must have the same shape."
        raise ValueError(msg)

    fits = [
        fit_binomial(
            int(s),
            int(h),
            max_likelihood_factor=max_likelihood_factor,
        )
        for s, h in zip(shots.ravel(), hits.ravel(), strict=True)
    ]
    low = np.array([f.low for f in fits], dtype=np.float64).reshape(shots.shape)
    best = np.array([f.best for f in fits], dtype=np.float64).reshape(shots.shape)
    high = np.array([f.high for f in fits], dtype=np.float64).reshape(shots.shape)
    return low, best, high


# --- Binomial likelihood fit of the per-round model -------------------------
#
# The functions below fit the memory model
#
#     LEP(r) = (1 - (1 - 2 * spam) * (1 - 2 * eps) ** r) / 2
#
# to the raw counts by maximising the total binomial likelihood, then profile
# each parameter to obtain an asymmetric interval. Profiling means fixing the
# parameter of interest, re-optimising the other one, and walking outward until
# the negative log-likelihood has risen by ``num_sigmas ** 2 / 2`` (so a 1-sigma
# interval is reached at a rise of 0.5). This is the chi-square = 1 / MINOS rule
# that follows from Wilks' theorem -- ``-2 ln(likelihood ratio)`` is
# asymptotically chi-square distributed -- see
# https://statproofbook.github.io/P/ci-wilks.html. Unlike a weighted
# least-squares fit with symmetric sigmas, it keeps the bounds inside the
# physical range and reflects the genuine skew of the likelihood near zero.

# Physical bounds on a per-round/SPAM error probability: strictly above zero (so
# the log-likelihood stays finite) and strictly below 0.5 (the maximally mixed
# rate, where the model becomes degenerate).
_RATE_FLOOR = 1e-12
_RATE_CEIL = 0.5 - 1e-9


def _model_lep(
    rounds: npt.NDArray[np.float64], eps: float, spam: float
) -> npt.NDArray[np.float64]:
    return (1 - (1 - 2 * spam) * (1 - 2 * eps) ** rounds) / 2


def _total_neg_log_likelihood(
    eps: float,
    spam: float,
    rounds: npt.NDArray[np.float64],
    num_fails: npt.NDArray[np.int_],
    shots: npt.NDArray[np.int_],
) -> float:
    # The binomial "success" event here is a *logical failure*, so the per-point
    # success count is the number of logical failures ``num_fails`` and ``p`` is
    # the modelled logical error probability ``_model_lep``.
    model = _model_lep(rounds, eps, spam)
    return -sum(
        log_binomial(p=float(p), num_trials=int(n), num_successes=int(k))
        for p, n, k in zip(model, shots, num_fails, strict=True)
    )


def _best_fit(
    rounds: npt.NDArray[np.float64],
    num_fails: npt.NDArray[np.int_],
    shots: npt.NDArray[np.int_],
    eps_guess: float,
    spam_guess: float,
) -> tuple[float, float]:
    """Maximum-likelihood ``(eps, spam)``, optimised in log space for scaling.

    Args:
        rounds: Number of QEC rounds for each point.
        num_fails: Number of logical failures observed at each point.
        shots: Number of shots at each point.
        eps_guess: Initial guess for the per-round error.
        spam_guess: Initial guess for the SPAM error.

    Returns:
        The maximum-likelihood ``(eps, spam)``.
    """

    def cost(log_params: npt.NDArray[np.float64]) -> float:
        eps = float(np.clip(np.exp(log_params[0]), _RATE_FLOOR, _RATE_CEIL))
        spam = float(np.clip(np.exp(log_params[1]), _RATE_FLOOR, _RATE_CEIL))
        return _total_neg_log_likelihood(eps, spam, rounds, num_fails, shots)

    # Nelder-Mead is derivative-free and robust for this smooth, low-dimensional
    # cost; it is not guaranteed to find the global optimum, but the two rates
    # are well scaled in log space and the subsequent profiling re-evaluates the
    # likelihood around the returned point, so a small offset does not bias the
    # reported interval.
    start = np.log([max(eps_guess, _RATE_FLOOR), max(spam_guess, _RATE_FLOOR)])
    result = optimize.minimize(cost, x0=start, method="Nelder-Mead")
    eps = float(np.clip(np.exp(result.x[0]), _RATE_FLOOR, _RATE_CEIL))
    spam = float(np.clip(np.exp(result.x[1]), _RATE_FLOOR, _RATE_CEIL))
    return eps, spam


def _profiled_neg_log_likelihood(
    fixed_eps: bool,
    fixed_value: float,
    rounds: npt.NDArray[np.float64],
    num_fails: npt.NDArray[np.int_],
    shots: npt.NDArray[np.int_],
) -> float:
    """Negative log-likelihood with one parameter fixed and the other refitted.

    Args:
        fixed_eps: If True the per-round error is held at ``fixed_value`` and the
            SPAM error is re-optimised; if False the roles are swapped.
        fixed_value: Value of the fixed parameter.
        rounds: Number of QEC rounds for each point.
        num_fails: Number of logical failures observed at each point.
        shots: Number of shots at each point.

    Returns:
        The minimised negative log-likelihood over the free parameter.
    """

    def cost(other: float) -> float:
        eps, spam = (fixed_value, other) if fixed_eps else (other, fixed_value)
        return _total_neg_log_likelihood(eps, spam, rounds, num_fails, shots)

    result = optimize.minimize_scalar(
        cost, bounds=(_RATE_FLOOR, _RATE_CEIL), method="bounded"
    )
    return float(result.fun)


def _profile_from_cost(
    cost: Callable[[float], float],
    best_value: float,
    num_sigmas: float,
) -> ConfidenceInterval:
    """Turn a profiled negative-log-likelihood into an asymmetric interval.

    Walks outward from ``best_value`` until ``cost`` has risen by
    ``num_sigmas ** 2 / 2`` above its value at the best fit (the chi-square = 1
    rule), then locates that crossing on each side with a bracketed root find.

    Args:
        cost: The profiled negative log-likelihood as a function of the parameter
            of interest, minimal at ``best_value``.
        best_value: The maximum-likelihood value of the parameter.
        num_sigmas: Width of the interval, in sigmas.

    Returns:
        The ``(low, best, high)`` confidence interval.
    """
    # Anchor the target to the profiled likelihood at the best value (the global
    # minimum), so the root is always bracketed regardless of optimiser noise.
    target = cost(best_value) + 0.5 * num_sigmas**2

    def excess(value: float) -> float:
        return cost(value) - target

    def cross(bound: float) -> float:
        # If the likelihood never rises enough before the bound, the parameter is
        # consistent with that bound and the interval is open there.
        if excess(bound) <= 0:
            return bound
        low_x, high_x = (
            (bound, best_value) if bound < best_value else (best_value, bound)
        )
        return float(optimize.brentq(excess, low_x, high_x, xtol=1e-15))

    return ConfidenceInterval(
        low=cross(_RATE_FLOOR), best=best_value, high=cross(_RATE_CEIL)
    )


def _profile_interval(
    fixed_eps: bool,
    best_value: float,
    rounds: npt.NDArray[np.float64],
    num_fails: npt.NDArray[np.int_],
    shots: npt.NDArray[np.int_],
    num_sigmas: float,
) -> ConfidenceInterval:
    """Profile one parameter (refitting the other) for an asymmetric interval.

    Args:
        fixed_eps: If True profile the per-round error (refitting SPAM); if False
            profile the SPAM error (refitting the per-round error).
        best_value: The maximum-likelihood value of the profiled parameter.
        rounds: Number of QEC rounds for each point.
        num_fails: Number of logical failures observed at each point.
        shots: Number of shots at each point.
        num_sigmas: Width of the interval, in sigmas.

    Returns:
        The ``(low, best, high)`` confidence interval for the profiled parameter.
    """

    def cost(value: float) -> float:
        return _profiled_neg_log_likelihood(fixed_eps, value, rounds, num_fails, shots)

    return _profile_from_cost(cost, best_value, num_sigmas)


def fit_leppr_and_spam(
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    num_fails: npt.NDArray[np.int_] | Sequence[int],
    num_shots: npt.NDArray[np.int_] | Sequence[int],
    *,
    num_sigmas: float = 1.0,
    fixed_spam: float | None = None,
) -> tuple[ConfidenceInterval, ConfidenceInterval]:
    """Fit the per-round error and SPAM error with asymmetric intervals.

    The model ``LEP(r) = (1 - (1 - 2 * spam) * (1 - 2 * eps) ** r) / 2`` is fitted
    to the raw counts by maximum binomial likelihood, and each parameter is
    profiled to get an asymmetric confidence interval. ``num_sigmas = 1``
    corresponds to a chi-square = 1 interval.

    The per-round error and the SPAM error are correlated, so when the data only
    weakly constrains the per-round error its interval can extend down to zero. If
    the SPAM error is known independently, pass it as ``fixed_spam`` to remove the
    correlation and obtain a tighter per-round interval. A warning is emitted when
    the per-round interval is left open towards zero.

    Args:
        num_rounds: Number of QEC rounds for each measured point.
        num_fails: Number of logical failures observed at each round count.
        num_shots: Number of shots at each round count.
        num_sigmas: Width of the interval, in sigmas.
        fixed_spam: A known SPAM error to hold fixed. When given, only the
            per-round error is fitted.

    Returns:
        ``(leppr_fit, spam_fit)`` as :class:`ConfidenceInterval` instances.

    Raises:
        ValueError: If the inputs have different lengths.
    """
    rounds = np.asarray(num_rounds, dtype=np.float64)
    fails = np.asarray(num_fails, dtype=np.int_)
    shots = np.asarray(num_shots, dtype=np.int_)
    if not len(rounds) == len(fails) == len(shots):
        msg = "num_rounds, num_fails and num_shots must have the same length."
        raise ValueError(msg)

    lep = fails / shots
    eps_guess = float(
        np.clip(np.mean(lep / np.maximum(rounds, 1)), _RATE_FLOOR, _RATE_CEIL)
    )

    if fixed_spam is not None:
        spam_best = float(np.clip(fixed_spam, _RATE_FLOOR, _RATE_CEIL))
        # Fit in log space so the small rates are well scaled.
        eps_only = optimize.minimize_scalar(
            lambda log_eps: _total_neg_log_likelihood(
                float(np.clip(np.exp(log_eps), _RATE_FLOOR, _RATE_CEIL)),
                spam_best,
                rounds,
                fails,
                shots,
            ),
            bounds=(float(np.log(_RATE_FLOOR)), float(np.log(_RATE_CEIL))),
            method="bounded",
        )
        eps_best = float(np.clip(np.exp(eps_only.x), _RATE_FLOOR, _RATE_CEIL))
        # SPAM is held fixed, so the per-round profile is just the total
        # likelihood as a function of eps alone.
        leppr_fit = _profile_from_cost(
            lambda eps: _total_neg_log_likelihood(eps, spam_best, rounds, fails, shots),
            eps_best,
            num_sigmas,
        )
        spam_fit = ConfidenceInterval(low=spam_best, best=spam_best, high=spam_best)
        return leppr_fit, spam_fit

    spam_guess = float(np.clip(lep[0] if lep.size else 0.0, _RATE_FLOOR, _RATE_CEIL))
    eps_best, spam_best = _best_fit(rounds, fails, shots, eps_guess, spam_guess)
    leppr_fit = _profile_interval(True, eps_best, rounds, fails, shots, num_sigmas)
    spam_fit = _profile_interval(False, spam_best, rounds, fails, shots, num_sigmas)

    if leppr_fit.low <= _RATE_FLOOR * 10 < leppr_fit.best:
        warnings.warn(
            "The per-round error interval is open towards zero: the data weakly "
            "constrains it, most likely because of correlation with the SPAM error. "
            "Pass `fixed_spam` if the SPAM error is known independently.",
            stacklevel=2,
        )
    return leppr_fit, spam_fit
