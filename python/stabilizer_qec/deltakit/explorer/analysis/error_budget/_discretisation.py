# (c) Copyright Riverlane 2020-2026. All rights reserved.
from enum import Enum, auto
from typing import Protocol

import numpy as np
import numpy.typing as npt
import scipy.optimize

# Number of candidate points the c-optimal optimiser searches over. In testing, this seemed like
# a healthy balance of precision and cost.
_C_OPTIMAL_CANDIDATE_GRID_SIZE = 50

# We have to set a maximum threshold in our c optimal design to prevent it from returning
# a singular matrix (i.e just sample 1 point for every run)
_C_OPTIMAL_COND_THRESHOLD = 1e10


class GradientFitDiscretisationGenerator(Protocol):
    def __call__(
        self, a: float, b: float, c: float, num_points: int, degree: int, /
    ) -> npt.NDArray[np.floating]:
        """
        Find a good set of points to estimate the gradient at point ``c`` by fitting a
        polynomial of degree ``degree`` on the interval ``[a, b]``.

        Functions implementing that interface find ``num_points`` unique points ``x``
        between ``a`` and ``b`` that will be used to fit a polynomial of degree
        ``degree``.

        Different implementation might choose to minimise different quantities. For
        example:

        - minimise code complexity, by simply returning linear (or log-spaced) points,
        - minimise the variance of the resulting gradient estimation, for example using
          a D-optimal design (sees
          https://en.wikipedia.org/wiki/Optimal_experimental_design).

        Implementations should raise a ``ValueError`` if ``a < c < b`` is not verified.

        Args:
            a: lower bound of the interval in which fitting points should be computed.
            b: upper bound of the interval in which fitting points should be computed.
            c: point at which the gradient will be estimated.
            num_points: number of points to return within ``[a, b]``.
            degree: degree of the polynomial that will be used to fit the values
                computed at each of the points returned by this function.

        Returns:
            a sorted array of ``num_points`` points within ``[a, b]`` and without duplicates
            that should be used to evaluate the function to fit with a degree ``degree``
            polynomial.
        """
        ...


def _check_interval(a: float, b: float, c: float) -> None:
    if not a < c < b:
        msg = f"Expected {a=} < {c=} < {b=}"
        raise ValueError(msg)


def get_linear_points(
    a: float, b: float, c: float, num_points: int, _: int
) -> npt.NDArray[np.floating]:
    """Returns ``num_points`` linearly spaced between ``a`` and ``b``.

    Args:
        a: lower bound of the interval.
        b: upper bound of the interval.
        c: point at which the gradient will be estimated.
        num_points: number of points to return within ``[a, b]``.

    Returns:
        ``num_points`` linearly spaced points from ``a`` to ``b``, both included.
    """
    _check_interval(a, b, c)
    return np.linspace(a, b, num_points)


def get_logarithmic_points(
    a: float, b: float, c: float, num_points: int, _: int
) -> npt.NDArray[np.floating]:
    """Returns ``num_points`` logarithmically spaced between ``a`` and ``b``.

    Args:
        a: lower bound of the interval. Must be strictly positive.
        b: upper bound of the interval.
        c: point at which the gradient will be estimated.
        num_points: number of points to return within ``[a, b]``.

    Returns:
        ``num_points`` logarithmically spaced points from ``a`` to ``b``, both
        included.

    Raises:
        ValueError: if ``a < c < b`` is not verified or if ``a <= 0``.
    """
    _check_interval(a, b, c)
    if a <= 0:
        msg = (
            "Cannot get logarithmically-spaced points for negative values. "
            f"Got [{a}, {b}]."
        )
        raise ValueError(msg)
    return np.logspace(np.log10(a), np.log10(b), num_points, base=10)


def _get_variance_of_gradient_estimation_at_point(
    cov: npt.NDArray[np.floating], c: float
) -> float:
    """Reusing this function created by Adrien Suau (nelimee) that used to live in
    _gradient.py but was removed recently.

    Get the variance of the gradient estimation at the point ``c`` for a polynomial
    with uncertainties on its coefficients provided by the covariance matrix ``cov``.

    Args:
        cov: an array of shape ``(d + 1, d + 1)`` representing the covariance
            matrix of the coefficients defining the degree-d polynomial used to
            estimate the gradient.
        c: point at which the degree-d polynomial will be used to estimate the
            gradient value.

    Returns:
        The variance of the gradient estimation at point ``c``.
    """
    # From https://en.wikipedia.org/wiki/Covariance#Covariance_of_linear_combinations we
    # have an easy formula for the variance involving the covariance matrix.
    n = cov.shape[0]
    coeff_matrix = np.array(
        [[(i + 1) * (j + 1) * c ** (i + j) for i in range(n - 1)] for j in range(n - 1)]
    )
    return float(np.sum(coeff_matrix * cov[1:, 1:]))


def _c_optimal_objective(
    indices: npt.NDArray[np.floating],
    candidate_grid: npt.NDArray[np.floating],
    degree: int,
    c: float,
) -> float:
    """Slope variance at ``c`` for the design at the given candidate indices.

    The coefficient covariance is computed as cov = (X.T @ X)^-1, where
    X is the Vandermonde matrix
    (https://en.wikipedia.org/wiki/Vandermonde_matrix); up to the assumed constant
    noise variance this is the covariance of the fitted polynomial
    coefficients.

    The variance is computed in rescaled ``[-1, 1]`` coordinates so the
    Vandermonde columns are all ``O(1)`` and ``cond(X.T @ X)`` is a meaningful
    measure of design quality rather than absolute x-scale.

    It's worth noting in this objective function there's two points of discontuinity,
    the first is where where we round `indices` to integers, the second is when
    we return np.inf. We use `differential_evolution` to optimize this function
    and it behaves fine but other optimisation methods may not.

    Args:
        indices: candidate-grid indices (floats, rounded internally) selecting
            the design points to evaluate.
        candidate_grid: the full grid of candidate x-values to choose from in
            sorted order (increasing or decreasing).
        degree: polynomial degree of the downstream fit.
        c: evaluation point at which the slope variance is computed.

    Returns:
        the slope-variance objective at ``c`` for this design, or ``inf`` if the
        design is ill-conditioned.
    """

    x = candidate_grid[indices.astype(int)]

    # Rescale to [-1, 1]. Without rescaling, the Vandermonde columns at typical
    # noise-parameter scales (x ~ 1e-3) span 12+ orders of magnitude and
    # cond(X.T @ X) ~ 1e15+ even for well-spread designs — making any
    # conditioning check meaningless.
    a, b = candidate_grid[0], candidate_grid[-1]
    u = 2 * (x - a) / (b - a) - 1
    uc = 2 * (c - a) / (b - a) - 1

    X = np.vander(u, degree + 1, increasing=True)
    XTX = X.T @ X

    # Reject ill-conditioned (effectively singular) designs. The optimiser
    # sees +inf and naturally avoids these regions.
    if np.linalg.cond(XTX) > _C_OPTIMAL_COND_THRESHOLD:
        return float("inf")
    try:
        cov = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        return float("inf")

    return _get_variance_of_gradient_estimation_at_point(cov, uc)


def get_c_optimal_points(
    a: float, b: float, c: float, num_points: int, degree: int
) -> npt.NDArray[np.floating]:
    """Returns ``num_points`` in ``[a, b]`` minimising slope variance at ``c``.

    Implements the c-optimal experimental design criterion for polynomial
    regression. Produces a design that minimises the variance of the Gauss-
    Markov best linear unbiased estimator of the polynomial's slope at the
    evaluation point ``c`` for a degree-``degree`` fit.

    Optimisation runs ``scipy.optimize.differential_evolution`` with
    ``seed=0`` over indices into a linear candidate grid of size
    `_C_OPTIMAL_CANDIDATE_GRID_SIZE` — results are deterministic across
    calls with identical inputs.

    The returned design may contain duplicate values (replication), which
    encode "spend more shots at this x-value". For replication to translate
    into actual precision in the error-budget pipeline, the post-processing
    must aggregate fails and shots across duplicate sweep rows — this is
    handled by
    :func:`_post_processing._compute_logical_error_rate_per_round_from_results`.

    Args:
        a: lower bound of the interval.
        b: upper bound of the interval.
        c: evaluation point at which slope variance is minimised. Must satisfy
            ``a < c < b``.
        num_points: number of design points to pick. Must be >= ``degree + 1``.
        degree: polynomial degree of the downstream fit.

    Returns:
        a sorted array of ``num_points`` values in ``[a, b]``. May contain
        duplicates.

    Raises:
        ValueError: if ``c`` is not a scalar, if ``a < c < b`` is not verified,
            or if ``num_points`` is below ``degree + 1``.
    """
    # Contract check: ``c`` must be a scalar. A non-scalar (e.g. a length-1
    # array) silently produces the wrong objective via broadcasting in
    # ``_get_variance_of_gradient_estimation_at_point``, so fail loudly here
    # rather than coercing and hiding a caller bug.
    if np.ndim(c) != 0:
        msg = f"c must be a scalar, got an array of shape {np.shape(c)}."
        raise ValueError(msg)
    c = float(c)

    _check_interval(a, b, c)
    if num_points < degree + 1:
        msg = f"For a degree {degree} polynomial, you must sample at least {degree + 1} points."
        raise ValueError(msg)

    candidate_grid = np.linspace(a, b, _C_OPTIMAL_CANDIDATE_GRID_SIZE)
    bounds = [(0, _C_OPTIMAL_CANDIDATE_GRID_SIZE - 1) for _ in range(num_points)]

    # Find the optimal x via differential evolution algorithm. Differential evolution
    # is a standard approach to this kind of problem. It doesn't rely on finding the
    # gradient of the objective function and it tests many regions of the solution
    # space simultaneously (unlike simulated annealing).
    # The mutation and recombination parameters were chosen to be near the SciPy defaults
    # with some brief testing to verify. The workers and seed parameters make the output
    # deterministic (given the same package versions)
    result = scipy.optimize.differential_evolution(
        _c_optimal_objective,
        bounds=bounds,
        args=(candidate_grid, degree, c),
        mutation=(0.5, 1.5),
        recombination=0.7,
        workers=1,
        seed=0,
    )

    optimal_indices = result.x.astype(int)
    return np.sort(candidate_grid[optimal_indices])


class DiscretisationStrategy(Enum):
    """Strategy to use to generate discretisation point for fitting a noisy function
    with a polynomial.

    Attributes:
        LINEAR: Linearly spaced points between the discretisation space boundaries.
        LOGARITHMIC: Logarithmically spaced points between the discretisation space
            boundaries.
        C_OPTIMAL: Points chosen to minimise the variance of the gradient estimate
            at the central evaluation point ``c``. Implements the c-optimal
            experimental design criterion for polynomial regression. Empirically
            achieves ~0.5x the slope variance of :attr:`LINEAR` at the same shot
            budget for typical parameter ranges, but produces clustered designs
            that may be sensitive to model misspecification. Recommended when
            slope-at-``c`` precision is the primary target and the polynomial
            degree is well-validated for the noise model. Implementation uses
            :func:`scipy.optimize.differential_evolution` and is deterministic
            across calls.
    """

    LINEAR = auto()
    LOGARITHMIC = auto()
    C_OPTIMAL = auto()

    def __call__(
        self, a: float, b: float, c: float, num_points: int, degree: int, /
    ) -> npt.NDArray[np.floating]:
        match self:
            case DiscretisationStrategy.LINEAR:
                return get_linear_points(a, b, c, num_points, degree)
            case DiscretisationStrategy.LOGARITHMIC:
                return get_logarithmic_points(a, b, c, num_points, degree)
            case DiscretisationStrategy.C_OPTIMAL:
                return get_c_optimal_points(a, b, c, num_points, degree)
            case _:
                msg = f"Discretisation {self} is not implemented yet."
                raise NotImplementedError(msg)
