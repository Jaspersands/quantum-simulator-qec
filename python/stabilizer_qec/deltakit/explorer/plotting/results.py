# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""Result types for plotting LEPPR and Lambda data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from stabilizer_qec.deltakit.explorer.analysis import LambdaData
from stabilizer_qec.deltakit.explorer.analysis import (
    LogicalErrorProbabilityPerRoundData as LEPPRData,
)


def _lambda_interpolated(
    lambda0: float | npt.NDArray[np.floating],
    lambda_: float | npt.NDArray[np.floating],
    distances: npt.NDArray[np.int_],
) -> npt.NDArray[np.floating]:
    """Interpolate the logical error probability per round for the given parameters.

    The interpolation is based on the formula

        ε = (1 / Λ₀) * Λ**(-(d + 1) / 2)

    where:
      - ε is the logical error probability per round,
      - Λ₀ is a normalisation constant,
      - Λ is the error suppression factor,
      - d is the code distance.

    For each distance, this function computes the corresponding
    logical error probability using the supplied Λ and Λ₀.

    Args:
        lambda0: Normalisation constant Λ₀.
        lambda_: Error suppression factor Λ.
        distances: Iterable of code distances d.

    Returns:
        An array containing the estimated logical error probability per round
        for each provided distance.
    """
    return lambda_ ** (-(distances + 1) / 2) / lambda0


def _lep_interpolated(
    spam: float, leppr: float, rounds_interpolated: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Compute the logical error probability for the given parameters.

    The expected computation fidelity is modelled as

        F = Fs * Fε**r

    where:
      - F is the overall fidelity of the computation,
      - Fs is the fidelity of SPAM-related operations,
       Fε is the fidelity of a single quantum error-correction round,
      - r is the number of error-correction rounds performed.

    Each fidelity value is derived from its associated error probability using

        f = 1 - 2e

    where:
      - f represents F, Fs, or Fε,
      - e represents the corresponding logical error probability
        (overall, SPAM-related, or per-round).

    Args:
        spam: Error probability associated with SPAM operations.
        leppr: Error probability per error-correction round.
        rounds_interpolated: Number of error-correction rounds performed.

    Returns:
        Logical error probability of the full computation.
    """
    expected_fidelity = (1 - 2 * spam) * (1 - 2 * leppr) ** rounds_interpolated
    return (1 - expected_fidelity) / 2


def _error_rate_band(
    value: float, stddev: float, num_sigmas: float
) -> tuple[float, float]:
    """Asymmetric ``num_sigmas`` band for an error probability.

    The error probability ``e`` is fitted through the fidelity ``f = 1 - 2 e``,
    which is the quantity that is Gaussian after the log-linear fit. Propagating
    the band in that space and mapping back gives bounds that stay below ``0.5``
    and are not symmetric around ``value`` once ``value`` is small.

    Args:
        value: The fitted error probability.
        stddev: Its standard deviation.
        num_sigmas: Width of the band, in standard deviations.

    Returns:
        The ``(low, high)`` bounds. The caller is expected to clip to ``[0, 1]``.
    """
    fidelity = 1 - 2 * value
    if fidelity <= 0:
        # Outside the modelled regime; fall back to a symmetric band.
        return value - num_sigmas * stddev, value + num_sigmas * stddev
    sigma_log = 2 * stddev / fidelity
    low = (1 - fidelity * np.exp(num_sigmas * sigma_log)) / 2
    high = (1 - fidelity * np.exp(-num_sigmas * sigma_log)) / 2
    return low, high


def _suppression_band(
    value: float, stddev: float, num_sigmas: float
) -> tuple[float, float]:
    """Asymmetric ``num_sigmas`` band for a positive suppression factor.

    Λ and Λ₀ are exponentials of the linear fit parameters, so the natural band
    is multiplicative. This keeps both bounds strictly positive, unlike a plain
    ``value ± num_sigmas · stddev``.

    Args:
        value: The fitted suppression factor (Λ or Λ₀).
        stddev: Its standard deviation.
        num_sigmas: Width of the band, in standard deviations.

    Returns:
        The ``(low, high)`` bounds.
    """
    if value <= 0:
        return value - num_sigmas * stddev, value + num_sigmas * stddev
    sigma_log = stddev / value
    return value * np.exp(-num_sigmas * sigma_log), value * np.exp(
        num_sigmas * sigma_log
    )


@dataclass(frozen=True)
class Interpolated:
    """Container for interpolated plotting data and associated confidence bounds.

    Stores the interpolated central values together with their lower and upper
    confidence interval boundaries, along with labels describing the fit and
    its confidence interval for visualisation purposes.

    Attributes:
        interpolated: Array of interpolated central values (e.g., fitted curve values).
        lower_boundary: Array representing the lower bound of the confidence interval
            corresponding to ``interpolated``.
        upper_boundary: Array representing the upper bound of the confidence interval
            corresponding to ``interpolated``.
        fit_label: Label describing the fitted/interpolated data (e.g., legend entry).
        confidence_interval_label: Label describing the confidence interval region (e.g., legend entry).
    """

    interpolated: npt.NDArray[np.floating]
    lower_boundary: npt.NDArray[np.floating]
    upper_boundary: npt.NDArray[np.floating]
    fit_label: str
    confidence_interval_label: str

    def __post_init__(self) -> None:
        """Validate consistency of interpolated data and confidence bounds.

        Ensures that ``interpolated``, ``lower_boundary``, and ``upper_boundary``
        arrays all have identical shapes so that each interpolated value has
        corresponding lower and upper confidence bounds.

        Raises:
            ValueError: If the shapes of ``interpolated``,
                ``lower_boundary``, and ``upper_boundary`` do not match.
        """
        if not (
            self.interpolated.shape
            == self.lower_boundary.shape
            == self.upper_boundary.shape
        ):
            msg = (
                "The 'interpolated', 'lower_boundary', and 'upper_boundary' arrays "
                f"must have the same shape. Got {self.interpolated.shape}, "
                f"{self.lower_boundary.shape}, and {self.upper_boundary.shape} respectively."
            )
            raise ValueError(msg)

        # Check that provided interpolated is within [0, 1]
        # boundaries are also within [0, 1]
        # Since the fit could technically exceed it slightly or we just want to warn/clip.
        # Provided `interpolated` is within `[0, 1)`, boundaries are also within `[0, 1)`.
        if not np.all((self.interpolated >= 0) & (self.interpolated <= 1)):
            msg = "Interpolated values must be within [0, 1]"
            raise ValueError(msg)
        if not np.all((self.lower_boundary >= 0) & (self.lower_boundary <= 1)):
            msg = "Lower boundary values must be within [0, 1]"
            raise ValueError(msg)
        if not np.all((self.upper_boundary >= 0) & (self.upper_boundary <= 1)):
            msg = "Upper boundary values must be within [0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True)
class LambdaResult(Interpolated):
    """Result type holding the data needed to plot a Lambda fit.

    Attributes:
        distances: Interpolated distance grid for the fit curve.
    """

    distances: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.all(self.distances > 0):
            msg = "Distances must be positive."
            raise ValueError(msg)
        if self.distances.shape != self.interpolated.shape:
            msg = (
                f"The 'distances' array shape {self.distances.shape} must match the "
                f"'interpolated' array shape {self.interpolated.shape}."
            )
            raise ValueError(msg)


def interpolate_lambda(
    lambda_data: LambdaData,
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
) -> LambdaResult:
    """Interpolate the Λ fit and compute confidence bands.

    Args:
        lambda_data: Result of a fit containing Λ, Λ₀, their standard deviations, and the original data.
        num_sigmas: Number of standard deviations for the error band. Default 3.
        num_points: Number of interpolation points. Default 200.

    Returns:
        A container for the interpolated fit data with error boundaries and confidence interval.
    """
    # Reshape distances into a (1, num_points) array for vectorisation
    distances_interpolated = np.linspace(
        lambda_data.distances[0], lambda_data.distances[-1], num_points
    ).reshape(1, num_points)

    # Use the bounds from an asymmetric fit if they are available, otherwise
    # build a multiplicative band from the symmetric standard deviations (Λ and Λ₀
    # are exponentials of the fit parameters, so the band is not symmetric).
    if lambda_data.has_asymmetric_bounds:
        assert lambda_data.lambda_interval is not None
        assert lambda_data.lambda0_interval is not None
        lambda_low, lambda_high = (
            lambda_data.lambda_interval.low,
            lambda_data.lambda_interval.high,
        )
        lambda0_low, lambda0_high = (
            lambda_data.lambda0_interval.low,
            lambda_data.lambda0_interval.high,
        )
    else:
        lambda_low, lambda_high = _suppression_band(
            lambda_data.lambda_, lambda_data.lambda_std, num_sigmas
        )
        lambda0_low, lambda0_high = _suppression_band(
            lambda_data.lambda0, lambda_data.lambda0_std, num_sigmas
        )

    # The error probability decreases with both Λ and Λ₀, so the largest factors
    # give the lower boundary of the band and the smallest give the upper one.
    interpolated = _lambda_interpolated(
        lambda_data.lambda0, lambda_data.lambda_, distances_interpolated
    ).ravel()
    lower_boundary = _lambda_interpolated(
        lambda0_high, lambda_high, distances_interpolated
    ).ravel()
    upper_boundary = _lambda_interpolated(
        lambda0_low, lambda_low, distances_interpolated
    ).ravel()

    fit_label = (
        f"Fit, Λ={lambda_data.lambda_:.4f} "
        f"(+{lambda_high - lambda_data.lambda_:.4f} / "
        f"-{lambda_data.lambda_ - lambda_low:.4f}, {num_sigmas}σ)"  # noqa: RUF001
    )

    return LambdaResult(
        distances=distances_interpolated.ravel(),  # Reshape back into 1D array.
        interpolated=np.clip(interpolated, 0, 1),
        lower_boundary=np.clip(lower_boundary, 0, 1),
        upper_boundary=np.clip(upper_boundary, 0, 1),
        fit_label=fit_label,
        confidence_interval_label=f"Confidence interval ({num_sigmas}σ) on Λ fit",  # noqa: RUF001
    )


@dataclass(frozen=True)
class LogicalErrorProbabilityPerRoundResult(Interpolated):
    """Result type holding the data needed to plot a LogicalErrorProbabilityPerRound (LEPPR) fit.

    Attributes:
        rounds: Interpolated rounds grid for the fit curve.
    """

    rounds: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.all(self.rounds > 0):
            msg = "Rounds must be positive."
            raise ValueError(msg)
        if self.rounds.shape != self.interpolated.shape:
            msg = (
                f"The 'rounds' array shape {self.rounds.shape} must match the "
                f"'interpolated' array shape {self.interpolated.shape}."
            )
            raise ValueError(msg)


def interpolate_leppr(
    leppr_data: LEPPRData,
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
) -> LogicalErrorProbabilityPerRoundResult:
    """Compute the interpolated LEPPR fit curve and its error band.

    Args:
        leppr_data: Results from compute_logical_error_per_round.
        num_sigmas: Number of standard deviations for the error band. Default 3.
        num_points: Number of interpolation points. Default 200.

    Returns:
        The interpolated fit data with error boundaries.

    """
    # Reshape rounds into a (1, num_points) array for vectorisation
    rounds_interpolated = np.linspace(
        leppr_data.num_rounds[0], leppr_data.num_rounds[-1], num_points
    ).reshape(1, num_points)

    # Build asymmetric bands for the SPAM error and the per-round error, then
    # combine them through the fidelity model. Both bands are ordered
    # (low, best, high), and the experiment error grows with both quantities, so
    # the first row gives the lower boundary and the last row the upper one. The
    # bounds from an asymmetric fit are used when available, otherwise they are
    # derived from the symmetric standard deviations.
    if (
        leppr_data.leppr_low is not None
        and leppr_data.leppr_high is not None
        and leppr_data.spam_error_low is not None
        and leppr_data.spam_error_high is not None
    ):
        spam_low, spam_high = leppr_data.spam_error_low, leppr_data.spam_error_high
        leppr_low, leppr_high = leppr_data.leppr_low, leppr_data.leppr_high
    else:
        spam_low, spam_high = _error_rate_band(
            leppr_data.spam_error, leppr_data.spam_error_stddev, num_sigmas
        )
        leppr_low, leppr_high = _error_rate_band(
            leppr_data.leppr, leppr_data.leppr_stddev, num_sigmas
        )
    spam_vals = np.array([spam_low, leppr_data.spam_error, spam_high]).reshape(3, 1)
    leppr_vals = np.array([leppr_low, leppr_data.leppr, leppr_high]).reshape(3, 1)

    lower_boundary, interpolated, upper_boundary = _lep_interpolated(
        spam_vals, leppr_vals, rounds_interpolated
    )

    fit_label = (
        f"Fit, ε={leppr_data.leppr:.4f} "
        f"(+{leppr_high - leppr_data.leppr:.4f} / "
        f"-{leppr_data.leppr - leppr_low:.4f}, {num_sigmas}σ)"  # noqa: RUF001
    )

    return LogicalErrorProbabilityPerRoundResult(
        rounds=rounds_interpolated.ravel(),  # Reshape back into 1D array
        interpolated=np.clip(interpolated, 0, 1),
        lower_boundary=np.clip(lower_boundary, 0, 1),
        upper_boundary=np.clip(upper_boundary, 0, 1),
        fit_label=fit_label,
        confidence_interval_label=f"Confidence interval ({num_sigmas}σ) on ε fit",  # noqa: RUF001
    )
