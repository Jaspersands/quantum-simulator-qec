# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from stabilizer_qec.deltakit.circuit._circuit import Circuit

from stabilizer_qec.deltakit.explorer.analysis.error_budget._gradient import inverse_lambda_gradient_at
from stabilizer_qec.deltakit.explorer.analysis.error_budget._memory import (
    MemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from stabilizer_qec.deltakit.explorer.analysis.error_budget._parameters import (
    FittingParameters,
    SamplingParameters,
)


@dataclass
class ErrorBudgetResult:
    """Result of an error budgeting computation.

    Attributes:
        contributions: contributions for each of the noise parameters to the error budget.
        contribution_stddevs: estimation of the standard deviation of each of the ``contributions``.
    """

    contributions: tuple[float, ...]
    contribution_stddevs: tuple[float, ...]

    @property
    def lambda_estimate(self) -> float:
        """Returns the estimation of Λ according to the computed budget."""
        return float(np.sum(self.contributions))

    @property
    def lambda_stddev_estimate(self) -> float:
        """Returns an estimation of the standard deviation on Λ according to the computed budget."""
        return float(np.sqrt(np.sum(np.asarray(self.contribution_stddevs) ** 2)))

    @staticmethod
    def from_gradient(
        gradient: npt.NDArray[np.floating],
        gradient_stddev: npt.NDArray[np.floating],
        noise_parameters: npt.NDArray[np.floating],
    ) -> ErrorBudgetResult:
        """Create an instance from gradient and noise parameters.

        Args:
            gradient: gradient of 1 / Λ with respect to each noise parameter.
            gradient_stddev: standard deviation of each entry in ``gradient``.
            noise_parameters: noise parameters the contributions are computed for.

        Returns:
            an error-budget result whose contributions are ``|gradient *
            noise_parameters|`` and whose standard deviations are
            ``|gradient_stddev * noise_parameters|``.
        """
        contributions = np.abs(gradient * noise_parameters)
        stddevs = np.abs(gradient_stddev * noise_parameters)
        return ErrorBudgetResult(
            tuple(map(float, contributions.ravel())), tuple(map(float, stddevs.ravel()))
        )


def get_error_budget(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    noise_parameters_exploration_bounds: list[tuple[float, float]],
    fitting_parameters: FittingParameters = FittingParameters(),
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
) -> ErrorBudgetResult:
    """Compute the error budget of the provided ``noise_model``.

    Args:
        noise_model: a callable adding noise to the provided circuit, according to the
            parameters provided.
        noise_parameters: valid parameters to forward to ``noise_model`` representing
            the point at which the gradient should be computed.
        num_rounds_by_distances: a mapping from each code distance that should be tested
            to the number of rounds that should be sampled in order to estimate the
            logical error-probability per round, to ultimately get 1 / Λ.
        noise_parameters_exploration_bounds: ``(min, max)`` bounds for each noise
            parameter of the provided ``noise_model``. A degree ``fitting_degree``
            polynomial will be fitted on the interval ``[min, max]``. The corresponding
            noise parameter from the provided ``noise_model`` should be strictly
            contained in ``[min, max]`` (i.e., for any valid ``i``, the following is
            true:
            ``noise_parameters_exploration_bounds[i][0] <
            noise_model.noise_parameters[i] <
            noise_parameters_exploration_bounds[i][1]``).
            Ideally, the lower (resp. upper) bound provided must be such that the
            logical error probability when replacing the parameter with its lower (resp.
            upper) bound is above ``100 / max_shots`` to ensure enough fails are
            observed with ``max_shots`` shots (resp. below ``1 / 2`` to ensure that we
            can compute the logical error probability per round).
        fitting_parameters: additional parameters relating to how the gradient is
            estimated.
        sampling_parameters: additional parameters relating to the sampling tasks used to
            estimate 1 / Λ indirectly.
        memory_generator: a callable that can generate a memory experiment. The
            resulting circuit will go through the provided ``noise_model`` for different
            values of the noise parameters.

    Returns:
        the error-budgeting result, which consists of an array of contributions for each
        of the noise parameters of the provided ``noise_model`` along with their
        associated standard deviations.
    """
    # We will compute the gradient at the half point following the methodology outlined in
    # https://doi.org/10.1038/s41586-021-03588-y (Supplementary materials, Section VIII.C.).
    point = np.asarray(noise_parameters) / 2
    # Evaluate the gradient.
    gradient, gradient_stddev = inverse_lambda_gradient_at(
        noise_model,
        point,
        num_rounds_by_distances,
        noise_parameters_exploration_bounds,
        fitting_parameters,
        sampling_parameters,
        memory_generator,
    )
    return ErrorBudgetResult.from_gradient(
        gradient, gradient_stddev, np.asarray(noise_parameters)
    )
