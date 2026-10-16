# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from stabilizer_qec.deltakit.circuit._circuit import Circuit
from stabilizer_qec.deltakit.decode.analysis import RunAllAnalysisEngine
from uncertainties import ufloat

from stabilizer_qec.deltakit.explorer.analysis._binomial_fit import ConfidenceInterval
from stabilizer_qec.deltakit.explorer.analysis.error_budget._generation import (
    generate_decoder_managers_for_lambda,
)
from stabilizer_qec.deltakit.explorer.analysis.error_budget._memory import (
    MemoryGenerator,
    PreComputedMemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from stabilizer_qec.deltakit.explorer.analysis.error_budget._parameters import SamplingParameters
from stabilizer_qec.deltakit.explorer.analysis.error_budget._post_processing import (
    _filter_non_close_noise_parameters,
    compute_lambda_and_stddev_from_results,
    compute_lambda_interval_from_results,
)


def _run_lambda_engine(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    sampling_parameters: SamplingParameters,
    memory_generator: MemoryGenerator | Mapping[int, Mapping[int, Circuit]],
) -> tuple[npt.NDArray[np.floating], list[str], pd.DataFrame]:
    """Sample the memory experiments needed to estimate Λ at a single point.

    Args:
        noise_model: a callable adding noise to a circuit from the parameters.
        noise_parameters: the parameters forwarded to ``noise_model``.
        num_rounds_by_distances: a mapping from each code distance to the number
            of rounds to sample.
        sampling_parameters: parameters relating to the sampling tasks.
        memory_generator: a callable that generates a memory experiment.

    Returns:
        The reshaped noise parameters, their identifier names, and the report
        frame produced by the analysis engine.
    """
    if isinstance(memory_generator, Mapping):
        memory_generator = PreComputedMemoryGenerator(memory_generator)

    point = np.asarray(noise_parameters).reshape((-1, 1))

    # Create unique identifiers for noise parameters that will be used to
    # discriminate between them in the CSV file storing the simulation results.
    noise_parameter_names = [str(i) for i in range(point.size)]

    decoder_managers = generate_decoder_managers_for_lambda(
        point,
        noise_model,
        num_rounds_by_distances,
        sampling_parameters.max_workers,
        memory_generator=memory_generator,
        noise_parameter_names=noise_parameter_names,
    )
    engine = RunAllAnalysisEngine(
        experiment_name="Estimating 1 / Λ",
        decoder_managers=decoder_managers,
        max_shots=sampling_parameters.max_shots,
        batch_size=sampling_parameters.batch_size,
        # Early stopping when we have a low-enough standard deviation
        loop_condition=RunAllAnalysisEngine.loop_until_observable_rse_below_threshold(
            sampling_parameters.lep_target_rse,
            sampling_parameters.lep_computation_min_fails,
        ),
        num_parallel_processes=sampling_parameters.max_workers,
    )
    return point, noise_parameter_names, engine.run()


def reciprocal_stddev(value: float, stddev: float) -> float:
    """Standard deviation of ``1 / value`` via the ``uncertainties`` package.

    Args:
        value: Nominal value.
        stddev: Standard deviation of the nominal value.

    Returns:
        Standard deviation of the reciprocal.
    """
    return float((1 / ufloat(value, stddev)).std_dev)


def inverse_lambda_at(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
) -> ConfidenceInterval:
    """Compute 1 / Λ.

    Warning:
        This is a helper function to compute 1 / Λ when you need a **single**
        evaluation.
        For error budgeting, :func:`~deltakit_explorer.analysis.error_budget.get_error_budget`
        will be able to parallelise more efficiently, while also performing several
        checks and optimisations.

    Args:
        noise_model: a callable adding noise to the provided circuit, according to the
            parameters provided.
        noise_parameters: valid parameters to forward to ``noise_model`` representing
            the point at which the gradient should be computed.
        num_rounds_by_distances: a mapping from each code distance that should be tested
            to the number of rounds that should be sampled in order to estimate the
            logical error-probability per round, to ultimately get 1 / Λ.
        sampling_parameters: additional parameters relating to the sampling tasks used to
            estimate 1 / Λ indirectly.
        memory_generator: a callable that can generate a memory experiment. The
            resulting circuit will go through the provided ``noise_model`` for different
            values of the noise parameters.

    Returns:
        A :class:`ConfidenceInterval` for 1 / Λ. Its bounds are the symmetric
        ``value ± stddev`` (a 1-sigma interval); the asymmetric counterpart
        :func:`inverse_lambda_interval_at` profiles the likelihood for genuinely
        asymmetric bounds.
    """
    point, noise_parameter_names, report = _run_lambda_engine(
        noise_model,
        noise_parameters,
        num_rounds_by_distances,
        sampling_parameters,
        memory_generator,
    )
    lambdas, lambda_stddevs = compute_lambda_and_stddev_from_results(
        point, noise_parameter_names, num_rounds_by_distances, report
    )
    lambda_reciprocals = 1 / lambdas
    lambda_reciprocal_stddevs = np.vectorize(reciprocal_stddev)(lambdas, lambda_stddevs)

    value = float(lambda_reciprocals[0, 0])
    stddev = float(lambda_reciprocal_stddevs[0, 0])
    return ConfidenceInterval(low=value - stddev, best=value, high=value + stddev)


def inverse_lambda_interval_at(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
) -> ConfidenceInterval:
    """Compute 1 / Λ with an asymmetric confidence interval.

    This is the asymmetric counterpart to :func:`inverse_lambda_at`. It samples
    the same memory experiments but fits Λ with a binomial likelihood and
    propagates the asymmetric bounds into 1 / Λ. Because 1 / Λ decreases with Λ,
    the largest Λ gives the lower bound on 1 / Λ.

    Args:
        noise_model: a callable adding noise to the provided circuit, according to
            the parameters provided.
        noise_parameters: valid parameters to forward to ``noise_model``
            representing the point at which 1 / Λ should be computed.
        num_rounds_by_distances: a mapping from each code distance to the number
            of rounds used to estimate the logical error probability per round.
        sampling_parameters: additional parameters relating to the sampling tasks.
        memory_generator: a callable that generates a memory experiment.

    Returns:
        The estimate of 1 / Λ together with its lower and upper bounds.
    """
    point, noise_parameter_names, report = _run_lambda_engine(
        noise_model,
        noise_parameters,
        num_rounds_by_distances,
        sampling_parameters,
        memory_generator,
    )
    # Keep only the rows matching this single noise point before fitting. The
    # symmetric ``compute_lambda_and_stddev_from_results`` does the same filtering
    # internally inside its per-point loop; here we evaluate one point, so the
    # caller selects it and ``compute_lambda_interval_from_results`` takes the
    # already-filtered frame.
    filtered = _filter_non_close_noise_parameters(
        report, point[:, 0], noise_parameter_names
    )
    lambda_data = compute_lambda_interval_from_results(
        num_rounds_by_distances, filtered
    )
    assert lambda_data.lambda_interval is not None
    lambda_interval = lambda_data.lambda_interval
    return ConfidenceInterval(
        low=1 / lambda_interval.high,
        best=1 / lambda_data.lambda_,
        high=1 / lambda_interval.low,
    )
