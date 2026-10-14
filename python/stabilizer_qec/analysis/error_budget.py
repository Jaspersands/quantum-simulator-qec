"""Error budgets: how much each noise parameter contributes to 1/Λ, by deltakit's method
(deltakit-explorer 0.9.4's ``error_budget``, Copyright Riverlane, Apache License 2.0; see
THIRD_PARTY_NOTICES.md), which follows Google's (Nature 595 (2021), supplement VIII.C).

1/Λ is estimated at points along each noise parameter (the others held at half their values),
a polynomial is fitted to those estimates, and its derivative at half the parameter, times the
parameter, is that parameter's contribution. Each estimate of Λ samples memory experiments of
the given distances and round counts, decodes them by matching, fits the logical error per
round of each distance, then fits Λ across distances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from stabilizer_qec.analysis._dk._estimate import Estimate
from stabilizer_qec.analysis._dk._lambda import calculate_lambda_and_lambda_stddev
from stabilizer_qec.analysis._dk._leppr import calculate_lep_and_lep_stddev, compute_logical_error_per_round
from stabilizer_qec.analysis._uncertain import correlated_values

__all__ = [
    "ErrorBudgetResult",
    "FittingParameters",
    "SamplingParameters",
    "get_error_budget",
    "get_rotated_surface_code_memory_circuit",
    "inverse_lambda_at",
    "inverse_lambda_gradient_at",
    "standard_noise_model",
]


@dataclass(frozen=True)
class FittingParameters:
    """How the gradient is estimated: ``num_points_per_parameters`` values of each parameter,
    spaced ``"logarithmic"`` or ``"linear"`` across its bounds, fitted by a polynomial of
    ``fitting_degree``."""

    num_points_per_parameters: int = 10
    discretisation_strategy: str = "logarithmic"
    fitting_degree: int = 3

    def __post_init__(self) -> None:
        if self.num_points_per_parameters + 1 < self.fitting_degree + 2:
            raise ValueError(
                f"Estimation of the standard deviation requires at least fitting_degree + 2 = {self.fitting_degree + 2} discretisation points, "
                f"but only {self.num_points_per_parameters} + 1 are provided."
            )
        if self.discretisation_strategy not in ("logarithmic", "linear"):
            raise ValueError("discretisation_strategy must be 'logarithmic' or 'linear'")

    def get_discretisation(self, a: float, b: float, c: float) -> np.ndarray:
        if self.discretisation_strategy == "linear":
            return np.linspace(a, b, self.num_points_per_parameters)
        return np.geomspace(a, b, self.num_points_per_parameters)


@dataclass(frozen=True)
class SamplingParameters:
    """When each sampling task stops: after ``max_shots``, or once its relative standard error
    is at most ``lep_target_rse`` with at least ``lep_computation_min_fails`` failures."""

    max_shots: int = 10_000_000
    batch_size: int = 10_000
    lep_target_rse: float = 1e-4
    lep_computation_min_fails: int = 10
    max_workers: int = 1


@dataclass
class ErrorBudgetResult:
    """Each noise parameter's contribution to 1/Λ, and its standard deviation."""

    contributions: Tuple[float, ...]
    contribution_stddevs: Tuple[float, ...]

    @property
    def lambda_estimate(self) -> float:
        """1/Λ according to the budget (the sum of the contributions)."""
        return float(np.sum(self.contributions))

    @property
    def lambda_stddev_estimate(self) -> float:
        return float(np.sqrt(np.sum(np.asarray(self.contribution_stddevs) ** 2)))

    @staticmethod
    def from_gradient(gradient: Any, gradient_stddev: Any, noise_parameters: Any) -> "ErrorBudgetResult":
        contributions = np.abs(np.asarray(gradient) * np.asarray(noise_parameters))
        stddevs = np.abs(np.asarray(gradient_stddev) * np.asarray(noise_parameters))
        return ErrorBudgetResult(tuple(map(float, contributions.ravel())), tuple(map(float, stddevs.ravel())))


def get_rotated_surface_code_memory_circuit(distance: int, num_rounds: int) -> Any:
    """A noiseless rotated surface-code memory experiment (Z basis)."""
    from stabilizer_qec import Circuit

    return Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=num_rounds)


def standard_noise_model(circuit: Any, noise_parameters: Sequence[float]) -> Any:
    """Circuit-level noise on a noiseless circuit, from ``[p_1q, p_2q, p_reset, p_meas]`` (or
    one ``p`` for all four): one-qubit depolarizing after each one-qubit Clifford gate,
    two-qubit after each two-qubit gate, a flip after each reset and before each measurement,
    in the basis of the reset or measurement."""
    from stabilizer_qec import Circuit, CircuitRepeatBlock, gate_data

    params = list(np.ravel(noise_parameters))
    p1, p2, pr, pm = params * 4 if len(params) == 1 else params[:4]
    flip = {"X": "Z_ERROR", "Y": "X_ERROR", "Z": "X_ERROR"}

    def basis(name: str) -> str:
        return {"RX": "X", "MX": "X", "MRX": "X", "RY": "Y", "MY": "Y", "MRY": "Y"}.get(name, "Z")

    def walk(c: Any) -> Any:
        out = Circuit()
        for item in c:
            if isinstance(item, CircuitRepeatBlock):
                out.append(CircuitRepeatBlock(item.repeat_count, walk(item.body_copy())))
                continue
            name = item.name
            targets = [t.value for t in item.targets_copy() if t.is_qubit_target]
            data = gate_data(name)
            if data.produces_measurements and pm > 0 and targets and not data.is_noisy_gate and name in ("M", "MX", "MY", "MR", "MRX", "MRY"):
                out.append(flip[basis(name)], targets, pm)
            out.append(item)
            if not targets or data.is_noisy_gate:
                continue
            if data.is_reset and pr > 0:
                out.append(flip[basis(name)], targets, pr)
            elif data.is_unitary and data.is_single_qubit_gate and p1 > 0 and name != "I":
                out.append("DEPOLARIZE1", targets, p1)
            elif data.is_unitary and data.is_two_qubit_gate and p2 > 0:
                out.append("DEPOLARIZE2", targets, p2)
        return out

    return walk(Circuit(circuit))


def _sample_lep(circuit: Any, sampling: SamplingParameters, seed: Optional[int]) -> Tuple[int, int]:
    from stabilizer_qec import Matching
    from stabilizer_qec.analysis.decoding import run_decoding_on_circuit

    decoder = Matching(circuit.detector_error_model(decompose_errors=True))
    result = run_decoding_on_circuit(
        circuit, sampling.max_shots, decoder, max_batch_size=sampling.batch_size, target_rse=sampling.lep_target_rse, min_fails=sampling.lep_computation_min_fails, seed=seed
    )
    return result["fails"], result["shots"]


def inverse_lambda_at(
    noise_model: Callable[[Any, Any], Any],
    noise_parameters: Any,
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: Callable[[int, int], Any] = get_rotated_surface_code_memory_circuit,
    *,
    seed: Optional[int] = None,
) -> Estimate:
    """1/Λ, and its standard deviation, at one point of the noise parameters."""
    distances, lepprs, leppr_stds = [], [], []
    for k, (distance, rounds) in enumerate(sorted(num_rounds_by_distances.items())):
        leps, stds = [], []
        for j, r in enumerate(rounds):
            circuit = noise_model(memory_generator(distance, r), np.asarray(noise_parameters))
            fails, shots = _sample_lep(circuit, sampling_parameters, None if seed is None else seed + 1000 * k + j)
            lep, std = calculate_lep_and_lep_stddev(max(fails, 1), shots)
            leps.append(lep)
            stds.append(std)
        fit = compute_logical_error_per_round(np.asarray(rounds), np.asarray(leps), np.asarray(stds), force_include_single_round=True)
        distances.append(distance)
        lepprs.append(fit.leppr)
        leppr_stds.append(fit.leppr_stddev)
    lam = calculate_lambda_and_lambda_stddev(np.asarray(distances), np.asarray(lepprs), np.asarray(leppr_stds))
    return Estimate(1 / lam.lambda_, lam.lambda_std / lam.lambda_**2)


def inverse_lambda_gradient_at(
    noise_model: Callable[[Any, Any], Any],
    noise_parameters: Any,
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    noise_parameters_exploration_bounds: Sequence[Tuple[float, float]],
    fitting_parameters: FittingParameters = FittingParameters(),
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: Callable[[int, int], Any] = get_rotated_surface_code_memory_circuit,
    *,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """The gradient of 1/Λ with respect to each noise parameter at ``noise_parameters``, and
    each entry's standard deviation."""
    point = np.asarray(noise_parameters, dtype=float).ravel()
    base = inverse_lambda_at(noise_model, point, num_rounds_by_distances, sampling_parameters, memory_generator, seed=seed)
    gradient: List[float] = []
    stddev: List[float] = []
    for i, (lo, hi) in enumerate(noise_parameters_exploration_bounds):
        xs, ys, ss = [point[i]], [base.value], [base.stddev]
        for j, x in enumerate(fitting_parameters.get_discretisation(lo, hi, point[i])):
            p = point.copy()
            p[i] = x
            est = inverse_lambda_at(noise_model, p, num_rounds_by_distances, sampling_parameters, memory_generator, seed=None if seed is None else seed + 7919 * (i + 1) + 104729 * (j + 1))
            xs.append(x)
            ys.append(est.value)
            ss.append(est.stddev)
        coefficients, cov = np.polyfit(np.asarray(xs), np.asarray(ys), deg=fitting_parameters.fitting_degree, cov="unscaled", w=1 / np.asarray(ss))
        coefficients, cov = np.flip(coefficients), np.flip(cov)
        uncertain = correlated_values(coefficients, cov)
        derivative = sum((power + 1) * coeff * point[i] ** power for power, coeff in enumerate(uncertain[1:]))
        gradient.append(float(derivative.nominal_value))
        stddev.append(float(derivative.std_dev))
    return np.asarray(gradient), np.asarray(stddev)


def get_error_budget(
    noise_model: Callable[[Any, Any], Any],
    noise_parameters: Any,
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    noise_parameters_exploration_bounds: Sequence[Tuple[float, float]],
    fitting_parameters: FittingParameters = FittingParameters(),
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: Callable[[int, int], Any] = get_rotated_surface_code_memory_circuit,
    *,
    seed: Optional[int] = None,
) -> ErrorBudgetResult:
    """Each noise parameter's contribution to 1/Λ: the gradient of 1/Λ at half the noise
    parameters, times the parameters. ``noise_model(circuit, parameters)`` adds noise to a
    noiseless circuit (``standard_noise_model`` is one); ``memory_generator(distance,
    rounds)`` makes the noiseless memory experiments."""
    point = np.asarray(noise_parameters, dtype=float) / 2
    gradient, gradient_stddev = inverse_lambda_gradient_at(
        noise_model, point, num_rounds_by_distances, noise_parameters_exploration_bounds, fitting_parameters, sampling_parameters, memory_generator, seed=seed
    )
    return ErrorBudgetResult.from_gradient(gradient, gradient_stddev, np.asarray(noise_parameters))


def plot_error_budget(result: ErrorBudgetResult, names: Sequence[str], ax: Any = None) -> Any:
    """The contributions as a bar chart with error bars. Returns the axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.bar(range(len(names)), result.contributions, yerr=result.contribution_stddevs, tick_label=list(names))
    ax.set_ylabel("contribution to 1/Λ")
    return ax

