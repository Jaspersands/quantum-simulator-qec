# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from stabilizer_qec.deltakit.explorer.analysis.error_budget._post_processing import (
    _compute_logical_error_rate_per_round_from_results,
    _filter_non_close_noise_parameters,
    compute_lambda_interval_from_results,
)


def test_compute_lambda_interval_from_results() -> None:
    lambda_true, lambda0_true, spam = 3.0, 2.0, 0.005
    distances = [3, 5, 7]
    rounds = [2, 6, 10, 14]
    shots = 1_000_000
    rows = []
    for distance in distances:
        eps = lambda_true ** (-(distance + 1) / 2) / lambda0_true
        for num_rounds in rounds:
            fidelity = (1 - 2 * spam) * (1 - 2 * eps) ** num_rounds
            fails = round((1 - fidelity) / 2 * shots)
            rows.append(
                {
                    "distance": distance,
                    "num_rounds": num_rounds,
                    "fails": fails,
                    "shots": shots,
                }
            )
    data = pd.DataFrame(rows)
    result = compute_lambda_interval_from_results(
        dict.fromkeys(distances, rounds), data
    )
    assert result.has_asymmetric_bounds
    assert result.lambda_interval is not None
    assert result.lambda_interval.low <= result.lambda_ <= result.lambda_interval.high
    assert result.lambda_ == pytest.approx(lambda_true, rel=0.1)


@dataclass
class ErrorBudgetingResults:
    data: pd.DataFrame
    noise_parameters: npt.NDArray[np.floating]
    parameter_names: list[str]

    def to_tuple(self) -> tuple[pd.DataFrame, npt.NDArray[np.floating], list[str]]:
        return self.data, self.noise_parameters, self.parameter_names


@pytest.fixture
def error_parameters() -> npt.NDArray[np.floating]:
    return np.array([5e-4, 4e-3, 5e-3, 5e-3, 5e-3])


@pytest.fixture
def error_names() -> list[str]:
    return ["1q error", "2q error", "Reset error", "Measurement error", "Readout flip"]


@pytest.fixture
def simulated_error_budgeting_data(
    error_parameters: npt.NDArray[np.floating], error_names: list[str]
) -> ErrorBudgetingResults:
    _resources_folder = Path(__file__).parent.parent.parent / "resources"
    _data_file = _resources_folder / "error_budget_data.csv"
    data = pd.read_csv(_data_file)
    return ErrorBudgetingResults(data, error_parameters, error_names)


def test_data_matches_csv(
    simulated_error_budgeting_data: ErrorBudgetingResults,
) -> None:
    """Test that the hard-coded parts in the data fixture agree with the CSV file."""
    header_names: frozenset[str] = frozenset(
        simulated_error_budgeting_data.data.columns.values
    )
    for name in simulated_error_budgeting_data.parameter_names:
        assert f"noise_{name}" in header_names

    for noise_name, noise_value in zip(
        simulated_error_budgeting_data.parameter_names,
        simulated_error_budgeting_data.noise_parameters,
        strict=True,
    ):
        values = simulated_error_budgeting_data.data[f"noise_{noise_name}"]
        assert np.any(np.isclose(values, noise_value))


def test_filter_non_close_noise_parameters(
    simulated_error_budgeting_data: ErrorBudgetingResults,
) -> None:
    data, parameters, names = simulated_error_budgeting_data.to_tuple()
    all_noises_index = [f"noise_{name}" for name in names]
    data_frame = _filter_non_close_noise_parameters(data, parameters, names)
    filtered_columns_data_frame = data_frame[all_noises_index]
    for row in filtered_columns_data_frame.to_numpy():
        np.testing.assert_allclose(row, parameters)


def test_filter_non_close_noise_parameters_random(
    simulated_error_budgeting_data: ErrorBudgetingResults,
    random_generator: np.random.Generator,
) -> None:
    data, _, names = simulated_error_budgeting_data.to_tuple()
    all_noises_index = [f"noise_{name}" for name in names]
    random_row_index = random_generator.integers(data.shape[0])
    random_parameters = data[all_noises_index].to_numpy()[random_row_index, :]
    data_frame = _filter_non_close_noise_parameters(data, random_parameters, names)
    filtered_columns_data_frame = data_frame[all_noises_index]
    for row in filtered_columns_data_frame.to_numpy():
        np.testing.assert_allclose(row, random_parameters)


def test_duplicate_rows_aggregate_via_sum() -> None:
    """Regression test: when several rows share the same ``(distance,
    num_rounds)``, the post-processing must sum their fails/shots rather than
    take only the first row. Without this, sweep designs that intentionally
    replicate (e.g., c-optimal) silently lose every batch beyond the first.
    """
    # Two batches per (distance, num_rounds), drawn from a synthetic model
    # with LEPpR ~ 0.05, so LEP(r) = (1 - (1 - 2*LEPpR)**r) / 2. The buggy
    # ``[0]``-indexed path would use only the first batch; the correct
    # ``.sum()`` path produces a result equivalent to a single batch with the
    # pooled counts.
    data = pd.DataFrame(
        [
            {"distance": 3, "num_rounds": 3, "fails": 135, "shots": 1000},
            {"distance": 3, "num_rounds": 3, "fails": 140, "shots": 1000},
            {"distance": 3, "num_rounds": 5, "fails": 205, "shots": 1000},
            {"distance": 3, "num_rounds": 5, "fails": 210, "shots": 1000},
            {"distance": 3, "num_rounds": 7, "fails": 261, "shots": 1000},
            {"distance": 3, "num_rounds": 7, "fails": 265, "shots": 1000},
        ]
    )
    result = _compute_logical_error_rate_per_round_from_results([3, 5, 7], data)

    pooled = pd.DataFrame(
        [
            {"distance": 3, "num_rounds": 3, "fails": 275, "shots": 2000},
            {"distance": 3, "num_rounds": 5, "fails": 415, "shots": 2000},
            {"distance": 3, "num_rounds": 7, "fails": 526, "shots": 2000},
        ]
    )
    expected = _compute_logical_error_rate_per_round_from_results([3, 5, 7], pooled)

    assert pytest.approx(expected.leppr) == result.leppr
    assert pytest.approx(expected.leppr_stddev) == result.leppr_stddev
