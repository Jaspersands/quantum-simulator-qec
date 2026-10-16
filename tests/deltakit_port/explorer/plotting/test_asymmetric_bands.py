# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pytest

from stabilizer_qec.deltakit.explorer.analysis import (
    calculate_lambda_and_lambda_stddev,
    calculate_lambda_asymmetric,
    calculate_lep_and_lep_stddev,
    compute_logical_error_per_round,
    fit_logical_error_per_round_asymmetric,
    predict_quops_at_distance,
    predict_quops_interval,
)
from stabilizer_qec.deltakit.explorer.plotting import (
    interpolate_lambda,
    interpolate_leppr,
    plot_lambda,
    plot_leppr,
)


@pytest.fixture
def leppr_data():
    # Generate counts from a known model so that the largest-round error rate is
    # close to 0.4 and the fit is clean. This keeps compute_logical_error_per_round
    # from warning (which would be turned into an error by the test configuration).
    eps_true, spam_true, shots_per_point = 0.05, 0.01, 200_000
    rounds = np.array([2, 6, 10, 14])
    fidelity = (1 - 2 * spam_true) * (1 - 2 * eps_true) ** rounds
    fails = np.round((1 - fidelity) / 2 * shots_per_point).astype(int)
    shots = np.full(len(rounds), shots_per_point)
    lep, lep_stddev = calculate_lep_and_lep_stddev(fails, shots)
    return (
        compute_logical_error_per_round(rounds, lep, lep_stddev),
        fails,
        shots,
        rounds,
    )


@pytest.fixture
def lambda_data():
    return calculate_lambda_and_lambda_stddev(
        distances=[5, 7, 9],
        leppr=[0.15, 0.1, 0.05],
        leppr_std=[0.01, 0.008, 0.005],
    )


class TestLepprBand:
    def test_band_is_ordered_and_clipped(self, leppr_data):
        result = interpolate_leppr(leppr_data[0])
        assert np.all(result.lower_boundary <= result.interpolated)
        assert np.all(result.interpolated <= result.upper_boundary)
        assert np.all(result.lower_boundary >= 0)
        assert np.all(result.upper_boundary <= 1)

    def test_band_is_asymmetric(self, leppr_data):
        result = interpolate_leppr(leppr_data[0])
        lower_margin = result.interpolated - result.lower_boundary
        upper_margin = result.upper_boundary - result.interpolated
        # The two margins should not be identical everywhere.
        assert not np.allclose(lower_margin, upper_margin)


class TestLambdaBand:
    def test_band_is_ordered_and_positive(self, lambda_data):
        result = interpolate_lambda(lambda_data)
        assert np.all(result.lower_boundary <= result.interpolated)
        assert np.all(result.interpolated <= result.upper_boundary)
        assert np.all(result.lower_boundary >= 0)

    def test_shapes_match(self, lambda_data):
        result = interpolate_lambda(lambda_data, num_points=50)
        assert result.interpolated.shape == result.distances.shape == (50,)


class TestPlotsRun:
    def test_leppr_plot_runs_with_symmetric_fit(self, leppr_data):
        data, *_ = leppr_data
        fig, _ = plot_leppr(interpolate_leppr(data))
        assert fig is not None

    def test_leppr_plot_runs_with_asymmetric_fit(self):
        rounds, fails, shots = _model_counts(0.02, 0.005, [2, 6, 10, 14], 500_000)
        data = fit_logical_error_per_round_asymmetric(rounds, fails, shots)
        fig, _ = plot_leppr(interpolate_leppr(data))
        assert fig is not None

    def test_lambda_plot_runs(self, lambda_data):
        fig, _ = plot_lambda(interpolate_lambda(lambda_data))
        assert fig is not None


def _model_counts(eps, spam, rounds, shots):
    rounds_arr = np.asarray(rounds)
    fidelity = (1 - 2 * spam) * (1 - 2 * eps) ** rounds_arr
    fails = np.round((1 - fidelity) / 2 * shots).astype(int)
    return rounds_arr, fails, np.full(len(rounds), shots)


class TestAsymmetricFitPipeline:
    def test_leppr_fit_populates_bounds(self):
        rounds, fails, shots = _model_counts(0.02, 0.005, [2, 6, 10, 14], 500_000)
        data = fit_logical_error_per_round_asymmetric(rounds, fails, shots)
        assert data.leppr_low is not None
        assert data.leppr_low <= data.leppr <= data.leppr_high
        assert data.leppr_low >= 0

    def test_leppr_band_uses_fit_bounds(self):
        rounds, fails, shots = _model_counts(1e-6, 1e-7, [2, 6, 10, 14], 1_000_000)
        data = fit_logical_error_per_round_asymmetric(rounds, fails, shots)
        result = interpolate_leppr(data)
        assert np.all(result.lower_boundary <= result.interpolated)
        assert np.all(result.interpolated <= result.upper_boundary)

    def test_lambda_fit_recovers_value(self):
        lambda_true, lambda0_true = 3.0, 2.0
        distances = [3, 5, 7, 9]
        rounds = np.array([2, 6, 10, 14])
        leppr, low, high = [], [], []
        for d in distances:
            eps_d = lambda_true ** (-(d + 1) / 2) / lambda0_true
            r, fails, shots = _model_counts(eps_d, 0.005, rounds, 1_000_000)
            data = fit_logical_error_per_round_asymmetric(r, fails, shots)
            leppr.append(data.leppr)
            low.append(data.leppr_low)
            high.append(data.leppr_high)
        result = calculate_lambda_asymmetric(distances, leppr, low, high)
        assert result.lambda_ == pytest.approx(lambda_true, rel=0.05)
        assert result.lambda_interval is not None
        assert (
            result.lambda_interval.low <= result.lambda_ <= result.lambda_interval.high
        )
        assert result.lambda_interval.low > 0

    def test_lambda_asymmetric_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            calculate_lambda_asymmetric([3, 5], [0.1], [0.09], [0.11])

    def test_quops_interval_from_asymmetric_lambda(self):
        distances = [3, 5, 7, 9]
        leppr = [3.0 ** (-(d + 1) / 2) / 2.0 for d in distances]
        low = [e * 0.9 for e in leppr]
        high = [e * 1.1 for e in leppr]
        result = calculate_lambda_asymmetric(distances, leppr, low, high)
        quops = predict_quops_interval(result, 11)
        assert quops.low <= quops.best <= quops.high
        assert quops.best == pytest.approx(
            predict_quops_at_distance(result.lambda0, result.lambda_, 11)
        )

    def test_quops_interval_requires_asymmetric_bounds(self):
        symmetric = calculate_lambda_and_lambda_stddev(
            [3, 5, 7], [1e-2, 1e-3, 1e-4], [1e-3, 1e-4, 1e-5]
        )
        with pytest.raises(ValueError):
            predict_quops_interval(symmetric, 11)
