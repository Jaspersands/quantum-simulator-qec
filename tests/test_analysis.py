"""stabilizer_qec.analysis without a reference to compare with: its circuit noise against
Stim's generated noise, error budgets, decoding runs and plots, and its numpy stand-ins for
scipy.optimize and uncertainties against those packages."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import analysis as ours
from stabilizer_qec.analysis import _scipy, _uncertain, pij


def lambda_data(seed: int):
    rng = np.random.default_rng(seed)
    distances = np.array([3, 5, 7, 9][: int(rng.integers(2, 5))])
    lam, lam0 = float(rng.uniform(1.5, 6)), float(rng.uniform(5, 50))
    leppr = 1 / lam0 * lam ** (-(distances + 1) / 2) * np.exp(rng.normal(0, 0.05, len(distances)))
    return distances, leppr, leppr * rng.uniform(0.02, 0.1, len(distances))


def test_run_decoding_on_circuit():
    c = sq.Circuit.generated("repetition_code:memory", distance=5, rounds=5, after_clifford_depolarization=0.05)
    r = ours.decoding.run_decoding_on_circuit(c, 20_000, sq.Matching(c.detector_error_model(decompose_errors=True)), max_batch_size=5000, seed=1)
    assert r["shots"] == 20_000 and 0 < r["fails"] == r["fails_log_0"] < 20_000


def test_brentq_and_bounded_minimisation_are_scipys():
    optimize = pytest.importorskip("scipy.optimize")

    for k in range(10):
        f = lambda x, k=k: math.exp(x) - 2 - k  # noqa: E731
        assert _scipy.brentq(f, -5, 5) == pytest.approx(optimize.brentq(f, -5, 5), abs=1e-11)
        g = lambda x, k=k: (x - 0.3 * k) ** 2 + math.sin(3 * x)  # noqa: E731
        assert _scipy.minimize_scalar(g, bounds=(-2, 4), method="bounded").x == pytest.approx(optimize.minimize_scalar(g, bounds=(-2, 4), method="bounded").x, abs=1e-6)
        assert _scipy.minimize_scalar(g).fun == pytest.approx(optimize.minimize_scalar(g).fun, abs=1e-9)
        h = lambda v, k=k: (v[0] - k) ** 2 + 3 * (v[1] + 0.5 * k) ** 2 + v[0] * v[1]  # noqa: E731
        assert np.allclose(_scipy.minimize(h, [0.5, 0.5], method="Nelder-Mead").x, optimize.minimize(h, [0.5, 0.5], method="Nelder-Mead").x, atol=1e-6)


def test_curve_fit_is_scipys():
    optimize = pytest.importorskip("scipy.optimize")

    rng = np.random.default_rng(0)
    x = np.linspace(1, 4, 9)
    y = 3 * 1.7 ** (-x) * np.exp(rng.normal(0, 0.03, len(x)))
    s = 0.03 * y
    f = lambda x, a, b: 1 / a * b ** (-x)  # noqa: E731
    pa, ca = optimize.curve_fit(f, x, y, p0=(1, 1), sigma=s, absolute_sigma=True, bounds=(0, np.inf))
    pb, cb = _scipy.curve_fit(f, x, y, p0=(1, 1), sigma=s, absolute_sigma=True, bounds=(0, np.inf))
    assert np.allclose(pa, pb, rtol=1e-6) and np.allclose(ca, cb, rtol=1e-3)


def test_uncertainty_propagation_is_uncertainties():
    unc = pytest.importorskip("uncertainties")
    from uncertainties import umath

    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    a1, b1 = unc.correlated_values([1.5, -0.3], cov)
    a2, b2 = _uncertain._correlated_values([1.5, -0.3], cov)
    for e1, e2 in (
        (umath.exp(-2 * a1), _uncertain._exp(-2 * a2)),
        (umath.exp(-b1 - umath.log(a1) / 2), _uncertain._exp(-b2 - _uncertain._log(a2) / 2)),
        ((1 - (1 - 2 * a1 / 10) ** (1 / 3)) / 2, (1 - (1 - 2 * a2 / 10) ** (1 / 3)) / 2),
        (a1 * b1 + b1 / a1, a2 * b2 + b2 / a2),
    ):
        assert e2.nominal_value == pytest.approx(e1.nominal_value, rel=1e-12)
        assert e2.std_dev == pytest.approx(e1.std_dev, rel=1e-9)



@pytest.mark.parametrize("task", ["surface_code:rotated_memory_z", "repetition_code:memory", "color_code:memory_xyz", "surface_code:unrotated_memory_x"])
def test_standard_noise_model_is_stims_generated_noise(task):
    from stabilizer_qec.analysis import error_budget as eb

    clean = sq.Circuit.generated(task, distance=3, rounds=3)
    noisy = sq.Circuit.generated(task, distance=3, rounds=3, after_clifford_depolarization=0.01, after_reset_flip_probability=0.02, before_measure_flip_probability=0.03)
    assert eb.standard_noise_model(clean, [0.01, 0.01, 0.02, 0.03]).detector_error_model() == noisy.detector_error_model()


def test_error_budget_adds_up_to_about_one_over_lambda():
    from stabilizer_qec.analysis import error_budget as eb

    rounds = {3: [2, 4, 6], 5: [2, 4, 6]}
    sampling = eb.SamplingParameters(max_shots=20_000, lep_target_rse=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        budget = eb.get_error_budget(eb.standard_noise_model, [0.02] * 4, rounds, [(0.006, 0.03)] * 4, eb.FittingParameters(num_points_per_parameters=5, fitting_degree=2), sampling, seed=1)
        direct = eb.inverse_lambda_at(eb.standard_noise_model, [0.02] * 4, rounds, sampling, seed=1).value
    # Google's linearisation at half the noise: the sum approximates 1/Λ at the full noise.
    assert len(budget.contributions) == 4 and max(budget.contributions) == budget.contributions[1]
    assert abs(budget.lambda_estimate - direct) < 0.35 * direct


def test_plots(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from stabilizer_qec.analysis import plotting

    d, l, s = lambda_data(0)
    plotting.plot_lambda(ours.calculate_lambda_and_lambda_stddev(d, l, s))
    rounds, shots, fails = np.array([2, 4, 8, 16, 32]), np.full(5, 20000), np.array([150, 290, 560, 1050, 1900])
    lep, std = ours.calculate_lep_and_lep_stddev(fails, shots)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plotting.plot_leppr(rounds, lep, std, ours.compute_logical_error_per_round(rounds, lep, std))
    c = sq.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=4, after_clifford_depolarization=0.01)
    model = c.detector_error_model()
    rates = plotting.defect_rates(c.compile_detector_sampler(seed=1).sample(2000), model)
    assert len(rates) == 8 and all(len(v) >= 3 for v in rates.values())
    plotting.plot_defect_rates(rates)
    plotting.plot_detection_probability_on_patch(model, rates)
    # Every check needs the same number of rounds for the matrix (as in deltakit).
    rep = sq.Circuit.generated("repetition_code:memory", distance=5, rounds=4, after_clifford_depolarization=0.01)
    rep_model = rep.detector_error_model(decompose_errors=True)
    matrix, mapping = pij.create_correlation_matrix(pij.calculate_pij_values(pij.generate_expectation_data(rep.compile_detector_sampler(seed=2).sample(2000)), rep_model), rep_model)
    assert matrix.shape == (20, 20) and len(mapping) == 4
    pij.plot_correlation_matrix(matrix, mapping)
    plt.savefig(tmp_path / "plots.png")
    plt.close("all")
