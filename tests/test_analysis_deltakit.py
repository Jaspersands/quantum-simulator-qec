"""stabilizer_qec.analysis's port of deltakit's analysis against deltakit itself (Λ, logical
error per round, binomial and quops fits, p_ij, error distributions), and its numpy stand-ins
for scipy.optimize and uncertainties against those packages."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import analysis as ours
from stabilizer_qec.analysis import pij

dk = pytest.importorskip("deltakit_explorer.analysis")
dkd = pytest.importorskip("deltakit_decode.utils")


def lambda_data(seed: int):
    rng = np.random.default_rng(seed)
    distances = np.array([3, 5, 7, 9][: int(rng.integers(2, 5))])
    lam, lam0 = float(rng.uniform(1.5, 6)), float(rng.uniform(5, 50))
    leppr = 1 / lam0 * lam ** (-(distances + 1) / 2) * np.exp(rng.normal(0, 0.05, len(distances)))
    return distances, leppr, leppr * rng.uniform(0.02, 0.1, len(distances))


def same(a, b, rel=1e-9):
    return a == pytest.approx(b, rel=rel, abs=1e-15)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("method", ["lin", "shifted", "curve"])
def test_lambda(seed, method):
    d, l, s = lambda_data(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = dk.calculate_lambda_and_lambda_stddev(d, l, s, method=dk._lambda.LambdaFitMethod(method))
        b = ours.calculate_lambda_and_lambda_stddev(d, l, s, method=ours._dk._lambda.LambdaFitMethod(method))
    for f in ("lambda_", "lambda_std", "lambda0", "lambda0_std"):
        assert same(getattr(b, f), getattr(a, f), 1e-7), f


@pytest.mark.parametrize("seed", range(10))
def test_asymmetric_lambda_and_quops(seed):
    d, l, s = lambda_data(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = dk.calculate_lambda_asymmetric(d, l, l - s, l + s)
        b = ours.calculate_lambda_asymmetric(d, l, l - s, l + s)
    for f in ("low", "best", "high"):
        assert same(getattr(b.lambda_interval, f), getattr(a.lambda_interval, f), 1e-6)
        assert same(getattr(b.lambda0_interval, f), getattr(a.lambda0_interval, f), 1e-6)
    qa, qb = dk.predict_quops_interval(a, 11), ours.predict_quops_interval(b, 11)
    assert same(qb.best, qa.best, 1e-6) and same(qb.low, qa.low, 1e-5) and same(qb.high, qa.high, 1e-5)
    if a.lambda_ > 1:
        assert ours.predict_distance_for_quops(b.lambda0, b.lambda_, 1e9) == dk.predict_distance_for_quops(a.lambda0, a.lambda_, 1e9)


@pytest.mark.parametrize("seed", range(10))
def test_logical_error_per_round(seed):
    rng = np.random.default_rng(seed)
    rounds = np.array([2, 4, 8, 16, 32])
    eps, spam = float(rng.uniform(0.002, 0.02)), float(rng.uniform(0.0, 0.02))
    shots = np.full(len(rounds), 20_000)
    fails = rng.binomial(shots, (1 - (1 - 2 * spam) * (1 - 2 * eps) ** rounds) / 2)
    lep, std = dk.calculate_lep_and_lep_stddev(fails, shots)
    assert np.array_equal(ours.calculate_lep_and_lep_stddev(fails, shots)[0], lep)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = dk.compute_logical_error_per_round(rounds, lep, std)
        b = ours.compute_logical_error_per_round(rounds, lep, std)
        for f in ("leppr", "leppr_stddev", "spam_error", "spam_error_stddev"):
            assert same(getattr(b, f), getattr(a, f), 1e-7), f
        a = dk.fit_logical_error_per_round_asymmetric(rounds, fails, shots)
        b = ours.fit_logical_error_per_round_asymmetric(rounds, fails, shots)
    for f in ("leppr", "leppr_low", "leppr_high", "spam_error"):
        assert same(getattr(b, f), getattr(a, f), 1e-6), f
    for x, y in zip(dk.calculate_lep_asymmetric(fails, shots), ours.calculate_lep_asymmetric(fails, shots)):
        assert np.allclose(x, y, rtol=1e-12)


def test_binomial_fits():
    for shots, hits in ((1000, 0), (1000, 7), (10**6, 123_456), (50, 50)):
        a, b = dk.fit_binomial(shots, hits), ours.fit_binomial(shots, hits)
        assert (b.low, b.best, b.high) == pytest.approx((a.low, a.best, a.high), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("task", ["repetition_code:memory", "surface_code:rotated_memory_z"])
def test_pij_values_and_models(task):
    stim = pytest.importorskip("stim")
    from deltakit_core.decoding_graphs import dem_to_decoding_graph_and_logicals

    c = stim.Circuit.generated(task, distance=3, rounds=3, after_clifford_depolarization=0.02)
    dets = c.compile_detector_sampler(seed=4).sample(20_000)
    samples = [np.flatnonzero(row).tolist() for row in dets]
    exp_a = dkd.generate_expectation_data(samples)
    exp_b = pij.generate_expectation_data(dets)
    assert exp_a == exp_b
    assert dkd.calculate_pij_values(exp_a) == pytest.approx(pij.calculate_pij_values(exp_b), nan_ok=True)
    dem = c.detector_error_model(decompose_errors=True)
    graph, _ = dem_to_decoding_graph_and_logicals(dem)
    a = dkd.calculate_pij_values(exp_a, graph)
    b = pij.calculate_pij_values(exp_b, sq.DetectorErrorModel(str(dem)))
    assert set(a) == set(b)
    assert all(math.isclose(a[k], b[k], rel_tol=1e-12, abs_tol=1e-15) for k in a)
    assert dkd.dem_and_pij_edges_max_diff(dem, a) == pytest.approx(pij.dem_and_pij_edges_max_diff(sq.DetectorErrorModel(str(dem)), b), rel=1e-9)


def test_error_distribution():
    from deltakit_decode.analysis import EmpiricalDecodingErrorDistribution as Theirs

    rng = np.random.default_rng(2)
    corr, target = rng.integers(0, 2, (500, 3)).astype(np.uint8), rng.integers(0, 2, (500, 3)).astype(np.uint8)
    a, b = Theirs(3), ours.decoding.EmpiricalDecodingErrorDistribution(3)
    a.batch_record_errors(corr, target)
    b.batch_record_errors(corr, target)
    assert a.to_dict() == b.to_dict() and (a.shots, a.fails) == (b.shots, b.fails)
    assert np.array_equal(a.fails_per_logical, b.fails_per_logical)


_WITHOUT = r"""
import json, sys, warnings
for name in ("scipy", "uncertainties", "lmfit"):
    sys.modules[name] = None  # importing them now fails, as when not installed
import numpy as np
from stabilizer_qec import analysis as a
from stabilizer_qec.analysis import _scipy
assert not hasattr(_scipy.optimize, "linprog"), "scipy was used"
warnings.simplefilter("ignore")
d, l, s = (np.array(x) for x in json.loads(sys.argv[1]))
out = {}
for m in ("lin", "shifted", "curve"):
    r = a.calculate_lambda_and_lambda_stddev(d, l, s, method=a._dk._lambda.LambdaFitMethod(m))
    out[m] = [r.lambda_, r.lambda_std, r.lambda0, r.lambda0_std]
r = a.calculate_lambda_asymmetric(d, l, l - s, l + s)
out["asym"] = [r.lambda_interval.low, r.lambda_interval.high, r.lambda0_interval.low, r.lambda0_interval.high]
rounds = np.array([2, 4, 8, 16, 32]); shots = np.full(5, 20000); fails = np.array([150, 290, 560, 1050, 1900])
lep, std = a.calculate_lep_and_lep_stddev(fails, shots)
r = a.compute_logical_error_per_round(rounds, lep, std)
out["leppr"] = [r.leppr, r.leppr_stddev, r.spam_error, r.spam_error_stddev]
r = a.fit_logical_error_per_round_asymmetric(rounds, fails, shots)
out["leppr_asym"] = [r.leppr, r.leppr_low, r.leppr_high, r.spam_error]
print(json.dumps(out))
"""


@pytest.mark.parametrize("seed", range(3))
def test_without_scipy_or_uncertainties(seed):
    import json
    import subprocess
    import sys

    d, l, s = lambda_data(seed)
    run = subprocess.run([sys.executable, "-c", _WITHOUT, json.dumps([d.tolist(), l.tolist(), s.tolist()])], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    mine = json.loads(run.stdout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in ("lin", "shifted", "curve"):
            r = dk.calculate_lambda_and_lambda_stddev(d, l, s, method=dk._lambda.LambdaFitMethod(m))
            assert mine[m] == pytest.approx([r.lambda_, r.lambda_std, r.lambda0, r.lambda0_std], rel=1e-5), m
        r = dk.calculate_lambda_asymmetric(d, l, l - s, l + s)
        assert mine["asym"] == pytest.approx([r.lambda_interval.low, r.lambda_interval.high, r.lambda0_interval.low, r.lambda0_interval.high], rel=1e-4)
        rounds, shots, fails = np.array([2, 4, 8, 16, 32]), np.full(5, 20000), np.array([150, 290, 560, 1050, 1900])
        lep, std = dk.calculate_lep_and_lep_stddev(fails, shots)
        r = dk.compute_logical_error_per_round(rounds, lep, std)
        assert mine["leppr"] == pytest.approx([r.leppr, r.leppr_stddev, r.spam_error, r.spam_error_stddev], rel=1e-6)
        r = dk.fit_logical_error_per_round_asymmetric(rounds, fails, shots)
        assert mine["leppr_asym"] == pytest.approx([r.leppr, r.leppr_low, r.leppr_high, r.spam_error], rel=1e-4)
