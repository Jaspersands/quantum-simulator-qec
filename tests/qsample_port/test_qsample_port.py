"""stabilizer_qec.qsample where it departs from qsample 0.0.2 on purpose (this package's tests):
each protocol owns its check functions, a save that fails leaves the file it would replace as it
was, and subset probabilities stay finite for any number of fault locations."""

from __future__ import annotations

import os
import pickle
import sys
from fractions import Fraction
import math

import numpy as np
import pytest

from stabilizer_qec import qsample as qs
from stabilizer_qec.qsample import math as qmath
from stabilizer_qec.qsample import utils


def ghz_protocol(check_functions=None):
    ghz = qs.Circuit([{"init": {0, 1, 2, 3, 4}}, {"H": {0}}, {"CNOT": {(0, 1)}}, {"CNOT": {(1, 2)}}, {"CNOT": {(2, 3)}}, {"CNOT": {(3, 4)}}, {"CNOT": {(0, 4)}}, {"measure": {4}}])
    meas = qs.Circuit([{"measure": {0, 1, 2, 3}}], noisy=False)
    protocol = qs.Protocol(check_functions={"logErr": lambda m: m not in [0, 0b1111]} if check_functions is None else check_functions)
    protocol.add_nodes_from(["ghz", "meas"], circuits=[ghz, meas])
    protocol.add_edge("START", "ghz", check="True")
    protocol.add_edge("ghz", "meas", check="ghz[-1]==0")
    protocol.add_edge("ghz", "FAIL", check="ghz[-1]==1")
    protocol.add_edge("meas", "FAIL", check="logErr(meas[-1])")
    return protocol


# Check functions

def test_protocols_built_without_check_functions_own_theirs():
    # qsample's default is one dict shared by them all: a check function one adds was the other's too.
    a, b = qs.Protocol(), qs.Protocol(fault_tolerant=True)
    assert a.check_functions == {} and a.check_functions is not b.check_functions
    a.check_functions["odd"] = lambda m: m % 2 == 1
    assert "odd" not in b.check_functions
    assert "odd" not in qs.Protocol().check_functions


def test_protocol_keeps_the_check_functions_it_is_given():
    functions = {"logErr": lambda m: m != 0}
    protocol = qs.Protocol(check_functions=functions)
    assert protocol.check_functions is functions  # as qsample: the caller's dict, not a copy


# Saving

def _direct_sampler(**kwargs):
    np.random.seed(5)
    sampler = qs.DirectSampler(protocol=ghz_protocol(**kwargs), simulator=qs.StabilizerSimulator, err_model=qs.E1, err_params={"q": [0.05]})
    sampler.run(200)
    return sampler


def _log_err(m):
    return m not in [0, 0b1111]


class _Unpicklable:
    def __reduce__(self):
        raise RuntimeError("refuses to be pickled")


def _leftovers(directory, name):
    return sorted(f for f in os.listdir(directory) if f != name)


def test_failed_save_leaves_the_file_it_would_replace(tmp_path):
    path = tmp_path / "checkpoint.pkl"
    utils.save({"shots": list(range(1000))}, path)
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="refuses"):
        utils.save({"shots": list(range(2000)), "bad": _Unpicklable()}, path)
    assert path.read_bytes() == before  # qsample opens it 'wb' first: 0 bytes
    assert utils.load(path) == {"shots": list(range(1000))}
    assert _leftovers(tmp_path, "checkpoint.pkl") == []


def test_failed_save_creates_no_file(tmp_path):
    with pytest.raises(RuntimeError, match="refuses"):
        utils.save(_Unpicklable(), str(tmp_path / "new.pkl"))
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("which", ["direct", "subset"])
def test_samplers_with_lambda_check_functions_save_with_dill(tmp_path, which):
    pytest.importorskip("dill")
    if which == "direct":
        sampler = _direct_sampler()
    else:
        np.random.seed(5)
        sampler = qs.SubsetSampler(protocol=ghz_protocol(), simulator=qs.StabilizerSimulator, p_max={"q": 0.1}, err_model=qs.E1, err_params={"q": [1e-3, 1e-2]}, L=2)
        sampler.run(100)
    path = tmp_path / f"{which}.pkl"
    sampler.save(str(path))
    loaded = utils.load(str(path))
    for a, b in zip(loaded.stats(), sampler.stats()):  # the subset sampler's sums may take another order
        assert np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-12, atol=0)
    check = loaded.protocol.check_functions["logErr"]
    assert (check(0), check(0b1111), check(0b0100)) == (False, False, True)
    assert _leftovers(tmp_path, f"{which}.pkl") == []


def test_without_dill_a_lambda_save_fails_and_keeps_the_old_file(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "dill", None)  # import dill raises ImportError
    path = tmp_path / "direct.pkl"
    good = _direct_sampler(check_functions={"logErr": _log_err})  # pickle takes a top-level function
    good.save(str(path))
    before = path.read_bytes()
    with pytest.raises((pickle.PicklingError, AttributeError)):
        _direct_sampler().save(str(path))
    assert path.read_bytes() == before
    assert np.asarray(utils.load(str(path)).stats()[0]).tolist() == np.asarray(good.stats()[0]).tolist()
    assert _leftovers(tmp_path, "direct.pkl") == []


# Subset probabilities

def _exact_binom(k, n, p):
    p = Fraction(p)
    return float(math.comb(n, k) * p**k * (1 - p) ** (n - k))


def test_binomial_coefficients_beyond_a_float_are_inf():
    assert qmath.comb(1029, 514) == float(math.comb(1029, 514))
    assert qmath.comb(1030, 515) == math.inf  # was OverflowError
    assert np.array_equal(qmath.comb([1030, 4, 3], [515, 2, 5]), [math.inf, 6.0, 0.0])


@pytest.mark.parametrize("n", [1029, 1030, 1100, 20000])
def test_binomial_probabilities_stay_finite_for_large_n(n):
    p = 0.01
    k = np.arange(n + 1)
    values = qmath.binom(k, n, p)
    assert values.shape == (n + 1,) and np.all(np.isfinite(values)) and np.all(values >= 0)
    assert abs(values.sum() - 1) < 1e-9
    for j in (0, 1, n // 100, n // 50, n // 4, n // 2):
        want = _exact_binom(j, n, p)
        assert values[j] == pytest.approx(want, rel=1e-12, abs=0)


def test_binomial_probabilities_below_the_normal_floats():
    # The coefficient is a float, but a power falls below the normal floats: the product lost
    # digits (1e-5 of binom(160, 1000, 0.01)) or was 0 (binom(200, 1029, 0.02) is 3.3e-129).
    for k, n, p in [(160, 1000, 0.01), (200, 1029, 0.02), (300, 1029, 0.05), (150, 1029, 0.01), (1030, 1030, 0.5), (514, 1029, 0.5)]:
        assert float(qmath.binom(k, n, p)) == pytest.approx(_exact_binom(k, n, p), rel=1e-12)
    assert float(qmath.binom(3, 1100, 0.0)) == 0.0 and float(qmath.binom(0, 1100, 0.0)) == 1.0
    assert float(qmath.binom(1100, 1100, 1.0)) == 1.0 and float(qmath.binom(5, 1100, 1.0)) == 0.0


def test_binomial_probabilities_broadcast_as_before():
    assert np.allclose(qmath.binom([1, 2], [3, 4], 0.1), [0.243, 0.0486])
    assert isinstance(qmath.binom(1, 3, 0.1), np.float64)
    joint = qmath.joint_binom([515, 2], [1030, 4], [[0.01, 0.1], [0.5, 0.2]])
    assert joint.shape == (2,)
    assert joint[0] == pytest.approx(_exact_binom(515, 1030, 0.01) * _exact_binom(2, 4, 0.1), rel=1e-12, abs=0)
    assert joint[1] == pytest.approx(_exact_binom(515, 1030, 0.5) * _exact_binom(2, 4, 0.2), rel=1e-12)


@pytest.mark.parametrize("ncnot", [1030, 1100])
def test_subset_sampler_takes_a_group_of_over_1029_fault_locations(ncnot):
    # qsample 0.0.2 gives NaN probabilities from 171 locations; the port raised OverflowError from 1030.
    ticks = [{"init": {0, 1}}] + [{"CNOT": {(0, 1)}} for _ in range(ncnot)] + [{"measure": {0, 1}}]
    protocol = qs.Protocol()
    protocol.add_nodes_from(["c"], circuits=[qs.Circuit(ticks)])
    protocol.add_edge("START", "c", check="True")
    protocol.add_edge("c", "FAIL", check="c[-1] != 0")
    sampler = qs.SubsetSampler(protocol=protocol, simulator=qs.StabilizerSimulator, p_max={"q": 0.01}, err_model=qs.E1, err_params={"q": [1e-4, 1e-3]}, L=1)
    probs = qmath.subset_probs(protocol.get_circuit("c"), qs.E1(), 0.01)
    assert len(probs) == ncnot + 1 and all(np.isfinite(v) for v in probs.values())
    assert abs(sum(probs.values()) - 1) < 1e-9
    np.random.seed(0)
    sampler.run(50)
    p_L, std, p_up, std_up = (np.asarray(x, dtype=float) for x in sampler.stats())
    assert np.all(np.isfinite(p_L)) and np.all(np.isfinite(p_up)) and np.all(p_L <= p_up)
