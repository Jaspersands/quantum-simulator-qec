"""stabilizer_qec.qsample where it departs from qsample 0.0.2 on purpose (this package's tests):
each protocol owns its check functions, and a save that fails leaves the file it would replace as
it was."""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pytest

from stabilizer_qec import qsample as qs
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
