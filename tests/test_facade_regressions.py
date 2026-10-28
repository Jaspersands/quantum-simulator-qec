"""The Python facade's Circuit, DetectorErrorModel, PauliString, Tableau and Matching where they
once departed from Stim and PyMatching: copies keep their class (and Deltakit's circuit its
leakage), the reference signs read tagged Clifford-angle gates as the reference run does, a
model's tags may hold '#', indexing parses once, model files with a stray non-UTF-8 byte in a
comment load, the WDIMACS writer takes a certain error, the exact distance skips faults that
never happen, and heavy networkx / rustworkx edges are left out with PyMatching's warning."""

from __future__ import annotations

import copy
import io
import pickle
import warnings

import numpy as np
import pytest

import stabilizer_qec as sq
import stabilizer_qec.deltakit_stim as dk
from stabilizer_qec import _circuit, _dem


# Subclasses at module level, so pickle finds them by name (as a user's would be).
class MyCircuit(sq.Circuit):
    pass


class MyModel(sq.DetectorErrorModel):
    pass


class MyPauliString(sq.PauliString):
    pass


class MyTableau(sq.Tableau):
    pass


class FlippedMatching(sq.Matching):
    def decode(self, z, **kwargs):
        return 1 - super().decode(z, **kwargs)


_ROUND_TRIPS = [("pickle", lambda x: pickle.loads(pickle.dumps(x))), ("copy", copy.copy), ("deepcopy", copy.deepcopy)]


@pytest.mark.parametrize("how,again", _ROUND_TRIPS, ids=[h for h, _ in _ROUND_TRIPS])
def test_subclasses_copy_and_pickle_as_themselves(how, again):
    # As Stim's: each comes back as its own class, equal to what it was.
    for obj in (MyCircuit("H 0\nM 0"), MyModel("error(0.1) D0"), MyPauliString("+XYZ"), MyPauliString("-i_ZZ"), MyTableau(2), MyTableau.random(3)):
        y = again(obj)
        assert type(y) is type(obj) and y == obj, (how, obj)
    # As PyMatching's copy: the subclass, its overrides and its attributes kept.
    m = FlippedMatching(np.array([[1, 1, 0], [0, 1, 1]]))
    m.note = "kept"
    y = again(m)
    assert type(y) is FlippedMatching and y.note == "kept"
    assert y.decode([1, 0]).tolist() == m.decode([1, 0]).tolist() == [0, 1, 1]
    # The base classes are as they were.
    assert type(again(sq.Matching(np.array([[1, 1]])))) is sq.Matching
    assert type(again(sq.Tableau(1))) is sq.Tableau


@pytest.mark.parametrize("how,again", _ROUND_TRIPS, ids=[h for h, _ in _ROUND_TRIPS])
def test_deltakit_circuit_copies_keep_their_leakage(how, again):
    c = dk.Circuit("R 0 1\nLEAKAGE(0.2) 0\nCX 0 1\nHERALD_LEAKAGE_EVENT 0\nM 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]")
    y = again(c)
    assert type(y) is dk.Circuit and y == c
    assert str(y) == str(c) and str(y.detector_error_model()) == str(c.detector_error_model()) == "error(0.5) D0 D1"
    rates = y.compile_detector_sampler(seed=1).sample(20000).mean(0)
    assert np.array_equal(rates, c.compile_detector_sampler(seed=1).sample(20000).mean(0))
    assert rates[1] > 0.15  # the herald fires on about 20% of shots


def test_deltakit_parallel_engine_samples_the_leakage():
    # Deltakit's engine sends its decoder managers, circuits and all, to a process pool.
    pytest.importorskip("pathos")
    import pandas  # noqa: F401  (the engine's results)

    from stabilizer_qec.deltakit.decode import PyMatchingDecoder
    from stabilizer_qec.deltakit.decode.analysis import RunAllAnalysisEngine, StimDecoderManager
    from stabilizer_qec.deltakit.decode.utils import parse_stim_circuit

    c = dk.Circuit.generated("repetition_code:memory", distance=5, rounds=5, after_clifford_depolarization=0.002, after_reset_leakage=0.03, after_clifford_leakage_and_relaxation=0.01)
    fails = {}
    for processes in (1, 2):
        g, logicals, sc = parse_stim_circuit(c)
        manager = StimDecoderManager(sc, PyMatchingDecoder(g, logicals), seed=1, batch_size=2000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = RunAllAnalysisEngine("x", [manager], num_parallel_processes=processes, batch_size=2000, max_shots=4000).run()
        fails[processes] = int(df["fails"][0])
    # About 3% of shots fail with the leakage (Deltakit's own run: 582 and 598 of 20,000);
    # without it, none would.
    assert fails[1] > 50 and fails[2] > 50, fails
