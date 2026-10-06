"""``explain_detector_error_model_errors``: where each fault of a circuit's error model comes
from, as Stim explains it, character for character: Stim's generated memories under each kind
of noise, filtered and unfiltered, one representative or all, and random circuits using every
gate and noise channel."""

from __future__ import annotations

import itertools

import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")

TASKS = ["repetition_code:memory", "surface_code:rotated_memory_x", "surface_code:unrotated_memory_z", "color_code:memory_xyz"]
NOISES = [
    dict(after_clifford_depolarization=0.01),
    dict(before_measure_flip_probability=0.01, after_reset_flip_probability=0.02, before_round_data_depolarization=0.03),
]


def circuits():
    for task, noise in itertools.product(TASKS, NOISES):
        yield stim.Circuit.generated(task, distance=3, rounds=3, **noise)


def texts(explained):
    return [str(e).rstrip("\n") for e in explained]


@pytest.mark.parametrize("reduce", [False, True])
def test_explanations_are_stims(reduce):
    for c in circuits():
        want = texts(c.explain_detector_error_model_errors(reduce_to_one_representative_error=reduce))
        got = texts(sq.Circuit(str(c)).explain_detector_error_model_errors(reduce_to_one_representative_error=reduce))
        assert got == want, str(c)[:200]


def test_filter_and_fields():
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.001)
    f = "error(1) D0\nerror(1) D0 D1"
    want = c.explain_detector_error_model_errors(dem_filter=stim.DetectorErrorModel(f))
    got = sq.Circuit(str(c)).explain_detector_error_model_errors(dem_filter=sq.DetectorErrorModel(f))
    assert texts(got) == texts(want)
    loc, ref = got[0].circuit_error_locations[0], want[0].circuit_error_locations[0]
    assert loc.tick_offset == ref.tick_offset and loc.noise_tag == ref.noise_tag
    frames = lambda l: [(s.instruction_offset, s.iteration_index, s.instruction_repetitions_arg) for s in l.stack_frames]  # noqa: E731
    assert frames(loc) == frames(ref)
    span = lambda l: (l.instruction_targets.gate, l.instruction_targets.target_range_start, l.instruction_targets.target_range_end)  # noqa: E731
    assert span(loc) == span(ref)


SMALL = [
    "R 0 1\nX_ERROR(0.1) 0\nM(0.2) 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    "QUBIT_COORDS(1, 2) 0\nH 0\nTICK\nZ_ERROR[mytag](0.1) 0\nH 0\nM 0\nDETECTOR(4, 5) rec[-1]",
    "R 0 1 2\nE(0.1) X0 X1 Z2\nELSE_CORRELATED_ERROR(0.2) X2\nM 0 1 2\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nDETECTOR rec[-3]",
    "R 0 1\nPAULI_CHANNEL_1(0.1, 0.2, 0.05) 0\nPAULI_CHANNEL_2(0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02) 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
    "R 0 1\nREPEAT 3 {\n  X_ERROR(0.1) 0\n  MR 0\n  DETECTOR rec[-1]\n}\nMPP(0.05) X0*X1 Z0*Z1\nDETECTOR rec[-1]",
    "RX 0\nHERALDED_ERASE(0.1) 0\nMX 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
]


@pytest.mark.parametrize("text", SMALL)
def test_small_circuits(text):
    c = stim.Circuit(text)
    for reduce in (False, True):
        want = texts(c.explain_detector_error_model_errors(reduce_to_one_representative_error=reduce))
        got = texts(sq.Circuit(text).explain_detector_error_model_errors(reduce_to_one_representative_error=reduce))
        assert got == want


def test_random_circuits_using_every_gate():
    import numpy as np
    from test_stim_gates import random_case, tagged

    checked = 0
    for seed in range(300):
        # Plain, with disjoint channels, and tagged, in turn; every location and the
        # representative Stim picks (by its order of locations, tags reduced separately).
        text = random_case(seed, disjoint=seed % 3 == 1)
        if seed % 3 == 2:
            text = tagged(np.random.default_rng(seed), text)
        for reduce in (False, True):
            try:
                want = texts(stim.Circuit(text).explain_detector_error_model_errors(reduce_to_one_representative_error=reduce))
            except ValueError:
                continue
            assert texts(sq.Circuit(text).explain_detector_error_model_errors(reduce_to_one_representative_error=reduce)) == want, text
            checked += 1
    assert checked >= 300
