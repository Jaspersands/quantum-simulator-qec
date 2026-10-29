"""``explain_detector_error_model_errors``: where each fault of a circuit's error model comes
from, as Stim explains it, character for character: Stim's generated memories under each kind
of noise, filtered and unfiltered, one representative or all, and random circuits using every
gate and noise channel."""

from __future__ import annotations

import itertools

import pytest

import stabilizer_qec as sq

# Stim 1.16 orders a fault's locations as these do; Stim 1.15 (the last for Python 3.9) does not.
stim = pytest.importorskip("stim", minversion="1.16")

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


def parts(explained):
    """Every piece of an explanation as ``str()`` gives it: terms, locations, their instructions,
    and the targets of each."""
    out = []
    for e in explained:
        out.append([str(t) for t in e.dem_error_terms])
        for loc in e.circuit_error_locations:
            out.append(str(loc))
            out.append(str(loc.instruction_targets))
            out.append([str(t) for t in loc.flipped_pauli_product])
            out.append([str(t) for t in loc.instruction_targets.targets_in_range])
    return out


def test_pieces_print_as_stims():
    import numpy as np
    from test_stim_gates import random_case

    cases = [str(c) for c in circuits()] + SMALL + ["QUBIT_COORDS(0.5, -2) 0\nQUBIT_COORDS 1\nX_ERROR(0.125) 0 1\nM 0 1\nDETECTOR(1e-5, 3) rec[-1]\nDETECTOR rec[-2]"]
    cases += [random_case(seed, disjoint=False) for seed in range(60)]
    for text in cases:
        try:
            want = parts(stim.Circuit(text).explain_detector_error_model_errors())
        except ValueError:
            continue
        assert parts(sq.Circuit(text).explain_detector_error_model_errors()) == want, text
    del np


GAUGE = [
    # Detectors that depend on a random outcome, which Stim's matcher allows: the collapse is a
    # fault of the measurement that makes it.
    "MX 0\nM 0\nDETECTOR rec[-1]",
    "MPP X0*Z1\nMXX 0 1\nDETECTOR rec[-1]",
    "MRX 0\nMR 0\nDETECTOR rec[-1]\n",
    "MPP X0\nM 0\nDETECTOR rec[-1]\nX_ERROR(0.25) 0\nM 0\nDETECTOR rec[-1]",
    "MX 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]",
    "MX 0 1\nCX 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
    "MY 0\nM 0\nDETECTOR rec[-1]",
    "MRY 0\nMR 0\nDETECTOR rec[-1]",
    "MYY 0 1\nM 0\nDETECTOR rec[-1]",
    # A measurement's own flip and its collapse: the site reads the lesser.
    "MX(0.1) 0\nM 0\nDETECTOR rec[-1]",
    "MX(0.1) 0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]",
    # A reset's collapse waits for the next fault site the backward walk meets, which reads it
    # when it is the lesser (a noise case, a measurement), or loses it (a lesser fault of its own).
    # (Where the collapse and that site carry different tags, Stim prints the collapse's as the
    # noise tag and the site's on its instruction; the Python objects keep one tag for both, so
    # those cases are pinned on the Rust text, in tests/rust_api.rs.)
    "RX 0\nM 0\nDETECTOR rec[-1]",
    "X_ERROR(0.1) 1\nRX 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
    "X_ERROR(0.1) 1\nRX 0\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
    "X_ERROR(0.1) 0\nRX 0\nM 0\nDETECTOR rec[-1]",
    "M 1\nRX 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
    "X_ERROR(0.1) 1\nRY 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
    "DEPOLARIZE1(0.1) 1 2\nRX 0\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
    "DEPOLARIZE2(0.1) 1 2\nRX 0\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
    "REPEAT 2 {\n RX 0\n X_ERROR(0.1) 1\n M 0 1\n DETECTOR rec[-1]\n DETECTOR rec[-2]\n}",
    # A heralded target empties what waits.
    "HERALDED_ERASE(0.1) 1\nRX 0\nX_ERROR(0.1) 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nDETECTOR rec[-3]",
    "X_ERROR(0.1) 1\nHERALDED_ERASE(0.1) 2\nRX 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nDETECTOR rec[-3]",
    # Stim's matcher makes a site of any E (even of probability 0) and none of an MPAD, whose
    # flip waits like a collapse.
    "E(0) X0\nM 0\nDETECTOR rec[-1]",
    "MPAD(0.1) 0\nDETECTOR rec[-1]",
    "X_ERROR(0.1) 1\nMPAD(0.1) 0\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
]


@pytest.mark.parametrize("text", GAUGE)
def test_gauge_detectors_are_explained_as_stim_explains_them(text):
    c = stim.Circuit(text)
    for reduce in (False, True):
        want = texts(c.explain_detector_error_model_errors(reduce_to_one_representative_error=reduce))
        got = texts(sq.Circuit(text).explain_detector_error_model_errors(reduce_to_one_representative_error=reduce))
        assert got == want


def test_a_nondeterministic_observable_is_refused_as_stim_refuses_it():
    text = "MX 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    with pytest.raises(ValueError, match="non-deterministic observables"):
        stim.Circuit(text).explain_detector_error_model_errors()
    with pytest.raises(ValueError, match="non-deterministic observables"):
        sq.Circuit(text).explain_detector_error_model_errors()


def random_collapsing_case(rng):
    """A random circuit of collapses in every basis, resets, Clifford gates, noise of every kind
    and detectors on random records: most of its detectors depend on random outcomes."""
    n = rng.randint(2, 4)
    lines, measured = [], 0
    for _ in range(rng.randint(4, 14)):
        q = rng.randrange(n)
        a, b = rng.sample(range(n), 2)
        kind = rng.random()
        if kind < 0.3:
            lines.append(rng.choice(["M", "MX", "MY", "MR", "MRX", "MRY"]) + rng.choice(["", "(0.125)"]) + f" {q}")
            measured += 1
        elif kind < 0.4:
            lines.append(rng.choice([f"MPP X{a}*Z{b}", f"MXX {a} {b}", f"MZZ {a} {b}", f"MYY {a} {b}"]))
            measured += 1
        elif kind < 0.5:
            lines.append(rng.choice(["R", "RX", "RY"]) + f" {q}")
        elif kind < 0.65:
            lines.append(rng.choice([f"H {q}", f"S {q}", f"CX {a} {b}", f"CZ {a} {b}", f"SQRT_X {q}"]))
        elif kind < 0.85:
            lines.append(
                rng.choice(
                    [f"X_ERROR(0.1) {q}", f"Z_ERROR(0.2) {q} {a}", f"DEPOLARIZE1(0.05) {q}", f"DEPOLARIZE2(0.05) {a} {b}", f"E(0.1) X{a} Z{b}",
                     f"PAULI_CHANNEL_1(0.1, 0, 0.05) {q}", f"HERALDED_ERASE(0.1) {q}", "MPAD(0.1) 0"]
                )
            )  # fmt: skip
            measured += lines[-1].startswith(("HERALDED", "MPAD"))
        elif measured:
            lines.append("DETECTOR " + " ".join(f"rec[-{k}]" for k in rng.sample(range(1, measured + 1), rng.randint(1, min(2, measured)))))
    lines.append("M " + " ".join(map(str, range(n))))
    measured += n
    for _ in range(3):
        lines.append("DETECTOR " + " ".join(f"rec[-{k}]" for k in rng.sample(range(1, measured + 1), rng.randint(1, min(2, measured)))))
    return "\n".join(lines)


def test_random_gauge_circuits_are_explained_as_stim_explains_them():
    import random

    rng = random.Random(7)
    compared = gauged = 0
    for _ in range(600):
        text = random_collapsing_case(rng)
        c = stim.Circuit(text)
        try:
            want = [texts(c.explain_detector_error_model_errors(reduce_to_one_representative_error=r)) for r in (False, True)]
        except ValueError:
            continue
        got = [texts(sq.Circuit(text).explain_detector_error_model_errors(reduce_to_one_representative_error=r)) for r in (False, True)]
        assert got == want, text
        compared += 1
        try:
            c.detector_error_model()
        except ValueError:
            gauged += 1
    assert compared >= 500 and gauged >= 200, (compared, gauged)


@pytest.mark.parametrize(
    "text",
    [
        "MX(0.1) 0\nZ_ERROR(0.1) 0\nMX 0\nDETECTOR rec[-1] rec[-2]",
        "RX 0\nZ_ERROR(0.1) 0\nMX(0.1) 0\nDETECTOR rec[-1]",
        "R 0 1\nMPP(0.1) Z0*Z1\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1] rec[-2]",
        "R 0 1\nX_ERROR(0.1) 0\nMPP(0.1) Z0*Z1\nDETECTOR rec[-1]",
        "R 0\nX_ERROR[a](0.1) 0\nX_ERROR[b](0.1) 0\nM(0.1) 0\nDETECTOR rec[-1]",
    ],
)
def test_the_representative_is_the_one_stims_matcher_keeps(text):
    # Stim keeps the first location it meets (walking back) unless a later one is simpler, and
    # its test of simpler never prefers across measured observables of different sizes.
    want = texts(stim.Circuit(text).explain_detector_error_model_errors(reduce_to_one_representative_error=True))
    assert texts(sq.Circuit(text).explain_detector_error_model_errors(reduce_to_one_representative_error=True)) == want
