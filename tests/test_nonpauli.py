"""Tagged non-Pauli instructions (`I_ERROR[R_Z(theta=0.1)] 0`): Stim-readable, inert in every
Clifford engine, and an error when a known tag is malformed."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

TAGS = [
    "I_ERROR[R_Z(theta=0.1)] 0 1",
    "I_ERROR[R_X(theta=-0.05)] 2",
    "I_ERROR[R_Y(theta=0.2)] 3",
    "II_ERROR[R_ZZ(theta=0.03)] 0 1",
    "II_ERROR[R_XX(theta=0.03)] 2 3",
    "I_ERROR[R_PAULI(theta=0.01, pauli=XZY)] 0 1 2",
    "I[T] 0",
    "I[T_DAG] 1",
    "I[U3(theta=0.1, phi=0.2, lambda=0.3)] 2",
    "I_ERROR[LEAK(p=0.01)] 3",
    "I_ERROR[SEEP(p=0.1)] 3",
    "II_ERROR[LEAK_TRANSPORT(p=0.1)] 2 3",
    "I_ERROR[AMPLITUDE_DAMPING(gamma=0.02)] 1",
    "I_ERROR[someone-else's tag] 0",
]


def with_and_without(base: sq.Circuit):
    """The circuit with every tag line inserted after each TICK, and the same circuit bare."""
    text = str(base)
    tagged = text.replace("TICK", "TICK\n" + "\n".join(TAGS))
    return sq.Circuit(tagged), base


@pytest.fixture(scope="module")
def pair():
    base = sq.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    return with_and_without(base)


def test_tags_round_trip(pair):
    tagged, _ = pair
    text = str(tagged)
    for line in TAGS:
        assert line in text
    assert str(sq.Circuit(text)) == text


def test_clifford_engines_ignore_the_tags(pair):
    tagged, bare = pair
    assert str(tagged.detector_error_model(decompose_errors=True)) == str(bare.detector_error_model(decompose_errors=True))
    a = tagged.compile_detector_sampler(seed=5).sample(2000, separate_observables=True)
    b = bare.compile_detector_sampler(seed=5).sample(2000, separate_observables=True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    m = bare.compile_sampler(seed=3).sample(500)
    assert np.array_equal(tagged.compile_sampler(seed=3).sample(500), m)
    assert np.array_equal(
        tagged.compile_m2d_converter().convert(measurements=m, separate_observables=True)[0],
        bare.compile_m2d_converter().convert(measurements=m, separate_observables=True)[0],
    )
    assert tagged.num_detectors == bare.num_detectors and tagged.num_measurements == bare.num_measurements


def test_stim_reads_the_tags(pair):
    stim = pytest.importorskip("stim")
    tagged, bare = pair
    ours = stim.Circuit(str(tagged))
    assert str(ours.detector_error_model(decompose_errors=True)) == str(stim.Circuit(str(bare)).detector_error_model(decompose_errors=True))


@pytest.mark.parametrize(
    "line, message",
    [
        ("I_ERROR[R_Z(angle=0.1)] 0", "missing argument 'theta'"),
        ("I_ERROR[R_Z(theta=0.1, phi=1)] 0", "unknown argument 'phi'"),
        ("I_ERROR[R_Z(theta=abc)] 0", "must be a number"),
        ("II_ERROR[R_Z(theta=0.1)] 0 1", "single qubits"),
        ("I_ERROR[R_ZZ(theta=0.1)] 0 1", "pairs"),
        ("I_ERROR[LEAK(p=1.5)] 0", "probability"),
        ("I_ERROR[R_PAULI(theta=0.1, pauli=XX)] 0", "letters"),
    ],
)
def test_malformed_known_tags_are_errors(line, message):
    with pytest.raises(ValueError, match=message):
        sq.Circuit("H 0\n" + line)
