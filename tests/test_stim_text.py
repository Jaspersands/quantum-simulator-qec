"""Error models print as Stim prints them, character for character: the same fault classes
(a fault's pieces in the order they arose), the same order, the same probabilities to the last
digit (accumulated and rounded as Stim's build for this machine accumulates and rounds them),
the same declarations, and loops folded the same way."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")

from test_stim_gates import random_case, tagged  # noqa: E402


def generated(code, d, rounds):
    return stim.Circuit.generated(
        code, distance=d, rounds=rounds, after_clifford_depolarization=0.001, before_round_data_depolarization=0.002,
        before_measure_flip_probability=0.003, after_reset_flip_probability=0.001,
    )


def assert_same_text(text, **kwargs):
    try:
        want = str(stim.Circuit(text).detector_error_model(**kwargs))
    except ValueError:
        return
    got = sq.Circuit(text).detector_error_model(**kwargs)
    assert str(got).strip() == want.strip()
    assert got.num_errors == stim.DetectorErrorModel(want).num_errors


CODES = ["surface_code:rotated_memory_z", "surface_code:rotated_memory_x", "surface_code:unrotated_memory_z", "repetition_code:memory", "color_code:memory_xyz"]


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("decompose", [False, True])
def test_generated_memories(code, decompose):
    for d, rounds in [(3, 2), (3, 7), (5, 20)]:
        assert_same_text(str(generated(code, d, rounds)), decompose_errors=decompose)


@pytest.mark.parametrize("seed", range(40))
def test_random_circuits(seed):
    rng = np.random.default_rng(seed)
    texts = [
        random_case(6000 + seed),
        random_case(7000 + seed, disjoint=True),
        random_case(8000 + seed, paulis=True),
        tagged(rng, random_case(9000 + seed)),
    ]
    case = random_case(10_000 + seed, disjoint=seed % 2 == 1)
    texts.append(f"REPEAT {5 + seed % 4} {{\n{case}\n}}\nREPEAT 3 {{\n{case}\n}}")
    for text in texts:
        for decompose in (False, True):
            assert_same_text(text, decompose_errors=decompose, approximate_disjoint_errors=True)


def test_numbers_print_with_stims_sixteen_digits():
    text = "R 0\nX_ERROR(0.1) 0\nX_ERROR(0.2) 0\nX_ERROR(0.0000123) 0\nM 0\nDETECTOR(1.5, 0.0001, 1e-05) rec[-1]"
    assert_same_text(text)
    assert "e-05" in str(sq.Circuit(text).detector_error_model())


@pytest.mark.parametrize("d", [5, 7])
def test_ignoring_decomposition_failures_is_stims(d):
    c = stim.Circuit.generated("color_code:memory_xyz", distance=d, rounds=4, after_clifford_depolarization=0.001, before_measure_flip_probability=0.002)
    want = str(c.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True))
    got = str(sq.Circuit(str(c)).detector_error_model(decompose_errors=True, ignore_decomposition_failures=True))
    assert got == want
    # Without it, a model Stim cannot decompose is refused here too.
    try:
        c.detector_error_model(decompose_errors=True)
    except ValueError:
        with pytest.raises(ValueError):
            sq.Circuit(str(c)).detector_error_model(decompose_errors=True)
