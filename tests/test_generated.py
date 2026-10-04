"""``Circuit.generated`` writes Stim's generated memory experiments character for character:
every task, at every distance and round count tried, with and without each kind of noise, and
refuses what Stim refuses."""

from __future__ import annotations

import itertools

import pytest

import stabilizer_qec as sq

TASKS = [
    "repetition_code:memory",
    "surface_code:rotated_memory_x",
    "surface_code:rotated_memory_z",
    "surface_code:unrotated_memory_x",
    "surface_code:unrotated_memory_z",
    "color_code:memory_xyz",
]

NOISES = [
    {},
    dict(after_clifford_depolarization=0.001, before_round_data_depolarization=0.002, before_measure_flip_probability=0.003, after_reset_flip_probability=0.004),
    dict(after_clifford_depolarization=1e-5),
    dict(before_round_data_depolarization=0.125, after_reset_flip_probability=0.3),
    dict(before_measure_flip_probability=2.5e-4, after_reset_flip_probability=1e-7),
]


@pytest.mark.parametrize("task", TASKS)
def test_generated_circuits_are_stims(task):
    stim = pytest.importorskip("stim")
    compared = 0
    for d, rounds, noise in itertools.product([2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 7], NOISES):
        try:
            want = str(stim.Circuit.generated(task, distance=d, rounds=rounds, **noise))
        except ValueError:
            with pytest.raises(ValueError):
                sq.Circuit.generated(task, distance=d, rounds=rounds, **noise)
            continue
        # This package's text ends in a newline, Stim's does not; the rest is the same.
        got = str(sq.Circuit.generated(task, distance=d, rounds=rounds, **noise))
        assert got == want + "\n", f"{task} d={d} rounds={rounds} {noise}"
        compared += 1
    assert compared >= 60


def test_a_large_surface_code_is_stims():
    stim = pytest.importorskip("stim")
    kw = dict(distance=25, rounds=50, after_clifford_depolarization=0.001, before_measure_flip_probability=0.001)
    assert str(sq.Circuit.generated("surface_code:rotated_memory_x", **kw)) == str(stim.Circuit.generated("surface_code:rotated_memory_x", **kw)) + "\n"


def test_text_keeps_what_six_digits_would_lose():
    # Stim writes 1/3 as 0.333333; this package writes it exactly, so its text round-trips.
    c = sq.Circuit.generated("repetition_code:memory", distance=3, rounds=2, after_clifford_depolarization=1 / 3)
    assert "DEPOLARIZE2(0.3333333333333333)" in str(c)
    assert sq.Circuit(str(c)) == c


def test_generated_circuits_work_here():
    c = sq.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=5, after_clifford_depolarization=0.001)
    assert (c.num_qubits, c.num_detectors, c.num_observables) == (26, 40, 1)
    dem = c.detector_error_model(decompose_errors=True)
    assert dem.num_errors > 0
    dets, obs = c.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    predicted = sq.Matching(dem).decode_batch(dets)
    assert (predicted != obs).mean() < 0.05


@pytest.mark.parametrize(
    "task, kw, message",
    [
        ("surface_code:hexagonal", {}, "unknown code task"),
        ("color_code:memory_xyz", dict(rounds=1), "rounds >= 2"),
        ("color_code:memory_xyz", dict(distance=4), "odd distance"),
        ("repetition_code:memory", dict(distance=1), "distance >= 2"),
        ("repetition_code:memory", dict(rounds=0), "rounds"),
        ("repetition_code:memory", dict(after_clifford_depolarization=1.5), "outside"),
    ],
)
def test_bad_arguments_are_refused(task, kw, message):
    args = dict(distance=3, rounds=3)
    args.update(kw)
    with pytest.raises(ValueError, match=message):
        sq.Circuit.generated(task, **args)
    with pytest.raises(TypeError):
        sq.Circuit.generated(3, distance=3, rounds=3)
