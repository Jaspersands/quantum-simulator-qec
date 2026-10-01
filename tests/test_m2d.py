from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

SWEPT = """R 0 1
CX sweep[0] 0
X_ERROR(0.2) 0 1
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def test_matches_stims_m2d(d5):
    stim = pytest.importorskip("stim")
    sc = stim.Circuit(str(d5))
    meas = sc.compile_sampler(seed=4).sample(2000)
    want_d, want_o = sc.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
    conv = d5.compile_m2d_converter()
    d, o = conv.convert(measurements=meas, separate_observables=True)
    assert np.array_equal(d, want_d) and np.array_equal(o, want_o)
    packed = np.packbits(meas, axis=1, bitorder="little")
    pd = conv.convert(measurements=packed, bit_packed=True)
    assert np.array_equal(np.unpackbits(pd, axis=1, count=d5.num_detectors, bitorder="little").astype(bool), want_d)


def test_sweep_bits():
    stim = pytest.importorskip("stim")
    sc = stim.Circuit(SWEPT)
    meas = sc.compile_sampler(seed=1).sample(500)
    sweeps = np.random.default_rng(1).random((500, 1)) < 0.5
    want = sc.compile_m2d_converter().convert(measurements=meas, sweep_bits=sweeps, append_observables=True)
    got = sq.Circuit(SWEPT).compile_m2d_converter().convert(measurements=meas, sweep_bits=sweeps, append_observables=True)
    assert np.array_equal(got, want)


def test_shape_errors(d3):
    conv = d3.compile_m2d_converter()
    with pytest.raises(ValueError):
        conv.convert(measurements=np.zeros((4, d3.num_measurements + 1), dtype=bool))
    with pytest.raises(ValueError):
        conv.convert(measurements=np.zeros(d3.num_measurements, dtype=bool))
    with pytest.raises(ValueError, match="dense tableau"):
        sq.Circuit("M 20000").compile_m2d_converter()
