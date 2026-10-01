from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq
from conftest import assert_same_model


@pytest.mark.parametrize("code", ["rotated", "xzzx"])
@pytest.mark.parametrize("basis", ["z", "x"])
def test_memory_circuits(code, basis):
    c = sq.memory_circuit(code, distance=3, rounds=4, p=0.001, basis=basis)
    assert c.num_observables == 1 and c.num_detectors > 0
    stim = pytest.importorskip("stim")
    assert stim.Circuit(str(c)).num_detectors == c.num_detectors


def test_memory_circuit_arguments():
    for bad in (dict(code="toric"), dict(noise="si1000"), dict(basis="y")):
        with pytest.raises(ValueError):
            sq.memory_circuit(**{**dict(distance=3, rounds=3, p=0.001), **bad})
    with pytest.raises(ValueError):
        sq.memory_circuit(distance=3, rounds=3, p=1.5)
    with pytest.raises(ValueError):
        sq.memory_circuit(distance=13, rounds=3, p=0.001)
    with pytest.raises(ValueError):
        sq.memory_circuit(distance=3, rounds=10**6, p=0.001)


@pytest.mark.parametrize("name,n", [("gross", 144), ("72", 72)])
def test_bivariate_bicycle_codes(name, n):
    code = sq.BivariateBicycleCode(name)
    assert (code.n, code.k) == (n, 12)
    hx, hz = code.check_matrices()
    lx, lz = code.logicals()
    assert hx.shape == (n // 2, n) and not ((hx.astype(int) @ hz.T) % 2).any()
    assert np.array_equal((lx.astype(int) @ lz.T) % 2, np.eye(12, dtype=int))
    assert not ((hz.astype(int) @ lx.T) % 2).any() and not ((hx.astype(int) @ lz.T) % 2).any()
    autos = code.automorphisms()
    assert len(autos) == n and all(a.action.shape == (24, 24) for a in autos)
    identity = [a for a in autos if a.shift == (0, 0) and not a.dual][0]
    assert np.array_equal(identity.action, np.eye(24, dtype=np.uint8))


def test_bivariate_bicycle_memory():
    code = sq.BivariateBicycleCode("72")
    for basis in ("z", "x"):
        c = code.memory_circuit(3, 0.003, basis=basis)
        assert c.num_observables == 12
    stim = pytest.importorskip("stim")
    c = code.memory_circuit(3, 0.003)
    assert_same_model(str(c.detector_error_model()), str(stim.Circuit(str(c)).detector_error_model().flattened()))
    with pytest.raises(ValueError):
        code.gauging("f")


def test_gross_logical_measurement():
    code = sq.BivariateBicycleCode("gross")
    g = code.gauging("f")
    assert len(g.support) == 12 and g.worst_cut[1] > 0
    assert not ((g.hx.astype(int) @ g.hz.T) % 2).any()
    expanded = code.gauging("f", expanded=True)
    assert len(expanded.extra_edges) > 0
    c = code.logical_measurement_circuit("f", "x", pre=1, merged=2, post=1, p=0.003)
    assert c.num_observables == 13
    with pytest.raises(ValueError):
        code.gauging("nope")


@pytest.mark.parametrize(
    "make",
    [
        lambda: sq.surgery.zz_measurement(3, merged=3, p=0.002),
        lambda: sq.surgery.zz_measurement(3, merged=3, p=0.002, basis="x", pre=1, post=2),
        lambda: sq.surgery.xx_measurement(3, merged=3, p=0.002),
        lambda: sq.surgery.cnot(3, merged=3, p=0.002, inputs="x"),
        lambda: sq.surgery.repeated_zz(3, k=2, merged=2, p=0.002),
        lambda: sq.surgery.line(3, n=3, merged=2, p=0.002),
    ],
)
def test_surgery_circuits_equal_stims_models(make):
    stim = pytest.importorskip("stim")
    c = make()
    assert_same_model(
        str(c.detector_error_model(decompose_errors=True)),
        str(stim.Circuit(str(c)).detector_error_model(decompose_errors=True).flattened()),
    )


def test_surgery_arguments():
    with pytest.raises(ValueError):
        sq.surgery.zz_measurement(4, merged=3, p=0.001)
    with pytest.raises(ValueError):
        sq.surgery.line(3, n=1, merged=3, p=0.001)
    with pytest.raises(ValueError):
        sq.surgery.cnot(3, merged=3, p=0.001, inputs="y")


def test_stream_memory():
    kw = dict(distance=3, rounds=200, p=0.003, commit=3, buffer=3, shots=128, seed=7)
    r = sq.stream_memory(**kw, threads=1)
    assert r.shots == 128 and 0 <= r.failures <= 128
    assert r.window_seconds.shape == (2, len(r.windows))
    assert sq.stream_memory(**kw, threads=4).failures == r.failures
