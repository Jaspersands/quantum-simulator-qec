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


@pytest.mark.parametrize("name,n,k", [("gross", 144, 12), ("72", 72, 12), ("90", 90, 8), ("108", 108, 8), ("288", 288, 12), (144, 144, 12)])
def test_bivariate_bicycle_codes(name, n, k):
    code = sq.BivariateBicycleCode(name)
    assert (code.n, code.k) == (n, k)
    hx, hz = code.check_matrices()
    lx, lz = code.logicals()
    assert hx.shape == (n // 2, n) and not ((hx.astype(int) @ hz.T) % 2).any()
    assert (hx.sum(axis=1) == 6).all() and (hz.sum(axis=1) == 6).all()
    assert np.array_equal((lx.astype(int) @ lz.T) % 2, np.eye(k, dtype=int))
    assert not ((hz.astype(int) @ lx.T) % 2).any() and not ((hx.astype(int) @ lz.T) % 2).any()
    if n <= 144:
        autos = code.automorphisms()
        assert len(autos) == n and all(a.action.shape == (2 * k, 2 * k) for a in autos)
        identity = [a for a in autos if a.shift == (0, 0) and not a.dual][0]
        assert np.array_equal(identity.action, np.eye(2 * k, dtype=np.uint8))


@pytest.mark.parametrize("name", ["90", "108", "288"])
def test_every_bivariate_bicycle_cycle_is_valid(name):
    # Without noise, every detector of the depth-8 cycle is quiet and no observable flips: the
    # schedule measures each code's checks, not something else.
    code = sq.BivariateBicycleCode(name)
    for basis in ("z", "x"):
        c = code.memory_circuit(2, 0.0, basis=basis)
        assert c.num_observables == code.k
        dets, obs = c.compile_detector_sampler(seed=3).sample(64, separate_observables=True)
        assert not dets.any() and not obs.any()
        assert c.detector_error_model().num_errors == 0
    c = code.memory_circuit(2, 0.001)
    assert c.detector_error_model().num_errors > 0


def test_bivariate_bicycle_codes_from_polynomials():
    code = sq.BivariateBicycleCode.from_polynomials(9, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    named = sq.BivariateBicycleCode("108")
    assert (code.n, code.k) == (108, 8) and code.name is None
    assert np.array_equal(code.check_matrices()[0], named.check_matrices()[0])
    assert code.polynomials == (9, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    assert eval(repr(code), {"stabilizer_qec": sq}).polynomials == code.polynomials
    gross = sq.BivariateBicycleCode.from_polynomials(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    assert len(gross.gauging("f").support) == 12  # the gross code however it was made
    # A code of other polynomials: its noiseless memory is quiet too.
    other = sq.BivariateBicycleCode.from_polynomials(6, 6, [(2, 0), (0, 1), (0, 3)], [(0, 2), (1, 0), (3, 0)])
    assert (other.n, other.k) == (72, 4)
    dets = other.memory_circuit(2, 0.0).compile_detector_sampler(seed=1).sample(16)
    assert not dets.any()
    with pytest.raises(ValueError, match="differ"):
        sq.BivariateBicycleCode.from_polynomials(6, 6, [(1, 0), (7, 0), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    with pytest.raises(ValueError, match="no logical"):
        sq.BivariateBicycleCode.from_polynomials(6, 6, [(1, 0), (0, 1), (2, 3)], [(0, 1), (1, 0), (3, 2)])
    with pytest.raises(ValueError, match="three monomials"):
        sq.BivariateBicycleCode.from_polynomials(6, 6, [(1, 0), (0, 1)], [(0, 3), (1, 0), (2, 0)])
    with pytest.raises(ValueError):
        sq.BivariateBicycleCode("100")
    with pytest.raises(ValueError, match="gross code only"):
        sq.BivariateBicycleCode("108").gauging("f")


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
