"""MbpDecoder, memory BP over GF(4), against ldpc's mbp.hpp: decodings, convergence, iteration
counts and log-likelihood ratios recorded from the C++ built by tools/mbp_reference.cpp (ldpc
2.4.1 cannot build it), on CSS codes and on GF(4) matrices with Y entries, product-sum and
min-sum, several alpha, beta and gamma."""

from __future__ import annotations

import json
import pathlib
import pickle
import sys

import numpy as np
import pytest

import stabilizer_qec as sq

CASES = json.loads((pathlib.Path(__file__).parent / "data" / "mbp_reference.json").read_text())


def build(case):
    kw = {k: case[k] for k in ("error_rate", "xyz_bias", "alpha_parameter", "beta_parameter", "gamma_parameter", "bp_method", "max_iter")}
    if "Hgf4" in case:
        return sq.MbpDecoder(Hgf4=np.array(case["Hgf4"]), **kw)
    return sq.MbpDecoder(HX_CSS=np.array(case["HX_CSS"], dtype=np.uint8), HZ_CSS=np.array(case["HZ_CSS"], dtype=np.uint8), **kw)


@pytest.mark.parametrize("index", range(len(CASES)))
def test_equals_ldpcs_cpp(index):
    case = CASES[index]
    dec = build(case)
    for syndrome, want in zip(case["syndromes"], case["expected"]):
        dec.decode(np.array(syndrome, dtype=np.uint8))
        assert dec.decoding.tolist() == want["decoding"]
        assert (dec.converge, dec.iterations) == (want["converge"], want["iterations"])
        llr = np.array(want["llr"])
        if sys.platform == "darwin":
            # Recorded on macOS: the same libm, so the same bits.
            assert np.array_equal(dec.log_prob_ratios, llr, equal_nan=True)
        else:
            assert np.allclose(dec.log_prob_ratios, llr, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_css_outputs_and_union_find_fallback():
    # The hypergraph product of two length-4 repetition codes (a planar surface code).
    rep = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    hx = np.hstack([np.kron(rep, np.eye(4, dtype=np.uint8)), np.kron(np.eye(3, dtype=np.uint8), rep.T)])
    hz = np.hstack([np.kron(np.eye(4, dtype=np.uint8), rep), np.kron(rep.T, np.eye(3, dtype=np.uint8))])
    rng = np.random.default_rng(1)
    dec = sq.MbpDecoder(HX_CSS=hx, HZ_CSS=hz, error_rate=0.05, max_iter=3, alpha_parameter=np.array([0.8, 0.8, 0.8]))
    assert dec.max_iter == 3 and np.allclose(dec.alpha, 0.8) and np.allclose(dec.error_channel, 0.05 / 3)
    for _ in range(20):
        ex, ez = (rng.random(hx.shape[1]) < 0.1).astype(np.uint8), (rng.random(hx.shape[1]) < 0.1).astype(np.uint8)
        sx, sz = hz @ ex % 2, hx @ ez % 2
        outx, outz = dec.decode(sx=sx, sz=sz)
        d = dec.decoding
        assert np.array_equal(outx, np.isin(d, (1, 2))) and np.array_equal(outz, np.isin(d, (2, 3)))
        assert np.array_equal(dec.syndrome, np.concatenate([sx, sz]))
        same = dec.decode(np.concatenate([sx, sz]))
        assert np.array_equal(same[0], outx) and np.array_equal(same[1], outz)
        # Union-find by inversion where MBP does not converge: always explains the syndrome.
        ux, uz = dec.uf_decode(sx=sx, sz=sz)
        assert np.array_equal(hz @ ux % 2, sx) and np.array_equal(hx @ uz % 2, sz)
    copy = pickle.loads(pickle.dumps(dec))
    s = np.zeros(hx.shape[0] + hz.shape[0], dtype=np.uint8)
    s[0] = 1
    assert np.array_equal(copy.decode(s)[0], dec.decode(s)[0])


def test_arguments():
    h = np.array([[1, 3, 3, 1, 0], [0, 1, 3, 3, 1], [1, 0, 1, 3, 3], [3, 1, 0, 1, 3]])  # the five-qubit code
    dec = sq.MbpDecoder(Hgf4=h, error_rate=0.03, xyz_bias=[1, 0, 1], bp_method="ms", gamma_parameter=0.8)
    assert dec.bp_method == 1 and dec.max_iter == 5
    assert np.allclose(dec.error_channel[1], 0) and np.allclose(dec.error_channel[0], 0.015)
    out = dec.decode(np.array([1, 0, 0, 1]))
    assert out.shape == (5,) and set(out.tolist()) <= {0, 1, 2, 3}
    dec.update_alpha(0.5)
    assert dec.alpha_parameter == [[0.5] * 5] * 3
    with pytest.raises(ValueError):
        dec.update_alpha(np.ones(4))
    with pytest.raises(ValueError):
        dec.update_alpha("big")
    with pytest.raises(ValueError):
        dec.decode(np.zeros(3))
    with pytest.raises(ValueError):
        dec.decode()
    with pytest.raises(ValueError):
        dec.uf_decode(sx=np.zeros(2), sz=np.zeros(2))  # GF(4) input
    for bad in (dict(), dict(HX_CSS=np.eye(2, dtype=np.uint8), HZ_CSS=np.eye(3, dtype=np.uint8)), dict(Hgf4=h, bp_method="nope"), dict(Hgf4=np.array([[4, 1]]))):
        with pytest.raises(ValueError):
            sq.MbpDecoder(**bad)
    with pytest.warns(UserWarning):
        sq.MbpDecoder(Hgf4=h, error_rate=0.1, error_channel=np.full((3, 5), 0.01))
