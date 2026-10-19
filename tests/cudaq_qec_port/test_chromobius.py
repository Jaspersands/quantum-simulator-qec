# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# Ported from CUDA-Q QEC's libs/qec/python/tests/test_chromobius.py and its test_data/chromobius
# reference case (itself Chromobius's main_predict.test.cc basic case) (NVIDIA/cudaqx; Apache License
# 2.0); modified for stabilizer_qec: cudaq_qec is stabilizer_qec.cudaq_qec, the reference data inline,
# plus the plugin's options and a comparison with the chromobius package on colour-code models.
import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec

BASIC_DEM = """error(0.1) D0 L0
error(0.1) D0 D1 L1
error(0.1) D1 L2
detector(0, 0, 0, 0) D0
detector(0, 0, 0, 1) D1
"""

# name, syndrome bits (D0..Dn), expected observable bits (L0..Lm)
BASIC_SHOTS = [("empty", "00", "000"), ("d0", "10", "100"), ("d1", "01", "001"), ("d0_d1", "11", "010")]


def _bits(s):
    return [float(c == "1") for c in s]


def test_chromobius_matches_upstream_predict_reference():
    decoder = qec.get_decoder("chromobius", BASIC_DEM)
    for name, syndrome, expected in BASIC_SHOTS:
        result = decoder.decode(_bits(syndrome))
        assert result.converged, name
        assert result.result.tolist() == _bits(expected), name


def test_chromobius_batch_matches_upstream_predict_reference():
    decoder = qec.get_decoder("chromobius", BASIC_DEM)
    batch = decoder.decode_batch([_bits(s) for _, s, _ in BASIC_SHOTS])
    for r, (name, _, expected) in zip(batch, BASIC_SHOTS):
        assert r.converged, name
        assert r.result.tolist() == _bits(expected), name


def test_chromobius_sizes_and_version():
    decoder = qec.get_decoder("chromobius", BASIC_DEM)
    assert decoder.get_syndrome_size() == 2
    assert decoder.get_block_size() == 3  # block_size is the number of observables
    assert decoder.get_version() == "CUDA-Q QEC Chromobius Decoder wrapper"
    assert decoder.decode([0, 1]).opt_results is None


def test_chromobius_return_weight():
    decoder = qec.get_decoder("chromobius", BASIC_DEM, return_weight=True)
    r = decoder.decode([1.0, 1.0])
    assert set(r.opt_results) == {"weight"}
    # (The Möbius graph doubles each detector, so the one edge counts twice.)
    assert r.opt_results["weight"] == pytest.approx(2 * np.log(0.9 / 0.1), rel=1e-6)


def test_chromobius_needs_dem_text():
    with pytest.raises(RuntimeError, match="requires a Stim detector error model string"):
        qec.get_decoder("chromobius", np.eye(2, dtype=np.uint8))


def test_chromobius_errors():
    decoder = qec.get_decoder("chromobius", BASIC_DEM)
    with pytest.raises(RuntimeError, match="syndrome length must match the number of detectors"):
        decoder.decode([1.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="Chromobius Stim DEM parse failed"):
        qec.get_decoder("chromobius", "error(0.1) Q0")
    too_many = "".join(f"error(0.1) D0 L{k}\n" for k in range(65)) + "detector(0, 0, 0, 0) D0\n"
    with pytest.raises(RuntimeError, match="at most 64 observables"):
        qec.get_decoder("chromobius", too_many)
    # Options the port's Chromobius cannot honour are refused, never ignored.
    with pytest.raises(RuntimeError, match="drop_mobius_errors_involving_remnant_errors=False"):
        qec.get_decoder("chromobius", BASIC_DEM, drop_mobius_errors_involving_remnant_errors=False)
    with pytest.raises(RuntimeError, match="write_mobius_match_to_stderr"):
        qec.get_decoder("chromobius", BASIC_DEM, write_mobius_match_to_stderr=True)
    with pytest.raises(RuntimeError, match="expected bool"):
        qec.get_decoder("chromobius", BASIC_DEM, return_weight=1)
    qec.get_decoder("chromobius", BASIC_DEM, drop_mobius_errors_involving_remnant_errors=True, include_coords_in_mobius_dem=True, ignore_decomposition_failures=True)


def test_chromobius_matches_chromobius_package():
    chromobius = pytest.importorskip("chromobius")
    stim = pytest.importorskip("stim")
    import stabilizer_qec as sq

    for d, rounds, p in ((3, 3, 0.01), (5, 3, 0.005)):
        c = stim.Circuit(str(sq.CssCode.color_code(d).memory_circuit(rounds=rounds, p=p, annotate_colors=True)))
        dem = c.detector_error_model()
        dets = c.compile_detector_sampler(seed=3).sample(300)
        ref = chromobius.compile_decoder_for_dem(dem)
        packed = np.packbits(dets.astype(np.uint8), axis=1, bitorder="little")
        obs_ref, w_ref = ref.predict_weighted_obs_flips_from_dets_bit_packed(packed)
        obs_ref = np.unpackbits(obs_ref, axis=1, bitorder="little")[:, : dem.num_observables]
        ours = qec.get_decoder("chromobius", str(dem), return_weight=True)
        batch = ours.decode_batch(dets.astype(np.float64))
        w = np.array([o["weight"] for o in batch.opt_results], dtype=np.float32)
        np.testing.assert_array_equal(w, w_ref)  # the matching weights, bit for bit
        same = (batch.result.astype(np.uint8) == obs_ref).all(axis=1)
        assert same.mean() >= 0.95  # the rest are ties between equally light matchings
        assert batch.converged.all()
