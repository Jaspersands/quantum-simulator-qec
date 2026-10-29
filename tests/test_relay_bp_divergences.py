"""Where stabilizer_qec.relay_bp answers inputs IBM's relay_bp 0.2.2 fails on: a check matrix
storing an entry twice (relay_bp panics, "Indices are not sorted"), and a model with a fault
of probability 1 under the sinter decoders' default pruning (relay_bp's pruning builds its
biases as np.matrix, and from the wrong matrix, so its binding raises TypeError). Neither needs
relay_bp: the answers are checked against the matrix and the model."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

import stabilizer_qec as sq  # noqa: E402
import stabilizer_qec.relay_bp as relay_bp  # noqa: E402


def test_a_check_stored_twice_sets_off_nothing():
    # Column 0 stores row 0 twice (1 + 1 = 2): that entry is even, so no correction sets off
    # row 0 through it, and success says whether the correction's real checks are the syndrome.
    # It is still an edge, as the dense matrix's 2 is; an explicit stored zero is one more.
    a = sp.csr_matrix((np.array([1, 1, 1, 1, 1], dtype=np.uint8), np.array([0, 0, 1, 1, 2]), np.array([0, 3, 5])), shape=(2, 3))
    with_zero = sp.csr_matrix((np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8), np.array([0, 0, 1, 0, 1, 2]), np.array([0, 3, 6])), shape=(2, 3))
    true = a.toarray() % 2
    p = np.array([0.3, 0.01, 0.01])
    rng = np.random.default_rng(0)
    for s in [np.array([1, 0], dtype=np.uint8)] + [rng.integers(0, 2, 2).astype(np.uint8) for _ in range(8)]:
        results = []
        for matrix in (a, a.tocsc(), a.toarray(), with_zero):
            for dec in (relay_bp.MinSumBPDecoderF64(matrix, p, max_iter=20), relay_bp.RelayDecoderF64(matrix, p, pre_iter=10, num_sets=4, set_max_iter=10, seed=1)):
                r = dec.decode_detailed(s)
                assert np.array_equal(r.decoded_detectors, true @ r.decoding % 2)
                assert r.success == np.array_equal(true @ r.decoding % 2, s)
                results.append((r.decoding.tolist(), r.success))
        # The three spellings of the one graph decode alike.
        assert results[0:6:2] == [results[0]] * 3 and results[1:6:2] == [results[1]] * 3


DEM = """error(0.05) D0 D1
error(0.05) D1 D2
error(0.05) D2 D3 L0
error(0.05) D0
error(0.05) D3
error(1) D1 D3 L0
"""


@pytest.mark.parametrize("certain", ["D1 D3 L0", "D0 D2 L0", "D1 D3"])
def test_sinter_decoders_with_a_certain_fault(tmp_path: pathlib.Path, certain: str):
    # A fault of probability 1 is pruned and its detectors and observables become biases: the
    # predictions are as good as on the model without it (whose detectors and observables are
    # the biases' complements), not their complements.
    from stabilizer_qec.relay_bp.stim import CheckMatrices, sinter_decoders

    dem = sq.DetectorErrorModel(DEM.replace("D1 D3 L0", certain))
    m = CheckMatrices.from_dem(dem)
    flip = np.zeros(4, dtype=np.uint8)
    flip[[int(t[1:]) for t in certain.split() if t[0] == "D"]] = 1
    assert np.array_equal(m.syndrome_bias, flip) and m.syndrome_bias.dtype == np.uint8 and type(m.syndrome_bias) is np.ndarray
    assert np.array_equal(m.observables_bias, [int("L0" in certain)]) and type(m.observables_bias) is np.ndarray
    dets, obs, _ = dem.compile_sampler(seed=7).sample(2000, bit_packed=True)
    obs = np.unpackbits(obs, axis=1, bitorder="little", count=1)[:, 0]
    (tmp_path / "m.dem").write_text(str(dem))
    (tmp_path / "dets.b8").write_bytes(dets.tobytes())
    for name, dec in sinter_decoders().items():
        out = tmp_path / f"{name}.b8"
        dec.decode_via_files(num_shots=2000, num_dets=4, num_obs=1, dem_path=tmp_path / "m.dem", dets_b8_in_path=tmp_path / "dets.b8", obs_predictions_b8_out_path=out, tmp_dir=tmp_path)
        predicted = np.frombuffer(out.read_bytes(), dtype=np.uint8)
        assert set(predicted.tolist()) <= {0, 1}, name
        assert (predicted != obs).mean() < 0.01, name
        compiled = dec.compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(bit_packed_detection_event_data=dets)
        assert type(compiled) is np.ndarray and compiled.shape == (2000, 1), name
        assert np.array_equal(compiled[:, 0], predicted), name
