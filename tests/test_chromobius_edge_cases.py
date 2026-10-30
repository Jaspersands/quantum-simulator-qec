"""Saved ignored-detector ties and safe recovery from the macOS arm64 audit.

These fixtures record equal matching weights, not a proof about global lifting optimality.
They supplement (and do not loosen) the existing macOS exact reference corpus.
"""
from __future__ import annotations

import gzip
import json
import pathlib

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import chromobius

DATA = json.loads(gzip.decompress((pathlib.Path(__file__).parent / "data" / "chromobius_cli_edges.json.gz").read_bytes()))


def test_saved_ignored_detector_ties_keep_equal_weights_and_stable_predictions():
    dem = sq.DetectorErrorModel(DATA["ignored_dem"])
    decoder = chromobius.compile_decoder_for_dem(dem)
    rows = np.array([r["dets"] for r in DATA["ignored"]], dtype=np.uint8)
    batch_p, batch_w = decoder.predict_weighted_obs_flips_from_dets_bit_packed(rows)
    for k, r in enumerate(DATA["ignored"]):
        fresh = chromobius.compile_decoder_for_dem(dem)
        p, weight = fresh.predict_weighted_obs_flips_from_dets_bit_packed(rows[k])
        assert p.tolist() == r["port_prediction"] == batch_p[k].tolist()
        assert float(weight) == r["port_weight"] == r["reference_weight"] == float(batch_w[k])


def test_saved_ignored_detector_ties_differ_from_the_recorded_reference():
    """Each saved tie's prediction is the port's, which differs from the one Chromobius's macOS
    arm64 build recorded for it (equal weights, two liftings)."""
    dem = sq.DetectorErrorModel(DATA["ignored_dem"])
    decoder = chromobius.compile_decoder_for_dem(dem)
    for r in DATA["ignored"]:
        assert decoder.predict_obs_flips_from_dets_bit_packed(np.array(r["dets"], dtype=np.uint8)).tolist() == r["port_prediction"]
        assert r["port_prediction"] != r["reference_prediction"]


def test_decode_via_files_refuses_an_unwritable_output_and_negative_counts(tmp_path):
    # As Chromobius refuses them: an output it cannot write is a ValueError naming the path, and a
    # negative count is a TypeError, raised before anything is written.
    dem = sq.DetectorErrorModel(DATA["ignored_dem"])
    dem_path = tmp_path / "m.dem"
    dem_path.write_text(DATA["ignored_dem"], encoding="utf-8")
    dets = tmp_path / "d.b8"
    dets.write_bytes(bytes((dem.num_detectors + 7) // 8))
    base = dict(num_shots=1, num_dets=dem.num_detectors, num_obs=dem.num_observables, dem_path=dem_path,
                dets_b8_in_path=dets, tmp_dir=tmp_path)
    decoder = chromobius.sinter_decoders()["chromobius"]
    missing = tmp_path / "no_such_dir" / "o.b8"
    with pytest.raises(ValueError) as e:
        decoder.decode_via_files(obs_predictions_b8_out_path=missing, **base)
    assert str(e.value) == f"Failed to open '{missing}' for writing."
    out = tmp_path / "o.b8"
    for count in ({"num_shots": -1}, {"num_obs": -1}):
        with pytest.raises(TypeError, match="non-negative"):
            decoder.decode_via_files(obs_predictions_b8_out_path=out, **{**base, **count})
    assert not out.exists()


def test_lifting_failure_does_not_corrupt_subsequent_shots():
    dem = sq.DetectorErrorModel(DATA["recovery_dem"])
    kept = chromobius.compile_decoder_for_dem(dem)
    for r in DATA["recovery"]:
        row = np.array(r["dets"], dtype=np.uint8)
        if isinstance(r["result"], str):
            with pytest.raises(ValueError, match="shot 0: no lifting") as ex:
                kept.predict_obs_flips_from_dets_bit_packed(row)
            assert str(ex.value) == r["result"]
        else:
            got = kept.predict_obs_flips_from_dets_bit_packed(row)
            fresh = chromobius.compile_decoder_for_dem(dem).predict_obs_flips_from_dets_bit_packed(row)
            assert got.tolist() == fresh.tolist() == r["result"]


def test_sinter_decoder_refuses_data_whose_rows_are_not_adjacent_bytes():
    """Chromobius's sinter decoder reads each row of bit-packed data in place: a view whose
    row's bytes are strided, or a column-major array, is refused with Chromobius's message, an
    empty batch too. Any integer dtype, contiguous, decodes to the same predictions."""
    dem = sq.DetectorErrorModel(DATA["ignored_dem"])
    rows = np.array([r["dets"] for r in DATA["ignored"]], dtype=np.uint8)
    want = [r["port_prediction"] for r in DATA["ignored"]]
    sinter = chromobius.sinter_decoders()["chromobius"].compile_decoder_for_dem(dem=dem)
    assert sinter.decode_shots_bit_packed(bit_packed_detection_event_data=rows).tolist() == want
    assert sinter.decode_shots_bit_packed(bit_packed_detection_event_data=rows.astype(np.uint16)).tolist() == want
    wide = np.zeros((rows.shape[0], 2 * rows.shape[1]), dtype=np.uint8)
    wide[:, ::2] = rows
    for bad in (wide[:, ::2], np.asfortranarray(rows), rows[:0]):
        with pytest.raises(ValueError, match="^bit_packed_detection_event_data.strides\\(1\\) != 1$"):
            sinter.decode_shots_bit_packed(bit_packed_detection_event_data=bad)
