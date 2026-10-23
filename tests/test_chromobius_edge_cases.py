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
        assert r["port_prediction"] != r["reference_prediction"]


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
