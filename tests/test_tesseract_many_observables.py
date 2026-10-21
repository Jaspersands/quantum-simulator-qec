"""Tesseract's A* decoder on models of any number of observables, as Tesseract's takes them: a
model of 100 observables decodes to Tesseract's errors and predictions (recorded from its
2026-10-13 build in data/tesseract_wide_reference.json, and live where it installs), and to the
predictions of its two halves of at most 64 observables each; so does this package's own search
decoder, whose predictions come back packed at any width."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re

import numpy as np
import pytest

import stabilizer_qec as sq
import stabilizer_qec.tesseract_decoder as O
from stabilizer_qec.tesseract_decoder.tesseract import TesseractConfig, TesseractDecoder

with open(os.path.join(os.path.dirname(__file__), "data", "tesseract_wide_reference.json")) as f:
    REF = json.load(f)
DEM = REF["dem"]


def shots() -> np.ndarray:
    dem = sq.DetectorErrorModel(DEM)
    rows = np.zeros((len(REF["fired"]), dem.num_detectors), dtype=bool)
    for k, fired in enumerate(REF["fired"]):
        rows[k, fired] = True
    return rows


def half(text: str, lo: int, hi: int) -> str:
    """The model with only observables lo..hi-1, renumbered from 0."""
    def keep(line: str) -> str:
        if line.startswith("logical_observable"):
            return f"logical_observable L{hi - lo - 1}"
        targets = [t for t in line.split()[1:] if not t.startswith("L") or lo <= int(t[1:]) < hi]
        targets = [f"L{int(t[1:]) - lo}" if t.startswith("L") else t for t in targets]
        return " ".join([line.split()[0]] + targets)
    return "\n".join(keep(line) for line in text.splitlines()) + f"\nlogical_observable L{hi - lo - 1}"


def test_100_observables_decode():
    rows = shots()
    full = TesseractDecoder(TesseractConfig(sq.DetectorErrorModel(DEM))).decode_batch(rows)
    assert full.shape == (rows.shape[0], 100)
    assert full[:, 64:].any()
    lo = TesseractDecoder(TesseractConfig(sq.DetectorErrorModel(half(DEM, 0, 50)))).decode_batch(rows)
    hi = TesseractDecoder(TesseractConfig(sq.DetectorErrorModel(half(DEM, 50, 100)))).decode_batch(rows)
    np.testing.assert_array_equal(full, np.hstack([lo, hi]))


def test_100_observables_are_tesseracts():
    rows = shots()
    decoder = TesseractDecoder(TesseractConfig(sq.DetectorErrorModel(DEM)))
    assert [np.flatnonzero(p).tolist() for p in decoder.decode_batch(rows)] == REF["predictions"]
    for k in range(rows.shape[0]):
        assert [decoder.decode_to_errors(rows[k]), int(decoder.low_confidence_flag)] == REF["errors"][k]
    assert decoder.num_observables == 100
    assert sorted({o for e in decoder.errors for o in e.symptom.observables})[-1] >= 64


def test_100_observables_verbose_log():
    rows = shots()
    out = io.StringIO()
    with contextlib.redirect_stdout(out), O.ostream_redirect(stdout=True):
        decoder = TesseractDecoder(TesseractConfig(sq.DetectorErrorModel(DEM), verbose=True, num_det_orders=2))
        out.seek(0)
        out.truncate()
        decoder.decode_to_errors(rows[REF["verbose_shot"]])
    assert out.getvalue() == REF["verbose_log"]
    assert re.search(r"obs_mask \[[^]]*\b7[0-9]\b", out.getvalue())


def test_100_observables_live():
    from conftest import reference

    T = reference("tesseract_decoder")
    stim = pytest.importorskip("stim")
    import tesseract_decoder.tesseract  # noqa: F401

    rows = shots()
    theirs = T.tesseract.TesseractDecoder(T.tesseract.TesseractConfig(stim.DetectorErrorModel(DEM), num_det_orders=3, det_beam=7)).decode_batch(rows)
    ours = TesseractDecoder(TesseractConfig(stim.DetectorErrorModel(DEM), num_det_orders=3, det_beam=7)).decode_batch(rows)
    np.testing.assert_array_equal(ours, theirs)


def test_cli_100_observables(tmp_path, capsys):
    # (Tesseract's program on the same model and shots, their observables appended.)
    from stabilizer_qec.tesseract_decoder.__main__ import main

    rows = shots()
    obs = np.zeros((rows.shape[0], 100), dtype=bool)
    for k, predicted in enumerate(REF["predictions"]):
        obs[k, predicted] = True
    (tmp_path / "wide.dem").write_text(DEM)
    (tmp_path / "shots.b8").write_bytes(np.packbits(np.hstack([rows, obs]), axis=1, bitorder="little").tobytes())
    code = main(["--dem", str(tmp_path / "wide.dem"), "--in", str(tmp_path / "shots.b8"), "--in-format", "b8", "--in-includes-appended-observables",
                 "--out", str(tmp_path / "p.01"), "--out-format", "01", "--threads", "2"])
    out = capsys.readouterr()
    assert (code, out.err) == (0, "")
    lines = (tmp_path / "p.01").read_text().splitlines()
    assert [[k for k, c in enumerate(line) if c == "1"] for line in lines] == REF["predictions"] and {len(line) for line in lines} == {100}
    low = sum(flag for _, flag in REF["errors"])
    assert re.fullmatch(rf"num_shots = {rows.shape[0]} num_low_confidence = {low} num_errors = 0 total_time_seconds = \S+\n", out.out)


@pytest.mark.parametrize("threads", [1, 3])
def test_search_decoder_100_observables(threads):
    rows = shots()
    full = sq.SearchDecoder(DEM).decode_batch(rows, threads=threads)
    assert full.shape == (rows.shape[0], 100) and full.dtype == np.uint8
    lo = sq.SearchDecoder(half(DEM, 0, 50)).decode_batch(rows)
    hi = sq.SearchDecoder(half(DEM, 50, 100)).decode_batch(rows)
    np.testing.assert_array_equal(full, np.hstack([lo, hi]))
    packed = sq.SearchDecoder(DEM).decode_batch(rows, bit_packed_predictions=True, threads=threads)
    np.testing.assert_array_equal(packed, np.packbits(full, axis=1, bitorder="little"))
    # The same answers as Tesseract's decoder under the same settings.
    np.testing.assert_array_equal(full.astype(bool), TesseractDecoder(TesseractConfig(sq.DetectorErrorModel(DEM))).decode_batch(rows))


def test_search_decoder_narrow_models_unchanged():
    # (Packed rows at widths 0, 1 and 64 read as the 64-bit masks did.)
    for k in (0, 1, 64):
        lines = ["error(0.1) D0" + (f" L{k - 1}" if k else ""), "error(0.1) D0 D1", "error(0.2) D1"]
        rows = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=bool)
        out = sq.SearchDecoder("\n".join(lines)).decode_batch(rows)
        assert out.shape == (4, k)
        if k:
            assert out[:, k - 1].tolist() == [1, 0, 0, 0]


def test_flat_model_decoders_still_refuse_past_64():
    # (The refusal names what refuses and what takes any number.)
    for make in (sq.UnionFind, sq.BpOsd, sq.BeliefMatching):
        with pytest.raises(ValueError, match=r"take at most 64 \(the Python search decoder, "):
            make(DEM)
