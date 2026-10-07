"""Shot data files in Stim's six result formats: written byte for byte as Stim writes them, and
Stim's files read back to the same shots."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")

FORMATS = ["01", "b8", "r8", "ptb64", "hits", "dets"]


@pytest.mark.parametrize("fmt", FORMATS)
def test_files_are_stims(fmt, tmp_path):
    rng = np.random.default_rng(1)
    for shots, (m, d, o), density in itertools.product([1, 63, 64, 65, 128, 200], [(0, 1, 0), (0, 7, 2), (5, 0, 0), (0, 64, 1), (0, 130, 2), (0, 600, 0), (70, 0, 0)], [0.0, 0.05, 0.5, 1.0]):
        if fmt == "ptb64" and shots % 64:
            with pytest.raises(ValueError, match="multiple of 64"):
                sq.write_shot_data_file(data=np.zeros((shots, m + d + o), bool), path=tmp_path / "x", format=fmt, num_measurements=m, num_detectors=d, num_observables=o)
            continue
        data = rng.random((shots, m + d + o)) < density
        kw = dict(format=fmt, num_measurements=m, num_detectors=d, num_observables=o)
        ours, theirs = tmp_path / "ours", tmp_path / "theirs"
        sq.write_shot_data_file(data=data, path=ours, **kw)
        stim.write_shot_data_file(data=data, path=str(theirs), **kw)
        assert ours.read_bytes() == theirs.read_bytes(), (shots, m, d, o, density)
        assert np.array_equal(sq.read_shot_data_file(path=theirs, **kw), data)
        packed = sq.read_shot_data_file(path=theirs, bit_packed=True, **kw)
        assert np.array_equal(packed, stim.read_shot_data_file(path=str(theirs), bit_packed=True, **kw))
        sq.write_shot_data_file(data=packed, path=ours, **kw)
        assert ours.read_bytes() == theirs.read_bytes()


def test_bad_input(tmp_path):
    with pytest.raises(ValueError, match="format"):
        sq.write_shot_data_file(data=np.zeros((1, 1), bool), path=tmp_path / "x", format="csv", num_detectors=1)
    with pytest.raises(ValueError):
        sq.write_shot_data_file(data=np.zeros((1, 3), bool), path=tmp_path / "x", format="01", num_detectors=2)
    with pytest.raises(ValueError, match="not both"):
        sq.write_shot_data_file(data=np.zeros((1, 3), bool), path=tmp_path / "x", format="01", num_measurements=1, num_detectors=2)
    (tmp_path / "bad").write_bytes(b"0102\n")
    with pytest.raises(ValueError):
        sq.read_shot_data_file(path=tmp_path / "bad", format="01", num_detectors=4)
    (tmp_path / "bad").write_bytes(b"shot D9\n")
    with pytest.raises(ValueError):
        sq.read_shot_data_file(path=tmp_path / "bad", format="dets", num_detectors=4)
