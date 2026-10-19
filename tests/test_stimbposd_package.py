"""stabilizer_qec.stimbposd against stimbposd 0.2.0 on ldpc 2.4.1: BPOSD and BPLSD in each of
stimbposd's sinter variants and more of their options give stimbposd's predictions bit for bit,
on the recorded reference (tests/data/stimbposd_reference.json.gz, recorded by
tools/stimbposd_reference.py, which replays stimbposd's decoders from its source on real ldpc)
and, where they install, on real ldpc, Stim and beliefmatching here, or on stimbposd itself.
Also its check matrices, its batch options and its sinter decoders. (stimbposd's own tests,
run on the port, are in tests/stimbposd_port.)"""

from __future__ import annotations

import gzip
import inspect
import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

import stabilizer_qec as sq  # noqa: E402
import stabilizer_qec.stimbposd as ours  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = json.loads(gzip.decompress((ROOT / "tests" / "data" / "stimbposd_reference.json.gz").read_bytes()))
sys.path.insert(0, str(ROOT / "tools"))
import stimbposd_reference as recorded  # noqa: E402

SINTER_NAMES = ["bposd", "bposd-serial", "bposd-minsum", "bposd-serial-minsum", "bplsd", "bplsd-serial", "bplsd-minsum", "bplsd-serial-minsum"]


def unpacked(hexed: str, rows: int, bits: int) -> np.ndarray:
    raw = np.frombuffer(bytes.fromhex(hexed), dtype=np.uint8).reshape(rows, -1)
    return np.unpackbits(raw, axis=1, bitorder="little")[:, :bits]


def ours_predicting(kind: str, model, **kw):
    return (ours.BPOSD if kind == "osd" else ours.BPLSD)(model, **kw)


def variants():
    for kind, table in (("osd", recorded.OSD_VARIANTS), ("lsd", recorded.LSD_VARIANTS)):
        for name, kw in table.items():
            yield kind, name, kw


@pytest.mark.parametrize("problem", REF["problems"], ids=[p["name"] for p in REF["problems"]])
def test_recorded_reference(problem):
    model = sq.DetectorErrorModel(problem["model"])
    shots = unpacked(problem["shots"], problem["num_shots"], model.num_detectors)
    packed = np.packbits(shots, axis=1, bitorder="little")
    for kind, name, kw in variants():
        dec = ours_predicting(kind, model, **kw)
        got = dec.decode_batch(shots)
        assert got.dtype == bool and got.shape == (len(shots), model.num_observables)
        assert recorded.packed_hex(got) == problem["predictions"][name], (problem["name"], name)
        # bit-packed in and out, as sinter asks for them
        assert np.array_equal(dec.decode_batch(packed[:8], bit_packed_shots=True, bit_packed_predictions=True), np.packbits(got[:8], axis=1, bitorder="little"))


def test_live_against_ldpc():
    """The same replay of stimbposd on the real ldpc, Stim and beliefmatching, on fresh shots."""
    stim = pytest.importorskip("stim")
    ldpc = pytest.importorskip("ldpc")
    beliefmatching = pytest.importorskip("beliefmatching")
    if ldpc.__version__ != "2.4.1":
        pytest.skip(f"ldpc {ldpc.__version__} is installed; the test reproduces 2.4.1")
    for name, text, shots in recorded.problems(stim):
        model = stim.DetectorErrorModel(text)
        shots = shots[:30]
        for kind, vname, kw in variants():
            theirs = recorded.replay(ldpc, beliefmatching.detector_error_model_to_check_matrices, model, kind, shots, **dict(kw))
            # Ours given Stim's own model object, and given this package's.
            assert np.array_equal(ours_predicting(kind, model, **kw).decode_batch(shots), theirs), (name, vname)
            assert np.array_equal(ours_predicting(kind, sq.DetectorErrorModel(text), **kw).decode_batch(shots), theirs), (name, vname)


def test_against_stimbposd_itself():
    """Where stimbposd itself installs (it is not a dependency of anything here)."""
    stimbposd = pytest.importorskip("stimbposd")
    stim = pytest.importorskip("stim")
    if stimbposd.__version__ != "0.2.0":
        pytest.skip(f"stimbposd {stimbposd.__version__} is installed; the test reproduces 0.2.0")
    for name, text, shots in recorded.problems(stim):
        model = stim.DetectorErrorModel(text)
        for kind, vname, kw in variants():
            theirs = (stimbposd.BPOSD if kind == "osd" else stimbposd.BPLSD)(model, **kw).decode_batch(shots)
            assert np.array_equal(ours_predicting(kind, model, **kw).decode_batch(shots), theirs), (name, vname)
        theirs, mine = stimbposd.sinter_decoders(), ours.sinter_decoders()
        assert list(theirs) == list(mine)


def test_check_matrices():
    model = sq.DetectorErrorModel(REF["problems"][2]["model"])
    m = ours.detector_error_model_to_check_matrices(model)
    assert m.check_matrix.shape == (model.num_detectors, len(m.priors))
    assert m.observables_matrix.shape == (model.num_observables, len(m.priors))
    stim = pytest.importorskip("stim")
    beliefmatching = pytest.importorskip("beliefmatching")
    for problem in REF["problems"]:
        theirs = beliefmatching.detector_error_model_to_check_matrices(stim.DetectorErrorModel(problem["model"]), allow_undecomposed_hyperedges=True)
        mine = ours.detector_error_model_to_check_matrices(sq.DetectorErrorModel(problem["model"]))
        for field in ("check_matrix", "observables_matrix", "edge_check_matrix", "edge_observables_matrix", "hyperedge_to_edge_matrix"):
            a, b = getattr(theirs, field), getattr(mine, field)
            assert a.shape == b.shape and (a != b).nnz == 0, field
        assert np.array_equal(theirs.priors, mine.priors)
    # stimbposd's default (unlike beliefmatching's) keeps undecomposed hyperedges
    assert inspect.signature(ours.detector_error_model_to_check_matrices).parameters["allow_undecomposed_hyperedges"].default is True


def test_interface():
    assert ours.__version__ == "0.2.0"
    decoders = ours.sinter_decoders()
    assert list(decoders) == SINTER_NAMES
    assert all(isinstance(d, sq.sinter.Decoder) for d in decoders.values())
    assert isinstance(decoders["bposd"], ours.SinterDecoder_BPOSD) and isinstance(decoders["bplsd"], ours.SinterDecoder_BPLSD)
    assert decoders["bposd-serial-minsum"].bposd_kwargs == dict(schedule="serial", random_schedule_seed=0) and decoders["bposd-serial-minsum"].bp_method == "minimum_sum"
    assert decoders["bplsd-serial"].bplsd_kwargs == dict(schedule="serial", random_schedule_seed=0)
    defaults = {k: v.default for k, v in inspect.signature(ours.BPOSD).parameters.items() if v.default is not inspect.Parameter.empty}
    assert defaults == dict(max_bp_iters=30, bp_method="product_sum", osd_order=60, osd_method="osd_cs")
    defaults = {k: v.default for k, v in inspect.signature(ours.BPLSD).parameters.items() if v.default is not inspect.Parameter.empty}
    assert defaults == dict(max_bp_iters=30, bp_method="product_sum", lsd_order=0, lsd_method="lsd_0", bits_per_step=1, always_run_lsd=False)


def test_single_shots_and_sinter(tmp_path: pathlib.Path):
    problem = REF["problems"][1]
    model = sq.DetectorErrorModel(problem["model"])
    shots = unpacked(problem["shots"], problem["num_shots"], model.num_detectors)
    for kind, name in (("osd", "bposd-serial"), ("lsd", "bplsd-minsum")):
        expected = unpacked(problem["predictions"][name], len(shots), model.num_observables)
        dec = ours_predicting(kind, model, **(recorded.OSD_VARIANTS if kind == "osd" else recorded.LSD_VARIANTS)[name])
        assert dec.num_detectors == model.num_detectors and dec.num_errors == model.num_errors
        for i in range(5):
            assert np.array_equal(dec.decode(shots[i]), expected[i])
        # the compiled sinter decoder, and decoding through files
        sinter_dec = ours.sinter_decoders()[name]
        compiled = sinter_dec.compile_decoder_for_dem(dem=model)
        packed = np.packbits(shots, axis=1, bitorder="little")
        assert compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed).tobytes().hex() == problem["predictions"][name]
        dem_path, dets_path, out_path = tmp_path / f"{name}.dem", tmp_path / f"{name}.b8", tmp_path / f"{name}.out.b8"
        model.to_file(dem_path)
        dets_path.write_bytes(packed.tobytes())
        sinter_dec.decode_via_files(num_shots=len(shots), num_dets=model.num_detectors, num_obs=model.num_observables, dem_path=dem_path, dets_b8_in_path=dets_path, obs_predictions_b8_out_path=out_path, tmp_dir=tmp_path)
        assert out_path.read_bytes().hex() == problem["predictions"][name]
