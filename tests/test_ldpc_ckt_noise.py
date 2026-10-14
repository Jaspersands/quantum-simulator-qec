"""stabilizer_qec.ldpc's circuit-noise layer against ldpc 2.4.1 on Stim: check matrices from
detector error models, the overlapping-window decoders and ldpc's sinter decoders give ldpc's
results on the same models and shots."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip("scipy.sparse")
ldpc = pytest.importorskip("ldpc", minversion="2.4")
stim = pytest.importorskip("stim")

import stabilizer_qec as sq  # noqa: E402
from stabilizer_qec.ldpc.ckt_noise import BpOsdOverlappingWindowDecoder, LsdOverlappingWindowDecoder, detector_error_model_to_check_matrices  # noqa: E402
from stabilizer_qec.ldpc.sinter_decoders import SinterBeliefFindDecoder, SinterBpOsdDecoder, SinterLsdDecoder  # noqa: E402

CIRCUITS = [
    ("surface_code:rotated_memory_x", 3, 4, 0.004),
    ("surface_code:rotated_memory_z", 5, 3, 0.002),
    ("repetition_code:memory", 5, 6, 0.02),
    ("color_code:memory_xyz", 3, 3, 0.003),
]


def models(decompose: bool):
    for task, d, rounds, p in CIRCUITS:
        theirs = stim.Circuit.generated(task, distance=d, rounds=rounds, after_clifford_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p)
        mine = sq.Circuit(str(theirs))
        yield theirs, mine, theirs.detector_error_model(decompose_errors=decompose), mine.detector_error_model(decompose_errors=decompose)


def same_sparse(a, b) -> bool:
    return a.shape == b.shape and (a != b).nnz == 0


@pytest.mark.parametrize("decompose", [False, True])
def test_check_matrices(decompose):
    from ldpc.ckt_noise import detector_error_model_to_check_matrices as theirs_fn

    for _, _, dt, dm in models(decompose):
        assert str(dt) == str(dm)
        for hyper in (True, False):
            try:
                a = theirs_fn(dt, allow_undecomposed_hyperedges=hyper)
            except ValueError:
                with pytest.raises(ValueError):
                    detector_error_model_to_check_matrices(dm, allow_undecomposed_hyperedges=hyper)
                continue
            b = detector_error_model_to_check_matrices(dm, allow_undecomposed_hyperedges=hyper)
            assert same_sparse(a.check_matrix, b.check_matrix) and same_sparse(a.observables_matrix, b.observables_matrix)
            assert same_sparse(a.edge_check_matrix, b.edge_check_matrix) and same_sparse(a.edge_observables_matrix, b.edge_observables_matrix)
            assert same_sparse(a.hyperedge_to_edge_matrix, b.hyperedge_to_edge_matrix)
            # This package's error models print Stim's text, but a merged probability can differ
            # from Stim's double in the last bit.
            assert np.allclose(a.priors, b.priors, rtol=1e-12, atol=0)


def test_overlapping_window_decoders():
    from ldpc.ckt_noise import BpOsdOverlappingWindowDecoder as TheirsOsd, LsdOverlappingWindowDecoder as TheirsLsd

    circuit = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    dt = circuit.detector_error_model()
    dm = sq.DetectorErrorModel(str(dt))
    shots, _ = circuit.compile_detector_sampler(seed=3).sample(200, separate_observables=True)
    kw = dict(decodings=2, window=4, commit=2, num_checks=4, bp_method="ms", ms_scaling_factor=0.625, max_iter=20)
    for theirs_cls, mine_cls, extra in ((TheirsOsd, BpOsdOverlappingWindowDecoder, dict(osd_method="osd_cs", osd_order=2)), (TheirsLsd, LsdOverlappingWindowDecoder, dict(lsd_method="lsd_cs", lsd_order=2))):
        theirs, mine = theirs_cls(dt, **kw, **extra), mine_cls(dm, **kw, **extra)
        # (Both take uint8 shots, and change them in place.)
        a, b = theirs.decode_batch(shots.astype(np.uint8)), mine.decode_batch(shots.astype(np.uint8))
        agree = np.mean(np.all(a == b, axis=1))
        # BP+LSD's merges of many clusters at once are the only difference allowed.
        assert agree == 1.0 or (theirs_cls is TheirsLsd and agree > 0.97)
        assert np.array_equal(theirs.decode(shots[0].astype(np.uint8)), mine.decode(shots[0].astype(np.uint8)))


def test_sinter_decoders(tmp_path: pathlib.Path):
    import ldpc.sinter_decoders as theirs

    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.005, before_measure_flip_probability=0.005)
    dem = circuit.detector_error_model()
    dem_path = tmp_path / "model.dem"
    dem.to_file(dem_path)
    dets, _ = circuit.compile_detector_sampler(seed=5).sample(300, separate_observables=True, bit_packed=True)
    dets_path = tmp_path / "dets.b8"
    dets_path.write_bytes(dets.tobytes())
    pairs = [(theirs.SinterBpOsdDecoder(osd_method="osd_cs", osd_order=3), SinterBpOsdDecoder(osd_method="osd_cs", osd_order=3)), (theirs.SinterLsdDecoder(lsd_order=2), SinterLsdDecoder(lsd_order=2)), (theirs.SinterBeliefFindDecoder(), SinterBeliefFindDecoder())]
    for t, m in pairs:
        out = []
        for k, dec in enumerate((t, m)):
            path = tmp_path / f"pred{k}.b8"
            dec.decode_via_files(num_shots=300, num_dets=dem.num_detectors, num_obs=dem.num_observables, dem_path=dem_path, dets_b8_in_path=dets_path, obs_predictions_b8_out_path=path, tmp_dir=tmp_path)
            out.append(path.read_bytes())
        agree = np.mean(np.frombuffer(out[0], dtype=np.uint8) == np.frombuffer(out[1], dtype=np.uint8))
        assert agree == 1.0 or (isinstance(m, SinterLsdDecoder) and agree > 0.97), type(m).__name__
    # (Neither implements sinter's compile_decoder_for_dem.)
    for dec in (theirs.SinterBpOsdDecoder(), SinterBpOsdDecoder()):
        with pytest.raises(NotImplementedError):
            dec.compile_decoder_for_dem(dem=dem)
