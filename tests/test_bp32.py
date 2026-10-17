"""Single-precision min-sum BP (`BpDecoder(device="cpu-f32")`, the program the GPU runs) against
the f64 decoder, and the GPU against the CPU bit for bit when the build has it."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import BpDecoder


def _gpu() -> bool:
    return sq.gpu_device() is not None


needs_gpu = pytest.mark.skipif(not _gpu(), reason="no gpu build or device")


@pytest.fixture(scope="module")
def problem():
    """A d = 5 memory's undecomposed model as a check matrix, with 4,000 sampled syndromes."""
    from stabilizer_qec.beliefmatching import detector_error_model_to_check_matrices

    c = sq.memory_circuit(distance=5, rounds=5, p=0.004)
    m = detector_error_model_to_check_matrices(c.detector_error_model(decompose_errors=True), allow_undecomposed_hyperedges=True)
    H = m.check_matrix
    H = np.asarray(H.todense() if hasattr(H, "todense") else H, dtype=np.uint8)
    dets, obs = c.compile_detector_sampler(seed=5).sample(4000, separate_observables=True)
    return H, np.asarray(m.priors, dtype=float), dets.astype(np.uint8)


def test_f32_min_sum_agrees_with_f64(problem):
    H, priors, dets = problem
    f64 = BpDecoder(H, error_channel=priors, max_iter=20, bp_method="minimum_sum", ms_scaling_factor=0.625)
    f32 = BpDecoder(H, error_channel=priors, max_iter=20, bp_method="minimum_sum", ms_scaling_factor=0.625, device="cpu-f32")
    assert f32.device == "cpu-f32"
    want = np.stack([f64.decode(s) for s in dets[:1500]])
    got = f32.decode_batch(dets[:1500])
    same = (want == got).all(axis=1).mean()
    assert same > 0.98, same
    one = f32.decode(dets[7])
    assert np.array_equal(one, got[7]) and isinstance(f32.converge, bool)
    # The adaptive scale (0) and a batch across threads give the same bytes.
    adaptive = BpDecoder(H, error_channel=priors, max_iter=12, bp_method="minimum_sum", ms_scaling_factor=0, device="cpu-f32")
    assert np.array_equal(adaptive.decode_batch(dets[:500], threads=1), adaptive.decode_batch(dets[:500], threads=8))
    # A zero syndrome decodes to nothing.
    assert not f32.decode_batch(np.zeros((3, H.shape[0]), dtype=np.uint8)).any()


def test_f32_device_errors(problem):
    H, priors, dets = problem
    with pytest.raises(ValueError, match="minimum_sum"):
        BpDecoder(H, error_channel=priors, max_iter=10, bp_method="product_sum", device="cpu-f32")
    with pytest.raises(ValueError, match="parallel"):
        BpDecoder(H, error_channel=priors, max_iter=10, bp_method="minimum_sum", schedule="serial", device="cpu-f32")
    with pytest.raises(ValueError, match="device must be"):
        BpDecoder(H, error_channel=priors, max_iter=10, device="tpu")
    f32 = BpDecoder(H, error_channel=priors, max_iter=10, bp_method="minimum_sum", device="cpu-f32")
    with pytest.raises(ValueError):
        f32.decode_batch(dets[:2, :-1])
    if not _gpu():
        with pytest.raises(ValueError):
            BpDecoder(H, error_channel=priors, max_iter=10, bp_method="minimum_sum", device="gpu").decode_batch(dets[:2])
    cpu = BpDecoder(H, error_channel=priors, max_iter=10, bp_method="minimum_sum")
    assert np.array_equal(cpu.decode_batch(dets[:5]), np.stack([cpu.decode(s) for s in dets[:5]]))


def test_gpu_device_name():
    name = sq.gpu_device()
    assert name is None or (isinstance(name, str) and name)


@needs_gpu
def test_gpu_bp_equals_cpu_f32(problem):
    H, priors, dets = problem
    for scale, iters in ((0.625, 20), (0, 12), (1.0, 5)):
        cpu = BpDecoder(H, error_channel=priors, max_iter=iters, bp_method="minimum_sum", ms_scaling_factor=scale, device="cpu-f32")
        gpu = BpDecoder(H, error_channel=priors, max_iter=iters, bp_method="minimum_sum", ms_scaling_factor=scale, device="gpu")
        a = cpu.decode_batch(dets, _details=True)
        b = gpu.decode_batch(dets, _details=True)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]), scale
    assert np.array_equal(gpu.decode(dets[3]), cpu.decode(dets[3]))
