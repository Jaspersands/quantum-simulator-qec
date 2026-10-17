"""The lane sampler (`device="cpu-lanes"`: 32 shots a word, integer draws, the program a GPU
runs): the same bytes on any thread count and batching, the same distribution as the word
sampler, Stim's measurement semantics, and the device argument's errors. `device="gpu"` is
tested here too when the build has it and a device answers."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq


def _gpu_available() -> bool:
    try:
        sq.Circuit("R 0\nM 0").compile_detector_sampler(seed=1, device="gpu")
        return True
    except ValueError:
        return False


needs_gpu = pytest.mark.skipif(not _gpu_available(), reason="no gpu build or device")
DEVICES = ["cpu-lanes"] + (["gpu"] if _gpu_available() else [])


@pytest.fixture(scope="module")
def memory():
    return sq.memory_circuit(distance=5, rounds=5, p=0.005)


@pytest.mark.parametrize("device", DEVICES)
def test_same_seed_same_bytes_any_threads_and_batching(memory, device):
    one = memory.compile_detector_sampler(seed=11, device=device)
    assert one.device == device
    d1, o1 = one.sample(1000, separate_observables=True, bit_packed=True, threads=1)
    d8, o8 = memory.compile_detector_sampler(seed=11, device=device).sample(1000, separate_observables=True, bit_packed=True, threads=8)
    assert np.array_equal(d1, d8) and np.array_equal(o1, o8)
    split = memory.compile_detector_sampler(seed=11, device=device)
    a = split.sample(320, separate_observables=True, bit_packed=True)
    b = split.sample(680, separate_observables=True, bit_packed=True)
    assert np.array_equal(np.concatenate([a[0], b[0]]), d1) and np.array_equal(np.concatenate([a[1], b[1]]), o1)
    # A call that ends mid-word drops the word's other lanes: the next call starts a new word.
    c = memory.compile_detector_sampler(seed=11, device=device)
    first = c.sample(10, bit_packed=True)
    whole = memory.compile_detector_sampler(seed=11, device=device).sample(52, bit_packed=True)
    assert np.array_equal(first, whole[:10]) and np.array_equal(c.sample(20, bit_packed=True), whole[32:52])


@pytest.mark.parametrize("device", DEVICES)
def test_lanes_sample_the_word_samplers_distribution(memory, device):
    n = 40_000
    dets_w, obs_w = memory.compile_detector_sampler(seed=3).sample(n, separate_observables=True)
    dets_l, obs_l = memory.compile_detector_sampler(seed=3, device=device).sample(n, separate_observables=True)
    assert dets_l.shape == dets_w.shape and obs_l.shape == obs_w.shape
    rate_w, rate_l = dets_w.mean(), dets_l.mean()
    assert abs(rate_l - rate_w) < 0.05 * rate_w, (rate_l, rate_w)
    # Per detector, the two rates agree within 6 standard errors (every detector, so a loose bound).
    per_w, per_l = dets_w.mean(axis=0), dets_l.mean(axis=0)
    se = np.sqrt(per_w * (1 - per_w) / n * 2) + 1e-9
    assert (np.abs(per_w - per_l) < 6 * se + 0.002).all()
    m = sq.Matching(memory.detector_error_model(decompose_errors=True))
    fail_w = (m.decode_batch(dets_w) != obs_w).any(axis=1).mean()
    fail_l = (m.decode_batch(dets_l) != obs_l).any(axis=1).mean()
    assert abs(fail_w - fail_l) < 4 * np.sqrt(fail_w * (1 - fail_w) / n * 2) + 0.002


@pytest.mark.parametrize("device", DEVICES)
def test_every_channel_runs_on_the_lanes(device):
    text = """R 0 1 2 3
H 0
CX 0 1
CZ 1 2
S 2
X_ERROR(0.1) 0
Y_ERROR(0.1) 1
Z_ERROR(0.1) 2
DEPOLARIZE1(0.2) 0 1
DEPOLARIZE2(0.2) 0 1
PAULI_CHANNEL_1(0.05, 0.1, 0.15) 2
PAULI_CHANNEL_2(0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03) 2 3
E(0.3) X0 Z1
ELSE_CORRELATED_ERROR(0.5) Y2
HERALDED_ERASE(0.2) 3
MPAD 0 1
M(0.05) 0 1
MR 2
MX 3
DETECTOR rec[-1]
DETECTOR rec[-2] rec[-3]
REPEAT 4 {
    X_ERROR(0.1) 0
    CX 1 0
    M 0
    DETECTOR rec[-1] rec[-2]
    CX rec[-1] 1
}
OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]
OBSERVABLE_INCLUDE(1) Z0 X1"""
    c = sq.Circuit(text)
    n = 20_000
    dets_w, obs_w = c.compile_detector_sampler(seed=5).sample(n, separate_observables=True)
    dets_l, obs_l = c.compile_detector_sampler(seed=5, device=device).sample(n, separate_observables=True)
    assert dets_l.shape == (n, c.num_detectors) and obs_l.shape == (n, c.num_observables)
    se = np.sqrt(0.25 / n * 2)
    assert (np.abs(dets_w.mean(axis=0) - dets_l.mean(axis=0)) < 6 * se).all()
    assert (np.abs(obs_w.mean(axis=0) - obs_l.mean(axis=0)) < 6 * se).all()


@pytest.mark.parametrize("device", DEVICES)
def test_measurement_sampler_on_the_lanes(memory, device):
    s = memory.compile_sampler(seed=2, device=device)
    assert s.device == device
    rows = s.sample(100)
    assert rows.shape == (100, memory.num_measurements)
    # Noiseless and with every outcome determined (a repetition code's Z checks on |0…0⟩):
    # every shot is the reference sample; with skip_reference_sample, the flips alone, none.
    quiet = sq.Circuit.generated("repetition_code:memory", distance=5, rounds=4)
    ref = quiet.reference_sample()
    assert np.array_equal(quiet.compile_sampler(seed=1, device=device).sample(7), np.tile(ref, (7, 1)))
    assert not quiet.compile_sampler(seed=1, device=device, skip_reference_sample=True).sample(7).any()
    # With a flip before each measurement, the flips show on top of the reference.
    noisy = sq.Circuit.generated("repetition_code:memory", distance=5, rounds=4, before_measure_flip_probability=0.5)
    flips = noisy.compile_sampler(seed=1, device=device, skip_reference_sample=True).sample(2000)
    assert 0.45 < flips.mean() < 0.55
    assert np.array_equal(noisy.compile_sampler(seed=1, device=device).sample(2000), flips ^ ref)
    # Its detection events through m2d equal the detector sampler's rates.
    conv = memory.compile_m2d_converter()
    dets = conv.convert(measurements=memory.compile_sampler(seed=9, device=device).sample(20_000), separate_observables=False)
    rate = dets[:, : memory.num_detectors].mean()
    rate_w = memory.compile_detector_sampler(seed=9).sample(20_000).mean()
    assert abs(rate - rate_w) < 0.05 * rate_w


def test_device_errors(memory):
    with pytest.raises(ValueError, match="device must be"):
        memory.compile_detector_sampler(seed=1, device="tpu")
    with pytest.raises(ValueError, match="device must be"):
        memory.compile_sampler(seed=1, device="tpu")
    t = sq.Circuit("R 0\nI[T] 0\nM 0")
    with pytest.raises(ValueError, match="non-Clifford"):
        t.compile_detector_sampler(seed=1, device="cpu-lanes")
    with pytest.raises(ValueError, match="non-Clifford"):
        t.compile_sampler(seed=1, device="cpu-lanes")
    if not _gpu_available():
        with pytest.raises(ValueError):
            memory.compile_detector_sampler(seed=1, device="gpu")
    assert memory.compile_detector_sampler(seed=1).device == "cpu" and memory.compile_sampler(seed=1).device == "cpu"


@needs_gpu
def test_gpu_long_runs_are_paced_past_the_watchdog():
    """A run of seconds in one dispatch trips macOS's GPU watchdog and the rest comes back zero;
    the dispatches are paced to a fraction of a second, so a large run equals the CPU lanes."""
    c = sq.Circuit.generated("surface_code:rotated_memory_z", distance=11, rounds=11, after_clifford_depolarization=0.001, before_measure_flip_probability=0.001)
    shots = 400_000
    cpu = c.compile_detector_sampler(seed=3, device="cpu-lanes").sample(shots, bit_packed=True)
    gpu = c.compile_detector_sampler(seed=3, device="gpu").sample(shots, bit_packed=True)
    assert np.array_equal(cpu, gpu)


@needs_gpu
def test_gpu_equals_the_cpu_lanes_bit_for_bit(memory):
    for circuit in (memory, sq.Circuit.generated("color_code:memory_xyz", distance=3, rounds=2, after_clifford_depolarization=0.01), sq.surgery.cnot(3, merged=3, p=0.003)):
        for shots in (1, 31, 32, 33, 1000):
            cpu = circuit.compile_detector_sampler(seed=4, device="cpu-lanes").sample(shots, separate_observables=True, bit_packed=True)
            gpu = circuit.compile_detector_sampler(seed=4, device="gpu").sample(shots, separate_observables=True, bit_packed=True)
            assert np.array_equal(cpu[0], gpu[0]) and np.array_equal(cpu[1], gpu[1]), (circuit.num_qubits, shots)
        assert np.array_equal(circuit.compile_sampler(seed=4, device="cpu-lanes").sample(100), circuit.compile_sampler(seed=4, device="gpu").sample(100))
