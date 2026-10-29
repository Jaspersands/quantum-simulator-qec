"""Objects shared between threads decode as they do alone. Under free-threaded Python (3.14t)
the threads run at once; elsewhere this checks the GIL is released and retaken cleanly."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import stabilizer_qec as sq


def test_the_module_keeps_the_gil_off():
    is_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_enabled is not None and sysconfig_free_threaded():
        assert not is_enabled(), "importing stabilizer_qec turned the GIL back on"


def sysconfig_free_threaded() -> bool:
    import sysconfig

    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def test_decoders_shared_between_threads():
    c = sq.memory_circuit(distance=5, rounds=5, p=0.004)
    dem = c.detector_error_model(decompose_errors=True)
    dets = c.compile_detector_sampler(seed=1).sample(4096)
    chunks = np.array_split(dets, 16)
    correlated = sq.Matching(dem, enable_correlations=True)
    decoders = [sq.Matching(dem).decode_batch, lambda x: correlated.decode_batch(x, enable_correlations=True), sq.BeliefMatching(dem, max_bp_iters=5).decode_batch]
    for decode in decoders:
        alone = [decode(x) for x in chunks]
        with ThreadPoolExecutor(8) as pool:
            together = list(pool.map(decode, chunks))
        assert all(np.array_equal(a, b) for a, b in zip(alone, together))


def test_circuits_models_and_samplers_from_many_threads():
    def work(seed):
        c = sq.memory_circuit(distance=3, rounds=3, p=0.01)
        dem = c.detector_error_model(decompose_errors=True)
        dets, obs = c.compile_detector_sampler(seed=seed).sample(512, separate_observables=True)
        return int((sq.Matching(dem).decode_batch(dets) != obs).any(axis=1).sum())

    alone = [work(s) for s in range(12)]
    with ThreadPoolExecutor(6) as pool:
        assert list(pool.map(work, range(12))) == alone


def test_one_sampler_used_by_many_threads_stays_consistent():
    # Shots drawn from one sampler by several threads at once: each call gets whole shots, and
    # together they are the stream's shots in some order.
    c = sq.memory_circuit(distance=3, rounds=3, p=0.01)
    s = c.compile_detector_sampler(seed=3)
    with ThreadPoolExecutor(4) as pool:
        parts = list(pool.map(lambda _: s.sample(64, bit_packed=True), range(8)))
    got = sorted(map(bytes, (row for part in parts for row in part)))
    want = sorted(map(bytes, c.compile_detector_sampler(seed=3).sample(512, bit_packed=True)))
    assert got == want


def _hypergraph_product_of_repetition_codes(n: int) -> np.ndarray:
    rep = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        rep[i, i] = rep[i, i + 1] = 1
    m1, n1 = rep.shape
    return np.hstack([np.kron(rep, np.eye(n1, dtype=np.uint8)), np.kron(np.eye(m1, dtype=np.uint8), rep.T)]) % 2


def _shared_by_threads(decode, syndromes, threads=8):
    # Each thread decodes every syndrome, in its own order, on the one decoder; the switch
    # interval small, so a decode is interrupted where another thread's can come in.
    alone = [np.asarray(decode(s)).copy() for s in syndromes]
    wrong, errors = [0] * threads, []

    def run(t):
        try:
            for i in np.random.default_rng(t).permutation(len(syndromes)):
                wrong[t] += not np.array_equal(decode(syndromes[i]), alone[i])
        except Exception as e:  # pragma: no cover - reported below
            errors.append(repr(e))

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(run, range(threads)))
    finally:
        sys.setswitchinterval(interval)
    return wrong, errors


def test_check_matrix_decoders_shared_between_threads():
    # ldpc's family keeps each decode's results on the decoder (converge, log_prob_ratios, ...):
    # a decode answers with its own, whatever another thread's decode leaves there meanwhile.
    h = _hypergraph_product_of_repetition_codes(5)
    m, n = h.shape
    rng = np.random.default_rng(1)
    syndromes = [(h @ (rng.random(n) < 0.08)) % 2 for _ in range(300)]
    weights = list(np.full(n, np.log(0.95 / 0.05)))
    decoders = {
        "bp": sq.BpDecoder(h, error_rate=0.05, max_iter=5, bp_method="ms").decode,
        "belief_find": sq.BeliefFindDecoder(h, error_rate=0.05, max_iter=3, bp_method="ms").decode,
        "bp_osd": sq.BpOsdDecoder(h, error_rate=0.05, max_iter=3, bp_method="ms", osd_method="osd_cs", osd_order=2).decode,
        "bp_lsd": sq.BpLsdDecoder(h, error_rate=0.05, max_iter=3, bp_method="ms", lsd_order=2).decode,
        "bp_flip": sq.BpFlipDecoder(h, error_rate=0.05, max_iter=3, bp_method="ms", flip_iterations=3).decode,
        "union_find": sq.UnionFindDecoder(h, uf_method="inversion").decode,
        "lsd": (lambda d: lambda s: d.decode(s, weights))(sq.LsdDecoder(h)),
        "soft_info_bp": (lambda d: lambda s: d.decode(np.where(s, -1.0, 1.0)))(sq.SoftInfoBpDecoder(h, error_rate=0.05, max_iter=3)),
    }
    for name, decode in decoders.items():
        wrong, errors = _shared_by_threads(decode, syndromes)
        assert (wrong, errors) == ([0] * len(wrong), []), name
