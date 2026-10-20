"""stabilizer_qec.ldpc against ldpc 2.4.1, shot for shot, where 2.1 closed the gaps: OSD's
candidate weights summed in ldpc's order and its columns sorted by the C library's qsort, LSD's
method and order set after construction, received vectors through BP+OSD, BP+LSD and
BeliefFind, union-find with LLRs, and the exceptions ldpc raises, type and message. Each test
but the recorded one is skipped where ldpc is not installed."""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")

import stabilizer_qec.ldpc as S  # noqa: E402
import stabilizer_qec.ldpc.lsd_decoder  # noqa: E402

DATA = pathlib.Path(__file__).parent / "data"


def real_ldpc():
    return pytest.importorskip("ldpc", minversion="2.4")


def bits(text: str) -> np.ndarray:
    return np.array([int(c) for c in text], dtype=np.uint8)


def random_ldpc(m: int, n: int, column_weight: int, rng) -> np.ndarray:
    H = np.zeros((m, n), dtype=np.uint8)
    for j in range(n):
        H[rng.choice(m, size=column_weight, replace=False), j] = 1
    return H


def ring(d: int) -> np.ndarray:
    return ((np.eye(d, dtype=np.uint8) + np.roll(np.eye(d, dtype=np.uint8), 1, axis=1)) % 2).astype(np.uint8)


def hamming(r: int) -> np.ndarray:
    n = 2**r - 1
    return np.array([[(c + 1) >> (r - 1 - j) & 1 for c in range(n)] for j in range(r)], dtype=np.uint8)


def hgp(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    m1, n1 = h1.shape
    m2, n2 = h2.shape
    return (np.hstack([np.kron(h1, np.eye(n2, dtype=np.uint8)), np.kron(np.eye(m1, dtype=np.uint8), h2.T)]) % 2).astype(np.uint8)


# -- OSD ------------------------------------------------------------------------------------


def test_osd_keeps_ldpcs_candidate_among_equal_weights_recorded():
    """ldpc 2.4.1's answer, recorded, on a shot where two candidates weigh the same."""
    e = json.loads((DATA / "ldpc_osd_sum_order.json").read_text())
    H = np.array([[int(c) for c in row] for row in e["H"]], dtype=np.uint8)
    d = S.BpOsdDecoder(H, **e["kw"])
    out = d.decode(bits(e["syndrome"]))
    assert np.array_equal(out, bits(e["ldpc_decoding"]))
    assert not np.array_equal(out, bits(e["other_candidate"]))


@pytest.mark.parametrize("method,order", [("osd_e", 3), ("osd_cs", 3), ("osd_cs", 6)])
def test_osd_equals_ldpc_with_uniform_priors(method, order):
    """Uniform priors make equal candidate weights and equal posteriors common: both the
    weights' summation order and the column sort's order among equals decide the answer."""
    ldpc = real_ldpc()
    rng = np.random.default_rng(5)
    for _ in range(6):
        H = random_ldpc(30, 60, 3, rng)
        for p in (0.01, 0.05):
            kw = dict(error_rate=p, bp_method=str(rng.choice(["ps", "ms"])), max_iter=int(rng.choice([1, 2, 3])), osd_method=method, osd_order=order)
            ours, theirs = S.BpOsdDecoder(H, **kw), ldpc.BpOsdDecoder(H, **kw)
            for _ in range(40):
                s = (H @ (rng.random(60) < 0.06) % 2).astype(np.uint8)
                a, b = ours.decode(s), theirs.decode(s)
                assert np.array_equal(a, b), kw
                assert np.array_equal(ours.osd0_decoding, theirs.osd0_decoding), kw


# -- LSD ------------------------------------------------------------------------------------

SETTINGS = [("lsd_0", "lsd_cs", 3), ("lsd_0", "lsd_e", 2), ("lsd_cs", "osd_off", 3), ("lsd_e", "lsd_off", 2), ("lsd_cs", "lsd_cs", 5), ("lsd_cs", "lsd_0", 0), ("lsd_e", "osd_cs", 0)]


@pytest.mark.parametrize("built,method,order", SETTINGS)
def test_lsd_method_and_order_set_after_construction(built, method, order):
    """Set, they decode as ldpc's set the same way (OSD_OFF only reachable so), and as a
    decoder built with them."""
    ldpc = real_ldpc()
    import ldpc.lsd_decoder

    rng = np.random.default_rng(3)
    for H in (hgp(hamming(3), hamming(3)), hgp(ring(5), ring(5))):
        n = H.shape[1]
        pair = []
        for M, L in ((S, stabilizer_qec.ldpc.lsd_decoder), (ldpc, ldpc.lsd_decoder)):
            d = M.BpLsdDecoder(H, error_rate=0.05, max_iter=2, bp_method="ms", lsd_method=built, lsd_order=0 if built == "lsd_0" else 1)
            d.lsd_method, d.lsd_order = method, order
            d.set_do_stats(True)
            l = L.LsdDecoder(H, lsd_method=built, lsd_order=0 if built == "lsd_0" else 1)
            l.lsd_method, l.lsd_order = method, order
            pair.append((d, l))
        (a, la), (b, lb) = pair
        assert (a.lsd_method, a.lsd_order, la.lsd_method, la.lsd_order) == (b.lsd_method, b.lsd_order, lb.lsd_method, lb.lsd_order)
        fresh = None
        if method in ("lsd_0", "lsd_e", "lsd_cs"):
            fresh = S.BpLsdDecoder(H, error_rate=0.05, max_iter=2, bp_method="ms", lsd_method=method, lsd_order=order)
        for _ in range(120):
            s = (H @ (rng.random(n) < 0.1) % 2).astype(np.uint8)
            x, y = a.decode(s), b.decode(s)
            if not a.last_tied:
                assert np.array_equal(x, y), (built, method, order)
            assert (a.statistics["lsd_method"], a.statistics["lsd_order"]) == (b.statistics["lsd_method"], b.statistics["lsd_order"])
            if fresh is not None:
                assert np.array_equal(fresh.decode(s), x)
            w = rng.uniform(0.5, 4, n)
            x, y = la.decode(s, w), lb.decode(s, w)
            if not la.last_tied:
                assert np.array_equal(x, y), (built, method, order)


# -- received vectors -----------------------------------------------------------------------


@pytest.mark.parametrize("cls", ["BpOsdDecoder", "BpLsdDecoder", "BeliefFindDecoder"])
def test_received_vectors_decode_as_ldpcs(cls):
    """Codewords of the ring code with errors, where BP often fails: the decoding and BP's
    run (iterations, convergence) are ldpc's."""
    ldpc = real_ldpc()
    rng = np.random.default_rng(11)
    H = ring(8)
    kw = dict(error_rate=0.1, max_iter=8, bp_method="ps")
    if cls == "BeliefFindDecoder":
        kw["uf_method"] = "inversion"
    ours, theirs = getattr(S, cls)(H, **kw), getattr(ldpc, cls)(H, **kw)
    ours.input_vector_type = theirs.input_vector_type = "received_vector"
    failed = 0
    for _ in range(300):
        codeword = np.zeros(8, np.uint8) if rng.random() < 0.5 else np.ones(8, np.uint8)
        received = codeword ^ (rng.random(8) < 0.2).astype(np.uint8)
        a, b = ours.decode(received), theirs.decode(received)
        assert np.array_equal(a, b), received
        assert (ours.converge, ours.iter) == (theirs.converge, theirs.iter), received
        failed += not theirs.converge
    assert failed > 0


# -- union-find -----------------------------------------------------------------------------


def test_union_find_with_llrs_equals_ldpcs_algorithms():
    """ldpc's uf_method="peeling" is inversion (its C bool): this one's "inversion" equals it on
    every shot; ldpc's peeling is uf_method="", which this one's "peeling" equals but where it
    flags last_stuck (where ldpc's can hang, so those shots are not given to ldpc)."""
    ldpc = real_ldpc()
    rng = np.random.default_rng(21)
    H = hgp(ring(5), ring(5))
    n = H.shape[1]
    inversion, peeling = ldpc.UnionFindDecoder(H, uf_method="peeling"), ldpc.UnionFindDecoder(H, uf_method="")
    oi, op = S.UnionFindDecoder(H, uf_method="inversion"), S.UnionFindDecoder(H, uf_method="peeling")
    p = rng.uniform(0.005, 0.08, n)
    for kind in ("uniform", "integer", "prior", None):
        for bps in (0, 1, 2):
            for _ in range(60):
                s = (H @ (rng.random(n) < 0.05) % 2).astype(np.uint8)
                llrs = {"uniform": lambda: rng.uniform(0.5, 4, n), "integer": lambda: rng.integers(0, 4, n).astype(float), "prior": lambda: np.log((1 - p) / p), None: lambda: None}[kind]()
                y = oi.decode(s, llrs=llrs, bits_per_step=bps)
                if not oi.last_tied:
                    assert np.array_equal(inversion.decode(s, llrs=llrs, bits_per_step=bps), y), (kind, bps)
                y = op.decode(s, llrs=llrs, bits_per_step=bps)
                if not (op.last_stuck or op.last_tied):
                    assert np.array_equal(peeling.decode(s, llrs=llrs, bits_per_step=bps), y), (kind, bps)


# -- exceptions -----------------------------------------------------------------------------


def outcome(f) -> str:
    try:
        f()
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def set_(obj, name, value):
    setattr(obj, name, value)


def v2_cases(M, L):
    H = hamming(3)
    yield "bp_method", lambda: M.BpDecoder(H, error_rate=0.1, bp_method="bogus")
    yield "schedule", lambda: M.BpDecoder(H, error_rate=0.1, schedule="bogus")
    yield "input type", lambda: M.BpDecoder(H, error_rate=0.1, input_vector_type="bad")
    yield "square, input type not given", lambda: M.BpDecoder(ring(5), error_rate=0.1)
    yield "no channel", lambda: M.BpDecoder(H)
    yield "matrix type", lambda: M.BpDecoder("abc", error_rate=0.1)
    yield "omp -1", lambda: M.BpDecoder(H, error_rate=0.1, omp_thread_count=-1)
    yield "seed -3", lambda: M.BpDecoder(H, error_rate=0.1, random_schedule_seed=-3)
    yield "ms 'x'", lambda: M.BpDecoder(H, error_rate=0.1, ms_scaling_factor="x")
    yield "ms None", lambda: M.BpDecoder(H, error_rate=0.1, ms_scaling_factor=None)
    yield "bp_method None", lambda: M.BpDecoder(H, error_rate=0.1, bp_method=None)
    yield "input type None", lambda: M.BpDecoder(H, error_rate=0.1, input_vector_type=None)
    yield "osd method", lambda: M.BpOsdDecoder(H, error_rate=0.1, osd_method="bogus")
    yield "osd method set", lambda: set_(M.BpOsdDecoder(H, error_rate=0.1), "osd_method", "bogus")
    yield "osd_order '2'", lambda: M.BpOsdDecoder(H, error_rate=0.1, osd_method="osd_cs", osd_order="2")
    yield "osd_order None", lambda: M.BpOsdDecoder(H, error_rate=0.1, osd_order=None)
    yield "osd_order set 2.0", lambda: set_(M.BpOsdDecoder(H, error_rate=0.1, osd_method="osd_cs", osd_order=2), "osd_order", 2.0)
    yield "osd_order set '2'", lambda: set_(M.BpOsdDecoder(H, error_rate=0.1, osd_method="osd_cs", osd_order=2), "osd_order", "2")
    yield "osd input type None", lambda: M.BpOsdDecoder(H, error_rate=0.1, input_vector_type=None)
    yield "lsd method 2", lambda: M.BpLsdDecoder(H, error_rate=0.1, lsd_method=2)
    yield "lsd method osde", lambda: M.BpLsdDecoder(H, error_rate=0.1, lsd_method="osde", lsd_order=2)
    yield "lsd method bogus", lambda: M.BpLsdDecoder(H, error_rate=0.1, lsd_method="bogus")
    yield "lsd order -1", lambda: M.BpLsdDecoder(H, error_rate=0.1, lsd_order=-1)
    yield "lsd method set", lambda: set_(M.BpLsdDecoder(H, error_rate=0.1), "lsd_method", "bogus")
    yield "lsd method set osd_e", lambda: set_(M.BpLsdDecoder(H, error_rate=0.1), "lsd_method", "osde")
    yield "lsd order set -1", lambda: set_(M.BpLsdDecoder(H, error_rate=0.1, lsd_method="lsd_cs"), "lsd_order", -1)
    yield "lsd order set under LSD_0", lambda: set_(M.BpLsdDecoder(H, error_rate=0.1), "lsd_order", 2)
    yield "lsd order set 2.0", lambda: set_(M.BpLsdDecoder(H, error_rate=0.1, lsd_method="lsd_cs"), "lsd_order", 2.0)
    yield "lsd order set True", lambda: set_(M.BpLsdDecoder(H, error_rate=0.1, lsd_method="lsd_cs"), "lsd_order", True)
    yield "LsdDecoder method 2", lambda: L.LsdDecoder(H, lsd_method=2)
    yield "LsdDecoder matrix", lambda: L.LsdDecoder("abc")
    yield "LsdDecoder method set", lambda: set_(L.LsdDecoder(H), "lsd_method", "bogus")
    yield "UF matrix", lambda: M.UnionFindDecoder("abc", uf_method="")
    yield "BeliefFind matrix", lambda: M.BeliefFindDecoder("abc", error_rate=0.1)
    yield "BeliefFind method", lambda: M.BeliefFindDecoder(H, error_rate=0.1, uf_method="bogus")
    yield "BeliefFind method 1", lambda: M.BeliefFindDecoder(H, error_rate=0.1, uf_method=1)


def test_v2_exceptions_are_ldpcs():
    ldpc = real_ldpc()
    import ldpc.lsd_decoder

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (label, ours), (_, theirs) in zip(v2_cases(S, stabilizer_qec.ldpc.lsd_decoder), v2_cases(ldpc, ldpc.lsd_decoder)):
            assert outcome(ours) == outcome(theirs), label


V1_CASES = [
    ((), dict(error_rate=0.1, max_iter=2.0)),
    ((), dict(error_rate=0.1, max_iter=np.int64(3))),
    ((), dict(error_rate=0.1, bp_method=0)),
    ((), dict(error_rate=0.1, schedule=0)),
    ((), dict(error_rate=0.1, omp_thread_count=1.0)),
    ((), dict(error_rate=0.1, serial_schedule_order=(0, 1, 2, 3, 4, 5, 6))),
    ((), dict(error_rate=0.1, input_vector_type=0)),
    ((), dict(error_rate=np.float64(0.1))),
    ((), dict(error_rate=1)),
    ((), dict(error_rate=True)),
    ((), dict(error_rate="0.1")),
    ((), dict(error_rate=0.1, bp_method=b"ps")),
    ((0.05,), {}),
    ((0.05, 4), dict(bp_method="ms")),
    ((), dict(error_rate=0.05, foo=1)),
    ((), dict(foo=1)),
    ((), {}),
]


@pytest.mark.parametrize("which", ["bp_decoder", "bposd_decoder"])
def test_v1_exceptions_are_ldpcs(which):
    """ldpc's v1 classes subclass the Cython ones in Python: Cython's exact argument types and
    its one positional argument, then the v1 __init__'s own checks."""
    real_ldpc()
    import importlib

    theirs = getattr(importlib.import_module(f"ldpc._legacy_ldpc_v1._legacy_{which}"), which)
    ours = getattr(importlib.import_module(f"stabilizer_qec.ldpc._legacy_ldpc_v1._legacy_{which}"), which)
    H = hamming(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for args, kwargs in V1_CASES:
            assert outcome(lambda: ours(H, *args, **kwargs)) == outcome(lambda: theirs(H, *args, **kwargs)), (args, kwargs)
        kw = dict(parity_check_matrix=H, error_rate=0.05)
        assert outcome(lambda: ours(**kw)) == outcome(lambda: theirs(**kw))
        if which == "bposd_decoder":
            for kwargs in (dict(error_rate=0.1, osd_order=2.0, osd_method="osd_cs"), dict(error_rate=0.1, osd_order="2", osd_method="osd_cs"), dict(error_rate=0.1, osd_order=None), dict(error_rate=0.1, osd_method=None)):
                assert outcome(lambda: ours(H, **kwargs)) == outcome(lambda: theirs(H, **kwargs)), kwargs
