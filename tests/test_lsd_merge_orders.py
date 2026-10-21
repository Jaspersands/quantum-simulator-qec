"""BP+LSD against ldpc 2.4.1 where ldpc's own answer varies: the merge orders ldpc can take.

ldpc merges the clusters a growth step meets in the order of a `tsl::robin_set<LsdCluster *>`
(src_cpp/lsd.hpp:47, :193), the order of their hashed heap addresses, so one syndrome decoded
again in one process can give another answer. tests/ldpc_lsd_merge/captured.json holds the QUITS
sweep's window decodes (tools/lsd_merge_capture.py) where ldpc's answers over 50 decodes varied:
every one of them, and ours, must be among the corrections `_merge_order_outcomes` finds over
the merge orders ldpc can take (every order that departs from ours at one merge, at least), and
`decode` must still take the order it took before the search existed.
"""

import hashlib
import json
import pathlib
import platform

import numpy as np
import pytest

from stabilizer_qec.ldpc import BpLsdDecoder

# ldpc's answers, and the sweep digest, were recorded on macOS. Elsewhere BP's posteriors are
# the platform's libm's (tanh, log), and LSD's sorts leave equal weights in the order the
# platform's std::sort does (ours reproduces each), so a window need not be the same decode
# there; tests/test_quits_differential.py checks reachability against ldpc itself on every
# platform. Only the tests on the recorded decodes are skipped there; the argument and
# unexplainable-syndrome tests depend on no platform behaviour.
macos_recorded = pytest.mark.skipif(platform.system() != "Darwin", reason="ldpc's answers were recorded on macOS (libc++, Apple's libm)")

CAPTURED = json.loads((pathlib.Path(__file__).parent / "ldpc_lsd_merge" / "captured.json").read_text(encoding="utf-8"))

# Our decodes of _sweep() before the merge-order search existed (2.1.0's).
SWEEP_DIGEST = "d814fce05ab6f2b640a1458f645548d56dc4e2d501a199513a7d8ed0b3fba0e1"


def _matrix(m):
    H = np.zeros((m["rows"], len(m["columns"])), dtype=np.uint8)
    for j, col in enumerate(m["columns"]):
        H[col, j] = 1
    return H


def _dense(indices, n):
    x = np.zeros(n, dtype=np.uint8)
    x[indices] = 1
    return x.tobytes()


@pytest.fixture(scope="module")
def windows():
    """Each captured window: (our answer, the corrections 64 merge orders reach, fewest
    departures from ours first, how far that went, ldpc's answers, our answer as recorded, our
    answer after the search)."""
    matrices = [_matrix(m) for m in CAPTURED["matrices"]]
    out = []
    for w in CAPTURED["windows"]:
        H = matrices[w["matrix"]]
        m, n = H.shape
        dec = BpLsdDecoder(H, **w["kw"])
        s = np.frombuffer(_dense(w["syndrome"], m), dtype=np.uint8)
        ours = np.asarray(dec.decode(s), dtype=np.uint8).tobytes()
        theirs = [_dense(a["correction"], n) for a in w["ldpc"]]
        reachable = dec._merge_order_outcomes(s, max_leaves=64)
        again = np.asarray(dec.decode(s), dtype=np.uint8).tobytes()
        out.append((ours, reachable, dec._merge_search, theirs, _dense(w["ours"], n), again))
    return out


@macos_recorded
def test_ours_and_every_ldpc_answer_reachable(windows):
    assert len(windows) == 15
    for k, (ours, reachable, _, theirs, _, _) in enumerate(windows):
        assert ours in reachable, k
        assert len(theirs) == 2, k
        for a in theirs:
            assert a in reachable, k


@macos_recorded
def test_reachable_set_small(windows):
    for k, (ours, reachable, (paths, departures, complete), theirs, _, _) in enumerate(windows):
        assert 2 <= len(reachable) <= 8, (k, len(reachable))
        # (Every order departing from ours at one merge was walked: at most 31 here.)
        assert departures >= 1 or complete, (k, paths, departures)
        # No answer ldpc did not give (nor with every order walked, where that is under 4,096).
        assert reachable == {ours, *theirs}, k


@macos_recorded
def test_search_stops_at_the_target():
    """Sought, a correction ends the walk at the first order that reaches it: ours (``decode``'s)
    at the first order, and each of ldpc's where the orders before it have not reached it, within
    the 32 orders that depart from ours at one merge."""
    matrices = [_matrix(m) for m in CAPTURED["matrices"]]
    for k, w in enumerate(CAPTURED["windows"]):
        H = matrices[w["matrix"]]
        m, n = H.shape
        dec = BpLsdDecoder(H, **w["kw"])
        s = np.frombuffer(_dense(w["syndrome"], m), dtype=np.uint8)
        ours = np.asarray(dec.decode(s), dtype=np.uint8)
        assert dec._merge_order_outcomes(s, target=ours) == {ours.tobytes()}, k
        assert dec._merge_search == (1, 0, False), k
        for a in w["ldpc"]:
            answer = np.frombuffer(_dense(a["correction"], n), dtype=np.uint8)
            found = dec._merge_order_outcomes(s, target=answer)
            paths = dec._merge_search[0]
            assert answer.tobytes() in found and len(found) <= 2 and paths <= 32, (k, paths)
            assert paths == 1 or answer.tobytes() not in dec._merge_order_outcomes(s, max_leaves=paths - 1), (k, paths)


def _sweep():
    """Our corrections on random matrices for LSD-0 (one bit a step and all), LSD-E and LSD-CS,
    LSD on every shot: 2,071 of its 2,400 decodes make a merge choice ldpc may make otherwise."""
    rng = np.random.default_rng(14)
    configs = [("minimum_sum", "lsd_0", 0, 1), ("product_sum", "lsd_0", 0, 0), ("minimum_sum", "lsd_e", 4, 1), ("product_sum", "lsd_cs", 6, 1)]
    out, tied = [], 0
    for _ in range(6):
        pcm = (rng.random((24, 48)) < 0.12).astype(np.uint8)
        pcm[rng.integers(24, size=48), np.arange(48)] = 1
        channel = rng.uniform(0.01, 0.1, 48)
        for method, lsd, order, per_step in configs:
            dec = BpLsdDecoder(pcm, error_channel=list(channel), max_iter=3, bp_method=method, lsd_method=lsd, lsd_order=order, bits_per_step=per_step, always_run_lsd=True)
            for _ in range(100):
                s = (pcm @ (rng.random(48) < channel)) % 2
                out.append(np.asarray(dec.decode(s), dtype=np.uint8).tobytes())
                tied += dec.last_tied
    return out, tied


@macos_recorded
def test_default_order_unchanged(windows):
    for k, (ours, _, _, _, recorded, again) in enumerate(windows):
        assert ours == recorded == again, k
    out, tied = _sweep()
    assert tied > 2000
    assert hashlib.sha256(b"".join(out)).hexdigest() == SWEEP_DIGEST


def test_merge_order_outcomes_arguments():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    dec = BpLsdDecoder(H, error_rate=0.1, max_iter=2)
    with pytest.raises(ValueError, match="The syndrome must have length 2. Not 3."):
        dec._merge_order_outcomes(np.zeros(3, dtype=np.uint8))
    with pytest.raises(ValueError, match="max_leaves must be at least 1, not 0"):
        dec._merge_order_outcomes(np.zeros(2, dtype=np.uint8), max_leaves=0)
    with pytest.raises(ValueError, match="The target must have length 3. Not 2."):
        dec._merge_order_outcomes(np.zeros(2, dtype=np.uint8), target=np.zeros(2, dtype=np.uint8))
    # BP's answer where it converges: one order, no merge.
    bp = np.asarray(dec.decode(np.array([1, 0])), dtype=np.uint8)
    assert dec._merge_order_outcomes(np.array([1, 0])) == {bp.tobytes()}
    assert dec._merge_search == (1, 0, True)
    assert dec._merge_order_outcomes(np.array([1, 0]), target=1 - bp) == {bp.tobytes()}
    assert dec._merge_search == (1, 0, True)


def test_an_unexplainable_syndrome_raises_as_decode():
    """A syndrome no correction explains raises ``decode``'s ValueError. No merge order explains
    it either, so the walk ends at the first order, whatever choices are left."""
    message = "^no correction explains the syndrome$"
    H = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    s = np.array([0, 0, 1])
    for always_run_lsd in (True, False):
        dec = BpLsdDecoder(H, error_rate=0.1, max_iter=2, always_run_lsd=always_run_lsd)
        with pytest.raises(ValueError, match=message):
            dec.decode(s)
        for target in (None, np.zeros(3, dtype=np.uint8)):
            with pytest.raises(ValueError, match=message):
                dec._merge_order_outcomes(s, target=target)
            assert dec._merge_search == (1, 0, True)
    # Shots with a fired check no fault touches, whose other clusters still merge with choices:
    # each search walks one order, and stops with orders left where that order made a choice.
    rng = np.random.default_rng(14)
    pcm = (rng.random((24, 48)) < 0.12).astype(np.uint8)
    pcm[rng.integers(24, size=48), np.arange(48)] = 1
    H = np.vstack([pcm, np.zeros((1, 48), dtype=np.uint8)])
    dec = BpLsdDecoder(H, error_rate=0.05, max_iter=3, bits_per_step=1, always_run_lsd=True)
    stopped = 0
    for _ in range(20):
        s = np.append((pcm @ (rng.random(48) < 0.05)) % 2, 1)
        with pytest.raises(ValueError, match=message):
            dec.decode(s)
        with pytest.raises(ValueError, match=message):
            dec._merge_order_outcomes(s)
        paths, departures, complete = dec._merge_search
        assert (paths, departures) == (1, 0)
        stopped += not complete
    assert stopped >= 15, stopped  # (all 20 on macOS arm64)
