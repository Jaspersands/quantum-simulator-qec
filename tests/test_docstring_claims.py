"""Docstrings that state behaviour the 2.2 fixes changed, each read beside the behaviour it states:
the merged twirl's handling of gates, the stabilizer-rank probabilities over every record bit and
a reset's unread outcome, explain on gauge detectors, and the Tsim module's refusals. A change
to either side that makes them disagree fails here."""

from __future__ import annotations

import math
import re

import pytest

import stabilizer_qec as sq


def doc(obj) -> str:
    """A docstring with its line breaks and runs of spaces folded to single spaces."""
    return re.sub(r"\s+", " ", obj.__doc__ or "")


def test_twirled_says_gates_are_kept_and_never_merged_into_errors():
    # A gate between errors of its own axis: the errors merge, the gate's angle is not added.
    merged = sq.Circuit("I_ERROR[R_X(theta=0.1)] 0\nI[R_X(theta=0.3)] 0\nI_ERROR[R_X(theta=0.2)] 0\nM 0").twirled(merge=True)
    assert str(merged) == "I[R_PAULI(theta=0.3, pauli=X)] 0\nX_ERROR(0.0873322) 0\nM 0"
    assert math.isclose(merged[1].gate_args_copy()[0], math.sin(0.3) ** 2)
    # Errors either side of a gate they anticommute with stay apart; a T elsewhere does not stop them.
    assert str(sq.Circuit("I_ERROR[R_Z(theta=0.1)] 0\nI[R_X(theta=0.3)] 0\nI_ERROR[R_Z(theta=0.2)] 0\nM 0").twirled(merge=True)).count("Z_ERROR") == 2
    assert str(sq.Circuit("I_ERROR[R_X(theta=0.1)] 0\nI[T] 1\nI_ERROR[R_X(theta=0.2)] 0\nM 0").twirled(merge=True)).count("X_ERROR") == 1
    # A multi-target gate is kept once.
    assert str(sq.Circuit("I[T] 0 1\nM 0").twirled()) == "I[T] 0 1\nM 0"
    d = doc(sq.Circuit.twirled)
    assert "are kept as they are, once" in d
    assert "a T on another qubit, or R_X between X errors" in d
    assert "a non-Clifford gate's angle is never added into an error" in d
    assert "(Z either side of R_X) stay apart" in d


def test_probability_of_says_every_bit_and_resets_count():
    s = sq.Circuit("MPAD 0\nM 0").compile_stabilizer_rank_sampler(seed=1)
    assert s.probability_of([[False, False], [True, False]]).tolist() == [1.0, 0.0]
    s = sq.Circuit("H 0\nCX 0 1\nR 0\nM 1").compile_stabilizer_rank_sampler(seed=1)
    assert s.probability_of([[False], [True]]).tolist() == pytest.approx([0.5, 0.5], abs=1e-15)
    d = doc(sq.StabilizerRankSampler.probability_of)
    assert "every bit of it (``MPAD`` bits and heralds as measurements)" in d
    assert "a reset's unread outcome summed over where the qubit is entangled with the rest (up to 10 such on a path" in d
    assert "neither summed nor drawn" in d


def test_explain_says_detectors_may_be_gauge_and_observables_may_not():
    sq.Circuit("RX 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]").explain_detector_error_model_errors()
    with pytest.raises(ValueError, match="non-deterministic observables"):
        sq.Circuit("RX 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]").explain_detector_error_model_errors()
    d = doc(sq.Circuit.explain_detector_error_model_errors)
    assert "As in Stim, detectors may depend on random outcomes" in d and "observables may not" in d


def test_tsim_docstring_says_what_every_sampler_refuses():
    import stabilizer_qec.tsim as tsim
    from stabilizer_qec.tsim import zx

    with pytest.raises(zx.parse.MeasurementRecordError):
        tsim.Circuit("M 0\nDETECTOR rec[-2]").compile_detector_sampler()
    with pytest.raises(ValueError, match="anti-Hermitian"):
        tsim.Circuit("TPP Y0*X0\nM 0").compile_sampler()
    with pytest.raises(ValueError, match=r"^line 1: .*Malformed parametric tag"):
        tsim.Circuit("I[R_Z(theta=0.3*pi*pi)] 0")
    assert str(tsim.Circuit("I[R_Z (theta=0.3*pi)] 0")) == "I 0"
    d = re.sub(r"\s+", " ", tsim.__doc__)
    assert "from the graph, ``tcount``, ``to_matrix`` and every sampler" in d
    assert "is refused by every sampler, with Tsim's message, before any sampler is built" in d
    assert "is refused when the text is read, with Tsim's message after the line" in d
    assert "is the identity, as in Tsim, held as ``I 0``" in d
    assert tsim.Circuit("I[R_Z(theta=0.3*pi)#x] 0").stim_circuit == sq.Circuit("I 0")
