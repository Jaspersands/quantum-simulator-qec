"""src/gate_data.rs (tools/gen_gate_data.py) against Stim: every instruction, with every
attribute ``stim.gate_data`` exposes, equal."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")

ATTRS = ["name", "aliases", "is_noisy_gate", "is_reset", "is_single_qubit_gate", "is_symmetric_gate", "is_two_qubit_gate", "is_unitary", "num_parens_arguments_range", "produces_measurements", "takes_measurement_record_targets", "takes_pauli_targets"]


def test_same_gates():
    assert sorted(sq.gate_data()) == sorted(stim.gate_data())


@pytest.mark.parametrize("name", sorted(stim.gate_data()))
def test_same_data(name):
    s, o = stim.gate_data(name), sq.gate_data(name)
    for a in ATTRS:
        want = getattr(s, a)
        got = getattr(o, a)
        assert (sorted(got) if a == "aliases" else got) == (sorted(want) if a == "aliases" else want), a
    def get(obj, attr):
        try:
            return getattr(obj, attr)
        except IndexError as e:
            return ("IndexError", str(e).split(" ")[0])

    st, ot = get(s, "tableau"), get(o, "tableau")
    assert (str(st) if st is not None else None) == (str(ot) if ot is not None else None)
    su, ou = get(s, "unitary_matrix"), get(o, "unitary_matrix")
    if isinstance(su, np.ndarray):
        assert np.array_equal(su, ou)
    else:
        assert type(su) is type(ou)
    assert [str(f) for f in (s.flows or [])] == [str(f) for f in (o.flows or [])]
    for f in ("inverse", "generalized_inverse"):
        a, b = get(s, f), get(o, f)
        name = lambda g: g.name if isinstance(g, (stim.GateData, sq.GateData)) else g  # noqa: E731
        assert name(a) == name(b), f
    for unsigned in (False, True):
        a, b = s.hadamard_conjugated(unsigned=unsigned), o.hadamard_conjugated(unsigned=unsigned)
        assert (a.name if a is not None else None) == (b.name if b is not None else None)
