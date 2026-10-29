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


def _line(name, g):
    """One instruction of gate ``g`` written with ``name`` (its own or an alias), with
    arguments and targets it takes."""
    args = ""
    if g.num_parens_arguments_range and 0 not in g.num_parens_arguments_range:
        args = "(" + ",".join(["0.01"] * g.num_parens_arguments_range[0]) + ")"
    if g.is_two_qubit_gate:
        targets = "0 1"
    elif name in ("MPP", "SPP", "SPP_DAG"):
        targets = "X0*Z1"
    elif name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
        targets = "rec[-1]"
        args = "(0)" if name == "OBSERVABLE_INCLUDE" else args
    elif name == "TICK":
        targets = ""
    elif name == "SHIFT_COORDS":
        targets, args = "", "(1)"
    elif name in ("CORRELATED_ERROR", "ELSE_CORRELATED_ERROR", "E"):
        targets = "X0"
    else:
        targets = "0"
    return f"{name}{args} {targets}".rstrip()


@pytest.mark.parametrize("name", sorted({a for g in stim.gate_data().values() for a in [g.name, *g.aliases]} - {"REPEAT"}))
def test_every_name_and_alias_reads_as_stim_reads_it(name):
    # Every gate's name and every alias Stim gives it (`H_XZ` for H, `CNOT` for CX, ...) parses,
    # appends and prints as Stim's circuit does.
    text = "M 0\n" + _line(name, stim.gate_data(name))
    assert str(sq.Circuit(text)) == str(stim.Circuit(text))
    if name not in ("DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "SHIFT_COORDS", "MPP", "SPP", "SPP_DAG", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR", "E", "MPAD"):
        g = stim.gate_data(name)
        targets = [0, 1] if g.is_two_qubit_gate else [0]
        args = [0.01] * g.num_parens_arguments_range[0] if g.num_parens_arguments_range and 0 not in g.num_parens_arguments_range else []
        theirs, ours = stim.Circuit(), sq.Circuit()
        theirs.append(name, targets, args)
        ours.append(name, targets, args)
        assert str(ours) == str(theirs)
