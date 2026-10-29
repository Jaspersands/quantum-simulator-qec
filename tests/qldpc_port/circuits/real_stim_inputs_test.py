"""The circuit, code-deformation and transversal-gate functions take Stim's own circuits, tableaus
and Pauli strings (this package's tests, not qLDPC's).

qLDPC 0.4.1 runs on Stim, so its users hand these functions ``stim`` objects; in the port, ``stim``
is this package, and a real Stim object failed the port's ``isinstance`` checks or mixed with this
package's objects (``circuit + identity``), and raised. Each now reads a Stim object exactly first,
and answers as it does given this package's copy."""

from __future__ import annotations

import numpy as np
import pytest
import stabilizer_qec as sq

from stabilizer_qec.qldpc import circuits, codes
from stabilizer_qec.qldpc._stim_inputs import ours
from stabilizer_qec.qldpc.circuits import transversal

stim = pytest.importorskip("stim", minversion="1.16")


def _same(f, real, mine) -> None:
    """``f`` on a Stim object answers as on this package's copy of it."""
    got, want = f(real), f(mine)
    assert str(got) == str(want)


@pytest.fixture(scope="module")
def steane() -> codes.QuditCode:
    return codes.SteaneCode()


@pytest.fixture(scope="module")
def encoder_text(steane: codes.QuditCode) -> str:
    return str(circuits.get_encoding_circuit(steane, only_zero=True))


def test_ours_reads_stim_objects_exactly() -> None:
    tableau = stim.Tableau.random(5)
    assert isinstance(ours(tableau), sq.Tableau) and str(ours(tableau)) == str(tableau)
    for text in ("+XYZ_", "-iX_Z", "+i", "-_Y"):
        assert ours(stim.PauliString(text)) == sq.PauliString(text)
    instruction = stim.Circuit("X_ERROR(0.123456789012) 0 1")[0]  # its text rounds to 6 digits
    assert isinstance(ours(instruction), sq.CircuitInstruction)
    assert ours(instruction).gate_args_copy() == [0.123456789012]
    block = stim.Circuit("REPEAT 3 {\n    H 0\n    TICK\n}")[0]
    assert isinstance(ours(block), sq.CircuitRepeatBlock) and ours(block).repeat_count == 3
    circuit = sq.Circuit("H 0")
    assert ours(circuit) is circuit and ours("H 0") == "H 0"


def test_as_noiseless_circuit_takes_stim_circuits() -> None:
    text = "H 0\nX_ERROR(0.1) 0\nM 0"
    _same(circuits.as_noiseless_circuit, stim.Circuit(text), sq.Circuit(text))


def test_state_stabilizers_take_stim_circuits(steane: codes.QuditCode, encoder_text: str) -> None:
    _same(lambda c: circuits.get_state_stabilizers(c, 7), stim.Circuit(encoder_text), sq.Circuit(encoder_text))
    _same(lambda c: circuits.get_logical_state_stabilizers(c, steane), stim.Circuit(encoder_text), sq.Circuit(encoder_text))


def test_state_prep_diagnostics_take_stim_objects(steane: codes.QuditCode, encoder_text: str) -> None:
    tasks = lambda c: [str(task.circuit) for task in circuits.get_state_prep_diagnostic_tasks(steane, c, [1e-3])]
    _same(tasks, stim.Circuit(encoder_text), sq.Circuit(encoder_text))
    logical_z = "+ZZZZZZZ"  # stabilizes the |0> the encoder prepares
    diagnostic = lambda obs: circuits.get_state_prep_diagnostic_circuit(steane, sq.Circuit(encoder_text), observables=obs)[0]
    _same(diagnostic, [stim.PauliString(logical_z)], [sq.PauliString(logical_z)])


@pytest.mark.parametrize("make", [codes.SteaneCode, lambda: codes.SurfaceCode(3)])
def test_deformed_takes_stim_circuits(make) -> None:
    code = make()
    real, mine = code.deformed(stim.Circuit("H 0 1")), code.deformed(sq.Circuit("H 0 1"))
    assert type(real) is type(mine)
    assert np.array_equal(real.matrix, mine.matrix)


def test_deformed_preserving_logicals_takes_stim_circuits(steane: codes.QuditCode) -> None:
    text = "H 0 1 2 3 4 5 6"  # transversal: the logical operators stay valid
    real = steane.deformed(stim.Circuit(text), preserve_logicals=True)
    mine = steane.deformed(sq.Circuit(text), preserve_logicals=True)
    assert np.array_equal(real.matrix, mine.matrix)
    assert np.array_equal(real.get_logical_ops(), mine.get_logical_ops())


def test_logical_tableaus_take_stim_tableaus(steane: codes.QuditCode) -> None:
    text = "H 0 1 2 3 4 5 6"
    _same(lambda t: circuits.get_logical_tableau(steane, t), stim.Tableau.from_circuit(stim.Circuit(text)), sq.Tableau.from_circuit(sq.Circuit(text)))
    _same(lambda c: circuits.get_logical_tableau(steane, c), stim.Circuit(text), sq.Circuit(text))
    _same(lambda t: circuits.get_encoder_and_decoder(steane, t), stim.Tableau.from_circuit(stim.Circuit("H 0")), sq.Tableau.from_circuit(sq.Circuit("H 0")))


def test_pauli_product_measurements_take_stim_pauli_strings() -> None:
    strings = ["+XYZ", "-Z_Z"]
    _same(circuits.get_pauli_product_measurements, [stim.PauliString(s) for s in strings], [sq.PauliString(s) for s in strings])
    _same(lambda ps: circuits.get_pauli_product_measurements(ps, qubits=[4, 2, 7]), [stim.PauliString(s) for s in strings], [sq.PauliString(s) for s in strings])


class _Searched(Exception):
    """Raised in place of the automorphism search, which needs GAP or Magma."""


@pytest.mark.parametrize("which", ["circuit", "tableau"])
def test_transversal_circuits_take_stim_objects(steane: codes.QuditCode, monkeypatch: pytest.MonkeyPatch, which: str) -> None:
    def search(*args, **kwargs):
        raise _Searched

    monkeypatch.setattr(transversal, "get_transversal_automorphism_group", search)
    real = stim.Circuit("H 0") if which == "circuit" else stim.Tableau.from_circuit(stim.Circuit("H 0"))
    for f in (lambda op: circuits.get_transversal_circuit(steane, op), lambda op: circuits.get_transversal_circuits(steane, [op, op])):
        with pytest.raises(_Searched):  # the logical tableaus are read before the search
            f(real)
    with pytest.raises(ValueError, match="logical tableau on 1 qubits, got 2"):
        circuits.get_transversal_circuit(steane, stim.Tableau(2))
