"""A tag on Deltakit's own instructions (``LEAKAGE[site](0.1) 0``, ``CX[t](p1, p2, p3, p4) 0 1``),
which deltakit_stim 0.2.7 reads and ``deltakit.circuit``'s tagged noise channels write, is refused
by stabilizer_qec.deltakit_stim (the lowered line's tag is where the instruction is held). The
refusal says so, the same way from text, from ``CircuitInstruction`` and from Deltakit's circuits,
and the documentation says so rather than that Deltakit's Stim has no such tags."""

import pytest

import stabilizer_qec.deltakit_stim as stim
from stabilizer_qec.deltakit.circuit import Circuit, GateLayer, NoiseLayer, Qubit
from stabilizer_qec.deltakit.circuit.gates import HERALD_LEAKAGE_EVENT, MZ, RZ
from stabilizer_qec.deltakit.circuit.noise_channels import Leakage, Relax

REFUSAL = r"takes no tag here \(deltakit_stim keeps one; this package does not carry it\)"

MAKERS = {
    "text LEAKAGE": lambda: stim.Circuit("R 0\nLEAKAGE[t](0.1) 0\nM 0"),
    "text RELAX": lambda: stim.Circuit("RELAX[t](0.1) 0"),
    "text RL": lambda: stim.Circuit("RL[x] 0"),
    "text herald": lambda: stim.Circuit("HERALD_LEAKAGE_EVENT[h](0.1) 0"),
    "text transport": lambda: stim.Circuit("CX[foo](0.1, 0.1, 0.1, 0.1) 0 1"),
    "append": lambda: stim.Circuit().append("LEAKAGE", [0], 0.1, tag="t"),
    "instruction": lambda: stim.CircuitInstruction("LEAKAGE", [0], [0.1], tag="x"),
    "instruction transport": lambda: stim.CircuitInstruction("CX", [0, 1], [0.1, 0.1, 0.1, 0.1], tag="x"),
    "deltakit.circuit": lambda: Circuit([GateLayer(RZ(Qubit(0))), NoiseLayer([Leakage(Qubit(0), 0.1, tag="leak-site"), Relax(Qubit(0), 0.2, tag="r")]), GateLayer(HERALD_LEAKAGE_EVENT(Qubit(0))), GateLayer(MZ(Qubit(0)))]).as_stim_circuit(),
}


@pytest.mark.parametrize("name", list(MAKERS))
def test_a_tag_on_deltakits_instruction_is_refused_and_says_why(name):
    with pytest.raises(ValueError, match=REFUSAL):
        MAKERS[name]()


def test_the_documentation_states_the_refusal():
    assert "no tags in its Stim" not in stim.Circuit.without_tags.__doc__
    assert "LEAKAGE[t](0.1) 0" in stim.Circuit.without_tags.__doc__
    assert "LEAKAGE[site](0.1) 0" in stim.__doc__ and "refuses one with ``ValueError``" in stim.__doc__
    # Tags on Stim's own instructions are kept and removed as Stim's are.
    c = stim.Circuit("H[a] 0\nLEAKAGE(0.1) 0")
    assert str(c) == "H[a] 0\nLEAKAGE(0.1) 0"
    assert str(c.without_tags()) == "H 0\nLEAKAGE(0.1) 0"
