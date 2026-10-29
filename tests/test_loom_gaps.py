"""The rest of Loom's API in the port (`stabilizer_qec.loom`) against el-loom 0.4.0, from fixtures
recorded with Loom on PYTHONHASHSEED=0 (tests/loom_port/fixtures_gaps.json): the converters'
``(circuit, qubits, records)`` and their readings of a run's outcome, Crumble's polygons, the
Executor, the lattices (poly_2d, oct_2d, cube_3d, linear on a tuple), the circuit utilities
(as_gate, from_circuits, detailed_str, padding), Loom's refusals of bad blocks, Pauli strings,
logical states and operations, the validator's DebugData, the ancilla resets in Loom's order,
the repetition, Shor, Steane and five-qubit codes with the repetition chain's grow, shrink,
merge and split (Stim text byte for byte; a shrink orders the chain's checks by a set of
strings, so its text is compared byte for byte on Loom's hash seed, in a fresh interpreter, and
up to that order here), and Loom's module paths. Then the circuits: deterministic detectors and
observables, and a noisy version of each surgery decoded."""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

import stabilizer_qec as sq
import stabilizer_qec.deltakit_stim as dstim
from stabilizer_qec import loom
from stabilizer_qec.loom import Block, Channel, Circuit, Eka, Lattice, LogicalState, PauliOperator, Stabilizer

HERE = pathlib.Path(__file__).parent
FIXTURES = json.loads((HERE / "loom_port" / "fixtures_gaps.json").read_text())
CONVERTERS = FIXTURES["converters"]
EXPERIMENTS = FIXTURES["experiments"]


def message(f) -> str | None:
    try:
        f()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return None


def call_message(text: str) -> str:
    """A bad call's TypeError as this Python words it: before 3.10 an unexpected keyword's names
    the method without its class."""
    if sys.version_info >= (3, 10) or "unexpected keyword" not in text:
        return text
    return re.sub(r"\w+\.(?=\w+\(\))", "", text)


def loom_message(text: str) -> str:
    """Loom's message without pydantic's wrapping (its first error's text)."""
    found = re.search(r"Value error, (.*?) \[type=", text, re.S)
    return found.group(1) if found else text.split(": ", 1)[1]


def label(cbit) -> str:
    return f"{cbit[0]}_{cbit[1]}"


# ---------------------------------------------------------------------------------------------
# The converters and the executor
# ---------------------------------------------------------------------------------------------


def rep_memory():
    lat = Lattice.linear((4,))
    b = Block(unique_label="q1", stabilizers=[Stabilizer("ZZ", ((1 + i, 0), (2 + i, 0)), ancilla_qubits=((1 + i, 1),)) for i in range(2)],
              logical_x_operators=[PauliOperator("XXX", ((1, 0), (2, 0), (3, 0)))], logical_z_operators=[PauliOperator("Z", ((1, 0),))])
    ops = [loom.ResetAllDataQubits("q1", state="0"), loom.MeasureBlockSyndromes("q1", n_cycles=1), loom.LogicalX("q1"), loom.MeasureBlockSyndromes("q1", n_cycles=1), loom.MeasureLogicalZ("q1")]
    return loom.interpret_eka(Eka(lat, blocks=[b], operations=ops))


def rsc_memory():
    lat = Lattice.square_2d((10, 10))
    b = loom.RotatedSurfaceCode.create(3, 3, lat, unique_label="q")
    return loom.interpret_eka(Eka(lat, blocks=[b], operations=[loom.ResetAllDataQubits("q"), loom.MeasureBlockSyndromes("q", n_cycles=2), loom.MeasureLogicalZ("q")]))


def samples(n_shots, n_measurements):
    return np.array([[(3 * i + 2 * j + i * j) % 2 for j in range(n_measurements)] for i in range(n_shots)], dtype=bool)


@pytest.fixture(scope="module")
def step():
    return rep_memory()


def test_the_stim_converter_returns_looms_triple(step):
    ref = CONVERTERS["stim"]
    circuit, qubits, records = loom.EkaToStimConverter().convert(interpreted_eka=step)
    assert str(circuit) == ref["text"]
    assert len(qubits) == ref["n_qubits"] and circuit.num_measurements == ref["n_measurements"]
    assert [[ch.label, idx] for ch, idx in records.items()] == ref["records"]
    assert len(loom.EkaToStimConverter().convert(step)) == ref["convert_kw"] == 3
    n = ref["n_measurements"]
    assert loom.EkaToStimConverter.parse_target_run_outcome((samples(3, n), records)) == ref["parse_3"]
    assert loom.EkaToStimConverter.parse_target_run_outcome((samples(1, n), records)) == ref["parse_1"]
    # A sample of the circuit itself reads as Loom reads it.
    shots = circuit.compile_sampler(seed=0).sample(shots=4)
    parsed = loom.EkaToStimConverter.parse_target_run_outcome((shots, records))
    assert set(parsed) == {name for name, _ in ref["records"]} and all(len(v) == 4 for v in parsed.values())


class FakeMimiq:
    def __init__(self, histogram):
        self._h = histogram

    def histogram(self):
        return self._h


class FakeCudaq:
    def __init__(self, data):
        self.register_names = ["__global__"] + list(data)
        self._data = data

    def get_sequential_data(self, name):
        return self._data[name]


class Shot:
    def __init__(self, entries):
        self.entries = entries


class Results:
    def __init__(self, shots):
        self.results = shots


def test_the_other_converters_read_a_run_as_loom(step):
    ref = CONVERTERS["qasm"]
    _, _, bits = loom.EkaToQasmConverter().convert(step)
    assert [[ch.label, idx] for ch, idx in bits.items()] == ref["bits"]
    assert loom.EkaToQasmConverter.parse_target_run_outcome((ref["counts"], bits)) == ref["parse"]
    assert message(lambda: loom.EkaToQasmConverter.parse_target_run_outcome(({}, bits))) == ref["empty"]
    ref = CONVERTERS["mimiq"]
    _, _, bits = loom.EkaToMimiqConverter().convert(step)
    assert [[ch.label, idx] for ch, idx in bits.items()] == ref["bits"]
    assert loom.EkaToMimiqConverter.parse_target_run_outcome((FakeMimiq(CONVERTERS["qasm"]["counts"]), bits)) == ref["parse"]
    assert message(lambda: loom.EkaToMimiqConverter.parse_target_run_outcome((FakeMimiq({}), bits))) == ref["empty"]
    ref = CONVERTERS["cudaq"]
    _, _, registers = loom.EkaToCudaqConverter().convert(step)
    assert [[ch.label, name] for ch, name in registers.items()] == ref["names"]
    assert loom.EkaToCudaqConverter.parse_target_run_outcome(FakeCudaq(ref["registers"]), {v: k for k, v in registers.items()}) == ref["parse"]
    ref = CONVERTERS["pennylane"]
    assert loom.EkaToPennylaneConverter.parse_target_run_outcome({k: np.array(v) for k, v in ref["outcome"].items()}) == ref["parse"]
    ref = CONVERTERS["guppy"]
    assert loom.EkaToGuppylangConverter.parse_target_run_outcome(Results([Shot([tuple(e) for e in s]) for s in ref["shots"]])) == ref["parse_result"]
    assert loom.EkaToGuppylangConverter.parse_target_run_outcome(Shot([tuple(e) for e in ref["shots"][1]])) == ref["parse_shot"]
    assert loom.EkaToGuppylangConverter.parse_target_run_outcome(object()) == ref["parse_other"]


def test_crumble_polygons_are_looms(step):
    assert loom.EkaToStimConverter().stim_polygons(step) == CONVERTERS["polygons_rep"]
    assert loom.EkaToStimConverter.stim_polygons(rsc_memory()) == CONVERTERS["polygons_rsc"]
    assert loom.EkaToStimConverter().print_stim_circuit_for_crumble(step) == CONVERTERS["crumble_rep"]
    lat = Lattice.square_2d((5, 5))
    noncss = Block(unique_label="n", stabilizers=[Stabilizer("XZ", ((0, 0, 0), (1, 0, 0)), ancilla_qubits=((0, 0, 1),)), Stabilizer("ZX", ((1, 0, 0), (2, 0, 0)), ancilla_qubits=((1, 0, 1),))],
                   logical_x_operators=[PauliOperator("X", ((0, 0, 0),))], logical_z_operators=[PauliOperator("ZXZ", ((0, 0, 0), (1, 0, 0), (2, 0, 0)))])
    noncss_step = loom.interpret_eka(Eka(lat, blocks=[noncss], operations=[loom.ResetAllDataQubits("n"), loom.MeasureBlockSyndromes("n")]))
    assert message(lambda: loom.EkaToStimConverter().stim_polygons(noncss_step)) == CONVERTERS["polygons_noncss"]


def test_the_executor_is_looms(step):
    T = loom.TargetLanguage
    assert [m.name for m in T] == CONVERTERS["target_languages"]
    assert str(loom.Executor(target_language=T.STIM).export_eka(step)[0]) == CONVERTERS["executor_stim"]
    assert loom.Executor(target_language=T.OPEN_QASM3).export_eka(step)[0] == CONVERTERS["executor_qasm"]
    assert loom.Executor(target_language=T.OPEN_QASM3).export_eka_circuit(step.final_circuit)[0] == CONVERTERS["executor_qasm_circuit"]
    assert [loom.Executor(target_language=T.PENNYLANE).converter.is_catalyst, loom.Executor(target_language=T.PENNYLANE_CATALYST).converter.is_catalyst] == CONVERTERS["executor_catalyst"]
    records = loom.EkaToStimConverter().convert(step)[2]
    assert loom.Executor(target_language=T.STIM).parse_outcome((samples(2, CONVERTERS["stim"]["n_measurements"]), records)) == CONVERTERS["executor_parse"]
    for language in T:
        assert isinstance(loom.Executor(target_language=language).converter, loom.Converter)
    with pytest.raises(ValueError, match="target_language"):
        loom.Executor(target_language="stim")
    assert message(lambda: loom.EkaToPennylaneConverter().convert(step, is_catalyst=True)) == call_message(CONVERTERS["converter_bad_keyword"])
    assert message(lambda: loom.EkaToStimConverter().convert()) == call_message(CONVERTERS["converter_no_step"])


# ---------------------------------------------------------------------------------------------
# Lattices and circuits
# ---------------------------------------------------------------------------------------------

LATTICES = {
    "linear_tuple": lambda: Lattice.linear((3,)),
    "poly_5_4": lambda: Lattice.poly_2d((2, 2), 5, anc=4),
    "poly_6": lambda: Lattice.poly_2d((2, 1), 6),
    "poly_3_custom": lambda: Lattice.poly_2d((1, 2), 3, poly_radius=0.7, poly_offset=0.3, cell_dist_factor=3.0, anc=2, anc_radius=0.1, anc_offset=0.5, anc_disp=(0.25, -0.5)),
    "oct": lambda: Lattice.oct_2d((2, 2)),
    "oct_anc": lambda: Lattice.oct_2d((2, 1), r=0.3, anc=2),
    "cube": lambda: Lattice.cube_3d((2, 2, 2)),
}


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_lattices_are_looms(name):
    ref = FIXTURES["lattices"][name]
    lat = LATTICES[name]()
    assert lat.lattice_type.value == ref["type"] and list(lat.size) == ref["size"]
    assert (lat.n_dimensions, lat.unit_cell_size) == (ref["n_dimensions"], ref["unit_cell_size"])
    assert [list(q) for q in lat.all_qubits()] == ref["all_qubits"]
    assert [list(q) for q in lat.all_qubits(force_including_basis=True)][:12] == ref["with_basis"]
    for ours, theirs in ((lat.basis_vectors, ref["basis"]), (lat.lattice_vectors, ref["vectors"])):
        assert len(ours) == len(theirs)
        for a, b in zip(ours, theirs):
            assert len(a) == len(b) and all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12) for x, y in zip(a, b))


def test_lattice_refusals_are_looms():
    ref = FIXTURES["lattice_errors"]
    cases = {
        "negative": lambda: Lattice.linear((-1,)),
        "dimensions": lambda: Lattice.square_2d((3,)),
        "no_size": lambda: Lattice.square_2d().all_unit_cells(),
        "partial_size": lambda: Lattice.square_2d().all_unit_cells((3, None)),
        "poly_n": lambda: Lattice.poly_2d((2, 2), 0),
    }
    for name, f in cases.items():
        with pytest.raises(ValueError) as info:
            f()
        assert str(info.value) == loom_message(ref[name]), name


def test_circuit_utilities_are_looms():
    ref = FIXTURES["circuits"]
    assert repr(Circuit.as_gate("cnot", nr_qchannels=2)) == ref["as_gate"]
    g = Circuit.as_gate("measure", 1, 1, duration=2)
    assert [[c.type.value, c.label] for c in g.channels] + [g.duration] == ref["as_gate_channels"]
    bell = Circuit.from_circuits("bell", [(Circuit.as_gate("h", nr_qchannels=1), [0]), (Circuit.as_gate("cnot", nr_qchannels=2), [0, 1])])
    assert repr(bell) == ref["from_circuits"] and len(bell.channels) == ref["from_circuits_channels"]
    assert bell.circuit[0][0].channels[0] is bell.circuit[1][0].channels[0]
    nested = Circuit.from_circuits("x", [[(Circuit.as_gate("cnot", 2), [0, 1]), (Circuit.as_gate("h", 1), [2])], [(Circuit.as_gate("measure", 1, 1), [1, 3])]])
    assert [repr(nested), nested.detailed_str(), nested.duration, [c.type.value for c in nested.channels]] == ref["from_circuits_nested"]
    assert message(lambda: Circuit.from_circuits("x", [(Circuit.as_gate("cnot", 2), [0, 1]), (Circuit.as_gate("measure", 1, 1), [1, 0])])) == ref["from_circuits_type"]
    assert message(lambda: Circuit.from_circuits("x", [(Circuit.as_gate("cnot", 2), [0, 1]), (Circuit.as_gate("measure", 1, 1), [1, 2, 3])])) == ref["from_circuits_count"]
    assert message(lambda: Circuit.from_circuits("x", ((Circuit.as_gate("h", 1), [0]), (Circuit.as_gate("h", 1), [0])))) == ref["from_circuits_tuple"]
    assert message(lambda: Circuit.from_circuits("x", [(Circuit.as_gate("h", 1), [0])])) == ref["from_circuits_one"]
    c1, c2, c3 = Channel(label="my_channel_1"), Channel(label="my_channel_2"), Channel(label="c3")
    g1, g2, g12 = Circuit(name="g1", channels=[c1]), Circuit(name="g2", channels=[c2]), Circuit(name="g12", channels=[c1, c2])
    assert Circuit(name="p_circuit", circuit=((g1, g2), (g12,))).detailed_str() == ref["detailed_str"]
    assert g1.detailed_str() == ref["detailed_str_gate"]
    long_sub = Circuit(name="long_subcircuit", circuit=(Circuit(name="gate_1", channels=[c1]), Circuit(name="gate_2", channels=[c2])))
    cnot2, h3, h1 = Circuit("cnot", channels=[c1, c2], duration=2), Circuit("h", channels=[c3]), Circuit("h", channels=[c1])
    seqs = {
        "long_then_gates": ((long_sub,), (Circuit(name="gate_1", channels=[c1]), Circuit(name="gate_2", channels=[c2]))),
        "h_then_cnot": ((h1,), (cnot2,)),
        "cnot_then_h": ((cnot2,), (h1,)),
        "cnot_with_h": ((cnot2, h3),),
        "cnot_free_h_then_h": ((cnot2,), (h3,), (h1,)),
        "empty": (),
    }
    assert {k: [[g.name for g in tick] for tick in Circuit.construct_padded_circuit_time_sequence(v)] for k, v in seqs.items()} == ref["padded"]
    assert repr(Circuit(name="valid_circuit", circuit=Circuit.construct_padded_circuit_time_sequence(seqs["long_then_gates"]))) == ref["padded_repr"]


# ---------------------------------------------------------------------------------------------
# Refusals, logical states, DebugData
# ---------------------------------------------------------------------------------------------

d0, d1, d2 = (0, 0), (1, 0), (2, 0)


def ZZ(a, b):
    return Stabilizer("ZZ", (a, b))


REFUSALS = {
    "one_sided_logicals": lambda: Block(stabilizers=[ZZ(d0, d1), ZZ(d1, d2)], logical_x_operators=[PauliOperator("XXX", (d0, d1, d2))], logical_z_operators=[]),
    "no_stabilizers": lambda: Block(stabilizers=[], logical_x_operators=[PauliOperator("XXX", (d0, d1, d2))], logical_z_operators=[PauliOperator("Z", (d0,))]),
    "logicals_counts": lambda: Block(stabilizers=[ZZ(d0, d1)], logical_x_operators=[PauliOperator("XX", (d0, d1)), PauliOperator("XX", (d0, d1))], logical_z_operators=[PauliOperator("Z", (d0,))]),
    "logical_outside": lambda: Block(stabilizers=[ZZ(d0, d1), ZZ(d1, d2)], logical_x_operators=[PauliOperator("XXXX", (d0, d1, d2, (3, 0)))], logical_z_operators=[PauliOperator("Z", (d0,))]),
    "lowercase_pauli": lambda: PauliOperator("xz", ((0, 0), (1, 0))),
    "lowercase_stabilizer": lambda: Stabilizer("zz", ((0, 0), (1, 0))),
    "bad_pauli": lambda: Stabilizer("ZQ", ((0, 0), (1, 0))),
    "pauli_length": lambda: Stabilizer("ZZZ", ((0, 0), (1, 0))),
    "duplicate_qubits": lambda: Stabilizer("ZZ", ((0, 0), (0, 0))),
    "coordinate_lengths": lambda: Stabilizer("ZZ", ((0, 0), (1, 0, 0))),
    "duplicate_ancillas": lambda: Stabilizer("ZZ", ((0, 0), (1, 0)), ancilla_qubits=((0, 1), (0, 1))),
    "unsigned_state": lambda: LogicalState(["Z0"]),
    "state_bad_pauli": lambda: LogicalState(["+Q0"]),
    "state_repeated_qubit": lambda: LogicalState(["+Z0X0"]),
    "state_out_of_range": lambda: LogicalState(["+Z0", "+X3"]),
    "merge_one_block": lambda: loom.Merge(["a"], "m"),
    "merge_three_blocks": lambda: loom.Merge(["a", "b", "c"], "m"),
    "split_one_name": lambda: loom.Split("a", ("x",), loom.Orientation.VERTICAL, 1),
    "grow_zero": lambda: loom.Grow("a", "left", 0),
    "transformation_not_tableau": lambda: loom.logical_state_transformations_to_check(["Z0"], ["Z0"]),
    "transformation_out_of_range": lambda: loom.logical_state_transformations_to_check(["X1"], ["Z0"]),
}
# pydantic's own messages (a tuple's length): the port says so in its words.
OWN_WORDS = {"merge_one_block", "merge_three_blocks", "split_one_name"}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_refusals_are_looms(name):
    ref = FIXTURES["refusals"][name]
    with pytest.raises(ValueError) as info:
        REFUSALS[name]()
    if name not in OWN_WORDS:
        assert str(info.value) == loom_message(ref)


def test_aux_cnot_takes_exactly_two_block_names():
    # Loom types AuxCNOT's input_blocks_name tuple[str, str] and refuses any other length when the operation is built
    # (pydantic's "Tuple should have at most 2 items after validation, not 3", "Field required"; a bare string, "Input
    # should be a valid tuple"); the port says so in its words, as for Merge and Split. (It took three names and
    # dropped the third, and failed on one later, with an IndexError.)
    for names in (("a", "b", "c"), ("a",), (), "ct"):
        with pytest.raises(ValueError, match="input_blocks_name must be two block names"):
            loom.AuxCNOT(names)
    assert loom.AuxCNOT(["a", "b"]).input_blocks_name == ("a", "b")
    lat = Lattice.square_2d((15, 15))
    blocks = [loom.RotatedSurfaceCode.create(3, 3, lat, unique_label=name, position=(5 * i, 0)) for i, name in enumerate("abc")]
    with pytest.raises(ValueError):
        loom.interpret_eka(Eka(lat, blocks=blocks, operations=[loom.AuxCNOT(("a", "b", "c"))]))


def test_eka_is_a_frozen_dataclass_as_looms():
    # Loom's Eka is a frozen (pydantic) dataclass: equal field by field, so it reads back from its dict equal; its
    # fields cannot be assigned (which would skip its checks); and the dataclass repr.
    lat = Lattice.square_2d((10, 10))

    def build():
        b = loom.RotatedSurfaceCode.create(3, 3, lat, unique_label="q")
        return Eka(lat, blocks=[b], operations=[loom.ResetAllDataQubits("q", state="0"), loom.MeasureBlockSyndromes("q", n_cycles=1), loom.MeasureLogicalZ("q")])

    e = build()
    assert e == build() and hash(e) == hash(build())
    assert Eka.fromdict(e.asdict(), block_class=loom.RotatedSurfaceCode) == e
    assert e != Eka(lat, blocks=e.blocks, operations=e.operations[:1])
    assert e != Eka(Lattice.square_2d((9, 9)), blocks=e.blocks, operations=e.operations)
    assert repr(Eka(lat)) == f"Eka(lattice={lat!r}, blocks=(), operations=())"
    far = loom.RotatedSurfaceCode.create(5, 5, Lattice.square_2d((20, 20)), unique_label="far", position=(12, 12))
    for field_name, value in (("blocks", (far,)), ("operations", ()), ("lattice", Lattice.square_2d((2, 2)))):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(e, field_name, value)
    assert e == build()
    with pytest.raises(ValueError, match="too large for the lattice"):
        Eka(lat, blocks=[far])
    with pytest.raises(ValueError, match="unique labels"):
        Eka(Lattice.square_2d((30, 30)), blocks=[e.blocks[0], loom.RotatedSurfaceCode.create(3, 3, lat, unique_label="q", position=(5, 5))])


def test_logical_states_and_transformations_are_looms():
    ref = FIXTURES["accepted"]
    state = LogicalState(("+Z0", "-X1"))
    assert [list(state.sparse_logical_paulistrings), state.n_logical_qubits] == ref["logical_state"]
    as_text = lambda pairs: [[str(a), [str(b) for b in bs]] for a, bs in pairs]  # noqa: E731
    assert as_text(loom.logical_state_transformations_to_check(["Z0"], ["X0"])) == ref["transformations"]
    assert as_text(loom.logical_state_transformations_to_check(["X0X1", "X1"], ["Z0", "Z0Z1"])) == ref["cnot"]
    assert as_text(loom.logical_state_transformations_to_check(["X0"], ["Z0"])) == ref["x_as_x"]
    assert message(lambda: loom.logical_state_transformations_to_check(["X0"], ["Z0", "Z1"])) == ref["lengths"]
    assert message(lambda: loom.logical_state_transformations_to_check(["x0"], ["z0"])) == ref["lowercase"]


def steane_syndrome_extraction():
    """The validator notebook's Steane block and Reichardt's syndrome extraction."""
    qubit_labels = [(i,) for i in range(7)]
    stabs = {"X": [], "Z": []}
    for lab in ([(0,), (2,), (4,), (6,)], [(1,), (2,), (5,), (6,)], [(3,), (4,), (5,), (6,)]):
        stabs["X"].append(Stabilizer("XXXX", lab))
        stabs["Z"].append(Stabilizer("ZZZZ", lab))
    steane = Block(stabilizers=stabs["X"] + stabs["Z"], logical_x_operators=[PauliOperator("X" * 7, qubit_labels)], logical_z_operators=[PauliOperator("Z" * 7, qubit_labels)])
    dq = [Channel(label=str(q)) for q in steane.data_qubits]
    aq = [Channel(label=f"ancilla_qubit_{i}") for i in range(3)]
    cc = [Channel(type="classical", label=f"c_{i}_0") for i in range(3)]
    C = Circuit
    sec = C("first_round", circuit=[
        [C("H", channels=[aq[0]])],
        [C("CNOT", channels=[aq[0], dq[4]]), C("CNOT", channels=[dq[6], aq[1]]), C("CNOT", channels=[dq[5], aq[2]])],
        [C("CNOT", channels=[aq[0], aq[2]])],
        [C("CNOT", channels=[aq[0], dq[0]]), C("CNOT", channels=[dq[4], aq[1]]), C("CNOT", channels=[dq[1], aq[2]])],
        [C("CNOT", channels=[aq[0], dq[2]]), C("CNOT", channels=[dq[3], aq[1]]), C("CNOT", channels=[dq[6], aq[2]])],
        [C("CNOT", channels=[aq[0], aq[1]])],
        [C("CNOT", channels=[aq[0], dq[6]]), C("CNOT", channels=[dq[5], aq[1]]), C("CNOT", channels=[dq[2], aq[2]])],
        [C("H", channels=[aq[0]])],
        [C("Measurement", channels=[aq[0], cc[0]]), C("Measurement", channels=[aq[1], cc[1]]), C("Measurement", channels=[aq[2], cc[2]])],
    ])
    m_ok = {cc[0].label: stabs["X"][0], cc[1].label: stabs["Z"][2], cc[2].label: stabs["Z"][1]}
    m_wrong = {cc[0].label: stabs["X"][0], cc[1].label: stabs["Z"][0], cc[2].label: stabs["Z"][1]}
    return steane, {
        "ok": (sec, m_ok),
        "wrong_map": (sec, m_wrong),
        "extra_cnot": (C("circuit_with_added_cnot", circuit=sec.circuit + ((C("CNOT", channels=[dq[0], dq[6]]),),)), m_ok),
        "logical_x": (C("circuit_with_log_x", circuit=sec.circuit + tuple((C("X", channels=[dq[i]]),) for i in range(7))), m_ok),
        "destabilizer": (C("circuit_w_destab", circuit=sec.circuit + tuple((C("Z", channels=[dq[i]]),) for i in [2, 6])), m_ok),
    }


@pytest.mark.parametrize("name", sorted(FIXTURES["debug_data"]))
def test_debug_data_is_looms(name):
    ref = FIXTURES["debug_data"][name]
    steane, cases = steane_syndrome_extraction()
    circuit, mapping = cases[name]
    d = loom.is_syndrome_extraction_circuit_valid(circuit, steane, measurement_to_input_stabilizer_map=mapping)
    assert bool(d.valid) == ref["valid"] and str(d) == ref["str"]
    assert [[type(c).__name__, bool(c.valid), len(c.output), str(c), type(c.output).__name__, sorted(f.name for f in dataclasses.fields(c.output))] for c in d.checks] == ref["checks"]
    assert sorted(f.name for f in dataclasses.fields(d.checks)) == ref["field_names"]


def test_ancilla_resets_are_looms():
    lat = Lattice.square_2d((10, 10))
    b = loom.RotatedSurfaceCode.create(3, 3, lat, unique_label="q")
    step = loom.interpret_eka(Eka(lat, blocks=[b], operations=[loom.ResetAllDataQubits("q"), loom.ResetAllAncillaQubits("q", state="+"), loom.MeasureBlockSyndromes("q"), loom.ResetAllAncillaQubits("q"), loom.MeasureLogicalZ("q")]))
    assert loom.EkaToQasmConverter().convert(step)[0] == FIXTURES["resets"]["qasm"]
    assert str(loom.EkaToStimConverter().convert(step)[0]) == FIXTURES["resets"]["stim"]


# ---------------------------------------------------------------------------------------------
# The other codes
# ---------------------------------------------------------------------------------------------


def block_dump(block) -> dict:
    names = {c.uuid: c.name for c in block.syndrome_circuits}
    return {
        "label": block.unique_label,
        "stabilizers": sorted([s.pauli, [list(q) for q in s.data_qubits], [list(q) for q in s.ancilla_qubits], names[block.stabilizer_to_circuit[s.uuid]]] for s in block.stabilizers),
        "logical_x": [[o.pauli, [list(q) for q in o.data_qubits]] for o in block.logical_x_operators],
        "logical_z": [[o.pauli, [list(q) for q in o.data_qubits]] for o in block.logical_z_operators],
        "syndrome_circuits": sorted([c.name, c.pauli, [[g.name for g in tick] for tick in c.circuit.circuit]] for c in block.syndrome_circuits),
    }


def build(spec) -> Eka:
    """An experiment as the recorder wrote it: code blocks by kind, operations by name."""
    blocks_spec, ops = spec
    lattices = {"linear": lambda: Lattice.linear((20,)), "square": lambda: Lattice.square_2d((10, 10)), "poly": lambda: Lattice.poly_2d((10, 10), 5, anc=4)}
    codes = {"rep": loom.RepetitionCode, "shor": loom.ShorCode, "steane": loom.SteaneCode, "five": loom.FiveQubitPerfectCode}
    blocks, lattice = [], None
    for kind, kw in blocks_spec:
        kw = {k: tuple(v) if isinstance(v, list) else v for k, v in kw.items()}
        lattice = lattices[kw.pop("lattice")]()
        blocks.append(codes[kind].create(lattice=lattice, **kw))
    return Eka(lattice, blocks=blocks, operations=[getattr(loom, name)(**{k: tuple(v) if isinstance(v, list) else v for k, v in kw.items()}) for name, kw in ops])


def stim(step, ticks: bool = False) -> str:
    return loom.EkaToStimConverter().convert(step, with_ticks=ticks).text


def layered(text: str):
    """A circuit up to the order of the gates within a tick (see test_loom_ops_a.py)."""
    layers: list[list[str]] = [[]]
    records: list[tuple] = []
    detectors, observables = [], []
    for line in text.splitlines():
        if "(" in line.split(" ", 1)[0]:
            head, _, rest = line.partition(")")
            head += ")"
        else:
            head, _, rest = line.partition(" ")
        targets = rest.split()
        if head == "TICK":
            layers.append([])
        elif head.startswith("DETECTOR"):
            detectors.append((head, sorted(records[int(t[4:-1])] for t in targets)))
        elif head.startswith("OBSERVABLE_INCLUDE"):
            observables.append((head, sorted(records[int(t[4:-1])] for t in targets)))
        elif head.startswith("QUBIT_COORDS"):
            layers[-1].append(line)
        elif head in ("CX", "CY", "CZ"):
            layers[-1].extend(f"{head} {a} {b}" for a, b in zip(targets[::2], targets[1::2]))
        else:
            layers[-1].extend(f"{head} {t}" for t in targets)
            if head in ("M", "MX", "MY"):
                records.extend((len(layers) - 1, head, t) for t in targets)
    return [sorted(layer) for layer in layers], sorted(detectors), observables


def shrinks(spec) -> bool:
    return any(name == "Shrink" for name, _ in spec[1])


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_code_experiments_are_looms(name):
    ref = EXPERIMENTS[name]
    eka = build(ref["spec"])
    assert [block_dump(b) for b in eka.blocks] == ref["initial_blocks"]
    step = loom.interpret_eka(eka)
    assert (len(step.syndromes), len(step.detectors)) == (ref["n_syndromes"], ref["n_detectors"])
    assert [[label(m) for m in o.measurements] for o in step.logical_observables] == ref["observables"]
    assert sorted((block_dump(b) for b in step.get_blocks_at_index(-1)), key=lambda d: d["label"]) == ref["final_blocks"]
    assert repr(step.final_circuit) == ref["final_circuit"]
    if "stim_error" in ref:
        # Loom's Stim coordinates stop at a cell's third site; the five-qubit cell has nine.
        assert message(lambda: loom.EkaToStimConverter().convert(step)) == ref["stim_error"]
        assert loom.EkaToQasmConverter().convert(step)[0] == ref["qasm"]
    elif shrinks(ref["spec"]):
        assert layered(stim(step)) == layered(ref["stim"])
    else:
        assert stim(step) == ref["stim"]


SHRINKS = sorted(name for name, ref in EXPERIMENTS.items() if shrinks(ref["spec"]))

SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import test_loom_gaps as t
print(json.dumps({name: t.stim(t.loom.interpret_eka(t.build(t.EXPERIMENTS[name]["spec"]))) for name in t.SHRINKS}))
"""


@pytest.fixture(scope="module")
def on_looms_hash_seed():
    if sys.version_info < (3, 11):
        pytest.skip("the fixtures' set orders are those of the string hash of Python 3.11 and later")
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(HERE)], env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize("name", SHRINKS)
def test_a_shrink_is_looms_byte_for_byte_on_its_hash_seed(on_looms_hash_seed, name):
    assert on_looms_hash_seed[name] == EXPERIMENTS[name]["stim"]


def code_refusals() -> dict:
    lin = Lattice.linear((20,))
    sq_lat = Lattice.square_2d((5, 5))
    RC, Shor, Steane, Five = loom.RepetitionCode, loom.ShorCode, loom.SteaneCode, loom.FiveQubitPerfectCode
    rep = RC.create(d=4, check_type="X", lattice=lin, unique_label="q", position=(3,))

    def interp(blocks, ops):
        return loom.interpret_eka(Eka(lin, blocks=blocks, operations=ops))
    r = RC.create(d=5, check_type="Z", lattice=lin, unique_label="r", position=(1,))
    r_far = RC.create(d=3, check_type="Z", lattice=lin, unique_label="r_far", position=(7,))
    rx = RC.create(d=3, check_type="X", lattice=lin, unique_label="rx", position=(12,))
    g = Block(stabilizers=[Stabilizer("ZZ", [(15, 0), (16, 0)], ancilla_qubits=[(15, 1)])], logical_x_operators=[PauliOperator("XX", [(15, 0), (16, 0)])], logical_z_operators=[PauliOperator("Z", [(15, 0)])], unique_label="g")
    return {
        "rep_d0": lambda: RC.create(d=0, check_type="X", lattice=lin),
        "rep_check": lambda: RC.create(d=3, check_type="Y", lattice=lin),
        "rep_lattice": lambda: RC.create(d=3, check_type="X", lattice=sq_lat),
        "rep_position": lambda: RC.create(d=3, check_type="X", lattice=lin, position=[1]),
        "rep_position_len": lambda: RC.create(d=3, check_type="X", lattice=lin, position=(1, 2)),
        "rep_lx_support": lambda: RC.create(d=3, check_type="Z", lattice=lin, logical_x_operator=PauliOperator("XX", [(0, 0), (1, 0)])),
        "rep_lz_support": lambda: RC.create(d=3, check_type="X", lattice=lin, logical_z_operator=PauliOperator("ZZ", [(0, 0), (1, 0)])),
        "rep_boundary_top": lambda: rep.boundary_qubits("top"),
        "rep_shift_not_data": lambda: rep.get_shifted_equivalent_logical_operator((0, 0)),
        "rep_eq_other": lambda: rep == 3,
        "shor_lattice": lambda: Shor.create(lattice=sq_lat),
        "shor_position": lambda: Shor.create(lattice=lin, position=[0]),
        "shor_eq_other": lambda: Shor.create(lattice=lin) == 1,
        "steane_lattice": lambda: Steane.create(lattice=lin),
        "steane_eq_other": lambda: Steane.create(lattice=Lattice.square_2d((5, 5))) == 1,
        "five_lattice": lambda: Five.create(lattice=sq_lat),
        "five_cell": lambda: Five.create(lattice=Lattice.poly_2d((2, 2), 5, anc=1)),
        "five_position": lambda: Five.create(lattice=Lattice.poly_2d((2, 2), 5, anc=4), position=(0.5, 0)),
        "five_eq_other": lambda: Five.create(lattice=Lattice.poly_2d((2, 2), 5, anc=4)) == 1,
        "grow_top": lambda: interp([r], [loom.Grow("r", "top", 1)]),
        "grow_left_beyond": lambda: interp([r], [loom.Grow("r", "left", 2)]),
        "shrink_too_large": lambda: interp([r], [loom.Shrink("r", "left", 4)]),
        "shrink_bottom": lambda: interp([r], [loom.Shrink("r", "bottom", 1)]),
        "merge_check_types": lambda: interp([r, rx], [loom.Merge(["r", "rx"], "m", orientation="horizontal")]),
        "merge_vertical": lambda: interp([r, r_far], [loom.Merge(["r", "r_far"], "m", orientation="vertical")]),
        "merge_no_orientation": lambda: interp([r, r_far], [loom.Merge(["r", "r_far"], "m")]),
        "split_outside": lambda: interp([r], [loom.Split("r", ("a", "b"), "vertical", 7)]),
        "split_edge": lambda: interp([r], [loom.Split("r", ("a", "b"), "vertical", 4)]),
        "split_short": lambda: interp([r], [loom.Split("r", ("a", "b"), "vertical", 1)]),
        "split_horizontal": lambda: interp([r], [loom.Split("r", ("a", "b"), "horizontal", 2)]),
        "mixed_blocks": lambda: interp([r, g], [loom.MeasureBlockSyndromes("r")]),
        "unsupported_op": lambda: interp([r], [loom.TransversalHadamard("r")]),
        "shor_unsupported": lambda: loom.interpret_eka(Eka(lin, blocks=[Shor.create(lattice=lin, unique_label="s")], operations=[loom.Grow("s", "right", 1)])),
    }


def test_code_refusals_are_looms():
    ref = FIXTURES["code_refusals"]
    for name, f in code_refusals().items():
        assert message(f) == ref[name], name


def test_repetition_chain_queries_are_looms():
    ref = FIXTURES["code_refusals"]
    rep = loom.RepetitionCode.create(d=4, check_type="X", lattice=Lattice.linear((20,)), unique_label="q", position=(3,))
    assert [rep.check_type, list(rep.boundary_qubits("left")), list(rep.boundary_qubits(loom.Direction.RIGHT))] == ref["rep_queries"]
    o, st = rep.get_shifted_equivalent_logical_operator((5, 0))
    assert [o.pauli, [list(q) for q in o.data_qubits], [[s.pauli, [list(q) for q in s.data_qubits]] for s in st]] == ref["rep_shifted"]
    o, st = rep.get_shifted_equivalent_logical_operator((3, 0))
    assert [o.pauli, [list(q) for q in o.data_qubits], len(st)] == ref["rep_unshifted"]
    assert rep == rep and hash(rep) == hash(rep) and isinstance(rep.rename("z"), loom.RepetitionCode) and isinstance(rep.shift((1,)), loom.RepetitionCode)


# ---------------------------------------------------------------------------------------------
# Loom's module paths
# ---------------------------------------------------------------------------------------------

# Loom's names the port leaves out: the Tanner graphs, parity-check matrices and syndrome-
# circuit algorithms built on them, if-else circuits, the Pauli-array algebra and pydantic
# helpers of loom.eka.utilities, the JSON helpers, the operation signatures, and the plugin
# machinery (one dispatch table here).
NOT_PORTED = {
    "ClassicalParityCheckMatrix", "ClassicalTannerGraph", "ParityCheckMatrix", "TannerGraph", "cardinal_circuit", "cartesian_product_tanner_graphs", "coloration_circuit", "extract_syndrome_circuit",
    "generate_stabilizer_and_syndrome_circuits_from_algorithm", "verify_css_code_stabilizers", "IfElseCircuit",
    "AntiCommutationError", "BoolOp", "PauliArray", "PauliOp", "SignedPauliOp", "StabArray", "UnsignedPauliOp", "anti_commutes_npfunc", "binary_gaussian_elimination", "cardinality_distribution",
    "extract_subgraphs_from_edge_labels", "find_destabarray", "find_logical_operator_set", "find_maximum_matching", "findall", "g", "g_npfunc", "invert_bookkeeping_matrix", "is_stabarray_equivalent",
    "is_subset_of_stabarray", "is_tableau_valid", "merge_stabarrays", "minimum_edge_coloring", "ndarray_rowsum", "paulichar_to_xz", "paulichar_to_xz_npfunc", "pauliops_anti_commute", "paulis_anti_commute",
    "paulixz_to_char", "paulixz_to_char_npfunc", "reduce_stabarray", "reduce_stabarray_with_bookkeeping", "reindex_stabarray", "rowsum", "sparse_formatter", "stabarray_bge", "stabarray_bge_with_bookkeeping",
    "stabarray_standard_form", "subtract_stabarrays", "swap_stabarray_rows", "tableau_generates_pauli_group", "verify_css_code_condition",
    "dataclass_config", "distinct_error", "ensure_tuple", "larger_than_zero_error", "no_name_error", "retrieve_field", "uuid_error", "dumps", "loads",
    "ALL_EKA_OP_SIGNATURES", "CLIFFORD_GATES_SIGNATURE", "OpSignature", "OpType", "USUAL_CLIFFORD_GATES", "USUAL_QUANTUM_GATES",
    "BaseApplicator", "CodeApplicator", "load_plugin_options",
}


def port_path(loom_module: str) -> str:
    """``loom.eka.operations`` is ``stabilizer_qec.loom.eka.operations``, ``loom_shor_code.code_factory``
    is ``stabilizer_qec.loom.shor_code.code_factory``."""
    return "stabilizer_qec.loom." + (loom_module[len("loom."):] if loom_module.startswith("loom.") else loom_module[len("loom_"):])


@pytest.mark.parametrize("loom_module", sorted(FIXTURES["modules"]))
def test_looms_module_paths_import(loom_module):
    module = importlib.import_module(port_path(loom_module))
    missing = sorted(n for n in FIXTURES["modules"][loom_module] if not hasattr(module, n) and n not in NOT_PORTED)
    assert missing == []
    for n in FIXTURES["modules"][loom_module]:
        if hasattr(module, n) and hasattr(loom, n) and not n.startswith("_"):
            assert getattr(module, n) is getattr(loom, n), n


def test_documented_imports():
    from stabilizer_qec.loom.eka import ChannelType, LogicalState as LS
    from stabilizer_qec.loom.eka.circuit import Circuit as C
    from stabilizer_qec.loom.eka.operations import Grow
    from stabilizer_qec.loom.eka.operations.code_operation import Split
    from stabilizer_qec.loom.eka.utilities import Direction
    from stabilizer_qec.loom.interpreter.interpreter import interpret_eka
    from stabilizer_qec.loom.repetition_code.code_factory import RepetitionCode
    from stabilizer_qec.loom.rotated_surface_code.code_factory.rotated_surface_code import RotatedSurfaceCode
    from stabilizer_qec.loom.rotated_surface_code.operations import AuxCNOT
    from stabilizer_qec.loom.rotated_surface_code.utilities import FourBodySchedule

    assert (ChannelType, LS, C, Grow, Split, Direction, interpret_eka) == (loom.ChannelType, loom.LogicalState, loom.Circuit, loom.Grow, loom.Split, loom.Direction, loom.interpret_eka)
    assert (RepetitionCode, RotatedSurfaceCode, AuxCNOT, FourBodySchedule) == (loom.RepetitionCode, loom.RotatedSurfaceCode, loom.AuxCNOT, loom.FourBodySchedule)
    import stabilizer_qec.loom.eka.operations as ops

    assert ops.Merge is loom.Merge and sys.modules["stabilizer_qec.loom.eka.operations"] is ops


# ---------------------------------------------------------------------------------------------
# The circuits
# ---------------------------------------------------------------------------------------------

STIM = sorted(name for name, ref in EXPERIMENTS.items() if "stim" in ref)


def noisy(text: str, p: float) -> sq.Circuit:
    """The circuit with this package's Deltakit noise: depolarizing after every Clifford gate,
    flips before measurements and after resets."""
    if sys.version_info < (3, 11):
        pytest.skip("stabilizer_qec.deltakit needs Python 3.11, as Deltakit")
    from stabilizer_qec.deltakit import circuit as dk

    circuit = dk.Circuit.from_stim_circuit(dstim.Circuit(text))
    circuit.apply_gate_noise(dk.after_clifford_depolarisation(p), dk.Circuit.LayerAdjacency.AFTER)
    circuit.apply_gate_noise(dk.before_measure_flip_probability(p), dk.Circuit.LayerAdjacency.BEFORE)
    circuit.apply_gate_noise(dk.after_reset_flip_probability(p), dk.Circuit.LayerAdjacency.AFTER)
    return sq.Circuit(str(circuit.as_stim_circuit()))


@pytest.mark.parametrize("name", STIM)
def test_code_circuits_have_deterministic_detectors_and_observables(name):
    step = loom.interpret_eka(build(EXPERIMENTS[name]["spec"]))
    circuit = sq.Circuit(stim(step, ticks=True))
    circuit.detector_error_model()  # raises on a detector or observable that is not deterministic
    d, o = circuit.compile_detector_sampler(seed=1).sample(256, separate_observables=True)
    assert not d.any() and not o.any()


@pytest.mark.parametrize("name", sorted(name for name in STIM if name.startswith("rep")))
def test_a_noisy_repetition_chain_decodes(name):
    step = loom.interpret_eka(build(EXPERIMENTS[name]["spec"]))
    circuit = noisy(stim(step, ticks=True), 0.001)
    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem.num_errors > 0 and dem.num_detectors == len(step.detectors)
    d, o = circuit.compile_detector_sampler(seed=5).sample(2000, separate_observables=True)
    predicted = sq.Matching.from_detector_error_model(dem).decode_batch(d)
    assert (predicted != o).any(axis=1).mean() < 0.03
