"""The Loom port (`stabilizer_qec.loom`): the rotated-surface-code factory, the interpreter's
circuits, syndromes, detectors and observables, the exporters and the validator, against
fixtures recorded from el-loom 0.4.0 (tests/loom_port/fixtures.json), and against Loom itself
where it is installed (it needs Python 3.11 or 3.12)."""

from __future__ import annotations

import json
import pathlib
import re

import pytest

import stabilizer_qec as sq
from stabilizer_qec import loom
from stabilizer_qec.loom import (
    ConditionalLogicalX,
    Direction,
    Eka,
    Grow,
    Lattice,
    LogicalMeasurement,
    LogicalX,
    MeasureBlockSyndromes,
    MeasureLogicalX,
    MeasureLogicalZ,
    Merge,
    Orientation,
    ResetAllDataQubits,
    RotatedSurfaceCode,
    Shrink,
    SingleQubitPauliEigenstate,
    Split,
)

FIXTURES = json.loads((pathlib.Path(__file__).parent / "loom_port" / "fixtures.json").read_text())

try:
    import loom as their_loom  # noqa: F401
    from loom.executor import EkaToStimConverter as TheirStim
    from loom.interpreter import interpret_eka as their_interpret
    from loom_rotated_surface_code.code_factory import RotatedSurfaceCode as TheirRSC
except ImportError:  # pragma: no cover
    their_loom = None

needs_loom = pytest.mark.skipif(their_loom is None, reason="el-loom is not installed")

LATTICE = Lattice.square_2d((15, 15))
PLUS = SingleQubitPauliEigenstate.PLUS


def rsc(label, pos=(0, 0), dx=3, dz=3):
    return RotatedSurfaceCode.create(dx, dz, LATTICE, unique_label=label, position=pos)


def experiment(name):
    """The blocks and operations of a recorded experiment, as the fixture generator wrote them."""
    b1, b2, b3 = rsc("b1"), rsc("b2", (4, 0)), rsc("b3", (0, 4))
    cases = {
        "memory_z_d3": ([b1], [ResetAllDataQubits("b1"), MeasureBlockSyndromes("b1", n_cycles=3), MeasureLogicalZ("b1")]),
        "memory_x_d3": ([b1], [ResetAllDataQubits("b1", state=PLUS), MeasureBlockSyndromes("b1", n_cycles=3), MeasureLogicalX("b1")]),
        "memory_z_d5": ([rsc("b5", dx=5, dz=5)], [ResetAllDataQubits("b5"), MeasureBlockSyndromes("b5", n_cycles=2), MeasureLogicalZ("b5")]),
        "merge_h": ([b1, b2], [ResetAllDataQubits("b1"), ResetAllDataQubits("b2"), MeasureBlockSyndromes("b1"), MeasureBlockSyndromes("b2"), Merge(["b1", "b2"], "m"), MeasureBlockSyndromes("m", n_cycles=2), MeasureLogicalZ("m")]),
        "merge_v": ([b1, b3], [ResetAllDataQubits("b1", state=PLUS), ResetAllDataQubits("b3", state=PLUS), MeasureBlockSyndromes("b1"), MeasureBlockSyndromes("b3"), Merge(["b1", "b3"], "m"), MeasureBlockSyndromes("m", n_cycles=2), MeasureLogicalX("m")]),
        "grow_right": ([b1], [ResetAllDataQubits("b1"), MeasureBlockSyndromes("b1"), Grow("b1", Direction.RIGHT, 2), MeasureBlockSyndromes("b1", n_cycles=2), MeasureLogicalZ("b1")]),
        "grow_bottom": ([b1], [ResetAllDataQubits("b1"), MeasureBlockSyndromes("b1"), Grow("b1", Direction.BOTTOM, 2), MeasureBlockSyndromes("b1", n_cycles=2), MeasureLogicalZ("b1")]),
        "shrink_left": ([rsc("w", dx=5, dz=3)], [ResetAllDataQubits("w"), MeasureBlockSyndromes("w"), Shrink("w", Direction.LEFT, 2), MeasureBlockSyndromes("w", n_cycles=2), MeasureLogicalZ("w")]),
        "logical_x_then_z": ([b1], [ResetAllDataQubits("b1"), MeasureBlockSyndromes("b1"), LogicalX("b1"), MeasureBlockSyndromes("b1"), MeasureLogicalZ("b1")]),
        "merge_split_h_1": ([b1, b2], [ResetAllDataQubits("b1"), ResetAllDataQubits("b2"), MeasureBlockSyndromes("b1"), MeasureBlockSyndromes("b2"), Merge(["b1", "b2"], "m"), MeasureBlockSyndromes("m", n_cycles=2), Split("m", ("c1", "c2"), Orientation.VERTICAL, 1), MeasureBlockSyndromes("c1"), MeasureBlockSyndromes("c2"), MeasureLogicalZ("c1"), MeasureLogicalZ("c2")]),
        "merge_split_h_3": ([b1, b2], [ResetAllDataQubits("b1"), ResetAllDataQubits("b2"), MeasureBlockSyndromes("b1"), MeasureBlockSyndromes("b2"), Merge(["b1", "b2"], "m"), MeasureBlockSyndromes("m", n_cycles=2), Split("m", ("c1", "c2"), Orientation.VERTICAL, 3), MeasureBlockSyndromes("c1"), MeasureBlockSyndromes("c2"), MeasureLogicalZ("c1"), MeasureLogicalZ("c2")]),
        "merge_d5_gap2": ([rsc("a", dx=5, dz=5), rsc("b", (8, 0), dx=5, dz=5)], [ResetAllDataQubits("a", state=PLUS), ResetAllDataQubits("b", state=PLUS), MeasureBlockSyndromes("a"), MeasureBlockSyndromes("b"), Merge(["a", "b"], "m"), MeasureBlockSyndromes("m"), MeasureLogicalX("m")]),
        "merge_conditional": ([b1, b2], [ResetAllDataQubits("b1"), ResetAllDataQubits("b2", state=PLUS), MeasureBlockSyndromes("b1"), MeasureBlockSyndromes("b2"), Merge(["b1", "b2"], "m"), MeasureBlockSyndromes("m"), Split("m", ("c1", "c2"), Orientation.VERTICAL, 1), ConditionalLogicalX("c2", LogicalMeasurement(("b1", "b2"), "ZZ")), MeasureBlockSyndromes("c2"), MeasureLogicalZ("c2"), MeasureLogicalZ("c1")]),
        "parallel_ops": ([b1, b2], [(ResetAllDataQubits("b1"), ResetAllDataQubits("b2")), (MeasureBlockSyndromes("b1", n_cycles=2), MeasureBlockSyndromes("b2", n_cycles=2)), (MeasureLogicalZ("b1"), MeasureLogicalZ("b2"))]),
    }
    blocks, operations = cases[name]
    return Eka(LATTICE, blocks, operations)


def canonical(text: str) -> str:
    return str(sq.Circuit(text)).strip()


def structure(text: str):
    """A circuit's gate applications as a multiset of (gate, targets) — each two-qubit gate a
    pair, each one-qubit gate a single, so that Stim's fusing of neighbouring instructions does
    not matter — with its detector coordinates and its observables' sizes: what two circuits
    share when only the order of commuting gates within a layer differs."""
    gates, detectors, observables = [], [], []
    for line in canonical(text).splitlines():
        head, _, rest = line.partition(" ")
        targets = rest.split()
        if head.startswith("DETECTOR"):
            detectors.append((head, len(targets)))
        elif head.startswith("OBSERVABLE_INCLUDE"):
            observables.append(len(targets))
        elif head.startswith("QUBIT_COORDS"):
            gates.append(line)
        elif head in ("CX", "CY", "CZ", "SWAP"):
            gates.extend(f"{head} {a} {b}" for a, b in zip(targets[::2], targets[1::2]))
        else:
            gates.extend(f"{head} {t}" for t in targets)
    return sorted(gates), sorted(map(str, detectors)), sorted(observables)


# ---------------------------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(FIXTURES["blocks"]))
def test_rotated_surface_code_blocks_are_looms(key):
    dx, dz, x_boundary, first_row, px, py = key.split(",")
    block = RotatedSurfaceCode.create(int(dx), int(dz), Lattice.square_2d((20, 20)), unique_label="b", position=(int(px), int(py)), x_boundary=Orientation(x_boundary), weight_2_stab_is_first_row=(first_row == "True"))
    ref = FIXTURES["blocks"][key]
    circuit_names = {c.uuid: c.name for c in block.syndrome_circuits}
    assert sorted([s.pauli, [list(q) for q in s.data_qubits], [list(q) for q in s.ancilla_qubits], circuit_names[block.stabilizer_to_circuit[s.uuid]]] for s in block.stabilizers) == ref["stabilizers"]
    assert [{"pauli": o.pauli, "data": [list(q) for q in o.data_qubits]} for o in block.logical_x_operators] == ref["logical_x"]
    assert [{"pauli": o.pauli, "data": [list(q) for q in o.data_qubits]} for o in block.logical_z_operators] == ref["logical_z"]
    assert {str(list(q)): c for q, c in block.pauli_charges.items()} == ref["charges"]
    assert {d: block.boundary_type(d) for d in ("top", "bottom", "left", "right")} == ref["boundary_types"]
    assert list(block.size) == ref["size"] and list(block.upper_left_qubit) == ref["upper_left"]
    assert sorted(list(q) for q in block.topological_corners) == ref["topological_corners"]
    assert [list(q) for q in block.geometric_corners] == ref["geometric_corners"]
    assert sorted([c.name, c.pauli, [[g.name for g in slot] for slot in c.circuit.circuit]] for c in block.syndrome_circuits) == ref["syndrome_circuits"]


def test_block_validation():
    block = rsc("b")
    assert block.n_data_qubits == 9 and block.n_logical_qubits == 1 and block.n_irreducible_stabs == 8
    assert block.x_boundary == Orientation.HORIZONTAL and block.weight_2_stab_is_first_row
    with pytest.raises(ValueError, match="must commute"):
        loom.Block([loom.Stabilizer("XX", [(0, 0, 0), (1, 0, 0)]), loom.Stabilizer("ZZ", [(1, 0, 0), (2, 0, 0)])], [loom.PauliOperator("X", [(2, 0, 0)])], [loom.PauliOperator("Z", [(0, 0, 0)])])
    with pytest.raises(ValueError, match="anticommute"):
        loom.Block([loom.Stabilizer("ZZ", [(0, 0, 0), (1, 0, 0)])], [loom.PauliOperator("Z", [(0, 0, 0)])], [loom.PauliOperator("Z", [(1, 0, 0)])])
    shifted = block.shift((2, 3))
    assert shifted.upper_left_qubit == (2, 3, 0) and shifted.size == (3, 3)
    with pytest.raises(ValueError, match="share the data qubits"):
        Eka(LATTICE, [rsc("a"), rsc("b", (1, 0))])


# ---------------------------------------------------------------------------------------------
# The interpreter and the Stim output
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES["experiments"]))
def test_interpreted_experiments_are_looms(name):
    ref = FIXTURES["experiments"][name]
    step = loom.interpret_eka(experiment(name))
    assert (len(step.syndromes), len(step.detectors), len(step.logical_observables)) == (ref["n_syndromes"], ref["n_detectors"], ref["n_observables"])
    assert step.timeslice_durations == ref["timeslice_durations"]
    ours = loom.EkaToStimConverter().convert(step).text
    if name == "shrink_left":
        # Loom orders the shrunk block's stabilizers by a set of strings, which Python's hash
        # randomisation permutes from run to run: the gates of each layer, the detectors and
        # the observables are the same.
        assert structure(ours) == structure(ref["stim"])
    else:
        assert canonical(ours) == canonical(ref["stim"])


def test_stim_output_is_a_working_circuit():
    circuit = loom.eka_to_circuit(loom.interpret_eka(experiment("merge_h")))
    assert circuit.num_detectors == 76 and circuit.num_observables == 1
    # Noiseless, every detector and observable is quiet.
    d, o = circuit.compile_detector_sampler(seed=3).sample(32, separate_observables=True)
    assert not d.any() and not o.any()
    # With a flip before every measurement the model has errors, and the observable decodes.
    lines = []
    for line in str(circuit).splitlines():
        if line.startswith("M "):
            lines.append("X_ERROR(0.001) " + line.split(" ", 1)[1])
        lines.append(line)
    noisy = sq.Circuit("\n".join(lines))
    dem = noisy.detector_error_model(decompose_errors=True)
    assert dem.num_errors > 0
    d, o = noisy.compile_detector_sampler(seed=5).sample(2000, separate_observables=True)
    predicted = sq.Matching.from_detector_error_model(dem).decode_batch(d)
    assert (predicted != o).any(axis=1).mean() < 0.02


def test_eka_dict_round_trips_looms_json():
    eka = Eka.fromdict(FIXTURES["eka_dict"], block_class=RotatedSurfaceCode)
    assert [b.unique_label for b in eka.blocks] == ["b1"] and len(eka.blocks[0].stabilizers) == 8
    assert [op.__class__.__name__ for step in eka.operations for op in step] == ["ResetAllDataQubits", "MeasureBlockSyndromes", "MeasureLogicalZ"]
    text = loom.EkaToStimConverter().convert(loom.interpret_eka(eka)).text
    assert canonical(text) == canonical(FIXTURES["experiments"]["memory_z_d3"]["stim"])
    again = Eka.fromdict(eka.asdict(), block_class=RotatedSurfaceCode)
    assert canonical(loom.EkaToStimConverter().convert(loom.interpret_eka(again)).text) == canonical(text)


def test_refusals():
    eka = Eka(LATTICE, [rsc("a"), rsc("b", (4, 0))], [loom.AuxCNOT(("a", "b"))])
    with pytest.raises(NotImplementedError, match="Loom"):
        loom.interpret_eka(eka)
    with pytest.raises(ValueError, match="Y basis"):
        loom.interpret_eka(Eka(LATTICE, [rsc("a")], [loom.MeasureLogicalY("a")]))
    with pytest.raises(ValueError, match="alternate pattern"):
        loom.interpret_eka(Eka(LATTICE, [rsc("a", dx=5, dz=5), rsc("b", (7, 0), dx=5, dz=5)], [Merge(["a", "b"], "m")]))
    with pytest.raises(ValueError, match="not disjoint"):
        Eka(LATTICE, [rsc("a")], [(MeasureBlockSyndromes("a"), LogicalX("a"))])


# ---------------------------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------------------------


def test_exporters_match_loom():
    ref = FIXTURES["exporters"]["memory_z_d3"]
    step = loom.interpret_eka(experiment("memory_z_d3"))
    assert loom.EkaToQasmConverter().convert(step)[0].strip() == ref["qasm"].strip()
    assert loom.EkaToCudaqConverter().convert(step)[0].strip() == ref["cudaq"].strip()
    ours = [line for line in loom.EkaToPennylaneConverter().convert(step)[0].splitlines() if not line.startswith("qml.registers")]
    theirs = [line for line in ref["pennylane"].splitlines() if not line.startswith("qml.registers")]
    # Loom keys the wire register by channel uuids; the program is otherwise the same.
    assert ours == theirs


# ---------------------------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------------------------


def syndrome_map(step, block):
    out = {}
    for s in step.syndromes:
        stab = next(st for st in block.stabilizers if st.uuid == s.stabilizer)
        for cbit in s.measurements:
            out[f"{cbit[0]}_{cbit[1]}"] = stab
    return out


def test_validator_verdicts_are_looms():
    block = rsc("v")
    step = loom.interpret_eka(Eka(LATTICE, [block], [MeasureBlockSyndromes("v")]))
    verdict = loom.is_syndrome_extraction_circuit_valid(step.final_circuit, block, syndrome_map(step, block))
    assert verdict.valid is FIXTURES["validator"]["syndrome_extraction_valid"] is True
    # A logical X sends Z to -Z: Loom's signless map ["X0"], ["Z0"] fails, as Loom reports.
    step_x = loom.interpret_eka(Eka(LATTICE, [block], [LogicalX("v")]))
    verdict_x = loom.is_logical_operation_circuit_valid(step_x.final_circuit, block, ["X0"], ["Z0"])
    assert verdict_x.valid is FIXTURES["validator"]["logical_x_valid_as_X"] is False
    assert "Actual: LogicalState(('-Z0',))" in str(verdict_x)
    # A wrong measurement map is caught, and names the stabilizer measured instead.
    wrong = syndrome_map(step, block)
    a, b = list(wrong)[:2]
    wrong[a], wrong[b] = wrong[b], wrong[a]
    bad = loom.is_syndrome_extraction_circuit_valid(step.final_circuit, block, wrong)
    assert not bad.valid and "did not measure the assigned stabilizer" in str(bad)
    # Two rounds are a valid syndrome extraction too; a logical Z is a valid logical operation.
    two = loom.interpret_eka(Eka(LATTICE, [block], [MeasureBlockSyndromes("v", n_cycles=2)]))
    assert loom.is_syndrome_extraction_circuit_valid(two.final_circuit, block, syndrome_map(two, block)).valid
    step_z = loom.interpret_eka(Eka(LATTICE, [block], [loom.LogicalZ("v")]))
    assert loom.is_logical_operation_circuit_valid(step_z.final_circuit, block, ["X0"], ["Z0"]).valid is False
    assert loom.is_logical_operation_circuit_valid(step_z.final_circuit, block, ["X0"], ["Z0"]).checks.code_stabilizers.valid


def test_logical_state_transformations():
    pairs = loom.logical_state_transformations_to_check(["X0X1", "X1"], ["Z0", "Z0Z1"])
    assert [(a.sparse_logical_paulistrings, b[0].sparse_logical_paulistrings) for a, b in pairs] == [
        (("+Z0", "+Z1"), ("+Z0", "+Z0Z1")),
        (("+X0", "+X1"), ("+X0X1", "+X1")),
        (("+Z0", "+X1"), ("+Z0", "+X1")),
        (("+X0", "+Z1"), ("+X0X1", "+Z0Z1")),
    ]
    with pytest.raises(ValueError, match="not irreducible"):
        loom.LogicalState(["+Z0Z1", "+Z0Z1"])
    with pytest.raises(ValueError, match="does not match"):
        loom.LogicalState(["+Z0", "+Z0"])


# ---------------------------------------------------------------------------------------------
# Against Loom installed
# ---------------------------------------------------------------------------------------------


@needs_loom
def test_memory_experiment_against_loom_live():
    from loom.eka import Eka as TheirEka, Lattice as TheirLattice
    from loom.eka.operations import MeasureBlockSyndromes as TheirMBS, MeasureLogicalZ as TheirMLZ, ResetAllDataQubits as TheirReset

    lattice = TheirLattice.square_2d((15, 15))
    block = TheirRSC.create(3, 3, lattice, unique_label="b1")
    theirs = TheirStim().convert(their_interpret(TheirEka(lattice, blocks=[block], operations=[TheirReset("b1"), TheirMBS("b1", n_cycles=3), TheirMLZ("b1")])))
    text = str(theirs[0] if isinstance(theirs, tuple) else theirs.circuit)
    ours = loom.EkaToStimConverter().convert(loom.interpret_eka(experiment("memory_z_d3"))).text
    assert canonical(ours) == canonical(text)
