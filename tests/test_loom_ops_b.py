"""The Loom port's move, Y-wall phase gate and rotate applicators (`stabilizer_qec.loom`
MoveBlock, LogicalPhaseViaYwall, RotateBlock) against fixtures recorded from el-loom 0.4.0
with PYTHONHASHSEED=0 (tests/loom_port/fixtures_ops_b.json): the Stim circuits, the syndromes,
detectors and observables, the final blocks, Loom's refusals, and the circuits themselves (a
detector error model builds, the noiseless shots never flip, the logical value is the
operation's)."""

from __future__ import annotations

import builtins
import functools
import json
import os
import pathlib
import subprocess
import sys

import pytest

import stabilizer_qec as sq
from stabilizer_qec import loom
from stabilizer_qec.loom import (
    Direction,
    Eka,
    Lattice,
    LogicalPhaseViaYwall,
    LogicalX,
    MeasureBlockSyndromes,
    MeasureLogicalX,
    MeasureLogicalZ,
    MoveBlock,
    Orientation,
    ResetAllDataQubits,
    RotateBlock,
    RotatedSurfaceCode,
    SingleQubitPauliEigenstate,
)

FIXTURES = json.loads((pathlib.Path(__file__).parent / "loom_port" / "fixtures_ops_b.json").read_text())
PORTED = ("move",)
EXPERIMENTS = {name: ref for name, ref in FIXTURES["experiments"].items() if name.startswith(PORTED)}
REFUSALS = {name: ref for name, ref in FIXTURES["refusals"].items() if name.startswith(PORTED)}


def build(spec: dict) -> Eka:
    """The experiment a fixture records, from its spec (the recorder builds Loom's the same way)."""
    lattice = Lattice.square_2d(tuple(spec["lattice"]))
    blocks = [RotatedSurfaceCode.create(b["dx"], b["dz"], lattice, unique_label=b["label"], position=tuple(b["position"]), x_boundary=Orientation(b["x_boundary"]), weight_2_stab_is_first_row=b["first_row"]) for b in spec["blocks"]]
    makers = {
        "ResetAllDataQubits": lambda label, state: ResetAllDataQubits(label, state=SingleQubitPauliEigenstate[state]),
        "MeasureBlockSyndromes": lambda label, n: MeasureBlockSyndromes(label, n_cycles=n),
        "MeasureLogicalX": MeasureLogicalX,
        "MeasureLogicalZ": MeasureLogicalZ,
        "LogicalX": LogicalX,
        "MoveBlock": lambda label, d: MoveBlock(label, Direction(d)),
        "LogicalPhaseViaYwall": lambda label, d: LogicalPhaseViaYwall(label, Direction(d)),
        "RotateBlock": lambda label, d: RotateBlock(label, Direction(d)),
    }
    return Eka(lattice, blocks=blocks, operations=[makers[name](*args) for name, *args in spec["ops"]])


def hash_dependent(name: str) -> bool:
    """The phase gate and the rotation end in a shrink, whose block Loom builds from a set of
    stabilizers: the order of the later syndrome rounds' gates and measurements follows
    Python's string hashing, so their text is Loom's only under the same PYTHONHASHSEED."""
    return not name.startswith("move")


@functools.lru_cache(maxsize=None)
def interpreted(name: str):
    step = loom.interpret_eka(build(EXPERIMENTS[name]["spec"]))
    return step, loom.EkaToStimConverter().convert(step).text


def stim_text(name: str) -> str:
    return interpreted(name)[1]


def canonical(text: str) -> str:
    return str(sq.Circuit(text)).strip()


def structure(text: str):
    """The gate applications as a multiset (each two-qubit gate a pair), the detectors'
    coordinates and sizes and the observables' sizes: what two circuits share when only the
    order within a layer differs."""
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


def label(v):
    return list(v) if isinstance(v, tuple) else v


def block_dump(b) -> dict:
    names = {c.uuid: c.name for c in b.syndrome_circuits}
    return {
        "label": b.unique_label,
        "stabilizers": sorted([s.pauli, [list(q) for q in s.data_qubits], [list(q) for q in s.ancilla_qubits], names[b.stabilizer_to_circuit[s.uuid]].lower()] for s in b.stabilizers),
        "logical_x": [[o.pauli, [list(q) for q in o.data_qubits]] for o in b.logical_x_operators],
        "logical_z": [[o.pauli, [list(q) for q in o.data_qubits]] for o in b.logical_z_operators],
        "syndrome_circuits": sorted([c.name.lower(), c.pauli, [[g.name for g in slot] for slot in c.circuit.circuit]] for c in b.syndrome_circuits if c.uuid in set(b.stabilizer_to_circuit.values())),
    }


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_experiments_are_looms(name):
    ref = EXPERIMENTS[name]
    step, ours = interpreted(name)
    assert (len(step.syndromes), len(step.detectors), len(step.logical_observables)) == (ref["n_syndromes"], ref["n_detectors"], ref["n_observables"])
    assert step.timeslice_durations == ref["timeslice_durations"]
    assert sorted([[s.round for s in d.syndromes], label(d.labels.get("space_coordinates")), label(d.labels.get("time_coordinate"))] for d in step.detectors) == ref["detectors"]
    assert [[o.label, len(o.measurements), sum(1 for m in o.measurements if not isinstance(m, tuple))] for o in step.logical_observables] == ref["observables"]
    assert sorted((block_dump(b) for b in step.get_blocks_at_index(-1)), key=lambda d: d["label"]) == sorted(ref["final_blocks"], key=lambda d: d["label"])
    assert structure(ours) == structure(ref["stim"])
    if not hash_dependent(name):
        assert canonical(ours) == canonical(ref["stim"])


@pytest.fixture(scope="module")
def texts_at_hash_seed_0():
    """Our Stim text of each hash-dependent experiment under PYTHONHASHSEED=0, the seed the
    fixtures were recorded with (in a subprocess unless this run already has it)."""
    names = sorted(n for n in EXPERIMENTS if hash_dependent(n))
    if os.environ.get("PYTHONHASHSEED") == "0":
        return {n: stim_text(n) for n in names}
    code = f"import json, sys; sys.path.insert(0, {str(pathlib.Path(__file__).parent)!r}); import test_loom_ops_b as t; print(json.dumps({{n: t.stim_text(n) for n in {names!r}}}))"
    run = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONHASHSEED="0"), capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, run.stderr[-3000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", sorted(n for n in EXPERIMENTS if hash_dependent(n)))
def test_stim_text_is_looms_under_the_same_hash_seed(name, texts_at_hash_seed_0):
    assert canonical(texts_at_hash_seed_0[name]) == canonical(EXPERIMENTS[name]["stim"])


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_circuits_are_deterministic_and_keep_the_logical(name):
    spec = EXPERIMENTS[name]["spec"]
    circuit = sq.Circuit(stim_text(name))
    circuit.detector_error_model()  # raises on a non-deterministic detector or observable
    d, o = circuit.compile_detector_sampler(seed=7).sample(64, separate_observables=True)
    assert not d.any() and not o.any()
    # |0> measured in Z stays 0; |+> measured in X picks up a sign from each pair of S gates.
    n_phase = sum(1 for op in spec["ops"] if op[0] == "LogicalPhaseViaYwall")
    expected = [(n_phase // 2) % 2 if op[0] == "MeasureLogicalX" else 0 for op in spec["ops"] if op[0].startswith("MeasureLogical")]
    _, obs_signs = circuit.reference_detector_and_observable_signs()
    assert [int(v) for v in obs_signs] == expected


@pytest.mark.parametrize("name", ["move_d3_twice"])
def test_noisy_circuit_decodes(name):
    lines = []
    for line in str(sq.Circuit(stim_text(name))).splitlines():
        if line.startswith(("M ", "MX ", "MY ")):
            lines.append("X_ERROR(0.001) " + line.split(" ", 1)[1] if line.startswith("M ") else "Z_ERROR(0.001) " + line.split(" ", 1)[1])
        lines.append(line)
    noisy = sq.Circuit("\n".join(lines))
    dem = noisy.detector_error_model(decompose_errors=True)
    assert dem.num_errors > 0
    d, o = noisy.compile_detector_sampler(seed=5).sample(2000, separate_observables=True)
    predicted = sq.Matching.from_detector_error_model(dem).decode_batch(d)
    assert (predicted != o).any(axis=1).mean() < 0.02


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_refusals_are_looms(name):
    error = REFUSALS[name]["error"]
    kind = error.partition(": ")[0]
    with pytest.raises(getattr(builtins, kind)) as info:
        loom.interpret_eka(build(REFUSALS[name]["spec"]))
    assert f"{type(info.value).__name__}: {info.value}" == error


def test_operation_fields_are_looms():
    assert MoveBlock("q", "right").direction == Direction.RIGHT
    assert LogicalPhaseViaYwall("q", "top").growth_direction == Direction.TOP
