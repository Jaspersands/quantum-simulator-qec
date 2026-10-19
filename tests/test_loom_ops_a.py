"""The Loom port's transversal Hadamard, state injection and auxiliary CNOT (`stabilizer_qec.loom`)
against el-loom 0.4.0, from fixtures recorded with Loom on PYTHONHASHSEED=0
(tests/loom_port/fixtures_ops_a.json): the Stim text byte for byte (for a T injection, which
Loom's Stim converter refuses, the QASM text and the detectors), the syndromes, observables,
joint measurements and final blocks. The auxiliary CNOT's shrink orders the target's
stabilizers by a set of strings, so its text is compared byte for byte on Loom's hash seed, in
a fresh interpreter, and up to that order here. Then the circuits themselves: deterministic
detectors, the observables each operation should leave (a Bell pair's product fixed, an
injected T state's logical X read +1 with probability (1 + 1/√2)/2), and a noisy version's
error model decomposed and decoded."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys

import pytest

import stabilizer_qec as sq
import stabilizer_qec.deltakit_stim as dstim
from stabilizer_qec import loom
from stabilizer_qec.loom import executor

HERE = pathlib.Path(__file__).parent
FIXTURES = json.loads((HERE / "loom_port" / "fixtures_ops_a.json").read_text())
EXPERIMENTS = FIXTURES["experiments"]
SUMMARY = ("n_syndromes", "n_detectors", "n_observables", "timeslice_durations", "observables", "logical_measurements", "final_blocks")


def build(spec: dict) -> loom.Eka:
    """An experiment as the recorder wrote it: blocks on a 15 x 15 lattice, operations by name
    (each time step a list of them)."""
    lattice = loom.Lattice.square_2d((15, 15))
    blocks = [loom.RotatedSurfaceCode.create(b["dx"], b["dz"], lattice, unique_label=b["label"], position=tuple(b["position"]), x_boundary=loom.Orientation(b["x_boundary"]), weight_2_stab_is_first_row=b["w2"]) for b in spec["blocks"]]
    steps = [[getattr(loom, name)(**{k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}) for name, kwargs in step] for step in spec["operations"]]
    return loom.Eka(lattice, blocks, steps)


def operations(spec: dict) -> list:
    return [(name, kwargs) for step in spec["operations"] for name, kwargs in step]


def has(spec: dict, name: str) -> bool:
    return any(n == name for n, _ in operations(spec))


def label(cbit) -> str:
    return f"{cbit[0]}_{cbit[1]}"


def block_dump(block) -> dict:
    names = {c.uuid: c.name for c in block.syndrome_circuits}
    return {
        "label": block.unique_label,
        "stabilizers": sorted([s.pauli, [list(q) for q in s.data_qubits], [list(q) for q in s.ancilla_qubits], names[block.stabilizer_to_circuit[s.uuid]]] for s in block.stabilizers),
        "logical_x": [[o.pauli, [list(q) for q in o.data_qubits]] for o in block.logical_x_operators],
        "logical_z": [[o.pauli, [list(q) for q in o.data_qubits]] for o in block.logical_z_operators],
        "syndrome_circuits": sorted([c.name, c.pauli, [[g.name for g in tick] for tick in c.circuit.circuit]] for c in block.syndrome_circuits),
    }


def summary(step) -> dict:
    return {
        "n_syndromes": len(step.syndromes),
        "n_detectors": len(step.detectors),
        "n_observables": len(step.logical_observables),
        "timeslice_durations": list(step.timeslice_durations),
        "observables": [[o.label, [label(m) for m in o.measurements]] for o in step.logical_observables],
        "logical_measurements": sorted([sorted(map(list, zip(k.blocks, k.observable))), [label(m) for m in v]] for k, v in step.logical_measurements.items()),
        "final_blocks": sorted((block_dump(b) for b in step.get_blocks_at_index(-1)), key=lambda d: d["label"]),
    }


UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def without_uuids(summary: dict) -> dict:
    """A summary with uuid labels (an auxiliary block's, when its usual label is taken) as one."""
    summary = json.loads(UUID.sub("<uuid4>", json.dumps(summary)))
    summary["logical_measurements"] = sorted([sorted(blocks), cbits] for blocks, cbits in summary["logical_measurements"])
    return summary


def stim(step, ticks: bool = False) -> str:
    return loom.EkaToStimConverter().convert(step, with_ticks=ticks).text


def detectors_apart(text: str):
    """The text but its detectors, and its detectors as a multiset."""
    lines = text.splitlines()
    return [line for line in lines if not line.startswith("DETECTOR")], sorted(line for line in lines if line.startswith("DETECTOR"))


def layered(text: str):
    """A circuit with TICKs up to the order of the gates within a tick: each tick's gate
    applications as a multiset, measurements named by tick, gate and qubit, detectors (with
    their coordinates) and observables as the measurements they read."""
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


# ---------------------------------------------------------------------------------------------
# Against Loom
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_interpreted_experiments_are_looms(name):
    ref = EXPERIMENTS[name]
    step = loom.interpret_eka(build(ref["spec"]))
    assert without_uuids(summary(step)) == without_uuids({key: ref[key] for key in SUMMARY})
    if "qasm" in ref:
        # Loom's Stim converter has no T gate; its QASM is the whole circuit, names included.
        assert loom.EkaToQasmConverter().convert(step)[0] == ref["qasm"]
        assert [[label(m) for m in d.measurements] for d in step.detectors] == ref["detectors"]
        with pytest.raises(KeyError, match="Operation t not found"):
            loom.EkaToStimConverter().convert(step)
    elif not has(ref["spec"], "AuxCNOT"):
        assert stim(step) == ref["stim"]
    elif "stim_ticks" in ref:
        # Rounds after the CNOT: the target's checks in the order of Python's hashes.
        assert layered(stim(step, ticks=True)) == layered(ref["stim_ticks"])
    elif "stim" in ref:
        # Only the final readout's detectors of the target follow the set's order.
        assert detectors_apart(stim(step)) == detectors_apart(ref["stim"])


AUXCNOT = sorted(name for name, ref in EXPERIMENTS.items() if has(ref["spec"], "AuxCNOT"))

SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import test_loom_ops_a as t
print(json.dumps(t.auxcnot_texts()))
"""


def auxcnot_texts() -> dict:
    """Each auxiliary-CNOT experiment's Stim text, with and without TICKs (for the test that
    runs this module on Loom's hash seed)."""
    out = {}
    for name in AUXCNOT:
        step = loom.interpret_eka(build(EXPERIMENTS[name]["spec"]))
        out[name] = {"stim": stim(step), "stim_ticks": stim(step, ticks=True)}
    return out


@pytest.fixture(scope="module")
def on_looms_hash_seed():
    if sys.version_info < (3, 11):
        pytest.skip("the fixtures' set orders are those of the string hash of Python 3.11 and later")
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(HERE)], env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize("name", AUXCNOT)
def test_auxcnot_is_looms_byte_for_byte_on_its_hash_seed(on_looms_hash_seed, name):
    ref, ours = EXPERIMENTS[name], on_looms_hash_seed[name]
    if "stim" in ref:
        assert ours["stim"] == ref["stim"]
    if "stim_sha256" in ref:
        assert hashlib.sha256(ours["stim"].encode()).hexdigest() == ref["stim_sha256"]
    if "stim_ticks" in ref:
        assert ours["stim_ticks"] == ref["stim_ticks"]


def normalized(message: str) -> str:
    """Loom's wording, but for a channel's id and an enum's rendering (Python's)."""
    message = re.sub(r"\([0-9a-f]{6}\.\.\)", "(..)", message)
    return re.sub(r"Orientation\.(\w+)", lambda m: m.group(1).lower(), message)


@pytest.mark.parametrize("name", sorted(FIXTURES["refusals"]))
def test_refusals_are_looms(name):
    ref = FIXTURES["refusals"][name]
    with pytest.raises(ValueError) as info:
        loom.interpret_eka(build(ref["spec"]))
    # A 3 x 5 block's centre is in a quadrant too: Loom's circuit check (wrapped by pydantic)
    # refuses the two resets, as this one does.
    assert normalized(str(info.value)) in normalized(ref["message"])


def test_eka_dict_round_trips_the_operations():
    eka = build(EXPERIMENTS["si_s_then_hadamard"]["spec"])
    again = loom.Eka.fromdict(json.loads(json.dumps(eka.asdict())), block_class=loom.RotatedSurfaceCode)
    assert [op.__class__.__name__ for step in again.operations for op in step] == ["StateInjection", "MeasureBlockSyndromes", "TransversalHadamard", "MeasureBlockSyndromes", "MeasureLogicalZ"]
    assert stim(loom.interpret_eka(again)) == EXPERIMENTS["si_s_then_hadamard"]["stim"]
    aux = build(EXPERIMENTS["auxcnot_tl_h_then_hadamard"]["spec"])
    again = loom.Eka.fromdict(json.loads(json.dumps(aux.asdict())), block_class=loom.RotatedSurfaceCode)
    assert detectors_apart(stim(loom.interpret_eka(again))) == detectors_apart(EXPERIMENTS["auxcnot_tl_h_then_hadamard"]["stim"])


# ---------------------------------------------------------------------------------------------
# The circuits
# ---------------------------------------------------------------------------------------------


def bell_pair(spec: dict) -> bool:
    """A CNOT from |+⟩ onto |0⟩: only the product of the two blocks' readings is fixed."""
    states = {kw["input_block_name"]: kw["state"] for n, kw in operations(spec) if n == "ResetAllDataQubits"}
    return has(spec, "AuxCNOT") and states.get("c") == "+"


def fixed_observables(spec: dict, n: int) -> list[list[int]]:
    """The observables an experiment fixes, as groups whose parity is fixed: none for an
    injection (|+i⟩ reads X or Z at random), the pair's product for a Bell pair, else each."""
    if has(spec, "StateInjection"):
        return []
    if bell_pair(spec):
        return [[0, 1]]
    return [[i] for i in range(n)]


def with_observables(text: str, groups: list[list[int]]) -> str:
    """The circuit with its observables replaced by the parities of groups of them (a
    measurement read twice cancels)."""
    lines = text.splitlines()
    observables = [line.split()[1:] for line in lines if line.startswith("OBSERVABLE_INCLUDE")]
    out = [line for line in lines if not line.startswith("OBSERVABLE_INCLUDE")]
    for k, group in enumerate(groups):
        parity: dict[str, int] = {}
        for i in group:
            for target in observables[i]:
                parity[target] = parity.get(target, 0) ^ 1
        out.append(f"OBSERVABLE_INCLUDE({k}) " + " ".join(t for t, v in parity.items() if v))
    return "\n".join(out)


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


CLIFFORD = sorted(name for name, ref in EXPERIMENTS.items() if "qasm" not in ref)


@pytest.mark.parametrize("name", CLIFFORD)
def test_detectors_are_deterministic_and_observables_what_the_operation_leaves(name):
    spec = EXPERIMENTS[name]["spec"]
    step = loom.interpret_eka(build(spec))
    text = stim(step, ticks=True)
    clean = sq.Circuit(with_observables(text, fixed_observables(spec, len(step.logical_observables))))
    clean.detector_error_model()  # raises on a detector or observable that is not deterministic
    d, o = clean.compile_detector_sampler(seed=1).sample(256, separate_observables=True)
    assert not d.any() and not o.any()
    if has(spec, "StateInjection") or bell_pair(spec):
        # Each reading alone is a coin: X or Z of |+i⟩, or one half of a Bell pair.
        _, o = sq.Circuit(text).compile_detector_sampler(seed=2).sample(2000, separate_observables=True)
        assert all(0.4 < m < 0.6 for m in o.mean(axis=0))


@pytest.mark.parametrize("name", sorted(name for name in CLIFFORD if "stim" in EXPERIMENTS[name]))
def test_a_noisy_version_decomposes_and_decodes(name):
    spec = EXPERIMENTS[name]["spec"]
    step = loom.interpret_eka(build(spec))
    circuit = noisy(with_observables(stim(step, ticks=True), fixed_observables(spec, len(step.logical_observables))), 0.001)
    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem.num_errors > 0 and dem.num_detectors == len(step.detectors)
    if dem.num_observables:
        d, o = circuit.compile_detector_sampler(seed=5).sample(2000, separate_observables=True)
        predicted = sq.Matching.from_detector_error_model(dem).decode_batch(d)
        assert (predicted != o).any(axis=1).mean() < 0.03


@pytest.mark.parametrize("name, basis, shots", [("si_t_d3_first_row_false", "X", 4000), ("si_t_d3_measure_z", "Z", 4000), ("si_t_d5_measure_x", "X", 1000)])
def test_an_injected_t_state_reads_as_one(name, basis, shots, monkeypatch):
    # Loom's Stim converter has no T gate; this package's circuits have it (I[T]), and its
    # stabilizer-rank sampler runs them exactly. |T⟩ = T|+⟩ has ⟨X⟩ = 1/√2 and ⟨Z⟩ = 0.
    monkeypatch.setitem(executor._STIM_OPS, "t", "I[T]")
    circuit = loom.eka_to_circuit(loom.interpret_eka(build(EXPERIMENTS[name]["spec"])))
    assert circuit.tcount() == 1
    d, o = circuit.compile_stabilizer_rank_sampler(seed=3).sample(shots, separate_observables=True)
    assert not d.any()
    expected = (1 + 1 / math.sqrt(2)) / 2 if basis == "X" else 0.5
    assert abs((1 - o.mean()) - expected) < 0.04
