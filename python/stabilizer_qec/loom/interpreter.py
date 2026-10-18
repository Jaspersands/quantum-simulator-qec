# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom/interpreter/{interpreter,interpretation_step,block_history,
# syndrome,detector,logical_observable}.py, src/loom/interpreter/applicator/*.py and
# src/loom_rotated_surface_code/applicator/{grow,shrink,merge,split}.py; Copyright 2024 Entropica Labs Pte Ltd,
# Apache License 2.0; see LICENSE in this directory), modified for stabilizer_qec: plain classes in place of
# pydantic, no plugin discovery, and the rotated surface code's other applicators (auxiliary CNOT, transversal
# Hadamard, move, rotate, Y-wall phase gate, state injection) refused with Loom named.
"""Loom's interpreter: operations on blocks become a circuit (in Loom's IR) with syndromes,
detectors and logical observables, bookkept through the blocks' evolution (which stabilizer
grew or shrank into which, which measurements correct a logical operator after a merge or
split)."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from itertools import compress, product
from typing import Any, Callable, Iterable, Optional, Sequence
from uuid import uuid4

from .circuit import Channel, ChannelType, Circuit
from .eka import (
    Block,
    CodeOperation,
    ConditionalLogicalX,
    ConditionalLogicalY,
    ConditionalLogicalZ,
    Coord,
    Direction,
    Eka,
    Grow,
    LogicalMeasurement,
    LogicalX,
    LogicalY,
    LogicalZ,
    MeasureBlockSyndromes,
    MeasureLogicalX,
    MeasureLogicalY,
    MeasureLogicalZ,
    Merge,
    Operation,
    Orientation,
    PauliOperator,
    ResetAllAncillaQubits,
    ResetAllDataQubits,
    Shrink,
    Split,
    Stabilizer,
)
from .rotated_surface_code import RotatedSurfaceCode

Cbit = tuple[str, int]


class SyndromeMissingError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------


@dataclass
class Syndrome:
    """One measurement of a stabilizer of a block in a round: the classical bits whose parity
    it is, the corrections (bits of data measurements the stabilizer absorbed in a surgery),
    and labels (space and time coordinates)."""

    stabilizer: str
    measurements: tuple[Cbit, ...]
    block: str
    round: int
    corrections: tuple[Cbit, ...] = ()
    labels: dict = field(default_factory=dict)
    uuid: str = field(default_factory=lambda: str(uuid4()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Syndrome):
            return NotImplemented
        return self.stabilizer == other.stabilizer and set(self.measurements) == set(other.measurements) and set(self.corrections) == set(other.corrections) and self.block == other.block and self.round == other.round

    def __hash__(self) -> int:
        return hash((self.stabilizer, frozenset(self.measurements), self.block, self.round))

    def __repr__(self) -> str:
        return f"Syndrome(Measurements: {self.measurements}, Corrections: {self.corrections}, Round: {self.round}, Labels: {self.labels})"


@dataclass
class Detector:
    """The parity of consecutive syndromes of one stabilizer: its measurements are theirs, plus
    the last one's corrections."""

    syndromes: tuple[Syndrome, ...]
    labels: dict = field(default_factory=dict)

    @property
    def measurements(self) -> tuple[Cbit, ...]:
        out: list[Cbit] = []
        for s in self.syndromes:
            out.extend(s.measurements)
        out.extend(self.syndromes[-1].corrections)
        return tuple(out)

    def rounds(self) -> tuple[int, ...]:
        return tuple(sorted({s.round for s in self.syndromes}))

    def stabilizer(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.stabilizer for s in self.syndromes))


@dataclass
class LogicalObservable:
    label: str
    measurements: tuple[Cbit, ...]


# ---------------------------------------------------------------------------------------------
# Block history
# ---------------------------------------------------------------------------------------------


class BlockHistory:
    """Which blocks (by uuid) exist at each timestamp."""

    def __init__(self, blocks_at_0: set[str]) -> None:
        self._by_timestamp: dict[int, set[str]] = {0: set(blocks_at_0)}
        self._timestamps: list[int] = [0]
        self._all: set[str] = set(blocks_at_0)

    def blocks_at(self, timestamp: int) -> set[str]:
        idx = bisect.bisect_right(self._timestamps, timestamp) - 1
        if idx < 0:
            raise IndexError(f"No blocks before timestamp {timestamp}.")
        return set(self._by_timestamp[self._timestamps[idx]])

    def block_uuids_at_index(self, index: int) -> set[str]:
        return set(self._by_timestamp[self._timestamps[index]])

    def update_blocks(self, timestamp: int, old_blocks: set[str], new_blocks: set[str]) -> None:
        current = self.blocks_at(timestamp)
        missing = old_blocks - current
        if missing:
            raise RuntimeError(f"Failed to update block history. Some old_blocks are not present in the current block configuration: {sorted(missing)}.")
        seen = new_blocks & self._all
        if seen:
            raise RuntimeError(f"Failed to update block history. Some new_blocks have already been present in the block history: {sorted(seen)}.")
        updated = (current - old_blocks) | new_blocks
        if timestamp not in self._by_timestamp:
            bisect.insort(self._timestamps, timestamp)
        self._by_timestamp[timestamp] = updated
        for later in self._timestamps:
            if later > timestamp:
                later_set = self._by_timestamp[later]
                if not old_blocks <= later_set:
                    raise RuntimeError(f"Inconsistent block update detected at timestamp {later}.")
                self._by_timestamp[later] = (later_set - old_blocks) | new_blocks
        self._all |= new_blocks


# ---------------------------------------------------------------------------------------------
# The interpretation step
# ---------------------------------------------------------------------------------------------


@dataclass
class CompositeOperationSession:
    start_timeslice_index: int
    same_timeslice: bool
    circuit_name: str


class InterpretationStep:
    """The state of an interpretation: the circuit so far as time slices of composite circuits,
    the blocks and their history, the syndromes, detectors and logical observables, and the
    evolutions (stabilizer to its predecessors, logical to its predecessors, block to its
    predecessors) and updates (classical bits that correct a stabilizer's or a logical's next
    reading) the applicators leave behind."""

    def __init__(self, initial_blocks: Sequence[Block]) -> None:
        if not isinstance(initial_blocks, Sequence) or not all(isinstance(b, Block) for b in initial_blocks):
            raise TypeError("All elements of initial_blocks must be Block objects.")
        if len(initial_blocks) != len({b.uuid for b in initial_blocks}):
            raise ValueError("All blocks in initial_blocks must have distinct UUIDs.")
        self.intermediate_circuit_sequence: tuple[tuple[Circuit, ...], ...] = ()
        self.final_circuit: Optional[Circuit] = None
        self.block_history = BlockHistory({b.uuid for b in initial_blocks})
        self.block_registry: dict[str, Block] = {b.uuid: b for b in initial_blocks}
        self.syndromes: tuple[Syndrome, ...] = ()
        self.detectors: tuple[Detector, ...] = ()
        self.logical_observables: tuple[LogicalObservable, ...] = ()
        self.stabilizer_evolution: dict[str, tuple[str, ...]] = {}
        self.logical_x_evolution: dict[str, tuple[str, ...]] = {}
        self.logical_z_evolution: dict[str, tuple[str, ...]] = {}
        self.block_evolution: dict[str, tuple[str, ...]] = {}
        self.block_qec_rounds: dict[str, int] = {}
        self.cbit_counter: dict[str, int] = {}
        self.logical_x_operator_updates: dict[str, tuple[Cbit, ...]] = {}
        self.logical_z_operator_updates: dict[str, tuple[Cbit, ...]] = {}
        self.stabilizer_updates: dict[str, tuple[Cbit, ...]] = {}
        self.reset_single_qubit_stabilizers: dict[str, set[Stabilizer]] = {}
        self.measured_single_qubit_stabilizers: dict[str, set[Stabilizer]] = {}
        self.channel_dict: dict[str, Channel] = {}
        self.logical_measurements: dict[LogicalMeasurement, tuple[Cbit, ...]] = {}
        self.composite_operation_session_stack: list[CompositeOperationSession] = []
        self.timeslice_durations: list[int] = []
        self.is_frozen = False

    @classmethod
    def create(cls, initial_blocks: Sequence[Block]) -> "InterpretationStep":
        return cls(initial_blocks)

    # Blocks
    def get_block(self, label: str) -> Block:
        for uuid in self.block_history.blocks_at(self.get_timestamp()):
            block = self.block_registry[uuid]
            if block.unique_label == label:
                return block
        raise RuntimeError(f"No block with label '{label}' found in the current configuration.")

    def get_blocks_at_index(self, index: int) -> tuple[Block, ...]:
        return tuple(self.block_registry[u] for u in self.block_history.block_uuids_at_index(index))

    @property
    def final_blocks(self) -> tuple[Block, ...]:
        """The blocks at the end, in insertion order of their uuids' first appearance."""
        uuids = self.block_history.block_uuids_at_index(-1)
        return tuple(self.block_registry[u] for u in self.block_registry if u in uuids)

    def update_block_history_and_evolution_MUT(self, new_blocks: Sequence[Block] = (), old_blocks: Sequence[Block] = (), update_evolution: bool = True) -> None:
        for arg in (new_blocks, old_blocks):
            if not all(isinstance(b, Block) for b in arg):
                raise TypeError("All elements of new_blocks/old_blocks must be Block objects.")
            if len(arg) != len({b.uuid for b in arg}):
                raise ValueError("All blocks in new_blocks/old_blocks must have distinct UUIDs.")
        self.block_registry.update({b.uuid: b for b in new_blocks})
        self.block_history.update_blocks(self.get_timestamp(), {b.uuid for b in old_blocks}, {b.uuid for b in new_blocks})
        if old_blocks and new_blocks and update_evolution:
            for nb in new_blocks:
                self.block_evolution[nb.uuid] = tuple(b.uuid for b in old_blocks)

    # Logical operator updates
    def update_logical_operator_updates_MUT(self, operator_type: str, logical_operator_id: str, new_updates: tuple[Cbit, ...], inherit_updates: bool) -> None:
        if operator_type == "X":
            evolution, updates = self.logical_x_evolution, self.logical_x_operator_updates
        elif operator_type == "Z":
            evolution, updates = self.logical_z_evolution, self.logical_z_operator_updates
        else:
            raise ValueError("Operator type must be labelled either 'X' or 'Z'.")
        new_updates = tuple(new_updates)
        if inherit_updates:
            old_ids = evolution.get(logical_operator_id, ())
            new_updates += tuple(cbit for old in old_ids for cbit in updates.get(old, ()))
        if new_updates:
            updates[logical_operator_id] = updates.get(logical_operator_id, ()) + new_updates

    # Channels, circuits and time
    def get_channel_MUT(self, label, channel_type=ChannelType.QUANTUM) -> Channel:
        label = str(label)
        if label not in self.channel_dict:
            self.channel_dict[label] = Channel(type=ChannelType(channel_type), label=label)
        return self.channel_dict[label]

    def append_circuit_MUT(self, circuit: Circuit, same_timeslice: bool = False) -> None:
        if not isinstance(circuit, Circuit):
            raise TypeError("The circuit must be a Circuit object")
        first_of_sessions = [s for s in self.composite_operation_session_stack if s.start_timeslice_index == len(self.intermediate_circuit_sequence)]
        if same_timeslice and first_of_sessions:
            raise ValueError(f"The first circuit of a composite operation session cannot be added to the same timeslice as the previous circuit. Please set same_timeslice to False for the first circuit of composite operation with circuit name '{first_of_sessions[-1].circuit_name}'.")
        if same_timeslice and self.intermediate_circuit_sequence:
            existing = [ch for c in self.intermediate_circuit_sequence[-1] for ch in c.channels]
            if any(ch in existing for ch in circuit.channels):
                raise ValueError("The channels of the new circuit are already in use in the current timeslice. Please use a new timeslice.")
            self.intermediate_circuit_sequence = self.intermediate_circuit_sequence[:-1] + (self.intermediate_circuit_sequence[-1] + (circuit,),)
            self.timeslice_durations[-1] = max(self.timeslice_durations[-1], circuit.duration)
        else:
            self.intermediate_circuit_sequence += ((circuit,),)
            self.timeslice_durations.append(circuit.duration)

    def get_timestamp(self) -> int:
        omit = {s.start_timeslice_index - 1 for s in self.composite_operation_session_stack if s.same_timeslice}
        n = len(self.intermediate_circuit_sequence)
        timestamp = sum(compress(self.timeslice_durations, (i not in omit for i in range(n - 1))))
        if self.intermediate_circuit_sequence:
            timestamp += self.intermediate_circuit_sequence[-1][-1].duration
        return timestamp

    def get_new_cbit_MUT(self, register_name: str) -> Cbit:
        count = self.cbit_counter.get(register_name, 0)
        self.cbit_counter[register_name] = count + 1
        return (register_name, count)

    def _pop_intermediate_circuit_MUT(self, length: int) -> tuple[tuple[Circuit, ...], ...]:
        if length > len(self.intermediate_circuit_sequence):
            raise ValueError("The number of timeslices to pop exceeds the number of timeslices in the intermediate circuit sequence.")
        if length == 0:
            raise ValueError("The number of timeslices to pop must be greater than 0.")
        popped = self.intermediate_circuit_sequence[-length:]
        self.intermediate_circuit_sequence = self.intermediate_circuit_sequence[:-length]
        self.timeslice_durations = self.timeslice_durations[:-length]
        return popped

    def begin_composite_operation_session_MUT(self, same_timeslice: bool, circuit_name: str) -> None:
        self.composite_operation_session_stack.append(CompositeOperationSession(len(self.intermediate_circuit_sequence), same_timeslice, circuit_name))

    def end_composite_operation_session_MUT(self) -> Circuit:
        if not self.composite_operation_session_stack:
            raise ValueError("No composite operation session to end. Please begin a session first.")
        session = self.composite_operation_session_stack.pop()
        length = len(self.intermediate_circuit_sequence) - session.start_timeslice_index
        sequence = self._pop_intermediate_circuit_MUT(length)
        wrapped: list[tuple[Circuit, ...]] = []
        for timeslice in sequence:
            span = max(c.duration for c in timeslice)
            wrapped.append(tuple(timeslice))
            wrapped.extend([()] * (span - 1))
        return Circuit(name=session.circuit_name, circuit=tuple(wrapped))

    # Syndromes and detectors
    def append_syndromes_MUT(self, syndromes) -> None:
        if isinstance(syndromes, Syndrome):
            syndromes = (syndromes,)
        syndromes = tuple(syndromes)
        if any(not isinstance(s, Syndrome) for s in syndromes):
            raise TypeError("All elements in the tuple must be Syndrome objects.")
        self.syndromes += syndromes

    def append_detectors_MUT(self, detectors) -> None:
        if isinstance(detectors, Detector):
            detectors = (detectors,)
        detectors = tuple(detectors)
        if any(not isinstance(d, Detector) for d in detectors):
            raise TypeError("Some elements in the input tuple are not of type Detector")
        self.detectors += detectors

    def get_all_syndromes(self, stab_id: str, block_id: str) -> list[Syndrome]:
        return [s for s in self.syndromes if s.stabilizer == stab_id and s.block == block_id]

    def get_prev_syndrome(self, stabilizer_id: str, block_id: str, current_round: Optional[int] = None) -> list[Syndrome]:
        """The latest syndromes of a stabilizer in a block, following the stabilizer's and the
        block's evolutions back when it has none of its own."""
        found: list[Syndrome] = []
        stabs = [stabilizer_id]
        while stabs and not found:
            blocks = [block_id]
            while blocks and not found:
                found = [s for b in blocks for st in stabs for s in self.get_all_syndromes(st, b)]
                blocks = [prev for b in blocks for prev in self.block_evolution.get(b, ())]
            stabs = [prev for st in stabs for prev in self.stabilizer_evolution.get(st, ())]
        if current_round is not None:
            found = [s for s in found if s.round < current_round]
        if not found:
            return []
        latest = max(s.round for s in found)
        return [s for s in found if s.round == latest]

    def retrieve_cbits_from_stabilizers(self, stabs_required: Sequence[Stabilizer], current_block: Block) -> tuple[Cbit, ...]:
        last = [self.get_prev_syndrome(s.uuid, current_block.uuid) for s in stabs_required]
        missing = [s for s, l in zip(stabs_required, last) if not l]
        if missing:
            raise SyndromeMissingError(f"Could not find a syndrome for some stabilizers. Stabilizers without syndrome: {missing}")
        return tuple(cbit for l in last for cbit in l[0].measurements)

    def update_reset_single_qubit_stabilizers_MUT(self, block_id: str, new_single_qubit_stabilizers: set[Stabilizer]) -> None:
        self._update_single(self.reset_single_qubit_stabilizers, block_id, new_single_qubit_stabilizers)

    def update_measured_single_qubit_stabilizers_MUT(self, block_id: str, new_single_qubit_stabilizers: set[Stabilizer]) -> None:
        self._update_single(self.measured_single_qubit_stabilizers, block_id, new_single_qubit_stabilizers)

    def _update_single(self, store: dict[str, set[Stabilizer]], block_id: str, stabs: set[Stabilizer]) -> None:
        for s in stabs:
            if not isinstance(s, Stabilizer) or len(s.data_qubits) != 1:
                raise ValueError("Each single-qubit stabilizer must contain exactly one data qubit.")
        if block_id not in self.block_history.block_uuids_at_index(-1):
            raise ValueError(f"Block {block_id} not present at current timestep in block history.")
        store[block_id] = store.get(block_id, set()) | set(stabs)


# ---------------------------------------------------------------------------------------------
# Generic applicators
# ---------------------------------------------------------------------------------------------


def generate_syndromes(step: InterpretationStep, stabilizers: Sequence[Stabilizer], block: Block, stab_measurements: Sequence[tuple[Cbit, ...]]) -> tuple[Syndrome, ...]:
    """New syndromes of the stabilizers in the block's next round, each with the corrections
    waiting for its stabilizer (popped from the step)."""
    if block.uuid in step.block_qec_rounds:
        new_round = step.block_qec_rounds[block.uuid]
        step.block_qec_rounds[block.uuid] += 1
    else:
        new_round = 0
        step.block_qec_rounds[block.uuid] = 1
    time_stamp = step.get_timestamp()
    return tuple(
        Syndrome(stabilizer=stab.uuid, measurements=tuple(meas), block=block.uuid, round=new_round, labels={**block.get_stabilizer_label(stab.uuid), "time_coordinate": (time_stamp,)}, corrections=step.stabilizer_updates.pop(stab.uuid, ()))
        for stab, meas in zip(stabilizers, stab_measurements)
    )


def generate_detectors(step: InterpretationStep, new_syndromes: Sequence[Syndrome]) -> tuple[Detector, ...]:
    """A detector for each new syndrome that has a previous one to compare with."""
    out = []
    for s in new_syndromes:
        old = step.get_prev_syndrome(s.stabilizer, s.block)
        if old:
            out.append(Detector(syndromes=tuple(old) + (s,), labels=dict(s.labels)))
    return tuple(out)


def reset_all_data_qubits(step: InterpretationStep, operation: ResetAllDataQubits, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    before = step.get_block(operation.input_block_name)
    block = before.rename(operation.input_block_name)
    reset_circuit = Circuit(name=f"reset all data qubits of block {block.unique_label} to |{operation.state.value}>", circuit=[[Circuit(f"reset_{operation.state.value}", channels=step.get_channel_MUT(q)) for q in block.data_qubits]])
    single = {Stabilizer(pauli=operation.state.pauli_basis, data_qubits=(q,)) for q in block.data_qubits}
    relevant = [s for s in block.stabilizers if set(s.pauli) == {operation.state.pauli_basis}]
    values = [((1,) if (operation.state.basis_expectation_value == -1 and len(s.pauli) % 2 == 1) else ()) for s in relevant]
    new_syndromes = generate_syndromes(step, relevant, block, values)
    step.append_syndromes_MUT(new_syndromes)
    step.append_circuit_MUT(reset_circuit, same_timeslice)
    step.update_block_history_and_evolution_MUT(old_blocks=(before,), new_blocks=(block,), update_evolution=False)
    step.update_reset_single_qubit_stabilizers_MUT(block.uuid, single)
    return step


def reset_all_ancilla_qubits(step: InterpretationStep, operation: ResetAllAncillaQubits, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    block = step.get_block(operation.input_block_name)
    circuit = Circuit(name=f"reset all ancilla qubits of block {block.unique_label} to |{operation.state.value}>", circuit=[[Circuit(f"reset_{operation.state.value}", channels=step.get_channel_MUT(q)) for q in block.ancilla_qubits]])
    step.append_circuit_MUT(circuit, same_timeslice)
    return step


def measureblocksyndromes(step: InterpretationStep, operation: MeasureBlockSyndromes, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """``n_cycles`` rounds of every stabilizer's syndrome circuit, woven tick by tick."""
    step.begin_composite_operation_session_MUT(same_timeslice=same_timeslice, circuit_name=f"measure {operation.input_block_name} syndromes {operation.n_cycles} time(s)")
    block = step.get_block(operation.input_block_name)
    stabilizers = block.stabilizers
    by_uuid = {c.uuid: c for c in block.syndrome_circuits}
    templates = [by_uuid[block.stabilizer_to_circuit[s.uuid]] for s in stabilizers]
    data_channels = [[step.get_channel_MUT(str(q), "quantum") for q in s.data_qubits] for s in stabilizers]
    ancilla_channels = [[step.get_channel_MUT(str(q), "quantum") for q in s.ancilla_qubits] for s in stabilizers]
    cbit_labels = [[str(q) for q in s.ancilla_qubits] for s in stabilizers]
    for idx in range(operation.n_cycles):
        cbit_channels, measurements = [], []
        for labels in cbit_labels:
            cbit = step.get_new_cbit_MUT("c_" + labels[0])
            cbit_channels.append(step.get_channel_MUT(f"{cbit[0]}_{cbit[1]}", "classical"))
            measurements.append(cbit)
        mapped = [t.circuit.clone(data_channels[i] + ancilla_channels[i] + [cbit_channels[i]]) for i, t in enumerate(templates)]
        if not all(len(c.circuit) == len(mapped[0].circuit) for c in mapped):
            raise ValueError("All syndrome circuits must be of the same length.")
        woven = [[gate for c in mapped for gate in c.circuit[i]] for i in range(len(mapped[0].circuit))]
        step.append_circuit_MUT(Circuit(name=f"measure {block.unique_label} syndromes - cycle {idx}", circuit=woven), same_timeslice=False)
        new_syndromes = generate_syndromes(step, stabilizers, block, [(m,) for m in measurements])
        new_detectors = generate_detectors(step, new_syndromes)
        step.append_syndromes_MUT(new_syndromes)
        step.append_detectors_MUT(new_detectors)
    woven_circuit = step.end_composite_operation_session_MUT()
    step.append_circuit_MUT(woven_circuit, same_timeslice)
    return step


def measurelogicalpauli(step: InterpretationStep, operation, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """Measure every data qubit in the basis (H first for X); the stabilizers of that type get
    syndromes and detectors from the data bits; the logical observable is the logical
    operator's data bits plus its corrections."""
    block = step.get_block(operation.input_block_name)
    index = operation.logical_qubit
    name = operation.__class__.__name__
    if name == "MeasureLogicalX":
        basis = "X"
    elif name == "MeasureLogicalZ":
        basis = "Z"
    elif name == "MeasureLogicalY":
        raise ValueError("Logical measurement in Y basis is not supported")
    else:
        raise ValueError(f"Operation {name} not supported")
    seq = []
    if basis == "X":
        seq.append([Circuit("H", channels=step.get_channel_MUT(str(q), "quantum")) for q in block.data_qubits])
    measurements = tuple(step.get_new_cbit_MUT(f"c_{q}") for q in block.data_qubits)
    seq.append([Circuit("measurement", channels=[step.get_channel_MUT(str(q), "quantum"), step.get_channel_MUT(f"{c[0]}_{c[1]}", "classical")]) for q, c in zip(block.data_qubits, measurements)])
    step.append_circuit_MUT(Circuit(f"Measure logical {basis} of {block.unique_label}", circuit=seq), same_timeslice)
    single = {Stabilizer(pauli=basis, data_qubits=(q,)) for q in block.data_qubits}
    step.update_measured_single_qubit_stabilizers_MUT(block.uuid, single)
    relevant = [s for s in block.stabilizers if set(s.pauli) == {basis}]
    stab_cbits = [[c for c in measurements if c[0].split("_", 1)[1] in map(str, s.data_qubits)] for s in relevant]
    new_syndromes = generate_syndromes(step, relevant, block, stab_cbits)
    new_detectors = generate_detectors(step, new_syndromes)
    if basis == "X":
        logical, updates = block.logical_x_operators[index], step.logical_x_operator_updates
    else:
        logical, updates = block.logical_z_operators[index], step.logical_z_operator_updates
    corrections = list(updates.get(logical.uuid, ()))
    qubits = [str(q) for q in logical.data_qubits]
    logical_measurements = [c for c in measurements if c[0].split("_", 1)[1] in qubits]
    observable = LogicalObservable(label=f"{block.unique_label}_{basis}_{index}", measurements=tuple(logical_measurements + corrections))
    step.logical_measurements[LogicalMeasurement(blocks=(block.unique_label,), observable=basis)] = tuple(logical_measurements + corrections)
    step.append_detectors_MUT(new_detectors)
    step.append_syndromes_MUT(new_syndromes)
    step.logical_observables += (observable,)
    return step


def logical_pauli(step: InterpretationStep, operation, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    block = step.get_block(operation.input_block_name)
    index = operation.logical_qubit
    if index >= block.n_logical_qubits:
        raise ValueError(f"Logical qubit {index} does not exist in block {block.unique_label}")
    name = operation.__class__.__name__
    if name == "LogicalX":
        operators = [block.logical_x_operators[index]]
    elif name == "LogicalZ":
        operators = [block.logical_z_operators[index]]
    elif name == "LogicalY":
        operators = [block.logical_x_operators[index], block.logical_z_operators[index]]
    else:
        raise ValueError(f"Operation {name} not supported")
    circuit = Circuit(name=f"{name} on block {block.unique_label}, logical qubit {index}", circuit=[[Circuit(p, channels=[step.get_channel_MUT(q)]) for q, p in zip(op.data_qubits, op.pauli)] for op in operators])
    step.append_circuit_MUT(circuit, same_timeslice)
    return step


def conditional_logical_pauli(step: InterpretationStep, operation, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    block = step.get_block(operation.input_block_name)
    index = operation.logical_qubit
    if index >= block.n_logical_qubits:
        raise ValueError(f"Logical qubit {index} does not exist in block {block.unique_label}")
    update = step.logical_measurements.get(operation.condition)
    if update is None:
        raise ValueError(f"Logical Measurement {operation.condition} has not yet been made")
    name = operation.__class__.__name__
    if name in ("ConditionalLogicalX", "ConditionalLogicalY"):
        step.update_logical_operator_updates_MUT("Z", block.logical_z_operators[index].uuid, update, inherit_updates=False)
    if name in ("ConditionalLogicalZ", "ConditionalLogicalY"):
        step.update_logical_operator_updates_MUT("X", block.logical_x_operators[index].uuid, update, inherit_updates=False)
    if name not in ("ConditionalLogicalX", "ConditionalLogicalY", "ConditionalLogicalZ"):
        raise ValueError(f"Operation {name} not supported")
    return step


# ---------------------------------------------------------------------------------------------
# Rotated surface code: grow, shrink, merge, split
# ---------------------------------------------------------------------------------------------


def _rsc(block: Block, what: str) -> RotatedSurfaceCode:
    if not isinstance(block, RotatedSurfaceCode):
        raise TypeError(f"The {what} operation is not supported for {type(block).__name__} blocks.")
    return block


def grow(step: InterpretationStep, operation: Grow, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    block = _rsc(step.get_block(operation.input_block_name), "grow")
    stabilizers = block.stabilizers
    direction, length = operation.direction, operation.length
    old_x, old_z = block.logical_x_operators, block.logical_z_operators
    horizontal = direction in (Direction.LEFT, Direction.RIGHT)
    dx = length + 1 if horizontal else block.size[0]
    dz = length + 1 if not horizontal else block.size[1]
    x_sched, z_sched = block.weight_4_x_schedule, block.weight_4_z_schedule
    pos = list(block.upper_left_qubit[:2])
    if direction == Direction.LEFT:
        pos[0] -= length
    elif direction == Direction.RIGHT:
        pos[0] += block.size[0] - 1
    elif direction == Direction.TOP:
        pos[1] -= length
    else:
        pos[1] += block.size[1] - 1
    parity = length * (direction in (Direction.TOP, Direction.LEFT)) + (block.size[0] - 1) * (direction == Direction.RIGHT) + (block.size[1] - 1) * (direction == Direction.BOTTOM)
    if parity % 2 == 0:
        new_ul_pauli = block.upper_left_4body_stabilizer.pauli
    else:
        new_ul_pauli = "ZZZZ" if block.upper_left_4body_stabilizer.pauli == "XXXX" else "XXXX"
    new_ul = RotatedSurfaceCode.generate_weight4_stabs(dx, dz, new_ul_pauli, x_sched if new_ul_pauli == "XXXX" else z_sched, True, pos)
    new_rest = RotatedSurfaceCode.generate_weight4_stabs(dx, dz, "XXXX" if new_ul_pauli == "ZZZZ" else "ZZZZ", x_sched if new_ul_pauli != "XXXX" else z_sched, False, pos)
    new_4 = new_ul + new_rest
    new_top_left_is_xxxx = new_ul_pauli == "XXXX"
    lr_is_x = block.x_boundary == Orientation.HORIZONTAL
    new_w2_first_row = new_top_left_is_xxxx != lr_is_x
    lr = "XX" if lr_is_x else "ZZ"
    tb = "ZZ" if lr_is_x else "XX"
    if direction != Direction.RIGHT:
        num = (dz - 1) / 2 if dz % 2 == 1 else dz / 2 - (not new_w2_first_row)
        stabs_left = RotatedSurfaceCode.generate_weight2_stabs(lr, (pos[0], pos[1] + (not new_w2_first_row)), num, Orientation.VERTICAL, False)
    else:
        stabs_left = []
    if direction != Direction.LEFT:
        right_first_row = new_w2_first_row != (dx % 2)
        num = (dz - 1) / 2 if dz % 2 == 1 else dz / 2 - (not right_first_row)
        stabs_right = RotatedSurfaceCode.generate_weight2_stabs(lr, (pos[0] + dx - 1, pos[1] + (not right_first_row)), num, Orientation.VERTICAL, True)
    else:
        stabs_right = []
    if direction != Direction.BOTTOM:
        num = (dx - 1) / 2 if dx % 2 == 1 else dx / 2 - new_w2_first_row
        stabs_top = RotatedSurfaceCode.generate_weight2_stabs(tb, (pos[0] + new_w2_first_row, pos[1]), num, Orientation.HORIZONTAL, False)
    else:
        stabs_top = []
    if direction != Direction.TOP:
        bottom_first_col = new_w2_first_row == (dz % 2)
        num = (dx - 1) / 2 if dx % 2 == 1 else dx / 2 - (not bottom_first_col)
        stabs_bottom = RotatedSurfaceCode.generate_weight2_stabs(tb, (pos[0] + (not bottom_first_col), pos[1] + dz - 1), num, Orientation.HORIZONTAL, True)
    else:
        stabs_bottom = []
    boundary = block.boundary_qubits(direction)
    to_remove = [s for s in stabilizers if set(s.data_qubits) <= set(boundary) and len(s.data_qubits) == 2]
    remove_uuids = [s.uuid for s in to_remove]
    replace_uuids = [s.uuid for old in to_remove for s in new_4 if set(old.data_qubits) <= set(s.data_qubits) and old.pauli in s.pauli]
    step.stabilizer_evolution.update({replace_uuids[i]: (u,) for i, u in enumerate(remove_uuids)})
    new_2 = stabs_left + stabs_right + stabs_top + stabs_bottom
    new_stabilizers = [s for s in stabilizers if s not in to_remove] + new_2 + new_4
    reset_state = "0" if block.boundary_type(direction) == "X" else "+"
    # Loom collects these in a set; the order of the resets follows (a set of int tuples
    # iterates the same way here, so the circuits come out gate for gate the same).
    new_data = tuple(set(q for s in new_4 for q in s.data_qubits if q not in boundary))
    existing_ancillas = tuple(q for s in to_remove for q in s.ancilla_qubits)
    grow_circuit = Circuit(name=f"grow {block.unique_label} by {length} to the {direction.value}", circuit=[[Circuit(name=f"reset_{reset_state}", channels=step.get_channel_MUT(q, "quantum")) for q in new_data]])
    step.append_circuit_MUT(grow_circuit, same_timeslice)
    det_pauli = "X" if reset_state == "+" else "Z"
    relevant = [s for s in new_2 + new_4 if set(s.pauli) == {det_pauli} and not all(a in existing_ancillas for a in s.ancilla_qubits)]
    step.append_syndromes_MUT(generate_syndromes(step, relevant, block, [() for _ in relevant]))
    circ = {c.name: c for c in block.syndrome_circuits}
    new_stc = {}
    new_stc.update({s.uuid: (circ["xxxx"].uuid if new_ul[0].pauli == "XXXX" else circ["zzzz"].uuid) for s in new_ul} if new_ul else {})
    new_stc.update({s.uuid: (circ["zzzz"].uuid if new_rest[0].pauli == "ZZZZ" else circ["xxxx"].uuid) for s in new_rest} if new_rest else {})
    new_stc.update({s.uuid: circ[f"top-{tb.lower()}"].uuid for s in stabs_top})
    new_stc.update({s.uuid: circ[f"bottom-{tb.lower()}"].uuid for s in stabs_bottom})
    new_stc.update({s.uuid: circ[f"left-{lr.lower()}"].uuid for s in stabs_left})
    new_stc.update({s.uuid: circ[f"right-{lr.lower()}"].uuid for s in stabs_right})
    new_stc.update({k: v for k, v in block.stabilizer_to_circuit.items() if k not in remove_uuids})
    if block.boundary_type(direction) == "X":
        update_type, untouched, to_modify = "Z", old_x, old_z
    else:
        update_type, untouched, to_modify = "X", old_z, old_x
    involved = list(to_modify[0].data_qubits)
    extend = next(q for q in boundary if q in involved)
    if direction == Direction.LEFT:
        new_qubits = involved + [(extend[0] - i, extend[1], extend[2]) for i in range(1, length + 1)]
    elif direction == Direction.RIGHT:
        new_qubits = involved + [(extend[0] + i, extend[1], extend[2]) for i in range(1, length + 1)]
    elif direction == Direction.TOP:
        new_qubits = involved + [(extend[0], extend[1] - i, extend[2]) for i in range(1, length + 1)]
    else:
        new_qubits = involved + [(extend[0], extend[1] + i, extend[2]) for i in range(1, length + 1)]
    new_logical = PauliOperator(update_type * len(new_qubits), new_qubits)
    if update_type == "X":
        new_x, new_z = (new_logical,), untouched
        step.logical_x_evolution[new_logical.uuid] = (to_modify[0].uuid,)
        step.update_logical_operator_updates_MUT("X", new_logical.uuid, (), True)
    else:
        new_x, new_z = untouched, (new_logical,)
        step.logical_z_evolution[new_logical.uuid] = (to_modify[0].uuid,)
        step.update_logical_operator_updates_MUT("Z", new_logical.uuid, (), True)
    new_block = RotatedSurfaceCode(unique_label=block.unique_label, stabilizers=new_stabilizers, logical_x_operators=tuple(new_x), logical_z_operators=tuple(new_z), syndrome_circuits=block.syndrome_circuits, stabilizer_to_circuit=new_stc, skip_validation=not debug_mode)
    step.update_block_history_and_evolution_MUT((new_block,), (block,))
    return step


def shrink(step: InterpretationStep, operation: Shrink, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    block = _rsc(step.get_block(operation.input_block_name), "shrink")
    horizontal = Orientation.from_direction(operation.direction) == Orientation.HORIZONTAL
    distance = block.size[0] if horizontal else block.size[1]
    if operation.length < 1 or operation.length > distance - 2:
        raise ValueError(f"Shrink length {operation.length} is not valid. Must be between 1 and {distance - 2} for the selected block.")
    boundary_type = block.boundary_type(operation.direction)
    shifts = {Direction.LEFT: [(i, 0, 0) for i in range(operation.length)], Direction.RIGHT: [(-i, 0, 0) for i in range(operation.length)], Direction.TOP: [(0, i, 0) for i in range(operation.length)], Direction.BOTTOM: [(0, -i, 0) for i in range(operation.length)]}[operation.direction]
    to_measure = [tuple(a + b for a, b in zip(q, v)) for q, v in product(block.boundary_qubits(operation.direction), shifts)]
    cbits = [step.get_new_cbit_MUT(f"c_{q}") for q in to_measure]
    cbit_channels = [step.get_channel_MUT(f"{c[0]}_{c[1]}", "classical") for c in cbits]
    measure_seq = [[Circuit("Measurement", channels=[step.get_channel_MUT(q), cbit_channels[i]]) for i, q in enumerate(to_measure)]]
    circuit_list = measure_seq if boundary_type == "X" else [[Circuit("H", channels=[step.get_channel_MUT(q)]) for q in to_measure]] + measure_seq
    step.append_circuit_MUT(Circuit(name=f"Shrink {block.unique_label} by {operation.length} from {operation.direction.value}", circuit=circuit_list), same_timeslice)
    to_remove = [s for s in block.stabilizers if any(q in to_measure for q in s.data_qubits)]
    to_reduce = [s for s in block.stabilizers if s.pauli[0] != boundary_type and len([q for q in s.data_qubits if q in to_measure]) == 2 and len(s.data_qubits) == 4]
    reduced = [Stabilizer(pauli="".join(s.pauli[i] for i, q in enumerate(s.data_qubits) if q not in to_measure), data_qubits=[q for q in s.data_qubits if q not in to_measure], ancilla_qubits=s.ancilla_qubits) for s in to_reduce]
    step.stabilizer_evolution.update({reduced[i].uuid: (s.uuid,) for i, s in enumerate(to_reduce)})
    for i, s in enumerate(reduced):
        previous = step.stabilizer_updates.get(s.uuid, ())
        old = to_reduce[i]
        measured = [q for q in old.data_qubits if q in to_measure]
        step.stabilizer_updates[s.uuid] = previous + tuple(cbits[to_measure.index(q)] for q in measured)
    new_stabs = set(block.stabilizers) - set(to_remove) | set(reduced)
    fully = [s for s in block.stabilizers if all(q in to_measure for q in s.data_qubits) and set(s.pauli) == ({"Z"} if boundary_type == "X" else {"X"})]
    syndromes = generate_syndromes(step, fully, block, [tuple(c for q, c in product(s.data_qubits, cbits) if c[0].split("_", 1)[1] == str(q)) for s in fully])
    detectors = generate_detectors(step, syndromes)
    step.append_syndromes_MUT(syndromes)
    step.append_detectors_MUT(detectors)

    def shorten(old_op: PauliOperator) -> tuple[PauliOperator, tuple[Cbit, ...]]:
        new_op = PauliOperator("".join(old_op.pauli[i] for i, q in enumerate(old_op.data_qubits) if q not in to_measure), [q for q in old_op.data_qubits if q not in to_measure], uuid=old_op.uuid)
        return new_op, tuple(cbits[to_measure.index(q)] for q in old_op.data_qubits if q in to_measure)

    def new_log_op(old_op: PauliOperator, new_ul: Coord):
        measured_paulis = {old_op.pauli[i] for i, q in enumerate(old_op.data_qubits) if q in to_measure}
        if len(measured_paulis) > 1:
            raise RuntimeError(f"Cannot update logical operator {old_op} during the shrink because it contains multiple different Paulis inside the region of data qubits that are measured in the shrink.")
        if measured_paulis == {"Y"}:
            raise RuntimeError(f"Cannot update logical operator {old_op} during the shrink because it contains a Y Pauli inside the region of data qubits that are measured in the shrink.")
        if next(iter(measured_paulis)) != boundary_type:
            new_op, meas = shorten(old_op)
            return new_op, meas, {new_op.uuid: (old_op.uuid,)}
        new_op, stabs = block.get_shifted_equivalent_logical_operator(old_op, new_ul)
        return new_op, step.retrieve_cbits_from_stabilizers(stabs, block), {new_op.uuid: (old_op.uuid,) + tuple(s.uuid for s in stabs)}

    new_qubits = [q for s in new_stabs for q in s.data_qubits]
    new_ul = min(new_qubits, key=lambda q: q[0] + q[1])
    new_x_ops, new_z_ops = [], []
    for ops, out, kind, evolution in ((block.logical_x_operators, new_x_ops, "X", step.logical_x_evolution), (block.logical_z_operators, new_z_ops, "Z", step.logical_z_evolution)):
        for op in ops:
            if any(q in to_measure for q in op.data_qubits):
                new_op, meas, evo = new_log_op(op, new_ul)
                out.append(new_op)
                evolution.update(evo)
                step.update_logical_operator_updates_MUT(kind, new_op.uuid, meas, inherit_updates=(new_op.uuid != op.uuid))
            else:
                out.append(op)
    new_stc = dict(block.stabilizer_to_circuit)
    for s in to_remove:
        new_stc.pop(s.uuid)
    for s in reduced:
        name = f"{operation.direction.value}-{s.pauli.lower()}"
        new_stc[s.uuid] = next(c for c in block.syndrome_circuits if c.name == name).uuid
    new_block = RotatedSurfaceCode(stabilizers=list(new_stabs), logical_x_operators=new_x_ops, logical_z_operators=new_z_ops, unique_label=block.unique_label, syndrome_circuits=block.syndrome_circuits, stabilizer_to_circuit=new_stc, skip_validation=not debug_mode)
    step.update_block_history_and_evolution_MUT(new_blocks=(new_block,), old_blocks=(block,))
    return step


def _merge_consistency_check(step: InterpretationStep, operation: Merge) -> tuple[RotatedSurfaceCode, RotatedSurfaceCode]:
    name1, name2 = operation.input_blocks_name
    block1, block2 = step.get_block(name1), step.get_block(name2)
    if not isinstance(block1, RotatedSurfaceCode) or not isinstance(block2, RotatedSurfaceCode):
        raise TypeError("The merge operation is not supported for these blocks.")
    context = f"Operation Merge on {operation._inputs} failed:\n"
    if len(set(block1.data_qubits + block2.data_qubits)) < len(block1.data_qubits) + len(block2.data_qubits):
        raise ValueError(context + "The blocks overlap.")
    link = [b - a for a, b in zip(block1.upper_left_qubit, block2.upper_left_qubit)]
    if sum(c != 0 for c in link) != 1:
        raise ValueError(context + "The blocks' upper left corners are not aligned.")
    index = next(i for i, c in enumerate(link) if c != 0)
    if link[index] < 0:
        block1, block2 = block2, block1
    if index == 0:
        default, horizontal = Orientation.HORIZONTAL, True
    elif index == 1:
        default, horizontal = Orientation.VERTICAL, False
    else:
        raise ValueError(context + "The blocks corners have same x and y coordinates but differ in another way.")
    orientation = operation.orientation or default
    if orientation != default:
        raise ValueError(context + "The orientation specified in the operation does not match the default orientation.")
    if horizontal:
        if block1.size[1] != block2.size[1]:
            raise ValueError(context + "The blocks have different sizes in the vertical direction.")
        if block1.boundary_type("right") != block2.boundary_type("left"):
            raise ValueError(context + "The boundaries to be merged are of different types.")
        distance = block2.upper_left_qubit[0] - block1.upper_left_qubit[0]
    else:
        if block1.size[0] != block2.size[0]:
            raise ValueError(context + "The blocks have different sizes in the horizontal direction.")
        if block1.boundary_type("bottom") != block2.boundary_type("top"):
            raise ValueError(context + "The boundaries to be merged are of different types.")
        distance = block2.upper_left_qubit[1] - block1.upper_left_qubit[1]
    same_type = block1.upper_left_4body_stabilizer.pauli == block2.upper_left_4body_stabilizer.pauli
    if (distance % 2 == 0 and not same_type) or (distance % 2 == 1 and same_type):
        raise ValueError(context + "The alternate pattern of stabilizers is not preserved.")
    gap = distance - (block1.size[0] if horizontal else block1.size[1])
    if gap < 1:
        raise ValueError(context + "There needs to be at least one row/column of data qubits between the two blocks.")
    if len(block1.logical_x_operators) != 1 or len(block2.logical_x_operators) != 1:
        raise ValueError(context + "The blocks must have a single logical operator.")
    return block1, block2


def _create_merge_2_body_stabilizers(blocks, orientation: Orientation, merge_distance: int, filling_ul: tuple[int, int], pauli: str):
    block1 = blocks[0]
    horizontal = orientation == Orientation.HORIZONTAL
    size_index = 0 if horizontal else 1
    same_first = block1.size[not size_index] % 2 == 0
    if horizontal:
        b1_first_col = not block1.weight_2_stab_is_first_row
        top_first = b1_first_col ^ ((block1.size[size_index] % 2) == 0)
        bot_first = top_first if same_first else not top_first
        n_top = merge_distance // 2 + (top_first and (merge_distance % 2))
        n_bot = merge_distance // 2 + (bot_first and (merge_distance % 2))
        top = RotatedSurfaceCode.generate_weight2_stabs(pauli, (filling_ul[0] + (not top_first), filling_ul[1]), n_top, Orientation.HORIZONTAL, False)
        bottom = RotatedSurfaceCode.generate_weight2_stabs(pauli, (filling_ul[0] + (not bot_first), filling_ul[1] + block1.size[1] - 1), n_bot, Orientation.HORIZONTAL, True)
        return top, bottom
    left_first = block1.weight_2_stab_is_first_row ^ ((block1.size[size_index] % 2) == 0)
    right_first = left_first if same_first else not left_first
    n_left = merge_distance // 2 + (left_first and (merge_distance % 2))
    n_right = merge_distance // 2 + (right_first and (merge_distance % 2))
    left = RotatedSurfaceCode.generate_weight2_stabs(pauli, (filling_ul[0], filling_ul[1] + (not left_first)), n_left, Orientation.VERTICAL, False)
    right = RotatedSurfaceCode.generate_weight2_stabs(pauli, (filling_ul[0] + block1.size[0] - 1, filling_ul[1] + (not right_first)), n_right, Orientation.VERTICAL, True)
    return left, right


def _merge_stabilizers(blocks, horizontal: bool):
    block1, block2 = blocks
    x_sched, z_sched = block1.weight_4_x_schedule, block1.weight_4_z_schedule
    boundary1 = block1.boundary_qubits("right") if horizontal else block1.boundary_qubits("bottom")
    boundary2 = block2.boundary_qubits("left") if horizontal else block2.boundary_qubits("top")
    ul = min(boundary1, key=lambda q: q[0] + q[1])
    dx = block2.upper_left_qubit[0] - ul[0] + 1 if horizontal else block1.size[0]
    dz = block2.upper_left_qubit[1] - ul[1] + 1 if not horizontal else block1.size[1]
    adjacent = next(s for s in block1.stabilizers if ul in s.data_qubits and len(s.pauli) == 4)
    ul_pauli = "XXXX" if adjacent.pauli[0] == "Z" else "ZZZZ"
    new_ul = RotatedSurfaceCode.generate_weight4_stabs(dx, dz, ul_pauli, x_sched if ul_pauli == "XXXX" else z_sched, True, ul)
    new_rest = RotatedSurfaceCode.generate_weight4_stabs(dx, dz, "XXXX" if ul_pauli == "ZZZZ" else "ZZZZ", x_sched if ul_pauli != "XXXX" else z_sched, False, ul)
    new_4 = new_ul + new_rest
    lr_is_x = block1.x_boundary == Orientation.HORIZONTAL
    lr = "XX" if lr_is_x else "ZZ"
    tb = "ZZ" if lr_is_x else "XX"
    if horizontal:
        top, bottom = _create_merge_2_body_stabilizers(blocks, Orientation.HORIZONTAL, dx - 1, ul, tb)
        left, right = [], []
    else:
        top, bottom = [], []
        left, right = _create_merge_2_body_stabilizers(blocks, Orientation.VERTICAL, dz - 1, ul, lr)
    new_2 = top + bottom + left + right
    old_to_lengthen = [s for s in block1.stabilizers + block2.stabilizers if set(s.data_qubits) <= set(boundary1 + boundary2) and len(s.data_qubits) == 2]
    increased = [s for old, s in product(old_to_lengthen, new_4) if set(old.data_qubits) <= set(s.data_qubits) and old.pauli in s.pauli]
    # The stabilizer-to-circuit map of the merged block.
    b2_to_b1 = {c2.uuid: c1.uuid for c1 in block1.syndrome_circuits for c2 in block2.syndrome_circuits if c1.name == c2.name}
    conserved = [s.uuid for s in block1.stabilizers + block2.stabilizers if s not in old_to_lengthen]
    stc = {s: c for s, c in block1.stabilizer_to_circuit.items() if s in conserved}
    stc.update({s: b2_to_b1[c] for s, c in block2.stabilizer_to_circuit.items() if s in conserved})
    circuits = block1.syndrome_circuits + block2.syndrome_circuits
    for group, prefix in ((left, "left-"), (right, "right-"), (top, "top-"), (bottom, "bottom-")):
        for s in group:
            stc[s.uuid] = next(c.uuid for c in circuits if c.name == f"{prefix}{s.pauli.lower()}")
    for s in new_4:
        stc[s.uuid] = next(c.uuid for c in circuits if c.name == s.pauli.lower())
    new_block_stabilizers = [s for s in block1.stabilizers + block2.stabilizers if s not in old_to_lengthen] + new_4 + new_2
    return new_block_stabilizers, old_to_lengthen, increased, stc


def _data_qubits_between(q1: Coord, q2: Coord) -> list[Coord]:
    x1, y1 = q1[:2]
    x2, y2 = q2[:2]
    if x1 == x2:
        a, b = sorted([y1, y2])
        return [(x1, y, 0) for y in range(a + 1, b)]
    if y1 == y2:
        a, b = sorted([x1, x2])
        return [(x, y1, 0) for x in range(a + 1, b)]
    raise ValueError("The qubits are not aligned horizontally or vertically.")


def _merge_logical_operators(step: InterpretationStep, blocks, to_reset, horizontal: bool):
    block1, block2 = blocks
    x_horizontal = block1.x_boundary == Orientation.HORIZONTAL
    if x_horizontal == horizontal:
        pauli, merged_ops, untouched = "X", (block1.logical_x_operators[0], block2.logical_x_operators[0]), (block1.logical_z_operators[0], block2.logical_z_operators[0])
    else:
        pauli, merged_ops, untouched = "Z", (block1.logical_z_operators[0], block2.logical_z_operators[0]), (block1.logical_x_operators[0], block2.logical_x_operators[0])
    aligned1, extra1 = block1.get_shifted_equivalent_logical_operator(merged_ops[0], block1.upper_left_qubit)
    aligned2, extra2 = block2.get_shifted_equivalent_logical_operator(merged_ops[1], block2.upper_left_qubit)
    cbits1 = step.retrieve_cbits_from_stabilizers(extra1, block1)
    cbits2 = step.retrieve_cbits_from_stabilizers(extra2, block2)
    between = _data_qubits_between(max(aligned1.data_qubits, key=lambda q: q[0] + q[1]), min(aligned2.data_qubits, key=lambda q: q[0] + q[1]))
    if not set(between) <= set(to_reset):
        raise RuntimeError("The qubits chosen are not included in the reset qubits.")
    qubits = aligned1.data_qubits + aligned2.data_qubits + tuple(between)
    new_logical = PauliOperator(pauli * len(qubits), qubits)
    untouched_logical = untouched[0]
    new_x, new_z = (new_logical, untouched_logical) if pauli == "X" else (untouched_logical, new_logical)
    lineage = (merged_ops[0].uuid, merged_ops[1].uuid) + tuple(s.uuid for s in extra1 + extra2)
    if pauli == "X":
        step.logical_x_evolution[new_logical.uuid] = lineage
        return new_x, cbits1 + cbits2, new_z, ()
    step.logical_z_evolution[new_logical.uuid] = lineage
    return new_x, (), new_z, cbits1 + cbits2


def merge(step: InterpretationStep, operation: Merge, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """Reset the data qubits between the blocks (|0⟩ for an X boundary, |+⟩ for Z), join the
    stabilizers (the boundary weight-2 checks grow to weight 4, new checks fill the gap), run
    one round of the merged block, and file the joint logical measurement's bits."""
    block1, block2 = _merge_consistency_check(step, operation)
    link = [b - a for a, b in zip(block1.upper_left_qubit, block2.upper_left_qubit)]
    index = next(i for i, c in enumerate(link) if c != 0)
    horizontal = index == 0
    if horizontal:
        boundary1 = block1.boundary_qubits("right")
        distance = block2.upper_left_qubit[0] - (block1.upper_left_qubit[0] + block1.size[0])
    else:
        boundary1 = block1.boundary_qubits("bottom")
        distance = block2.upper_left_qubit[1] - (block1.upper_left_qubit[1] + block1.size[1])
    to_reset = [(q[0] + i * horizontal, q[1] + i * (not horizontal), 0) for q in boundary1 for i in range(1, distance + 1)]
    boundary_type = block1.boundary_type("right") if horizontal else block1.boundary_type("bottom")
    reset_state = "0" if boundary_type == "X" else "+"
    merge_circuit = Circuit(name=f"Merge {block1.unique_label} and {block2.unique_label} into {operation.output_block_name}", circuit=[[Circuit(f"Reset_{reset_state}", channels=[step.get_channel_MUT(q)]) for q in to_reset]])
    step.append_circuit_MUT(merge_circuit, same_timeslice)
    new_stabilizers, old_to_lengthen, increased, new_stc = _merge_stabilizers((block1, block2), horizontal)
    step.stabilizer_evolution.update({new.uuid: (old.uuid,) for new, old in zip(increased, old_to_lengthen)})
    new_x, cbits_x, new_z, cbits_z = _merge_logical_operators(step, (block1, block2), to_reset, horizontal)
    step.update_logical_operator_updates_MUT("X", new_x.uuid, cbits_x, inherit_updates=new_x.uuid not in (block1.logical_x_operators[0].uuid, block2.logical_x_operators[0].uuid))
    step.update_logical_operator_updates_MUT("Z", new_z.uuid, cbits_z, inherit_updates=new_z.uuid not in (block1.logical_z_operators[0].uuid, block2.logical_z_operators[0].uuid))
    merged = RotatedSurfaceCode(stabilizers=list(new_stabilizers), logical_x_operators=[new_x], logical_z_operators=[new_z], unique_label=operation.output_block_name, syndrome_circuits=block1.syndrome_circuits, stabilizer_to_circuit=new_stc, skip_validation=not debug_mode)
    step.update_block_history_and_evolution_MUT(new_blocks=(merged,), old_blocks=(block1, block2))
    reset_type = "X" if boundary_type == "Z" else "Z"
    reset_stabs = tuple(s for s in new_stabilizers if all(q in to_reset for q in s.data_qubits) and s not in increased and set(s.pauli) == {reset_type})
    step.append_syndromes_MUT(generate_syndromes(step, reset_stabs, merged, [() for _ in reset_stabs]))
    step = measureblocksyndromes(step, MeasureBlockSyndromes(merged.unique_label, 1), same_timeslice=False, debug_mode=debug_mode)
    if boundary_type == "X":
        new_op, old_op2, updates = merged.logical_x_operators[0], block2.logical_x_operators[0], step.logical_x_operator_updates
    else:
        new_op, old_op2, updates = merged.logical_z_operators[0], block2.logical_z_operators[0], step.logical_z_operator_updates
    _, stabs_between = merged.get_shifted_equivalent_logical_operator(new_op, min(old_op2.data_qubits, key=lambda q: q[0] + q[1]))
    between_cbits = step.retrieve_cbits_from_stabilizers(stabs_between, merged)
    corrections = updates.get(new_op.uuid, ()) + updates.get(old_op2.uuid, ())
    step.logical_measurements[LogicalMeasurement(blocks=(block1.unique_label, block2.unique_label), observable=boundary_type * 2)] = between_cbits + corrections
    return step


def _split_stabilizers(block: RotatedSurfaceCode, operation: Split, to_measure):
    boundary = Direction.LEFT if operation.orientation == Orientation.VERTICAL else Direction.TOP
    boundary_type = block.boundary_type(boundary)
    vertical = operation.orientation == Orientation.VERTICAL
    position = operation.split_position
    to_remove = [s for s in block.stabilizers if any(q in to_measure for q in s.data_qubits)]
    to_reduce = [s for s in to_remove if s.pauli[0] != boundary_type and len([q for q in s.data_qubits if q in to_measure]) == 2 and len(s.data_qubits) == 4]
    reduced = [Stabilizer(pauli="".join(s.pauli[i] for i, q in enumerate(s.data_qubits) if q not in to_measure), data_qubits=[q for q in s.data_qubits if q not in to_measure], ancilla_qubits=s.ancilla_qubits) for s in to_reduce]
    new_stabilizers = [s for s in block.stabilizers if s not in to_remove] + reduced
    axis = int(not vertical)
    cut = position + block.upper_left_qubit[axis]
    qubits1 = {q for q in block.data_qubits if q[axis] < cut}
    qubits2 = {q for q in block.data_qubits if q[axis] > cut}
    stabs1 = [s for s in new_stabilizers if set(s.data_qubits) <= qubits1]
    stabs2 = [s for s in new_stabilizers if set(s.data_qubits) <= qubits2]
    remaining = set(new_stabilizers) - set(stabs1) - set(stabs2)
    if remaining:
        raise ValueError(f"Stabilizers {remaining} are not assigned to any block.")
    return stabs1, stabs2, to_reduce, reduced


def _split_stab_to_circ(block: RotatedSurfaceCode, new_stabs, new_boundary: Direction) -> dict[str, str]:
    ids = [s.uuid for s in new_stabs]
    stc = {s: c for s, c in block.stabilizer_to_circuit.items() if s in ids}
    boundary_stabs = [s for s in new_stabs if s.uuid not in stc]
    if boundary_stabs:
        pauli = boundary_stabs[0].pauli
        circuit = next(c for c in block.syndrome_circuits if c.pauli == pauli and c.name == f"{new_boundary.value.lower()}-{pauli.lower()}")
        stc.update({s.uuid: circuit.uuid for s in boundary_stabs})
    return stc


def _split_logical_operators(step: InterpretationStep, block: RotatedSurfaceCode, to_measure, operation: Split, ul1: Coord, ul2: Coord):
    vertical = operation.orientation == Orientation.VERTICAL
    axis = int(not vertical)
    cut = operation.split_position + block.upper_left_qubit[axis]
    out = {}
    for kind, ops, evolution in (("X", block.logical_x_operators, step.logical_x_evolution), ("Z", block.logical_z_operators, step.logical_z_evolution)):
        preserved = [op for op in ops if not any(q in op.data_qubits for q in to_measure) or all(q in op.data_qubits for q in to_measure)]
        split_ops = [op for op in ops if any(q in op.data_qubits for q in to_measure) and not all(q in op.data_qubits for q in to_measure)]
        if preserved and split_ops:
            raise NotImplementedError(f"Splitting blocks with more than one logical {kind} operator is not supported.")
        if split_ops:
            op = split_ops[0]
            q1 = [q for q in op.data_qubits if q[axis] < cut]
            q2 = [q for q in op.data_qubits if q[axis] > cut]
            op1 = PauliOperator("".join(op.pauli[i] for i in range(len(q1))), q1)
            op2 = PauliOperator("".join(op.pauli[i] for i in range(len(q2))), q2)
            stabs1: tuple = ()
            stabs2: tuple = ()
            updates: dict = {}
        else:
            op1, stabs1 = block.get_shifted_equivalent_logical_operator(preserved[0], ul1)
            op2, stabs2 = block.get_shifted_equivalent_logical_operator(preserved[0], ul2)
            updates = {op1.uuid: step.retrieve_cbits_from_stabilizers(stabs1, block), op2.uuid: step.retrieve_cbits_from_stabilizers(stabs2, block)}
        evolution[op1.uuid] = tuple([ops[0].uuid] + [s.uuid for s in stabs1])
        evolution[op2.uuid] = tuple([ops[0].uuid] + [s.uuid for s in stabs2])
        out[kind] = (op1, op2, updates)
    return out


def split(step: InterpretationStep, operation: Split, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """Measure a row or column of data qubits (in the basis of the boundaries it joins), reduce
    the checks across it, and part the block in two with their logicals."""
    block = _rsc(step.get_block(operation.input_block_name), "split")
    position = operation.split_position
    if len(block.logical_x_operators) >= 2 or len(block.logical_z_operators) >= 2:
        raise NotImplementedError("Splitting blocks with more than one logical operator is not supported.")
    vertical = operation.orientation == Orientation.VERTICAL
    extent = block.size[0] if vertical else block.size[1]
    if position >= extent:
        raise ValueError(f"Split position {position} is larger than the {'height' if vertical else 'width'} of the block ({extent}).")
    if position in (0, extent - 1):
        raise ValueError("Split position cannot be at the edge of the block.")
    boundary = Direction.LEFT if vertical else Direction.TOP
    boundary_type = block.boundary_type(boundary)
    shift = (vertical * position, (not vertical) * position, 0)
    to_measure = tuple(tuple(a + b for a, b in zip(q, shift)) for q in block.boundary_qubits(boundary))
    axis = int(not vertical)
    cut = position + block.upper_left_qubit[axis]
    ul1 = min([q for q in block.data_qubits if q[axis] < cut], key=lambda q: q[0] + q[1])
    ul2 = min([q for q in block.data_qubits if q[axis] > cut], key=lambda q: q[0] + q[1])
    cbits = [step.get_new_cbit_MUT(f"c_{q}") for q in to_measure]
    cbit_channels = [step.get_channel_MUT(f"{c[0]}_{c[1]}", "classical") for c in cbits]
    measure_seq = [[Circuit("Measurement", channels=[step.get_channel_MUT(q), cbit_channels[i]]) for i, q in enumerate(to_measure)]]
    circuit_list = measure_seq if boundary_type == "X" else [[Circuit("H", channels=[step.get_channel_MUT(q)]) for q in to_measure]] + measure_seq
    step.append_circuit_MUT(Circuit(name=f"Split {block.unique_label} at {'row' if operation.orientation == Orientation.HORIZONTAL else 'column'} {position}", circuit=circuit_list), same_timeslice)
    stabs1, stabs2, to_reduce, reduced = _split_stabilizers(block, operation, to_measure)
    for i, s in enumerate(reduced):
        previous = step.stabilizer_updates.get(s.uuid, ())
        measured = [q for q in to_reduce[i].data_qubits if q in to_measure]
        new_updates = tuple(cbits[to_measure.index(q)] for q in measured)
        if previous + new_updates:
            step.stabilizer_updates[s.uuid] = previous + new_updates
    step.stabilizer_evolution.update({reduced[i].uuid: (s.uuid,) for i, s in enumerate(to_reduce)})
    stc1 = _split_stab_to_circ(block, stabs1, Direction.RIGHT if vertical else Direction.BOTTOM)
    stc2 = _split_stab_to_circ(block, stabs2, Direction.LEFT if vertical else Direction.TOP)
    ops = _split_logical_operators(step, block, to_measure, operation, ul1, ul2)
    (x1, x2, x_updates), (z1, z2, z_updates) = ops["X"], ops["Z"]
    aligned = block.x_boundary == operation.orientation
    x_meas = tuple(c for c in cbits if c[0].split("_", 1)[1] in map(str, block.logical_x_operators[0].data_qubits)) if not aligned else ()
    z_meas = tuple(c for c in cbits if c[0].split("_", 1)[1] in map(str, block.logical_z_operators[0].data_qubits)) if aligned else ()
    step.update_logical_operator_updates_MUT("X", x1.uuid, x_updates.get(x1.uuid, ()) + x_meas, inherit_updates=(x1.uuid != block.logical_x_operators[0].uuid))
    step.update_logical_operator_updates_MUT("Z", z1.uuid, z_updates.get(z1.uuid, ()) + z_meas, inherit_updates=(z1.uuid != block.logical_z_operators[0].uuid))
    step.update_logical_operator_updates_MUT("X", x2.uuid, x_updates.get(x2.uuid, ()), inherit_updates=aligned and (x2.uuid != block.logical_x_operators[0].uuid))
    step.update_logical_operator_updates_MUT("Z", z2.uuid, z_updates.get(z2.uuid, ()), inherit_updates=(not aligned) and (z2.uuid != block.logical_z_operators[0].uuid))
    new1 = RotatedSurfaceCode(stabilizers=stabs1, logical_x_operators=[x1], logical_z_operators=[z1], syndrome_circuits=block.syndrome_circuits, stabilizer_to_circuit=stc1, unique_label=operation.output_blocks_name[0], skip_validation=not debug_mode)
    new2 = RotatedSurfaceCode(stabilizers=stabs2, logical_x_operators=[x2], logical_z_operators=[z2], syndrome_circuits=block.syndrome_circuits, stabilizer_to_circuit=stc2, unique_label=operation.output_blocks_name[1], skip_validation=not debug_mode)
    step.update_block_history_and_evolution_MUT(new_blocks=(new1, new2), old_blocks=(block,))
    return step


# ---------------------------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------------------------

Applicator = Callable[[InterpretationStep, Any, bool, bool], InterpretationStep]

CODE_APPLICATORS: dict[str, Applicator] = {
    "MeasureBlockSyndromes": measureblocksyndromes,
    "MeasureLogicalX": measurelogicalpauli,
    "MeasureLogicalY": measurelogicalpauli,
    "MeasureLogicalZ": measurelogicalpauli,
    "ResetAllDataQubits": reset_all_data_qubits,
    "LogicalX": logical_pauli,
    "LogicalY": logical_pauli,
    "LogicalZ": logical_pauli,
    "ResetAllAncillaQubits": reset_all_ancilla_qubits,
    "ConditionalLogicalX": conditional_logical_pauli,
    "ConditionalLogicalY": conditional_logical_pauli,
    "ConditionalLogicalZ": conditional_logical_pauli,
}

ROTATED_SURFACE_CODE_APPLICATORS: dict[str, Applicator] = {**CODE_APPLICATORS, "Grow": grow, "Shrink": shrink, "Merge": merge, "Split": split}

_NOT_PORTED = {"AuxCNOT", "TransversalHadamard", "StateInjection"}


def interpret_operation(eka: Eka, step: InterpretationStep, op: Operation, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    name = op.__class__.__name__
    if not isinstance(op, CodeOperation):
        raise NotImplementedError(f"Operation {name} is not supported by this interpreter (Loom's base and logical operations are not ported).")
    classes = {step.get_block(label).__class__ for label in op._inputs}
    if len(classes) > 1:
        raise NotImplementedError(f"Operations including blocks of different classes are not supported at the moment. Block classes: {[c.__name__ for c in classes]}")
    klass = classes.pop()
    if name in _NOT_PORTED:
        raise NotImplementedError(f"Operation {name} is not ported; Loom (pip install el-loom) interprets it.")
    table = ROTATED_SURFACE_CODE_APPLICATORS if issubclass(klass, RotatedSurfaceCode) else CODE_APPLICATORS
    applicator = table.get(name)
    if applicator is None:
        raise NotImplementedError(f"Operation {name} is not supported by {'RotatedSurfaceCodeApplicator' if issubclass(klass, RotatedSurfaceCode) else 'CodeApplicator'}")
    step = applicator(step, op, same_timeslice, debug_mode)
    if step.composite_operation_session_stack:
        raise ValueError(f"Please ensure that all composite operation sessions have been ended. Found live sessions: {step.composite_operation_session_stack}")
    return step


def cleanup_final_step(step: InterpretationStep) -> InterpretationStep:
    """The final circuit: every time slice padded to its longest member, channels sorted by
    label."""
    full: list[tuple[Circuit, ...]] = []
    for timeslice in step.intermediate_circuit_sequence:
        span = max(c.duration for c in timeslice)
        full.append(tuple(timeslice))
        full.extend([()] * (span - 1))
    channels = sorted({ch for ts in full for c in ts for ch in c.channels}, key=lambda ch: ch.label)
    step.final_circuit = Circuit(name="Final circuit", circuit=tuple(full), channels=channels) if full else Circuit(name="Final circuit")
    step.channel_dict = {ch.id: ch for ch in step.channel_dict.values() if ch in step.final_circuit.channels}
    return step


def interpret_eka(eka: Eka, debug_mode: bool = True) -> InterpretationStep:
    """Interpret an Eka: every operation in order (a time step's operations share a time slice),
    then the final circuit."""
    step = InterpretationStep.create(eka.blocks)
    for timestep in eka.operations:
        same = False
        for op in timestep:
            step = interpret_operation(eka, step, op, same, debug_mode)
            same = True
    step = cleanup_final_step(step)
    step.is_frozen = True
    return step


# The rotated surface code's move applicator, ported in its own module (it builds on the
# functions above, so it is imported last).
from ._move_block import move_block  # noqa: E402

ROTATED_SURFACE_CODE_APPLICATORS.update({"MoveBlock": move_block})
