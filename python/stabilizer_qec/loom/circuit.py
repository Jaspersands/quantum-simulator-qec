# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom/eka/channel.py, circuit.py and syndrome_circuit.py;
# Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for
# stabilizer_qec: plain dataclasses in place of pydantic, the if-else circuits left out.
"""Loom's circuit IR: channels (quantum or classical wires), gates (leaf circuits named after
their operation) and composite circuits (time slices of sub-circuits, each slice one tick, a
sub-circuit lasting its duration), with Loom's timing, unrolling and cloning."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from typing import Iterable, Optional, Sequence, Union
from uuid import uuid4

log = logging.getLogger(__name__)


class ChannelType(str, Enum):
    QUANTUM = "quantum"
    CLASSICAL = "classical"


@dataclass(frozen=True)
class Channel:
    """A wire: quantum or classical, with a label (a qubit's coordinates, a classical bit's
    name) and an identity of its own (two channels with one label are two wires)."""

    type: ChannelType = ChannelType.QUANTUM
    label: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not isinstance(self.type, ChannelType):
            object.__setattr__(self, "type", ChannelType(self.type))
        if self.label is None:
            object.__setattr__(self, "label", "data_qubit" if self.type == ChannelType.QUANTUM else "classical_bit")

    def is_quantum(self) -> bool:
        return self.type == ChannelType.QUANTUM

    def is_classical(self) -> bool:
        return self.type == ChannelType.CLASSICAL

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Channel) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def asdict(self) -> dict:
        return {"type": self.type.value, "label": self.label, "id": self.id}


class Circuit:
    """A gate (no sub-circuits; ``name`` the operation, ``channels`` its wires, ``duration``
    one tick unless given) or a composite circuit: ``circuit`` a tuple of time slices, each a
    tuple of sub-circuits starting at that tick; a list of circuits becomes consecutive slices,
    each followed by the empty slices its duration fills. The channels and the duration of a
    composite circuit are derived from its parts; two circuits are equal when they unroll to
    the same gates on the same wires, tick by tick (labels and ids aside)."""

    __slots__ = ("name", "circuit", "channels", "duration", "description", "id")

    def __init__(self, name: str, circuit: Union[None, "Circuit", Sequence] = None, channels: Union[None, Channel, Sequence[Channel]] = None, duration: Optional[int] = None, description: str = "", id: Optional[str] = None) -> None:
        self.name = name.lower()
        self.circuit = self._format(circuit)
        self._validate_timing(self.circuit)
        if channels is None:
            channels = ()
        elif isinstance(channels, Channel):
            channels = (channels,)
        channels = tuple(channels)
        if self.circuit:
            derived = self._derive_channels(self.circuit)
            if channels and set(derived) != set(channels):
                raise ValueError("Error while setting up composite circuit: Provided channels do not match the channels of the sub-circuits.")
            channels = channels or derived
        elif len(set(channels)) != len(channels):
            raise ValueError("Channels must be distinct.")
        self.channels = channels
        if self.circuit:
            derived_duration = self._derive_duration(self.circuit)
            if duration is not None and duration != derived_duration:
                raise ValueError(f"Error while setting up composite circuit: Provided duration ({duration}) does not match the duration of the sub-circuits ({derived_duration}).")
            self.duration = derived_duration
        else:
            if duration is None:
                duration = 1
            if duration < 1:
                raise ValueError("Duration must be a positive integer.")
            self.duration = duration
        self.description = description
        self.id = id or str(uuid4())

    @staticmethod
    def _format(circuit) -> tuple[tuple["Circuit", ...], ...]:
        if circuit is None:
            return ()
        if isinstance(circuit, Circuit):
            return ((circuit,),)
        circuit = list(circuit)
        if not circuit:
            return ()
        if all(isinstance(gate, Circuit) for gate in circuit):
            return reduce(lambda x, y: x + ((y,),) + ((),) * (y.duration - 1), circuit, ())
        return tuple(tuple(tick) for tick in circuit)

    @staticmethod
    def _validate_timing(circuit: tuple[tuple["Circuit", ...], ...]) -> None:
        occupancy: dict[str, int] = {}
        for tick, time_step in enumerate(circuit):
            for gate in time_step:
                for channel in gate.channels:
                    if occupancy.get(channel.id, -1) < tick:
                        occupancy[channel.id] = tick + gate.duration - 1
                    else:
                        raise ValueError(f"Error while setting up composite circuit: Channel {channel.label}({channel.id[0:6]}..) is subject to more than one operation at tick {tick}.")

    @staticmethod
    def _derive_channels(circuit: tuple[tuple["Circuit", ...], ...]) -> tuple[Channel, ...]:
        all_channels = {channel for tick in circuit if tick for circ in tick for channel in circ.channels}
        order = (ChannelType.QUANTUM, ChannelType.CLASSICAL)
        return tuple(sorted(all_channels, key=lambda x: order.index(x.type)))

    @staticmethod
    def _derive_duration(circuit: tuple[tuple["Circuit", ...], ...]) -> int:
        ends = [op.duration + t for t, tick in enumerate(circuit) for op in tick]
        return max(ends + [len(circuit)])

    @property
    def is_gate(self) -> bool:
        return not self.circuit

    def nr_of_qubits_in_circuit(self) -> int:
        return sum(1 for c in self.channels if c.is_quantum())

    def circuit_seq(self) -> tuple["Circuit", ...]:
        return reduce(lambda x, y: x + y, self.circuit, ())

    def flatten(self) -> "Circuit":
        """The gates of the circuit, nesting removed, in a list (the time structure too)."""
        flat: list[Circuit] = []
        queue = [self]
        while queue:
            nxt = queue.pop()
            if not nxt.circuit:
                flat.append(nxt)
            else:
                for tick in nxt.circuit:
                    for circ in tick:
                        queue.append(circ)
        flat.reverse()
        return Circuit(self.name, circuit=flat, channels=self.channels)

    @classmethod
    def unroll(cls, circuit: "Circuit") -> tuple[tuple["Circuit", ...], ...]:
        """The gates tick by tick, nesting removed and the time structure kept."""
        out: list[tuple[Circuit, ...]] = [() for _ in range(max(len(circuit.circuit), circuit.duration))]
        stack = [(0, circuit)]
        while stack:
            time, circ = stack.pop()
            if not circ.circuit:
                out[time] += (circ,)
            else:
                for i, tick in enumerate(circ.circuit):
                    for sub in reversed(tick):
                        stack.append((time + i, sub))
        return tuple(out)

    def clone(self, channels: Optional[Sequence[Channel]] = None) -> "Circuit":
        """A copy on new channels (positionally; missing ones made fresh), nesting kept."""
        new_channels = list(channels or [])
        mapped: list[Channel] = []
        for i, old in enumerate(self.channels):
            if i < len(new_channels):
                if old.is_quantum() != new_channels[i].is_quantum():
                    raise ValueError("Error while cloning circuit: CLASSICAL channels cannot be assigned to QUANTUM channels and vice versa.")
                mapped.append(new_channels[i])
            else:
                mapped.append(Channel(type=old.type))
        channel_map = dict(zip((c.id for c in self.channels), mapped))

        def update(circ: Circuit) -> Circuit:
            chans = [channel_map[c.id] for c in circ.channels]
            if not circ.circuit:
                return Circuit(circ.name, channels=chans, duration=circ.duration, description=circ.description)
            return Circuit(circ.name, circuit=tuple(tuple(update(sub) for sub in tick) for tick in circ.circuit), channels=chans, description=circ.description)

        return update(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        seq1, seq2 = Circuit.unroll(self), Circuit.unroll(other)
        if len(seq1) != len(seq2):
            return False
        channel_map: dict[str, str] = {}
        for slice1, slice2 in zip(seq1, seq2):
            if len(slice1) != len(slice2):
                return False
            for g1, g2 in zip(sorted(slice1, key=lambda x: x.name), sorted(slice2, key=lambda x: x.name)):
                if g1.name != g2.name:
                    return False
                for c1, c2 in zip(g1.channels, g2.channels):
                    channel_map.setdefault(c1.id, c2.id)
                if [c.id for c in g2.channels] != [channel_map.get(c.id) for c in g1.channels]:
                    return False
        return True

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        if not self.circuit:
            return f"{self.name} (base gate)"
        lines = [f"{self.name} ({len(self.circuit)} ticks)"]
        for i, tick in enumerate(self.circuit):
            lines.extend(f"{i}: {gate.name}" for gate in tick)
        return "\n".join(lines)

    def asdict(self) -> dict:
        return {"name": self.name, "circuit": [[sub.asdict() for sub in tick] for tick in self.circuit], "channels": [c.asdict() for c in self.channels], "duration": self.duration, "description": self.description, "id": self.id}

    @classmethod
    def fromdict(cls, data: dict, channels: Optional[dict[str, Channel]] = None) -> "Circuit":
        """A circuit from its dict (Loom's ``dataclasses.asdict`` form), channels shared by id."""
        channels = {} if channels is None else channels

        def chan(d: dict) -> Channel:
            if d["id"] not in channels:
                channels[d["id"]] = Channel(type=ChannelType(d["type"]), label=d.get("label"), id=d["id"])
            return channels[d["id"]]

        sub = tuple(tuple(cls.fromdict(s, channels) for s in tick) for tick in data.get("circuit", []))
        chans = [chan(d) for d in data.get("channels", [])]
        if sub:
            return cls(data["name"], circuit=sub, channels=chans, description=data.get("description", ""), id=data.get("id"))
        return cls(data["name"], channels=chans, duration=data.get("duration", 1), description=data.get("description", ""), id=data.get("id"))


def default_circuit(pauli_str: str) -> Circuit:
    """Loom's default syndrome circuit of a stabilizer: ancilla reset, H, a controlled Pauli
    to each data qubit in order, H, measurement; channels ``d0..``, ``a0``, ``c0``."""
    weight = len(pauli_str)
    ancilla = Channel(type=ChannelType.QUANTUM, label="a0")
    data = [Channel(type=ChannelType.QUANTUM, label=f"d{i}") for i in range(weight)]
    cbit = Channel(type=ChannelType.CLASSICAL, label="c0")
    ops = [[Circuit("Reset_0", channels=[ancilla])], [Circuit("H", channels=[ancilla])]]
    ops += [[Circuit(f"C{p}", channels=[ancilla, data[i]])] for i, p in enumerate(pauli_str)]
    ops += [[Circuit("H", channels=[ancilla])], [Circuit("Measurement", channels=[ancilla, cbit])]]
    return Circuit(name=f"stab_{pauli_str}", circuit=ops, channels=data + [ancilla, cbit])


@dataclass
class SyndromeCircuit:
    """A circuit that measures stabilizers of one Pauli string, by name; without a circuit,
    Loom's default."""

    pauli: str
    name: str = "SyndromeCircuit"
    circuit: Optional[Circuit] = None
    uuid: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not self.pauli:
            raise ValueError("A syndrome circuit needs a Pauli string.")
        if not self.name:
            raise ValueError("A syndrome circuit needs a name.")
        # Loom keeps circuit names in lower case (its applicators look them up as "xxxx",
        # "left-xx", ...).
        self.name = self.name.lower()
        if self.circuit is None:
            self.circuit = default_circuit(self.pauli)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SyndromeCircuit) and self.pauli == other.pauli and self.name == other.name and self.circuit == other.circuit

    def __hash__(self) -> int:
        return hash((self.pauli, self.name))

    def asdict(self) -> dict:
        return {"pauli": self.pauli, "name": self.name, "circuit": self.circuit.asdict() if self.circuit else None, "uuid": self.uuid}

    @classmethod
    def fromdict(cls, data: dict) -> "SyndromeCircuit":
        circuit = Circuit.fromdict(data["circuit"]) if data.get("circuit") else None
        return cls(pauli=data["pauli"], name=data.get("name", "SyndromeCircuit"), circuit=circuit, uuid=data.get("uuid", str(uuid4())))
