# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from collections.abc import Mapping
from typing import Protocol

from stabilizer_qec.deltakit.circuit import Circuit
from stabilizer_qec.deltakit.circuit.gates import PauliBasis

from stabilizer_qec.deltakit.explorer.codes import RotatedPlanarCode, css_code_memory_circuit


class MemoryGenerator(Protocol):
    def __call__(self, distance: int, num_rounds: int) -> Circuit: ...


class PreComputedMemoryGenerator(MemoryGenerator):
    """A memory generator that uses pre-computed circuits.

    Args:
        circuits: a mapping from distance values to another mapping that maps num_rounds
            values to actual quantum circuits. Will be used as
            ``circuits[distance][num_rounds]``.
    """

    def __init__(self, circuits: Mapping[int, Mapping[int, Circuit]]) -> None:
        super().__init__()
        self._circuits = circuits

    def __call__(self, distance: int, num_rounds: int) -> Circuit:
        if distance not in self._circuits:
            msg = f"No circuit provided for {distance=}."
            raise RuntimeError(msg)
        circuits_of_distance = self._circuits[distance]
        if num_rounds not in circuits_of_distance:
            msg = f"No circuit provided for {num_rounds=}."
            raise RuntimeError(msg)
        return circuits_of_distance[num_rounds]


def get_rotated_surface_code_memory_circuit(distance: int, num_rounds: int) -> Circuit:
    """Returns a rotated surface code Z memory experiment.

    Args:
        distance: distance of the rotated surface code.
        num_rounds: number of stabiliser measurement rounds of the memory experiment.

    Returns:
        the noiseless Z-basis memory experiment circuit.
    """
    return css_code_memory_circuit(
        RotatedPlanarCode(distance, distance), num_rounds, PauliBasis.Z
    )
