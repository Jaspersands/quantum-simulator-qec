# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec/python/cudaq_qec/patch.py; Copyright (c) 2024
# NVIDIA Corporation & Affiliates, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: a ``qvector`` is a sequence of qubit indices of this package's circuits.
"""CUDA-Q QEC's ``patch``: a logical qubit's data, X-ancilla and Z-ancilla qubits."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class patch:
    """A logical qubit patch representation for surface code quantum error correction.

    This class represents a logical qubit encoded in a 2D patch, containing
    data qubits and both X and Z ancilla qubits arranged in a 2D lattice pattern.
    In CUDA-Q each field is a ``qvector``; here each is a sequence of qubit
    indices (a memory circuit lays out the data qubits first, then the X
    ancillas, then the Z ancillas).

    Attributes
    ----------
    data : sequence of int
        The data qubits that encode the logical qubit state.
    ancx : sequence of int
        The X-basis ancilla qubits used for syndrome measurement.
    ancz : sequence of int
        The Z-basis ancilla qubits used for syndrome measurement.
    """

    data: Sequence[int]
    ancx: Sequence[int]
    ancz: Sequence[int]
