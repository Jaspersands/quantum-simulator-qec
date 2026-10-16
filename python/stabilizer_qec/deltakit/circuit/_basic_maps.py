# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""
This module defines commonly-used maps between objects.
"""

from stabilizer_qec.deltakit.circuit import PauliX, PauliY, PauliZ
from stabilizer_qec.deltakit.circuit._basic_types import CP
from stabilizer_qec.deltakit.circuit._qubit_identifiers import PauliGate
from stabilizer_qec.deltakit.circuit.gates import CX, CY, CZ, PauliBasis

BASIS_TO_PAULI = {PauliBasis.X: PauliX, PauliBasis.Y: PauliY, PauliBasis.Z: PauliZ}

PAULI_TO_CP = {
    PauliX: CX,
    PauliY: CY,
    PauliZ: CZ,
}

GATE_TO_PAULI: dict[CP, PauliGate] = {
    CX: PauliX,
    CZ: PauliZ,
}
