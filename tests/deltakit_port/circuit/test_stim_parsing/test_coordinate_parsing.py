# (c) Copyright Riverlane 2020-2025.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from stabilizer_qec.deltakit import circuit as sp


def test_deltakit_circuit_produces_stim_files_with_coordinates_at_start_of_file_if_qubits_are_defined_with_coordinates():
    deltakit_circuit_circuit = sp.Circuit()
    deltakit_circuit_circuit.append_layers(
        [
            sp.GateLayer(sp.gates.X(sp.Coordinate(0, 0))),
            sp.GateLayer(sp.gates.CX(sp.Coordinate(0, 0), sp.Coordinate(1, 0))),
            sp.GateLayer(sp.gates.MZ(sp.Coordinate(1, 0))),
        ]
    )
    assert str(deltakit_circuit_circuit.as_stim_circuit()).startswith(
        "QUBIT_COORDS(1, 0) 0\nQUBIT_COORDS(0, 0) 1\n"
    )
