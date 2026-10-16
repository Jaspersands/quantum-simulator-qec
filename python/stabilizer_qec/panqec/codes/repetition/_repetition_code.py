# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
from typing import Tuple, Dict, List
import numpy as np
from stabilizer_qec.panqec.codes import StabilizerCode

Operator = Dict[Tuple, str]  # Location to pauli ('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations


class RepetitionCode(StabilizerCode):
    dimension = 2
    deformation_names = []

    @property
    def label(self) -> str:
        return 'Repetition {}'.format(self.size[0])

    @property
    def d(self):
        return self.size[0]

    def get_qubit_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, _ = self.size

        for x in range(0, 2*Lx, 2):
            coordinates.append((x, 0))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        coordinates: Coordinates = []
        Lx, Ly = self.size

        # Vertices
        for x in range(1, 2*Lx, 2):
            coordinates.append((x, 0))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        return 'vertex'

    def get_stabilizer(self, location) -> Operator:
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        pauli = 'Z'

        delta = [(-1, 0), (1, 0)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d) %
                                   (2*np.array(self.size)))

            if self.is_qubit(qubit_location):
                operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location) -> str:
        return 'x'

    def get_logicals_x(self) -> List[Operator]:
        """The 2 logical X operators."""

        Lx, Ly = self.size
        logicals = []

        operator: Operator = dict()
        for x in range(0, 2*Lx, 2):
            operator[(x, 0)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self) -> List[Operator]:
        """The 2 logical Z operators."""

        Lx, Ly = self.size
        logicals = []

        operator: Operator = dict()
        operator[(0, 0)] = 'Z'
        logicals.append(operator)

        return logicals
