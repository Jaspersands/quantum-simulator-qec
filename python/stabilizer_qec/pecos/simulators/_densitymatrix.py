# modified for stabilizer_qec: new in this package (PECOS 0.2.0 declares the DensityMatrix and
# ProcessMatrix simulator classes, sim_class_types, and ships neither).

"""Density-matrix and process-matrix simulators on the PECOS simulator interface (``run_gate``,
``run_circuit``, the gate symbols PECOS's simulators share), run on stabilizer_qec's density
matrix (up to 12 qubits; a process matrix tracks its Choi state, up to 6)."""

import numpy as np

from ... import _density, _process
from .sim_class_types import DensityMatrix as _DensityMatrixBase
from .sim_class_types import ProcessMatrix as _ProcessMatrixBase

# PECOS's one-qubit Cliffords as Stim's gates, matched by the images of X and Z PECOS's own
# SparseSim gives them (a density matrix does not see a global phase).
_CLIFFORD_1 = {
    'I': 'I', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
    'Q': 'SQRT_X', 'Qd': 'SQRT_X_DAG', 'R': 'SQRT_Y', 'Rd': 'SQRT_Y_DAG', 'S': 'S', 'Sd': 'S_DAG',
    'H': 'H', 'H1': 'H', 'H2': 'H_NXZ', 'H3': 'H_XY', 'H4': 'H_NXY', 'H5': 'H_YZ', 'H6': 'H_NYZ',
    'H+z+x': 'H', 'H-z-x': 'H_NXZ', 'H+y-z': 'H_XY', 'H-y-z': 'H_NXY', 'H-x+y': 'H_YZ', 'H-x-y': 'H_NYZ',
    'F1': 'C_XYZ', 'F1d': 'C_ZYX', 'F2': 'C_ZYNX', 'F2d': 'C_NXYZ', 'F3': 'C_XYNZ', 'F3d': 'C_NZYX',
    'F4': 'C_ZNYX', 'F4d': 'C_XNYZ',
}

# PECOS's CY is S, CNOT, S† on the target: controlled −Y, Z on the control after Stim's CY.
_CLIFFORD_2 = {
    'CNOT': 'CX {0} {1}', 'CZ': 'CZ {0} {1}', 'CY': 'CY {0} {1}\nZ {0}', 'SWAP': 'SWAP {0} {1}',
    'G': 'CZ {0} {1}\nH {0} {1}\nCZ {0} {1}', 'G2': 'CZ {0} {1}\nH {0} {1}\nCZ {0} {1}',
    'II': 'I {0} {1}', 'SqrtXX': 'SQRT_XX {0} {1}', 'MS': 'SQRT_XX {0} {1}', 'MSXX': 'SQRT_XX {0} {1}',
}

_INIT = {
    'init |0>': 'R {0}', 'init |1>': 'R {0}\nX {0}', 'init |+>': 'RX {0}', 'init |->': 'RX {0}\nZ {0}',
    'init |+i>': 'RY {0}', 'init |-i>': 'RY {0}\nX {0}',
}


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def _rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def _controlled(u):
    """The control first, the target second (matrix index: control the more significant)."""
    out = np.eye(4, dtype=complex)
    out[2:, 2:] = u
    return out


_SQRT_SWAP = np.array([[1, 0, 0, 0], [0, (1 + 1j) / 2, (1 - 1j) / 2, 0], [0, (1 - 1j) / 2, (1 + 1j) / 2, 0], [0, 0, 0, 1]])
_TOFFOLI = np.eye(8, dtype=complex)
_TOFFOLI[6:, 6:] = [[0, 1], [1, 0]]


def _text(template):
    def gate(state, location, **params):
        qubits = (location,) if isinstance(location, int) else tuple(location)
        state._run(template.format(*qubits), qubits)
    return gate


def _matrix(make, angle=False):
    """A gate given by its matrix, the first location the most significant bit."""
    def gate(state, location, **params):
        qubits = [location] if isinstance(location, int) else list(location)
        m = make(params['angle']) if angle else make
        state._unitary(qubits, m)
    return gate


def _rotation(axis, two=False):
    def gate(state, location, angle=None, **params):
        qubits = (location,) if isinstance(location, int) else tuple(location)
        tag = f"II[R_{axis}{axis}(theta={angle / 2!r})]" if two else f"I[R_{axis}(theta={angle / 2!r})]"
        state._run(f"{tag} {' '.join(map(str, qubits))}", qubits)
    return gate


def _entangle(state, location, **params):
    qubits = list(location)
    state._run(f"H {qubits[0]}\n" + "".join(f"CX {qubits[0]} {q}\n" for q in qubits[1:]), qubits)


def _measure(basis):
    def gate(state, location, forced_outcome=-1, **params):
        forced = None if forced_outcome not in (0, 1) else bool(forced_outcome)
        return int(state._measure(location, basis, forced))
    return gate


def _force_output(state, location, forced_output=-1, **params):
    return forced_output


def _gate_dict():
    d = {}
    for symbol, gate in _CLIFFORD_1.items():
        d[symbol] = _text(gate + ' {0}')
    d['idle'] = d['noop'] = d['I']
    for symbol, text in _CLIFFORD_2.items():
        d[symbol] = _text(text)
    for symbol, text in _INIT.items():
        d[symbol] = _text(text)
    d.update({
        'T': _text('I[T] {0}'), 'Td': _text('I[T_DAG] {0}'),
        'RX': _rotation('X'), 'RY': _rotation('Y'), 'RZ': _rotation('Z'),
        'RXX': _rotation('X', True), 'RYY': _rotation('Y', True), 'RZZ': _rotation('Z', True),
        'PhaseRot': _matrix(lambda t: np.diag([1, np.exp(1j * t)]), angle=True),
        'CRZ': _matrix(lambda t: _controlled(_rz(t)), angle=True),
        'CRX': _matrix(lambda t: _controlled(_rx(t)), angle=True),
        'CRY': _matrix(lambda t: _controlled(_ry(t)), angle=True),
        'TOFFOLI': _matrix(_TOFFOLI), 'sqrtSWAP': _matrix(_SQRT_SWAP), 'Entangle': _entangle,
        'measure X': _measure('X'), 'measure Y': _measure('Y'), 'measure Z': _measure('Z'),
        'force output': _force_output,
    })
    return d


gate_dict = _gate_dict()


class _Gates:
    """The gate dictionary on a density matrix (for a process matrix, its system qubits, the
    Choi state's first ``num_qubits``)."""

    def _q(self, q):
        return q

    def _run(self, text, qubits):
        from ... import Circuit

        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise IndexError(f'qubit {q} is not one of the simulator\'s {self.num_qubits}')
        self._sim.do(Circuit(text))

    def _unitary(self, qubits, matrix):
        self._sim.apply_unitary([self._q(q) for q in qubits], matrix, endian='big')

    def _measure(self, qubit, basis, forced):
        return self._sim.measure(self._q(qubit), basis=basis, forced=forced)

    def add_gate(self, symbol, gate_obj, make_func=True):
        """Adds a gate: a function (state, location, **params), or with ``make_func`` a unitary
        matrix, the first location its most significant bit."""
        if symbol in self.bindings:
            print('WARNING: Can not add gate as the symbol has already been taken.')
        else:
            self.bindings[symbol] = _matrix(np.asarray(gate_obj)) if make_func else gate_obj

    def logical_sign(self, logical_op, allow_float=False):
        """0 if the state is in the +1 eigenspace of the one-tick Pauli ``logical_op``, 1 if in
        the −1 eigenspace (its expectation if ``allow_float``), as PECOS's ProjectQ wrapper."""
        if len(logical_op) != 1:
            raise Exception('Logical operators are expected to only have one tick.')
        letters = {}
        for symbol, locations, _ in logical_op.items():
            if symbol not in ('X', 'Y', 'Z'):
                raise Exception('Can not currently handle logical operator with operator "%s"!' % symbol)
            for q in locations:
                letters[self._q(q)] = symbol
        from ... import PauliString

        width = max(letters) + 1 if letters else 0
        p = PauliString(''.join(letters.get(q, '_') for q in range(width)))
        result = self._sim.peek_observable_expectation(p)
        if allow_float:
            return result
        result = round(result, 5)
        if result == -1:
            return 1
        if result == 1:
            return 0
        raise Exception('Unexpected result found!')


class DensityMatrix(_Gates, _DensityMatrixBase):
    """A density matrix on ``num_qubits`` qubits (at most 12), starting in |0…0⟩, with the gate
    symbols PECOS's simulators share (its Cliffords, initializations and measurements, with
    forced outcomes) and ProjectQ's further gates (T, rotations, controlled rotations, Toffoli).
    ``density_matrix()`` is ρ, qubit 0 the least significant bit of an index. An
    initialization is a reset, the qubit's state traced out, where PECOS's stabilizer simulators
    measure the qubit forcing the outcome (the same on a qubit not entangled with others)."""

    def __init__(self, num_qubits, seed=None):
        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int.``')
        super().__init__()
        self.bindings = dict(gate_dict)
        self.num_qubits = num_qubits
        self._sim = _density.DensityMatrixSimulator(num_qubits, seed=seed)

    def density_matrix(self, *, endian='little'):
        return self._sim.density_matrix(endian=endian)

    def probabilities(self, *, endian='little'):
        return self._sim.probabilities(endian=endian)

    def purity(self):
        return self._sim.purity()

    def apply_kraus(self, qubits, operators):
        """A channel by its Kraus operators, the first qubit the most significant bit."""
        self._sim.apply_kraus(list(qubits), operators, endian='big')


class ProcessMatrix(_Gates, _ProcessMatrixBase):
    """The process the gates run so far make on ``num_qubits`` qubits (at most 6), kept as its
    Choi state: the qubits maximally entangled with as many reference qubits, the gates acting
    on the qubits. ``choi()``, ``superoperator()``, ``ptm()`` and ``chi()`` give it in each
    form (stabilizer_qec's process conventions); a measurement makes it the process given that
    outcome."""

    def __init__(self, num_qubits, seed=None):
        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int.``')
        if num_qubits > _process.MAX_QUBITS:
            raise ValueError(f'at most {_process.MAX_QUBITS} qubits')
        super().__init__()
        self.bindings = dict(gate_dict)
        self.num_qubits = num_qubits
        self._sim = _density.DensityMatrixSimulator(2 * num_qubits, seed=seed)
        d = 1 << num_qubits
        omega = np.zeros(d * d, dtype=complex)
        for i in range(d):
            omega[i | (i << num_qubits)] = 1 / np.sqrt(d)
        self._sim.set_density_matrix(np.outer(omega, omega.conj()))

    def choi(self):
        """J = Σ_ij |i⟩⟨j| ⊗ Λ(|i⟩⟨j|), normalized to trace d."""
        rho = self._sim.density_matrix()
        return rho * ((1 << self.num_qubits) / np.real(np.trace(rho)))

    def superoperator(self):
        return _process.choi_to_superop(self.choi())

    def ptm(self):
        return _process.choi_to_ptm(self.choi())

    def chi(self):
        return _process.choi_to_chi(self.choi())

    def apply_kraus(self, qubits, operators):
        """A channel by its Kraus operators, the first qubit the most significant bit."""
        self._sim.apply_kraus([self._q(q) for q in qubits], operators, endian='big')
