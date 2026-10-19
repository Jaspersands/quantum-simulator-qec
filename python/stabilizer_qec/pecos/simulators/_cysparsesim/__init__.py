#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #
# modified for stabilizer_qec: PECOS's C++ SparseSim (cysparsesim.pyx over sparsesim.cpp, which
# PECOS uses where it has been compiled) is this package's Rust one (stabilizer_qec::pecos_sparse);
# this module is its Cython wrapper's interface in Python. Its gates are the C++ gates' images of
# X and Z, read off the compiled C++ (and so its CY, S CNOT Sd in that order, is controlled -Y).
# Random outcomes come from NumPy's global generator, drawn where PECOS's Python SparseSim draws
# them, so that np.random.seed reproduces a run (the C++ calls rand()).
# Where PECOS's Python SparseSim has more than the C++ one, this one has it too: print_stabs's
# print_y and print_destabs, copy, refactor, find_stab, run_direct and 'G2'. Two bugs are fixed:
# 'init |-i>' prepares |-i> (the C++'s, |1> then H6, prepares |+i>), and Y's sign prints right
# (the Python one's print_y drops the -1 of i*i).

import copy as _copy
from types import SimpleNamespace

import numpy as np

from stabilizer_qec._core import PecosSparseSim as _Rust

from .logical_sign import find_logical_signs

# Each gate's images of X and Z on its qubits, as (phase, x bits, z bits) for i^phase X^x Z^z
# (bit j for the gate's j-th qubit), from PECOS's compiled C++ SparseSim.
IMAGES = {
    'I': ((0, 1, 0), (0, 0, 1)),
    'X': ((0, 1, 0), (2, 0, 1)),
    'Y': ((2, 1, 0), (2, 0, 1)),
    'Z': ((2, 1, 0), (0, 0, 1)),
    'Q': ((0, 1, 0), (3, 1, 1)),
    'Qd': ((0, 1, 0), (1, 1, 1)),
    'R': ((2, 0, 1), (0, 1, 0)),
    'Rd': ((0, 0, 1), (2, 1, 0)),
    'S': ((1, 1, 1), (0, 0, 1)),
    'Sd': ((3, 1, 1), (0, 0, 1)),
    'H': ((0, 0, 1), (0, 1, 0)),
    'H2': ((2, 0, 1), (2, 1, 0)),
    'H3': ((1, 1, 1), (2, 0, 1)),
    'H4': ((3, 1, 1), (2, 0, 1)),
    'H5': ((2, 1, 0), (1, 1, 1)),
    'H6': ((2, 1, 0), (3, 1, 1)),
    'F1': ((1, 1, 1), (0, 1, 0)),
    'F2': ((2, 0, 1), (1, 1, 1)),
    'F3': ((1, 1, 1), (2, 1, 0)),
    'F4': ((0, 0, 1), (3, 1, 1)),
    'F1d': ((0, 0, 1), (1, 1, 1)),
    'F2d': ((3, 1, 1), (2, 1, 0)),
    'F3d': ((2, 0, 1), (3, 1, 1)),
    'F4d': ((3, 1, 1), (0, 1, 0)),
    'CNOT': ((0, 3, 0), (0, 0, 1), (0, 2, 0), (0, 0, 3)),
    'CZ': ((0, 1, 2), (0, 0, 1), (0, 2, 1), (0, 0, 2)),
    'CY': ((3, 3, 2), (0, 0, 1), (0, 2, 1), (0, 0, 3)),
    'SWAP': ((0, 2, 0), (0, 0, 2), (0, 1, 0), (0, 0, 1)),
    'G': ((0, 2, 0), (0, 1, 2), (0, 1, 0), (0, 2, 1)),
    'SqrtXX': ((0, 1, 0), (3, 3, 1), (0, 2, 0), (3, 3, 2)),
}
# The C++ wrapper's other names for these.
ALIASES = {'H1': 'H', 'H+z+x': 'H', 'H-z-x': 'H2', 'H+y-z': 'H3', 'H-y-z': 'H4', 'H-x+y': 'H5', 'H-x-y': 'H6', 'MS': 'SqrtXX', 'MSXX': 'SqrtXX',
           'G2': 'G'}  # ('G2': the Python SparseSim's name for G, which is the same gate)

_PHASES = ('  ', ' i', ' -', '-i')  # i^p for p = 0, 1, 2, 3


def y_letters(stab_letters, print_signs):
    """A generator's sign and letters, W = XZ = -iY printed as Y: k of them take i^p to i^(p + 3k).
    (stabilizer_qec: PECOS's Python SparseSim's print_y drops the -1 of i*i, printing -Y for |+i>'s Y.)"""
    k = stab_letters.count('W')
    letters = [stab_letters[0]] + ['Y' if letter == 'W' else letter for letter in stab_letters[1:]]
    if print_signs:
        letters[0] = _PHASES[(_PHASES.index(letters[0]) + 3 * k) % 4]
    return letters


def _one(name):
    images = IMAGES[ALIASES.get(name, name)]

    def gate(sim, qubit, **params):
        sim._state.gate([qubit], images)
    gate.__name__ = name
    return gate


def _two(name):
    images = IMAGES[ALIASES.get(name, name)]

    def gate(sim, qubits, **params):
        sim._state.gate([qubits[0], qubits[1]], images)
    gate.__name__ = name
    return gate


class SparseSim:
    """PECOS's C++ sparse stabilizer simulator (cySparseSim), on this package's Rust one."""

    def __init__(self, num_qubits, reserve_buckets=0):
        self._state = _Rust(int(num_qubits))
        self.num_qubits = int(num_qubits)
        self.reserve_buckets = reserve_buckets  # (Bucket reservation is the C++'s; kept for its interface.)
        self.bindings = bindings

    # ---- Measurements and initializations (as the Cython wrapper composes them) ---- #

    def _outcome(self, qubit, forced_outcome):
        if forced_outcome in (0, 1):
            return forced_outcome
        if forced_outcome == -1 or forced_outcome is None:  # (None: as PECOS's Python SparseSim)
            return 0 if self._state.is_deterministic(qubit) else int(np.random.randint(2))
        raise ValueError('forced_outcome was not 0, 1, or -1')

    def measure(self, qubit, forced_outcome=-1, collapse=True):
        return self._state.measure(qubit, self._outcome(qubit, forced_outcome), collapse)

    def measureX(self, qubit, forced_outcome=-1, collapse=True):
        self._state.gate([qubit], IMAGES['H'])
        result = self.measure(qubit, forced_outcome, collapse)
        self._state.gate([qubit], IMAGES['H'])
        return result

    def measureY(self, qubit, forced_outcome=-1, collapse=True):
        self._state.gate([qubit], IMAGES['H5'])
        result = self.measure(qubit, forced_outcome, collapse)
        self._state.gate([qubit], IMAGES['H5'])
        return result

    def initzero(self, qubit, **params):
        if self._state.measure(qubit, 0, True) == 1:
            self._state.gate([qubit], IMAGES['X'])

    def initone(self, qubit, **params):
        if self._state.measure(qubit, 1, True) == 0:
            self._state.gate([qubit], IMAGES['X'])

    def initplus(self, qubit, **params):
        self.initzero(qubit)
        self._state.gate([qubit], IMAGES['H'])

    def initminus(self, qubit, **params):
        self.initone(qubit)
        self._state.gate([qubit], IMAGES['H'])

    def initplusi(self, qubit, **params):
        self.initzero(qubit)
        self._state.gate([qubit], IMAGES['H5'])

    def initminusi(self, qubit, **params):
        # (stabilizer_qec: |0> then H6, which takes Z to -Y, as the Python SparseSim's init_minusi; PECOS's C++
        # starts from |1>, preparing |+i>.)
        self.initzero(qubit)
        self._state.gate([qubit], IMAGES['H6'])

    def force_output(self, qubit, forced_output=-1):
        """
        Outputs value.

        Used for error generators to generate outputs when replacing measurements.
        """
        return forced_output

    def II(self, qubits, **params):
        pass

    # ---- Running gates and circuits ---- #

    def logical_sign(self, logical_op):
        return find_logical_signs(self, logical_op)

    def run_gate(self, symbol, locations, **params):
        output = {}
        for location in locations:
            results = self.bindings[symbol](self, location, **params)
            if results:
                output[location] = results
        return output

    def run_direct(self, symbol, location, **gate_kwargs):
        """(As the Python SparseSim's) one gate on one location, with no output collected."""
        self.bindings[symbol](self, location, **gate_kwargs)

    def run_circuit(self, circuit, removed_locations=None):
        if removed_locations is None:
            removed_locations = set([])
        results = {}
        for symbol, locations, params in circuit.items():
            gate_results = self.run_gate(symbol, locations - removed_locations, **params)
            results.update(gate_results)
        return results

    # ---- The state ---- #

    def clear(self):
        self._state.clear()

    def copy(self):
        """(As the Python SparseSim's, which forgets to return it) a copy of this state."""
        new = SparseSim.__new__(SparseSim)
        new.__dict__.update(self.__dict__)
        new._state = _copy.copy(self._state)
        return new

    def _gens(self):
        # The generators as the Python SparseSim's Gens: lists of sets, the signs on the stabilizers.
        stabs, destabs = (SimpleNamespace(**gen) for gen in (self.stabs, self.destabs))
        stabs.signs_minus, stabs.signs_i = self.signs_minus, self.signs_i
        destabs.signs_minus, destabs.signs_i = set(), set()
        return SimpleNamespace(stabs=stabs, destabs=destabs, num_qubits=self.num_qubits)

    def find_stab(self, xs, zs):
        """(As the Python SparseSim's) whether X on xs and Z on zs is in the stabilizer group, and the ids of the
        stabilizer generators whose product it is."""
        from .._sparsesim.refactor import find_stab
        return find_stab(self._gens(), set(xs), set(zs))

    def refactor(self, xs, zs, choose=None, prefer=None, protected=None):
        """(As the Python SparseSim's) makes X on xs and Z on zs a stabilizer generator, if it is in the group."""
        from .._sparsesim.refactor import refactor
        gens = self._gens()
        result = refactor(gens, set(xs), set(zs), choose, prefer, protected)
        if result[0]:
            self._state.set_rows([sorted(r) for r in gens.stabs.row_x], [sorted(r) for r in gens.stabs.row_z],
                                 [sorted(r) for r in gens.destabs.row_x], [sorted(r) for r in gens.destabs.row_z])
            self._state.set_signs(sorted(gens.stabs.signs_minus), sorted(gens.stabs.signs_i))
        return result

    @property
    def signs_minus(self):
        return self._state.signs_minus

    @property
    def signs_i(self):
        return self._state.signs_i

    @property
    def stabs(self):
        return self._state.stabs

    @property
    def destabs(self):
        return self._state.destabs

    def _pauli_sign(self, gen, i_gen):
        if i_gen in self.signs_minus:
            sign = '-i' if i_gen in self.signs_i else ' -'
        else:
            sign = ' i' if i_gen in self.signs_i else '  '
        return sign

    # (stabilizer_qec: num_qubits and print_y as the Python SparseSim's.)
    def col_string(self, gen, num_qubits=None, print_signs=True, print_y=False):
        col_x, col_z = gen['col_x'], gen['col_z']
        n = self.num_qubits if num_qubits is None else num_qubits
        result = []
        for i_gen in range(n):
            stab_letters = [self._pauli_sign(gen, i_gen) if print_signs else '  ']
            for qubit in range(n):
                x, z = i_gen in col_x[qubit], i_gen in col_z[qubit]
                stab_letters.append('W' if x and z else 'X' if x else 'Z' if z else 'I')
            result.append(''.join(y_letters(stab_letters, print_signs) if print_y else stab_letters))
        return result

    def row_string(self, gen, num_qubits=None, print_signs=True, print_y=False):
        row_x, row_z = gen['row_x'], gen['row_z']
        n = self.num_qubits if num_qubits is None else num_qubits
        result = []
        for i_gen in range(n):
            stab_letters = [self._pauli_sign(gen, i_gen) if print_signs else '  ']
            for qubit in range(n):
                x, z = qubit in row_x[i_gen], qubit in row_z[i_gen]
                stab_letters.append('W' if x and z else 'X' if x else 'Z' if z else 'I')
            result.append(''.join(y_letters(stab_letters, print_signs) if print_y else stab_letters))
        return result

    def get_largest_degree(self):
        stabs, destabs = self.stabs, self.destabs
        out = {}
        for name, gen in (('stabs', stabs), ('destabs', destabs)):
            for part in ('col_x', 'col_z', 'row_x', 'row_z'):
                kind, axis = part.split('_')
                out[f'size_{name}_{kind}_{axis}'] = [len(s) for s in gen[part]]
        out = {k: out[k] for k in ('size_stabs_col_x', 'size_stabs_col_z', 'size_destabs_col_x', 'size_destabs_col_z',
                                    'size_stabs_row_x', 'size_stabs_row_z', 'size_destabs_row_x', 'size_destabs_row_z')}
        out['size_signs_minus'] = len(self.signs_minus)
        out['size_signs_i'] = len(self.signs_i)
        return out

    def set_stabs_destabs(self, stabs_row_x, stabs_row_z, destabs_row_x, destabs_row_z):
        for name, rows in (('stabs_row_x', stabs_row_x), ('stabs_row_z', stabs_row_z),
                           ('destabs_row_x', destabs_row_x), ('destabs_row_z', destabs_row_z)):
            if len(rows) != self.num_qubits:
                raise Exception('Size of `%s` must equal `num_qubits`.' % name)
        self._state.set_rows([sorted(r) for r in stabs_row_x], [sorted(r) for r in stabs_row_z],
                             [sorted(r) for r in destabs_row_x], [sorted(r) for r in destabs_row_z])

    def print_tableau(self, gen, verbose=True, print_signs=True, print_y=True):
        col_str = self.col_string(gen, print_signs=print_signs, print_y=print_y)
        row_str = self.row_string(gen, print_signs=print_signs, print_y=print_y)
        if col_str != row_str:
            print('col')
            for line in col_str:
                print(line)
            print('\nrow')
            for line in row_str:
                print(line)
            raise Exception('Something bad happened! String representation of the row-wise vs column-wile '
                            'spare stabilizers do not match!')
        if verbose:
            for line in col_str:
                print(line)
        return col_str

    def print_stabs(self, verbose=True, print_y=True, print_destabs=False):
        # (stabilizer_qec: as the Python SparseSim's, which PECOS's docs show: the destabilizers on print_destabs,
        # Y for W = XZ. PECOS's C++ one takes no arguments, always prints both, and prints W.)
        str_s = self.print_tableau(self.stabs, verbose=verbose, print_y=print_y)
        if print_destabs:
            if verbose:
                print('-------------------------------')
            str_d = self.print_tableau(self.destabs, verbose=verbose, print_signs=False, print_y=print_y)
            return str_s, str_d
        return str_s


bindings = {
    'init |0>': SparseSim.initzero,
    'init |1>': SparseSim.initone,
    'init |+>': SparseSim.initplus,
    'init |->': SparseSim.initminus,
    'init |+i>': SparseSim.initplusi,
    'init |-i>': SparseSim.initminusi,
    'II': SparseSim.II,
    'measure X': SparseSim.measureX,
    'measure Y': SparseSim.measureY,
    'measure Z': SparseSim.measure,
    'force output': SparseSim.force_output,
}
for _name in ['I', 'X', 'Y', 'Z', 'Q', 'Qd', 'R', 'Rd', 'S', 'Sd', 'H', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
              'H+z+x', 'H-z-x', 'H+y-z', 'H-y-z', 'H-x+y', 'H-x-y', 'F1', 'F2', 'F3', 'F4', 'F1d', 'F2d', 'F3d', 'F4d']:
    bindings[_name] = _one(_name)
for _name in ['CNOT', 'CZ', 'CY', 'SWAP', 'G', 'G2', 'SqrtXX', 'MS', 'MSXX']:
    bindings[_name] = _two(_name)
