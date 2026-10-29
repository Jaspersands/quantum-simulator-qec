# stabilizer_qec's fixes to PECOS 0.2.0's SparseSim refactor (both simulators), and the default SparseSim's state
# (this package's Rust one, PECOS's C++ SparseSim) as PECOS's Python SparseSim's: its misc.stabilizer_funcs, deepcopy,
# pickling and num_qubits.
import itertools
import random

import pytest

from stabilizer_qec.pecos.circuits import QuantumCircuit
from stabilizer_qec.pecos.simulators import DensityMatrix, SparseSim, cySparseSim, pySparseSim

SIMS = [cySparseSim, pySparseSim]
ONE = ['X', 'Y', 'Z', 'Q', 'R', 'S', 'Sd', 'H', 'H5', 'F1']
TWO = ['CNOT', 'CZ', 'SWAP']


def _random_state(sim, rng):
    n = rng.randint(2, 4)
    state = sim(n)
    for _ in range(rng.randint(3, 30)):
        if rng.random() < 0.5:
            state.run_gate(rng.choice(ONE), {rng.randrange(n)})
        else:
            state.run_gate(rng.choice(TWO), {tuple(rng.sample(range(n), 2))})
    return state, n


def _signed_group(state, n):
    # Every Pauli in the stabilizer group (but the identity), with its eigenvalue: 0 for +1, 1 for -1.
    group = {}
    for letters in itertools.product('IXYZ', repeat=n):
        xs = {q for q, p in enumerate(letters) if p in 'XY'}
        zs = {q for q, p in enumerate(letters) if p in 'ZY'}
        if (xs or zs) and state.find_stab(xs, zs)[0]:
            tick = {p: {q for q, l in enumerate(letters) if l == p} for p in 'XYZ'}
            group[''.join(letters)] = state.logical_sign(QuantumCircuit([{p: qs for p, qs in tick.items() if qs}]))
    return group


@pytest.mark.parametrize('sim', SIMS)
def test_refactor_keeps_the_signed_stabilizer_group(sim):
    # stabilizer_qec: refactor makes a group element a generator, and leaves the group, signs and all, as it was
    # (PECOS's dropped the -1s of reordering the generators' X^a Z^b X^c Z^d, flipping the new generator's sign).
    changed = 0
    for seed in range(60):
        rng = random.Random(seed)
        state, n = _random_state(sim, rng)
        before = _signed_group(state, n)
        letters = rng.choice(sorted(before))
        xs = {q for q, p in enumerate(letters) if p in 'XY'}
        zs = {q for q, p in enumerate(letters) if p in 'ZY'}
        assert state.refactor(xs, zs, choose=0 if seed % 2 else None)[0]
        changed += _signed_group(state, n) != before
    assert changed == 0


@pytest.mark.parametrize('sim', SIMS)
def test_refactor_then_measure_as_the_density_matrix(sim):
    # The reviewer's program: Z0 Z1 = +1 for certain (the density matrix's [0.5, 0, 0, 0.5]); after refactoring onto
    # Z0 Z1, measuring Z0 and Z1 gives even parity.
    program = [('CNOT', (0, 1)), ('CZ', (1, 0)), ('CNOT', (1, 0)), ('CNOT', (1, 0)), ('CZ', (1, 0)), ('Y', 1), ('SWAP', (1, 0)),
               ('Q', 1), ('X', 0), ('CNOT', (1, 0)), ('CNOT', (0, 1)), ('SWAP', (1, 0)), ('H5', 0), ('R', 1), ('CNOT', (1, 0)),
               ('Q', 0), ('Z', 0), ('CZ', (1, 0))]
    state, truth = sim(2), DensityMatrix(2)
    for gate, q in program:
        state.run_gate(gate, {q})
        truth.run_gate(gate, {q})
    zz = QuantumCircuit([{'Z': {0, 1}}])
    assert truth.logical_sign(zz) == 0
    assert state.logical_sign(zz) == 0
    assert state.refactor(set(), {0, 1}, choose=0) == (True, 0)
    assert state.print_stabs(verbose=False, print_y=False)[0] == '  ZZ'
    assert state.logical_sign(zz) == 0
    for forced in (0, 1):
        copy = state.copy()
        outcomes = [copy.run_gate('measure Z', {q}, forced_outcome=forced).get(q, 0) for q in (0, 1)]
        assert outcomes[0] == outcomes[1]


def _bell(sim):
    state = sim(2)
    state.run_gate('H', {0})
    state.run_gate('CNOT', {(0, 1)})
    return state


@pytest.mark.parametrize('sim', SIMS)
def test_stabilizer_funcs_take_either_simulator(sim):
    # stabilizer_qec: misc.stabilizer_funcs reads PECOS's Python SparseSim's Gens; it takes the default SparseSim (the
    # C++ one's, whose stabs and destabs are dicts and whose signs are the simulator's) too.
    from stabilizer_qec.pecos.misc import stabilizer_funcs as funcs

    state = _bell(sim)
    assert funcs.find_stab(state, {0, 1}, set()) is True
    assert funcs.find_stab(state, {0}, set()) is False
    assert funcs.op_commutes(set(), {0, 1}, state.stabs) is True
    assert funcs.op_commutes({0}, set(), state.stabs) is False
    assert funcs.is_not_stabilizer(state, set(), {0, 1}) == 0
    assert funcs.is_not_stabilizer(state, {0}, set()) == 1
    assert funcs.is_not_stabilizer(state, {0}, {0}) == 1
    state.run_gate('X', {1})  # Z0 Z1 = -1
    funcs.remove_stab(state, set(), {0, 1}, {0}, set())
    stabs = [s for s in state.print_stabs(verbose=False, print_y=False) if s.strip(' -i') != 'II']
    assert stabs == ['  XX']
    assert state.logical_sign(QuantumCircuit([{'X': {0, 1}}])) == 0


def test_default_sparsesim_copies_pickles_and_checks_num_qubits():
    # stabilizer_qec: as PECOS's Python SparseSim, the default one deep-copies and pickles, and refuses a num_qubits
    # that is not an integer.
    import copy
    import pickle

    import numpy as np

    state = _bell(SparseSim)
    state.run_gate('S', {0})
    expected = state.print_stabs(verbose=False, print_y=False, print_destabs=True)
    for twin in (copy.deepcopy(state), pickle.loads(pickle.dumps(state)), copy.copy(state)):
        assert type(twin) is SparseSim and twin.num_qubits == 2
        assert twin.print_stabs(verbose=False, print_y=False, print_destabs=True) == expected
        twin.run_gate('measure Z', {0}, forced_outcome=1)
        assert state.print_stabs(verbose=False, print_y=False, print_destabs=True) == expected
        assert twin.run_gate('measure Z', {1}) == {1: 1}
    with pytest.raises(TypeError, match='num_qubits'):
        SparseSim(2.5)
    assert SparseSim(np.int64(3)).num_qubits == 3
