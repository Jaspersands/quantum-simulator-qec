# stabilizer_qec's fixes to PECOS 0.2.0 and its additions to its C++ SparseSim, against PECOS's docs and its
# Python SparseSim (see stabilizer_qec.pecos's docstring).
import pytest

from stabilizer_qec.pecos.circuits import QuantumCircuit
from stabilizer_qec.pecos.simulators import SparseSim, cySparseSim, pySparseSim

SIMS = [cySparseSim, pySparseSim]


def test_sparsesim_is_the_cpp_one():
    assert SparseSim is cySparseSim


def test_sparsesim_print_stabs_docs_example(capsys):
    # PECOS's docs, api_guide/simulators.rst, on its default SparseSim.
    state = SparseSim(3)
    assert state.run_gate('CNOT', {(0, 1)}) == {}
    assert state.run_gate('X', {0}) == {}
    assert state.print_stabs(print_destabs=True) == ([' -ZII', ' -ZZI', '  IIZ'], ['  XXI', '  IXI', '  IIX'])
    assert capsys.readouterr().out == ' -ZII\n -ZZI\n  IIZ\n-------------------------------\n  XXI\n  IXI\n  IIX\n'
    assert state.logical_sign(QuantumCircuit([{'Z': {0, 1}}])) == 1
    assert state.logical_sign(QuantumCircuit([{'Z': {2}}])) == 0
    assert state.print_stabs(verbose=False) == [' -ZII', ' -ZZI', '  IIZ']
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('sim', SIMS)
def test_sparsesim_inits_and_y_signs(sim):
    # Each init's stabilizer, Y printed with its sign (W = XZ = -iY), and the Y measurement it determines.
    expected = {'init |0>': '  Z', 'init |1>': ' -Z', 'init |+>': '  X', 'init |->': ' -X',
                'init |+i>': '  Y', 'init |-i>': ' -Y'}
    for init, stab in expected.items():
        state = sim(1)
        state.run_gate(init, {0})
        assert state.print_stabs(verbose=False) == [stab], init
    for init, raw, outcome in [('init |+i>', ' iW', 0), ('init |-i>', '-iW', 1)]:
        state = sim(1)
        state.run_gate(init, {0})
        assert state.print_stabs(verbose=False, print_y=False) == [raw]
        assert state.run_gate('measure Y', {0}, forced_outcome=0) == ({0: 1} if outcome else {})
    # Y on two qubits: S on both halves of a Bell pair takes XX to YY (-WW) and ZZ to ZZ.
    state = sim(2)
    state.run_gate('H', {0})
    state.run_gate('CNOT', {(0, 1)})
    state.run_gate('S', {0, 1})
    assert state.print_stabs(verbose=False) == ['  YY', '  ZZ']
    assert state.print_stabs(verbose=False, print_y=False) == [' -WW', '  ZZ']
    state.run_gate('Z', {0})
    assert state.print_stabs(verbose=False) == [' -YY', '  ZZ']


@pytest.mark.parametrize('sim', SIMS)
def test_sparsesim_copy(sim):
    state = sim(3)
    state.run_gate('H', {0})
    state.run_gate('CNOT', {(0, 1)})
    state.run_gate('S', {2})
    before = state.print_stabs(verbose=False, print_destabs=True)
    copied = state.copy()
    assert type(copied) is sim and copied.print_stabs(verbose=False, print_destabs=True) == before
    copied.run_gate('X', {0, 2})
    assert state.print_stabs(verbose=False, print_destabs=True) == before
    assert copied.print_stabs(verbose=False) == ['  XXI', ' -ZZI', ' -IIZ']


def _ghz(sim):
    state = sim(4)
    state.run_gate('H', {0})
    for pair in [(0, 1), (1, 2), (2, 3)]:
        state.run_gate('CNOT', {pair})
    state.run_gate('X', {3})
    return state


def test_cysparsesim_find_stab_refactor_run_direct_as_pysparsesim():
    cy, py = _ghz(cySparseSim), _ghz(pySparseSim)
    assert cy.print_stabs(verbose=False, print_destabs=True) == py.print_stabs(verbose=False, print_destabs=True)
    for xs, zs in [(set(), {0, 3}), (set(), {0, 2}), ({0, 1, 2, 3}, set()), ({0}, set()), (set(), {1, 2})]:
        assert cy.find_stab(xs, zs) == py.find_stab(xs, zs), (xs, zs)
    assert cy.find_stab(set(), {0, 3}) == (True, {1, 2, 3})
    assert cy.find_stab({0}, set())[0] is False
    # (With neither choose nor prefer, PECOS's refactor raised KeyError; the port's picks a generator.)
    for args, picked in [(((set(), {0, 3}), {}), 1), (((set(), {1, 3}), {'choose': 0}), 2),
                         ((({0, 1, 2, 3}, set()), {'protected': {1}}), 0), (((set(), {0, 2}), {'prefer': [2], 'protected': {1}}), 3)]:
        assert cy.refactor(*args[0], **args[1]) == py.refactor(*args[0], **args[1]) == (True, picked), args
        assert cy.print_stabs(verbose=False, print_destabs=True) == py.print_stabs(verbose=False, print_destabs=True), args
    assert cy.print_stabs(verbose=False) == ['  XXXX', ' -ZIIZ', ' -IZIZ', '  ZIZI']
    assert cy.refactor({0}, set()) == py.refactor({0}, set()) == (False, None)
    for state in (cy, py):  # every generator it could pick protected: as PECOS's
        with pytest.raises(KeyError):
            state.refactor({0, 1, 2, 3}, set(), protected={0})
    for state in (cy, py):
        assert state.run_direct('X', 0) is None
        assert state.run_direct('measure Z', 1, forced_outcome=0) is None
    assert cy.print_stabs(verbose=False, print_destabs=True) == py.print_stabs(verbose=False, print_destabs=True)


def test_cysparsesim_g2_is_g():
    for gate in ['G', 'G2']:
        cy, py = cySparseSim(2), pySparseSim(2)
        for state in (cy, py):
            state.run_gate('H', {0})
            state.run_gate('S', {1})
            state.run_gate('H', {1})
            state.run_gate(gate if state is cy else 'G2', {(0, 1)})
        assert cy.print_stabs(verbose=False, print_destabs=True) == py.print_stabs(verbose=False, print_destabs=True)
    assert cySparseSim(2).bindings['G2'].__name__ == 'G2'


@pytest.mark.parametrize('code', ['SurfaceMedial4444', 'Surface4444', 'Color488'])
def test_instruction_plot_on_networkx_3(code, monkeypatch, tmp_path):
    # PECOS's docs' `syn_extract.plot()` (and their ZRepetition's): networkx 3 refuses the edge_labels that networkx 2
    # ignored in draw_networkx_edges; the picture is the same, an arrow per CNOT, labelled by its tick.
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    import stabilizer_qec.pecos as pc

    monkeypatch.setattr(plt, 'show', lambda: None)
    instr = getattr(pc.qeccs, code)(distance=3).instruction('instr_syn_extract')
    instr.plot(filename=str(tmp_path / 'syn.png'))
    ax = plt.gca()
    edges = {}
    for tick in range(len(instr.circuit)):
        for sym, locations, _ in instr.circuit.items(tick=tick):
            if sym in ('CNOT', 'CZ', 'CY'):
                edges.update({edge: str(tick) for edge in locations})
    arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    labels = sorted(t.get_text() for t in ax.texts if t.get_text() in edges.values())
    assert len(arrows) == len(edges) and labels == sorted(edges.values())
    assert (tmp_path / 'syn.png').stat().st_size > 10000
    plt.close('all')


@pytest.mark.parametrize('after', [True, False])
def test_gatewise_two_qubit_error_set_on_numpy_2(after):
    # PECOS's docs, examples/gatewise_error_models.rst: a set of tuples of one-qubit errors and two-qubit gates, which
    # NumPy 1.24 refuses as a ragged array (and, drawn before the gate, PECOS refused its tuples).
    import numpy as np

    import stabilizer_qec.pecos as pc

    errors = pc.error_gens.GatewiseGen()
    errors.set_gate_error('CNOT', {('I', 'X'), ('X', 'X'), 'CNOT'}, error_param='r', after=after)
    np.random.seed(0)
    locations = {(2 * i, 2 * i + 1) for i in range(40)}
    circuits = errors.get_gate_error('CNOT', locations, error_params={'r': True})
    drawn, other, replace = circuits if after else (circuits[1], circuits[0], circuits[2])
    assert replace == set() and not list(other.items())
    gates = {sym: locs for sym, locs, _ in drawn.items()}
    assert set(gates) == {'CNOT', 'X'} and gates['CNOT'] <= locations
    kinds = {'CNOT' if (a, b) in gates['CNOT'] else 'XX' if a in gates['X'] else 'IX' for a, b in locations}
    assert kinds == {'CNOT', 'XX', 'IX'}
    for a, b in locations:
        assert ((a, b) in gates['CNOT']) != (b in gates['X'])
        assert a not in gates['X'] or b in gates['X']


@pytest.mark.parametrize('gen', ['DepolarGen', 'XGen', 'ZGen', 'XZGen'])
def test_error_gens_phenomenological(gen):
    # Data errors before each logical instruction and errors before measurements, nothing else (PECOS raised KeyError).
    import stabilizer_qec.pecos as pc

    qecc = pc.qeccs.Surface4444(distance=3)
    logic = pc.circuits.LogicalCircuit()
    logic.append(qecc.gate('ideal init |0>'))
    logic.append(qecc.gate('I'))
    state = pc.simulators.SparseSim(qecc.num_qudits)
    error_gen = getattr(pc.error_gens, gen)(model_level='phenomenological')
    _, errors = pc.circuit_runners.Standard(seed=3).run(state, logic, error_gen=error_gen, error_params={'p': 0.3})
    instr = qecc.instruction('instr_syn_extract')
    last = len(instr.circuit) - 1
    measured = {q for sym, locs, _ in instr.circuit.items(tick=last) if sym == 'measure Z' for q in locs}
    allowed = {'XGen': {'X'}, 'ZGen': {'Z'}}.get(gen, {'X', 'Y', 'Z'})
    seen = set()
    for (_, _, inner), faults in errors.items():
        for where, circuit in faults.items():
            qudits = {q for _, locs, _ in circuit.items() for q in locs}
            assert (inner, where) in {(0, 'after'), (last, 'before')}
            assert qudits <= (qecc.data_qudit_set if inner == 0 else measured)
            assert {sym for sym, _, _ in circuit.items()} <= allowed
            seen.add(where)
    assert seen == {'after', 'before'}


def test_timing_runner_total_time():
    # PECOS's docs, api_guide/circuit_runners.rst (PECOS's total_time stayed 0.0: its run never timed a gate).
    import stabilizer_qec.pecos as pc

    circ_runner = pc.circuit_runners.TimingRunner()
    state = pc.simulators.SparseSim(4)
    qc = pc.circuits.QuantumCircuit()
    qc.append('X', {0, 1, 2, 3})
    assert circ_runner.run(state, qc) == ({}, {})
    assert circ_runner.total_time > 0.0 and circ_runner.num_gates == 4
    assert state.print_stabs(verbose=False) == [' -ZIII', ' -IZII', ' -IIZI', ' -IIIZ']
    circ_runner.reset_time()
    assert circ_runner.total_time == 0.0 and circ_runner.num_gates == 0

    # On a clock that ticks once a call: a unit per circuit the state runs, the errors' too; outputs as Standard's.
    ticks = iter(range(10 ** 6))
    qecc = pc.qeccs.Surface4444(distance=3)
    logic = pc.circuits.LogicalCircuit()
    logic.append(qecc.gate('ideal init |0>'))
    logic.append(qecc.gate('I'))
    runs = []
    for timing in (True, False):
        state = pc.simulators.SparseSim(qecc.num_qudits)
        # (each seeds NumPy's generator)
        runner = pc.circuit_runners.TimingRunner(seed=3, timer=lambda: next(ticks)) if timing else pc.circuit_runners.Standard(seed=3)
        timed = runner if timing else timed
        calls = []
        run_circuit = state.run_circuit
        state.run_circuit = lambda circuit, removed_locations=None: calls.append(1) or run_circuit(circuit, removed_locations)
        output, errors = runner.run(state, logic, error_gen=pc.error_gens.DepolarGen(), error_params={'p': 0.1})
        runs.append((dict(output), state.print_stabs(verbose=False), len(calls)))
    assert runs[0] == runs[1] and timed.total_time == runs[0][2] > 0
