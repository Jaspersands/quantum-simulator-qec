"""stabilizer_qec.analysis's port of qec-util against qec-util 0.5 itself, on generated circuits,
their models and sampled data: the same circuits, models, instructions, matrices and numbers."""

from __future__ import annotations

import random

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import analysis as ours

qec_util = pytest.importorskip("qec_util")
stim = pytest.importorskip("stim", minversion="1.16")

TASKS = [
    ("surface_code:rotated_memory_x", 3, 3),
    ("surface_code:rotated_memory_z", 3, 2),
    ("repetition_code:memory", 5, 3),
    ("color_code:memory_xyz", 3, 2),
    ("surface_code:unrotated_memory_z", 3, 2),
]


def norm(x):
    """Stim's objects and ours as text; containers element by element."""
    if isinstance(x, (stim.Circuit, stim.DetectorErrorModel, stim.DemInstruction, stim.DemTarget, stim.CircuitInstruction)):
        return str(x)
    if isinstance(x, (sq.Circuit, sq.DetectorErrorModel, sq.DemInstruction, sq.DemTarget, sq.CircuitInstruction)):
        return str(x).replace("stabilizer_qec.", "stim.")
    if isinstance(x, dict):
        return {norm(k): norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(norm(v) for v in x)
    if isinstance(x, set):
        return {norm(v) for v in x}
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def both(module: str, name: str, make_args):
    """``name`` from ``module`` run on Stim's objects and on ours; outcomes normalised (an
    exception's type, where one is raised)."""
    out = []
    for lib, mk in ((qec_util, stim), (ours, sq)):
        fn = getattr(getattr(lib, module), name)
        try:
            out.append(norm(fn(*make_args(mk))))
        except Exception as e:  # noqa: BLE001 - the same failure is the expectation
            out.append(("raised", type(e).__name__))
    return out


def circuit(mk, task, d, r):
    c = stim.Circuit.generated(task, distance=d, rounds=r, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01, after_reset_flip_probability=0.005)
    return mk.Circuit(str(c))


CIRCUIT_FUNCS = [
    ("remove_gauge_detectors", lambda c: (c,)),
    ("remove_detectors", lambda c: (c, [0, 2])),
    ("remove_observables", lambda c: (c,)),
    ("observables_to_detectors", lambda c: (c,)),
    ("move_observables_to_end", lambda c: (c,)),
    ("move_first_resets_to_beginning", lambda c: (c,)),
    ("format_rec_targets", lambda c: (c,)),
    ("merge_observable_definitions", lambda c: (c,)),
]


@pytest.mark.parametrize("task,d,r", TASKS)
@pytest.mark.parametrize("name,args", CIRCUIT_FUNCS, ids=[f[0] for f in CIRCUIT_FUNCS])
def test_circuit_tools(task, d, r, name, args):
    a, b = both("circuits", name, lambda mk: args(circuit(mk, task, d, r)))
    assert a == b


def test_circuit_text_tools():
    for task, d, r in TASKS:
        c = stim.Circuit.generated(task, distance=d, rounds=r, after_clifford_depolarization=0.01).flattened()
        text = str(c)
        assert ours.circuits.remove_non_native_instrs(text) == qec_util.circuits.remove_non_native_instrs(text)
        a, b = both("circuits", "format_rec_targets", lambda mk: (mk.Circuit(text),))
        assert a == b
        if isinstance(a, str):
            qubits = {f"Q{q}": q for q in range(c.num_qubits)}
            x, y = both("circuits", "format_to_rec_targets", lambda mk: (a, qubits))
            assert x == y
    c = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2)
    a, b = both("circuits", "redefine_observables", lambda mk: (mk.Circuit(str(c)), {0: [0, 1]}))
    assert a == b


def dem(mk, task, d, r, decompose=True):
    c = stim.Circuit.generated(task, distance=d, rounds=r, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    return mk.DetectorErrorModel(str(c.detector_error_model(decompose_errors=decompose)))


DEM_FUNCS = [
    ("remove_gauge_detectors", lambda m: (m,)),
    ("get_max_weight_hyperedge", lambda m: (m,)),
    ("disjoint_graphs", lambda m: (m,)),
    ("get_flippable_detectors", lambda m: (m,)),
    ("get_flippable_observables", lambda m: (m,)),
    ("contains_only_edges", lambda m: (m,)),
    ("observables_to_detectors", lambda m: (m,)),
    ("get_errors_triggering_detectors", lambda m: (m,)),
    ("only_errors", lambda m: (m,)),
    ("remove_hyperedges", lambda m: (m,)),
    ("separate_edges_and_hyperedges", lambda m: (m,)),
    ("decomposed_graphlike_dem", lambda m: (m,)),
    ("get_dem_subgraph", lambda m: (m, [0, 1, 2, 5])),
    ("dem_difference", lambda m: (m, m[:-3] if len(m) > 3 else m)),
    ("detectors_to_observables", lambda m: (m, 0)),
    ("prepare_distance2_dem_for_pymatching", lambda m: (m,)),
]


@pytest.mark.parametrize("task,d,r", TASKS)
@pytest.mark.parametrize("decompose", [True, False])
@pytest.mark.parametrize("name,args", DEM_FUNCS, ids=[f[0] for f in DEM_FUNCS])
def test_model_tools(task, d, r, decompose, name, args):
    a, b = both("dems", name, lambda mk: args(dem(mk, task, d, r, decompose)))
    assert a == b


@pytest.mark.parametrize("task,d,r", TASKS[:3])
def test_hyperedge_decomposition(task, d, r):
    a, b = both("dems", "decompose_hyperedges_to_edges", lambda mk: (dem(mk, task, d, r, False), None, True))
    # The same pieces; their order is the order PyMatching's decode_to_edges_array lists a
    # matching's edges in, which this package's need not follow.
    assert pieces_unordered(a) == pieces_unordered(b)


def pieces_unordered(text):
    if not isinstance(text, str):
        return text
    out = []
    for line in text.split("\n"):
        if line.startswith("error"):
            head, _, rest = line.partition(") ")
            line = head + ") " + " ^ ".join(sorted(p.strip() for p in rest.split("^")))
        out.append(line)
    return out


@pytest.mark.parametrize("seed", range(20))
def test_instruction_tools(seed):
    rng = random.Random(seed)
    c = stim.Circuit.generated(rng.choice(TASKS)[0], distance=3, rounds=2, after_clifford_depolarization=0.02)
    model = c.detector_error_model(decompose_errors=True)
    errors = [i for i in model.flattened() if i.type == "error"]
    for instr in rng.sample(errors, min(6, len(errors))):
        mine = sq.DemInstruction(*_args(instr))
        for name, extra in (("get_detectors", ()), ("get_observables", ()), ("has_separator", ()), ("decomposed_detectors", ()), ("decomposed_observables", ()), ("decomposed_instrs", ()), ("sorted_dem_instr", ()), ("remove_detectors", ([0, 1, 2],))):
            outcomes = []
            for lib, x in ((qec_util, instr), (ours, mine)):
                try:
                    outcomes.append(norm(getattr(lib.dem_instrs, name)(x, *extra)))
                except ValueError as e:
                    outcomes.append(("raised", str(e)))
            assert outcomes[0] == outcomes[1], name
    instr = rng.choice(errors)
    pair = [instr, stim.DemInstruction("error", [0.2], instr.targets_copy())]
    assert norm(qec_util.dem_instrs.merge_instrs(*pair)) == norm(ours.dem_instrs.merge_instrs(*(sq.DemInstruction(*_args(i)) for i in pair)))
    probs = [rng.random() * 0.4 for _ in range(4)]
    assert ours.dem_instrs.xor_probs(*probs) == qec_util.dem_instrs.xor_probs(*probs)
    assert ours.dem_instrs.xor_lists([1, 2, 3], [2, 5]) == qec_util.dem_instrs.xor_lists([1, 2, 3], [2, 5])
    for p in (0.001, 0.1, 0.3):
        assert ours.dem_instrs.prob_indep_depol1(p) == qec_util.dem_instrs.prob_indep_depol1(p)
        assert ours.dem_instrs.prob_indep_depol2(p) == qec_util.dem_instrs.prob_indep_depol2(p)


def _args(instr):
    return instr.type, instr.args_copy(), [sq.DemTarget(str(t)) if str(t) != "^" else sq.target_separator() for t in instr.targets_copy()]


@pytest.mark.parametrize("seed", range(5))
def test_pij_matrices(seed):
    rng = np.random.default_rng(seed)
    defects = (rng.random((4000, 12)) < 0.1).astype(np.uint8)
    defects[:, 3] ^= defects[:, 4] & (rng.random(4000) < 0.5)
    for name in ("get_pij_matrix", "get_approx_pij_matrix"):
        a = getattr(qec_util.data_processing, name)(defects)
        b = getattr(ours.data_processing, name)(defects)
        assert np.array_equal(a, b, equal_nan=True)


def test_syndromes_and_defects_with_xarray_and_numpy():
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(1)
    meas = rng.integers(0, 2, (50, 4, 3)).astype(bool)
    for reset in (True, False):
        anc = xr.DataArray(meas, dims=["shot", "qec_round", "anc_qubit"], coords=dict(qec_round=[1, 2, 3, 4], anc_qubit=["X1", "X2", "Z1"]), attrs=dict(meas_reset=reset))
        a = qec_util.data_processing.get_syndromes(anc)
        b = ours.data_processing.get_syndromes(anc)
        assert np.array_equal(a.values, b.values)
        assert np.array_equal(ours.data_processing.get_syndromes(meas, meas_reset=reset), a.values)
        frame = xr.DataArray([1, 0, 1], dims=["anc_qubit"], coords=dict(anc_qubit=["X1", "X2", "Z1"])).astype(bool)
        da = qec_util.data_processing.get_defects(a, frame)
        assert np.array_equal(ours.data_processing.get_defects(a, frame).values, da.values)
        assert np.array_equal(ours.data_processing.get_defects(a.values, frame.values), da.values)
        proj = a.isel(qec_round=-1) ^ True
        assert np.array_equal(ours.data_processing.get_final_defects(a, proj).values, qec_util.data_processing.get_final_defects(a, proj).values)
        assert np.array_equal(ours.data_processing.get_final_defects(a.values, proj.values), qec_util.data_processing.get_final_defects(a, proj).values)


@pytest.mark.parametrize("seed", range(10))
def test_mod2(seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, (6, 9))
    x = rng.integers(0, 2, 9)
    b = (a @ x) % 2
    assert np.array_equal(ours.mod2.gauss_elimination_rows(a.copy()), qec_util.mod2.gauss_elimination_rows(a.copy()))
    mine = ours.mod2.solve_linear_system(a, b)
    assert np.array_equal((a @ mine) % 2, b)
    basis = [[0, 1], [1, 2], [2, 3]]
    assert ours.mod2.decompose_into_basis([0, 2], basis) in ([0, 1],)
    try:
        import galois  # noqa: F401 - qec-util solves with it
    except ImportError:
        return
    try:
        theirs = qec_util.mod2.solve_linear_system(a, b)
    except IndexError:
        return  # qec-util overruns its rows on wide systems of full rank; the port does not
    assert np.array_equal(mine, np.array(theirs))
    assert ours.mod2.decompose_into_basis([0, 2], basis) == qec_util.mod2.decompose_into_basis([0, 2], basis)


def test_performance():
    rng = np.random.default_rng(3)
    pred, true = rng.integers(0, 2, (100, 3)), rng.integers(0, 2, (100, 3))
    assert ours.performance.logical_error_prob(pred, true) == qec_util.performance.logical_error_prob(pred, true)
    assert ours.performance.confidence_interval_binomial(17, 1000) == qec_util.performance.confidence_interval_binomial(17, 1000)
    rounds = np.array([1, 2, 3, 4, 6, 8, 10, 14])
    for seed in range(5):
        data = ours.performance.logical_error_prob_decay(rounds, 0.02 + 0.01 * seed, 0.3) + np.random.default_rng(seed).normal(0, 0.002, len(rounds))
        # qec-util passes its fit a 'guess' keyword, which lmfit ignores with a warning.
        with pytest.warns(UserWarning, match="keyword argument guess"):
            a = qec_util.performance.metrics.get_error_rate(rounds, data, return_round_offset=True)
        with pytest.warns(UserWarning, match="keyword argument guess"):
            b = ours.performance.metrics.get_error_rate(rounds, data, return_round_offset=True)
        for x, y in zip(a, b):
            assert y.nominal_value == pytest.approx(x.nominal_value, rel=1e-5)
            assert y.std_dev == pytest.approx(x.std_dev, rel=1e-3)


def test_decay_fit_without_lmfit_is_lmfits():
    # This package's least squares, used when lmfit is not installed.
    from stabilizer_qec.analysis.performance import _fitting

    rounds = np.array([1, 2, 3, 5, 8, 12])
    for seed in range(5):
        data = 0.5 * (1 - (1 - 2 * 0.03) ** (rounds - 0.2)) + np.random.default_rng(seed).normal(0, 0.003, len(rounds))
        for vary in (True, False):
            theirs = qec_util.performance.LogicalErrorProbDecayModel(vary_round_offset=vary).fit(data, rounds)
            model = _fitting._Model(ours.performance.logical_error_prob_decay)
            model.set_param_hint("error_rate", min=0, max=0.5, vary=True)
            model.set_param_hint("round_offset", value=0, vary=vary)
            mine = model.fit(data, rounds=rounds)
            # The same optimum (lmfit's MINPACK stops a hair short of it at times).
            assert mine.chisqr <= theirs.chisqr * (1 + 1e-9)
            for name in ("error_rate", "round_offset"):
                err = theirs.params[name].stderr
                if err:
                    assert abs(mine.params[name].value - theirs.params[name].value) <= 1e-3 * err
                    assert mine.params[name].stderr == pytest.approx(err, rel=1e-4)


_DECAY_WITHOUT_LMFIT = r"""
import json, sys, warnings
for name in ("lmfit", "uncertainties", "scipy"):
    sys.modules[name] = None  # importing them now fails, as when not installed
import numpy as np
from stabilizer_qec import analysis as a
from stabilizer_qec.analysis.performance import _fitting
assert _fitting.Model is _fitting._Model, "lmfit was used"
rounds = np.array([1, 2, 3, 4, 6, 8, 10, 14])
data = a.performance.logical_error_prob_decay(rounds, 0.03, 0.3) + np.random.default_rng(1).normal(0, 0.002, len(rounds))
weights = np.array([1, 1, 1, 1, 1, 1, 100, 100.0])
M = a.performance.LogicalErrorProbDecayModel
def params(r):
    return {**{k: [p.value, p.stderr] for k, p in r.params.items()}, "success": r.success}
def outcome(f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = f()
        except Exception as e:
            out = ["raised", type(e).__name__, str(e)]
    return [out, sorted({str(x.message) for x in w})]
nan, inf = data.copy(), data.copy()
nan[3], inf[3] = np.nan, np.inf
out = {
    "plain": outcome(lambda: params(M().fit(data, rounds))),
    "weights": outcome(lambda: params(M().fit(data, rounds, weights=weights))),
    "fixed offset": outcome(lambda: params(M(vary_round_offset=False).fit(data, rounds, weights=weights))),
    "nan": outcome(lambda: params(M().fit(nan, rounds))),
    "inf": outcome(lambda: params(M().fit(inf, rounds))),
    "omit": outcome(lambda: params(M().fit(nan, rounds, nan_policy="omit"))),
    "empty": outcome(lambda: params(M().fit(data, rounds, min_round_fit=100))),
    "one point": outcome(lambda: params(M().fit(data, rounds, min_round_fit=14))),
    "max_nfev": outcome(lambda: params(M().fit(data, rounds, max_nfev=5))),
    "unknown": outcome(lambda: params(M().fit(data, rounds, colour=1))),
    "error rate": outcome(lambda: [[x.nominal_value, x.std_dev] for x in a.performance.metrics.get_error_rate(rounds, data, return_round_offset=True)]),
}
print(json.dumps(out))
"""


def test_decay_model_without_lmfit_is_qec_utils():
    """Without lmfit (or scipy), LogicalErrorProbDecayModel fits as lmfit does: weighted, with
    max_nfev, lmfit's errors for NaN, inf and too few points, and its warning for an unknown
    keyword."""
    import json
    import subprocess
    import sys
    import warnings

    run = subprocess.run([sys.executable, "-c", _DECAY_WITHOUT_LMFIT], capture_output=True, text=True)
    assert run.returncode == 0 and run.stderr == "", run.stderr
    mine = json.loads(run.stdout)

    rounds = np.array([1, 2, 3, 4, 6, 8, 10, 14])
    data = qec_util.performance.logical_error_prob_decay(rounds, 0.03, 0.3) + np.random.default_rng(1).normal(0, 0.002, len(rounds))
    weights = np.array([1, 1, 1, 1, 1, 1, 100, 100.0])
    M = qec_util.performance.LogicalErrorProbDecayModel

    def params(r):
        return {**{k: [p.value, p.stderr] for k, p in r.params.items()}, "success": r.success}

    def outcome(f):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                out = f()
            except Exception as e:  # noqa: BLE001
                out = ["raised", type(e).__name__, str(e)]
        return [out, sorted({str(x.message) for x in w})]

    nan, inf = data.copy(), data.copy()
    nan[3], inf[3] = np.nan, np.inf
    theirs = {
        "plain": outcome(lambda: params(M().fit(data, rounds))),
        "weights": outcome(lambda: params(M().fit(data, rounds, weights=weights))),
        "fixed offset": outcome(lambda: params(M(vary_round_offset=False).fit(data, rounds, weights=weights))),
        "nan": outcome(lambda: params(M().fit(nan, rounds))),
        "inf": outcome(lambda: params(M().fit(inf, rounds))),
        "omit": outcome(lambda: params(M().fit(nan, rounds, nan_policy="omit"))),
        "empty": outcome(lambda: params(M().fit(data, rounds, min_round_fit=100))),
        "one point": outcome(lambda: params(M().fit(data, rounds, min_round_fit=14))),
        "max_nfev": outcome(lambda: params(M().fit(data, rounds, max_nfev=5))),
        "unknown": outcome(lambda: params(M().fit(data, rounds, colour=1))),
        "error rate": outcome(lambda: [[x.nominal_value, x.std_dev] for x in qec_util.performance.metrics.get_error_rate(rounds, data, return_round_offset=True)]),
    }
    assert set(mine) == set(theirs)
    for case, (got, warned) in mine.items():
        want, want_warned = theirs[case]
        assert warned == want_warned, case
        if isinstance(want, list) and want[0] == "raised":
            assert got == want, case
            continue
        if case == "error rate":
            assert np.allclose(got, want, rtol=1e-7), case
            continue
        assert got["success"] == want["success"], case
        for name in ("error_rate", "round_offset"):
            (v, e), (w, f) = got[name], want[name]
            assert v == pytest.approx(w, rel=1e-7, abs=1e-12), (case, name)
            assert (e is None) == (f is None) and (e is None or e == pytest.approx(f, rel=1e-6, abs=1e-15)), (case, name)


def test_threshold_fit():
    rng = np.random.default_rng(7)
    ps = np.linspace(0.005, 0.015, 7)
    data = {}
    for d in (3, 5, 7):
        x = (ps - 0.01) * d ** (1 / 1.5)
        rate = np.clip(0.2 + 2 * x + 30 * x**2, 1e-4, 0.5)
        n = np.full(len(ps), 100_000)
        data[d] = (ps, rng.binomial(n, rate).astype(float), n.astype(float))
    for name in ("poly-2", "poly-3"):
        a = qec_util.threshold.get_threshold(data, fit_func_name=name, num_samples_bootstrap=0)
        b = ours.threshold.get_threshold(data, fit_func_name=name, num_samples_bootstrap=0)
        assert np.allclose(a, b, rtol=1e-6)
    t, lo, hi = ours.threshold.get_threshold(data, num_samples_bootstrap=50)
    assert lo <= t <= hi and abs(t - 0.01) < 0.001


def test_fit_files_round_trip(tmp_path):
    popt, pcov = np.array([0.01, 1.5, 0.2, 2.0, 30.0]), np.eye(5) * 1e-6
    ours.threshold.save_fit_information(tmp_path / "f.txt", "poly-2", popt, pcov, np.array([0.01, 0.011]))
    libs = [ours]
    try:
        import yaml  # noqa: F401 - qec-util reads the file with it
    except ImportError:
        pass
    else:
        libs.append(qec_util)
    for lib in libs:
        name, p, c, boot = lib.threshold.load_fit_information(tmp_path / "f.txt")
        assert name == "poly-2" and np.allclose(p, popt) and np.allclose(c, pcov) and np.allclose(boot, [0.01, 0.011])


@pytest.mark.parametrize("task,d", [("surface_code:rotated_memory_z", 3), ("repetition_code:memory", 5), ("surface_code:rotated_memory_x", 5)])
def test_circuit_distance(task, d):
    pytest.importorskip("scipy")
    c = sq.Circuit(str(stim.Circuit.generated(task, distance=d, rounds=2, after_clifford_depolarization=0.01)))
    model = c.detector_error_model()
    dist, errors = ours.distance.get_circuit_distance(model)
    assert dist == d == model.distance()
    assert len(errors) == d
    upper, _ = ours.distance.get_upper_bound_circuit_distance(model)
    assert upper >= d
