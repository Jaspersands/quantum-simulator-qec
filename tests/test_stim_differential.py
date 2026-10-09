"""Stim's methods against ours on random circuits: the same output, character for character
(tableaus, flows, transformations), or the same verdict."""

from __future__ import annotations

import random

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")

ONE = ["H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "H_XY", "H_YZ", "C_XYZ", "C_ZYX", "X", "Y", "Z", "I"]
TWO = ["CX", "CY", "CZ", "SWAP", "ISWAP", "ISWAP_DAG", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "SQRT_XX", "SQRT_YY", "SQRT_ZZ", "SQRT_ZZ_DAG", "CXSWAP", "CZSWAP", "SWAPCX"]
MEAS = ["M", "MX", "MY", "MR", "MRX", "MRY", "R", "RX", "RY"]


def unitary_circuit(rng: random.Random, n: int, depth: int) -> str:
    lines = []
    for _ in range(depth):
        if rng.random() < 0.5:
            g = rng.choice(ONE)
            qs = rng.sample(range(n), rng.randint(1, n))
            lines.append(f"{g} {' '.join(map(str, qs))}")
        elif n >= 2:
            g = rng.choice(TWO)
            a, b = rng.sample(range(n), 2)
            lines.append(f"{g} {a} {b}")
    return "\n".join(lines)


def dissipative_circuit(rng: random.Random, n: int, depth: int) -> str:
    lines = []
    for _ in range(depth):
        r = rng.random()
        if r < 0.4:
            lines.append(unitary_circuit(rng, n, 1))
        elif r < 0.7:
            g = rng.choice(MEAS)
            qs = rng.sample(range(n), rng.randint(1, n))
            lines.append(f"{g} {' '.join(map(str, qs))}")
        elif r < 0.8 and n >= 2:
            a, b = rng.sample(range(n), 2)
            lines.append(f"{rng.choice(['MXX', 'MYY', 'MZZ'])} {a} {b}")
        elif r < 0.9:
            k = rng.randint(1, min(3, n))
            qs = rng.sample(range(n), k)
            lines.append("MPP " + "*".join(f"{rng.choice('XYZ')}{q}" for q in qs))
        else:
            lines.append("TICK")
    return "\n".join(x for x in lines if x)


@pytest.mark.parametrize("seed", range(150))
def test_tableaus_and_inverses(seed):
    rng = random.Random(seed)
    text = unitary_circuit(rng, rng.randint(1, 5), rng.randint(1, 12))
    if not text:
        return
    s, o = stim.Circuit(text), sq.Circuit(text)
    assert str(o.to_tableau()) == str(s.to_tableau())
    assert str(o.inverse()) == str(s.inverse())
    assert str(o.decomposed()) == str(s.decomposed())
    t = s.to_tableau()
    for method in ["elimination", "graph_state", "mpp_state", "mpp_state_unsigned"]:
        assert str(sq.Tableau.from_conjugated_generators(xs=[sq.PauliString(str(t.x_output(k))) for k in range(len(t))], zs=[sq.PauliString(str(t.z_output(k))) for k in range(len(t))]).to_circuit(method)) == str(t.to_circuit(method)), method
    if len(t) <= 4:
        ours = sq.Tableau.from_conjugated_generators(xs=[sq.PauliString(str(t.x_output(k))) for k in range(len(t))], zs=[sq.PauliString(str(t.z_output(k))) for k in range(len(t))])
        np.testing.assert_allclose(ours.to_unitary_matrix(endian="little"), t.to_unitary_matrix(endian="little"), atol=1e-5)
        np.testing.assert_allclose(ours.to_state_vector(endian="big"), t.to_state_vector(endian="big"), atol=1e-5)
        sv = t.to_state_vector(endian="little")
        assert str(sq.Tableau.from_state_vector(sv, endian="little")) == str(stim.Tableau.from_state_vector(sv, endian="little"))
        stabs = t.to_stabilizers()
        assert str(sq.Tableau.from_stabilizers([sq.PauliString(str(p)) for p in stabs])) == str(stim.Tableau.from_stabilizers(stabs))


@pytest.mark.parametrize("seed", range(150))
def test_flows(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(1, 4)
    text = dissipative_circuit(rng, n, rng.randint(1, 10))
    if not text:
        return
    s, o = stim.Circuit(text), sq.Circuit(text)
    gens = [str(f) for f in s.flow_generators()]
    assert [str(f) for f in o.flow_generators()] == gens
    sflows = s.flow_generators()
    oflows = [sq.Flow(str(f)) for f in sflows]
    assert o.has_all_flows(oflows) == s.has_all_flows(sflows)
    assert o.has_all_flows(oflows, unsigned=True) == s.has_all_flows(sflows, unsigned=True)
    for f, g in zip(sflows, oflows):
        assert o.has_flow(g) == s.has_flow(f)
        flipped = stim.Flow(input=f.input_copy(), output=f.output_copy() * -1, measurements=f.measurements_copy())
        assert o.has_flow(sq.Flow(str(flipped))) == s.has_flow(flipped)
    try:
        want = s.solve_flow_measurements([stim.Flow(f"Z{q} -> Z{q}") for q in range(n)])
    except ValueError:
        return
    assert o.solve_flow_measurements([sq.Flow(f"Z{q} -> Z{q}") for q in range(n)]) == want
    assert str(o.missing_detectors()) == str(s.missing_detectors())
    assert str(o.missing_detectors(unknown_input=True)) == str(s.missing_detectors(unknown_input=True))
    assert o.count_determined_measurements() == s.count_determined_measurements()


@pytest.mark.parametrize("task", ["repetition_code:memory", "surface_code:rotated_memory_x", "surface_code:rotated_memory_z", "color_code:memory_xyz"])
def test_generated_circuits(task):
    kw = dict(distance=3, rounds=3, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    s = stim.Circuit.generated(task, **kw)
    o = sq.Circuit.generated(task, **kw)
    for name in ["flattened", "without_noise", "decomposed", "with_inlined_feedback"]:
        assert str(getattr(o, name)()) == str(getattr(s, name)()), name
    assert str(o.missing_detectors()) == str(s.missing_detectors())
    assert o.count_determined_measurements() == s.count_determined_measurements()
    assert o.get_detector_coordinates() == s.get_detector_coordinates()
    assert o.get_final_qubit_coordinates() == s.get_final_qubit_coordinates()
    for a, b in zip(o.reference_detector_and_observable_signs(), s.reference_detector_and_observable_signs()):
        assert np.array_equal(a, b)
    flows = [stim.Flow(str(f)) for f in s.without_noise().flow_generators()[:5]]
    try:
        sc, sf = s.without_noise().time_reversed_for_flows(flows)
    except ValueError:
        return
    oc, of = o.without_noise().time_reversed_for_flows([sq.Flow(str(f)) for f in flows])
    assert str(oc) == str(sc) and [str(f) for f in of] == [str(f) for f in sf]


def test_feedback_is_inlined_as_stim_inlines_it():
    text = """
        R 0 1 2
        H 0
        CX 0 1
        M 1
        CX rec[-1] 0
        M 0
        DETECTOR rec[-1]
        X_ERROR(0.1) 2
        M 2
        CX rec[-1] 2
        CZ rec[-2] 2
        M 2
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        REPEAT 3 {
            R 1
            X_ERROR(0.1) 1
            M 1
            CX rec[-1] 1
            M 1
            DETECTOR rec[-1]
        }
    """
    assert str(sq.Circuit(text).with_inlined_feedback()) == str(stim.Circuit(text).with_inlined_feedback())


def outcome(f):
    """A call's result, or its error's type and message."""
    try:
        return f()
    except Exception as e:  # noqa: BLE001 - errors are compared too
        return f"{type(e).__name__}: {e}"


@pytest.mark.parametrize("seed", range(150))
def test_exports_of_random_circuits(seed):
    rng = random.Random(2000 + seed)
    n = rng.randint(1, 5)
    text = dissipative_circuit(rng, n, rng.randint(1, 12))
    if not text:
        return
    s, o = stim.Circuit(text), sq.Circuit(text)
    for v in (2, 3):
        assert outcome(lambda: o.to_qasm(open_qasm_version=v)) == outcome(lambda: s.to_qasm(open_qasm_version=v))
        assert outcome(lambda: o.to_qasm(open_qasm_version=v, skip_dets_and_obs=True)) == outcome(lambda: s.to_qasm(open_qasm_version=v, skip_dets_and_obs=True))
    assert outcome(o.to_quirk_url) == outcome(s.to_quirk_url)
    assert o.to_crumble_url() == s.to_crumble_url()


@pytest.mark.parametrize("task", ["repetition_code:memory", "surface_code:rotated_memory_x", "surface_code:unrotated_memory_z", "color_code:memory_xyz"])
def test_exports_of_generated_circuits(task):
    kw = dict(distance=3, rounds=3, after_clifford_depolarization=0.01, before_measure_flip_probability=0.02, after_reset_flip_probability=0.03)
    s, o = stim.Circuit.generated(task, **kw), sq.Circuit.generated(task, **kw)
    assert o.without_noise().to_qasm(open_qasm_version=3) == s.without_noise().to_qasm(open_qasm_version=3)
    assert o.without_noise().to_qasm(open_qasm_version=2, skip_dets_and_obs=True) == s.without_noise().to_qasm(open_qasm_version=2, skip_dets_and_obs=True)
    assert o.to_crumble_url() == s.to_crumble_url()
    assert o.to_crumble_url(skip_detectors=True) == s.to_crumble_url(skip_detectors=True)
    sf, of = s.flattened(), o.flattened()
    marked_s, marked_o = sf.explain_detector_error_model_errors()[:6], of.explain_detector_error_model_errors()[:6]
    assert of.to_crumble_url(mark={2: marked_o, 5: marked_o[:1]}) == sf.to_crumble_url(mark={2: marked_s, 5: marked_s[:1]})
    assert o.shortest_error_sat_problem() == s.shortest_error_sat_problem()
    for q in (1, 7, 100):
        assert o.likeliest_error_sat_problem(quantization=q) == s.likeliest_error_sat_problem(quantization=q)

    def regions(c, **kw):
        return {str(k): {t: str(p) for t, p in v.items()} for k, v in c.detecting_regions(**kw).items()}

    assert regions(o) == regions(s)
    assert regions(o, targets=["L0", "D1", (1,)], ticks=[2, 3, 4, 99]) == regions(s, targets=["L0", "D1", (1,)], ticks=[2, 3, 4, 99])
    assert regions(o, targets=["L"], ticks=range(3)) == regions(s, targets=["L"], ticks=range(3))
