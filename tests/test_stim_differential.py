"""Stim's methods against ours on random circuits: the same output, character for character
(tableaus, flows, transformations), or the same verdict."""

from __future__ import annotations

import random
import re
import sys

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim", minversion="1.16")

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


def qasm(text: str) -> str:
    """QASM text to compare. Stim's Euler angles for its custom one-qubit gates round at a ±pi
    boundary, so its Windows build writes some differently (e.g. ``sydg`` as
    ``U(pi/2, -pi/2, -pi/2)``); ours are its Linux and macOS builds'. On Windows the angles are
    left out of the comparison."""
    if sys.platform == "win32":
        return re.sub(r"U\([^)]*\)", "U(...)", text)
    return text


@pytest.mark.parametrize("seed", range(150))
def test_exports_of_random_circuits(seed):
    rng = random.Random(2000 + seed)
    n = rng.randint(1, 5)
    text = dissipative_circuit(rng, n, rng.randint(1, 12))
    if not text:
        return
    s, o = stim.Circuit(text), sq.Circuit(text)
    for v in (2, 3):
        assert qasm(outcome(lambda: o.to_qasm(open_qasm_version=v))) == qasm(outcome(lambda: s.to_qasm(open_qasm_version=v)))
        assert qasm(outcome(lambda: o.to_qasm(open_qasm_version=v, skip_dets_and_obs=True))) == qasm(outcome(lambda: s.to_qasm(open_qasm_version=v, skip_dets_and_obs=True)))
    assert outcome(o.to_quirk_url) == outcome(s.to_quirk_url)
    assert o.to_crumble_url() == s.to_crumble_url()


@pytest.mark.parametrize("task", ["repetition_code:memory", "surface_code:rotated_memory_x", "surface_code:unrotated_memory_z", "color_code:memory_xyz"])
def test_exports_of_generated_circuits(task):
    kw = dict(distance=3, rounds=3, after_clifford_depolarization=0.01, before_measure_flip_probability=0.02, after_reset_flip_probability=0.03)
    s, o = stim.Circuit.generated(task, **kw), sq.Circuit.generated(task, **kw)
    assert qasm(o.without_noise().to_qasm(open_qasm_version=3)) == qasm(s.without_noise().to_qasm(open_qasm_version=3))
    assert qasm(o.without_noise().to_qasm(open_qasm_version=2, skip_dets_and_obs=True)) == qasm(s.without_noise().to_qasm(open_qasm_version=2, skip_dets_and_obs=True))
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


ALL_TWO = TWO + ["ISWAP_DAG", "SQRT_XX_DAG", "SQRT_YY_DAG", "CXSWAP"]


@pytest.mark.parametrize("seed", range(100))
def test_flip_simulator_propagates_flips_as_stim(seed):
    """No noise and no stabilizer randomization: flips set by hand travel through the circuit
    (every Clifford, measurement, reset, MPP, SPP, feedback) exactly as in Stim."""
    rng = random.Random(3000 + seed)
    n = rng.randint(1, 5)
    lines = [dissipative_circuit(rng, n, rng.randint(1, 12))]
    if n >= 2 and rng.random() < 0.5:
        lines.append(f"M 0\nCX rec[-1] {rng.randrange(1, n)}\nCZ {rng.randrange(n)} rec[-1]")
    if rng.random() < 0.5:
        lines.append("SPP " + "*".join(f"{rng.choice('XYZ')}{q}" for q in rng.sample(range(n), rng.randint(1, n))))
    text = "\n".join(x for x in lines if x)
    if not text:
        return
    batch = 70
    s = stim.FlipSimulator(batch_size=batch, num_qubits=n, disable_stabilizer_randomization=True)
    o = sq.FlipSimulator(batch_size=batch, num_qubits=n, disable_stabilizer_randomization=True)
    for _ in range(40):
        p, q, k = rng.choice("XYZ"), rng.randrange(n), rng.randrange(batch)
        s.set_pauli_flip(p, qubit_index=q, instance_index=k)
        o.set_pauli_flip(p, qubit_index=q, instance_index=k)
    s.do(stim.Circuit(text))
    o.do(sq.Circuit(text))
    assert [str(p) for p in o.peek_pauli_flips()] == [str(p) for p in s.peek_pauli_flips()]
    assert np.array_equal(o.get_measurement_flips(), s.get_measurement_flips())


NOISY = """
    R 0 1 2 3
    X_ERROR(0.1) 0
    Y_ERROR(0.05) 1
    DEPOLARIZE1(0.2) 2
    PAULI_CHANNEL_1(0.05, 0.1, 0.15) 3
    CX 0 1 2 3
    DEPOLARIZE2(0.3) 0 2
    PAULI_CHANNEL_2(0.01, 0.02, 0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.04, 0.05) 1 3
    E(0.1) X0 Z1
    ELSE_CORRELATED_ERROR(0.2) Y2
    HERALDED_ERASE(0.1) 1
    HERALDED_PAULI_CHANNEL_1(0.05, 0.1, 0.05, 0.02) 3
    M(0.05) 0 1
    MX(0.1) 2
    MY 3
    MPAD(0.25) 0
    MPP(0.1) X0*Z1 Y2
    MZZ(0.05) 0 1
    MRX 2
    DETECTOR rec[-1]
    DETECTOR rec[-2] rec[-3]
    DETECTOR rec[-4]
    DETECTOR rec[-5]
    DETECTOR rec[-6]
    DETECTOR rec[-7]
    DETECTOR rec[-8]
    DETECTOR rec[-9]
    DETECTOR rec[-10]
    DETECTOR rec[-11]
    OBSERVABLE_INCLUDE(0) rec[-11] Z0
"""


def test_flip_simulator_noise_matches_stim_statistically():
    batch = 40000
    s = stim.FlipSimulator(batch_size=batch, seed=1)
    o = sq.FlipSimulator(batch_size=batch, seed=1)
    s.do(stim.Circuit(NOISY))
    o.do(sq.Circuit(NOISY))
    for a, b in [(o.get_measurement_flips(), s.get_measurement_flips()), (o.get_detector_flips(), s.get_detector_flips()), (o.get_observable_flips(), s.get_observable_flips())]:
        assert a.shape == b.shape
        ra, rb = a.mean(axis=1), b.mean(axis=1)
        tol = 5 * np.sqrt(np.maximum(rb * (1 - rb), 0.01) / batch) * np.sqrt(2)
        assert np.all(np.abs(ra - rb) < tol), (ra, rb)


@pytest.mark.parametrize("seed", range(30))
def test_clifford_strings_as_stim(seed):
    rng = random.Random(4000 + seed)
    names = str(stim.CliffordString.all_cliffords_string()).split(",")
    a = ",".join(rng.choice(names) for _ in range(rng.randint(0, 9)))
    b = ",".join(rng.choice(names) for _ in range(rng.randint(0, 9)))
    sa, sb, oa, ob = stim.CliffordString(a), stim.CliffordString(b), sq.CliffordString(a), sq.CliffordString(b)
    assert str(oa * ob) == str(sa * sb)
    assert str(ob * oa) == str(sb * sa)
    e = rng.randint(-30, 30)
    assert str(oa**e) == str(sa**e)
    for which in ("x_outputs", "y_outputs", "z_outputs"):
        (op, osg), (sp, ssg) = getattr(oa, which)(), getattr(sa, which)()
        assert str(op) == str(sp) and np.array_equal(osg, ssg)
    text = unitary_circuit(rng, rng.randint(1, 5), 8)
    ones = "\n".join(line for line in text.split("\n") if line.split()[0] in ONE)
    assert str(sq.CliffordString(sq.Circuit(ones))) == str(stim.CliffordString(stim.Circuit(ones)))


DETERMINISTIC = """
    X_ERROR(1) 0 3
    R 5
    X 6
    CX 6 7
    M 0 1 2 3 4 5 6 7
    DETECTOR rec[-8]
    DETECTOR rec[-7] rec[-6]
    DETECTOR rec[-5]
    DETECTOR rec[-1] rec[-2]
    OBSERVABLE_INCLUDE(0) rec[-8]
    OBSERVABLE_INCLUDE(2) rec[-5] rec[-4]
"""


@pytest.mark.parametrize("fmt", ["01", "b8", "r8", "ptb64", "hits", "dets"])
def test_sampler_files_as_stim(fmt, tmp_path):
    s, o = stim.Circuit(DETERMINISTIC), sq.Circuit(DETERMINISTIC)
    shots = 64

    def both(write):
        write(s, tmp_path / "s")
        write(o, tmp_path / "o")
        for name in sorted(p.name[1:] for p in tmp_path.iterdir() if p.name.startswith("s")):
            assert (tmp_path / ("o" + name)).read_bytes() == (tmp_path / ("s" + name)).read_bytes(), name
        for p in list(tmp_path.iterdir()):
            p.unlink()

    both(lambda c, p: c.compile_detector_sampler().sample_write(shots, filepath=str(p) + "d", format=fmt, obs_out_filepath=str(p) + "o", obs_out_format=fmt))
    both(lambda c, p: c.compile_detector_sampler().sample_write(shots, filepath=str(p) + "d", format=fmt, append_observables=True))
    both(lambda c, p: c.compile_detector_sampler().sample_write(shots, filepath=str(p) + "d", format=fmt, prepend_observables=True))
    both(lambda c, p: c.compile_sampler(skip_reference_sample=True).sample_write(shots, filepath=str(p) + "m", format=fmt))
    both(lambda c, p: c.compile_sampler(reference_sample=np.array([1, 0, 1, 0, 0, 0, 1, 1], dtype=np.bool_)).sample_write(shots, filepath=str(p) + "m", format=fmt))
    dem = "error(1) D0 L0\nerror(0) D1\nerror(1) D1 D3 L2\ndetector D4"
    if fmt != "ptb64":  # Stim's model sampler refuses ptb64 output
        both(lambda c, p: (sq if c is o else stim).DetectorErrorModel(dem).compile_sampler().sample_write(shots, det_out_file=str(p) + "d", det_out_format=fmt, obs_out_file=str(p) + "o", obs_out_format=fmt, err_out_file=str(p) + "e", err_out_format=fmt))
    if fmt == "ptb64":  # Stim's file conversion refuses ptb64 too
        return
    rng = np.random.default_rng(5)
    meas = rng.random((shots, 8)) < 0.5
    sq.write_shot_data_file(data=meas, path=tmp_path / "in", format=fmt, num_measurements=8)
    data = (tmp_path / "in").read_bytes()
    for p in list(tmp_path.iterdir()):
        p.unlink()

    def convert(c, p, **kw):
        (tmp_path / "in").write_bytes(data)
        c.compile_m2d_converter(**kw).convert_file(measurements_filepath=str(tmp_path / "in"), measurements_format=fmt, detection_events_filepath=str(p) + "d", detection_events_format=fmt, obs_out_filepath=str(p) + "o", obs_out_format=fmt)
        (tmp_path / "in").unlink()

    both(convert)
    both(lambda c, p: convert(c, p, skip_reference_sample=True))


NOISE = ["X_ERROR(0.1)", "Y_ERROR(0.1)", "Z_ERROR(0.1)", "DEPOLARIZE1(0.1)", "PAULI_CHANNEL_1(0.1, 0.1, 0.1)", "I_ERROR(0.1)", "HERALDED_ERASE(0.1)", "HERALDED_PAULI_CHANNEL_1(0.1, 0.1, 0.1, 0.1)"]


def drawing_circuit(rng: random.Random, n: int) -> str:
    """A circuit with every kind of thing a diagram draws: gates, noise, feedback, products,
    correlated errors, detectors, coordinates and loops."""
    lines = [f"QUBIT_COORDS({q % 3}, {q // 3}) {q}" for q in range(n) if rng.random() < 0.7]
    lines.append(dissipative_circuit(rng, n, rng.randint(1, 6)))
    for _ in range(rng.randint(0, 4)):
        r = rng.random()
        if r < 0.3:
            lines.append(f"{rng.choice(NOISE)} {rng.randrange(n)}")
        elif r < 0.5 and n >= 2:
            a, b = rng.sample(range(n), 2)
            lines.append(f"{rng.choice(['DEPOLARIZE2(0.1)', 'II', 'II_ERROR(0.1)', 'SWAP', 'ISWAP', 'MXX', 'SQRT_YY_DAG'])} {a} {b}")
        elif r < 0.7:
            lines.append(f"E(0.1) X{rng.randrange(n)} Z{rng.randrange(n)}\nELSE_CORRELATED_ERROR(0.1) Y{rng.randrange(n)}")
        elif r < 0.85:
            lines.append("SPP " + "*".join(f"{rng.choice('XYZ')}{q}" for q in rng.sample(range(n), rng.randint(1, n))))
        else:
            lines.append(f"M {rng.randrange(n)}\nCX rec[-1] {rng.randrange(n)}\nCZ sweep[0] {rng.randrange(n)}")
    body = dissipative_circuit(rng, n, rng.randint(1, 5))
    if body:
        lines.append(f"REPEAT {rng.randint(1, 3)} {{\n{body}\nTICK\n}}")
    lines.append(f"M {' '.join(map(str, range(n)))}")
    lines.append("DETECTOR(1, 2) rec[-1]")
    if n >= 2:
        lines.append("DETECTOR(0, 1, 5) rec[-1] rec[-2]")
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    return "\n".join(x for x in lines if x)


@pytest.mark.parametrize("seed", range(60))
def test_diagrams_of_random_circuits(seed):
    rng = random.Random(5000 + seed)
    text = drawing_circuit(rng, rng.randint(1, 5))
    s, o = stim.Circuit(text), sq.Circuit(text)
    assert str(o.diagram("timeline-3d")) == str(s.diagram("timeline-3d"))
    for tick in range(s.num_ticks + 1):
        assert str(o.diagram("detslice-text", tick=tick)) == str(s.diagram("detslice-text", tick=tick)), tick
    assert str(o.diagram("detslice-text", filter_coords=["L0", (0,)])) == str(s.diagram("detslice-text", filter_coords=["L0", (0,)]))
    try:
        dem_s = s.detector_error_model(approximate_disjoint_errors=True, allow_gauge_detectors=True)
    except ValueError:  # a random observable that isn't deterministic
        return
    dem_o = sq.DetectorErrorModel(str(dem_s))
    assert str(dem_o.diagram("matchgraph-3d")) == str(dem_s.diagram("matchgraph-3d"))


@pytest.mark.parametrize("task", ["repetition_code:memory", "surface_code:rotated_memory_x", "surface_code:unrotated_memory_z", "color_code:memory_xyz"])
def test_diagrams_of_generated_circuits(task):
    kw = dict(distance=3, rounds=3, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    s, o = stim.Circuit.generated(task, **kw), sq.Circuit.generated(task, **kw)
    for t in [0, 1, 4, range(2, 5), None]:
        assert str(o.diagram("detslice-text", tick=t)) == str(s.diagram("detslice-text", tick=t)), t
    assert str(o.diagram("detslice-text", tick=4, filter_coords=["L0", (1,), "D3"])) == str(s.diagram("detslice-text", tick=4, filter_coords=["L0", (1,), "D3"]))
    assert str(o.diagram("timeline-3d")) == str(s.diagram("timeline-3d"))
    assert str(o.diagram("matchgraph-3d")) == str(s.diagram("matchgraph-3d"))
    assert o.diagram("timeline-svg-html")._repr_html_().startswith('<iframe style="width: 100%; height: 300px;')
    assert str(o.diagram("timeline-svg-html")) == o.diagram("timeline-svg-html")._repr_html_()


@pytest.mark.parametrize("task", ["repetition_code:memory", "surface_code:rotated_memory_x", "color_code:memory_xyz"])
def test_explained_errors_as_stim(task):
    kw = dict(distance=3, rounds=2, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01, after_reset_flip_probability=0.01)
    s, o = stim.Circuit.generated(task, **kw), sq.Circuit.generated(task, **kw)

    def shown(errors):
        return [repr(e).replace("stabilizer_qec.", "stim.") for e in errors], [str(e) for e in errors]

    assert shown(o.shortest_graphlike_error()) == shown(s.shortest_graphlike_error())
    assert shown(o.explain_detector_error_model_errors(reduce_to_one_representative_error=True)) == shown(s.explain_detector_error_model_errors(reduce_to_one_representative_error=True))
    for e in o.shortest_graphlike_error():
        rebuilt = sq.ExplainedError(dem_error_terms=e.dem_error_terms, circuit_error_locations=e.circuit_error_locations)
        assert rebuilt == e and hash(rebuilt) == hash(e) and str(rebuilt) == str(e)
