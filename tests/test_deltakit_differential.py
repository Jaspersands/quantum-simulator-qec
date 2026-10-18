"""stabilizer_qec.deltakit against Deltakit itself (deltakit-circuit 0.9.2 and deltakit-explorer
0.9.4, where they are installed): every code's memory circuits in both bases (rotated and
unrotated planar codes with their schedules, the toric, repetition and bivariate bicycle codes),
compiled to native gate sets and given each of Explorer's noise models (leakage too), equal as
text and as detector error models; and the circuits Deltakit parses back from Stim equal. Its
stability circuits are Deltakit's cloud's alone (its local call raises), as are the port's."""

import pytest

from conftest import reference

reference("deltakit_stim")
dk_circuit = pytest.importorskip("deltakit_circuit")
dk_explorer = pytest.importorskip("deltakit_explorer")

from deltakit_circuit import gates as dk_gates  # noqa: E402
from deltakit_explorer import codes as dk_codes  # noqa: E402
from deltakit_explorer import qpu as dk_qpu  # noqa: E402

from stabilizer_qec.deltakit import circuit as our_circuit  # noqa: E402
from stabilizer_qec.deltakit.circuit import gates as our_gates  # noqa: E402
from stabilizer_qec.deltakit.explorer import codes as our_codes  # noqa: E402
from stabilizer_qec.deltakit.explorer import qpu as our_qpu  # noqa: E402

DK = (dk_circuit, dk_gates, dk_codes, dk_qpu)
OURS = (our_circuit, our_gates, our_codes, our_qpu)


def _codes(pkg):
    _, gates, codes, _ = pkg
    out = {}
    for w, h in ((3, 3), (3, 5), (4, 4)):
        for st in codes.ScheduleType:
            out[f"rotated {w}x{h} {st.name}"] = codes.RotatedPlanarCode(w, h, schedule_type=st)
            out[f"unrotated {w}x{h} {st.name}"] = codes.UnrotatedPlanarCode(w, h, schedule_type=st)
    for order in codes.ScheduleOrder:
        try:
            out[f"rotated 3x3 {order.name}"] = codes.RotatedPlanarCode(3, 3, schedule_order=order)
        except Exception:  # noqa: BLE001 - an order a code does not take raises in both
            out[f"rotated 3x3 {order.name}"] = None
    out["toric 3x3"] = codes.UnrotatedToricCode(3, 3)
    out["toric 4x2"] = codes.UnrotatedToricCode(4, 2)
    for d in (2, 3, 5):
        out[f"repetition {d} Z"] = codes.RepetitionCode(d)
        out[f"repetition {d} X"] = codes.RepetitionCode(d, stabiliser_type=gates.PauliBasis.X)
    out["repetition 4 looping"] = codes.RepetitionCode(4, use_looping_stabiliser=True)
    out["bivariate bicycle 6x6"] = codes.BivariateBicycleCode(6, 6, [3, 1, 2], [3, 1, 2])
    out["bivariate bicycle 3x3"] = codes.BivariateBicycleCode(3, 3, [1, 0, 2], [1, 2, 0], validate=False)
    return out


def _memories(pkg):
    _, gates, codes, _ = pkg
    out = {}
    for name, code in _codes(pkg).items():
        for basis in (gates.PauliBasis.X, gates.PauliBasis.Z):
            key = f"{name} memory {basis.name}"
            if code is None:
                out[key] = "raises"
                continue
            try:
                out[key] = codes.css_code_memory_circuit(code, 2, basis)
            except Exception as e:  # noqa: BLE001
                out[key] = f"raises {type(e).__name__}"
    return out


@pytest.fixture(scope="module")
def memories():
    return _memories(DK), _memories(OURS)


def _text(c):
    return c if isinstance(c, str) else str(c.as_stim_circuit())


def test_memory_circuits_equal(memories):
    a, b = memories
    assert a.keys() == b.keys()
    for key in a:
        assert _text(a[key]) == _text(b[key]), key


def _noise_models(pkg):
    _, _, _, qpu = pkg
    return {
        "none": qpu.NoiseParameters(),
        "SI1000": qpu.SI1000Noise(p=0.001),
        "SI1000 leakage": qpu.SI1000Noise(p=0.001, pL=0.0002),
        "SD6": qpu.SD6Noise(p=0.002),
        "toy": qpu.ToyNoise(p=0.003),
        "toy flips": qpu.ToyNoise(p=0.003, p_measurement_flip=0.01),
        "physical": qpu.PhysicalNoise(t1=20e-6, t2=30e-6, p_1_qubit_gate_error=0.001, p_2_qubit_gate_error=0.005, p_reset_error=0.002, p_meas_qubit_error=0.003, p_readout_flip=0.004),
    }


def _gate_sets(pkg):
    _, gates, _, qpu = pkg
    return {
        "exhaustive": qpu.ExhaustiveGateSet(),
        "CZ": qpu.NativeGateSet(one_qubit_gates={gates.H, gates.S, gates.SQRT_X, gates.X, gates.Z}, two_qubit_gates={gates.CZ}, reset_gates={gates.RZ}, measurement_gates={gates.MZ}),
        "CX timed": qpu.NativeGateSetAndTimes.from_times(time_1_qubit_gate=2e-8, time_2_qubit_gate=4e-8, time_reset=1e-7, time_measurement=5e-7, native_gates=qpu.NativeGateSet(two_qubit_gates={gates.CX}, reset_gates={gates.RZ, gates.RX}, measurement_gates={gates.MZ, gates.MX})),
    }


def _compiled(pkg):
    _, gates, codes, qpu = pkg
    bases = (gates.PauliBasis.X, gates.PauliBasis.Z)
    experiments = {f"{name} {b.name}": codes.css_code_memory_circuit(code, 2, b) for name, code in (("rotated 3x3", codes.RotatedPlanarCode(3, 3)), ("repetition 3", codes.RepetitionCode(3)), ("unrotated 2x3", codes.UnrotatedPlanarCode(2, 3))) for b in bases}
    out = {}
    for gname, gate_set in _gate_sets(pkg).items():
        for nname, noise in _noise_models(pkg).items():
            for ename, experiment in experiments.items():
                key = f"{ename} on {gname} with {nname}"
                try:
                    out[key] = qpu.QPU(experiment.qubits, native_gates_and_times=gate_set, noise_model=noise).compile_and_add_noise_to_circuit(experiment)
                except Exception as e:  # noqa: BLE001
                    out[key] = f"raises {type(e).__name__}: {e}"
    return out


@pytest.fixture(scope="module")
def compiled():
    return _compiled(DK), _compiled(OURS)


def test_compiled_noisy_circuits_equal(compiled):
    a, b = compiled
    assert a.keys() == b.keys()
    for key in a:
        assert _text(a[key]) == _text(b[key]), key


def test_compiled_noisy_detector_error_models_equal(compiled):
    a, b = compiled
    for key in a:
        if isinstance(a[key], str):
            continue
        da = a[key].as_stim_circuit().detector_error_model(approximate_disjoint_errors=True)
        db = b[key].as_stim_circuit().detector_error_model(approximate_disjoint_errors=True)
        assert str(da) == str(db), key


def test_circuits_parsed_from_stim_equal(memories):
    # Each package's circuit, through its Stim and back (which orders a layer's qubits
    # afresh), prints its Stim alike.
    a, b = memories
    for key in a:
        if isinstance(a[key], str):
            continue
        ra = dk_circuit.Circuit.from_stim_circuit(a[key].as_stim_circuit())
        rb = our_circuit.Circuit.from_stim_circuit(b[key].as_stim_circuit())
        assert str(ra.as_stim_circuit()) == str(rb.as_stim_circuit()), key


def test_stability_is_the_clouds_in_both():
    for codes, gates in ((dk_codes, dk_gates), (our_codes, our_gates)):
        with pytest.raises(NotImplementedError, match="client"):
            codes.css_code_stability_circuit(codes.RotatedPlanarCode(3, 3), 2, gates.PauliBasis.Z)


def test_decode_names_are_deltakits():
    # deltakit's `deltakit.decode` is deltakit_decode's names plus Explorer's six cloud decoders.
    deltakit_decode = pytest.importorskip("deltakit.decode")
    from stabilizer_qec.deltakit import decode as ours
    from stabilizer_qec.deltakit.explorer import _cloud_decoders

    assert sorted(ours.__all__) == sorted(deltakit_decode.__all__)
    for name in ("ACDecoder", "BPOSDecoder", "BeliefMatchingDecoder", "CCDecoder", "LCDecoder", "MWPMDecoder"):
        assert getattr(ours, name) is getattr(_cloud_decoders, name) and name in dir(ours)
