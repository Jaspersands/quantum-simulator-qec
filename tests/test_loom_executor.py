"""The rest of Loom's executor in the port (`stabilizer_qec.loom`): its circuit error models,
the noise they write into the Stim circuit, and its Mimiq and Guppy exporters, against fixtures
recorded from el-loom 0.4.0 (tests/loom_port/fixtures_executor.json): the noisy Stim text of
each error-model type and application mode on four interpreted experiments and a gate zoo,
the Mimiq and Guppy texts and Loom's validation messages; and the noisy circuits decode."""

from __future__ import annotations

import json
import pathlib
import re

import pytest

import stabilizer_qec as sq
from stabilizer_qec import loom
from stabilizer_qec.loom import (
    ApplicationMode,
    AsymmetricDepolarizeCEM,
    Channel,
    ChannelType,
    Circuit,
    CircuitErrorModel,
    Direction,
    Eka,
    ErrorType,
    Grow,
    HomogeneousTimeDependentCEM,
    HomogeneousTimeIndependentCEM,
    Lattice,
    MeasureBlockSyndromes,
    MeasureLogicalZ,
    Merge,
    Orientation,
    ResetAllDataQubits,
    RotatedSurfaceCode,
    Split,
)

FIXTURES = json.loads((pathlib.Path(__file__).parent / "loom_port" / "fixtures_executor.json").read_text())
NOISE = {"X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1", "DEPOLARIZE1", "DEPOLARIZE2"}
LATTICE = Lattice.square_2d((15, 15))


def experiment(name):
    """The recorded experiments, as the recorder made them."""
    b1 = RotatedSurfaceCode.create(3, 3, LATTICE, unique_label="b1")
    b2 = RotatedSurfaceCode.create(3, 3, LATTICE, unique_label="b2", position=(4, 0))
    cases = {
        "memory_z_d3": ([b1], [ResetAllDataQubits("b1"), MeasureBlockSyndromes("b1", n_cycles=3), MeasureLogicalZ("b1")]),
        "merge_h": ([b1, b2], [ResetAllDataQubits("b1"), ResetAllDataQubits("b2"), MeasureBlockSyndromes("b1"), MeasureBlockSyndromes("b2"), Merge(["b1", "b2"], "m"), MeasureBlockSyndromes("m", n_cycles=2), MeasureLogicalZ("m")]),
        "merge_split_h_1": ([b1, b2], [ResetAllDataQubits("b1"), ResetAllDataQubits("b2"), MeasureBlockSyndromes("b1"), MeasureBlockSyndromes("b2"), Merge(["b1", "b2"], "m"), MeasureBlockSyndromes("m", n_cycles=2), Split("m", ("c1", "c2"), Orientation.VERTICAL, 1), MeasureBlockSyndromes("c1"), MeasureBlockSyndromes("c2"), MeasureLogicalZ("c1"), MeasureLogicalZ("c2")]),
        "grow_right": ([b1], [ResetAllDataQubits("b1"), MeasureBlockSyndromes("b1"), Grow("b1", Direction.RIGHT, 2), MeasureBlockSyndromes("b1", n_cycles=2), MeasureLogicalZ("b1")]),
    }
    blocks, operations = cases[name]
    return loom.interpret_eka(Eka(LATTICE, blocks, operations))


def fn(spec):
    """The probability functions the recorded configurations name."""
    p, T = spec["p"], spec["T"]
    if spec["fn"] == "drift":
        return lambda t, dt=None: [q * (1.0 + (t or 0.0) / T) for q in p]
    if spec["fn"] == "idle":
        return lambda t, dt=None: [q * (dt or 0.0) / T for q in p]
    raise KeyError(spec["fn"])


def build_models(specs, circuit):
    out = []
    for s in specs:
        kw = {"circuit": circuit}
        if "gate_durations" in s:
            kw["gate_durations"] = FIXTURES["durations"]
        if "error_type" in s:
            kw["error_type"] = ErrorType[s["error_type"]]
        if "application_mode" in s:
            kw["application_mode"] = ApplicationMode(s["application_mode"])
        if "target_gates" in s:
            kw["target_gates"] = s["target_gates"]
        if s["cls"] == "HomogeneousTimeIndependentCEM":
            out.append(HomogeneousTimeIndependentCEM(error_probability=s["error_probability"], **kw))
        elif s["cls"] == "HomogeneousTimeDependentCEM":
            out.append(HomogeneousTimeDependentCEM(error_probability=fn(s["fn"]), **kw))
        elif s["cls"] == "AsymmetricDepolarizeCEM":
            out.append(AsymmetricDepolarizeCEM(t1=s["t1"], t2=s["t2"], **kw))
        else:
            if "gate_fns" in s:
                kw["gate_error_probabilities"] = {g: fn(f) for g, f in s["gate_fns"].items()}
            if "global_fn" in s:
                kw["global_time_error_probability"] = fn(s["global_fn"])
            out.append(CircuitErrorModel(is_time_dependent=s["is_time_dependent"], **kw))
    return out


def build_zoo(with_t):
    zoo = FIXTURES["zoo"]
    chans = {}

    def gate(name, labels):
        for label in labels:
            chans.setdefault(label, Channel(type=ChannelType.CLASSICAL if label.startswith("c_") else ChannelType.QUANTUM, label=label))
        return Circuit(name, channels=[chans[x] for x in labels])

    ticks = [[gate(*g) for g in tick] for tick in zoo["ticks"]]
    if with_t:
        ticks.append([gate(*g) for g in zoo["t_tick"]])
    ticks.append([Circuit("inner", circuit=[[gate(*g) for g in tick] for tick in zoo["inner"]])])
    return Circuit("zoo", circuit=ticks)


def canonical(text: str) -> str:
    return str(sq.Circuit(text)).strip()


def noise_args(text: str):
    return [op.gate_args_copy() for op in sq.Circuit(text).flattened() if op.name in NOISE]


def assert_noisy_text_is_looms(ours: str, ref: dict) -> None:
    # Loom's text (Stim's, arguments to 6 digits), and the probabilities themselves.
    assert canonical(ours) == canonical(ref["stim"])
    got = noise_args(ours)
    assert len(got) == len(ref["args"])
    for a, b in zip(got, ref["args"]):
        assert a == pytest.approx(b, rel=1e-12, abs=0)


STEPS = {}


def step_of(name):
    if name not in STEPS:
        STEPS[name] = experiment(name)
    return STEPS[name]


# ---------------------------------------------------------------------------------------------
# Noisy Stim text
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,config", [(n, c) for n in sorted(FIXTURES["experiments"]) for c in sorted(FIXTURES["experiments"][n]["noisy"])])
def test_noisy_stim_is_looms(name, config):
    step = step_of(name)
    spec = FIXTURES["configs"][config]
    program = loom.EkaToStimConverter().convert(step, error_models=build_models(spec["models"], step.final_circuit), with_ticks=spec.get("with_ticks", False))
    assert_noisy_text_is_looms(program.text, FIXTURES["experiments"][name]["noisy"][config])
    # The detectors and observables are the noiseless circuit's, and the record indices too.
    plain = loom.EkaToStimConverter().convert(step)
    assert program.records == plain.records and program.num_measurements == plain.num_measurements
    circuit = program.circuit
    assert circuit.num_detectors == plain.circuit.num_detectors and circuit.num_observables == plain.circuit.num_observables


@pytest.mark.parametrize("config", sorted(c for c in FIXTURES["zoo_noise"] if c != "noiseless"))
def test_noisy_gate_zoo_is_looms(config):
    zoo = build_zoo(False)
    spec = FIXTURES["configs"][config]
    program = loom.EkaToStimConverter().convert_circuit(zoo, error_models=build_models(spec["models"], zoo), with_ticks=spec.get("with_ticks", False))
    assert_noisy_text_is_looms(program.text, FIXTURES["zoo_noise"][config])
    assert canonical(loom.EkaToStimConverter().convert_circuit(zoo).text) == canonical(FIXTURES["zoo_noise"]["noiseless"]["stim"])


def test_zero_probabilities_write_no_noise():
    zoo = build_zoo(False)
    models = [
        HomogeneousTimeIndependentCEM(circuit=zoo, error_type=ErrorType.DEPOLARIZING1, application_mode=ApplicationMode.AFTER_GATE, error_probability=0.0, target_gates=["h", "x"]),
        HomogeneousTimeDependentCEM(circuit=zoo, error_type=ErrorType.PAULI_CHANNEL, application_mode=ApplicationMode.IDLE_END_OF_TICK, error_probability=lambda t, dt=None: [0.0, 0.0, 0.0], gate_durations=FIXTURES["durations"], target_gates=["cx"]),
    ]
    assert loom.EkaToStimConverter().convert_circuit(zoo, error_models=models).text == loom.EkaToStimConverter().convert_circuit(zoo).text


def test_fused_text_is_stims():
    # The text the converter writes is already fused as Stim fuses it: equal arguments join.
    step = step_of("memory_z_d3")
    spec = FIXTURES["configs"]["idle"]
    text = loom.EkaToStimConverter().convert(step, error_models=build_models(spec["models"], step.final_circuit), with_ticks=True).text
    assert len(text.splitlines()) == len(str(sq.Circuit(text)).splitlines())


def test_noisy_memory_decodes_below_the_physical_rate():
    # Circuit-level noise at p = 1e-3 on Loom's d=3 memory (three rounds): the detector error
    # model this package builds from the noisy circuit decodes, with Matching, to a logical
    # error rate per round well below p.
    step = step_of("memory_z_d3")
    p = 1e-3
    circuit = loom.eka_to_circuit(step, error_models=[
        HomogeneousTimeIndependentCEM(circuit=step.final_circuit, error_type=ErrorType.DEPOLARIZING1, application_mode=ApplicationMode.AFTER_GATE, error_probability=p, target_gates=["h"]),
        HomogeneousTimeIndependentCEM(circuit=step.final_circuit, error_type=ErrorType.DEPOLARIZING2, application_mode=ApplicationMode.AFTER_GATE, error_probability=p, target_gates=["cx", "cz"]),
        HomogeneousTimeIndependentCEM(circuit=step.final_circuit, error_type=ErrorType.BIT_FLIP, application_mode=ApplicationMode.BEFORE_GATE, error_probability=p, target_gates=["measurement"]),
        HomogeneousTimeIndependentCEM(circuit=step.final_circuit, error_type=ErrorType.PAULI_X, application_mode=ApplicationMode.AFTER_GATE, error_probability=p, target_gates=["reset_0"]),
    ])
    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem.num_errors > 0 and dem.num_detectors == 24 and dem.num_observables == 1
    d, o = circuit.compile_detector_sampler(seed=11).sample(20000, separate_observables=True)
    predicted = sq.Matching.from_detector_error_model(dem).decode_batch(d)
    logical = (predicted != o).any(axis=1).mean()
    per_round = (1 - (1 - 2 * logical) ** (1 / 3)) / 2
    assert per_round < p / 2 and logical < o.any(axis=1).mean() / 10


def test_time_dependent_noise_decodes():
    # T1/T2 idling and depolarizing gates: the circuit's model decodes better than no decoder.
    step = step_of("merge_h")
    spec = FIXTURES["configs"]["t1t2_idle"]
    circuit = loom.EkaToStimConverter().convert(step, error_models=build_models(spec["models"], step.final_circuit)).circuit
    dem = circuit.detector_error_model(decompose_errors=True)
    d, o = circuit.compile_detector_sampler(seed=7).sample(4000, separate_observables=True)
    predicted = sq.Matching.from_detector_error_model(dem).decode_batch(d)
    assert (predicted != o).any(axis=1).mean() < 0.5 * o.any(axis=1).mean()


def test_with_ticks_positional_is_deprecated():
    step = step_of("memory_z_d3")
    with pytest.warns(DeprecationWarning, match="with_ticks by keyword"):
        old = loom.EkaToStimConverter().convert(step, True)
    assert old.text == loom.EkaToStimConverter().convert(step, with_ticks=True).text
    assert "TICK" in old.text


def test_legacy_noise_annotation_is_looms():
    step = step_of("memory_z_d3")
    plain = loom.EkaToStimConverter().convert(step)
    for given in (plain, plain.circuit, plain.text):
        out = loom.noise_annotated_stim_circuit(given, before_measure_flip_probability=0.001, after_clifford_depolarization=0.002, after_reset_flip_probability=0.003)
        assert canonical(str(out)) == canonical(FIXTURES["experiments"]["memory_z_d3"]["legacy_noise_annotated"])
    assert loom.noise_annotated_stim_circuit(plain) == plain.circuit


def test_refusals_are_looms():
    errors = FIXTURES["errors"]
    step = step_of("memory_z_d3")
    # A tick-wise model at probability zero: Loom writes DEPOLARIZE1 with no arguments, which
    # Stim refuses; so does this package's parser.
    assert errors["zero_tick_probability"][0] == "ValueError"
    zero = HomogeneousTimeIndependentCEM(circuit=step.final_circuit, error_type=ErrorType.DEPOLARIZING1, application_mode=ApplicationMode.END_OF_TICK, error_probability=0.0)
    program = loom.EkaToStimConverter().convert(step, error_models=[zero])
    assert "\nDEPOLARIZE1 0 1 2" in program.text
    with pytest.raises(ValueError, match="DEPOLARIZE1"):
        program.circuit
    with pytest.raises(KeyError, match=re.escape(errors["guppy_t_in_stim"][1].strip("'"))):
        loom.EkaToStimConverter().convert_circuit(build_zoo(True))
    with pytest.raises(ValueError, match=re.escape("import_prefix 'guppyQ' must end with a dot (.)")):
        loom.EkaToGuppylangConverter(quantum_prefix="guppyQ")
    with pytest.raises(TypeError, match="import_prefix must be a string"):
        loom.EkaToGuppylangConverter(builtins_prefix=3)
    with pytest.raises(ValueError, match=re.escape(errors["mimiq_unsupported"][1])):
        loom.EkaToMimiqConverter().convert_circuit(Circuit("zz", circuit=[[Circuit("hadamard_2", channels=[Channel(label="a")])]]))


# ---------------------------------------------------------------------------------------------
# Mimiq and Guppy
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["memory_z_d3", "merge_split_h_1"])
def test_mimiq_and_guppy_programs_are_looms(name):
    step = step_of(name)
    ref = FIXTURES["experiments"][name]
    assert loom.EkaToMimiqConverter().convert(step)[0] == ref["mimiq"]
    assert loom.EkaToGuppylangConverter().convert(step)[0] == ref["guppy"]


def test_mimiq_and_guppy_gate_zoo_is_looms():
    ref = FIXTURES["zoo_text"]
    zoo = build_zoo(True)
    assert loom.EkaToMimiqConverter().convert_circuit(zoo)[0] == ref["mimiq"]
    assert loom.EkaToMimiqConverter(mimiq_import_prefix="mymc", circuit_varname="my_circuit").convert_circuit(zoo)[0] == ref["mimiq_custom"]
    assert loom.EkaToGuppylangConverter().convert_circuit(zoo)[0] == ref["guppy"]
    assert loom.EkaToGuppylangConverter(quantum_prefix="", qsystem_prefix="qs.", builtins_prefix="b.").convert_circuit(zoo)[0] == ref["guppy_custom"]
    single = Circuit("h", channels=[Channel(label="(0, 0, 0)")])
    assert loom.EkaToMimiqConverter().convert_circuit(single)[0] == ref["mimiq_single_gate"]
    assert loom.EkaToGuppylangConverter().convert_circuit(single)[0] == ref["guppy_single_gate"]
    assert loom.EkaToMimiqConverter().convert_circuit(Circuit("empty"))[0] == ref["mimiq_empty"]
    assert repr(loom.EkaToGuppylangConverter().convert_circuit(Circuit("empty"))) == ref["guppy_empty"]
    # The registers: qubits and bits in label order.
    _, qmap, cmap = loom.EkaToGuppylangConverter().convert_circuit(zoo)
    assert [c.label for c in sorted(qmap, key=lambda c: qmap[c])] == sorted(c.label for c in qmap)
    assert sorted(cmap.values()) == [f"c[{i}]" for i in range(4)]


# ---------------------------------------------------------------------------------------------
# The error models: Loom's checks, messages and probabilities
# ---------------------------------------------------------------------------------------------

Q0, Q1 = Channel(label="q0"), Channel(label="q1")
C0 = Channel(type=ChannelType.CLASSICAL, label="c0")
SMALL = Circuit("c", circuit=[[Circuit("h", channels=[Q0])], [Circuit("cx", channels=[Q0, Q1])], [Circuit("measurement", channels=[Q1, C0])]])
D = {"h": 1, "cx": 2, "measurement": 3}


def htic(**kw):
    return HomogeneousTimeIndependentCEM(**{"circuit": SMALL, "error_type": ErrorType.DEPOLARIZING1, "application_mode": ApplicationMode.AFTER_GATE, "error_probability": 0.1, **kw})


def htdc(**kw):
    return HomogeneousTimeDependentCEM(**{"circuit": SMALL, "error_type": ErrorType.DEPOLARIZING1, "application_mode": ApplicationMode.END_OF_TICK, "error_probability": lambda a, b: [0.1], "gate_durations": D, **kw})


def base(**kw):
    return CircuitErrorModel(**{"circuit": SMALL, "error_type": ErrorType.PAULI_X, "is_time_dependent": False, "application_mode": ApplicationMode.AFTER_GATE, **kw})


# (Loom's messages, as its probes printed them; pydantic wraps the ValueErrors in a
# ValidationError, a ValueError too.)
REFUSED = [
    (lambda: htic(error_type=ErrorType.PAULI_CHANNEL, error_probability=0.001, target_gates=["h"]), ValueError, "Gate 'h' error probability function failed validation: PAULI_CHANNEL expects 3 parameter(s), got 1."),
    (lambda: htic(error_type=ErrorType.PAULI_CHANNEL, error_probability=0.001), ValueError, "global_time_error_probability function failed validation: PAULI_CHANNEL expects 3 parameter(s), got 1."),
    (lambda: htic(error_type=ErrorType.PAULI_CHANNEL, error_probability=[0.5, 0.5, 0.5]), ValueError, "global_time_error_probability function failed validation: PAULI_CHANNEL parameters sum must not exceed 1.0."),
    (lambda: htic(application_mode=ApplicationMode.END_OF_TICK, error_probability=1.5), ValueError, "global_time_error_probability function failed validation: All parameters for DEPOLARIZING1 must be in [0, 1]."),
    (lambda: htic(application_mode=ApplicationMode.IDLE_END_OF_TICK), ValueError, "Idle application mode can only be used with time-dependent error models."),
    (lambda: htic(application_mode=ApplicationMode.IDLE_END_OF_TICK, is_time_dependent=True), ValueError, "gate_durations must be provided for time-dependent error models."),
    (lambda: htic(gate_durations={"h": 1}), ValueError, "Missing gate durations for: "),
    (lambda: htic(circuit="x"), ValueError, "circuit: Input should be"),
    (lambda: htic(application_mode="whenever"), ValueError, "application_mode: Input should be 'before_gate', 'after_gate', 'end_of_tick' or 'idle_end_of_tick'"),
    (lambda: htic(error_type="pauli_x"), ValueError, "error_type: Input should be ('pauli_x', 1), ('pauli_y', 1), ('pauli_z', 1), ('pauli_channel', 3), ('bit-flip', 1), ('phase-flip', 1), ('depolarizing1', 1) or ('depolarizing2', 1)"),
    (lambda: htic(gate_error_probabilities={"cx": 3}), TypeError, "Value for 'cx' must be callable."),
    (lambda: htic(error_probability=None), ValueError, "error_probability must be a float or a list of floats."),
    (lambda: htdc(gate_durations=None), ValueError, "gate_durations must be provided for time-dependent error models."),
    (lambda: htdc(gate_durations={"h": 1}), ValueError, "Gate cx not found in gate_durations dictionary."),
    (lambda: htdc(application_mode=ApplicationMode.AFTER_GATE, target_gates=["h"]), ValueError, "Gate 'h' error probability function failed validation: "),
    (lambda: htdc(error_probability=0.1), ValueError, "error_probability: Input should be an instance of ErrorProbProtocol"),
    (lambda: htdc(error_probability=lambda a, b: 0.1), ValueError, "global_time_error_probability function failed validation: object of type 'float' has no len()"),
    (lambda: htdc(error_probability=lambda a, b: [0.1, 0.2]), ValueError, "global_time_error_probability function failed validation: DEPOLARIZING1 expects 1 parameter(s), got 2."),
    (lambda: htdc(gate_durations={"h": "x", "cx": 2, "measurement": 3}), ValueError, "gate_durations.h: Input should be a valid number"),
    (lambda: htdc(gate_durations=[1, 2]), ValueError, "gate_durations: Input should be a valid dictionary"),
    (lambda: AsymmetricDepolarizeCEM(circuit=SMALL, t1=1.0, t2=15.0, gate_durations=D), ValueError, "t2 must be smaller or equal than 2*t1."),
    (lambda: AsymmetricDepolarizeCEM(circuit=SMALL, t1=-1.0, t2=15.0, gate_durations=D), ValueError, "t1 must be non-negative."),
    (lambda: AsymmetricDepolarizeCEM(circuit=SMALL, t1=10.0, t2=15.0), ValueError, "gate_durations must be provided for time-dependent error models."),
    (lambda: AsymmetricDepolarizeCEM(circuit=SMALL, t1=1.0, t2=15.0, gate_durations=D, is_time_dependent=False, application_mode="idle_end_of_tick"), ValueError, "Idle application mode can only be used with time-dependent error models."),
    (lambda: AsymmetricDepolarizeCEM(circuit=SMALL, t1=10.0, t2=15.0, gate_durations=D, error_type=ErrorType.DEPOLARIZING1), ValueError, "global_time_error_probability function failed validation: DEPOLARIZING1 expects 1 parameter(s), got 3."),
    (lambda: base(gate_error_probabilities=[1]), TypeError, "gate_error_probabilities must be a dictionary"),
    (lambda: base(gate_error_probabilities={"h": 0.1}), TypeError, "Value for 'h' must be callable."),
    (lambda: base(gate_error_probabilities={"h": lambda: [0.1]}), ValueError, "Gate 'h' error probability function failed validation: "),
    (lambda: base(gate_error_probabilities={"h": lambda _: [0.1, 0.2]}), ValueError, "Gate 'h' error probability function failed validation: PAULI_X expects 1 parameter(s), got 2."),
    (lambda: base(gate_error_probabilities={"h": lambda _: ["a"]}), ValueError, "Gate 'h' error probability function failed validation: All parameters for PAULI_X must be numbers."),
    (lambda: base(application_mode=ApplicationMode.END_OF_TICK, global_time_error_probability=0.1), TypeError, "global_time_error_probability must be a callable"),
    (lambda: base(application_mode=ApplicationMode.END_OF_TICK, global_time_error_probability=lambda a, b: [1]), ValueError, "global_time_error_probability callable failed: global_time_error_probability must return a list offloats between 0 and 1"),
    (lambda: base(application_mode=ApplicationMode.END_OF_TICK, global_time_error_probability=lambda a, b: 0.1), ValueError, "global_time_error_probability callable failed: global_time_error_probability must return a list of floats"),
    (lambda: base(application_mode=ApplicationMode.END_OF_TICK, global_time_error_probability=lambda a, b: [0.1, 0.1]), ValueError, "global_time_error_probability function failed validation: PAULI_X expects 1 parameter(s), got 2."),
    (lambda: base(application_mode=ApplicationMode.END_OF_TICK, global_time_error_probability=lambda a, b: [a * 0.0 + 0.1]), ValueError, "global_time_error_probability function failed validation: unsupported operand type(s) for *: 'NoneType' and 'float'"),
    (lambda: base(application_mode=ApplicationMode.IDLE_END_OF_TICK, gate_durations={"h": 1}, gate_error_probabilities={"h": lambda _: [2.0]}), ValueError, "Idle application mode can only be used with time-dependent error models."),
    (lambda: base(gate_error_probabilities={"h": lambda _: [2.0]}, global_time_error_probability=lambda a, b: [0.1, 0.2]), ValueError, "Gate 'h' error probability function failed validation: All parameters for PAULI_X must be in [0, 1]."),
]


@pytest.mark.parametrize("k", range(len(REFUSED)))
def test_error_model_checks_are_looms(k):
    make, error, message = REFUSED[k]
    with pytest.raises(error, match=re.escape(message)):
        make()


def test_error_model_coercions_are_looms():
    assert htic(error_probability=0).error_probability == [0.0]
    assert htic(error_probability=1).error_probability == [1.0]
    assert htic(error_probability=[0]).error_probability == [0.0]
    assert htic(error_probability="0.1").error_probability == [0.1]
    assert htic(error_probability=(0.1,)).error_probability == [0.1]
    assert htic(application_mode="after_gate").application_mode is ApplicationMode.AFTER_GATE
    assert htic(error_type=("pauli_x", 1)).error_type is ErrorType.PAULI_X
    assert htic(target_gates=("h",)).target_gates == ["h"]
    model = htic(application_mode=ApplicationMode.IDLE_END_OF_TICK, is_time_dependent=True, gate_durations={"h": 1, "cx": 2, "measurement": 3})
    assert model.gate_durations == {"h": 1.0, "cx": 2.0, "measurement": 3.0} and model._tick_durations == [1.0, 2.0, 3.0]
    assert dict(model._idle_times) == {Q0.id: [0.0, 0.0, 3.0], Q1.id: [1.0, 0.0, 0.0]}
    assert AsymmetricDepolarizeCEM(circuit=SMALL, t1=10, t2="15", gate_durations=D).t2 == 15.0
    # Loom's default zero for a Pauli channel is three zeros.
    pauli = base(error_type=ErrorType.PAULI_CHANNEL, application_mode=ApplicationMode.END_OF_TICK)
    assert pauli.global_time_error_probability(None, None) == [0.0, 0.0, 0.0] and pauli.get_tick_error_probability() is None
    with pytest.raises(TypeError):
        HomogeneousTimeIndependentCEM(SMALL, ErrorType.PAULI_X, ApplicationMode.AFTER_GATE, 0.1)


def test_error_model_probabilities_are_looms():
    h, cx = SMALL.circuit[0][0], SMALL.circuit[1][0]
    m = htic(error_type=ErrorType.PAULI_CHANNEL, error_probability=[0.1, 0.2, 0.3], target_gates=["h"])
    assert m.get_gate_error_probability(h) == [0.1, 0.2, 0.3] and m.get_gate_error_probability(cx) is None
    assert m.gate_error_probabilities["zz"](None) == [0.0, 0.0, 0.0] and m.get_tick_error_probability(0) is None
    tick = htic(error_type=ErrorType.PAULI_CHANNEL, application_mode=ApplicationMode.END_OF_TICK, error_probability=[0.1, 0.2, 0.3])
    assert tick.get_tick_error_probability() == tick.get_tick_error_probability(7) == [0.1, 0.2, 0.3] and tick.get_gate_error_probability(h) is None
    idle = AsymmetricDepolarizeCEM(circuit=SMALL, t1=10.0, t2=15.0, gate_durations=D, application_mode="idle_end_of_tick")
    got = [[idle.get_idle_tick_error_probability(i, ch.id) for i in range(3)] for ch in (Q0, Q1)]
    assert got[0][:2] == [None, None] and got[1][1:] == [None, None]
    assert got[0][2] == pytest.approx([0.06479544482957053, 0.06479544482957053, 0.02583917863143856], rel=1e-14)
    assert got[1][0] == pytest.approx([0.02379064549101012, 0.02379064549101012, 0.00845586199318099], rel=1e-14)
    with pytest.raises(IndexError, match="Tick index out of range."):
        idle.get_idle_tick_error_probability(3, Q0.id)
    with pytest.raises(ValueError, match="Channel nope not found in idle times mapping."):
        idle.get_idle_tick_error_probability(0, "nope")
    assert idle.get_tick_error_probability(0) is None
    t1t2 = AsymmetricDepolarizeCEM(circuit=SMALL, t1=10.0, t2=15.0, gate_durations=D)
    assert t1t2.error_type is ErrorType.PAULI_CHANNEL and t1t2.application_mode is ApplicationMode.END_OF_TICK
    assert t1t2.get_tick_error_probability(1) == pytest.approx([0.045317311730504545, 0.045317311730504545, 0.017096028748021713], rel=1e-14)
    with pytest.raises(ValueError, match="tick_index must be provided for time-dependent models."):
        t1t2.get_tick_error_probability()
    with pytest.raises(IndexError, match="Tick index out of range."):
        t1t2.get_tick_error_probability(3)
    assert t1t2.total_time == 6.0
    assert AsymmetricDepolarizeCEM(circuit=SMALL, t1=0, t2=0, gate_durations=D).get_tick_error_probability(0) == [0.25, 0.25, 0.25]
    gates = htdc(application_mode=ApplicationMode.AFTER_GATE, error_probability=lambda t, dt=None: [0.01 * (1 + t)], target_gates=["h", "cx"])
    assert gates._op_time == {h.id: 0, cx.id: 1.0, SMALL.circuit[2][0].id: 3.0}
    assert gates.get_gate_error_probability(h) == [0.02] and gates.get_gate_error_probability(cx) == pytest.approx([0.04])
    plain = base(gate_error_probabilities={"h": lambda _: [0.1]})
    assert plain.get_gate_error_probability(h) == [0.1] and plain.get_gate_error_probability(cx) is None
    with pytest.raises(ValueError, match="Circuit is not time-dependent or tick durations are not set."):
        plain.total_time
    with pytest.raises(ValueError, match=re.escape("gate must have empty children (no sub-circuit)")):
        plain.get_gate_error_probability(SMALL)
    with pytest.raises(TypeError, match="gate must be an instance of Circuit"):
        plain.get_gate_error_probability("h")
    with pytest.raises(AttributeError):
        plain.error_type = ErrorType.PAULI_Z


def test_update_gate_durations_recomputes_the_times():
    # (Loom's update_gate_durations raises AttributeError; here it is the copy its docstring
    # describes.)
    model = AsymmetricDepolarizeCEM(circuit=SMALL, t1=10.0, t2=15.0, gate_durations=D)
    faster = model.update_gate_durations({"h": 0.5, "cx": 1, "measurement": 1})
    assert faster is not model and faster.t1 == 10.0 and faster._tick_durations == [0.5, 1.0, 1.0] and faster.total_time == 2.5
    assert model.total_time == 6.0
    with pytest.raises(ValueError, match="Gate cx not found in gate_durations dictionary."):
        model.update_gate_durations({"h": 1})


def test_error_type_parameters_are_looms():
    assert [e.label for e in ErrorType] == ["pauli_x", "pauli_y", "pauli_z", "pauli_channel", "bit-flip", "phase-flip", "depolarizing1", "depolarizing2"]
    assert [e.param_count for e in ErrorType] == [1, 1, 1, 3, 1, 1, 1, 1]
    assert ErrorType.PAULI_X.validate_params([True]) is None
    for et, params, message in [(ErrorType.PAULI_X, [0.1, 0.2], "PAULI_X expects 1 parameter(s), got 2."), (ErrorType.PAULI_X, [-0.1], "All parameters for PAULI_X must be in [0, 1]."), (ErrorType.PAULI_CHANNEL, [0.5, 0.6, 0.0], "PAULI_CHANNEL parameters sum must not exceed 1.0.")]:
        with pytest.raises(ValueError, match=re.escape(message)):
            et.validate_params(params)
    assert loom.EkaToStimConverter.SUPPORTED_ERROR_TYPES == frozenset(ErrorType)
    assert loom.EkaToStimConverter.ALLOW_ERROR_MODELS and not loom.EkaToMimiqConverter.ALLOW_ERROR_MODELS and not loom.EkaToGuppylangConverter.ALLOW_ERROR_MODELS


# ---------------------------------------------------------------------------------------------
# Against Loom installed
# ---------------------------------------------------------------------------------------------

try:
    from loom.executor import EkaToStimConverter as TheirStim  # noqa: F401

    their_loom = True
except ImportError:  # pragma: no cover
    their_loom = False


@pytest.mark.skipif(not their_loom, reason="el-loom is not installed")
def test_noisy_memory_against_loom_live():
    from loom.eka import Eka as TheirEka, Lattice as TheirLattice
    from loom.eka.operations import MeasureBlockSyndromes as TheirMBS, MeasureLogicalZ as TheirMLZ, ResetAllDataQubits as TheirReset
    from loom.executor import ApplicationMode as TheirMode, AsymmetricDepolarizeCEM as TheirT1T2, ErrorType as TheirType, HomogeneousTimeIndependentCEM as TheirHTIC
    from loom.interpreter import interpret_eka as their_interpret
    from loom_rotated_surface_code.code_factory import RotatedSurfaceCode as TheirRSC

    lattice = TheirLattice.square_2d((15, 15))
    theirs = their_interpret(TheirEka(lattice, blocks=[TheirRSC.create(3, 3, lattice, unique_label="b1")], operations=[TheirReset("b1"), TheirMBS("b1", n_cycles=3), TheirMLZ("b1")]))
    their_models = [TheirHTIC(circuit=theirs.final_circuit, error_type=TheirType.DEPOLARIZING2, application_mode=TheirMode.AFTER_GATE, error_probability=0.001, target_gates=["cx", "cz"]), TheirT1T2(circuit=theirs.final_circuit, t1=20000.0, t2=30000.0, gate_durations=FIXTURES["durations"], application_mode=TheirMode.IDLE_END_OF_TICK)]
    text = str(TheirStim().convert(theirs, error_models=their_models, with_ticks=True)[0])
    step = step_of("memory_z_d3")
    ours = [HomogeneousTimeIndependentCEM(circuit=step.final_circuit, error_type=ErrorType.DEPOLARIZING2, application_mode=ApplicationMode.AFTER_GATE, error_probability=0.001, target_gates=["cx", "cz"]), AsymmetricDepolarizeCEM(circuit=step.final_circuit, t1=20000.0, t2=30000.0, gate_durations=FIXTURES["durations"], application_mode=ApplicationMode.IDLE_END_OF_TICK)]
    assert canonical(loom.EkaToStimConverter().convert(step, error_models=ours, with_ticks=True).text) == canonical(text)
