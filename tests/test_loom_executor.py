"""The rest of Loom's executor in the port (`stabilizer_qec.loom`): its circuit error models,
the noise they write into the Stim circuit, and its Mimiq and Guppy exporters, against fixtures
recorded from el-loom 0.4.0 (tests/loom_port/fixtures_executor.json): the noisy Stim text of
each error-model type and application mode on four interpreted experiments and a gate zoo,
the Mimiq and Guppy texts and Loom's validation messages; and the noisy circuits decode."""

from __future__ import annotations

import builtins
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
# If/else circuits, against el-loom 0.4.0 (tests/loom_port/fixtures_ifelse.json)
# ---------------------------------------------------------------------------------------------

IFELSE = json.loads((pathlib.Path(__file__).parent / "loom_port" / "fixtures_ifelse.json").read_text())

# ----- build_ifelse() is the recorder's, verbatim; ns is a namespace of Loom's names -----
IFELSE_CIRCUITS = ["rsc_match_paulis", "rsc_not_else", "rsc_nested_or", "rep_and_nor_measure", "rep_xor_nand_else", "rep_match_not_feedforward"]


def ifelse_base(ns, code):
    """A distance-3 memory, reset and syndrome rounds, interpreted: its final circuit and its
    channels by label."""
    if code == "rsc":
        lattice = ns.Lattice.square_2d((15, 15))
        block, cycles = ns.RotatedSurfaceCode.create(3, 3, lattice, unique_label="b"), 1
    else:
        lattice = ns.Lattice.linear((10,))
        block, cycles = ns.RepetitionCode.create(3, "Z", lattice, unique_label="b"), 2
    step = ns.interpret_eka(ns.Eka(lattice, [block], [ns.ResetAllDataQubits("b"), ns.MeasureBlockSyndromes("b", n_cycles=cycles)]))
    return step.final_circuit, {c.label: c for c in step.final_circuit.channels}


def build_ifelse(ns, name):
    """The recorded if/else circuits: an interpreted memory followed by branches on its syndrome
    bits (and on bits measured in the branches)."""
    C, IE, B = ns.Circuit, ns.IfElseCircuit, ns.BoolOp
    base, ch = ifelse_base(ns, name[:3])
    new = {}

    def chan(label):
        if label not in ch and label not in new:
            new[label] = ns.Channel(type=ns.ChannelType.CLASSICAL, label=label)
        return ch.get(label) or new[label]

    def g(op, *labels):
        return C(op, channels=[chan(x) for x in labels])

    if name == "rsc_match_paulis":
        # Single-bit conditions on single Paulis: what Stim and Mimiq take. Two conditional Ys
        # share a tick, and a plain gate sits between the branches.
        body = C("corrections", circuit=[
            [IE(if_circuit=g("x", "(1, 1, 0)"), condition_circuit=g(B.MATCH, "c_(1, 1, 1)_0")), IE(if_circuit=g("z", "(2, 2, 0)"), condition_circuit=g("match", "c_(2, 1, 1)_0"))],
            [IE(if_circuit=g("y", "(0, 0, 0)"), condition_circuit=g("MATCH", "c_(0, 1, 1)_0")), IE(if_circuit=g("y", "(2, 0, 0)"), condition_circuit=g(B.MATCH, "c_(2, 0, 1)_0"))],
            [g("h", "(1, 1, 0)")],
            [IE(if_circuit=g("x", "(1, 1, 0)"), condition_circuit=g(B.MATCH, "c_(1, 2, 1)_0"))],
        ])
    elif name == "rsc_not_else":
        # NOT, else branches, a composite if branch, a measurement in a branch, an empty if branch.
        flip = C("flip", circuit=[[g("x", "(0, 0, 0)"), g("z", "(1, 0, 0)")], [g("h", "(0, 0, 0)")]])
        body = C("branches", circuit=[
            [IE(if_circuit=flip, else_circuit=g("h", "(1, 0, 0)"), condition_circuit=g(B.NOT, "c_(1, 2, 1)_0"))],
            [IE(if_circuit=C("measure_data", circuit=[g("measurement", "(2, 2, 0)", "c_extra")]), condition_circuit=g(B.NOT, "c_(3, 2, 1)_0")), IE(else_circuit=g("reset_+", "(0, 2, 0)"), condition_circuit=g(B.MATCH, "c_(1, 3, 1)_0"))],
        ])
    elif name == "rsc_nested_or":
        # OR over two bits, an if/else three deep in the if branch, an AND if/else (measuring)
        # in a composite else branch; then a plain gate.
        deep = IE(if_circuit=IE(if_circuit=g("x", "(1, 1, 0)"), condition_circuit=g(B.MATCH, "c_(2, 0, 1)_0")), condition_circuit=g(B.NOT, "c_(2, 2, 1)_0"))
        inner_else = IE(if_circuit=g("measure_y", "(1, 1, 0)", "c_m1"), else_circuit=g("z", "(1, 1, 0)"), condition_circuit=g(B.AND, "c_(1, 1, 1)_0", "c_(2, 1, 1)_0"))
        outer = IE(if_circuit=deep, else_circuit=C("else_block", circuit=[inner_else, g("h", "(1, 1, 0)")]), condition_circuit=g(B.OR, "c_(0, 1, 1)_0", "c_(1, 2, 1)_0"))
        body = C("after", circuit=[outer, g("y", "(0, 0, 0)")])
    elif name == "rep_and_nor_measure":
        # AND on a measurement, NOR on a Pauli, MATCH on the bit measured in the branch.
        body = C("feedforward", circuit=[
            IE(if_circuit=g("measurement", "(1, 0)", "c_m"), condition_circuit=g(B.AND, "c_(0, 1)_0", "c_(1, 1)_0")),
            IE(if_circuit=g("x", "(2, 0)"), condition_circuit=g(B.NOR, "c_(0, 1)_1", "c_(1, 1)_1")),
            IE(if_circuit=g("z", "(0, 0)"), condition_circuit=g(B.MATCH, "c_m")),
        ])
    elif name == "rep_xor_nand_else":
        # XOR and NAND over two and three bits, multi-instruction resets and measurements.
        body = C("parities", circuit=[
            IE(if_circuit=g("x", "(1, 0)"), condition_circuit=g(B.XOR, "c_(0, 1)_0", "c_(1, 1)_0")),
            IE(if_circuit=C("fix", circuit=[g("z", "(0, 0)"), g("z", "(2, 0)")]), else_circuit=g("reset_-i", "(1, 0)"), condition_circuit=g(B.NAND, "c_(0, 1)_1", "c_(1, 1)_1", "c_(0, 1)_0")),
            IE(if_circuit=g("measure_x", "(2, 0)", "c_x"), else_circuit=g("phaseinv", "(2, 0)"), condition_circuit=g(B.XOR, "c_(0, 1)_0", "c_(1, 1)_0", "c_(1, 1)_1")),
        ])
    elif name == "rep_match_not_feedforward":
        # MATCH and NOT on single Paulis (Stim and Mimiq take them), on syndrome bits and on a
        # bit measured after the first branches.
        body = C("feedforward", circuit=[
            [IE(if_circuit=g("x", "(1, 0)"), condition_circuit=g(B.MATCH, "c_(0, 1)_0")), IE(if_circuit=g("z", "(2, 0)"), condition_circuit=g(B.MATCH, "c_(1, 1)_0"))],
            [g("measurement", "(1, 0)", "c_m")],
            [IE(if_circuit=g("y", "(1, 0)"), condition_circuit=g(B.NOT, "c_m")), IE(if_circuit=g("x", "(0, 0)"), condition_circuit=g(B.NOT, "c_(0, 1)_1"))],
        ])
    else:
        raise KeyError(name)
    return C(name, circuit=[base, body])
# ----- end of the recorder's build_ifelse() -----


IFELSE_STEPS = {}


def ifelse_circuit(name):
    if name not in IFELSE_STEPS:
        IFELSE_STEPS[name] = build_ifelse(loom, name)
    return IFELSE_STEPS[name]


def text_converters():
    return {"qasm": loom.EkaToQasmConverter(), "cudaq": loom.EkaToCudaqConverter(), "pennylane": loom.EkaToPennylaneConverter(), "catalyst": loom.EkaToPennylaneConverter(is_catalyst=True, import_prefix="pl"), "mimiq": loom.EkaToMimiqConverter(), "guppy": loom.EkaToGuppylangConverter()}


def by_label(mapping):
    return {channel.label: value for channel, value in mapping.items()}


def unid(text, circuit):
    """The text with channel ids as labels, as recorded (Loom's PennyLane registers name the
    qubits by id)."""
    for channel in circuit.channels:
        text = text.replace(channel.id, channel.label)
    return text


def loom_message(recorded):
    """The message of a recorded refusal: pydantic's ValidationError wraps a validator's
    ValueError message as 'Value error, <message> [type=...'."""
    kind, text = recorded
    if kind == "ValidationError":
        return re.search(r"Value error, (.*) \[type=", text, re.S).group(1)
    return text


def raises_looms(recorded, make):
    """Raises the recorded error's type (pydantic's ValidationError, a ValueError, here a
    ValueError)."""
    kind = ValueError if recorded[0] == "ValidationError" else getattr(builtins, recorded[0])
    with pytest.raises(kind) as info:
        make()
    return info


@pytest.mark.parametrize("target", ["qasm", "cudaq", "pennylane", "catalyst", "mimiq", "guppy"])
@pytest.mark.parametrize("name", IFELSE_CIRCUITS)
def test_ifelse_text_is_looms(name, target):
    ref = IFELSE["circuits"][name][target]
    converter = text_converters()[target]
    if "error" in ref:
        info = raises_looms(ref["error"], lambda: converter.convert_circuit(ifelse_circuit(name)))
        assert str(info.value) == ref["error"][1]
        return
    circuit = ifelse_circuit(name)
    text, qubits, bits = converter.convert_circuit(circuit)
    assert unid(text, circuit) == ref["text"]
    assert by_label(qubits) == ref["qmap"] and by_label(bits) == ref["cmap"]


@pytest.mark.parametrize("name", IFELSE_CIRCUITS)
def test_ifelse_stim_is_looms(name):
    ref = IFELSE["circuits"][name]["stim"]
    if "error" in ref:
        info = raises_looms(ref["error"], lambda: loom.EkaToStimConverter().convert_circuit(ifelse_circuit(name)))
        assert str(info.value) == ref["error"][1]
        return
    lines, _, records, counter = loom.EkaToStimConverter().emit_circuit_program(ifelse_circuit(name))
    assert lines == ref["lines"]
    assert by_label(records) == ref["records"] and counter == ref["num_measurements"]
    program = loom.EkaToStimConverter().convert_circuit(ifelse_circuit(name))
    assert str(program.circuit) == ref["stim"]
    assert ifelse_circuit(name).duration == IFELSE["circuits"][name]["duration"]


def test_ifelse_noisy_stim_is_looms():
    # A time-independent model gives the conditional Paulis no noise (their gate is Loom's
    # classical_controlled_<pauli>, not a gate of the circuit); a time-dependent one cannot
    # place them in time, and wants a duration for the if/else itself.
    circuit = ifelse_circuit("rsc_match_paulis")
    ref = IFELSE["circuits"]["rsc_match_paulis"]["stim_noisy"]
    models = [HomogeneousTimeIndependentCEM(circuit=circuit, error_type=ErrorType.DEPOLARIZING1, application_mode=ApplicationMode.AFTER_GATE, error_probability=0.01, target_gates=["x", "y", "z", "h"]), HomogeneousTimeIndependentCEM(circuit=circuit, error_type=ErrorType.PAULI_X, application_mode=ApplicationMode.END_OF_TICK, error_probability=0.002)]
    lines, _, records, _ = loom.EkaToStimConverter().emit_circuit_program(circuit, error_models=models, with_ticks=True)
    assert lines == ref["lines"] and by_label(records) == ref["records"]
    assert str(loom.EkaToStimConverter().convert_circuit(circuit, error_models=models, with_ticks=True).circuit) == ref["stim"]
    durations = {"x": 1, "y": 1, "z": 1, "h": 1, "cx": 2, "cz": 2, "reset": 1, "reset_0": 1, "measurement": 5, "if-else_circuit": 1}
    t1t2 = AsymmetricDepolarizeCEM(circuit=circuit, t1=1000.0, t2=1500.0, gate_durations=durations, application_mode=ApplicationMode.AFTER_GATE)
    uuid = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    recorded = IFELSE["errors"]["stim_time_dependent"]
    assert recorded[0] == "ValueError"
    with pytest.raises(ValueError) as info:
        loom.EkaToStimConverter().emit_circuit_program(circuit, error_models=[t1t2])
    # The id is the conditional Pauli's, a gate Loom makes for it, so in no time mapping.
    assert re.sub(uuid, "UUID", str(info.value)) == re.sub(uuid, "UUID", recorded[1])
    with pytest.raises(ValueError, match=re.escape(loom_message(IFELSE["errors"]["cem_missing_ifelse_duration"]))):
        AsymmetricDepolarizeCEM(circuit=circuit, t1=1000.0, t2=1500.0, gate_durations={k: v for k, v in durations.items() if k != "if-else_circuit"}, application_mode=ApplicationMode.AFTER_GATE)
    # Noise on a conditional Pauli: Loom's noise line cannot take its record target.
    q0, q1, c0 = Channel(label="q0"), Channel(label="q1"), Channel(type=ChannelType.CLASSICAL, label="c0")
    small = Circuit("o", circuit=[[Circuit("measurement", channels=[q1, c0])], [loom.IfElseCircuit(if_circuit=Circuit("x", channels=[q0]), condition_circuit=Circuit("match", channels=[c0]))]])
    model = CircuitErrorModel(circuit=small, error_type=ErrorType.DEPOLARIZING1, application_mode=ApplicationMode.AFTER_GATE, is_time_dependent=False, gate_error_probabilities={"classical_controlled_x": lambda t: [0.01]})
    with pytest.raises(AttributeError, match="'int' object has no attribute 'value'"):
        loom.EkaToStimConverter().convert_circuit(small, error_models=[model])


def small_ifelse():
    """The recorder's small circuits, on labelled channels."""
    q0, q1 = Channel(label="q0"), Channel(label="q1")
    c0, c1, c2 = (Channel(type=ChannelType.CLASSICAL, label=f"c{i}") for i in range(3))
    small = loom.IfElseCircuit(if_circuit=Circuit("x", channels=[q0]), else_circuit=Circuit("long", circuit=[Circuit("y", channels=[q1]), Circuit("z", channels=[q1])]), condition_circuit=Circuit("xor", channels=[c0, c1]))
    outer = Circuit("outer", circuit=[[Circuit("measurement", channels=[q0, c0]), Circuit("measurement", channels=[q1, c1])], [small]])
    return (q0, q1, c0, c1, c2), small, outer


def test_ifelse_circuit_is_looms():
    (q0, q1, c0, c1, c2), small, outer = small_ifelse()
    ref = IFELSE["strings"]
    assert repr(small) == ref["repr"] and small.detailed_str() == ref["detailed"]
    assert repr(outer) == ref["outer_repr"] and outer.detailed_str() == ref["outer_detailed"]
    assert repr(loom.IfElseCircuit()) == ref["default_repr"] and repr(outer.flatten()) == ref["flatten_repr"]
    assert small.duration == ref["duration"] and [c.label for c in small.channels if c.is_classical()] == ref["channels"]
    assert [[repr(g) for g in tick] for tick in Circuit.unroll(outer)] == ref["unroll"]
    unrolled = loom.IfElseCircuit.unroll(small)[0]
    assert [unrolled.if_circuit.name, repr(unrolled.else_circuit)] == ref["unrolled_branches"]
    wrapped = loom.IfElseCircuit(if_circuit=Circuit("cx", channels=[q0, q1], duration=2))
    assert [wrapped.if_circuit.name, wrapped.duration] == ref["wrapped"]
    assert str(loom.BoolOp("XOR")) == ref["boolop_upper"] and loom.BoolOp("XOR") is loom.BoolOp.XOR
    assert loom.EkaToQasmConverter().convert_circuit(small)[0] == ref["qasm_top"]
    info = raises_looms(ref["stim_unmeasured"], lambda: loom.EkaToStimConverter().convert_circuit(loom.IfElseCircuit(if_circuit=Circuit("x", channels=[q0]), condition_circuit=Circuit("match", channels=[c0]))))
    assert str(info.value) == ref["stim_unmeasured"][1]
    # The channels: the condition's, then the branches' (quantum first); a condition bit a
    # branch also uses appears twice, as in Loom.
    assert small.channels[:2] == (c0, c1) and set(small.channels[2:]) == {q0, q1}
    shared = loom.IfElseCircuit(if_circuit=Circuit("measurement", channels=[q0, c0]), condition_circuit=Circuit("match", channels=[c0]))
    assert shared.channels == (c0, q0, c0)
    # Equality: branches and conditions, labels and ids aside; Circuit's equality stops at the
    # first if/else it meets, as Loom's does.
    again = loom.IfElseCircuit(if_circuit=Circuit("x", channels=[Channel()]), else_circuit=Circuit("long", circuit=[Circuit("y", channels=[q1]), Circuit("z", channels=[q1])]), condition_circuit=Circuit("xor", channels=[c0, c2]))
    assert again == small and small != loom.IfElseCircuit(if_circuit=Circuit("z", channels=[q0]), condition_circuit=Circuit("match", channels=[c0])) and small != Circuit("x", channels=[q0]) and Circuit("x", channels=[q0]) != small
    later = [Circuit("o", circuit=[[small], [], [Circuit(g, channels=[q0])]]) for g in ("h", "x")]
    assert later[0] == later[1] and outer == outer.clone()
    assert hash(small) == hash(small.id)
    # Cloning keeps the branches and maps their channels.
    cloned = outer.clone()
    assert cloned.circuit[1][0].if_circuit.channels[0] is cloned.circuit[0][0].channels[0]
    assert cloned.circuit[1][0].condition_circuit.channels == (cloned.circuit[0][0].channels[1], cloned.circuit[0][1].channels[1])
    assert isinstance(small.clone([c2, Channel(type=ChannelType.CLASSICAL), Channel(), Channel()]), loom.IfElseCircuit)


def test_ifelse_refusals_are_looms():
    (q0, q1, c0, c1, c2), small, _ = small_ifelse()
    errors = IFELSE["errors"]
    uuid = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    cases = {
        "cond_quantum": lambda: loom.IfElseCircuit(condition_circuit=Circuit("match", channels=[q0])),
        "cond_and_one": lambda: loom.IfElseCircuit(condition_circuit=Circuit("and", channels=[c0])),
        "cond_not_two": lambda: loom.IfElseCircuit(condition_circuit=Circuit("not", channels=[c0, c1])),
        "cond_no_channel": lambda: loom.IfElseCircuit(condition_circuit=Circuit("match")),
        "cond_unknown": lambda: loom.IfElseCircuit(condition_circuit=Circuit("maybe", channels=[c0])),
        "as_gate": lambda: loom.IfElseCircuit.as_gate(),
        "circuit_seq": lambda: small.circuit_seq(),
        "padded": lambda: loom.IfElseCircuit.construct_padded_circuit_time_sequence(),
        "boolop_unknown": lambda: loom.BoolOp("Maybe"),
        "stim_non_pauli": lambda: loom.EkaToStimConverter().convert_circuit(Circuit("o", circuit=[[Circuit("measurement", channels=[q1, c0])], [loom.IfElseCircuit(if_circuit=Circuit("h", channels=[q0]), condition_circuit=Circuit("match", channels=[c0]))]])),
        "mimiq_t": lambda: loom.EkaToMimiqConverter().convert_circuit(Circuit("o", circuit=[[Circuit("measurement", channels=[q1, c0])], [loom.IfElseCircuit(if_circuit=Circuit("t", channels=[q0]), condition_circuit=Circuit("match", channels=[c0]))]])),
        "mimiq_or": lambda: loom.EkaToMimiqConverter().convert_circuit(loom.IfElseCircuit(if_circuit=Circuit("x", channels=[q0]), condition_circuit=Circuit("or", channels=[c0, c1]))),
        "qasm_composite_condition": lambda: loom.EkaToQasmConverter().convert_circuit(loom.IfElseCircuit(if_circuit=Circuit("x", channels=[q0]), condition_circuit=Circuit("match", circuit=[Circuit("match", channels=[c0])]))),
        "qasm_x_two_qubits": lambda: loom.EkaToQasmConverter().convert_circuit(Circuit("o", circuit=[[Circuit("x", channels=[q0, q1])]])),
        "qasm_measurement_no_bit": lambda: loom.EkaToQasmConverter().convert_circuit(Circuit("o", circuit=[[Circuit("measurement", channels=[q0])]])),
        "guppy_x_with_bit": lambda: loom.EkaToGuppylangConverter().convert_circuit(Circuit("o", circuit=[[Circuit("x", channels=[q0, c0])]])),
        "cudaq_cx_one_qubit": lambda: loom.EkaToCudaqConverter().convert_circuit(Circuit("o", circuit=[[Circuit("cx", channels=[q0])]])),
        "pennylane_composite_gate": lambda: loom.EkaToPennylaneConverter().convert_circuit(Circuit("o", circuit=[[Circuit("x", circuit=[Circuit("x", channels=[q0])])]])),
    }
    assert set(cases) | {"stim_time_dependent", "cem_missing_ifelse_duration"} == set(errors)
    for key, make in cases.items():
        info = raises_looms(errors[key], make)
        expected = loom_message(errors[key])
        if key == "cond_quantum":
            # The channel's repr carries its random id.
            assert re.sub(uuid, "UUID", str(info.value)) == re.sub(uuid, "UUID", expected)
        else:
            assert str(info.value) == expected, key
    with pytest.raises(TypeError):
        loom.IfElseCircuit(Circuit("x", channels=[q0]))
    with pytest.raises(ValueError, match="if_circuit: Input should be an instance of Circuit"):
        loom.IfElseCircuit(if_circuit="x")


def test_ifelse_gates_outside_a_condition_are_looms():
    # Bool operations and control flow as plain gates: Loom writes their text where they stand.
    q0, q1 = Channel(label="q0"), Channel(label="q1")
    c0, c1, c2 = (Channel(type=ChannelType.CLASSICAL, label=f"c{i}") for i in range(3))
    loose = Circuit("loose", circuit=[[Circuit("measurement", channels=[q0, c0]), Circuit("measurement", channels=[q1, c1])], [Circuit("and", channels=[c0, c1])], [Circuit("classical_if", description="flag"), Circuit("not", channels=[c2])], [Circuit("end_if", description="flag, is_else=False")]])
    converters = text_converters()
    for target, ref in IFELSE["strings"]["loose"].items():
        text, qubits, bits = converters[target].convert_circuit(loose)
        assert text == ref["text"] and by_label(qubits) == ref["qmap"] and by_label(bits) == ref["cmap"]


def test_ifelse_feedforward_runs_on_the_simulator():
    # Stim's classically controlled Paulis, sampled here: X on the measured 1 flips the target.
    q0, q1, c0, c1 = Channel(label="(0, 0)"), Channel(label="(1, 0)"), Channel(type=ChannelType.CLASSICAL, label="c0"), Channel(type=ChannelType.CLASSICAL, label="c1")
    circuit = Circuit("ff", circuit=[[Circuit("reset", channels=[q0]), Circuit("reset", channels=[q1])], [Circuit("x", channels=[q0])], [Circuit("measurement", channels=[q0, c0])], [loom.IfElseCircuit(if_circuit=Circuit("x", channels=[q1]), condition_circuit=Circuit("match", channels=[c0]))], [Circuit("measurement", channels=[q1, c1])]])
    program = loom.EkaToStimConverter().convert_circuit(circuit)
    assert "CX rec[-1] 1" in program.text
    shots = program.circuit.compile_sampler(seed=1).sample(20)
    assert shots[:, program.records[c1]].all() and shots[:, program.records[c0]].all()


def test_ifelse_round_trips_and_loom_paths():
    (q0, q1, c0, c1, c2), small, outer = small_ifelse()
    again = Circuit.fromdict(json.loads(json.dumps(outer.asdict())))
    assert isinstance(again.circuit[1][0], loom.IfElseCircuit) and again == outer
    assert loom.EkaToGuppylangConverter().convert_circuit(again)[0] == loom.EkaToGuppylangConverter().convert_circuit(outer)[0]
    from stabilizer_qec.loom.eka import IfElseCircuit
    from stabilizer_qec.loom.eka.ifelse_circuit import IfElseCircuit as Same
    from stabilizer_qec.loom.eka.utilities import BoolOp

    assert IfElseCircuit is Same is loom.IfElseCircuit and BoolOp is loom.BoolOp
    assert [str(b) for b in BoolOp.multi_bit_list()] == ["and", "nand", "or", "nor", "xor"] and BoolOp.mono_bit_list() == [BoolOp.NOT, BoolOp.MATCH]
    assert small.is_condition_single_bit is False and small.is_single_gate_conditioned is False
    single = IfElseCircuit(if_circuit=Circuit("x", channels=[q0]), condition_circuit=Circuit("not", channels=[c2]))
    assert single.is_condition_single_bit and single.is_single_gate_conditioned and single.if_circuit.name == "wrapped_x"
    flat = single.flatten()
    assert isinstance(flat, IfElseCircuit) and flat.condition_circuit.circuit and flat == single


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


@pytest.mark.skipif(not their_loom, reason="el-loom is not installed")
@pytest.mark.parametrize("name", IFELSE_CIRCUITS)
def test_ifelse_against_loom_live(name):
    from types import SimpleNamespace

    import loom.eka as their_eka
    import loom.executor as their_executor
    from loom.eka.operations import MeasureBlockSyndromes as TheirMBS, ResetAllDataQubits as TheirReset
    from loom.eka.utilities import BoolOp as TheirBoolOp
    from loom.interpreter import interpret_eka as their_interpret
    from loom_repetition_code.code_factory import RepetitionCode as TheirRep
    from loom_rotated_surface_code.code_factory import RotatedSurfaceCode as TheirRSC

    ns = SimpleNamespace(Channel=their_eka.Channel, ChannelType=their_eka.ChannelType, Circuit=their_eka.Circuit, Eka=their_eka.Eka, IfElseCircuit=their_eka.IfElseCircuit, Lattice=their_eka.Lattice, MeasureBlockSyndromes=TheirMBS, ResetAllDataQubits=TheirReset, BoolOp=TheirBoolOp, interpret_eka=their_interpret, RepetitionCode=TheirRep, RotatedSurfaceCode=TheirRSC)
    theirs, ours = build_ifelse(ns, name), ifelse_circuit(name)
    their_converters = {"qasm": their_executor.EkaToQasmConverter(), "cudaq": their_executor.EkaToCudaqConverter(), "pennylane": their_executor.EkaToPennylaneConverter(), "catalyst": their_executor.EkaToPennylaneConverter(is_catalyst=True, import_prefix="pl"), "mimiq": their_executor.EkaToMimiqConverter(), "guppy": their_executor.EkaToGuppylangConverter()}
    for target, converter in text_converters().items():
        try:
            expected = unid(their_converters[target].convert_circuit(theirs)[0], theirs)
        except ValueError as error:
            with pytest.raises(ValueError, match=re.escape(str(error))):
                converter.convert_circuit(ours)
            continue
        assert unid(converter.convert_circuit(ours)[0], ours) == expected, target
    try:
        expected = their_executor.EkaToStimConverter().emit_circuit_program(theirs)[0]
    except ValueError as error:
        with pytest.raises(ValueError, match=re.escape(str(error))):
            loom.EkaToStimConverter().emit_circuit_program(ours)
        return
    assert loom.EkaToStimConverter().emit_circuit_program(ours)[0] == list(expected)
