"""The ``stabilizer-qec`` command line, after Stim's: the same commands, the same flags, the same
file formats, and (where the output is deterministic) the same bytes.

    stabilizer-qec gen --code surface_code --task rotated_memory_z --distance 5 --rounds 10 \\
        --after_clifford_depolarization 0.001 > c.stim
    stabilizer-qec analyze_errors --in c.stim --decompose_errors > c.dem
    stabilizer-qec detect --in c.stim --shots 1000 --out_format b8 --obs_out obs.01 > d.b8
    stabilizer-qec decode --dem c.dem --in d.b8 --in_format b8 > predictions.01

Also ``python -m stabilizer_qec``. ``stabilizer-qec help <command>`` describes each command.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

COMMANDS = {
    "analyze_errors": "Converts a circuit into a detector error model.",
    "convert": "Convert data between result formats.",
    "decode": "Decode detection events with a detector error model (after PyMatching's predict).",
    "detect": "Sample detection events and observable flips from a circuit.",
    "diagram": "Produces various kinds of diagrams.",
    "explain_errors": "Find circuit errors that produce certain detection events.",
    "gen": "Generates example circuits.",
    "help": "Prints helpful information about using stabilizer-qec.",
    "m2d": "Convert measurement data into detection event data.",
    "repl": "Read operations from stdin and run them, printing measurement results as they happen.",
    "sample": "Samples measurements from a circuit.",
    "sample_dem": "Samples detection events from a detector error model.",
}

FORMATS = ["01", "b8", "r8", "ptb64", "hits", "dets"]


class UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # an exit code of 1, as Stim's
        raise UsageError(message)


def _read_text(path: Optional[str]) -> str:
    if path is None:
        return sys.stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def _read_bytes(path: Optional[str]) -> bytes:
    if path is None:
        return sys.stdin.buffer.read()
    with open(path, "rb") as f:
        return f.read()


def _write(path: Optional[str], data) -> None:
    # Text (circuits, models, explanations, diagrams) ends its lines as the platform does, as
    # Stim's text-mode output does (\r\n on Windows); shot data is written as it is.
    raw = data.replace("\n", os.linesep).encode() if isinstance(data, str) else data
    if path is None:
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
    else:
        with open(path, "wb") as f:
            f.write(raw)


def _parser(command: str) -> _Parser:
    p = _Parser(prog=f"stabilizer-qec {command}", description=COMMANDS[command], add_help=command != "help")
    add = p.add_argument
    if command in ("sample", "detect", "analyze_errors", "explain_errors", "diagram", "sample_dem"):
        add("--in", dest="input", help="the input file (default: stdin)")
    if command != "help":
        add("--out", help="the output file (default: stdout)")
    if command in ("sample", "detect", "sample_dem"):
        add("--shots", type=int, default=1)
        add("--seed", type=int, default=None)
    if command in ("sample", "detect", "m2d", "convert", "sample_dem", "decode"):
        add("--out_format", choices=FORMATS, default="01")
    if command in ("detect", "m2d", "convert", "sample_dem"):
        add("--obs_out")
        add("--obs_out_format", choices=FORMATS, default="01")
    if command in ("detect", "m2d"):
        add("--append_observables", action="store_true")
    if command == "detect":
        add("--prepend_observables", action="store_true", help="deprecated (Stim's); observables before the detectors")
    if command == "sample":
        add("--skip_reference_sample", action="store_true")
        add("--skip_loop_folding", action="store_true", help="accepted for Stim's sake; loops are never unrolled here")
        add("--frame0", action="store_true", help="deprecated (Stim's): --skip_reference_sample")
    if command == "m2d":
        add("--circuit", required=True)
        add("--in", dest="input")
        add("--in_format", choices=FORMATS, default="01")
        add("--sweep")
        add("--sweep_format", choices=FORMATS, default="01")
        add("--skip_reference_sample", action="store_true")
        add("--ran_without_feedback", action="store_true")
    if command == "analyze_errors":
        add("--decompose_errors", action="store_true")
        add("--approximate_disjoint_errors", nargs="?", const=1.0, default=None, type=float)
        add("--fold_loops", action="store_true")
        add("--ignore_decomposition_failures", action="store_true")
        add("--allow_gauge_detectors", action="store_true")
        add("--block_decompose_from_introducing_remnant_edges", action="store_true")
        add("--detector_hypergraph", action="store_true", help="deprecated (Stim's); no effect")
    if command == "explain_errors":
        add("--dem_filter")
        add("--single", action="store_true")
    if command == "diagram":
        add("--type", required=True)
        add("--tick", default=None)
        add("--filter_coords", default=None)
        add("--remove_noise", action="store_true")
    if command == "gen":
        add("--code", required=True, choices=["color_code", "repetition_code", "surface_code"])
        add("--task", required=True)
        add("--distance", type=int, required=True)
        add("--rounds", type=int, required=True)
        for name in ("after_clifford_depolarization", "after_reset_flip_probability", "before_measure_flip_probability", "before_round_data_depolarization"):
            add(f"--{name}", type=float, default=0.0)
    if command == "convert":
        add("--in", dest="input")
        add("--in_format", choices=FORMATS, required=True)
        add("--num_measurements", type=int, default=0)
        add("--num_detectors", type=int, default=0)
        add("--num_observables", type=int, default=0)
        add("--bits_per_shot", type=int, default=None)
        add("--circuit")
        add("--dem")
        add("--types", default=None)
    if command == "sample_dem":
        add("--err_out")
        add("--err_out_format", choices=FORMATS, default="01")
        add("--replay_err_in")
        add("--replay_err_in_format", choices=FORMATS, default="01")
    if command == "decode":
        add("--dem", required=True)
        add("--in", dest="input")
        add("--in_format", choices=FORMATS, default="01")
        add("--in_includes_appended_observables", action="store_true")
        add("--decoder", default="matching", choices=["matching", "correlated_matching", "belief_matching", "union_find", "bposd", "bplsd", "relay_bp", "color_matching", "search"])
    if command == "help":
        add("topic", nargs="?")
        add("--help", dest="topic_flag", nargs="?", const="", default=None)
    return p


def _circuit(text: str):
    from ._circuit import Circuit

    return Circuit(text)


def cmd_gen(a) -> None:
    from . import _core
    from ._circuit import Circuit

    task = f"{a.code}:{a.task}"
    noise = dict(
        after_clifford_depolarization=a.after_clifford_depolarization,
        before_round_data_depolarization=a.before_round_data_depolarization,
        before_measure_flip_probability=a.before_measure_flip_probability,
        after_reset_flip_probability=a.after_reset_flip_probability,
    )
    circuit = Circuit.generated(task, distance=a.distance, rounds=a.rounds, **noise)
    header = _core.generated_header(task, a.distance, a.rounds, *noise.values())
    _write(a.out, header + str(circuit) + "\n")


def cmd_sample(a) -> None:
    from ._shots import encode_shots

    c = _circuit(_read_text(a.input))
    if a.frame0:
        print("[DEPRECATION] Use `--skip_reference_sample` instead of `--frame0`", file=sys.stderr)
    data = c.compile_sampler(skip_reference_sample=a.skip_reference_sample or a.frame0, seed=a.seed).sample(a.shots)
    _write(a.out, encode_shots(data, a.out_format, num_measurements=c.num_measurements))


def _detections_out(a, dets: np.ndarray, obs: np.ndarray, nd: int, no: int) -> None:
    from ._shots import encode_shots

    prepend = getattr(a, "prepend_observables", False)
    if a.append_observables or prepend:
        rows = np.hstack(([obs] if prepend else []) + [dets] + ([obs] if a.append_observables else []))
        names = [f"L{k}" for k in range(no)] * prepend + [f"D{k}" for k in range(nd)] + [f"L{k}" for k in range(no)] * a.append_observables
        _write(a.out, encode_shots(rows, a.out_format, num_detectors=rows.shape[1], names=names))
    else:
        _write(a.out, encode_shots(dets, a.out_format, num_detectors=nd))
    if a.obs_out:
        _write(a.obs_out, encode_shots(obs, a.obs_out_format, num_observables=no))


def cmd_detect(a) -> None:
    if a.prepend_observables:
        print("[DEPRECATION] Avoid using `--prepend_observables`. Data readers assume observables are appended, not prepended.", file=sys.stderr)
    if a.out_format == "dets" and not a.append_observables:
        a.prepend_observables = True  # as Stim: dets output names the observables, first
    if a.prepend_observables + a.append_observables + (a.obs_out is not None) > 1:
        raise UsageError("Can't combine --prepend_observables, --append_observables, or --obs_out")
    c = _circuit(_read_text(a.input))
    dets, obs = c.compile_detector_sampler(seed=a.seed).sample(a.shots, separate_observables=True)
    _detections_out(a, dets, obs, c.num_detectors, c.num_observables)


def cmd_m2d(a) -> None:
    from ._shots import decode_shots

    c = _circuit(_read_text(a.circuit))
    if a.ran_without_feedback:
        # As Stim: the results are of the circuit with its feedback left out.
        c = c.with_inlined_feedback()
    meas = decode_shots(_read_bytes(a.input), a.in_format, num_measurements=c.num_measurements)
    if a.skip_reference_sample:
        meas = meas ^ c.reference_sample()
    sweep = None
    if a.sweep:
        sweep = decode_shots(_read_bytes(a.sweep), a.sweep_format, num_measurements=c.num_sweep_bits)
    dets, obs = c.compile_m2d_converter().convert(measurements=meas, sweep_bits=sweep, separate_observables=True)
    _detections_out(a, dets, obs, c.num_detectors, c.num_observables)


def cmd_analyze_errors(a) -> None:
    if a.detector_hypergraph:
        print("[DEPRECATION] Use `stabilizer-qec analyze_errors` instead of `--detector_hypergraph`", file=sys.stderr)
    c = _circuit(_read_text(a.input))
    dem = c.detector_error_model(
        decompose_errors=a.decompose_errors,
        approximate_disjoint_errors=False if a.approximate_disjoint_errors is None else a.approximate_disjoint_errors,
        flatten_loops=not a.fold_loops,
        ignore_decomposition_failures=a.ignore_decomposition_failures,
        allow_gauge_detectors=a.allow_gauge_detectors,
        block_decomposition_from_introducing_remnant_edges=a.block_decompose_from_introducing_remnant_edges,
    )
    _write(a.out, str(dem) + "\n")


def cmd_explain_errors(a) -> None:
    from ._circuit import DetectorErrorModel

    c = _circuit(_read_text(a.input))
    f = None if a.dem_filter is None else DetectorErrorModel(_read_text(a.dem_filter))
    out = c.explain_detector_error_model_errors(dem_filter=f, reduce_to_one_representative_error=a.single)
    _write(a.out, "".join(str(e).rstrip("\n") + "\n" for e in out))


def cmd_diagram(a) -> None:
    from ._circuit import DetectorErrorModel

    if a.filter_coords is not None:
        raise UsageError("--filter_coords is not supported")
    kind = {"detector-slice-svg": "detslice-svg", "detector-slice-text": "detslice-text", "match-graph-svg": "matchgraph-svg"}.get(a.type, a.type)
    text = _read_text(a.input)
    if kind == "matchgraph-svg":
        try:
            d = DetectorErrorModel(text).diagram("matchgraph-svg")
        except ValueError:
            d = _circuit(text).diagram("matchgraph-svg")
    else:
        c = _circuit(text)
        tick = None if a.tick is None else int(a.tick)
        d = c.diagram(kind, tick=tick)
    out = str(d)
    _write(a.out, out if out.endswith("\n") else out + "\n")


def _counts(a):
    from ._circuit import DetectorErrorModel

    m, d, o = a.num_measurements, a.num_detectors, a.num_observables
    if a.circuit or a.dem:
        if a.circuit:
            c = _circuit(_read_text(a.circuit))
            have = dict(M=c.num_measurements, D=c.num_detectors, L=c.num_observables)
        else:
            dem = DetectorErrorModel(_read_text(a.dem))
            have = dict(M=0, D=dem.num_detectors, L=dem.num_observables)
        types = a.types if a.types is not None else ("M" if a.circuit else "DL")
        if set(types) - set("MDL"):
            raise UsageError("--types must be letters of M, D and L")
        m, d, o = (have["M"] if "M" in types else 0), (have["D"] if "D" in types else 0), (have["L"] if "L" in types else 0)
    elif a.bits_per_shot is not None:
        m = a.bits_per_shot
    return m, d, o


def cmd_convert(a) -> None:
    from ._shots import decode_shots, encode_shots

    m, d, o = _counts(a)
    data = decode_shots(_read_bytes(a.input), a.in_format, m, d, o)
    if a.obs_out and o:
        _write(a.out, encode_shots(data[:, : m + d], a.out_format, m, d, 0))
        _write(a.obs_out, encode_shots(data[:, m + d :], a.obs_out_format, num_observables=o))
    else:
        _write(a.out, encode_shots(data, a.out_format, m, d, o))


def cmd_sample_dem(a) -> None:
    from ._circuit import DetectorErrorModel
    from ._shots import decode_shots, encode_shots

    dem = DetectorErrorModel(_read_text(a.input))
    s = dem.compile_sampler(seed=a.seed)
    replay = None
    if a.replay_err_in:
        replay = decode_shots(_read_bytes(a.replay_err_in), a.replay_err_in_format, num_measurements=s.num_errors)
    dets, obs, errs = s.sample(a.shots, return_errors=a.err_out is not None, recorded_errors_to_replay=replay)
    _write(a.out, encode_shots(dets, a.out_format, num_detectors=s.num_detectors))
    if a.obs_out:
        _write(a.obs_out, encode_shots(obs, a.obs_out_format, num_observables=s.num_observables))
    if a.err_out:
        _write(a.err_out, encode_shots(errs, a.err_out_format, num_measurements=s.num_errors))


def cmd_decode(a) -> None:
    from ._circuit import DetectorErrorModel
    from ._shots import decode_shots, encode_shots
    from ._decoders import BeliefMatching, BpLsd, BpOsd, ColorMatching, Matching, RelayBp, SearchDecoder, UnionFind

    dem = DetectorErrorModel(_read_text(a.dem))
    nd, no = dem.num_detectors, dem.num_observables
    data = decode_shots(_read_bytes(a.input), a.in_format, num_detectors=nd, num_observables=no if a.in_includes_appended_observables else 0)
    decoder = {
        "matching": lambda: Matching(dem),
        "correlated_matching": lambda: Matching(dem, enable_correlations=True),
        "belief_matching": lambda: BeliefMatching(dem),
        "union_find": lambda: UnionFind(dem),
        "bposd": lambda: BpOsd(dem),
        "bplsd": lambda: BpLsd(dem),
        "relay_bp": lambda: RelayBp(dem),
        "color_matching": lambda: ColorMatching(dem),
        "search": lambda: SearchDecoder(dem),
    }[a.decoder]()
    predictions = decoder.decode_batch(data[:, :nd]).astype(bool)
    _write(a.out, encode_shots(predictions, a.out_format, num_observables=no))


FORMAT_HELP = {
    "01": "The default format. Each shot is a line of '0' and '1' characters, one per bit, ending with a newline.",
    "b8": "Binary. Each shot is a whole number of bytes, the bits packed eight to a byte, least significant first.",
    "r8": "Binary, for sparse data. Each shot is a run of bytes, each the number of 0s before the next 1 (a byte of 255 adds 255 and continues the run), ending with the run before an imagined 1 past the end.",
    "ptb64": "Binary, partially transposed. Shots in groups of 64: for each bit, a little-endian 64-bit word whose bit k is that bit of the group's shot k. The number of shots must be a multiple of 64.",
    "hits": "Text. Each shot is a line listing the indices of its 1 bits, comma-separated.",
    "dets": "Text. Each shot is a line starting 'shot', then the 1 bits named by kind and index: M for measurements, D for detectors, L for observables (e.g. 'shot D0 D5 L1').",
}


def _gate_help(name: str) -> str:
    """A gate's help, from the gate table (our own text)."""
    from . import _stim

    g = _stim.GateData(name)
    d = g._d
    out = [f"### The '{g.name}' Instruction", ""]
    others = [a for a in g.aliases if a != g.name]
    if others:
        out += ["Alternative names:", ""] + [f"    {a}" for a in others] + [""]
    out.append(f"Category: {d['category'][2:]}")
    lo, hi = d["args"]
    if hi <= 1:
        out.append("Parens arguments: none")
    elif lo + 1 == hi:
        out.append(f"Parens arguments: exactly {lo}")
    else:
        out.append(f"Parens arguments: {lo} or more" if hi > 64 else f"Parens arguments: {lo} to {hi - 1}")
    if d["takes_pauli_targets"]:
        targets = "Pauli products (e.g. X0*Y1)" if d["flags"] & (1 << 12) else "Pauli targets (e.g. X0)"
    elif d["is_two_qubit_gate"]:
        targets = "qubit pairs"
    elif d["flags"] & (1 << 10):
        targets = "none"
    elif d["flags"] & (1 << 8):
        targets = "measurement records (rec[-k])"
    else:
        targets = "qubits"
    if d["takes_measurement_record_targets"] and d["is_two_qubit_gate"]:
        targets += "; a measurement record or sweep bit may be a control"
    out.append(f"Targets: {targets}")
    kinds = [k for k, f in (("unitary", "is_unitary"), ("noisy", "is_noisy_gate"), ("a reset", "is_reset"), ("a measurement", "produces_measurements")) if d[f]]
    if kinds:
        out.append("It is " + ", ".join(kinds) + ".")
    if d["tableau"]:
        out += ["", "Stabilizer generators (how it conjugates each Pauli):", ""]
        n = len(d["tableau"]) // 2
        for k in range(n):
            out.append(f"    X{k} -> {d['tableau'][2 * k]}")
            out.append(f"    Z{k} -> {d['tableau'][2 * k + 1]}")
    if d["flows"]:
        out += ["", "Flows:", ""] + [f"    {f}" for f in d["flows"]]
    if d["unitary"]:
        dim = int(round(len(d["unitary"]) ** 0.5))
        out += ["", "Unitary matrix (little-endian):", ""]

        def c(re_im: tuple) -> str:
            re, im = re_im
            if abs(im) < 1e-9:
                return f"{re:+.3f}"
            if abs(re) < 1e-9:
                return f"{im:+.3f}i"
            return f"{re:+.3f}{im:+.3f}i"

        for r in range(dim):
            out.append("    [" + ", ".join(c(d["unitary"][r * dim + col]) for col in range(dim)) + "]")
    if d["decomposition"]:
        out += ["", "Decomposition (into H, S, CX, M, R):", ""] + [f"    {line}" for line in d["decomposition"].splitlines()]
    if d["inverse"]:
        out += ["", f"Inverse: {d['inverse']}"]
    return "\n".join(out) + "\n"


def _gates_index(markdown: bool) -> str:
    from . import _stim

    _stim._load_gates()
    categories: dict = {}
    for d in _stim._GATES.values():
        categories.setdefault(d["category"], set()).update(d["aliases"])
    if not markdown:
        lines = ["Gates supported by stabilizer-qec", "================================="]
        for cat in sorted(categories):
            lines.append(cat[2:] + ":")
            lines += [f"    {n}" for n in sorted(categories[cat])]
        return "\n".join(lines) + "\n"
    lines = ["# Gates supported by stabilizer-qec", ""]
    for cat in sorted(categories):
        lines.append("- " + cat[2:])
        lines += [f"    - [{n}](#{n})" for n in sorted(categories[cat])]
    lines.append("")
    for cat in sorted(categories):
        lines += [f"## {cat[2:]}", ""]
        for n in sorted(categories[cat]):
            if _stim.GateData(n).name == n:
                lines.append(_gate_help(n))
    return "\n".join(lines) + "\n"


def _formats_index(markdown: bool) -> str:
    if markdown:
        return "# Result formats\n\n" + "".join(f"## {k}\n\n{v}\n\n" for k, v in FORMAT_HELP.items())
    lines = ["Result formats supported by stabilizer-qec", "==========================================="]
    lines += [f"    {k:<6} {v.split('. ')[0]}." for k, v in FORMAT_HELP.items()]
    return "\n".join(lines) + "\n"


def _commands_index() -> str:
    lines = ["Available stabilizer-qec commands:", ""]
    lines += [f"    stabilizer-qec {name:<16} # {doc}" for name, doc in COMMANDS.items()]
    return "\n".join(lines) + "\n"


def help_for(topic: str) -> str:
    """The help on a topic: '' (an overview), commands, gates, formats, a command, a gate or a
    format; the *_markdown topics as Markdown. Empty for an unknown topic."""
    from . import _stim

    key = topic.strip()
    upper = key.upper()
    if key == "":
        return _commands_index() + """
Use `stabilizer-qec help [topic]` for help on specific topics. Available topics include:

    stabilizer-qec help commands  # List all commands.
    stabilizer-qec help gates     # List all circuit instructions.
    stabilizer-qec help formats   # List all result formats.
    stabilizer-qec help [command] # Print information about a command, e.g. "sample".
    stabilizer-qec help [gate]    # Print information about a gate, e.g. "CNOT".
    stabilizer-qec help [format]  # Print information about a result format, e.g. "01".
"""
    if upper == "COMMANDS":
        return _commands_index()
    if upper == "COMMANDS_MARKDOWN":
        return "# stabilizer-qec command line reference\n\n" + "".join(f"## {c}\n\n```\n{_parser(c).format_help()}```\n\n" for c in COMMANDS)
    if upper == "GATES":
        return _gates_index(False)
    if upper == "GATES_MARKDOWN":
        return _gates_index(True)
    if upper == "FORMATS":
        return _formats_index(False)
    if upper == "FORMATS_MARKDOWN":
        return _formats_index(True)
    if key.lower() in COMMANDS:
        return _parser(key.lower()).format_help()
    if key.lower() in FORMAT_HELP:
        return f"Result format '{key.lower()}'\n\n{FORMAT_HELP[key.lower()]}\n"
    try:
        return _gate_help(key)
    except (IndexError, KeyError, ValueError):
        return ""


def cmd_help(a) -> None:
    topic = a.topic if a.topic is not None else (a.topic_flag or "")
    msg = help_for(topic)
    if not msg:
        raise UsageError(f"Unrecognized help topic '{topic}'.")
    _write(None, msg)


def cmd_repl(a) -> None:
    """Stim's REPL: each instruction (a REPEAT block once it closes) runs as soon as it's read,
    on a tableau simulator; each instruction that measures prints its results as 0s and 1s
    and a newline. An instruction that fails is reported (in red) and skipped."""
    from ._circuit import Circuit
    from ._stim import TableauSimulator

    sim = TableauSimulator()
    out = sys.stdout
    pending: List[str] = []
    depth = 0
    for line in sys.stdin:
        body = line.split("#", 1)[0]
        depth += body.count("{") - body.count("}")
        pending.append(line)
        if depth > 0 or not body.strip():
            if depth <= 0:
                pending = []
            continue
        text, pending, depth = "".join(pending), [], 0
        try:
            ops = list(Circuit(text).flattened())
        except ValueError as ex:
            print(f"\033[31m{ex}\033[0m", file=sys.stderr)
            continue
        for op in ops:
            before = len(sim.current_measurement_record())
            try:
                sim.do(op)
            except ValueError as ex:
                print(f"\033[31m{ex}\033[0m", file=sys.stderr)
                break
            record = sim.current_measurement_record()
            if len(record) > before:
                out.write("".join("1" if b else "0" for b in record[before:]) + "\n")
                out.flush()
    out.write("\n")
    out.flush()


_LEGACY_MODES = {
    "--repl": "repl",
    "--sample": "sample",
    "--detect": "detect",
    "--analyze_errors": "analyze_errors",
    "--detector_hypergraph": "analyze_errors",
    "--gen": "gen",
    "--m2d": "m2d",
    "--explain_errors": "explain_errors",
    "--convert": "convert",
    "--help": "help",
}


def _legacy(argv: List[str]) -> Optional[List[str]]:
    """Stim's old invocations (`--sample=10 --in c.stim`, `--detect`, `--help`, ...) as a
    command and its arguments."""
    found = [(k, a) for k, a in enumerate(argv) if a.split("=", 1)[0] in _LEGACY_MODES]
    if not found:
        return None
    k, flag = found[0]
    name, _, value = flag.partition("=")
    command = _LEGACY_MODES[name]
    rest = argv[:k] + argv[k + 1 :]
    if name == "--detector_hypergraph":
        print("[DEPRECATION] Use `stabilizer-qec analyze_errors` instead of `--detector_hypergraph`", file=sys.stderr)
    if name in ("--sample", "--detect"):
        if not value and k + 1 < len(argv) and argv[k + 1].isdigit():
            value = argv[k + 1]
            rest = argv[:k] + argv[k + 2 :]
        rest = ["--shots", value or "1"] + rest
    if command == "help":
        rest = [value] if value else rest[:1]
    return [command] + rest


def main(argv: Optional[List[str]] = None, *, command_line_args: Optional[List[str]] = None) -> int:
    """Runs the command line (as ``stim.main``): ``command_line_args`` are the arguments after
    the program's name. Returns the exit code."""
    if command_line_args is not None:
        argv = list(command_line_args)
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        argv = ["help"]
    if argv[0].startswith("-"):
        legacy = _legacy(argv)
        if legacy is None:
            print("No mode was given.\n\n" + help_for(""), file=sys.stderr)
            return 1
        argv = legacy
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"Unrecognized command '{command}'. Use `stabilizer-qec help` for the commands.", file=sys.stderr)
        return 1
    if command != "help" and rest and rest[0] in ("help", "--help") and len(rest) == 1:
        command, rest = "help", [command]
    try:
        args = _parser(command).parse_args(rest)
        globals()[f"cmd_{command}"](args)
    except UsageError as ex:
        print(f"stabilizer-qec {command}: {ex}", file=sys.stderr)
        return 1
    except (ValueError, TypeError, OSError) as ex:
        print(f"stabilizer-qec {command}: {ex}", file=sys.stderr)
        return 1
    except SystemExit as ex:  # argparse's own --help
        return int(ex.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
