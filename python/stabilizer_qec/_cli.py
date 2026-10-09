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
    p = _Parser(prog=f"stabilizer-qec {command}", description=COMMANDS[command])
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
    if command == "sample":
        add("--skip_reference_sample", action="store_true")
        add("--skip_loop_folding", action="store_true", help="accepted for Stim's sake; loops are never unrolled here")
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
    data = c.compile_sampler(skip_reference_sample=a.skip_reference_sample, seed=a.seed).sample(a.shots)
    _write(a.out, encode_shots(data, a.out_format, num_measurements=c.num_measurements))


def _detections_out(a, dets: np.ndarray, obs: np.ndarray, nd: int, no: int) -> None:
    from ._shots import encode_shots

    if a.append_observables:
        _write(a.out, encode_shots(np.hstack([dets, obs]), a.out_format, num_detectors=nd, num_observables=no))
    else:
        _write(a.out, encode_shots(dets, a.out_format, num_detectors=nd))
    if a.obs_out:
        _write(a.obs_out, encode_shots(obs, a.obs_out_format, num_observables=no))


def cmd_detect(a) -> None:
    c = _circuit(_read_text(a.input))
    dets, obs = c.compile_detector_sampler(seed=a.seed).sample(a.shots, separate_observables=True)
    _detections_out(a, dets, obs, c.num_detectors, c.num_observables)


def cmd_m2d(a) -> None:
    from ._shots import decode_shots

    if a.ran_without_feedback:
        raise UsageError("--ran_without_feedback is not supported")
    c = _circuit(_read_text(a.circuit))
    meas = decode_shots(_read_bytes(a.input), a.in_format, num_measurements=c.num_measurements)
    if a.skip_reference_sample:
        meas = meas ^ c.reference_sample()
    sweep = None
    if a.sweep:
        sweep = decode_shots(_read_bytes(a.sweep), a.sweep_format, num_measurements=c.num_sweep_bits)
    dets, obs = c.compile_m2d_converter().convert(measurements=meas, sweep_bits=sweep, separate_observables=True)
    _detections_out(a, dets, obs, c.num_detectors, c.num_observables)


def cmd_analyze_errors(a) -> None:
    if a.allow_gauge_detectors:
        raise UsageError("--allow_gauge_detectors is not supported: a detector must be deterministic")
    if a.block_decompose_from_introducing_remnant_edges:
        raise UsageError("--block_decompose_from_introducing_remnant_edges is not supported")
    c = _circuit(_read_text(a.input))
    dem = c.detector_error_model(
        decompose_errors=a.decompose_errors,
        approximate_disjoint_errors=False if a.approximate_disjoint_errors is None else a.approximate_disjoint_errors,
        flatten_loops=not a.fold_loops,
        ignore_decomposition_failures=a.ignore_decomposition_failures,
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


def cmd_help(a) -> None:
    if a.topic in COMMANDS:
        _write(None, _parser(a.topic).format_help())
        return
    lines = ["Available stabilizer-qec commands:", ""]
    lines += [f"    stabilizer-qec {name:<16} # {doc}" for name, doc in COMMANDS.items()]
    lines += ["", "Use `stabilizer-qec help [command]` for help on a command.", ""]
    _write(None, "\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        argv = ["help"]
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"Unrecognized command '{command}'. Use `stabilizer-qec help` for the commands.", file=sys.stderr)
        return 1
    try:
        args = _parser(command).parse_args(rest)
        globals()[f"cmd_{command}"](args)
    except UsageError as ex:
        print(f"stabilizer-qec {command}: {ex}", file=sys.stderr)
        return 1
    except (ValueError, TypeError, OSError) as ex:
        print(f"stabilizer-qec {command}: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
