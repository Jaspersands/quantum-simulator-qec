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
import functools
import os
import re
import sys
from pathlib import Path
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
    "sinter": "sinter's commands: collect, combine, plot and predict (`stabilizer-qec sinter collect ...`).",
    "predict": "Predict observable flips from detection events by matching (PyMatching's `predict`).",
    "count_mistakes": "Count the shots matching mispredicts (PyMatching's `count_mistakes`).",
    "animate": "Draw matching's run on a shot as SVG frames (PyMatching's `animate`).",
    "repl": "Read operations from stdin and run them, printing measurement results as they happen.",
    "sample": "Samples measurements from a circuit.",
    "sample_dem": "Samples detection events from a detector error model.",
    "serve": "Serve a decoder to clients over TCP or over stdin/stdout (stabilizer_qec.realtime.DecoderClient).",
    "mqt.qecc.cc-decoder": "MQT QECC's colour-code decoder runs (`stabilizer-qec mqt.qecc.cc-decoder DISTANCE ERROR_RATE ...`).",
    "panqec": "PanQEC's command line: ls, run, start-gui, generate-input, ... (`stabilizer-qec panqec ls`).",
    "qecsim": "qecsim's command line: run, run-ftp and merge (`stabilizer-qec qecsim run ...`).",
}

# The ports whose own command lines (click groups) run as `stabilizer-qec <name> ...`, with the
# same arguments as `panqec ...` and `qecsim ...` (no `panqec` or `qecsim` scripts here: they
# would shadow the real packages').
PORT_CLIS = {"panqec": "stabilizer_qec.panqec.cli", "qecsim": "stabilizer_qec.qecsim.cli"}


def _port_cli(command: str):
    import importlib

    return importlib.import_module(PORT_CLIS[command]).cli


def _port_cli_help(command: str) -> str:
    """The port's own help page (its command line needs click, an extra's dependency)."""
    try:
        import click

        cli = _port_cli(command)
    except ImportError:
        return f"{COMMANDS[command]}\n(It needs `pip install stabilizer-qec[{command}]`.)\n"
    with click.Context(cli, info_name=f"stabilizer-qec {command}") as ctx:
        return cli.get_help(ctx) + "\n"


def _run_port_cli(command: str, rest: List[str]) -> int:
    """Runs PanQEC's or qecsim's command line on `rest`, as `stabilizer-qec <command>`."""
    try:
        cli = _port_cli(command)
    except ImportError as ex:
        print(f"stabilizer-qec {command}: {ex} (`pip install stabilizer-qec[{command}]`)", file=sys.stderr)
        return 1
    try:
        cli.main(args=rest, prog_name=f"stabilizer-qec {command}")
    except SystemExit as ex:  # click's standalone mode ends this way
        return ex.code if isinstance(ex.code, int) else int(ex.code is not None)
    return 0

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
    if command == "serve":
        from .realtime import DECODERS

        add("--dem", help="the detector error model file the decoder is built on")
        add("--circuit", help="a circuit file; its error model is built here (decomposed for the graph decoders)")
        add("--decoder", choices=list(DECODERS), default="matching")
        add("--tcp", help="host:port to listen on (port 0 picks one and prints it to stderr)")
        add("--stdio", action="store_true", help="serve one client over stdin and stdout")
        add("--workers", type=int, default=1, help="batches decoded at once")
        add("--threads", type=int, default=0, help="threads per batch, for decoders that take them (0: every core)")
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
        add("--decoder", default="matching", choices=["matching", "correlated_matching", "belief_matching", "union_find", "bposd", "bplsd", "relay_bp", "color_matching", "search", "ambiguity_clustering", "collision_clustering", "local_clustering"])
    if command in ("predict", "count_mistakes"):
        add("--dem", required=True)
        add("--in", dest="input")
        # PyMatching's defaults: predict reads b8, count_mistakes 01.
        add("--in_format", choices=FORMATS, default="b8" if command == "predict" else "01")
        add("--in_includes_appended_observables", action="store_true")
        add("--enable_correlations", action="store_true")
    if command == "predict":
        add("--out_format", choices=FORMATS, default="01")
    if command == "count_mistakes":
        add("--obs_in")
        add("--obs_in_format", choices=FORMATS, default="01")
        add("--time", action="store_true")
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

    filters = None
    if a.filter_coords is not None:
        # Stim's form: filters separated by ':', each a detector or observable (D5, L0) or a
        # coordinate prefix of numbers separated by ','.
        filters = []
        for part in a.filter_coords.split(":"):
            part = part.strip()
            if part[:1] in ("D", "L") and part[1:].isdigit():
                filters.append(part)
                continue
            try:
                filters.append([float(c) for c in part.split(",")])
            except ValueError:
                raise UsageError(f"--filter_coords: {part!r} is not a detector (D5), an observable (L0) or coordinates (2,3)") from None
    kind ={"detector-slice-svg": "detslice-svg", "detector-slice-text": "detslice-text", "match-graph-svg": "matchgraph-svg"}.get(a.type, a.type)
    text = _read_text(a.input)
    if kind == "matchgraph-svg":
        try:
            d = DetectorErrorModel(text).diagram("matchgraph-svg")
        except ValueError:
            d = _circuit(text).diagram("matchgraph-svg")
    else:
        c = _circuit(text)
        if a.remove_noise:
            c = c.without_noise()
        tick = None
        # Stim's command line ignores a tick for the text timeline.
        if a.tick is not None and kind != "timeline-text":
            # A tick (`--tick 3`) or, as Stim writes a range, `--tick A:B` for ticks A to B - 1;
            # a text slice is drawn at one tick, a range's first, as Stim's command line draws it.
            start, sep, stop = a.tick.partition(":")
            tick = range(int(start), int(stop)) if sep and kind != "detslice-text" else int(start)
        d = c.diagram(kind, tick=tick, filter_coords=filters)
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
        if len(replay) < a.shots:
            raise ValueError("Expected more error data for the requested number of shots.")
        replay = replay[: a.shots]
    dets, obs, errs = s.sample(a.shots, return_errors=a.err_out is not None, recorded_errors_to_replay=replay)
    _write(a.out, encode_shots(dets, a.out_format, num_detectors=s.num_detectors))
    if a.obs_out:
        _write(a.obs_out, encode_shots(obs, a.obs_out_format, num_observables=s.num_observables))
    if a.err_out:
        _write(a.err_out, encode_shots(errs, a.err_out_format, num_measurements=s.num_errors))


def cmd_serve(a) -> None:
    from ._circuit import DetectorErrorModel
    from .realtime import make_decoder, serve

    if (a.dem is None) == (a.circuit is None):
        raise UsageError("give --dem or --circuit, one of the two")
    if bool(a.tcp) == bool(a.stdio):
        raise UsageError("give --tcp host:port or --stdio, one of the two")
    if a.tcp:
        host, _, port = a.tcp.rpartition(":")
        if not host or not port.isdigit() or not 0 <= int(port) <= 65535:
            raise UsageError(f"--tcp takes host:port with a port from 0 to 65535, not {a.tcp!r}")
    if a.threads < 0:
        raise UsageError(f"--threads is 0 (every core) or positive, not {a.threads}")
    if a.dem is not None:
        decoder = make_decoder(a.decoder, DetectorErrorModel(_read_text(a.dem)))
    else:
        decoder = make_decoder(a.decoder, circuit=_read_text(a.circuit))

    def ready(server) -> None:
        host, port = server.server_address[:2]
        print(f"stabilizer-qec serve: {a.decoder} on {host}:{port}", file=sys.stderr, flush=True)

    serve(decoder, tcp=a.tcp, stdio=a.stdio, workers=a.workers, threads=a.threads, ready=ready)


def cmd_decode(a) -> None:
    from ._circuit import DetectorErrorModel
    from ._shots import decode_shots, encode_shots
    from ._decoders import AmbiguityClustering, BeliefMatching, CollisionClustering, LocalClustering, BpLsd, BpOsd, ColorMatching, Matching, RelayBp, SearchDecoder, UnionFind
    from ._matching import _CorrelatedMatching

    dem = DetectorErrorModel(_read_text(a.dem))
    nd, no = dem.num_detectors, dem.num_observables
    data = decode_shots(_read_bytes(a.input), a.in_format, num_detectors=nd, num_observables=no if a.in_includes_appended_observables else 0)
    decoder = {
        "matching": lambda: Matching(dem),
        "correlated_matching": lambda: _CorrelatedMatching(dem),
        "belief_matching": lambda: BeliefMatching(dem),
        "union_find": lambda: UnionFind(dem),
        "bposd": lambda: BpOsd(dem),
        "bplsd": lambda: BpLsd(dem),
        "relay_bp": lambda: RelayBp(dem),
        "color_matching": lambda: ColorMatching(dem),
        "search": lambda: SearchDecoder(dem),
        "ambiguity_clustering": lambda: AmbiguityClustering(dem),
        "collision_clustering": lambda: CollisionClustering(dem),
        "local_clustering": lambda: LocalClustering(dem),
    }[a.decoder]()
    predictions = decoder.decode_batch(data[:, :nd]).astype(bool)
    _write(a.out, encode_shots(predictions, a.out_format, num_observables=no))


def _matching_shots(a):
    """PyMatching's CLI inputs: the model's matching, the detection events, and the observables
    appended to them (or None)."""
    from ._circuit import DetectorErrorModel
    from ._matching import Matching
    from ._shots import decode_shots

    dem = DetectorErrorModel(_read_text(a.dem))
    nd, no = dem.num_detectors, dem.num_observables
    data = decode_shots(_read_bytes(a.input), a.in_format, num_detectors=nd, num_observables=no if a.in_includes_appended_observables else 0)
    m = Matching.from_detector_error_model(dem, enable_correlations=a.enable_correlations)
    return m, data[:, :nd], (data[:, nd:] if a.in_includes_appended_observables else None), no


def cmd_predict(a) -> None:
    from ._shots import encode_shots

    m, dets, _, no = _matching_shots(a)
    predictions = m.decode_batch(dets, enable_correlations=a.enable_correlations).astype(bool)
    _write(a.out, encode_shots(predictions, a.out_format, num_observables=no))


def cmd_count_mistakes(a) -> None:
    import time

    from ._shots import decode_shots

    m, dets, obs, no = _matching_shots(a)
    if obs is None:
        if a.obs_in is None:
            raise UsageError("Observables must be appended (--in_includes_appended_observables) or given with --obs_in")
        obs = decode_shots(_read_bytes(a.obs_in), a.obs_in_format, num_observables=no)
    m.decode_batch(dets[:0])  # the decoder is built before the clock starts, as PyMatching's is
    start = time.perf_counter()
    predictions = m.decode_batch(dets, enable_correlations=a.enable_correlations).astype(bool)
    elapsed = time.perf_counter() - start
    mistakes = int((predictions != obs.astype(bool)).any(axis=1).sum())
    _write(a.out, f"{mistakes} / {len(dets)}\n".encode())  # bare \n on every platform, as PyMatching's
    if a.time:
        micros = elapsed * 1e6
        per = micros / max(1, len(dets))
        _write(None, f"Total decoding time: {micros:.0f}us\nDecoding time per shot: {per:.3g}us\n")


@functools.lru_cache(maxsize=None)
def _stim_help() -> dict:
    """Stim's help pages (Stim 1.16, Apache-2.0; tools/stim_help.py)."""
    import gzip
    import json

    return json.loads(gzip.decompress((Path(__file__).parent / "_stim_help.json.gz").read_bytes()))


def _as_ours(page: str) -> str:
    """A Stim help page with its `stim <command>` invocations as this command line's."""
    names = "|".join(sorted(_stim_help()["commands"], key=len, reverse=True))
    return re.sub(rf"\bstim (?=(?:{names})\b)", "stabilizer-qec ", page)


def _commands_markdown(program: str, pages: dict) -> str:
    """The command line reference in Stim's layout: an index, then each command's page."""
    out = [f"# {'Stim' if program == 'stim' else program} command line reference\n\n## Index\n\n"]
    out += [f"- [{program} {c}](#{c})\n" for c in pages]
    out.append("## Commands\n\n")
    out += [f'<a name="{c}"></a>\n### {program} {c}\n\n```\n{page}```\n\n' for c, page in pages.items()]
    return "".join(out)


_ANIMATE_HELP = """NAME
    stabilizer-qec animate

SYNOPSIS
    stabilizer-qec animate \\
        --dem_in filepath \\
        --dets_in filepath \\
        --dets_in_format 01|b8|r8|ptb64|hits|dets \\
        [--dets_in_includes_appended_observables] \\
        --held_frames_at_end int \\
        --held_frames_at_start int \\
        --held_frames_per_event int \\
        --max_growth_between_frames int \\
        --out_dir directory \\
        --pixels_per_unit_length int \\
        [--quiet]

DESCRIPTION
    PyMatching's `animate`: matching (PyMatching 2.4's sparse blossom) at work on the first shot
    of --dets_in, drawn as SVG frames frame_00000.svg, frame_00001.svg, ... in --out_dir (made if
    it does not exist; one level only), with PyMatching's flags, messages and frames. Every flag
    but the two switches is required. Regions are red while growing, green once matched (dark
    green when matched to the boundary) and blue while shrinking; detection events are red dots,
    collisions about to happen orange, alternating trees black, blossom cycles blue, matches thick
    black, and the event about to be handled is highlighted. A run with no perfect matching stops
    with PyMatching's `Unknown tentative event type.` (exit 1) after the frames before it. An
    --out_dir that cannot be made, or a shot index too big for 64 bits, is reported as
    `stabilizer-qec animate: <message>` (exit 1), where PyMatching's program aborts.
    Frame write/close failures stop with a diagnostic (exit 1), instead of native ofstream's
    silent truncated success. Frame equality is limited to supported finite drawing coordinates;
    float32 overflow can produce NaN polygons whose native ordered-set behavior is not total.
    Also `python -m stabilizer_qec.pymatching animate ...`.

OPTIONS
    --dem_in
        The detector error model whose graph is matched (with 1000 weight levels, as PyMatching's
        animation builds it); all detector coordinates place them on the page: the first two as
        x/y, further coordinates projected onto them; an unplaced detector uses its index.

    --dets_in
        The detection events; only the first shot is drawn, and the file is not read to its end
        (a pipe's writer need not finish).

    --dets_in_format
        The format of --dets_in: 01, b8, r8, ptb64, hits or dets.

    --dets_in_includes_appended_observables
        The shots in --dets_in carry the observables after the detectors.

    --held_frames_at_end
        Frames of the final matching (0 to 1000000).

    --held_frames_at_start
        Frames of the detection events before anything grows (0 to 1000000).

    --held_frames_per_event
        Frames of each event the matcher handles (0 to 1000000).

    --max_growth_between_frames
        The most the regions grow between in-between frames (0 to 1000000; 0 draws none but one
        after each event's frames).

    --out_dir
        The directory the frames are written to.

    --pixels_per_unit_length
        The page's scale: pixels per unit of detector coordinate (4 to 2048).

    --quiet
        Do not print `wrote <frame>` to stderr for each frame.

EXAMPLES
    Example #1
        >>> stabilizer-qec gen --code surface_code --task rotated_memory_x --distance 5 --rounds 5 --after_clifford_depolarization 0.02 > c.stim
        >>> stabilizer-qec analyze_errors --in c.stim --decompose_errors > c.dem
        >>> stabilizer-qec detect --in c.stim --shots 1 --out d.01
        >>> stabilizer-qec animate --dem_in c.dem --dets_in d.01 --dets_in_format 01 --out_dir frames --held_frames_per_event 2 --held_frames_at_start 3 --held_frames_at_end 10 --max_growth_between_frames 50 --pixels_per_unit_length 20
"""


def _command_help(command: str) -> str:
    """A command's help: Stim's page for Stim's commands, the flags for this package's own."""
    pages = _stim_help()["commands"]
    if command in PORT_CLIS:
        return _port_cli_help(command)
    if command == "animate":
        return _ANIMATE_HELP
    return _as_ours(pages[command]) if command in pages else _parser(command).format_help()


def _commands_index() -> str:
    lines = ["Available stabilizer-qec commands:", ""]
    lines += [f"    stabilizer-qec {name:<16} # {doc}" for name, doc in COMMANDS.items()]
    return "\n".join(lines) + "\n"


def help_for(topic: str) -> str:
    """The help on a topic: '' (an overview), commands, gates, formats, a command, a gate or a
    format; the *_markdown topics as Markdown. Stim's pages for its gates, formats and commands
    (`stim <command>` reading `stabilizer-qec <command>`). Empty for an unknown topic."""
    from . import _stim

    key = topic.strip()
    lower = key.lower()
    data = _stim_help()
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
    if lower == "commands":
        return _commands_index()
    if lower == "commands_markdown":
        return _commands_markdown("stabilizer-qec", {c: _command_help(c) for c in COMMANDS})
    if lower in data["indexes"]:
        return data["indexes"][lower]
    if lower in COMMANDS:
        return _command_help(lower)
    if lower in data["formats"]:
        return data["formats"][lower]
    try:
        name = _stim.GateData(key).name
    except (IndexError, KeyError, ValueError):
        return ""
    return data["gates"].get(name, "")


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
    out = sys.stdout.buffer  # bare \n on every platform, as Stim writes it
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
                out.write(("".join("1" if b else "0" for b in record[before:]) + "\n").encode())
                out.flush()
    out.write(b"\n")
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
    if command == "sinter":
        from .sinter._command._main import main as sinter_main

        return sinter_main(command_line_args=rest) or 0
    if command == "mqt.qecc.cc-decoder":  # MQT QECC's console script, as a command (its flags, its help)
        from .mqt_qecc.cc_decoder.cli import cli as cc_decoder_cli

        try:
            cc_decoder_cli(rest, prog=f"stabilizer-qec {command}")
        except SystemExit as ex:  # argparse's --help and usage errors
            return ex.code if isinstance(ex.code, int) else int(ex.code is not None)
        return 0
    if command in PORT_CLIS:
        return _run_port_cli(command, rest)
    if command not in COMMANDS:
        print(f"Unrecognized command '{command}'. Use `stabilizer-qec help` for the commands.", file=sys.stderr)
        return 1
    if command != "help" and rest and rest[0] in ("help", "--help") and len(rest) == 1:
        command, rest = "help", [command]
    if command == "animate":  # PyMatching's flags, messages and exit codes
        from .pymatching import _animate

        try:
            return _animate([command] + rest)
        except (RuntimeError, OSError) as ex:  # what PyMatching's program aborts on (and its cli raises), said as ours are
            print(f"stabilizer-qec {command}: {ex}", file=sys.stderr)
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
    except SystemExit as ex:  # argparse's own --help
        return int(ex.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
