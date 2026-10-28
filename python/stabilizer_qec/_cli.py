"""The ``stabilizer-qec`` command line, after Stim's: the same commands, the same flags, the same
file formats, and (where the output is deterministic) the same bytes.

    stabilizer-qec gen --code surface_code --task rotated_memory_z --distance 5 --rounds 10 \\
        --after_clifford_depolarization 0.001 > c.stim
    stabilizer-qec analyze_errors --in c.stim --decompose_errors > c.dem
    stabilizer-qec detect --in c.stim --shots 1000 --out_format b8 --obs_out obs.01 > d.b8
    stabilizer-qec decode --dem c.dem --in d.b8 --in_format b8 > predictions.01

Also ``python -m stabilizer_qec``. ``stabilizer-qec help <command>`` describes each command.

As Stim's, the commands that sample, convert or decode shots stream them, 1,024 shots at a time
(read, sampled, decoded and written a batch at a time), so their memory does not grow with the
number of shots; the bytes are those of one pass over all of them.
"""

from __future__ import annotations

import argparse
import contextlib
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
    """A circuit or model file (stdin by default) as text. Stim reads the bytes, so a byte that
    is not UTF-8 (in a comment, say) is read as U+FFFD, as the Chromobius command line reads it,
    not refused; lines end as in text mode (\\r\\n and \\r read as \\n), whatever the locale."""
    if path is None:
        raw = sys.stdin.buffer.read()
    else:
        with open(path, "rb") as f:
            raw = f.read()
    return raw.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")


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


# Shot data is sampled, read, decoded and written this many shots at a time, so the memory a
# command needs does not grow with its shots, as Stim's command line streams them. A multiple of
# 64: whole ptb64 blocks, and the samplers draw 64 shots to a word, so the batches of a seeded
# run are the shots one call for all of them would draw.
_BATCH = 1024


class _Output:
    """A shot data output (a file, or stdout), written a batch at a time. A file is opened at
    its first write, or made empty at the end of a run that wrote nothing; a run that fails
    before writing leaves no file, as when the whole output was written at once."""

    def __init__(self, path: Optional[str]) -> None:
        self.path, self.f = path, None

    def _open(self):
        if self.f is None:
            self.f = sys.stdout.buffer if self.path is None else open(self.path, "wb")
        return self.f

    def write(self, raw: bytes) -> None:
        f = self._open()
        f.write(raw)
        if self.path is None:
            f.flush()

    def close(self, failed: bool = False) -> None:
        if failed:
            if self.f is not None and self.path is not None:
                try:
                    self.f.close()
                except OSError:
                    pass  # the failure that stopped the run is the one reported
            return
        f = self._open()
        if self.path is None:
            f.flush()
        else:
            f.close()


@contextlib.contextmanager
def _streamed(*outputs: Optional[_Output]):
    """Closes a streamed command's outputs (None for one not asked for) when it ends: made
    whole if it succeeds, and if it fails, the files it opened closed (the failure is the one
    reported)."""
    try:
        yield outputs
    except BaseException:
        for o in outputs:
            if o is not None:
                o.close(failed=True)
        raise
    for o in outputs:
        if o is not None:
            o.close()


def _optional_reader(path: Optional[str], format: str, **counts):
    return contextlib.nullcontext(None) if path is None else _ShotReader(path, format, **counts)


def _count_rest(reader: "_ShotReader") -> int:
    """The shots left in ``reader``, read a batch at a time to count them."""
    total = 0
    while True:
        k = len(reader.read(_BATCH))
        total += k
        if k < _BATCH:
            return total


class _ShotReader:
    """A shot data file (stdin by default) in one of Stim's formats, read a batch at a time:
    ``read(k)`` returns the next k shots as a (k, n) bool array, fewer only at the end of the
    data. Each batch is cut at whole shots and decoded by ``decode_shots``, so the shots and
    the messages for bad data are the ones the whole file read at once gives."""

    _BLOCK = 1 << 16

    def __init__(self, path: Optional[str], format: str, num_measurements: int = 0, num_detectors: int = 0, num_observables: int = 0) -> None:
        from ._shots import _format, _width

        self.format = _format(format)
        self.counts = (num_measurements, num_detectors, num_observables)
        self.n = _width(num_measurements, num_detectors, num_observables)
        self.f = sys.stdin.buffer if path is None else open(path, "rb")
        self.owned = path is not None
        self.buf, self.eof, self.consumed, self.pending = b"", False, 0, None

    def __enter__(self) -> "_ShotReader":
        return self

    def __exit__(self, *exc) -> None:
        if self.owned:
            self.f.close()

    def _decode(self, raw: bytes) -> np.ndarray:
        from ._shots import decode_shots

        return decode_shots(raw, self.format, *self.counts)

    def _take(self, size: int) -> bytes:
        raw = self.f.read(size)
        self.consumed += len(raw)
        if len(raw) < size:
            self.eof = True
        return raw

    def _whole_length(self) -> int:
        """The data's length in bytes: those read and those left (read to count them)."""
        while self._take(self._BLOCK):
            pass
        return self.consumed

    def _next(self) -> Optional[np.ndarray]:
        """The next run of whole shots (any number), or None at the end of the data."""
        if self.eof and not self.buf:
            return None
        n, fmt = self.n, self.format
        if fmt in ("b8", "ptb64"):
            size = (n + 7) // 8 if fmt == "b8" else 8 * n
            if size == 0:  # no bits a shot: read as the whole file is (b8: no shots; ptb64: refused)
                self.eof = True
                return self._decode(self.f.read())
            per = _BATCH if fmt == "b8" else _BATCH // 64
            raw = self._take(per * size)
            if len(raw) % size:
                what = "shots" if fmt == "b8" else "blocks"
                raise ValueError(f"{fmt} data of {self._whole_length()} bytes is not whole {what} of {size} bytes")
            return self._decode(raw) if raw else None
        while True:
            # Up to a batch of whole shots from what is buffered, reading more only when it
            # holds none; at the end of the data, the rest (a last line without its newline,
            # or the data's fault, which decoding reports).
            end = _r8_end(self.buf, n, _BATCH) if fmt == "r8" else _lines_end(self.buf, _BATCH)
            if not end and self.eof:
                end = len(self.buf)
            if end:
                chunk, self.buf = self.buf[:end], self.buf[end:]
                return self._decode(chunk)
            if self.eof:
                return None
            self.buf += self._take(self._BLOCK)

    def read(self, k: int) -> np.ndarray:
        parts, have = [], 0
        while have < k:
            if self.pending is not None:
                rows, self.pending = self.pending, None
            else:
                rows = self._next()
            if rows is None:
                break
            if len(rows) > k - have:
                rows, self.pending = rows[: k - have], rows[k - have :]
            parts.append(rows)
            have += len(rows)
        return np.concatenate(parts) if parts else np.zeros((0, self.n), dtype=bool)


def _lines_end(buf: bytes, shots: int) -> int:
    """The end of the first ``shots`` whole lines in ``buf``, or of as many as it holds (0 if none)."""
    end = 0
    for _ in range(shots):
        i = buf.find(b"\n", end)
        if i < 0:
            break
        end = i + 1
    return end


def _r8_end(buf: bytes, n: int, shots: int) -> int:
    """The end of the first ``shots`` whole r8 shots in ``buf``, or of as many as it holds (0 if
    none), or the end of the first byte that runs past its shot (so decoding up to it reports
    that)."""
    end = k = 0
    for i, b in enumerate(buf):
        k += b
        if b == 255:
            continue
        if k == n:
            end, k, shots = i + 1, 0, shots - 1
            if not shots:
                break
        elif k > n:
            return i + 1
        else:
            k += 1
    return end


def _sample_batches(shots: int, *formats: str):
    """The sizes of the batches a run of ``shots`` samples, after checking ptb64 outputs take
    them (a negative count is passed on, for the sampler's own message)."""
    if shots < 0:
        yield shots
        return
    if any(f == "ptb64" for f in formats) and shots % 64:
        raise ValueError("shots must be a multiple of 64 to use ptb64 format.")
    while shots > 0:
        yield min(shots, _BATCH)
        shots -= _BATCH


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
    sampler = c.compile_sampler(skip_reference_sample=a.skip_reference_sample or a.frame0, seed=a.seed)
    with _streamed(_Output(a.out)) as (out,):
        for k in _sample_batches(a.shots, a.out_format):
            out.write(encode_shots(sampler.sample(k), a.out_format, num_measurements=c.num_measurements))


def _detection_outputs(a):
    """The outputs of detect and m2d: the detection events (stdout by default) and --obs_out."""
    return _streamed(_Output(a.out), _Output(a.obs_out) if a.obs_out else None)


def _detections_out(a, outs, dets: np.ndarray, obs: np.ndarray, nd: int, no: int) -> None:
    """Writes a batch of detection events and observables to ``outs`` (``_detection_outputs``)."""
    from ._shots import encode_shots

    out, obs_out = outs
    prepend = getattr(a, "prepend_observables", False)
    if a.append_observables or prepend:
        rows = np.hstack(([obs] if prepend else []) + [dets] + ([obs] if a.append_observables else []))
        names = [f"L{k}" for k in range(no)] * prepend + [f"D{k}" for k in range(nd)] + [f"L{k}" for k in range(no)] * a.append_observables
        out.write(encode_shots(rows, a.out_format, num_detectors=rows.shape[1], names=names))
    else:
        out.write(encode_shots(dets, a.out_format, num_detectors=nd))
    if obs_out is not None:
        obs_out.write(encode_shots(obs, a.obs_out_format, num_observables=no))


def cmd_detect(a) -> None:
    if a.prepend_observables:
        print("[DEPRECATION] Avoid using `--prepend_observables`. Data readers assume observables are appended, not prepended.", file=sys.stderr)
    if a.out_format == "dets" and not a.append_observables:
        a.prepend_observables = True  # as Stim: dets output names the observables, first
    if a.prepend_observables + a.append_observables + (a.obs_out is not None) > 1:
        raise UsageError("Can't combine --prepend_observables, --append_observables, or --obs_out")
    c = _circuit(_read_text(a.input))
    sampler = c.compile_detector_sampler(seed=a.seed)
    with _detection_outputs(a) as outs:
        for k in _sample_batches(a.shots, a.out_format, a.obs_out_format if a.obs_out else a.out_format):
            dets, obs = sampler.sample(k, separate_observables=True)
            _detections_out(a, outs, dets, obs, c.num_detectors, c.num_observables)


def cmd_m2d(a) -> None:
    c = _circuit(_read_text(a.circuit))
    if a.ran_without_feedback:
        # As Stim: the results are of the circuit with its feedback left out.
        c = c.with_inlined_feedback()
    reference = c.reference_sample() if a.skip_reference_sample else None
    converter = c.compile_m2d_converter()
    nd, no = c.num_detectors, c.num_observables
    with _ShotReader(a.input, a.in_format, num_measurements=c.num_measurements) as meas_in, _optional_reader(a.sweep, a.sweep_format, num_measurements=c.num_sweep_bits) as sweep_in, _detection_outputs(a) as outs:
        done = 0
        while True:
            meas = meas_in.read(_BATCH)
            last = len(meas) < _BATCH
            sweep = None
            if sweep_in is not None:
                # The sweep data must end where the measurements do: checked before the batch is
                # written, what is left of the longer one counted for the message.
                sweep = sweep_in.read(len(meas))
                if len(sweep) < len(meas):
                    raise ValueError(f"{done + len(sweep)} shots of sweep bits for {done + len(meas) + _count_rest(meas_in)} shots of measurements")
                if last and len(sweep_in.read(1)):
                    raise ValueError(f"{done + len(sweep) + 1 + _count_rest(sweep_in)} shots of sweep bits for {done + len(meas)} shots of measurements")
            if len(meas):
                if reference is not None:
                    meas = meas ^ reference
                dets, obs = converter.convert(measurements=meas, sweep_bits=sweep, separate_observables=True)
                _detections_out(a, outs, dets, obs, nd, no)
                done += len(meas)
            if last:
                break


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
    from ._circuit import _DIAGRAM_KIND, DetectorErrorModel

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
    if _DIAGRAM_KIND.get(kind) == "matchgraph":
        # Every match graph type (svg, 3d, their html pages and match-graph-* names) draws a
        # detector error model, or a circuit's, as Stim's command line does; its noise makes it.
        if a.remove_noise:
            raise UsageError("--remove_noise is incompatible with match graph diagrams, because the noise is needed to produce the match graph.")
        text = _read_text(a.input)
        try:
            d = DetectorErrorModel(text).diagram(kind)
        except ValueError:
            d = _circuit(text).diagram(kind)
    else:
        text = _read_text(a.input)
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
    from ._shots import encode_shots

    m, d, o = _counts(a)
    split = bool(a.obs_out and o)
    with _ShotReader(a.input, a.in_format, m, d, o) as shots_in, _streamed(_Output(a.out), _Output(a.obs_out) if split else None) as (out, obs_out):
        while True:
            data = shots_in.read(_BATCH)
            if len(data):
                if split:
                    out.write(encode_shots(data[:, : m + d], a.out_format, m, d, 0))
                    obs_out.write(encode_shots(data[:, m + d :], a.obs_out_format, num_observables=o))
                else:
                    out.write(encode_shots(data, a.out_format, m, d, o))
            if len(data) < _BATCH:
                break


def cmd_sample_dem(a) -> None:
    from ._circuit import DetectorErrorModel
    from ._shots import encode_shots

    dem = DetectorErrorModel(_read_text(a.input))
    s = dem.compile_sampler(seed=a.seed)
    formats = [a.out_format] + [a.obs_out_format] * bool(a.obs_out) + [a.err_out_format] * bool(a.err_out)
    outputs = _streamed(_Output(a.out), _Output(a.obs_out) if a.obs_out else None, _Output(a.err_out) if a.err_out else None)
    with _optional_reader(a.replay_err_in, a.replay_err_in_format, num_measurements=s.num_errors) as replay_in, outputs as (out, obs_out, err_out):
        for k in _sample_batches(a.shots, *formats):
            replay = None
            if replay_in is not None:
                replay = replay_in.read(max(k, 0))  # extra error data is not read, as in Stim
                if len(replay) < k:
                    raise ValueError("Expected more error data for the requested number of shots.")
            dets, obs, errs = s.sample(k, return_errors=err_out is not None, recorded_errors_to_replay=replay)
            out.write(encode_shots(dets, a.out_format, num_detectors=s.num_detectors))
            if obs_out is not None:
                obs_out.write(encode_shots(obs, a.obs_out_format, num_observables=s.num_observables))
            if err_out is not None:
                err_out.write(encode_shots(errs, a.err_out_format, num_measurements=s.num_errors))


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


# Relay-BP's shot s of a batch draws its memory strengths from (seed + (s + 1) * this) (src/batch.rs
# shot_seed, mod 2**64), so the batch from shot k on is decoded as a whole run decodes it by a
# decoder whose seed is k times this.
_RELAY_SHOT_STEP = 0x9E3779B97F4A7C15


def _decoded_batches(a, dem, decode):
    """Reads --in a batch at a time (the observables appended to it, with
    --in_includes_appended_observables, left out) and yields (detection events, observables
    or None, predictions from ``decode(events, first shot)``)."""
    nd, no = dem.num_detectors, dem.num_observables
    appended = a.in_includes_appended_observables
    with _ShotReader(a.input, a.in_format, num_detectors=nd, num_observables=no if appended else 0) as shots_in:
        done = 0
        while True:
            data = shots_in.read(_BATCH)
            if len(data):
                dets = data[:, :nd]
                yield dets, (data[:, nd:] if appended else None), decode(dets, done).astype(bool)
                done += len(data)
            if len(data) < _BATCH:
                return


def cmd_decode(a) -> None:
    from ._circuit import DetectorErrorModel
    from ._shots import encode_shots
    from ._decoders import AmbiguityClustering, BeliefMatching, CollisionClustering, LocalClustering, BpLsd, BpOsd, ColorMatching, Matching, RelayBp, SearchDecoder, UnionFind
    from ._matching import _CorrelatedMatching

    dem = DetectorErrorModel(_read_text(a.dem))
    make = {
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
    }[a.decoder]
    decoder = make()

    def decode(dets: np.ndarray, first: int) -> np.ndarray:
        if a.decoder == "relay_bp" and first:
            # Relay-BP draws each shot's strengths from its index in the batch: this batch
            # by a decoder seeded as the whole run's would be at its first shot.
            return RelayBp(dem, seed=(first * _RELAY_SHOT_STEP) % (1 << 64)).decode_batch(dets)
        return decoder.decode_batch(dets)

    with _streamed(_Output(a.out)) as (out,):
        for _, _, predictions in _decoded_batches(a, dem, decode):
            out.write(encode_shots(predictions, a.out_format, num_observables=dem.num_observables))


def _matching(a):
    """PyMatching's CLI inputs: the model, and its matching."""
    from ._circuit import DetectorErrorModel
    from ._matching import Matching

    dem = DetectorErrorModel(_read_text(a.dem))
    return dem, Matching.from_detector_error_model(dem, enable_correlations=a.enable_correlations)


def cmd_predict(a) -> None:
    from ._shots import encode_shots

    dem, m = _matching(a)
    with _streamed(_Output(a.out)) as (out,):
        for _, _, predictions in _decoded_batches(a, dem, lambda dets, _: m.decode_batch(dets, enable_correlations=a.enable_correlations)):
            out.write(encode_shots(predictions, a.out_format, num_observables=dem.num_observables))


def cmd_count_mistakes(a) -> None:
    """PyMatching's count_mistakes: the shots whose predicted observables differ from the
    actual ones, read from --obs_in when it is given (even with appended observables), else
    the appended ones, else none (all zero: PyMatching's --obs_in defaults to stdin, which it
    never reads). Extra --obs_in shots are not read; too few is PyMatching's error. --time
    writes PyMatching's two timing lines to stderr."""
    import time

    dem, m = _matching(a)
    no = dem.num_observables
    m.decode_batch(np.zeros((0, dem.num_detectors), dtype=bool))  # the decoder is built before the clock starts, as PyMatching's is
    mistakes = shots = 0
    start = time.perf_counter()
    with _optional_reader(a.obs_in, a.obs_in_format, num_observables=no) as obs_in:
        for dets, appended, predictions in _decoded_batches(a, dem, lambda dets, _: m.decode_batch(dets, enable_correlations=a.enable_correlations)):
            if obs_in is not None:
                obs = obs_in.read(len(dets))
                if len(obs) < len(dets):
                    raise ValueError("Obs data ended before shot data ended.")
            else:
                obs = appended if appended is not None else np.zeros((len(dets), no), dtype=bool)
            mistakes += int((predictions != obs).any(axis=1).sum())
            shots += len(dets)
    _write(a.out, f"{mistakes} / {shots}\n".encode())  # bare \n on every platform, as PyMatching's
    if a.time:
        # As PyMatching: whole microseconds, and the time per shot as C++ streams print a double.
        micros = float(int((time.perf_counter() - start) * 1e6))
        per = micros / shots if shots else (float("inf") if micros else float("nan"))
        sys.stderr.write(f"Total decoding time: {int(micros)}us\nDecoding time per shot: {per:g}us\n")
        sys.stderr.flush()


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
