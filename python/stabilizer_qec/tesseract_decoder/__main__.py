# Ported from Tesseract (quantumlib/tesseract-decoder, src/tesseract_main.cc; Copyright 2025 Google
# LLC, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec. Its flags are
# read, and its parsing errors and help printed, as the library it parses them with prints them
# (p-ranav/argparse 3.1, MIT License); this is a reimplementation of that behaviour, not its code.
"""Tesseract's ``tesseract`` program: ``python -m stabilizer_qec.tesseract_decoder`` with its
flags (``--circuit``, ``--dem``, ``--sample-num-shots``, ``--in``, ``--out``, the detector-order
flags, ``--beam``, ``--pqlimit``, ``--multipass``, ``--stats-out``, ...), its output (the
``num_shots = ... total_time_seconds = ...`` line, ``--print-stats``, ``--verbose``, the
``--dem-out`` model and the ``--stats-out`` JSON) and its errors. ``--help`` lists the flags.

Where it differs from Tesseract's program:

- ``--sample-num-shots`` samples with this package's detector sampler from ``--sample-seed``:
  the same seed gives the same shots here on any machine, but not Stim's shots.
- An exception that leaves Tesseract's ``main`` (each check after parsing, a file it cannot
  open, ...) ends it through the C++ runtime: its message (libc++'s words on macOS, libstdc++'s
  elsewhere) and an abort. Here the same message, and exit status 134 (an abort's, as a shell
  reports it) with no core dump.
- ``--out`` to a file that cannot be opened: Tesseract's program writes through a null file
  (and crashes); here it stops with ``Failed to open <path>``, as for ``--dem-out``.
- ``--max-errors``: Tesseract's program stops once the shots it has taken in order reach the
  limit, while its threads may have decoded (and counted for ``--dem-out``) shots past it,
  depending on timing; here decoding stops at the shot that reaches the limit.
- ``--threads``: the shots are decoded across threads by one decoder (Tesseract's program
  makes one per thread, and with ``--verbose`` each prints its errors); with ``--verbose`` or
  ``--multipass`` the shots are decoded in order on one thread.
- A circuit, model or shot file that cannot be read is refused with this package's message,
  not Stim's.
- Any number of observables, as Tesseract's.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SIZE_MAX = 2**64 - 1
_INT_MIN, _INT_MAX = -(2**31), 2**31 - 1
_INF_DET_BEAM = 65535
# (Stim's format names, in the order of its std::map.)
_FORMATS = ("01", "b8", "dets", "hits", "ptb64", "r8")
_FORMAT_LIST = "/".join(_FORMATS)
_TEXT_FORMATS = ("01", "dets", "hits")
_METHODS = {"bfs": 0, "index": 1, "coordinate": 2}


class _ParseError(Exception):
    """What argparse throws while parsing: Tesseract's program prints it and its help."""


class _Abort(Exception):
    """An exception out of Tesseract's ``main``: the C++ runtime's message, then abort."""

    def __init__(self, message: str, kind: str = "std::invalid_argument") -> None:
        super().__init__(message)
        self.kind = kind


class _Exit(Exception):
    """argparse's ``std::exit(0)`` after ``--help`` or ``--version``."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text


def _terminate(kind: str, message: str) -> str:
    if sys.platform == "darwin":
        return f"libc++abi: terminating due to uncaught exception of type {kind}: {message}\n"
    return f"terminate called after throwing an instance of '{kind}'\n  what():  {message}\n"


# -- argparse's number parsing ---------------------------------------------------------------

_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def _from_chars(s: str, base: int, lo: int, hi: int) -> int:
    """``std::from_chars`` of an integer as argparse reads one: the whole string, or its error."""
    i = 1 if lo < 0 and s[:1] == "-" else 0
    j = i
    while j < len(s) and s[j].isascii() and s[j].lower() in _DIGITS[:base]:
        j += 1
    if j == i:
        raise _ParseError(f"pattern '{s}' not found")
    value = int(s[i:j], base) * (-1 if i else 1)
    if not lo <= value <= hi:
        raise _ParseError(f"'{s}' not representable")
    if j != len(s):
        raise _ParseError(f"pattern '{s}' does not match to the end")
    return value


def _decimal(lo: int, hi: int) -> Callable[[str], int]:
    return lambda s: _from_chars(s, 10, lo, hi)


def _scanned_int(s: str) -> int:
    """``scan<'i', int>``: hexadecimal after ``0x``, binary after ``0b``, octal after a ``0``,
    else decimal."""
    for prefixes, base, name in ((("0x", "0X"), 16, "hexadecimal"), (("0b", "0B"), 2, "binary")):
        if s[:2] in prefixes:
            try:
                return _from_chars(s[2:], base, _INT_MIN, _INT_MAX)
            except _ParseError as e:
                raise _ParseError(f"Failed to parse '{s}' as {name}: {e}") from None
    base, name = (8, "octal") if s.startswith("0") else (10, "decimal integer")
    try:
        return _from_chars(s, base, _INT_MIN, _INT_MAX)
    except _ParseError as e:
        raise _ParseError(f"Failed to parse '{s}' as {name}: {e}") from None


_STRTOD = re.compile(
    r"[+-]?(?:inf(?:inity)?|nan(?:\([0-9a-z_]*\))?|0x(?:[0-9a-f]+\.?[0-9a-f]*|\.[0-9a-f]+)(?:p[+-]?[0-9]+)?|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:e[+-]?[0-9]+)?)",
    re.IGNORECASE,
)


def _strtod(s: str) -> float:
    """A double as argparse reads one (``strtod``, refusing a leading space or ``+``, and hex)."""
    if s[:2] in ("0x", "0X"):
        raise _ParseError("chars_format::general does not parse hexfloat")
    if s[:2] in ("0b", "0B"):
        raise _ParseError("chars_format::general does not parse binfloat")
    if s[:1] in (" ", "\t", "\n", "\v", "\f", "\r", "+"):
        raise _ParseError(f"pattern '{s}' not found")
    m = _STRTOD.match(s)
    if m is None:
        # (No number: strtod reads nothing, which is all of an empty string.)
        if s:
            raise _ParseError(f"pattern '{s}' does not match to the end")
        return 0.0
    text = m.group(0)
    body = text.lstrip("+-").lower()
    sign = -1.0 if text.startswith("-") else 1.0
    if body.startswith("inf"):
        value = sign * math.inf
    elif body.startswith("nan"):
        value = math.nan
    elif body.startswith("0x"):
        try:
            value = float.fromhex(text)
        except OverflowError:
            raise _ParseError(f"'{s}' not representable") from None
    else:
        value = float(text)
    mantissa = body.split("p")[0] if body.startswith("0x") else body.split("e")[0]
    nonzero = any(c not in "0x." for c in mantissa)
    if (math.isinf(value) and not body.startswith("inf")) or (nonzero and abs(value) < 2.2250738585072014e-308):
        raise _ParseError(f"'{s}' not representable")
    if m.end() != len(s):
        raise _ParseError(f"pattern '{s}' does not match to the end")
    return value


# -- argparse's arguments ----------------------------------------------------------------------

_NO_DEFAULT = object()


class _Arg:
    """One of argparse's optional arguments: its names (sorted by length, then text), help,
    metavar, number of values and actions."""

    def __init__(self, names: Sequence[str], help: str, actions: Sequence[Callable[[str], Any]] = (), metavar: str = "", default: Any = _NO_DEFAULT,
                 flag: bool = False, repeatable: bool = False) -> None:
        self.names = sorted(names, key=lambda n: (len(n), n))
        self.help, self.metavar, self.repeatable, self.flag = help, metavar, repeatable, flag
        self.actions = list(actions)
        self.nargs = (0, 0) if flag else ((0, 1) if default is not _NO_DEFAULT else (1, 1))
        self.default = None if default is _NO_DEFAULT else default
        self.used = False

    def longest_name(self) -> str:
        out = self.names[0]
        for n in self.names:
            if len(n) > len(out):
                out = n
        return out

    def inline_usage(self) -> str:
        out = "[" + self.longest_name() + (" " + (self.metavar or "VAR") if self.nargs[1] > 0 else "") + "]"
        return out + ("..." if self.repeatable else "")

    def usage_full(self) -> str:
        return "/".join(self.names) + (" " + (self.metavar or "VAR") if self.nargs[1] > 0 else "")

    def length(self) -> int:
        size = sum(len(n) for n in self.names) + 2 * (len(self.names) - 1)
        if self.metavar and self.nargs == (1, 1):
            size += len(self.metavar) + 1
        return size + 2

    def help_line(self, width: int) -> str:
        name = "  " + ", ".join(self.names)
        if self.metavar and self.nargs == (1, 1):
            name += " " + self.metavar
        line = name.ljust(width) + "  " + self.help + (" " if self.help else "")
        lo, hi = self.nargs
        if lo != hi:
            line += f"[nargs={lo}..{hi}] "
        elif lo not in (0, 1):
            line += f"[nargs: {lo}] "
        shown = self.default is not None and self.nargs != (0, 0)
        if shown:
            line += f"[default: {self.default}]"
        if self.repeatable:
            line += (" " if shown else "") + "[may be repeated]"
        return line

    def consume(self, args: List[str], start: int, used_name: str) -> int:
        if not self.repeatable and self.used:
            raise _ParseError("Duplicate argument " + used_name)
        lo, hi = self.nargs
        if hi == 0:
            self.used = True
            for action in self.actions:
                action("")
            return start
        if len(args) - start >= lo:
            values = args[start : start + hi]
            cut = next((k for k, v in enumerate(values) if not _is_positional(v)), len(values))
            values = values[:cut]
            if len(values) < lo:
                raise _ParseError(f"Too few arguments for '{used_name}'.")
            for action in self.actions:
                for v in values:
                    action(v)
            self.used = True
            return start + len(values)
        if self.default is not None:
            self.used = True
            return start
        raise _ParseError(f"Too few arguments for '{used_name}'.")


_DECIMAL_LITERAL = re.compile(r"0|[1-9][0-9]*|[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?|\.[0-9]+(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+")


def _is_positional(arg: str) -> bool:
    """argparse's test: a value, not an option (``-3`` and ``-.5`` are values)."""
    if not arg.startswith("-"):
        return True
    return arg == "-" or _DECIMAL_LITERAL.fullmatch(arg[1:]) is not None


class _Args:
    """Tesseract's program's settings (its ``Args``), at their defaults."""

    def __init__(self) -> None:
        self.multipass = False
        self.print_multipass_plan = False
        self.multipass_strategy = "causal"
        self.num_passes = 2
        self.circuit_path = ""
        self.dem_path = ""
        self.no_merge_errors = False
        self.det_order_seed = 518278944
        self.num_orders = 1
        self.sources: List[Tuple[str, str]] = []
        self.sample_num_shots = 0
        self.max_errors = _SIZE_MAX
        self.sample_seed = random.SystemRandom().getrandbits(32)
        self.shot_range_begin = 0
        self.shot_range_end = 0
        self.in_fname = ""
        self.in_format = ""
        self.obs_in_fname = ""
        self.obs_in_format = ""
        self.append_observables = False
        self.out_fname = ""
        self.out_format = ""
        self.dem_out_fname = ""
        self.stats_out_fname = ""
        self.num_threads = os.cpu_count() or 1
        self.det_beam = _INF_DET_BEAM
        self.det_penalty = 0.0
        self.beam_climbing = False
        self.no_revisit_dets = False
        self.pqlimit = _SIZE_MAX
        self.verbose = False
        self.print_stats = False
        self.sparsify_errors = False
        self.sparsify_base_degree = -1
        self.sparsify_max_degree = -1
        self.sparsify_reactivate_limit = -1

    def has_observables(self) -> bool:
        return self.append_observables or bool(self.obs_in_fname) or self.sample_num_shots > 0


def _store(args: _Args, field: str, parse: Callable[[str], Any] = str) -> Callable[[str], None]:
    return lambda v: setattr(args, field, parse(v))


def _set(args: _Args, field: str) -> Callable[[str], None]:
    return lambda v: setattr(args, field, True)


def _strategy(args: _Args) -> Callable[[str], None]:
    def parse(value: str) -> None:
        if value not in ("static", "causal"):
            raise _ParseError(f"Invalid multi-pass scheduling strategy '{value}'; expected 'static' or 'causal'.")
        args.multipass_strategy = value

    return parse


def _quoted(s: str) -> str:
    return '"' + s + '"'


class _Parser:
    """Tesseract's program's ``argparse::ArgumentParser("tesseract")``."""

    def __init__(self, args: _Args) -> None:
        a = args
        size_t, uint64 = _decimal(0, _SIZE_MAX), _decimal(0, 2**64 - 1)
        source = lambda method: (lambda v: a.sources.append((method, "")))  # noqa: E731
        shot_range = ("Useful for processing a fragment of a file. If shot_range_begin == 0 and shot_range_end == 0 (the default), then all available shots will be "
                      "decoded. Otherwise, only those in the range [shot_range_begin, shot_range_end) will be decoded.")
        self.arguments = [
            _Arg(["-h", "--help"], "shows help message and exits", [self._help], default="false", flag=True),
            _Arg(["-v", "--version"], "prints version information and exits", [self._version], default="false", flag=True),
            _Arg(["--circuit"], "Stim circuit file path", [_store(a, "circuit_path")]),
            _Arg(["--dem"], "Stim dem file path", [_store(a, "dem_path")]),
            _Arg(["--no-merge-errors"], "If provided, will not merge identical error mechanisms. Multi-pass supports this only with --num-passes=1; two-pass "
                 "reweighting requires merged mechanisms.", [_set(a, "no_merge_errors")], flag=True),
            _Arg(["--num-det-orders"], "Number of orders generated by each selected detector-order method", [_store(a, "num_orders", size_t)], "N", "1"),
            _Arg(["--det-order-bfs"], "Add BFS-based detector orders", [source("bfs")], flag=True),
            _Arg(["--det-order-index"], "Add randomly increasing or decreasing detector index orders (default when no source is specified)", [source("index")], flag=True),
            _Arg(["--det-order-coordinate"], "Add random geometric detector orientation orders", [source("coordinate")], flag=True),
            _Arg(["--det-order-seed"], "Base seed used independently by each generated detector-order method.", [_store(a, "det_order_seed", uint64)], "N", "518278944"),
            _Arg(["--detector-orders"], "Add detector-ID permutations from a JSON file. May be repeated and combined with generated detector-order methods.",
                 [lambda v: a.sources.append(("literal", v))], "FILE", repeatable=True),
            _Arg(["--sample-num-shots"], "If provided, will sample the requested number of shots from the Stim circuit and decode them. May end early if "
                 "--max-errors errors are reached before decoding all shots.", [_store(a, "sample_num_shots", size_t)]),
            _Arg(["--max-errors"], "If provided, will sample at least this many errors from the Stim circuit and decode them.", [_store(a, "max_errors", size_t)]),
            _Arg(["--sample-seed"], "Seed used when initializing the random number generator for sampling shots", [_store(a, "sample_seed", uint64)], "N",
                 str(a.sample_seed)),
            _Arg(["--shot-range-begin"], shot_range, [_store(a, "shot_range_begin", size_t)], default="0"),
            _Arg(["--shot-range-end"], shot_range, [_store(a, "shot_range_end", size_t)], default="0"),
            _Arg(["--in"], "File to read detection events (and possibly observable flips) from", [_store(a, "in_fname")], "filename", _quoted("")),
            _Arg(["--in-format", "--in_format"], f"Format of the file to read detection events from ({_FORMAT_LIST})", [_store(a, "in_format")], _FORMAT_LIST, _quoted("")),
            _Arg(["--in-includes-appended-observables", "--in_includes_appended_observables"], "If present, assumes that the observable flips are appended to the end of each shot.",
                 [_set(a, "append_observables")], default="false", flag=True),
            _Arg(["--obs_in", "--obs-in"], "File to read observable flips from", [_store(a, "obs_in_fname")], "filename", _quoted("")),
            _Arg(["--obs-in-format", "--obs_in_format"], f"Format of the file to read observable flips from ({_FORMAT_LIST})", [_store(a, "obs_in_format")], _FORMAT_LIST, _quoted("")),
            _Arg(["--out"], "File to write observable flip predictions to (or - for stdout)", [_store(a, "out_fname")], "filename", _quoted("")),
            _Arg(["--out-format"], f"Format of the file to write observable flip predictions to ({_FORMAT_LIST})", [_store(a, "out_format")], _FORMAT_LIST, _quoted("")),
            _Arg(["--dem-out"], "File to write matching frequency dem to", [_store(a, "dem_out_fname")], "filename", _quoted("")),
            _Arg(["--stats-out"], "File to write high-level statistics and metadata to", [_store(a, "stats_out_fname")], "filename", _quoted("")),
            _Arg(["--threads"], "Number of decoder threads to use", [_store(a, "num_threads", size_t)], "N", str(a.num_threads)),
            _Arg(["--beam"], "Beam to use for truncation (default = infinity)", [_store(a, "det_beam", size_t)], "N", str(_INF_DET_BEAM)),
            _Arg(["--det-penalty"], "Penalty cost to add per activated detector in the residual syndrome.", [_store(a, "det_penalty", _strtod)], "D", "0"),
            _Arg(["--beam-climbing"], "Use beam-climbing heuristic", [_set(a, "beam_climbing")], flag=True),
            _Arg(["--no-revisit-dets"], "Use no-revisit-dets heuristic", [_set(a, "no_revisit_dets")], flag=True),
            _Arg(["--pqlimit"], "Maximum size of the priority queue (default = infinity)", [_store(a, "pqlimit", size_t)], "N", str(_SIZE_MAX)),
            _Arg(["--verbose"], "Increases output verbosity", [_set(a, "verbose")], flag=True),
            _Arg(["--print-stats"], "Prints out the number of shots (and number of errors, if known) during decoding.", [_set(a, "print_stats")], flag=True),
            _Arg(["--multipass"], "Enable multi-pass graph shattering for correlated error decoding", [_set(a, "multipass")], flag=True),
            _Arg(["--print-multipass-plan"], "Print the multi-pass components, dependencies, and schedule to stderr", [_set(a, "print_multipass_plan")], flag=True),
            _Arg(["--multipass-strategy", "--multipass_strategy"], "Multi-pass scheduling strategy: static or causal (default = causal). Note: static scheduling is "
                 "experimental and was never systematically benchmarked.", [_strategy(a)], default=_quoted("causal")),
            _Arg(["--num-passes", "--num_passes"], "Number of prior propagation passes: 1 (uncorrelated independent CSS decoding) or 2 (standard causally reweighted "
                 "decoding, default = 2).", [_store(a, "num_passes", size_t)], default="2"),
            _Arg(["--sparsify-errors"], "Enables per-shot sparse error activation.", [_set(a, "sparsify_errors")], flag=True),
            _Arg(["--sparsify-base-degree"], "Maximum detector degree for mandatory errors. Errors with degree <= K are always enabled for every shot.",
                 [_scanned_int, _store(a, "sparsify_base_degree", _decimal(_INT_MIN, _INT_MAX))], "K"),
            _Arg(["--sparsify-max-degree"], "Maximum detector degree for optional errors that may be reactivated. Errors with degree > M are never enabled.",
                 [_scanned_int, _store(a, "sparsify_max_degree", _decimal(_INT_MIN, _INT_MAX))], "M"),
            _Arg(["--sparsify-reactivate-limit"], "Maximum number of optional errors to reactivate per shot. Use -1 for auto.",
                 [_scanned_int, _store(a, "sparsify_reactivate_limit", _decimal(_INT_MIN, _INT_MAX))], "N"),
        ]
        self.by_name: Dict[str, _Arg] = {n: arg for arg in self.arguments for n in arg.names}

    def help(self) -> str:
        width = max(arg.length() for arg in self.arguments)
        usage = "Usage: tesseract" + "".join(" " + arg.inline_usage() for arg in self.arguments)
        return usage + "\n\nOptional arguments:\n" + "".join(arg.help_line(width) + "\n" for arg in self.arguments)

    def _help(self, _: str) -> None:
        raise _Exit(self.help())

    def _version(self, _: str) -> None:
        raise _Exit("1.0\n")

    def used(self, name: str) -> bool:
        return self.by_name[name].used

    def parse(self, raw: Sequence[str]) -> None:
        args: List[str] = []
        for arg in raw:
            # (``--name=value`` for a known name is ``--name value``.)
            if arg not in self.by_name and arg.startswith("--") and "=" in arg and arg[: arg.index("=")] in self.by_name:
                args += [arg[: arg.index("=")], arg[arg.index("=") + 1 :]]
            else:
                args.append(arg)
        k = 0
        while k < len(args):
            current = args[k]
            if _is_positional(current):
                for arg in self.arguments:
                    if not arg.flag and not arg.used:
                        raise _ParseError("Zero positional arguments expected, did you mean " + arg.usage_full())
                raise _ParseError("Zero positional arguments expected")
            if current in self.by_name:
                k = self.by_name[current].consume(args, k + 1, current)
            elif len(current) > 1 and current[1] != "-":
                # (Single-letter options run together, ``-hv``.)
                k += 1
                for c in current[1:]:
                    arg = self.by_name.get("-" + c)
                    if arg is None:
                        raise _ParseError("Unknown argument: " + current)
                    k = arg.consume(args, k, "-" + c)
            else:
                raise _ParseError("Unknown argument: " + current)


# -- Tesseract's program ---------------------------------------------------------------------


def _validate(a: _Args, parser: _Parser) -> None:
    if not a.circuit_path and not a.dem_path:
        raise _Abort("Must provide at least one of --circuit or --dem")
    if not a.sources:
        a.sources.append(("index", ""))
    generated = any(method != "literal" for method, _ in a.sources)
    if not generated and (parser.used("--num-det-orders") or parser.used("--det-order-seed")):
        raise _Abort(
            "--num-det-orders and --det-order-seed only apply to generated detector orders. Select --det-order-bfs, --det-order-index, or "
            "--det-order-coordinate to combine generated orders with --detector-orders files."
        )
    if generated and a.num_orders == 0:
        raise _Abort("--num-det-orders must be at least 1.")
    if int(a.sample_num_shots > 0) + int(bool(a.in_fname)) != 1:
        raise _Abort("Requires exactly 1 source of shots.")
    if a.in_fname and not a.in_format:
        raise _Abort("If --in is provided, must also specify --in-format.")
    if a.out_fname and not a.out_format:
        raise _Abort("If --out is provided, must also specify --out-format.")
    for fmt in (a.in_format, a.obs_in_format, a.out_format):
        if fmt and fmt not in _FORMATS:
            raise _Abort("Invalid format: " + fmt)
    if a.obs_in_fname and not a.in_fname:
        raise _Abort("Cannot load observable flips without a corresponding detection event data file.")
    if a.num_threads == 0:
        raise _Abort("--threads must be at least 1.")
    if not 1 <= a.num_passes <= 2:
        raise _Abort("--num-passes must be 1 or 2.")
    if a.print_multipass_plan and not a.multipass:
        raise _Abort("--print-multipass-plan requires --multipass.")
    if a.multipass and a.dem_out_fname:
        raise _Abort("--dem-out is not supported when --multipass is enabled.")
    if a.num_threads > 1000:
        raise _Abort(f"There is a maximum limit of 1000 threads imposed to avoid accidentally overloading a host. You specified {a.num_threads} threads.")
    if (a.shot_range_begin or a.shot_range_end) and a.shot_range_end < a.shot_range_begin:
        raise _Abort("Provided shot range must have end >= begin.")
    if a.sample_num_shots > 0 and not a.circuit_path:
        raise _Abort("Cannot sample shots without a circuit.")
    if a.beam_climbing and a.det_beam == _INF_DET_BEAM:
        raise _Abort("Beam climbing requires a finite beam")
    has_base, has_max, has_limit = (parser.used(n) for n in ("--sparsify-base-degree", "--sparsify-max-degree", "--sparsify-reactivate-limit"))
    if not a.sparsify_errors:
        if has_base or has_max or has_limit:
            raise _Abort("Cannot use --sparsify-base-degree, --sparsify-max-degree, or --sparsify-reactivate-limit without --sparsify-errors")
    else:
        if not has_base:
            raise _Abort("Must specify --sparsify-base-degree when --sparsify-errors is enabled.")
        if a.sparsify_base_degree <= 0:
            raise _Abort("--sparsify-base-degree must be > 0.")
        if has_limit and a.sparsify_reactivate_limit < -1:
            raise _Abort("--sparsify-reactivate-limit must be >= -1.")
        if has_max and a.sparsify_max_degree < a.sparsify_base_degree:
            raise _Abort("--sparsify-max-degree must be >= --sparsify-base-degree.")


def _read(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        raise _Abort("Could not open the file: " + path) from None


class _NlohmannInt(int):
    """A JSON integer as nlohmann's parser keeps it: unsigned unless written with a sign or
    too large for 64 bits."""

    unsigned = True


def _json_int(text: str) -> Any:
    value = _NlohmannInt(int(text))
    value.unsigned = not text.startswith("-") and int(text) < 2**64
    return value


def _no_constant(name: str) -> Any:
    raise ValueError(name)


def _load_orders(path: str, num_detectors: int) -> List[List[int]]:
    """Tesseract's ``load_detector_orders``: a JSON array of detector orders, each checked."""
    raw = _read(path)
    try:
        text = raw.decode("utf-8")
        if text.startswith("﻿"):
            text = text[1:]
        orders = json.loads(text, parse_int=_json_int, parse_constant=_no_constant)
    except ValueError:
        raise _Abort("Detector orders file is not valid JSON: " + path) from None
    if not isinstance(orders, list) or not orders:
        raise _Abort("Detector orders file must contain at least one order.")
    for order in orders:
        if not isinstance(order, list):
            raise _Abort("Each detector order must be an array.")
        for d in order:
            if not (isinstance(d, _NlohmannInt) and d.unsigned):
                raise _Abort("Detector IDs must be nonnegative integers.")
    for order in orders:
        _check_order(order, num_detectors)
    return [[int(d) for d in order] for order in orders]


def _check_order(order: Sequence[int], n: int) -> None:
    if len(order) != n:
        raise _Abort(f"Detector order has size {len(order)}, but the detector error model has {n} detectors.")
    seen = [False] * n
    for position, d in enumerate(order):
        if d >= n:
            raise _Abort(f"Detector order contains out-of-range detector ID {d} at position {position}.")
        if seen[d]:
            raise _Abort(f"Detector order contains detector ID {d} more than once.")
        seen[d] = True


def _g(x: float, digits: int) -> str:
    """``std::cout << x`` at precision ``digits``."""
    return "%.*g" % (digits, x)


def _nlohmann_float(x: float) -> str:
    """A double as nlohmann's JSON writer writes one: its shortest digits, as ``1.5``, ``2.0``,
    ``0.0001`` or ``1e-05``, and ``null`` for an infinity or NaN."""
    if not math.isfinite(x):
        return "null"
    sign = "-" if math.copysign(1.0, x) < 0 else ""
    if x == 0:
        return sign + "0.0"
    t = Decimal(repr(abs(x))).normalize().as_tuple()
    digits = "".join(map(str, t.digits))
    k = len(digits)
    n = k + int(t.exponent)
    if k <= n <= 15:
        out = digits + "0" * (n - k) + ".0"
    elif 0 < n <= 15:
        out = digits[:n] + "." + digits[n:]
    elif -4 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        out = (digits if k == 1 else digits[0] + "." + digits[1:]) + "e" + ("-" if e < 0 else "+") + ("%02d" % abs(e))
    return sign + out


def _nlohmann_dump(value: Any) -> str:
    """nlohmann's compact ``dump()``: keys sorted, no spaces."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _nlohmann_float(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _nlohmann_dump(value[k]) for k in sorted(value)) + "}"
    return "[" + ",".join(_nlohmann_dump(v) for v in value) + "]"


def _plan(mp: Any, strategy: str, merge: bool) -> str:
    """Tesseract's ``MultiPassExecutionPlan::str()``."""
    from .common import merge_indistinguishable_errors, remove_zero_probability_errors
    from .utils import get_errors_from_dem

    model = mp.monolithic_dem
    if merge:
        model = merge_indistinguishable_errors(model)
    model = remove_zero_probability_errors(model)
    errors = get_errors_from_dem(model)
    nd = model.num_detectors
    degree = 0.0 if nd == 0 else sum(len(e.symptom.detectors) for e in errors) / nd
    out = [
        "Multi-pass execution plan", f"strategy: {strategy}", f"passes: {mp.num_passes}",
        f"monolithic input DEM: detectors={nd}, error_mechanisms={len(errors)}, average_detector_degree={_g(degree, 6)}", f"components: {len(mp.decoders)}",
    ]
    for c, decoder in enumerate(mp.decoders):
        active = mp.active_detector_counts[c]
        degree = 0.0 if active == 0 else sum(len(e.symptom.detectors) for e in decoder.errors) / active
        limit = decoder.config.sparsify_reactivate_limit if decoder.config.sparsify_errors else "disabled"
        out.append(
            f"  component {c}: label={mp.assignment_labels[c]}, active_detectors={active}, decoder_detectors={decoder.num_detectors}, "
            f"observable={'yes' if mp.affects_observable[c] else 'no'}, error_mechanisms={len(decoder.errors)}, average_active_detector_degree={_g(degree, 6)}, "
            f"sparsify_reactivate_limit={limit}"
        )
    out.append("dependencies:")
    counts: Dict[Tuple[int, int], int] = {}
    for source, rules in enumerate(mp.reweight_rules):
        for entries in rules.values():
            for rule in entries:
                counts[(source, rule[0])] = counts.get((source, rule[0]), 0) + 1
    if not counts:
        out.append("  none")
    for (source, target), n in sorted(counts.items()):
        out.append(f"  component {source} -> component {target}: {n} rules")
    out.append("schedule:")
    for p, components in enumerate(mp.pass_schedule):
        out.append(f"  pass {p + 1}: [" + ", ".join(map(str, components)) + "]")
    return "\n".join(out) + "\n"


class _Writer:
    """Stim's record writer for the predictions (result type ``L``), to a file or stdout."""

    def __init__(self, path: str, fmt: str, num_observables: int) -> None:
        self.fmt, self.num_observables, self.file = fmt, num_observables, None
        if path != "-":
            try:
                self.file = open(path, "wb")
            except OSError:
                self.file = None
                if fmt != "ptb64":
                    raise _Abort("Failed to open " + path) from None
        if fmt == "ptb64":
            raise _Abort("SAMPLE_FORMAT_PTB64 incompatible with SingleMeasurementRecord")

    def write(self, row: np.ndarray) -> None:
        from .._shots import encode_shots

        data = encode_shots(row.reshape(1, -1), self.fmt, num_observables=self.num_observables)
        if self.file is not None:
            self.file.write(data)
        elif self.fmt in _TEXT_FORMATS or not hasattr(sys.stdout, "buffer"):
            sys.stdout.write(data.decode("latin-1"))
        else:
            sys.stdout.flush()
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


def _shot_detector_count(circuit: Any, dem: Any) -> int:
    """The detectors a shot holds: the circuit's, when there is one (its model may append
    detectors that never fire), else the model's."""
    if circuit is None:
        return dem.num_detectors
    if circuit.num_detectors > dem.num_detectors:
        raise _Abort(
            f"Circuit has {circuit.num_detectors} detectors, but the decoding DEM has only {dem.num_detectors}. When both are supplied, circuit detector IDs "
            "must be preserved in the DEM; the DEM may only append detectors whose shot values are implicitly zero."
        )
    if circuit.num_observables != dem.num_observables:
        raise _Abort(
            f"Circuit has {circuit.num_observables} observables, but the decoding DEM has {dem.num_observables}; the counts must match when both are supplied."
        )
    return circuit.num_detectors


def _shot_data(a: _Args, circuit: Any, nd: int, no: int) -> Tuple[np.ndarray, np.ndarray]:
    """The shots to decode: (detection events, observable flips), as bool rows."""
    from .._shots import decode_shots

    dets = np.zeros((0, nd), dtype=bool)
    obs = np.zeros((0, no), dtype=bool)
    if a.sample_num_shots > 0:
        dets, obs = circuit.compile_detector_sampler(seed=a.sample_seed).sample(a.sample_num_shots, separate_observables=True)
        dets, obs = np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)
    if a.in_fname:
        raw = _read(a.in_fname)
        data = decode_shots(raw, a.in_format, num_detectors=nd, num_observables=no if a.append_observables else 0)
        dets = data[:, :nd]
        obs = data[:, nd:] if a.append_observables else np.zeros((data.shape[0], no), dtype=bool)
    if a.obs_in_fname:
        raw = _read(a.obs_in_fname)
        flips = decode_shots(raw, a.obs_in_format, num_observables=no)
        if flips.shape[0] > dets.shape[0]:
            raise _Abort("Shot data ended before obs data.")
        if flips.shape[0] < dets.shape[0]:
            raise _Abort("Obs data ended before shot data ended.")
        obs = flips
    if a.shot_range_begin or a.shot_range_end:
        if a.shot_range_end > dets.shape[0]:
            raise _Abort(f"Shot range end is past end of shots array (size {dets.shape[0]}).")
        dets, obs = dets[a.shot_range_begin : a.shot_range_end], obs[a.shot_range_begin : a.shot_range_end]
    return dets, obs


def _orders_for(sources: Sequence[Tuple[str, Any]], model: Any, count: int, seed: int) -> List[List[int]]:
    """The detector orders of each source in turn, generated ones resolved against ``model``
    (each from the same seed, as Tesseract's program makes them)."""
    from .utils import DetectorOrderMethod, _orders

    out: List[List[int]] = []
    for method, payload in sources:
        out += payload if method == "literal" else _orders(model, count, DetectorOrderMethod(_METHODS[method]), seed)
    return out


def _run(a: _Args, parser: _Parser) -> None:
    import stabilizer_qec as sq

    from .utils import get_detector_coords

    _validate(a, parser)
    circuit = sq.Circuit(_read(a.circuit_path).decode("utf-8")) if a.circuit_path else None
    if a.dem_path:
        dem = sq.DetectorErrorModel(_read(a.dem_path).decode("utf-8"))
    else:
        # (Tesseract's analysis: loops folded, gauge detectors allowed, disjoint errors approximated.)
        dem = circuit.detector_error_model(allow_gauge_detectors=True, approximate_disjoint_errors=True)
    nd = _shot_detector_count(circuit, dem)
    if a.verbose:
        lines = []
        for d, coords in enumerate(get_detector_coords(dem)):
            lines.append(f"Detector D{d} coordinate (" + ", ".join(_g(x, 16) for x in coords[:3]) + ")\n")
        sys.stdout.write("".join(lines))
    sources: List[Tuple[str, Any]] = []
    for method, path in a.sources:
        sources.append((method, _load_orders(path, dem.num_detectors)) if method == "literal" else (method, None))
    dets, obs = _shot_data(a, circuit, nd, dem.num_observables)
    writer = _Writer(a.out_fname, a.out_format, dem.num_observables) if a.out_fname else None
    try:
        _decode(a, dem, sources, dets, obs, writer)
    finally:
        if writer is not None:
            writer.close()


def _decode(a: _Args, dem: Any, sources: Any, dets: np.ndarray, obs: np.ndarray, writer: Optional[_Writer]) -> None:
    from .._util import call
    from .common import dem_from_counts
    from .tesseract import TesseractConfig, TesseractDecoder
    from .utils import DetectorOrderMethod

    no = dem.num_observables
    literal = all(method == "literal" for method, _ in sources)
    settings = dict(
        # (Tesseract's config holds the beam in an int: the low 32 bits of the one given.)
        det_beam=(a.det_beam + 2**31) % 2**32 - 2**31, beam_climbing=a.beam_climbing, no_revisit_dets=a.no_revisit_dets, verbose=a.verbose,
        merge_errors=not a.no_merge_errors, pqlimit=a.pqlimit, det_penalty=a.det_penalty, sparsify_errors=a.sparsify_errors,
        sparsify_base_degree=a.sparsify_base_degree, sparsify_max_degree=a.sparsify_max_degree, sparsify_reactivate_limit=a.sparsify_reactivate_limit,
    )
    if literal:
        config = TesseractConfig(dem, det_orders=[o for _, orders in sources for o in orders], **settings)
    else:
        # (Generated orders are resolved against each decoder's own model, below.)
        config = TesseractConfig(dem, num_det_orders=1, det_order_method=DetectorOrderMethod.Index, seed=0, **settings)
    mp = None
    if a.multipass:
        from ._multi_pass import MultiPassTesseractDecoder

        mp = MultiPassTesseractDecoder(config, a.num_passes, (), 0 if a.multipass_strategy == "static" else 1)
        decoders = mp.decoders
    else:
        decoders = [TesseractDecoder(config)]
    num_orders = 0
    for decoder in decoders:
        orders = _orders_for(sources, decoder._flat, a.num_orders, a.det_order_seed)
        num_orders = len(orders)
        if not literal:
            decoder.config.det_orders = orders
    if mp is not None and a.print_multipass_plan:
        sys.stdout.flush()
        sys.stderr.write(_plan(mp, a.multipass_strategy, not a.no_merge_errors))
    original = dem.flattened()
    error_use = [0] * original.num_errors
    has_obs = a.has_observables()
    state = {"shots": 0, "errors": 0, "low": 0, "time": 0.0, "precision": 16}

    def consume(k: int, flipped: Sequence[int], errors: Optional[Sequence[int]], low: bool, cost: float, micros: int) -> bool:
        predicted = np.zeros(no, dtype=bool)
        for o in flipped:
            predicted[o] ^= True
        if errors is not None and (not has_obs or np.array_equal(predicted, obs[k])):
            for e in errors:
                error_use[e] += 1
        if writer is not None:
            writer.write(predicted)
        state["shots"] = k + 1
        if low:
            state["low"] += 1
        elif has_obs and not np.array_equal(predicted, obs[k]):
            state["errors"] += 1
        state["time"] += micros / 1e6
        if a.print_stats:
            p = state["precision"]
            sys.stdout.write(
                f"num_shots = {k + 1} num_low_confidence = {state['low']}" + (f" num_errors = {state['errors']}" if has_obs else "")
                + f" total_time_seconds = {_g(state['time'], p)}\ncost = {_g(cost, p)}\n"
            )
        return not has_obs or state["errors"] < a.max_errors

    num_shots = dets.shape[0]
    if a.verbose or mp is not None:
        scheduled = set() if mp is None else {c for components in mp.pass_schedule for c in components}
        for k in range(num_shots):
            hits = np.flatnonzero(dets[k]).tolist()
            start = time.perf_counter()
            if mp is None:
                flipped, errors, low, cost = decoders[0]._decode_result(hits)
            else:
                flipped, low, cost = mp.decode_result(hits)
                errors = None
            micros = int((time.perf_counter() - start) * 1e6)
            # (A verbose search leaves Tesseract's std::cout at precision 13.)
            if a.verbose and hits and (mp is None or any(mp._component_of[d] in scheduled for d in hits)):
                state["precision"] = 13
            if not consume(k, flipped, errors, low, cost, micros):
                break
    else:
        decoder = decoders[0]
        decoder._sync()
        rows = np.zeros((num_shots, decoder.num_detectors), dtype=bool)
        rows[:, : dets.shape[1]] = dets
        packed = np.packbits(rows, axis=1, bitorder="little")
        observables = [e.symptom._observables for e in decoder.errors]
        to_dem = decoder._error_to_dem
        chunk = max(256, 32 * a.num_threads) if has_obs and a.max_errors < _SIZE_MAX else max(num_shots, 1)
        going = True
        for begin in range(0, num_shots, chunk):
            part = np.ascontiguousarray(packed[begin : begin + chunk])
            results = call(decoder._x.decode_shots, part.tobytes(), part.shape[0], a.num_threads)
            for j, (faults, low, cost, micros) in enumerate(results):
                # (The cost is summed as Tesseract's decode_result sums it: from +0, in the
                # order the search found the errors.)
                flipped: set = set()
                for f in faults:
                    flipped ^= set(observables[f])
                if not consume(begin + j, sorted(flipped), [to_dem[f] for f in faults], low, cost, micros):
                    going = False
                    break
            if not going:
                break
    shots = state["shots"]
    if a.dem_out_fname:
        estimate = dem_from_counts(original, error_use, shots - state["errors"] if has_obs else shots)
        try:
            with open(a.dem_out_fname, "w") as f:
                f.write(str(estimate) + "\n")
        except OSError:
            raise _Abort("Failed to open " + a.dem_out_fname) from None
    final = True
    if a.stats_out_fname:
        stats = {
            "circuit_path": a.circuit_path, "dem_path": a.dem_path, "max_errors": a.max_errors, "sample_seed": a.sample_seed, "det_beam": a.det_beam,
            "det_penalty": a.det_penalty, "beam_climbing": a.beam_climbing, "no_revisit_dets": a.no_revisit_dets, "pqlimit": a.pqlimit, "num_det_orders": num_orders,
            "num_det_orders_per_generated_source": a.num_orders, "det_order_seed": a.det_order_seed,
            "detector_orders_paths": [path for method, path in a.sources if method == "literal"], "total_time_seconds": state["time"],
            "num_errors": state["errors"] if has_obs else None, "num_low_confidence": state["low"], "num_shots": shots, "num_threads": a.num_threads,
            "multipass": a.multipass, "multipass_strategy": a.multipass_strategy, "multipass_num_passes": a.num_passes, "sample_num_shots": a.sample_num_shots,
            "sparsify_errors": a.sparsify_errors, "sparsify_base_degree": a.sparsify_base_degree, "sparsify_max_degree": a.sparsify_max_degree,
            "sparsify_reactivate_limit": a.sparsify_reactivate_limit,
        }
        text = _nlohmann_dump(stats) + "\n"
        if a.stats_out_fname == "-":
            sys.stdout.write(text)
            final = False
        else:
            # (As Tesseract's ofstream, a file that cannot be opened is left unwritten.)
            try:
                with open(a.stats_out_fname, "w") as f:
                    f.write(text)
            except OSError:
                pass
    if final:
        sys.stdout.write(
            f"num_shots = {shots} num_low_confidence = {state['low']}" + (f" num_errors = {state['errors']}" if has_obs else "")
            + f" total_time_seconds = {_g(state['time'], state['precision'])}\n"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Tesseract's program on ``argv`` (the command line's arguments by default): its exit
    status."""
    from .common import _redirects

    args = _Args()
    parser = _Parser(args)
    try:
        parser.parse(list(sys.argv[1:] if argv is None else argv))
    except _Exit as e:
        sys.stdout.write(e.text)
        return 0
    except _ParseError as e:
        sys.stderr.write(f"{e}\n{parser.help()}")
        return 1
    # (What Tesseract writes to std::cout goes to sys.stdout, in order with the rest.)
    _redirects[0] += 1
    try:
        _run(args, parser)
    except _Abort as e:
        sys.stdout.flush()
        sys.stderr.write(_terminate(e.kind, str(e)))
        return 134
    except (ValueError, IndexError, OverflowError, UnicodeDecodeError) as e:
        sys.stdout.flush()
        sys.stderr.write(_terminate("std::invalid_argument", str(e)))
        return 134
    except RuntimeError as e:
        sys.stdout.flush()
        sys.stderr.write(_terminate("std::runtime_error", str(e)))
        return 134
    finally:
        _redirects[0] -= 1
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
