# Ported from Tesseract (quantumlib/tesseract-decoder; Copyright 2025 Google LLC, Apache License
# 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec.
"""Tesseract's decoders under Tesseract's names (``import stabilizer_qec.tesseract_decoder as
tesseract_decoder``): the A* search decoder and its configuration, the simplex decoder, the
error and model helpers, detector orders, the visualizer log, the sinter decoders and the
multi-pass decoder, with Tesseract's answers.

Where it differs from Tesseract:

- Models: each function takes Stim's models or this package's, and gives back the kind it was
  given (Tesseract's always give Stim's); ``Symptom.as_dem_instruction_targets`` gives this
  package's targets.
- Observables: the A* decoder (``TesseractDecoder``, and the sinter and multi-pass decoders
  over it) takes models of at most 64 observables, refusing more with a ``ValueError``;
  Tesseract's takes any number. The simplex decoder and the helpers take any number.
- The simplex decoder solves each integer program by an exact search, not HiGHS (see
  ``simplex``).
- No command line: Tesseract's ``tesseract`` program (``--circuit``, ``--dem``,
  ``--sample-num-shots``, ...) is not ported; its decoders are, through Python and sinter.
- A decoder's configuration is read at every decode, as Tesseract's is, except its sparsifying
  settings (which Tesseract's decoder crashes on when changed after construction): those stay
  as the decoder was made.
- Detector orders are checked when read, compiled or next decoded (Tesseract's decoder reads
  past an order of the wrong size, or crashes, at its next decode; this one raises
  ``ValueError``).
- Refused arguments raise the exception Tesseract's binding raises (pybind11's ``TypeError``
  for an argument of the wrong kind or range), with this package's message."""

from typing import Any as _Any

from . import common, demutil, simplex, tesseract, tesseract_sinter_compat, utils, viz
from ._multi_pass import _compile_multi_pass_decoder_for_dem, _MultiPassSinterCompiledDecoder
from .tesseract_sinter_compat import TesseractSinterDecoder, make_tesseract_sinter_decoders_dict

# (Tesseract's own tests use the configuration at the top level, though its 2026-10-13 build
# does not export it there.)
TesseractConfig = tesseract.TesseractConfig


class SchedulingStrategy(utils._Enum, metaclass=utils._EnumType):
    """How the multi-pass decoder schedules its two components: both in every pass (Static),
    or each pass only those whose answers the next pass needs (Causal)."""

    __slots__ = ()
    _entries = (("Static", 0), ("Causal", 1))


Static, Causal = SchedulingStrategy.Static, SchedulingStrategy.Causal


class ostream_redirect:
    """Tesseract's context manager for its C++ output. Outside one, what Tesseract writes to
    C++'s standard output (a decoder's errors when ``verbose``, the search log of
    ``decode_batch`` and the sinter decoders, the warning about errors that flip no
    detector) goes straight to the process's standard output, past ``sys.stdout``; inside
    one made with ``stdout=True`` it goes to ``sys.stdout``. (``decode``,
    ``decode_to_errors`` and ``decode_from_detection_events`` send theirs to
    ``sys.stdout`` either way, as Tesseract's do. Tesseract writes nothing to C++'s standard
    error, so ``stderr`` changes nothing.)"""

    def __init__(self, stdout: bool = True, stderr: bool = True) -> None:
        self._stdout, self._stderr = bool(stdout), bool(stderr)

    def __enter__(self) -> None:
        if self._stdout:
            common._redirects[0] += 1

    def __exit__(self, *exc: _Any) -> None:
        if self._stdout:
            common._redirects[0] -= 1


def __getattr__(name: str) -> _Any:
    if name == "demutil":
        import importlib

        return importlib.import_module(__name__ + ".demutil")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Causal", "SchedulingStrategy", "Static", "TesseractConfig", "TesseractSinterDecoder", "common", "demutil", "make_tesseract_sinter_decoders_dict",
    "ostream_redirect", "simplex", "tesseract", "tesseract_sinter_compat", "utils", "viz",
]
