# Ported from Tesseract (quantumlib/tesseract-decoder; Copyright 2025 Google LLC, Apache License
# 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec.
"""Tesseract's decoders under Tesseract's names (``import stabilizer_qec.tesseract_decoder as
tesseract_decoder``): the A* search decoder and its configuration, the simplex decoder, the
error and model helpers, detector orders, the visualizer log, the sinter decoders and the
multi-pass decoder, with Tesseract's answers."""

import contextlib as _contextlib
import enum as _enum
from typing import Any as _Any
from typing import Iterator as _Iterator

from . import common, simplex, tesseract, tesseract_sinter_compat, utils, viz
from ._multi_pass import _compile_multi_pass_decoder_for_dem, _MultiPassSinterCompiledDecoder
from .tesseract_sinter_compat import TesseractSinterDecoder, make_tesseract_sinter_decoders_dict

# (Tesseract's own tests use the configuration at the top level, though its 2026-10-13 build
# does not export it there.)
TesseractConfig = tesseract.TesseractConfig


class SchedulingStrategy(_enum.IntEnum):
    """How the multi-pass decoder schedules its two components: both in every pass (Static),
    or each pass only those whose answers the next pass needs (Causal)."""

    Static = 0
    Causal = 1

    def __str__(self) -> str:
        return f"SchedulingStrategy.{self.name}"


Static, Causal = SchedulingStrategy.Static, SchedulingStrategy.Causal


@_contextlib.contextmanager
def ostream_redirect(stdout: bool = True, stderr: bool = True) -> _Iterator[None]:
    """Tesseract's context manager for its C++ output; this package's output is Python's
    already, so it changes nothing."""
    yield


def __getattr__(name: str) -> _Any:
    if name == "demutil":
        import importlib

        return importlib.import_module(__name__ + ".demutil")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Causal", "SchedulingStrategy", "Static", "TesseractConfig", "TesseractSinterDecoder", "common", "demutil", "make_tesseract_sinter_decoders_dict",
    "ostream_redirect", "simplex", "tesseract", "tesseract_sinter_compat", "utils", "viz",
]
