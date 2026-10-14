# Ported from Tesseract (quantumlib/tesseract-decoder, src/visualization.cc; Copyright 2025 Google
# LLC, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec.
"""Tesseract's search log for its visualizer (``tesseract_decoder.viz``)."""

from __future__ import annotations

from typing import List, Sequence

__all__ = ["Visualizer"]


class Visualizer:
    """The lines a decoder with ``create_visualization`` records: its detectors' coordinates,
    its errors, and each search node's errors and lit detectors."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def _add_detector_coords(self, coords: Sequence[Sequence[float]]) -> None:
        for d, c in enumerate(coords):
            self.lines.append(f"Detector D{d} coordinate (" + ", ".join(f"{x:g}" for x in list(c)[:3]) + ")")

    def _add_errors(self, errors: Sequence[object]) -> None:
        self.lines.extend(str(e) for e in errors)

    def write(self, fpath: str) -> None:
        """Write the lines to ``fpath``."""
        with open(fpath, "w") as f:
            for line in self.lines:
                f.write(line + "\n")
