# Ported from tqec (tqec/tqec 0.2.0, src/tqec/simulation/io_utils/csv_files.py; Apache License
# 2.0, see LICENSE in the tqec directory), modified for stabilizer_qec: this package's sinter.
"""Write sinter's statistics to CSV files (read them with ``sinter.read_stats_from_csv_files``)."""

from __future__ import annotations

import io
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TextIO, Union

from .... import sinter
from ..._errors import TQECError


def write_sinter_stats_to_csv(filepath: Union[str, Path], stats: Iterable["sinter.TaskStats"], if_file_exists: Literal["merge", "overwrite", "raise"] = "raise") -> None:
    """Write ``stats`` to a CSV file in sinter's format. If the file exists: ``"raise"`` (the
    default) raises a TQECError, ``"overwrite"`` replaces it, ``"merge"`` folds ``stats`` into
    the file's statistics (by strong id, as sinter reads several files)."""
    filepath = Path(filepath)
    if filepath.exists():
        if if_file_exists == "raise":
            raise TQECError(f"File '{filepath}' already exists.")
        elif if_file_exists == "merge":
            in_memory_file = io.StringIO()
            _write_sinter_stats(in_memory_file, stats)
            _ = in_memory_file.seek(0)
            stats = sinter.read_stats_from_csv_files(filepath, in_memory_file)
    with open(filepath, "w") as f:
        _write_sinter_stats(f, stats)


def _write_sinter_stats(file: TextIO, stats: Iterable["sinter.TaskStats"]) -> None:
    _ = file.write(sinter.CSV_HEADER + "\n")
    for sample in stats:
        _ = file.write(sample.to_csv_line() + "\n")
