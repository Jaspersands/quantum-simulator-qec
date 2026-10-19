# Ported from tqec (tqec/tqec 0.2.0, src/tqec/simulation/simulation.py; Apache License 2.0, see
# LICENSE in the tqec directory), modified for stabilizer_qec: this package's compiler and sinter.
"""tqec's entry point for simulations: compile, generate the tasks, collect with sinter."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from ... import sinter
from ..compile import FIXED_BULK_CONVENTION, Convention, compile_block_graph
from ..scale import LinearFunction
from .generation import generate_sinter_tasks
from .split import heuristic_custom_error_key, split_stats_for_observables

if TYPE_CHECKING:
    from ..block_graph import BlockGraph
    from ..correlation import CorrelationSurface
    from ..noise_model import NoiseModel

_DEFAULT_BLOCK_REPETITIONS = LinearFunction(2, -1)


def start_simulation_using_sinter(
    block_graph: "BlockGraph",
    ks: Sequence[int],
    ps: Sequence[float],
    noise_model_factory: Callable[[float], "NoiseModel"],
    manhattan_radius: int,
    convention: Convention = FIXED_BULK_CONVENTION,
    observables: Optional[list["CorrelationSurface"]] = None,
    detector_database=None,
    database_path: Union[str, Path, None] = None,
    num_workers: int = multiprocessing.cpu_count(),
    progress_callback: Optional[Callable[["sinter.Progress"], None]] = None,
    max_shots: Optional[int] = None,
    max_errors: Optional[int] = None,
    decoders: Iterable[str] = ("pymatching",),
    print_progress: bool = False,
    custom_decoders: Optional[dict] = None,
    save_resume_filepath: Union[str, Path, None] = None,
    existing_data_filepaths: Iterable[Union[str, Path]] = (),
    split_observable_stats: bool = True,
    block_temporal_height: LinearFunction = _DEFAULT_BLOCK_REPETITIONS,
) -> list[list["sinter.TaskStats"]]:
    """Simulate the computation with sinter, as tqec does: the circuits at every ``k`` in
    ``ks`` (distance 2k + 1) under ``noise_model_factory(p)`` for every ``p`` in ``ps``,
    collected by :func:`stabilizer_qec.sinter.collect` with the other arguments, counting each
    combination of observables a shot gets wrong.

    ``observables`` are the correlation surfaces to measure, every one the graph has when
    ``None`` (here, unlike ``compile_block_graph``, as in tqec). With ``split_observable_stats``
    (the default) the result has one list of statistics per observable, each task's errors those
    of that observable, and sinter samples until the observable likeliest to be short of errors
    has ``max_errors`` (:func:`heuristic_custom_error_key`); otherwise it is ``[stats]``.

    The detectors are the surgery engine's: ``manhattan_radius`` only leaves them out when
    negative, and ``detector_database`` and ``database_path`` are taken for tqec's signature."""
    if observables is None:
        observables = block_graph.find_correlation_surfaces()
    custom_error_count_key: Optional[str] = None
    if split_observable_stats and len(observables) > 1:
        custom_error_count_key = heuristic_custom_error_key(observables)

    compiled_graph = compile_block_graph(block_graph, convention, observables, block_temporal_height)
    stats = sinter.collect(
        num_workers=num_workers,
        tasks=generate_sinter_tasks(compiled_graph, ks, ps, noise_model_factory, manhattan_radius, detector_database, database_path),
        existing_data_filepaths=existing_data_filepaths,
        save_resume_filepath=save_resume_filepath,
        progress_callback=progress_callback,
        max_shots=max_shots,
        max_errors=max_errors,
        decoders=decoders,
        print_progress=print_progress,
        custom_decoders=custom_decoders,
        hint_num_tasks=len(ks) * len(ps),
        count_observable_error_combos=True,
        custom_error_count_key=custom_error_count_key,
    )
    if split_observable_stats:
        return split_stats_for_observables(stats, len(observables))
    return [stats]
