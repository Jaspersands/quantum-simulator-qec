# Ported from tqec (tqec/tqec 0.2.0, src/tqec/simulation/generation.py; Apache License 2.0, see
# LICENSE in the tqec directory), modified for stabilizer_qec: this package's circuits and sinter.
"""The circuits and sinter tasks of a compiled computation at each scale and noise strength."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from ... import sinter

if TYPE_CHECKING:
    from ..compile import TopologicalComputationGraph
    from ..noise_model import NoiseModel


def generate_stim_circuits_with_detectors(
    compiled_graph: "TopologicalComputationGraph",
    ks: Iterable[int],
    ps: Iterable[float],
    noise_model_factory: Callable[[float], "NoiseModel"],
    manhattan_radius: int,
    detector_database=None,
    database_path: Union[str, Path, None] = None,
    do_not_use_database: bool = False,
    only_use_database: bool = False,
) -> Iterator[tuple]:
    """``(circuit, k, p)`` for every ``k`` and ``p``: the computation's noiseless circuit at
    each ``k`` (distance 2k + 1) with ``noise_model_factory(p)`` laid over it, the circuits
    grouped by ``k`` as tqec yields them.

    The detectors are the surgery engine's, so ``manhattan_radius`` only leaves them out when
    negative, and the detector database arguments are taken for tqec's signature."""
    noise_models = {p: noise_model_factory(p) for p in ps}
    circuits = {
        k: compiled_graph.generate_stim_circuit(
            k,
            manhattan_radius=manhattan_radius,
            detector_database=detector_database,
            database_path=database_path,
            do_not_use_database=do_not_use_database,
            only_use_database=only_use_database,
        )
        for k in ks
    }
    yield from ((nm.noisy_circuit(circuit), k, p) for k, circuit in circuits.items() for p, nm in noise_models.items())


def generate_sinter_tasks(
    compiled_graph: "TopologicalComputationGraph",
    ks: Iterable[int],
    ps: Iterable[float],
    noise_model_factory: Callable[[float], "NoiseModel"],
    manhattan_radius: int,
    detector_database=None,
    database_path: Union[str, Path, None] = None,
    do_not_use_database: bool = False,
    only_use_database: bool = False,
) -> Iterator["sinter.Task"]:
    """A ``sinter.Task`` for each circuit of :func:`generate_stim_circuits_with_detectors`,
    with tqec's metadata ``{"d": 2k + 1, "r": 2k + 1, "p": p}``."""
    yield from (
        sinter.Task(circuit=circuit, json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p})
        for circuit, k, p in generate_stim_circuits_with_detectors(
            compiled_graph,
            ks,
            ps,
            noise_model_factory,
            manhattan_radius,
            detector_database,
            database_path=database_path,
            do_not_use_database=do_not_use_database,
            only_use_database=only_use_database,
        )
    )
