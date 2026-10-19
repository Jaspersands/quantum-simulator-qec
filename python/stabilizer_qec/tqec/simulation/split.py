# Ported from tqec (tqec/tqec 0.2.0, src/tqec/simulation/split.py; Apache License 2.0, see
# LICENSE in the tqec directory), modified for stabilizer_qec: this package's sinter.
"""Each observable's statistics, from sinter's counts of observable error combinations."""

from __future__ import annotations

import collections
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ... import sinter
from ...sinter._data import ExistingData

if TYPE_CHECKING:
    from ..correlation import CorrelationSurface


def split_counts_for_observables(counts: Mapping[str, int], num_observables: int) -> list[int]:
    """The error count of each observable, from the counts of each combination of observables
    a shot got wrong (sinter's ``count_observable_error_combos=True``)."""
    split_counts: list[int] = [0] * num_observables
    for key, count in counts.items():
        if not key.startswith("obs_mistake_mask="):
            continue
        comb = key.split("=")[1]
        assert num_observables == len(comb)
        for i in range(num_observables):
            if comb[i] == "E":
                split_counts[i] += count
    return split_counts


def split_stats_for_observables(stats: list["sinter.TaskStats"], num_observables: int) -> list[list["sinter.TaskStats"]]:
    """For each observable, every task's statistics with its errors that observable's (the
    statistics of a task are combined first; sinter's ``count_observable_error_combos=True``)."""
    data = ExistingData()
    for s in stats:
        data.add_sample(s)
    combined_stats = list(data.data.values())

    stats_by_observables: list[list[sinter.TaskStats]] = [[] for _ in range(num_observables)]
    for task_stats in combined_stats:
        split_counts = split_counts_for_observables(task_stats.custom_counts, num_observables)
        for obs_idx, count in enumerate(split_counts):
            stats_by_observables[obs_idx].append(
                task_stats.with_edits(
                    errors=count,
                    custom_counts=collections.Counter({k: v for k, v in task_stats.custom_counts.items() if not k.startswith("obs_mistake_mask=")}),
                )
            )
    return stats_by_observables


def heuristic_custom_error_key(observables: list["CorrelationSurface"]) -> str:
    """tqec's ``custom_error_count_key`` for sinter: of the combinations of observables that
    include the one of least area, the one the most fragments of that surface flip (an error
    crossing a fragment flips every observable whose surface holds it), so that sinter samples
    until the observable with the fewest errors has enough."""
    minimal_area_obs = min(observables, key=lambda x: x.area)
    obs_comb_to_prob: dict[str, int] = {}
    for fragment in minimal_area_obs.span:
        obs_comb = ["E" if fragment in obs.span else "_" for obs in observables]
        key = "obs_mistake_mask=" + "".join(obs_comb)
        obs_comb_to_prob[key] = obs_comb_to_prob.get(key, 0) + 1
    return max(obs_comb_to_prob.keys(), key=lambda x: obs_comb_to_prob[x])
