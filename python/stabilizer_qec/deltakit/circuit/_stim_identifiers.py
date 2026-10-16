# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""Module which provides ways to identify instructions to stim."""

from typing import NamedTuple

from stabilizer_qec import deltakit_stim as stim


class NoiseStimIdentifier(NamedTuple):
    """Collection of information which uniquely identifies a noise channel to
    stim. Differs from the `GateStimIdentifier` in that noise channels can
    have multiple probabilities which need to be taken into account."""

    stim_string: str
    probabilities: tuple[float, ...]
    tag: str | None = None


class AppendArguments(NamedTuple):
    """Collection of items used when appending to a stim circuit. This object
    could be destructured and passed to the `stim.Circuit.append` method.
    """

    stim_string: str
    stim_targets: tuple[stim.GateTarget, ...]
    arguments: tuple[float, ...]
    tag: str | None = None
