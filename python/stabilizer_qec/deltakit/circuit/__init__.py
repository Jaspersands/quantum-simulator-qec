# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""``deltakit.circuit`` provides classes to represent Stim circuit elements
and functions/methods for interacting with them."""

import importlib.metadata
from typing import TYPE_CHECKING

from stabilizer_qec.deltakit.circuit._annotations._detector import Detector
from stabilizer_qec.deltakit.circuit._annotations._observable import Observable
from stabilizer_qec.deltakit.circuit._annotations._shift_coordinates import ShiftCoordinates
from stabilizer_qec.deltakit.circuit._basic_types import Coord2D, Coord2DDelta
from stabilizer_qec.deltakit.circuit._circuit import Circuit, Layer
from stabilizer_qec.deltakit.circuit._detector_manipulation import trim_detectors
from stabilizer_qec.deltakit.circuit._gate_layer import GateLayer
from stabilizer_qec.deltakit.circuit._noise_factory import (
    GateReplacementPolicy,
    NoiseChannelGen,
    NoiseContext,
    NoiseProfile,
    after_clifford_depolarisation,
    after_reset_flip_probability,
    before_measure_flip_probability,
    measurement_noise_profile,
    noise_profile_with_inverted_noise,
)
from stabilizer_qec.deltakit.circuit._noise_layer import NoiseLayer
from stabilizer_qec.deltakit.circuit._qubit_identifiers import (
    Coordinate,
    InvertiblePauliX,
    InvertiblePauliY,
    InvertiblePauliZ,
    MeasurementPauliProduct,
    MeasurementRecord,
    PauliProduct,
    PauliX,
    PauliY,
    PauliZ,
    Qubit,
    SweepBit,
)

# Recursively import submodules for type checking, linting and code completion
if TYPE_CHECKING:  # pragma: no cover
    from stabilizer_qec.deltakit.circuit import _annotations, gates, noise_channels

__version__ = "0.9.2"

# Prevent import of non-public objects from this module.
del importlib, TYPE_CHECKING

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
