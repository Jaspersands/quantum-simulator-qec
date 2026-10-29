"""Stim's own objects as this package's, for the qLDPC port's public functions (not in qLDPC).

qLDPC runs on Stim; in the port, ``stim`` is this package, so a real ``stim.Circuit``,
``stim.Tableau`` or ``stim.PauliString`` handed to a qLDPC function failed its ``isinstance``
checks or mixed with this package's objects (``circuit + identity``), and raised. Each public
function that takes one reads it through :func:`ours` first."""

from __future__ import annotations

import sys
from typing import Any

import stabilizer_qec as stim
from stabilizer_qec._util import ours as _ours

__all__ = ["ours"]

_TABLEAU_ARRAYS = ("x2x", "x2z", "z2x", "z2z", "x_signs", "z_signs")


def ours(obj: Any) -> Any:
    """A Stim (or deltakit_stim) ``Circuit``, ``DetectorErrorModel``, ``DemInstruction``,
    ``CircuitInstruction``, ``CircuitRepeatBlock``, ``Tableau`` or ``PauliString`` as this
    package's, exactly; anything else as given."""
    t = type(obj)
    root = t.__module__.split(".")[0]
    if root not in ("stim", "deltakit_stim"):
        return obj
    name = t.__name__
    if name == "Tableau":
        return stim.Tableau.from_numpy(**dict(zip(_TABLEAU_ARRAYS, obj.to_numpy())))
    if name == "PauliString":
        return stim.PauliString(str(obj))  # its text is exact: a sign and one letter per qubit
    if name in ("CircuitInstruction", "CircuitRepeatBlock"):
        circuit = sys.modules[root].Circuit()  # an instruction's text rounds its arguments
        circuit.append(obj)
        return _ours(circuit)[0]
    return _ours(obj)
