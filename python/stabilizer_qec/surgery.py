"""Lattice surgery on rotated surface-code patches (Horsman, Fowler, Devitt and Van Meter
2012), each operation written as one circuit with SD6 noise at ``p``: patches prepared, held,
merged across a seam for ``merged`` rounds, split, held and read out. Every function returns
a ``Circuit`` whose error model equals Stim's for the same text."""

from __future__ import annotations

from typing import Optional

from . import _core
from ._circuit import Circuit
from ._util import call, count, probability

__all__ = ["zz_measurement", "xx_measurement", "cnot", "repeated_zz", "line"]


def _basis(basis: str) -> str:
    if basis not in ("z", "x"):
        raise ValueError(f"basis must be 'z' or 'x', not {basis!r}")
    return basis


def zz_measurement(
    distance: int, *, merged: int, p: float, basis: str = "z", pre: Optional[int] = None, post: Optional[int] = None
) -> Circuit:
    """Z⊗Z on two patches side by side, merged across a column of seam qubits. With
    ``basis="z"`` both start in |0⟩ and the observables are the outcome Z₁Z₂ (L0), Z₁ and Z₂;
    with ``"x"``, both in |+⟩ and the observable is X₁X₂, which the measurement keeps.
    ``pre`` and ``post`` rounds apart default to ``distance``."""
    pre = distance if pre is None else pre
    post = distance if post is None else post
    return Circuit(
        call(_core.surgery_circuit, count(distance, "distance"), count(merged, "merged"), probability(p), _basis(basis), count(pre, "pre"), count(post, "post"))
    )


def xx_measurement(distance: int, *, merged: int, p: float, basis: str = "x") -> Circuit:
    """X⊗X, the mirror of ``zz_measurement``: two patches one above the other, merged across a
    row of seam qubits. ``"x"``: the outcome and each patch's X; ``"z"``: Z₁Z₂."""
    return Circuit(call(_core.surgery_vertical, count(distance, "distance"), count(merged, "merged"), probability(p), _basis(basis)))


def cnot(distance: int, *, merged: int, p: float, inputs: str = "z") -> Circuit:
    """A logical CNOT by surgery: control, an ancilla in |+⟩ and target in an L; Z_C Z_A, then
    X_A X_T, then the ancilla read in Z. ``inputs="z"``: |0⟩|0⟩, observables Z_C and Z_T with
    its Pauli frame; ``"x"``: |+⟩|+⟩, observables X_T and X_C X_T with its frame."""
    return Circuit(call(_core.surgery_cnot, count(distance, "distance"), count(merged, "merged"), probability(p), _basis(inputs)))


def repeated_zz(distance: int, *, k: int, merged: int, p: float) -> Circuit:
    """``k`` Z⊗Z measurements in a row on two patches in |0⟩|0⟩, each of ``merged`` rounds and
    followed by a round apart. Observables: each outcome, then Z₁ and Z₂."""
    return Circuit(call(_core.surgery_repeated, count(distance, "distance"), count(k, "k"), count(merged, "merged"), probability(p)))


def line(distance: int, *, n: int, merged: int, p: float) -> Circuit:
    """``n`` patches in a row, all in |0⟩, merged at once: each neighbouring pair's Z⊗Z, n − 1
    parities (not the n-body product). Observables: each seam's outcome, then each patch's Z."""
    return Circuit(call(_core.surgery_line, count(distance, "distance"), count(n, "n"), count(merged, "merged"), probability(p)))
