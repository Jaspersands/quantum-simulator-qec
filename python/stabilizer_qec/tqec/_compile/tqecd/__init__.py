# Ported from tqecd (tqec/tqecd 0.2.1, src/tqecd/__init__.py; Apache License 2.0, see LICENSE here),
# modified for stabilizer_qec.
"""Automatic computation of detectors in quantum circuits representing fault-tolerant
quantum computations based on surface code and lattice surgery.

This package provides a set of functions and data-structures to help anyone
computing automatically the detectors (i.e., sets of deterministic measurements)
contained in a quantum circuit.
"""

from __future__ import annotations

# tqecd's construction module (annotate_detectors_automatically) is not part of the port: tqec's
# compile pipeline calls the functions below it directly.
