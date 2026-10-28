#  ==================================================================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
#   DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
#   the License. You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
#   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
#   specific language governing permissions and limitations under the License.
#  =========================================================================  #
# modified for stabilizer_qec: pecos is stabilizer_qec.pecos.

"""
Performance Estimator of Codes On Surfaces (PECOS)
==================================================

A framework for developing, studying, and evaluating quantum error-correcting codes.

(stabilizer_qec: PECOS 0.2.0, ported. Where PECOS has a bug, the port fixes it; each fix is marked where it is made:

* ``simulators.SparseSim`` (PECOS's C++ simulator, here in Rust): ``'init |-i>'`` prepares :math:`|-i\\rangle` (the
  C++ prepared :math:`|+i\\rangle`); it has the Python ``SparseSim``'s ``print_stabs(print_destabs=..., print_y=...)``,
  ``copy``, ``refactor``, ``find_stab``, ``run_direct`` and ``'G2'``, which PECOS's docs use on it.
* ``simulators.pySparseSim``: ``print_y`` prints Y's sign right (PECOS's printed ``-Y`` for :math:`|+i\\rangle`);
  ``copy`` returns the copy; ``refactor`` works with neither ``choose`` nor ``prefer``, and keeps the stabilizer
  group's signs (PECOS's dropped the -1s of reordering the generators it multiplies, flipping the new generator's
  sign in about one refactor in five; ``SparseSim.refactor`` is this one too); a random measurement's
  ``forced_outcome=-1``, its gates' default, draws the outcome.
* ``error_gens``: ``model_level='phenomenological'`` works (PECOS raised KeyError); ``GatewiseGen`` takes a set mixing
  gates and tuples, e.g. ``{('I', 'X'), ('X', 'X'), 'CNOT'}``, on NumPy 1.24 and later.
* ``circuit_runners.TimingRunner``: ``total_time`` times ``run``'s gates (PECOS's stayed 0.0).
* ``tools.fault_tolerance_check`` decodes an ideal round after each faulty one, and passes the shipped codes;
  ``fault_tolerance_checks``' ``distance_check``, ``t_errors_check`` and ``fault_check`` run (PECOS's raised).
* ``qeccs``: the medial surface and 4.8.8 colour codes' initializations run; instructions' ``plot`` draws on
  networkx 3.)
"""

# from __future__ import annotations

from . import (
    version,
    types,
)

from .misc.symbol_library import SymbolLibrary

# PECOS namespaces
from . import (
    circuits,
    qeccs,
    simulators,
    error_gens,
    circuit_runners,
    decoders,
    circuit_converters,
    misc,
    tools,
)

__version__ = version.__version__
__date__ = ''
__author__ = version.__author__
__copyright__ = version.__copyright__
