# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code; Copyright 2023 Google LLC, Apache
# License 2.0; see LICENSE in the parent directory), modified for stabilizer_qec; see _models.py.
"""Qualtran's surface-code cost models (``qualtran.surface_code`` 0.7.0): a QEC scheme (logical error
rate from distance), physical parameters, data blocks (Gidney–Fowler's simple block; Litinski's
compact, intermediate and fast blocks), magic-state factories (Gidney–Fowler's catalyzed CCZ-to-2T
factory, Litinski's 15-to-1 with its density-matrix error model and its channels, several factories
at once), rotation cost models, the `PhysicalCostModel` that combines them, and the Gidney–Fowler
spreadsheet's and Beverland et al.'s estimates.

Every name is here, as in ``qualtran.surface_code``, and in the submodule Qualtran keeps it in, under
the same submodule names: ``algorithm_summary``, ``beverland_et_al_model``, ``ccz2t_factory``,
``data_block``, ``fifteen_to_one_factory``, ``gidney_fowler_model``, ``magic_state_factory``,
``multi_factory``, ``physical_cost_model``, ``physical_cost_summary``, ``physical_parameters``,
``qec_scheme``, ``rotation_cost_model``, ``t_factory_utils`` and ``ui`` (the Dash app's estimates and
figures without Dash; imported on its own, as Qualtran's is). ``GateCounts`` (Qualtran's
``qualtran.resource_counting``) and ``TComplexity`` (``qualtran.cirq_interop``) are here too.

>>> from stabilizer_qec.estimate.qualtran.gidney_fowler_model import get_ccz2t_costs_from_grid_search
>>> from stabilizer_qec.estimate.qualtran import GateCounts
>>> cost, factory, data_block = get_ccz2t_costs_from_grid_search(n_logical_gates=GateCounts(t=10**8, toffoli=10**8), n_algo_qubits=100)
>>> factory, data_block, cost.footprint
(CCZ2TFactory(distillation_l1_d=15, distillation_l2_d=25), SimpleDataBlock(data_d=29, routing_overhead=0.5), 378700)
"""

from . import _models

# Every name of the implementation, its private helpers too (``from stabilizer_qec.estimate.qualtran import
# _plain_count`` worked when this was one module).
globals().update({_k: _v for _k, _v in vars(_models).items() if not _k.startswith("__")})

# Qualtran's submodules, after the names they re-export; ``ui``, as Qualtran's, is imported on its own.
from . import (  # noqa: E402
    algorithm_summary,
    beverland_et_al_model,
    ccz2t_factory,
    data_block,
    fifteen_to_one_factory,
    gidney_fowler_model,
    magic_state_factory,
    multi_factory,
    physical_cost_model,
    physical_cost_summary,
    physical_parameters,
    qec_scheme,
    rotation_cost_model,
    t_factory_utils,
)

__all__ = sorted(_k for _k, _v in vars(_models).items() if not _k.startswith("_") and getattr(_v, "__module__", None) == _models.__name__)
