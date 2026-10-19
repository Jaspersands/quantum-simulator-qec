# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code/gidney_fowler_model.py; Copyright 2023 Google LLC,
# Apache License 2.0; see LICENSE in the parent directory), modified for stabilizer_qec; see _models.py.
"""Qualtran's ``qualtran.surface_code.gidney_fowler_model``. The Gidney–Fowler spreadsheet's estimate: its costs, from an error budget and from a grid search over factories and data blocks."""

from ._models import (  # noqa: F401
    AlgorithmSummary,
    CCZ2TFactory,
    DataBlock,
    get_ccz2t_costs,
    get_ccz2t_costs_from_error_budget,
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
    iter_simple_data_blocks,
    MagicStateFactory,
    MultiFactory,
    PhysicalCostModel,
    PhysicalCostsSummary,
    PhysicalParameters,
    QECScheme,
    SimpleDataBlock,
)

__all__ = ["get_ccz2t_costs", "get_ccz2t_costs_from_error_budget", "get_ccz2t_costs_from_grid_search", "iter_ccz2t_factories", "iter_simple_data_blocks"]
