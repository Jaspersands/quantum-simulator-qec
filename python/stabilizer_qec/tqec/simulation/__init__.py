"""Simulations of block graphs with sinter, as tqec's ``tqec.simulation`` (tqec 0.2.0, ported;
Apache License 2.0), on this package's compiler and sinter (``stabilizer_qec.sinter``):

- :func:`start_simulation_using_sinter`: the computation at each ``k`` (distance 2k + 1) under
  ``noise_model_factory(p)`` for each ``p``, collected by sinter with each observable's
  statistics split out (``split_stats_for_observables``, ``heuristic_custom_error_key``);
- ``generation``: ``generate_stim_circuits_with_detectors``, ``generate_sinter_tasks``;
- ``threshold``: ``binary_search_threshold``, ``get_logical_error_rate_per_shot``;
- ``plotting``: ``plot_observable_as_inset``, ``plotting.inset.plot_threshold_as_inset``
  (matplotlib);
- ``io_utils.csv_files``: ``write_sinter_stats_to_csv``.

>>> from stabilizer_qec import tqec
>>> from stabilizer_qec.tqec.simulation import start_simulation_using_sinter
>>> stats = start_simulation_using_sinter(  # doctest: +SKIP
...     tqec.gallery.cnot(tqec.Basis.Z), ks=[1, 2], ps=[0.001, 0.002],
...     noise_model_factory=tqec.NoiseModel.uniform_depolarizing, manhattan_radius=2,
...     max_shots=100_000, max_errors=500)
"""

from .plotting import plot_observable_as_inset as plot_observable_as_inset
from .simulation import start_simulation_using_sinter as start_simulation_using_sinter
from .split import heuristic_custom_error_key as heuristic_custom_error_key
from .split import split_stats_for_observables as split_stats_for_observables
