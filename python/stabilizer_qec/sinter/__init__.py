# Ported from sinter 1.16.0 (Copyright Google LLC, Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: Stim and PyMatching are this package's, and scipy is not needed.
"""sinter (Stim's tool for Monte Carlo sweeps of decoding experiments), whole, on this package.

Everything sinter 1.16 has, by sinter's names: ``Task``, ``collect`` / ``iter_collect`` with
worker processes, resuming from a CSV file and post-selection, ``TaskStats`` and its CSV
format, the fits, the plots, ``predict_*``, and the command line (``python -m
stabilizer_qec.sinter collect|combine|plot|predict``, or ``stabilizer-qec sinter ...``). It
needs no Stim, PyMatching or scipy (matplotlib only to plot). The built-in decoders are
sinter's (``pymatching`` is this package's PyMatching) and this package's (``KINDS``)::

    from stabilizer_qec import sinter

    stats = sinter.collect(tasks=tasks, decoders=["pymatching", "bposd"], num_workers=8, max_shots=100_000)

As in 1.x, ``Decoder(kind, **options)`` and ``Sampler(kind, **options)`` make one of this
package's decoders with options (``Decoder("bposd", osd_order=10)``), and ``decoders()``,
``samplers()`` and ``sinter_decoders()`` name them all, for this sinter or for sinter itself
(``sinter collect --custom_decoders_module_function "stabilizer_qec.sinter:sinter_decoders"``).
"""

# The version of sinter ported.
__version__ = '1.16.0'

from stabilizer_qec.sinter._collection import (
    collect,
    iter_collect,
    post_selection_mask_from_4th_coord,
    Progress,
)
from stabilizer_qec.sinter._data import (
    AnonTaskStats,
    CollectionOptions,
    CSV_HEADER,
    read_stats_from_csv_files,
    stats_from_csv_files,
    Task,
    TaskStats,
)
from stabilizer_qec.sinter._decoding import (
    CompiledDecoder,
    Decoder,
    BUILT_IN_DECODERS,
    BUILT_IN_SAMPLERS,
    Sampler,
    CompiledSampler,
)
from stabilizer_qec.sinter._probability_util import (
    comma_separated_key_values,
    Fit,
    fit_binomial,
    fit_line_slope,
    fit_line_y_at_x,
    log_binomial,
    log_factorial,
    shot_error_rate_to_piece_error_rate,
)
from stabilizer_qec.sinter._plotting import (
    better_sorted_str_terms,
    plot_discard_rate,
    plot_error_rate,
    plot_custom,
    group_by,
)
from stabilizer_qec.sinter._predict import (
    predict_discards_bit_packed,
    predict_observables_bit_packed,
    predict_on_disk,
    predict_observables,
)
from stabilizer_qec.sinter._legacy import (
    KINDS,
    decoders,
    samplers,
    sinter_decoders,
)
