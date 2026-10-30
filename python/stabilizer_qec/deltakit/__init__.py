"""Deltakit 0.10 (Riverlane's QEC toolkit) on this package: ``deltakit.circuit`` (its circuit
model), ``deltakit.core`` (decoding graphs, data formats, plotting colours), ``deltakit.decode``
(its decoders, noise sources and analysis) and ``deltakit.explorer`` (its codes, experiments,
QPU noise models and native gate sets, analysis, types and plotting), each its own package
(deltakit-circuit and -core 0.9.2, -decode 0.9.2, -explorer 0.9.4; the deltakit 0.10.2
distribution also requires deltakit-compile 0.1.0 and deltakit-visualise 0.4.0, which are not
ported), on ``stabilizer_qec.deltakit_stim`` (Deltakit's Stim, its leakage instructions native)
and this package's PyMatching. ``stabilizer_qec.deltakit.circuit`` is ``deltakit.circuit`` and so
on.

Deltakit's cloud client (``Client``) is its own: it calls Riverlane's server only when given a
token, for what Deltakit does there alone (its stability circuits, its decoders as a service).
Everything else runs here, leakage included: ``explorer.simulation.simulate_with_stim`` samples a
circuit with leakage locally, on Deltakit's leakage model.

The ``deltakit`` extra installs what Deltakit needs beyond this package (Python 3.11).
"""

import sys as _sys

if _sys.version_info < (3, 11):
    raise ImportError(
        f"stabilizer_qec.deltakit needs Python 3.11 or later, as Deltakit 0.10 does; this is Python {_sys.version_info[0]}.{_sys.version_info[1]}"
    )

__version__ = "0.10.2"


def __getattr__(name):
    if name == "Client":
        from .explorer import Client

        return Client
    raise AttributeError(f"module 'stabilizer_qec.deltakit' has no attribute '{name}'")


__all__ = ["Client"]
