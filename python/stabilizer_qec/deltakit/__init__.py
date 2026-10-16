"""Deltakit 0.10 (Riverlane's QEC toolkit) on this package: ``deltakit.circuit`` (its circuit
model), ``deltakit.core`` (decoding graphs, data formats, plotting colours), ``deltakit.decode``
(its decoders, noise sources and analysis) and ``deltakit.explorer`` (its codes, experiments,
QPU noise models and native gate sets, analysis, types and plotting), each its own package
whole (deltakit-circuit and -core 0.9.2, -decode 0.9.2, -explorer 0.9.4), on
``stabilizer_qec.deltakit_stim`` (Deltakit's Stim, its leakage instructions native) and this
package's PyMatching. ``stabilizer_qec.deltakit.circuit`` is ``deltakit.circuit`` and so on.

Deltakit's cloud client (``Client``) is its own: it calls Riverlane's server only when given a
token, for what Deltakit does there alone (its stability circuits, its decoders as a service).
Everything else runs here, leakage included: ``explorer.simulation.simulate_with_stim`` samples a
circuit with leakage locally, on Deltakit's leakage model.

The ``deltakit`` extra installs what Deltakit needs beyond this package (Python 3.11).
"""

__version__ = "0.10.2"


def __getattr__(name):
    if name == "Client":
        from .explorer import Client

        return Client
    raise AttributeError(f"module 'stabilizer_qec.deltakit' has no attribute '{name}'")


__all__ = ["Client"]
