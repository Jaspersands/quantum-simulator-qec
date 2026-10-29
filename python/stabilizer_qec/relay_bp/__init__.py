"""IBM's relay_bp 0.2.2, whole, on this package: ``import stabilizer_qec.relay_bp as relay_bp`` and
relay_bp's code runs unchanged. Relay-BP and min-sum BP in every number type relay_bp is built for
(``RelayDecoderF32``/``F64``/``I32``/``I64``, ``MinSumBPDecoderF32``/``F64``/``I8``/``I16``/
``I32``/``I64``/``Fixed``), ``DecodeResult``, ``ObservableDecoderRunner`` and, in
``relay_bp.stim``, its sinter decoders and noise tools, with the same results bit for bit.

Where relay_bp fails, this answers: a sparse check matrix storing an entry twice (relay_bp
panics) is the matrix it sums to, mod 2, so a check stored twice is an edge that sets nothing
off; and the sinter decoders' pruning of a fault of probability 1 (relay_bp's builds its biases
as an ``np.matrix``, which its decoders refuse, and the observables' from the check matrix)
flips its detectors and observables. Calls on one decoder from several threads take turns, as
relay_bp's do under the GIL."""

from .bp import *  # noqa: F401,F403
from .decoder import *  # noqa: F401,F403
from .observable_decoder import *  # noqa: F401,F403

__version__ = "0.2.2"  # the relay_bp release this follows
