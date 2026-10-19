# Ported from stimbposd 0.2.0 (oscarhiggott/stimbposd; Oscar Higgott, Apache License 2.0; see LICENSE
# in this directory and THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: imports point at this
# package's Stim, ldpc and sinter, and this docstring.
"""stimbposd 0.2.0, whole, on this package: ``import stabilizer_qec.stimbposd as stimbposd`` and
stimbposd's code runs unchanged. BP+OSD (``BPOSD``) and BP+LSD (``BPLSD``) on a detector error
model, predicting observables; their sinter decoders (``SinterDecoder_BPOSD``,
``SinterDecoder_BPLSD``, and ``sinter_decoders()``: "bposd", "bposd-serial", "bposd-minsum",
"bposd-serial-minsum" and the same four for "bplsd"); and
``detector_error_model_to_check_matrices``. The decoders are this package's ldpc
(``stabilizer_qec.ldpc``), so the predictions are stimbposd's on ldpc 2.4.1 bit for bit.
With sinter's command line: ``--custom_decoders_module_function stabilizer_qec.stimbposd:sinter_decoders``."""

__version__ = "0.2.0"

from stabilizer_qec.stimbposd.bp_osd import BPOSD  # noqa: F401
from stabilizer_qec.stimbposd.sinter_bp_osd import SinterDecoder_BPOSD, sinter_decoders  # noqa: F401
from stabilizer_qec.stimbposd.dem_to_matrices import detector_error_model_to_check_matrices  # noqa: F401
from stabilizer_qec.stimbposd.bp_lsd import BPLSD  # noqa: F401
from stabilizer_qec.stimbposd.sinter_bp_lsd import SinterDecoder_BPLSD  # noqa: F401
