# Ported from bposd 2.1 (quantumgizmos/bp_osd; Joschka Roffe, MIT; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec:
# its ldpc is this package's.
from stabilizer_qec.ldpc import bposd_decoder
import os
from stabilizer_qec import bposd as bposd


def get_include():
    path = os.path.dirname(bposd.__file__)
    return path


with open(f"{get_include()}/VERSION", "r") as f:
    __version__ = f.read()
