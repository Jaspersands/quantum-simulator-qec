# Ported from ldpc 2.4.1 (Copyright (c) 2022 Joschka Roffe, MIT License; see THIRD_PARTY_NOTICES.md).
"""ldpc 2.4.1, whole, on this package: ``import stabilizer_qec.ldpc as ldpc`` and ldpc's code runs
unchanged. The decoders are this package's (bit for bit ldpc's, `stabilizer_qec._ldpc`); its
pure-Python modules (`code_util`, `codes`, `alist`, `protograph`, `noise_models`,
`monte_carlo_simulation`, `ckt_noise`, `sinter_decoders`) are ported, with Stim, PyMatching and
sinter replaced by this package's; and `mod2`, Cython and C++ in ldpc, is reimplemented in numpy
with ldpc's answers. It needs scipy, as ldpc does."""

__version__ = "2.4.1"  # the ldpc release this follows

from stabilizer_qec.ldpc.bp_decoder import BpDecoder, SoftInfoBpDecoder, bp_decoder  # noqa: E402
from stabilizer_qec.ldpc.bposd_decoder import BpOsdDecoder, bposd_decoder  # noqa: E402
from stabilizer_qec.ldpc.bplsd_decoder import BpLsdDecoder  # noqa: E402
from stabilizer_qec.ldpc.belief_find_decoder import BeliefFindDecoder  # noqa: E402
from stabilizer_qec.ldpc.union_find_decoder import UnionFindDecoder  # noqa: E402
from stabilizer_qec.ldpc.sinter_decoders import SinterBpOsdDecoder  # noqa: E402

__all__ = ["BpDecoder", "BpOsdDecoder", "BpLsdDecoder", "SoftInfoBpDecoder", "BeliefFindDecoder", "SinterBpOsdDecoder", "UnionFindDecoder", "bp_decoder", "bposd_decoder"]
