# Ported from sinter 1.16.0 (Copyright Google LLC, Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: Stim and PyMatching are this package's, and scipy is not needed.
from typing import Dict
from typing import Union

from stabilizer_qec.sinter._decoding._decoding_decoder_class import Decoder
from stabilizer_qec.sinter._decoding._decoding_fusion_blossom import FusionBlossomDecoder
from stabilizer_qec.sinter._decoding._decoding_pymatching import PyMatchingDecoder
from stabilizer_qec.sinter._decoding._decoding_vacuous import VacuousDecoder
from stabilizer_qec.sinter._decoding._perfectionist_sampler import PerfectionistSampler
from stabilizer_qec.sinter._decoding._sampler import Sampler
from stabilizer_qec.sinter._decoding._decoding_mwpf import HyperUFDecoder, MwpfDecoder
from stabilizer_qec.sinter._legacy import KINDS, KindDecoder

BUILT_IN_DECODERS: Dict[str, Decoder] = {
    'vacuous': VacuousDecoder(),
    'pymatching': PyMatchingDecoder(),
    'pymatching-correlated': PyMatchingDecoder(use_correlated_decoding=True),
    'fusion_blossom': FusionBlossomDecoder(),
    # an implementation of (weighted) hypergraph UF decoder (https://arxiv.org/abs/2103.08049)
    'hypergraph_union_find': HyperUFDecoder(),
    # Minimum-Weight Parity Factor using similar primal-dual method the blossom algorithm (https://pypi.org/project/mwpf/)
    'mw_parity_factor': MwpfDecoder(),
    # This package's decoders, by kind (beside 'pymatching', which is this package's too).
    # Colour matching needs Chromobius's detector annotations: it is `Decoder('color_matching')`.
    **{kind: KindDecoder(kind) for kind in KINDS if kind != 'color_matching'},
}

BUILT_IN_SAMPLERS: Dict[str, Union[Decoder, Sampler]] = {
    **BUILT_IN_DECODERS,
    'perfectionist': PerfectionistSampler(),
}
