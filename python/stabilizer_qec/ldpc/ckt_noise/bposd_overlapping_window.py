# Ported from ldpc 2.4.1 (Copyright (c) 2022 Joschka Roffe, MIT License; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: imports are this package's (ldpc is stabilizer_qec.ldpc; Stim's API is stabilizer_qec's).
import numpy as np
import stabilizer_qec as stim
from stabilizer_qec.ldpc.ckt_noise.base_overlapping_window_decoder import BaseOverlappingWindowDecoder
from stabilizer_qec.ldpc.bposd_decoder import BpOsdDecoder
from stabilizer_qec.ldpc.ckt_noise.config import (
    DEFAULT_BPOSD_DECODER_ARGS,
)


class BpOsdOverlappingWindowDecoder(BaseOverlappingWindowDecoder):
    def __init__(self, model: stim.DetectorErrorModel, **kwargs):
        self.decoder_config = DEFAULT_BPOSD_DECODER_ARGS | kwargs.pop(
            "decoder_config", {}
        )

        super().__init__(
            model=model,
            **kwargs,
        )

    def _get_dcm(self):
        """
        Set the detector check matrix for the decoder.

        """
        return self.dem_matrices.check_matrix

    def _get_logical_observables_matrix(self):
        """
        Set the logical observables matrix for the decoder.

        """
        return self.dem_matrices.observables_matrix

    @property
    def _min_weight(self):
        """
        Return the minimum weight of the error channel for the decoder.
        """
        return 0.0

    def _get_weights(self):
        """
        Return the weights for the error channel of the decoder obtained from the detector error model.
        """
        return self.dem_matrices.priors

    def _init_decoder(self, round_dcm: np.ndarray, weights: np.ndarray):
        """
        Initialize the decoder for a given round.
        """
        decoder = BpOsdDecoder(
            round_dcm,
            error_channel=list(weights),
            **self.decoder_config,
        )

        return decoder
