# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec/python/cudaq_qec/plugins/decoders/example.py;
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates, Apache License 2.0; see
# THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: ``cudaq_qec`` is this package's.
import numpy as np

from stabilizer_qec import cudaq_qec as qec


@qec.decoder("example_byod")
class ExampleDecoder:

    def __init__(self, H, **kwargs):
        qec.Decoder.__init__(self, H)
        self.H = H
        if 'weights' in kwargs:
            print(kwargs['weights'])

    def decode(self, syndrome):
        res = qec.DecoderResult()
        res.converged = True
        res.result = np.random.random(len(syndrome)).tolist()
        res.opt_results = None
        return res
