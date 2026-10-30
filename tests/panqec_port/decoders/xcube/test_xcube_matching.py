# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import threading

import numpy as np
import pytest
from stabilizer_qec.panqec.codes import XCubeCode
from stabilizer_qec.panqec.decoders import XCubeMatchingDecoder
from stabilizer_qec.panqec.error_models import PauliErrorModel
from panqec_port.decoders.decoder_test import DecoderTest


class TestXCubeMatchingDecoder(DecoderTest):
    @pytest.fixture
    def code(self):
        return XCubeCode(4)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.1
        return XCubeMatchingDecoder(code, error_model, error_rate)


def test_decode_returns_at_a_high_error_rate():
    # Past 0.75 the matching weights of X and Y noise go negative, and the correction can then hold
    # a cycle that the walk over the matched pairs used to circle forever. The decode runs on a
    # thread so that a regression fails the test rather than hanging the run.
    code = XCubeCode(3)
    decoder = XCubeMatchingDecoder(code, PauliErrorModel(1/3, 1/3, 1/3), 0.9)
    error = np.zeros(2 * code.n, dtype='uint8')
    error[0] = 1
    done = {}

    def run():
        done['correction'] = decoder.decode(code.measure_syndrome(error))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive(), 'decode did not return'
    assert done['correction'].shape[0] == 2 * code.n
