# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import pytest
from stabilizer_qec.panqec.codes import XCubeCode
from stabilizer_qec.panqec.decoders import XCubeMatchingDecoder
from panqec_port.decoders.decoder_test import DecoderTest


class TestXCubeMatchingDecoder(DecoderTest):
    @pytest.fixture
    def code(self):
        return XCubeCode(4)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.1
        return XCubeMatchingDecoder(code, error_model, error_rate)
