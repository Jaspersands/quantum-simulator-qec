# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import pytest
from stabilizer_qec.panqec.codes import Toric2DCode
from stabilizer_qec.panqec.decoders import MemoryBeliefPropagationDecoder
from panqec_port.decoders.decoder_test import DecoderTest


class TestMemoryBeliefPropagationDecoder(DecoderTest):

    @pytest.fixture
    def code(self):
        return Toric2DCode(4)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.1
        return MemoryBeliefPropagationDecoder(code, error_model, error_rate)
