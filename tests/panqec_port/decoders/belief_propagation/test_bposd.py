# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import pytest
from stabilizer_qec.panqec.codes import Toric3DCode
from stabilizer_qec.panqec.decoders import BeliefPropagationOSDDecoder
from panqec_port.decoders.decoder_test import DecoderTest


class TestBeliefPropagationOSDDecoder(DecoderTest):

    @pytest.fixture
    def code(self):
        return Toric3DCode(4)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.1
        return BeliefPropagationOSDDecoder(code, error_model, error_rate)

    # stabilizer_qec: PanQEC 0.1.7 fails this (slow) test on ldpc 2 as well (on ldpc 2.4.1 its
    # decoder returns no correction for a single Z on the d = 4 3D toric code); this port, on this
    # package's ldpc, does the same.
    @pytest.mark.slow
    @pytest.mark.xfail(reason="PanQEC 0.1.7 fails it on ldpc 2 too", strict=False)
    def test_decode_single_qubit_error(self, code, decoder, allowed_paulis):
        super().test_decode_single_qubit_error(code, decoder, allowed_paulis)
