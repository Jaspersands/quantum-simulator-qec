# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec; the
# decoder's correction no longer carries over from the previous shot (below), so it passes the single-error test.
import numpy as np
import pytest
from stabilizer_qec.panqec.codes import Toric2DCode, Toric3DCode
from stabilizer_qec.panqec.decoders import BeliefPropagationOSDDecoder
from stabilizer_qec.panqec.error_models import PauliErrorModel
from panqec_port.decoders.decoder_test import DecoderTest


class TestBeliefPropagationOSDDecoder(DecoderTest):

    @pytest.fixture
    def code(self):
        return Toric3DCode(4)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.1
        return BeliefPropagationOSDDecoder(code, error_model, error_rate)


def _error(code, xs=(), zs=()):
    error = np.zeros(2 * code.n, dtype=np.uint8)
    for q in xs:
        error[q] = 1
    for q in zs:
        error[code.n + q] = 1
    return error


def _xzzx_toric(size):
    code = Toric2DCode(size)
    code.deform('XZZX')
    return code


@pytest.mark.parametrize('make_code', [Toric2DCode, _xzzx_toric], ids=['css', 'non-css'])
def test_decode_depends_only_on_the_syndrome(make_code):
    # stabilizer_qec: a decoder's correction depends only on the syndrome it is given. PanQEC 0.1.7 read ldpc's
    # osdw_decoding after decode, which ldpc 2 leaves as the previous shot's correction when a syndrome is zero, so
    # an X error then a Z error decoded as X and Z (not explaining the syndrome); this reads decode's return.
    code = make_code(4)
    model = PauliErrorModel(1 / 3, 1 / 3, 1 / 3)
    decoder = BeliefPropagationOSDDecoder(code, model, 0.1)
    rng = np.random.default_rng(5)
    errors = [_error(code, xs=[5]), _error(code, zs=[2]), _error(code)] + [model.generate(code, 0.05, rng=rng) for _ in range(20)]
    for error in errors:
        syndrome = code.measure_syndrome(error)
        correction = decoder.decode(syndrome)
        fresh = BeliefPropagationOSDDecoder(code, model, 0.1).decode(syndrome)
        np.testing.assert_array_equal(correction, fresh)
        np.testing.assert_array_equal(code.measure_syndrome(correction), syndrome)
