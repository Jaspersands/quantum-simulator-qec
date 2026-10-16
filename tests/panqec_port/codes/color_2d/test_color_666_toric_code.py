# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import pytest
from stabilizer_qec.panqec.codes import Color666ToricCode
from panqec_port.codes.stabilizer_code_test import StabilizerCodeTest


class TestColor666ToricCode(StabilizerCodeTest):

    @pytest.fixture(params=[1, 2, 3])
    def code(self, request):
        return Color666ToricCode(request.param)
