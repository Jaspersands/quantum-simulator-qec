# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import pytest
from stabilizer_qec.panqec.codes import Color666PlanarCode
from panqec_port.codes.stabilizer_code_test import StabilizerCodeTest


class TestColor666PlanarCode(StabilizerCodeTest):

    @pytest.fixture(params=[1, 2, 3])
    def code(self, request):
        return Color666PlanarCode(request.param)
