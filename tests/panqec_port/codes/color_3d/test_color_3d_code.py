# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import pytest
from stabilizer_qec.panqec.codes import Color3DCode
from panqec_port.codes.stabilizer_code_test import StabilizerCodeTest


class TestColor3DCode(StabilizerCodeTest):

    @pytest.fixture(params=[2, 4])
    def code(self, request):
        return Color3DCode(request.param)
