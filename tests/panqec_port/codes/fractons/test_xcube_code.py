# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import pytest
from stabilizer_qec.panqec.codes import XCubeCode
from panqec_port.codes.stabilizer_code_test import StabilizerCodeTest


class TestXCubeCode(StabilizerCodeTest):

    @pytest.fixture(params=[2, 3])
    def code(self, request):
        return XCubeCode(request.param)
