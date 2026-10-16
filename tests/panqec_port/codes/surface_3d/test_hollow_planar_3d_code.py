# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import pytest
from stabilizer_qec.panqec.codes import HollowPlanar3DCode
from panqec_port.codes.stabilizer_code_test import StabilizerCodeTest


class TestHollowPlanar3DCode(StabilizerCodeTest):

    @pytest.fixture(params=[(2, 2, 2), (3, 3, 3), (2, 3, 4)])
    def code(self, request):
        return HollowPlanar3DCode(*request.param)
