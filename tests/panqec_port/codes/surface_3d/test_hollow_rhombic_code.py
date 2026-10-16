# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import pytest
from stabilizer_qec.panqec.codes import HollowRhombicCode
from panqec_port.codes.stabilizer_code_test import StabilizerCodeTest


class TestHollowRhombicCode(StabilizerCodeTest):

    @pytest.fixture(params=[4, 6])
    def code(self, request):
        return HollowRhombicCode(request.param)
