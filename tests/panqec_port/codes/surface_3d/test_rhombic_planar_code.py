# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import pytest
from stabilizer_qec.panqec.codes import RhombicPlanarCode
from panqec_port.codes.stabilizer_code_test import StabilizerCodeTest


class TestRhombicPlanarCode(StabilizerCodeTest):

    @pytest.fixture
    def code(self):
        return RhombicPlanarCode(4)

    def test_stabilizer_index(self, code):
        assert all(
            len(index) in [3, 4] for index in code.stabilizer_index
        )
