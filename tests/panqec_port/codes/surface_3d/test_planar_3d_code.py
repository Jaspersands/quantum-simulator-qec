# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
import pytest
from stabilizer_qec.panqec.codes import Planar3DCode
from panqec_port.codes.stabilizer_code_test import StabilizerCodeTest


class TestPlanar3DCode(StabilizerCodeTest):

    @pytest.fixture
    def code(self):
        return Planar3DCode(4)
