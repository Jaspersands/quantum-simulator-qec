# modified for stabilizer_qec: its Stim and ldpc are this package's, and quits is stabilizer_qec.quits.
import numpy as np

PCM = __import__('os').path.join(__import__('os').path.dirname(__file__), 'parity_check_matrices')
import pytest
import stabilizer_qec as stim
from stabilizer_qec.quits.circuit import check_overlapping_CX
from stabilizer_qec.quits.noise import ErrorModel
from stabilizer_qec.quits.qldpc_code import BbCode, HgpCode, LcsCode, QlpCode


@pytest.mark.parametrize("strategy", ["cardinal", "cardinalNSmerge"])
def test_check_overlapping_cx_hgp_prints_when_verbose(strategy):
    h = np.loadtxt(
        PCM + "/n=12_dv=3_dc=4_dist=6.txt",
        dtype=int,
    )
    code = HgpCode(h, h)
    em = ErrorModel(5e-4, 5e-4, 5e-4, 5e-4)

    circuit = code.build_circuit(
        strategy=strategy,
        error_model=em,
        num_rounds=1,
        basis="Z",
        seed=1,
    )

    overlaps = check_overlapping_CX(circuit, verbose=True)
    print({"overlaps": [(i, d.tolist()) for i, d in overlaps]})

    assert overlaps == []
