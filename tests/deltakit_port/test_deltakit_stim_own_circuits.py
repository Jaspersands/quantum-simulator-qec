"""Deltakit's ``Circuit.from_stim_circuit`` (and ``parse_stim_circuit`` through it) reads this
package's own ``stabilizer_qec.Circuit``, which is the same circuit to Stim, and a pickled
Deltakit circuit, which comes back as one: as it reads the same circuit given as
``stabilizer_qec.deltakit_stim.Circuit`` or as Stim's own, not as an empty circuit."""

import pickle

import pytest

import stabilizer_qec as sq
import stabilizer_qec.deltakit_stim as stim
from stabilizer_qec.deltakit.circuit import Circuit as DeltakitCircuit
from stabilizer_qec.deltakit.decode.utils import parse_stim_circuit


PLAIN = "R 0 1 2\nX_ERROR(0.1) 0 1 2\nCX 0 1\nM 0 1 2\nDETECTOR rec[-1] rec[-2]\nDETECTOR rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]"
LEAKY = "R 0 1\nLEAKAGE(0.1) 0\nCX(0.1, 0.1, 0.1, 0.1) 0 1\nHERALD_LEAKAGE_EVENT 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]"


@pytest.mark.parametrize("text", [PLAIN, LEAKY])
def test_from_stim_circuit_reads_this_packages_circuits(text):
    want = str(DeltakitCircuit.from_stim_circuit(stim.Circuit(text)).as_stim_circuit())
    assert want  # the Deltakit circuit itself reads as itself
    for given in [sq.Circuit(stim.Circuit(text)._stim_exact_text()), pickle.loads(pickle.dumps(stim.Circuit(text)))]:
        assert str(DeltakitCircuit.from_stim_circuit(given).as_stim_circuit()) == want


def test_parse_stim_circuit_reads_this_packages_circuit():
    def shape(c):
        g, logicals, _ = parse_stim_circuit(c)
        return len(g.nodes), len(g.edges), [len(x) for x in logicals]

    assert shape(sq.Circuit(PLAIN)) == shape(stim.Circuit(PLAIN)) == (3, 2, [1])
