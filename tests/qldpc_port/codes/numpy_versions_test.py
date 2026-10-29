"""Logical-operator reduction on every NumPy the port supports (this package's tests, not qLDPC's).

qLDPC writes a reduced logical operator back through ``np.reshape(ops, shape, copy=False)``, whose
``copy`` needs NumPy 2.1 (a TypeError before, so ``reduce_logical_op(s)`` failed on NumPy 1.x and
2.0). The port writes the same entries by indexing. (Its group lifts likewise stacked with
``np.vstack(..., dtype=int)``, which needs NumPy 1.24; the lift tests in ``abstract/groups_test.py``
cover that on an older NumPy.)"""

from __future__ import annotations

import numpy as np
import pytest

from stabilizer_qec.qldpc import codes
from stabilizer_qec.qldpc.objects import Pauli


@pytest.mark.parametrize("order", ["C", "F"])
def test_reduce_logical_ops_writes_through_any_layout(order: str) -> None:
    reference = codes.ToricCode(4)
    ops = np.array(reference.get_logical_ops().view(np.ndarray), order=order)
    code = codes.ToricCode(4)
    code.set_logical_ops(ops)
    assert code.get_logical_ops().flags.f_contiguous == (order == "F")
    reference.reduce_logical_ops()
    assert code.reduce_logical_ops() is code
    assert np.array_equal(code.get_logical_ops().view(np.ndarray), reference.get_logical_ops().view(np.ndarray))
    assert np.all(np.count_nonzero(code.get_logical_ops().view(np.ndarray), axis=1) >= 4)  # the distance


def test_reduce_logical_op_writes_only_its_own_row() -> None:
    code = codes.SurfaceCode(3)
    before = code.get_logical_ops().view(np.ndarray).copy()
    code.reduce_logical_op(Pauli.Z, 0)
    after = code.get_logical_ops().view(np.ndarray)
    n, k = len(code), code.dimension
    assert np.array_equal(np.delete(after, k, axis=0), np.delete(before, k, axis=0))  # Z-type row 0 is row k
    assert not np.any(after[k, :n])  # a Z-type operator has no X part
    assert np.count_nonzero(after[k, n:]) == 3
