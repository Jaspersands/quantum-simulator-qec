"""stabilizer_qec.qsample where it departs from qsample 0.0.2 on purpose (this package's tests):
each protocol owns its check functions."""

from __future__ import annotations


import numpy as np
import pytest

from stabilizer_qec import qsample as qs


def ghz_protocol(check_functions=None):
    ghz = qs.Circuit([{"init": {0, 1, 2, 3, 4}}, {"H": {0}}, {"CNOT": {(0, 1)}}, {"CNOT": {(1, 2)}}, {"CNOT": {(2, 3)}}, {"CNOT": {(3, 4)}}, {"CNOT": {(0, 4)}}, {"measure": {4}}])
    meas = qs.Circuit([{"measure": {0, 1, 2, 3}}], noisy=False)
    protocol = qs.Protocol(check_functions={"logErr": lambda m: m not in [0, 0b1111]} if check_functions is None else check_functions)
    protocol.add_nodes_from(["ghz", "meas"], circuits=[ghz, meas])
    protocol.add_edge("START", "ghz", check="True")
    protocol.add_edge("ghz", "meas", check="ghz[-1]==0")
    protocol.add_edge("ghz", "FAIL", check="ghz[-1]==1")
    protocol.add_edge("meas", "FAIL", check="logErr(meas[-1])")
    return protocol


# Check functions

def test_protocols_built_without_check_functions_own_theirs():
    # qsample's default is one dict shared by them all: a check function one adds was the other's too.
    a, b = qs.Protocol(), qs.Protocol(fault_tolerant=True)
    assert a.check_functions == {} and a.check_functions is not b.check_functions
    a.check_functions["odd"] = lambda m: m % 2 == 1
    assert "odd" not in b.check_functions
    assert "odd" not in qs.Protocol().check_functions


def test_protocol_keeps_the_check_functions_it_is_given():
    functions = {"logErr": lambda m: m != 0}
    protocol = qs.Protocol(check_functions=functions)
    assert protocol.check_functions is functions  # as qsample: the caller's dict, not a copy
