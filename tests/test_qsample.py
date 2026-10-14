"""stabilizer_qec.qsample, the port of qsample's dynamical subset sampling: subset sampling
agrees with direct sampling (which is right by construction) at every error rate, on a flagged
GHZ preparation and on a repeat-until-success protocol, its upper bound holds, and its results
agree with qsample's own (recorded from qsample 0.0.2, which needs packages this test
environment lacks)."""

from __future__ import annotations

import numpy as np
import pytest

from stabilizer_qec import qsample as qs
from stabilizer_qec.qsample._graph import simple_eval


def ghz_protocol():
    ghz = qs.Circuit([{"init": {0, 1, 2, 3, 4}}, {"H": {0}}, {"CNOT": {(0, 1)}}, {"CNOT": {(1, 2)}}, {"CNOT": {(2, 3)}}, {"CNOT": {(3, 4)}}, {"CNOT": {(0, 4)}}, {"measure": {4}}])
    meas = qs.Circuit([{"measure": {0, 1, 2, 3}}], noisy=False)
    protocol = qs.Protocol(check_functions={"logErr": lambda m: m not in [0, 0b1111]})
    protocol.add_nodes_from(["ghz", "meas"], circuits=[ghz, meas])
    protocol.add_edge("START", "ghz", check="True")
    protocol.add_edge("ghz", "meas", check="ghz[-1]==0")
    protocol.add_edge("ghz", "FAIL", check="ghz[-1]==1")
    protocol.add_edge("meas", "FAIL", check="logErr(meas[-1])")
    return protocol


def repeat_until_success_protocol():
    # Prepare a Bell pair with a check; on a failed check, try again (at most three tries).
    prep = qs.Circuit([{"init": {0, 1, 2}}, {"H": {0}}, {"CNOT": {(0, 1)}}, {"CNOT": {(0, 2)}}, {"CNOT": {(1, 2)}}, {"measure": {2}}])
    meas = qs.Circuit([{"measure": {0, 1}}], noisy=False)
    protocol = qs.Protocol(check_functions={"odd": lambda m: m in (1, 2)})
    protocol.add_nodes_from(["prep", "meas"], circuits=[prep, meas])
    protocol.add_edge("START", "prep", check="True")
    protocol.add_edge("prep", "prep", check="prep[-1]==1 and len(prep) < 3")
    protocol.add_edge("prep", "FAIL", check="prep[-1]==1 and len(prep) >= 3")
    protocol.add_edge("prep", "meas", check="prep[-1]==0")
    protocol.add_edge("meas", "FAIL", check="odd(meas[-1])")
    return protocol


@pytest.mark.parametrize("make", [ghz_protocol, repeat_until_success_protocol])
def test_subset_sampling_agrees_with_direct_sampling(make):
    np.random.seed(3)
    q = [3e-3, 3e-2, 1e-1]
    direct = qs.DirectSampler(protocol=make(), simulator=qs.StabilizerSimulator, err_model=qs.E1, err_params={"q": q})
    direct.run(6000)
    subset = qs.SubsetSampler(protocol=make(), simulator=qs.StabilizerSimulator, p_max={"q": 0.1}, err_model=qs.E1, err_params={"q": q}, L=3)
    subset.run(3000)
    p_d, std_d = direct.stats()
    p_s, std_s, p_up, _ = subset.stats()
    for a, sa, b, sb, up in zip(p_d, std_d, p_s, std_s, p_up):
        assert abs(a - b) < 4.5 * np.hypot(sa, sb) + 1e-4, (a, b)
        assert up >= b
    # One run gives every rate below p_max: the low rates are far more precise than direct
    # sampling's, which sees few failures there.
    assert std_s[0] < std_d[0] / 3


def test_agrees_with_qsample():
    # qsample 0.0.2 on ghz_protocol, SubsetSampler(p_max 0.1, E1, L=3), 4000 shots, seed 1:
    # p_L and its standard deviation at q = 1e-3, 1e-2, 1e-1.
    theirs = [(0.0036528741743950565, 0.00010052819302458594), (0.03600386033957163, 0.0009621231180243829), (0.3108616793207027, 0.007004398128569192)]
    np.random.seed(1)
    s = qs.SubsetSampler(protocol=ghz_protocol(), simulator=qs.StabilizerSimulator, p_max={"q": 0.1}, err_model=qs.E1, err_params={"q": [1e-3, 1e-2, 1e-1]}, L=3)
    s.run(4000)
    p, std, _, _ = s.stats()
    for (a, sa), b, sb in zip(theirs, p, std):
        assert abs(a - b) < 4.5 * np.hypot(sa, sb)
        assert 0.6 < sb / sa < 1.6


def test_protocol_graph_and_checks():
    p = ghz_protocol()
    assert p.root == "START"
    assert sorted(p.successors("ghz")) == ["FAIL", "meas"]
    assert p.out_degree("FAIL") == 0 and dict(p.in_degree)["START"] == 0
    assert p.successor("START", {})[0] == "ghz"
    assert p.successor("ghz", {"ghz": [1]}) == ("FAIL", None)
    assert p.successor("meas", {"meas": [0b1111]}) == (None, None)
    hist = {"a": [1, 0, 3]}
    names = lambda n: hist.get(n.id)  # noqa: E731
    assert simple_eval("a[-1] == 3 and len(a) > 2", names=names) is True
    assert simple_eval("f(a[0]) + 2 ** 3", names=names, functions={"f": lambda x: x * 10}) == 18
    with pytest.raises(ValueError):
        simple_eval("__import__('os')", names=names)


def test_simulator_gates_are_qsamples():
    from stabilizer_qec import Tableau, TableauSimulator

    sim = qs.StabilizerSimulator(2, seed=1)
    for gate in ("H", "S", "Q", "R", "Sd", "Qd", "Rd", "X", "Y", "Z"):
        getattr(sim, gate)(0)
    sim.MSd(0, 1)
    # Q = H S H, R = Sd Q S as qsample composes them; MSd its five gates.
    ref = TableauSimulator()
    ref.set_num_qubits(2)
    for name in ("h", "s", "h_s_h", "sd_q_s", "s_dag", "q3", "r3", "x", "y", "z"):
        if name == "h_s_h":
            ref.h(0), ref.s(0), ref.h(0)
        elif name == "sd_q_s":
            ref.s_dag(0), ref.h(0), ref.s(0), ref.h(0), ref.s(0)
        elif name == "q3":
            for _ in range(3):
                ref.h(0), ref.s(0), ref.h(0)
        elif name == "r3":
            for _ in range(3):
                ref.s_dag(0), ref.h(0), ref.s(0), ref.h(0), ref.s(0)
        else:
            getattr(ref, name)(0)
    for _ in range(3):  # Rd(0)
        ref.s_dag(0), ref.h(0), ref.s(0), ref.h(0), ref.s(0)
    ref.cx(0, 1)
    ref.s_dag(0), ref.h(0), ref.s(0), ref.h(0), ref.s(0)  # R(0)
    for q in (0, 1):
        for _ in range(3):  # Qd
            ref.h(q), ref.s(q), ref.h(q)
    assert sim._sim.current_inverse_tableau() == ref.current_inverse_tableau()
    with pytest.raises(NotImplementedError):
        sim.T(0)
    assert isinstance(Tableau(1), Tableau)
