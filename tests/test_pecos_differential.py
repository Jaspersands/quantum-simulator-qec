"""stabilizer_qec.pecos against PECOS 0.2.0 itself, from tests/data/pecos_reference.json (PECOS
0.2.0 needs NumPy 1 and networkx 2; tools/pecos_reference.py records it): its Python simulator on
random circuits of every gate, initialization and measurement (forced and on seeds), and PECOS's
C++ one against this package's Rust one; every instruction of every code; the fault-tolerance
checks; error generators and the standard runner on seeds, with MWPM2D's recoveries; logical
error rates. Equal, except where PECOS 0.2.0 raises and the port does not (each checked here),
where networkx 3 breaks a matching's ties otherwise than networkx 2 (an equal-weight recovery
with the same syndrome), and where the port fixes PECOS's C++ 'init |-i>' and TimingRunner's
total_time (each run as PECOS's, or checked, here)."""

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
pytest.importorskip("networkx")

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = json.loads((ROOT / "tests" / "data" / "pecos_reference.json").read_text())

# Run as the reference was: PECOS draws Paulis from sets of strings, in the order of the strings'
# hashes, so on a fixed hash seed (in a fresh interpreter; Python 3.11's string hash on).
# The port's C++ SparseSim prepares |-i> for 'init |-i>'; PECOS's C++ (|1> then H6) prepares |+i>,
# so against the C++'s record it runs the C++'s composition (test_cysparsesim_init_minusi checks the fix).
SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import pecos_reference
from stabilizer_qec import pecos
from stabilizer_qec.pecos.simulators import _cysparsesim as cy


def cpp_init_minusi(sim, qubit, **params):
    sim.initone(qubit)
    sim._state.gate([qubit], cy.IMAGES['H6'])


class CppSparseSim(cy.SparseSim):
    def __init__(self, num_qubits):
        super().__init__(num_qubits)
        self.bindings = dict(self.bindings, **{'init |-i>': cpp_init_minusi})


print(json.dumps(pecos_reference.compute(pecos, CppSparseSim)))
"""


@pytest.fixture(scope="module")
def ours():
    if sys.version_info < (3, 11):
        pytest.skip("the reference's string hashes are Python 3.11's")
    env = dict(os.environ, PYTHONHASHSEED="0")
    out = subprocess.run([sys.executable, "-s", "-c", SCRIPT, str(ROOT / "tools")], env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def raises(o, kind=None):
    return isinstance(o, list) and len(o) == 3 and o[0] == "raises" and (kind is None or o[1] == kind)


def test_simulators(ours):
    # pySparseSim, forced and seeded, and PECOS's C++ SparseSim against this package's Rust one.
    for kind in ("py_forced", "py_seeded", "cy_forced"):
        assert len(REF["sims"][kind]) == 30
        for i, (a, b) in enumerate(zip(REF["sims"][kind], ours["sims"][kind])):
            assert a == b, (kind, i)


# PECOS 0.2.0's medial surface and 4.8.8 colour codes' initializations update a circuit's
# `params`, since renamed `metadata`, and raise; the port's do not.
FIXED_INSTRUCTIONS = {(code, instr) for code in ("SurfaceMedial4444 3", "SurfaceMedial4444 5", "Color488 3", "Color488 5") for instr in ("instr_init_zero", "instr_init_plus")}


def test_codes(ours):
    assert set(REF["codes"]) == set(ours["codes"])
    for name, ref in REF["codes"].items():
        got = ours["codes"][name]
        for key in ("num_qudits", "data", "ancilla", "layout", "gates"):
            assert ref[key] == got[key], (name, key)
        assert set(ref["instructions"]) == set(got["instructions"])
        for symbol, instr in ref["instructions"].items():
            if (name, symbol) in FIXED_INSTRUCTIONS:
                assert raises(instr, "AttributeError") and "params" in instr[2]
                assert isinstance(got["instructions"][symbol], dict)
            else:
                assert instr == got["instructions"][symbol], (name, symbol)
    # PECOS labels the 4.8.8 code's initialized logical Z 'X' (and its X 'Z'); the port does not.
    for d in (3, 5):
        ops = dict(ours["codes"][f"Color488 {d}"]["instructions"]["instr_init_zero"]["final_logical_ops"][0])
        assert ops["Z"][1][0][0][0] == "Z" and ops["X"][1][0][0][0] == "X"


def logical_weight(found):
    xs, zs = re.fullmatch(r"Logical error found: Xs - \((.*)\) Zs - \((.*)\)", found).groups()
    support = {q for part in (xs, zs) for q in part.replace(",", " ").split()}
    return len(support)


def test_fault_tolerance_checks(ours):
    ref, got = REF["ft_checks"], ours["ft_checks"]
    assert set(ref) == set(got)
    for key, a in ref.items():
        b = got[key]
        if key.startswith("distance"):
            # PECOS reads the initialization's `logical_stabs`, which no instruction has, and
            # raises; the port finds the codes' (d = 3) weight-3 logical operators.
            assert raises(a, "AttributeError"), key
            assert logical_weight(b) == 3, (key, b)
        elif key.startswith(("t_errors", "fault")):
            assert raises(a, "AttributeError"), key
            assert b == [True, 1], key  # every weight-1 error corrected
        elif key.startswith("tool collection"):
            # PECOS raises on the decoder's input; the port finds what this one-round check
            # cannot pass, a fault within the round (a measurement's, or a data qudit's).
            assert raises(a), key
            assert raises(b, "Exception") and b[2].startswith("Decoder failed to correct error: {"), key
        else:
            assert a == b, key  # MWPM2D, built or not (on the 4.8.8 code it raises in both)


def test_runs(ours):
    ref, got = REF["runs"], ours["runs"]
    assert set(ref) == set(got)
    for key, a in ref.items():
        b = got[key]
        if key.startswith(("SurfaceMedial4444", "Color488")):
            # PECOS's initializations of these codes raise (see FIXED_INSTRUCTIONS).
            assert raises(a, "AttributeError"), key
            assert len(b) == 4 and all({"output", "errors", "recovery"} <= set(run) for run in b), key
            continue
        assert len(a) == len(b) == 4
        for i, (x, y) in enumerate(zip(a, b)):
            assert x["output"] == y["output"] and x["errors"] == y["errors"], (key, i)
            rx, ry = x["recovery"], y["recovery"]
            if rx == ry:
                continue
            if raises(rx, "TypeError"):
                # PECOS's MWPM2D raises on no syndrome; the port's recovers nothing.
                assert x["output"] == [] and ry == ["circuit", [[]]], (key, i)
            else:
                # The same syndrome, matched otherwise among ties (networkx 3, not 2): a recovery
                # of the same weight with the same syndrome.
                weight = lambda r: sum(len(locs) for _, locs, _ in r[1][0])  # noqa: E731
                assert weight(rx) == weight(ry), (key, i)
                assert x["recovery_syndrome"] == y["recovery_syndrome"], (key, i)


def test_rates(ours):
    ref, got = REF["rates"], ours["rates"]
    for p in ("0.05", "0.1"):
        # (rate, total_time): PECOS's TimingRunner timed no gate, its total_time 0.0; the port's times them.
        (rate_ref, time_ref), (rate_got, time_got) = ref[f"codecapacity {p}"], got[f"codecapacity {p}"]
        assert rate_ref == rate_got and time_ref == 0.0 < time_got
    a, b = dict(ref["pseudo"]), dict(got["pseudo"])
    assert a["distance"] == b["distance"] and a["ps"] == b["ps"]
    # 30 runs a rate, on the same errors; matching's ties (networkx 3, not 2) move a few.
    pa, pb = (list(map(float, d["plog"].strip("[]").split())) for d in (a, b))
    assert all(abs(x - y) <= 0.1 for x, y in zip(pa, pb)), (pa, pb)
