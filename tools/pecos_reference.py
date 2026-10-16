"""PECOS 0.2.0's results, recorded for tests/test_pecos_differential.py.

PECOS 0.2.0 needs NumPy 1 and networkx 2 (and so Python 3.12 or older), which this package's own
tests do not run beside; so its results are recorded here, run against PECOS itself, and the
test runs the same `compute` against `stabilizer_qec.pecos` and compares.

    PYTHONHASHSEED=0 python3.12 -s -P tools/pecos_reference.py --cysparsesim DIR
    # with PECOS 0.2.0, NumPy 1.26 and networkx 2.8.8 installed

PECOS draws Paulis from sets of strings, whose order is the string hashes', so both sides run
with PYTHONHASHSEED=0 on Python 3.11 or later (whose string hash this is; not under -I or -E,
which ignore it).

`--cysparsesim` names the directory holding PECOS's compiled C++ simulator (its
`_cysparsesim/src`, built by Cython), whose results are recorded too. Writes
tests/data/pecos_reference.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "pecos_reference.json"

ONE = ['I', 'X', 'Y', 'Z', 'Q', 'Qd', 'R', 'Rd', 'S', 'Sd', 'H', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H+z+x', 'H-z-x',
       'H+y-z', 'H-y-z', 'H-x+y', 'H-x-y', 'F1', 'F2', 'F3', 'F4', 'F1d', 'F2d', 'F3d', 'F4d']
TWO = ['II', 'CNOT', 'CZ', 'CY', 'SWAP', 'G', 'SqrtXX', 'MS', 'MSXX']
INIT = ['init |0>', 'init |1>', 'init |+>', 'init |->', 'init |+i>', 'init |-i>']
MEAS = ['measure X', 'measure Y', 'measure Z']


def ser(o):
    """A JSON form, the same for equal values in either package (sets sorted, floats rounded)."""
    import numpy as np

    if isinstance(o, dict):
        return sorted(([ser(k), ser(v)] for k, v in o.items()), key=repr)
    if isinstance(o, (set, frozenset)):
        return sorted((ser(x) for x in o), key=repr)
    if isinstance(o, (list, tuple)):
        return [ser(x) for x in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return round(float(o), 10)
    if o is None or isinstance(o, str):
        return o
    if hasattr(o, "iter_ticks") and hasattr(o, "items"):  # a QuantumCircuit
        return ["circuit", [ser(list(o.items(tick=t))) for t in range(len(o))]]
    return str(o)


def attempt(fn, raw=False):
    """`fn()` (serialized unless `raw`), or how it raises: PECOS's own failures are recorded, to
    be the same."""
    import contextlib
    import io

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            got = fn()
    except Exception as e:  # noqa: BLE001
        return ["raises", type(e).__name__, str(e)]
    return got if raw else ser(got)


def failed(o):
    """Whether `o` is `attempt`'s record of a raise."""
    return isinstance(o, list) and len(o) == 3 and o[0] == "raises"


def decode(decoder, output):
    """The decoder's recovery, or the raise that made the decoder (a code MWPM2D does not decode)."""
    return decoder if failed(decoder) else attempt(lambda: decoder.decode(output))


def canonical(state):
    """The stabilizer group in reduced row-echelon form, each generator (phase, x, z) for
    i^phase X^x Z^z (X before Z on each qubit), whichever simulator holds it."""
    n = state.num_qubits
    if isinstance(state.stabs, dict):  # PECOS's C++ simulator (and this package's Rust one)
        rows_x, rows_z, minus, ii = state.stabs['row_x'], state.stabs['row_z'], state.signs_minus, state.signs_i
    else:
        rows_x, rows_z, minus, ii = state.stabs.row_x, state.stabs.row_z, state.stabs.signs_minus, state.stabs.signs_i
    gens = []
    for g in range(n):
        x = sum(1 << q for q in rows_x[g])
        z = sum(1 << q for q in rows_z[g])
        gens.append([(g in ii) + 2 * (g in minus), x, z])

    def times(a, b):
        return [(a[0] + b[0] + 2 * bin(a[2] & b[1]).count("1")) % 4, a[1] ^ b[1], a[2] ^ b[2]]

    r = 0
    for bit in [(0, q) for q in range(n)] + [(1, q) for q in range(n)]:
        part, q = 1 + bit[0], bit[1]
        piv = next((i for i in range(r, n) if gens[i][part] >> q & 1), None)
        if piv is None:
            continue
        gens[r], gens[piv] = gens[piv], gens[r]
        for i in range(n):
            if i != r and gens[i][part] >> q & 1:
                gens[i] = times(gens[i], gens[r])
        r += 1
    return gens


def random_ops(seed, forced):
    rng = random.Random(seed)
    n = rng.randint(2, 7)
    ops = []
    for _ in range(60):
        kind = rng.random()
        if kind < 0.45:
            ops.append([rng.choice(ONE), rng.randrange(n), None])
        elif kind < 0.75 and n > 1:
            a, b = rng.sample(range(n), 2)
            ops.append([rng.choice(TWO), [a, b], None])
        elif kind < 0.85:
            ops.append([rng.choice(INIT), rng.randrange(n), None])
        else:
            ops.append([rng.choice(MEAS), rng.randrange(n), rng.randrange(2) if forced else "random"])
    return n, ops


def run_ops(sim_class, n, ops):
    state = sim_class(n)
    outcomes = []
    for sym, loc, forced in ops:
        loc = tuple(loc) if isinstance(loc, list) else loc
        if forced is None:
            state.run_gate(sym, {loc})
        else:
            # (The Python SparseSim draws an outcome for forced_outcome=None; its default, -1, raises.)
            out = state.run_gate(sym, {loc}, forced_outcome=None if forced == "random" else forced)
            outcomes.append(int(out.get(loc, 0)))
    return {"outcomes": outcomes, "stabilizers": canonical(state)}


def compute(pc, cysim=None):
    """Everything compared, from the PECOS package `pc` (PECOS or stabilizer_qec.pecos), with
    `cysim`, the C++ (or Rust) SparseSim class, if any."""
    import numpy as np

    out = {}

    # Simulators: random circuits of every gate, initialization and measurement, outcomes forced
    # (the C++'s random ones come from rand()) and, for the Python one, drawn from NumPy's
    # generator on a seed.
    sims = {"py_forced": [], "py_seeded": [], "cy_forced": []}
    for seed in range(30):
        n, ops = random_ops(seed, forced=True)
        sims["py_forced"].append(run_ops(pc.simulators.pySparseSim, n, ops))
        if cysim is not None:
            sims["cy_forced"].append(run_ops(cysim, n, ops))
        n, ops = random_ops(1000 + seed, forced=False)
        np.random.seed(seed)
        sims["py_seeded"].append(run_ops(pc.simulators.pySparseSim, n, ops))
    out["sims"] = sims

    # Codes: their qudits, layout, and every instruction's circuit and logical operators.
    codes = {}
    for cls in (pc.qeccs.Surface4444, pc.qeccs.SurfaceMedial4444, pc.qeccs.Color488):
        for d in (3, 5):
            q = cls(distance=d)
            entry = {
                "num_qudits": q.num_qudits,
                "data": ser(q.data_qudit_set),
                "ancilla": ser(q.ancilla_qudit_set),
                "layout": ser(q.layout),
                "gates": ser(sorted(q.available_gates)),
                "instructions": {},
            }
            for symbol in sorted(q.available_instructions):
                try:
                    instr = q.instruction(symbol)
                except Exception as e:  # noqa: BLE001 - PECOS's own failures, recorded to be the same
                    entry["instructions"][symbol] = ["raises", type(e).__name__, str(e)]
                    continue
                entry["instructions"][symbol] = {
                    "circuit": ser(instr.circuit),
                    "abstract": ser(instr.abstract_circuit),
                    "logical_stabs": ser(getattr(instr, "logical_stabs", None)),
                    "final_logical_ops": ser(getattr(instr, "final_logical_ops", None)),
                    "initial_logical_ops": ser(getattr(instr, "initial_logical_ops", None)),
                }
            codes[f"{cls.__name__} {d}"] = entry
    out["codes"] = codes

    # Fault-tolerance checks.
    ft = {}
    checks = pc.tools.fault_tolerance_checks
    for cls in (pc.qeccs.Surface4444, pc.qeccs.SurfaceMedial4444, pc.qeccs.Color488):
        q = cls(distance=3)
        ft[f"distance {cls.__name__}"] = attempt(lambda: checks.distance_check(q))
        for mode in ("x", "z", "power"):
            ft[f"distance {mode} {cls.__name__}"] = attempt(lambda: checks.distance_check(q, mode=mode))
        decoder = attempt(lambda: pc.decoders.MWPM2D(q), raw=True)
        ft[f"decoder {cls.__name__}"] = decoder if failed(decoder) else "MWPM2D"
        if failed(decoder):
            continue
        for name, check in (("t_errors", checks.t_errors_check), ("fault", checks.fault_check)):
            ft[f"{name} {cls.__name__}"] = attempt(lambda: check(q, decoder=decoder, verbose=False))
        ft[f"t_errors ancillas {cls.__name__}"] = attempt(lambda: checks.t_errors_check(q, decoder=decoder, verbose=False, ancilla_errors=True))
        ft[f"tool collection {cls.__name__}"] = attempt(lambda: pc.tools.fault_tolerance_check(q, decoder))
    out["ft_checks"] = ft

    # Error generators and the standard runner, on seeds; MWPM2D's recoveries of the results.
    def code_runs(cls_name, name, gen):
        q = getattr(pc.qeccs, cls_name)(distance=3)
        decoder = attempt(lambda: pc.decoders.MWPM2D(q), raw=True)
        syn = pc.circuits.LogicalCircuit(suppress_warning=True)
        syn.append(q.gate('I', num_syn_extract=1))  # (MWPM2D decodes one round)
        init = pc.circuits.LogicalCircuit(suppress_warning=True)
        init.append(q.gate('ideal init |0>'))
        got, recoveries = [], []
        runner = pc.circuit_runners.Standard(seed=7)
        for _ in range(4):
            state = pc.simulators.pySparseSim(q.num_qudits)
            runner.run(state, init)
            output, errors = runner.run(state, syn, error_gen=gen, error_params={'p': 0.05})
            got.append({"output": ser(dict(output)), "errors": ser(errors), "recovery": decode(decoder, output)})
            recoveries.append(None if failed(decoder) else attempt(lambda: decoder.decode(output), raw=True))
        # Each recovery's own syndrome (from the code's state, after the runs: their draws stay
        # the same), by which recoveries matching chose among ties are compared.
        for entry, recovery in zip(got, recoveries):
            if recovery is not None and not failed(recovery):
                state = pc.simulators.pySparseSim(q.num_qudits)
                runner.run(state, init)
                runner.run(state, recovery)
                output, _ = runner.run(state, syn)
                entry["recovery_syndrome"] = ser(output.simplified(True))
        return got

    runs = {}
    gens = [("depolar cc", lambda: pc.error_gens.DepolarGen(model_level='code_capacity')),
            ("depolar circuit", lambda: pc.error_gens.DepolarGen(model_level='circuit')),
            ("depolar perp", lambda: pc.error_gens.DepolarGen(model_level='code_capacity', perp_errors=True)),
            ("x circuit", lambda: pc.error_gens.XGen(model_level='circuit')),
            ("z cc", lambda: pc.error_gens.ZGen(model_level='code_capacity')),
            ("xz circuit", lambda: pc.error_gens.XZGen(model_level='circuit'))]
    for cls_name in ("Surface4444", "SurfaceMedial4444", "Color488"):
        for name, gen in gens:
            runs[f"{cls_name} {name}"] = attempt(lambda: code_runs(cls_name, name, gen()), raw=True)
    out["runs"] = runs

    # Logical error rates and pseudo-thresholds, on seeds.
    rates = {}
    surface = pc.qeccs.Surface4444(distance=3)
    for p in (0.05, 0.1):
        r = pc.tools.codecapacity_logical_rate(40, surface, 3, pc.error_gens.DepolarGen(model_level='code_capacity'), {'p': p}, pc.decoders.MWPM2D(surface), seed=11, verbose=False)
        rates[f"codecapacity {p}"] = ser(r)
    np.random.seed(3)
    random.seed(3)
    rates["pseudo"] = attempt(lambda: pc.tools.pseudo_threshold_tools.pseudo_threshold_code_capacity([0.08, 0.12, 0.16], 3, 30, verbose=False, plotting=False, mode=1, circuit_runner=pc.circuit_runners.Standard(seed=3)))
    out["rates"] = rates
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cysparsesim", help="directory holding PECOS's compiled cysparsesim module")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    import pecos

    assert pecos.__version__ == "0.2.0", pecos.__version__
    cysim = None
    if a.cysparsesim:
        sys.path.insert(0, a.cysparsesim)
        from cysparsesim import SparseSim as cysim
    result = compute(pecos, cysim)
    pathlib.Path(a.out).write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
