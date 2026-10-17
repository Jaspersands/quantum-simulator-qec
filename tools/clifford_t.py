"""Clifford+T circuits: the stabilizer-rank sampler against Tsim (bloqade-tsim), and the 2.0
headline experiment (Clifford+T circuits with coherent noise on every gate).

    python tools/clifford_t.py check [--shots N] [--out data/clifford_t/tsim_check.json]
    python tools/clifford_t.py inject [--eps 0.01 0.02 ...] [--shots N]   # -> data/clifford_t/inject.json
    python tools/clifford_t.py distill [--eps ...] [--shots N] [--encoded-shots N]   # -> data/clifford_t/distill.json

`inject` encodes a magic state |T⟩ into the Steane code (d = 3) and the [[17,1,5]] colour code
(d = 5) with an over-rotation after every gate, and reads the encoded state's infidelity from
its X_L and Y_L expectations (postselected on no detected error) three ways: the Pauli twirl
of the errors (the stabilizer-rank sampler on the twirled circuit), the joint coherent sampler
(gates exact, errors twirled and reweighted), and the state vector (the truth; 7 and 17
qubits). `distill` runs Tsim's 5-qubit distillation protocol under the same noise: unencoded
(5 qubits, the state vector the truth) and encoded in the [[17,1,5]] colour code (85 qubits,
beyond any state vector, where the twirl and the joint sampler are all there is).

The noise: after every one-qubit gate exp(−iεZ) on its qubit, after every two-qubit gate
exp(−iεZZ) on its pair, after every reset exp(−iεX) (exp(−iεZ) for an X-basis reset).

`check` samples Tsim's own demo circuits (docs/demos/magic_state_distillation.ipynb: a magic
state's preparation, the 5-qubit distillation protocol noiseless and noisy, and the protocol
encoded in the [[17,1,5]] colour code through `ColorEncoder5`) with both samplers and compares
the distributions: a chi-square test against our exact probabilities where the circuit has no
noise, a two-sample test and per-bit z-scores where it has; and times each sampler per shot
(Tsim's second call, after its JAX compilation). Tsim must be importable (`pip install
bloqade-tsim`); nothing here is read by the package.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np

import stabilizer_qec as sq
from stabilizer_qec import tsim as ours_tsim
from stabilizer_qec.tsim import ColorEncoder5

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "clifford_t"

THETA = -np.arccos(np.sqrt(1 / 3)) / np.pi  # Tsim's distillation angle, in half-turns.
P_PREP = 0.05


def prep_text(p: float = P_PREP) -> str:
    """A magic state prepared, depolarized, and undone: the measurement reads its infidelity."""
    return f"R 0\nR_X({THETA}) 0\nT_DAG 0\nDEPOLARIZE1({p}) 0\nT 0\nR_X({-THETA}) 0\nM 0"


def distillation_text(p: float = P_PREP, noise: float = 0.0) -> str:
    """Tsim's 5-qubit distillation circuit (Rodriguez et al. 2025), with depolarizing noise of
    strength `noise` on its gates."""
    n = lambda qs: f"DEPOLARIZE1({noise}) {qs}\n" if noise else ""  # noqa: E731
    n2 = lambda qs: f"DEPOLARIZE2({noise}) {qs}\n" if noise else ""  # noqa: E731
    return (
        f"R 0 1 2 3 4\nR_X({THETA}) 0 1 2 3 4\nT_DAG 0 1 2 3 4\nDEPOLARIZE1({p}) 0 1 2 3 4\n"
        + n("0 1 2 3 4")
        + "SQRT_X 0 1 4\n" + n("0 1 4")
        + "CZ 0 1 2 3\n" + n2("0 1 2 3")
        + "SQRT_Y 0 3\n" + n("0 3")
        + "CZ 0 2 3 4\n" + n2("0 2 3 4")
        + "TICK\nSQRT_X_DAG 0\n" + n("0")
        + "CZ 0 4 1 3\n" + n2("0 4 1 3")
        + "TICK\nSQRT_X_DAG 0 1 2 3 4\n" + n("0 1 2 3 4")
        + f"T 0\nR_X({-THETA}) 0\nM 0 1 2 3 4"
    )


def encoded_distillation(basis: str = "X", p_prep: float = P_PREP, noise: float = P_PREP / 5):
    """The distillation circuit encoded in the [[17,1,5]] colour code, as Tsim's demo builds
    it, by this package's encoder."""
    return encoded_distillation_with(ColorEncoder5, basis, p_prep, noise)


def encoded_distillation_with(encoder_cls, basis: str = "X", p_prep: float = P_PREP, noise: float = P_PREP / 5):
    """The same through any encoder with Tsim's interface (Tsim's own included)."""

    def build(encoder_cls):
        encoder = encoder_cls()
        encoder.initialize(f"R 0 1 2 3 4\nR_X({THETA}) 0 1 2 3 4\nT_DAG 0 1 2 3 4\nDEPOLARIZE1({p_prep}) 0 1 2 3 4\n")
        encoder.encode_transversally(
            f"SQRT_X 0 1 4\nDEPOLARIZE1({noise}) 0 1 4\nCZ 0 1 2 3\nDEPOLARIZE2({noise}) 0 1 2 3\nSQRT_Y 0 3\nDEPOLARIZE1({noise}) 0 3\n"
            f"CZ 0 2 3 4\nDEPOLARIZE2({noise}) 0 2 3 4\nTICK\nSQRT_X_DAG 0\nDEPOLARIZE1({noise}) 0\nCZ 0 4\nDEPOLARIZE2({noise}) 0 4\nTICK\n"
            f"CZ 1 3\nDEPOLARIZE2({noise}) 1 3\nTICK\nSQRT_X_DAG 0 1 2 3 4\nDEPOLARIZE1({noise}) 0 1 2 3 4\n"
            + ("H 0\n" if basis == "X" else "H_YZ 0\n" if basis == "Y" else "")
            + "M 0 1 2 3 4\nDETECTOR rec[-5]\nDETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
            "OBSERVABLE_INCLUDE(0) rec[-5]\nOBSERVABLE_INCLUDE(1) rec[-4]\nOBSERVABLE_INCLUDE(2) rec[-3]\nOBSERVABLE_INCLUDE(3) rec[-2]\nOBSERVABLE_INCLUDE(4) rec[-1]\n"
        )
        return encoder.circuit

    return build(encoder_cls)


def post_select(samples: np.ndarray) -> tuple[float, float]:
    """(infidelity of the distilled output, acceptance rate) from the 5-qubit records."""
    sel = np.all(samples[:, 1:] == np.array([1, 0, 1, 1], dtype=bool), axis=1)
    kept = samples[sel, 0]
    return (float(kept.mean()) if kept.size else float("nan")), float(sel.mean())


def chi_square(counts: np.ndarray, expected: np.ndarray) -> dict:
    """Pearson's chi-square of observed counts against expected ones, bins with fewer than 5
    expected pooled; the p-value through scipy where it is installed."""
    order = np.argsort(expected)
    pooled_o, pooled_e, o_acc, e_acc = [], [], 0.0, 0.0
    for i in order:
        o_acc += counts[i]
        e_acc += expected[i]
        if e_acc >= 5:
            pooled_o.append(o_acc)
            pooled_e.append(e_acc)
            o_acc = e_acc = 0.0
    if e_acc > 0 and pooled_e:
        pooled_o[-1] += o_acc
        pooled_e[-1] += e_acc
    o, e = np.array(pooled_o), np.array(pooled_e)
    stat = float(((o - e) ** 2 / e).sum())
    dof = max(len(e) - 1, 1)
    return {"chi2": stat, "dof": dof, "p_value": p_value(stat, dof)}


def two_sample_chi_square(a: np.ndarray, b: np.ndarray) -> dict:
    """Pearson's test of homogeneity between two samples' histograms (bins pooled below 5
    expected in either)."""
    n_a, n_b = a.sum(), b.sum()
    total = a + b
    e_a, e_b = total * n_a / (n_a + n_b), total * n_b / (n_a + n_b)
    keep = (e_a >= 5) & (e_b >= 5)
    rest_a, rest_b = a[~keep].sum(), b[~keep].sum()
    o_a, o_b = np.append(a[keep], rest_a), np.append(b[keep], rest_b)
    tot = o_a + o_b
    mask = tot > 0
    o_a, o_b, tot = o_a[mask], o_b[mask], tot[mask]
    ea, eb = tot * n_a / (n_a + n_b), tot * n_b / (n_a + n_b)
    stat = float(((o_a - ea) ** 2 / ea).sum() + ((o_b - eb) ** 2 / eb).sum())
    dof = max(len(tot) - 1, 1)
    return {"chi2": stat, "dof": dof, "p_value": p_value(stat, dof)}


def p_value(stat: float, dof: int):
    try:
        from scipy.stats import chi2

        return float(chi2.sf(stat, dof))
    except ImportError:
        return None


def z_scores(p_a: np.ndarray, n_a: int, p_b: np.ndarray, n_b: int) -> np.ndarray:
    var = p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b
    return np.abs(p_a - p_b) / np.sqrt(np.maximum(var, 1e-300))


def histogram(samples: np.ndarray) -> np.ndarray:
    """Counts of each record, records read as little-endian integers."""
    idx = (samples.astype(np.int64) << np.arange(samples.shape[1])).sum(axis=1)
    return np.bincount(idx, minlength=1 << samples.shape[1]).astype(float)


def timed(fn) -> tuple[Any, float]:
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def check(shots: int, out: pathlib.Path) -> dict:
    import tsim

    results: dict = {"tsim": getattr(tsim, "__version__", "0.1.5"), "shots": shots, "cases": {}}
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        out.write_text(json.dumps(results, indent=1) + "\n")
        case = results["cases"][name]
        print(name, json.dumps({k: v for k, v in case.items() if k in ("chi2", "dof", "p_value", "z", "max_z", "ours_s_per_shot", "tsim_s_per_shot", "peak_terms")}), flush=True)

    def both_records(text: str, n: int, seed: int = 0):
        ours = ours_tsim.Circuit(text).compile_stabilizer_rank_sampler(seed=seed)
        ours.sample_measurements(100)  # warm
        o, t_o = timed(lambda: ours.sample_measurements(n, threads=1))
        theirs = tsim.Circuit(text).compile_sampler(seed=seed)
        _, t_first = timed(lambda: theirs.sample(n, batch_size=n))
        t, t_t = timed(lambda: theirs.sample(n, batch_size=n))
        return o, t, {"ours_s_per_shot": t_o / n, "tsim_s_per_shot": t_t / n, "tsim_first_call_s": t_first, "peak_terms": ours.peak_terms}

    # 1. The magic state's infidelity: exact here, sampled by Tsim.
    text = prep_text()
    c = ours_tsim.Circuit(text + "\nDETECTOR rec[-1]")
    p1 = c.exact_distribution(method="stabilizer_rank").get(((True,), ()), 0.0)
    _, t, timing = both_records(text, shots)
    rate = float(t.mean())
    results["cases"]["prep"] = {"exact_infidelity": p1, "tsim_infidelity": rate, "z": float(z_scores(np.array([p1]), 10**12, np.array([rate]), shots)[0]), **timing}
    save("prep")

    # 2. Distillation without noise: our exact record probabilities against Tsim's histogram.
    text = distillation_text()
    sampler = ours_tsim.Circuit(text).compile_stabilizer_rank_sampler(seed=0)
    records = np.array([[(i >> k) & 1 for k in range(5)] for i in range(32)], dtype=bool)
    probs = sampler.probability_of(records)
    o, t, timing = both_records(text, shots)
    hist_t = histogram(t)
    inf_o, acc_o = post_select(o)
    inf_t, acc_t = post_select(t)
    exact_acc = float(sum(p for r, p in zip(records, probs) if np.array_equal(r[1:], [1, 0, 1, 1])))
    exact_inf = float(sum(p for r, p in zip(records, probs) if np.array_equal(r[1:], [1, 0, 1, 1]) and r[0]) / exact_acc)
    results["cases"]["distillation"] = {
        **chi_square(hist_t, probs * shots),
        "exact": {"infidelity": exact_inf, "acceptance": exact_acc},
        "ours": {"infidelity": inf_o, "acceptance": acc_o},
        "tsim": {"infidelity": inf_t, "acceptance": acc_t},
        **timing,
    }
    save("distillation")

    # 3. Distillation with noise on its gates: both sampled.
    text = distillation_text(noise=P_PREP / 5)
    o, t, timing = both_records(text, shots)
    inf_o, acc_o = post_select(o)
    inf_t, acc_t = post_select(t)
    results["cases"]["distillation_noisy"] = {
        **two_sample_chi_square(histogram(o), histogram(t)),
        "ours": {"infidelity": inf_o, "acceptance": acc_o},
        "tsim": {"infidelity": inf_t, "acceptance": acc_t},
        **timing,
    }
    save("distillation_noisy")

    # 4. The protocol encoded in the [[17,1,5]] colour code (85 data qubits), detectors and
    # observables by both: per-bit z-scores and the two-sample test over the detector words
    # (Tsim's batches kept small: its compiled graph of 85 qubits is large).
    n_enc = max(shots // 25, 500)
    c = encoded_distillation()
    ours = c.compile_stabilizer_rank_sampler(seed=0)
    ours.sample(50)
    (d_o, l_o), t_o = timed(lambda: ours.sample(n_enc, separate_observables=True, threads=1))
    bits_o = np.hstack([d_o, l_o])
    case = {
        "qubits": c.num_qubits,
        "non_clifford_gates": c.num_non_clifford_gates(),
        "detector_rates_ours": bits_o.mean(0).tolist(),
        "shots": n_enc,
        "ours_s_per_shot": t_o / n_enc,
        "peak_terms": ours.peak_terms,
    }
    results["cases"]["encoded_distillation"] = case
    save("encoded_distillation")
    # Tsim's side, from its own encoder, in a process of its own: compiling its graph of the
    # 85-qubit circuit was killed for memory on a 16 GB machine.
    tsim_out = out.with_name("tsim_encoded.npz")
    script = (
        "import sys, time, numpy as np\nsys.path.insert(0, sys.argv[1])\nfrom clifford_t import encoded_distillation_with\n"
        "from tsim.utils.encoder import ColorEncoder5\nc = encoded_distillation_with(ColorEncoder5)\nn, batch = int(sys.argv[2]), int(sys.argv[3])\n"
        "s = c.compile_detector_sampler(seed=0)\nt = time.perf_counter(); s.sample(batch, batch_size=batch, separate_observables=True); first = time.perf_counter() - t\n"
        "t = time.perf_counter(); d, l = s.sample(n, batch_size=batch, separate_observables=True); dt = time.perf_counter() - t\n"
        "np.savez(sys.argv[4], d=d, l=l, first=first, dt=dt)\n"
    )
    import subprocess

    proc = subprocess.run([sys.executable, "-c", script, str(ROOT / "tools"), str(n_enc), str(min(n_enc, 250)), str(tsim_out)], capture_output=True, text=True)
    if proc.returncode != 0 or not tsim_out.exists():
        case["tsim"] = f"failed: exit code {proc.returncode}" + (" (killed: out of memory)" if proc.returncode in (-9, 137) else "") + (f"; {proc.stderr.strip().splitlines()[-1]}" if proc.stderr.strip() else "")
        save("encoded_distillation")
        return results
    with np.load(tsim_out) as f:
        d_t, l_t, t_first, t_t = f["d"], f["l"], float(f["first"]), float(f["dt"])
    tsim_out.unlink()
    bits_t = np.hstack([d_t, l_t])
    z = z_scores(bits_o.mean(0), n_enc, bits_t.mean(0), n_enc)
    case.update({"detector_rates_tsim": bits_t.mean(0).tolist(), "max_z": float(z.max()), **two_sample_chi_square(histogram(d_o), histogram(d_t)), "tsim_s_per_shot": t_t / n_enc, "tsim_first_call_s": t_first})
    save("encoded_distillation")
    return results


def over_rotate(circuit: sq.Circuit, eps: float) -> sq.Circuit:
    """The circuit with an over-rotation of angle `eps` after every gate and reset."""
    out = sq.Circuit()
    for ins in circuit.flattened():
        out.append(ins)
        gd = sq.gate_data(ins.name)
        targets = [t.value for t in ins.targets_copy() if not t.is_combiner]
        if gd.is_unitary and targets:
            if gd.is_two_qubit_gate:
                out.append("II_ERROR", targets, [], tag=f"R_ZZ(theta={eps!r})")
            else:
                out.append("I_ERROR", sorted(set(targets)), [], tag=f"R_Z(theta={eps!r})")
        elif gd.is_reset and targets:
            out.append("I_ERROR", targets, [], tag=f"R_{'Z' if ins.name == 'RX' else 'X'}(theta={eps!r})")
    return out


def weighted_mean(values: np.ndarray, weights) -> float:
    w = np.ones(len(values)) if weights is None else weights
    return float((w * values).sum() / w.sum()) if w.sum() > 0 else float("nan")


def three_ways(circuit: sq.Circuit, shots: int, exact: bool, seed: int = 0, threads: int = 0) -> dict:
    """Detection events and observables of a circuit with coherent errors by the twirl, the
    joint sampler and (where it fits) the state vector: {"twirl": (d, o, None), "coherent":
    (d, o, w), "exact": distribution or None}, with timings."""
    out: dict = {}
    t = time.perf_counter()
    tw = circuit.twirled().compile_stabilizer_rank_sampler(seed=seed)
    d, o = tw.sample(shots, separate_observables=True, threads=threads)
    out["twirl"] = (d, o, None, time.perf_counter() - t, tw.peak_terms)
    t = time.perf_counter()
    cs = circuit.compile_coherent_sampler(seed=seed)
    d, o, w = cs.sample(shots, threads=threads)
    out["coherent"] = (d, o, w, time.perf_counter() - t, cs.num_locations, cs.num_exact, cs.num_generators)
    out["exact"] = circuit.exact_distribution(max_branches=1 << 24) if exact else None
    return out


def rates(d: np.ndarray, o: np.ndarray, w, accept: np.ndarray) -> dict:
    """Acceptance and the accepted shots' observable rates (weighted)."""
    ww = np.ones(len(d)) if w is None else w
    acc = weighted_mean(accept.astype(float), ww)
    kept = ww * accept
    obs = [float((kept * o[:, k]).sum() / kept.sum()) if kept.sum() > 0 else float("nan") for k in range(o.shape[1])]
    return {"acceptance": acc, "observables": obs, "ess": float(ww.sum() ** 2 / (ww**2).sum())}


def exact_rates(dist: dict, accept_fn) -> dict:
    acc = sum(p for (dets, obs), p in dist.items() if accept_fn(dets, obs))
    no = max((len(obs) for (_, obs) in dist), default=0)
    obs = [sum(p for (dets, ob), p in dist.items() if accept_fn(dets, ob) and ob[k]) / acc if acc else float("nan") for k in range(no)]
    return {"acceptance": acc, "observables": obs}


def inject(eps_list, shots: int, out: pathlib.Path) -> dict:
    """The encoded magic state's infidelity: F = (1 + (⟨X_L⟩ + ⟨Y_L⟩)/√2) / 2 from two circuits,
    one measuring X_L and one Y_L transversally, postselected on no detected error; signs
    calibrated on the noiseless circuit (|T_L⟩ has ⟨X_L⟩ = ⟨Y_L⟩ = 1/√2)."""
    from stabilizer_qec.tsim import SteaneEncoder

    results: dict = {"shots": shots, "codes": {}}
    for name, encoder_cls in (("steane_d3", SteaneEncoder), ("color_d5", ColorEncoder5)):
        base = {}
        for basis in "XY":
            enc = encoder_cls()
            enc.initialize("R 0\nH 0\nT 0\n")
            enc.encode_transversally(f"M{basis} 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
            c = enc.circuit
            ideal = exact_rates(c.exact_distribution(), lambda dets, obs: not any(dets))
            # The observable flip rate of the ideal state is (1 ∓ 1/√2)/2: the sign that makes it +1/√2.
            sign = 1.0 if ideal["observables"][0] < 0.5 else -1.0
            base[basis] = (c, sign)
        entry = {"qubits": base["X"][0].num_qubits, "eps": {}}
        for eps in eps_list:
            row: dict = {}
            for way in ("twirl", "coherent", "exact"):
                row[way] = {}
            per_basis = {}
            for basis in "XY":
                c, sign = base[basis]
                noisy = over_rotate(c, eps)
                r = three_ways(noisy, shots, exact=True)
                per_basis[basis] = {}
                for way in ("twirl", "coherent"):
                    d, o, w = r[way][:3]
                    rr = rates(d, o, w, ~d.any(axis=1))
                    per_basis[basis][way] = {"expectation": sign * (1 - 2 * rr["observables"][0]), "acceptance": rr["acceptance"], "ess": rr["ess"]}
                er = exact_rates(r["exact"], lambda dets, obs: not any(dets))
                per_basis[basis]["exact"] = {"expectation": sign * (1 - 2 * er["observables"][0]), "acceptance": er["acceptance"]}
                row.setdefault("timing", {})[basis] = {"twirl_s": r["twirl"][3], "coherent_s": r["coherent"][3], "locations": r["coherent"][4], "exact_gates": r["coherent"][5], "generators": r["coherent"][6]}
            for way in ("twirl", "coherent", "exact"):
                ex, ey = per_basis["X"][way]["expectation"], per_basis["Y"][way]["expectation"]
                row[way] = {"infidelity": 1 - (1 + (ex + ey) / np.sqrt(2)) / 2, "x": ex, "y": ey, "acceptance": per_basis["X"][way]["acceptance"]}
                if way != "exact":
                    row[way]["ess"] = min(per_basis["X"][way]["ess"], per_basis["Y"][way]["ess"])
            entry["eps"][repr(eps)] = row
            print(name, eps, json.dumps({w: round(row[w]["infidelity"], 5) for w in ("twirl", "coherent", "exact")}), "ess", round(row["coherent"]["ess"]), flush=True)
        results["codes"][name] = entry
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    return results


def distill(eps_list, shots: int, encoded_shots: int, out: pathlib.Path) -> dict:
    """Tsim's 5-qubit distillation under over-rotation on every gate: the distilled state's
    infidelity (P(output = 1) given the syndrome [1, 0, 1, 1]) and the acceptance, unencoded
    (the state vector the truth) and encoded in the [[17,1,5]] colour code (85 qubits, with
    every stabilizer check of the final transversal measurements clean)."""
    results: dict = {"shots": shots, "encoded_shots": encoded_shots, "unencoded": {}, "encoded": {}}
    plain = ours_tsim.Circuit(distillation_text(p=0.0) + "\n" + "\n".join(f"OBSERVABLE_INCLUDE({k}) rec[{k - 5}]" for k in range(5)))
    encoded = encoded_distillation(p_prep=0.0, noise=0.0)
    syndrome = np.array([True, False, True, True])

    def accept_fn(dets, obs):
        return not any(dets) and tuple(obs[1:5]) == tuple(syndrome)

    for label, circuit, n, exact in (("unencoded", plain, shots, True), ("encoded", encoded, encoded_shots, False)):
        # Observable flips are relative to the reference run: its signs give the outcomes back.
        _, ref = circuit.reference_detector_and_observable_signs()
        ref = np.asarray(ref, dtype=bool)
        for eps in eps_list:
            noisy = over_rotate(circuit, eps)
            r = three_ways(noisy, n, exact=exact)
            row: dict = {"qubits": noisy.num_qubits, "timing": {"twirl_s": r["twirl"][3], "coherent_s": r["coherent"][3], "locations": r["coherent"][4], "exact_gates": r["coherent"][5], "generators": r["coherent"][6], "twirl_peak_terms": r["twirl"][4]}}
            for way in ("twirl", "coherent"):
                d, o, w = r[way][:3]
                outcomes = o ^ ref[None, :]
                accept = ~d.any(axis=1) & np.all(outcomes[:, 1:5] == syndrome, axis=1)
                rr = rates(d, outcomes, w, accept)
                row[way] = {"infidelity": rr["observables"][0], "acceptance": rr["acceptance"], "ess": rr["ess"]}
            if exact:
                er = exact_rates({(dets, tuple(np.array(obs) ^ ref)): p for (dets, obs), p in r["exact"].items()}, accept_fn)
                row["exact"] = {"infidelity": er["observables"][0], "acceptance": er["acceptance"]}
            results[label][repr(eps)] = row
            print(label, eps, json.dumps({w: {k: round(v, 5) for k, v in row[w].items() if k != "ess"} for w in row if w not in ("qubits", "timing")}), "ess", round(row["coherent"]["ess"]), f"{row['timing']['coherent_s'] / n * 1e3:.1f} ms/shot", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=1) + "\n")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="Tsim's demo circuits by both samplers")
    p.add_argument("--shots", type=int, default=50_000)
    p.add_argument("--out", type=pathlib.Path, default=OUT / "tsim_check.json")
    p = sub.add_parser("inject", help="an encoded magic state under over-rotation on every gate")
    p.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1])
    p.add_argument("--shots", type=int, default=20_000)
    p.add_argument("--out", type=pathlib.Path, default=OUT / "inject.json")
    p = sub.add_parser("distill", help="Tsim's distillation protocol under over-rotation on every gate")
    p.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    p.add_argument("--shots", type=int, default=20_000)
    p.add_argument("--encoded-shots", type=int, default=2_000)
    p.add_argument("--out", type=pathlib.Path, default=OUT / "distill.json")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        check(a.shots, a.out)
    elif a.cmd == "inject":
        inject(a.eps, a.shots, a.out)
    elif a.cmd == "distill":
        distill(a.eps, a.shots, a.encoded_shots, a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
