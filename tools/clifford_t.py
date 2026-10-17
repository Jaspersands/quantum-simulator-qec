"""Clifford+T circuits: the stabilizer-rank sampler against Tsim (bloqade-tsim), and the 2.0
headline experiment (Clifford+T circuits with coherent noise on every gate).

    python tools/clifford_t.py check [--shots N] [--out data/clifford_t/tsim_check.json]

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
    it: (our circuit, the same program text for Tsim)."""

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

    return build(ColorEncoder5)


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
    ea, eb = tot * n_a / (n_a + n_b), tot * n_b / (n_a + n_b)
    mask = tot > 0
    stat = float((((o_a - ea) ** 2 / ea)[mask]).sum() + (((o_b - eb) ** 2 / eb)[mask]).sum())
    dof = max(int(mask.sum()) - 1, 1)
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

    # 4. The protocol encoded in the [[17,1,5]] colour code (85 data qubits), detectors and
    # observables by both: per-bit z-scores and the two-sample test over the detector words.
    n_enc = max(shots // 10, 1000)
    c = encoded_distillation()
    ours = c.compile_stabilizer_rank_sampler(seed=0)
    ours.sample(50)
    (d_o, l_o), t_o = timed(lambda: ours.sample(n_enc, separate_observables=True, threads=1))
    theirs = tsim.Circuit(str(c)).compile_detector_sampler(seed=0)
    _, t_first = timed(lambda: theirs.sample(n_enc, batch_size=n_enc, separate_observables=True))
    (d_t, l_t), t_t = timed(lambda: theirs.sample(n_enc, batch_size=n_enc, separate_observables=True))
    bits_o, bits_t = np.hstack([d_o, l_o]), np.hstack([d_t, l_t])
    z = z_scores(bits_o.mean(0), n_enc, bits_t.mean(0), n_enc)
    results["cases"]["encoded_distillation"] = {
        "qubits": c.num_qubits,
        "non_clifford_gates": c.num_non_clifford_gates(),
        "detector_rates_ours": bits_o.mean(0).tolist(),
        "detector_rates_tsim": bits_t.mean(0).tolist(),
        "max_z": float(z.max()),
        **two_sample_chi_square(histogram(d_o), histogram(d_t)),
        "shots": n_enc,
        "ours_s_per_shot": t_o / n_enc,
        "tsim_s_per_shot": t_t / n_enc,
        "tsim_first_call_s": t_first,
        "peak_terms": ours.peak_terms,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="Tsim's demo circuits by both samplers")
    p.add_argument("--shots", type=int, default=50_000)
    p.add_argument("--out", type=pathlib.Path, default=OUT / "tsim_check.json")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        r = check(a.shots, a.out)
        for name, case in r["cases"].items():
            summary = {k: v for k, v in case.items() if k in ("chi2", "dof", "p_value", "z", "max_z", "ours_s_per_shot", "tsim_s_per_shot", "peak_terms")}
            print(name, json.dumps(summary))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
