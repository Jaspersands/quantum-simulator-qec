"""Coherent errors at circuit level, against their Pauli twirl: the measurements behind the
"Beyond Pauli noise" section, written to data/coherent/.

    python tools/coherent.py validate   # the coherent sampler against exact answers
    python tools/coherent.py sweep      # surface-code memory under over-rotation, d = 3 to 9
    python tools/coherent.py capacity   # code capacity, d = 3 to 15 (Bravyi et al.'s setting)
    python tools/coherent.py crosstalk  # ZZ crosstalk on every CNOT
    python tools/coherent.py all

Every coherent rate is the coherent sampler's weighted estimate (stabilizer_qec.CoherentSampler),
decoded by matching on the twirled model and on the merged (coherence-aware) model; every twirl
rate is the detector sampler's, decoded on the twirled model. Exact answers come from the state
vector: its exact distribution for small repetition codes, its samples for the d = 3 surface
code.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import stabilizer_qec as sq

OUT = Path(__file__).resolve().parent.parent / "data" / "coherent"
NOISE = 0.002  # SD-style circuit noise besides the rotations
THREADS = 0


def data_qubits(text: str) -> list[str]:
    """The qubits the memory measures at the end."""
    return [line for line in text.splitlines() if line.startswith(("M ", "MX "))][-1].split()[1:]


def memory(d: int, theta: float, *, rounds: int | None = None, noise: float = NOISE, crosstalk: bool = False) -> tuple[sq.Circuit, sq.Circuit]:
    """The rotated surface code's X memory with a coherent Z over-rotation by θ on every data
    qubit after every tick (or, with ``crosstalk``, a ZZ rotation on the pair of every CNOT),
    and its twirl."""
    base = sq.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=rounds or d,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        after_reset_flip_probability=noise,
    )
    text = str(base)
    if crosstalk:
        lines = []
        for line in text.splitlines():
            lines.append(line)
            stripped = line.strip()
            if stripped.startswith("CX "):
                targets = stripped.split()[1:]
                indent = line[: len(line) - len(line.lstrip())]
                lines.append(f"{indent}II_ERROR[R_ZZ(theta={theta!r})] " + " ".join(targets))
        text = "\n".join(lines)
    else:
        text = text.replace("TICK", f"TICK\nI_ERROR[R_Z(theta={theta!r})] " + " ".join(data_qubits(text)))
    c = sq.Circuit(text)
    return c, c.twirled()


def capacity(d: int, theta: float) -> tuple[sq.Circuit, sq.Circuit]:
    """Code capacity: perfect measurements, one Z over-rotation by θ on every data qubit before
    the first round."""
    text = str(sq.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=1))
    text = text.replace("TICK", f"TICK\nI_ERROR[R_Z(theta={theta!r})] " + " ".join(data_qubits(text)), 1)
    c = sq.Circuit(text)
    return c, c.twirled()


def decoder(circuit: sq.Circuit) -> sq.Matching:
    return sq.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))


def coherent_rates(c: sq.Circuit, *, min_failures: float = 300, max_shots: int = 4_000_000, batch: int = 100_000, seed: int = 1) -> dict:
    """The coherent logical error rate (twirl-model and merged-model decoders) by weighted shots,
    until enough weighted failures or the shot budget."""
    dec_twirl, dec_merged = decoder(c.twirled()), decoder(c.twirled(merge=True))
    sampler = c.compile_coherent_sampler(seed=seed)
    ws, ft, fm = [], [], []
    shots = 0
    t0 = time.time()
    while shots < max_shots:
        d, o, w = sampler.sample(batch, threads=THREADS)
        ws.append(w)
        ft.append((dec_twirl.decode_batch(d) != o).any(axis=1))
        fm.append((dec_merged.decode_batch(d) != o).any(axis=1))
        shots += batch
        w_all = np.concatenate(ws)
        if (w_all * np.concatenate(ft)).sum() / w_all.mean() >= min_failures:
            break
    w = np.concatenate(ws)
    pt, st = sq.weighted_rate(np.concatenate(ft), w)
    pm, sm = sq.weighted_rate(np.concatenate(fm), w)
    return {
        "shots": shots,
        "ess": sq.effective_sample_size(w) / len(w),
        "rate": pt,
        "stderr": st,
        "rate_merged": pm,
        "stderr_merged": sm,
        "seconds": round(time.time() - t0, 1),
        "generators": sampler.num_generators,
        "locations": sampler.num_locations,
    }


def twirl_rate(c: sq.Circuit, tw: sq.Circuit, *, min_failures: int = 300, max_shots: int = 20_000_000, batch: int = 1_000_000, seed: int = 2) -> dict:
    dec = decoder(tw)
    sampler = tw.compile_detector_sampler(seed=seed)
    fails, shots = 0, 0
    while shots < max_shots:
        d, o = sampler.sample(batch, separate_observables=True, threads=THREADS)
        fails += int((dec.decode_batch(d) != o).any(axis=1).sum())
        shots += batch
        if fails >= min_failures:
            break
    p = fails / shots
    return {"shots": shots, "failures": fails, "rate": p, "stderr": float(np.sqrt(max(p, 1 / shots) * (1 - p) / shots))}


def save(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"version": sq.__version__, **payload}
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")


def validate() -> None:
    """The coherent sampler against exact answers."""
    rows = []
    # Repetition codes: the state vector's exact distribution, no noise besides the rotations.
    for d, rounds, theta in [(3, 2, 0.1), (3, 3, 0.1), (5, 2, 0.1), (3, 3, 0.05), (5, 2, 0.05), (3, 4, 0.05)]:
        text = str(sq.Circuit.generated("repetition_code:memory", distance=d, rounds=rounds))
        data = [line for line in text.splitlines() if line.startswith("M ")][-1].split()[1:]
        c = sq.Circuit(text.replace("TICK", f"TICK\nI_ERROR[R_X(theta={theta!r})] " + " ".join(data)))
        tw = c.twirled()
        dec = decoder(tw)
        dist = c.exact_distribution(max_branches=1 << 24)
        keys = list(dist)
        dets = np.array([k[0] for k in keys], dtype=bool)
        obs = np.array([k[1] for k in keys], dtype=bool)
        probs = np.array([dist[k] for k in keys])
        exact = float((probs * (dec.decode_batch(dets) != obs).any(axis=1)).sum())
        coh = coherent_rates(c, min_failures=1e9, max_shots=1_000_000)
        tr = twirl_rate(c, tw, min_failures=10**9, max_shots=2_000_000)
        rows.append({"code": "repetition", "d": d, "rounds": rounds, "theta": theta, "noise": 0.0, "exact": exact, "exact_stderr": 0.0, "coherent": coh, "twirl": tr})
        print(f"repetition d{d} r{rounds} θ{theta}: exact {exact:.5f} coherent {coh['rate']:.5f}±{coh['stderr']:.5f} twirl {tr['rate']:.5f}", flush=True)
    # The d = 3 surface code: state-vector samples (17 qubits).
    for theta, shots in [(0.02, 60_000), (0.04, 60_000)]:
        c, tw = memory(3, theta, rounds=2)
        de, oe = c.compile_exact_sampler(seed=7).sample(shots, separate_observables=True, threads=THREADS)
        fail = (decoder(tw).decode_batch(de) != oe).any(axis=1)
        pe = float(fail.mean())
        coh = coherent_rates(c, min_failures=1e9, max_shots=2_000_000)
        tr = twirl_rate(c, tw, min_failures=10**9, max_shots=2_000_000)
        rows.append({"code": "surface", "d": 3, "rounds": 2, "theta": theta, "noise": NOISE, "exact": pe, "exact_stderr": float(np.sqrt(pe * (1 - pe) / shots)), "exact_shots": shots, "coherent": coh, "twirl": tr})
        print(f"surface d3 θ{theta}: exact {pe:.5f} coherent {coh['rate']:.5f}±{coh['stderr']:.5f} twirl {tr['rate']:.5f}", flush=True)
    save("validate", {"rows": rows})


SWEEP = {3: [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03], 5: [0.0025, 0.005, 0.01, 0.015, 0.02], 7: [0.0025, 0.005, 0.01, 0.015], 9: [0.0025, 0.005, 0.01]}


def sweep(crosstalk: bool = False) -> None:
    rows = []
    grid = {3: [0.01, 0.02, 0.04, 0.06], 5: [0.01, 0.02, 0.04], 7: [0.01, 0.02]} if crosstalk else SWEEP
    for d, thetas in grid.items():
        for theta in thetas:
            c, tw = memory(d, theta, crosstalk=crosstalk)
            coh = coherent_rates(c)
            tr = twirl_rate(c, tw)
            rows.append({"d": d, "rounds": d, "theta": theta, "coherent": coh, "twirl": tr})
            print(f"{'crosstalk' if crosstalk else 'sweep'} d{d} θ{theta}: coherent {coh['rate']:.3e}±{coh['stderr']:.1e} (merged {coh['rate_merged']:.3e}) twirl {tr['rate']:.3e}  ESS {coh['ess']:.2f}  {coh['seconds']}s", flush=True)
            save("crosstalk" if crosstalk else "sweep", {"noise": NOISE, "rows": rows})


def code_capacity() -> None:
    rows = []
    for theta in (0.2, 0.25, 0.3, 0.35):
        for d in (3, 5, 7, 9, 11, 13, 15):
            c, tw = capacity(d, theta)
            coh = coherent_rates(c, max_shots=2_000_000)
            tr = twirl_rate(c, tw, max_shots=4_000_000)
            rows.append({"d": d, "theta": theta, "coherent": coh, "twirl": tr})
            print(f"capacity d{d} θ{theta}: coherent {coh['rate']:.3e}±{coh['stderr']:.1e} twirl {tr['rate']:.3e}  ESS {coh['ess']:.2f}", flush=True)
            save("capacity", {"rows": rows})


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if what in ("validate", "all"):
        validate()
    if what in ("capacity", "all"):
        code_capacity()
    if what in ("sweep", "all"):
        sweep()
    if what in ("crosstalk", "all"):
        sweep(crosstalk=True)
