"""Stim's `search_for_undetectable_logical_errors` on 1,200 random models at five sets of limits,
recorded for tests/test_distance.py (`test_faults_unchanged_on_corpus`).

The models are 2.1.0's corpus (the one its CHANGELOG counts: 3,804 searches that find faults,
2,196 that find none): three seeds of 400, each a circuit with a qubit per detector and
observable and each fault an `E(0.01)` on its qubits. Each search's answer is the faults' terms,
or Stim's message where it finds nothing. This package's answers are checked equal to Stim's
before anything is written, so the file holds 2.1.0's answers too.

    python tools/search_corpus.py   # with stim and stabilizer_qec

Writes tests/data/search_corpus.json.gz.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "data" / "search_corpus.json.gz"
LIMITS = [(4, 4, False), (3, 3, True), (2, 2, False), (6, 3, False), (5, 4, True)]
KEYS = ("dont_explore_detection_event_sets_with_size_above", "dont_explore_edges_with_degree_above", "dont_explore_edges_increasing_symptom_degree")


def models(seed: int, trials: int = 400) -> list:
    """(detectors, observables, faults) for each trial; a fault is (its detectors, its observables)."""
    rng = random.Random(seed)
    out = []
    for _ in range(trials):
        n = rng.randint(2, 9)
        nobs = rng.randint(1, 3)
        faults = []
        for _ in range(rng.randint(2, 18)):
            dets = sorted(rng.sample(range(n), rng.randint(0 if rng.random() < 0.05 else 1, min(4, n))))
            obs = [k for k in range(nobs) if rng.random() < 0.3]
            if not dets and not obs:
                continue
            if rng.random() >= 0.05:  # a fault of probability 0 is left out of the circuit
                faults.append((dets, obs))
        out.append((n, nobs, faults))
    return out


def circuit_text(n: int, nobs: int, faults: list) -> str:
    lines = ["E(0.01) " + " ".join(f"X{q}" for q in dets + [n + k for k in obs]) for dets, obs in faults]
    lines.append("M " + " ".join(str(q) for q in range(n + nobs)))
    lines += [f"DETECTOR rec[{q - n - nobs}]" for q in range(n)]
    lines += [f"OBSERVABLE_INCLUDE({k}) rec[{k - nobs}]" for k in range(nobs)]
    return "\n".join(lines)


def answer(module, text: str, limits) -> object:
    """The faults' terms, one string per fault, or the message where the search finds nothing."""
    try:
        found = module.Circuit(text).search_for_undetectable_logical_errors(**dict(zip(KEYS, limits)))
    except ValueError as ex:
        return str(ex)
    return [" ".join(str(t.dem_target) for t in e.dem_error_terms) for e in found]


def main() -> None:
    import stim

    import stabilizer_qec as sq

    corpus = [m for seed in range(3) for m in models(seed)]
    answers = []
    for n, nobs, faults in corpus:
        text = circuit_text(n, nobs, faults)
        row = [answer(stim, text, limits) for limits in LIMITS]
        assert [answer(sq, text, limits) for limits in LIMITS] == row, text
        answers.append(row)
    found = sum(isinstance(a, list) for row in answers for a in row)
    data = {"stim": stim.__version__, "limits": LIMITS, "models": corpus, "answers": answers}
    OUT.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0))
    print(f"{len(corpus)} models, {found} searches that find faults, {len(corpus) * len(LIMITS) - found} that find none: {OUT}")


if __name__ == "__main__":
    main()
