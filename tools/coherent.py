"""Coherent errors at circuit level, against their Pauli twirl: the measurements behind the
"Beyond Pauli noise" section, written to data/coherent/.

    python tools/coherent.py validate   # the coherent sampler against exact answers
    python tools/coherent.py sweep      # surface-code memory under over-rotation, d = 3 to 9
    python tools/coherent.py capacity   # code capacity, d = 3 to 15 (Bravyi et al.'s setting)
    python tools/coherent.py crosstalk  # ZZ crosstalk on every CNOT
    python tools/coherent.py all
    python tools/coherent.py sweep --only d=3          # re-measure these rows, keep the rest
    python tools/coherent.py validate --only code=surface

Every coherent rate is the coherent sampler's weighted estimate (stabilizer_qec.CoherentSampler),
decoded by matching on the twirled model and on the merged (coherence-aware) model; every twirl
rate is the detector sampler's, decoded on the twirled model. Exact answers come from the state
vector: its exact distribution for small repetition codes, its samples for the d = 3 surface
code.

A weighted rate is sampled until its error bar is as narrow as an unweighted run's when that run
stops (300 failures, and at least one batch of 100,000 shots): see `enough`.

Every row says what measured it: `version`, the version of the core that measured it (a row
--only keeps keeps its own), and `coherent.exact`, whether its weights summed each shot's whole
class (the sampler's `exact_weights`; validate's own `exact` is the exact answer). A file's
`version` is the core that last wrote it.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import stabilizer_qec as sq

OUT = Path(__file__).resolve().parent.parent / "data" / "coherent"
NOISE = 0.002  # SD-style circuit noise besides the rotations
THREADS = 0
# With --only field=value[,value...]: the rows to re-measure; every other row is kept as the file
# has it.
ONLY: dict = {}
KEPT: dict = {}
SECTIONS = ("validate", "sweep", "crosstalk", "capacity")
# The fields that name a row, by file.
KEYS = {"validate": ("code", "d", "rounds", "theta"), "sweep": ("d", "rounds", "theta"), "crosstalk": ("d", "rounds", "theta"), "capacity": ("d", "theta")}
# Each file's rows: repetition codes (d, rounds, θ) and the d = 3 surface code (θ, state-vector
# shots) for validate; θ by d for the sweep and the crosstalk; θ and d for code capacity.
REPETITION = [(3, 2, 0.1), (3, 3, 0.1), (5, 2, 0.1), (3, 3, 0.05), (5, 2, 0.05), (3, 4, 0.05)]
SURFACE = [(0.02, 60_000), (0.04, 60_000)]
SWEEP = {3: [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03], 5: [0.0025, 0.005, 0.01, 0.015, 0.02], 7: [0.0025, 0.005, 0.01, 0.015], 9: [0.0025, 0.005, 0.01]}
CROSSTALK = {3: [0.01, 0.02, 0.04, 0.06], 5: [0.01, 0.02, 0.04], 7: [0.01, 0.02]}
CAPACITY_THETAS = (0.2, 0.25, 0.3, 0.35)
CAPACITY_DS = (3, 5, 7, 9, 11, 13, 15)


class UsageError(ValueError):
    """A command line this tool refuses, before it measures anything."""


def grid(name: str) -> list:
    """The keys of a file's rows, in the order the tool measures them."""
    if name == "validate":
        return [{"code": "repetition", "d": d, "rounds": r, "theta": t} for d, r, t in REPETITION] + [{"code": "surface", "d": 3, "rounds": 2, "theta": t} for t, _ in SURFACE]
    if name in ("sweep", "crosstalk"):
        return [{"d": d, "rounds": d, "theta": t} for d, thetas in (SWEEP if name == "sweep" else CROSSTALK).items() for t in thetas]
    return [{"d": d, "theta": t} for t in CAPACITY_THETAS for d in CAPACITY_DS]


def label(key: dict) -> str:
    return ", ".join(f"{f}={v}" for f, v in key.items())


def wanted(key: dict) -> bool:
    return all(str(key[field]) in values for field, values in ONLY.items())


def find(name: str, key: dict):
    """The file's row for `key` as it was before this run, or None."""
    return next((r for r in KEPT[name]["rows"] if all(r[f] == v for f, v in key.items())), None)


def kept(name: str, row: dict) -> dict:
    """A row --only keeps, with its stamp: the version that measured it, or (a row from before
    rows had their own) the file's."""
    return {**row, "version": row.get("version", KEPT[name]["version"])}


def previous(name: str, key: dict) -> dict:
    """The file's row for `key` as it was before this run, kept when --only leaves it out."""
    row = find(name, key)
    if row is None:
        raise UsageError(f"{OUT / name}.json has no row {label(key)} to keep: re-measure it too, or run {name} without --only")
    return kept(name, row)


def measured_by() -> str:
    """The version of the core measuring, for the rows' stamps. 2.2's development core reports
    the last release, 2.1.0, until 2.2.0's version bump; this tool needs a core that contracts
    (the sampler's `contraction_width`, new in 2.2), so a core that reports an earlier version is
    that one, and a row it measures is stamped 2.2.0, never 2.1.0's truncating sampler."""
    version = sq.__version__
    try:
        release = tuple(int(part) for part in version.split("+")[0].split(".")[:2])
    except ValueError:
        return version
    return "2.2.0" if release < (2, 2) else version


def parse_args(argv: list) -> tuple:
    """(the sections to run, --only's rows: {field: {values}}) from the command line."""
    what, only, args = None, {}, list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--only":
            if not args:
                raise UsageError("--only needs field=value[,value...], such as --only d=3")
            spec = args.pop(0)
            field, eq, values = spec.partition("=")
            if not eq or not field or "" in values.split(","):
                raise UsageError(f"--only {spec}: expected field=value[,value...], such as --only d=3 or --only theta=0.02,0.04")
            only[field] = set(values.split(","))
        elif arg.startswith("-"):
            raise UsageError(f"unknown option {arg}")
        elif arg not in (*SECTIONS, "all"):
            raise UsageError(f"unknown section {arg!r}: run one of {', '.join(SECTIONS)} or all")
        elif what is not None:
            raise UsageError(f"one section at a time ({what}, then {arg}), or all")
        else:
            what = arg
    return (list(SECTIONS) if what in (None, "all") else [what]), only


def check_only(names: list) -> None:
    """--only's rows, checked before anything is measured: its fields name the rows of every
    section run, each value names at least one of their rows, and each file holds every row it
    keeps."""
    for field, values in ONLY.items():
        for name in names:
            if field not in KEYS[name]:
                raise UsageError(f"--only {field}=...: {name} rows have no field {field!r} (they are named by {', '.join(KEYS[name])}); run only the sections whose rows have it")
        have = {str(key[field]) for name in names for key in grid(name)}
        for value in sorted(values - have):
            raise UsageError(f"--only {field}={value}: no {' or '.join(names)} row has {field}={value} (they have {', '.join(sorted(have, key=numeric))})")
    for name in names:
        for key in grid(name):
            if not wanted(key) and not isinstance(previous(name, key)["coherent"].get("exact"), bool):
                raise UsageError(f"{OUT / name}.json's row {label(key)} does not say whether its weights were exact (coherent.exact): re-measure it too")


def numeric(value: str) -> tuple:
    """A sort key: numbers by value, then words."""
    try:
        return (0, float(value), value)
    except ValueError:
        return (1, 0.0, value)


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


def worth(total: float, total2: float, hits: float, hits2: float) -> tuple[float, float, float]:
    """A weighted rate from running sums (Σ w and Σ w² over every shot, then over the hits), its
    standard error as sq.weighted_rate gives it, and the unweighted shots it is worth: those whose
    rate would carry the same error bar, p(1 − p)/stderr². For unit weights that is the number of
    shots, and p times it the number of hits."""
    p = float(hits / total)
    se = math.sqrt((1 - p) ** 2 * hits2 + p**2 * max(total2 - hits2, 0.0)) / total
    # With no hits (or nothing else) the error bar is zero: count the weights' own worth, Kish's
    # (Σ w)² / Σ w², so that p times it is the hits, none or all.
    return p, se, (p * (1 - p) / se**2 if se > 0 else total**2 / total2)


def enough(total: float, total2: float, hits: float, hits2: float, *, min_failures: float, min_shots: float) -> bool:
    """Whether a weighted failure rate is worth what an unweighted run is worth when it stops: at
    least ``min_failures`` failures over at least ``min_shots`` shots, both counted by the error
    bar (`worth`). Exact weights (2.2's contraction) have a heavier tail than 2.1's truncated
    ones, and it follows the failures: Σ w·fail / mean w, the count 2.1 stopped on, read code
    capacity d = 15, θ = 0.35's first 100,000 shots as 22,600 failures, when their error bar was
    that of 74 failures in 327 unweighted shots. As p → 0 the failures counted here are Kish's
    (Σ_fail w)² / Σ_fail w²; for unit weights both counts are the plain ones, so an unweighted
    run stops where it always did."""
    p, _, shots = worth(total, total2, hits, hits2)
    # To rounding: for unit weights these are the plain counts, but only to the last bit.
    return bool(shots >= min_shots * (1 - 1e-9) and p * shots >= min_failures * (1 - 1e-9))


def coherent_rates(c: sq.Circuit, *, min_failures: float = 300, min_shots: float = 100_000, max_shots: int = 4_000_000, batch: int = 100_000, seed: int = 1) -> dict:
    """The coherent logical error rate (twirl-model and merged-model decoders) by weighted shots,
    until both rates are `enough` or the shot budget is spent."""
    dec_twirl, dec_merged = decoder(c.twirled()), decoder(c.twirled(merge=True))
    sampler = c.compile_coherent_sampler(seed=seed)
    # Σ w, Σ w², then Σ w and Σ w² over the twirl-model decoder's failures and the merged one's:
    # all the rule and the rates need, however many shots it takes.
    sums = np.zeros(6)
    shots = 0
    t0 = time.time()
    while shots < max_shots:
        d, o, w = sampler.sample(batch, threads=THREADS)
        w = np.asarray(w, dtype=float)
        ft = (dec_twirl.decode_batch(d, threads=THREADS) != o).any(axis=1)
        fm = (dec_merged.decode_batch(d, threads=THREADS) != o).any(axis=1)
        sums += [w.sum(), (w * w).sum(), w[ft].sum(), (w[ft] ** 2).sum(), w[fm].sum(), (w[fm] ** 2).sum()]
        shots += batch
        if all(enough(*sums[:2], *sums[i : i + 2], min_failures=min_failures, min_shots=min_shots) for i in (2, 4)):
            break
    pt, st, _ = worth(*sums[:4])
    pm, sm, _ = worth(*sums[:2], *sums[4:])
    return {
        "shots": shots,
        "ess": float(sums[0] ** 2 / sums[1] / shots),
        "rate": pt,
        "stderr": st,
        "rate_merged": pm,
        "stderr_merged": sm,
        "seconds": round(time.time() - t0, 1),
        "generators": sampler.num_generators,
        "locations": sampler.num_locations,
        "exact": sampler.exact_weights,
        "width": sampler.contraction_width,
    }


def twirl_rate(c: sq.Circuit, tw: sq.Circuit, *, min_failures: int = 300, max_shots: int = 20_000_000, batch: int = 1_000_000, seed: int = 2) -> dict:
    dec = decoder(tw)
    sampler = tw.compile_detector_sampler(seed=seed)
    fails, shots = 0, 0
    while shots < max_shots:
        d, o = sampler.sample(batch, separate_observables=True, threads=THREADS)
        fails += int((dec.decode_batch(d, threads=THREADS) != o).any(axis=1).sum())
        shots += batch
        if fails >= min_failures:
            break
    p = fails / shots
    return {"shots": shots, "failures": fails, "rate": p, "stderr": float(np.sqrt(max(p, 1 / shots) * (1 - p) / shots))}


def save(name: str, payload: dict) -> None:
    """The file, stamped with this version, each row with the version that measured it (this
    one for a row measured now); with --only, every row it had, those re-measured so far in
    place of theirs, the others kept with their stamps."""
    OUT.mkdir(parents=True, exist_ok=True)
    version = measured_by()
    rows = [r if "version" in r else {**r, "version": version} for r in payload["rows"]]
    if name in KEPT:
        new = {tuple(r[f] for f in KEYS[name]): r for r in rows}
        rows = [new.get(tuple(r[f] for f in KEYS[name])) or kept(name, r) for r in KEPT[name]["rows"]]
    for r in rows:
        if not isinstance(r["coherent"].get("exact"), bool):
            raise ValueError(f"{name} row {label({f: r[f] for f in KEYS[name]})} does not say whether its weights were exact (coherent.exact)")
    payload = {"version": version, **{k: v for k, v in payload.items() if k != "rows"}, "rows": rows}
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")


def validate() -> None:
    """The coherent sampler against exact answers."""
    rows = []
    # Repetition codes: the state vector's exact distribution, no noise besides the rotations.
    for d, rounds, theta in REPETITION:
        key = {"code": "repetition", "d": d, "rounds": rounds, "theta": theta}
        if not wanted(key):
            rows.append(previous("validate", key))
            continue
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
    for theta, shots in SURFACE:
        key = {"code": "surface", "d": 3, "rounds": 2, "theta": theta}
        if not wanted(key):
            rows.append(previous("validate", key))
            continue
        c, tw = memory(3, theta, rounds=2)
        if ONLY and previous("validate", key).get("exact_shots") == shots:
            # The state vector's samples are seeded, so re-measuring gives the file's again, and
            # they take most of the row's time (60,000 shots of 17 qubits): with --only, keep them.
            pe = previous("validate", key)["exact"]
        else:
            de, oe = c.compile_exact_sampler(seed=7).sample(shots, separate_observables=True, threads=THREADS)
            fail = (decoder(tw).decode_batch(de) != oe).any(axis=1)
            pe = float(fail.mean())
        # A fixed 4,000,000 shots: exact weights are worth fewer unweighted shots than 2.1's
        # truncated ones (at θ = 0.04, 0.27 of a shot each against 0.43), so 2.1's 2,000,000
        # would widen its error bar.
        coh = coherent_rates(c, min_failures=1e9, max_shots=4_000_000)
        tr = twirl_rate(c, tw, min_failures=10**9, max_shots=2_000_000)
        rows.append({"code": "surface", "d": 3, "rounds": 2, "theta": theta, "noise": NOISE, "exact": pe, "exact_stderr": float(np.sqrt(pe * (1 - pe) / shots)), "exact_shots": shots, "coherent": coh, "twirl": tr})
        print(f"surface d3 θ{theta}: exact {pe:.5f} coherent {coh['rate']:.5f}±{coh['stderr']:.5f} twirl {tr['rate']:.5f}", flush=True)
    save("validate", {"rows": rows})


def sweep(crosstalk: bool = False) -> None:
    rows = []
    for d, thetas in (CROSSTALK if crosstalk else SWEEP).items():
        for theta in thetas:
            key = {"d": d, "rounds": d, "theta": theta}
            if not wanted(key):
                rows.append(previous("crosstalk" if crosstalk else "sweep", key))
                continue
            c, tw = memory(d, theta, crosstalk=crosstalk)
            coh = coherent_rates(c)
            tr = twirl_rate(c, tw)
            rows.append({"d": d, "rounds": d, "theta": theta, "coherent": coh, "twirl": tr})
            print(f"{'crosstalk' if crosstalk else 'sweep'} d{d} θ{theta}: coherent {coh['rate']:.3e}±{coh['stderr']:.1e} (merged {coh['rate_merged']:.3e}) twirl {tr['rate']:.3e}  ESS {coh['ess']:.2f}  {coh['seconds']}s", flush=True)
            save("crosstalk" if crosstalk else "sweep", {"noise": NOISE, "rows": rows})


def code_capacity() -> None:
    rows = []
    for theta in CAPACITY_THETAS:
        for d in CAPACITY_DS:
            key = {"d": d, "theta": theta}
            if not wanted(key):
                rows.append(previous("capacity", key))
                continue
            c, tw = capacity(d, theta)
            # At d = 15, θ = 0.35 a weighted shot is worth 0.0035 of an unweighted one: the rule
            # took 28,700,000 of them.
            coh = coherent_rates(c, max_shots=40_000_000)
            tr = twirl_rate(c, tw, max_shots=4_000_000)
            rows.append({"d": d, "theta": theta, "coherent": coh, "twirl": tr})
            print(f"capacity d{d} θ{theta}: coherent {coh['rate']:.3e}±{coh['stderr']:.1e} twirl {tr['rate']:.3e}  ESS {coh['ess']:.2f}", flush=True)
            save("capacity", {"rows": rows})


def main(argv: list) -> int:
    try:
        names, only = parse_args(argv)
        ONLY.clear()
        ONLY.update(only)
        KEPT.clear()
        if ONLY:
            for name in names:
                path = OUT / f"{name}.json"
                if not path.exists():
                    raise UsageError(f"--only keeps the other rows of {path}, which does not exist: run {name} without --only")
                KEPT[name] = json.loads(path.read_text())
            check_only(names)
    except UsageError as e:
        print(f"tools/coherent.py: error: {e}", file=sys.stderr)
        return 2
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if "validate" in names:
        validate()
    if "capacity" in names:
        code_capacity()
    if "sweep" in names:
        sweep()
    if "crosstalk" in names:
        sweep(crosstalk=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
