"""PyMatching 2.4.0's ``animate`` frames and messages, recorded from PyMatching itself as the
fixtures of ``stabilizer_qec.pymatching``'s ``animate``.

    python -I tools/pymatching_animate_fixtures.py --pymatching BUILD/pymatching \\
        --out tests/pymatching_port/animate

The frames come from PyMatching 2.4.0's macOS arm64 wheel: ``pymatching.cli`` (what PyMatching's
``pymatching`` script runs), exiting with what it returns. ``--pymatching`` is PyMatching 2.4.0's
C++ command line, built from its sdist with CMake (target ``pymatching``; its CMakeLists pins Stim
at 1320ad7e and googletest 1.12.1): the same ``pm::main``, whose exit codes and stderr must be the
wheel's, and whose frames each case notes as equal or not (``program_frames_equal``). Built with
a newer clang (Apple clang 21 here; the wheel's SDK is 15.5), it does not fuse ``lerp_pos``'s
multiply and add as the wheel does, and some polygon corners differ in a sixth digit, so the
shipped wheel is the reference for the drawing. The wheel also gives the exceptions the program aborts
on (an ``--out_dir`` it cannot create) and the message of an unknown command. Stim 1.16 makes the
models and shots.

Each frame case runs in a fresh directory with relative paths (``model.dem``, ``shot.<format>``,
``--out_dir out``), so stderr's ``wrote out/frame_00000.svg`` lines are the same wherever the
fixtures are checked. ``<case>.json.gz`` holds the case's model and shot file, its arguments, the
exit code and stderr, the frame names, and each frame as its lines sorted (an index into the
case's table of distinct lines): PyMatching draws its regions in heap-address order
(``std::set<GraphFillRegion *>``, ``mwpm_diagram.cc:218``), which no port can reproduce, so a
frame is compared as its sorted lines. ``messages.json`` holds the command-line misuse cases:
arguments, the program's exit code, stderr and frame count, and what ``pymatching.cli`` returned
or raised (an exception the program aborts on, the wheel raises).

The cases:

- ``d3_varied``: rotated surface-code memory (X), distance 3, 3 rounds, each error's probability
  redrawn (so every edge has its own weight), shot as ``01``;
- ``d5_varied``: the same (Z) at distance 5, 5 rounds, shot as ``b8``: 31 detection events, with
  alternating trees, blossoms and a blossom shattering;
- ``d3_uniform``: distance 3, 3 rounds, the generated (uniform) noise, shot as ``dets`` with the
  observables appended, ``--quiet``, no growth frames;
- ``repetition``: a repetition code, distance 7, 4 rounds, shot as ``r8``, no event frames;
- ``coords``: a small model whose detectors have no, one, two and four coordinates;
- ``empty``: a shot without detection events (only the held start and end frames), one detector
  at the average of all (its boundary drawn in PyMatching's fixed direction);
- ``no_matching``: one detection event in a component without a boundary: the queue runs dry
  and PyMatching stops with ``Unknown tentative event type.`` after the frames it drew;
- ``irregular``: a grid of detectors at irregular (non-integer) places with three and five
  coordinates, its edges' probabilities all different, shot as ``hits`` listed out of order;
- ``irregular_ptb64``: the same model, another shot, the first of a group of 64 in ``ptb64``.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import tempfile

PARAMS = ["--held_frames_per_event", "--held_frames_at_start", "--held_frames_at_end", "--max_growth_between_frames", "--pixels_per_unit_length"]


def _varied(dem_text: str, seed: int, top: float) -> str:
    """Each ``error(p)`` given its own probability, up to ``top``."""
    rng = random.Random(seed)
    return re.sub(r"^(\s*)error\([^)]*\)", lambda m: f"{m.group(1)}error({rng.uniform(top / 20, top):.4g})", dem_text, flags=re.M)


def _shot(stim, dem, *, seed: int, lo: int, hi: int):
    """The first sampled shot with ``lo`` to ``hi`` detection events: (detectors, observables)."""
    dets, obs, _ = dem.compile_sampler(seed=seed).sample(4096)
    for d, o in zip(dets, obs):
        if lo <= int(d.sum()) <= hi:
            return d, o
    raise ValueError(f"no shot with {lo} to {hi} detection events")


def _file(stim, dets, obs, fmt: str, *, appended: bool) -> bytes:
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "shot")
        data = np.concatenate([dets, obs]) if appended else dets
        stim.write_shot_data_file(data=data[None, :], path=path, format=fmt, num_detectors=len(dets), num_observables=len(obs) if appended else 0)
        return pathlib.Path(path).read_bytes()


def frame_cases(stim) -> list:
    out = []

    def add(name, dem_text, shot, fmt, params, *, appended=False, quiet=False):
        args = ["animate", "--dem_in", "model.dem", "--dets_in", f"shot.{fmt}", "--dets_in_format", fmt, "--out_dir", "out"]
        for flag, v in zip(PARAMS, params):
            args += [flag, str(v)]
        if appended:
            args.append("--dets_in_includes_appended_observables")
        if quiet:
            args.append("--quiet")
        out.append({"name": name, "dem": dem_text, "shot_name": f"shot.{fmt}", "shot": base64.b64encode(shot).decode(), "args": args})

    def memory(task, d, rounds, *, varied_seed=None, top=0.0, shot_seed, lo, hi, fmt, params, appended=False, quiet=False, name):
        c = stim.Circuit.generated(task, distance=d, rounds=rounds, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
        text = str(c.detector_error_model(decompose_errors=True))
        if varied_seed is not None:
            text = _varied(text, varied_seed, top)
        dem = stim.DetectorErrorModel(text)
        dets, obs = _shot(stim, dem, seed=shot_seed, lo=lo, hi=hi)
        add(name, text, _file(stim, dets, obs, fmt, appended=appended), fmt, params, appended=appended, quiet=quiet)

    memory("surface_code:rotated_memory_x", 3, 3, varied_seed=3, top=0.03, shot_seed=1, lo=6, hi=8, fmt="01", params=[2, 3, 3, 20, 20], name="d3_varied")
    memory("surface_code:rotated_memory_z", 5, 5, varied_seed=5, top=0.03, shot_seed=10, lo=30, hi=34, fmt="b8", params=[1, 1, 2, 40, 12], name="d5_varied")
    memory("surface_code:rotated_memory_x", 3, 3, shot_seed=4, lo=6, hi=8, fmt="dets", params=[1, 0, 1, 0, 25], appended=True, quiet=True, name="d3_uniform")
    memory("repetition_code:memory", 7, 4, shot_seed=6, lo=5, hi=7, fmt="r8", params=[0, 1, 1, 15, 30], name="repetition")
    coords = "\n".join(
        [
            "error(0.1) D0",
            "error(0.2) D0 D1",
            "error(0.15) D1 D2",
            "error(0.05) D2 D3",
            "error(0.3) D3 D4",
            "error(0.1) D4",
            "error(0.12) D1 D4",
            "detector(0) D0",
            "detector D1",
            "detector(1, 2, 3, 4) D3",
            "detector(2, 1) D4",
        ]
    )
    add("coords", coords + "\n", b"11011\n", "01", [1, 1, 1, 7, 16])
    # D1 is at the average, so its boundary takes PyMatching's fixed direction.
    add("empty", "error(0.1) D0 D1\nerror(0.1) D1 D2\nerror(0.2) D1\nerror(0.1) D2\ndetector(0, 0) D0\ndetector(1, 0) D1\ndetector(2, 0) D2\n", b"000\n", "01", [1, 2, 3, 5, 10])
    add("no_matching", "error(0.1) D0 D1\nerror(0.2) D1 D2\nerror(0.3) D3\ndetector(0, 0) D0\ndetector(1, 0) D1\ndetector(2, 0) D2\ndetector(3, 0) D3\n", b"1001\n", "01", [1, 1, 1, 50, 10])
    irregular = _irregular_model(seed=7)
    # The hits in the file's order (not sorted), as PyMatching creates its detection events.
    add("irregular", irregular, b"9,2,13,6,4\n", "hits", [1, 1, 1, 6, 37])
    add("irregular_ptb64", irregular, _ptb64_first_shot([1, 5, 8, 10, 14], 15, seed=8), "ptb64", [2, 0, 1, 11, 5])
    return out


def _irregular_model(*, seed: int) -> str:
    """A 5 x 3 grid of detectors at irregular places, three coordinates each (one with five),
    its edges' probabilities all different, the first and last columns on the boundary."""
    rng = random.Random(seed)
    lines = []
    for k in range(15):
        x, y = k % 5, k // 5
        if x + 1 < 5:
            lines.append(f"error({rng.uniform(0.01, 0.2):.5g}) D{k} D{k + 1}")
        if y + 1 < 3:
            lines.append(f"error({rng.uniform(0.01, 0.2):.5g}) D{k} D{k + 5}")
        if x in (0, 4):
            lines.append(f"error({rng.uniform(0.01, 0.2):.5g}) D{k}")
    for k in range(15):
        x, y = k % 5, k // 5
        cs = [x + rng.uniform(-0.3, 0.3), y * 1.1 + rng.uniform(-0.3, 0.3), rng.uniform(0, 2)] + ([rng.uniform(-1, 1), 0.37] if k == 7 else [])
        lines.append(f"detector({', '.join(repr(c) for c in cs)}) D{k}")
    return "\n".join(lines) + "\n"


def _ptb64_first_shot(hits: list, n: int, *, seed: int) -> bytes:
    """A group of 64 shots in ptb64 whose first has ``hits``; the other 63 random."""
    rng = random.Random(seed)
    words = [rng.getrandbits(64) & ~1 | (1 if k in hits else 0) for k in range(n)]
    return b"".join(w.to_bytes(8, "little") for w in words)


def run_case(cmd: list, case: dict, tmp: pathlib.Path) -> dict:
    (tmp / "model.dem").write_text(case["dem"])
    (tmp / case["shot_name"]).write_bytes(base64.b64decode(case["shot"]))
    r = subprocess.run(cmd + case["args"], cwd=tmp, capture_output=True)
    names = sorted(os.listdir(tmp / "out")) if (tmp / "out").exists() else []
    table: dict = {}
    frames = []
    for n in names:
        lines = sorted((tmp / "out" / n).read_text().split("\n"))
        frames.append([table.setdefault(line, len(table)) for line in lines])
    return {**case, "exit": r.returncode, "stdout": r.stdout.decode(), "stderr": r.stderr.decode(), "frame_names": names, "lines": list(table), "frames": frames}


def _frames_equal(a: dict, b: dict) -> bool:
    return a["frame_names"] == b["frame_names"] and all([a["lines"][i] for i in x] == [b["lines"][i] for i in y] for x, y in zip(a["frames"], b["frames"]))


MESSAGE_MODEL = "error(0.1) D0 D1\nerror(0.1) D1\ndetector(0, 0) D0\ndetector(1, 0) D1\n"
WHEEL_EXIT = "import sys, pymatching\nsys.exit(pymatching.cli(command_line_args=sys.argv[1:]))\n"
WHEEL = "import sys, pymatching\ntry:\n    print('returned', pymatching.cli(command_line_args=sys.argv[1:]))\nexcept Exception as ex:\n    print(type(ex).__name__ + ': ' + str(ex))\n"


def message_cases() -> list:
    """Command-line misuse: (name, arguments after ``animate``), in a directory holding
    ``model.dem`` (``MESSAGE_MODEL``), ``shot.01`` (``11``) and an empty ``empty.01``."""
    good = {"--dem_in": "model.dem", "--dets_in": "shot.01", "--dets_in_format": "01", "--out_dir": "out", "--held_frames_per_event": "1", "--held_frames_at_start": "1", "--held_frames_at_end": "1", "--max_growth_between_frames": "100", "--pixels_per_unit_length": "10"}

    def argv(drop=(), **over):
        out = []
        for k, v in good.items():
            if k in drop:
                continue
            v = over.get(k.lstrip("-"), v)
            out += [k] if v is None else [k, v]
        return out

    cases = []
    for flag in good:
        cases.append((f"missing{flag}", argv(drop=(flag,))))
    for flag in PARAMS:
        cases.append((f"empty{flag}", argv(**{flag.lstrip("-"): None})))
    cases += [
        ("max_edge_weight", argv() + ["--max_edge_weight", "1000"]),
        ("unknown_flag_first", ["--bogus"] + argv()),
        ("positional", ["foo"] + argv()),
        ("pixels_3", argv(pixels_per_unit_length="3")),
        ("pixels_4", argv(pixels_per_unit_length="4")),
        ("pixels_2048", argv(pixels_per_unit_length="2048")),
        ("pixels_2049", argv(pixels_per_unit_length="2049")),
        ("held_negative", argv(held_frames_at_start="-1")),
        ("held_too_many", argv(held_frames_at_end="1000001")),
        ("held_not_int", argv(held_frames_per_event="2.5")),
        ("held_inline", argv(drop=("--held_frames_per_event",)) + ["--held_frames_per_event=0"]),
        ("format_unknown", argv(dets_in_format="xyz")),
        ("format_empty", argv(dets_in_format=None)),
        ("quiet_value", argv() + ["--quiet=1"]),
        ("dets_in_missing_file", argv(dets_in="nope.01")),
        ("dem_in_missing_file", argv(dem_in="nope.dem")),
        ("dets_in_empty", argv(dets_in="")),
        ("empty_shot_file", argv(dets_in="empty.01")),
        ("out_dir_is_a_file", argv(out_dir="model.dem")),
        ("out_dir_parent_missing", argv(out_dir="missing/out")),
        ("out_dir_slash", argv(out_dir="out2/")),
        ("double_dash", argv() + ["--", "--bogus"]),
    ]
    return cases


def main() -> None:
    import stim

    ap = argparse.ArgumentParser()
    ap.add_argument("--pymatching", required=True, help="PyMatching 2.4.0's C++ command line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert stim.__version__ == "1.16.0", stim.__version__
    for case in frame_cases(stim):
        with tempfile.TemporaryDirectory() as tmp:
            rec = run_case([sys.executable, "-I", "-c", WHEEL_EXIT], case, pathlib.Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            program = run_case([a.pymatching], case, pathlib.Path(tmp))
        assert (program["exit"], program["stderr"]) == (rec["exit"], rec["stderr"]), case["name"]
        rec["program_frames_equal"] = _frames_equal(rec, program)
        # (No name or time in the gzip header, so recording again gives the same bytes.)
        with open(out / f"{case['name']}.json.gz", "wb") as raw, gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0) as f:
            f.write(json.dumps(rec, separators=(",", ":")).encode())
        print(case["name"], "exit", rec["exit"], "frames", len(rec["frames"]), "distinct lines", len(rec["lines"]), "program's frames equal", rec["program_frames_equal"], file=sys.stderr)
    messages = []
    for name, args in message_cases():
        rec = {"name": name, "args": args}
        for who in ("binary", "wheel"):
            with tempfile.TemporaryDirectory() as tmp:
                t = pathlib.Path(tmp)
                (t / "model.dem").write_text(MESSAGE_MODEL)
                (t / "shot.01").write_text("11\n")
                (t / "empty.01").write_text("")
                cmd = [a.pymatching] if who == "binary" else [sys.executable, "-I", "-c", WHEEL]
                r = subprocess.run(cmd + ["animate"] + args, cwd=t, capture_output=True)
                frames = sum(len(os.listdir(t / d)) for d in ("out", "out2") if (t / d).is_dir())
            if who == "binary":
                rec.update(exit=r.returncode, stdout=r.stdout.decode(), stderr=r.stderr.decode(), frames=frames)
            else:
                # What pymatching.cli returned or raised; its stderr is the binary's (none where
                # the binary aborts on the exception).
                rec.update(wheel=r.stdout.decode(), wheel_stderr=r.stderr.decode(), wheel_frames=frames)
        messages.append(rec)
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, "-I", "-c", WHEEL, "nope"], cwd=tmp, capture_output=True)
        messages.append({"name": "unrecognized_command", "args": None, "wheel": r.stdout.decode(), "wheel_stderr": r.stderr.decode()})
    (out / "messages.json").write_text(json.dumps(messages, indent=1) + "\n")
    print("messages", len(messages), file=sys.stderr)


if __name__ == "__main__":
    main()
