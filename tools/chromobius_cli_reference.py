"""Chromobius 1.1.1's command line, recorded for tests/test_chromobius_cli.py.

Runs the ``chromobius`` program built from Chromobius 1.1.1's C++ sources (with the Stim and
PyMatching 2.2.2 it pins: Stim at 1320ad7, its ``src/main.cc`` and ``src/chromobius/commands``)
on each case: its arguments, its standard input and the files it starts with (written to an empty
directory it runs in). Records its exit status, standard output and error, and every file it
creates or changes.

    python tools/chromobius_cli_reference.py path/to/chromobius   # with stabilizer_qec importable

Writes tests/data/chromobius_cli_reference.json.gz: the input files (shared by the cases, as
base64) and the cases. The inputs are this package's colour-code memories and their shots, in
every Stim format; the cases cover each command, its flags and defaults, and its misuse (missing,
empty, unknown and malformed flags, files that cannot be opened, malformed records part way).
"""

from __future__ import annotations

import base64
import gzip
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

import stabilizer_qec as sq
from stabilizer_qec._shots import encode_shots

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "data" / "chromobius_cli_reference.json.gz"
FORMATS = ("01", "b8", "r8", "ptb64", "hits", "dets")
WRITABLE = ("01", "b8", "r8", "hits", "dets")


def memory(d, rounds, p, basis, shots, seed):
    c = sq.CssCode.color_code(d).memory_circuit(rounds=rounds, p=p, basis=basis, annotate_colors=True)
    dem = c.detector_error_model()
    dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return str(c), str(dem), dets, obs, dem.num_detectors, dem.num_observables


def inputs():
    files = {}
    circuit, dem, dets, obs, nd, no = memory(3, 3, 0.01, "z", 128, 5)
    files["d3.stim"] = circuit.encode()
    files["d3.dem"] = dem.encode()
    for f in FORMATS:
        files[f"d3.{f}"] = encode_shots(dets, f, num_detectors=nd)
        files[f"d3obs.{f}"] = encode_shots(np.hstack([dets, obs]), f, num_detectors=nd, num_observables=no)
        files[f"obs.{f}"] = encode_shots(obs, f, num_observables=no)
    files["short_obs.01"] = encode_shots(obs[:100], "01", num_observables=no)
    # Without its observables: predictions 0 bits wide.
    files["d3_noobs.dem"] = "\n".join(line.replace(" L0", "") for line in dem.splitlines()).encode() + b"\n"
    circuit5, dem5, dets5, obs5, nd5, no5 = memory(5, 5, 0.02, "x", 256, 6)
    files["d5.dem"] = dem5.encode()
    files["d5.b8"] = encode_shots(dets5, "b8", num_detectors=nd5)
    files["d5obs.dets"] = encode_shots(np.hstack([dets5, obs5]), "dets", num_detectors=nd5, num_observables=no5)
    files["empty"] = b""
    # Records that go wrong part way: good records, then a bad one.
    good01 = encode_shots(dets[:5], "01", num_detectors=nd)
    files["bad_short.01"] = good01 + b"0101\n"
    files["bad_char.01"] = good01 + b"2" * nd + b"\n"
    files["bad_nonl.01"] = good01 + b"0" * nd
    files["bad_long.01"] = good01 + b"0" * (nd + 1) + b"\n"
    files["crlf.01"] = good01.replace(b"\n", b"\r\n")
    files["bad_cut.b8"] = encode_shots(dets[:5], "b8", num_detectors=nd) + b"\x00"
    files["bad_cut.ptb64"] = encode_shots(dets[:64], "ptb64", num_detectors=nd) + b"\x00" * 9
    goodhits = encode_shots(dets[:5], "hits", num_detectors=nd)
    files["bad_big.hits"] = goodhits + f"1,{nd}\n".encode()
    files["bad_sep.hits"] = goodhits + b"1;2\n"
    files["bad_comma.hits"] = goodhits + b"1,\n"
    files["bad_huge.hits"] = goodhits + b"99999999999999999999\n"
    files["crlf.hits"] = goodhits.replace(b"\n", b"\r\n")
    gooddets = encode_shots(dets[:5], "dets", num_detectors=nd)
    files["bad_word.dets"] = gooddets + b"shots D1\n"
    files["bad_prefix.dets"] = gooddets + b"shot X1\n"
    files["bad_space.dets"] = gooddets + b"shot  D1\n"
    files["bad_trail.dets"] = gooddets + b"shot D1 \n"
    files["bad_noint.dets"] = gooddets + b"shot D\n"
    files["bad_big.dets"] = gooddets + f"shot D{nd}\n".encode()
    files["bad_obs.dets"] = gooddets + b"shot L0\n"
    files["loose.dets"] = b"\n  \tshot D1 D3\n\n" + gooddets
    goodr8 = encode_shots(dets[:5], "r8", num_detectors=nd)
    files["bad_past.r8"] = goodr8 + bytes([nd + 1])
    files["bad_eof.r8"] = goodr8 + b"\x00"
    # A record with a bit set past the last detector, in the padding of its last byte.
    padded = np.packbits(dets[:4], axis=1, bitorder="little")
    if nd % 8:
        padded[3, -1] |= 0x80
    files["bad_pad.b8"] = padded.tobytes()
    # Observables with padding bits set: Chromobius compares a record's first 64 bits whole.
    obs_padded = np.packbits(obs, axis=1, bitorder="little")
    if no % 8:
        obs_padded[::3, -1] |= 0x80
    files["pad_obs.b8"] = obs_padded.tobytes()
    # Models: malformed, uncoloured, a declaration without coordinates, a fault Chromobius cannot
    # split, and small ones exercising shifts, loops, ignored detectors and faults above 1/2.
    files["bad_syntax.dem"] = b"error(0.1) D0 Q1\n"
    files["bad_prob.dem"] = b"error(1.5) D0\ndetector(0, 0, 0, 0) D0\n"
    files["uncoloured.dem"] = b"error(0.1) D0 D1\ndetector(0, 0, 0) D0\ndetector(1, 0, 0) D1\n"
    files["nocoords.dem"] = b"error(0.1) D0\ndetector D0\n"
    files["shifted_uncoloured.dem"] = b"shift_detectors(1, 2, 3, 0.5) 4\ndetector(0.25, 1.123456789, 0, 0) D1\n"
    files["sevenfold.dem"] = b"error(0.1) D0 D1 D2 D3 D4 D5 D6\n" + b"".join(f"detector(0, {k}, 0, {k % 3}) D{k}\n".encode() for k in range(7))
    files["small.dem"] = (
        b"error(0.1) D0 D1 D2\nerror(0.2) D0 L0\nerror(0.05) D1\nerror(0.7) D2 D3\nerror(0.01) D3 D4 D5 L0\n"
        b"error(0.02) D4\nerror(0.03) D5\nerror(0.04) D3\n"
        b"detector(0, 0, 0, 0) D0\ndetector(1, 0, 0, 1) D1\ndetector(2, 0, 0, 2) D2\n"
        b"shift_detectors(0, 1.5) 3\ndetector(0, 0, 0, 0) D0\ndetector(1, 0, 0, 1) D1\ndetector(2, 0, 0, 2) D2\n"
    )
    files["looped.dem"] = (
        b"detector(0, 0, 0, 3) D0\ndetector(1, 0, 0, 4) D1\ndetector(2, 0, 0, 5) D2\nerror(0.1) D0 D1 D2\nerror(0.01) D0\nerror(0.01) D1 L0\nerror(0.01) D2\n"
        b"repeat 2 {\n    error(0.001) D0 D3\n    error(0.001) D1 D4\n    error(0.001) D2 D5\n    shift_detectors(0, 0, 1) 3\n"
        b"    detector(0, 0, 0, 3) D0\n    detector(1, 0, 0, 4) D1\n    detector(2, 0, 0, 5) D2\n    error(0.1) D0 D1 D2\n    error(0.01) D0\n    error(0.01) D1 L0\n    error(0.01) D2\n}\n"
    )
    files["ignored.dem"] = (
        b"error(0.1) D0 D1 D2\nerror(0.1) D0\nerror(0.1) D1\nerror(0.1) D2 L0\nerror(0.2) D2 D3\nerror(0.3) D3\n"
        b"detector(0, 0, 0, 0) D0\ndetector(1, 0, 0, 1) D1\ndetector(2, 0, 0, 2) D2\ndetector(3, 0, 0, -1) D3\n"
    )
    files["ignored.01"] = b"0000\n1000\n1100\n1110\n0001\n1111\n0010\n"
    files["small.01"] = b"000000\n100000\n111000\n001100\n000111\n110011\n011110\n"
    # Shots the Möbius matching cannot match (PyMatching's "No perfect matching"): a detector no
    # fault touches fires (the third record), or a component without a boundary has odd parity.
    files["unmatchable.dem"] = b"error(0.1) D1 L0\ndetector(0, 0, 0, 0) D0\ndetector(1, 0, 0, 1) D1\n"
    files["unmatchable.01"] = b"00\n01\n10\n01\n"
    files["unmatchable_obs.01"] = b"000\n011\n100\n011\n"
    files["odd_component.dem"] = b"error(0.1) D0 D1 D2 L0\ndetector(0, 0, 0, 0) D0\ndetector(1, 0, 0, 1) D1\ndetector(2, 0, 0, 2) D2\n"
    files["odd_component.01"] = b"000\n111\n100\n"
    return files


def cases():
    out = []

    def case(name, args, stdin=None, outputs=()):
        out.append(dict(name=name, args=list(args), stdin=stdin, outputs=list(outputs)))

    # The commands, help and misuse of the command itself.
    case("no command", [])
    for h in ("help", "--help", "-help", "-h"):
        case(f"help {h}", [h])
    case("unknown command", ["bogus", "--dem", "d3.dem"])
    # predict: every input format to every output format, appended observables, files, stdin.
    case("predict defaults", ["predict", "--dem", "d3.dem", "--in", "d3.b8"])
    for fin in FORMATS:
        for fout in WRITABLE:
            case(f"predict {fin} to {fout}", ["predict", "--dem", "d3.dem", "--in", f"d3.{fin}", "--in_format", fin, "--out_format", fout])
        case(f"predict {fin} with observables", ["predict", "--dem", "d3.dem", "--in", f"d3obs.{fin}", "--in_format", fin, "--in_includes_appended_observables"])
    case("predict to a file", ["predict", "--dem", "d3.dem", "--in", "d3.01", "--in_format", "01", "--out", "pred.b8", "--out_format", "b8"], outputs=["pred.b8"])
    case("predict from stdin", ["predict", "--dem", "d3.dem", "--in_format", "01"], stdin="d3.01")
    case("predict flags with =", ["predict", "--dem=d3.dem", "--in=d3.dets", "--in_format=dets", "--out_format=hits"])
    case("predict after --", ["predict", "--dem", "d3.dem", "--in", "d3.b8", "--", "--bogus"])
    case("predict d5", ["predict", "--dem", "d5.dem", "--in", "d5.b8", "--out_format", "dets"])
    case("predict d5 dets with observables", ["predict", "--in", "d5obs.dets", "--in_format", "dets", "--dem", "d5.dem", "--in_includes_appended_observables", "--out_format", "r8"])
    case("predict no shots", ["predict", "--dem", "d3.dem", "--in", "empty"])
    case("predict no observables", ["predict", "--dem", "d3_noobs.dem", "--in", "d3.01", "--in_format", "01"])
    case("predict no observables b8", ["predict", "--dem", "d3_noobs.dem", "--in", "d3.01", "--in_format", "01", "--out_format", "b8"])
    case("predict no observables dets", ["predict", "--dem", "d3_noobs.dem", "--in", "d3.01", "--in_format", "01", "--out_format", "dets"])
    case("predict small", ["predict", "--dem", "small.dem", "--in", "small.01", "--in_format", "01"])
    case("predict ignored", ["predict", "--dem", "ignored.dem", "--in", "ignored.01", "--in_format", "01", "--out_format", "hits"])
    case("predict crlf 01", ["predict", "--dem", "d3.dem", "--in", "crlf.01", "--in_format", "01"])
    case("predict crlf hits", ["predict", "--dem", "d3.dem", "--in", "crlf.hits", "--in_format", "hits"])
    case("predict loose dets", ["predict", "--dem", "d3.dem", "--in", "loose.dets", "--in_format", "dets"])
    # predict misuse.
    case("predict nothing", ["predict"])
    case("predict empty dem", ["predict", "--dem"])
    case("predict empty dem =", ["predict", "--dem="])
    case("predict dem then flag", ["predict", "--dem", "--in", "d3.b8"])
    case("predict missing dem file", ["predict", "--dem", "nope.dem"])
    case("predict unknown flag", ["predict", "--dem", "d3.dem", "--bogus"])
    case("predict unknown flag first", ["predict", "--bogus", "--dem", "d3.dem"])
    case("predict unknown flag with value", ["predict", "--dem", "d3.dem", "--obs_in", "obs.01"])
    case("predict stray word", ["predict", "--dem", "d3.dem", "--in", "d3.b8", "stray"])
    case("predict bad in_format", ["predict", "--dem", "d3.dem", "--in", "d3.b8", "--in_format", "xyz"])
    case("predict empty in_format", ["predict", "--dem", "d3.dem", "--in", "d3.b8", "--in_format"])
    case("predict bad out_format", ["predict", "--dem", "d3.dem", "--in", "d3.b8", "--out_format", "B8"])
    case("predict ptb64 out", ["predict", "--dem", "d3.dem", "--in", "d3.b8", "--out_format", "ptb64"])
    case("predict bool with value", ["predict", "--dem", "d3.dem", "--in", "d3.b8", "--in_includes_appended_observables=yes"])
    case("predict missing in file", ["predict", "--dem", "d3.dem", "--in", "nope.b8"])
    case("predict empty in", ["predict", "--dem", "d3.dem", "--in", "--out_format", "01"])
    case("predict unwritable out", ["predict", "--dem", "d3.dem", "--in", "d3.b8", "--out", "nodir/pred.01"])
    case("predict out is a directory", ["predict", "--dem", "d3.dem", "--in", "d3.b8", "--out", "."])
    case("predict out made before dem fails", ["predict", "--dem", "nope.dem", "--in", "d3.b8", "--out", "made.01"], outputs=["made.01"])
    case("predict out kept after a bad record", ["predict", "--dem", "d3.dem", "--in", "bad_char.01", "--in_format", "01", "--out", "part.01"], outputs=["part.01"])
    for name in ("bad_short.01", "bad_char.01", "bad_nonl.01", "bad_long.01", "bad_cut.b8", "bad_cut.ptb64", "bad_big.hits", "bad_sep.hits", "bad_comma.hits",
                 "bad_word.dets", "bad_prefix.dets", "bad_space.dets", "bad_trail.dets", "bad_noint.dets", "bad_big.dets", "bad_obs.dets", "bad_past.r8", "bad_eof.r8"):
        case(f"predict {name}", ["predict", "--dem", "d3.dem", "--in", name, "--in_format", name.split(".")[1]])
    case("predict huge hit", ["predict", "--dem", "d3.dem", "--in", "bad_huge.hits", "--in_format", "hits"])
    case("predict negative-looking value", ["predict", "--dem", "d3.dem", "--in", "-5"])
    case("predict flag given twice", ["predict", "--dem", "nope.dem", "--dem", "d3.dem", "--in", "d3.b8"])
    case("predict flag prefix", ["predict", "--de", "d3.dem"])
    case("predict flags after --", ["predict", "--", "--dem", "d3.dem"])
    case("predict dem is a directory", ["predict", "--dem", ".", "--in", "d3.b8"])
    case("predict in is a directory", ["predict", "--dem", "d3.dem", "--in", "."])
    case("help ignores the rest", ["help", "--bogus"])
    case("predict padding bit", ["predict", "--dem", "d3.dem", "--in", "bad_pad.b8"])
    case("predict malformed dem", ["predict", "--dem", "bad_syntax.dem", "--in", "d3.b8"])
    case("predict probability above 1", ["predict", "--dem", "bad_prob.dem", "--in", "d3.b8"])
    case("predict uncoloured dem", ["predict", "--dem", "uncoloured.dem", "--in", "d3.b8"])
    case("predict detector without coordinates", ["predict", "--dem", "nocoords.dem", "--in", "d3.b8"])
    case("predict shifted uncoloured", ["predict", "--dem", "shifted_uncoloured.dem", "--in", "empty"])
    # benchmark: its statistics (the timings vary), its observables from either place.
    case("benchmark appended", ["benchmark", "--dem", "d3.dem", "--in", "d3obs.01", "--in_includes_appended_observables"])
    case("benchmark obs file", ["benchmark", "--dem", "d3.dem", "--in", "d3.b8", "--in_format", "b8", "--obs_in", "obs.b8", "--obs_in_format", "b8"])
    case("benchmark obs file default formats", ["benchmark", "--dem", "d3.dem", "--in", "d3.01", "--obs_in", "obs.01"])
    for f in FORMATS:
        case(f"benchmark obs {f}", ["benchmark", "--dem", "d3.dem", "--in", f"d3.{f}", "--in_format", f, "--obs_in", f"obs.{f}", "--obs_in_format", f])
    case("benchmark to a file", ["benchmark", "--dem", "d5.dem", "--in", "d5obs.dets", "--in_format", "dets", "--in_includes_appended_observables", "--out", "stats.txt"], outputs=["stats.txt"])
    case("benchmark from stdin", ["benchmark", "--dem", "d3.dem", "--in_includes_appended_observables"], stdin="d3obs.01")
    case("benchmark no shots", ["benchmark", "--dem", "d3.dem", "--in", "empty", "--in_includes_appended_observables"])
    case("benchmark nothing", ["benchmark"])
    case("benchmark neither observables", ["benchmark", "--dem", "d3.dem", "--in", "d3.01"])
    case("benchmark neither observables, no dem", ["benchmark", "--in", "d3.01"])
    case("benchmark short obs", ["benchmark", "--dem", "d3.dem", "--in", "d3.01", "--obs_in", "short_obs.01"])
    case("benchmark empty obs_in", ["benchmark", "--dem", "d3.dem", "--in", "d3.01", "--obs_in"])
    case("benchmark missing obs file", ["benchmark", "--dem", "d3.dem", "--in", "d3.01", "--obs_in", "nope.01"])
    case("benchmark unknown flag", ["benchmark", "--dem", "d3.dem", "--in", "d3.01", "--out_format", "01"])
    case("benchmark bad obs_in_format", ["benchmark", "--dem", "d3.dem", "--in", "d3.01", "--obs_in", "obs.01", "--obs_in_format", "x"])
    case("benchmark bad record", ["benchmark", "--dem", "d3.dem", "--in", "bad_char.01", "--in_includes_appended_observables"])
    case("benchmark bad obs record", ["benchmark", "--dem", "d3.dem", "--in", "d3.01", "--obs_in", "bad_char.01"])
    case("benchmark unwritable out", ["benchmark", "--dem", "d3.dem", "--in", "d3obs.01", "--in_includes_appended_observables", "--out", "nodir/stats.txt"])
    case("benchmark obs padding bits", ["benchmark", "--dem", "d3.dem", "--in", "d3.b8", "--in_format", "b8", "--obs_in", "pad_obs.b8", "--obs_in_format", "b8"])
    case("benchmark both observables", ["benchmark", "--dem", "d3.dem", "--in", "d3obs.01", "--in_includes_appended_observables", "--obs_in", "obs.01"])
    case("benchmark padding bit", ["benchmark", "--dem", "d3.dem", "--in", "bad_pad.b8", "--in_format", "b8", "--obs_in", "obs.01"])
    case("benchmark huge hit", ["benchmark", "--dem", "d3.dem", "--in", "bad_huge.hits", "--in_format", "hits", "--obs_in", "obs.01"])
    # describe_decoder.
    case("describe d3", ["describe_decoder", "--in", "d3.dem"])
    case("describe d3 circuit", ["describe_decoder", "--circuit", "d3.stim"])
    case("describe circuit wins", ["describe_decoder", "--in", "uncoloured.dem", "--circuit", "d3.stim"])
    case("describe from stdin", ["describe_decoder"], stdin="small.dem")
    case("describe to a file", ["describe_decoder", "--in", "looped.dem", "--out", "described.txt"], outputs=["described.txt"])
    for name in ("small.dem", "looped.dem", "ignored.dem", "d3_noobs.dem", "d5.dem", "empty"):
        case(f"describe {name}", ["describe_decoder", "--in", name])
    case("describe unknown flag", ["describe_decoder", "--dem", "d3.dem"])
    case("describe empty circuit flag", ["describe_decoder", "--circuit"])
    case("describe missing circuit", ["describe_decoder", "--circuit", "nope.stim"])
    case("describe missing in", ["describe_decoder", "--in", "nope.dem"])
    case("describe empty out", ["describe_decoder", "--in", "d3.dem", "--out="])
    case("describe unwritable out", ["describe_decoder", "--in", "d3.dem", "--out", "nodir/x.txt"])
    case("describe out made before the model fails", ["describe_decoder", "--in", "uncoloured.dem", "--out", "made.txt"], outputs=["made.txt"])
    case("describe uncoloured", ["describe_decoder", "--in", "uncoloured.dem"])
    case("describe no coordinates", ["describe_decoder", "--in", "nocoords.dem"])
    case("describe shifted uncoloured", ["describe_decoder", "--in", "shifted_uncoloured.dem"])
    case("describe malformed", ["describe_decoder", "--in", "bad_syntax.dem"])
    case("describe sevenfold", ["describe_decoder", "--in", "sevenfold.dem"])
    case("predict sevenfold", ["predict", "--dem", "sevenfold.dem", "--in", "empty"])
    case("describe in is a directory", ["describe_decoder", "--in", "."])
    case("describe circuit is a directory", ["describe_decoder", "--circuit", "."])
    # Shots with no perfect matching: PyMatching's message, after the predictions before them.
    case("predict unmatchable", ["predict", "--dem", "unmatchable.dem", "--in", "unmatchable.01", "--in_format", "01"])
    case("predict unmatchable to a file", ["predict", "--dem", "unmatchable.dem", "--in", "unmatchable.01", "--in_format", "01", "--out", "part.01"], outputs=["part.01"])
    case("predict odd component", ["predict", "--dem", "odd_component.dem", "--in", "odd_component.01", "--in_format", "01"])
    case("benchmark unmatchable", ["benchmark", "--dem", "unmatchable.dem", "--in", "unmatchable_obs.01", "--in_includes_appended_observables"])
    case("benchmark unmatchable obs file", ["benchmark", "--dem", "unmatchable.dem", "--in", "unmatchable.01", "--obs_in", "obs.01"])
    return out


def run(binary, files, c):
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        for name, data in files.items():
            (d / name).write_bytes(data)
        stdin = files[c["stdin"]] if c["stdin"] else b""
        r = subprocess.run([binary, *c["args"]], cwd=d, input=stdin, capture_output=True, timeout=600)
        made = {}
        for p in sorted(d.rglob("*")):
            name = p.relative_to(d).as_posix()
            if p.is_file() and (name not in files or p.read_bytes() != files[name]):
                made[name] = base64.b64encode(p.read_bytes()).decode()
        assert sorted(made) == sorted(c["outputs"]), (c["name"], sorted(made))
        return dict(returncode=r.returncode, stdout=base64.b64encode(r.stdout).decode(), stderr=r.stderr.decode("utf-8", "replace"), files=made)


def main():
    binary = str(pathlib.Path(sys.argv[1]).resolve())
    files = inputs()
    recorded = []
    for c in cases():
        recorded.append(dict(c, **run(binary, files, c)))
    data = dict(
        reference="chromobius 1.1.1 (with Stim 1320ad7 and PyMatching 2.2.2), built from source",
        platform=sys.platform,
        files={k: base64.b64encode(v).decode() for k, v in files.items()},
        cases=recorded,
    )
    OUT.write_bytes(gzip.compress(json.dumps(data, indent=0, sort_keys=True).encode(), mtime=0))
    print(len(recorded), "cases;", sum(r["returncode"] != 0 for r in recorded), "fail;", OUT)


if __name__ == "__main__":
    main()
