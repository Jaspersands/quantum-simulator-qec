"""``python -m stabilizer_qec.tesseract_decoder``: Tesseract's ``tesseract`` program
(src/tesseract_main.cc). Tesseract's wheel has no binary, and building it needs Bazel, so its
flags are checked against Tesseract's Python API: data/tesseract_cli_reference.json holds what
Tesseract's 2026-10-13 build computes, through its Python API, for each flag set on stored shots
and the model's text (predictions, low-confidence flags, costs, counts, the --dem-out model,
the multi-pass decoder's answers and the verbose log); the text is checked against
tesseract_main.cc and p-ranav/argparse 3.1, which it parses its flags with."""

from __future__ import annotations

import errno
import gc
import io
import json
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec.tesseract_decoder.__main__ import main
from stabilizer_qec.tesseract_decoder.tesseract import TesseractConfig, TesseractDecoder
from stabilizer_qec.tesseract_decoder.utils import DetectorOrderMethod
from tesseract_platform_helpers import assert_dem_text, assert_search_log

with open(os.path.join(os.path.dirname(__file__), "data", "tesseract_cli_reference.json")) as f:
    REF = json.load(f)
MODELS = {m["name"]: m for m in REF["models"]}
TERMINATE = "libc++abi: terminating due to uncaught exception of type std::invalid_argument: " if sys.platform == "darwin" else None


def run(args, capsys):
    code = main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


def failure(message: str, kind: str = "std::invalid_argument") -> str:
    """What Tesseract's program prints when an exception leaves main: the C++ runtime's words."""
    if sys.platform == "darwin":
        return f"libc++abi: terminating due to uncaught exception of type {kind}: {message}\n"
    return f"terminate called after throwing an instance of '{kind}'\n  what():  {message}\n"


def write_model(tmp_path, model):
    (tmp_path / "c.stim").write_text(model["circuit"])
    (tmp_path / "c.dem").write_text(model["dem"])
    (tmp_path / "shots.b8").write_bytes(bytes.fromhex(model["shots_b8"]))
    (tmp_path / "orders.json").write_text(json.dumps(model["orders_json"]))


def final_line(out: str) -> dict:
    line = out.splitlines()[-1]
    m = re.fullmatch(r"num_shots = (\d+) num_low_confidence = (\d+)(?: num_errors = (\d+))? total_time_seconds = (\S+)", line)
    assert m, line
    return {"num_shots": int(m[1]), "num_low_confidence": int(m[2]), "num_errors": None if m[3] is None else int(m[3]), "time": float(m[4])}


CASES = [(m["name"], k) for m in REF["models"] for k in range(len(m["cases"]))]


@pytest.mark.parametrize("source", ["dem", "circuit"])
@pytest.mark.parametrize("name,k", CASES)
def test_cli_equals_tesseracts_python_api(name, k, source, tmp_path, capsys):
    # (The fixture is what Tesseract's program computes with --dem c.dem, the model's text; its
    # predictions, counts and --dem-out are the same from the circuit's model.)
    model = MODELS[name]
    case = model["cases"][k]
    write_model(tmp_path, model)
    args = [a.replace("ORDERS", str(tmp_path / "orders.json")) for a in case["args"]]
    models = ["--circuit", tmp_path / "c.stim"] + (["--dem", tmp_path / "c.dem"] if source == "dem" else [])
    code, out, err = run(
        [*models, "--in", tmp_path / "shots.b8", "--in-format", "b8", "--in-includes-appended-observables",
         "--out", tmp_path / "p.01", "--out-format", "01", "--dem-out", tmp_path / "out.dem", "--threads", "2", "--print-stats", *args],
        capsys,
    )
    assert (code, err) == (0, "")
    lines = (tmp_path / "p.01").read_text().splitlines()
    nobs = sq.DetectorErrorModel(model["dem"]).num_observables
    assert lines == ["".join("1" if o in shot[0] else "0" for o in range(nobs)) for shot in case["shots"]]
    stats = out.splitlines()
    assert len(stats) == 2 * len(case["shots"]) + 1
    low = errors = 0
    for s, (predicted, is_low, cost) in enumerate(case["shots"]):
        low += is_low
        assert re.fullmatch(rf"num_shots = {s + 1} num_low_confidence = {low} num_errors = \d+ total_time_seconds = \S+", stats[2 * s])
        if source == "dem":
            assert stats[2 * s + 1] == "cost = %.16g" % cost
        else:
            # (This package's model of the circuit can differ from Stim's in its probabilities'
            # last digits, and its costs with them: test_cli_equals_python_api checks the costs
            # from --circuit exactly against this package's decoder.)
            assert re.fullmatch(r"cost = [0-9.e+-]+", stats[2 * s + 1])
    got = final_line(out)
    assert (got["num_shots"], got["num_low_confidence"], got["num_errors"]) == (model["num_shots"], case["num_low_confidence"], case["num_errors"])
    assert_dem_text((tmp_path / "out.dem").read_text(), case["dem_out"])


def test_cli_dem_from_circuit_is_tesseracts(tmp_path):
    # (--circuit alone builds the model Tesseract's analyzer builds: folded loops, gauge
    # detectors allowed, disjoint errors approximated.)
    for model in REF["models"]:
        dem = sq.Circuit(model["circuit"]).detector_error_model(allow_gauge_detectors=True, approximate_disjoint_errors=True)
        assert_dem_text(str(dem), model["dem"])


def test_cli_equals_python_api(tmp_path, capsys):
    model = MODELS["surface_d3"]
    write_model(tmp_path, model)
    args = ["--circuit", tmp_path / "c.stim", "--sample-num-shots", 150, "--sample-seed", 5, "--det-order-bfs", "--num-det-orders", 2, "--det-order-seed", 9,
            "--beam", 6, "--pqlimit", 30000, "--no-revisit-dets", "--out", tmp_path / "p.b8", "--out-format", "b8", "--threads", 3, "--print-stats"]
    code, out, err = run(args, capsys)
    assert (code, err) == (0, "")
    circuit = sq.Circuit(model["circuit"])
    dets, obs = circuit.compile_detector_sampler(seed=5).sample(150, separate_observables=True)
    dem = circuit.detector_error_model(allow_gauge_detectors=True, approximate_disjoint_errors=True)
    decoder = TesseractDecoder(TesseractConfig(dem, det_beam=6, pqlimit=30000, no_revisit_dets=True, num_det_orders=2, det_order_method=DetectorOrderMethod.BFS, seed=9))
    predicted = decoder.decode_batch(dets)
    got = np.unpackbits(np.frombuffer((tmp_path / "p.b8").read_bytes(), dtype=np.uint8).reshape(150, -1), axis=1, count=dem.num_observables, bitorder="little")
    np.testing.assert_array_equal(got, predicted)
    low, costs = [], []
    for row in dets:
        errors = decoder.decode_to_errors(row)
        low.append(decoder.low_confidence_flag)
        costs.append("cost = %.16g" % decoder.cost_from_errors(errors))
    low = np.array(low)
    assert out.splitlines()[1:-1:2] == costs
    assert "cost = 0" in costs
    wrong = (~low & (predicted != obs).any(axis=1)).sum()
    stats = final_line(out)
    assert (stats["num_shots"], stats["num_low_confidence"], stats["num_errors"]) == (150, low.sum(), wrong)
    # The same seed, the same shots.
    code, out2, _ = run(args, capsys)
    assert final_line(out2)["num_errors"] == wrong


@pytest.mark.parametrize("run_index", [0, 1])
def test_cli_multipass_equals_tesseracts(run_index, tmp_path, capsys):
    model = MODELS["surface_d3"]
    mp = model["multipass"]["runs"][run_index]
    write_model(tmp_path, model)
    (tmp_path / "a.dem").write_text(model["multipass"]["dem"])
    strategy = "causal" if "Causal" in mp["strategy"] else "static"
    code, out, err = run(
        ["--dem", tmp_path / "a.dem", "--in", tmp_path / "shots.b8", "--in-format", "b8", "--in-includes-appended-observables", "--out", "-", "--out-format", "hits",
         "--beam", 5, "--no-revisit-dets", "--pqlimit", 100000, "--multipass", "--num-passes", mp["passes"], "--multipass-strategy", strategy, "--print-multipass-plan"],
        capsys,
    )
    assert code == 0
    lines = out.splitlines()
    assert lines[:-1] == [",".join(map(str, p)) for p in mp["predictions"]]
    got = final_line(out)
    assert (got["num_low_confidence"], got["num_errors"]) == (mp["num_low_confidence"], mp["num_errors"])
    assert err.startswith(f"Multi-pass execution plan\nstrategy: {strategy}\npasses: {mp['passes']}\nmonolithic input DEM: detectors=24, error_mechanisms=")
    assert re.search(r"\ncomponents: 2\n  component 0: label=0, active_detectors=\d+, decoder_detectors=24, observable=(yes|no), error_mechanisms=\d+, "
                     r"average_active_detector_degree=[0-9.e+-]+, sparsify_reactivate_limit=disabled\n", err)
    assert err.endswith("]\n") and "\nschedule:\n  pass 1: [" in err
    # The plan's dependencies and schedule: the schedule is the one Tesseract builds for the
    # strategy asked for (build_static_schedule: every component in every pass;
    # build_causal_schedule: the components that touch an observable last, before them those
    # with a rule into the next pass), from the plan's own components and dependencies, and
    # both are those of the Python API's decoder for that strategy.
    observable = [m[1] == "yes" for m in re.finditer(r"\n  component \d+: .*?observable=(yes|no),", err)]
    dependencies = {(int(m[1]), int(m[2])): int(m[3]) for m in re.finditer(r"\n  component (\d+) -> component (\d+): (\d+) rules", err)}
    schedule = [[int(c) for c in m[2].split(", ") if c] for m in re.finditer(r"\n  pass (\d+): \[([0-9, ]*)\]", err)]
    # (Reweighting rules exist only for a second pass to use.)
    assert len(observable) == 2 and bool(dependencies) == (mp["passes"] == 2) and len(schedule) == mp["passes"]
    assert schedule == tesseracts_schedule(strategy, mp["passes"], observable, dependencies)
    # (On this model the two strategies' schedules differ, so a swap cannot pass.)
    other = "static" if strategy == "causal" else "causal"
    assert tesseracts_schedule(other, mp["passes"], observable, dependencies) != schedule
    from stabilizer_qec.tesseract_decoder import SchedulingStrategy
    from stabilizer_qec.tesseract_decoder._multi_pass import MultiPassTesseractDecoder

    config = TesseractConfig(sq.DetectorErrorModel(model["multipass"]["dem"]), det_beam=5, no_revisit_dets=True, pqlimit=100000)
    api = MultiPassTesseractDecoder(config, mp["passes"], (), int(getattr(SchedulingStrategy, strategy.capitalize())))
    assert api.pass_schedule == schedule and api.affects_observable == observable
    counts: dict = {}
    for source, rules in enumerate(api.reweight_rules):
        for entries in rules.values():
            for rule in entries:
                counts[(source, rule[0])] = counts.get((source, rule[0]), 0) + 1
    assert counts == dependencies


def tesseracts_schedule(strategy, passes, observable, dependencies):
    """Tesseract's MultiPassTesseractDecoder::build_static_schedule and build_causal_schedule
    (src/multi_pass/multi_pass_tesseract_decoder.cc), from a plan's components (whether each
    affects an observable) and its dependencies (source component, target component)."""
    n = len(observable)
    if strategy == "static":
        return [list(range(n)) for _ in range(passes)]
    sets = [set() for _ in range(passes)]
    sets[-1] = {c for c in range(n) if observable[c]}
    for p in range(passes - 2, -1, -1):
        sets[p] = {source for (source, target) in dependencies if target in sets[p + 1]}
    return [sorted(s) for s in sets]


def test_cli_stats_out_json_keys(tmp_path, capsys):
    model = MODELS["repetition_d3"]
    write_model(tmp_path, model)
    base = ["--dem", tmp_path / "c.dem", "--in", tmp_path / "shots.b8", "--in-format", "b8", "--in-includes-appended-observables", "--threads", 1]
    code, out, err = run(base + ["--stats-out", tmp_path / "stats.json"], capsys)
    assert (code, err) == (0, "")
    text = (tmp_path / "stats.json").read_text()
    stats = json.loads(text)
    assert list(stats) == [
        "beam_climbing", "circuit_path", "dem_path", "det_beam", "det_order_seed", "det_penalty", "detector_orders_paths", "max_errors", "multipass",
        "multipass_num_passes", "multipass_strategy", "no_revisit_dets", "num_det_orders", "num_det_orders_per_generated_source", "num_errors",
        "num_low_confidence", "num_shots", "num_threads", "pqlimit", "sample_num_shots", "sample_seed", "sparsify_base_degree", "sparsify_errors",
        "sparsify_max_degree", "sparsify_reactivate_limit", "total_time_seconds",
    ]
    assert text.endswith("}\n") and ": " not in text and ", " not in text  # (nlohmann's compact dump)
    assert text.startswith('{"beam_climbing":false,"circuit_path":"","dem_path":' + json.dumps(str(tmp_path / "c.dem")) + ',"det_beam":65535,"det_order_seed":518278944,"det_penalty":0.0,"detector_orders_paths":[],"max_errors":18446744073709551615,')
    assert stats["num_shots"] == model["num_shots"] and stats["num_errors"] == model["cases"][0]["num_errors"] and stats["pqlimit"] == 2**64 - 1
    assert stats["multipass_strategy"] == "causal" and stats["multipass_num_passes"] == 2 and stats["num_det_orders"] == 1
    assert 0 <= stats["sample_seed"] < 2**32 and stats["sparsify_reactivate_limit"] == -1
    assert final_line(out)["num_shots"] == model["num_shots"]
    # "-": the JSON on standard output, and no final line.
    code, out, err = run(base + ["--stats-out", "-", "--det-penalty", "0.25", "--sample-seed", 7], capsys)
    assert (code, err) == (0, "")
    assert out.count("\n") == 1 and json.loads(out)["det_penalty"] == 0.25 and '"det_penalty":0.25,' in out and '"sample_seed":7,' in out
    # Without observables: num_errors is null and the final line leaves it out.
    (tmp_path / "dets.b8").write_bytes(b"")
    code, out, err = run(["--dem", tmp_path / "c.dem", "--in", tmp_path / "dets.b8", "--in-format", "b8", "--stats-out", tmp_path / "s2.json"], capsys)
    assert json.loads((tmp_path / "s2.json").read_text())["num_errors"] is None
    assert re.fullmatch(r"num_shots = 0 num_low_confidence = 0 total_time_seconds = 0\n", out)


def test_cli_max_errors_and_shot_range(tmp_path, capsys):
    model = MODELS["surface_d3"]
    case = model["cases"][0]
    write_model(tmp_path, model)
    base = ["--dem", tmp_path / "c.dem", "--in", tmp_path / "shots.b8", "--in-format", "b8", "--in-includes-appended-observables", *case["args"]]
    wrong = [k for k, (p, low, _) in enumerate(case["shots"]) if not low and p != np.flatnonzero(np.unpackbits(np.frombuffer(bytes.fromhex(model["shots_b8"]), dtype=np.uint8).reshape(model["num_shots"], -1), axis=1, bitorder="little")[k, 24:25]).tolist()]
    assert len(wrong) == case["num_errors"] >= 2
    # (Decoding stops at the shot that reaches the limit.)
    code, out, _ = run(base + ["--max-errors", 2], capsys)
    assert final_line(out)["num_shots"] == wrong[1] + 1 and final_line(out)["num_errors"] == 2
    code, out, _ = run(base + ["--shot-range-begin", 10, "--shot-range-end", 40], capsys)
    assert final_line(out)["num_shots"] == 30
    assert final_line(out)["num_errors"] == sum(1 for k in wrong if 10 <= k < 40)
    code, out, err = run(base + ["--shot-range-end", 500], capsys)
    assert code != 0 and err == failure(f"Shot range end is past end of shots array (size {model['num_shots']}).")


def test_cli_verbose(tmp_path, capsys):
    v = REF["verbose"]
    (tmp_path / "v.dem").write_text(v["dem"])
    (tmp_path / "v.01").write_text("101\n")
    code, out, err = run(["--dem", tmp_path / "v.dem", "--in", tmp_path / "v.01", "--in-format", "01", "--verbose"], capsys)
    assert (code, err) == (0, "")
    head = "Detector D0 coordinate (0, 0)\nDetector D1 coordinate (1, 0)\nDetector D2 coordinate (2, 0)\n"
    assert out.startswith(head)
    log, stats = out[len(head):].rsplit("num_shots = ", 1)
    if sys.platform == "darwin":
        assert log == v["log"]
    # The four error descriptions stay exact; the native heap/order may change
    # the subsequent search (src/search.rs), whose parity/cost/log structure do not.
    prefix = v["log"].split("len(pq)", 1)[0]
    assert log.startswith(prefix)
    decoder = TesseractDecoder(TesseractConfig(sq.DetectorErrorModel(v["dem"])))
    assert_search_log(log[len(prefix):], v["log"][len(prefix):], decoder._flat, [0, 2])
    assert re.fullmatch(r"1 num_low_confidence = 0 total_time_seconds = \S+\n", stats)


def test_cli_help_and_version(capsys):
    code, out, err = run(["--help"], capsys)
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0].startswith("Usage: tesseract [--help] [--version] [--circuit VAR] [--dem VAR] [--no-merge-errors] [--num-det-orders N] [--det-order-bfs]")
    assert "[--detector-orders FILE]... [--sample-num-shots VAR]" in lines[0]
    assert "[--in-format 01/b8/dets/hits/ptb64/r8]" in lines[0]
    assert (lines[1], lines[2]) == ("", "Optional arguments:")
    width = len("  --in-includes-appended-observables, --in_includes_appended_observables")
    assert lines[3] == "  -h, --help".ljust(width) + "  shows help message and exits "
    assert "  --num-det-orders".ljust(width) + "  Number of orders generated by each selected detector-order method [nargs=0..1] [default: 1]" in lines
    assert "  --detector-orders FILE".ljust(width) + "  Add detector-ID permutations from a JSON file. May be repeated and combined with generated detector-order methods. [may be repeated]" in lines
    assert "  --beam".ljust(width) + "  Beam to use for truncation (default = infinity) [nargs=0..1] [default: 65535]" in lines
    assert "  --pqlimit".ljust(width) + "  Maximum size of the priority queue (default = infinity) [nargs=0..1] [default: 18446744073709551615]" in lines
    assert "  --sparsify-base-degree K".ljust(width) + "  Maximum detector degree for mandatory errors. Errors with degree <= K are always enabled for every shot. " in lines
    assert "  --multipass-strategy, --multipass_strategy".ljust(width) + '  Multi-pass scheduling strategy: static or causal (default = causal). Note: static scheduling is experimental and was never systematically benchmarked. [nargs=0..1] [default: "causal"]' in lines
    assert out.endswith("\n") and not out.endswith("\n\n")
    code, out, err = run(["--version"], capsys)
    assert (code, out, err) == (0, "1.0\n", "")


def test_cli_flag_errors(tmp_path, capsys):
    model = MODELS["repetition_d3"]
    write_model(tmp_path, model)
    dem, shots = tmp_path / "c.dem", tmp_path / "shots.b8"
    nd = sq.DetectorErrorModel(model["dem"]).num_detectors
    _, help_text, _ = run(["--help"], capsys)
    # (Tesseract's --sample-seed defaults to a fresh random number, which its help shows.)
    seedless = lambda text: re.sub(r"(--sample-seed .*\[default: )\d+\]", r"\1N]", text)  # noqa: E731

    # Parsing errors: the message, then the help, on standard error; exit status 1.
    for args, message in [
        (["--frobnicate"], "Unknown argument: --frobnicate"),
        (["--dem"], "Too few arguments for '--dem'."),
        (["--dem", "--circuit", "x"], "Too few arguments for '--dem'."),
        (["stray"], "Zero positional arguments expected, did you mean --circuit VAR"),
        (["--circuit", "c", "stray"], "Zero positional arguments expected, did you mean --dem VAR"),
        (["--dem", "a", "--dem", "b"], "Duplicate argument --dem"),
        (["--in-format", "b8", "--in_format", "01"], "Duplicate argument --in_format"),
        (["--sample-num-shots", "-3"], "pattern '-3' not found"),
        (["--sample-num-shots", "12x"], "pattern '12x' does not match to the end"),
        (["--sample-num-shots", "99999999999999999999999"], "'99999999999999999999999' not representable"),
        (["--det-penalty", "x1"], "pattern 'x1' does not match to the end"),
        (["--det-penalty", "+1"], "pattern '+1' not found"),
        (["--sparsify-base-degree", "abc"], "Failed to parse 'abc' as decimal integer: pattern 'abc' not found"),
        (["--multipass-strategy", "eager"], "Invalid multi-pass scheduling strategy 'eager'; expected 'static' or 'causal'."),
    ]:
        code, out, err = run(args, capsys)
        assert (code, out, seedless(err)) == (1, "", seedless(message + "\n" + help_text)), args

    # Checks after parsing (an exception out of main: the C++ runtime's message, and abort).
    base = ["--dem", dem, "--in", shots, "--in-format", "b8"]
    for args, message in [
        ([], "Must provide at least one of --circuit or --dem"),
        (["--circuit", ""], "Must provide at least one of --circuit or --dem"),
        (["--dem", dem], "Requires exactly 1 source of shots."),
        (["--dem", dem, "--in", shots], "If --in is provided, must also specify --in-format."),
        (base + ["--out", "x"], "If --out is provided, must also specify --out-format."),
        (["--dem", dem, "--in", shots, "--in-format", "b9"], "Invalid format: b9"),
        (base + ["--out", "-", "--out-format", "bin"], "Invalid format: bin"),
        (["--dem", dem, "--obs_in", shots, "--obs-in-format", "01", "--sample-num-shots", "3"], "Cannot load observable flips without a corresponding detection event data file."),
        (["--dem", dem, "--in", shots, "--in-format", "b8", "--sample-num-shots", "3"], "Requires exactly 1 source of shots."),
        (base + ["--threads", "0"], "--threads must be at least 1."),
        (base + ["--threads", "1001"], "There is a maximum limit of 1000 threads imposed to avoid accidentally overloading a host. You specified 1001 threads."),
        (base + ["--num-passes", "3"], "--num-passes must be 1 or 2."),
        (base + ["--print-multipass-plan"], "--print-multipass-plan requires --multipass."),
        (base + ["--multipass", "--dem-out", "x"], "--dem-out is not supported when --multipass is enabled."),
        (base + ["--shot-range-begin", "5", "--shot-range-end", "2"], "Provided shot range must have end >= begin."),
        (["--dem", dem, "--sample-num-shots", "3"], "Cannot sample shots without a circuit."),
        (base + ["--beam-climbing"], "Beam climbing requires a finite beam"),
        (base + ["--sparsify-base-degree", "2"], "Cannot use --sparsify-base-degree, --sparsify-max-degree, or --sparsify-reactivate-limit without --sparsify-errors"),
        (base + ["--sparsify-errors"], "Must specify --sparsify-base-degree when --sparsify-errors is enabled."),
        (base + ["--sparsify-errors", "--sparsify-base-degree", "0"], "--sparsify-base-degree must be > 0."),
        (base + ["--sparsify-errors", "--sparsify-base-degree", "3", "--sparsify-max-degree", "2"], "--sparsify-max-degree must be >= --sparsify-base-degree."),
        (base + ["--detector-orders", tmp_path / "orders.json", "--num-det-orders", "2"],
         "--num-det-orders and --det-order-seed only apply to generated detector orders. Select --det-order-bfs, --det-order-index, or --det-order-coordinate to combine generated orders with --detector-orders files."),
        (base + ["--num-det-orders", "0"], "--num-det-orders must be at least 1."),
        (["--dem", tmp_path / "missing.dem", "--in", shots, "--in-format", "b8"], f"Could not open the file: {tmp_path / 'missing.dem'}"),
        (["--dem", dem, "--in", tmp_path / "missing.b8", "--in-format", "b8"], f"Could not open the file: {tmp_path / 'missing.b8'}"),
        (base + ["--obs_in", tmp_path / "missing.01", "--obs-in-format", "01"], f"Could not open the file: {tmp_path / 'missing.01'}"),
        (base + ["--out", tmp_path / "no" / "such" / "dir.01", "--out-format", "01"], f"Failed to open {tmp_path / 'no' / 'such' / 'dir.01'}"),
        (base + ["--dem-out", tmp_path / "no" / "dir.dem"], f"Failed to open {tmp_path / 'no' / 'dir.dem'}"),
        (base + ["--out", tmp_path / "p.ptb64", "--out-format", "ptb64"], "SAMPLE_FORMAT_PTB64 incompatible with SingleMeasurementRecord"),
        (base + ["--detector-orders", tmp_path / "missing.json"], f"Could not open the file: {tmp_path / 'missing.json'}"),
    ]:
        code, out, err = run(args, capsys)
        assert code != 0 and err == failure(message), (args, err)
    # --obs_in without --obs-in-format passes the checks; once the file is open, Stim's format
    # map (a std::map) has no "" (map::at).
    code, out, err = run(base + ["--obs_in", tmp_path / "missing.01"], capsys)
    assert code != 0 and err == failure(f"Could not open the file: {tmp_path / 'missing.01'}"), err
    code, out, err = run(base + ["--obs_in", shots], capsys)
    assert code != 0 and err == failure("map::at:  key not found" if sys.platform == "darwin" else "map::at", "std::out_of_range"), err
    # --out-format ptb64: Stim's writer refuses it after Tesseract's program has created (and
    # emptied) the file, which is closed again, not left open.
    (tmp_path / "p.ptb64").write_bytes(b"old")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        code, out, err = run(base + ["--out", tmp_path / "p.ptb64", "--out-format", "ptb64"], capsys)
        gc.collect()
    assert code != 0 and err == failure("SAMPLE_FORMAT_PTB64 incompatible with SingleMeasurementRecord")
    assert (tmp_path / "p.ptb64").read_bytes() == b""
    assert [str(w.message) for w in caught if issubclass(w.category, ResourceWarning)] == []
    # Detector-order files: JSON arrays of arrays of nonnegative integers, each a permutation.
    for text, message in [
        ("[[0, 1", "Detector orders file is not valid JSON: {}"), ("NaN", "Detector orders file is not valid JSON: {}"), ("[]", "Detector orders file must contain at least one order."),
        ("{}", "Detector orders file must contain at least one order."), ("[3]", "Each detector order must be an array."),
        ("[[0, -1]]", "Detector IDs must be nonnegative integers."), ("[[0, 1.0]]", "Detector IDs must be nonnegative integers."),
        ("[[0, 1]]", f"Detector order has size 2, but the detector error model has {nd} detectors."),
        (json.dumps([list(range(nd - 1)) + [nd + 5]]), f"Detector order contains out-of-range detector ID {nd + 5} at position {nd - 1}."),
        (json.dumps([[1] + list(range(1, nd))]), "Detector order contains detector ID 1 more than once."),
        ("[[0, -0]]", "Detector IDs must be nonnegative integers."), ("[[18446744073709551616]]", "Detector IDs must be nonnegative integers."),
    ]:
        (tmp_path / "bad.json").write_text(text)
        code, out, err = run(base + ["--detector-orders", tmp_path / "bad.json"], capsys)
        assert code != 0 and err == failure(message.format(tmp_path / "bad.json")), text
    # An unwritable --stats-out is left unwritten, as Tesseract's ofstream leaves it.
    code, out, err = run(base + ["--stats-out", tmp_path / "no" / "s.json"], capsys)
    assert (code, err) == (0, "") and final_line(out)["num_shots"] == model["num_shots"]


def test_cli_python_m(tmp_path):
    exe = [sys.executable, "-m", "stabilizer_qec.tesseract_decoder"]
    r = subprocess.run(exe + ["--version"], capture_output=True, text=True)
    assert (r.returncode, r.stdout) == (0, "1.0\n")
    r = subprocess.run(exe + ["--nope"], capture_output=True, text=True)
    assert r.returncode == 1 and r.stderr.startswith("Unknown argument: --nope\nUsage: tesseract [--help]")
    r = subprocess.run(exe, capture_output=True, text=True)
    assert r.returncode != 0 and r.stderr == failure("Must provide at least one of --circuit or --dem")



def _prediction_stdout_args(tmp_path):
    dem, shots = tmp_path / "stdout.dem", tmp_path / "stdout.01"
    dem.write_text("error(0.1) D0 L0\n")
    shots.write_text("1\n" * 8)
    return ["--dem", str(dem), "--in", str(shots), "--in-format", "01", "--out", "-", "--out-format", "01"]


@pytest.mark.parametrize("limited", [False, True], ids=["success", "file-size-failure"])
def test_prediction_stdout_process_shutdown(limited, tmp_path):
    if limited:
        resource = pytest.importorskip("resource")
        if not hasattr(resource, "RLIMIT_FSIZE"):
            pytest.skip("platform has no RLIMIT_FSIZE")
    args = _prediction_stdout_args(tmp_path)
    # Import before applying a tiny, child-only limit. Execute the actual module boundary,
    # then inspect final process status and complete stderr after interpreter shutdown.
    script = """
import runpy, sys
import stabilizer_qec.tesseract_decoder
if sys.argv[1] == 'limited':
    import resource, signal
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))
sys.argv = ['stabilizer_qec.tesseract_decoder', *sys.argv[2:]]
runpy.run_module('stabilizer_qec.tesseract_decoder', run_name='__main__')
"""
    path = tmp_path / "predictions.01"
    with path.open("wb") as stdout:
        proc = subprocess.run([sys.executable, "-c", script, "limited" if limited else "success", *args], stdout=stdout, stderr=subprocess.PIPE, timeout=15)
    if limited:
        expected = f"tesseract: {OSError(errno.EFBIG, os.strerror(errno.EFBIG))}\n".encode()
        assert (proc.returncode, proc.stderr, path.read_bytes()) == (1, expected, b"1")
    else:
        assert (proc.returncode, proc.stderr) == (0, b"")
        out = path.read_text()
        assert out.splitlines()[:8] == ["1"] * 8
        assert final_line(out)["num_shots"] == 8


def test_prediction_stdout_callable_preserves_stream_ownership(tmp_path, monkeypatch, capsys):
    class FailedStdout(io.StringIO):
        def flush(self):
            raise OSError(errno.EFBIG, "caller-owned flush failure")
    stream = FailedStdout()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", stream)
        assert main(_prediction_stdout_args(tmp_path)) == 1
        assert sys.stdout is stream and not stream.closed
    assert capsys.readouterr().err == f"tesseract: {OSError(errno.EFBIG, 'caller-owned flush failure')}\n"
    stream.close()  # cleanup belongs to the caller


def test_prediction_close_failure_inside_callers_except_block(tmp_path, monkeypatch, capsys):
    # A prediction file whose close fails (a full disk, say) is reported, and the exit status is
    # 1, also when main is called while its caller handles an exception of its own.
    from stabilizer_qec.tesseract_decoder import __main__ as cli

    closed = []
    real_close = cli._Writer.close

    def failing_close(self):
        real_close(self)
        closed.append(True)
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(cli._Writer, "close", failing_close)
    args = _prediction_stdout_args(tmp_path)
    args[args.index("-")] = str(tmp_path / "p.01")
    for inside in (False, True):
        closed.clear()
        if inside:
            try:
                raise LookupError("the caller's own error")
            except LookupError:
                code = main(args)
        else:
            code = main(args)
        out = capsys.readouterr()
        assert (code, out.err, closed) == (1, f"tesseract: {OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))}\n", [True]), inside
    # An error already in flight (here --dem-out's, after the predictions) is kept over a
    # failing close, inside or outside the caller's except block.
    bad = str(tmp_path / "no" / "x.dem")
    for inside in (False, True):
        closed.clear()
        if inside:
            try:
                raise LookupError("the caller's own error")
            except LookupError:
                code = main(args + ["--dem-out", bad])
        else:
            code = main(args + ["--dem-out", bad])
        assert (code, capsys.readouterr().err, closed) == (134, failure("Failed to open " + bad), [True]), inside


SMALL_DEM = "error(0.1) D0 L0\nerror[t](0.1) D0 D1\nerror(0.2) D1 L1\ndetector(0, 0) D0\ndetector(1, 0) D1\n"


def tesseracts_nan() -> str:
    """How Tesseract's program prints the NaN 0.0/0.0 makes (Stim writes a model's arguments
    to a C++ stream): its sign bit is set on x86 ("-nan" from Linux's C library, "-nan(ind)"
    from Windows'); clear on ARM64, and macOS's C library prints no sign for a NaN ("nan")."""
    import platform

    x86 = platform.machine().lower() in ("x86_64", "amd64")
    if x86 and sys.platform.startswith("linux"):
        return "-nan"
    if x86 and sys.platform == "win32":
        return "-nan(ind)"
    return "nan"


@pytest.mark.parametrize("shots", ["1000\n", "1000\n1000\n", ""], ids=["one-shot-wrong", "every-shot-wrong", "no-shots"])
def test_cli_dem_out_without_usable_shots(shots, tmp_path, capsys):
    # With no shot to count from (every shot a logical error, or none), Tesseract's
    # dem_from_counts divides 0 by 0: each estimate is NaN, which Stim's validate lets
    # through and its stream prints; the stats and the final line follow, and it exits 0.
    (tmp_path / "s.dem").write_text(SMALL_DEM)
    (tmp_path / "s.01").write_text(shots)
    code, out, err = run(
        ["--dem", tmp_path / "s.dem", "--in", tmp_path / "s.01", "--in-format", "01", "--in-includes-appended-observables",
         "--dem-out", tmp_path / "o.dem", "--stats-out", tmp_path / "st.json", "--out", tmp_path / "p.01", "--out-format", "01"],
        capsys,
    )
    n = shots.count("\n")
    assert (code, err) == (0, "")
    nan = tesseracts_nan()
    assert (tmp_path / "o.dem").read_text() == (
        f"error({nan}) D0 L0\nerror[t]({nan}) D0 D1\nerror({nan}) D1 L1\ndetector(0, 0) D0\ndetector(1, 0) D1\n"
    )
    assert (tmp_path / "p.01").read_text() == "10\n" * n
    stats = json.loads((tmp_path / "st.json").read_text())
    assert (stats["num_shots"], stats["num_errors"]) == (n, n)
    assert (final_line(out)["num_shots"], final_line(out)["num_errors"]) == (n, n)
    if not shots:
        # Without observables the count is over every shot: none here.
        code, out, err = run(["--dem", tmp_path / "s.dem", "--in", tmp_path / "s.01", "--in-format", "01", "--dem-out", tmp_path / "o2.dem"], capsys)
        assert (code, err) == (0, "") and (tmp_path / "o2.dem").read_text() == (tmp_path / "o.dem").read_text()
        assert final_line(out) == {"num_shots": 0, "num_low_confidence": 0, "num_errors": None, "time": 0.0}
    # One shot decoded right: an estimate of 0 or 1, as before.
    (tmp_path / "ok.01").write_text("0000\n")
    code, out, err = run(["--dem", tmp_path / "s.dem", "--in", tmp_path / "ok.01", "--in-format", "01", "--in-includes-appended-observables", "--dem-out", tmp_path / "o3.dem"], capsys)
    assert (code, err) == (0, "") and (tmp_path / "o3.dem").read_text() == "error(0) D0 L0\nerror[t](0) D0 D1\nerror(0) D1 L1\ndetector(0, 0) D0\ndetector(1, 0) D1\n"


def test_cli_reads_models_with_non_utf8_comment_bytes(tmp_path, capsys):
    # Stim reads a model's or circuit's bytes and passes over those in a comment, so Tesseract's
    # program reads such a file; here a byte that is not UTF-8 is replaced, as the Chromobius
    # program replaces it, and the file decodes as the same file without it.
    (tmp_path / "plain.dem").write_text(SMALL_DEM)
    (tmp_path / "high.dem").write_bytes(SMALL_DEM.replace("error(0.2) D1 L1\n", "error(0.2) D1 L1 # caf\xe9 \xff\n").encode("latin-1"))
    (tmp_path / "s.01").write_text("1000\n0110\n0101\n")
    got = []
    for name in ("plain", "high"):
        code, out, err = run(["--dem", tmp_path / f"{name}.dem", "--in", tmp_path / "s.01", "--in-format", "01", "--in-includes-appended-observables",
                              "--out", tmp_path / f"{name}.01", "--out-format", "01", "--dem-out", tmp_path / f"{name}_out.dem"], capsys)
        assert (code, err) == (0, ""), name
        got.append(((tmp_path / f"{name}.01").read_text(), (tmp_path / f"{name}_out.dem").read_text(), final_line(out)["num_errors"]))
    assert got[0] == got[1]
    # A tag's UTF-8 is written back to --dem-out as it was read (in a text file, its lines ended as
    # the platform's text mode ends them, \r\n on Windows, as Tesseract's std::ofstream writes it).
    (tmp_path / "tag.dem").write_bytes(SMALL_DEM.replace("[t]", "[caf\xe9]").encode("utf-8"))
    code, out, err = run(["--dem", tmp_path / "tag.dem", "--in", tmp_path / "s.01", "--in-format", "01", "--in-includes-appended-observables", "--dem-out", tmp_path / "tag_out.dem"], capsys)
    raw = (tmp_path / "tag_out.dem").read_bytes()
    assert (code, err) == (0, "") and raw.count(b"\n") == raw.count(os.linesep.encode())
    assert raw.decode("utf-8").replace(os.linesep, "\n") == got[0][1].replace("[t]", "[caf\xe9]")
    circuit = MODELS["repetition_d3"]["circuit"]
    (tmp_path / "plain.stim").write_text(circuit)
    (tmp_path / "high.stim").write_bytes(b"# \xe9t\xe9 \xff\n" + circuit.encode())
    got = []
    for name in ("plain", "high"):
        code, out, err = run(["--circuit", tmp_path / f"{name}.stim", "--sample-num-shots", 20, "--sample-seed", 3, "--out", tmp_path / f"{name}.c01", "--out-format", "01"], capsys)
        assert (code, err) == (0, ""), name
        got.append(((tmp_path / f"{name}.c01").read_text(), final_line(out)["num_errors"]))
    assert got[0] == got[1]
