"""tqec's compile pipeline, ported (``stabilizer_qec.tqec._compile``): every circuit equals the
one tqec 0.2.0 (with tqecd 0.2.1) writes for the same graph, byte for byte.

The fixtures are tqec's own circuits, recorded by ``tools/tqec_fixtures.py`` (each case in a
fresh process, ``do_not_use_database=True``); ``fixtures/manifest.json`` lists each case's graph,
convention, scale, observables and block height with the circuit's SHA-256 and counts, or the
exception tqec raises. The gallery at k = 1 runs memory, stability and the CNOT by default and
the rest with ``STABILIZER_QEC_SLOW=1``, as do the cases at k = 2."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

from stabilizer_qec import tqec
from stabilizer_qec.tqec import LinearFunction
from stabilizer_qec.tqec._compile import compile as _tqec_compile

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
CASES = {c["id"]: c for c in MANIFEST["cases"]}
SLOW = pytest.mark.skipif(not os.environ.get("STABILIZER_QEC_SLOW"), reason="slow: set STABILIZER_QEC_SLOW=1")
GALLERY = ("memory", "stability", "cnot", "move_rotation", "three_cnots", "steane_encoding", "cz")
DEFAULT_GALLERY = ("memory", "stability", "cnot")


def build_graph(graph: str):
    """The case's graph, built as ``tools/tqec_fixtures.py`` builds it with tqec."""
    parts = graph.split(":")
    if parts[0] in ("temporal_hadamard", "y_cube"):
        g = tqec.BlockGraph(parts[0])
        g.add_cube(tqec.Position3D(0, 0, 0), "ZXZ" if parts[0] == "temporal_hadamard" else "Y")
        g.add_cube(tqec.Position3D(0, 0, 1), "XZX" if parts[0] == "temporal_hadamard" else "ZXZ")
        g.add_pipe(tqec.Position3D(0, 0, 0), tqec.Position3D(0, 0, 1))
        return g
    if parts[0] == "two_cnots":
        g = tqec.gallery.cnot().compose(tqec.gallery.cnot(), "Out_Control", "In_Control")
        g.fill_ports(tqec.ZXCube.from_str(parts[1]))
        return g
    if parts[0] == "rotated":
        return build_graph(":".join(parts[1:])).rotate(tqec.Direction3D.Z)
    if parts[0] == "cz":
        return tqec.gallery.cz(parts[1])
    return getattr(tqec.gallery, parts[0])(tqec.Basis(parts[1]))


def compile_case(case: dict):
    """The case compiled as a user compiles it: ``compile_block_graph(...)`` and
    ``generate_stim_circuit(k)`` with their defaults."""
    g = build_graph(case["graph"])
    observables = case["observables"]
    if isinstance(observables, list):
        surfaces = g.find_correlation_surfaces()
        observables = [surfaces[i] for i in observables]
    kwargs = {}
    if case["block_temporal_height"] is not None:
        kwargs["block_temporal_height"] = LinearFunction(*case["block_temporal_height"])
    compiled = tqec.compile_block_graph(g, convention=case["convention"], observables=observables, **kwargs)
    return compiled, compiled.generate_stim_circuit(case["k"])


def reference_text(case: dict) -> str:
    data = gzip.decompress((FIXTURES / case["file"]).read_bytes())
    assert hashlib.sha256(data).hexdigest() == case["sha256"], f"{case['file']} is not the file the manifest recorded"
    return data.decode()


def leaf_detectors(compiled, k: int) -> list[list[str]]:
    """Each leaf layer's detectors, sorted, as the manifest records them for the CNOT."""
    tree = compiled._tqec_graph().to_layer_tree()
    tree.generate_circuit(k)
    out: list[list[str]] = []

    def walk(node) -> None:
        if node.is_leaf:
            out.append(sorted(str(d.to_instruction()) for d in node.get_annotations(k).detectors))
        else:
            for child in node._children:
                walk(child)

    walk(tree._root)
    return out


def check_case(case_id: str) -> None:
    case = CASES[case_id]
    compiled, circuit = compile_case(case)
    text = str(circuit)
    want = reference_text(case)
    if text != want and "layer_detectors" in case:
        ours = leaf_detectors(compiled, case["k"])
        for layer, (a, b) in enumerate(zip(ours, case["layer_detectors"])):
            assert a == b, f"{case_id}: the detectors of layer {layer} differ from tqec's"
    assert text == want, f"{case_id}: the circuit differs from tqec's"
    assert (circuit.num_qubits, circuit.num_detectors, circuit.num_observables) == (case["qubits"], case["detectors"], case["observable_count"])


def check_refusal(case_id: str) -> None:
    """tqec's exception: its type, and its message (an object's address and module path aside)."""
    case = CASES[case_id]
    with pytest.raises(Exception) as info:
        compile_case(case)

    def normal(message: str) -> str:
        return re.sub(r"<[\w.]+\.(\w+) object at 0x[0-9a-fA-F]+>", r"<\1 object>", message)

    assert type(info.value).__name__ == case["error"]["type"]
    assert normal(str(info.value)) == normal(case["error"]["message"])


def gallery_params() -> list:
    """The gallery's graphs at k = 1, each convention, with tqec's defaults."""
    params = []
    for case in MANIFEST["cases"]:
        name = case["graph"].split(":")[0]
        if name not in GALLERY or case["k"] != 1 or case["observables"] != "auto" or case["block_temporal_height"] is not None:
            continue
        marks = () if name in DEFAULT_GALLERY else (SLOW,)
        params.append(pytest.param(case["id"], id=f"{case['graph'].replace(' -> ', '').replace(':', '_')}-{case['convention']}", marks=marks))
    return params


@pytest.mark.parametrize("case_id", gallery_params())
def test_gallery_k1_byte_identical(case_id):
    check_case(case_id)


def test_gallery_covers_tqecs_gallery():
    assert len(gallery_params()) == 28
    assert {c["graph"].split(":")[0] for c in MANIFEST["cases"]} >= set(GALLERY)


@SLOW
@pytest.mark.parametrize("case_id", [c["id"] for c in MANIFEST["cases"] if c["k"] == 2])
def test_k2_selected(case_id):
    check_case(case_id)


@pytest.mark.parametrize("convention", ["fixed_bulk", "fixed_boundary"])
def test_temporal_hadamard(convention):
    check_case(f"temporal_hadamard_{convention}_k1")


@pytest.mark.parametrize("case_id", [c["id"] for c in MANIFEST["cases"] if c["graph"].startswith("rotated")])
def test_rotated(case_id):
    check_case(case_id)


@pytest.mark.parametrize("case_id", [c["id"] for c in MANIFEST["cases"] if c["graph"].startswith("two_cnots")])
def test_two_cnots(case_id):
    check_case(case_id)


@pytest.mark.parametrize("case_id", ["cnot_Z_fixed_bulk_k1_obs_none", "cnot_Z_fixed_bulk_k1_obs_1", "cnot_X_fixed_boundary_k1_obs_0"])
def test_observables_none_and_explicit(case_id):
    check_case(case_id)


@pytest.mark.parametrize("case_id", [c["id"] for c in MANIFEST["cases"] if "_height" in c["id"]])
def test_block_temporal_height(case_id):
    if "error" in CASES[case_id]:
        check_refusal(case_id)  # a constant height: tqec refuses it
    else:
        check_case(case_id)


@pytest.mark.parametrize("convention", ["fixed_bulk", "fixed_boundary"])
def test_y_cube_raises_tqecs_error(convention):
    case = CASES[f"y_cube_{convention}_k1"]
    assert case["error"] == {"type": "NotImplementedError", "message": "Y cube is not implemented."}
    check_refusal(case["id"])


_HASH_SEED_SCRIPT = """
import sys
from stabilizer_qec import tqec
g = tqec.gallery.cnot(tqec.Basis.Z)
sys.stdout.write(str(tqec.compile_block_graph(g, convention="fixed_boundary").generate_stim_circuit(1)))
"""


def test_detector_order_independent_of_hash_seed():
    """Strings hash differently in every process; the detectors come out in the same order."""
    texts = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        texts.append(subprocess.run([sys.executable, "-c", _HASH_SEED_SCRIPT], env=env, capture_output=True, text=True, check=True).stdout)
    assert texts[0] == texts[1] == reference_text(CASES["cnot_Z_fixed_boundary_k1"])


def test_each_compilation_starts_from_fresh_caches(monkeypatch):
    """tqec's circuit for a graph depends on what its process compiled before: it reschedules
    the measurements of cached plaquettes in place. Move/rotation (fixed boundary) reschedules
    plaquettes the memory, the rotated memory and the temporal Hadamard share, so a tqec process
    that compiled it first writes other circuits for them than a fresh process does. Here every
    compilation is a fresh process's."""
    check_case("move_rotation_Z_fixed_boundary_k1")
    # The control: with the cache clear skipped, the memory compiled now is the circuit of a tqec
    # process that compiled move/rotation first, not the fixture. So the checks below need the clear.
    memory = CASES["memory_Z_fixed_boundary_k1"]
    with monkeypatch.context() as patch:
        patch.setattr(_tqec_compile, "_start_from_fresh_caches", lambda: None)
        _, stale = compile_case(memory)
    assert str(stale) != reference_text(memory), "move/rotation no longer changes the memory's cached plaquettes: this test no longer tests the cache clear"
    check_case("memory_Z_fixed_boundary_k1")
    check_case("rotated_memory_Z_fixed_boundary_k1")
    check_case("temporal_hadamard_fixed_boundary_k1")
