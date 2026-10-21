"""tqec 0.2.0's circuits, recorded from tqec itself as the port's fixtures.

    python -I tools/tqec_fixtures.py --out tests/tqec_port/fixtures [--only ID,ID] [--jobs N]

Runs real tqec (and tqecd 0.2.1) on the cases below, each with ``do_not_use_database=True`` (no
detector database is read or written), and writes each circuit's text gzipped as
``<id>.stim.gz`` with ``manifest.json``: per case the graph, convention, scale ``k``, observables
and ``block_temporal_height`` given (``[slope, offset]``), and the circuit's SHA-256 and qubit, detector and observable
counts (or the exception tqec raises). For the CNOT at k = 1 the manifest also holds the detectors
of every layer, sorted, to localise a mismatch. ``tests/tqec_port/test_compile_fixtures.py``
compiles the same cases with ``stabilizer_qec.tqec`` and compares the text byte for byte.

Every case runs in a process of its own: tqec reschedules the measurements of the plaquettes it
caches in place, so a circuit it compiles can depend on the circuits the process compiled before.
A fresh process gives the circuit of a program that compiles one graph.

The case graphs (``graph`` in the manifest), built the same way on both sides:

- ``NAME:BASIS`` a gallery graph (``cz:FLOWS`` the CZ with its flow string);
- ``rotated:NAME:BASIS`` the gallery graph rotated 90 degrees counterclockwise about z;
- ``two_cnots:KIND`` ``cnot().compose(cnot(), "Out_Control", "In_Control")``, ports filled
  with ``KIND``;
- ``temporal_hadamard`` a ZXZ cube under an XZX cube, joined by a temporal Hadamard pipe;
- ``y_cube`` a Y half cube under a ZXZ cube (tqec raises ``NotImplementedError``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import pathlib
import sys
import time

GALLERY = [(n, b) for n in ("memory", "stability", "cnot", "move_rotation", "three_cnots", "steane_encoding") for b in ("Z", "X")] + [("cz", "XI -> XZ"), ("cz", "ZZ -> ZZ")]
CONVENTIONS = ("fixed_bulk", "fixed_boundary")


def _slug(graph: str) -> str:
    return graph.replace(" -> ", "").replace(":", "_")


def cases() -> list[dict]:
    out = []

    def add(graph, convention, k, observables="auto", height=None, tag=""):
        cid = f"{_slug(graph)}_{convention}_k{k}{tag}"
        out.append({"id": cid, "graph": graph, "convention": convention, "k": k, "observables": observables, "block_temporal_height": height})

    for name, basis in GALLERY:
        for conv in CONVENTIONS:
            add(f"{name}:{basis}", conv, 1)
    for graph in ("cnot:Z", "cz:ZZ -> ZZ", "move_rotation:Z"):
        for conv in CONVENTIONS:
            add(graph, conv, 2)
    for conv in CONVENTIONS:
        add("temporal_hadamard", conv, 1)
        add("rotated:memory:Z", conv, 1)
        add("rotated:cnot:Z", conv, 1)
        add("two_cnots:ZXZ", conv, 1)
        add("two_cnots:ZXX", conv, 1)
        add("y_cube", conv, 1)
    add("cnot:Z", "fixed_bulk", 1, observables=None, tag="_obs_none")
    add("cnot:Z", "fixed_bulk", 1, observables=[1], tag="_obs_1")
    add("cnot:X", "fixed_boundary", 1, observables=[0], tag="_obs_0")
    for height in (1, 3):
        # a constant height (tqec refuses it) and 2k + height
        add("memory:Z", "fixed_bulk", 1, height=[0, height], tag=f"_height{height}")
        add("cnot:Z", "fixed_boundary", 1, height=[0, height], tag=f"_height{height}")
        add("memory:Z", "fixed_bulk", 1, height=[2, height], tag=f"_height2k{height}")
        add("cnot:Z", "fixed_boundary", 1, height=[2, height], tag=f"_height2k{height}")
    return out


def build_graph(tqec_module, graph: str):
    """The case's graph, from ``tqec`` or ``stabilizer_qec.tqec`` (``tqec_module``): the gallery,
    the block graph, the positions and ZXCube of that package."""
    gallery = tqec_module.gallery
    Basis = tqec_module.Basis
    parts = graph.split(":")
    if parts[0] == "temporal_hadamard" or parts[0] == "y_cube":
        g = tqec_module.BlockGraph(parts[0])
        bottom = "ZXZ" if parts[0] == "temporal_hadamard" else "Y"
        top = "XZX" if parts[0] == "temporal_hadamard" else "ZXZ"
        g.add_cube(tqec_module.Position3D(0, 0, 0), bottom)
        g.add_cube(tqec_module.Position3D(0, 0, 1), top)
        g.add_pipe(tqec_module.Position3D(0, 0, 0), tqec_module.Position3D(0, 0, 1))
        return g
    if parts[0] == "two_cnots":
        g = gallery.cnot().compose(gallery.cnot(), "Out_Control", "In_Control")
        g.fill_ports(tqec_module.ZXCube.from_str(parts[1]))
        return g
    if parts[0] == "rotated":
        return build_graph(tqec_module, ":".join(parts[1:])).rotate(tqec_module.Direction3D.Z)
    if parts[0] == "cz":
        return gallery.cz(parts[1])
    return getattr(gallery, parts[0])(Basis(parts[1]))


class _TheirTqec:
    """tqec's names, gathered as ``stabilizer_qec.tqec`` exports them."""

    def __init__(self) -> None:
        import tqec
        from tqec import gallery
        from tqec.computation.cube import ZXCube
        from tqec.utils.enums import Basis
        from tqec.utils.position import Direction3D, Position3D

        self.gallery, self.Basis, self.BlockGraph, self.ZXCube = gallery, Basis, tqec.BlockGraph, ZXCube
        self.Position3D, self.Direction3D = Position3D, Direction3D


def _leaf_detectors(tree, k: int) -> list[list[str]]:
    """Each leaf layer's detectors (``DETECTOR`` lines), sorted, in the tree's order."""
    out = []

    def walk(node):
        if node.is_leaf:
            out.append(sorted(str(d.to_instruction()) for d in node.get_annotations(k).detectors))
        else:
            for child in node._children:
                walk(child)

    walk(tree._root)
    return out


def run_case(case: dict, outdir: str) -> dict:
    from tqec.compile.compile import compile_block_graph
    from tqec.compile.convention import ALL_CONVENTIONS
    from tqec.utils.scale import LinearFunction

    t = _TheirTqec()
    record = dict(case)
    start = time.time()
    try:
        g = build_graph(t, case["graph"])
        observables = case["observables"]
        if isinstance(observables, list):
            surfaces = g.find_correlation_surfaces()
            observables = [surfaces[i] for i in observables]
        kwargs = {}
        if case["block_temporal_height"] is not None:
            kwargs["block_temporal_height"] = LinearFunction(*case["block_temporal_height"])
        compiled = compile_block_graph(g, convention=ALL_CONVENTIONS[case["convention"]], observables=observables, **kwargs)
        circuit = compiled.generate_stim_circuit(case["k"], do_not_use_database=True)
        if case["id"].startswith("cnot_Z_") and case["k"] == 1 and case["observables"] == "auto" and case["block_temporal_height"] is None:
            tree = compiled.to_layer_tree()
            again = tree.generate_circuit(case["k"], do_not_use_database=True)
            assert str(again) == str(circuit)
            record["layer_detectors"] = _leaf_detectors(tree, case["k"])
    except Exception as e:  # noqa: BLE001 - the reference's exception is the fixture
        record.update(error={"type": type(e).__name__, "message": str(e)}, seconds=round(time.time() - start, 1))
        return record
    text = str(circuit)
    data = text.encode()
    with open(os.path.join(outdir, case["id"] + ".stim.gz"), "wb") as f:
        f.write(gzip.compress(data, compresslevel=9, mtime=0))
    try:
        circuit.detector_error_model()
        deterministic = True
    except Exception:  # noqa: BLE001 - tqec's two CZ bugs: a nondeterministic detector or observable
        deterministic = False
    record.update(file=case["id"] + ".stim.gz", sha256=hashlib.sha256(data).hexdigest(), qubits=circuit.num_qubits, detectors=circuit.num_detectors, observable_count=circuit.num_observables, deterministic=deterministic, seconds=round(time.time() - start, 1))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    import tqec
    import importlib.metadata

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.json"
    old = json.loads(manifest_path.read_text())["cases"] if manifest_path.exists() else []
    wanted = [c for c in cases() if not args.only or c["id"] in args.only.split(",")]
    records = {r["id"]: r for r in old}
    # Each case in a fresh process: tqec's output depends on what the process compiled before (its
    # cached plaquettes are rescheduled in place). tqec's detector search runs a pool of its own.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs, max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_case, c, str(outdir)): c["id"] for c in wanted}
        for fut in concurrent.futures.as_completed(futures):
            rec = fut.result()
            records[rec["id"]] = rec
            print(rec["id"], rec.get("detectors", rec.get("error", {}).get("type")), rec["seconds"], "s", flush=True)
    order = [c["id"] for c in cases()]
    manifest = {
        "reference": {"tqec": tqec.__version__, "tqecd": importlib.metadata.version("tqecd"), "stim": importlib.metadata.version("stim"), "python": sys.version.split()[0]},
        "cases": [records[i] for i in order if i in records],
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
