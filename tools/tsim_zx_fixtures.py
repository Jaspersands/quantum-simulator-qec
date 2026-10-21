"""Record Tsim's ZX graphs, tensors and T-counts of a corpus of 40 circuits into
tests/tsim_port/zx_graphs.json.gz, which tests/test_tsim_zx.py compares this package's against.

    python tools/tsim_zx_fixtures.py            # rewrite the fixtures from the installed Tsim
    python tools/tsim_zx_fixtures.py --check    # fail if they differ from what Tsim gives now

Needs bloqade-tsim (``tsim``) and pyzx_param (a dependency of it); run it on the machine the fixtures
are recorded on (macOS: the reductions of the diagrams are pyzx_param's, deterministic). For each
circuit it records, from Tsim's ``Circuit``:

- ``graph``: ``get_graph()``, and ``built``: the vertices of the measurement record, the detectors and
  observables, the error bit count and each noise channel's probabilities;
- ``sampling`` and ``sampling_dets``: ``get_sampling_graph(False)`` and ``get_sampling_graph(True)``;
- ``tcount``: ``tcount()``;
- ``matrix``: ``to_matrix()`` for the circuits with measurements chosen below;
- ``diagrams``: the graphs ``diagram("pyzx")``, ``diagram("pyzx-dets")`` and ``diagram("pyzx-meas")``
  return.

A graph is its vertices ``[id, type, phase, [phase variables], qubit, row]`` in the graph's order,
its edges ``[s, t, type]`` in the graph's order, its scalar ``[power2, phase]``, its inputs and its
outputs, each phase a Fraction written ``"p/q"`` (units of pi).
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import warnings
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "tsim_port" / "zx_graphs.json.gz"


def repetition_code() -> str:
    """Distance 3, two rounds of syndrome extraction, noisy, with detectors and an observable."""
    lines = ["R 0 1 2 3 4", "TICK"]
    for r in range(2):
        lines += ["X_ERROR(0.01) 0 1 2", "CX 0 3 1 3 1 4 2 4", "DEPOLARIZE2(0.02) 0 3", "MR(0.01) 3 4"]
        if r == 0:
            lines += ["DETECTOR rec[-2]", "DETECTOR rec[-1]"]
        else:
            lines += ["DETECTOR rec[-2] rec[-4]", "DETECTOR rec[-1] rec[-3]"]
        lines.append("TICK")
    lines += ["M 0 1 2", "DETECTOR rec[-3] rec[-2] rec[-5]", "DETECTOR rec[-2] rec[-1] rec[-4]", "OBSERVABLE_INCLUDE(0) rec[-3]"]
    return "\n".join(lines)


# (name, program text), in Tsim's shorthand.
CORPUS = [
    ("bell_measured", "R 0 1\nH 0\nCX 0 1\nM 0 1"),
    ("clifford_single_qubit", "H 0\nS 0\nS_DAG 0\nSQRT_X 0\nSQRT_X_DAG 0\nSQRT_Y 0\nSQRT_Y_DAG 0\nC_XYZ 0\nC_NXYZ 0\nC_XNYZ 0\nC_XYNZ 0\nC_ZYX 0\nC_NZYX 0\nC_ZNYX 0\nC_ZYNX 0\nH_XY 0\nH_NXY 0\nH_NXZ 0\nH_YZ 0\nH_NYZ 0\nX 0\nY 0\nZ 0\nI 0"),
    ("clifford_two_qubit", "CX 0 1\nCY 0 1\nCZ 0 1\nSWAP 0 1\nISWAP 0 1\nISWAP_DAG 0 1\nCXSWAP 0 1\nCZSWAP 0 1\nSWAPCX 0 1\nSWAPCZ 0 1\nSQRT_XX 0 1\nSQRT_XX_DAG 0 1\nSQRT_YY 0 1\nSQRT_YY_DAG 0 1\nSQRT_ZZ 0 1\nSQRT_ZZ_DAG 0 1\nXCX 0 1\nXCY 0 1\nXCZ 0 1\nYCX 0 1\nYCY 0 1\nYCZ 0 1\nII 0 1"),
    ("ghz_fanout", "H 0\nCX 0 1 1 2\nCZ 2 0"),
    ("t_gate", "H 0\nT 0\nH 0"),
    ("t_chain", "H 0\nT 0\nT_DAG 1\nCX 0 1\nT 1\nT 0 1"),
    ("tpp", "TPP X0*Z1\nTPP_DAG Y0*Y1*Z2\nTPP Z3"),
    ("spp", "SPP X0*Y1\nSPP_DAG Z0\nSPP !Z1*X2"),
    ("rotations", "R_X(0.1) 0\nR_Y(0.7) 1\nR_Z(0.3) 2\nR_Z(1e-05) 0\nR_X(-0.45) 1"),
    ("rotations_clifford_angles", "R_Z(0.5) 0\nR_X(1.0) 1\nR_Y(-0.5) 0\nR_Z(1.5) 1\nR_Z(2) 0"),
    ("pauli_rotations", "R_XX(0.3) 0 1\nR_YY(0.4) 0 1\nR_ZZ(0.5) 1 2\nR_PAULI(0.345) X0*Y1*Z2"),
    ("u3", "U3(0.3, 0.24, 0.49) 0\nU3(0.5, 0.5, 1.0) 1"),
    ("ccz_ccx", "CCZ 0 1 2\nCCX 0 1 2"),
    ("measure_twice", "H 0\nM 0\nM 0"),
    ("measure_bases", "R 0 1 2\nMX 0\nMY 1\nMZ 2"),
    ("measure_reset", "H 0\nMR 0\nMRX 0\nMRY 0\nMRZ 0"),
    ("reset_mid_circuit", "H 0\nT 0\nR 0\nH 0\nRX 1\nRY 2\nRZ 1"),
    ("mpp", "H 0\nMPP X0*Z1 !Y2\nMPP Z0*Z1"),
    ("two_qubit_measurements", "MXX 0 1\nMYY 0 1\nMZZ 1 2"),
    ("mpad", "MPAD 1 0\nM 0\nDETECTOR rec[-1] rec[-3]"),
    ("feedback_controlled", "H 0\nM 0\nCX rec[-1] 1\nCZ rec[-1] 2\nCY rec[-1] 3"),
    ("feedback_target_first", "M 0\nXCZ 1 rec[-1]\nYCZ 2 rec[-1]"),
    ("detector", "H 0\nM 0\nDETECTOR rec[-1]"),
    ("detector_parity", "H 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nDETECTOR(1, 2) rec[-1]"),
    ("observables", "R 0 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"),
    ("observable_index_gap", "M 0\nOBSERVABLE_INCLUDE(2) rec[-1]"),
    ("annotations_without_records", "M 0\nDETECTOR\nOBSERVABLE_INCLUDE(1)"),
    ("repeated_record_targets", "M 0\nDETECTOR rec[-1] rec[-1]\nM 1\nDETECTOR rec[-1] rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-1]"),
    ("noise_paulis", "X_ERROR(0.1) 0\nY_ERROR(0.2) 1\nZ_ERROR(0.3) 0 1\nM 0 1"),
    ("noise_depolarizing", "H 0\nDEPOLARIZE1(0.1) 0\nDEPOLARIZE2(0.2) 0 1\nM 0 1"),
    ("noise_pauli_channels", "PAULI_CHANNEL_1(0.1, 0.05, 0.02) 0\nPAULI_CHANNEL_2(0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03) 0 1\nM 0 1\nDETECTOR rec[-1]"),
    ("noise_correlated", "E(0.1) X0 Y1\nELSE_CORRELATED_ERROR(0.2) Z1\nE(0.3) Z0\nM 0 1\nDETECTOR rec[-1] rec[-2]"),
    ("noise_heralded", "HERALDED_ERASE(0.1) 0\nHERALDED_PAULI_CHANNEL_1(0.1, 0.2, 0.3, 0.4) 1\nM 0 1\nDETECTOR rec[-1]"),
    ("noisy_measurements", "M(0.1) 0\nMR(0.2) 1\nMX(0.05) 0\nMPP(0.1) X0*X1\nDETECTOR rec[-1]"),
    ("loop_with_annotations", "REPEAT 3 {\n    R 0\n    H 0\n    T 0\n    H 0\n    M 0\n    DETECTOR(1) rec[-1]\n    SHIFT_COORDS(0, 1)\n}\nOBSERVABLE_INCLUDE(0) rec[-1]"),
    ("ticks_and_coordinates", "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(1, 0) 1\nH 0\nTICK\nCX 0 1\nTICK\nT 1\nM 0 1"),
    ("repetition_code_noisy", repetition_code()),
    ("teleportation", "R 0 1 2\nH 1\nCX 1 2\nCX 0 1\nH 0\nM 0 1\nCX rec[-1] 2\nCZ rec[-2] 2\nM 2"),
    ("inverted_results", "RY 0\nM !0 1\nMX !1\nMY !0\nMRY !1\nMPP !X0*Z1\nDETECTOR rec[-1] rec[-2]"),
    ("magic_state_check", "RX 0 1 2 3 4\nT 0 1 2 3 4\nMPP X0*X1*X2*X3\nMPP Z0*Z1\nDETECTOR rec[-1]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"),
]

assert len(CORPUS) == 40, len(CORPUS)

# The circuits with measurements whose to_matrix is recorded (small ones: at most 2^10 entries).
MATRIX = [
    "bell_measured",
    "measure_twice",
    "measure_bases",
    "measure_reset",
    "reset_mid_circuit",
    "mpp",
    "mpad",
    "feedback_controlled",
    "detector_parity",
    "observables",
    "noisy_measurements",
    "teleportation",
]
assert len(MATRIX) == 12 and set(MATRIX) <= {n for n, _ in CORPUS}


def frac(x) -> str:
    f = Fraction(x)
    return f"{f.numerator}/{f.denominator}"


def record_graph(g) -> dict:
    return {
        "vertices": [[int(v), int(g.type(v)), frac(g.phase(v)), sorted(g._phaseVars[v]), g.qubit(v), g.row(v)] for v in g.vertices()],
        "edges": [[int(e[0]), int(e[1]), int(g.edge_type(e))] for e in g.edges()],
        "scalar": [int(g.scalar.power2), frac(g.scalar.phase)],
        "inputs": [int(v) for v in g.inputs()],
        "outputs": [int(v) for v in g.outputs()],
    }


def record_built(circuit) -> dict:
    from tsim.core.parse import parse_stim_circuit

    b = parse_stim_circuit(circuit._stim_circ)
    return {
        "rec": [int(v) for v in b.rec],
        "silent_rec": [int(v) for v in b.silent_rec],
        "detectors": [int(v) for v in b.detectors],
        "observables": {str(k): int(v) for k, v in b.observables_dict.items()},
        "first_vertex": {str(k): int(v) for k, v in b.first_vertex.items()},
        "last_vertex": {str(k): int(v) for k, v in b.last_vertex.items()},
        "num_error_bits": int(b.num_error_bits),
        "channel_probs": [[float(x) for x in p] for p in b.channel_probs],
    }


def record() -> dict:
    import numpy as np
    import pyzx_param
    import tsim

    circuits = []
    for name, text in CORPUS:
        c = tsim.Circuit(text)
        entry = {
            "name": name,
            "text": text,
            "graph": record_graph(c.get_graph()),
            "built": record_built(c),
            "sampling": record_graph(c.get_sampling_graph(sample_detectors=False)),
            "sampling_dets": record_graph(c.get_sampling_graph(sample_detectors=True)),
            "tcount": int(c.tcount()),
            "diagrams": {kind: record_graph(c.diagram(kind)) for kind in ("pyzx", "pyzx-dets", "pyzx-meas")},
        }
        if name in MATRIX:
            m = np.asarray(c.to_matrix(), dtype=complex)
            entry["matrix"] = {"shape": list(m.shape), "real": m.real.ravel().tolist(), "imag": m.imag.ravel().tolist()}
        circuits.append(entry)
    from importlib.metadata import version

    return {"tsim": version("bloqade-tsim"), "pyzx_param": version("pyzx-param"), "circuits": circuits}


def dump(data: dict) -> bytes:
    text = json.dumps(data, separators=(",", ":"))
    return gzip.compress(text.encode(), compresslevel=9, mtime=0)


def main() -> int:
    data = record()
    blob = dump(data)
    if "--check" in sys.argv:
        current = json.loads(gzip.decompress(OUT.read_bytes()))
        if current != json.loads(gzip.decompress(blob)):
            print(f"{OUT} differs from what Tsim {data['tsim']} gives now", file=sys.stderr)
            return 1
        print(f"{OUT} is current ({len(data['circuits'])} circuits, Tsim {data['tsim']}, pyzx_param {data['pyzx_param']})")
        return 0
    OUT.write_bytes(blob)
    print(f"wrote {OUT} ({len(blob)} bytes, {len(data['circuits'])} circuits, Tsim {data['tsim']}, pyzx_param {data['pyzx_param']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
