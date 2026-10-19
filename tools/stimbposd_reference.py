"""stimbposd 0.2.0's predictions on ldpc 2.4.1, recorded for tests/test_stimbposd_package.py.

stimbposd is not installed beside this package's tests, so its decoders are replayed here from its
source (`src/stimbposd/bp_osd.py`, `bp_lsd.py`): its check matrices are beliefmatching's
`detector_error_model_to_check_matrices` (stimbposd's `dem_to_matrices.py` is that function with
`allow_undecomposed_hyperedges=True` by default), its decoders ldpc's `BpOsdDecoder` and
`BpLsdDecoder` built with its arguments, and its predictions `observables_matrix @ correction % 2`.
`compute` takes the modules, so the test runs the same replay on real ldpc, Stim and
beliefmatching where they are installed.

    python tools/stimbposd_reference.py   # with stim, ldpc 2.4.1 and beliefmatching 0.2.0

Writes tests/data/stimbposd_reference.json.gz: each problem's model, its shots (bit-packed,
little-endian, as hex) and, for each of stimbposd's decoder variants, the predictions.
"""

from __future__ import annotations

import gzip
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "data" / "stimbposd_reference.json.gz"
BB72 = ROOT / "tests" / "stimbposd_port" / "data" / "bivariate_bicycle_nkd_72_12_6_p_0.001_round_6"

# stimbposd's sinter variants (sinter_decoders()) and more of its options, by name.
OSD_VARIANTS = {
    "bposd": {},
    "bposd-serial": dict(schedule="serial", random_schedule_seed=0),
    "bposd-minsum": dict(bp_method="minimum_sum"),
    "bposd-serial-minsum": dict(schedule="serial", random_schedule_seed=0, bp_method="minimum_sum"),
    "osd_e 3": dict(osd_method="osd_e", osd_order=3),
    "osd0": dict(osd_method="osd0", osd_order=0),
    "osd order 5": dict(osd_order=5),
    "ms 0.625": dict(bp_method="minimum_sum", ms_scaling_factor=0.625),
    "max_bp_iters 5": dict(max_bp_iters=5),
}
LSD_VARIANTS = {
    "bplsd": {},
    "bplsd-serial": dict(schedule="serial", random_schedule_seed=0),
    "bplsd-minsum": dict(bp_method="minimum_sum"),
    "bplsd-serial-minsum": dict(schedule="serial", random_schedule_seed=0, bp_method="minimum_sum"),
    "lsd_cs 3": dict(lsd_method="lsd_cs", lsd_order=3),
    "lsd_e 2": dict(lsd_method="lsd_e", lsd_order=2),
    "bits_per_step 2": dict(bits_per_step=2),
    "always_run_lsd": dict(always_run_lsd=True),
}


def problems(stim):
    """(name, model text, shots as a uint8 array) for each problem."""
    import numpy as np

    out = []
    for name, task, kw, shots, seed in [
        ("surface d5 x p0.007", "surface_code:rotated_memory_x", dict(distance=5, rounds=5, p=0.007), 50, 1),
        ("surface d3 z p0.01", "surface_code:rotated_memory_z", dict(distance=3, rounds=3, p=0.01), 100, 4),
        ("repetition d9 r25 p0.05", "repetition_code:memory", dict(distance=9, rounds=25, after_clifford_depolarization=0.05), 50, 7),
    ]:
        p = kw.pop("p", None)
        if p is not None:
            kw.update(after_clifford_depolarization=p, before_round_data_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p)
        circuit = stim.Circuit.generated(task, **kw)
        dets = circuit.compile_detector_sampler(seed=seed).sample(shots)
        out.append((name, str(circuit.detector_error_model()), np.asarray(dets, dtype=np.uint8)))
    circuit = stim.Circuit.from_file(str(BB72) + ".stim")
    data = stim.read_shot_data_file(path=str(BB72) + ".b8", format="b8", num_detectors=circuit.num_detectors, num_observables=circuit.num_observables)
    out.append(("bivariate bicycle 72 (stimbposd's test data)", pathlib.Path(str(BB72) + ".dem").read_text(), np.asarray(data[:, : circuit.num_detectors], dtype=np.uint8)))
    return out


def replay(ldpc, matrices_fn, model, kind: str, shots, **kwargs):
    """stimbposd's BPOSD (kind "osd") or BPLSD ("lsd") `decode_batch(shots)`, as its source does it."""
    import numpy as np

    m = matrices_fn(model, allow_undecomposed_hyperedges=True)
    if kind == "osd":
        max_bp_iters, bp_method = kwargs.pop("max_bp_iters", 30), kwargs.pop("bp_method", "product_sum")
        osd_order, osd_method = kwargs.pop("osd_order", 60), kwargs.pop("osd_method", "osd_cs")
        h = m.check_matrix.shape
        dec = ldpc.BpOsdDecoder(
            pcm=m.check_matrix, error_channel=list(m.priors), max_iter=max_bp_iters, bp_method=bp_method,
            osd_order=max(0, min(osd_order, h[1] - h[0])), osd_method=osd_method, input_vector_type="syndrome", **kwargs,
        )
    else:
        max_bp_iters, bp_method = kwargs.pop("max_bp_iters", 30), kwargs.pop("bp_method", "product_sum")
        dec = ldpc.BpLsdDecoder(
            pcm=m.check_matrix, max_iter=max_bp_iters, bp_method=bp_method, error_channel=list(m.priors),
            lsd_order=kwargs.pop("lsd_order", 0), lsd_method=kwargs.pop("lsd_method", "lsd_0"), bits_per_step=kwargs.pop("bits_per_step", 1),
            always_run_lsd=kwargs.pop("always_run_lsd", False), input_vector_type="syndrome", **kwargs,
        )
    out = np.zeros((shots.shape[0], m.observables_matrix.shape[0]), dtype=bool)
    for i in range(shots.shape[0]):
        out[i, :] = (m.observables_matrix @ dec.decode(shots[i, :])) % 2
    return out


def packed_hex(a) -> str:
    import numpy as np

    return np.packbits(np.asarray(a, dtype=np.uint8), axis=1, bitorder="little").tobytes().hex()


def compute(stim, ldpc, matrices_fn) -> list:
    out = []
    for name, text, shots in problems(stim):
        model = stim.DetectorErrorModel(text)
        record = dict(name=name, model=text, num_shots=int(shots.shape[0]), shots=packed_hex(shots), predictions={})
        for kind, variants in (("osd", OSD_VARIANTS), ("lsd", LSD_VARIANTS)):
            for vname, kw in variants.items():
                record["predictions"][vname] = packed_hex(replay(ldpc, matrices_fn, model, kind, shots, **dict(kw)))
        out.append(record)
    return out


def main() -> None:
    import beliefmatching
    import ldpc
    import stim

    assert ldpc.__version__ == "2.4.1", ldpc.__version__
    data = dict(stim=stim.__version__, ldpc=ldpc.__version__, problems=compute(stim, ldpc, beliefmatching.detector_error_model_to_check_matrices))
    OUT.write_bytes(gzip.compress(json.dumps(data, indent=0).encode(), mtime=0))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
