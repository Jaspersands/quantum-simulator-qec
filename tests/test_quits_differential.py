"""stabilizer_qec.quits against QUITS 1.2.0 itself, where it is installed (with Stim and ldpc):
every code family's parity-check matrices and logicals, the circuits of every strategy each
family supports (text equal to QUITS' own on Stim), and the sliding-window decodes (BP+OSD and
BP+LSD, circuit-level and phenomenological) of the same detection events, equal shot for shot.
"""

import pathlib

import numpy as np
import pytest

from conftest import reference

pytest.importorskip("networkx")
stim = pytest.importorskip("stim")
pytest.importorskip("ldpc")
quits = reference("quits")

import stabilizer_qec as sq  # noqa: E402
import stabilizer_qec.quits as ours  # noqa: E402
from stabilizer_qec.quits.qldpc_code.mitten import sample_mitten_base_matrices as our_sample  # noqa: E402

PCM = pathlib.Path(__file__).parent / "quits_port" / "parity_check_matrices"


def _mitten(pkg, sample):
    c = tuple([(i + 1) % 5 for i in range(5)] + list(range(5, 25)))
    y = tuple([(3 * i) % 5 for i in range(5)] + [5 + (j + 1) % 20 for j in range(20)])
    G = pkg.gf2_util.MultTableGroup.from_generators([c, y], "C5 : C20")
    A, B = sample(G, np.random.default_rng(1))
    return pkg.qldpc_code.MittenCode(A, B, G)


def _codes(pkg):
    """name -> (constructor, strategies) for the package ``pkg`` (quits or ours)."""
    import importlib

    importlib.import_module(pkg.__name__ + ".gf2_util")
    mitten = importlib.import_module(pkg.__name__ + ".qldpc_code.mitten")
    q = importlib.import_module(pkg.__name__ + ".qldpc_code")
    h = np.loadtxt(PCM / "n=12_dv=3_dc=4_dist=6.txt", dtype=int)
    rep = np.array([[1, 1, 0], [0, 1, 1]])
    b = np.array([[0, 0, 0, 0, 0], [0, 2, 4, 7, 11], [0, 3, 10, 14, 15]])
    lifted = ("zxcoloration", "cardinal", "cardinalNSmerge")
    return {
        "HGP rep": (lambda: q.HgpCode(rep, rep), lifted),
        "HGP n=12": (lambda: q.HgpCode(h, h), lifted),
        "QLP 7": (lambda: q.QlpCode(np.array([[0, 0, 0], [0, 1, 3]]), np.array([[0, 0, 0], [0, 1, 3]]), 7), lifted),
        "QLP 16": (lambda: q.QlpCode(b, b, 16), lifted),
        "BPC 15": (lambda: q.BpcCode([0, 1, 5], [0, 8, 13], 15, 3), lifted),
        "BPC 6": (lambda: q.BpcCode([0, 1, 2], [0, 4, 5], lift_size=6, factor=3), lifted),
        "LCS 5,3": (lambda: q.LcsCode(5, 3), lifted),
        "BB 12,6 (gross)": (lambda: q.BbCode(12, 6, [3], [1, 2], [1, 2], [3]), ("custom", "zxcoloration")),
        "BB 15,3": (lambda: q.BbCode(15, 3, [9], [1, 2], [2, 7], [0]), ("custom", "zxcoloration")),
        "Mitten C5:C20": (lambda: _mitten(pkg, mitten.sample_mitten_base_matrices), ("custom", "zxcoloration", "hook_error_free")),
    }


NAMES = list(_codes(ours))
# QLP 16's circuits take minutes to colour; its matrices are checked, QLP 7's circuits.
CIRCUIT_NAMES = [n for n in NAMES if n != "QLP 16"]


@pytest.mark.parametrize("name", NAMES)
def test_codes_equal(name):
    a, b = _codes(ours)[name][0](), _codes(quits)[name][0]()
    for attr in ("hz", "hx", "lz", "lx"):
        np.testing.assert_array_equal(np.asarray(getattr(a, attr)), np.asarray(getattr(b, attr)), err_msg=attr)
    assert a.verify_css_logicals()["all_tests_passed"]


@pytest.mark.parametrize("name", CIRCUIT_NAMES)
def test_circuits_equal(name):
    make, strategies = _codes(ours)[name]
    make_ref = _codes(quits)[name][0]
    for strategy in strategies:
        for basis in ("Z", "X"):
            for noisy in (False, True):
                kwargs = dict(strategy=strategy, num_rounds=2, basis=basis)
                if strategy in ("cardinal", "cardinalNSmerge"):
                    kwargs["seed"] = 3
                if noisy:
                    em = dict(idle_error=1e-4, sqgate_error=2e-4, tqgate_error=1e-3, spam_error=3e-3)
                    a = make().build_circuit(error_model=ours.ErrorModel(**em), circuit_build_options=ours.CircuitBuildOptions(get_all_detectors=True), **kwargs)
                    b = make_ref().build_circuit(error_model=quits.ErrorModel(**em), circuit_build_options=quits.CircuitBuildOptions(get_all_detectors=True), **kwargs)
                else:
                    a, b = make().build_circuit(**kwargs), make_ref().build_circuit(**kwargs)
                assert isinstance(a, sq.Circuit) and isinstance(b, stim.Circuit)
                assert str(a) == str(b), (strategy, basis, noisy)


def _memory(code_factory, rounds, p, shots, seed):
    code = code_factory()
    circuit = code.build_circuit(strategy="zxcoloration", error_model=ours.ErrorModel(p / 10, p / 10, p, p), num_rounds=rounds, basis="Z")
    dets, obs = ours.get_stim_mem_result(circuit, shots, seed=seed)
    return code, circuit, dets, obs


@pytest.mark.parametrize("decoder", ["bposd", "bplsd"])
@pytest.mark.parametrize("name", ["HGP n=12", "BPC 6", "BB 15,3"])
def test_circuit_window_decodes_equal(name, decoder):
    code, circuit, dets, obs = _memory(_codes(ours)[name][0], rounds=4, p=3e-3, shots=60, seed=5)
    order = "osd_order" if decoder == "bposd" else "lsd_order"
    mine = getattr(ours, f"sliding_window_{decoder}_circuit_mem")
    ref = getattr(quits, f"sliding_window_{decoder}_circuit_mem")
    a = mine(dets, circuit, code.hz, code.lz, 3, 1, max_iter=10, **{order: 2})
    b = ref(dets, stim.Circuit(str(circuit)), code.hz, code.lz, 3, 1, max_iter=10, **{order: 2})
    # Equal shot for shot but for ties: OSD's higher-order search ranks candidates by sums the
    # reference forms in numpy, whose last bit moves between numpy builds (one shot of sixty
    # has flipped on one CI platform at a time: Windows 3.11 once, macOS 3.12 and 3.14 once).
    mismatched = int((a != b).any(axis=1).sum())
    assert mismatched <= 1, (name, decoder, mismatched, np.flatnonzero((a != b).any(axis=1)))
    assert np.mean((obs != a).any(axis=1)) < 0.5


@pytest.mark.parametrize("decoder", ["bposd", "bplsd"])
@pytest.mark.parametrize("name", ["HGP n=12", "BPC 6", "LCS 5,3"])
def test_phenomenological_window_decodes_equal(name, decoder):
    code, circuit, dets, obs = _memory(_codes(ours)[name][0], rounds=6, p=1e-3, shots=200, seed=7)
    order = "osd_order" if decoder == "bposd" else "lsd_order"
    rate = 1e-3 * (code.depth + 3)
    mine = getattr(ours, f"sliding_window_{decoder}_phenom_mem")
    ref = getattr(quits, f"sliding_window_{decoder}_phenom_mem")
    a = mine(dets, code.hz, code.lz, 4, 2, eff_error_rate_per_fault=rate, max_iter=10, **{order: 1})
    b = ref(dets, code.hz, code.lz, 4, 2, eff_error_rate_per_fault=rate, max_iter=10, **{order: 1})
    np.testing.assert_array_equal(a, b)


def test_detector_error_model_matrices_and_spacetime_distance_equal():
    from quits.simulation import estimate_spacetime_distance as ref_distance
    from quits.simulation import spacetime_matrices as ref_matrices

    from stabilizer_qec.quits.simulation import estimate_spacetime_distance, spacetime_matrices

    code, circuit, _, _ = _memory(_codes(ours)["BPC 6"][0], rounds=2, p=1e-3, shots=1, seed=0)
    ref_circuit = stim.Circuit(str(circuit))
    a = ours.detector_error_model_to_matrix(circuit.detector_error_model())
    b = quits.detector_error_model_to_matrix(ref_circuit.detector_error_model())
    for x, y in zip(a, b):
        if hasattr(x, "toarray"):
            np.testing.assert_array_equal(x.toarray(), y.toarray())
        else:
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-12)
    (H, L), (Hr, Lr) = spacetime_matrices(circuit), ref_matrices(ref_circuit)
    np.testing.assert_array_equal(np.asarray(H.todense() if hasattr(H, "todense") else H), np.asarray(Hr.todense() if hasattr(Hr, "todense") else Hr))
    np.testing.assert_array_equal(np.asarray(L.todense() if hasattr(L, "todense") else L), np.asarray(Lr.todense() if hasattr(Lr, "todense") else Lr))
    assert estimate_spacetime_distance(H, L, num_trials=5, seed=1) == ref_distance(Hr, Lr, num_trials=5, seed=1)


def test_mitten_sampling_equal():
    from quits.gf2_util import direct_product, cyclic, symmetric
    from quits.qldpc_code.mitten import sample_mitten_base_matrices as ref_sample

    from stabilizer_qec.quits import gf2_util as g

    for seed in range(4):
        A, B = our_sample(g.direct_product(g.cyclic(5), g.symmetric(3)), np.random.default_rng(seed))
        Ar, Br = ref_sample(direct_product(cyclic(5), symmetric(3)), np.random.default_rng(seed))
        np.testing.assert_array_equal(np.asarray(A), np.asarray(Ar))
        np.testing.assert_array_equal(np.asarray(B), np.asarray(Br))
