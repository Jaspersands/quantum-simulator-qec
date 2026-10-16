"""stabilizer_qec.panqec against PanQEC 0.1.7 itself, where it is installed (with ldpc and
PyMatching): every registered code's stabilizers, logicals and coordinates (plain and
Clifford-deformed), the Pauli noise drawn on the same seeds, every decoder's corrections on the
same errors, direct simulations, and the GUI's answers, all equal."""

import json

import numpy as np
import pytest

from conftest import reference

pytest.importorskip("scipy")
pytest.importorskip("ldpc")
pytest.importorskip("pymatching")
panqec = reference("panqec")

import stabilizer_qec.panqec as ours  # noqa: E402
import stabilizer_qec.panqec.config as our_config  # noqa: E402
import stabilizer_qec.panqec.simulation  # noqa: E402,F401
import panqec.config as ref_config  # noqa: E402
import panqec.simulation  # noqa: E402,F401
from panqec.decoders.sweepmatch import _rotated_sweep_decoder, _sweep_decoder_3d  # noqa: E402


def _default_direction(self):
    return int(self._rng.choice([0, 1, 2], size=1)[0])


# PanQEC 0.1.7's sweep decoders convert a one-element array to an int, which NumPy 2.4 refuses;
# the reference takes the port's fix (the same draw).
_sweep_decoder_3d.SweepDecoder3D.get_default_direction = _default_direction
_rotated_sweep_decoder.RotatedSweepDecoder3D.get_default_direction = _default_direction

CODE_NAMES = list(ref_config.CODES)
SIZES = [(3,), (4,), (2, 3, 4), (3, 4, 3)]


def _make(config, name, size):
    try:
        return config.CODES[name](*size), None
    except Exception as e:  # noqa: BLE001 - a size a code does not take raises in both
        return None, type(e)


def _dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m)


def _assert_codes_equal(a, b):
    np.testing.assert_array_equal(_dense(a.stabilizer_matrix), _dense(b.stabilizer_matrix))
    np.testing.assert_array_equal(np.asarray(a.logicals_x), np.asarray(b.logicals_x))
    np.testing.assert_array_equal(np.asarray(a.logicals_z), np.asarray(b.logicals_z))
    assert a.qubit_coordinates == b.qubit_coordinates
    assert a.stabilizer_coordinates == b.stabilizer_coordinates
    assert (a.n, a.k, a.label) == (b.n, b.k, b.label)
    if a.is_css:
        np.testing.assert_array_equal(_dense(a.Hx), _dense(b.Hx))
        np.testing.assert_array_equal(_dense(a.Hz), _dense(b.Hz))


def test_repetition_code_equal():
    # In PanQEC's wheel, not registered in its config (nor its source on GitHub).
    from panqec.codes.repetition._repetition_code import RepetitionCode as Ref
    from stabilizer_qec.panqec.codes.repetition._repetition_code import RepetitionCode as Ours

    for size in (3, 4, 7):
        _assert_codes_equal(Ours(size), Ref(size))


@pytest.mark.parametrize("name", CODE_NAMES)
def test_codes_equal(name):
    made = 0
    for size in SIZES:
        (a, ea), (b, eb) = _make(our_config, name, size), _make(ref_config, name, size)
        assert ea == eb, size
        if a is None:
            continue
        made += 1
        _assert_codes_equal(a, b)
        assert sorted(type(a).deformation_names) == sorted(type(b).deformation_names)
        for deformation in type(b).deformation_names:
            da, db = our_config.CODES[name](*size), ref_config.CODES[name](*size)
            da.deform(deformation)
            db.deform(deformation)
            _assert_codes_equal(da, db)
    assert made


@pytest.mark.parametrize("name", ["Toric2DCode", "RotatedPlanar3DCode", "XCubeCode", "RhombicToricCode"])
def test_noise_equal(name):
    a, b = our_config.CODES[name](3), ref_config.CODES[name](3)
    for rates, deformation in [((1 / 3, 1 / 3, 1 / 3), None), ((0.1, 0.2, 0.7), None), ((0.0, 0.0, 1.0), "XZZX"), ((0.05, 0.05, 0.9), "XY")]:
        try:
            mb = panqec.error_models.PauliErrorModel(*rates, deformation_name=deformation)
            pb = mb.probability_distribution(b, 0.1)
        except Exception as e:  # noqa: BLE001 - a deformation a code does not have raises in both
            ma = ours.error_models.PauliErrorModel(*rates, deformation_name=deformation)
            with pytest.raises(type(e)):
                ma.probability_distribution(a, 0.1)
            continue
        ma = ours.error_models.PauliErrorModel(*rates, deformation_name=deformation)
        for x, y in zip(ma.probability_distribution(a, 0.1), pb):
            np.testing.assert_array_equal(x, y)
        for seed in range(3):
            np.testing.assert_array_equal(ma.generate(a, 0.1, rng=np.random.default_rng(seed)), mb.generate(b, 0.1, rng=np.random.default_rng(seed)))


DECODER_CASES = [
    ("MatchingDecoder", "Toric2DCode", (4,), {}),
    ("MatchingDecoder", "Planar2DCode", (3, 4), {}),
    ("MatchingDecoder", "RotatedPlanar2DCode", (5,), {}),
    ("SweepMatchDecoder", "Toric3DCode", (3,), {}),
    ("SweepMatchDecoder", "Planar3DCode", (3,), {}),
    ("RotatedSweepMatchDecoder", "RotatedPlanar3DCode", (3,), {}),
    ("RotatedSweepMatchDecoder", "RotatedToric3DCode", (4, 4, 3), {}),
    ("XCubeMatchingDecoder", "XCubeCode", (3,), {}),
    ("BeliefPropagationOSDDecoder", "Toric2DCode", (4,), {}),
    ("BeliefPropagationOSDDecoder", "Color666PlanarCode", (3,), {}),
    ("BeliefPropagationOSDDecoder", "RhombicToricCode", (2,), {"channel_update": True, "osd_order": 2}),
    ("UnionFindDecoder", "Toric2DCode", (5,), {}),
]
MATCHING = {"MatchingDecoder", "SweepMatchDecoder", "RotatedSweepMatchDecoder", "XCubeMatchingDecoder"}
IDS = [f"{d}-{c}" for d, c, _, _ in DECODER_CASES]


def _decodes(decoder, code, size, kwargs, shots=25, rate=0.05):
    """Each package's corrections on the same errors (asserted equal), with its code."""
    ca, cb = our_config.CODES[code](*size), ref_config.CODES[code](*size)
    ma, mb = ours.error_models.PauliErrorModel(0.2, 0.3, 0.5), panqec.error_models.PauliErrorModel(0.2, 0.3, 0.5)
    da, db = our_config.DECODERS[decoder](ca, ma, rate, **kwargs), ref_config.DECODERS[decoder](cb, mb, rate, **kwargs)
    ra, rb = np.random.default_rng(11), np.random.default_rng(11)
    if code == "RotatedToric3DCode":
        shots = 8  # its sweeps are slow
    out = []
    for _ in range(shots):
        ea, eb = ma.generate(ca, rate, rng=ra), mb.generate(cb, rate, rng=rb)
        np.testing.assert_array_equal(ea, eb)
        sa, sb = ca.measure_syndrome(ea), cb.measure_syndrome(eb)
        np.testing.assert_array_equal(sa, sb)
        out.append((sa, np.asarray(da.decode(sa)), np.asarray(db.decode(sb))))
    return ca, ma, rate, out


@pytest.fixture
def pymatching_matcher(monkeypatch):
    """PanQEC's matching decoders on PyMatching itself, whose ties between equally good matchings
    this package's matcher does not break alike."""
    import pymatching

    monkeypatch.setattr("stabilizer_qec.panqec.decoders.matching._matching_decoder.Matching", pymatching.Matching)


@pytest.mark.parametrize("decoder,code,size,kwargs", DECODER_CASES, ids=IDS)
def test_decoders_equal(decoder, code, size, kwargs, pymatching_matcher, capsys):
    for _, a, b in _decodes(decoder, code, size, kwargs)[3]:
        np.testing.assert_array_equal(a, b)
    capsys.readouterr()


@pytest.mark.parametrize("decoder,code,size,kwargs", [c for c in DECODER_CASES if c[0] in MATCHING], ids=[i for i, c in zip(IDS, DECODER_CASES) if c[0] in MATCHING])
def test_matching_decoders_on_this_packages_matcher(decoder, code, size, kwargs, capsys):
    # On this package's matcher: the same syndrome left, and for plain matching, corrections of
    # the same weight (both minimum weight), where ties allow different ones.
    c, model, rate, out = _decodes(decoder, code, size, kwargs)
    wx, wz = model.get_weights(c, rate)
    for _, a, b in out:
        np.testing.assert_array_equal(c.measure_syndrome(a), c.measure_syndrome(b))
        if decoder == "MatchingDecoder":
            n = c.n
            assert np.dot(a[:n], wx) + np.dot(a[n:], wz) == pytest.approx(np.dot(b[:n], wx) + np.dot(b[n:], wz))
    capsys.readouterr()


def test_memory_belief_propagation(capsys):
    # PanQEC 0.1.7's MBP raises on NumPy 2 (-1 to an unsigned syndrome's power); this port's
    # takes the syndrome as an integer, and corrects single errors.
    c = our_config.CODES["Toric2DCode"](3)
    model = ours.error_models.PauliErrorModel(1 / 3, 1 / 3, 1 / 3)
    d = our_config.DECODERS["MemoryBeliefPropagationDecoder"](c, model, 0.1, max_bp_iter=10)
    for q in (0, 5):
        error = np.zeros(2 * c.n, dtype=np.uint)
        error[c.n + q] = 1
        correction = d.decode(c.measure_syndrome(error))
        np.testing.assert_array_equal(c.measure_syndrome(correction), c.measure_syndrome(error))
    if int(np.__version__.split(".")[0]) >= 2:
        cb = ref_config.CODES["Toric2DCode"](3)
        db = ref_config.DECODERS["MemoryBeliefPropagationDecoder"](cb, panqec.error_models.PauliErrorModel(1 / 3, 1 / 3, 1 / 3), 0.1, max_bp_iter=10)
        with pytest.raises(OverflowError):
            db.decode(cb.measure_syndrome(error))
    capsys.readouterr()


def test_direct_simulation_equal(pymatching_matcher, capsys):
    results = []
    for pkg, config in ((ours, our_config), (panqec, ref_config)):
        code = config.CODES["Toric3DCode"](3)
        model = pkg.error_models.PauliErrorModel(1 / 3, 1 / 3, 1 / 3)
        decoder = config.DECODERS["SweepMatchDecoder"](code, model, 0.08)
        sim = pkg.simulation.DirectSimulation(code, model, decoder, 0.08, verbose=False, rng=np.random.default_rng(3))
        sim.run(40)
        r = sim.get_results()
        results.append((r["n_fail"], r["n_runs"], [list(map(int, np.ravel(e))) for e in sim.results["effective_error"]], list(map(bool, sim.results["codespace"]))))
    assert results[0] == results[1]
    capsys.readouterr()


def test_gui_answers_equal():
    pytest.importorskip("flask")
    from panqec.gui import GUI as RefGUI

    from stabilizer_qec.panqec.gui import GUI

    clients = [GUI().app.test_client(), RefGUI().app.test_client()]

    def both(path, payload=None):
        a, b = (c.post(path, json=payload) if payload is not None else c.get(path) for c in clients)
        assert a.status_code == b.status_code == 200, path
        return a.data, b.data

    for path in ("/", "/2d", "/3d", "/main.css", "/js/main.js"):
        a, b = both(path)
        # A Windows checkout ends the port's files' lines CRLF.
        assert a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")
    for dim in (2, 3):
        a, b = both("/code-names", {"dimension": dim})
        assert json.loads(a) == json.loads(b)
    for name, size, rotated in [("Toric 2D", (4, 4), False), ("Rotated Planar 3D", (3, 3, 3), True)]:
        names = json.loads(both("/code-names", {"dimension": len(size)})[0])
        if name not in names:
            name = names[0]
        req = {"Lx": size[0], "Ly": size[1], "code_name": name, "code_deformation_name": "None", "rotated_picture": rotated}
        if len(size) == 3:
            req["Lz"] = size[2]
        a, b = both("/code-data", req)
        assert json.loads(a) == json.loads(b)
        a, b = both("/decoder-names", {"code_name": name})
        assert json.loads(a) == json.loads(b)
        data = json.loads(a)
        H = np.array(json.loads(both("/code-data", req)[0])["H"])
        n = H.shape[1] // 2
        error = np.zeros(2 * n, dtype=int)
        error[[0, n + 1]] = 1
        syndrome = ((H[:, :n] @ error[n:] + H[:, n:] @ error[:n]) % 2).tolist()
        for decoder in data:
            if decoder == "MBP":
                continue
            payload = {**req, "syndrome": syndrome, "p": 0.1, "noise_deformation_name": "None", "max_bp_iter": 10, "alpha": 0.4, "beta": 0, "decoder": decoder, "error_model": list(_noise_names())[0]}
            a, b = both("/decode", payload)
            assert json.loads(a) == json.loads(b), decoder


def _noise_names():
    from panqec.gui import _gui

    return _gui.noise_directions
