"""stabilizer_qec.tesseract_decoder given Stim's own models: the configurations, model helpers
and demutil functions give back Stim's models (as Tesseract's do) when given Stim's, and this
package's when given this package's; and a decoder's configuration is read at every decode.
(test_tesseract_package.py compares the same against Tesseract itself.)"""

from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")

import stabilizer_qec as sq  # noqa: E402
import stabilizer_qec.tesseract_decoder as O  # noqa: E402

CIRCUIT = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.01)


def kinds(theirs, ours):
    assert isinstance(theirs, stim.DetectorErrorModel) and isinstance(ours, sq.DetectorErrorModel)
    assert theirs == stim.DetectorErrorModel(str(ours))


def test_configurations_keep_the_kind_of_model():
    dem = CIRCUIT.detector_error_model()
    mine = sq.DetectorErrorModel(str(dem))
    for make in (O.tesseract.TesseractConfig, O.simplex.SimplexConfig):
        a, b = make(dem), make(mine)
        kinds(a.dem, b.dem)
        assert a.dem is not a.dem and b.dem is not b.dem  # (copies, as Tesseract's)
        kinds(a.compile_decoder().config.dem, b.compile_decoder().config.dem)
        a.dem, b.dem = mine, dem
        kinds(b.dem, a.dem)
    assert isinstance(O.tesseract.TesseractConfig().dem, sq.DetectorErrorModel)
    assert isinstance(O.tesseract.TesseractConfig("error(0.1) D0").dem, sq.DetectorErrorModel)


def test_model_helpers_keep_the_kind_of_model():
    dem = CIRCUIT.detector_error_model()
    mine = sq.DetectorErrorModel(str(dem))
    for helper in (O.common.merge_indistinguishable_errors, O.common.remove_zero_probability_errors):
        kinds(helper(dem), helper(mine))
    counts = list(range(dem.num_errors))
    kinds(O.common.dem_from_counts(dem, counts, 1000), O.common.dem_from_counts(mine, counts, 1000))
    assert O.common.dem_from_counts(dem, counts, 1000).approx_equals(O.common.dem_from_counts(dem, counts, 1000), atol=0)
    # (Stim's own model operations take what the helpers give back.)
    assert (dem + O.common.remove_zero_probability_errors(dem)).num_errors == 2 * dem.num_errors


def test_demutil_keeps_the_kind_of_model():
    d = O.demutil
    dem = CIRCUIT.detector_error_model()
    mine = sq.DetectorErrorModel(str(dem))
    classifier = d.stim_surface_code_detector_basis_classifier
    kinds(d.decompose_errors(dem), d.decompose_errors(mine))
    kinds(d.decompose_errors(dem=dem), d.decompose_errors(dem=mine))
    kinds(d.decompose_errors_using_detector_basis_classifier(dem, classifier), d.decompose_errors_using_detector_basis_classifier(mine, classifier))
    kinds(d.annotate_detector_bases(dem, detector_basis_classifier=classifier), d.annotate_detector_bases(mine, detector_basis_classifier=classifier))
    assert d.classify_detector_bases(dem, detector_basis_classifier=classifier) == d.classify_detector_bases(mine, detector_basis_classifier=classifier)
    flat = dem.flattened()
    kinds(d.regeneralize_spatial_dem([flat], flat), d.regeneralize_spatial_dem([sq.DetectorErrorModel(str(flat))], sq.DetectorErrorModel(str(flat))))


def test_decoder_config_is_read_at_every_decode():
    dem = sq.DetectorErrorModel(str(CIRCUIT.detector_error_model()))
    shots = CIRCUIT.compile_detector_sampler(seed=11).sample(40)
    # (Beam climbing finds the same errors on this model; tests/test_tesseract_package.py
    # checks it live against Tesseract.)
    for field, value, base in (("det_beam", 0, {}), ("pqlimit", 3, {}), ("det_penalty", 5.0, {})):
        d = O.tesseract.TesseractConfig(dem, **base).compile_decoder()
        before = [d.decode_to_errors(s) for s in shots]
        setattr(d.config, field, value)
        fresh = O.tesseract.TesseractConfig(dem, **base, **{field: value}).compile_decoder()
        after = [(d.decode_to_errors(s), d.low_confidence_flag) for s in shots]
        assert after == [(fresh.decode_to_errors(s), fresh.low_confidence_flag) for s in shots]
        assert [a for a, _ in after] != before, field
    d = O.tesseract.TesseractConfig(dem).compile_decoder()
    d.config.det_orders = [list(range(dem.num_detectors))[::-1]]
    fresh = O.tesseract.TesseractConfig(dem, det_orders=[list(range(dem.num_detectors))[::-1]]).compile_decoder()
    assert np.array_equal(d.decode_batch(shots), fresh.decode_batch(shots))
