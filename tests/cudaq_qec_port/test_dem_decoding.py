# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# Ported from CUDA-Q QEC 0.8.1's libs/qec/python/tests/test_dem.py (NVIDIA/cudaqx; Apache License
# 2.0): its two PyMatching decoding tests, which decode to the observables because the model has
# an observable matrix. Modified for stabilizer_qec: cudaq_qec is stabilizer_qec.cudaq_qec, the
# noise model is its NoiseModel, and the sampler takes its seed (for cudaq.set_random_seed(13)).
# Added: the same Stim model given to get_decoder as text, with no merge_strategy.
import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec


def test_pymatching_decode_to_observable_surface_code_dem():
    """Test PyMatching with O (observables) matrix: decoder returns observable
    flips directly."""
    code = qec.get_code('surface_code', distance=5)
    Lz = code.get_observables_z()
    p = 0.003
    noise = qec.NoiseModel()
    noise.add_all_qubit_channel("x", qec.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 5
    nShots = 2000

    syndromes, data = qec.z_sample_memory_circuit(code, statePrep, nShots, nRounds, noise, seed=13)

    logical_measurements = (Lz @ data.transpose()) % 2
    logical_measurements = logical_measurements.flatten()

    dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)

    decoder = qec.get_decoder(
        'pymatching',
        dem.detector_error_matrix,
        O=dem.observables_flips_matrix,
        error_rate_vec=np.array(dem.error_rates),
    )

    dr = decoder.decode_batch(syndromes)
    # With decode_to_observables=True, each row is observable flips
    # (length num_observables), not error predictions.
    obs_per_shot = np.asarray(dr.result, dtype=np.float64)
    data_predictions = np.round(obs_per_shot).astype(np.uint8).T

    nLogicalErrorsWithoutDecoding = np.sum(logical_measurements)
    nLogicalErrorsWithDecoding = np.sum(data_predictions ^ logical_measurements)
    assert nLogicalErrorsWithDecoding < nLogicalErrorsWithoutDecoding


def _stim_dem_to_arrays(stim_dem):
    # Split decomposed Stim errors into graphlike H/O columns for PyMatching.
    n_dets = stim_dem.num_detectors
    n_obs = stim_dem.num_observables
    h_cols, o_cols, rates = [], [], []
    for inst in stim_dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        components = [[]]
        for tgt in inst.targets_copy():
            if tgt.is_separator():
                components.append([])
            else:
                components[-1].append(tgt)
        for comp in components:
            h_col = np.zeros(n_dets, dtype=np.uint8)
            o_col = np.zeros(n_obs, dtype=np.uint8)
            for tgt in comp:
                if tgt.is_relative_detector_id():
                    h_col[tgt.val] = 1
                elif tgt.is_logical_observable_id():
                    o_col[tgt.val] = 1
            # Matching cannot use a purely-logical component with no syndrome.
            if h_col.sum() == 0:
                continue
            h_cols.append(h_col)
            o_cols.append(o_col)
            rates.append(prob)
    H = np.stack(h_cols, axis=1) if h_cols else np.zeros((n_dets, 0), dtype=np.uint8)
    O = np.stack(o_cols, axis=1) if o_cols else np.zeros((n_obs, 0), dtype=np.uint8)
    return H, O, np.asarray(rates, dtype=np.float64)


def _stim_surface_code(stim_mod):
    distance = 5
    n_rounds = 5
    p = 0.003
    return stim_mod.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=n_rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def _logical_errors(decoder, circuit, n_shots=2000):
    # Sample syndromes and true observable flips from the same Stim circuit.
    sampler = circuit.compile_detector_sampler(seed=13)
    syndromes_bool, obs_bool = sampler.sample(shots=n_shots, separate_observables=True)
    syndromes = syndromes_bool.astype(np.uint8)
    logical_measurements = obs_bool.astype(np.uint8).flatten()

    dr = decoder.decode_batch(syndromes)
    # With O provided, the decoder returns predicted observable flips.
    obs_per_shot = np.asarray(dr.result, dtype=np.float64)
    data_predictions = np.round(obs_per_shot).astype(np.uint8).flatten()
    return int(np.sum(data_predictions ^ logical_measurements)), int(np.sum(logical_measurements))


def test_pymatching_decodes_stim_surface_code_dem():
    # End-to-end smoke test: feed a Stim-generated rotated surface-code DEM
    # (which contains parallel matching edges) through cudaq-qec's PyMatching
    # plugin and confirm decoding reduces the logical error count.
    stim_mod = pytest.importorskip("stim", reason="stim not installed; skipping Stim-based PyMatching decode test")
    circuit = _stim_surface_code(stim_mod)
    # Matching plugin expects graphlike columns (1 or 2 detectors).
    stim_dem = circuit.detector_error_model(decompose_errors=True)

    H, O, rates = _stim_dem_to_arrays(stim_dem)

    # Surface-code DEMs have parallel edges; let PyMatching merge them.
    decoder = qec.get_decoder('pymatching', H, O=O, error_rate_vec=rates, merge_strategy='independent')

    # ASSERT: decoding the observable corrects more logical flips than leaving
    # the raw measurement uncorrected, i.e. the matching decode is useful.
    n_errors_with_decoding, n_errors_without_decoding = _logical_errors(decoder, circuit)
    assert n_errors_with_decoding < n_errors_without_decoding, (
        f"PyMatching did not reduce logical errors: with_decoding={n_errors_with_decoding}, without_decoding={n_errors_without_decoding}"
    )


def _graphlike_text(H, O, rates):
    # The columns of _stim_dem_to_arrays as model text, one error per column (CUDA-Q's get_decoder
    # reads a model's text without decomposition suggestions, so `^` would join the components).
    lines = []
    for c in range(H.shape[1]):
        targets = [f"D{d}" for d in np.flatnonzero(H[:, c])] + [f"L{o}" for o in np.flatnonzero(O[:, c])]
        lines.append(f"error({float(rates[c])!r}) " + " ".join(targets))
    return "\n".join(lines) + "\n"


def test_pymatching_decodes_stim_surface_code_dem_text_merging_by_default():
    # (Added.) The same model as text: make_pcm_decoder gives pymatching the model's O, so it
    # decodes to the observable, merging the parallel edges as independent without being asked.
    stim_mod = pytest.importorskip("stim", reason="stim not installed; skipping Stim-based PyMatching decode test")
    circuit = _stim_surface_code(stim_mod)
    H, O, rates = _stim_dem_to_arrays(circuit.detector_error_model(decompose_errors=True))

    from_text = qec.get_decoder('pymatching', _graphlike_text(H, O, rates))
    from_matrices = qec.get_decoder('pymatching', H, O=O, error_rate_vec=rates, merge_strategy='independent')
    syndromes = circuit.compile_detector_sampler(seed=13).sample(shots=200).astype(np.uint8)
    got = from_text.decode_batch(syndromes).result
    assert got.shape == (200, 1)
    assert np.array_equal(got, from_matrices.decode_batch(syndromes).result)

    n_errors_with_decoding, n_errors_without_decoding = _logical_errors(from_text, circuit)
    assert n_errors_with_decoding < n_errors_without_decoding
