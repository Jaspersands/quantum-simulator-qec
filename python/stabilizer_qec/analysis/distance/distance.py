# Ported from qec-util 0.5.0 (Copyright (c) 2023 MarcSerraPeralta, MIT License; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: Stim and PyMatching are this package's, and numpy stands in for its other dependencies.
from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
import numpy.typing as npt
import stabilizer_qec as stim

from ..dems import (
    get_errors_triggering_detectors,
    observables_to_detectors,
    only_errors,
)


class Decoder:
    def __init__(self, dem: stim.DetectorErrorModel, **kargs: object) -> None: ...
    def decode_to_faults_array(
        self, syndrome: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.integer]: ...


def get_circuit_distance(
    dem: stim.DetectorErrorModel,
) -> tuple[int, stim.DetectorErrorModel]:
    """Returns the circuit distance of the given circuit.

    Parameters
    ----------
    dem
        Detector error model. This function works better if 'dem' does not
        contain gauge detectors.

    Returns
    -------
    d_circ
        Circuit distance of the given circuit.
    errors
        Set of faults that makes the circuit distance be ``d_circ``.

    Notes
    -----
    Solved exactly as an integer program by scipy's HiGHS (qec-util uses Gurobi): this
    needs scipy.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise ValueError(
            f"'dem' must be a 'stim.DetectorErrorModel', but {type(dem)} was given."
        )
    if dem.num_observables == 0:
        return 0, stim.DetectorErrorModel()

    d_circ: int = np.inf
    error: stim.DetectorErrorModel = None
    for o in range(dem.num_observables):
        _d_circ, _error = get_circuit_distance_observable(dem, obs_inds=[o])
        if _d_circ < d_circ:
            d_circ, error = _d_circ, _error
    return d_circ, error


def get_circuit_distance_observable(
    dem: stim.DetectorErrorModel, obs_inds: int | Sequence[int]
) -> tuple[int, stim.DetectorErrorModel]:
    """Returns the minimum number of faults to flip (all) the specified logical
    observable(s) without triggering any detectors given the detector error model.
    Note that the returned error can trigger other observables that are not in ``obs_inds``.

    Parameters
    ----------
    dem
        Detector error model.
    obs_inds
        Index(s) of the logical observable in the ``dem``.

    Returns
    -------
    d_circ
        Circuit distance of the ``obs_ind``.
    errors
        Set of faults that makes the circuit distance be ``d_circ``.

    Notes
    -----
    Solved exactly as an integer program by scipy's HiGHS (qec-util uses Gurobi): this
    needs scipy.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )
    if isinstance(obs_inds, int):
        obs_inds = [obs_inds]
    if not isinstance(obs_inds, Sequence):
        raise TypeError(
            f"'obs_inds' must be an int or a list[int], but {type(obs_inds)} was given."
        )

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as ex:
        raise ImportError("an exact circuit distance needs scipy: pip install scipy") from ex

    dem = dem.flattened()
    dem = only_errors(dem)
    obs_det_inds = [dem.num_detectors + i for i, _ in enumerate(obs_inds)]
    new_dem = observables_to_detectors(dem, obs_inds=obs_inds, det_inds=obs_det_inds)
    det_support = get_errors_triggering_detectors(new_dem)

    # Variables: a 0/1 per error, then an integer per detector. Each detector's errors sum to
    # twice its integer, plus one for the observables' detectors.
    num_errors, num_dets = new_dem.num_errors, new_dem.num_detectors
    rows = [(det_id, support) for det_id, support in det_support.items() if len(support)]
    a = lil_matrix((len(rows), num_errors + num_dets))
    rhs = np.zeros(len(rows))
    for r, (det_id, support) in enumerate(rows):
        for e in support:
            a[r, e] = 1
        a[r, num_errors + det_id] = -2
        rhs[r] = 1 if det_id in obs_det_inds else 0
    cost = np.concatenate([np.ones(num_errors), np.zeros(num_dets)])
    result = milp(
        cost,
        integrality=np.ones(num_errors + num_dets),
        bounds=Bounds(np.zeros(num_errors + num_dets), np.concatenate([np.ones(num_errors), np.full(num_dets, np.inf)])),
        constraints=[LinearConstraint(a.tocsr(), rhs, rhs)] if rows else [],
    )
    if result.x is None:
        raise ValueError(f"no set of errors flips the observables {list(obs_inds)} and no detector ({result.message})")
    error_ids = [k for k in range(num_errors) if result.x[k] > 0.5]

    d_circ = len(error_ids)
    errors = stim.DetectorErrorModel()
    for error_id in error_ids:
        errors.append(dem[error_id])

    return d_circ, errors


def get_upper_bound_circuit_distance(
    dem: stim.DetectorErrorModel, decoder: type[Decoder] | None = None, **kargs: object
) -> tuple[int, stim.DetectorErrorModel]:
    """Returns an upper bound for the circuit distance.

    Parameters
    ----------
    dem
        Detector error model.
    decoder
        Decoder used to obtain the upper bound. The inputs for the initialization
        must be ``dem`` and ``**kargs``. It must have the ``decode_to_faults_array``
        which must return an array of the predicted faults/errors for the given syndrome.
        By default ``None``, which loads BPOSD.
    **kargs
        Extra arguments for the initialization of the decoder with ``dem``.

    Returns
    -------
    int
        Upper bound for the circuit distance.
    error
        Error corresponding to the upper bound.

    Notes
    -----
    The default decoder is this package's BP+OSD (qec-util's is ``dem_decoders``'), with the
    same options.
    """
    if kargs == {} and (decoder is None):
        decoder = _BpOsdFaults

        # default arguments for BP_OSD
        kargs = {
            "max_iter": 50,
            "bp_method": "ps",
            "osd_order": 5,
            "osd_method": "osd_cs",
        }

    num_obs = dem.num_observables
    if num_obs == 0:
        raise ValueError("'dem' does not contain any logical observable.")

    dem = dem.flattened()
    new_dem = observables_to_detectors(dem)
    decoder_dem = decoder(new_dem, **kargs)

    num_faults: int = np.inf
    error = stim.DetectorErrorModel()
    for comb in product([False, True], repeat=num_obs):
        if not any(comb):
            # skip [0,0,0...] case
            continue

        syndrome = np.zeros(new_dem.num_detectors, dtype=bool)
        syndrome[-num_obs:] = np.array(comb, dtype=bool)

        faults = decoder_dem.decode_to_faults_array(syndrome)
        if np.sum(faults) > num_faults:
            continue

        fault_inds = np.where(faults)[0]
        error = stim.DetectorErrorModel()
        for fault_ind in fault_inds:
            error += dem[fault_ind : fault_ind + 1]  # object is a DetectorErrorModel
        num_faults = len(error)

    num_faults = num_faults if num_faults != np.inf else -1

    return num_faults, error


class _BpOsdFaults:
    """BP+OSD over a model's errors, as ``dem_decoders``' ``BP_OSD``: ``decode_to_faults_array``
    gives the errors (one per instruction of the flattened model) explaining a syndrome."""

    def __init__(self, dem: stim.DetectorErrorModel, **kargs: object) -> None:
        from stabilizer_qec import BpOsdDecoder

        errors = [instr for instr in dem.flattened() if instr.type == "error"]
        pcm = np.zeros((dem.num_detectors, len(errors)), dtype=np.uint8)
        for k, instr in enumerate(errors):
            for t in instr.targets_copy():
                if t.is_relative_detector_id():
                    pcm[t.val, k] ^= 1
        priors = [instr.args_copy()[0] for instr in errors]
        self._decoder = BpOsdDecoder(pcm, error_channel=priors, **kargs)

    def decode_to_faults_array(self, syndrome: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        return self._decoder.decode(np.asarray(syndrome, dtype=np.uint8)).astype(bool)
