# This file is from IBM's relay_bp 0.2.2 (Copyright IBM 2025, Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: Stim, sinter, relay_bp and beliefmatching are this package's.
# relay_bp/stim/relay_sinter_decoder_logging.py
# (c) 2026 Relay-BP contributors.
#
# Sinter decoder that:
#   * builds a Relay-BP decoder directly from a Stim DEM (no external helpers),
#   * writes standard observable predictions (.b8 little-endian),
#   * writes a side-channel ".relay_converged.b8" (1 bit/shot: 1=converged),
#   * and MIRRORS the side-channel to a persistent directory so it survives
#     Sinter worker temp cleanup (env RELAY_SIDECAR_DIR or ./_relay_sidecars).
#
# Compatibility: Tested with Relay 0.2.1 where RelayDecoderF32/F64 constructors
# take (check_matrix, error_priors, ...) and NOT observable_matrix.
# Logical flips are computed as (H_obs @ error_bits) % 2.
#
# DEM parsing: We flatten the DEM then, for each `error(p) ...` line,
# every D# sets H_det[D, col]=1 and every L# sets H_obs[L, col]=1.

from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import re
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np
from stabilizer_qec import sinter
import stabilizer_qec as stim
from scipy.sparse import csc_matrix, csr_matrix


# --------------------------- Configuration ---------------------------

@dataclasses.dataclass
class RelayDecoderConfig:
    gamma0: float = 0.1
    pre_iter: int = 80
    num_sets: int = 60
    set_max_iter: int = 60
    gamma_dist_interval: Tuple[float, float] = (-0.24, 0.66)
    stop_nconv: int = 1
    show_progress: bool = False   # kept for future compatibility; not passed to 0.2.1 ctor
    dtype: str = "f32"            # "f32" or "f64"


# --------------------------- DEM -> Matrices ---------------------------

# Matches: error(p) <targets>  (optionally tagged like error[foo](p) <targets>)
_DEM_ERROR_RE = re.compile(r"^\s*error(?:\[[^\]]*\])?\s*\(([^)]*)\)\s*(.*)$")

def _dem_to_parity_checks(
    dem: stim.DetectorErrorModel,
    *,
    merge_duplicates: bool = False,
) -> Tuple[csr_matrix, csr_matrix, np.ndarray]:
    """Convert a flattened Stim DEM into (H_det, H_obs, error_probabilities).

    H_det:  (num_detectors x num_errors) CSR bool
    H_obs:  (num_observables x num_errors) CSR bool
    p:      (num_errors,) float probabilities

    Strategy:
      - Iterate lines of the flattened DEM program text.
      - For each `error(p) <targets>` line:
          * parse p (float),
          * parse tokens D# / L# among <targets> (ignore '^', comments),
          * add a new error column with 1s at those D/L rows.
    """
    dem = dem.flattened()  # inline repeats/shifts for simple text

    num_det = dem.num_detectors
    num_obs = dem.num_observables

    det_rows: List[int] = []
    det_cols: List[int] = []
    obs_rows: List[int] = []
    obs_cols: List[int] = []
    probs: List[float] = []

    # Optional merging: map (tuple(sorted D), tuple(sorted L)) -> col index
    pattern_to_idx: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}

    for raw_line in str(dem).splitlines():
        # strip inline comments
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        m = _DEM_ERROR_RE.match(line)
        if not m:
            continue  # skip non-error lines (detector declarations, coords, etc.)

        p_str, targets_str = m.groups()
        p = float(p_str.strip()) if p_str.strip() else 0.0

        # targets may include '^' separators (graphlike decomposition); ignore them.
        targets_str = targets_str.replace("^", " ")
        tokens = [t for t in targets_str.split() if t]

        det_ids: List[int] = []
        obs_ids: List[int] = []
        for t in tokens:
            if t[0] in ("D", "d"):
                det_ids.append(int(t[1:]))
            elif t[0] in ("L", "l"):
                obs_ids.append(int(t[1:]))
            else:
                # ignore stray tokens on error line; DEM error targets should be D#/L#
                pass

        if not det_ids and not obs_ids:
            continue

        if merge_duplicates:
            key = (tuple(sorted(det_ids)), tuple(sorted(obs_ids)))
            if key in pattern_to_idx:
                j = pattern_to_idx[key]
                # Independent XOR-combination: p <- p + p_new - 2 p p_new
                probs[j] = probs[j] + p - 2.0 * probs[j] * p
                continue
            else:
                j = len(probs)
                pattern_to_idx[key] = j
                probs.append(p)
        else:
            j = len(probs)
            probs.append(p)

        for d_id in det_ids:
            if 0 <= d_id < num_det:
                det_rows.append(d_id)
                det_cols.append(j)
        for l_id in obs_ids:
            if 0 <= l_id < num_obs:
                obs_rows.append(l_id)
                obs_cols.append(j)

    n_err = len(probs)
    H_det = csc_matrix(
        (np.ones(len(det_rows), dtype=np.uint8), (np.array(det_rows), np.array(det_cols))),
        shape=(num_det, n_err),
        dtype=np.uint8,
    ).tocsr()
    H_obs = csc_matrix(
        (np.ones(len(obs_rows), dtype=np.uint8), (np.array(obs_rows), np.array(obs_cols))),
        shape=(num_obs, n_err),
        dtype=np.uint8,
    ).tocsr()
    return H_det, H_obs, np.asarray(probs, dtype=np.float64)


# --------------------------- Sinter Decoder ---------------------------

class RelaySinterDecoderWithConvergence(sinter.Decoder):
    """Sinter decoder that logs per-shot convergence as a side-channel."""

    def __init__(self, **relay_kwargs: Any):
        self.cfg = RelayDecoderConfig(**relay_kwargs)
        # Cache per-DEM path
        self._compiled: Dict[str, Tuple[object, int, int]] = {}

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        runner, cached_dets, cached_obs = self._get_or_build_runner(dem_path=dem_path)
        if cached_dets != num_dets or cached_obs != num_obs:
            raise ValueError(
                f"[Relay] DEM sizes mismatch: runner (dets={cached_dets}, obs={cached_obs}) "
                f"vs Sinter request (dets={num_dets}, obs={num_obs})."
            )

        # Load b8-packed detection events and unpack to dense bits (little-endian).
        dets_b8 = np.fromfile(dets_b8_in_path, dtype=np.uint8)
        expected = num_shots * math.ceil(num_dets / 8)
        if dets_b8.size != expected:
            raise ValueError(f"[Relay] Unexpected dets_b8 size {dets_b8.size} (expected {expected}).")
        dets_b8 = dets_b8.reshape(num_shots, math.ceil(num_dets / 8))
        dets = np.unpackbits(dets_b8, axis=1, bitorder="little")[:, :num_dets].astype(np.uint8)

        preds = np.zeros((num_shots, num_obs), dtype=np.uint8)
        conv_bits = np.zeros((num_shots,), dtype=np.uint8)

        for s in range(num_shots):
            res = runner.decode_observables_detailed(dets[s])
            obs = np.asarray(res.observables, dtype=np.uint8)
            if obs.size != num_obs:
                raise ValueError(f"[Relay] Shot {s}: got {obs.size} observables, expected {num_obs}.")
            preds[s] = obs
            conv_bits[s] = 1 if bool(res.converged) else 0

        # Write standard predictions (b8 little-endian).
        np.packbits(preds, axis=1, bitorder="little").tofile(obs_predictions_b8_out_path)

        # Side-channel per-shot convergence (1=converged).
        side_path = _derive_side_channel_path(obs_predictions_b8_out_path, suffix=".relay_converged.b8")
        np.packbits(conv_bits, bitorder="little").tofile(side_path)

        # Optional tiny JSON summary (best-effort).
        try:
            import json
            summary = {
                "shots": int(num_shots),
                "num_dets": int(num_dets),
                "num_obs": int(num_obs),
                "converged": int(conv_bits.sum()),
                "non_converged": int(num_shots - conv_bits.sum()),
                "convergence_rate": float(conv_bits.mean()) if num_shots else None,
            }
            json_path = side_path.with_suffix(side_path.suffix + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            json_path = None

        # Mirror the side-channel (.b8 and optional .json) to a persistent dir so it won't be
        # deleted with Sinter's temporary worker directories.
        persist_dir = pathlib.Path(os.getenv("RELAY_SIDECAR_DIR", "./_relay_sidecars"))
        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
            # copy .b8
            persist_path = persist_dir / side_path.name
            shutil.copy2(side_path, persist_path)
            # copy JSON if present
            if json_path and pathlib.Path(json_path).exists():
                shutil.copy2(json_path, persist_dir / pathlib.Path(json_path).name)
            print(f"[Relay] Side-channel mirrored to: {persist_path}")
        except Exception as e:
            # non-fatal
            print(f"[Relay] WARNING: could not mirror side-channel: {e}")

    # ----------------------- internals -----------------------

    def _get_or_build_runner(self, dem_path: pathlib.Path) -> Tuple[object, int, int]:
        key = str(dem_path.resolve())
        if key in self._compiled:
            return self._compiled[key]

        dem = stim.DetectorErrorModel.from_file(dem_path)
        runner, num_dets, num_obs = self._build_runner_for_dem(dem)
        self._compiled[key] = (runner, num_dets, num_obs)
        return runner, num_dets, num_obs

    def _build_runner_for_dem(self, dem: stim.DetectorErrorModel) -> Tuple[object, int, int]:
        """Build a Relay observable decoder runner from a DEM (Relay 0.2.1 compatible)."""
        from stabilizer_qec.relay_bp import RelayDecoderF32, RelayDecoderF64  # constructor: check_matrix + error_priors

        H_det, H_obs, priors = _dem_to_parity_checks(dem, merge_duplicates=False)
        Dec = RelayDecoderF32 if self.cfg.dtype == "f32" else RelayDecoderF64

        decoder = Dec(
            check_matrix=H_det,
            error_priors=priors,
            gamma0=self.cfg.gamma0,
            pre_iter=self.cfg.pre_iter,
            num_sets=self.cfg.num_sets,
            set_max_iter=self.cfg.set_max_iter,
            gamma_dist_interval=self.cfg.gamma_dist_interval,
            stop_nconv=self.cfg.stop_nconv,
        )

        class _Runner:
            """Expose decode_observables_detailed(det_vec) expected by the logger."""
            def __init__(self, dec, Hobs):
                self.dec = dec
                self.Hobs = Hobs

            def decode_observables_detailed(self, det_vec: np.ndarray):
                det_vec = np.asarray(det_vec, dtype=np.uint8)
                dres = self.dec.decode_detailed(det_vec)              # detailed API with .decoding/.success
                err = np.asarray(dres.decoding, dtype=np.uint8).reshape(-1)
                obs = (self.Hobs @ err) % 2                           # compute logical flips from H_obs
                obs = np.asarray(obs, dtype=np.uint8).reshape(-1)
                return type("DecodeResult", (), {
                    "observables": obs,
                    "converged": bool(getattr(dres, "success", False)),
                })

        runner = _Runner(decoder, H_obs)
        return runner, dem.num_detectors, dem.num_observables


def _derive_side_channel_path(base: pathlib.Path, suffix: str) -> pathlib.Path:
    """Sibling filename for side-channel next to .b8 predictions."""
    return base.with_suffix(base.suffix.replace(".b8", "") + suffix)


# --------------------------- Sinter factory ---------------------------

def sinter_decoders_with_logging(**relay_kwargs: Any) -> Dict[str, sinter.Decoder]:
    """Return a dict usable as Sinter's `custom_decoders=` argument."""
    return {"relay-bp-logged": RelaySinterDecoderWithConvergence(**relay_kwargs)}
