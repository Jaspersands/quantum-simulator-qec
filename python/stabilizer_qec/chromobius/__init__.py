# Ported from Chromobius 1.1.1 (quantumlib/chromobius, src/chromobius/pybind/; Copyright 2023
# Google LLC, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: the
# decoder is this package's port (``ColorMatching``, src/color/).
"""Chromobius under its own names (``import stabilizer_qec.chromobius as chromobius``): the
colour-code decoder compiled for a model, its weighted predictions, and its sinter decoder.
Predictions are Chromobius's but for ties between equally light matchings; weights are
Chromobius's (PyMatching's discretized Möbius matching weight, as float32) bit for bit."""

from __future__ import annotations

import pathlib as _pathlib
from typing import Any as _Any

import numpy as _np

__version__ = "1.1.1"
__all__ = ["CompiledDecoder", "compile_decoder_for_dem", "sinter_decoders"]


def _model(dem: _Any) -> _Any:
    from .._circuit import DetectorErrorModel

    if isinstance(dem, DetectorErrorModel):
        return dem
    name = str(type(dem))
    if "stim." not in name or ".DetectorErrorModel" not in name:
        raise ValueError("dem must be a stim.DetectorErrorModel.")
    return DetectorErrorModel(str(dem))


class CompiledDecoder:
    """A Chromobius decoder for one model (``chromobius.CompiledDecoder``)."""

    def __init__(self, *args: _Any, **kwargs: _Any) -> None:
        raise TypeError("chromobius.CompiledDecoder: No constructor defined! Use compile_decoder_for_dem or CompiledDecoder.from_dem.")

    @staticmethod
    def from_dem(dem: _Any) -> "CompiledDecoder":
        """Compiles a decoder for a detector error model (its detectors annotated with
        Chromobius's 4th coordinate: 0–2 red, green, blue X; 3–5 the same in Z; −1 ignored)."""
        from .._decoders import ColorMatching

        model = _model(dem)
        out = object.__new__(CompiledDecoder)
        out._decoder = ColorMatching(model)
        out._num_detector_bytes = (model.num_detectors + 7) // 8
        out._num_observable_bytes = (model.num_observables + 7) // 8
        return out

    def _predict(self, dets: _Any, include_weight: bool) -> _Any:
        if not isinstance(dets, _np.ndarray) or dets.dtype != _np.uint8:
            raise ValueError("Expected bit packed detection event data, but dets.dtype wasn't np.uint8.")
        if dets.ndim == 2:
            if dets.shape[0] and dets.strides[1] != 1:
                raise ValueError(f"Bit packed shot data must be contiguous in memory, but dets.stride[1] wasn't equal to 1.\nIt was {dets.strides[1]}.")
            rows = dets
        elif dets.ndim == 1:
            rows = dets.reshape(1, -1)
        else:
            raise ValueError("dets.shape not in [1, 2]")
        if rows.shape[1] != self._num_detector_bytes:
            raise ValueError(
                f"Expected dets.shape[1]={rows.shape[1]} == num_detector_bytes={self._num_detector_bytes} because dets.shape == 2 "
                "and dets.dtype==np.uint8 indicating bit packed shots."
            )
        preds, weights = self._decoder.decode_batch(rows, bit_packed_shots=True, bit_packed_predictions=True, return_weights=True)
        # Chromobius keeps a 64-bit observable mask; bytes past it are zero.
        obs = _np.zeros((rows.shape[0], self._num_observable_bytes), dtype=_np.uint8)
        k = min(self._num_observable_bytes, 8, preds.shape[1])
        obs[:, :k] = preds[:, :k]
        w = weights.astype(_np.float32)
        if dets.ndim == 1:
            obs, w = obs[0], _np.array(w[0], dtype=_np.float32)
        return (obs, w) if include_weight else obs

    def predict_obs_flips_from_dets_bit_packed(self, dets: _Any) -> _np.ndarray:
        """Bit-packed predicted observable flips from bit-packed detection events (one shot as
        a 1-D array, or one per row)."""
        return self._predict(dets, False)

    def predict_weighted_obs_flips_from_dets_bit_packed(self, dets: _Any) -> tuple:
        """The predictions and each shot's weight: the Möbius matching's weight as PyMatching
        reports it (float32), not accounting for the lifting."""
        return self._predict(dets, True)


def compile_decoder_for_dem(dem: _Any) -> CompiledDecoder:
    """Compiles a decoder for a detector error model (see ``CompiledDecoder.from_dem``)."""
    return CompiledDecoder.from_dem(dem)


class _ChromobiusSinterCompiledDecoder:
    """Chromobius for one model, as sinter calls it."""

    def __init__(self, decoder: CompiledDecoder) -> None:
        self._decoder = decoder

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: _Any) -> _np.ndarray:
        data = bit_packed_detection_event_data
        if not isinstance(data, _np.ndarray) or data.ndim != 2:
            raise ValueError("bit_packed_detection_event_data.ndim() != 2")
        if data.shape[0] and data.strides[1] != 1:
            raise ValueError("bit_packed_detection_event_data.strides(1) != 1")
        if data.shape[1] != self._decoder._num_detector_bytes:
            raise ValueError("bit_packed_detection_event_data.shape(1) != num_detector_bytes")
        return self._decoder.predict_obs_flips_from_dets_bit_packed(data.astype(_np.uint8, copy=False))


class _ChromobiusSinterDecoder:
    """Chromobius as a sinter decoder."""

    def __init__(self, sub_decoder: int = 0) -> None:
        self._sub_decoder = int(sub_decoder)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ChromobiusSinterDecoder) or NotImplemented

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, _ChromobiusSinterDecoder)

    __hash__ = None  # type: ignore[assignment]

    def __getstate__(self) -> int:
        return self._sub_decoder

    def __setstate__(self, state: int) -> None:
        self._sub_decoder = int(state)

    def compile_decoder_for_dem(self, *, dem: _Any) -> _ChromobiusSinterCompiledDecoder:
        """A decoder for the model ``dem``."""
        from .._circuit import DetectorErrorModel

        return _ChromobiusSinterCompiledDecoder(CompiledDecoder.from_dem(DetectorErrorModel(str(dem))))

    def decode_via_files(self, *, num_shots: int, num_dets: int, num_obs: int, dem_path: _Any, dets_b8_in_path: _Any, obs_predictions_b8_out_path: _Any, tmp_dir: _Any) -> None:
        """Decodes bit-packed detection events read from a file, writing the predictions to
        another (sinter's file protocol)."""
        from .._circuit import DetectorErrorModel

        decoder = CompiledDecoder.from_dem(DetectorErrorModel(_pathlib.Path(str(dem_path)).read_text()))
        nd = (int(num_dets) + 7) // 8
        raw = _pathlib.Path(str(dets_b8_in_path)).read_bytes()[: int(num_shots) * nd]
        rows = _np.frombuffer(raw, dtype=_np.uint8).reshape(int(num_shots), nd)
        preds = decoder.predict_obs_flips_from_dets_bit_packed(rows) if int(num_shots) else _np.zeros((0, (int(num_obs) + 7) // 8), _np.uint8)
        _pathlib.Path(str(obs_predictions_b8_out_path)).write_bytes(_np.ascontiguousarray(preds[:, : (int(num_obs) + 7) // 8]).tobytes())


def sinter_decoders() -> dict:
    """``{'chromobius': ...}``, a sinter decoder, for ``sinter.collect(custom_decoders=...)``."""
    return {"chromobius": _ChromobiusSinterDecoder(0)}
