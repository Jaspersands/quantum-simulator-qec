# Ported from Chromobius 1.1.1 (quantumlib/chromobius, src/chromobius/pybind/; Copyright 2023
# Google LLC, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: the
# decoder is this package's port (``ColorMatching``, src/color/).
"""Chromobius under its own names (``import stabilizer_qec.chromobius as chromobius``): the
colour-code decoder compiled for a model, its weighted predictions, and its sinter decoder.
The Möbius matching is PyMatching 2.2.2's sparse blossom, which Chromobius links, to the edge
and its order (``pm_exact``), so weights (PyMatching's discretized matching weight, as float32)
are Chromobius's bit for bit and predictions are Chromobius's: on macOS on every one of 65,000
colour-code shots checked. (Chromobius walks some tables in std::unordered_map order, which is
the C++ library's own; with another library it may drag charge another way on some shots.)

Chromobius's command-line program is ``python -m stabilizer_qec.chromobius`` (``predict``,
``benchmark``, ``describe_decoder``; see its ``__main__``).

Where it differs from Chromobius:

- A model with a fault of probability 1 is refused (Chromobius's weights are then NaN);
  probabilities above 1/2 are negative-weight edges, as in Chromobius.
- A model declaring an observable past L63 that no error names (``logical_observable L64``) is
  refused with this package's 64-observable message, where Chromobius compiles it (its
  predictions past L63 are undefined). An error naming one is refused with Chromobius's
  message.
- ``decode_via_files`` with records wider than the model refuses a set bit past its detectors,
  as Chromobius does, but may name another detector."""

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
        out._num_detectors = model.num_detectors
        out._num_detector_bytes = (model.num_detectors + 7) // 8
        out._num_observable_bytes = (model.num_observables + 7) // 8
        return out

    def _predict(self, dets: _Any, include_weight: bool) -> _Any:
        if not isinstance(dets, _np.ndarray) or dets.dtype != _np.uint8:
            raise ValueError("Expected bit packed detection event data, but dets.dtype wasn't np.uint8.")
        if dets.ndim == 2:
            # (An empty batch is refused too: Chromobius's binding sees its stride as 0.)
            stride = dets.strides[1] if dets.shape[0] else 0
            if stride != 1:
                raise ValueError(f"Bit packed shot data must be contiguous in memory, but dets.stride[1] wasn't equal to 1.\nIt was {stride}.")
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
        _check_padding(rows, self._num_detectors)
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


def _check_padding(rows: _Any, num_detectors: int) -> None:
    """Refuse a shot with bits set past the last detector, as Chromobius does (its matcher is
    handed both Möbius copies of each such detection event, beyond the graph's nodes): the
    first such shot's lowest such bit is named."""
    if not rows.size or 8 * rows.shape[1] <= num_detectors:
        return
    first, used = num_detectors // 8, num_detectors % 8
    extra = rows[:, first:].copy()
    extra[:, 0] &= 0xFF ^ ((1 << used) - 1)
    shots = _np.flatnonzero(extra.any(axis=1))
    if shots.size:
        row = extra[shots[0]]
        k = int(_np.flatnonzero(row)[0])
        low = int(row[k])
        d = 8 * (first + k) + (low & -low).bit_length() - 1
        raise ValueError(f"The detection event with index {2 * d} does not correspond to a node in the graph, which only has {2 * num_detectors} nodes.")


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
        # Chromobius's binding casts the data to bytes (any integer dtype, any layout); an
        # empty batch it refuses, its stride then reading as 0.
        if not data.shape[0]:
            raise ValueError("bit_packed_detection_event_data.strides(1) != 1")
        data = _np.ascontiguousarray(data.astype(_np.uint8, copy=False))
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
        shots, nd, nob = int(num_shots), (int(num_dets) + 7) // 8, (int(num_obs) + 7) // 8
        try:
            raw = _pathlib.Path(str(dets_b8_in_path)).read_bytes()[: shots * nd]
        except OSError:
            raise ValueError(f"Failed to open '{dets_b8_in_path}' for reading.") from None
        if nd and len(raw) % nd:
            raise ValueError(
                f"b8 data ended in middle of record at byte position {len(raw) % nd}.\nExpected bytes per record was {nd} ({int(num_dets)} bits padded)."
            )
        # As Chromobius's: shots past the end of the file repeat its last record (none with an
        # empty file), and each prediction is ``num_obs`` bits wide (zero past the model's
        # observables).
        rows = _np.zeros((shots, nd), _np.uint8)
        read = len(raw) // nd if nd else 0
        if read and shots:
            rows[:read] = _np.frombuffer(raw, dtype=_np.uint8).reshape(-1, nd)
            rows[read:] = rows[read - 1]
        # (Each record is handed to the decoder whole: bits set past the model's detectors are
        # refused, and a record narrower than the model is padded with zeros.)
        _check_padding(rows, decoder._num_detectors)
        width = decoder._num_detector_bytes
        rows = _np.ascontiguousarray(rows[:, :width] if nd >= width else _np.pad(rows, ((0, 0), (0, width - nd))))
        out = _np.zeros((shots, nob), _np.uint8)
        if shots:
            preds = decoder.predict_obs_flips_from_dets_bit_packed(rows)
            k = min(nob, preds.shape[1])
            out[:, :k] = preds[:, :k]
        _pathlib.Path(str(obs_predictions_b8_out_path)).write_bytes(out.tobytes())


def sinter_decoders() -> dict:
    """``{'chromobius': ...}``, a sinter decoder, for ``sinter.collect(custom_decoders=...)``."""
    return {"chromobius": _ChromobiusSinterDecoder(0)}
