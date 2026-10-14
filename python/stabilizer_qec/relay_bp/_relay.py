"""IBM's relay_bp 0.2.2 decoders on this package's Relay-BP engine (`src/relay.rs`), with
relay_bp's classes, arguments, defaults and results: Relay-BP and plain (Mem-)BP min-sum in
every number type relay_bp is built for, `DecodeResult`, `ObservableDecoderRunner` and
`ObservableDecodeResult`. The decodings, posteriors and iteration counts are relay_bp's bit for
bit (tests/test_relay_bp_package.py)."""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional, Tuple

import numpy as np

from .. import _core
from .._util import call

__all__ = [
    "DecodeResult",
    "RelayDecoderF32",
    "RelayDecoderF64",
    "RelayDecoderI32",
    "RelayDecoderI64",
    "MinSumBPDecoderF32",
    "MinSumBPDecoderF64",
    "MinSumBPDecoderI8",
    "MinSumBPDecoderI16",
    "MinSumBPDecoderI32",
    "MinSumBPDecoderI64",
    "MinSumBPDecoderFixed",
    "ObservableDecoderRunner",
    "ObservableDecodeResult",
]


def _sparse_bits(matrix: Any) -> Tuple[int, int, List[List[int]], Any]:
    """relay_bp's reading of a check matrix: ``astype("uint8")``, then a dense array or a csr or
    csc matrix, each stored entry an edge. (rows, columns, each column's rows, the matrix)."""
    m = matrix.astype("uint8")
    if isinstance(m, np.ndarray):
        if m.ndim != 2:
            raise TypeError("Input must be a numpy array or scipy sparse matrix (csr_matrix or csc_matrix)")
        rows, cols = m.shape
        return rows, cols, [np.flatnonzero(m[:, j]).tolist() for j in range(cols)], m
    if not hasattr(m, "tocsc"):
        raise TypeError("Input must be a numpy array or scipy sparse matrix (csr_matrix or csc_matrix)")
    name = type(m).__name__
    if name not in ("csr_matrix", "csc_matrix"):
        raise TypeError(f"Unsupported sparse format '{name}'. Only CSR and CSC are supported.")
    csc = m.tocsc()
    rows, cols = csc.shape
    return rows, cols, [sorted(int(r) for r in csc.indices[csc.indptr[j] : csc.indptr[j + 1]]) for j in range(cols)], m


def _bits_vector(x: Any, n: int, what: str) -> List[int]:
    v = np.asarray(x)
    if v.ndim != 1 or len(v) != n:
        raise ValueError(f"{what} must have {n} entries, not shape {v.shape}")
    return [int(b) for b in v.astype(np.uint8)]


def _parity(matrix: Any, vector: np.ndarray) -> np.ndarray:
    """relay_bp's ``mul_mod2``: the matrix (its stored values) times the vector, mod 2."""
    prod = matrix @ vector.astype(np.int64)
    return (np.asarray(prod).ravel() % 2).astype(np.uint8)


def _rust_debug(x: float) -> str:
    """Rust's ``{:?}`` of an f64 (as relay_bp writes it in its log)."""
    r = repr(float(x))
    if "e" in r:
        mantissa, exponent = r.split("e")
        r = f"{mantissa}e{int(exponent)}"
    return r


class _Progress:
    """A progress bar on stderr while a batch decodes (relay_bp shows indicatif's)."""

    def __init__(self, total: int, show: bool, leave: bool) -> None:
        self.total, self.show, self.leave, self.done = total, show and total > 0 and sys.stderr.isatty(), leave, 0

    def step(self) -> None:
        self.done += 1
        if self.show:
            width = 40
            filled = width * self.done // self.total
            sys.stderr.write(f"\r[{'#' * filled}{'-' * (width - filled)}] {self.done}/{self.total}")
            sys.stderr.flush()

    def close(self) -> None:
        if self.show:
            sys.stderr.write("\n" if self.leave else "\r" + " " * 70 + "\r")
            sys.stderr.flush()


class DecodeResult:
    """A decode's result, as relay_bp's: ``decoding``, ``decoded_detectors`` (the checks the
    decoding sets off), ``posterior_ratios``, ``success``, ``iterations`` and ``max_iter``."""

    __slots__ = ("_decoding", "_decoded", "_posterior", "_success", "_iterations", "_max_iter", "_quality")

    def __init__(self, raw: tuple) -> None:
        correction, decoded, posterior, success, iterations, max_iter, quality = raw
        self._decoding = np.asarray(correction, dtype=np.uint8)
        self._decoded = np.asarray(decoded, dtype=np.uint8)
        self._posterior = np.asarray(posterior, dtype=np.float64)
        self._success, self._iterations, self._max_iter, self._quality = bool(success), int(iterations), int(max_iter), float(quality)

    @property
    def decoding(self) -> np.ndarray:
        return self._decoding.copy()

    @property
    def decoded_detectors(self) -> np.ndarray:
        return self._decoded.copy()

    @property
    def posterior_ratios(self) -> np.ndarray:
        return self._posterior.copy()

    @property
    def success(self) -> bool:
        return self._success

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def max_iter(self) -> int:
        return self._max_iter

    def __repr__(self) -> str:
        return f"DecodeResult(success={self._success}, iterations={self._iterations}, max_iter={self._max_iter})"


class _Decoder:
    """What every relay_bp decoder shares (its ``DynDecoder``)."""

    _x: Any
    _m: int
    _n: int
    _log: Optional[str] = None

    def _run(self, detectors: Any) -> tuple:
        raw = call(self._x.decode, _bits_vector(detectors, self._m, "detectors"))
        self._write_log()
        return raw

    def _write_log(self) -> None:
        if self._log is not None:
            with open(self._log, "a") as f:
                for line in self._x.log_lines():
                    f.write(line + "\n")

    def decode(self, detectors: Any) -> np.ndarray:
        """The correction for one shot's detectors."""
        return np.asarray(self._run(detectors)[0], dtype=np.uint8)

    def decode_detailed(self, detectors: Any) -> DecodeResult:
        """The correction with how it was found."""
        return DecodeResult(self._run(detectors))

    def decode_batch(self, detectors: Any) -> np.ndarray:
        """Each row's correction, decoded in turn."""
        rows = np.asarray(detectors)
        out = np.zeros((rows.shape[0], self._n), dtype=np.uint8)
        for i, row in enumerate(rows):
            out[i] = self.decode(row)
        return out

    def decode_detailed_batch(self, detectors: Any) -> List[DecodeResult]:
        return [self.decode_detailed(row) for row in np.asarray(detectors)]


def _engine(self: _Decoder, check_matrix: Any, error_priors: Any, kind: str, frac_bits: Optional[int], relay: bool, **options: Any) -> None:
    self._m, self._n, cols, self._matrix = _sparse_bits(check_matrix)
    priors = np.asarray(error_priors, dtype=np.float64).ravel()
    if len(priors) != self._n:
        raise ValueError(f"error_priors must have {self._n} entries (one per column), not {len(priors)}")
    self._priors = priors
    self._x = call(_core.RelayEngine, self._m, cols, [float(p) for p in priors], kind, frac_bits, relay, *options.values())


_STOPPING = {"pre_iter", "nconv", "all"}


class _Relay(_Decoder):
    _kind = "f64"

    def __init__(
        self,
        check_matrix: Any,
        error_priors: Any,
        alpha: Optional[float] = None,
        alpha_iteration_scaling_factor: float = 1.0,
        gamma0: Optional[float] = 0.1,
        data_scale_value: Optional[float] = None,
        max_data_value: Optional[float] = None,
        pre_iter: int = 80,
        num_sets: int = 300,
        set_max_iter: int = 60,
        gamma_dist_interval: Tuple[float, float] = (-0.24, 0.66),
        explicit_gammas: Any = None,
        stop_nconv: int = 1,
        stopping_criterion: str = "nconv",
        logging: bool = False,
        seed: int = 0,
    ) -> None:
        gammas = None
        if explicit_gammas is not None:
            g = np.asarray(explicit_gammas, dtype=np.float64)
            if g.ndim != 2:
                raise ValueError("explicit_gammas must be two-dimensional: a row of memory strengths per leg")
            if num_sets > g.shape[0]:
                import warnings

                warnings.warn(f"Number of different gamma sets {g.shape[0]} is smaller than the number of Relay legs {num_sets}. Legs will be reused.")
            gammas = g.tolist()
        # relay_bp's reading: an unknown criterion is "nconv" stopping after one solution,
        # whatever stop_nconv says.
        crit = stopping_criterion if stopping_criterion in _STOPPING else "nconv"
        solutions = None if crit == "all" else (stop_nconv if stopping_criterion == "nconv" else 1)
        _engine(
            self, check_matrix, error_priors, self._kind, None, True,
            pre_iter=pre_iter, legs=num_sets, leg_iter=set_max_iter, solutions=solutions, stop_after_first=crit == "pre_iter",
            gamma0=gamma0, gamma_range=tuple(map(float, gamma_dist_interval)), gammas=gammas, alpha=alpha, alpha_scaling=float(alpha_iteration_scaling_factor),
            data_scale=data_scale_value, max_data=max_data_value, logging=bool(logging), seed=int(seed),
        )
        if logging:
            # relay_bp writes its log to relay_logging.out in the working directory.
            self._log = os.path.abspath("relay_logging.out")
            with open(self._log, "w") as f:
                f.write(
                    f"# pre_iter: {pre_iter}: sets: {num_sets} set_max_iter: {set_max_iter}\n"
                    f"# gamma_distribution: ({_rust_debug(gamma_dist_interval[0])}, {_rust_debug(gamma_dist_interval[1])}) # set_idx, num_iter, converged, unique_best_solution\n"
                )


class _MinSum(_Decoder):
    _kind = "f64"

    def __init__(
        self,
        check_matrix: Any,
        error_priors: Any,
        max_iter: int = 200,
        alpha: Optional[float] = None,
        alpha_iteration_scaling_factor: float = 1.0,
        gamma0: Optional[float] = None,
        data_scale_value: Optional[float] = None,
        max_data_value: Optional[float] = None,
        int_bits: Optional[int] = None,
        frac_bits: Optional[int] = None,
    ) -> None:
        self.int_bits, self.frac_bits = int_bits, frac_bits
        _engine(
            self, check_matrix, error_priors, self._kind, frac_bits, False,
            pre_iter=max_iter, legs=0, leg_iter=0, solutions=None, stop_after_first=False,
            gamma0=gamma0, gamma_range=(-0.24, 0.66), gammas=None, alpha=alpha, alpha_scaling=float(alpha_iteration_scaling_factor),
            data_scale=data_scale_value, max_data=max_data_value, logging=False, seed=0,
        )


def _subclass(base: type, name: str, kind: str, doc: str) -> type:
    return type(name, (base,), {"_kind": kind, "__doc__": doc, "__module__": "stabilizer_qec.relay_bp.bp"})


_RELAY_DOC = """Relay-BP (Müller et al., IBM, 2025) in {t}, as relay_bp's ``{name}``: a first Mem-BP leg of
``pre_iter`` iterations with memory strength ``gamma0``, then up to ``num_sets`` legs of
``set_max_iter`` iterations, each with strengths drawn from ``gamma_dist_interval`` (or rows of
``explicit_gammas``) and starting from the posteriors the last ended with; the least-weight
solution among those found, stopping per ``stopping_criterion`` ("nconv": after ``stop_nconv``
solutions; "pre_iter": if the first leg converges; "all"). ``data_scale_value`` and
``max_data_value`` scale and bound values for integer types; ``logging`` writes relay_bp's
``relay_logging.out``. Its random strengths continue from decode to decode."""

_MINSUM_DOC = """Min-sum BP in {t}, as relay_bp's ``{name}``: ``max_iter`` iterations, scaled by ``alpha``
(0: ``1 − 2^−(t/alpha_iteration_scaling_factor)``), with memory (Mem-BP) when ``gamma0`` is set;
``data_scale_value`` and ``max_data_value`` scale and bound values for integer types."""

RelayDecoderF32 = _subclass(_Relay, "RelayDecoderF32", "f32", _RELAY_DOC.format(t="f32", name="RelayDecoderF32"))
RelayDecoderF64 = _subclass(_Relay, "RelayDecoderF64", "f64", _RELAY_DOC.format(t="f64", name="RelayDecoderF64"))
RelayDecoderI32 = _subclass(_Relay, "RelayDecoderI32", "i32", _RELAY_DOC.format(t="i32", name="RelayDecoderI32"))
RelayDecoderI64 = _subclass(_Relay, "RelayDecoderI64", "i64", _RELAY_DOC.format(t="i64", name="RelayDecoderI64"))
MinSumBPDecoderF32 = _subclass(_MinSum, "MinSumBPDecoderF32", "f32", _MINSUM_DOC.format(t="f32", name="MinSumBPDecoderF32"))
MinSumBPDecoderF64 = _subclass(_MinSum, "MinSumBPDecoderF64", "f64", _MINSUM_DOC.format(t="f64", name="MinSumBPDecoderF64"))
MinSumBPDecoderI8 = _subclass(_MinSum, "MinSumBPDecoderI8", "i8", _MINSUM_DOC.format(t="i8", name="MinSumBPDecoderI8"))
MinSumBPDecoderI16 = _subclass(_MinSum, "MinSumBPDecoderI16", "i16", _MINSUM_DOC.format(t="i16", name="MinSumBPDecoderI16"))
MinSumBPDecoderI32 = _subclass(_MinSum, "MinSumBPDecoderI32", "i32", _MINSUM_DOC.format(t="i32", name="MinSumBPDecoderI32"))
MinSumBPDecoderI64 = _subclass(_MinSum, "MinSumBPDecoderI64", "i64", _MINSUM_DOC.format(t="i64", name="MinSumBPDecoderI64"))
MinSumBPDecoderFixed = _subclass(
    _MinSum, "MinSumBPDecoderFixed", "fixed",
    _MINSUM_DOC.format(t="64-bit fixed point with ``frac_bits`` (0 to 31) fractional bits", name="MinSumBPDecoderFixed"),
)


class ObservableDecodeResult:
    """An observable decode's result, as relay_bp's: ``observables``, ``converged``,
    ``iterations``, the physical result when asked for, and when decoded from a known error,
    whether the logical outcome was wrong and how (otherwise ``None``)."""

    __slots__ = ("observables_", "converged", "iterations", "_truth", "_physical")

    def __init__(self, observables: np.ndarray, result: DecodeResult, truth: Optional[dict], include: bool) -> None:
        self.observables_ = observables
        self.converged = result.success
        self.iterations = result.iterations
        self._truth = truth
        self._physical = result if include else None

    @property
    def observables(self) -> np.ndarray:
        return self.observables_.copy()

    def _t(self, key: str) -> Optional[bool]:
        return None if self._truth is None else self._truth[key]

    @property
    def error_detected(self) -> Optional[bool]:
        return self._t("error_detected")

    @property
    def error_mismatch_detected(self) -> Optional[bool]:
        return self._t("error_mismatch_detected")

    @property
    def unconverged_no_error(self) -> Optional[bool]:
        return self._t("unconverged_no_error")

    @property
    def better_decoding_quality_error(self) -> Optional[bool]:
        return self._t("better_decoding_quality_error")

    @property
    def worse_decoding_quality_error(self) -> Optional[bool]:
        return self._t("worse_decoding_quality_error")

    @property
    def physical_decode_result(self) -> Optional[DecodeResult]:
        return self._physical


class ObservableDecoderRunner:
    """A decoder and the matrix of which faults flip which observables, as relay_bp's: decodes
    detectors to observables, or samples' known errors to observables with how the decoding
    went. It decodes on a copy of ``decoder`` as it is when given. In a batch, ``parallel``
    decodes every row from the decoder's present state (as relay_bp's parallel runs do, each
    row on a copy), and ``progress_bar`` shows a bar while it runs."""

    def __init__(self, decoder: _Decoder, observable_error_matrix: Any, include_decode_result: bool = False) -> None:
        if not isinstance(decoder, _Decoder):
            raise TypeError(f"decoder must be one of relay_bp's decoders, not {type(decoder).__name__}")
        self._x = decoder._x.copy()
        self._m, self._n, self._matrix = decoder._m, decoder._n, decoder._matrix
        _, n, _, self._obs = _sparse_bits(observable_error_matrix)
        if n != self._n:
            raise ValueError(f"the observable error matrix has {n} columns; the decoder has {self._n} faults")
        self.include_decode_result = bool(include_decode_result)

    def _decode(self, detectors: Any, engine: Any = None) -> DecodeResult:
        return DecodeResult(call((engine or self._x).decode, _bits_vector(detectors, self._m, "detectors")))

    def compute_observables(self, errors: Any) -> np.ndarray:
        return _parity(self._obs, np.asarray(errors))

    def decode(self, detectors: Any) -> np.ndarray:
        return self._decode(detectors).decoding

    def decode_detailed(self, detectors: Any) -> DecodeResult:
        return self._decode(detectors)

    def decode_observables(self, detectors: Any) -> np.ndarray:
        return self.compute_observables(self._decode(detectors).decoding)

    def _observables_detailed(self, detectors: Any, engine: Any = None) -> ObservableDecodeResult:
        result = self._decode(detectors, engine)
        return ObservableDecodeResult(self.compute_observables(result.decoding), result, None, self.include_decode_result)

    def _from_errors_detailed(self, errors: Any, engine: Any = None) -> ObservableDecodeResult:
        errors = np.asarray(errors).astype(np.uint8)
        detectors = _parity(self._matrix, errors)
        result = self._decode(detectors, engine)
        observables = self.compute_observables(errors)
        decoded = self.compute_observables(result.decoding)
        detected = not np.array_equal(observables, decoded)
        better = result._quality < call(self._x.quality, [int(e) for e in errors])
        truth = dict(
            error_detected=detected,
            error_mismatch_detected=not np.array_equal(errors, result.decoding),
            unconverged_no_error=not detected and not result.success,
            better_decoding_quality_error=detected and better,
            worse_decoding_quality_error=detected and not better,
        )
        return ObservableDecodeResult(observables, result, truth, self.include_decode_result)

    def from_errors_decode_observables_detailed(self, errors: Any) -> ObservableDecodeResult:
        return self._from_errors_detailed(errors)

    def _batch(self, rows: Any, one, parallel: bool, progress_bar: bool, leave: bool) -> list:
        rows = np.asarray(rows)
        bar = _Progress(len(rows), progress_bar, leave)
        out = []
        for row in rows:
            out.append(one(row, self._x.copy() if parallel else None))
            bar.step()
        bar.close()
        return out

    def decode_batch(self, detectors: Any, parallel: bool = False, progress_bar: bool = True, leave_progress_bar_on_finish: bool = False) -> np.ndarray:
        results = self._batch(detectors, lambda r, e: self._decode(r, e).decoding, parallel, progress_bar, leave_progress_bar_on_finish)
        return np.array(results, dtype=np.uint8).reshape(len(results), self._n)

    def decode_detailed_batch(self, detectors: Any, parallel: bool = False, progress_bar: bool = True, leave_progress_bar_on_finish: bool = False) -> List[DecodeResult]:
        return self._batch(detectors, self._decode, parallel, progress_bar, leave_progress_bar_on_finish)

    def decode_observables_batch(self, detectors: Any, parallel: bool = False, progress_bar: bool = True, leave_progress_bar_on_finish: bool = False) -> np.ndarray:
        results = self._batch(detectors, lambda r, e: self.compute_observables(self._decode(r, e).decoding), parallel, progress_bar, leave_progress_bar_on_finish)
        return np.array(results, dtype=np.uint8).reshape(len(results), self._obs.shape[0])

    def decode_observables_detailed_batch(self, detectors: Any, parallel: bool = False, progress_bar: bool = True, leave_progress_bar_on_finish: bool = False) -> List[ObservableDecodeResult]:
        return self._batch(detectors, self._observables_detailed, parallel, progress_bar, leave_progress_bar_on_finish)

    def from_errors_decode_observables_batch(self, errors: Any, parallel: bool = False, progress_bar: bool = True, leave_progress_bar_on_finish: bool = False) -> np.ndarray:
        def one(row: Any, engine: Any) -> np.ndarray:
            detectors = _parity(self._matrix, np.asarray(row))
            return self.compute_observables(self._decode(detectors, engine).decoding)

        results = self._batch(errors, one, parallel, progress_bar, leave_progress_bar_on_finish)
        return np.array(results, dtype=np.uint8).reshape(len(results), self._obs.shape[0])

    def from_errors_decode_observables_detailed_batch(self, errors: Any, parallel: bool = False, progress_bar: bool = True, leave_progress_bar_on_finish: bool = False) -> List[ObservableDecodeResult]:
        return self._batch(errors, self._from_errors_detailed, parallel, progress_bar, leave_progress_bar_on_finish)
