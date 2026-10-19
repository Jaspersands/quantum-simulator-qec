# Follows CUDA-Q QEC's nv-qldpc-decoder as documented (NVIDIA/cudaqx 0.8.1,
# docs/sphinx/api/qec/nv_qldpc_decoder_api.rst and python/cudaq_qec/_compat.py; Copyright (c) 2024 -
# 2026 NVIDIA Corporation & Affiliates, Apache License 2.0; see THIRD_PARTY_NOTICES.md): its
# parameters, mapped onto this package's ldpc (BP, BP+OSD) and relay_bp (Mem-BP, Relay-BP) ports.
"""CUDA-Q QEC's ``nv-qldpc-decoder`` on this package's decoders.

CUDA-Q's decoder is closed-source GPU code; this one takes its documented parameters and runs
the algorithm they name on this package's ports of the reference implementations it cites:

- ``bp_method`` 0 (sum-product) and 1 (min-sum, ``scale_factor``; 0 the dynamic factor), with
  ``use_osd`` (``osd_method`` 1 OSD-0, 2 exhaustive, 3 combination sweep, ``osd_order``): ldpc's
  ``BpDecoder`` / ``BpOsdDecoder`` (``tests/test_ldpc_package.py`` checks those against ldpc);
- ``bp_method`` 2 (min-sum with memory ``gamma0``): relay_bp's ``MinSumBPDecoderF64`` (``F32``
  with ``proc_float="fp32"``);
- ``composition`` 1 with ``bp_method`` 3 (sequential relay: ``srelay_config``'s ``pre_iter``
  warm-up iterations at ``gamma0``, then up to ``num_sets`` legs of ``max_iterations``, each
  with memory strengths from ``gamma_dist`` or a row of ``explicit_gammas``, stopping per
  ``stopping_criterion``): relay_bp's ``RelayDecoderF64`` (``F32`` with fp32), seeded with
  ``bp_seed``. Random strengths are relay_bp's draws, not CUDA-Q's GPU ones.

``converged`` is BP's (or a relay leg's) convergence, as CUDA-Q reports it, also when OSD
supplied the correction. With ``O`` the result is the observable flips. ``opt_results`` may ask
for ``num_iter``. What the ports cannot run is refused, never ignored: ``bp_method`` 4 and 5,
``bp_method`` 3 outside sequential relay, ``use_osd`` with memory BP, ``clip_value`` (and so
``repeatable``), ``iter_per_check`` other than 1, ``gamma_ensemble_size`` above 1, the
``bp_llr_history`` and ``relay_solutions`` results, and fp32 sum-product or BP+OSD.
``use_sparsity``, ``n_threads``, ``bp_batch_size`` and ``osd_batch_size`` choose GPU kernels and
batching in CUDA-Q and change nothing here (beyond the ``use_sparsity`` requirements CUDA-Q
checks).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ._decoders import BatchDecoderResult, DecoderResult, _batch, _hard, _Inputs, _make_result, _NativeDecoder

NAME = "nv-qldpc-decoder"


def _err(msg: str) -> RuntimeError:
    return RuntimeError(f"{NAME}: {msg}")


def _get(params: dict, key: str, default: Any, kind: str) -> Any:
    if key not in params:
        return default
    v = params[key]
    ok = {
        "bool": isinstance(v, (bool, np.bool_)),
        "int": isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)),
        "float": isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)),
        "str": isinstance(v, str),
        "dict": isinstance(v, dict),
    }[kind]
    if not ok:
        raise _err(f"parameter '{key}' must be a {kind}, not {type(v).__name__}")
    return {"bool": bool, "int": int, "float": float, "str": str, "dict": dict}[kind](v)


def _unsupported(what: str) -> RuntimeError:
    return _err(f"{what} is not supported by this package's decoders (see stabilizer_qec.cudaq_qec._qldpc)")


_OSD = {1: "osd_0", 2: "osd_e", 3: "osd_cs"}
_STOPPING = {"All": "all", "FirstConv": "nconv", "NConv": "nconv"}
_OPT_RESULTS = ("bp_llr_history", "num_iter", "relay_solutions")


class _NvQldpc(_NativeDecoder):
    """CUDA-Q's ``nv-qldpc-decoder`` (see the module notes)."""

    _raw_output = True

    def __init__(self, inputs: _Inputs, output: Optional[str], params: dict) -> None:
        to_obs = inputs.O is not None and inputs.num_observables > 0
        if output == "observables" and not to_obs:
            raise _err("constructed for observable output but its model supplies no observable mapping")
        if output == "errors":
            to_obs = False
        super().__init__(inputs, "observables" if to_obs else "errors", params)
        n = inputs.num_cols
        p = params
        use_sparsity = _get(p, "use_sparsity", False, "bool")
        error_rate = _get(p, "error_rate", 0.001, "float")
        rates = np.asarray(inputs.rates, dtype=np.float64) if inputs.rates else np.full(n, error_rate)
        if rates.size != n:
            raise _err(f"error_rate_vec must have block_size ({n}) entries, not {rates.size}")
        if not np.all(np.isfinite(rates)) or np.any((rates < 0.0) | (rates > 1.0)):
            raise _err("error rates must be in the 0-1 range")
        max_iter = _get(p, "max_iterations", 30, "int")
        if max_iter < 1:
            raise _err("max_iterations must be positive")
        for key in ("n_threads", "bp_batch_size", "osd_batch_size"):
            if _get(p, key, 1, "int") < 0:
                raise _err(f"{key} must be non-negative")
        use_osd = _get(p, "use_osd", False, "bool")
        osd_method = _get(p, "osd_method", 1, "int")
        osd_order = _get(p, "osd_order", 0, "int")
        if use_osd and osd_method not in (0, 1, 2, 3):
            raise _err(f"osd_method must be 1 (OSD-0), 2 (exhaustive) or 3 (combination sweep), not {osd_method}")
        if osd_order < 0:
            raise _err("osd_order must be non-negative")
        iter_per_check = _get(p, "iter_per_check", 1, "int")
        if not 1 <= iter_per_check <= max_iter:
            raise _err(f"iter_per_check must be between 1 and max_iterations ({max_iter})")
        clip = _get(p, "clip_value", 0.0, "float")
        if clip < 0.0:
            raise _err("clip_value must be non-negative")
        if _get(p, "repeatable", False, "bool") and clip == 0.0:
            raise _err("repeatable=True requires a non-zero clip_value")
        bp_method = _get(p, "bp_method", 0, "int")
        if bp_method not in range(6):
            raise _err(f"bp_method must be 0-5, not {bp_method}")
        composition = _get(p, "composition", 0, "int")
        if composition not in (0, 1):
            raise _err(f"composition must be 0 (standard) or 1 (sequential relay), not {composition}")
        scale = _get(p, "scale_factor", 1.0, "float")
        proc = _get(p, "proc_float", "fp64", "str")
        if proc not in ("fp32", "fp64"):
            raise _err(f"proc_float must be 'fp32' or 'fp64', not '{proc}'")
        gamma0 = _get(p, "gamma0", None, "float")
        gamma_dist = p.get("gamma_dist")
        explicit = p.get("explicit_gammas")
        srelay = _get(p, "srelay_config", None, "dict")
        seed = _get(p, "bp_seed", 42, "int")
        ensemble = _get(p, "gamma_ensemble_size", 1, "int")
        if ensemble not in (1, 2, 4, 8):
            raise _err(f"gamma_ensemble_size must be 1, 2, 4 or 8, not {ensemble}")
        opt = _get(p, "opt_results", {}, "dict")
        for key in opt:
            if key not in _OPT_RESULTS:
                raise _err(f"unknown opt_results entry '{key}'; choose from {', '.join(_OPT_RESULTS)}")
        llr_history = opt.get("bp_llr_history", 0)
        if isinstance(llr_history, (bool, np.bool_)) or not isinstance(llr_history, (int, np.integer)) or not 0 <= int(llr_history) <= max_iter:
            raise _err(f"opt_results bp_llr_history must be an integer between 0 and max_iterations ({max_iter})")
        self._num_iter = bool(opt.get("num_iter", False))

        # CUDA-Q's own requirements.
        if bp_method >= 2 and not use_sparsity:
            raise _err(f"bp_method={bp_method} requires use_sparsity=True")
        if composition == 1:
            if bp_method not in (3, 5):
                raise _err("composition=1 (sequential relay) requires bp_method=3 or bp_method=5")
            if not use_sparsity:
                raise _err("composition=1 (sequential relay) requires use_sparsity=True")
            if srelay is None:
                raise _err("composition=1 (sequential relay) requires srelay_config")
            if gamma0 is None:
                raise _err("composition=1 (sequential relay) requires gamma0")
            if gamma_dist is None and explicit is None:
                raise _err("composition=1 (sequential relay) requires gamma_dist or explicit_gammas")
        if bp_method in (2, 4) and gamma0 is None:
            raise _err(f"bp_method={bp_method} requires gamma0")
        if bp_method in (3, 5) and gamma_dist is None and explicit is None:
            raise _err(f"bp_method={bp_method} requires gamma_dist or explicit_gammas")
        if ensemble > 1 and not (composition == 1 and bp_method in (3, 5)):
            raise _err("gamma_ensemble_size > 1 requires composition=1 with bp_method=3 or bp_method=5")

        # What this package's decoders cannot run.
        if bp_method in (4, 5):
            raise _unsupported(f"bp_method={bp_method} (sum-product with memory)")
        if bp_method == 3 and composition == 0:
            raise _unsupported("bp_method=3 (disordered memory) outside sequential relay (composition=1)")
        if use_osd and osd_method != 0 and bp_method >= 2:
            raise _unsupported(f"use_osd with bp_method={bp_method} (memory BP)")
        if clip != 0.0:
            raise _unsupported("clip_value (message clipping)")
        if iter_per_check != 1:
            raise _unsupported("iter_per_check other than 1")
        if ensemble > 1:
            raise _unsupported("gamma_ensemble_size > 1")
        if int(llr_history) > 0:
            raise _unsupported("opt_results bp_llr_history")
        if opt.get("relay_solutions", False) not in (False, 0):
            raise _unsupported("opt_results relay_solutions (recording every relay convergence)")
        if proc == "fp32" and bp_method == 0:
            raise _unsupported("proc_float='fp32' with sum-product BP")
        if proc == "fp32" and use_osd and osd_method != 0:
            raise _unsupported("proc_float='fp32' with OSD")

        H = np.zeros((inputs.num_rows, n), dtype=np.uint8)
        for j, col in enumerate(inputs.columns):
            H[col, j] = 1
        self._H = H
        self._relay = bp_method >= 2
        if bp_method <= 1:
            from .. import ldpc

            method = "product_sum" if bp_method == 0 else "minimum_sum"
            common = dict(error_channel=rates, max_iter=max_iter, bp_method=method, ms_scaling_factor=scale, schedule="parallel")
            if use_osd and osd_method != 0:
                self._bp = ldpc.BpOsdDecoder(H, osd_method=_OSD[osd_method], osd_order=osd_order if osd_method != 1 else 0, **common)
            else:
                self._bp = ldpc.BpDecoder(H, device="cpu-f32" if proc == "fp32" else "cpu", **common)
        elif bp_method == 2:
            from .. import relay_bp

            cls = relay_bp.MinSumBPDecoderF32 if proc == "fp32" else relay_bp.MinSumBPDecoderF64
            self._bp = cls(H, rates, max_iter=max_iter, alpha=scale, gamma0=gamma0)
        else:
            from .. import relay_bp

            for key in ("pre_iter", "num_sets"):
                if key not in srelay:
                    raise _err(f"srelay_config requires {key}")
            pre_iter = _get(srelay, "pre_iter", 0, "int")
            num_sets = _get(srelay, "num_sets", 0, "int")
            criterion = _get(srelay, "stopping_criterion", "All", "str")
            if criterion not in _STOPPING:
                raise _err(f"srelay_config stopping_criterion must be 'All', 'FirstConv' or 'NConv', not '{criterion}'")
            for key in srelay:
                if key not in ("pre_iter", "num_sets", "stopping_criterion", "stop_nconv"):
                    raise _err(f"unknown srelay_config entry '{key}'")
            if pre_iter < 0 or num_sets < 0:
                raise _err("srelay_config pre_iter and num_sets must be non-negative")
            stop_nconv = 1
            if criterion == "NConv":
                if "stop_nconv" not in srelay:
                    raise _err("srelay_config stopping_criterion 'NConv' requires stop_nconv")
                stop_nconv = _get(srelay, "stop_nconv", 1, "int")
                if stop_nconv < 1:
                    raise _err("srelay_config stop_nconv must be positive")
            kw: Dict[str, Any] = {}
            if explicit is not None:
                g = np.asarray(explicit, dtype=np.float64)
                if g.ndim != 2 or g.shape[1] != n:
                    raise _err(f"explicit_gammas must be rows of block_size ({n}) memory strengths")
                if g.shape[0] != num_sets:
                    raise _err(f"explicit_gammas must have num_sets ({num_sets}) rows, one per relay leg")
                # relay_bp gives leg k (from 1) row k mod rows: rotate so leg k gets row k - 1.
                kw["explicit_gammas"] = np.roll(g, 1, axis=0) if num_sets else g
            if gamma_dist is not None:
                d = np.asarray(gamma_dist, dtype=np.float64).reshape(-1)
                if d.size != 2 or not d[0] < d[1]:
                    raise _err("gamma_dist must be an interval [min, max] with min < max")
                kw["gamma_dist_interval"] = (float(d[0]), float(d[1]))
            cls = relay_bp.RelayDecoderF32 if proc == "fp32" else relay_bp.RelayDecoderF64
            if num_sets == 0 and "explicit_gammas" in kw:
                del kw["explicit_gammas"]
            self._bp = cls(H, rates, alpha=scale, gamma0=gamma0, pre_iter=pre_iter, num_sets=num_sets, set_max_iter=max_iter, stop_nconv=stop_nconv, stopping_criterion=_STOPPING[criterion], seed=seed, **kw)

    def get_version(self) -> str:
        return "CUDA-Q QEC nv-qldpc-decoder (stabilizer_qec: this package's ldpc and relay_bp ports)"

    def decode(self, syndrome: Any) -> DecoderResult:
        hard = _hard(syndrome)
        if hard.size != self._syndrome_size:
            raise _err(f"syndrome length ({hard.size}) must equal syndrome_size ({self._syndrome_size})")
        s = hard.astype(np.uint8)
        if self._relay:
            r = self._bp.decode_detailed(s)
            correction, converged, iters = r.decoding, r.success, r.iterations
        else:
            correction = np.asarray(self._bp.decode(s), dtype=np.uint8)
            converged, iters = bool(self._bp.converge), int(self._bp.iter)
            if not s.any():
                iters = 0
        errors = np.asarray(correction, dtype=np.float64)
        result = self._project(errors) if self._output == "observables" else errors
        return _make_result(converged, result, {"num_iter": iters} if self._num_iter else None)

    def decode_batch(self, syndromes: Any) -> BatchDecoderResult:
        return _batch([self.decode(s) for s in syndromes])
