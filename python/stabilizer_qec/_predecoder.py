"""The AI pre-decoder: a 3D convolutional network over the detectors' space-time grid that
removes the faults it is confident of before a global decoder decodes the rest (NVIDIA's shape,
arXiv 2604.12841). Inference and training in Rust; training in numpy too, step for step the
same."""

from __future__ import annotations

import functools
import inspect
import os
import threading
from typing import Any, Union

import numpy as np

from . import _core
from ._circuit import DetectorErrorModel
from ._decoders import ModelLike, _DemDecoder, _failed, _inner, _model, _rebuild
from ._util import call, count, real, rows_to_b8, u64_to_rows

_PRETRAINED = os.path.join(os.path.dirname(__file__), "data", "predecoder_memory.npz")


def _serial(method: Any) -> Any:
    """A method run with the pre-decoder's other such calls kept out: training borrows the
    network with the GIL released, where a call from another thread would find it borrowed."""

    @functools.wraps(method)
    def run(self: "PreDecoder", *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return run


class PreDecoder(_DemDecoder):
    """A convolutional pre-decoder in front of a global decoder, in the shape of NVIDIA's
    (Chamberland, Olle, Li, Thornton and Baratta, arXiv 2604.12841): a fully convolutional 3D
    network reads the detection events on the grid of the detectors' coordinates (x, y, t) and
    gives each of the model's graph-like faults a probability; those above ``threshold`` are
    applied (their detection events removed, their observables flipped), and ``decoder``
    decodes what is left. The prediction is both together. Most faults are isolated, so the
    global decoder sees a far sparser syndrome.

    With ``reweight`` (the default for matching and union-find), the global decoder also weighs
    each fault by the network's probability q: ln((1 − q)/q), and for a fault the network applied
    ln(q/(1 − q)), the chance that applying it was wrong. At threshold 1/2 that is matching on the
    whole syndrome with the network's weights, negative ones included, on a sparser syndrome: what
    the network learnt that matching's own weights do not say (the X and Z halves of a Y fault
    together, a hook's two edges) reaches the global decoder even where the network is unsure.
    Without it (the paper's pipeline, and any decoder) the global decoder keeps its own weights.

    The model must be decomposed (``decompose_errors=True``), and every detector needs
    coordinates (x, y, t) on a regular grid. The network's inputs are, per sector (the
    components the model's graph-like faults connect: a CSS memory's X and Z detectors, or
    ``sectors``), its events and where its detectors are; its outputs are the graph-like faults,
    one channel per kind (``kinds``: sector, boundary, and the offset from the fault's first
    detector to its second), at the fault's first detector. Convolutions are local, so weights
    trained on one model run on another with the same kinds, any distance or number of rounds.

    ``weights`` are a ``weights`` dict, or a file ``save`` wrote; without them the network is
    fresh (``hidden`` filters per hidden layer, cubes of side ``kernel``) and ``train`` trains it
    on the model's own shots. ``PreDecoder.pretrained`` loads this package's weights for
    ``memory_circuit``'s rotated memories.

    >>> import stabilizer_qec as sq
    >>> dem = sq.memory_circuit(distance=5, rounds=5, p=0.006).detector_error_model(decompose_errors=True)
    >>> dets, obs, _ = dem.compile_sampler(seed=1).sample(20000)
    >>> pre = sq.PreDecoder.pretrained(dem)
    >>> left, _, applied = pre.predecode(dets, threads=4)
    >>> bool(dets.sum() > 3 * left.sum())
    True
    >>> plain = int((sq.Matching(dem).decode_batch(dets) != obs).any(axis=1).sum())
    >>> int((pre.decode_batch(dets, threads=4) != obs).any(axis=1).sum()) < 0.9 * plain
    True
    """

    __slots__ = ("_dec", "_threshold", "_np", "_reweight", "_lock")

    def __init__(
        self,
        model: ModelLike,
        weights: Any = None,
        *,
        decoder: Any = "matching",
        reweight: Union[bool, None] = None,
        threshold: float = 0.5,
        hidden: Any = (32, 32, 32),
        kernel: int = 3,
        sectors: Any = None,
        seed: int = 0,
    ) -> None:
        model = self._made(_model(model), decoder=decoder, reweight=reweight, threshold=threshold, sectors=sectors)
        graph = decoder if isinstance(decoder, str) and decoder in ("matching", "union_find") else None
        if reweight is None:
            reweight = graph is not None
        if reweight and graph is None:
            raise ValueError(f"reweight needs decoder 'matching' or 'union_find', not {decoder!r}")
        if isinstance(weights, (str, os.PathLike)):
            weights = _load(weights)
        if weights is not None:
            hidden = [int(c) for c in weights["channels"][1:-1]]
            kernel = int(weights["kernel"])
        hidden = [count(h, "hidden", 1) for h in hidden]
        secs = None if sectors is None else [count(s, "sectors") for s in sectors]
        self._lock = threading.RLock()
        self._x = call(_core.PreDecoder, model._d, hidden, count(kernel, "kernel", 1), count(seed, "seed"), secs, graph if reweight else None)
        self.threshold = threshold
        self._reweight = bool(reweight)
        self._dec = None if reweight else _inner(decoder)(model)
        self._np = None
        if weights is not None:
            self.weights = weights

    def __reduce__(self) -> tuple:
        (model,), kw = self._made_from
        return (_rebuild, (type(self), (model,), {**kw, "weights": self.weights}))

    @classmethod
    def pretrained(cls, model: ModelLike, **kwargs: Any) -> "PreDecoder":
        """This package's weights, for ``memory_circuit``'s rotated memories (Z basis, any
        distance and rounds): three hidden layers of 32 filters, 3×3×3 (receptive field 9),
        trained on its distance-9, 9-round memory under SD6 noise at p = 0.006 (near threshold,
        as the paper trains), 12,000 steps of 256 fresh shots by Lion; ``tools/predecoder.py``
        trains them again (the same on any number of threads)."""
        return cls(model, _PRETRAINED, **kwargs)

    @property
    def threshold(self) -> float:
        """The probability above which a fault is applied."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        t = real(value, "threshold")
        if not 0.0 < t < 1.0:
            raise ValueError(f"threshold must be in (0, 1), not {t}")
        self._threshold = t

    @property
    def reweight(self) -> bool:
        """Whether the global decoder weighs faults by the network's probabilities."""
        return self._reweight

    @property
    def dims(self) -> tuple:
        """The grid: (t, y, x)."""
        return self._x.dims

    @property
    def num_sectors(self) -> int:
        return self._x.num_sectors

    @property
    def sectors(self) -> list:
        """Per detector, its sector."""
        return self._x.sectors

    @property
    def kinds(self) -> list:
        """Per output channel, its kind of fault: (sector, boundary, dt, dy, dx)."""
        return self._x.kinds

    @property
    def faults(self) -> list:
        """The faults the network can apply: (detectors, observables mask, probability)."""
        return self._x.faults

    @property
    def num_params(self) -> int:
        return len(self._x.params) // 4

    @property
    def steps(self) -> int:
        """The training steps taken (by this engine's optimizer)."""
        return self._x.steps if self._np is None else self._np["step"]

    @property
    @_serial
    def weights(self) -> dict:
        """The network: its channels (inputs, hidden layers, outputs), its kernel, the kinds of
        its outputs, and per layer its weights (kernel × kernel × kernel × in × out) and biases,
        float32."""
        channels, k = self._x.channels, self._x.kernel
        params = np.frombuffer(self._x.params, dtype="<f4")
        return {"channels": list(channels), "kernel": k, "kinds": [tuple(x) for x in self._x.kinds], "layers": [(w.copy(), b.copy()) for w, b in _split(params, channels, k)]}

    @weights.setter
    @_serial
    def weights(self, weights: dict) -> None:
        channels, k = self._x.channels, self._x.kernel
        theirs = list(weights["channels"])
        if theirs[:-1] != channels[:-1] or int(weights["kernel"]) != k:
            raise ValueError(f"weights for channels {theirs[:-1]} and kernel {weights['kernel']} do not fit this network's {channels[:-1]} and {k}")
        layers = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in weights["layers"]]
        # The last layer's outputs are kinds: this layout's, each from the weights' of that kind;
        # a kind the weights do not know is never applied.
        of = {tuple(int(v) for v in kind): i for i, kind in enumerate(weights["kinds"])}
        w, b = layers[-1]
        nw, nb = np.zeros(w.shape[:-1] + (channels[-1],), dtype=np.float32), np.full(channels[-1], -30.0, dtype=np.float32)
        for j, kind in enumerate(self._x.kinds):
            i = of.get(tuple(int(v) for v in kind))
            if i is not None:
                nw[..., j], nb[j] = w[..., i], b[i]
        layers[-1] = (nw, nb)
        flat = np.concatenate([np.concatenate([w.ravel(), b.ravel()]) for w, b in layers]).astype("<f4")
        call(setattr, self._x, "params", flat.tobytes())
        if self._np is not None:
            self._np["params"] = flat.astype(np.float32)

    @_serial
    def save(self, path: Union[str, os.PathLike]) -> None:
        """Writes the weights to ``path`` (numpy's ``.npz``), for ``PreDecoder(model, path)``."""
        w = self.weights
        arrays = {f"w{i}": a for i, (a, _) in enumerate(w["layers"])}
        arrays.update({f"b{i}": b for i, (_, b) in enumerate(w["layers"])})
        np.savez(path, channels=np.array(w["channels"]), kernel=np.array(w["kernel"]), kinds=np.array(w["kinds"], dtype=np.int64).reshape(-1, 5), **arrays)

    @_serial
    def train(
        self,
        steps: int,
        *,
        batch: int = 256,
        optimizer: str = "lion",
        lr: Union[float, None] = None,
        warmup: int = 100,
        weight_decay: float = 1e-7,
        seed: int = 0,
        threads: int = 0,
        engine: str = "rust",
    ) -> np.ndarray:
        """Trains the network on fresh shots of the model: ``steps`` steps of ``batch`` shots,
        binary cross-entropy against the graph-like faults that fired, by Lion (the paper's,
        β 0.9 and 0.95, rate 3e-4) or Adam (rate 1e-3), the rate warmed up over ``warmup`` steps
        and ×0.7 at a quarter and at half of the steps. Returns each step's mean loss. Calls with
        the same optimizer continue its state.

        ``engine="numpy"`` trains in numpy instead, on the same shots: the two agree step for
        step but for rounding."""
        steps, batch, threads = count(steps, "steps"), count(batch, "batch", 1), count(threads, "threads")
        if optimizer not in ("lion", "adam"):
            raise ValueError(f"optimizer must be 'lion' or 'adam', not {optimizer!r}")
        lr = (3e-4 if optimizer == "lion" else 1e-3) if lr is None else real(lr, "lr")
        if engine == "rust":
            if self._np is not None:
                raise ValueError("this network was trained by the numpy engine: continue with engine='numpy'")
            return np.array(call(self._x.train, steps, batch, optimizer, lr, count(warmup, "warmup"), real(weight_decay, "weight_decay"), count(seed, "seed"), threads))
        if engine != "numpy":
            raise ValueError(f"engine must be 'rust' or 'numpy', not {engine!r}")
        if self._np is None or self._np["optimizer"] != optimizer:
            if self._x.steps:
                raise ValueError("this network was trained by the rust engine: continue with engine='rust'")
            n = self.num_params
            self._np = {"optimizer": optimizer, "params": np.frombuffer(self._x.params, dtype="<f4").astype(np.float32), "m": np.zeros(n, np.float32), "v": np.zeros(n, np.float32), "step": 0}
        return _train_numpy(self, steps, batch, optimizer, np.float32(lr), count(warmup, "warmup"), np.float32(real(weight_decay, "weight_decay")), count(seed, "seed"), threads)

    @_serial
    def loss_and_grad(self, dets: Any, errors: Any, *, engine: str = "rust", threads: int = 1) -> tuple:
        """The mean loss of shots (their detection events, and the model's faults that fired, as
        its sampler's ``return_errors`` gives them) and its gradient by the parameters, flat."""
        packed, n = rows_to_b8(dets, self.num_detectors, False, "dets")
        errs, m = rows_to_b8(errors, self._x.num_errors, False, "errors")
        if m != n:
            raise ValueError(f"{n} shots of detection events but {m} of errors")
        if engine == "rust":
            loss, grad = call(self._x.loss_and_grad, packed, errs, n, count(threads, "threads"))
            return loss, np.frombuffer(grad, dtype="<f4").copy()
        if engine != "numpy":
            raise ValueError(f"engine must be 'rust' or 'numpy', not {engine!r}")
        params = np.frombuffer(self._x.params, dtype="<f4").astype(np.float32)
        return _loss_and_grad_numpy(self, params, packed, errs, n)

    @_serial
    def predecode(self, shots: Any, *, bit_packed_shots: bool = False, threads: int = 1) -> tuple:
        """Per shot: the detection events the pre-decoder leaves (bool, shots × detectors), the
        observables of the faults it applied (uint8, shots × observables), and how many it
        applied."""
        packed, n = rows_to_b8(shots, self.num_detectors, bit_packed_shots, "shots")
        left, obs, applied = call(self._x.predecode, packed, n, np.float32(self._threshold), count(threads, "threads"))
        nd = self.num_detectors
        left = np.unpackbits(np.frombuffer(left, dtype=np.uint8).reshape(n, -1), axis=1, count=nd, bitorder="little").astype(bool)
        return left, u64_to_rows(obs, n, self.num_observables, False), np.frombuffer(applied, dtype="<u4").astype(np.int64)

    @_serial
    def decode_batch(
        self,
        shots: Any,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> np.ndarray:
        """As ``Matching.decode_batch``: the pre-decoder's faults, then the global decoder on the
        detection events left (on the network's weights, with ``reweight``), their observables
        together."""
        if self._reweight:
            packed, n, threads = self._shots(shots, bit_packed_shots, threads)
            raw, failed = call(self._x.decode_soft, packed, n, np.float32(self._threshold), threads)
            _failed([i for i, f in enumerate(failed) if f], "no correction explains its detection events")
            return self._predictions(raw, n, bit_packed_predictions)
        left, flips, _ = self.predecode(shots, bit_packed_shots=bit_packed_shots, threads=threads)
        dec = self._dec
        kw = {"threads": threads} if "threads" in inspect.signature(dec.decode_batch).parameters else {}
        pred = np.asarray(dec.decode_batch(left, **kw), dtype=np.uint8).reshape(len(left), self.num_observables) ^ flips
        return np.packbits(pred, axis=1, bitorder="little") if bit_packed_predictions else pred


def _load(path: Union[str, os.PathLike]) -> dict:
    with np.load(path) as f:
        n = len(f["channels"]) - 1
        return {
            "channels": [int(c) for c in f["channels"]],
            "kernel": int(f["kernel"]),
            "kinds": [tuple(int(v) for v in k) for k in f["kinds"]],
            "layers": [(f[f"w{i}"], f[f"b{i}"]) for i in range(n)],
        }


def _split(params: np.ndarray, channels: list, k: int) -> list:
    out, o = [], 0
    for cin, cout in zip(channels, channels[1:]):
        w = params[o : o + k**3 * cin * cout].reshape(k, k, k, cin, cout)
        o += k**3 * cin * cout
        out.append((w, params[o : o + cout]))
        o += cout
    return out


# -- The network in numpy -----------------------------------------------------

_S = np.float32(0.7978846)
_C = np.float32(0.044715)


def _gelu(x: np.ndarray) -> np.ndarray:
    return np.float32(0.5) * x * (np.float32(1) + np.tanh(_S * (x + _C * x * x * x)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    th = np.tanh(_S * (x + _C * x * x * x))
    return np.float32(0.5) * (np.float32(1) + th) + np.float32(0.5) * x * (np.float32(1) - th * th) * _S * (np.float32(1) + np.float32(3) * _C * x * x)


def _windows(a: np.ndarray, k: int) -> np.ndarray:
    """(n, t, y, x, c) zero-padded to (n, t, y, x, c, k, k, k): each voxel's neighbourhood."""
    h = k // 2
    p = np.pad(a, ((0, 0), (h, h), (h, h), (h, h), (0, 0)))
    return np.lib.stride_tricks.sliding_window_view(p, (k, k, k), axis=(1, 2, 3))


def _loss_and_grad_numpy(pre: PreDecoder, params: np.ndarray, packed: bytes, errs: bytes, n: int) -> tuple:
    x = pre._x
    channels, k = x.channels, x.kernel
    t, yy, xx = x.dims
    raw_x, raw_y = x.encode(packed, n, errs)
    inp = np.frombuffer(raw_x, dtype="<f4").reshape(n, t, yy, xx, channels[0])
    labels = np.frombuffer(raw_y, dtype="<f4").reshape(n, t, yy, xx, channels[-1])
    mask = np.frombuffer(x.mask, dtype=np.uint8).reshape(t, yy, xx, channels[-1]).astype(bool)
    layers = _split(params, channels, k)
    acts, pres = [inp], []
    for i, (w, b) in enumerate(layers):
        z = np.einsum("ntyxcabd,abdco->ntyxo", _windows(acts[-1], k), w, optimize=True) + b
        pres.append(z)
        if i + 1 < len(layers):
            acts.append(_gelu(z))
    z = pres[-1]
    loss = np.maximum(z, 0) - z * labels + np.log1p(np.exp(-np.abs(z)))
    loss = float(loss[:, mask].astype(np.float64).sum()) / n
    g = (np.float32(1) / (np.float32(1) + np.exp(-z)) - labels) * mask
    grads = []
    for i in range(len(layers) - 1, -1, -1):
        w, _ = layers[i]
        gw = np.einsum("ntyxcabd,ntyxo->abdco", _windows(acts[i], k), g, optimize=True)
        gb = g.sum(axis=(0, 1, 2, 3))
        grads.append(np.concatenate([gw.ravel(), gb]))
        if i:
            g = np.einsum("ntyxoabd,abdco->ntyxc", _windows(g, k), w[::-1, ::-1, ::-1], optimize=True) * _gelu_grad(pres[i - 1])
    return loss, (np.concatenate(grads[::-1]) / np.float32(n)).astype(np.float32)


def _train_numpy(pre: PreDecoder, steps: int, batch: int, optimizer: str, lr: np.float32, warmup: int, wd: np.float32, seed: int, threads: int) -> np.ndarray:
    st = pre._np
    p, m, v = st["params"], st["m"], st["v"]
    per = -(-batch // 64)
    losses = []
    for step in range(steps):
        dets, errs = pre._x.sample(batch, seed, (st["step"] + 1) * per, threads)
        loss, g = _loss_and_grad_numpy(pre, p, dets, errs, batch)
        rate = lr
        if step < warmup:
            rate = rate * (np.float32(step + 1) / np.float32(warmup))
        if 4 * step >= steps:
            rate = rate * np.float32(0.7)
        if 2 * step >= steps:
            rate = rate * np.float32(0.7)
        st["step"] += 1
        if optimizer == "lion":
            c = np.float32(0.9) * m + np.float32(0.1) * g
            p -= rate * (np.sign(c) + wd * p)
            m *= np.float32(0.95)
            m += np.float32(0.05) * g
        else:
            m *= np.float32(0.9)
            m += np.float32(0.1) * g
            v *= np.float32(0.999)
            v += np.float32(0.001) * g * g
            c1 = np.float32(1) - np.float32(0.9) ** st["step"]
            c2 = np.float32(1) - np.float32(0.999) ** st["step"]
            p -= rate * (m / c1) / (np.sqrt(v / c2) + np.float32(1e-8))
        losses.append(loss)
    call(setattr, pre._x, "params", p.astype("<f4").tobytes())
    return np.array(losses)
