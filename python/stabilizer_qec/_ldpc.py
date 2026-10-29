"""ldpc 2.4.1's BP decoders on a check matrix (``BpDecoder``, ``SoftInfoBpDecoder``,
``BpOsdDecoder``, ``BpLsdDecoder``): their arguments, defaults, properties and behaviour, with
every BP schedule (parallel, serial, serial-relative, random serial) and input type (a syndrome
or a received vector), bit for bit.

Errors are ldpc's, type and message (its messages' runs of spaces included). A few things are
looser than ldpc 2's classes: positional ``error_rate``/``error_channel``, integers where a float
is typed, NumPy scalars, and sequences where a list is typed are taken where ldpc's Cython
signatures refuse them. Its v1 classes (``bp_decoder``, ``bposd_decoder``) are held to ldpc's
exact types and one positional argument. And ``BpOsdDecoder.decoding`` returns the OSD decoding
(in ldpc it raises ``AttributeError``)."""

from __future__ import annotations

import functools
import sys
import threading
import warnings
from typing import Any, List, Optional, Union

import numpy as np

from . import _core
from ._decoders import _bp_method, _columns, _rebuild
from ._util import call, count

__all__ = ["BpDecoder", "BpDecoderBase", "BpFlipDecoder", "BpLsdDecoder", "BpOsdDecoder", "LsdDecoder", "SoftInfoBpDecoder"]

_SCHEDULES = {"parallel": "parallel", "p": "parallel", "0": "parallel", "serial": "serial", "s": "serial", "1": "serial", "serial_relative": "serial_relative", "sr": "serial_relative", "2": "serial_relative"}


def _is_matrix(pcm: Any) -> bool:
    return isinstance(pcm, np.ndarray) or hasattr(pcm, "tocsc")


# ldpc's base class reads BP's options from the arguments actually given, with these defaults,
# whatever defaults its subclasses' signatures show: so product-sum BP unless asked otherwise.
_BASE_DEFAULTS = dict(error_rate=None, error_channel=None, max_iter=0, bp_method=0, ms_scaling_factor=1.0, schedule=0, omp_thread_count=1, random_serial_schedule=False, random_schedule_seed=0, serial_schedule_order=None, channel_probs=[None])


class _Default:
    """A signature's default that was not given: it prints as ldpc's declared default, and
    the base class's default is used, as in ldpc."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)


# ldpc's messages keep the indentation of the source lines they continue onto: runs of spaces.
_S12, _S16, _S21, _S28 = " " * 12, " " * 16, " " * 21, " " * 28


def _tname(value: Any) -> str:
    """A value's type as Cython's messages name it (``int``, ``numpy.int64``)."""
    t = type(value)
    return t.__name__ if t.__module__ == "builtins" else f"{t.__module__}.{t.__name__}"


def _type_error(name: str, expected: str, value: Any) -> TypeError:
    """Cython's TypeError for an argument of the wrong type. ldpc's wheels for Python 3.10
    and 3.11 were built by a Cython that adds a note when the value's type subclasses the builtin
    expected."""
    message = f"Argument '{name}' has incorrect type (expected {expected}, got {_tname(value)})"
    builtin = {"int": int, "float": float, "str": str, "list": list}.get(expected)
    if sys.version_info < (3, 12) and builtin is not None and isinstance(value, builtin):
        message += (". Note that Cython is deliberately stricter than PEP-484 and rejects subclasses of builtin types."
                    " If you need to pass subclasses then set the 'annotation_typing' directive to False.")
    return TypeError(message)


def _typed(value: Any, kind: str, name: str = "value", strict: bool = False) -> None:
    """ldpc's typed arguments: a value of the wrong kind is a TypeError with Cython's message.
    Cython's check is exact (no bool for an int, no NumPy scalar, no int for a float); unless
    ``strict``, NumPy's scalars, and ints where a float is wanted, are taken."""
    if strict:
        ok = type(value) is (float if kind == "float" else int)
    else:
        ok = not isinstance(value, bool) and isinstance(value, (float, int, np.floating, np.integer) if kind == "float" else (int, np.integer))
    if not ok:
        raise _type_error(name, kind, value)


# The arguments ldpc's constructors declare as str, Optional[str] or Optional[List[int]], which
# Cython checks once the base class has set BP up: (types, None allowed). Unless strict (ldpc's
# v1 classes, which are its v2 classes), an integer is taken for a str and any sequence for a
# list, as this package's setters read them.
_CINIT_ARGS = {"bp_method": (str, True), "schedule": (str, True), "serial_schedule_order": (list, True), "input_vector_type": (str, False)}


def _arg_typed(value: Any, name: str, want: type, optional: bool, strict: bool) -> None:
    if value is None and optional:
        return
    if strict:
        ok = isinstance(value, want)
    elif want is str:
        ok = isinstance(value, (str, int, np.integer)) and not isinstance(value, bool)
    else:
        ok = value is not None
    if not ok:
        raise _type_error(name, want.__name__, value)


def _cinit_args(strict: bool, **given: Any) -> None:
    for name, value in given.items():
        if not isinstance(value, _Default):
            _arg_typed(value, name, *_CINIT_ARGS[name], strict)


def _osd_order_arg(order: Any) -> None:
    """The constructors' ``osd_order: int``, which Cython takes from any number."""
    if not isinstance(order, (int, float, np.integer, np.floating)):
        raise _type_error("osd_order", "int", order)


def _probability(p: Any) -> float:
    # ldpc takes any number; one outside [0, 1] only makes its posteriors NaN.
    if isinstance(p, (str, bytes)) or p is None:
        raise TypeError(f"must be real number, not {type(p).__name__}")
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"an error probability is between 0 and 1, not {p}")
    return p


# Fewer shots than this go to the cpu-f32 path on the gpu device (the bits are the same; a GPU
# earns its keep with hundreds of shots in flight).
_GPU_BATCH_MIN = 256


def _base(name: str, value: Any) -> Any:
    return _BASE_DEFAULTS[name] if isinstance(value, _Default) else value


def _declared(value: Any) -> Any:
    return value.value if isinstance(value, _Default) else value


def _one_at_a_time(method: Any) -> Any:
    """A decode with the decoder's other decodes kept out: each keeps its results on the decoder
    (``converge``, ``log_prob_ratios``, the decoding) and answers from them, so decodes from two
    threads on one decoder take turns (ldpc's hold the GIL throughout), and each answers its own
    syndrome."""

    @functools.wraps(method)
    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return run


class BpDecoderBase:
    """What ldpc's BP decoders share: the matrix, the channel, the BP options and the last
    run's results."""

    _core_factory: Any = None
    _defaults: dict = {}
    # ldpc's v1 classes check their arguments' types as ldpc's Cython does, exactly.
    _strict = False

    def _setup(self, pcm: Any, error_rate: Any, error_channel: Any, max_iter: Any, bp_method: Any, ms_scaling_factor: Any, schedule: Any, omp_thread_count: Any, random_schedule_seed: Any, serial_schedule_order: Any, random_serial_schedule: Any, channel_probs: Any, core_args: tuple) -> None:
        error_rate, error_channel, max_iter = _base("error_rate", error_rate), _base("error_channel", error_channel), _base("max_iter", max_iter)
        bp_method, ms_scaling_factor, schedule = _base("bp_method", bp_method), _base("ms_scaling_factor", ms_scaling_factor), _base("schedule", schedule)
        omp_thread_count, random_schedule_seed = _base("omp_thread_count", omp_thread_count), _base("random_schedule_seed", random_schedule_seed)
        serial_schedule_order, random_serial_schedule = _base("serial_schedule_order", serial_schedule_order), _base("random_serial_schedule", random_serial_schedule)
        if not _is_matrix(pcm) and not isinstance(pcm, (list, tuple)):
            raise TypeError(f"The input matrix is of an invalid type. Please input{_S12}a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")
        self._lock = threading.RLock()
        self.m, self.n, self._cols = _columns(pcm)
        self._channel = np.full(self.n, 0.5)
        self._method, self._scale, self._max_iter = "product_sum", 1.0, 0
        self._schedule = "parallel"
        self._omp = 1
        self._seed = 0
        self._random = False
        self._x = call(type(self)._core_factory, self.m, self._cols, list(self._channel), 0, "product_sum", 1.0, *core_args)
        self._converge, self._iter = False, 0
        self._llr = np.zeros(self.n)
        self._decoding = np.zeros(self.n, dtype=np.uint8)
        # In ldpc's order.
        self.bp_method = bp_method
        self.max_iter = max_iter
        self.ms_scaling_factor = ms_scaling_factor
        self.schedule = schedule
        self.serial_schedule_order = serial_schedule_order
        self.random_schedule_seed = random_schedule_seed
        self.omp_thread_count = omp_thread_count
        self.random_serial_schedule = random_serial_schedule
        if isinstance(channel_probs, (list, np.ndarray)) and len(channel_probs) > 0 and channel_probs[0] is not None:
            error_channel = channel_probs  # ldpc 1's name
        if error_channel is not None:
            self.error_channel = error_channel
        elif error_rate is not None:
            self.error_rate = error_rate
        else:
            # (ldpc's message, not an f-string there.)
            raise ValueError("Please specify the error channel. Either: 1) error_rate: float or 2) error_channel:" + _S12 + "list of floats of length equal to the block length of the code {self.n}.")

    # -- The channel ----------------------------------------------------------------------

    @property
    def error_rate(self) -> np.ndarray:
        return self._channel.copy()

    @error_rate.setter
    def error_rate(self, value: Optional[float]) -> None:
        if value is not None:
            _typed(value, "float", strict=self._strict)
            self._channel = np.full(self.n, _probability(value))
            call(self._x.set_priors, list(self._channel))

    @property
    def error_channel(self) -> np.ndarray:
        return self._channel.copy()

    @error_channel.setter
    def error_channel(self, value: Any) -> None:
        if value is not None:
            if len(value) != self.n:
                raise ValueError(f"The error channel vector must have length {self.n}, not {len(value)}.")
            self._channel = np.array([_probability(p) for p in (np.ravel(value) if isinstance(value, np.ndarray) else value)])
            call(self._x.set_priors, list(self._channel))

    def update_channel_probs(self, value: Any) -> None:
        self.error_channel = value

    @property
    def channel_probs(self) -> np.ndarray:
        return self._channel.copy()

    # -- The input ------------------------------------------------------------------------

    @property
    def input_vector_type(self) -> str:
        return self._input

    @input_vector_type.setter
    def input_vector_type(self, input_type: str) -> None:
        _arg_typed(input_type, "input_type", str, False, self._strict)
        t = str(input_type).lower()
        if t in ("auto", "a", "2"):
            if self.m == self.n:
                raise ValueError(f"Please specify the input vector type. Either: 1) input_vector_type: 'syndrome' or 2) input_vector_type:{_S16}'received_vector'.")
            self._input = "auto"
        elif t in ("syndrome", "s", "0"):
            self._input = "syndrome"
        elif t in ("received_vector", "r", "1"):
            self._input = "received_vector"
        else:
            raise ValueError(f"The input vector type '{input_type}' is invalid.{_S21}Please choose from the following methods:{_S21}'input_vector_type=syndrome', 'input_vector_type=received_vector'")

    # -- The last run ---------------------------------------------------------------------

    @property
    def log_prob_ratios(self) -> np.ndarray:
        return self._llr.copy()

    @property
    def converge(self) -> bool:
        return self._converge

    @property
    def iter(self) -> int:
        return self._iter

    @property
    def check_count(self) -> int:
        return self.m

    @property
    def bit_count(self) -> int:
        return self.n

    # -- BP's options ---------------------------------------------------------------------

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        _typed(value, "int", strict=self._strict)
        if value < 0:
            raise ValueError(f"max_iter input parameter must be a positive int. Not {value}.")
        self._max_iter = int(value) if value != 0 else self.n
        self._x.set_max_iter(self._max_iter)

    @property
    def bp_method(self) -> str:
        return self._method

    @bp_method.setter
    def bp_method(self, value: Union[str, int]) -> None:
        v = str(value).lower()
        if v in ("prod_sum", "product_sum", "ps", "0", "prod sum"):
            self._method = "product_sum"
        elif v in ("min_sum", "minimum_sum", "ms", "1", "minimum sum", "min sum"):
            self._method = "minimum_sum"
        else:
            raise ValueError(f"BP method '{value}' is invalid.{_S21}Please choose from the following methods:{_S21}'product_sum', 'minimum_sum'")
        call(self._x.set_method, self._method, float(self._scale))

    @property
    def schedule(self) -> str:
        return self._schedule

    @schedule.setter
    def schedule(self, value: Union[str, int]) -> None:
        v = _SCHEDULES.get(str(value).lower())
        if v is None:
            raise ValueError(f"The BP schedule method '{value}' is invalid.{_S21}Please choose from the following methods:{_S21}'schedule=parallel', 'schedule=serial', 'schedule=serial_relative'")
        self._schedule = v
        call(self._x.set_schedule, v)

    @property
    def serial_schedule_order(self) -> Optional[np.ndarray]:
        return np.asarray(self._x.order(), dtype=int)

    @serial_schedule_order.setter
    def serial_schedule_order(self, value: Any) -> None:
        if value is None:
            return
        if not len(value) == self.n:
            raise Exception("Input error. The `serial_schedule_order` input parameter must have length equal to the length of the code.")
        for i in range(self.n):
            if not isinstance(value[i], (int, np.integer)) or value[i] < 0 or value[i] >= self.n:
                raise ValueError(f"serial_schedule_order[{i}] is invalid. It must be a non-negative integer less than {self.n}.")
        call(self._x.set_order, [int(v) for v in value])
        self.random_serial_schedule = False

    @property
    def ms_scaling_factor(self) -> float:
        return self._scale

    @ms_scaling_factor.setter
    def ms_scaling_factor(self, value: float) -> None:
        # ldpc's is a C double: any real number (its own "must be specified as a float" is
        # never reached).
        if isinstance(value, (str, bytes)) or not hasattr(value, "__float__"):
            raise TypeError(f"must be real number, not {type(value).__name__}")
        self._scale = value
        call(self._x.set_method, self._method, float(value))

    @property
    def omp_thread_count(self) -> int:
        if self._omp != 1:
            warnings.warn("The OpenMP functionality is not yet implemented")
        return self._omp

    @omp_thread_count.setter
    def omp_thread_count(self, value: int) -> None:
        _typed(value, "int", strict=self._strict)
        if value < 1:
            raise TypeError(f"The omp_thread_count must be specified as a{_S12}positive integer.")
        self._omp = value
        if value != 1:
            warnings.warn("The OpenMP functionality is not yet implemented")

    @property
    def random_schedule_seed(self) -> int:
        return self._seed

    @random_schedule_seed.setter
    def random_schedule_seed(self, value: int) -> None:
        if self._strict or not isinstance(value, bool):
            _typed(value, "int", strict=self._strict)
        if value < -2:
            raise ValueError(f"The value of random_schedule_seed must{_S12}be a positive integer. Set as -1 to disable to the random{_S12}schedule. Set as 0 to use the system clock.")
        self._seed = int(value)
        self.random_serial_schedule = True
        self._x.seed(int(value) & 0xFFFFFFFF)

    @property
    def random_serial_schedule(self) -> bool:
        return self._random

    @random_serial_schedule.setter
    def random_serial_schedule(self, value: bool) -> None:
        self._random = bool(value)
        self._x.set_random(self._random)

    # -- Decoding -------------------------------------------------------------------------

    def _check_length(self, input_vector: Any) -> None:
        k = len(input_vector)
        if self._input == "syndrome" and k != self.m:
            raise ValueError(f"The input_vector must have length {self.m} (for syndrome decoding). Not length {k}.")
        if self._input == "received_vector" and k != self.n:
            raise ValueError(f"The input_vector must have length {self.n} (for received vector decoding). Not length {k}.")
        if self._input == "auto" and k not in (self.m, self.n):
            raise ValueError(f"The input_vector must have length {self.m} (for syndrome decoding) or length {self.n} (for received vector decoding). Not length {k}.")

    def _syndrome_of(self, vector: np.ndarray) -> List[int]:
        s = [0] * self.m
        for j in np.flatnonzero(vector):
            for r in self._cols[j]:
                s[r] ^= 1
        return s

    def _received(self, input_vector: Any) -> Optional[tuple]:
        """``input_vector_type`` set to ``"received_vector"`` after construction, which ldpc's
        BP+OSD, BP+LSD and BeliefFind honour as their BP does: BP decodes the vector's syndrome,
        and where it converges the answer is the codeword, BP's error taken off the vector. Where
        it does not, ldpc hands OSD, LSD or union-find the vector itself, read as a syndrome, with
        BP's posteriors, so the answer is then an error whose syndrome is the vector. The vector's
        length is checked as a syndrome's, so the matrix must be square. None for a syndrome;
        else (the vector's bits, their syndrome, the vector's dtype)."""
        if self._input != "received_vector":
            return None
        v = np.asarray(input_vector)
        if not len(v) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(v)}.")
        if self.m != self.n:
            raise ValueError(f"decoding a received vector here needs a square check matrix, not {self.m} by {self.n} (ldpc checks the vector's length against the checks and reads it as a word of the columns)")
        bits = (v != 0).astype(np.uint8)
        return bits, self._syndrome_of(bits), v.dtype

    def __reduce__(self) -> tuple:
        # Rebuilt from its matrix and options, then the settings changed since replayed on it,
        # as they were made on this one (OSD's candidates are those it was built with), and the
        # serial schedule's state carried over (its order, and a random one's generator, as the
        # decodes so far left them): the copy decodes on as this one does. The last results
        # come along.
        with self._lock:
            state = {k: v for k, v in self.__dict__.items() if k not in ("m", "n", "_cols", "_lock") and not type(v).__module__.startswith("stabilizer_qec._core")}
            state["_serial_state"] = self._x.serial_state()
            state.update(self._core_settings())
        args, kwargs = self._made_from
        return (_rebuild, (type(self), args, kwargs), state)

    def __setstate__(self, state: dict) -> None:
        state = dict(state)
        serial = state.pop("_serial_state", None)
        self.__dict__.update(state)
        call(self._x.set_priors, list(self._channel))
        self._x.set_max_iter(self._max_iter)
        call(self._x.set_method, self._method, float(self._scale))
        call(self._x.set_schedule, self._schedule)
        self._x.set_random(self._random)
        self._replay()
        if serial is not None:
            call(self._x.set_serial_state, serial)

    def _core_settings(self) -> dict:
        """Settings the core keeps alone, for a copy (beside the attributes)."""
        return {}

    def _replay(self) -> None:
        """The settings a subclass changes on the core after construction, made again on a
        copy's (from its attributes, ``_core_settings``' among them)."""


class BpDecoder(BpDecoderBase):
    """Belief propagation on a check matrix, as ldpc's ``BpDecoder``."""

    _core_factory = _core.Bp

    def __init__(
        self,
        pcm: Any,
        error_rate: Optional[float] = _Default(None),
        error_channel: Any = _Default(None),
        max_iter: int = _Default(0),
        bp_method: str = _Default("minimum_sum"),
        ms_scaling_factor: float = _Default(1.0),
        schedule: str = _Default("parallel"),
        omp_thread_count: int = _Default(1),
        random_schedule_seed: int = _Default(0),
        serial_schedule_order: Optional[List[int]] = _Default(None),
        input_vector_type: str = "auto",
        random_serial_schedule: bool = _Default(False),
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        if device not in ("cpu", "cpu-f32", "gpu"):
            raise ValueError(f"device must be 'cpu', 'cpu-f32' or 'gpu', not {device!r}")
        self._made_from = ((pcm, error_rate, error_channel), dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, schedule=schedule, omp_thread_count=omp_thread_count, random_schedule_seed=random_schedule_seed, serial_schedule_order=serial_schedule_order, input_vector_type=input_vector_type, random_serial_schedule=random_serial_schedule, device=device, **kwargs))
        self._input = "syndrome"
        if device != "cpu" and isinstance(bp_method, _Default):
            # The single-precision executors are min-sum; a device without a method asks for it
            # (ldpc's base default, product_sum, is what an unset method otherwise means).
            bp_method = "minimum_sum"
        # In ldpc's order: the base class's setup, the typed arguments, then the rest.
        self._setup(pcm, error_rate, error_channel, max_iter, bp_method, ms_scaling_factor, schedule, omp_thread_count, random_schedule_seed, serial_schedule_order, random_serial_schedule, kwargs.get("channel_probs", [None]), ())
        _cinit_args(self._strict, bp_method=bp_method, schedule=schedule, serial_schedule_order=serial_schedule_order, input_vector_type=input_vector_type)
        for key in kwargs:
            if key not in ("channel_probs",):
                raise ValueError(f"Unknown parameter '{key}' passed to the BpDecoder constructor.")
        self.input_vector_type = input_vector_type
        self.device = device
        if device != "cpu":  # the single-precision executors are min-sum, parallel schedule only
            if self.bp_method != "minimum_sum" or self.schedule != "parallel":
                raise ValueError(f"device {device!r} runs minimum_sum with the parallel schedule only, not {self.bp_method} / {self.schedule}")

    @_one_at_a_time
    def decode(self, input_vector: np.ndarray) -> np.ndarray:
        """The hard decision: for a syndrome, the error; for a received vector, the codeword.
        On a single-precision device (``"cpu-f32"``, ``"gpu"``) the shot goes through
        ``decode_batch``; the posteriors are not kept."""
        input_vector = np.asarray(input_vector)
        self._check_length(input_vector)
        dtype = input_vector.dtype
        received = self._input == "received_vector" or (self._input == "auto" and len(input_vector) == self.n)
        if not input_vector.any():
            self._converge = True
            return np.zeros(self.n, dtype=dtype)
        if received:
            vector = (input_vector != 0).astype(np.uint8)
            hard, llr, converged, iterations = self._decode_one(self._syndrome_of(vector))
            self._decoding = np.asarray(hard, dtype=np.uint8) ^ vector
        else:
            hard, llr, converged, iterations = self._decode_one([int(x != 0) for x in input_vector])
            self._decoding = np.asarray(hard, dtype=np.uint8)
        self._converge, self._iter, self._llr = bool(converged), int(iterations), np.asarray(llr, dtype=float)
        return self._decoding.astype(dtype)

    def _decode_one(self, syndrome: Any) -> tuple:
        if self.device == "cpu":
            return call(self._x.decode, syndrome)
        # One shot is one thread on either single-precision device: the CPU's, which the GPU
        # matches bit for bit, without a graph upload and a readback per call.
        hard, converged, iterations = self._decode_rows32(np.asarray(syndrome, dtype=np.uint8).reshape(1, -1), "cpu-f32", 1)
        return hard[0], np.full(self.n, np.nan), bool(converged[0]), int(iterations[0])

    def _f32_scale(self) -> float:
        scale = float(self._scale)
        as_f32 = float(np.float32(scale))
        if not np.isfinite(as_f32) or (as_f32 == 0.0) != (scale == 0.0):
            raise ValueError(f"ms_scaling_factor {scale!r} is not a single-precision number (0 means adaptive scaling): the {self.device!r} device cannot use it")
        return scale

    def _decode_rows32(self, rows: np.ndarray, device: str, threads: int) -> tuple:
        self._f32_scale()
        packed = (rows != 0).astype(np.uint8).tobytes()
        hard, converged, iterations = call(self._x.decode_batch32, packed, int(rows.shape[0]), device, int(threads))
        stride = (self.n + 7) // 8
        bits = np.unpackbits(np.frombuffer(hard, dtype=np.uint8).reshape(len(rows), stride), axis=1, bitorder="little")[:, : self.n]
        return bits, np.frombuffer(converged, dtype=np.uint8).astype(bool), np.asarray(iterations, dtype=np.int64)

    @_one_at_a_time
    def decode_batch(self, syndromes: Any, *, threads: int = 0, _details: bool = False) -> np.ndarray:
        """Many syndromes at once (one row each, ``m`` bits: syndromes whatever
        ``input_vector_type`` says): the hard decisions as rows of 0/1. On ``device="cpu"`` each
        row runs the double-precision decoder; on ``"cpu-f32"`` the single-precision min-sum of
        the GPU, ``threads`` wide; on ``"gpu"`` the same on the GPU, bit for bit. An all-zero
        row is the zero error on every device, as ``decode`` has it; the decoder's
        ``converge``, ``iter`` and ``log_prob_ratios`` are left as they were."""
        rows = np.asarray(syndromes)
        if rows.ndim != 2 or rows.shape[1] != self.m:
            raise ValueError(f"syndromes are rows of {self.m} bits, not shape {rows.shape}")
        threads = int(threads)
        if threads < 0:
            raise ValueError(f"threads must be 0 (every core) or positive, not {threads}")
        out = np.zeros((len(rows), self.n), dtype=np.uint8)
        converged = np.ones(len(rows), dtype=bool)
        iterations = np.zeros(len(rows), dtype=np.int64)
        live = np.flatnonzero(rows.any(axis=1))
        if len(live):
            if self.device == "cpu":
                for i in live:
                    hard, _llr, conv, iters = call(self._x.decode, [int(x != 0) for x in rows[i]])
                    out[i] = np.asarray(hard, dtype=np.uint8)
                    converged[i], iterations[i] = bool(conv), int(iters)
            else:
                # A GPU shot is one thread walking the whole graph: a small batch runs faster
                # on the CPU's threads, with the same bits (the device is still asked for, once,
                # so a machine without one hears so on the first decode).
                device = self.device
                if self.device == "gpu" and len(live) < _GPU_BATCH_MIN:
                    if not getattr(self, "_gpu_seen", False):
                        if call(_core.gpu_device) is None:
                            raise ValueError("device 'gpu': no GPU adapter is available to wgpu on this machine; use device='cpu-f32' (the same bits)")
                        self._gpu_seen = True
                    device = "cpu-f32"
                out[live], converged[live], iterations[live] = self._decode_rows32(rows[live], device, threads)
        return (out, converged, iterations) if _details else out

    @property
    def decoding(self) -> np.ndarray:
        return self._decoding.astype(int)


class SoftInfoBpDecoder(BpDecoderBase):
    """Soft-information BP (serial min-sum on analog syndromes, Raveendran et al.
    arXiv:2205.02341), as ldpc's ``SoftInfoBpDecoder``: each syndrome a value whose sign is
    the bit and whose magnitude its reliability, ``2s/σ²`` its log-likelihood ratio; those
    below ``cutoff`` take part as virtual bits and can flip."""

    _core_factory = _core.Bp

    def __init__(
        self,
        pcm: Any,
        error_rate: Optional[float] = _Default(None),
        error_channel: Any = _Default(None),
        max_iter: int = _Default(0),
        bp_method: str = _Default("minimum_sum"),
        ms_scaling_factor: float = _Default(1.0),
        cutoff: float = np.inf,
        sigma: float = 2.0,
        **kwargs: Any,
    ) -> None:
        self._made_from = ((pcm, error_rate, error_channel), dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, cutoff=cutoff, sigma=sigma, **kwargs))
        self._input = "syndrome"
        self._setup(
            pcm, error_rate, error_channel, max_iter, bp_method, ms_scaling_factor, kwargs.get("schedule", 0), kwargs.get("omp_thread_count", 1), kwargs.get("random_schedule_seed", 0),
            kwargs.get("serial_schedule_order"), kwargs.get("random_serial_schedule", False), kwargs.get("channel_probs", [None]), (),
        )
        self.cutoff = cutoff
        if not isinstance(sigma, (float, int)) or isinstance(sigma, bool) or sigma <= 0:
            raise ValueError("The sigma value must be a float greater than 0.")
        self.sigma = float(sigma)
        self.schedule = "serial"
        self.bp_method = "minimum_sum"
        self.input_vector_type = "syndrome"
        self._soft = np.zeros(self.m)

    @_one_at_a_time
    def decode(self, soft_info_syndrome: np.ndarray) -> np.ndarray:
        soft = [float(x) for x in np.asarray(soft_info_syndrome, dtype=float)[: self.m]]
        hard, llr, converged, iterations, out = call(self._x.decode_soft, soft, float(self.cutoff), float(self.sigma), float(self._scale), self._seed & 0xFFFFFFFF)
        self._converge, self._iter, self._llr = bool(converged), int(iterations), np.asarray(llr, dtype=float)
        self._decoding = np.asarray(hard, dtype=np.uint8)
        self._soft = np.asarray(out, dtype=float)
        return self._decoding.copy()

    @property
    def soft_syndrome(self) -> np.ndarray:
        return self._soft.copy()

    @property
    def decoding(self) -> np.ndarray:
        return self._decoding.astype(int)


def _osd_name(method: Any) -> Optional[str]:
    v = str(method).lower()
    if v in ("osd_0", "0", "osd0"):
        return "osd_0"
    if v in ("osd_e", "e", "exhaustive"):
        return "osd_e"
    if v in ("osd_cs", "1", "cs", "combination_sweep"):
        return "osd_cs"
    if v in ("off", "osd_off", "deactivated", "-1"):
        return "osd_off"
    return None


class BpOsdDecoder(BpDecoderBase):
    """BP+OSD on a check matrix, as ldpc's ``BpOsdDecoder``: BP, and where it does not
    converge, ordered-statistics decoding (OSD-0, exhaustive or combination sweep). As in ldpc,
    OSD's candidates are those of the method and order it was built with; a later
    ``osd_order`` of 0 turns them off."""

    _core_factory = _core.BpOsd

    def __init__(
        self,
        pcm: Any,
        error_rate: Optional[float] = _Default(None),
        error_channel: Any = _Default(None),
        max_iter: int = _Default(0),
        bp_method: str = _Default("minimum_sum"),
        ms_scaling_factor: float = _Default(1.0),
        schedule: str = _Default("parallel"),
        omp_thread_count: int = _Default(1),
        random_schedule_seed: int = _Default(0),
        serial_schedule_order: Optional[List[int]] = _Default(None),
        osd_method: Union[str, int, float] = 0,
        osd_order: int = 0,
        input_vector_type: str = "syndrome",
        **kwargs: Any,
    ) -> None:
        self._made_from = ((pcm, error_rate, error_channel), dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, schedule=schedule, omp_thread_count=omp_thread_count, random_schedule_seed=random_schedule_seed, serial_schedule_order=serial_schedule_order, osd_method=osd_method, osd_order=osd_order, **kwargs))
        # In ldpc's order: the base class's setup, the typed arguments, the unknown ones, then
        # the OSD method and order (OSD-0 until they are known good).
        name = _osd_name(osd_method)
        order = int(osd_order) if isinstance(osd_order, (int, float, np.integer, np.floating)) else 0
        good = name is not None and order >= 0 and (name != "osd_0" or order == 0)
        core_name = "osd_0" if not good or name in ("osd_0", "osd_off") else name
        self._input = "syndrome"
        self._setup(pcm, error_rate, error_channel, max_iter, bp_method, ms_scaling_factor, schedule, omp_thread_count, random_schedule_seed, serial_schedule_order, False, kwargs.get("channel_probs", [None]), (core_name, order if good else 0))
        _cinit_args(self._strict, bp_method=bp_method, schedule=schedule, serial_schedule_order=serial_schedule_order)
        _osd_order_arg(osd_order)
        _cinit_args(self._strict, input_vector_type=input_vector_type)
        for key in kwargs:
            if key not in ("channel_probs",):
                raise ValueError(f"Unknown parameter '{key}' passed to the BpDecoder constructor.")
        if name is None:
            raise ValueError(f"ERROR: OSD method '{osd_method}' invalid. Please choose from the following methods:{_S16}'OSD_0', 'OSD_E' or 'OSD_CS'.")
        self._check_order(name, order)
        self._osd, self._order = name, order
        self._bp_decoding = np.zeros(self.n, dtype=np.uint8)
        self._osd0 = np.zeros(self.n, dtype=np.uint8)
        self._osdw = np.zeros(self.n, dtype=np.uint8)

    @staticmethod
    def _check_order(name: str, order: int) -> None:
        if order < 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. Please choose a positive integer.")
        if name == "osd_0" and order != 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. The 'osd_method' is set to 'OSD_0'. The osd order must therefore be set to 0.")
        if name == "osd_e" and order > 15:
            warnings.warn("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth greater than 15 is not recommended. Use the 'osd_cs' method instead.")

    @_one_at_a_time
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        received = self._received(syndrome)
        if received is None:
            return self._decode_syndrome(syndrome)
        bits, bits_syndrome, dtype = received
        if not bits.any():
            self._converge = True
            return np.zeros(self.n, dtype=dtype)
        _, converged, iterations, hard, llr, _ = call(self._x.decode, bits_syndrome)
        self._converge, self._iter, self._llr = bool(converged), int(iterations), np.asarray(llr, dtype=float)
        self._bp_decoding = np.asarray(hard, dtype=np.uint8) ^ bits
        if converged:
            return self._bp_decoding.astype(dtype)
        osdw, osd0 = call(self._x.osd_on, [int(x) for x in bits], list(self._llr))
        self._osd0, self._osdw = np.asarray(osd0, dtype=np.uint8), np.asarray(osdw, dtype=np.uint8)
        return self._osdw.astype(dtype)

    def _decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = np.asarray(syndrome)
        if not len(syndrome) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        if not syndrome.any():
            self._converge = True
            return np.zeros(self.n, dtype=syndrome.dtype)
        correction, converged, iterations, hard, llr, osd0 = call(self._x.decode, [int(x != 0) for x in syndrome])
        self._converge, self._iter, self._llr = bool(converged), int(iterations), np.asarray(llr, dtype=float)
        self._bp_decoding = np.asarray(hard, dtype=np.uint8)
        if not converged:
            self._osd0 = np.asarray(osd0, dtype=np.uint8)
            self._osdw = np.asarray(correction, dtype=np.uint8)
        return np.asarray(correction, dtype=syndrome.dtype)

    def _replay(self) -> None:
        self._x.set_osd_order(self._order)

    @property
    def osd_method(self) -> Optional[str]:
        return {"osd_0": "OSD_0", "osd_e": "OSD_E", "osd_cs": "OSD_CS", "osd_off": "OSD_OFF"}.get(self._osd)

    @osd_method.setter
    def osd_method(self, method: Union[str, int, float]) -> None:
        name = _osd_name(method)
        if name is None:
            raise ValueError(f"ERROR: OSD method '{method}' invalid. Please choose from the following methods:{_S16}'OSD_0', 'OSD_E' or 'OSD_CS'.")
        self._osd = name
        if name == "osd_0":
            self._order = 0
            self._x.set_osd_order(0)

    @property
    def osd_order(self) -> int:
        return self._order

    @osd_order.setter
    def osd_order(self, order: int) -> None:
        _typed(order, "int", "order", self._strict)
        self._check_order(self._osd, order)
        self._order = int(order)
        self._x.set_osd_order(int(order))

    @property
    def decoding(self) -> np.ndarray:
        return self._osdw.astype(int)

    @property
    def bp_decoding(self) -> np.ndarray:
        return self._bp_decoding.astype(int)

    @property
    def osd0_decoding(self) -> np.ndarray:
        return (self._bp_decoding if self._converge else self._osd0).astype(int)

    @property
    def osdw_decoding(self) -> np.ndarray:
        return (self._bp_decoding if self._converge else self._osdw).astype(int)


# ldpc's lsd_method setter (which its constructors call): spellings of each method; OSD_OFF is
# the setter's alone.
_LSD_SPELLINGS = (("osd_0", ("osd_0", "0", "osd0", "lsd_0", "lsd0")), ("osd_e", ("osd_e", "e", "exhaustive", "lsd_e", "lsde")), ("osd_cs", ("osd_cs", "1", "cs", "combination_sweep", "lsd_cs")), ("osd_off", ("off", "osd_off", "deactivated", "lsd_off")))


def _lsd_setter_name(method: Any) -> str:
    m = str(method).lower()
    for name, spellings in _LSD_SPELLINGS:
        if m in spellings:
            return name
    raise ValueError(f"ERROR: OSD method '{method}' invalid. Please choose from the following methods:{_S16}'LSD_0', 'LSD_E' or 'LSD_CS'.")


def _lsd_ctor(lsd_method: Any, lsd_order: Any) -> tuple:
    """(method, order) as ldpc's LSD constructors read them: their own checks, then the
    ``lsd_method`` setter, which refuses some spellings the checks let through (``2``,
    ``"osde"``, ``"osdcs"``, ``"lsdcs"``) and sets the order to 0 for LSD-0."""
    if lsd_order < 0:
        raise ValueError(f"lsd_order must be greater than or equal to 0. Not {lsd_order}.")
    if isinstance(lsd_method, str):
        if lsd_method.lower() not in ("osd_0", "osd_e", "osd_cs", "osde", "osdcs", "osd0", "lsd_0", "lsd_e", "lsd_cs", "lsd0", "lsdcs", "lsde"):
            raise ValueError(f"lsd_method must be one of 'LSD_0', 'LSD_E', 'LSD_CS'. Not {lsd_method}.")
    elif isinstance(lsd_method, int):
        if lsd_method not in (0, 1, 2):
            raise ValueError(f"lsd_method must be one of 0, 1, 2. Not {lsd_method}.")
    else:
        raise ValueError(f"lsd_method must be one of 'LSD_0' (0), 'LSD_E' (1), 'LSD_CS' (2). Not {lsd_method}.")
    name = _lsd_setter_name(lsd_method)
    return name, 0 if name == "osd_0" else count(lsd_order, "lsd_order")


class _LsdOptions:
    """``lsd_method`` and ``lsd_order``, read and set as ldpc's (the next decode uses them)."""

    _lsd_method: str
    _lsd_order: int
    _x: Any

    @property
    def lsd_method(self) -> Optional[str]:
        return {"osd_0": "lSD_0", "osd_e": "LSD_E", "osd_cs": "LSD_CS", "osd_off": "LSD_OFF"}.get(self._lsd_method)

    @lsd_method.setter
    def lsd_method(self, method: Union[str, int, float]) -> None:
        name = _lsd_setter_name(method)
        self._lsd_method = name
        if name == "osd_0":
            self._lsd_order = 0
        call(self._x.set_lsd, name, self._lsd_order)

    @property
    def lsd_order(self) -> int:
        return self._lsd_order

    @lsd_order.setter
    def lsd_order(self, order: int) -> None:
        _typed(order, "int", "order")
        if order < 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. Please choose a positive integer.")
        if self._lsd_method == "osd_0" and order != 0:
            raise ValueError(f"ERROR: OSD order '{order}' invalid. The 'osd_method' is set to 'OSD_0'. The osd order must therefore be set to 0.")
        if self._lsd_method == "osd_e" and order > 15:
            warnings.warn("WARNING: Running the 'OSD_E' (Exhaustive method) with search depth greater than 15 is not recommended. Use the 'osd_cs' method instead.")
        self._lsd_order = int(order)
        call(self._x.set_lsd, self._lsd_method, self._lsd_order)


class BpLsdDecoder(_LsdOptions, BpDecoderBase):
    """BP+LSD on a check matrix, as ldpc's ``BpLsdDecoder``: BP, and where it does not converge
    (or always, with ``always_run_lsd``), localised statistics decoding: clusters grown from
    the syndrome by BP's posteriors, ``bits_per_step`` at a time, each solved by OSD of
    ``lsd_order``. ``osd_method``/``osd_order`` are accepted for ``lsd_method``/``lsd_order``."""

    _core_factory = _core.BpLsd

    def __init__(
        self,
        pcm: Any,
        error_rate: Optional[float] = _Default(None),
        error_channel: Any = _Default(None),
        max_iter: int = _Default(0),
        bp_method: str = _Default("minimum_sum"),
        ms_scaling_factor: float = _Default(1.0),
        schedule: str = _Default("parallel"),
        omp_thread_count: int = _Default(1),
        random_schedule_seed: int = _Default(0),
        serial_schedule_order: Optional[List[int]] = _Default(None),
        bits_per_step: int = 1,
        input_vector_type: str = "syndrome",
        lsd_order: int = 0,
        lsd_method: Union[str, int] = 0,
        always_run_lsd: bool = False,
        **kwargs: Any,
    ) -> None:
        self._made_from = ((pcm, error_rate, error_channel), dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, schedule=schedule, omp_thread_count=omp_thread_count, random_schedule_seed=random_schedule_seed, serial_schedule_order=serial_schedule_order, bits_per_step=bits_per_step, lsd_order=lsd_order, lsd_method=lsd_method, always_run_lsd=always_run_lsd, **kwargs))
        lsd_method = kwargs.pop("osd_method", lsd_method)
        lsd_order = kwargs.pop("osd_order", lsd_order)
        method, order = _lsd_ctor(lsd_method, lsd_order)
        bits = pcm.shape[1] if bits_per_step == 0 else int(bits_per_step)
        self._input = "syndrome"
        self._setup(pcm, error_rate, error_channel, max_iter, bp_method, ms_scaling_factor, schedule, omp_thread_count, random_schedule_seed, serial_schedule_order, kwargs.get("random_serial_schedule", _Default(False)), kwargs.get("channel_probs", [None]), (method, order, bits, bool(always_run_lsd)))
        _cinit_args(self._strict, bp_method=bp_method, schedule=schedule, serial_schedule_order=serial_schedule_order, input_vector_type=input_vector_type)
        self._lsd_method, self._lsd_order = method, order
        self.always_run_lsd = bool(always_run_lsd)
        self.bits_per_step = bits
        self._bp_decoding = np.zeros(self.n, dtype=np.uint8)
        self.last_tied = False
        self._stat_fields: dict = {}

    @_one_at_a_time
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        received = self._received(syndrome)
        if received is None:
            return self._decode_syndrome(syndrome)
        bits, bits_syndrome, dtype = received
        _, converged, iterations, _, _, _, hard, llr = call(self._x.decode, bits_syndrome)
        self._converge, self._iter, self._llr = bool(converged), int(iterations), np.asarray(llr, dtype=float)
        self._bp_decoding = np.asarray(hard, dtype=np.uint8) ^ bits
        self._stat_fields = {}
        self.last_tied = False
        if converged and not self.always_run_lsd:
            return self._bp_decoding.astype(dtype)
        correction, solved, ties, _ = call(self._x.decode_lsd, [int(x) for x in bits], list(self._llr))
        if not solved:
            raise ValueError("no correction explains the syndrome")
        self.last_tied = ties > 0
        return np.asarray(correction, dtype=dtype)

    def _decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        """The correction. ``last_tied`` says whether this decode made a choice ldpc might
        make otherwise (equal keys in a long sort, or a merge of many clusters at once)."""
        syndrome = np.asarray(syndrome)
        if not len(syndrome) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        # ldpc's BpLsdDecoder runs BP whatever the syndrome (its zero check never fires).
        correction, converged, iterations, solved, ties, _, hard, llr = call(self._x.decode, [int(x != 0) for x in syndrome])
        if not solved:
            raise ValueError("no correction explains the syndrome")
        self._converge, self._iter, self._llr = bool(converged), int(iterations), np.asarray(llr, dtype=float)
        self._bp_decoding = np.asarray(hard, dtype=np.uint8)
        self.last_tied = ties > 0
        self._stat_fields = {}  # (each decode starts the statistics afresh, as in ldpc)
        return np.asarray(correction, dtype=syndrome.dtype)

    def _core_settings(self) -> dict:
        return {"_do_stats": self.do_stats}

    def _replay(self) -> None:
        call(self._x.set_lsd, self._lsd_method, self._lsd_order)
        self.set_do_stats(self.__dict__.pop("_do_stats", False))

    # -- ldpc's statistics -----------------------------------------------------------------

    @property
    def statistics(self) -> dict:
        """The last decode's statistics, as ldpc's: each cluster (keyed by the check it grew
        from) with its growth steps, size history, merges, the timestep it became valid or was
        absorbed and by which cluster, its final bits, its matrix's entries and sparsity, and
        (at order 0) its solution; the bits each cluster took at each timestep; BP's
        posteriors; the method, order, syndrome and time taken in microseconds. Most of it only
        while ``do_stats`` is on."""
        st = call(self._x.statistics)
        return {
            "individual_cluster_stats": st["individual_cluster_stats"],
            "global_timestep_bit_history": st["global_timestep_bit_history"],
            "elapsed_time": st["elapsed_time"],
            "lsd_order": st["lsd_order"],
            "lsd_method": st["lsd_method"],
            "bit_llrs": st["bit_llrs"],
            "error": list(self._stat_fields.get("error", [])),
            "syndrome": list(self._stat_fields["syndrome"]) if "syndrome" in self._stat_fields else st["syndrome"],
            "compare_recover": list(self._stat_fields.get("compare_recover", [])),
        }

    @property
    def do_stats(self) -> bool:
        return bool(self._x.do_stats)

    def set_do_stats(self, value: bool) -> None:
        call(self._x.set_do_stats, bool(value))

    def set_additional_stat_fields(self, error: Any, syndrome: Any, compare_recover: Any) -> None:
        """Fields kept with the statistics until the next decode."""
        self._stat_fields = {"error": [int(x) for x in error], "syndrome": [int(x) for x in syndrome], "compare_recover": [int(x) for x in compare_recover]}

    def reset_cluster_stats(self) -> None:
        call(self._x.reset_stats)
        self._stat_fields = {}

    @_one_at_a_time
    def _cluster_bits(self, syndrome: Any) -> list:
        """Each final cluster's (id, bits in robin-set order): for checking against ldpc's
        statistics."""
        return call(self._x.decode, [int(x != 0) for x in np.asarray(syndrome)], True)[5]

    @_one_at_a_time
    def _merge_order_outcomes(self, syndrome: np.ndarray, max_leaves: int = 4096, target: Optional[np.ndarray] = None) -> set[bytes]:
        """Every correction a decode of ``syndrome`` reaches over the merge orders ldpc can take,
        each as ``np.uint8`` bytes, one per column: for checking against ldpc. ldpc merges the
        clusters a growth step meets in the order of their hashed heap addresses, so one
        syndrome can decode two ways in one process; ``decode`` takes one of the orders, and its
        answer is always in the set. The decode is the last one when that was of ``syndrome``
        (a random or relative serial schedule moves on at each decode: its BP runs again from
        where that decode's started), else the next one; the schedule is left as it was, so a
        search before ``decode(syndrome)`` and one after it are alike. A syndrome no correction
        explains raises ``decode``'s ValueError (no order explains it, so the walk ends at the
        first). BP runs once. The
        orders (any order of the clusters met, and whether each cluster met while merging is
        merged) are walked for at most ``max_leaves`` of them, fewest departures from
        ``decode``'s order first. With ``target`` (a correction, one entry per column), the walk
        stops at the first order that reaches it: each order is a whole LSD decode, so asking
        whether ldpc's answer is reachable should pass it. ``_merge_search`` then says how far:
        (orders walked, the departures up to which every order was walked, whether all were)."""
        syndrome = np.asarray(syndrome)
        if not len(syndrome) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        if target is not None:
            target = np.asarray(target)
            if not len(target) == self.n:
                raise ValueError(f"The target must have length {self.n}. Not {len(target)}.")
            target = [int(x != 0) for x in target]
        found, paths, departures, complete = call(self._x.merge_order_outcomes, [int(x != 0) for x in syndrome], count(max_leaves, "max_leaves", 1), target)
        self._merge_search = (paths, departures, complete)
        if not found:
            raise ValueError("no correction explains the syndrome")
        return {np.asarray(c, dtype=np.uint8).tobytes() for c in found}

    @property
    def bp_decoding(self) -> np.ndarray:
        return self._bp_decoding.astype(int)



class BpFlipDecoder(BpDecoderBase):
    """Bit flipping, then BP on what it leaves, as ldpc's ``BpFlipDecoder``: Gallager's
    flipping for ``flip_iterations`` passes (0: the block length), every ``pflip_frequency``-th
    pass flipping tied bits at random (``pflip_seed``; 0 is the clock), then BP decodes the
    remaining syndrome. ``osd_method`` and ``osd_order`` are accepted and unused, as in ldpc."""

    _core_factory = _core.Bp

    def __init__(
        self,
        pcm: Any,
        error_rate: Optional[float] = _Default(None),
        error_channel: Any = _Default(None),
        max_iter: int = _Default(0),
        bp_method: str = _Default("minimum_sum"),
        ms_scaling_factor: float = _Default(1.0),
        schedule: str = _Default("parallel"),
        omp_thread_count: int = _Default(1),
        random_schedule_seed: int = _Default(False),
        serial_schedule_order: Optional[List[int]] = _Default(None),
        osd_method: int = 0,
        osd_order: int = 0,
        flip_iterations: int = 0,
        pflip_frequency: int = 0,
        pflip_seed: int = 0,
    ) -> None:
        self._made_from = ((pcm, error_rate, error_channel), dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, schedule=schedule, omp_thread_count=omp_thread_count, random_schedule_seed=random_schedule_seed, serial_schedule_order=serial_schedule_order, osd_method=osd_method, osd_order=osd_order, flip_iterations=flip_iterations, pflip_frequency=pflip_frequency, pflip_seed=pflip_seed))
        self._input = "syndrome"
        self._setup(pcm, error_rate, error_channel, max_iter, bp_method, ms_scaling_factor, schedule, omp_thread_count, random_schedule_seed, serial_schedule_order, _Default(False), [None], ())
        self.flip_iterations = flip_iterations
        self._flip = call(_core.Flip, self.m, self._cols, int(flip_iterations), int(pflip_frequency), int(pflip_seed) & 0xFFFFFFFF)
        self._decoding = np.zeros(self.n, dtype=np.uint8)

    @_one_at_a_time
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = np.asarray(syndrome)
        if not len(syndrome) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        if not syndrome.any():
            self._converge = True
            return np.zeros(self.n, dtype=syndrome.dtype)
        flipped, _, _ = call(self._flip.decode, [int(x != 0) for x in syndrome])
        flipped = np.asarray(flipped, dtype=np.uint8)
        residual = [int(x != 0) ^ y for x, y in zip(syndrome, self._syndrome_of(flipped))]
        hard, llr, converged, iterations = call(self._x.decode, residual)
        self._converge, self._iter, self._llr = bool(converged), int(iterations), np.asarray(llr, dtype=float)
        self._decoding = np.asarray(hard, dtype=np.uint8) ^ flipped
        return self._decoding.astype(syndrome.dtype)

    @property
    def decoding(self) -> np.ndarray:
        # (ldpc's raises AttributeError.)
        return self._decoding.astype(int)


class LsdDecoder(_LsdOptions):
    """Localised statistics decoding alone, given each bit's weight (lower: likelier), as
    ldpc's ``LsdDecoder``: clusters grown from the syndrome ``bits_per_step`` bits at a time,
    lightest first, each solved by OSD of ``lsd_order`` (``osd_method``/``osd_order`` accepted for
    ``lsd_method``/``lsd_order``)."""

    def __init__(self, pcm: Any, bits_per_step: int = 1, lsd_order: int = 0, lsd_method: Union[str, int] = 0, **kwargs: Any) -> None:
        self._made_from = ((pcm,), dict(bits_per_step=bits_per_step, lsd_order=lsd_order, lsd_method=lsd_method, **kwargs))
        lsd_method = kwargs.get("osd_method", lsd_method)
        lsd_order = kwargs.get("osd_order", lsd_order)
        _lsd_ctor(lsd_method, lsd_order)  # (its checks come before the matrix's)
        if not _is_matrix(pcm) and not isinstance(pcm, (list, tuple)):
            raise TypeError(f"The input matrix is of an invalid type. Please input{_S12}a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")
        self.m, self.n, cols = _columns(pcm)
        method, order = _lsd_ctor(lsd_method, lsd_order)
        self.bits_per_step = self.n if bits_per_step == 0 else int(bits_per_step)
        self._lsd_method, self._lsd_order = method, order
        self._x = call(_core.BpLsd, self.m, cols, [0.5] * self.n, 1, "product_sum", 1.0, method, order, self.bits_per_step, True)
        self._decoding = np.zeros(self.n, dtype=np.uint8)
        self.last_tied = False
        self._lock = threading.RLock()

    def __reduce__(self) -> tuple:
        # Rebuilt from its matrix and options, with the LSD method and order it has now.
        args, kwargs = self._made_from
        return (_rebuild, (type(self), args, kwargs), {"_decoding": self._decoding, "last_tied": self.last_tied, "_lsd_method": self._lsd_method, "_lsd_order": self._lsd_order})

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        call(self._x.set_lsd, self._lsd_method, self._lsd_order)

    @_one_at_a_time
    def decode(self, syndrome: np.ndarray, bit_weights: Any) -> np.ndarray:
        syndrome = np.asarray(syndrome)
        if not len(bit_weights) == self.n:
            raise ValueError(f"The bit weights must have length {self.n}. Not {len(bit_weights)}.")
        if not len(syndrome) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        correction, solved, ties, _ = call(self._x.decode_lsd, [int(x != 0) for x in syndrome], [float(w) for w in bit_weights])
        if not solved:
            raise ValueError("no correction explains the syndrome")
        self.last_tied = ties > 0
        self._decoding = np.asarray(correction, dtype=np.uint8)
        return self._decoding.astype(syndrome.dtype)

    @property
    def decoding(self) -> np.ndarray:
        return self._decoding.copy()



def _per_step(bits_per_step: int, n: int) -> int:
    bits_per_step = int(bits_per_step)
    return n if bits_per_step == 0 else (bits_per_step if bits_per_step > 0 else 1 << 62)


class UnionFindDecoder:
    """Union-find decoding of any check matrix, as ldpc's ``UnionFindDecoder``: a cluster grows
    around each fired check until it can explain its checks, then each is solved alone, by
    elimination (``uf_method`` true: any matrix) or by peeling (false, the default: columns of at
    most two checks, a column with one check touching the boundary). Given ``llrs``, growth takes
    the ``bits_per_step`` lowest first (0: all). ``uf_method`` may also be a name,
    ``"inversion"`` or ``"peeling"``, and the name is what runs. ldpc's own differs here: its
    argument is typed ``str`` but kept in a C ``bool``, so any non-empty string is inversion
    there, ``"peeling"`` included; its only peeling is ``uf_method=""``; and it refuses a
    ``bool`` with a TypeError, its own default ``False`` included, so ``UnionFindDecoder(H)``
    fails there and peels here. So ldpc's ``UnionFindDecoder(H, uf_method="peeling")`` is this
    one's ``uf_method="inversion"``, shot for shot; against ldpc's ``uf_method=""``, this one's
    ``"peeling"`` differs only where flagged, with or without ``llrs``.
    Corrections are ldpc's but where ``last_tied`` says the last decode merged three or more
    clusters at once (ldpc's order there comes from hashed pointers), or ``last_stuck`` says it
    left ldpc's algorithm: where ldpc's peeling goes round forever, or (ldpc 2.4.1 peels the
    boundary through a bit its spanning tree may not have taken) peels to a correction that does
    not explain the syndrome, as on 78 of the distance-5 rotated surface code's 4096; there a
    cluster is peeled through the tree's boundary bit, or solved by inversion."""

    def __init__(self, pcm: Any, uf_method: Union[bool, str] = False) -> None:
        if not _is_matrix(pcm):
            raise TypeError(f"The input matrix is of an invalid type. Please input{_S12}a np.ndarray or spmatrix object, not {type(pcm)}")
        self._made_from = ((pcm,), dict(uf_method=uf_method))
        self.m, self.n, cols = _columns(pcm)
        self._x = call(_core.LdpcUf, self.m, cols)
        if isinstance(uf_method, str):
            self.uf_method = uf_method.lower() not in ("", "peel", "peeling")
        else:
            self.uf_method = bool(uf_method)
        self.bits_per_step = 0
        self._decoding = np.zeros(self.n, dtype=np.uint8)
        self.last_tied = self.last_stuck = False
        self._lock = threading.RLock()

    def __reduce__(self) -> tuple:
        args, kwargs = self._made_from
        return (_rebuild, (type(self), args, kwargs), {"_decoding": self._decoding, "last_tied": self.last_tied, "last_stuck": self.last_stuck, "bits_per_step": self.bits_per_step})

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    @_one_at_a_time
    def decode(self, syndrome: np.ndarray, llrs: Optional[np.ndarray] = None, bits_per_step: int = 0) -> np.ndarray:
        syndrome = np.asarray(syndrome)
        if not len(syndrome) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        if llrs is not None and not len(llrs) == self.n:
            raise ValueError(f"The llrs must have length {self.n}. Not {len(llrs)}.")
        if not syndrome.any():
            return np.zeros(self.n, dtype=syndrome.dtype)
        self.bits_per_step = self.n if int(bits_per_step) == 0 else int(bits_per_step)
        peel = not self.uf_method
        if peel and not self._x.max_degree_2:
            raise RuntimeError("Peel decoder only works for planar codes. Use the matrix_decode method for more general codes.")
        weights = None if llrs is None else [float(x) for x in llrs]
        correction, solved, ties, stuck = call(self._x.decode, [int(x != 0) for x in syndrome], weights, _per_step(bits_per_step, self.n), peel)
        if not solved:
            raise ValueError("no correction explains the syndrome")
        self.last_tied, self.last_stuck = ties > 0, stuck > 0
        self._decoding = np.asarray(correction, dtype=np.uint8)
        return self._decoding.astype(syndrome.dtype)

    @property
    def decoding(self) -> np.ndarray:
        # (ldpc's returns None.)
        return self._decoding.copy()


class BeliefFindDecoder(BpDecoderBase):
    """BP, and where it does not converge, union-find decoding guided by BP's posteriors, as
    ldpc's ``BeliefFindDecoder``: clusters grow ``bits_per_step`` (0: the block length) most
    likely columns at a time and are solved by ``uf_method`` ``"inversion"`` (any matrix) or
    ``"peeling"`` (columns of at most two checks)."""

    _core_factory = _core.Bp

    def __init__(
        self,
        pcm: Any,
        error_rate: Optional[float] = _Default(None),
        error_channel: Any = _Default(None),
        max_iter: int = _Default(0),
        bp_method: str = _Default("minimum_sum"),
        ms_scaling_factor: float = _Default(1.0),
        schedule: str = _Default("parallel"),
        omp_thread_count: int = _Default(1),
        random_schedule_seed: int = _Default(0),
        serial_schedule_order: Optional[List[int]] = _Default(None),
        uf_method: str = "peeling",
        bits_per_step: int = 0,
        input_vector_type: str = "syndrome",
    ) -> None:
        self._made_from = ((pcm, error_rate, error_channel), dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, schedule=schedule, omp_thread_count=omp_thread_count, random_schedule_seed=random_schedule_seed, serial_schedule_order=serial_schedule_order, uf_method=uf_method, bits_per_step=bits_per_step, input_vector_type=input_vector_type))
        self._input = "syndrome"
        self._setup(pcm, error_rate, error_channel, max_iter, bp_method, ms_scaling_factor, schedule, omp_thread_count, random_schedule_seed, serial_schedule_order, _Default(False), [None], ())
        self._uf = call(_core.LdpcUf, self.m, self._cols)
        _cinit_args(self._strict, bp_method=bp_method, schedule=schedule, serial_schedule_order=serial_schedule_order)
        if not isinstance(uf_method, str):
            raise _type_error("uf_method", "str", uf_method)
        _cinit_args(self._strict, input_vector_type=input_vector_type)
        if uf_method.lower() in ("inversion", "invert", "matrix"):
            self._uf_method = "inversion"
        elif uf_method.lower() in ("peeling", "peel"):
            self._uf_method = "peeling"
            for i, col in enumerate(self._cols):
                if len(col) > 2:
                    raise ValueError(f"The 'peeling' method is only suitable for LDPC codes with point like syndromes. Each column of the PCM must have at most 2 entries. Column {i} has degree {len(col)}.")
        else:
            raise ValueError(f"Invalid UF method: {uf_method}. Must be one of 'inversion' or 'peeling'.")
        self.bits_per_step = self.n if bits_per_step == 0 else int(bits_per_step)
        self.input_vector_type = "syndrome"
        self.last_tied = self.last_stuck = False

    @_one_at_a_time
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        received = self._received(syndrome)
        if received is None:
            return self._decode_syndrome(syndrome)
        bits, bits_syndrome, dtype = received
        hard, llr, converged, iterations = call(self._x.decode, bits_syndrome)
        self._converge, self._iter, self._llr = bool(converged), int(iterations), np.asarray(llr, dtype=float)
        self._decoding = np.asarray(hard, dtype=np.uint8) ^ bits
        self.last_tied = self.last_stuck = False
        if self._converge:
            return self._decoding.astype(dtype)
        correction, solved, ties, stuck = call(self._uf.decode, [int(x) for x in bits], list(self._llr), _per_step(self.bits_per_step, self.n), self._uf_method == "peeling")
        if not solved:
            raise ValueError("no correction explains the syndrome")
        self.last_tied, self.last_stuck = ties > 0, stuck > 0
        return np.asarray(correction, dtype=dtype)

    def _decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = np.asarray(syndrome)
        if not len(syndrome) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        # BP runs on a zero syndrome too (ldpc's iter and log_prob_ratios are that run's).
        hard, llr, converged, iterations = call(self._x.decode, [int(x != 0) for x in syndrome])
        self._converge, self._iter, self._llr = bool(converged), int(iterations), np.asarray(llr, dtype=float)
        self._decoding = np.asarray(hard, dtype=np.uint8)
        self.last_tied = self.last_stuck = False
        if self._converge:
            return self._decoding.astype(syndrome.dtype)
        correction, solved, ties, stuck = call(self._uf.decode, [int(x != 0) for x in syndrome], list(self._llr), _per_step(self.bits_per_step, self.n), self._uf_method == "peeling")
        if not solved:
            raise ValueError("no correction explains the syndrome")
        self.last_tied, self.last_stuck = ties > 0, stuck > 0
        return np.asarray(correction, dtype=syndrome.dtype)

    @property
    def uf_method(self) -> str:
        return self._uf_method


def _entries(pcm: Any, value: Optional[int] = None, offset: int = 0) -> list:
    """A matrix's nonzeros as (row + offset, column, value or the entry)."""
    if hasattr(pcm, "tocoo"):
        coo = pcm.tocoo()
        return [(int(r) + offset, int(c), int(value if value is not None else v)) for r, c, v in zip(coo.row, coo.col, coo.data) if v != 0]
    a = np.asarray(pcm)
    rows, cols = np.nonzero(a)
    return [(int(r) + offset, int(c), int(value if value is not None else a[r, c])) for r, c in zip(rows, cols)]


class MbpDecoder:
    """Memory belief propagation over GF(4) (Kuo and Lai), as ldpc's ``mbp_decoder``: X, Y and
    Z errors decoded together, each qubit summing its checks' messages scaled by
    ``1/alpha_parameter`` (a number, three, or 3 x n) but sending back only the unscaled ones;
    ``beta_parameter`` weighs commuting checks' messages and ``gamma_parameter`` scales
    min-sum's (``bp_method`` 1). The code is ``Hgf4`` (entries 1 = X, 2 = Y, 3 = Z; decodings
    over GF(4)) or ``HX_CSS`` and ``HZ_CSS`` (decodings as X and Z parts; the syndrome is the
    Z checks' then the X checks', or ``sx`` and ``sz``). ``uf_decode`` falls back on union-find
    by inversion, weighted by MBP's posteriors, where MBP does not converge.

    ldpc 2.4.1 ships this interface without its C++ (on ldpc's main branch, and no longer
    buildable); here the C++ is ported as written and the interface does what it describes.
    Where ldpc's Python would fail: ``Hgf4`` alone works, ``alpha_parameter`` defaults to 1,
    ``update_alpha`` takes effect, and ``uf_decode`` returns its decodings."""

    def __init__(
        self,
        Hgf4: Any = None,
        HX_CSS: Any = None,
        HZ_CSS: Any = None,
        error_rate: Optional[float] = None,
        xyz_bias: Any = (1, 1, 1),
        error_channel: Any = None,
        max_iter: int = 0,
        alpha_parameter: Any = None,
        beta_parameter: float = 0.0,
        bp_method: Union[int, str] = 0,
        gamma_parameter: float = 1.0,
    ) -> None:
        self._made_from = ((), dict(Hgf4=Hgf4, HX_CSS=HX_CSS, HZ_CSS=HZ_CSS, error_rate=error_rate, xyz_bias=xyz_bias, error_channel=error_channel, max_iter=max_iter, alpha_parameter=alpha_parameter, beta_parameter=beta_parameter, bp_method=bp_method, gamma_parameter=gamma_parameter))
        self._uf = None
        if Hgf4 is not None:
            self.stab_count, self.qubit_count = Hgf4.shape[0], Hgf4.shape[1]
            self._gf4 = True
            entries = _entries(Hgf4)
        elif HX_CSS is not None and HZ_CSS is not None:
            if HX_CSS.shape[1] != HZ_CSS.shape[1]:
                raise ValueError("The number of columns in HX_CSS should be equal to the number of columns in HZ_CSS.")
            self._gf4 = False
            self.stab_count, self.qubit_count = HZ_CSS.shape[0] + HX_CSS.shape[0], HX_CSS.shape[1]
            entries = _entries(HZ_CSS, 3) + _entries(HX_CSS, 1, HZ_CSS.shape[0])
            self._mz = HZ_CSS.shape[0]
            mz, _, cols_z = _columns(HZ_CSS)
            mx, _, cols_x = _columns(HX_CSS)
            self._uf_specs = ((mz, cols_z), (mx, cols_x))
        else:
            raise ValueError("Please enter either the GF4 parity check matrix, or the GF2 HX and HZ CSS parity check components.")
        n = self.qubit_count
        self.max_iter = n if max_iter == 0 else int(max_iter)
        bias = np.asarray(xyz_bias, dtype=float)
        total = float(np.sum(bias))
        self._xyz_bias = bias / total if total > 0 else np.zeros(3)
        channel = np.zeros((3, n))
        if error_rate is not None and error_channel is None:
            self.error_rate = error_rate
            channel[:] = ((self._xyz_bias if total > 0 else bias) * error_rate)[:, None]
        elif error_channel is not None:
            if error_rate is not None:
                warnings.warn(f"An error channel has been provided as input. This will override the 'error_rate={error_rate}' parameter that has also been inputted.")
            for w in range(3):
                for i in range(n):
                    channel[w, i] = error_channel[w][i]
        self._channel = channel
        self._alpha = np.ones((3, n))
        self.update_alpha(alpha_parameter)
        if str(bp_method).lower() in ["prod_sum", "product_sum", "ps", "0", "prod sum"]:
            self.bp_method = 0
        elif str(bp_method).lower() in ["min_sum", "minimum_sum", "ms", "1", "minimum sum", "min sum"]:
            self.bp_method = 1
        else:
            raise ValueError(f"BP method '{bp_method}' is invalid.{_S28}Please choose from the following methods:'product_sum',{_S28}'minimum_sum' or 'product_sum'")
        self._beta, self.gamma_parameter = float(beta_parameter), float(gamma_parameter)
        self._x = call(_core.Mbp, self.stab_count, n, entries, channel.tolist(), self._alpha.tolist(), self._beta, self.gamma_parameter, self.bp_method == 1, self.max_iter)
        self._syndrome = np.zeros(self.stab_count, dtype=np.uint8)
        self._decoding = np.zeros(n, dtype=np.uint8)
        self._llr = np.zeros((3, n))
        self._converge, self._iter = False, 0
        self._lock = threading.RLock()

    def __reduce__(self) -> tuple:
        args, kwargs = self._made_from
        kwargs = dict(kwargs, alpha_parameter=self._alpha.copy())
        return (_rebuild, (type(self), args, kwargs), None)

    def update_alpha(self, alpha: Any) -> None:
        if alpha is None:
            return
        n = self.qubit_count
        if isinstance(alpha, np.ndarray):
            if alpha.size == 3:
                new = np.repeat(alpha.reshape(3, 1).astype(float), n, axis=1)
            elif alpha.size == 3 * n:
                new = np.asarray(alpha, dtype=float).reshape(3, n)
            else:
                raise ValueError(f"The 'alpha' input must be either a single double variable or a 3xN np.ndarray of doubles, where N is the qubit count. The current input has dimensions {alpha.shape}.")
        elif isinstance(alpha, (float, int)):
            new = np.full((3, n), float(alpha))
        else:
            raise ValueError(f"The 'alpha' input must be either a single double variable or a 3xN np.ndarray of doubles, where N is the qubit count. The current input has of type {type(alpha)}.")
        self._alpha = new
        if hasattr(self, "_x"):
            call(self._x.set_alpha, new.tolist())

    @_one_at_a_time
    def decode(self, syndrome: Any = None, sx: Any = None, sz: Any = None) -> Any:
        if syndrome is not None:
            syndrome = np.asarray(syndrome)
            if not len(syndrome) == self.stab_count:
                raise ValueError(f"The syndrome must have length {self.stab_count}. Not {len(syndrome)}.")
            dtype = syndrome.dtype
            values = syndrome
        elif sx is not None and sz is not None:
            sx, sz = np.asarray(sx), np.asarray(sz)
            dtype = sx.dtype
            if len(sx) + len(sz) != self.stab_count:
                raise ValueError(f"sx and sz must have {self.stab_count} entries between them. Not {len(sx) + len(sz)}.")
            values = np.concatenate([sx, sz])
        else:
            raise ValueError("Invalid syndrome input.")
        self._syndrome = np.asarray(values).astype(np.uint8)
        d, converged, iterations, llr = call(self._x.decode, [int(v) for v in values])
        self._decoding = np.asarray(d, dtype=np.uint8)
        self._converge, self._iter, self._llr = bool(converged), int(iterations), np.asarray(llr, dtype=float)
        if self._gf4:
            return self._decoding.astype(dtype)
        outx = np.isin(self._decoding, (1, 2)).astype(dtype)
        outz = np.isin(self._decoding, (2, 3)).astype(dtype)
        return outx, outz

    @_one_at_a_time
    def uf_decode(self, sx: Any = None, sz: Any = None) -> Any:
        if self._gf4:
            raise ValueError("uf_decode needs the code as HX_CSS and HZ_CSS.")
        out = self.decode(sx=sx, sz=sz)
        if self._converge:
            return out
        if self._uf is None:
            self._uf = tuple(call(_core.LdpcUf, m, cols) for m, cols in self._uf_specs)

        def weights(a: np.ndarray, b: np.ndarray) -> list:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                w = 1 / (np.exp(a) + 1) + 1 / (np.exp(b) + 1)
                return [float(np.inf) if v == 0 else float(np.log((1 - v) / v)) for v in w]

        llr = self._llr
        sx, sz = np.asarray(sx), np.asarray(sz)
        outs = []
        for uf, s, w in ((self._uf[0], sx, weights(llr[1], llr[0])), (self._uf[1], sz, weights(llr[1], llr[2]))):
            correction, solved, _, _ = call(uf.decode, [int(v != 0) for v in s], w, 1, False)
            if not solved:
                raise ValueError("no correction explains the syndrome")
            outs.append(np.asarray(correction, dtype=sx.dtype))
        return outs[0], outs[1]

    @property
    def log_prob_ratios(self) -> np.ndarray:
        return self._llr.copy()

    @property
    def decoding(self) -> np.ndarray:
        return self._decoding.astype(int)

    @property
    def syndrome(self) -> np.ndarray:
        return self._syndrome.astype(int)

    @property
    def error_channel(self) -> np.ndarray:
        return self._channel.copy()

    @property
    def alpha(self) -> np.ndarray:
        return self._alpha.copy()

    @property
    def alpha_parameter(self) -> list:
        return self._alpha.tolist()

    @property
    def beta_parameter(self) -> float:
        return self._beta

    @property
    def iterations(self) -> int:
        return self._iter

    @property
    def iter(self) -> int:
        return self._iter

    @property
    def converge(self) -> bool:
        return self._converge


mbp_decoder = MbpDecoder


class SoftInfoBpOsdDecoder(SoftInfoBpDecoder):
    """Soft-information BP, and where it does not converge, OSD on the syndrome its soft values
    end at (a check is fired where its soft value is at most 0), weighted by BP's posteriors:
    what ldpc's ``SoftInfoBpOsdDecoder`` describes (in ldpc 2.4.1 the class exists but its
    body is commented out, so it cannot decode). ``osd_method`` and ``osd_order`` as
    ``BpOsdDecoder``'s."""

    def __init__(
        self,
        pcm: Any,
        error_rate: Optional[float] = _Default(None),
        error_channel: Any = _Default(None),
        max_iter: int = _Default(0),
        bp_method: str = _Default("minimum_sum"),
        ms_scaling_factor: float = _Default(1.0),
        schedule: str = _Default("parallel"),
        omp_thread_count: int = _Default(1),
        random_serial_schedule: Any = _Default(False),
        serial_schedule_order: Optional[List[int]] = _Default(None),
        osd_method: Union[str, int, float] = 0,
        osd_order: int = 0,
        cutoff: float = np.inf,
        sigma: float = 2.0,
    ) -> None:
        super().__init__(pcm, error_rate, error_channel, max_iter, bp_method, ms_scaling_factor, cutoff=cutoff, sigma=sigma, schedule=schedule, omp_thread_count=omp_thread_count, random_serial_schedule=random_serial_schedule, serial_schedule_order=serial_schedule_order)
        self._made_from = ((pcm, error_rate, error_channel), dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, schedule=schedule, omp_thread_count=omp_thread_count, random_serial_schedule=random_serial_schedule, serial_schedule_order=serial_schedule_order, osd_method=osd_method, osd_order=osd_order, cutoff=cutoff, sigma=sigma))
        name = _osd_name(osd_method)
        if name is None:
            raise ValueError(f"ERROR: OSD method '{osd_method}' invalid. Please choose from the following methods:{_S16}'OSD_0', 'OSD_E' or 'OSD_CS'.")
        order = int(osd_order)
        BpOsdDecoder._check_order(name, order)
        self._osd, self._order = name, order
        core_name = "osd_0" if name in ("osd_0", "osd_off") else name
        self._osdx = call(_core.BpOsd, self.m, self._cols, list(self._channel), 1, "product_sum", 1.0, core_name, order)
        self._osd0 = np.zeros(self.n, dtype=np.uint8)
        self._osdw = np.zeros(self.n, dtype=np.uint8)

    @_one_at_a_time
    def decode(self, soft_info_syndrome: np.ndarray) -> np.ndarray:
        if not len(soft_info_syndrome) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(soft_info_syndrome)}.")
        hard = super().decode(soft_info_syndrome)
        if self._converge:
            return hard.astype(np.uint8)
        syndrome = [int(v <= 0) for v in self._soft]
        call(self._osdx.set_priors, list(self._channel))
        osdw, osd0 = call(self._osdx.osd_on, syndrome, list(self._llr))
        self._osdw, self._osd0 = np.asarray(osdw, dtype=np.uint8), np.asarray(osd0, dtype=np.uint8)
        return self._osdw.copy()

    @property
    def osd_method(self) -> Optional[str]:
        return {"osd_0": "OSD_0", "osd_e": "OSD_E", "osd_cs": "OSD_CS", "osd_off": "OSD_OFF"}.get(self._osd)

    @property
    def osd_order(self) -> int:
        return self._order

    @property
    def osd0_decoding(self) -> np.ndarray:
        return self._osd0.astype(int)

    @property
    def osdw_decoding(self) -> np.ndarray:
        return self._osdw.astype(int)
