"""ldpc 2.4.1's BP decoders on a check matrix (``BpDecoder``, ``SoftInfoBpDecoder``,
``BpOsdDecoder``, ``BpLsdDecoder``): their arguments, defaults, properties and behaviour, with
every BP schedule (parallel, serial, serial-relative, random serial) and input type (a syndrome
or a received vector), bit for bit.

A few things are looser than ldpc: positional ``error_rate``/``error_channel`` and integer
values are taken where ldpc's typed signatures refuse them, and ``BpOsdDecoder.decoding``
returns the OSD decoding (in ldpc it raises ``AttributeError``)."""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Union

import numpy as np

from . import _core
from ._decoders import _bp_method, _columns, _lsd_method, _rebuild
from ._util import call

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


def _probability(p: Any) -> float:
    # ldpc takes any number; one outside [0, 1] only makes its posteriors NaN.
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"an error probability is between 0 and 1, not {p}")
    return p


def _base(name: str, value: Any) -> Any:
    return _BASE_DEFAULTS[name] if isinstance(value, _Default) else value


def _declared(value: Any) -> Any:
    return value.value if isinstance(value, _Default) else value


class BpDecoderBase:
    """What ldpc's BP decoders share: the matrix, the channel, the BP options and the last
    run's results."""

    _core_factory: Any = None
    _defaults: dict = {}

    def _setup(self, pcm: Any, error_rate: Any, error_channel: Any, max_iter: Any, bp_method: Any, ms_scaling_factor: Any, schedule: Any, omp_thread_count: Any, random_schedule_seed: Any, serial_schedule_order: Any, random_serial_schedule: Any, channel_probs: Any, core_args: tuple) -> None:
        error_rate, error_channel, max_iter = _base("error_rate", error_rate), _base("error_channel", error_channel), _base("max_iter", max_iter)
        bp_method, ms_scaling_factor, schedule = _base("bp_method", bp_method), _base("ms_scaling_factor", ms_scaling_factor), _base("schedule", schedule)
        omp_thread_count, random_schedule_seed = _base("omp_thread_count", omp_thread_count), _base("random_schedule_seed", random_schedule_seed)
        serial_schedule_order, random_serial_schedule = _base("serial_schedule_order", serial_schedule_order), _base("random_serial_schedule", random_serial_schedule)
        if not _is_matrix(pcm) and not isinstance(pcm, (list, tuple)):
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")
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
            raise ValueError(f"Please specify the error channel. Either: 1) error_rate: float or 2) error_channel: list of floats of length equal to the block length of the code {self.n}.")

    # -- The channel ----------------------------------------------------------------------

    @property
    def error_rate(self) -> np.ndarray:
        return self._channel.copy()

    @error_rate.setter
    def error_rate(self, value: Optional[float]) -> None:
        if value is not None:
            if not isinstance(value, (float, int, np.floating, np.integer)) or isinstance(value, bool):
                raise ValueError("The `error_rate` parameter must be specified as a single float value.")
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
            self._channel = np.array([_probability(p) for p in np.asarray(value, dtype=float).ravel()])
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
        t = str(input_type).lower()
        if t in ("auto", "a", "2"):
            if self.m == self.n:
                raise ValueError("Please specify the input vector type. Either: 1) input_vector_type: 'syndrome' or 2) input_vector_type: 'received_vector'.")
            self._input = "auto"
        elif t in ("syndrome", "s", "0"):
            self._input = "syndrome"
        elif t in ("received_vector", "r", "1"):
            self._input = "received_vector"
        else:
            raise ValueError(f"The input vector type '{input_type}' is invalid. Please choose from the following methods: 'input_vector_type=syndrome', 'input_vector_type=received_vector'")

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
        if not isinstance(value, (int, np.integer)):
            raise ValueError("max_iter input parameter is invalid. This must be specified as a positive int.")
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
            raise ValueError(f"BP method '{value}' is invalid. Please choose from the following methods: 'product_sum', 'minimum_sum'")
        call(self._x.set_method, self._method, float(self._scale))

    @property
    def schedule(self) -> str:
        return self._schedule

    @schedule.setter
    def schedule(self, value: Union[str, int]) -> None:
        v = _SCHEDULES.get(str(value).lower())
        if v is None:
            raise ValueError(f"The BP schedule method '{value}' is invalid. Please choose from the following methods: 'schedule=parallel', 'schedule=serial', 'schedule=serial_relative'")
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
        if not isinstance(value, (float, int, np.floating, np.integer)):
            raise TypeError("The ms_scaling factor must be specified as a float")
        self._scale = value
        call(self._x.set_method, self._method, float(value))

    @property
    def omp_thread_count(self) -> int:
        if self._omp != 1:
            warnings.warn("The OpenMP functionality is not yet implemented")
        return self._omp

    @omp_thread_count.setter
    def omp_thread_count(self, value: int) -> None:
        if not isinstance(value, int) or value < 1:
            raise TypeError("The omp_thread_count must be specified as a positive integer.")
        self._omp = value
        if value != 1:
            warnings.warn("The OpenMP functionality is not yet implemented")

    @property
    def random_schedule_seed(self) -> int:
        return self._seed

    @random_schedule_seed.setter
    def random_schedule_seed(self, value: int) -> None:
        if not isinstance(value, (int, np.integer)) or value < -2:
            raise ValueError("The value of random_schedule_seed must be a positive integer. Set as -1 to disable to the random schedule. Set as 0 to use the system clock.")
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

    def __reduce__(self) -> tuple:
        # Rebuilt from its matrix and options; the settings and last results come along.
        state = {k: v for k, v in self.__dict__.items() if k not in ("m", "n", "_cols") and not type(v).__module__.startswith("stabilizer_qec._core")}
        args, kwargs = self._made_from
        return (_rebuild, (type(self), args, kwargs), state)

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        call(self._x.set_priors, list(self._channel))
        self._x.set_max_iter(self._max_iter)
        call(self._x.set_method, self._method, float(self._scale))
        call(self._x.set_schedule, self._schedule)
        self._x.set_random(self._random)


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
        **kwargs: Any,
    ) -> None:
        for key in kwargs:
            if key not in ("channel_probs",):
                raise ValueError(f"Unknown parameter '{key}' passed to the BpDecoder constructor.")
        self._made_from = ((pcm, error_rate, error_channel), dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, schedule=schedule, omp_thread_count=omp_thread_count, random_schedule_seed=random_schedule_seed, serial_schedule_order=serial_schedule_order, input_vector_type=input_vector_type, random_serial_schedule=random_serial_schedule, **kwargs))
        self._input = "syndrome"
        self._setup(pcm, error_rate, error_channel, max_iter, bp_method, ms_scaling_factor, schedule, omp_thread_count, random_schedule_seed, serial_schedule_order, random_serial_schedule, kwargs.get("channel_probs", [None]), ())
        self.input_vector_type = input_vector_type

    def decode(self, input_vector: np.ndarray) -> np.ndarray:
        """The hard decision: for a syndrome, the error; for a received vector, the codeword."""
        input_vector = np.asarray(input_vector)
        self._check_length(input_vector)
        dtype = input_vector.dtype
        received = self._input == "received_vector" or (self._input == "auto" and len(input_vector) == self.n)
        if not input_vector.any():
            self._converge = True
            return np.zeros(self.n, dtype=dtype)
        if received:
            vector = (input_vector != 0).astype(np.uint8)
            hard, llr, converged, iterations = call(self._x.decode, self._syndrome_of(vector))
            self._decoding = np.asarray(hard, dtype=np.uint8) ^ vector
        else:
            hard, llr, converged, iterations = call(self._x.decode, [int(x != 0) for x in input_vector])
            self._decoding = np.asarray(hard, dtype=np.uint8)
        self._converge, self._iter, self._llr = bool(converged), int(iterations), np.asarray(llr, dtype=float)
        return self._decoding.astype(dtype)

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
        for key in kwargs:
            if key not in ("channel_probs",):
                raise ValueError(f"Unknown parameter '{key}' passed to the BpDecoder constructor.")
        self._made_from = ((pcm, error_rate, error_channel), dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, schedule=schedule, omp_thread_count=omp_thread_count, random_schedule_seed=random_schedule_seed, serial_schedule_order=serial_schedule_order, osd_method=osd_method, osd_order=osd_order, **kwargs))
        name = _osd_name(osd_method)
        if name is None:
            raise ValueError(f"ERROR: OSD method '{osd_method}' invalid. Please choose from the following methods: 'OSD_0', 'OSD_E' or 'OSD_CS'.")
        order = 0 if name == "osd_0" else int(osd_order)
        self._check_order(name, order)
        self._input = "syndrome"
        core_name = "osd_0" if name in ("osd_0", "osd_off") else name
        self._setup(pcm, error_rate, error_channel, max_iter, bp_method, ms_scaling_factor, schedule, omp_thread_count, random_schedule_seed, serial_schedule_order, False, kwargs.get("channel_probs", [None]), (core_name, order))
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

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
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

    @property
    def osd_method(self) -> Optional[str]:
        return {"osd_0": "OSD_0", "osd_e": "OSD_E", "osd_cs": "OSD_CS", "osd_off": "OSD_OFF"}.get(self._osd)

    @osd_method.setter
    def osd_method(self, method: Union[str, int, float]) -> None:
        name = _osd_name(method)
        if name is None:
            raise ValueError(f"ERROR: OSD method '{method}' invalid. Please choose from the following methods: 'OSD_0', 'OSD_E' or 'OSD_CS'.")
        self._osd = name
        if name == "osd_0":
            self._order = 0
            self._x.set_osd_order(0)

    @property
    def osd_order(self) -> int:
        return self._order

    @osd_order.setter
    def osd_order(self, order: int) -> None:
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


class BpLsdDecoder(BpDecoderBase):
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
        method, order = _lsd_method(lsd_method, lsd_order)
        bits = pcm.shape[1] if bits_per_step == 0 else int(bits_per_step)
        self._input = "syndrome"
        self._setup(pcm, error_rate, error_channel, max_iter, bp_method, ms_scaling_factor, schedule, omp_thread_count, random_schedule_seed, serial_schedule_order, False, kwargs.get("channel_probs", [None]), (method, order, bits, bool(always_run_lsd)))
        self._lsd_method, self._lsd_order = method, order
        self.always_run_lsd = bool(always_run_lsd)
        self.bits_per_step = bits
        self._bp_decoding = np.zeros(self.n, dtype=np.uint8)
        self.last_tied = False

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
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
        return np.asarray(correction, dtype=syndrome.dtype)

    def _cluster_bits(self, syndrome: Any) -> list:
        """Each final cluster's (id, bits in robin-set order): for checking against ldpc's
        statistics."""
        return call(self._x.decode, [int(x != 0) for x in np.asarray(syndrome)], True)[5]

    @property
    def bp_decoding(self) -> np.ndarray:
        return self._bp_decoding.astype(int)

    @property
    def lsd_method(self) -> Optional[str]:
        return {"osd_0": "lSD_0", "osd_e": "LSD_E", "osd_cs": "LSD_CS"}.get(self._lsd_method)

    @property
    def lsd_order(self) -> int:
        return self._lsd_order


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


class LsdDecoder:
    """Localised statistics decoding alone, given each bit's weight (lower: likelier), as
    ldpc's ``LsdDecoder``: clusters grown from the syndrome ``bits_per_step`` bits at a time,
    lightest first, each solved by OSD of ``lsd_order`` (``osd_method``/``osd_order`` accepted for
    ``lsd_method``/``lsd_order``)."""

    def __init__(self, pcm: Any, bits_per_step: int = 1, lsd_order: int = 0, lsd_method: Union[str, int] = 0, **kwargs: Any) -> None:
        self._made_from = ((pcm,), dict(bits_per_step=bits_per_step, lsd_order=lsd_order, lsd_method=lsd_method, **kwargs))
        lsd_method = kwargs.get("osd_method", lsd_method)
        lsd_order = kwargs.get("osd_order", lsd_order)
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
        if not _is_matrix(pcm) and not isinstance(pcm, (list, tuple)):
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")
        self.m, self.n, cols = _columns(pcm)
        method, order = _lsd_method(lsd_method, lsd_order)
        self.bits_per_step = self.n if bits_per_step == 0 else int(bits_per_step)
        self._lsd_method, self._lsd_order = method, order
        self._x = call(_core.BpLsd, self.m, cols, [0.5] * self.n, 1, "product_sum", 1.0, method, order, self.bits_per_step, True)
        self._decoding = np.zeros(self.n, dtype=np.uint8)
        self.last_tied = False

    def __reduce__(self) -> tuple:
        args, kwargs = self._made_from
        return (_rebuild, (type(self), args, kwargs), {"_decoding": self._decoding, "last_tied": self.last_tied})

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

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

    @property
    def lsd_method(self) -> Optional[str]:
        return {"osd_0": "lSD_0", "osd_e": "LSD_E", "osd_cs": "LSD_CS"}.get(self._lsd_method)

    @property
    def lsd_order(self) -> int:
        return self._lsd_order


def _per_step(bits_per_step: int, n: int) -> int:
    bits_per_step = int(bits_per_step)
    return n if bits_per_step == 0 else (bits_per_step if bits_per_step > 0 else 1 << 62)


class UnionFindDecoder:
    """Union-find decoding of any check matrix, as ldpc's ``UnionFindDecoder``: a cluster grows
    around each fired check until it can explain its checks, then each is solved alone, by
    elimination (``uf_method`` true: any matrix) or by peeling (false: columns of at most two
    checks, a column with one check touching the boundary). Given ``llrs``, growth takes the
    ``bits_per_step`` lowest first (0: all). ``uf_method`` may also be a name, ``"inversion"`` or
    ``"peeling"``. (ldpc's own takes any non-empty string as inversion, ``"peeling"`` too, and its
    declared default ``False`` is refused.)"""

    def __init__(self, pcm: Any, uf_method: Union[bool, str] = False) -> None:
        if not _is_matrix(pcm):
            raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or spmatrix object, not {type(pcm)}")
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

    def __reduce__(self) -> tuple:
        args, kwargs = self._made_from
        return (_rebuild, (type(self), args, kwargs), {"_decoding": self._decoding, "last_tied": self.last_tied, "last_stuck": self.last_stuck, "bits_per_step": self.bits_per_step})

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

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
        if not isinstance(uf_method, str):
            raise TypeError(f"Argument 'uf_method' has incorrect type (expected str, got {type(uf_method).__name__})")
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

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = np.asarray(syndrome)
        if not len(syndrome) == self.m:
            raise ValueError(f"The syndrome must have length {self.m}. Not {len(syndrome)}.")
        if not syndrome.any():
            self._converge = True
            return np.zeros(self.n, dtype=syndrome.dtype)
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
