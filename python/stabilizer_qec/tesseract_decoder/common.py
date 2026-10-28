# Ported from Tesseract (quantumlib/tesseract-decoder, src/common.cc and its bindings; Copyright
# 2025 Google LLC, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec:
# Python on this package's detector error models.
"""Tesseract's errors, symptoms and model transforms (``tesseract_decoder.common``)."""

from __future__ import annotations

import contextlib
import math
import os
import platform
import re
import sys
from typing import Any, Iterable, Iterator, List, Sequence, Tuple

import stabilizer_qec as _sq

__all__ = ["Error", "Symptom", "dem_from_counts", "merge_indistinguishable_errors", "remove_zero_probability_errors"]

_MAX = 2**64 - 1


def _is_stims(obj: Any) -> bool:
    """Whether ``obj`` is one of Stim's own objects."""
    return type(obj).__module__.split(".")[0] == "stim"


def _dem(dem: Any) -> "_sq.DetectorErrorModel":
    """A model as this package's. As Tesseract's binding does, anything else is read from its
    text (so a Stim model is read exactly, and a name that is no instruction is refused with
    Stim's error)."""
    if isinstance(dem, _sq.DetectorErrorModel):
        return dem
    try:
        return _sq.DetectorErrorModel(dem if _is_stims(dem) else str(dem))
    except ValueError as e:
        name = re.fullmatch(r"line \d+: unsupported instruction '(.*)'", str(e))
        if name is None:
            raise
        raise IndexError(f"Unrecognized instruction name: {name.group(1).lower()}") from None


def _as_given(out: "_sq.DetectorErrorModel", stims: Any) -> Any:
    """``out`` as Stim's model when ``stims`` (or the model it was made from) is Stim's, and as
    this package's otherwise: each function returns the kind of model it was given."""
    if stims is True or (stims is not False and _is_stims(stims)):
        import stim

        return stim.DetectorErrorModel(str(out))
    return out


# How many ``ostream_redirect`` blocks (or bindings that redirect for their own call) are
# sending Tesseract's C++ standard output to Python's ``sys.stdout``.
_redirects = [0]


@contextlib.contextmanager
def _redirected() -> Iterator[None]:
    """Tesseract's ``py::scoped_ostream_redirect`` for one call: its output goes to Python's
    ``sys.stdout``."""
    _redirects[0] += 1
    try:
        yield
    finally:
        _redirects[0] -= 1


def _cout(lines: Iterable[Any]) -> None:
    """Lines written as Tesseract writes them to ``std::cout``: to Python's ``sys.stdout``
    inside ``ostream_redirect`` (or a call that redirects its own output), and otherwise
    straight to the process's standard output (file descriptor 1), past any Python
    redirection, after what Python has buffered there."""
    text = "".join(f"{line}\n" for line in lines)
    if not text:
        return
    if _redirects[0]:
        sys.stdout.write(text)
        return
    try:
        sys.stdout.flush()
    except (AttributeError, OSError, ValueError):
        pass
    data = text.encode()
    try:
        while data:
            data = data[os.write(1, data) :]
    except OSError:
        pass


def _flat(dem: Any) -> "_sq.DetectorErrorModel":
    d = _dem(dem)
    if any(inst.type in ("repeat", "shift_detectors") for inst in d):
        return d.flattened()
    return d


def _vec(v: Sequence[int]) -> str:
    return "[" + " ".join(str(x) for x in v) + "]"


class Symptom:
    """The detectors and observables an error flips. (As Tesseract's binding does, reading
    ``detectors`` or ``observables`` gives a copy; assign the attribute to change it.)"""

    __hash__ = None  # type: ignore[assignment]
    __slots__ = ("_detectors", "_observables")

    def __init__(self, detectors: Sequence[int] = [], observables: Sequence[int] = []) -> None:
        self.detectors = detectors  # type: ignore[assignment]
        self.observables = observables  # type: ignore[assignment]

    @property
    def detectors(self) -> List[int]:
        return list(self._detectors)

    @detectors.setter
    def detectors(self, value: Sequence[int]) -> None:
        self._detectors = tuple(_int(d, "detectors") for d in value)

    @property
    def observables(self) -> List[int]:
        return list(self._observables)

    @observables.setter
    def observables(self, value: Sequence[int]) -> None:
        self._observables = tuple(_int(o, "observables") for o in value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Symptom) and self._detectors == other._detectors and self._observables == other._observables

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __str__(self) -> str:
        return f"Symptom{{detectors={_vec(self.detectors)}, observables={_vec(self.observables)}}}"

    def _key(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return self._detectors, self._observables

    def as_dem_instruction_targets(self) -> List[Any]:
        """The symptom as model targets (this package's ``DemTarget``s; Tesseract's are
        Stim's)."""
        return [_sq.target_relative_detector_id(d) for d in self._detectors] + [_sq.target_logical_observable_id(o) for o in self._observables]


def _int(value: Any, name: str, low: int = -(2**31), high: int = 2**31) -> int:
    """A C++ integer argument (``int`` by default), refused as pybind11 refuses one: a
    ``TypeError`` for a float, a non-number or a value out of range."""
    if isinstance(value, float) or not hasattr(type(value), "__index__"):
        raise TypeError(f"{name}: incompatible function arguments; expected an integer, got {value!r}.")
    v = value.__index__()
    if not low <= v < high:
        raise TypeError(f"{name}: incompatible function arguments; {v} is out of range for this argument.")
    return v


class Error:
    """An error: its symptom and its cost, ``−ln(p/(1 − p))``."""

    __slots__ = ("likelihood_cost", "symptom")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not args and not kwargs:
            self.likelihood_cost = 0.0
            self.symptom = Symptom()
            return
        if len(args) + len(kwargs) == 1:
            instruction = args[0] if args else kwargs["error"]
            self._from_instruction(instruction)
            return
        names = ("likelihood_cost", "detectors", "observables")
        values = dict(zip(names, args), **kwargs)
        self.likelihood_cost = float(values["likelihood_cost"])
        self.symptom = Symptom(values["detectors"], values["observables"])

    def _from_instruction(self, instruction: Any) -> None:
        if getattr(instruction, "type", None) != "error":
            raise ValueError(f"Error must be loaded from an error dem instruction, but received: {instruction}")
        probability = float(instruction.args_copy()[0])
        if probability < 0 or probability > 1:
            raise ValueError(f"Probability must be between 0 and 1, but received: {probability:f}")
        detectors: set = set()
        observables: set = set()
        for t in instruction.targets_copy():
            if t.is_logical_observable_id():
                observables ^= {t.val}
            elif t.is_relative_detector_id():
                detectors ^= {t.val}
        # −ln(p/(1 − p)) in IEEE arithmetic: ∞ at p = 0, −∞ at p = 1.
        self.likelihood_cost = -1 * _log(probability / (1 - probability)) if probability != 1 else -math.inf
        self.symptom = Symptom(sorted(detectors), sorted(observables))

    def __str__(self) -> str:
        return f"Error{{cost={self.likelihood_cost:.6f}, symptom={self.symptom}}}"

    def get_probability(self) -> float:
        return 1.0 / (1.0 + _exp(self.likelihood_cost))

    def set_with_probability(self, probability: float) -> None:
        p = float(probability)
        if p <= 0 or p >= 1:
            raise ValueError("Probability must be between 0 and 1.")
        self.likelihood_cost = -_log(p / (1.0 - p))


def _log(x: float) -> float:
    if x == 0:
        return -math.inf
    if x < 0 or math.isnan(x):
        return math.nan
    return math.log(x) if x != math.inf else math.inf


def _exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def merge_weights(a: float, b: float) -> float:
    """The cost of either of two independent errors with these costs (``common::merge_weights``)."""
    sgn = math.copysign(1, a) * math.copysign(1, b)
    signed_min = sgn * min(abs(a), abs(b))
    return signed_min + _log(1 + _exp(-abs(a + b))) - _log(1 + _exp(-abs(a - b)))


def _counts(dem: "_sq.DetectorErrorModel") -> Tuple[int, int]:
    return dem.num_detectors, dem.num_observables


def _preserve(out: "_sq.DetectorErrorModel", num_detectors: int, num_observables: int) -> None:
    if out.num_detectors < num_detectors:
        out.append("detector", [], [_sq.target_relative_detector_id(num_detectors - 1)])
    if out.num_observables < num_observables:
        out.append("logical_observable", [], [_sq.target_logical_observable_id(num_observables - 1)])


# Tesseract's NaN reaches Python as text, which Stim refuses: "nan" where the default NaN is
# positive (ARM64), "-nan" from Linux's C++ library on x86, where its sign bit is set.
_NAN_ARGUMENT = ("Not a real number: -" if sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64", "amd64")
                 else "Parens arguments for 'detector error model instruction' didn't end with a ')'.")


def _probability_argument(p: float) -> float:
    """``p`` as an error's argument, refused as Stim refuses the text Tesseract's C++ model
    turns into when it reaches Python."""
    if math.isnan(p):
        raise ValueError(_NAN_ARGUMENT)
    if not 0 <= p <= 1:
        raise ValueError(f"'error' instruction argument must be a probability (0 to 1) but got {p:f}")
    return p


def _merge(dem: Any, public: bool = False) -> Tuple["_sq.DetectorErrorModel", List[int]]:
    """Tesseract's merge of the errors of one symptom. Merging two errors of probability 0 (or
    two of probability 1, or one of each) makes a cost of NaN; Tesseract's decoder drops such
    an error with those of probability 0, which is what this does unless ``public``, where (as
    the model Tesseract returns cannot be read back) it is refused."""
    flat = _flat(dem)
    nd, no = _counts(flat)
    out = _sq.DetectorErrorModel()
    index_map: List[int] = []
    by_symptom: dict = {}
    merged: List[Error] = []
    for inst in flat:
        if inst.type == "error":
            e = Error(inst)
            if not e.symptom._detectors:
                _cout(["Warning: the circuit has errors that do not flip any detectors"])
            k = by_symptom.get(e.symptom._key())
            if k is not None:
                merged[k].likelihood_cost = merge_weights(e.likelihood_cost, merged[k].likelihood_cost)
                index_map.append(k)
            else:
                by_symptom[e.symptom._key()] = len(merged)
                index_map.append(len(merged))
                merged.append(e)
        elif inst.type in ("detector", "logical_observable"):
            out.append(inst)
        else:
            raise ValueError(f"Unrecognized instruction type: {inst}")
    for e in merged:
        p = e.get_probability()
        if math.isnan(p):
            if public:
                raise ValueError(_NAN_ARGUMENT)
            p = 0.0
        out.append("error", [p], e.symptom.as_dem_instruction_targets())
    _preserve(out, nd, no)
    return out, index_map


def _remove_zero(dem: Any) -> Tuple["_sq.DetectorErrorModel", List[int]]:
    flat = _flat(dem)
    nd, no = _counts(flat)
    out = _sq.DetectorErrorModel()
    index_map: List[int] = []
    k = 0
    for inst in flat:
        if inst.type == "error":
            if inst.args_copy()[0] > 0:
                out.append(inst)
                index_map.append(k)
                k += 1
            else:
                index_map.append(_MAX)
        elif inst.type in ("detector", "logical_observable"):
            out.append(inst)
        else:
            raise ValueError(f"Unrecognized instruction type: {inst}")
    _preserve(out, nd, no)
    return out, index_map


def merge_indistinguishable_errors(dem: Any) -> Any:
    """The model with errors of the same symptom merged into one (in first-seen order, after the
    detector and observable declarations); Stim's model when given Stim's."""
    return _as_given(_merge(dem, public=True)[0], dem)


def remove_zero_probability_errors(dem: Any) -> Any:
    """The model without its errors of probability 0; Stim's model when given Stim's."""
    return _as_given(_remove_zero(dem)[0], dem)


def _from_counts(orig_dem: Any, error_counts: Sequence[int], num_shots: int) -> Tuple["_sq.DetectorErrorModel", List[int]]:
    """Tesseract's ``common::dem_from_counts``: the model with each error's probability its
    count over ``num_shots``, and the indices (in the model) of the errors whose estimate is
    0/0, NaN, which this package's model cannot hold (each holds 0 in its place)."""
    flat = _flat(orig_dem)
    counts = [_int(c, "error_counts", 0, 2**64) for c in error_counts]
    shots = _int(num_shots, "num_shots", 0, 2**64)
    if flat.num_errors != len(counts):
        raise ValueError("Error hits array must be the same size as the number of errors in the original DEM.")
    out = _sq.DetectorErrorModel()
    nans: List[int] = []
    k = 0
    for inst in flat:
        if inst.type == "error":
            c = counts[k]
            # (Tesseract divides doubles: c/0 is ∞, 0/0 NaN.)
            p = c / shots if shots else (math.inf if c else math.nan)
            if math.isnan(p):
                nans.append(len(out))
                p = 0.0
            out.append("error", [_probability_argument(p)], inst.targets_copy(), tag=inst.tag)
            k += 1
        else:
            out.append(inst)
    return out, nans


def dem_from_counts(orig_dem: Any, error_counts: Sequence[int], num_shots: int) -> Any:
    """The model with each error's probability estimated as its count over ``num_shots``;
    Stim's model when given Stim's."""
    out, nans = _from_counts(orig_dem, error_counts, num_shots)
    if nans:
        # (Tesseract's model holds the NaN, but Stim refuses it as it reaches Python.)
        raise ValueError(_NAN_ARGUMENT)
    return _as_given(out, orig_dem)


def _nan_text() -> str:
    """How Tesseract's C++ stream (``out << model``) prints the NaN that 0.0/0.0 makes. Its
    sign bit is set on x86, so Linux's C library prints "-nan" and Windows' "-nan(ind)"; it
    is clear on ARM64, and macOS's C library prints no sign for a NaN in any case ("nan")."""
    x86 = platform.machine().lower() in ("x86_64", "amd64")
    if x86 and sys.platform.startswith("linux"):
        return "-nan"
    if x86 and sys.platform == "win32":
        return "-nan(ind)"
    return "nan"


def _dem_from_counts_text(orig_dem: Any, error_counts: Sequence[int], num_shots: int) -> str:
    """The text Tesseract's program writes for ``common::dem_from_counts`` (its ``--dem-out``),
    without a final newline: an estimate of 0/0 (no shot to count from) is NaN, which Stim's
    ``validate`` lets through and its stream prints (``error(nan) D0``)."""
    out, nans = _from_counts(orig_dem, error_counts, num_shots)
    if not nans:
        return str(out)
    # (The model is flat: one line per instruction. A tag's own ']' is escaped, so the first
    # ']' closes it, and the argument follows.)
    lines = [str(inst) for inst in out]
    nan = _nan_text()
    for k in nans:
        line = lines[k]
        start = line.index("]") + 1 if line.startswith("error[") else len("error")
        assert line.startswith("(0)", start), line
        lines[k] = f"{line[:start]}({nan}){line[start + len('(0)'):]}"
    return "\n".join(lines)


def _retained_maps(dem: Any, merge_errors: bool) -> Tuple["_sq.DetectorErrorModel", List[int], List[int]]:
    """Tesseract's preparation of a decoder's model: flattened, merged (if asked), errors of
    probability 0 removed, and the maps from the flattened model's errors to the retained ones
    and back (``2**64 − 1`` for none)."""
    flat = _flat(dem)
    dem_map = list(range(flat.num_errors))
    if merge_errors:
        flat, merge_map = _merge(flat)
        dem_map = [merge_map[i] if i != _MAX else _MAX for i in dem_map]
    flat, nonzero_map = _remove_zero(flat)
    dem_map = [nonzero_map[i] if i != _MAX else _MAX for i in dem_map]
    inverse = [_MAX] * flat.num_errors
    for i, m in enumerate(dem_map):
        if m != _MAX and inverse[m] == _MAX:
            inverse[m] = i
    return flat, dem_map, inverse


def _iter_errors(dem: Any) -> Iterable[Error]:
    for inst in _dem(dem):
        if inst.type == "error" and inst.args_copy()[0] > 0:
            yield Error(inst)
