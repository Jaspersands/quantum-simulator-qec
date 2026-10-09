"""Stim's detector-error-model objects: ``DemTarget`` and its ``target_*`` helpers,
``DemInstruction``, ``DemRepeatBlock``, and the list view of a ``DetectorErrorModel``.

A model's text (which this package writes exactly as Stim does) is parsed here into these
objects; anything built from them goes back through the engine as text.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, List, Optional, Tuple, Union

_SEPARATOR = -1


class DemTarget:
    """A target of a model instruction, as ``stim.DemTarget``: a detector (``D5``), an
    observable (``L2``), or the separator ``^`` between the parts of a decomposed error."""

    __slots__ = ("_kind", "_v")

    def __init__(self, arg: Any, /) -> None:
        if isinstance(arg, DemTarget):
            self._kind, self._v = arg._kind, arg._v
            return
        if type(arg).__name__ == "DemTarget":  # a stim.DemTarget
            arg = str(arg)
        if isinstance(arg, str):
            if arg == "^":
                self._kind, self._v = "^", 0
                return
            m = re.fullmatch(r"([DL])(\d+)", arg)
            if m:
                self._kind, self._v = m.group(1), int(m.group(2))
                return
        raise ValueError(f"Failed to parse as a stim.DemTarget: '{arg}'")

    @staticmethod
    def _make(kind: str, v: int) -> "DemTarget":
        out = DemTarget.__new__(DemTarget)
        out._kind, out._v = kind, int(v)
        return out

    @staticmethod
    def relative_detector_id(index: int) -> "DemTarget":
        return target_relative_detector_id(index)

    @staticmethod
    def logical_observable_id(index: int) -> "DemTarget":
        return target_logical_observable_id(index)

    @staticmethod
    def separator() -> "DemTarget":
        return target_separator()

    @property
    def val(self) -> int:
        """The detector or observable index (the separator's is a sentinel)."""
        if self._kind == "^":
            return 0xFFFFFFFFFFFFFFFF
        return self._v

    def is_relative_detector_id(self) -> bool:
        return self._kind == "D"

    def is_logical_observable_id(self) -> bool:
        return self._kind == "L"

    def is_separator(self) -> bool:
        return self._kind == "^"

    def __str__(self) -> str:
        return "^" if self._kind == "^" else f"{self._kind}{self._v}"

    def __repr__(self) -> str:
        if self._kind == "^":
            return "stabilizer_qec.target_separator()"
        return f"stabilizer_qec.DemTarget('{self}')"

    def _instruction_repr(self) -> str:
        if self._kind == "D":
            return f"stabilizer_qec.target_relative_detector_id({self._v})"
        if self._kind == "L":
            return f"stabilizer_qec.target_logical_observable_id({self._v})"
        return "stabilizer_qec.target_separator()"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DemTarget):
            return (self._kind, self._v) == (other._kind, other._v)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def __hash__(self) -> int:
        return hash(("DemTarget", self._kind, self._v))


def target_relative_detector_id(index: int) -> DemTarget:
    """``D<index>``."""
    if int(index) < 0 or int(index) >= 1 << 62:
        raise ValueError("detector index out of range")
    return DemTarget._make("D", int(index))


def target_logical_observable_id(index: int) -> DemTarget:
    """``L<index>``."""
    if int(index) < 0 or int(index) >= 1 << 32:
        raise ValueError("observable index out of range")
    return DemTarget._make("L", int(index))


def target_separator() -> DemTarget:
    """``^``, between the parts of a decomposed error."""
    return DemTarget._make("^", 0)


def _fmt(v: float) -> str:
    v = float(v)
    if v.is_integer() and abs(v) < 9.2e18:
        return str(int(v))
    return format(v, "g")


def _exact(v: float) -> str:
    v = float(v)
    if v.is_integer() and abs(v) < 9.2e18:
        return str(int(v))
    return repr(v)


def _escape_tag(tag: str) -> str:
    return tag.replace("\\", "\\B").replace("\n", "\\n").replace("\r", "\\r").replace("]", "\\C")


def _unescape_tag(tag: str) -> str:
    out, it = [], iter(tag)
    for c in it:
        if c == "\\":
            n = next(it, "")
            out.append({"n": "\n", "r": "\r", "B": "\\", "C": "]"}.get(n, "\\" + n))
        else:
            out.append(c)
    return "".join(out)


_TYPES = ("error", "detector", "logical_observable", "shift_detectors")


class DemInstruction:
    """One instruction of a detector error model, as ``stim.DemInstruction``."""

    __slots__ = ("_type", "_args", "_targets", "_tag")

    def __init__(self, type: str, args: Optional[Iterable[float]] = None, targets: Optional[Iterable[Any]] = None, *, tag: str = "") -> None:
        if args is None and targets is None and not tag and isinstance(type, str) and re.search(r"[\s(\[#]", type.strip()):
            items = parse_items(type)
            if len(items) != 1 or not isinstance(items[0], DemInstruction):
                raise ValueError(f"Expected exactly one instruction but got {type!r}")
            it = items[0]
            self._type, self._args, self._targets, self._tag = it._type, it._args, it._targets, it._tag
            return
        t = str(type).lower()
        if t not in _TYPES:
            raise ValueError(f"Unknown instruction type: '{type}'")
        self._type = t
        self._args = [float(a) for a in (args or [])]
        self._targets = [_dem_target(x, t) for x in (targets or [])]
        self._tag = str(tag)
        from ._circuit import DetectorErrorModel

        DetectorErrorModel(self._exact_text())  # validates, as Stim does on construction

    @staticmethod
    def _raw(t: str, args: List[float], targets: List[Any], tag: str) -> "DemInstruction":
        out = DemInstruction.__new__(DemInstruction)
        out._type, out._args, out._targets, out._tag = t, args, targets, tag
        return out

    @property
    def type(self) -> str:
        return self._type

    @property
    def tag(self) -> str:
        return self._tag

    def args_copy(self) -> List[float]:
        return list(self._args)

    def targets_copy(self) -> List[Any]:
        return list(self._targets)

    def target_groups(self) -> List[List[DemTarget]]:
        """The targets split at separators (``^``)."""
        groups: List[List[DemTarget]] = [[]]
        for t in self._targets:
            if isinstance(t, DemTarget) and t.is_separator():
                groups.append([])
            else:
                groups[-1].append(t if isinstance(t, DemTarget) else DemTarget._make("D", t))
        return groups

    def _head(self, exact: bool) -> str:
        s = self._type
        if self._tag:
            s += f"[{_escape_tag(self._tag)}]"
        if self._args:
            s += "(" + ", ".join((_exact if exact else _fmt)(a) for a in self._args) + ")"
        return s

    def _exact_text(self) -> str:
        return " ".join([self._head(True)] + [str(t) for t in self._targets])

    def __str__(self) -> str:
        return " ".join([self._head(False)] + [str(t) for t in self._targets])

    def __repr__(self) -> str:
        targets = ", ".join(t._instruction_repr() if isinstance(t, DemTarget) else str(t) for t in self._targets)
        args = ", ".join(_fmt(a) for a in self._args)
        tag = f", tag={self._tag!r}" if self._tag else ""
        return f"stabilizer_qec.DemInstruction({self._type!r}, [{args}], [{targets}]{tag})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DemInstruction):
            return NotImplemented
        return (self._type, self._args, self._targets, self._tag) == (other._type, other._args, other._targets, other._tag)

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    __hash__ = None  # type: ignore[assignment]


class DemRepeatBlock:
    """A ``repeat`` block of a detector error model, as ``stim.DemRepeatBlock``."""

    __slots__ = ("_count", "_body")

    def __init__(self, repeat_count: int, block: Any) -> None:
        from ._circuit import DetectorErrorModel

        if int(repeat_count) <= 0:
            raise ValueError("Can't repeat 0 times.")
        self._count = int(repeat_count)
        self._body = DetectorErrorModel(block).copy()

    @property
    def repeat_count(self) -> int:
        return self._count

    @property
    def type(self) -> str:
        return "repeat"

    def body_copy(self) -> Any:
        return self._body.copy()

    def _exact_text(self) -> str:
        return f"repeat {self._count} {{\n{_exact_text_of(self._body)}\n}}"

    def __repr__(self) -> str:
        return f"stabilizer_qec.DemRepeatBlock({self._count}, {self._body!r})"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DemRepeatBlock):
            return NotImplemented
        return self._count == other._count and self._body == other._body

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    __hash__ = None  # type: ignore[assignment]


def _dem_target(x: Any, t: str) -> Any:
    if t == "shift_detectors":
        if isinstance(x, DemTarget):
            raise ValueError("shift_detectors takes integer targets")
        return int(x)
    if isinstance(x, DemTarget):
        return x
    if type(x).__name__ == "DemTarget":
        return DemTarget(str(x))
    if isinstance(x, str):
        return DemTarget(x)
    raise ValueError(f"Expected a stim.DemTarget, not {x!r}")


_LINE = re.compile(r"^([A-Za-z_]+)(\[(?:[^\]\\]|\\.)*\])?(?:\(([^)]*)\))?(.*)$")


def parse_items(text: str) -> List[Any]:
    """The model's top-level items: DemInstruction and DemRepeatBlock objects."""
    lines = text.split("\n")
    pos = 0

    def block(nested: bool) -> List[Tuple]:
        nonlocal pos
        out: List[Tuple] = []
        while pos < len(lines):
            raw = lines[pos]
            pos += 1
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line == "}":
                return out
            m = _LINE.match(line)
            if not m:
                raise ValueError(f"Can't parse model line: {raw!r}")
            name, tag, args, rest = m.group(1).lower(), m.group(2), m.group(3), m.group(4).strip()
            tag = _unescape_tag(tag[1:-1]) if tag else ""
            if name == "repeat":
                count = int(rest.rstrip("{").strip())
                out.append(("repeat", count, block(True)))
                continue
            values = [float(a) for a in args.split(",")] if args and args.strip() else []
            targets: List[Any] = []
            for w in rest.split():
                targets.append(int(w) if name == "shift_detectors" else DemTarget(w))
            out.append(("op", name, values, targets, tag))
        return out

    return [_item_object(it) for it in block(False)]


def _item_object(it: Tuple) -> Any:
    from ._circuit import DetectorErrorModel

    if it[0] == "op":
        _, name, args, targets, tag = it
        return DemInstruction._raw(name, args, targets, tag)
    _, count, body = it
    rb = DemRepeatBlock.__new__(DemRepeatBlock)
    rb._count = count
    rb._body = DetectorErrorModel(items_exact_text([_item_object(b) for b in body]))
    return rb


def items_exact_text(items: List[Any]) -> str:
    return "\n".join(i._exact_text() for i in items)


def _exact_text_of(dem: Any) -> str:
    return items_exact_text(parse_items(str(dem)))
