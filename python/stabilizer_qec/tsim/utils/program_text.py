# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5; Apache License 2.0; see LICENSE in this directory),
# modified for stabilizer_qec: tsim is stabilizer_qec.tsim; annotations postponed (Python 3.9);
# stim_to_shorthand also reads this package's tagged forms (I[T], radians).
"""Conversion utilities between tsim shorthand and stim program text."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction

from stabilizer_qec.tsim.core.tags import decode_t_user_tag, encode_t_tag

# Matches valid numeric literals including scientific notation (e.g. 0.5, 4e-4, 1.2e3)
FLOAT_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

_TSIM_GATES = {
    "CCZ",
    "CCX",
    "R_X",
    "R_Y",
    "R_Z",
    "R_XX",
    "R_YY",
    "R_ZZ",
    "R_PAULI",
    "U3",
}
_GATE_NOT_FOUND_RE = re.compile(r"Gate not found: '(\w+)'")
_GATE_USAGE_RE = re.compile(
    r"(?<!\[)\b(CCZ\b|CCX\b|R_PAULI\([^)]*\)|R_[XYZ]{1,2}\([^)]*\)|R_[XYZ]\b|U3\([^)]*\)|U3\b)"
)


def controlled_gate_decomposition_lines(
    gate: str,
    control1: int | str,
    control2: int | str,
    target: int | str,
    *,
    tag: str = "",
) -> list[str]:
    """Return a Clifford+T decomposition for a controlled-controlled gate."""
    if gate not in ("CCZ", "CCX"):
        raise ValueError(f"Unsupported controlled-controlled gate: {gate!r}")

    def tagged(name: str) -> str:
        return f"{name}[{tag}]" if tag else name

    a, b, c = str(control1), str(control2), str(target)
    ccz_lines = [
        f"{tagged('CNOT')} {b} {c}",
        f"{tagged('T_DAG')} {c}",
        f"{tagged('CNOT')} {a} {c}",
        f"{tagged('T')} {c}",
        f"{tagged('CNOT')} {b} {c}",
        f"{tagged('T_DAG')} {c}",
        f"{tagged('CNOT')} {a} {c}",
        f"{tagged('T')} {b}",
        f"{tagged('T')} {c}",
        f"{tagged('CNOT')} {a} {b}",
        f"{tagged('T')} {a}",
        f"{tagged('T_DAG')} {b}",
        f"{tagged('CNOT')} {a} {b}",
    ]
    if gate == "CCZ":
        return ccz_lines
    return [f"{tagged('H')} {c}", *ccz_lines, f"{tagged('H')} {c}"]


def _expand_controlled_gates(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        body, sep, comment = line.partition("#")
        match = re.match(r"^(\s*)(CCZ|CCX)(?:\[([^\]\n]*)\])?\s+(.+?)\s*$", body)
        if not match:
            lines.append(line)
            continue

        indent, gate, tag, targets_text = match.groups()
        targets = targets_text.split()
        if len(targets) % 3 != 0 or not all(target.isdecimal() for target in targets):
            raise ValueError(
                f"{gate} expects bare qubit integer targets in groups of three."
            )

        if sep:
            lines.append(f"{indent}{sep}{comment}")
        for i in range(0, len(targets), 3):
            lines.extend(
                f"{indent}{decomp_line}"
                for decomp_line in controlled_gate_decomposition_lines(
                    gate, targets[i], targets[i + 1], targets[i + 2], tag=tag or ""
                )
            )
    return "\n".join(lines)


def _replace_t_family(stim_gate: str) -> Callable[[re.Match[str]], str]:
    def replace(m: re.Match[str]) -> str:
        return f"{stim_gate}[{encode_t_tag(m.group(1) or '')}]"

    return replace


def _replace_t_family_shorthand(tsim_gate: str) -> Callable[[re.Match[str]], str]:
    def replace(m: re.Match[str]) -> str:
        user_tag = decode_t_user_tag(m.group(1))
        tag_suffix = f"[{user_tag}]" if user_tag else ""
        return f"{tsim_gate}{tag_suffix}"

    return replace


def enriched_stim_error(exc: ValueError, converted_text: str) -> ValueError:
    """Improve stim parse errors for tsim-specific gates.

    When stim raises a 'Gate not found' error for a gate that should have been
    converted by shorthand_to_stim, this searches the converted text for the
    unconverted usage and returns a more helpful error message.
    """
    m = _GATE_NOT_FOUND_RE.search(str(exc))
    if not m or m.group(1) not in _TSIM_GATES:
        return exc
    # Successfully converted gates live inside brackets (e.g. I[R_Z(...)]) and won't match.
    usage = _GATE_USAGE_RE.search(converted_text)
    if not usage:
        return exc
    return ValueError(f"Could not parse '{usage.group()}' in program text.")


def shorthand_to_stim(text: str) -> str:
    """Convert tsim shorthand syntax to valid stim instructions.

    Converts:
        CCZ 0 1 2           → Clifford+T decomposition of CCZ
        CCX 0 1 2           → Clifford+T decomposition of CCX/Toffoli
        T 0 1               → S[T] 0 1
        T_DAG 0 1           → S_DAG[T] 0 1
        TPP X0*Y1           → SPP[T] X0*Y1
        TPP_DAG X0*Y1       → SPP_DAG[T] X0*Y1
        R_Z(0.3) 0          → I[R_Z(theta=0.3*pi)] 0
        R_X(0.25) 0         → I[R_X(theta=0.25*pi)] 0
        R_Y(-0.5) 0         → I[R_Y(theta=-0.5*pi)] 0
        R_XX(0.5) 0 1       → SPP[R_PAULI(theta=0.5*pi)] X0*X1
        R_YY(0.5) 0 1       → SPP[R_PAULI(theta=0.5*pi)] Y0*Y1
        R_ZZ(0.5) 0 1       → SPP[R_PAULI(theta=0.5*pi)] Z0*Z1
        R_PAULI(0.3) X0*Y1  → SPP[R_PAULI(theta=0.3*pi)] X0*Y1
        U3(0.3, 0.24, 0.49) 0 → I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 0

    """
    text = _expand_controlled_gates(text)

    # TPP_DAG/TPP must come before T_DAG/T to avoid partial matches
    # (?<!\[) ensures we don't match T inside [T]
    text = re.sub(
        r"(?<!\[)\bTPP_DAG(?:\[([^\]\n]*)\])?(?!\w)",
        _replace_t_family("SPP_DAG"),
        text,
    )
    text = re.sub(
        r"(?<!\[)\bTPP(?:\[([^\]\n]*)\])?(?!\w)",
        _replace_t_family("SPP"),
        text,
    )
    text = re.sub(
        r"(?<!\[)\bT_DAG(?:\[([^\]\n]*)\])?(?!\w)",
        _replace_t_family("S_DAG"),
        text,
    )
    text = re.sub(
        r"(?<!\[)\bT(?:\[([^\]\n]*)\])?(?!\w)",
        _replace_t_family("S"),
        text,
    )

    def replace_pauli_pair(m: re.Match) -> str:
        pauli = m.group(1)
        alpha = float(m.group(2))
        q0, q1 = m.group(3), m.group(4)
        if q0 == q1:
            raise ValueError(
                f"R_{pauli}{pauli} target qubits must be distinct, got {q0} {q1}."
            )
        return f"SPP[R_PAULI(theta={alpha}*pi)] {pauli}{q0}*{pauli}{q1}"

    text = re.sub(
        rf"\bR_([XYZ])\1\(({FLOAT_RE})\)\s+(\d+)\s+(\d+)", replace_pauli_pair, text
    )

    def replace_pauli(m: re.Match) -> str:
        alpha = float(m.group(1))
        return f"SPP[R_PAULI(theta={alpha}*pi)] {m.group(2)}"

    text = re.sub(
        rf"\bR_PAULI\(({FLOAT_RE})\)\s+((?:[XYZ]\d+)(?:\*[XYZ]\d+)*)",
        replace_pauli,
        text,
    )

    def replace_rotation(m: re.Match) -> str:
        axis = m.group(1)
        return f"I[R_{axis}(theta={float(m.group(2))}*pi)]"

    text = re.sub(rf"\bR_([XYZ])\(({FLOAT_RE})\)", replace_rotation, text)

    def replace_u3(m: re.Match) -> str:
        theta, phi, lam = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return f"I[U3(theta={theta}*pi, phi={phi}*pi, lambda={lam}*pi)]"

    text = re.sub(
        rf"\bU3\(({FLOAT_RE})\s*,\s*({FLOAT_RE})\s*,\s*({FLOAT_RE})\)",
        replace_u3,
        text,
    )

    # Canonicalize literals inside already-expanded parametric tags so that
    # equivalent inputs like `I[R_X(theta=0.5e-2*pi)]` and
    # `I[R_X(theta=0.005*pi)]` produce the same stim tag string. This keeps
    # `tsim.Circuit(str(c)) == c` round-trip stable across notations.
    def _canonicalize_param(m: re.Match) -> str:
        return f"{m.group(1)}={float(m.group(2))}*pi"

    text = re.sub(
        rf"\b(theta|phi|lambda)=({FLOAT_RE})\*pi",
        _canonicalize_param,
        text,
    )

    return text


_NATIVE_LINE = re.compile(r"^(\s*)(I|II)\[([^\]\n]*)\](.*)$")
_NATIVE_TAG = re.compile(r"^\s*(\w+)\s*(?:\((.*)\))?\s*$")


def _native_args(body: str) -> dict[str, str] | None:
    """A tag's ``key=value`` arguments (``theta=0.1, pauli=XZ``), as the core reads them
    (``nonpauli::split``): spaces around each part and around the ``=`` skipped, and an empty
    part (``theta=0.1,``) too; ``None`` if a part is not ``key=value``."""
    out: dict[str, str] = {}
    for part in body.split(","):
        if not part.strip():
            continue
        key, eq, value = part.partition("=")
        if not eq:
            return None
        out[key.strip()] = value.strip()
    return out


def _no_args(body: str | None) -> bool:
    """Whether a tag has no arguments: no parentheses (``T``) or only empty ones (``T()``, ``T( , )``),
    which the core reads alike."""
    return body is None or _native_args(body) == {}


def _turns(radians: float, scale: float) -> str:
    """The shortest number of half-turns h with ``float(h) * pi * scale == radians``, as
    ``tsim_to_native`` reads it back (``scale`` 1/2 for a rotation's exp(−iθP), 1 for U3)."""
    h = radians / (math.pi * scale)
    for digits in range(1, 18):
        candidate = float(f"{h:.{digits}g}")
        if candidate * math.pi * scale == radians:
            return repr(candidate)
    return repr(h)


# A rotation written in half-turns (Tsim's ``R_Z(a)``, ``I[R_Z(theta=a*pi)]``) is stored in radians,
# ``a * pi * scale`` as a float, where several half-turns can give the same float. Its radians are
# written as ``repr`` when ``_turns`` reads them back as the half-turns written, and otherwise with the
# further digits of the exact product ``a * pi * scale`` that give them back (``_read_turns``): the
# number is the same float, so the engines run the same rotation, and the half-turns are Tsim's, the
# literal as written (as ``repr`` of its float). A product rotation by +-pi/8 is ``TPP`` when stored as
# ``repr`` and Tsim's ``R_PAULI(+-0.25)`` (a different ZX graph, its scalar phase -+1/8) when stored with
# the further digits.


def _significant_digits(text: str) -> int:
    """The number of significant digits a decimal number is written with (``-0.3927e-1`` has 4)."""
    mantissa = text.strip().lstrip("+-").lower().split("e", 1)[0]
    return len(mantissa.replace(".", "").strip("0"))


def _has_further_digits(text: str) -> bool:
    """Whether a number is written with more digits than ``repr`` of its float."""
    return _significant_digits(text) > _significant_digits(repr(float(text)))


def _turns_of_digits(text: str, scale: float) -> float | None:
    """The half-turns whose radians a number written with further digits is: the float nearest to
    ``text / (pi * scale)``, if its radians are the float of ``text``; else ``None``."""
    try:
        turns = float(Fraction(text.strip()) / (Fraction(math.pi) * Fraction(scale)))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return turns if turns * math.pi * scale == float(text) else None


def _decimal_text(value: Fraction, digits: int) -> str:
    """``value`` rounded to ``digits`` significant digits (half to even), in decimal."""
    with localcontext() as ctx:
        ctx.prec = digits
        ctx.rounding = ROUND_HALF_EVEN
        d = Decimal(value.numerator) / Decimal(value.denominator)
    return format(d, "f") if -5 <= d.adjusted() < 16 else format(d, "e")


def _stored_radians(turns: float, scale: float, product: bool = False) -> str:
    """The radians this package stores for a rotation of ``turns`` half-turns (``scale`` 1/2 for a
    rotation's exp(−iθP), 1 for U3): ``repr`` of ``turns * pi * scale`` when ``_read_turns`` gives
    ``turns`` back from it (and, for a Pauli product, it is not +-pi/8, which is ``TPP``), else the
    exact product's decimal to the fewest further digits that give ``turns`` back."""
    radians = turns * math.pi * scale
    short = repr(radians)
    if not (product and abs(radians) == math.pi / 8) and _turns(radians, scale) == repr(turns):
        return short
    exact = Fraction(turns) * Fraction(math.pi) * Fraction(scale)
    for digits in list(range(_significant_digits(short) + 1, 40)) + [2000]:
        text = _decimal_text(exact, digits)
        if _has_further_digits(text) and float(text) == radians and repr(_turns_of_digits(text, scale)) == repr(turns):
            return text
    return short  # pragma: no cover - the exact product, written in full, has been the answer for every angle tried


def _read_turns(text: str, scale: float) -> str:
    """The half-turns of stored radians (``repr`` of the float, as Tsim writes them): those its further
    digits give, or else ``_turns``'s."""
    if _has_further_digits(text):
        turns = _turns_of_digits(text, scale)
        if turns is not None:
            return repr(turns)
    return _turns(float(text), scale)


def _is_tpp_radians(text: str) -> bool:
    """Whether a Pauli product's stored radians are ``TPP``'s (+-pi/8 as ``repr`` writes it) rather
    than ``R_PAULI(+-0.25)``'s (+-pi/8 with further digits)."""
    return abs(float(text)) == math.pi / 8 and not _has_further_digits(text)


# Tsim's product rotation (``SPP[R_PAULI(theta=a*pi)] P``, ``SPP_DAG[...]``, ``SPP[T]``, ``SPP_DAG[T]``) is
# stored one product a line, ``I[R_PAULI(theta=t, pauli=L)] q1 q2 ...``, exp(-i t P) with t the radians of
# the half-turns signed by the dagger and the product's ``!``s, so the engines run the rotation it is. The
# rest of Tsim's instruction is kept in how that is written, which the core reads past: a factor's letter
# is lower case where the factor has a ``!`` (``SPP[T] !X0*Y1`` is ``pauli=xY``, the core reads letters in
# either case), and the ``SPP_DAG`` of a rotation has its arguments in the other order
# (``SPP_DAG[R_PAULI(theta=0.3*pi)] X0`` is ``I[R_PAULI(pauli=X, theta=-0.47123889803846897)] 0``; a
# ``TPP``'s dagger is its sign). So the instruction read back is Tsim's, and Stim's fusion of it with its
# neighbours, R_PAULI's limits and the messages quoting it are Tsim's. The core reads ``R_PAULI`` on ``II``
# as on ``I`` (``nonpauli::parse``: one product over the targets, ``II[R_PAULI(theta=t, pauli=XZYY)] 0 1 2 3``
# is X0*Z1*Y2*Y3), and so it is read here.


def _product_rotation(args: dict, qubits: list) -> tuple | None:
    """Tsim's instruction of this package's ``I[R_PAULI(theta=t, pauli=L)] q1 q2 ...`` (or ``II[...]``, the
    same product; ``args`` its tag's arguments): ``(dagger, turns, factors)``, the instruction
    ``SPP_DAG`` when ``dagger`` else ``SPP``, its tag ``T`` when ``turns`` is ``None`` else
    ``R_PAULI(theta=<turns>*pi)``, its product's ``factors`` ``(inverted, letter, qubit)`` (``!`` on
    the inverted ones; ``I``, ``i`` and ``_`` letters dropped, identities as the core reads them in
    ``nonpauli::parse``). ``None`` if the letters are not one of these, X, Y, Z (either case) for
    each qubit."""
    letters = args["pauli"]
    if len(letters) != len(qubits) or not set(letters) <= set("IXYZixyz_"):
        return None
    factors = [(p.islower(), p.upper(), q) for p, q in zip(letters, qubits) if p not in "Ii_"]
    flipped = sum(inverted for inverted, _, _ in factors) % 2 == 1
    theta = args["theta"]
    if _is_tpp_radians(theta):
        return (float(theta) < 0) != flipped, None, factors
    dagger = list(args) == ["pauli", "theta"]
    turns = _read_turns(theta, 0.5)
    return dagger, (repr(-float(turns)) if dagger != flipped else turns), factors


def _product_text(factors: list) -> str:
    """A product's factors ``(inverted, letter, qubit)`` as Stim writes them (``!X0*Y1``)."""
    return "*".join(f"{'!' if inverted else ''}{p}{q}" for inverted, p, q in factors)


def _native_to_shorthand_line(line: str) -> str | None:
    m = _NATIVE_LINE.match(line)
    if not m:
        return None
    indent, gate, tag, rest = m.groups()
    t = _NATIVE_TAG.match(tag)
    if not t:
        return None
    name, body = t.group(1), t.group(2)
    targets = rest.split()
    if gate == "I" and _no_args(body) and name in ("T", "T_DAG") and targets:
        return f"{indent}{name} {' '.join(targets)}"
    args = _native_args(body) if body else None
    if args is None or not targets:
        return None
    try:
        if gate == "I" and name in ("R_X", "R_Y", "R_Z") and set(args) == {"theta"}:
            return f"{indent}{name}({_read_turns(args['theta'], 0.5)}) {' '.join(targets)}"
        if gate == "I" and name == "U3" and set(args) == {"theta", "phi", "lambda"}:
            a, b, c = (_read_turns(args[k], 1.0) for k in ("theta", "phi", "lambda"))
            return f"{indent}U3({a}, {b}, {c}) {' '.join(targets)}"
        if gate == "II" and name in ("R_XX", "R_YY", "R_ZZ") and set(args) == {"theta"} and len(targets) % 2 == 0:
            pairs = [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]
            if _is_tpp_radians(args["theta"]):
                # +-pi/8 as ``repr`` writes it is ``TPP``, as the graph reads it (``R_ZZ(0.25)`` is Tsim's R_PAULI).
                p, head = name[2], "TPP" if float(args["theta"]) > 0 else "TPP_DAG"
                return "\n".join(f"{indent}{head} {p}{a}*{p}{b}" for a, b in pairs)
            angle = _read_turns(args["theta"], 0.5)
            return "\n".join(f"{indent}{name}({angle}) {a} {b}" for a, b in pairs)
        if name == "R_PAULI" and set(args) == {"theta", "pauli"}:  # on I or II, one product as the core reads it
            if not all(q.isdecimal() for q in targets):
                return None
            found = _product_rotation(args, targets)
            if found is None or not found[2]:
                return None
            dagger, angle, factors = found
            product = _product_text(factors)
            if angle is None:
                return f"{indent}{'TPP_DAG' if dagger else 'TPP'} {product}"
            if dagger or any(inverted for inverted, _, _ in factors):
                # Tsim's tagged form, which its shorthand keeps (``SPP_DAG[R_PAULI(theta=0.3*pi)] X0``,
                # ``SPP[R_PAULI(theta=0.3*pi)] !X0``; ``X0*!X1`` becomes ``R_PAULI(0.3) X0*!X1`` below).
                return f"{indent}{'SPP_DAG' if dagger else 'SPP'}[R_PAULI(theta={angle}*pi)] {product}"
            if len(factors) == 2 and factors[0][1] == factors[1][1] and factors[0][2] != factors[1][2]:
                p = factors[0][1]
                return f"{indent}R_{p}{p}({angle}) {factors[0][2]} {factors[1][2]}"
            return f"{indent}R_PAULI({angle}) {product}"
    except ValueError:
        return None
    return None


def _native_to_shorthand(text: str) -> str:
    """This package's tagged forms (``I[T]``, ``I[R_X(theta=<radians>)]``, ``I[U3(...)]``,
    ``I[R_PAULI(theta=..., pauli=...)]`` or ``II[...]``, ``II[R_ZZ(theta=...)]``) as Tsim's shorthand."""
    out = []
    for line in text.split("\n"):
        converted = _native_to_shorthand_line(line)
        out.append(line if converted is None else converted)
    return "\n".join(out)


def stim_to_shorthand(text: str) -> str:
    """Convert expanded stim annotations back to tsim shorthand.

    Rewrites:
    - I[U3(theta=θ*pi, phi=φ*pi, lambda=λ*pi)] → U3(θ, φ, λ)
    - I[R_X(theta=α*pi)] / I[R_Y(...)] / I[R_Z(...)] → R_X(α) / R_Y(α) / R_Z(α)
    - SPP[R_PAULI(theta=α*pi)] P0*P1 → R_PP(α) 0 1 (P ∈ {X, Y, Z})
    - SPP[R_PAULI(theta=α*pi)] X0*Y1 → R_PAULI(α) X0*Y1
    - SPP[T] → TPP
    - SPP_DAG[T] → TPP_DAG
    - S[T] → T
    - S_DAG[T] → T_DAG

    and this package's tagged forms (stabilizer_qec's addition), angles in radians of
    exp(−iθP) written back in the half-turns they were written in (the shortest decimal number
    that gives the radians, or the one their further digits give):
    - I[T] / I[T_DAG] → T / T_DAG
    - I[R_X(theta=t)] → R_X(2t/π), and R_Y, R_Z
    - I[U3(theta=a, phi=b, lambda=c)] → U3(a/π, b/π, c/π)
    - I[R_PAULI(theta=±π/8, pauli=XY)] 0 1 → TPP / TPP_DAG X0*Y1 (±π/8 as ``repr`` writes it;
      written with further digits it is R_PAULI(±0.25) X0*Y1)
    - I[R_PAULI(theta=t, pauli=ZZ)] 0 1 → R_ZZ(2t/π) 0 1; any other product
      R_PAULI(2t/π) X0*Y1 (II[R_PAULI(...)] as I[R_PAULI(...)], one product over its targets, as
      the core reads it)
    - a product with a ``!`` (a lower-case letter, ``pauli=xY``) or the ``SPP_DAG`` of a rotation
      (its arguments in the other order, ``pauli=X, theta=t``) → Tsim's tagged form, as its
      shorthand writes it: SPP[R_PAULI(theta=α*pi)] !X0*Y1, SPP_DAG[R_PAULI(theta=α*pi)] X0,
      TPP !X0 (see ``_product_rotation``)
    - II[R_XX(theta=t)] 0 1 2 3 → R_XX(2t/π) 0 1 and R_XX(2t/π) 2 3, a line a pair (TPP / TPP_DAG
      X0*X1 at ±π/8 as ``repr`` writes it)
    """
    text = _native_to_shorthand(text)

    # Replace I[U3(theta=θ*pi, phi=φ*pi, lambda=λ*pi)] with U3(θ, φ, λ)
    def replace_u3(m: re.Match) -> str:
        theta, phi, lam = m.group(1), m.group(2), m.group(3)
        return f"U3({theta}, {phi}, {lam})"

    text = re.sub(
        rf"\bI\[U3\(theta=({FLOAT_RE})\*pi, phi=({FLOAT_RE})\*pi, lambda=({FLOAT_RE})\*pi\)\]",
        replace_u3,
        text,
    )

    # SPP[R_PAULI(...)] with a same-axis two-qubit product → R_XX/R_YY/R_ZZ shorthand.
    # Must precede the general R_PAULI rewrite below.
    def replace_pauli_pair(m: re.Match) -> str:
        alpha, pauli, q0, q1 = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"R_{pauli}{pauli}({alpha}) {q0} {q1}"

    text = re.sub(
        rf"\bSPP\[R_PAULI\(theta=({FLOAT_RE})\*pi\)\] ([XYZ])(\d+)\*\2(\d+)(?!\*)\b",
        replace_pauli_pair,
        text,
    )

    def replace_pauli(m: re.Match) -> str:
        alpha, product = m.group(1), m.group(2)
        return f"R_PAULI({alpha}) {product}"

    text = re.sub(
        rf"\bSPP\[R_PAULI\(theta=({FLOAT_RE})\*pi\)\] ((?:[XYZ]\d+)(?:\*[XYZ]\d+)*)",
        replace_pauli,
        text,
    )

    # Replace I[R_X(...)] / I[R_Y(...)] / I[R_Z(...)] with R_X(α) / R_Y(α) / R_Z(α)
    def replace_rotation(m: re.Match) -> str:
        axis = m.group(1)
        angle = m.group(2)
        return f"R_{axis}({angle})"

    text = re.sub(
        rf"\bI\[R_([XYZ])\(theta=({FLOAT_RE})\*pi\)\]",
        replace_rotation,
        text,
    )

    # Replace SPP[T] and SPP_DAG[T] with TPP and TPP_DAG
    # Must come before S[T]/S_DAG[T] to avoid partial matches
    text = re.sub(
        r"(?<!\w)SPP_DAG\[(T(?::[^\]\n]*)?)\](?!\w)",
        _replace_t_family_shorthand("TPP_DAG"),
        text,
    )
    text = re.sub(
        r"(?<!\w)SPP\[(T(?::[^\]\n]*)?)\](?!\w)",
        _replace_t_family_shorthand("TPP"),
        text,
    )

    # Replace S[T] and S_DAG[T] with T and T_DAG
    # Use non-word lookarounds because trailing ] is not a word character.
    text = re.sub(
        r"(?<!\w)S_DAG\[(T(?::[^\]\n]*)?)\](?!\w)",
        _replace_t_family_shorthand("T_DAG"),
        text,
    )
    text = re.sub(
        r"(?<!\w)S\[(T(?::[^\]\n]*)?)\](?!\w)",
        _replace_t_family_shorthand("T"),
        text,
    )

    return text
