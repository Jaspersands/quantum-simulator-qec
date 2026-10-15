"""Conformance checks for decoders (Shen and Zhong, arXiv 2609.07035): what can be checked about a
decoder's answers without knowing the right ones.

A decoder hands back a correction (the model's faults it believes fired), predicted observable
flips, or a weight. Counting logical errors reads only the observables, at one fixed presentation
of the model. This module checks the rest, on graphlike models:

- the correction explains the syndrome, in the caller's indices; observables returned with it
  are the correction's, and a weight returned with it is its weight (the interface's own
  contract, mandatory for every decoder that returns them);
- the same instance under another presentation (faults listed in another order, R1; detectors
  numbered otherwise, R2; an extra or shifted observable, R3 and R3b; decoded fresh, again or
  after other shots, R4; each fault split into two parallel halves, R5; every weight scaled,
  R6) gets an equivalent answer. Two feasible answers of different weight prove the heavier is
  not minimum-weight with no solver involved: a `Certificate`, which `verify` (or the standalone
  `tools/verify_certificates.py`) re-derives from the model's text alone;
- against an exact oracle (every fault subset, for up to 22 faults) the answer has the optimum's
  weight, and an error within the code's radius is corrected (bounded-distance correctness).

What a failure means is decided by what the decoder declares (`Declaration`): a capability
declared ``"yes"`` is held to it (a failure is a conformance failure), one declared ``"no"`` is a
legal skip, and one the documentation is ``"silent"`` on is measured but cannot be failed
(*unassessable*), as conformance testing has it (ISO/IEC 9646). The paper audited five libraries'
documentation and found 4 of 54 capability questions answered; `panel()` holds this package's
ports of them (PyMatching, fusion-blossom, ldpc, MWPF, Tesseract) to those audited declarations,
and `subjects()` holds this package's decoders to its own, which answer every question.

>>> from stabilizer_qec import conformance
>>> report = conformance.check("matching", conformance.population()[3])
>>> report.failures, report.counts["matching", "exact optimality"]["pass"]
([], 16)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, NamedTuple, Optional, Sequence, Union

import numpy as np

from ._circuit import DetectorErrorModel

YES, NO, SILENT = "yes", "no", "silent"

__all__ = [
    "Answer", "Certificate", "Declaration", "Instance", "Report", "Subject",
    "audit", "check", "panel", "population", "subjects", "width_family", "verify_certificate",
]


@dataclass(frozen=True)
class Declaration:
    """What a decoder's documentation says it guarantees, each ``"yes"``, ``"no"`` or
    ``"silent"``: ``bounded_distance`` (L1: an error within the code's radius is corrected),
    ``min_weight`` (L2: the correction is a minimum-weight one), ``max_likelihood`` (L3),
    ``deterministic`` (the answer is a function of the syndrome alone), ``randomised`` (a
    declared random schedule) and ``int_weights`` (it rounds weights to integers internally,
    each by at most ``weight_quantum``). ``source`` says where the declaration comes from."""

    bounded_distance: str = SILENT
    min_weight: str = SILENT
    max_likelihood: str = SILENT
    deterministic: str = SILENT
    randomised: str = SILENT
    int_weights: str = SILENT
    weight_quantum: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("bounded_distance", "min_weight", "max_likelihood", "deterministic", "randomised", "int_weights"):
            if getattr(self, name) not in (YES, NO, SILENT):
                raise ValueError(f"a declaration is 'yes', 'no' or 'silent', not {getattr(self, name)!r} for {name}")


class Answer(NamedTuple):
    """A decoder's answer for one syndrome: its correction (fault indices into the model it was
    given), the observables it predicts (if it computes them), and the weight it reports.
    ``gave_up`` marks an answer the decoder itself flags as failed (Relay-BP with no leg
    converged, a search that gave up): it claims no correction. ``raised`` is an exception's
    text, where decoding raised."""

    correction: Optional[tuple]
    observables: Optional[tuple] = None
    weight: Optional[float] = None
    raised: Optional[str] = None
    gave_up: bool = False


@dataclass(frozen=True)
class Subject:
    """A decoder under test: ``build(model)`` (a `DetectorErrorModel`) returns a function from a
    syndrome (uint8, one per detector) to an `Answer`, or to the correction's fault indices.
    ``computes_observables`` says its answers carry observables it computed itself."""

    name: str
    build: Callable[[DetectorErrorModel], Callable[[np.ndarray], Any]]
    declaration: Declaration = Declaration()
    computes_observables: bool = False
    description: str = ""


# ---------------------------------------------------------------------------------------------
# Instances and their presentations


def _weight(p: float) -> float:
    return math.log((1 - p) / p)


@dataclass
class Instance:
    """A graphlike decoding instance: faults each flipping one or two detectors and some
    observables, with probabilities in (0, 1/2). ``distance`` is the fewest faults flipping an
    observable and no detector, where known."""

    name: str
    num_detectors: int
    num_observables: int
    faults: list  # (detectors tuple, observables tuple, p)
    distance: Optional[int] = None

    def __post_init__(self) -> None:
        for dets, obs, p in self.faults:
            if not 1 <= len(dets) <= 2 or len(set(dets)) != len(dets):
                raise ValueError(f"{self.name}: a fault flips one or two detectors, not {list(dets)}")
            if not 0 < p < 0.5:
                raise ValueError(f"{self.name}: a fault's probability is in (0, 1/2), not {p}")

    @classmethod
    def from_model(cls, model: Union[DetectorErrorModel, str], name: str = "model", *, merge: bool = True) -> "Instance":
        """From a graphlike detector error model (decomposed parts and separators are not
        allowed); with ``merge``, faults flipping the same detectors are combined, and must flip
        the same observables."""
        model = model if isinstance(model, DetectorErrorModel) else DetectorErrorModel(model)
        faults: list = []
        seen: dict = {}
        for inst in model.flattened():
            if inst.type != "error":
                continue
            dets, obs = [], []
            for t in inst.targets_copy():
                if t.is_separator():
                    raise ValueError(f"{name}: a decomposed fault is outside a graphlike instance")
                (dets if t.is_relative_detector_id() else obs).append(t.val)
            p = float(inst.args_copy()[0])
            if p == 0:
                continue
            key = tuple(sorted(dets))
            o = tuple(sorted(set(x for x in obs if obs.count(x) % 2)))
            if merge and key in seen:
                i = seen[key]
                d0, o0, p0 = faults[i]
                if o0 != o:
                    raise ValueError(f"{name}: two faults flip detectors {list(key)} but different observables")
                faults[i] = (d0, o0, p0 + p - 2 * p0 * p)
                continue
            seen[key] = len(faults)
            faults.append((key, o, p))
        return cls(name, model.num_detectors, model.num_observables, faults)

    @classmethod
    def from_checks(cls, h: Any, logicals: Any, p: float, name: str, distance: Optional[int] = None) -> "Instance":
        """Code capacity: each column of ``h`` a fault of probability ``p`` flipping its rows'
        detectors and the ``logicals`` rows it meets; columns with the same detectors merge."""
        h, l = np.asarray(h, dtype=np.uint8), np.atleast_2d(np.asarray(logicals, dtype=np.uint8))
        text = "\n".join(
            f"error({p}) " + " ".join([f"D{d}" for d in np.flatnonzero(h[:, j])] + [f"L{o}" for o in np.flatnonzero(l[:, j])])
            for j in range(h.shape[1])
        )
        inst = cls.from_model(DetectorErrorModel(text), name)
        inst.num_detectors, inst.num_observables, inst.distance = h.shape[0], l.shape[0], distance
        return inst

    @property
    def num_faults(self) -> int:
        return len(self.faults)

    def model(self) -> DetectorErrorModel:
        """The instance as a detector error model, its faults in order."""
        lines = [f"detector D{d}" for d in range(self.num_detectors)]
        if self.num_observables:
            lines.append(f"logical_observable L{self.num_observables - 1}")
        lines += [f"error({p!r}) " + " ".join([f"D{d}" for d in dets] + [f"L{o}" for o in obs]) for dets, obs, p in self.faults]
        return DetectorErrorModel("\n".join(lines))

    def text(self) -> str:
        return str(self.model())

    def weights(self) -> np.ndarray:
        return np.array([_weight(p) for _, _, p in self.faults])

    def simple(self) -> bool:
        """No two faults flip the same detectors (a returned edge names one fault)."""
        return len({dets for dets, _, _ in self.faults}) == len(self.faults)

    def uniform(self) -> bool:
        """Every fault has the same weight, so minimum weight is fewest faults (Hamming order)."""
        w = self.weights()
        return bool(len(w) and np.ptp(w) <= 1e-12 * abs(w[0]))

    def syndrome_of(self, correction: Iterable[int]) -> np.ndarray:
        s = np.zeros(self.num_detectors, dtype=np.uint8)
        for j in correction:
            for d in self.faults[j][0]:
                s[d] ^= 1
        return s

    def observables_of(self, correction: Iterable[int]) -> np.ndarray:
        o = np.zeros(self.num_observables, dtype=np.uint8)
        for j in correction:
            for k in self.faults[j][1]:
                o[k] ^= 1
        return o

    def weight_of(self, correction: Iterable[int]) -> float:
        w = self.weights()
        return float(sum(w[j] for j in correction))

    # Presentations: each returns the new instance and how to carry an answer back.

    def permute_faults(self, perm: Sequence[int]) -> "Instance":
        """R1: fault ``i`` of the result is fault ``perm[i]`` of this one."""
        return Instance(self.name, self.num_detectors, self.num_observables, [self.faults[j] for j in perm], self.distance)

    def permute_detectors(self, perm: Sequence[int]) -> "Instance":
        """R2: detector ``d`` here is detector ``perm[d]`` in the result."""
        faults = [(tuple(sorted(perm[d] for d in dets)), obs, p) for dets, obs, p in self.faults]
        return Instance(self.name, self.num_detectors, self.num_observables, faults, self.distance)

    def widened(self) -> "Instance":
        """R3: one more observable, flipped by no fault."""
        return Instance(self.name, self.num_detectors, self.num_observables + 1, list(self.faults), self.distance)

    def offset(self) -> "Instance":
        """R3b: every observable moved up one, observable 0 flipped by no fault."""
        faults = [(dets, tuple(o + 1 for o in obs), p) for dets, obs, p in self.faults]
        return Instance(self.name, self.num_detectors, self.num_observables + 1, faults, self.distance)

    def scaled(self, gamma: float) -> "Instance":
        """R6: every weight times ``gamma``."""
        faults = [(dets, obs, 1 / (1 + math.exp(gamma * _weight(p)))) for dets, obs, p in self.faults]
        return Instance(self.name, self.num_detectors, self.num_observables, faults, self.distance)

    def split(self) -> "Instance":
        """R5: each fault ``j`` as faults ``2j`` and ``2j + 1``, parallel, each of probability q
        with 2q(1 − q) = p, so the model's distribution of detection events is unchanged."""
        faults = []
        for dets, obs, p in self.faults:
            q = (1 - math.sqrt(1 - 2 * p)) / 2
            faults += [(dets, obs, q), (dets, obs, q)]
        return Instance(self.name, self.num_detectors, self.num_observables, faults, self.distance)


def _rotated_checks(d: int) -> tuple:
    """A rotated surface code's Z checks (detecting X errors) and its Z logical (column 0)."""
    q = lambda r, c: r * d + c  # noqa: E731
    rows = []
    for r in range(d + 1):
        for c in range(d + 1):
            if (r + c) % 2:
                continue
            bulk = 0 < r < d and 0 < c < d
            edge = (r in (0, d)) and 0 < c < d
            if not (bulk or edge):
                continue
            row = np.zeros(d * d, dtype=np.uint8)
            for rr, cc in ((r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c)):
                if 0 <= rr < d and 0 <= cc < d:
                    row[q(rr, cc)] = 1
            rows.append(row)
    logical = np.zeros(d * d, dtype=np.uint8)
    logical[[q(r, 0) for r in range(d)]] = 1
    return np.array(rows), logical


def _repetition(d: int) -> np.ndarray:
    h = np.zeros((d - 1, d), dtype=np.uint8)
    for i in range(d - 1):
        h[i, i] = h[i, i + 1] = 1
    return h


def population(p: float = 0.01) -> list:
    """The paper's instances (its Table 3), X errors at code capacity: repetition codes of
    distance 3, 5 and 7, rotated surface codes of distance 3 and 5, and the unrotated surface
    code of distance 3. Small enough that every syndrome is tested and solved exactly."""
    from ._codes import CssCode

    out = []
    for d in (3, 5, 7):
        logical = np.zeros(d, dtype=np.uint8)
        logical[0] = 1
        out.append(Instance.from_checks(_repetition(d), logical, p, f"repetition d={d}", d))
    h, l = _rotated_checks(3)
    out.append(Instance.from_checks(h, l, p, "rotated surface d=3", 3))
    code = CssCode.hypergraph_product(_repetition(3))
    out.append(Instance.from_checks(code.check_matrices()[1], code.logicals()[1], p, "unrotated surface d=3", 3))
    h, l = _rotated_checks(5)
    out.append(Instance.from_checks(h, l, p, "rotated surface d=5", 5))
    return out


def width_family(k: int, p: float = 0.01) -> Instance:
    """The paper's observable-width family: fault ``j`` flips detector ``j`` and observable
    ``j``, for ``j < k``, so every syndrome has one explanation."""
    return Instance(f"width k={k}", k, k, [((j,), (j,), p) for j in range(k)], 1)


# ---------------------------------------------------------------------------------------------
# The exact oracle


class _Optima(NamedTuple):
    weight: np.ndarray  # per syndrome (as an integer of detector bits), the minimum weight
    unique: np.ndarray  # whether one subset attains it
    argmin: np.ndarray  # a subset attaining it, as a bit mask over faults
    observables: np.ndarray  # that subset's observables, as a bit mask
    distance: Optional[int]


MAX_ORACLE_FAULTS = 22


def _optima(inst: Instance, weights: Optional[np.ndarray] = None) -> Optional[_Optima]:
    """Every fault subset's syndrome and weight; per syndrome the lightest. None above
    ``MAX_ORACLE_FAULTS`` faults or 62 detectors or observables."""
    m = inst.num_faults
    if m > MAX_ORACLE_FAULTS or inst.num_detectors > 62 or inst.num_observables > 62:
        return None
    w = inst.weights() if weights is None else weights
    syn = np.zeros(1 << m, dtype=np.int64)
    obs = np.zeros(1 << m, dtype=np.int64)
    tot = np.zeros(1 << m)
    size = np.zeros(1 << m, dtype=np.int16)
    for j, (dets, o, _) in enumerate(inst.faults):
        half = 1 << j
        sm = sum(1 << d for d in dets)
        om = sum(1 << k for k in o)
        syn[half:2 * half] = syn[:half] ^ sm
        obs[half:2 * half] = obs[:half] ^ om
        tot[half:2 * half] = tot[:half] + w[j]
        size[half:2 * half] = size[:half] + 1
    logical = (syn == 0) & (obs != 0)
    distance = int(size[logical].min()) if logical.any() else None
    order = np.lexsort((tot, syn))
    s_sorted = syn[order]
    first = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    starts = np.flatnonzero(first)
    nsyn = 1 << inst.num_detectors
    best_w = np.full(nsyn, np.inf)
    unique = np.zeros(nsyn, dtype=bool)
    argmin = np.full(nsyn, -1, dtype=np.int64)
    best_o = np.zeros(nsyn, dtype=np.int64)
    best = order[starts]
    keys = syn[best]
    best_w[keys] = tot[best]
    argmin[keys] = best
    best_o[keys] = obs[best]
    # Unique when the next subset with the same syndrome is strictly heavier.
    nxt = starts + 1
    has_next = nxt < len(order)
    has_next[has_next] = s_sorted[nxt[has_next]] == s_sorted[starts[has_next]]
    tol = 1e-9 * np.maximum(1.0, np.abs(tot[best]))
    second = np.where(has_next, tot[order[np.minimum(nxt, len(order) - 1)]], np.inf)
    unique[keys] = second - tot[best] > tol
    return _Optima(best_w, unique, argmin, best_o, distance)


def _mask(bits: Iterable[int]) -> int:
    return sum(1 << int(b) for b in bits)


def _subset(mask: int) -> tuple:
    return tuple(j for j in range(mask.bit_length()) if mask >> j & 1)


# ---------------------------------------------------------------------------------------------
# Certificates


@dataclass(frozen=True)
class Certificate:
    """Two corrections one decoder returned for one syndrome under two presentations of the
    same instance, carried into the caller's indices: both explain the syndrome and the
    ``heavier`` weighs more, so it is not a minimum-weight correction (Lemma 5.3). ``model`` is
    the base presentation's text; ``syndrome`` its fired detectors."""

    subject: str
    instance: str
    relation: str
    model: str
    syndrome: tuple
    lighter: tuple
    heavier: tuple
    lighter_weight: float
    heavier_weight: float

    def verify(self) -> bool:
        """Re-derive the certificate from its model text alone."""
        return verify_certificate(self.to_dict())

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "Certificate":
        d = json.loads(text)
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def verify_certificate(cert: dict) -> bool:
    """A certificate (as a dict, as `Certificate.to_dict` or the JSON gives it) checked from
    first principles: parse its model's ``error`` lines, check both corrections are in range,
    unrepeated and explain the syndrome, and that the weights recomputed from the probabilities
    differ as claimed. Uses nothing from this package; `tools/verify_certificates.py` is the
    same check as a standalone program."""
    faults = []
    for line in cert["model"].splitlines():
        line = line.split("#")[0].strip()
        if not line.startswith("error("):
            continue
        p = float(line[len("error("):line.index(")")])
        dets = [int(t[1:]) for t in line[line.index(")") + 1:].split() if t.startswith("D")]
        faults.append((p, dets))
    fired = set(cert["syndrome"])

    def total(c: list) -> Optional[float]:
        if len(set(c)) != len(c) or any(not 0 <= j < len(faults) for j in c):
            return None
        flips: set = set()
        for j in c:
            flips ^= set(faults[j][1])
        if flips != fired:
            return None
        return sum(math.log((1 - faults[j][0]) / faults[j][0]) for j in c)

    a, b = total(list(cert["lighter"])), total(list(cert["heavier"]))
    if a is None or b is None:
        return False
    return b - a > 1e-9 * max(1.0, abs(b)) and abs(a - cert["lighter_weight"]) < 1e-6 * max(1.0, abs(a)) and abs(b - cert["heavier_weight"]) < 1e-6 * max(1.0, abs(b))


# ---------------------------------------------------------------------------------------------
# Findings and the report

PROPERTIES = (
    "residual",  # L0: the correction explains the syndrome, in range and unrepeated
    "observable agreement",  # L0: returned observables are the correction's
    "weight agreement",  # L0: a returned weight is the correction's
    "observable width",  # L0: R3 and R3b, for decoders computing observables
    "purity",  # det: R4
    "exact optimality",  # L2, oracle
    "matching-core equivariance",  # L2, oracle: a unique optimum is returned under R1, R2, R6
    "split faithfulness",  # L2, oracle: R5 where the split leaves the optimum's class
    "bounded distance",  # L1, ground truth
)
OUTCOMES = ("pass", "fail", "legal skip", "unassessable", "not applicable", "excluded", "refused", "gave up")


class Finding(NamedTuple):
    """One adverse result: a conformance failure (against a mandatory or declared capability)
    or an instance-measured observation (a capability the decoder is silent on)."""

    subject: str
    instance: str
    property: str
    relation: str
    syndrome: tuple
    detail: str


@dataclass
class Report:
    """Per (subject, property), counts of each outcome over judged answers; the conformance
    failures; the instance-measured observations (adverse results against capabilities the
    subject is silent on, or declines); the certificates; and, per (subject, instance), whether
    each property's gate opened directly, by a proved implication, or its hypotheses held."""

    counts: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    certificates: list = field(default_factory=list)
    cells: dict = field(default_factory=dict)
    max_findings: int = 1000

    def _count(self, subject: str, prop: str, outcome: str, n: int = 1) -> None:
        self.counts.setdefault((subject, prop), Counter())[outcome] += n

    def _find(self, finding: Finding, conformance: bool) -> None:
        target = self.failures if conformance else self.observations
        if len(target) < self.max_findings:
            target.append(finding)

    def merge(self, other: "Report") -> "Report":
        for k, c in other.counts.items():
            self.counts.setdefault(k, Counter()).update(c)
        self.failures += other.failures
        self.observations += other.observations
        self.certificates += other.certificates
        self.cells.update(other.cells)
        return self

    def yields(self) -> dict:
        """Per property, the paper's three dispatch yields (Definition 4.7) over the (subject,
        instance) cells: direct (the subject declares the gate), derived (or a proved
        implication carries a declared capability to it on this instance), applicable (the
        property's hypotheses hold, whatever is declared)."""
        out = {}
        for prop in PROPERTIES:
            cells = [v for (s, i, p), v in self.cells.items() if p == prop]
            if cells:
                n = len(cells)
                out[prop] = tuple(sum(c[k] for c in cells) / n for k in range(3))
        return out

    def table(self) -> str:
        """The counts as a table, one row per subject and property."""
        rows = [("subject", "property") + OUTCOMES]
        for (s, p), c in sorted(self.counts.items(), key=lambda x: (x[0][0], PROPERTIES.index(x[0][1]))):
            rows.append((s, p) + tuple(str(c.get(o, 0)) for o in OUTCOMES))
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        return "\n".join("  ".join(x.ljust(w) for x, w in zip(r, widths)).rstrip() for r in rows)

    def __str__(self) -> str:
        return (
            f"{len(self.failures)} conformance failures, {len(self.observations)} observations, "
            f"{len(self.certificates)} certificates\n" + self.table()
        )


# ---------------------------------------------------------------------------------------------
# Checking


def _gate(decl: Declaration, capability: str) -> str:
    """What a property gated on ``capability`` returns: a verdict, a legal skip, or an
    unassessable measurement."""
    state = getattr(decl, capability)
    return {YES: "verdict", NO: "legal skip", SILENT: "unassessable"}[state]


def _normalise(raw: Any, inst: Instance) -> Answer:
    if isinstance(raw, Answer):
        a = raw
    else:
        a = Answer(tuple(int(j) for j in raw))
    corr = None if a.correction is None else tuple(int(j) for j in a.correction)
    obs = None if a.observables is None else tuple(int(x) & 1 for x in a.observables)
    return Answer(corr, obs, None if a.weight is None else float(a.weight), a.raised, bool(a.gave_up))


def _well_formed(c: tuple, inst: Instance) -> bool:
    return len(set(c)) == len(c) and all(0 <= j < inst.num_faults for j in c)


class _Run:
    """One subject on one instance."""

    def __init__(self, subject: Subject, inst: Instance, report: Report, quantum: float) -> None:
        self.s, self.inst, self.r = subject, inst, report
        self.decl = subject.declaration
        self.quantum = quantum
        self.w = inst.weights()

    def decoder(self, inst: Instance, prop: str) -> Optional[Callable]:
        """The subject built on ``inst``, its answers' exceptions caught; None, counted as
        refused under ``prop``, if it will not build (refusing is not misdecoding)."""
        try:
            dec = self.s.build(inst.model())
        except Exception as e:  # noqa: BLE001 - any refusal is recorded, not raised
            self.r._count(self.s.name, prop, "refused")
            self.r._find(Finding(self.s.name, inst.name, prop, "build", (), f"refused: {type(e).__name__}: {e}"), False)
            return None

        def call(s: np.ndarray) -> Any:
            try:
                return dec(s)
            except Exception as e:  # noqa: BLE001 - a raise on a feasible syndrome is judged
                return Answer(None, raised=f"{type(e).__name__}: {e}")

        return call

    def tol(self, *corrections: tuple, inst: Optional[Instance] = None) -> float:
        """How far two totals may differ before a verdict counts it: the subject's declared
        rounding per fault, and six significant digits (a weight kept in single precision, or
        rounded to integers undeclared, is still its weight). A certificate needs only that the
        totals differ."""
        inst = self.inst if inst is None else inst
        n = sum(len(c) for c in corrections)
        scale = max([1.0] + [abs(inst.weight_of(c)) for c in corrections])
        return self.quantum * n + 1e-6 * scale

    def verdict(self, prop: str, capability: Optional[str], ok: bool, relation: str, syn: tuple, detail: str) -> None:
        gate = "verdict" if capability is None else _gate(self.decl, capability)
        if ok:
            self.r._count(self.s.name, prop, "pass" if gate == "verdict" else gate)
            return
        if gate == "verdict":
            self.r._count(self.s.name, prop, "fail")
            self.r._find(Finding(self.s.name, self.inst.name, prop, relation, syn, detail), True)
        else:
            self.r._count(self.s.name, prop, gate)
            self.r._find(Finding(self.s.name, self.inst.name, prop, relation, syn, detail), False)

    def judge(self, inst: Instance, a: Answer, syndrome: np.ndarray, relation: str, back: Callable[[tuple], tuple]) -> Optional[tuple]:
        """The interface contract on one answer, under presentation ``inst``; returns its
        correction carried back into the base presentation's indices, if it is feasible."""
        syn = tuple(int(d) for d in np.flatnonzero(syndrome))
        if a.raised is not None:
            self.verdict("residual", None, False, relation, syn, f"raised on a syndrome a correction explains: {a.raised}")
            return None
        if a.gave_up:
            self.r._count(self.s.name, "residual", "gave up")
            return None
        if a.correction is None:
            self.r._count(self.s.name, "residual", "not applicable")
            return None
        c = a.correction
        ok = _well_formed(c, inst) and np.array_equal(inst.syndrome_of(c), syndrome)
        self.verdict("residual", None, ok, relation, syn, f"correction {list(c)} does not explain the syndrome")
        if not ok:
            return None
        if a.observables is not None:
            want = tuple(int(x) for x in inst.observables_of(c))
            self.verdict("observable agreement", None, a.observables == want, relation, syn, f"observables {list(a.observables)}, the correction's {list(want)}")
        if a.weight is not None and not inst.simple():
            # Parallel faults merged into one edge: the weight is the edge's, not any one fault's.
            self.r._count(self.s.name, "weight agreement", "not applicable")
        elif a.weight is not None:
            mine = inst.weight_of(c)
            self.verdict("weight agreement", None, abs(a.weight - mine) <= self.tol(c, inst=inst), relation, syn, f"weight {a.weight}, the correction's {mine}")
        return tuple(sorted(back(c)))

    def certify(self, relation: str, syndrome: np.ndarray, c1: tuple, c2: tuple) -> None:
        w1, w2 = self.inst.weight_of(c1), self.inst.weight_of(c2)
        if abs(w1 - w2) <= 1e-9 * max(1.0, abs(w1), abs(w2)):
            return
        light, heavy = (c1, c2) if w1 < w2 else (c2, c1)
        syn = tuple(int(d) for d in np.flatnonzero(syndrome))
        cert = Certificate(self.s.name, self.inst.name, relation, self.inst.text(), syn, light, heavy, min(w1, w2), max(w1, w2))
        if len(self.r.certificates) < self.r.max_findings:
            self.r.certificates.append(cert)
        # A certificate is arithmetic; against a declared minimum-weight decoder, and beyond its
        # declared rounding, it is a conformance failure.
        beyond = abs(w1 - w2) > self.tol(c1, c2)
        if beyond:
            self.verdict("exact optimality", "min_weight", False, relation, syn, f"{relation}: corrections of weight {min(w1, w2):.6g} and {max(w1, w2):.6g}")


def check(
    subject: Union[str, Subject],
    instance: Union[Instance, DetectorErrorModel, str],
    *,
    syndromes: Any = None,
    relabellings: int = 2,
    gammas: Sequence[float] = (0.5, 2.0),
    purity: int = 32,
    oracle: bool = True,
    seed: int = 0,
    report: Optional[Report] = None,
) -> Report:
    """Check one decoder (a `Subject`, or a name from `subjects()` or `panel()`) on one
    instance: every syndrome, if the instance has at most 16 detectors, else ``syndromes``
    (rows of detection events, or a count of random fault patterns' syndromes). Each syndrome
    is decoded under the base presentation and ``relabellings`` random fault orders (R1) and
    detector numberings (R2), with weights scaled by each of ``gammas`` (R6), split (R5, with
    the oracle), and the first ``purity`` fresh, twice, and after other calls (R4); decoders
    computing observables are also given a widened and an offset observable set (R3, R3b).
    With ``oracle`` (and at most 22 faults) answers are also held to the exact optimum and the
    instance's radius."""
    subj = _resolve(subject)
    inst = instance if isinstance(instance, Instance) else Instance.from_model(instance)
    rep = report if report is not None else Report()
    rng = np.random.default_rng(seed)
    decl = subj.declaration
    quantum = decl.weight_quantum
    run = _Run(subj, inst, rep, quantum)
    opt = _optima(inst) if oracle else None
    if opt is not None and inst.distance is None:
        inst.distance = opt.distance

    if syndromes is None:
        if inst.num_detectors > 16:
            raise ValueError(f"{inst.name} has {inst.num_detectors} detectors: give syndromes to test")
        rows = ((np.arange(1 << inst.num_detectors)[:, None] >> np.arange(inst.num_detectors)) & 1).astype(np.uint8)
    elif np.ndim(syndromes) == 0:
        n = int(syndromes)
        hits = rng.random((n, inst.num_faults)) < np.array([p for _, _, p in inst.faults])
        rows = np.array([inst.syndrome_of(np.flatnonzero(h)) for h in hits], dtype=np.uint8).reshape(n, inst.num_detectors)
    else:
        rows = np.asarray(syndromes, dtype=np.uint8)
    keys = rows.astype(np.int64) @ (1 << np.arange(inst.num_detectors, dtype=np.int64)) if inst.num_detectors <= 62 else None

    def feasible(key: int) -> bool:
        return opt is None or np.isfinite(opt.weight[key])

    # The base presentation: the interface contract, and the oracle's verdicts.
    dec = run.decoder(inst, "residual")
    if dec is None:
        return rep
    base: list = []
    for i, s in enumerate(rows):
        key = None if keys is None else int(keys[i])
        if key is not None and not feasible(key):
            base.append(None)
            continue
        a = _normalise(dec(s), inst)
        c = run.judge(inst, a, s, "base", lambda c: c)
        base.append((a, c))
        if c is None or opt is None:
            continue
        syn = tuple(int(d) for d in np.flatnonzero(s))
        best = float(opt.weight[key])
        mine = inst.weight_of(c)
        ok = mine <= best + run.tol(c, _subset(int(opt.argmin[key])))
        run.verdict("exact optimality", "min_weight", ok, "oracle", syn, f"weight {mine:.6g}, the optimum {best:.6g}")
        if ok and mine > best + 1e-9 * max(1.0, abs(best)):
            run.certify("oracle", s, _subset(int(opt.argmin[key])), c)
    _record_cells(rep, subj, inst, opt)

    # R1 and R2: other presentations of the same instance.
    for t in range(relabellings):
        perm = rng.permutation(inst.num_faults)
        r1 = inst.permute_faults(perm)
        _relation(run, r1, rows, rows, base, opt, keys, "R1 fault order", lambda c, perm=perm: tuple(int(perm[j]) for j in c))
        dperm = rng.permutation(inst.num_detectors)
        r2 = inst.permute_detectors(dperm)
        rows2 = np.zeros_like(rows)
        rows2[:, dperm] = rows
        _relation(run, r2, rows2, rows, base, opt, keys, "R2 detector labels", lambda c: c)
    # R6: weights scaled leave the minimum where it was.
    for g in gammas:
        _relation(run, inst.scaled(g), rows, rows, base, opt, keys, f"R6 weights x{g}", lambda c: c)
    # R5: each fault split in two; where the split's optimum stays in the base optimum's
    # class, an exact decoder's answer must too.
    if opt is not None:
        split = inst.split()
        w_split = np.array([_weight((1 - math.sqrt(1 - 2 * p)) / 2) for _, _, p in inst.faults])
        opt5 = _optima(inst, w_split)
        dec5 = run.decoder(split, "split faithfulness")
        for i, s in enumerate(rows if dec5 is not None else ()):
            if base[i] is None or base[i][1] is None:
                continue
            key = int(keys[i])
            syn = tuple(int(d) for d in np.flatnonzero(s))
            if opt5.observables[key] != opt.observables[key] or not opt.unique[key]:
                rep._count(subj.name, "split faithfulness", "excluded")
                continue
            c = run.judge(split, _normalise(dec5(s), split), s, "R5 split", _unsplit)
            if c is None:
                continue
            same = _mask(inst.observables_of(c).nonzero()[0]) == int(opt.observables[key])
            run.verdict("split faithfulness", "min_weight", same, "R5 split", syn, "the split moved the answer to another observable class")
    # R3 and R3b: observable width and offset.
    if subj.computes_observables:
        for name, wide, check_bits in (("R3 width", inst.widened(), lambda o, b: o[:-1] == b and o[-1] == 0), ("R3b offset", inst.offset(), lambda o, b: o[0] == 0 and o[1:] == b)):
            dw = run.decoder(wide, "observable width")
            for i, s in enumerate(rows[:64] if dw is not None else ()):
                if base[i] is None or base[i][0].observables is None:
                    continue
                a = _normalise(dw(s), wide)
                syn = tuple(int(d) for d in np.flatnonzero(s))
                ok = a.observables is not None and len(a.observables) == wide.num_observables and check_bits(a.observables, base[i][0].observables)
                run.verdict("observable width", None, ok, name, syn, f"observables {a.observables} after {base[i][0].observables}")
    else:
        rep._count(subj.name, "observable width", "not applicable")
    # R4: fresh, twice, and after other calls.
    for i, s in enumerate(rows[:purity]):
        if base[i] is None:
            continue
        if decl.randomised == YES and decl.deterministic != YES:
            # A declared random schedule: not tested (Definition 6.1).
            rep._count(subj.name, "purity", "legal skip")
            continue
        fresh = run.decoder(inst, "purity")
        if fresh is None:
            break
        a1 = _normalise(fresh(s), inst)
        a2 = _normalise(fresh(s), inst)
        syn = tuple(int(d) for d in np.flatnonzero(s))
        same = a1.correction == a2.correction == base[i][0].correction and a1.observables == a2.observables == base[i][0].observables
        run.verdict("purity", "deterministic", same, "R4 call order", syn, "fresh, again and after other calls the answers differ")
    # Bounded distance: every error of at most (d − 1)/2 faults is corrected.
    if inst.distance is not None and inst.distance >= 3:
        _bounded(run, inst)
    else:
        rep._count(subj.name, "bounded distance", "not applicable")
    return rep


def _relation(run: _Run, pres: Instance, rows_p: np.ndarray, rows: np.ndarray, base: list, opt: Optional[_Optima], keys: Any, relation: str, back: Callable) -> None:
    dec = run.decoder(pres, "residual")
    if dec is None:
        return
    for i, s in enumerate(rows_p):
        if base[i] is None:
            continue
        a = _normalise(dec(s), pres)
        c2 = run.judge(pres, a, s, relation, back)
        c1 = base[i][1]
        if c1 is None or c2 is None:
            continue
        run.certify(relation, rows[i], c1, c2)
        if opt is not None and opt.unique[int(keys[i])]:
            want = _subset(int(opt.argmin[int(keys[i])]))
            syn = tuple(int(d) for d in np.flatnonzero(rows[i]))
            run.verdict("matching-core equivariance", "min_weight", c1 == want and c2 == want, relation, syn, f"unique optimum {list(want)}, returned {list(c1)} and {list(c2)}")


def _unsplit(c: tuple) -> tuple:
    """A correction on the split instance as one on the base: fault j where one of its halves
    is used (both cancel)."""
    halves = Counter(j // 2 for j in c)
    return tuple(j for j, n in halves.items() if n % 2)


def _bounded(run: _Run, inst: Instance) -> None:
    from itertools import combinations

    t = (inst.distance - 1) // 2
    dec = run.decoder(inst, "bounded distance")
    if dec is None:
        return
    decl = run.decl
    # The gate: L1 declared, or L2 declared on a code-capacity instance whose weights order
    # corrections by their size (Remark B.2), the derived assurance; else the declaration's.
    derived = decl.bounded_distance != YES and decl.min_weight == YES and inst.uniform()
    for size in range(1, t + 1):
        for e in combinations(range(inst.num_faults), size):
            s = inst.syndrome_of(e)
            a = _normalise(dec(s), inst)
            syn = tuple(int(d) for d in np.flatnonzero(s))
            ok = a.correction is not None and _well_formed(a.correction, inst) and np.array_equal(inst.syndrome_of(a.correction), s) and not (inst.observables_of(a.correction) ^ inst.observables_of(e)).any()
            if derived:
                run.verdict("bounded distance", None, ok, "injected", syn, f"error {list(e)} corrected by {list(a.correction or ())}")
            else:
                run.verdict("bounded distance", "bounded_distance", ok, "injected", syn, f"error {list(e)} corrected by {list(a.correction or ())}")


def _record_cells(rep: Report, subj: Subject, inst: Instance, opt: Optional[_Optima]) -> None:
    d = subj.declaration
    has_oracle = opt is not None
    unique = has_oracle and bool(opt.unique[np.isfinite(opt.weight)].any())
    radius = inst.distance is not None and inst.distance >= 3
    l2 = d.min_weight == YES
    for prop, direct, derived, applicable in (
        ("residual", True, True, True),
        ("exact optimality", l2, l2, has_oracle),
        ("matching-core equivariance", l2 and unique, l2 and unique, unique),
        ("bounded distance", d.bounded_distance == YES and radius, (d.bounded_distance == YES or l2 and inst.uniform()) and radius, radius),
    ):
        rep.cells[subj.name, inst.name, prop] = (bool(direct and applicable), bool(derived and applicable), bool(applicable))


def audit(subjects_: Optional[Iterable[Union[str, Subject]]] = None, instances: Optional[Iterable[Instance]] = None, **options: Any) -> Report:
    """`check` every subject (default: `panel()` and `subjects()`) on every instance (default:
    `population()`), into one report; and the observable-width family (k from 1 to 65, either
    side of the 32- and 64-bit words) for those computing observables."""
    names = list(subjects_) if subjects_ is not None else list(panel()) + list(subjects())
    insts = list(instances) if instances is not None else population()
    rep = Report()
    for s in names:
        subj = _resolve(s)
        for inst in insts:
            check(subj, inst, report=rep, **options)
        if subj.computes_observables:
            for k in (1, 2, 8, 30, 31, 32, 33, 62, 63, 64, 65):
                _width(subj, width_family(k), rep)
    return rep


def _width(subj: Subject, inst: Instance, rep: Report) -> None:
    run = _Run(subj, inst, rep, subj.declaration.weight_quantum)
    dec = run.decoder(inst, "observable width")
    if dec is None:
        return
    for j in (0, inst.num_faults - 1):
        s = np.zeros(inst.num_detectors, dtype=np.uint8)
        s[j] = 1
        a = _normalise(dec(s), inst)
        want = tuple(int(k == j) for k in range(inst.num_observables))
        run.verdict("observable width", None, a.observables == want, f"width {inst.num_observables}", (j,), f"observables {a.observables}, want {want}")


# ---------------------------------------------------------------------------------------------
# Subjects

_Q = 1 << 20  # integer weights for the solvers taking them: 2^20 a unit, even


def _int_weights(model: DetectorErrorModel) -> tuple:
    inst = Instance.from_model(model, merge=False)
    return inst, [2 * int(round(w * _Q / 2)) for w in inst.weights()]


def _matching(correlated: bool) -> Callable:
    def build(model: DetectorErrorModel) -> Callable:
        from ._matching import Matching

        inst = Instance.from_model(model, merge=False)
        m = Matching.from_detector_error_model(model, enable_correlations=correlated)
        index: dict = {}
        for j, (dets, _, _) in enumerate(inst.faults):
            index.setdefault((dets[0], dets[1] if len(dets) == 2 else -1), j)

        def decode(s: np.ndarray) -> Answer:
            edges = m.decode_to_edges_array(s, enable_correlations=correlated) if correlated else m.decode_to_edges_array(s)
            corr = tuple(index[(min(u, v), max(u, v)) if v >= 0 and u >= 0 else (max(u, v), -1)] for u, v in edges)
            obs, w = m.decode(s, return_weight=True, enable_correlations=correlated)
            return Answer(corr, tuple(int(x) for x in obs), w)

        return decode

    return build


def _fusion(model: DetectorErrorModel) -> Callable:
    from . import fusion_blossom as fb

    inst, w = _int_weights(model)
    nd = inst.num_detectors
    edges = [(dets[0], dets[1] if len(dets) == 2 else nd, wj) for (dets, _, _), wj in zip(inst.faults, w)]
    solver = fb.SolverSerial(fb.SolverInitializer(nd + 1, edges, [nd]))

    def decode(s: np.ndarray) -> tuple:
        solver.solve(fb.SyndromePattern([int(d) for d in np.flatnonzero(s)]))
        out = tuple(solver.subgraph())
        solver.clear()
        return out

    return decode


def _mwpf(kind: str) -> Callable:
    def build(model: DetectorErrorModel) -> Callable:
        from . import mwpf

        inst, w = _int_weights(model)
        edges = [mwpf.HyperEdge(list(dets), wj) for (dets, _, _), wj in zip(inst.faults, w)]
        solver = getattr(mwpf, kind)(mwpf.SolverInitializer(inst.num_detectors, edges))

        def decode(s: np.ndarray) -> tuple:
            solver.solve(mwpf.SyndromePattern([int(d) for d in np.flatnonzero(s)]))
            out = tuple(solver.subgraph())
            solver.clear()
            return out

        return decode

    return build


def _checks(model: DetectorErrorModel) -> tuple:
    inst = Instance.from_model(model, merge=False)
    h = np.zeros((inst.num_detectors, inst.num_faults), dtype=np.uint8)
    for j, (dets, _, _) in enumerate(inst.faults):
        h[list(dets), j] = 1
    return h, [p for _, _, p in inst.faults]


def _pcm(make: Callable) -> Callable:
    def build(model: DetectorErrorModel) -> Callable:
        h, p = _checks(model)
        dec = make(h, p)

        def decode(s: np.ndarray) -> Answer:
            corr = tuple(int(j) for j in np.flatnonzero(dec.decode(np.asarray(s, dtype=np.uint8))))
            # (Relay-BP says when no leg converged, and then returns a guess.)
            return Answer(corr, gave_up=getattr(dec, "converge", True) is False)

        return decode

    return build


def _search(**options: Any) -> Callable:
    def build(model: DetectorErrorModel) -> Callable:
        from ._decoders import SearchDecoder

        dec = SearchDecoder(model, merge_errors=False, **options)

        def decode(s: np.ndarray) -> Answer:
            faults = tuple(dec.decode_to_faults(s))
            # (A search that gave up returns no faults.)
            return Answer(faults, gave_up=not faults and bool(np.any(s)))

        return decode

    return build


def _tesseract(model: DetectorErrorModel) -> Callable:
    from .tesseract_decoder import tesseract

    dec = tesseract.TesseractConfig(dem=model, merge_errors=False).compile_decoder()

    def decode(s: np.ndarray) -> Answer:
        corr = tuple(int(j) for j in dec.decode_to_errors(np.asarray(s, dtype=bool)))
        return Answer(corr, gave_up=bool(dec.low_confidence_flag))

    return decode


def _random_serial(h: np.ndarray, p: list) -> Any:
    from .ldpc import BpOsdDecoder

    dec = BpOsdDecoder(h, error_channel=p, max_iter=h.shape[1], bp_method="product_sum", osd_method="osd0", schedule="serial")
    dec.random_schedule_seed = 1  # (ldpc's switch to a random serial order, drawn afresh each decode)
    return dec


_AUDIT = "the upstream library's documentation, as audited by Shen and Zhong (arXiv 2609.07035, Table 4)"


def panel() -> dict:
    """The paper's panel (its Table 4), on this package's ports of the five libraries, each held
    to the upstream library's audited declarations: PyMatching (``pm``, and ``pmc`` with
    correlations) and fusion-blossom (``fb``) declare minimum weight; ldpc's BP+OSD-0 (``ldpc``,
    and ``ldpc-r`` with a random serial schedule), MWPF's three solvers (``mwpf-j``, ``mwpf-s``,
    ``mwpf-u``) and Tesseract (``tess``) are silent on every capability."""
    from .ldpc import BpOsdDecoder

    mw = Declaration(min_weight=YES, source=_AUDIT)
    silent = Declaration(source=_AUDIT)
    return {
        "pm": Subject("pm", _matching(False), mw, True, "pymatching 2.4.0 (this package's Matching), sparse blossom"),
        "pmc": Subject("pmc", _matching(True), silent, True, "pymatching 2.4.0, correlations enabled"),
        "fb": Subject("fb", _fusion, mw, False, "fusion-blossom 0.2.13 (stabilizer_qec.fusion_blossom)"),
        "ldpc": Subject("ldpc", _pcm(lambda h, p: BpOsdDecoder(h, error_channel=p, max_iter=h.shape[1], bp_method="product_sum", osd_method="osd0")), silent, False, "ldpc 2.4.1, BP with OSD-0"),
        "ldpc-r": Subject("ldpc-r", _pcm(_random_serial), Declaration(randomised=YES, source=_AUDIT), False, "ldpc 2.4.1, randomised serial schedule"),
        "mwpf-j": Subject("mwpf-j", _mwpf("SolverSerialJointSingleHair"), silent, False, "mwpf 0.2.12, serial joint single-hair"),
        "mwpf-s": Subject("mwpf-s", _mwpf("SolverSerialSingleHair"), silent, False, "mwpf 0.2.12, serial single-hair"),
        "mwpf-u": Subject("mwpf-u", _mwpf("SolverSerialUnionFind"), silent, False, "mwpf 0.2.12, serial union-find"),
        "tess": Subject("tess", _tesseract, silent, False, "tesseract-decoder (stabilizer_qec.tesseract_decoder)"),
    }


_OURS = "this package's documentation"


def subjects() -> dict:
    """This package's decoders, each with a declaration answering every capability question.
    ``matching`` (the sparse-blossom matcher) is minimum-weight on its integer weights (each
    within 2^-20 of ln((1 − p)/p)) and deterministic; ``search_exact`` (A* with no beam, no queue
    bound, and revisits) is minimum-weight on the weights as given. Minimum weight is not
    bounded-distance correctness (an unlikely fault can outweigh several likely ones), and
    neither is maximum likelihood (which sums over a class); those are declined. The rest
    (correlated matching, Tesseract's defaults, BP+OSD, BP+LSD, union-find, belief-find,
    Relay-BP) decline minimum weight; all are deterministic but Relay-BP, whose memory strengths
    come from one seeded stream across its decodes (randomised)."""
    from ._decoders import RelayBpDecoder
    from .ldpc import BeliefFindDecoder, BpLsdDecoder, BpOsdDecoder, UnionFindDecoder

    exact = dict(bounded_distance=NO, min_weight=YES, max_likelihood=NO, deterministic=YES, randomised=NO, source=_OURS)
    approx = dict(bounded_distance=NO, min_weight=NO, max_likelihood=NO, deterministic=YES, randomised=NO, int_weights=NO, source=_OURS)
    return {
        "matching": Subject("matching", _matching(False), Declaration(int_weights=YES, weight_quantum=1 / _Q, **exact), True, "Matching: sparse blossom"),
        "matching_correlated": Subject("matching_correlated", _matching(True), Declaration(**{**approx, "int_weights": YES}), True, "Matching with enable_correlations"),
        "search_exact": Subject("search_exact", _search(beam=None, pqlimit=None, no_revisit_dets=False, num_det_orders=1), Declaration(int_weights=NO, **exact), False, "SearchDecoder, exact"),
        "search": Subject("search", _search(), Declaration(**approx), False, "SearchDecoder, Tesseract's defaults"),
        "fusion_blossom": Subject("fusion_blossom", _fusion, Declaration(int_weights=YES, weight_quantum=1 / _Q, **exact), False, "stabilizer_qec.fusion_blossom, serial"),
        "bposd": Subject("bposd", _pcm(lambda h, p: BpOsdDecoder(h, error_channel=p, max_iter=30, bp_method="product_sum", osd_method="osd_cs", osd_order=4)), Declaration(**approx), False, "BpOsdDecoder, BP+OSD-CS(4)"),
        "bplsd": Subject("bplsd", _pcm(lambda h, p: BpLsdDecoder(h, error_channel=p, max_iter=30, bp_method="product_sum", lsd_order=4)), Declaration(**approx), False, "BpLsdDecoder, BP+LSD-4"),
        "union_find": Subject("union_find", _pcm(lambda h, p: UnionFindDecoder(h, uf_method="peeling")), Declaration(**approx), False, "UnionFindDecoder, peeling"),
        "belief_find": Subject("belief_find", _pcm(lambda h, p: BeliefFindDecoder(h, error_channel=p, max_iter=30, bp_method="product_sum")), Declaration(**approx), False, "BeliefFindDecoder"),
        "relay_bp": Subject("relay_bp", _pcm(lambda h, p: RelayBpDecoder(h, error_channel=p, seed=1)), Declaration(**{**approx, "deterministic": NO, "randomised": YES}), False, "RelayBpDecoder, seeded"),
    }


def _resolve(subject: Union[str, Subject]) -> Subject:
    if isinstance(subject, Subject):
        return subject
    for table in (subjects, panel):
        t = table()
        if subject in t:
            return t[subject]
    raise ValueError(f"no subject {subject!r}; the subjects are {', '.join(list(subjects()) + list(panel()))}")
