"""The PyPI long description (python/README.md, frozen into every wheel) and the site (index.html)
say what the package does and what its committed data hold: their claims are checked against the
sampler's behaviour and against the rows of data/coherent/."""

from __future__ import annotations

import html
import importlib.util
import json
import pathlib
import re
import warnings

import stabilizer_qec as sq

ROOT = pathlib.Path(__file__).resolve().parents[1]
LONG_DESCRIPTION = ROOT / "python" / "README.md"
SITE = ROOT / "index.html"
COHERENT = ROOT / "data" / "coherent"
# The data files as the prose names them.
NAMES = {"sweep": "sweep", "crosstalk": "crosstalk", "validate": "validation", "capacity": "capacity"}


def prose(path: pathlib.Path) -> str:
    """The file's text without markup, its whitespace collapsed (so a sentence may wrap)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".html":
        text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return " ".join(text.split())


def sentences(text: str) -> list:
    # A full stop before a space and a capital ends a sentence (version numbers, 1.8.0, do not),
    # and a semicolon ends a clause.
    return re.split(r"(?<=\.)\s+(?=[A-Z`])|;\s+", text)


def current_revision() -> str:
    spec = importlib.util.spec_from_file_location("readme_tables", ROOT / "tools" / "readme_tables.py")
    tables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    return tables.COHERENT_GENERATION_REVISION


def archival_rows() -> dict:
    """Per data file, the rows the current tables refuse (no current generation revision)."""
    revision = current_revision()
    out = {}
    for name in NAMES:
        data = json.loads((COHERENT / f"{name}.json").read_text(encoding="utf-8"))
        rows = [dict(row, version=row.get("version", data.get("version"))) for row in data["rows"] if row.get("generation_revision") != revision]
        if rows:
            out[name] = rows
    return out


def pre_correction_sentences(text: str) -> list:
    """The sentences that say which coherent rows predate the merged-proposal correction."""
    return [s for s in sentences(text) if "merged-proposal correction" in s and re.search(r"\b(before|predate|pre-correction)", s)]


def check_archival_claims(path: pathlib.Path) -> None:
    archival = archival_rows()
    assert archival, "data/coherent/ has no archival rows: the prose's archival sentence needs rewriting"
    claims = pre_correction_sentences(prose(path))
    assert claims, f"{path.name} does not say which coherent rows predate the merged-proposal correction"
    for claim in claims:
        named = {name for name, word in NAMES.items() if re.search(rf"\b{word}\b", claim)}
        assert named == set(archival), f"{path.name} names {sorted(named)} as pre-correction data, but only {sorted(archival)} hold such rows: {claim!r}"
        versions = set(re.findall(r"\b\d+\.\d+\.\d+\b", claim))
        assert versions == {row["version"] for rows in archival.values() for row in rows}, f"{path.name}: {claim!r}"
        numbers = {int(n) for n in re.findall(r"(?<![\d.])\d+(?![\d.])", claim)}
        assert {len(rows) for rows in archival.values()} <= numbers, f"{path.name} does not count the archival rows: {claim!r}"
    text = prose(path)
    for stale in ("until remeasured", "labeled 2.2.0", "stamped 2.2.0", "must refresh", "will refresh"):
        assert stale not in text, f"{path.name} still describes the remeasurement as pending: {stale!r}"


def test_the_long_description_names_only_the_archival_coherent_rows():
    check_archival_claims(LONG_DESCRIPTION)


def test_the_site_names_only_the_archival_coherent_rows():
    check_archival_claims(SITE)


def joint_surface_memory() -> sq.Circuit:
    """A d = 3, three-round rotated X memory with a Z rotation on every data qubit each round, and a
    T, T-dagger pair (an identity, but non-Clifford gates): the joint sampler, past twelve generators."""
    lines = str(sq.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3)).splitlines()
    data = " ".join([line for line in lines if line.strip().startswith("MX ")][-1].split()[1:])
    out = []
    for i, line in enumerate(lines):
        if line.strip().startswith("H ") and i + 2 < len(lines) and lines[i + 1].strip() == "TICK" and lines[i + 2].strip().startswith("CX "):
            out.append(f"I_ERROR[R_Z(theta=0.02)] {data}")
        out.append(line)
    return sq.Circuit("\n".join(out).replace("TICK", "TICK\nI[T] 1\nI[T_DAG] 1", 1))


def test_the_joint_sampler_truncation_is_documented_as_silent():
    """The joint sampler has no contraction plan: past twelve generators it truncates its class sums
    whatever ``max_contraction`` is, without a warning, and only ``exact_weights`` says so. The long
    description and ``compile_coherent_sampler``'s docstring must not promise a warning there."""
    c = joint_surface_memory()
    assert c.num_non_clifford_gates() == 2
    for budget in (65536, 2**31, 0):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            s = c.compile_coherent_sampler(seed=1, max_contraction=budget)
        assert s.num_generators > 12 and s.kernel_complete and not s.exact_weights and s.contraction_width is None
        assert caught == []
    text = prose(LONG_DESCRIPTION)
    assert "larger/incomplete sums are approximations and warn" not in text
    joint = [s for s in sentences(text) if "joint sampler" in s]
    assert any("`max_contraction`" in s and "twelve generators" in s and "without a warning" in s for s in joint), joint
    doc = " ".join(sq.Circuit.compile_coherent_sampler.__doc__.split())
    joint = [s for s in sentences(doc) if "joint sampler" in s]
    assert any("``max_contraction``" in s and "twelve generators" in s and "without a warning" in s for s in joint), joint
