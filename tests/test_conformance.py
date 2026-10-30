"""Conformance checks for decoders (Shen and Zhong, arXiv 2609.07035), against the paper's claims
and on decoders broken on purpose: every interface defect the paper injects is caught by the
property that reads it, verdicts follow declarations, and certificates are re-derivable from the
model alone."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import conformance as cf

POP = cf.population()
REPO = Path(__file__).resolve().parents[1]


def test_the_papers_population():
    # Table 3: detectors and edges of each instance; every syndrome tested.
    shape = [(i.name, i.num_detectors, i.num_faults, i.distance) for i in POP]
    assert shape == [
        ("repetition d=3", 2, 3, 3), ("repetition d=5", 4, 5, 5), ("repetition d=7", 6, 7, 7),
        ("rotated surface d=3", 4, 7, 3), ("unrotated surface d=3", 6, 13, 3), ("rotated surface d=5", 12, 21, 5),
    ]
    assert sum(1 << i.num_detectors for i in POP) == 4260
    # The oracle: every rotated d=5 syndrome is explained, 2867 with one optimum, as the paper's.
    opt = cf._optima(POP[5])
    assert np.isfinite(opt.weight).all() and opt.distance == 5
    assert sum(int(cf._optima(i).unique.sum()) for i in POP) == 2867


@pytest.mark.parametrize("name", ["matching", "fusion_blossom", "search_exact", "pm", "fb"])
def test_exact_decoders_pass_everything(name):
    rep = cf.Report()
    for inst in POP:
        cf.check(name, inst, report=rep)
    assert rep.failures == [] and rep.certificates == []
    c = rep.counts
    assert c[name, "exact optimality"]["pass"] == 4260
    assert c[name, "matching-core equivariance"]["fail"] == 0 and c[name, "matching-core equivariance"]["pass"] > 10_000
    assert c[name, "residual"]["fail"] == 0


def test_mwpf_union_find_is_presentation_sensitive():
    # The paper's finding: mwpf's union-find returns the optimum under one numbering and a
    # heavier correction under another. Certified with no solver, and only an observation, as
    # mwpf declares nothing.
    rep = cf.check("mwpf-u", POP[5])
    assert rep.failures == []
    certs = [c for c in rep.certificates if c.relation.startswith(("R1", "R2"))]
    assert len(certs) > 20 and all(c.verify() for c in certs)
    assert max(c.heavier_weight / c.lighter_weight for c in certs) > 1.2
    assert rep.counts["mwpf-u", "matching-core equivariance"]["unassessable"] > 0


def broken(name, build, declaration=cf.Declaration(), observables=False):
    return cf.Subject(name, build, declaration, observables)


def matching_build(model):
    return cf.subjects()["matching"].build(model)


def test_injected_defects_are_caught_by_the_property_reading_them():
    inst = POP[4]

    # A correction in another index space (the faults reversed) fails the residual check.
    def reversed_indices(model):
        dec, m = matching_build(model), cf.Instance.from_model(model, merge=False).num_faults
        return lambda s: cf.Answer(tuple(m - 1 - j for j in dec(s).correction))

    rep = cf.check(broken("reversed", reversed_indices), inst)
    assert rep.counts["reversed", "residual"]["fail"] > 0

    # A misreported weight fails weight agreement, and nothing else.
    def heavy(model):
        dec = matching_build(model)
        return lambda s: dec(s)._replace(weight=(dec(s).weight or 0) * 1.01 + 0.5 * bool(np.any(s)))

    rep = cf.check(broken("heavy", heavy, observables=True), inst)
    assert rep.counts["heavy", "weight agreement"]["fail"] > 0
    assert {f.property for f in rep.failures} == {"weight agreement"}

    # Observables truncated to the first: the width relations catch it on the wide family.
    def truncated(model):
        dec = matching_build(model)
        return lambda s: dec(s)._replace(observables=dec(s).observables[:1])

    rep = cf.Report()
    cf._width(broken("truncated", truncated, observables=True), cf.width_family(8), rep)
    assert rep.counts["truncated", "observable width"]["fail"] > 0

    # Observables flipped against the correction fail observable agreement.
    def flipped(model):
        dec = matching_build(model)
        return lambda s: dec(s)._replace(observables=tuple(1 - x for x in dec(s).observables))

    rep = cf.check(broken("flipped", flipped, observables=True), inst, relabellings=0, gammas=())
    assert rep.counts["flipped", "observable agreement"]["fail"] > 0


def logical_path(model):
    """One fault per distinct detector set: on a repetition code, the whole path from boundary to
    boundary, flipping no detector and the observable, under any presentation."""
    me = cf.Instance.from_model(model, merge=False)
    first = {}
    for j, (dets, _, _) in enumerate(me.faults):
        first.setdefault(dets, j)
    return set(first.values())


def test_verdicts_follow_declarations():
    # A correction with the logical path added (still explaining the syndrome, much heavier) is
    # a failure against a decoder declaring minimum weight, unassessable for one silent on it,
    # and a legal skip for one declining it.
    inst = POP[1]

    def sloppy(model):
        dec, path = matching_build(model), logical_path(model)
        return lambda s: cf.Answer(tuple(sorted(set(dec(s).correction) ^ path)))

    for state, outcome in ((cf.YES, "fail"), (cf.SILENT, "unassessable"), (cf.NO, "legal skip")):
        rep = cf.check(broken("sloppy", sloppy, cf.Declaration(min_weight=state)), inst)
        assert rep.counts["sloppy", "residual"]["fail"] == 0
        assert rep.counts["sloppy", "exact optimality"][outcome] > 0
        assert (len(rep.failures) > 0) == (state == cf.YES)
        assert all(c.verify() for c in rep.certificates)

    # Purity: a decoder whose answers depend on its call count.
    def flaky(model):
        dec, path, calls = matching_build(model), logical_path(model), [0]

        def decode(s):
            calls[0] += 1
            c = set(dec(s).correction)
            return cf.Answer(tuple(sorted(c ^ path if calls[0] % 2 else c)))

        return decode

    for decl, outcome in ((cf.Declaration(deterministic=cf.YES), "fail"), (cf.Declaration(randomised=cf.YES), "legal skip"), (cf.Declaration(), "unassessable")):
        rep = cf.check(broken("flaky", flaky, decl), inst)
        assert rep.counts["flaky", "purity"][outcome] > 0


def test_bounded_distance_derived_from_minimum_weight_on_hamming_ordered_instances():
    # L2 => L1 where every weight is equal (Remark B.2): matching declines L1 in general, yet
    # on the repetition and unrotated codes it is held to it; on the rotated d=3 code, whose
    # merged faults weigh differently, the declaration's legal skip stands.
    rep = cf.Report()
    for inst in POP:
        cf.check("matching", inst, report=rep)
    c = rep.counts["matching", "bounded distance"]
    assert c["pass"] > 0 and c["fail"] == 0 and c["legal skip"] > 0
    y = rep.yields()
    assert y["bounded distance"][0] == 0 and y["bounded distance"][1] > 0 and y["residual"] == (1.0, 1.0, 1.0)


def test_certificates_round_trip_and_the_standalone_verifier(tmp_path):
    rep = cf.check("mwpf-u", POP[5], relabellings=1, gammas=())
    certs = rep.certificates[:20]
    assert certs and all(cf.Certificate.from_json(c.to_json()) == c for c in certs)
    # Tampered: the lighter correction no longer explains the syndrome, or the order is wrong.
    d = certs[0].to_dict()
    assert not cf.verify_certificate({**d, "lighter": d["lighter"][1:]})
    assert not cf.verify_certificate({**d, "lighter": d["heavier"], "heavier": d["lighter"]})
    path = tmp_path / "certs.jsonl"
    path.write_text("\n".join(c.to_json() for c in certs))
    out = subprocess.run([sys.executable, "-I", str(REPO / "tools" / "verify_certificates.py"), str(path)], capture_output=True, text=True)
    assert out.returncode == 0 and f"{len(certs)} verified, 0 failed" in out.stdout
    path.write_text(json.dumps([{**d, "lighter": d["lighter"][1:]}]))
    assert subprocess.run([sys.executable, "-I", str(REPO / "tools" / "verify_certificates.py"), str(path)], capture_output=True).returncode == 1


def test_found_by_these_checks():
    # Three defects these checks found in this package's ports, now fixed: ldpc's peeling
    # union-find left 78 rotated d=5 syndromes unexplained (ldpc 2.4.1 does too); correlated
    # matching's edges were the uncorrelated matching's; and Relay-BP's declared determinism
    # was wrong (its seeded stream runs across decodes).
    rep = cf.check("union_find", POP[5], relabellings=0, gammas=(), oracle=False)
    assert rep.counts["union_find", "residual"]["fail"] == 0
    rep = cf.check("matching_correlated", POP[4])
    assert rep.counts["matching_correlated", "observable agreement"]["fail"] == 0
    assert cf.subjects()["relay_bp"].declaration.randomised == cf.YES
    rep = cf.check("relay_bp", POP[2])
    assert rep.failures == [] and rep.counts["relay_bp", "residual"]["gave up"] > 0


def test_observable_width_across_machine_words():
    rep = cf.Report()
    for k in (1, 31, 32, 33, 63, 64, 65):
        cf._width(cf.subjects()["matching"], cf.width_family(k), rep)
        cf._width(cf.subjects()["matching_correlated"], cf.width_family(k), rep)
    assert rep.counts["matching", "observable width"] == {"pass": 14}
    # Correlated matching takes at most 64 observables, and says so: refused, not misdecoded.
    assert rep.counts["matching_correlated", "observable width"] == {"pass": 12, "refused": 1}


def test_models_sampled_syndromes_and_errors():
    inst = cf.Instance.from_model("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.2) D1")
    assert inst.num_faults == 3 and inst.distance is None
    rep = cf.check("matching", inst)
    assert rep.failures == [] and inst.distance == 3
    with pytest.raises(ValueError, match="decomposed"):
        cf.Instance.from_model("error(0.1) D0 D1 ^ D2")
    with pytest.raises(ValueError, match="one or two detectors"):
        cf.Instance.from_model("error(0.1) D0 D1 D2")
    with pytest.raises(ValueError, match="different observables"):
        cf.Instance.from_model("error(0.1) D0 L0\nerror(0.1) D0")
    big = cf.Instance("ring", 20, 1, [((j, (j + 1) % 20), (0,) if j == 0 else (), 0.01) for j in range(20)])
    with pytest.raises(ValueError, match="give syndromes"):
        cf.check("matching", big, oracle=False)
    rep = cf.check("matching", big, syndromes=300, oracle=False, seed=2)
    assert rep.failures == [] and rep.counts["matching", "residual"]["pass"] >= 300
    with pytest.raises(ValueError, match="no subject"):
        cf.check("nope", inst)
    with pytest.raises(ValueError, match="'yes', 'no' or 'silent'"):
        cf.Declaration(min_weight="maybe")


def test_audit_report():
    pytest.importorskip("scipy")  # (ldpc and bposd are the ldpc port's, which needs it)
    rep = cf.audit(["pm", "ldpc", "bposd"], POP[:3])
    text = str(rep)
    assert "pm" in text and "exact optimality" in text and "conformance failures" in text
    assert rep.failures == []
    y = rep.yields()
    assert y["exact optimality"][0] == pytest.approx(1 / 3)
