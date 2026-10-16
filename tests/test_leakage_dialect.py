"""Deltakit's leakage dialect (LEAKAGE, RELAX, HERALD_LEAKAGE_EVENT), its measurements and
leakage flags, its deinstrumentation (``without_leakage``), and leakage-aware decoding from
heralds around matching, union-find, local clustering and the BP decoders."""

import pickle

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec._decoders import _kept_detectors

SMALL = """
R 0 1
LEAKAGE(0.1) 0
HERALD_LEAKAGE_EVENT(0.01) 0 !1
DETECTOR rec[-2]
CX 0 1
M 1
DETECTOR rec[-1]
RELAX(0.5) 0
"""


def deltakit_memory(distance=5, rounds=5, p=0.001, pl=0.002, flip=0.0, herald_detectors=True):
    """A memory with Deltakit's leakage after every CX (and relaxation), and a herald (with a
    detector reading it, as Deltakit writes them) before every measurement; the memory's own
    lookbacks renumbered past the heralds."""
    base = sq.memory_circuit(distance=distance, rounds=rounds, p=p)
    out, new_of_old, new = [], [], 0
    for line in str(base).splitlines():
        words = line.split()
        name = words[0].split("(")[0] if words else ""
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            old = len(new_of_old)
            fixed = []
            for w in words:
                if w.startswith("rec[-"):
                    w = f"rec[-{new - new_of_old[old - int(w[5:-1])]}]"
                fixed.append(w)
            out.append(" ".join(fixed))
            continue
        if name == "M":
            qs = words[1:]
            out.append(f"HERALD_LEAKAGE_EVENT({flip}) " + " ".join(qs))
            new += len(qs)
            if herald_detectors:
                out.extend(f"DETECTOR rec[-{len(qs) - i}]" for i in range(len(qs)))
            out.append(line)
            new_of_old.extend(range(new, new + len(qs)))
            new += len(qs)
            continue
        out.append(line)
        if name == "CX":
            out.append(f"LEAKAGE({pl}) " + " ".join(words[1:]))
            out.append(f"RELAX({pl}) " + " ".join(words[1:]))
    return sq.Circuit("\n".join(out))


def test_the_dialect_is_read_as_tagged_lines_stim_runs_and_written_back():
    c = sq.Circuit(SMALL)
    assert str(c).splitlines()[1:3] == ["I_ERROR[LEAKAGE(p=0.1)] 0", "MPAD[HERALD_LEAKAGE_EVENT(qubits=0 1)](0.01) 0 1"]
    assert c.to_deltakit_text().split() == SMALL.split()
    assert sq.Circuit(c.to_deltakit_text()) == c
    # The engines that run no leakage see an identity and a padded record.
    assert c.num_measurements == 3 and c.num_detectors == 2
    # (Stim's Linux builds print the probability from a long double: 0.01000000000000000021.)
    error, *rest = str(c.detector_error_model().without_tags()).split()
    assert float(error[6:-1]) == pytest.approx(0.01) and rest == "D0 detector D1".split()
    assert str(c.without_leakage()).split() == "R 0 1 CX 0 1 M 1 DETECTOR rec[-1]".split()
    for bad, match in [
        ("LEAKAGE 0", "one argument"),
        ("RELAX(1.5) 0", "probability"),
        ("RELAX(0.1) !0", "inverted"),
        ("HERALD_LEAKAGE_EVENT(0.1, 0.2) 0", "at most one"),
        ("LEAKAGE[t](0.1) 0", "no tag"),
        ("MPAD[HERALD_LEAKAGE_EVENT(qubits=0 1)] 0", "2 qubits for 1"),
    ]:
        with pytest.raises(ValueError, match=match):
            sq.Circuit(bad)
    # Appended as an instruction, the same.
    d = sq.Circuit()
    d.append("LEAKAGE", [0, 2], 0.25)
    assert str(d).strip() == "I_ERROR[LEAKAGE(p=0.25)] 0 2"


def test_deltakits_model():
    # LEAKAGE(0.1) then a herald on qubit 0 (flipped 1%): its detector fires about 10.8%;
    # the inverted herald on qubit 1 only by its flips. A CX with a leaked qubit 0
    # depolarizes qubit 1, so M 1 fires half the time that 0 leaked.
    s = sq.Circuit(SMALL).compile_leakage_sampler(seed=3)
    d, _, h = s.sample(100_000, threads=4)
    leaked = 0.1 * 0.99 + 0.9 * 0.01
    assert abs(d[:, 0].mean() - leaked) < 0.004
    assert abs(d[:, 1].mean() - 0.05) < 0.003
    assert h.shape == (100_000, 3) and not h[:, :2].any()
    m, f = s.sample_measurements(100_000, threads=4)
    assert m.shape == (100_000, 1) and f.shape == (100_000, 2) and s.num_heralds == 2
    assert abs(f[:, 0].mean() - leaked) < 0.004 and abs((~f[:, 1]).mean() - 0.01) < 0.002


def test_measurements_convert_on_the_circuit_without_leakage_shot_for_shot():
    c = deltakit_memory(distance=3, rounds=4, pl=0.01, flip=0.01)
    heralds, kept, herald_dets = _kept_detectors(c)
    assert len(heralds) == c.num_measurements // 2 and len(herald_dets) == len(heralds)
    assert len(kept) + len(herald_dets) == c.num_detectors
    bare = c.without_leakage()
    assert bare.num_detectors == len(kept) and bare.num_measurements == len(heralds)
    assert bare == sq.memory_circuit(distance=3, rounds=4, p=0.001)
    dets, obs, _ = c.compile_leakage_sampler(seed=7).sample(3000, threads=3)
    meas, flags = c.compile_leakage_sampler(seed=7).sample_measurements(3000, threads=2)
    d2, o2 = bare.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
    np.testing.assert_array_equal(d2, dets[:, kept])
    np.testing.assert_array_equal(o2, obs)
    # The herald detectors are the leakage flags.
    np.testing.assert_array_equal(dets[:, [d for d, _ in herald_dets]], flags)
    assert flags.any() and dets[:, kept].any()


def test_loops_survive_deinstrumentation():
    rounds = """
R 0 1 2
M 0 2
REPEAT 50 {
    CX 0 1
    LEAKAGE(0.01) 0 1
    M 1
    HERALD_LEAKAGE_EVENT 1
    DETECTOR rec[-1]
    DETECTOR rec[-2] rec[-4]
}
M 2
OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]
"""
    c = sq.Circuit(rounds)
    bare = c.without_leakage()
    # The first round's detector reaches two measurements before the loop, with no herald
    # between them: that round is peeled off.
    assert str(bare).split() == "R 0 1 2 M 0 2 CX 0 1 M 1 DETECTOR rec[-1] rec[-3] REPEAT 49 { CX 0 1 M 1 DETECTOR rec[-1] rec[-2] } M 2 OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]".split()
    flat = c.flattened().without_leakage()
    assert bare.flattened() == flat
    assert str(bare.detector_error_model()) == str(flat.detector_error_model())


def test_leakage_aware_decoding_from_heralds_flags_and_herald_detectors():
    c = deltakit_memory(distance=5, rounds=5, pl=0.003)
    dets, obs, her = c.compile_leakage_sampler(seed=2).sample(8000, threads=4)
    meas, flags = c.compile_leakage_sampler(seed=2).sample_measurements(8000, threads=4)
    heralds, kept, _ = _kept_detectors(c)
    model = c.without_leakage().detector_error_model(decompose_errors=True)
    la = sq.LeakageAwareDecoder(c)
    assert la.num_detectors == model.num_detectors and la.num_triggers == c.num_measurements and la.unmatched == 0
    # Untriggered, it is matching on the model without leakage, shot for shot.
    np.testing.assert_array_equal(la.decode_batch(dets[:, kept], threads=4), sq.Matching(model).decode_batch(dets[:, kept]))
    # The three routes for heralds agree: the circuit's herald detectors, leakage flags, or the
    # records (only the heralds' records, here, as a device would see).
    by_dets, raised = la.decode_batch(dets, return_raised=True, threads=4)
    by_flags = la.decode_batch(dets[:, kept], leakage_flags=flags, threads=4)
    records = np.zeros_like(her)
    records[:, heralds] = flags
    by_records = la.decode_batch(dets[:, kept], heralds=records, threads=4)
    np.testing.assert_array_equal(by_dets, by_flags)
    np.testing.assert_array_equal(by_dets, by_records)
    assert (raised > 0).mean() > 0.3
    plain = (la.decode_batch(dets[:, kept], threads=4) != obs).any(axis=1).sum()
    aware = (by_dets != obs).any(axis=1).sum()
    assert aware < 0.6 * plain, (aware, plain)
    # Measurements that found their qubit leaked (the sampler's heralds) trigger too.
    assert ((la.decode_batch(dets[:, kept], heralds=her, threads=4) != obs).any(axis=1).sum()) < 0.6 * plain
    with pytest.raises(ValueError, match="one per herald"):
        la.decode_batch(dets[:, kept], leakage_flags=flags[:, 1:])
    again = pickle.loads(pickle.dumps(la))
    np.testing.assert_array_equal(again.decode_batch(dets[:500]), by_dets[:500])


@pytest.mark.parametrize("decoder,options,shots", [("union_find", {}, 8000), ("local_clustering", {}, 8000), ("bposd", dict(max_iter=20), 2000), ("bplsd", dict(max_iter=20), 2000), ("relay_bp", {}, 600)])
def test_around_other_decoders(decoder, options, shots):
    c = deltakit_memory(distance=5, rounds=5, pl=0.003)
    dets, obs, _ = c.compile_leakage_sampler(seed=4).sample(shots, threads=4)
    la = sq.LeakageAwareDecoder(c, decoder, **options)
    assert la.unmatched == 0 and max(la.trigger_sizes) > 0
    plain = (la.decode_batch(dets[:, _kept_detectors(c)[1]], threads=4) != obs).any(axis=1).sum()
    aware = (la.decode_batch(dets, threads=4) != obs).any(axis=1).sum()
    assert aware < 0.7 * plain, (aware, plain)
    if decoder == "local_clustering":
        lcd = sq.LocalClustering.from_circuit(c)
        assert lcd.num_detectors == la.num_detectors and lcd.adaptivity_sizes == la.trigger_sizes


def test_errors_and_the_state_vectors_heralds():
    with pytest.raises(ValueError, match="decoder"):
        sq.LeakageAwareDecoder(sq.Circuit(SMALL), "tesseract")
    with pytest.raises(TypeError, match="no options"):
        sq.LeakageAwareDecoder(sq.Circuit(SMALL), "matching", max_iter=3)
    # The state vector reads a herald on this package's leakage, and refuses Deltakit's model.
    c = sq.Circuit("R 0\nI_ERROR[LEAK(p=0.25)] 0\nHERALD_LEAKAGE_EVENT 0\nDETECTOR rec[-1]")
    assert c.exact_distribution() == pytest.approx({((False,), ()): 0.75, ((True,), ()): 0.25})
    with pytest.raises(ValueError, match="does not run Deltakit"):
        sq.Circuit(SMALL).compile_exact_sampler()
