"""Every gate of Stim's language the engine reads, checked against Stim on random circuits.

Each circuit applies a random Clifford circuit U drawn from every unitary gate, with noise
between its layers; measures some of the state's stabilizers (MPP, MXX/MYY/MZZ, MY, some
inverted); applies Stim's own inverse of U; and measures every qubit. Without noise every
measurement is then deterministic, so each carries a detector. A gate decomposed wrongly shows
up as a detector that is not deterministic, a model that differs from Stim's, raw measurements
converted differently, or detection rates apart from Stim's."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")

UNITARY_1 = sorted(n for n, g in stim.gate_data().items() if g.is_unitary and g.is_single_qubit_gate and n == g.name)
UNITARY_2 = sorted(n for n, g in stim.gate_data().items() if g.is_unitary and g.is_two_qubit_gate and n == g.name)
NOISE_1 = ["X_ERROR(0.02)", "Y_ERROR(0.03)", "Z_ERROR(0.02)", "DEPOLARIZE1(0.03)", "PAULI_CHANNEL_1(0.01, 0.02, 0.03)", "I_ERROR(0.1)"]
NOISE_2 = ["DEPOLARIZE2(0.03)", "II_ERROR(0.1)"]


def random_unitary(rng, n, layers):
    """Lines of a random Clifford circuit on n qubits, every unitary gate available."""
    out = []
    for _ in range(layers):
        if rng.random() < 0.1:
            # A Pauli-product rotation, sometimes of a negated product.
            qs = rng.choice(n, size=rng.integers(1, n + 1), replace=False)
            product = "*".join(f"{'XYZ'[rng.integers(3)]}{q}" for q in qs)
            out.append(f"{rng.choice(['SPP', 'SPP_DAG'])} {'!' if rng.random() < 0.3 else ''}{product}")
        elif rng.random() < 0.5:
            g = rng.choice(UNITARY_1)
            qs = rng.choice(n, size=rng.integers(1, n + 1), replace=False)
            out.append(f"{g} {' '.join(map(str, qs))}")
        else:
            g = rng.choice(UNITARY_2)
            a, b = rng.choice(n, size=2, replace=False)
            out.append(f"{g} {a} {b}")
    return out


def noise(rng, n, correlated):
    out = []
    for _ in range(rng.integers(1, 3)):
        if rng.random() < 0.6:
            out.append(f"{rng.choice(NOISE_1)} {rng.integers(n)}")
        elif correlated and rng.random() < 0.5:
            qs = rng.choice(n, size=rng.integers(1, min(n, 3) + 1), replace=False)
            out.append("E(0.04) " + " ".join(f"{'XYZ'[rng.integers(3)]}{q}" for q in qs))
        else:
            a, b = rng.choice(n, size=2, replace=False)
            out.append(f"{rng.choice(NOISE_2)} {a} {b}")
    return out


def stabilizer_measurements(rng, prefix, n):
    """Measurements of a few of the state's stabilizers, each written the way its form allows."""
    sim = stim.TableauSimulator()
    sim.do(stim.Circuit("\n".join(prefix)))
    out = []
    for s in rng.permutation(sim.canonical_stabilizers())[: rng.integers(1, 3)]:
        support = [(q, "_XYZ"[p]) for q, p in enumerate(s) if p]
        if not support:
            continue
        inv = "!" if rng.random() < 0.4 else ""
        flip = f"({rng.choice(['0', '0.01'])})" if rng.random() < 0.5 else ""
        if len(support) == 1:
            q, p = support[0]
            out.append(f"{ {'X': 'MX', 'Y': 'MY', 'Z': 'M'}[p] }{flip} {inv}{q}")
        elif len(support) == 2 and support[0][1] == support[1][1]:
            (a, p), (b, _) = support
            out.append(f"M{p}{p}{flip} {inv}{a} {b}")
        else:
            out.append(f"MPP{flip} {inv}" + "*".join(f"{p}{q}" for q, p in support))
        out.append("DETECTOR rec[-1]")
    return out


def random_case(seed, n=4, layers=12, correlated=True):
    rng = np.random.default_rng(seed)
    u = random_unitary(rng, n, layers)
    noisy, prefix = [], [f"R {' '.join(map(str, range(n)))}"]
    for k, line in enumerate(u):
        noisy.append(line)
        prefix.append(line)
        if k % 3 == 2:
            noisy.extend(noise(rng, n, correlated))
    lines = [prefix[0]] + noisy
    lines += stabilizer_measurements(rng, prefix, n)
    lines += [str(stim.Circuit("\n".join(u)).inverse())]
    lines += ["M " + " ".join(("!" if rng.random() < 0.3 else "") + str(q) for q in range(n))]
    lines += [f"DETECTOR rec[-{k + 1}]" for k in range(n)]
    lines += ["OBSERVABLE_INCLUDE(0) rec[-1]"]
    return "\n".join(lines)


def canon(text):
    from conftest import canonical_dem

    return canonical_dem(text)


@pytest.mark.parametrize("seed", range(60))
def test_random_circuits_against_stim(seed):
    text = random_case(seed)
    ours, theirs = sq.Circuit(text), stim.Circuit(text)
    assert ours.num_detectors == theirs.num_detectors and ours.num_measurements == theirs.num_measurements
    assert sq.Circuit(str(ours)) == ours
    # The error models agree fault for fault.
    a = canon(str(ours.detector_error_model()))
    b = canon(str(theirs.detector_error_model().flattened()))
    assert set(a) == set(b)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-15)
    # Raw measurements from Stim convert to Stim's detection events exactly.
    meas = theirs.compile_sampler(seed=seed).sample(300)
    want = theirs.compile_m2d_converter().convert(measurements=meas, append_observables=True)
    got = ours.compile_m2d_converter().convert(measurements=meas, append_observables=True)
    assert np.array_equal(got, want)
    # And the samplers agree on every detector's rate.
    shots = 20_000
    r1 = ours.compile_detector_sampler(seed=seed).sample(shots).mean(axis=0)
    r2 = theirs.compile_detector_sampler(seed=seed).sample(shots).mean(axis=0)
    sigma = np.sqrt((r1 * (1 - r1) + r2 * (1 - r2)) / shots) + 1e-9
    assert np.abs((r1 - r2) / sigma).max() < 5.5


@pytest.mark.parametrize("seed", range(8))
def test_channels_error_models_refuse_are_sampled_as_stim_samples(seed):
    rng = np.random.default_rng(100 + seed)
    probs = rng.dirichlet(np.ones(16))[:15] * 0.3
    pc2 = "PAULI_CHANNEL_2(" + ", ".join(f"{p:.6f}" for p in probs) + ") 0 1"
    text = random_case(seed, n=3, layers=6, correlated=False).replace(
        "\nM ", f"\n{pc2}\nE(0.1) X0 Z2\nELSE_CORRELATED_ERROR(0.2) Y1\nELSE_CORRELATED_ERROR(0.3) X0 X1\nMPAD(0.05) 0 1\nM ", 1
    )
    ours, theirs = sq.Circuit(text), stim.Circuit(text)
    with pytest.raises(ValueError):
        ours.detector_error_model()
    shots = 40_000
    r1 = ours.compile_detector_sampler(seed=seed).sample(shots, append_observables=True).mean(axis=0)
    r2 = theirs.compile_detector_sampler(seed=seed).sample(shots, append_observables=True).mean(axis=0)
    sigma = np.sqrt((r1 * (1 - r1) + r2 * (1 - r2)) / shots) + 1e-9
    assert np.abs((r1 - r2) / sigma).max() < 5.5


def test_y_basis_resets_and_measurements():
    text = (
        "RY 0 1\nMYY 0 !1\nMY 0 !1\nMRY 0\nMY 0\nH_YZ 1\nM 1\n"
        + "\n".join(f"DETECTOR rec[-{k}]" for k in range(1, 7))
    )
    ours, theirs = sq.Circuit(text), stim.Circuit(text)
    meas = theirs.compile_sampler(seed=1).sample(50)
    assert np.array_equal(
        ours.compile_m2d_converter().convert(measurements=meas),
        theirs.compile_m2d_converter().convert(measurements=meas, separate_observables=False, append_observables=False),
    )
    assert not ours.compile_detector_sampler(seed=1).sample(1000).any()


def test_gate_lines_print_as_written():
    text = "R 0 1\nSQRT_X_DAG 0\nISWAP 0 1\nMPP(0.01) !X0*Z1\nMXX 0 !1\nE(0.1) X0 Y1\nMPAD 1 0\nS 0"
    assert str(sq.Circuit(text)).strip().split("\n") == text.split("\n")


def test_unsupported_gates_say_so():
    for text in ["HERALDED_ERASE(0.1) 0", "CY sweep[0] 1", "MPP X0*Z0", "E(0.1) X0 Y0", "MXX 0 0"]:
        with pytest.raises(ValueError):
            sq.Circuit(text)
