# %% [markdown]
# # Beyond Pauli noise
#
# Stim and every decoder treat noise as random Pauli errors. Real hardware also over-rotates:
# a qubit meant to idle turns by a small angle θ about Z, the same way every time. That is a
# *coherent* error, exp(−iθZ), and its Pauli twirl (Z with probability sin²θ) is what the
# usual tools put in its place. The twirl is exact for one rotation alone, but rotations that
# repeat add up as amplitudes rather than probabilities: two turns of θ flip with probability
# sin²(2θ) ≈ 4θ², not 2 sin²θ ≈ 2θ².
#
# This package simulates coherent errors on every gate of a circuit, at sizes a state vector
# cannot reach (Tsim, arXiv 2604.01059, reaches any size for circuits with a few non-Clifford
# gates; noise on every gate is beyond it):
#
# - **Rotations are tagged identities** that Stim reads as identities, so circuits stay
#   Stim's: `I_ERROR[R_Z(theta=0.01)] 0`, `II_ERROR[R_ZZ(theta=0.02)] 0 1`, any Pauli rotation
#   as `I_ERROR[R_PAULI(theta=…, pauli=XZY)] 0 1 2`.
# - **`compile_coherent_sampler`** draws shots of the twirled circuit, as the detector sampler
#   does, each with a weight that restores the interference the twirl leaves out. Weighted,
#   the shots are distributed as the coherent circuit's.
# - **`compile_exact_sampler`** runs the full state vector (up to 24 qubits), the oracle the
#   coherent sampler is checked against; `exact_distribution` follows every branch exactly.

# %%
import numpy as np
import stabilizer_qec as sq

# %% [markdown]
# ## Two rotations that add up
#
# A Z over-rotation either side of a CZ whose control is the rotated qubit: the CZ leaves Z
# alone, so the two are one rotation by θ₁ + θ₂.

# %%
c = sq.Circuit("""
RX 0
R 1
I_ERROR[R_Z(theta=0.2)] 0
CZ 0 1
I_ERROR[R_Z(theta=0.15)] 0
MX 0
DETECTOR rec[-1]
""")
exact = c.exact_distribution()[((True,), ())]
d, o, w = c.compile_coherent_sampler(seed=1).sample(200_000)
weighted, err = sq.weighted_rate(d[:, 0], w)
twirled = c.twirled().compile_detector_sampler(seed=1).sample(200_000)[:, 0].mean()
print(f"exact (state vector)  {exact:.4f}    sin²(0.35) = {np.sin(0.35) ** 2:.4f}")
print(f"coherent sampler      {weighted:.4f} ± {err:.4f}")
print(f"Pauli twirl           {twirled:.4f}")

# %% [markdown]
# ## A surface code under over-rotation
#
# The d = 3 rotated surface code's X memory over two rounds, with circuit noise p = 0.2% and a coherent Z
# over-rotation by θ = 0.02 on every data qubit after every tick. The coherent sampler's
# logical error rate against the twirl's, both decoded by matching on the twirled model; and
# against the state vector's, which this circuit (17 qubits) is still small enough for.

# %%
def memory(theta, d=3, rounds=3, p=0.002):
    base = sq.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=rounds,
                                after_clifford_depolarization=p, before_measure_flip_probability=p,
                                after_reset_flip_probability=p)
    text = str(base)
    data = [line for line in text.splitlines() if line.startswith("MX ")][-1].split()[1:]
    return sq.Circuit(text.replace("TICK", f"TICK\nI_ERROR[R_Z(theta={theta})] " + " ".join(data)))


c = memory(0.02, rounds=2)
decoder = sq.Matching.from_detector_error_model(c.twirled().detector_error_model(decompose_errors=True))
d, o, w = c.compile_coherent_sampler(seed=2).sample(200_000, threads=0)
coherent, err = sq.weighted_logical_error_rate(decoder, d, o, w)
de, oe = c.compile_exact_sampler(seed=3).sample(2000, separate_observables=True, threads=0)
exact_rate = (decoder.decode_batch(de) != oe).any(axis=1).mean()
dt, ot = c.twirled().compile_detector_sampler(seed=4).sample(200_000, separate_observables=True)
twirl_rate = (decoder.decode_batch(dt) != ot).any(axis=1).mean()
print(f"coherent sampler   {coherent:.4f} ± {err:.4f}   (effective sample size {sq.effective_sample_size(w) / len(w):.0%})")
print(f"state vector       {exact_rate:.4f} ± {np.sqrt(exact_rate * (1 - exact_rate) / 2000):.4f}   (2,000 shots)")
print(f"Pauli twirl        {twirl_rate:.4f}")

# %% [markdown]
# The twirl underestimates the logical error rate several times over: the over-rotations of
# successive ticks, between the gates that would tell them apart, add coherently.
#
# ## A decoder that knows
#
# `twirled(merge=True)` adds up the rotations that are the same fault in different places
# before twirling: sin²(Σθ) rather than Σ sin²θ. Decoding on that model, the same shots fail
# less often.

# %%
aware = sq.Matching.from_detector_error_model(c.twirled(merge=True).detector_error_model(decompose_errors=True))
merged, merr = sq.weighted_logical_error_rate(aware, d, o, w)
print(f"decoded on the twirl          {coherent:.4f} ± {err:.4f}")
print(f"decoded on the merged model   {merged:.4f} ± {merr:.4f}")

# %% [markdown]
# ## Larger codes
#
# The state vector stops at 24 qubits; the coherent sampler does not. At d = 7 (97 qubits,
# 2,401 rotations) with θ = 0.01:

# %%
c7 = memory(0.01, d=7, rounds=7)
s7 = c7.compile_coherent_sampler(seed=5)
d, o, w = s7.sample(60_000, threads=0)
dec7 = sq.Matching.from_detector_error_model(c7.twirled().detector_error_model(decompose_errors=True))
rate7, err7 = sq.weighted_logical_error_rate(dec7, d, o, w)
dt, ot = c7.twirled().compile_detector_sampler(seed=6).sample(400_000, separate_observables=True)
print(f"{s7.num_locations} rotations, {s7.num_generators} interference generators")
print(f"coherent  {rate7:.2e} ± {err7:.1e}   twirl  {(dec7.decode_batch(dt) != ot).any(axis=1).mean():.2e}")

# %% [markdown]
# ## T gates and leakage
#
# The state vector also runs non-Clifford gates (`I[T] 0`, `I[U3(theta=…, phi=…, lambda=…)] 0`)
# and amplitude damping. Leakage is written the same Stim-readable way and runs in a frame
# sampler that also reports which measurements found their qubit leaked:

# %%
leaky = sq.Circuit("""
R 0 1 2
I_ERROR[LEAK(p=0.05)] 0 2
CX 0 1 2 1
MR 1
DETECTOR rec[-1]
M 0 2
""")
d, o, heralds = leaky.compile_leakage_sampler(seed=7).sample(10_000)
print(f"detector fires {d[:, 0].mean():.3f}; leaked final measurements {heralds[:, 1:].mean():.3f}")
