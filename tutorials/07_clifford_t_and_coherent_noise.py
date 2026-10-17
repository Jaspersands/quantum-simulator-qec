# %% [markdown]
# # Clifford+T circuits, and coherent noise on every gate
#
# A Clifford circuit is cheap to simulate at any size; a T gate is not. The **stabilizer-rank
# sampler** keeps the state as a sum of Clifford branches, one per coset the non-Clifford gates
# reach, so a circuit with a few T gates samples exactly at any qubit count (Tsim's territory,
# and Tsim's demos sample the same distributions here). Then the headline: that circuit with a
# small over-rotation after every gate, where the Pauli twirl every other tool uses misjudges
# the answer by two to three times.

# %%
import numpy as np
import stabilizer_qec as sq

# %% [markdown]
# ## A T gate, exactly
#
# `I[T] 0` is a T gate written as a tagged identity, so Stim reads the circuit as Clifford and
# this package runs it. The outcome of H·T·H on |0⟩ is 1 with probability sin²(π/8).

# %%
c = sq.Circuit("R 0\nH 0\nI[T] 0\nH 0\nM 0\nDETECTOR rec[-1]")
print("T count:", c.tcount(), "| is Clifford:", c.is_clifford)
dist = c.exact_distribution(method="stabilizer_rank")
p1 = dist[((True,), ())]
print(f"P(1) = {p1:.6f}, sin²(π/8) = {np.sin(np.pi / 8) ** 2:.6f}")
dets = c.compile_stabilizer_rank_sampler(seed=1).sample(20_000)
print(f"sampled: {dets.mean():.4f}")

# %% [markdown]
# ## Many qubits, few T gates
#
# Fourteen T gates on the Steane code's |0_L⟩ — a transversal T applied twice, which is a
# transversal S and so a logical S† — leave Z_L and the three Z checks exactly where they
# were. The sampler follows every branch the T gates open and lands on the empty detection
# pattern every shot: the Clifford gates cost nothing per term, and terms that land on one
# coset add, so seven T gates on a code state are eight terms rather than 128.

# %%
steane = [
    "R 0 1 2 3 4 5 6", "H 0 1 2",
    "CX 0 3 0 5 0 6", "CX 1 3 1 4 1 6", "CX 2 4 2 5 2 6",  # |0_L> from |+++0000>
    "I[T] 0 1 2 3 4 5 6", "I[T] 0 1 2 3 4 5 6",  # transversal T twice: logical S†
    "M 0 1 2 3 4 5 6",
    "DETECTOR rec[-7] rec[-4] rec[-2] rec[-1]",  # the Z checks Z0 Z3 Z5 Z6, Z1 Z3 Z4 Z6, Z2 Z4 Z5 Z6
    "DETECTOR rec[-6] rec[-4] rec[-3] rec[-1]",
    "DETECTOR rec[-5] rec[-3] rec[-2] rec[-1]",
    "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3] rec[-4] rec[-5] rec[-6] rec[-7]",  # Z_L
]
cs = sq.Circuit("\n".join(steane))
print("non-Clifford gates:", cs.num_non_clifford_gates())
dets7, obs7 = cs.compile_stabilizer_rank_sampler(seed=2).sample(5_000, separate_observables=True)
print(f"detection events: {int(dets7.sum())}, logical flips: {int(obs7.sum())} in 5,000 shots")

# %% [markdown]
# ## Coherent errors on every gate
#
# Now put exp(−iεZ) after every gate of a magic-state preparation and read the prepared
# state's infidelity three ways: the **Pauli twirl** (Z with probability sin²ε after each gate,
# what every stabilizer tool would do), the **joint sampler** (gates exact, the errors twirled
# and reweighted so the shots are distributed as the coherent circuit's), and the **state
# vector** (the truth, small enough here).

# %%
def preparation(eps: float, twirl: bool) -> sq.Circuit:
    err = f"Z_ERROR({np.sin(eps) ** 2:.12g}) 0" if twirl else f"I_ERROR[R_Z(theta={eps})] 0"
    lines = ["R 0", "H 0", err, "I[T] 0", err, "H 0", err, "I[T_DAG] 0", err, "M 0", "DETECTOR rec[-1]"]
    return sq.Circuit("\n".join(lines))

for eps in (0.02, 0.05, 0.1):
    twirled = preparation(eps, True).compile_stabilizer_rank_sampler(seed=3).sample(100_000).mean()
    exact = preparation(eps, False).exact_distribution(method="state_vector")[((True,), ())]
    d, _, w = preparation(eps, False).compile_coherent_sampler(seed=3).sample(100_000)
    joint, err = sq.weighted_rate(d[:, 0], w)
    print(f"ε = {eps}: twirl {twirled:.5f}, joint sampler {joint:.5f} ± {err:.5f}, exact {exact:.5f}")

# %% [markdown]
# The twirl under-reads the error (the two rotations about Z between the Hadamards add as
# amplitudes, 4ε² against 2ε²); the exact answer is what the hardware would show. At the scale
# of a 15-to-1 distillation in the [[17,1,5]] colour code (85 qubits, 801 rotations) no state
# vector reaches, the README's tables carry the same comparison by the joint sampler.
