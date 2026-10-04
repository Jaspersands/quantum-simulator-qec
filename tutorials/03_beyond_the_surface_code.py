# %% [markdown]
# # Beyond the surface code
#
# A surface code spends about 2d² physical qubits on one logical qubit. Codes with more
# logical qubits per physical qubit exist; their checks are larger and less local, and their
# faults set off three or more detectors at once, so they are decoded by BP+OSD rather than
# matching. This tutorial builds three families of them: IBM's bivariate bicycle codes,
# hypergraph products of classical codes, and colour codes.

# %%
import numpy as np
import stabilizer_qec as sq

# %% [markdown]
# ## Bivariate bicycle codes
#
# Bravyi et al. (Nature, 2024) list five bivariate bicycle codes. Each lives on a torus of
# qubits with weight-6 checks, and comes with a depth-8 syndrome cycle. The gross code,
# [[144, 12, 12]], keeps 12 logical qubits at distance 12 on 144 data qubits and 144 check
# qubits. Twelve distance-12 surface codes would need about 3,400.

# %%
print(f"{'code':>6} {'n':>4} {'k':>3}")
for name in ("72", "90", "108", "144", "288"):
    code = sq.BivariateBicycleCode(name)
    print(f"{name:>6} {code.n:4d} {code.k:3d}")

# %% [markdown]
# A memory of the gross code under the paper's circuit noise: six syndrome cycles at
# p = 0.3%, decoded by BP+OSD. The error model is not decomposed, because BP+OSD takes
# faults of any size.

# %%
gross = sq.BivariateBicycleCode("gross")
circuit = gross.memory_circuit(6, 0.003)
dem = circuit.detector_error_model()
dets, obs = circuit.compile_detector_sampler(seed=2).sample(500, separate_observables=True)
predicted = sq.BpOsd(dem, max_iter=100, osd_order=7).decode_batch(dets)
print(circuit.num_qubits, "qubits,", dem.num_errors, "faults,", circuit.num_observables, "logical observables")
print("shots where any of the 12 logical qubits failed:", int((predicted != obs).any(axis=1).sum()), "of", len(obs))
print("shots with a logical flip before decoding:", int(obs.any(axis=1).sum()))

# %% [markdown]
# The code's symmetries act on its logical qubits. Every shift of the torus, with or without
# exchanging X and Z, is an automorphism, a logical operation done by relabelling qubits:

# %%
autos = gross.automorphisms()
distinct = {a.action.tobytes() for a in autos}
print(len(autos), "automorphisms,", len(distinct), "distinct actions on the 12 logical qubits")

# %% [markdown]
# Other polynomials give other codes. `from_polynomials` builds any bivariate bicycle code
# and checks that it encodes something:

# %%
other = sq.BivariateBicycleCode.from_polynomials(6, 6, [(2, 0), (0, 1), (0, 3)], [(0, 2), (1, 0), (3, 0)])
print(other, "-> n =", other.n, ", k =", other.k)

# %% [markdown]
# ## Hypergraph products
#
# Any two classical codes give a quantum code by the hypergraph product (Tillich and Zémor).
# Two repetition codes give the surface code; two [7, 4, 3] Hamming codes give
# [[58, 16, 3]], sixteen logical qubits. `CssCode` takes any CSS code and writes a memory
# experiment that is correct for it: every X check is measured, then every Z check.

# %%
hamming = [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]]
hgp = sq.CssCode.hypergraph_product(hamming)
print(hgp)
surface = sq.CssCode.hypergraph_product(np.eye(4, 5, dtype=int) + np.eye(4, 5, 1, dtype=int))
print(surface, "(two distance-5 repetition codes: the unrotated surface code)")

c = hgp.memory_circuit(3, 0.002)
d, o = c.compile_detector_sampler(seed=4).sample(1_000, separate_observables=True)
p = sq.BpOsd(c.detector_error_model(), max_iter=100, osd_order=7).decode_batch(d)
print(f"[[58, 16, 3]] at p = 0.2%, 3 rounds: {np.mean((p != o).any(axis=1)):.3f} of shots fail, "
      f"against {np.mean(o.any(axis=1)):.3f} undecoded")

# %% [markdown]
# ## Colour codes
#
# The triangular 6.6.6 colour code has every hexagon as both an X and a Z check, and can do
# every Clifford gate transversally. At distance 3 it is the Steane code:

# %%
for d in (3, 5, 7):
    print(sq.CssCode.color_code(d))

# %% [markdown]
# Stim's own colour-code memory measures each hexagon in turn in the Z, X and Y bases,
# rotating the data with `C_XYZ` every round. Its detector slices show the rotation: the same
# detector compares Z on its hexagon (and its ancilla, qubit 2), then, past the `C_XYZ`, X on
# the data, then a mixture as the first CNOTs spread it.

# %%
xyz = sq.Circuit.generated("color_code:memory_xyz", distance=3, rounds=3, after_clifford_depolarization=0.001)
for tick in (1, 2, 3):
    first = str(xyz.diagram("detslice-text", tick=tick)).splitlines()[0]
    print(f"after TICK {tick}: {first}")
