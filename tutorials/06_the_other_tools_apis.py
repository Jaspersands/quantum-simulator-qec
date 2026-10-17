# %% [markdown]
# # The other tools' APIs, in one package
#
# Release 2.0 set out to hold every feature of every other QEC tool, under the names those
# tools use, with each port checked against its original. This notebook is a tour: Stim's
# object model, PyMatching's graph API, ldpc's decoders, CUDA-Q QEC's codes and decoders, and
# leakage-aware decoding, all from `import stabilizer_qec as sq`. The inventory of what was
# ported, tool by tool, is `docs/superpowers/notes/2026-10-09-parity-inventory.md`.

# %%
import numpy as np
import stabilizer_qec as sq

# %% [markdown]
# ## Stim's objects
#
# `Circuit`, `DetectorErrorModel`, `PauliString`, `Tableau` and `TableauSimulator` carry
# Stim 1.16's API member for member: the same methods, the same text, the same numbers.

# %%
c = sq.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.001)
print(c.num_qubits, "qubits,", c.num_detectors, "detectors;", len(c), "instructions")
dem = c.detector_error_model(decompose_errors=True)
print(dem.num_errors, "error mechanisms; shortest graph-like error:", len(c.shortest_graphlike_error()))
print(repr(sq.PauliString("XYZ") * sq.PauliString("ZZX")))
sim = sq.TableauSimulator()
sim.h(0)
sim.cx(0, 1)
print("a Bell pair's stabilizers:", sim.canonical_stabilizers())
print(c.diagram("timeline-text").__str__().splitlines()[0][:80], "…")

# %% [markdown]
# ## PyMatching's graph API
#
# `Matching` is PyMatching's class whole: build a graph edge by edge, load a check matrix or a
# detector error model, decode one syndrome or a batch, `decode_to_edges_array`, and the rest.

# %%
m = sq.Matching()
for i in range(4):
    m.add_edge(i, i + 1, weight=1.0, fault_ids={i})
m.add_boundary_edge(0, weight=1.0, fault_ids={4})
m.add_boundary_edge(4, weight=1.0, fault_ids={5})
print(m)
print("a single defect at node 1 is matched to the boundary through node 0:", m.decode([0, 1, 0, 0, 0]))
print("two defects pair up:", m.decode([0, 1, 0, 1, 0]))

# %% [markdown]
# ## ldpc's decoders
#
# `BpDecoder`, `BpOsdDecoder` and `BpLsdDecoder` take ldpc's arguments and give ldpc's
# corrections (bit for bit, but for the ties its sort leaves to the platform). Here BP+OSD on a
# repetition code's check matrix, from a syndrome of a two-bit error.

# %%
H = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]], dtype=np.uint8)
error = np.array([0, 1, 1, 0, 0], dtype=np.uint8)
syndrome = (H @ error) % 2
dec = sq.BpOsdDecoder(H, error_rate=0.1, max_iter=20, bp_method="minimum_sum", osd_method="osd_cs", osd_order=2)
print("syndrome", syndrome, "→ correction", dec.decode(syndrome), "(the error was", error, ")")

# %% [markdown]
# ## CUDA-Q QEC's codes and decoders
#
# `stabilizer_qec.cudaq_qec` is CUDA-Q QEC under its names: `get_code` with the surface code's
# stabilizer grid and CNOT schedules, `get_decoder` for its lookup-table, tensor-network and
# sliding-window decoders, its detector error models and check-matrix tools. CUDA-Q's kernels
# are this package's circuits; its GPUs are not needed.

# %%
import stabilizer_qec.cudaq_qec as qec

code = qec.get_code("surface_code", distance=3)
Hx, Hz = code.get_parity_x(), code.get_parity_z()
print("surface code d = 3: parity matrices", Hx.shape, Hz.shape, "|", code.get_num_data_qubits(), "data qubits")
lut = qec.get_decoder("single_error_lut", Hz)
e = np.zeros(Hz.shape[1], dtype=np.uint8)
e[2] = 1
result = lut.decode((Hz @ e) % 2)
print("a single X error on data qubit 2, decoded by the lookup table:", np.flatnonzero(np.asarray(result.result)), "converged:", result.converged)

# %% [markdown]
# ## Leakage, and a decoder that knows
#
# Deltakit's `LEAKAGE(p)` after every CX leaks a qubit; a leaked qubit reads 1 and leaves every
# partner a random Pauli until a reset returns it. The leakage sampler gives each shot's heralds
# (which measurements read a leaked qubit), and `LeakageAwareDecoder` uses them to raise the
# priors of the faults a leak makes likely before decoding, as Google's and Riverlane's
# leakage-aware decoders do.

# %%
lines = []
for line in str(sq.memory_circuit(distance=5, rounds=5, p=0.001)).splitlines():
    lines.append(line)
    if line.startswith("CX"):
        lines.append("LEAKAGE(0.003)" + line[2:])
leaky = sq.Circuit("\n".join(lines))
dets, obs, heralds = leaky.compile_leakage_sampler(seed=1).sample(20_000)
dec = sq.LeakageAwareDecoder(leaky)
blind = (dec.decode_batch(dets) != obs).any(axis=1).mean()
aware = (dec.decode_batch(dets, heralds=heralds) != obs).any(axis=1).mean()
print(f"logical error blind to the leakage: {blind:.3%}; with the heralds: {aware:.3%}")

# %% [markdown]
# ## And the rest
#
# The same pattern — the original's names, checked against the original — runs through the
# package: `stabilizer_qec.sinter` (collection and plotting), `stabilizer_qec.ldpc` (the whole
# library, codes and GF(2) tools included), `qsample` (subset sampling), `mqt_qecc` (code
# switching and surgery routing), `panqec`, `qecsim`, `pecos`, `quits`, `deltakit`, `tsim`,
# `tqec`, `loom`, `estimate`, `realtime` — each a module with its own notebook or README section.

# %%
import importlib

for name in ("sinter", "ldpc", "qsample", "mqt_qecc", "panqec", "qecsim", "pecos", "quits", "deltakit", "tsim", "tqec", "loom", "estimate", "realtime"):
    doc = (importlib.import_module(f"stabilizer_qec.{name}").__doc__ or "").strip().splitlines()
    print(f"{name:10s} {doc[0][:90] if doc else ''}")
