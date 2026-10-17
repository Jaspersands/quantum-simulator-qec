# %% [markdown]
# # Fault-tolerant compilation, and what it would cost
#
# A logical computation is a spacetime diagram before it is a circuit. Two tools draw them:
# tqec's **block graphs** (cubes and pipes in 3D, correlation surfaces for the observables) and
# Loom's **Eka** (blocks of a rotated surface code grown, shrunk, merged and split). Both are
# here, checked against their originals, and both land on this package's lattice-surgery
# engine or its circuit IR. At the other end, two resource estimators — Qualtran's surface-code
# cost models and Azure's Resource Estimator — say what a computation costs, to the bit the
# originals report.

# %%
import numpy as np
import stabilizer_qec as sq
from stabilizer_qec import loom, tqec
from stabilizer_qec.tqec import gallery

# %% [markdown]
# ## tqec: a CNOT as a block graph, compiled and decoded
#
# `gallery.cnot()` is tqec's logical CNOT: cubes joined by pipes in space and time, its ports
# filled here with Z-basis inputs and outputs. Its
# correlation surfaces are the observables; `compile_block_graph` runs it on the lattice-surgery
# engine at a code distance and noise rate and returns a circuit with detectors, which decodes
# like any other.

# %%
graph = gallery.cnot(tqec.Basis.Z)  # ports filled: both inputs and outputs in the Z basis
print(graph.num_cubes, "cubes,", graph.num_pipes, "pipes,", len(graph.find_correlation_surfaces()), "correlation surfaces")
compiled = tqec.compile_block_graph(graph, 3, p=0.002)
circuit = compiled.circuit
print(circuit.num_qubits, "qubits,", circuit.num_detectors, "detectors,", circuit.num_observables, "observables")
dem = circuit.detector_error_model(decompose_errors=True)
dets, obs = circuit.compile_detector_sampler(seed=1).sample(20_000, separate_observables=True)
pred = sq.Matching(dem).decode_batch(dets)
print(f"logical error of the compiled CNOT at d = 3, p = 0.2%: {(pred != obs).any(axis=1).mean():.2%}")

# %% [markdown]
# A graph can be written as tqec writes it — JSON, or a Collada `.dae` file SketchUp opens —
# and read back:

# %%
import tempfile, pathlib
with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp) / "cnot.json"
    graph.to_json(path)
    again = tqec.BlockGraph.from_json(path)
    graph.to_dae_file(pathlib.Path(tmp) / "cnot.dae")
print("round trip through tqec's JSON:", again == graph)

# %% [markdown]
# ## Loom: an Eka interpreted to a circuit
#
# Loom describes an experiment as blocks on a lattice and a list of operations. The interpreter
# turns it into a circuit with its syndromes, detectors and logical observables; the circuit
# here equals Loom's gate for gate.

# %%
lattice = loom.Lattice.square_2d((12, 12))
block = loom.RotatedSurfaceCode.create(3, 3, lattice, unique_label="a")
eka = loom.Eka(lattice, [block], [loom.MeasureBlockSyndromes("a"), loom.MeasureBlockSyndromes("a"), loom.MeasureLogicalZ("a")])
step = loom.interpret_eka(eka)
memory = loom.eka_to_circuit(step)
print(memory.num_qubits, "qubits,", memory.num_detectors, "detectors,", memory.num_observables, "observable")
d7, o7 = memory.compile_detector_sampler(seed=1).sample(1000, separate_observables=True)
print("noiseless, every detector and the observable are deterministic:", not d7.any() and not o7.any())

# %% [markdown]
# ## Qualtran: the Gidney–Fowler spreadsheet
#
# `stabilizer_qec.estimate` holds Qualtran's `surface_code` models: QEC schemes, data blocks,
# the CCZ-to-2T and 15-to-1 factories, rotation costs, and the grid search over factory and data
# block distances. The numbers are Qualtran's own.

# %%
from stabilizer_qec.estimate import GateCounts, get_ccz2t_costs_from_error_budget, get_ccz2t_costs_from_grid_search

cost = get_ccz2t_costs_from_error_budget(n_logical_gates=GateCounts(t=10**8, toffoli=10**8), n_algo_qubits=100, error_budget=0.01, phys_err=1e-3, cycle_time_us=1)
print(f"10^8 T + 10^8 Toffolis on 100 qubits: {cost.footprint:,} physical qubits, {cost.duration_hr:.2f} h, failure {cost.failure_prob:.3f}")
best, factory, block = get_ccz2t_costs_from_grid_search(n_logical_gates=GateCounts(t=10**8, toffoli=10**8), n_algo_qubits=100, phys_err=1e-3, error_budget=0.1, cycle_time_us=1)
print(f"grid search: factory distances {factory.distillation_l1_d}/{factory.distillation_l2_d}, data block d = {block.data_d}: {best.footprint:,} qubits, {best.duration_hr:.2f} h")

# %% [markdown]
# ## Azure's Resource Estimator
#
# `azure_estimate` takes the QDK's inputs — logical counts, a qubit preset or custom qubit, a
# QEC scheme, an error budget, constraints — and returns its result dict, physical qubits to
# runtime to the T factory's rounds, equal to the QDK's on every configuration recorded in the
# tests.

# %%
from stabilizer_qec.estimate import LogicalCounts, azure_estimate

counts = LogicalCounts(num_qubits=100, t_count=100_000, ccz_count=10_000, measurement_count=10_000)
for qubit in ("qubit_gate_ns_e3", "qubit_gate_ns_e4", "qubit_maj_ns_e6"):
    r = azure_estimate(counts, qubit)
    pc, tf = r["physicalCounts"], r["tfactory"]
    print(f"{qubit}: {pc['physicalQubits']:,} qubits, {pc['runtime'] / 1e9:.2f} s, code distance {r['logicalQubit']['codeDistance']}, "
          f"{pc['breakdown']['numTfactories']} factories of {tf['numRounds']} rounds {tf['codeDistancePerRound']}")
frontier = azure_estimate(counts, "qubit_gate_ns_e3", estimate_type="frontier")["frontierEntries"]
print(f"the Pareto frontier has {len(frontier)} points, from {frontier[0]['physicalCounts']['physicalQubits']:,} qubits in {frontier[0]['physicalCounts']['runtime'] / 1e9:.1f} s "
      f"to {frontier[-1]['physicalCounts']['physicalQubits']:,} in {frontier[-1]['physicalCounts']['runtime'] / 1e9:.1f} s")
