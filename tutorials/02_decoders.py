# %% [markdown]
# # Decoders, and decoding in real time
#
# One error model can be decoded many ways. This tutorial runs the package's decoders on the
# same shots and compares how often they fail and how fast they are. Then it turns to
# decoding as a real machine must: in windows, while the rounds are still arriving.

# %%
import time

import numpy as np
import stabilizer_qec as sq

circuit = sq.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=5,
    rounds=5,
    after_clifford_depolarization=0.004,
    before_round_data_depolarization=0.004,
    before_measure_flip_probability=0.004,
    after_reset_flip_probability=0.004,
)
dem = circuit.detector_error_model(decompose_errors=True)
dets, obs = circuit.compile_detector_sampler(seed=11).sample(10_000, separate_observables=True)
print(dem.num_detectors, "detectors,", dem.num_errors, "faults,", len(dets), "shots")

# %% [markdown]
# ## Five decoders on the same shots
#
# - `Matching`: minimum-weight perfect matching by sparse blossom (exact, as PyMatching's).
# - `Matching(..., enable_correlations=True)`: two passes, so that a Y error's X and Z halves
#   reweight each other.
# - `BeliefMatching`: belief propagation's posteriors as the matching weights.
# - `UnionFind`: weighted union-find, the classic near-linear baseline.
# - `BpOsd`: BP with ordered-statistics post-processing. It needs no decomposition, so it
#   also decodes codes whose faults flip three or more detectors.

# %%
decoders = {
    "matching": lambda: sq.Matching(dem),
    "correlated matching": lambda: sq.Matching(dem, enable_correlations=True),
    "belief-matching": lambda: sq.BeliefMatching(dem),
    "union-find": lambda: sq.UnionFind(dem),
    # BP rarely settles on the surface code's degenerate errors; capping its iterations keeps
    # it quick, and OSD finishes the job.
    "BP+OSD": lambda: sq.BpOsd(circuit.detector_error_model(), max_iter=30, osd_order=4),
}
print(f"{'decoder':22} {'logical error rate':>18} {'µs per shot':>12}")
for name, make in decoders.items():
    decoder = make()
    t0 = time.perf_counter()
    predicted = decoder.decode_batch(dets)
    seconds = time.perf_counter() - t0
    rate = (predicted != obs).any(axis=1).mean()
    print(f"{name:22} {rate:18.4f} {1e6 * seconds / len(dets):12.1f}")

# %% [markdown]
# Correlated matching and belief-matching fail less often than plain matching; correlated
# matching costs only twice the time, belief-matching hundreds of times (BP runs on every
# shot). Union-find fails a little more often than matching. BP+OSD is not built for the
# surface code, but it is the decoder for the codes in the next tutorial.
#
# ## Decoding in windows
#
# A real machine cannot wait for the experiment to end. A *window* decoder decodes a block of
# rounds as soon as they and a buffer after them have arrived, commits to the corrections in
# its first `commit` rounds, and slides on. In `"parallel"` mode the windows are decoded
# independently and stitched together, so many can run at once.

# %%
long = sq.Circuit.generated(
    "surface_code:rotated_memory_z", distance=5, rounds=60,
    after_clifford_depolarization=0.002, before_round_data_depolarization=0.002,
    before_measure_flip_probability=0.002, after_reset_flip_probability=0.002,
)
long_dem = long.detector_error_model(decompose_errors=True)
d, o = long.compile_detector_sampler(seed=3).sample(5_000, separate_observables=True)
whole = (sq.Matching(long_dem).decode_batch(d) != o).any(axis=1).mean()
for mode in ("sliding", "parallel"):
    w = sq.WindowMatching(long_dem, commit=5, buffer=5, mode=mode)
    windowed = (w.decode_batch(d) != o).any(axis=1).mean()
    print(f"{mode:8}: {windowed:.4f}   (the whole history at once: {whole:.4f})")

# %% [markdown]
# Windowed decoding costs little accuracy once the buffer covers the distance.
#
# ## A memory too long to write down
#
# `stream_memory` samples a memory round by round and decodes it in windows as it streams.
# The windows' graphs are built once, from a short template of the circuit's loop, so its
# length is limited by time, not memory. Here are 20,000 rounds of a distance-5 code, in
# 64 independent streams:

# %%
result = sq.stream_memory(
    circuit=sq.Circuit.generated(
        "surface_code:rotated_memory_z", distance=5, rounds=20_000,
        after_clifford_depolarization=0.001, before_round_data_depolarization=0.001,
        before_measure_flip_probability=0.001, after_reset_flip_probability=0.001,
    ),
    commit=5, buffer=5, shots=64, seed=1,
)
per_window = np.median(result.window_seconds) * 1e6
per_round = 1 - (1 - result.failures / result.shots) ** (1 / 20_000)
print(f"{result.failures} of {result.shots} streams failed over 20,000 rounds: about {per_round:.1e} per round")
print(f"{len(result.windows)} windows, median {per_window:.0f} µs per window; {result.seconds:.1f} s in all")

# %% [markdown]
# A logical error rate per round of a few times 10⁻⁵ means that holding a qubit for 20,000
# rounds fails about half the time: long computations need larger codes, or lower noise.
