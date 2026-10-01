# stabilizer-qec

A surface-code simulator and decoder written in Rust, with Python bindings.

- Circuits and detector error models in [Stim](https://github.com/quantumlib/Stim)'s text formats,
  and error models built by walking a circuit backwards, identical to Stim's on every circuit
  checked.
- Exact minimum-weight matching by sparse blossom, and correlated matching, agreeing with
  [PyMatching](https://github.com/oscarhiggott/PyMatching) shot for shot except for ties.
- A bit-parallel sampler, 64 shots per machine word, that runs `REPEAT` blocks without flattening.
- Raw measurements to detection events, reproducing Google's Willow and Sycamore files bit for bit.
- Sliding and parallel window decoders for real-time decoding.
- Belief propagation reproducing the [`ldpc`](https://github.com/quantumgizmos/ldpc) library's
  arithmetic bit for bit, and belief-matching equal to the `beliefmatching` package's.
- BP+OSD (OSD-0, OSD-E, OSD-CS), whose corrections equal `ldpc`'s `BpOsdDecoder`'s, and IBM's
  bivariate bicycle codes: the gross code [[144, 12, 12]] and [[72, 12, 6]], with the depth-8
  syndrome cycle of Bravyi et al. (Nature 627, 778, 2024).
- On the gross code, every automorphism's action on the logical qubits, and the gauging measurement
  of a logical operator as a circuit, its error model equal to Stim's.
- Lattice surgery as single circuits, their error models equal to Stim's: Z⊗Z and X⊗X merges, a
  logical CNOT, merges in a row, and lines of patches merged at once.

```python
import numpy as np, stabilizer_qec as sq

text = sq.generate_circuit("rotated", 5, 5, "sd6", 0.004)      # Stim-format circuit
dets, obs, _ = sq.sample_b8_batch(text, 100_000, seed=1)         # b8 rows
pred, _, errors, seconds = sq.decode_b8_own(text, dets, 100_000, threads=0, correlated=True)
failures = ((np.frombuffer(pred, "<u8") & 1) != (np.frombuffer(obs, np.uint8) & 1)).sum()
```

The gross code, decoded by BP+OSD:

```python
circuit = sq.bb_memory_circuit("gross", 12, 0.003)               # [[144, 12, 12]], 12 cycles
dem = sq.dem_from_circuit(circuit)                               # each fault its own column
dets, obs, _ = sq.sample_b8_batch(circuit, 2_000, seed=1)
pred, converged, seconds = sq.decode_b8_bposd(dem, dets, 2_000)  # BP+OSD-CS, order 7
truth = np.frombuffer(obs, np.uint8).reshape(-1, 2).view("<u2")[:, 0]
failures = (np.frombuffer(pred, "<u8") != truth).sum()           # any of the 12 logicals wrong
```

The explainer, its figures running the engine in the browser, is at
<https://qcompiler.jaspersands.com>, with a technical report. The source and the full write-up are at
<https://github.com/Jaspersands/quantum-simulator-qec>.
