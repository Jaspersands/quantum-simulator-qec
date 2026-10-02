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
- Stim's whole circuit language (every gate, all three bases, `MPP`, `SPP`, correlated and
  heralded noise, measurement feedback, instruction tags, Pauli-target observables), checked against Stim on random circuits that use
  every gate.
- Lattice surgery as single circuits, their error models equal to Stim's: Z⊗Z and X⊗X merges, a
  logical CNOT, merges in a row, and lines of patches merged at once.
- A [sinter](https://pypi.org/project/sinter/) adapter for threshold sweeps, and objects that
  pickle for `multiprocessing`.

```python
import stabilizer_qec as sq

circuit = sq.memory_circuit(distance=5, rounds=5, p=0.004)               # a d = 5 SD6 memory
dets, obs = circuit.compile_detector_sampler(seed=1).sample(100_000, separate_observables=True, threads=0)
dem = circuit.detector_error_model(decompose_errors=True)
pred = sq.Matching(dem, enable_correlations=True).decode_batch(dets, threads=0)
print((pred != obs).any(axis=1).mean())                                  # logical error rate
```

The gross code, decoded by BP+OSD:

```python
gross = sq.BivariateBicycleCode("gross")                                 # [[144, 12, 12]], 12 cycles
c = gross.memory_circuit(12, 0.003)
dets, obs = c.compile_detector_sampler(seed=2).sample(2_000, separate_observables=True)
pred = sq.BpOsd(c.detector_error_model()).decode_batch(dets, threads=0)
print((pred != obs).any(axis=1).mean())                                  # any of the 12 logicals wrong
```

In [sinter](https://pypi.org/project/sinter/)'s sweeps (`pip install "stabilizer-qec[sinter]"`):

```python
import sinter
from stabilizer_qec import sinter as sq_sinter

stats = sinter.collect(tasks=tasks, decoders=["sq_matching", "sq_belief_matching", "pymatching"],
                       custom_decoders=sq_sinter.sinter_decoders(), num_workers=8, max_shots=100_000)
```

The names follow Stim (`Circuit`, `compile_detector_sampler`, `separate_observables`),
PyMatching (`Matching`, `decode_batch`, `enable_correlations`) and `ldpc` (`BpDecoder`,
`BpOsdDecoder`, `error_channel`). A seed gives the same shots on any machine and thread count.
See the [changelog](https://github.com/Jaspersands/quantum-simulator-qec/blob/master/CHANGELOG.md).

The explainer, its figures running the engine in the browser, is at
<https://qcompiler.jaspersands.com>, with a technical report. The source and the full write-up are at
<https://github.com/Jaspersands/quantum-simulator-qec>.
