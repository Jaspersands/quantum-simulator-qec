# stabilizer-qec

A surface-code simulator and decoder written in Rust, with Python bindings.

- Circuits and detector error models in [Stim](https://github.com/quantumlib/Stim)'s text formats,
  and error models built by walking a circuit backwards, identical to Stim's on every circuit
  checked.
- Exact minimum-weight matching by sparse blossom, and correlated matching, checked against
  [PyMatching](https://github.com/oscarhiggott/PyMatching) on finite corpora. Plain `Matching`
  reproduces recorded tied choices; separate correlated routes may choose another equal-weight
  answer and require an unedited DEM/circuit loaded with `enable_correlations=True`, with
  actual observable IDs 0–63 (wider ensured output is zero-padded).
- A bit-parallel sampler, 64 shots per machine word, that runs `REPEAT` blocks without flattening.
- Raw measurements to detection events, reproducing the recorded Google Willow and Sycamore files bit for bit.
- Sliding and parallel window decoders for real-time decoding, and decoding as a service
  (`decode_async`, a decoder queue with statistics, a decoder server).
- The APIs of Stim, PyMatching, sinter, ldpc, Tsim, tqec, Deltakit, PECOS and seventeen more QEC
  tools under their own names, each port checked against its original.
- Clifford+T circuits with a coherent error on every gate: a stabilizer-rank sampler for the
  gates and a joint sampler that reweights the twirl of the errors around them.
- Fault-tolerant compilation (tqec's block graphs, Loom's lattice surgery), resource estimates
  (Azure's and Qualtran's), and sampling and min-sum BP on the GPU through wgpu.
- Belief propagation reproducing the [`ldpc`](https://github.com/quantumgizmos/ldpc) library's
  arithmetic on the recorded controls, and belief-matching checked against `beliefmatching`,
  subject to documented ties, platform arithmetic and API limits.
- BP+OSD (OSD-0, OSD-E, OSD-CS), checked against `ldpc`'s `BpOsdDecoder` on recorded inputs,
  with documented tie limits, and IBM's
  bivariate bicycle codes: the gross code [[144, 12, 12]] and [[72, 12, 6]], with the depth-8
  syndrome cycle of Bravyi et al. (Nature 627, 778, 2024).
- On the gross code, every automorphism's action on the logical qubits, and the gauging measurement
  of a logical operator as a circuit, its error model equal to Stim's.
- Stim's whole circuit language (every gate, all three bases, `MPP`, `SPP`, correlated and
  heralded noise, measurement feedback, instruction tags, Pauli-target observables), checked against Stim on random circuits that use
  every gate.
- Lattice surgery as single circuits, their error models equal to Stim's: Z⊗Z and X⊗X merges, a
  logical CNOT, merges in a row, and lines of patches merged at once.
- [sinter](https://pypi.org/project/sinter/), whole (`stabilizer_qec.sinter`), for threshold
  sweeps with nothing else installed, and objects that pickle for `multiprocessing`.

```python
import stabilizer_qec as sq

circuit = sq.memory_circuit(distance=5, rounds=5, p=0.004)               # a d = 5 SD6 memory
dets, obs = circuit.compile_detector_sampler(seed=1).sample(100_000, separate_observables=True, threads=0)
dem = circuit.detector_error_model(decompose_errors=True)
pred = sq.Matching(dem, enable_correlations=True).decode_batch(dets, enable_correlations=True, threads=0)
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

Threshold sweeps with `stabilizer_qec.sinter`, which is sinter whole (matplotlib to plot):

```python
from stabilizer_qec import sinter

stats = sinter.collect(tasks=tasks, decoders=["pymatching", "belief_matching", "bposd"],
                       num_workers=8, max_shots=100_000)
```

The names follow Stim (`Circuit`, `compile_detector_sampler`, `separate_observables`),
PyMatching (`Matching`, `decode_batch`, `enable_correlations`) and `ldpc` (`BpDecoder`,
`BpOsdDecoder`, `error_channel`). Seed reproducibility applies within the documented
sampler/version contract and supported
thread counts. Algorithm corrections, including 2.2's coherent merged-proposal and stable-group
arithmetic fixes, can change seeded outcomes. Coherent whole-class sums require a completed
kernel and a fitting contraction plan (or at most twelve generators); larger/incomplete sums
are approximations and warn. `exact_weights` describes coverage in floating-point arithmetic,
not universal numerical validation. Entry budgets are not byte limits and successful planning
can dominate compilation.

The [root guide](https://github.com/Jaspersands/quantum-simulator-qec/blob/master/README.md)
documents port-specific refusals, exceptions, tie choices and migration changes. Crashes and
hangs remain bugs to report; resource checks do not make all allocation failures recoverable.
All coherent validation and capacity rows, and the d = 3 sweep and crosstalk rows, were
remeasured with the actual 2.2.0 wheel and carry the generation revision `coherent-merged-proposal-v1`, which records
provenance independently of exactness and validation; current tables accept only those rows.
Only the 12 larger-distance sweep and 5 crosstalk rows, measured by 1.8.0 before the
merged-proposal correction, remain archival and are excluded from current tables. Earlier
development-2.2 measurements remain in Git history and the preserved raw evidence.
See the [changelog](https://github.com/Jaspersands/quantum-simulator-qec/blob/master/CHANGELOG.md).

The explainer, its figures running the engine in the browser, is at
<https://qcompiler.jaspersands.com>, with a technical report. The source and the full write-up are at
<https://github.com/Jaspersands/quantum-simulator-qec>.
