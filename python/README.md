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

```python
import numpy as np, stabilizer_qec as sq

text = sq.generate_circuit("rotated", 5, 5, "sd6", 0.004)      # Stim-format circuit
dets, obs, _ = sq.sample_b8_batch(text, 100_000, seed=1)         # b8 rows
pred, _, errors, seconds = sq.decode_b8_own(text, dets, 100_000, threads=0, correlated=True)
failures = ((np.frombuffer(pred, "<u8") & 1) != (np.frombuffer(obs, np.uint8) & 1)).sum()
```

The explainer, with every figure computed live in the browser, is at
<https://qcompiler.jaspersands.com>. The source and the full write-up are at
<https://github.com/Jaspersands/quantum-simulator-qec>.
