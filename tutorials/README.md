# Tutorials

Nine notebooks, each runnable top to bottom in a few seconds with `pip install stabilizer-qec
matplotlib` (Stim and PyMatching, where installed, are compared against):

1. [Getting started](01_getting_started.ipynb): a surface-code memory from circuit to logical
   error rate: diagrams, the detector error model, sampling, matching, and a threshold plot.
2. [Decoders, and decoding in real time](02_decoders.ipynb): five decoders on the same shots,
   window decoding, and a 20,000-round memory streamed and decoded as it runs.
3. [Beyond the surface code](03_beyond_the_surface_code.ipynb): IBM's bivariate bicycle codes,
   hypergraph products of classical codes, and colour codes, decoded by BP+OSD.
4. [More decoders](04_more_decoders.ipynb): BP+LSD, Relay-BP, colour-code matching and a search
   decoder, each checked against its authors' package, on a colour code and the gross code.
5. [Beyond Pauli noise](05_beyond_pauli.ipynb): coherent over-rotations against their Pauli
   twirl, by the coherent sampler and the state vector; a decoder that knows; T gates and
   leakage.
6. [The other tools' APIs](06_the_other_tools_apis.ipynb): Stim's objects, PyMatching's graph,
   ldpc's decoders, CUDA-Q QEC's codes and decoders, and leakage-aware decoding, under the
   names those tools use.
7. [Clifford+T circuits, and coherent noise on every gate](07_clifford_t_and_coherent_noise.ipynb):
   the stabilizer-rank sampler on T gates exactly, and a magic-state preparation under
   over-rotation, where the twirl, the joint sampler and the state vector part ways.
8. [Fault-tolerant compilation, and what it would cost](08_compilation_and_estimation.ipynb):
   tqec's block graphs compiled and decoded, Loom's Eka interpreted, and Qualtran's and Azure's
   resource estimates to the bit.
9. [Real time, and the GPU](09_real_time_and_the_gpu.ipynb): decodes in the background, a decoder
   queue with latency statistics, a decoder server, and sampling and BP on the GPU equal to the
   CPU bit for bit.

Each notebook is built from the Python file beside it (percent format: `# %%` starts a cell) by
running it:

```bash
python tools/notebooks.py           # rebuild every notebook with its outputs
python tools/notebooks.py --check   # run them; fail if one raises or is out of date
```

Edit the `.py` file, not the notebook. CI runs `--check` on every push.
