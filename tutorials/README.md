# Tutorials

Three notebooks, each runnable top to bottom in a few seconds with `pip install stabilizer-qec
matplotlib` (Stim and PyMatching, where installed, are compared against):

1. [Getting started](01_getting_started.ipynb): a surface-code memory from circuit to logical
   error rate: diagrams, the detector error model, sampling, matching, and a threshold plot.
2. [Decoders, and decoding in real time](02_decoders.ipynb): five decoders on the same shots,
   window decoding, and a 20,000-round memory streamed and decoded as it runs.
3. [Beyond the surface code](03_beyond_the_surface_code.ipynb): IBM's bivariate bicycle codes,
   hypergraph products of classical codes, and colour codes, decoded by BP+OSD.

Each notebook is built from the Python file beside it (percent format: `# %%` starts a cell) by
running it:

```bash
python tools/notebooks.py           # rebuild every notebook with its outputs
python tools/notebooks.py --check   # run them; fail if one raises or is out of date
```

Edit the `.py` file, not the notebook. CI runs `--check` on every push.
