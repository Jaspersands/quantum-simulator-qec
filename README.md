# Quantum Error Correction (QEC) Simulator

A Rust stabilizer circuit simulator and decoder for Rotated Surface Codes and XZZX Codes. Compiles to WebAssembly for an interactive browser visualizer and PyO3 Python bindings (`stabilizer_qec`) for Monte Carlo threshold benchmarking.

## Key Features

- **Stabilizer Engine**: Binary symplectic tableau simulator ($O(N^2)$ Pauli propagation) based on the Gottesman-Knill theorem.
- **Supported Codes**:
  - Rotated Surface Code ($d^2$ data qubits, $d^2-1$ ancillas).
  - XZZX Surface Code for $Z$-biased noise.
- **Noise Models**: Phenomenological noise, pure data noise, readout error, and $Z$-bias scaling.
- **Decoders**: Disjoint-set Union-Find (UF) cluster peeling decoder and Minimum-Weight Perfect Matching (MWPM).
- **Web Visualizer (`index.html`)**: Interactive WASM interface for stepping through syndrome extraction cycles and watching cluster growth decoding.
- **Python Extension (`stabilizer_qec.so`)**: PyO3 bindings used for threshold benchmarking.

## Repository Structure

- `src/tableau.rs` – Symplectic tableau representation and Clifford operations.
- `src/surface_code.rs` – Rotated and XZZX surface code lattice setup and error generation.
- `src/decoder.rs` – Union-Find cluster growth and peeling decoder.
- `src/lib.rs` – PyO3 module and WASM interface.
- `index.html` – Web visualizer interface.
- `run_benchmarks.py` – Phenomenological noise threshold benchmarking script.
- `run_data_benchmarks.py` – Data noise threshold benchmarking script.

## Building & Running

### Rust Library
```bash
cargo build --release
```

### WebAssembly Visualizer
Build the WASM binary (requires `wasm-pack`):

```bash
wasm-pack build --target web --release
```

Then serve `index.html`:

```bash
python3 -m http.server 8080
```

Open `http://localhost:8080` in your browser.

### Python Bindings & Benchmarks
Install the PyO3 package into your environment (e.g. using `maturin`):

```bash
pip install maturin
maturin develop --release
```

Run threshold benchmarks to generate plot outputs (`threshold_plot.png` & `data_threshold_plot.png`):

```bash
python3 run_benchmarks.py
python3 run_data_benchmarks.py
```

## License

MIT
