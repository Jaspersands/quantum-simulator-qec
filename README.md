# Quantum Error Correction (QEC) Stabilizer Simulator & Decoder ⚛️🦀

[![Language](https://img.shields.io/badge/language-Rust%202021-orange.svg)](Cargo.toml)
[![WASM](https://img.shields.io/badge/webassembly-supported-purple.svg)](stabilizer_qec.wasm)
[![Python](https://img.shields.io/badge/python-PyO3-blue.svg)](src/lib.rs)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A high-performance **Quantum Error Correction (QEC) Stabilizer Simulator and Union-Find Decoder** written in Rust, compiling to **WebAssembly (WASM)** for interactive browser visual debugging and **PyO3 C-extensions** for high-throughput Python benchmarking.

---

## 🌟 Overview & Highlights

Fault-tolerant quantum computing requires quantum error correction codes to protect fragile logical qubits against physical environmental noise. This repository implements a complete end-to-end stabilizer simulation engine based on the **Gottesman-Knill Theorem**, specialized for **Rotated Surface Codes** and **XZZX Codes**.

### Key Features
* **⚡ High-Performance Symplectic Stabilizer Engine**: $O(N^2)$ binary symplectic tableau simulation for $N$-qubit Pauli propagation and measurement projections.
* **🛡️ Surface Code Architectures**:
  * **Rotated Surface Code**: Standard rotated planar lattice requiring $d^2$ data qubits and $d^2-1$ syndrome measurement ancillas for code distance $d$.
  * **XZZX Code**: Biased-noise resilient surface code configuration offering superior thresholds under $Z$-biased noise.
* **🎲 Comprehensive Noise Models**:
  * **Phenomenological Noise**: Data qubit Pauli errors ($X, Y, Z$) combined with measurement ancilla readout bitflips over $d$ syndrome cycles.
  * **Pure Data Noise**: Single-shot data qubit errors under perfect syndrome extraction.
  * **Biased Noise**: Tunable $Z$-bias ratio $\eta = p_z / (p_x + p_y)$.
  * **Erasure & Leakage Noise**: Qubit loss and erasure error handling.
* **🔍 High-Speed Decoders**:
  * **Union-Find (UF) Decoder**: $O(N \alpha(N))$ disjoint-set syndrome cluster growth with tree peeling decoder.
  * **Minimum-Weight Perfect Matching (MWPM)** support.
* **🌐 Interactive WebAssembly Dashboard (`index.html`)**: Real-time visual web app for stepping through syndrome cycles, injecting physical errors, and watching cluster growth decoding.
* **🐍 PyO3 Python Module (`stabilizer_qec.so`)**: C-extension Python library (`import stabilizer_qec`) for Monte Carlo error threshold benchmarking.

---

## 📂 Repository Structure

```
quantum-simulator-qec/
├── Cargo.toml                  # Rust package manifest & dependencies (PyO3, rand)
├── index.html                  # Interactive WASM visual QEC web dashboard
├── stabilizer_qec.wasm         # Pre-compiled WebAssembly module
├── stabilizer_qec.so           # Compiled PyO3 C-extension binary module
├── run_benchmarks.py           # Phenomenological noise threshold benchmarking script
├── run_data_benchmarks.py      # Pure data noise threshold benchmarking script
├── threshold_plot.png          # Phenomenological noise threshold plot output
├── data_threshold_plot.png     # Data noise threshold plot output
└── src/
    ├── lib.rs                  # PyO3 module interface and WASM export bindings
    ├── tableau.rs              # Binary symplectic stabilizer tableau engine
    ├── simulator.rs            # Circuit execution & Pauli matrix operations
    ├── surface_code.rs         # Rotated & XZZX Surface code lattice logic
    └── decoder.rs              # Union-Find disjoint-set cluster growth decoder
```

---

## 📐 Mathematical Foundations & QEC Theory

### 1. Rotated Surface Code Lattice
For a rotated surface code of distance $d$, the code space encodes 1 logical qubit ($k=1$) into $N = d^2$ physical data qubits with $d^2-1$ stabilizer generators:
- **$X$-Check Stabilizers**: $A_v = \prod_{i \in \text{star}(v)} X_i$
- **$Z$-Check Stabilizers**: $B_f = \prod_{j \in \text{boundary}(f)} Z_j$

Logical operators correspond to non-trivial topological chains across the lattice boundaries:
- **Logical $X_L$**: Chain of $X$ operators spanning left to right.
- **Logical $Z_L$**: Chain of $Z$ operators spanning top to bottom.

```
       Z-check (Top)
   (q0) ----- (q1) ----- (q2)
    |           |          |
X-check       Z-check    X-check
    |           |          |
   (q3) ----- (q4) ----- (q5)
    |           |          |
X-check       Z-check    X-check
    |           |          |
   (q6) ----- (q7) ----- (q8)
```

### 2. Disjoint-Set Peeling Union-Find Decoding
When physical errors occur, syndrome measurement yields defects where stabilizer checks flip to $-1$. The Union-Find decoder resolves defects via two phases:
1. **Cluster Growth**: Defect nodes grow half-edges until clusters contain an even number of defect nodes or touch a boundary.
2. **Peeling Decoder**: Computes a spanning forest over clusters and eliminates syndrome defects from leaves to roots, establishing the correction chain.

---

## 🛠️ Build & Usage Guide

### Prerequisites
- **Rust Toolchain**: `rustc` and `cargo` 1.70+
- **Python**: 3.9+ with `numpy` and `matplotlib`
- **wasm-pack** (optional, for re-compiling WASM): `cargo install wasm-pack`

### 1. Building the Rust Library
```bash
cargo build --release
```

### 2. Compiling WebAssembly for Browser UI
To build the WASM module for `index.html`:
```bash
wasm-pack build --target web --release
```
Then serve `index.html` locally:
```bash
python3 -m http.server 8080
```
Open `http://localhost:8080` to interactively visualize rotated surface code syndrome extraction and Union-Find decoding!

### 3. Building Python PyO3 Bindings
To build and install the native Python module:
```bash
# Using maturin
pip install maturin
maturin develop --release
```

---

## 📊 Threshold Benchmarking

Run the included Monte Carlo scripts to simulate thousands of noise realizations across code distances $d \in \{3, 5, 7\}$:

### Phenomenological Noise Threshold Benchmark
```bash
python3 run_benchmarks.py
```
* Scans physical error rates $p \in [0.0001, 0.03]$ over $d$ measurement rounds.
* Generates `threshold_plot.png` demonstrating error suppression below the fault-tolerant threshold $p_{th} \approx 1.0\%$.

### Pure Data Noise Threshold Benchmark
```bash
python3 run_data_benchmarks.py
```
* Scans physical data error rates $p \in [0.005, 0.15]$ for single-shot syndrome extraction.
* Generates `data_threshold_plot.png` demonstrating data error threshold $p_{th} \approx 10.3\%$.

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
