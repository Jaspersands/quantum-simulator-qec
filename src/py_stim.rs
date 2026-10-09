//! Bindings for Stim's stabilizer objects (`clifford`): the core of `PauliString`, `Tableau`,
//! `TableauSimulator`, the gate table and the instruction-level circuit. The Python classes in
//! `python/stabilizer_qec/_stim.py` give them Stim's interface; matrices and vectors cross as
//! bytes of little-endian complex128.

#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use crate::clifford::convert::{self, C64};
use crate::clifford::ir::{self, GateTarget, Instruction, Item};
use crate::clifford::pauli_string::PauliString;
use crate::clifford::tableau::Tableau;
use crate::clifford::tableau_sim::{gate_tableaus, Rng, TableauSimulator};
use crate::gate_data;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

fn complex_bytes<'py>(py: Python<'py>, v: &[C64]) -> Bound<'py, PyBytes> {
    let mut out = Vec::with_capacity(v.len() * 16);
    for (re, im) in v {
        out.extend_from_slice(&re.to_le_bytes());
        out.extend_from_slice(&im.to_le_bytes());
    }
    PyBytes::new(py, &out)
}

fn complex_from_bytes(b: &[u8]) -> PyResult<Vec<C64>> {
    if b.len() % 16 != 0 {
        return Err(err("complex data must be pairs of float64".into()));
    }
    Ok(b.chunks(16).map(|c| (f64::from_le_bytes(c[..8].try_into().unwrap()), f64::from_le_bytes(c[8..].try_into().unwrap()))).collect())
}

#[pyclass(name = "PauliStringCore", module = "stabilizer_qec._core", from_py_object)]
#[derive(Clone)]
pub struct PyPauli {
    pub p: PauliString,
}

#[pymethods]
impl PyPauli {
    #[staticmethod]
    fn from_text(text: &str) -> PyResult<Self> {
        Ok(PyPauli { p: PauliString::from_text(text).map_err(err)? })
    }

    #[staticmethod]
    fn from_paulis(paulis: Vec<u8>, phase: u8) -> Self {
        PyPauli { p: PauliString::from_fn(paulis.len(), phase, |k| paulis[k] & 3) }
    }

    #[staticmethod]
    fn random(n: usize, allow_imaginary: bool, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let phase = if allow_imaginary { (rng.next_u64() & 3) as u8 } else { ((rng.next_u64() & 1) * 2) as u8 };
        PyPauli { p: PauliString::from_fn(n, phase, |_| (rng.next_u64() & 3) as u8) }
    }

    #[staticmethod]
    #[pyo3(signature = (data, little_endian, unsigned))]
    fn from_unitary(data: &[u8], little_endian: bool, unsigned: bool) -> PyResult<Self> {
        let m = complex_from_bytes(data)?;
        Ok(PyPauli { p: PauliString::from_unitary_matrix(&m, little_endian, unsigned).map_err(err)? })
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __str__(&self) -> String {
        self.p.to_string()
    }

    fn __eq__(&self, o: &PyPauli) -> bool {
        self.p == o.p
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.p.hash(&mut h);
        h.finish()
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.p.num_qubits()
    }

    #[getter]
    fn phase(&self) -> u8 {
        self.p.phase
    }

    #[setter]
    fn set_phase(&mut self, v: u8) {
        self.p.phase = v & 3;
    }

    fn get(&self, k: usize) -> u8 {
        self.p.get(k)
    }

    fn set(&mut self, k: usize, v: u8) {
        self.p.set(k, v & 3)
    }

    /// One byte per qubit, Stim's numbering 0..4.
    fn paulis<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let v: Vec<u8> = (0..self.p.num_qubits()).map(|k| self.p.get(k)).collect();
        PyBytes::new(py, &v)
    }

    fn mul(&self, o: &PyPauli) -> Self {
        PyPauli { p: self.p.mul(&o.p) }
    }

    fn tensor(&self, o: &PyPauli) -> Self {
        PyPauli { p: self.p.tensor(&o.p) }
    }

    fn tensor_power(&self, k: usize) -> Self {
        PyPauli { p: self.p.tensor_power(k) }
    }

    fn select(&self, indices: Vec<usize>) -> Self {
        PyPauli { p: self.p.select(&indices) }
    }

    fn commutes(&self, o: &PyPauli) -> bool {
        self.p.commutes(&o.p)
    }

    fn weight(&self) -> usize {
        self.p.weight()
    }

    fn pauli_indices(&self, included: &str) -> PyResult<Vec<usize>> {
        self.p.pauli_indices(included).map_err(err)
    }

    fn to_unitary<'py>(&self, py: Python<'py>, little_endian: bool) -> Bound<'py, PyBytes> {
        complex_bytes(py, &self.p.to_unitary_matrix(little_endian))
    }

    fn to_tableau(&self) -> PyTableauCore {
        PyTableauCore { t: Tableau::from_pauli_string(&self.p) }
    }

    /// Conjugated by a tableau on `targets`: forward (U P U†), or backward (U† P U).
    fn conjugate_by_tableau(&self, t: &PyTableauCore, targets: Vec<usize>, backward: bool) -> PyResult<Self> {
        if targets.len() != t.t.num_qubits() {
            return Err(err(format!("len(targets) != len(tableau): {} != {}", targets.len(), t.t.num_qubits())));
        }
        let mut p = self.p.clone();
        let need = targets.iter().map(|&q| q + 1).max().unwrap_or(0);
        if need > p.num_qubits() {
            return Err(err(format!("target {} is beyond the Pauli string's {} qubits", need - 1, p.num_qubits())));
        }
        if backward {
            t.t.inverse(false).apply_within(&mut p, &targets);
        } else {
            t.t.apply_within(&mut p, &targets);
        }
        Ok(PyPauli { p })
    }

    /// Conjugated by a Clifford circuit's operations, forward or backward.
    fn conjugate_by_circuit(&self, text: &str, backward: bool) -> PyResult<Self> {
        let c = ir::Circuit::parse(text).map_err(err)?;
        let mut ops = Vec::new();
        c.for_each_operation(&mut |op| ops.push(op.clone()));
        if backward {
            ops.reverse();
        }
        let mut p = self.p.clone();
        for op in &ops {
            conjugate_by_instruction(&mut p, op, backward)?;
        }
        Ok(PyPauli { p })
    }
}

fn conjugate_by_instruction(p: &mut PauliString, op: &Instruction, backward: bool) -> PyResult<()> {
    let f = op.gate.flags;
    if f & gate_data::FLAG_HAS_NO_EFFECT_ON_QUBITS != 0 || matches!(op.gate.name, "I" | "II" | "TICK") {
        return Ok(());
    }
    if !op.gate.is_unitary {
        return Err(err(format!("Operation is not unitary: {op}")));
    }
    let (fwd, inv) = gate_tableaus(op.gate.name).ok_or_else(|| err(format!("Not a tableau gate: {op}")))?;
    let t = if backward { inv } else { fwd };
    let width = t.num_qubits();
    let mut groups: Vec<&[GateTarget]> = op.targets.chunks(width).collect();
    if backward {
        groups.reverse();
    }
    for g in groups {
        let q: Vec<usize> = g.iter().map(|t| t.value() as usize).collect();
        if q.iter().any(|&x| x >= p.num_qubits()) {
            p.ensure_num_qubits(q.iter().max().unwrap() + 1);
        }
        t.apply_within(p, &q);
    }
    Ok(())
}

#[pyclass(name = "TableauCore", module = "stabilizer_qec._core", from_py_object)]
#[derive(Clone)]
pub struct PyTableauCore {
    pub t: Tableau,
}

#[pymethods]
impl PyTableauCore {
    #[staticmethod]
    fn identity(n: usize) -> Self {
        PyTableauCore { t: Tableau::identity(n) }
    }

    #[staticmethod]
    fn from_named_gate(name: &str) -> PyResult<Self> {
        Ok(PyTableauCore { t: Tableau::from_named_gate(name).map_err(err)? })
    }

    #[staticmethod]
    fn from_conjugated_generators(xs: Vec<PyPauli>, zs: Vec<PyPauli>) -> PyResult<Self> {
        Ok(PyTableauCore { t: Tableau::from_conjugated_generators(xs.into_iter().map(|p| p.p).collect(), zs.into_iter().map(|p| p.p).collect()).map_err(err)? })
    }

    #[staticmethod]
    fn from_stabilizers(stabilizers: Vec<PyPauli>, allow_redundant: bool, allow_underconstrained: bool, invert: bool) -> PyResult<Self> {
        let s: Vec<PauliString> = stabilizers.into_iter().map(|p| p.p).collect();
        if s.iter().any(|p| p.is_imaginary()) {
            return Err(err("Stabilizers can't have imaginary sign.".into()));
        }
        Ok(PyTableauCore { t: convert::stabilizers_to_tableau(&s, allow_redundant, allow_underconstrained, invert).map_err(err)? })
    }

    #[staticmethod]
    fn from_state_vector(data: &[u8], little_endian: bool) -> PyResult<Self> {
        Ok(PyTableauCore { t: convert::state_vector_to_tableau(&complex_from_bytes(data)?, little_endian).map_err(err)? })
    }

    #[staticmethod]
    fn from_unitary(data: &[u8], little_endian: bool) -> PyResult<Self> {
        Ok(PyTableauCore { t: convert::unitary_to_tableau(&complex_from_bytes(data)?, little_endian).map_err(err)? })
    }

    #[staticmethod]
    fn from_circuit(text: &str, ignore_noise: bool, ignore_measurement: bool, ignore_reset: bool) -> PyResult<Self> {
        let c = ir::Circuit::parse(text).map_err(err)?;
        Ok(PyTableauCore { t: convert::circuit_to_tableau(&c, ignore_noise, ignore_measurement, ignore_reset, false).map_err(err)? })
    }

    #[staticmethod]
    fn random(n: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        PyTableauCore { t: convert::random_tableau(n, &mut rng) }
    }

    /// Rows of bits: (x2x, x2z, z2x, z2z) each n×n row-major as bytes, then x_signs, z_signs.
    fn to_bits<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        let n = self.t.num_qubits();
        let quad = |rows: &[PauliString], z: bool| -> Vec<u8> {
            let mut v = Vec::with_capacity(n * n);
            for r in rows {
                for c in 0..n {
                    v.push(if z { r.z(c) } else { r.x(c) } as u8);
                }
            }
            v
        };
        let xs: Vec<u8> = self.t.xs.iter().map(|p| p.sign() as u8).collect();
        let zs: Vec<u8> = self.t.zs.iter().map(|p| p.sign() as u8).collect();
        PyTuple::new(
            py,
            [
                PyBytes::new(py, &quad(&self.t.xs, false)),
                PyBytes::new(py, &quad(&self.t.xs, true)),
                PyBytes::new(py, &quad(&self.t.zs, false)),
                PyBytes::new(py, &quad(&self.t.zs, true)),
                PyBytes::new(py, &xs),
                PyBytes::new(py, &zs),
            ],
        )
        .unwrap()
    }

    #[staticmethod]
    fn from_bits(n: usize, x2x: &[u8], x2z: &[u8], z2x: &[u8], z2z: &[u8], x_signs: &[u8], z_signs: &[u8]) -> PyResult<Self> {
        let row = |xb: &[u8], zb: &[u8], r: usize, s: u8| PauliString::from_fn(n, if s != 0 { 2 } else { 0 }, |c| crate::clifford::pauli_string::pauli_from_xz(xb[r * n + c] != 0, zb[r * n + c] != 0));
        let xs = (0..n).map(|r| row(x2x, x2z, r, x_signs[r])).collect();
        let zs = (0..n).map(|r| row(z2x, z2z, r, z_signs[r])).collect();
        Ok(PyTableauCore { t: Tableau::from_conjugated_generators(xs, zs).map_err(err)? })
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __str__(&self) -> String {
        self.t.to_string()
    }

    fn __eq__(&self, o: &PyTableauCore) -> bool {
        self.t == o.t
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.t.hash(&mut h);
        h.finish()
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.t.num_qubits()
    }

    fn x_output(&self, k: usize) -> PyPauli {
        PyPauli { p: self.t.xs[k].clone() }
    }

    fn y_output(&self, k: usize) -> PyPauli {
        PyPauli { p: self.t.y_output(k) }
    }

    fn z_output(&self, k: usize) -> PyPauli {
        PyPauli { p: self.t.zs[k].clone() }
    }

    fn output_pauli(&self, which: u8, input: usize, output: usize) -> u8 {
        self.t.output_pauli(which, input, output)
    }

    fn inverse_output_pauli(&self, which: u8, input: usize, output: usize) -> u8 {
        self.t.inverse_output_pauli(which, input, output)
    }

    fn inverse_output(&self, which: u8, input: usize, unsigned: bool) -> PyPauli {
        PyPauli { p: self.t.inverse_output(which, input, unsigned) }
    }

    fn inverse(&self, unsigned: bool) -> Self {
        PyTableauCore { t: self.t.inverse(unsigned) }
    }

    fn raised_to(&self, e: i64) -> Self {
        PyTableauCore { t: self.t.raised_to(e) }
    }

    fn then(&self, o: &PyTableauCore) -> PyResult<Self> {
        if o.t.num_qubits() != self.t.num_qubits() {
            return Err(err("len(self) != len(second)".into()));
        }
        Ok(PyTableauCore { t: self.t.then(&o.t) })
    }

    fn tensor(&self, o: &PyTableauCore) -> Self {
        PyTableauCore { t: self.t.tensor(&o.t) }
    }

    fn apply(&self, p: &PyPauli) -> PyResult<PyPauli> {
        Ok(PyPauli { p: self.t.apply(&p.p).map_err(err)? })
    }

    fn append(&mut self, op: &PyTableauCore, targets: Vec<usize>) -> PyResult<()> {
        check_targets(&targets, op.t.num_qubits(), self.t.num_qubits())?;
        self.t.append(&op.t, &targets);
        Ok(())
    }

    fn prepend(&mut self, op: &PyTableauCore, targets: Vec<usize>) -> PyResult<()> {
        check_targets(&targets, op.t.num_qubits(), self.t.num_qubits())?;
        self.t.prepend(&op.t, &targets);
        Ok(())
    }

    fn is_pauli_product(&self) -> bool {
        self.t.is_pauli_product()
    }

    fn to_pauli_string(&self) -> PyResult<PyPauli> {
        Ok(PyPauli { p: self.t.to_pauli_string().map_err(err)? })
    }

    fn stabilizers(&self, canonical: bool) -> Vec<PyPauli> {
        self.t.stabilizers(canonical).into_iter().map(|p| PyPauli { p }).collect()
    }

    fn to_circuit(&self, method: &str) -> PyResult<String> {
        Ok(convert::tableau_to_circuit(&self.t, method).map_err(err)?.exact_text())
    }

    fn to_state_vector<'py>(&self, py: Python<'py>, little_endian: bool) -> PyResult<Bound<'py, PyBytes>> {
        Ok(complex_bytes(py, &convert::tableau_to_state_vector(&self.t, little_endian).map_err(err)?))
    }

    fn to_unitary<'py>(&self, py: Python<'py>, little_endian: bool) -> PyResult<Bound<'py, PyBytes>> {
        Ok(complex_bytes(py, &convert::tableau_to_unitary(&self.t, little_endian).map_err(err)?))
    }
}

fn check_targets(targets: &[usize], width: usize, n: usize) -> PyResult<()> {
    if targets.len() != width {
        return Err(err(format!("len(targets) != len(gate): {} != {}", targets.len(), width)));
    }
    let mut seen = std::collections::BTreeSet::new();
    for &t in targets {
        if t >= n {
            return Err(err(format!("target {t} >= len(tableau) {n}")));
        }
        if !seen.insert(t) {
            return Err(err("Collision in targets.".into()));
        }
    }
    Ok(())
}

#[pyclass(name = "TableauSimulatorCore", module = "stabilizer_qec._core")]
pub struct PyTableauSimulatorCore {
    s: TableauSimulator,
}

#[pymethods]
impl PyTableauSimulatorCore {
    #[new]
    fn new(seed: u64) -> Self {
        PyTableauSimulatorCore { s: TableauSimulator::new(0, seed) }
    }

    /// A copy; with `seed`, its own random stream, else this one's state of the stream.
    #[pyo3(signature = (copy_rng, seed=None))]
    fn copy(&self, copy_rng: bool, seed: Option<u64>) -> Self {
        let mut s = self.s.clone();
        if !copy_rng {
            s.rng = Rng::new(seed.unwrap_or_else(rand::random));
        }
        PyTableauSimulatorCore { s }
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.s.num_qubits()
    }

    fn set_num_qubits(&mut self, n: usize) {
        self.s.set_num_qubits(n);
    }

    fn ensure(&mut self, n: usize) {
        self.s.ensure_large_enough_for_qubits(n);
    }

    fn do_circuit(&mut self, text: &str) -> PyResult<()> {
        let c = ir::Circuit::parse(text).map_err(err)?;
        self.s.do_circuit(&c).map_err(err)
    }

    /// One instruction from its parts; the targets are Stim's 32-bit encodings.
    fn do_instruction(&mut self, name: &str, args: Vec<f64>, targets: Vec<u32>, tag: &str) -> PyResult<()> {
        let inst = Instruction::new(name, args, targets.into_iter().map(GateTarget).collect(), tag).map_err(err)?;
        let n = inst.targets.iter().filter(|t| t.has_qubit_value() && inst.gate.name != "MPAD").map(|t| t.value() as usize + 1).max().unwrap_or(0);
        self.s.ensure_large_enough_for_qubits(n);
        self.s.do_gate(&inst).map_err(err)
    }

    fn do_tableau(&mut self, t: &PyTableauCore, targets: Vec<usize>) -> PyResult<()> {
        check_targets(&targets, t.t.num_qubits(), usize::MAX)?;
        let n = targets.iter().map(|&q| q + 1).max().unwrap_or(0);
        self.s.ensure_large_enough_for_qubits(n);
        self.s.inv_state.prepend(&t.t.inverse(false), &targets);
        Ok(())
    }

    fn do_pauli_string(&mut self, p: &PyPauli) {
        self.s.ensure_large_enough_for_qubits(p.p.num_qubits());
        for q in 0..p.p.num_qubits() {
            if p.p.x(q) {
                self.s.inv_state.zs[q].phase ^= 2;
            }
            if p.p.z(q) {
                self.s.inv_state.xs[q].phase ^= 2;
            }
        }
    }

    fn measure_pauli_string(&mut self, p: &PyPauli, flip_probability: f64) -> PyResult<bool> {
        if p.p.is_imaginary() {
            return Err(err("Observable isn't Hermitian; it has imaginary sign. Need observable.sign in [1, -1].".into()));
        }
        self.s.measure_pauli_string(&p.p, flip_probability).map_err(err)
    }

    fn measure_kickback(&mut self, q: usize, basis: u8) -> (bool, Option<PyPauli>) {
        self.s.ensure_large_enough_for_qubits(q + 1);
        let (b, k) = self.s.measure_kickback(q, basis, false);
        (b, k.map(|p| PyPauli { p }))
    }

    fn peek(&mut self, q: usize, basis: u8) -> i8 {
        self.s.ensure_large_enough_for_qubits(q + 1);
        match basis {
            1 => self.s.peek_x(q),
            2 => self.s.peek_y(q),
            _ => self.s.peek_z(q),
        }
    }

    fn peek_bloch(&mut self, q: usize) -> PyPauli {
        self.s.ensure_large_enough_for_qubits(q + 1);
        PyPauli { p: self.s.peek_bloch(q) }
    }

    fn peek_observable_expectation(&self, p: &PyPauli) -> PyResult<i8> {
        if p.p.is_imaginary() {
            return Err(err("Observable isn't Hermitian; it has imaginary sign. Need observable.sign in [1, -1].".into()));
        }
        Ok(self.s.peek_observable_expectation(&p.p))
    }

    fn postselect(&mut self, qubits: Vec<usize>, desired: bool, basis: u8) -> PyResult<()> {
        let n = qubits.iter().map(|&q| q + 1).max().unwrap_or(0);
        self.s.ensure_large_enough_for_qubits(n);
        self.s.postselect(&qubits, desired, basis).map_err(err)
    }

    fn postselect_observable(&mut self, p: &PyPauli, desired: bool) -> PyResult<()> {
        if p.p.is_imaginary() {
            return Err(err("Observable isn't Hermitian; it has imaginary sign. Need observable.sign in [1, -1].".into()));
        }
        self.s.postselect_observable(&p.p, desired).map_err(err)
    }

    fn canonical_stabilizers(&self) -> Vec<PyPauli> {
        self.s.canonical_stabilizers().into_iter().map(|p| PyPauli { p }).collect()
    }

    fn current_inverse_tableau(&self) -> PyTableauCore {
        PyTableauCore { t: self.s.inv_state.clone() }
    }

    fn set_inverse_tableau(&mut self, t: &PyTableauCore) {
        self.s.inv_state = t.t.clone();
    }

    fn current_measurement_record(&self) -> Vec<bool> {
        self.s.measurement_record.clone()
    }

    fn state_vector<'py>(&self, py: Python<'py>, little_endian: bool) -> PyResult<Bound<'py, PyBytes>> {
        let t = self.s.inv_state.inverse(false);
        Ok(complex_bytes(py, &convert::tableau_to_state_vector(&t, little_endian).map_err(err)?))
    }
}

/// Every gate's data, as dicts the Python `GateData` reads.
#[pyfunction]
fn gate_data_table<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for g in gate_data::GATE_INFO {
        let d = PyDict::new(py);
        d.set_item("name", g.name)?;
        d.set_item("aliases", g.aliases.to_vec())?;
        d.set_item("is_noisy_gate", g.is_noisy_gate)?;
        d.set_item("is_reset", g.is_reset)?;
        d.set_item("is_single_qubit_gate", g.is_single_qubit_gate)?;
        d.set_item("is_symmetric_gate", g.is_symmetric_gate)?;
        d.set_item("is_two_qubit_gate", g.is_two_qubit_gate)?;
        d.set_item("is_unitary", g.is_unitary)?;
        d.set_item("produces_measurements", g.produces_measurements)?;
        d.set_item("takes_measurement_record_targets", g.takes_measurement_record_targets)?;
        d.set_item("takes_pauli_targets", g.takes_pauli_targets)?;
        d.set_item("args", g.args)?;
        d.set_item("tableau", g.tableau.to_vec())?;
        d.set_item("unitary", g.unitary.to_vec())?;
        d.set_item("flows", g.flows.to_vec())?;
        d.set_item("inverse", g.inverse)?;
        d.set_item("generalized_inverse", g.generalized_inverse)?;
        d.set_item("hadamard_conjugated", g.hadamard_conjugated)?;
        d.set_item("hadamard_conjugated_unsigned", g.hadamard_conjugated_unsigned)?;
        d.set_item("flags", g.flags)?;
        d.set_item("arg_count", g.arg_count)?;
        d.set_item("category", g.category)?;
        d.set_item("decomposition", g.decomposition)?;
        out.append(d)?;
    }
    Ok(out)
}

/// The circuit text as Stim prints it (fused, canonical names, `%g` arguments).
#[pyfunction]
fn circuit_stim_text(text: &str) -> PyResult<String> {
    Ok(ir::Circuit::parse(text).map_err(err)?.to_string())
}

/// The circuit text fused and canonical but with exact arguments.
#[pyfunction]
fn circuit_exact_text(text: &str) -> PyResult<String> {
    Ok(ir::Circuit::parse(text).map_err(err)?.exact_text())
}

fn items_to_py<'py>(py: Python<'py>, c: &ir::Circuit) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for it in &c.items {
        match it {
            Item::Op(op) => {
                let t: Vec<u32> = op.targets.iter().map(|t| t.0).collect();
                out.append(("op", op.gate.name, op.tag.clone(), op.args.clone(), t))?;
            }
            Item::Repeat { count, body, tag } => {
                out.append(("repeat", *count, tag.clone(), items_to_py(py, body)?))?;
            }
        }
    }
    Ok(out)
}

/// Whether two circuits are equal up to arguments within `atol`.
#[pyfunction]
fn circuit_approx_equals(a: &str, b: &str, atol: f64) -> PyResult<bool> {
    let a = ir::Circuit::parse(a).map_err(err)?;
    let b = ir::Circuit::parse(b).map_err(err)?;
    Ok(a.approx_equals(&b, atol))
}

/// The circuit's items: ("op", name, tag, args, targets) or ("repeat", count, tag, items).
#[pyfunction]
fn circuit_items<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyList>> {
    let c = ir::Circuit::parse(text).map_err(err)?;
    items_to_py(py, &c)
}

/// One instruction's line (Stim's format) from its parts.
#[pyfunction]
#[pyo3(signature = (name, args, targets, tag, exact=false))]
fn instruction_text(name: &str, args: Vec<f64>, targets: Vec<u32>, tag: &str, exact: bool) -> PyResult<String> {
    let op = Instruction::new(name, args, targets.into_iter().map(GateTarget).collect(), tag).map_err(err)?;
    Ok(if exact { op.exact_line() } else { op.to_string() })
}

/// Parse one target's text into Stim's 32-bit encoding.
#[pyfunction]
fn parse_gate_target(text: &str) -> PyResult<u32> {
    if text == "*" {
        return Ok(ir::TARGET_COMBINER);
    }
    ir::parse_target(text).map(|t| t.0).map_err(err)
}

/// A target's succinct text.
#[pyfunction]
fn gate_target_text(bits: u32) -> String {
    GateTarget(bits).to_string()
}

/// How many measurements an instruction records.
#[pyfunction]
fn instruction_num_measurements(name: &str, targets: Vec<u32>) -> PyResult<u64> {
    let op = Instruction::new(name, vec![], targets.into_iter().map(GateTarget).collect(), "").map_err(err)?;
    Ok(op.count_measurement_results())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPauli>()?;
    m.add_class::<PyTableauCore>()?;
    m.add_class::<PyTableauSimulatorCore>()?;
    m.add_function(wrap_pyfunction!(gate_data_table, m)?)?;
    m.add_function(wrap_pyfunction!(circuit_stim_text, m)?)?;
    m.add_function(wrap_pyfunction!(circuit_exact_text, m)?)?;
    m.add_function(wrap_pyfunction!(circuit_items, m)?)?;
    m.add_function(wrap_pyfunction!(circuit_approx_equals, m)?)?;
    m.add_function(wrap_pyfunction!(instruction_text, m)?)?;
    m.add_function(wrap_pyfunction!(parse_gate_target, m)?)?;
    m.add_function(wrap_pyfunction!(gate_target_text, m)?)?;
    m.add_function(wrap_pyfunction!(instruction_num_measurements, m)?)?;
    Ok(())
}
