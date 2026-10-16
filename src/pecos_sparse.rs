//! PECOS's sparse stabilizer simulator (`SparseSim`, its C++ `sparsesim.cpp` that Cython wraps as
//! `cySparseSim`), in Rust.
//!
//! The state is the stabilizer and destabilizer generators, each kept both by rows (generator ->
//! qubits) and by columns (qubit -> generators), for X and Z parts, with each stabilizer's phase:
//! generator `g` is `i^(signs_i) (-1)^(signs_minus) prod_q X_q^x Z_q^z`, X before Z on each qubit.
//!
//! WHY GATES ARE TABLES
//! --------------------
//! The C++ writes out every gate (thirty one-qubit ones and their two-qubit compositions) by hand.
//! Each is a Clifford conjugation, and a generator's form above is unique, so the state after a
//! gate is fixed by the images of X and Z on the qubits it acts on: `SparseSim::gate` applies
//! any gate given those images (with their phases), and the images PECOS's own gates have are
//! read off its compiled simulator (`stabilizer_qec.pecos`'s wrapper holds them). Measurement,
//! which chooses which stabilizer to replace, follows the C++ step for step: the anticommuting
//! stabilizer whose X parts (as stabilizer and destabilizer) are lightest, the smallest id among
//! equals (the C++ takes the first its hash set yields).

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};

/// Generator and qubit ids hash as themselves (the C++ `TrivialHash`, here with a multiply so
/// that runs of ids spread).
#[derive(Default)]
pub struct IdHash(u64);

impl Hasher for IdHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8 | b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_u32(&mut self, v: u32) {
        self.0 = (v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

pub type Set = HashSet<u32, BuildHasherDefault<IdHash>>;

fn toggle(s: &mut Set, v: u32) {
    if !s.remove(&v) {
        s.insert(v);
    }
}

/// One generator matrix, by columns and rows.
#[derive(Clone, Debug, Default)]
pub struct Generators {
    pub col_x: Vec<Set>,
    pub col_z: Vec<Set>,
    pub row_x: Vec<Set>,
    pub row_z: Vec<Set>,
}

impl Generators {
    fn empty(n: usize) -> Vec<Set> {
        (0..n).map(|_| Set::default()).collect()
    }
    fn ones(n: usize) -> Vec<Set> {
        (0..n as u32).map(|i| std::iter::once(i).collect()).collect()
    }
    /// Sets the (generator, qubit) entries' X and Z bits.
    fn set(&mut self, g: u32, q: u32, x: bool, z: bool) {
        let (gi, qi) = (g as usize, q as usize);
        if x {
            self.row_x[gi].insert(q);
            self.col_x[qi].insert(g);
        } else {
            self.row_x[gi].remove(&q);
            self.col_x[qi].remove(&g);
        }
        if z {
            self.row_z[gi].insert(q);
            self.col_z[qi].insert(g);
        } else {
            self.row_z[gi].remove(&q);
            self.col_z[qi].remove(&g);
        }
    }
}

/// A Pauli on a gate's qubits: `i^phase prod X^x Z^z` (bit j of `x` and `z` for its j-th qubit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Image {
    pub phase: u8,
    pub x: u8,
    pub z: u8,
}

impl Image {
    /// `self * other`, both in X-before-Z form: moving `other`'s Xs past `self`'s Zs.
    fn times(self, other: Image) -> Image {
        let swaps = (self.z & other.x).count_ones() as u8;
        Image { phase: (self.phase + other.phase + 2 * swaps) % 4, x: self.x ^ other.x, z: self.z ^ other.z }
    }
}

/// PECOS's sparse stabilizer state.
#[derive(Clone, Debug)]
pub struct SparseSim {
    pub num_qubits: usize,
    pub stabs: Generators,
    pub destabs: Generators,
    pub signs_minus: Set,
    pub signs_i: Set,
}

impl SparseSim {
    /// `num_qubits` qubits in |0...0>.
    pub fn new(num_qubits: usize) -> SparseSim {
        let mut s = SparseSim { num_qubits, stabs: Generators::default(), destabs: Generators::default(), signs_minus: Set::default(), signs_i: Set::default() };
        s.clear();
        s
    }

    /// Back to |0...0> (the C++ `clear`).
    pub fn clear(&mut self) {
        let n = self.num_qubits;
        self.stabs = Generators { col_x: Generators::empty(n), col_z: Generators::ones(n), row_x: Generators::empty(n), row_z: Generators::ones(n) };
        self.destabs = Generators { col_x: Generators::ones(n), col_z: Generators::empty(n), row_x: Generators::ones(n), row_z: Generators::empty(n) };
        self.signs_minus.clear();
        self.signs_i.clear();
    }

    fn phase(&self, g: u32) -> u8 {
        self.signs_i.contains(&g) as u8 + 2 * self.signs_minus.contains(&g) as u8
    }

    fn set_phase(&mut self, g: u32, k: u8) {
        if k & 1 == 1 {
            self.signs_i.insert(g);
        } else {
            self.signs_i.remove(&g);
        }
        if k & 2 == 2 {
            self.signs_minus.insert(g);
        } else {
            self.signs_minus.remove(&g);
        }
    }

    /// The gate taking X and Z on `qubits[j]` to `images[2j]` and `images[2j + 1]`.
    ///
    /// # Panics
    /// When `qubits` repeats a qubit, is out of range or has more than 8 qubits, or `images` is
    /// not two per qubit.
    pub fn gate(&mut self, qubits: &[usize], images: &[Image]) {
        let k = qubits.len();
        assert!(k <= 8 && images.len() == 2 * k, "a gate needs the images of X and Z on each of its (at most 8) qubits");
        for (a, &q) in qubits.iter().enumerate() {
            assert!(q < self.num_qubits, "qubit {q} out of range");
            assert!(!qubits[..a].contains(&q), "a gate's qubits must differ");
        }
        for stabs in [true, false] {
            let gens = if stabs { &self.stabs } else { &self.destabs };
            let mut touched: Vec<u32> = Vec::new();
            for &q in qubits {
                touched.extend(gens.col_x[q].iter().copied());
                touched.extend(gens.col_z[q].iter().copied());
            }
            touched.sort_unstable();
            touched.dedup();
            for g in touched {
                let gens = if stabs { &self.stabs } else { &self.destabs };
                let mut out = Image { phase: 0, x: 0, z: 0 };
                for (j, &q) in qubits.iter().enumerate() {
                    if gens.row_x[g as usize].contains(&(q as u32)) {
                        out = out.times(images[2 * j]);
                    }
                    if gens.row_z[g as usize].contains(&(q as u32)) {
                        out = out.times(images[2 * j + 1]);
                    }
                }
                let gens = if stabs { &mut self.stabs } else { &mut self.destabs };
                for (j, &q) in qubits.iter().enumerate() {
                    gens.set(g, q as u32, out.x >> j & 1 == 1, out.z >> j & 1 == 1);
                }
                if stabs {
                    let p = (self.phase(g) + out.phase) % 4;
                    self.set_phase(g, p);
                }
            }
        }
    }

    /// Whether measuring Z on `qubit` has a determined outcome.
    pub fn is_deterministic(&self, qubit: usize) -> bool {
        self.stabs.col_x[qubit].is_empty()
    }

    /// Measures Z on `qubit` (the C++ `measure`): a determined outcome is returned; otherwise
    /// `outcome` (0 or 1) is, and with `collapse` the state becomes the one it heralds.
    ///
    /// # Panics
    /// When the outcome is not determined and `outcome` is not 0 or 1.
    pub fn measure(&mut self, qubit: usize, outcome: u8, collapse: bool) -> u8 {
        if self.is_deterministic(qubit) {
            return self.deterministic_measure(qubit);
        }
        assert!(outcome < 2, "forced_outcome was not 0 or 1");
        if collapse {
            self.nondeterministic_measure(qubit, outcome);
        }
        outcome
    }

    fn deterministic_measure(&self, qubit: usize) -> u8 {
        let mut cumulative_x = Set::default();
        let (mut minuses, mut is) = (0usize, 0usize);
        for &g in &self.destabs.col_x[qubit] {
            minuses += self.signs_minus.contains(&g) as usize;
            is += self.signs_i.contains(&g) as usize;
            minuses += self.stabs.row_z[g as usize].iter().filter(|q| cumulative_x.contains(q)).count();
            for &q in &self.stabs.row_x[g as usize] {
                toggle(&mut cumulative_x, q);
            }
        }
        if is % 4 != 0 {
            minuses += 1;
        }
        (minuses % 2) as u8
    }

    fn nondeterministic_measure(&mut self, qubit: usize, outcome: u8) {
        let weight = |s: &SparseSim, g: u32| s.stabs.row_x[g as usize].len() + s.destabs.row_x[g as usize].len();
        let removed = *self.stabs.col_x[qubit].iter().min_by_key(|&&g| (weight(self, g), g)).expect("an anticommuting stabilizer");
        let mut anticom_stabs: Vec<u32> = self.stabs.col_x[qubit].iter().copied().filter(|&g| g != removed).collect();
        anticom_stabs.sort_unstable();
        let mut anticom_destabs: Vec<u32> = self.destabs.col_x[qubit].iter().copied().filter(|&g| g != removed).collect();
        anticom_destabs.sort_unstable();
        let removed_row_x = self.stabs.row_x[removed as usize].clone();
        let removed_row_z = self.stabs.row_z[removed as usize].clone();

        if self.signs_minus.contains(&removed) {
            for &g in &anticom_stabs {
                toggle(&mut self.signs_minus, g);
            }
        }
        if self.signs_i.remove(&removed) {
            for &g in &anticom_stabs {
                if self.signs_i.remove(&g) {
                    toggle(&mut self.signs_minus, g);
                } else {
                    self.signs_i.insert(g);
                }
            }
        }
        for &g in &anticom_stabs {
            let gi = g as usize;
            let mut minuses = 0usize;
            for &q in &removed_row_z {
                minuses += self.stabs.row_x[gi].contains(&q) as usize;
                if self.stabs.row_z[gi].remove(&q) {
                    self.stabs.col_z[q as usize].remove(&g);
                } else {
                    self.stabs.row_z[gi].insert(q);
                    self.stabs.col_z[q as usize].insert(g);
                }
            }
            if minuses % 2 == 1 {
                toggle(&mut self.signs_minus, g);
            }
            for &q in &removed_row_x {
                if self.stabs.row_x[gi].remove(&q) {
                    self.stabs.col_x[q as usize].remove(&g);
                } else {
                    self.stabs.row_x[gi].insert(q);
                    self.stabs.col_x[q as usize].insert(g);
                }
            }
        }

        // The destabilizers.
        let r = removed as usize;
        for &q in &self.destabs.row_x[r] {
            self.destabs.col_x[q as usize].remove(&removed);
        }
        for &q in &self.destabs.row_z[r] {
            self.destabs.col_z[q as usize].remove(&removed);
        }
        for &q in &removed_row_x {
            let qi = q as usize;
            self.stabs.col_x[qi].remove(&removed);
            self.destabs.col_x[qi].insert(removed);
            for &row in &anticom_destabs {
                if self.destabs.col_x[qi].remove(&row) {
                    self.destabs.row_x[row as usize].remove(&q);
                } else {
                    self.destabs.col_x[qi].insert(row);
                    self.destabs.row_x[row as usize].insert(q);
                }
            }
        }
        for &q in &removed_row_z {
            let qi = q as usize;
            self.stabs.col_z[qi].remove(&removed);
            self.destabs.col_z[qi].insert(removed);
            for &row in &anticom_destabs {
                if self.destabs.col_z[qi].remove(&row) {
                    self.destabs.row_z[row as usize].remove(&q);
                } else {
                    self.destabs.col_z[qi].insert(row);
                    self.destabs.row_z[row as usize].insert(q);
                }
            }
        }

        // The measured stabilizer replaces the removed one.
        self.stabs.col_z[qubit].insert(removed);
        self.stabs.row_x[r].clear();
        self.stabs.row_z[r].clear();
        self.stabs.row_z[r].insert(qubit as u32);
        self.destabs.row_x[r] = removed_row_x;
        self.destabs.row_z[r] = removed_row_z;
        if outcome == 1 {
            self.signs_minus.insert(removed);
        } else {
            self.signs_minus.remove(&removed);
        }
    }

    /// Replaces the generators (rows; the columns follow), as `set_stabs_destabs`.
    pub fn set_rows(&mut self, stabs_x: Vec<Set>, stabs_z: Vec<Set>, destabs_x: Vec<Set>, destabs_z: Vec<Set>) {
        let n = self.num_qubits;
        let cols = |rows: &[Set]| {
            let mut c = Generators::empty(n);
            for (g, row) in rows.iter().enumerate() {
                for &q in row {
                    c[q as usize].insert(g as u32);
                }
            }
            c
        };
        self.stabs = Generators { col_x: cols(&stabs_x), col_z: cols(&stabs_z), row_x: stabs_x, row_z: stabs_z };
        self.destabs = Generators { col_x: cols(&destabs_x), col_z: cols(&destabs_z), row_x: destabs_x, row_z: destabs_z };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn im(phase: u8, x: u8, z: u8) -> Image {
        Image { phase, x, z }
    }
    // H: X -> Z, Z -> X. S: X -> Y = iXZ, Z -> Z. CNOT (control 0, target 1).
    const H: [Image; 2] = [im(0, 0, 1), im(0, 1, 0)];
    const S: [Image; 2] = [im(1, 1, 1), im(0, 0, 1)];
    const X: [Image; 2] = [im(0, 1, 0), im(2, 0, 1)];
    const CNOT: [Image; 4] = [im(0, 0b11, 0), im(0, 0, 0b01), im(0, 0b10, 0), im(0, 0, 0b11)];

    fn rows(s: &SparseSim) -> Vec<String> {
        (0..s.num_qubits as u32)
            .map(|g| {
                let sign = ["+", "i", "-", "-i"][s.phase(g) as usize];
                let p: String = (0..s.num_qubits as u32).map(|q| match (s.stabs.row_x[g as usize].contains(&q), s.stabs.row_z[g as usize].contains(&q)) {
                    (false, false) => 'I',
                    (true, false) => 'X',
                    (false, true) => 'Z',
                    (true, true) => 'W',
                }).collect();
                format!("{sign}{p}")
            })
            .collect()
    }

    fn consistent(s: &SparseSim) {
        for gens in [&s.stabs, &s.destabs] {
            for g in 0..s.num_qubits {
                for q in 0..s.num_qubits {
                    assert_eq!(gens.row_x[g].contains(&(q as u32)), gens.col_x[q].contains(&(g as u32)));
                    assert_eq!(gens.row_z[g].contains(&(q as u32)), gens.col_z[q].contains(&(g as u32)));
                }
            }
        }
    }

    #[test]
    fn bell_state_and_its_measurements() {
        let mut s = SparseSim::new(2);
        s.gate(&[0], &H);
        s.gate(&[0, 1], &CNOT);
        assert_eq!(rows(&s), ["+XX", "+ZZ"]);
        assert!(!s.is_deterministic(0));
        assert_eq!(s.measure(0, 1, true), 1);
        consistent(&s);
        assert!(s.is_deterministic(1));
        assert_eq!(s.measure(1, 0, true), 1);
        // X flips it back; S takes X to Y = iW.
        s.gate(&[1], &X);
        assert_eq!(s.measure(1, 0, true), 0);
        let mut t = SparseSim::new(1);
        t.gate(&[0], &H);
        t.gate(&[0], &S);
        assert_eq!(rows(&t), ["iW"]);
        t.gate(&[0], &S);
        assert_eq!(rows(&t), ["-X"]);
    }

    #[test]
    fn ghz_parities_are_determined() {
        let n = 5;
        let mut s = SparseSim::new(n);
        s.gate(&[0], &H);
        for q in 1..n {
            s.gate(&[0, q], &CNOT);
        }
        let first = s.measure(2, 1, true);
        for q in 0..n {
            assert!(s.is_deterministic(q));
            assert_eq!(s.measure(q, 0, true), first);
        }
        consistent(&s);
    }
}
