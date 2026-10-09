//! Stim's `FlipSimulator`: an interactive Pauli-frame simulator over a batch of instances.
//! Each qubit's X and Z flips, each measurement's, detector's and observable's flips are rows
//! of bits, one bit per instance, packed 64 to a word. Every instruction of Stim's language
//! acts on the rows: Cliffords through their tableaus (signs don't matter to a frame), noise
//! with each channel's disjoint semantics, measurements reading the anticommuting part of the
//! frame, and (unless disabled) stabilizer randomization after resets and measurements.

use super::ir::{self, GateTarget, Instruction, Item};
use super::tableau_sim::{decompose_mpp, decompose_spp, disjoint_pair_segments};
use crate::gate_data::{FLAG_IS_SINGLE_QUBIT_GATE, FLAG_IS_UNITARY, FLAG_TARGETS_PAIRS};

/// xoshiro256**, seeded through splitmix64.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

type Row = Vec<u64>;

#[derive(Clone, Debug)]
pub struct FlipSim {
    pub batch: usize,
    words: usize,
    pub xs: Vec<Row>,
    pub zs: Vec<Row>,
    pub meas: Vec<Row>,
    pub dets: Vec<Row>,
    pub obs: Vec<Row>,
    pub randomize: bool,
    pub rng: Rng,
    /// Which instances' correlated-error chain (`E` / `ELSE_CORRELATED_ERROR`) has fired.
    chain: Row,
}

fn xor_into(a: &mut Row, b: &Row) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= *y;
    }
}

fn and(a: &Row, b: &Row) -> Row {
    a.iter().zip(b).map(|(x, y)| x & y).collect()
}

/// A Pauli's (x, z) bits from its letter.
fn pauli_bits(c: u8) -> (bool, bool) {
    match c {
        b'X' => (true, false),
        b'Y' => (true, true),
        b'Z' => (false, true),
        _ => (false, false),
    }
}

impl FlipSim {
    pub fn new(batch: usize, num_qubits: usize, randomize: bool, seed: u64) -> FlipSim {
        let words = batch.div_ceil(64);
        let mut s = FlipSim { batch, words, xs: Vec::new(), zs: Vec::new(), meas: Vec::new(), dets: Vec::new(), obs: Vec::new(), randomize, rng: Rng::new(seed), chain: vec![0; words] };
        s.ensure_qubits(num_qubits);
        s
    }

    fn zero(&self) -> Row {
        vec![0; self.words]
    }

    /// Each instance's bit set with probability `p`.
    pub fn bernoulli(&mut self, p: f64) -> Row {
        self.bernoulli_bits(p, self.batch)
    }

    /// `n` bits, each set with probability `p` (bits past `n` in the last word are zero).
    pub fn bernoulli_bits(&mut self, p: f64, n: usize) -> Row {
        let mut r: Row = vec![0; n.div_ceil(64)];
        if p <= 0.0 || n == 0 {
            return r;
        }
        if p >= 1.0 {
            r.iter_mut().for_each(|w| *w = u64::MAX);
        } else if p == 0.5 {
            for w in r.iter_mut() {
                *w = self.rng.next_u64();
            }
        } else if p < 0.1 {
            // Skip ahead geometrically between hits.
            let ln = (1.0 - p).ln();
            let mut k: f64 = -1.0;
            loop {
                let u = self.rng.uniform();
                k += 1.0 + ((1.0 - u).ln() / ln).floor();
                if k >= n as f64 {
                    break;
                }
                let i = k as usize;
                r[i / 64] |= 1 << (i % 64);
            }
        } else {
            for i in 0..n {
                if self.rng.uniform() < p {
                    r[i / 64] |= 1 << (i % 64);
                }
            }
        }
        let extra = r.len() * 64 - n;
        if extra > 0 {
            if let Some(w) = r.last_mut() {
                *w &= u64::MAX >> extra;
            }
        }
        r
    }

    fn coin(&mut self) -> Row {
        self.bernoulli(0.5)
    }

    pub fn ensure_qubits(&mut self, n: usize) {
        while self.xs.len() < n {
            let z = if self.randomize { self.coin() } else { self.zero() };
            self.xs.push(self.zero());
            self.zs.push(z);
        }
    }

    pub fn ensure_observables(&mut self, n: usize) {
        while self.obs.len() < n {
            self.obs.push(self.zero());
        }
    }

    /// Back to |0> with empty records, keeping the size.
    pub fn clear(&mut self) {
        self.meas.clear();
        self.dets.clear();
        for o in self.obs.iter_mut() {
            o.iter_mut().for_each(|w| *w = 0);
        }
        self.chain = self.zero();
        for q in 0..self.xs.len() {
            self.xs[q] = self.zero();
            self.zs[q] = if self.randomize { self.coin() } else { self.zero() };
        }
    }

    pub fn get_bit(row: &Row, k: usize) -> bool {
        row[k / 64] >> (k % 64) & 1 != 0
    }

    pub fn set_bit(row: &mut Row, k: usize, v: bool) {
        if v {
            row[k / 64] |= 1 << (k % 64);
        } else {
            row[k / 64] &= !(1 << (k % 64));
        }
    }

    fn rec(&self, t: GateTarget) -> Result<Row, String> {
        let k = t.value() as usize;
        if k == 0 || k > self.meas.len() {
            return Err("Referred to a measurement record before the beginning of time.".into());
        }
        Ok(self.meas[self.meas.len() - k].clone())
    }

    fn apply_pauli(&mut self, q: usize, x: bool, z: bool, mask: &Row) {
        if x {
            xor_into(&mut self.xs[q], mask);
        }
        if z {
            xor_into(&mut self.zs[q], mask);
        }
    }

    fn push_measurement(&mut self, mut flip: Row, p: f64) {
        if p > 0.0 {
            let noise = self.bernoulli(p);
            xor_into(&mut flip, &noise);
        }
        self.meas.push(flip);
    }

    fn max_qubit(inst: &Instruction) -> usize {
        inst.targets.iter().filter(|t| t.has_qubit_value()).map(|t| t.value() as usize + 1).max().unwrap_or(0)
    }

    pub fn do_circuit(&mut self, c: &ir::Circuit) -> Result<(), String> {
        for it in &c.items {
            match it {
                Item::Op(op) => self.do_instruction(op)?,
                Item::Repeat { count, body, .. } => {
                    for _ in 0..*count {
                        self.do_circuit(body)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn do_instruction(&mut self, inst: &Instruction) -> Result<(), String> {
        self.ensure_qubits(Self::max_qubit(inst));
        let name = inst.gate.name;
        let p0 = inst.args.first().copied().unwrap_or(0.0);
        match name {
            "TICK" | "QUBIT_COORDS" | "SHIFT_COORDS" | "I" | "II" | "I_ERROR" | "II_ERROR" => {}
            "DETECTOR" => {
                let mut r = self.zero();
                for t in &inst.targets {
                    if t.is_record() {
                        xor_into(&mut r, &self.rec(*t)?);
                    }
                }
                self.dets.push(r);
            }
            "OBSERVABLE_INCLUDE" => {
                let k = p0 as usize;
                self.ensure_observables(k + 1);
                let mut r = self.zero();
                for t in &inst.targets {
                    if t.is_record() {
                        xor_into(&mut r, &self.rec(*t)?);
                    } else if t.is_pauli() {
                        let q = t.value() as usize;
                        if t.0 & ir::TARGET_PAULI_X_BIT != 0 {
                            xor_into(&mut r, &self.zs[q]);
                        }
                        if t.0 & ir::TARGET_PAULI_Z_BIT != 0 {
                            xor_into(&mut r, &self.xs[q]);
                        }
                    }
                }
                xor_into(&mut self.obs[k], &r);
            }
            "MPAD" => {
                for _ in &inst.targets {
                    let z = self.zero();
                    self.push_measurement(z, p0);
                }
            }
            "M" | "MX" | "MY" | "MR" | "MRX" | "MRY" => {
                let basis = match name.as_bytes()[name.len() - 1] {
                    b'X' => b'X',
                    b'Y' => b'Y',
                    _ => b'Z',
                };
                let reset = name.starts_with("MR");
                for t in &inst.targets {
                    self.measure(t.value() as usize, basis, p0, reset);
                }
            }
            "R" | "RX" | "RY" => {
                let basis = if name == "R" { b'Z' } else { name.as_bytes()[1] };
                for t in &inst.targets {
                    self.reset(t.value() as usize, basis);
                }
            }
            "MXX" | "MYY" | "MZZ" => {
                // As Stim: per segment of disjoint pairs, a CX / CY / XCZ onto the second, a
                // one-qubit measurement of the first, and the gate again.
                let (gate, basis) = match name {
                    "MXX" => ("CX", b'X'),
                    "MYY" => ("CY", b'Y'),
                    _ => ("XCZ", b'Z'),
                };
                for seg in disjoint_pair_segments(&inst.targets) {
                    let g = Instruction::new(gate, vec![], seg.iter().map(|t| GateTarget::qubit(t.value(), false)).collect(), "")?;
                    self.do_instruction(&g)?;
                    for pair in seg.chunks(2) {
                        self.measure(pair[0].value() as usize, basis, p0, false);
                    }
                    self.do_instruction(&g)?;
                }
            }
            "MPP" => {
                for op in decompose_mpp(inst, self.xs.len())? {
                    self.do_instruction(&op)?;
                }
            }
            "SPP" | "SPP_DAG" => {
                for op in decompose_spp(inst)? {
                    self.do_instruction(&op)?;
                }
            }
            "X_ERROR" | "Y_ERROR" | "Z_ERROR" => {
                let (x, z) = pauli_bits(name.as_bytes()[0]);
                for t in &inst.targets {
                    let r = self.bernoulli(p0);
                    self.apply_pauli(t.value() as usize, x, z, &r);
                }
            }
            "DEPOLARIZE1" | "PAULI_CHANNEL_1" => {
                let probs: Vec<f64> = if name == "DEPOLARIZE1" { vec![p0 / 3.0; 3] } else { inst.args.clone() };
                for t in &inst.targets {
                    let rows = self.disjoint(&probs);
                    let q = t.value() as usize;
                    for (k, r) in rows.iter().enumerate() {
                        let (x, z) = pauli_bits(b"XYZ"[k]);
                        self.apply_pauli(q, x, z, r);
                    }
                }
            }
            "DEPOLARIZE2" | "PAULI_CHANNEL_2" => {
                let probs: Vec<f64> = if name == "DEPOLARIZE2" { vec![p0 / 15.0; 15] } else { inst.args.clone() };
                for pair in inst.targets.chunks(2) {
                    let rows = self.disjoint(&probs);
                    let (a, b) = (pair[0].value() as usize, pair[1].value() as usize);
                    for (k, r) in rows.iter().enumerate() {
                        let (p1, p2) = ((k + 1) >> 2, (k + 1) & 3);
                        let (x1, z1) = pauli_bits(b"IXYZ"[p1]);
                        let (x2, z2) = pauli_bits(b"IXYZ"[p2]);
                        self.apply_pauli(a, x1, z1, r);
                        self.apply_pauli(b, x2, z2, r);
                    }
                }
            }
            "E" | "ELSE_CORRELATED_ERROR" => {
                let hit = self.bernoulli(p0);
                let fire = if name == "E" {
                    self.chain = hit.clone();
                    hit
                } else {
                    let f: Row = hit.iter().zip(&self.chain).map(|(h, c)| h & !c).collect();
                    for (c, x) in self.chain.iter_mut().zip(&f) {
                        *c |= x;
                    }
                    f
                };
                for t in &inst.targets {
                    let q = t.value() as usize;
                    self.apply_pauli(q, t.0 & ir::TARGET_PAULI_X_BIT != 0, t.0 & ir::TARGET_PAULI_Z_BIT != 0, &fire);
                }
            }
            "HERALDED_ERASE" | "HERALDED_PAULI_CHANNEL_1" => {
                let probs: Vec<f64> = if name == "HERALDED_ERASE" { vec![p0 / 4.0; 4] } else { inst.args.clone() };
                for t in &inst.targets {
                    let rows = self.disjoint(&probs);
                    let q = t.value() as usize;
                    let mut herald = self.zero();
                    for (k, r) in rows.iter().enumerate() {
                        xor_into(&mut herald, r);
                        let (x, z) = pauli_bits(b"IXYZ"[k]);
                        self.apply_pauli(q, x, z, r);
                    }
                    self.meas.push(herald);
                }
            }
            _ if inst.gate.flags & FLAG_IS_UNITARY != 0 => {
                if inst.gate.flags & FLAG_IS_SINGLE_QUBIT_GATE != 0 {
                    let (xi, zi) = (inst.gate.tableau[0].as_bytes(), inst.gate.tableau[1].as_bytes());
                    let (xx, xz) = pauli_bits(xi[1]);
                    let (zx, zz) = pauli_bits(zi[1]);
                    for t in &inst.targets {
                        let q = t.value() as usize;
                        let (x, z) = (self.xs[q].clone(), self.zs[q].clone());
                        self.xs[q] = x.iter().zip(&z).map(|(a, b)| (if xx { *a } else { 0 }) ^ (if zx { *b } else { 0 })).collect();
                        self.zs[q] = x.iter().zip(&z).map(|(a, b)| (if xz { *a } else { 0 }) ^ (if zz { *b } else { 0 })).collect();
                    }
                } else if inst.gate.flags & FLAG_TARGETS_PAIRS != 0 {
                    for pair in inst.targets.chunks(2) {
                        self.two_qubit(inst, pair[0], pair[1])?;
                    }
                } else {
                    return Err(format!("Not supported by the flip simulator: {inst}"));
                }
            }
            _ => return Err(format!("Not supported by the flip simulator: {inst}")),
        }
        Ok(())
    }

    fn random_row(&mut self) -> Row {
        self.coin()
    }

    /// A one-qubit measurement (and reset), updating the rows exactly as Stim's frame
    /// simulator does: the record reads the anticommuting part; the stabilizer part is
    /// re-randomized (overwritten) when randomization is on.
    fn measure(&mut self, q: usize, basis: u8, noise: f64, reset: bool) {
        match basis {
            b'X' => {
                let flip = self.zs[q].clone();
                self.push_measurement(flip, noise);
                if reset {
                    self.zs[q] = self.zero();
                }
                if self.randomize {
                    self.xs[q] = self.random_row();
                }
            }
            b'Y' => {
                let z = self.zs[q].clone();
                xor_into(&mut self.xs[q], &z);
                let flip = self.xs[q].clone();
                self.push_measurement(flip, noise);
                if self.randomize {
                    self.zs[q] = self.random_row();
                }
                if reset {
                    self.xs[q] = self.zs[q].clone();
                } else {
                    let z = self.zs[q].clone();
                    xor_into(&mut self.xs[q], &z);
                }
            }
            _ => {
                let flip = self.xs[q].clone();
                self.push_measurement(flip, noise);
                if reset {
                    self.xs[q] = self.zero();
                }
                if self.randomize {
                    self.zs[q] = self.random_row();
                }
            }
        }
    }

    /// A reset, as Stim's frame simulator does it.
    fn reset(&mut self, q: usize, basis: u8) {
        match basis {
            b'X' => {
                if self.randomize {
                    self.xs[q] = self.random_row();
                }
                self.zs[q] = self.zero();
            }
            b'Y' => {
                if self.randomize {
                    self.zs[q] = self.random_row();
                }
                self.xs[q] = self.zs[q].clone();
            }
            _ => {
                self.xs[q] = self.zero();
                if self.randomize {
                    self.zs[q] = self.random_row();
                }
            }
        }
    }

    /// A disjoint channel: one row per case, each instance in at most one.
    fn disjoint(&mut self, probs: &[f64]) -> Vec<Row> {
        let mut rows = vec![self.zero(); probs.len()];
        let total: f64 = probs.iter().sum();
        if total <= 0.0 {
            return rows;
        }
        let hit = self.bernoulli(total.min(1.0));
        for (w, &word) in hit.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let mut u = self.rng.uniform() * total;
                let mut case = probs.len() - 1;
                for (k, &p) in probs.iter().enumerate() {
                    if u < p {
                        case = k;
                        break;
                    }
                    u -= p;
                }
                rows[case][w] |= 1 << b;
            }
        }
        rows
    }

    fn two_qubit(&mut self, inst: &Instruction, a: GateTarget, b: GateTarget) -> Result<(), String> {
        let name = inst.gate.name;
        if !a.has_qubit_value() || !b.has_qubit_value() {
            // Classically controlled: a Pauli on the qubit where the bit (a measurement) flipped.
            let (bit, q, first_is_bit) = if a.has_qubit_value() { (b, a, false) } else { (a, b, true) };
            if !q.has_qubit_value() || bit.is_sweep() {
                return Ok(());
            }
            let pauli = match (name, first_is_bit) {
                ("CX", true) => b'X',
                ("CY", true) => b'Y',
                ("CZ", _) => b'Z',
                ("XCZ", false) => b'X',
                ("YCZ", false) => b'Y',
                _ => return Err(format!("Unsupported classical control in the flip simulator: {inst}")),
            };
            let r = self.rec(bit)?;
            let (x, z) = pauli_bits(pauli);
            self.apply_pauli(q.value() as usize, x, z, &r);
            return Ok(());
        }
        let (qa, qb) = (a.value() as usize, b.value() as usize);
        let images: Vec<(bool, bool, bool, bool)> = inst
            .gate
            .tableau
            .iter()
            .map(|s| {
                let s = s.as_bytes();
                let (x0, z0) = pauli_bits(s[1]);
                let (x1, z1) = pauli_bits(s[2]);
                (x0, z0, x1, z1)
            })
            .collect();
        let inputs = [self.xs[qa].clone(), self.zs[qa].clone(), self.xs[qb].clone(), self.zs[qb].clone()];
        let mut out = [self.zero(), self.zero(), self.zero(), self.zero()];
        for (k, input) in inputs.iter().enumerate() {
            let (x0, z0, x1, z1) = images[k];
            for (o, on) in out.iter_mut().zip([x0, z0, x1, z1]) {
                if on {
                    xor_into(o, input);
                }
            }
        }
        let [ox0, oz0, ox1, oz1] = out;
        self.xs[qa] = ox0;
        self.zs[qa] = oz0;
        self.xs[qb] = ox1;
        self.zs[qb] = oz1;
        Ok(())
    }

    /// Multiplies the frame by `pauli` (0=I, 1=X, 2=Y, 3=Z) on each qubit where `mask` says,
    /// each such place with probability `p`.
    pub fn broadcast(&mut self, pauli: u8, mask: &[Row], p: f64) {
        self.ensure_qubits(mask.len());
        let (x, z) = [(false, false), (true, false), (true, true), (false, true)][pauli as usize & 3];
        for (q, m) in mask.iter().enumerate() {
            let hit = if p >= 1.0 { m.clone() } else { and(m, &self.bernoulli(p)) };
            self.apply_pauli(q, x, z, &hit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(text: &str, batch: usize) -> FlipSim {
        let mut s = FlipSim::new(batch, 0, false, 5);
        s.do_circuit(&ir::Circuit::parse(text).unwrap()).unwrap();
        s
    }

    #[test]
    fn errors_propagate_through_cliffords() {
        // Stim's example: X errors on 0, 1, 3, then 5 rounds of H 0, C_XYZ 1 → Z, Z, _, X.
        let s = run("X_ERROR(1) 0 1 3\nREPEAT 5 {\nH 0\nC_XYZ 1\n}", 3);
        assert_eq!((s.xs[0][0], s.zs[0][0]), (0, 0b111));
        assert_eq!((s.xs[1][0], s.zs[1][0]), (0, 0b111));
        assert_eq!((s.xs[3][0], s.zs[3][0]), (0b111, 0));
    }

    #[test]
    fn measurements_detectors_and_observables_read_flips() {
        let s = run("X_ERROR(1) 0\nCX 0 1\nM 0 1 2\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(2) rec[-3] Z2", 70);
        assert_eq!(s.meas.len(), 3);
        assert_eq!(s.meas[0][1], u64::MAX >> (128 - 70));
        assert_eq!(s.dets[0], vec![u64::MAX, u64::MAX >> (128 - 70)]);
        assert_eq!(s.obs.len(), 3);
        assert_eq!(s.obs[2][0], u64::MAX);
    }

    #[test]
    fn feedback_and_pair_measurements() {
        let s = run("X_ERROR(1) 0\nM 0\nCX rec[-1] 1\nMZZ 1 2\nMPP X3*Z1", 4);
        assert_eq!(s.xs[1][0], 0b1111);
        assert_eq!(s.meas[1][0], 0b1111);
        assert_eq!(s.meas[2][0], 0b1111);
    }

    #[test]
    fn bernoulli_rates_are_right() {
        let mut s = FlipSim::new(100_000, 0, false, 9);
        for p in [0.001, 0.05, 0.3, 0.5, 0.9] {
            let r = s.bernoulli(p);
            let k: u32 = r.iter().map(|w| w.count_ones()).sum();
            let rate = k as f64 / 100_000.0;
            assert!((rate - p).abs() < 5.0 * (p * (1.0 - p) / 100_000.0).sqrt() + 1e-4, "{p} {rate}");
        }
    }
}
