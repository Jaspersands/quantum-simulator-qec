//! Stabilizer tableaus, as `stim.Tableau` holds them: for a Clifford operation C on n qubits,
//! the images C X_k C† and C Z_k C† of each qubit's X and Z (signed Pauli strings).

use std::fmt;

use super::pauli_string::{pauli_from_xz, PauliString};
use crate::gate_data;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Tableau {
    n: usize,
    /// `xs[k]` is the image of X_k, `zs[k]` of Z_k. Their phases are 0 or 2.
    pub xs: Vec<PauliString>,
    pub zs: Vec<PauliString>,
}

impl Tableau {
    /// The identity on `n` qubits.
    pub fn identity(n: usize) -> Tableau {
        let mut xs = Vec::with_capacity(n);
        let mut zs = Vec::with_capacity(n);
        for k in 0..n {
            let mut x = PauliString::new(n);
            x.set_x(k, true);
            let mut z = PauliString::new(n);
            z.set_z(k, true);
            xs.push(x);
            zs.push(z);
        }
        Tableau { n, xs, zs }
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// A unitary gate's tableau (one or two qubits), by name or alias.
    pub fn from_named_gate(name: &str) -> Result<Tableau, String> {
        let g = gate_data::info(name).ok_or_else(|| format!("Gate not found: '{name}'"))?;
        if g.tableau.is_empty() {
            return Err(format!("{} doesn't have 1q or 2q tableau data.", g.name));
        }
        let n = g.tableau.len() / 2;
        let mut t = Tableau::identity(n);
        for q in 0..n {
            t.xs[q] = PauliString::from_text(g.tableau[2 * q]).expect("gate data");
            t.zs[q] = PauliString::from_text(g.tableau[2 * q + 1]).expect("gate data");
        }
        Ok(t)
    }

    /// The tableau of a Pauli product: X_k picks up the sign of Z on k, and so on.
    pub fn from_pauli_string(p: &PauliString) -> Tableau {
        let mut t = Tableau::identity(p.num_qubits());
        for k in 0..p.num_qubits() {
            if p.z(k) {
                t.xs[k].phase = 2;
            }
            if p.x(k) {
                t.zs[k].phase = 2;
            }
        }
        t
    }

    /// From the images of each X and Z, checking that they form a Clifford tableau.
    pub fn from_conjugated_generators(xs: Vec<PauliString>, zs: Vec<PauliString>) -> Result<Tableau, String> {
        let n = xs.len();
        if zs.len() != n {
            return Err("len(xs) != len(zs)".into());
        }
        for p in xs.iter().chain(zs.iter()) {
            if p.num_qubits() != n {
                return Err("not all(len(p) == len(xs) for p in xs + zs)".into());
            }
            if p.is_imaginary() {
                return Err("Conjugated generator can't have imaginary sign.".into());
            }
        }
        let t = Tableau { n, xs, zs };
        if !t.satisfies_invariants() {
            return Err("The given generator outputs don't describe a valid Clifford operation.\nThey don't preserve commutativity.\nEverything must commute, except for X_k anticommuting with Z_k for each k.".into());
        }
        Ok(t)
    }

    /// Every pair commutes except each X_k with its Z_k.
    pub fn satisfies_invariants(&self) -> bool {
        for q1 in 0..self.n {
            if self.xs[q1].commutes(&self.zs[q1]) {
                return false;
            }
            for q2 in q1 + 1..self.n {
                let (x1, z1, x2, z2) = (&self.xs[q1], &self.zs[q1], &self.xs[q2], &self.zs[q2]);
                if !x1.commutes(x2) || !x1.commutes(z2) || !z1.commutes(x2) || !z1.commutes(z2) {
                    return false;
                }
            }
        }
        true
    }

    /// The image of the given Pauli string on the qubits `indices` of this tableau, as a
    /// string on all qubits (Stim's `scatter_eval`).
    pub fn scatter_eval(&self, gathered: &PauliString, indices: &[usize]) -> PauliString {
        let mut result = PauliString::new(self.n);
        result.phase = gathered.phase & 2;
        let mut extra = gathered.phase & 1;
        for (kg, &ks) in indices.iter().enumerate() {
            let (x, z) = (gathered.x(kg), gathered.z(kg));
            if x && z {
                // Y = i X Z.
                result.mul_assign(&self.xs[ks]);
                result.mul_assign(&self.zs[ks]);
                extra += 1;
            } else if x {
                result.mul_assign(&self.xs[ks]);
            } else if z {
                result.mul_assign(&self.zs[ks]);
            }
        }
        result.phase = (result.phase + extra) & 3;
        result
    }

    /// The image C P C† of a Pauli string on all qubits.
    pub fn apply(&self, p: &PauliString) -> Result<PauliString, String> {
        if p.num_qubits() != self.n {
            return Err("pauli_string.num_qubits != tableau.num_qubits".into());
        }
        let idx: Vec<usize> = (0..self.n).collect();
        Ok(self.scatter_eval(p, &idx))
    }

    /// Conjugate the Paulis of `target` on `qubits` by this tableau, in place.
    pub fn apply_within(&self, target: &mut PauliString, qubits: &[usize]) {
        let gathered = PauliString::from_fn(qubits.len(), 0, |k| target.get(qubits[k]));
        let out = self.apply(&gathered).expect("sizes match");
        for (k, &q) in qubits.iter().enumerate() {
            target.set(q, out.get(k));
        }
        target.phase = (target.phase + out.phase) & 3;
    }

    /// Apply `op` after this one, on `targets`: self ← op ∘ self.
    pub fn append(&mut self, op: &Tableau, targets: &[usize]) {
        for q in 0..self.n {
            op.apply_within(&mut self.xs[q], targets);
            op.apply_within(&mut self.zs[q], targets);
        }
    }

    /// Apply `op` before this one, on `targets`: self ← self ∘ op.
    pub fn prepend(&mut self, op: &Tableau, targets: &[usize]) {
        let new_x: Vec<PauliString> = (0..op.n).map(|q| self.scatter_eval(&op.xs[q], targets)).collect();
        let new_z: Vec<PauliString> = (0..op.n).map(|q| self.scatter_eval(&op.zs[q], targets)).collect();
        for (q, (x, z)) in new_x.into_iter().zip(new_z).enumerate() {
            self.xs[targets[q]] = x;
            self.zs[targets[q]] = z;
        }
    }

    /// This operation, then `second`: their composition second ∘ self.
    pub fn then(&self, second: &Tableau) -> Tableau {
        let xs = self.xs.iter().map(|p| second.apply(p).expect("sizes match")).collect();
        let zs = self.zs.iter().map(|p| second.apply(p).expect("sizes match")).collect();
        Tableau { n: self.n, xs, zs }
    }

    /// The inverse operation. With `unsigned`, the signs are left at +.
    pub fn inverse(&self, unsigned: bool) -> Tableau {
        let n = self.n;
        let mut r = Tableau::identity(n);
        // Transpose with Stim's xx/zz swap: inverse X_k image has x[q] = zs[q].z[k], z[q] = xs[q].z[k].
        for k in 0..n {
            let mut x = PauliString::new(n);
            let mut z = PauliString::new(n);
            for q in 0..n {
                x.set_x(q, self.zs[q].z(k));
                x.set_z(q, self.xs[q].z(k));
                z.set_x(q, self.zs[q].x(k));
                z.set_z(q, self.xs[q].x(k));
            }
            r.xs[k] = x;
            r.zs[k] = z;
        }
        if !unsigned {
            for k in 0..n {
                let mut single = PauliString::new(n);
                single.set_x(k, true);
                let xs_sign = self.apply(&r.apply(&single).unwrap()).unwrap().sign();
                single.set_x(k, false);
                single.set_z(k, true);
                let zs_sign = self.apply(&r.apply(&single).unwrap()).unwrap().sign();
                if xs_sign {
                    r.xs[k].phase ^= 2;
                }
                if zs_sign {
                    r.zs[k].phase ^= 2;
                }
            }
        }
        r
    }

    /// This operation applied `exponent` times (negative: the inverse's).
    pub fn raised_to(&self, exponent: i64) -> Tableau {
        let mut result = Tableau::identity(self.n);
        if exponent == 0 {
            return result;
        }
        let mut square = if exponent < 0 { self.inverse(false) } else { self.clone() };
        let mut e = exponent.unsigned_abs();
        loop {
            if e & 1 == 1 {
                result = result.then(&square);
            }
            e >>= 1;
            if e == 0 {
                break;
            }
            square = square.then(&square);
        }
        result
    }

    /// The direct sum: this one on the first qubits, `second` on the next.
    pub fn tensor(&self, second: &Tableau) -> Tableau {
        let n = self.n + second.n;
        let pad = |p: &PauliString, offset: usize| PauliString::from_fn(n, p.phase, |q| if q >= offset && q < offset + p.num_qubits() { p.get(q - offset) } else { 0 });
        let mut xs: Vec<PauliString> = self.xs.iter().map(|p| pad(p, 0)).collect();
        let mut zs: Vec<PauliString> = self.zs.iter().map(|p| pad(p, 0)).collect();
        xs.extend(second.xs.iter().map(|p| pad(p, self.n)));
        zs.extend(second.zs.iter().map(|p| pad(p, self.n)));
        Tableau { n, xs, zs }
    }

    pub fn x_output(&self, k: usize) -> &PauliString {
        &self.xs[k]
    }

    pub fn z_output(&self, k: usize) -> &PauliString {
        &self.zs[k]
    }

    /// The image of Y_k: i · X_k-image · Z_k-image.
    pub fn y_output(&self, k: usize) -> PauliString {
        let mut r = self.xs[k].mul(&self.zs[k]);
        r.phase = (r.phase + 1) & 3;
        r
    }

    /// Pauli (0..4) of the image of X/Y/Z of `input` on `output`; `which` is 1, 2 or 3.
    pub fn output_pauli(&self, which: u8, input: usize, output: usize) -> u8 {
        let (x, z) = (&self.xs[input], &self.zs[input]);
        match which {
            1 => pauli_from_xz(x.x(output), x.z(output)),
            2 => pauli_from_xz(x.x(output) ^ z.x(output), x.z(output) ^ z.z(output)),
            _ => pauli_from_xz(z.x(output), z.z(output)),
        }
    }

    /// As `output_pauli` for the inverse operation, without computing it.
    pub fn inverse_output_pauli(&self, which: u8, input: usize, output: usize) -> u8 {
        let (x, z) = (&self.xs[output], &self.zs[output]);
        match which {
            1 => pauli_from_xz(z.z(input), x.z(input)),
            2 => pauli_from_xz(z.z(input) ^ z.x(input), x.z(input) ^ x.x(input)),
            _ => pauli_from_xz(z.x(input), x.x(input)),
        }
    }

    /// The inverse operation's image of X/Y/Z (`which` 1, 2, 3) of qubit `input`.
    pub fn inverse_output(&self, which: u8, input: usize, unsigned: bool) -> PauliString {
        let n = self.n;
        let mut r = PauliString::new(n);
        for k in 0..n {
            let (x, z) = (&self.xs[k], &self.zs[k]);
            let (rx, rz) = match which {
                1 => (z.z(input), x.z(input)),
                2 => (z.z(input) ^ z.x(input), x.z(input) ^ x.x(input)),
                _ => (z.x(input), x.x(input)),
            };
            r.set_x(k, rx);
            r.set_z(k, rz);
        }
        if !unsigned {
            // The sign that makes C(r) the unsigned target Pauli.
            let image = self.apply(&r).unwrap();
            if image.sign() {
                r.phase = 2;
            }
        }
        r
    }

    /// Whether this is a Pauli product (every X_k ↦ ±X_k and Z_k ↦ ±Z_k).
    pub fn is_pauli_product(&self) -> bool {
        (0..self.n).all(|k| {
            let mut ex = PauliString::new(self.n);
            ex.set_x(k, true);
            let mut ez = PauliString::new(self.n);
            ez.set_z(k, true);
            ex.phase = self.xs[k].phase;
            ez.phase = self.zs[k].phase;
            self.xs[k] == ex && self.zs[k] == ez
        })
    }

    pub fn to_pauli_string(&self) -> Result<PauliString, String> {
        if !self.is_pauli_product() {
            return Err("The Tableau isn't equivalent to a Pauli product.".into());
        }
        let mut p = PauliString::new(self.n);
        for k in 0..self.n {
            p.set_x(k, self.zs[k].sign());
            p.set_z(k, self.xs[k].sign());
        }
        Ok(p)
    }

    /// The stabilizers of C|0…0⟩ (the Z images), reduced to Stim's canonical form if asked.
    pub fn stabilizers(&self, canonical: bool) -> Vec<PauliString> {
        let n = self.n;
        let mut s: Vec<PauliString> = self.zs.clone();
        if canonical {
            canonicalize(&mut s, n);
        }
        s
    }
}

/// Stim's canonical form of a list of commuting stabilizers on `n` qubits: Gaussian
/// elimination by qubit, X before Z, the pivot moved to the next free row.
pub fn canonicalize(s: &mut [PauliString], n: usize) {
    let mut min_pivot = 0;
    for q in 0..n {
        for b in 0..2 {
            let hit = |p: &PauliString| if b == 1 { p.z(q) } else { p.x(q) };
            let mut pivot = min_pivot;
            while pivot < s.len() && !hit(&s[pivot]) {
                pivot += 1;
            }
            if pivot == s.len() {
                continue;
            }
            let pv = s[pivot].clone();
            for k in 0..s.len() {
                if k != pivot && hit(&s[k]) {
                    s[k].mul_assign(&pv);
                }
            }
            if min_pivot != pivot {
                s.swap(min_pivot, pivot);
            }
            min_pivot += 1;
        }
    }
}

impl fmt::Display for Tableau {
    /// Stim's text: a header, the signs, then a row per output qubit.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.n;
        let mut s = String::from("+-");
        for _ in 0..n {
            s.push_str("xz-");
        }
        s.push_str("\n|");
        for k in 0..n {
            s.push(' ');
            s.push(if self.xs[k].sign() { '-' } else { '+' });
            s.push(if self.zs[k].sign() { '-' } else { '+' });
        }
        let ch = |p: &PauliString, q: usize| ['_', 'X', 'Z', 'Y'][p.x(q) as usize + 2 * p.z(q) as usize];
        for q in 0..n {
            s.push_str("\n|");
            for k in 0..n {
                s.push(' ');
                s.push(ch(&self.xs[k], q));
                s.push(ch(&self.zs[k], q));
            }
        }
        f.write_str(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ps(t: &str) -> PauliString {
        PauliString::from_text(t).unwrap()
    }

    #[test]
    fn gates_compose_and_invert() {
        let h = Tableau::from_named_gate("H").unwrap();
        let s = Tableau::from_named_gate("S").unwrap();
        let cx = Tableau::from_named_gate("CNOT").unwrap();
        assert_eq!(h.then(&h), Tableau::identity(1));
        assert_eq!(s.raised_to(4), Tableau::identity(1));
        assert_eq!(s.raised_to(-1), Tableau::from_named_gate("S_DAG").unwrap());
        assert_eq!(cx.apply(&ps("X_")).unwrap().to_string(), "+XX");
        assert_eq!(cx.apply(&ps("_Z")).unwrap().to_string(), "+ZZ");
        let sx = Tableau::from_named_gate("SQRT_X").unwrap();
        assert_eq!(sx.inverse(false), Tableau::from_named_gate("SQRT_X_DAG").unwrap());
        assert_eq!(s.y_output(0).to_string(), "-X");
        // A random-ish composite inverts.
        let mut t = Tableau::identity(3);
        t.append(&h, &[0]);
        t.append(&cx, &[0, 2]);
        t.append(&s, &[2]);
        t.append(&Tableau::from_named_gate("ISWAP").unwrap(), &[1, 2]);
        assert_eq!(t.then(&t.inverse(false)), Tableau::identity(3));
        let mut p = Tableau::identity(3);
        p.prepend(&Tableau::from_named_gate("ISWAP").unwrap(), &[1, 2]);
        p.prepend(&s, &[2]);
        p.prepend(&cx, &[0, 2]);
        p.prepend(&h, &[0]);
        assert_eq!(p, t);
        assert_eq!(t.to_string().lines().count(), 5);
    }
}
