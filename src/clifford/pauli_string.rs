//! Pauli strings with a phase in {+1, +i, −1, −i}, as `stim.PauliString` holds them: one bit
//! of X and one of Z per qubit (both set is Y), and the coefficient i^phase.

use std::fmt;

/// One qubit's Pauli in Stim's numbering: 0 = I, 1 = X, 2 = Y, 3 = Z.
pub fn pauli_from_xz(x: bool, z: bool) -> u8 {
    match (x, z) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

/// The (x, z) bits of a Pauli in Stim's numbering.
pub fn xz_from_pauli(p: u8) -> (bool, bool) {
    match p & 3 {
        0 => (false, false),
        1 => (true, false),
        2 => (true, true),
        _ => (false, true),
    }
}

#[inline]
pub(crate) fn words(n: usize) -> usize {
    n.div_ceil(64)
}

/// A Pauli string on `n` qubits, i^phase times a tensor product of I, X, Y and Z.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PauliString {
    n: usize,
    xs: Vec<u64>,
    zs: Vec<u64>,
    /// The coefficient is i^phase; 0 is +1, 1 is +i, 2 is −1, 3 is −i.
    pub phase: u8,
}

impl PauliString {
    /// The identity on `n` qubits, with sign +1.
    pub fn new(n: usize) -> Self {
        PauliString { n, xs: vec![0; words(n)], zs: vec![0; words(n)], phase: 0 }
    }

    /// From a function giving each qubit's Pauli (0..4, Stim's numbering).
    pub fn from_fn(n: usize, phase: u8, mut f: impl FnMut(usize) -> u8) -> Self {
        let mut p = PauliString::new(n);
        p.phase = phase & 3;
        for k in 0..n {
            p.set(k, f(k));
        }
        p
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    pub fn x(&self, k: usize) -> bool {
        (self.xs[k / 64] >> (k % 64)) & 1 == 1
    }

    pub fn z(&self, k: usize) -> bool {
        (self.zs[k / 64] >> (k % 64)) & 1 == 1
    }

    pub fn set_x(&mut self, k: usize, v: bool) {
        let m = 1u64 << (k % 64);
        if v {
            self.xs[k / 64] |= m
        } else {
            self.xs[k / 64] &= !m
        }
    }

    pub fn set_z(&mut self, k: usize, v: bool) {
        let m = 1u64 << (k % 64);
        if v {
            self.zs[k / 64] |= m
        } else {
            self.zs[k / 64] &= !m
        }
    }

    /// Qubit `k`'s Pauli: 0 = I, 1 = X, 2 = Y, 3 = Z.
    pub fn get(&self, k: usize) -> u8 {
        pauli_from_xz(self.x(k), self.z(k))
    }

    pub fn set(&mut self, k: usize, p: u8) {
        let (x, z) = xz_from_pauli(p);
        self.set_x(k, x);
        self.set_z(k, z);
    }

    pub fn xs_words(&self) -> &[u64] {
        &self.xs
    }

    pub fn zs_words(&self) -> &[u64] {
        &self.zs
    }

    /// Whether the sign is −1 or −i.
    pub fn sign(&self) -> bool {
        self.phase & 2 != 0
    }

    pub fn is_imaginary(&self) -> bool {
        self.phase & 1 != 0
    }

    /// Grow to `n` qubits (identities on the new ones); never shrinks.
    pub fn ensure_num_qubits(&mut self, n: usize) {
        if n > self.n {
            self.n = n;
            self.xs.resize(words(n), 0);
            self.zs.resize(words(n), 0);
        }
    }

    /// Parse Stim's text: an optional sign (`+`, `-`, then `i`), then a dense string of
    /// `_IXYZ` (lower case too) or a sparse product such as `X2*Y6`.
    pub fn from_text(text: &str) -> Result<PauliString, String> {
        let mut t = text;
        let mut phase = 0u8;
        if let Some(r) = t.strip_prefix('-') {
            phase = 2;
            t = r;
        } else if let Some(r) = t.strip_prefix('+') {
            t = r;
        }
        if let Some(r) = t.strip_prefix('i') {
            phase += 1;
            t = r;
        }
        let bad = || format!("Not a valid Pauli string shorthand: '{t}'");
        let sparse = sparse_size(t).map_err(|_| bad())?;
        let n = if sparse > 0 { sparse } else { t.chars().count() };
        let mut out = PauliString::new(n);
        out.phase = phase;
        if sparse > 0 {
            parse_sparse(t, &mut out).map_err(|_| bad())?;
        } else {
            for (k, c) in t.chars().enumerate() {
                let p = match c {
                    'I' | '_' => 0,
                    'x' | 'X' => 1,
                    'y' | 'Y' => 2,
                    'z' | 'Z' => 3,
                    _ => return Err(bad()),
                };
                out.set(k, p);
            }
        }
        Ok(out)
    }

    /// Right-multiply by `rhs` (padding the shorter with identities): self ← self · rhs.
    pub fn mul_assign(&mut self, rhs: &PauliString) {
        self.ensure_num_qubits(rhs.n);
        let log_i = self.mul_paulis_returning_log_i(rhs);
        self.phase = (self.phase + log_i + rhs.phase) & 3;
    }

    /// Multiplies the Paulis of `rhs` into these (rhs may be shorter) and returns the power of
    /// i the product of the Paulis picks up, not counting either coefficient. Stim's
    /// `inplace_right_mul_returning_log_i_scalar`, word by word.
    fn mul_paulis_returning_log_i(&mut self, rhs: &PauliString) -> u8 {
        let mut cnt1: u32 = 0;
        let mut cnt2: u32 = 0;
        for w in 0..rhs.xs.len() {
            let (x2, z2) = (rhs.xs[w], rhs.zs[w]);
            let (old_x1, old_z1) = (self.xs[w], self.zs[w]);
            let x1 = old_x1 ^ x2;
            let z1 = old_z1 ^ z2;
            self.xs[w] = x1;
            self.zs[w] = z1;
            let x1z2 = old_x1 & z2;
            let anti = (x2 & old_z1) ^ x1z2;
            // Each anticommuting position contributes i^(1 + 2s), s = x1 ^ z1 ^ x1z2 after the
            // update (Stim's per-bit mod-4 counter, from zero, steps by +1 or -1); the total is
            // a sum mod 4, so per-word popcounts add up the same.
            let c1_bits = anti;
            let c2_bits = (x1 ^ z1 ^ x1z2) & anti;
            cnt1 += c1_bits.count_ones();
            cnt2 += c2_bits.count_ones();
        }
        ((cnt1 + 2 * cnt2) & 3) as u8
    }

    /// The product self · rhs as a new string.
    pub fn mul(&self, rhs: &PauliString) -> PauliString {
        let mut out = self.clone();
        out.mul_assign(rhs);
        out
    }

    /// Whether the two commute (the shorter padded with identities).
    pub fn commutes(&self, other: &PauliString) -> bool {
        let m = self.xs.len().min(other.xs.len());
        let mut c = 0u32;
        for w in 0..m {
            c += ((self.xs[w] & other.zs[w]) ^ (other.xs[w] & self.zs[w])).count_ones();
        }
        c & 1 == 0
    }

    /// How many qubits have a non-identity Pauli.
    pub fn weight(&self) -> usize {
        self.xs.iter().zip(&self.zs).map(|(x, z)| (x | z).count_ones() as usize).sum()
    }

    /// The qubits whose Pauli is one of `included` (characters of "XYZ", "I" or "_").
    pub fn pauli_indices(&self, included: &str) -> Result<Vec<usize>, String> {
        let mut want = [false; 4];
        for c in included.chars() {
            match c {
                'I' | '_' | 'i' => want[0] = true,
                'X' | 'x' => want[1] = true,
                'Y' | 'y' => want[2] = true,
                'Z' | 'z' => want[3] = true,
                _ => return Err(format!("Invalid character in include string: {c}")),
            }
        }
        Ok((0..self.n).filter(|&k| want[self.get(k) as usize]).collect())
    }

    /// The tensor product self ⊗ rhs (rhs's qubits after these); the phases multiply.
    pub fn tensor(&self, rhs: &PauliString) -> PauliString {
        let n = self.n + rhs.n;
        let mut out = PauliString::new(n);
        for k in 0..self.n {
            out.set(k, self.get(k));
        }
        for k in 0..rhs.n {
            out.set(self.n + k, rhs.get(k));
        }
        out.phase = (self.phase + rhs.phase) & 3;
        out
    }

    /// The tensor power: `power` copies side by side, the coefficient raised to `power`.
    pub fn tensor_power(&self, power: usize) -> PauliString {
        let phase = ((self.phase as usize * power) & 3) as u8;
        PauliString::from_fn(self.n * power, phase, |k| self.get(k % self.n.max(1)))
    }

    /// The Paulis at `indices`, with sign +1 (Stim's slicing).
    pub fn select(&self, indices: &[usize]) -> PauliString {
        PauliString::from_fn(indices.len(), 0, |k| self.get(indices[k]))
    }

    /// The matrix in the computational basis, row-major, as (re, im). `little_endian` puts
    /// qubit 0 in the lowest bit of the index.
    pub fn to_unitary_matrix(&self, little_endian: bool) -> Vec<(f64, f64)> {
        let n = self.n;
        let dim = 1usize << n;
        let bit = |k: usize| if little_endian { k } else { n - 1 - k };
        let (mut xmask, mut zmask, mut ys) = (0usize, 0usize, 0u32);
        for k in 0..n {
            if self.x(k) {
                xmask |= 1 << bit(k);
            }
            if self.z(k) {
                zmask |= 1 << bit(k);
            }
            if self.x(k) && self.z(k) {
                ys += 1;
            }
        }
        let mut m = vec![(0.0, 0.0); dim * dim];
        for col in 0..dim {
            let row = col ^ xmask;
            // Y|b> = i(-1)^b |b^1>, Z|b> = (-1)^b |b>: the sign counts Z (and Y) on 1s.
            let neg = (col & zmask).count_ones();
            let p = (self.phase as u32 + ys + 2 * neg) & 3;
            m[row * dim + col] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][p as usize];
        }
        m
    }

    /// The Pauli string whose matrix is `m` (row-major, dimension 2^n), up to the matrix
    /// being exactly a Pauli times a unit phase in {±1, ±i}. With `unsigned`, any unit phase
    /// is allowed and dropped.
    pub fn from_unitary_matrix(m: &[(f64, f64)], little_endian: bool, unsigned: bool) -> Result<PauliString, String> {
        let dim = (m.len() as f64).sqrt().round() as usize;
        if dim * dim != m.len() || dim == 0 || !dim.is_power_of_two() {
            return Err("The given matrix isn't a square matrix with power-of-2 dimensions.".into());
        }
        let n = dim.trailing_zeros() as usize;
        let bit = |k: usize| if little_endian { k } else { n - 1 - k };
        // Column 0 has one nonzero entry, at row xmask.
        let close = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4;
        let nonzero = |v: (f64, f64)| v.0.abs() > 1e-4 || v.1.abs() > 1e-4;
        let rows: Vec<usize> = (0..dim).filter(|&r| nonzero(m[r * dim])).collect();
        if rows.len() != 1 {
            return Err("The given matrix isn't a Pauli string matrix.".into());
        }
        let xmask = rows[0];
        let v0 = m[xmask * dim];
        let mut zmask = 0usize;
        for k in 0..n {
            let col = 1usize << bit(k);
            let v = m[(col ^ xmask) * dim + col];
            // v = v0 * (-1)^{z_k}: the ratio is ±1.
            if close(v, (-v0.0, -v0.1)) {
                zmask |= col;
            } else if !close(v, v0) {
                return Err("The given matrix isn't a Pauli string matrix.".into());
            }
        }
        let mut p = PauliString::new(n);
        for k in 0..n {
            p.set_x(k, xmask >> bit(k) & 1 == 1);
            p.set_z(k, zmask >> bit(k) & 1 == 1);
        }
        // The phase-0 string's entry in column 0 is w; the matrix is r times that string's.
        let w = p.to_unitary_matrix(little_endian)[xmask * dim];
        let den = w.0 * w.0 + w.1 * w.1;
        let r = ((v0.0 * w.0 + v0.1 * w.1) / den, (v0.1 * w.0 - v0.0 * w.1) / den);
        let units = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        match units.iter().position(|&u| close(u, r)) {
            Some(k) if !unsigned => p.phase = k as u8,
            _ if unsigned && ((r.0 * r.0 + r.1 * r.1) - 1.0).abs() < 1e-3 => {}
            _ => return Err("The given matrix isn't a Pauli string matrix.".into()),
        }
        let base = p.to_unitary_matrix(little_endian);
        let scale = if unsigned { r } else { (1.0, 0.0) };
        for (have, want) in m.iter().zip(&base) {
            let expect = (want.0 * scale.0 - want.1 * scale.1, want.0 * scale.1 + want.1 * scale.0);
            if !close(*have, expect) {
                return Err("The given matrix isn't a Pauli string matrix.".into());
            }
        }
        Ok(p)
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.sign() { "-" } else { "+" })?;
        if self.is_imaginary() {
            f.write_str("i")?;
        }
        let mut s = String::with_capacity(self.n);
        for k in 0..self.n {
            s.push(['_', 'X', 'Y', 'Z'][self.get(k) as usize]);
        }
        f.write_str(&s)
    }
}

/// The qubit count a sparse string (`X2*Y6`) names, or 0 when the text has no digits (dense).
fn sparse_size(text: &str) -> Result<usize, ()> {
    let mut n = 0usize;
    let mut cur: Option<u64> = None;
    for c in text.chars() {
        if let Some(d) = c.to_digit(10) {
            cur = Some(cur.unwrap_or(0).saturating_mul(10).saturating_add(d as u64));
        } else if let Some(v) = cur.take() {
            if v >= (usize::MAX as u64) {
                return Err(());
            }
            n = n.max(v as usize + 1);
        }
    }
    if let Some(v) = cur {
        if v >= (usize::MAX as u64) {
            return Err(());
        }
        n = n.max(v as usize + 1);
    }
    Ok(n)
}

fn parse_sparse(text: &str, out: &mut PauliString) -> Result<(), ()> {
    let mut cur: Option<u64> = None;
    let mut pauli: Option<char> = None;
    let flush = |pauli: &mut Option<char>, cur: &mut Option<u64>, out: &mut PauliString| -> Result<(), ()> {
        let (p, idx) = match (pauli.take(), cur.take()) {
            (Some(p), Some(i)) => (p, i),
            _ => return Err(()),
        };
        if idx as usize >= out.n {
            return Err(());
        }
        if p != 'I' {
            let mut single = PauliString::new(out.n);
            single.set(idx as usize, match p {
                'X' => 1,
                'Y' => 2,
                _ => 3,
            });
            out.mul_assign(&single);
        }
        Ok(())
    };
    for c in text.chars() {
        match c {
            '*' => flush(&mut pauli, &mut cur, out)?,
            'I' | 'x' | 'X' | 'y' | 'Y' | 'z' | 'Z' => {
                if pauli.is_some() {
                    return Err(());
                }
                pauli = Some(c.to_ascii_uppercase());
            }
            '0'..='9' => {
                if pauli.is_none() {
                    return Err(());
                }
                let d = c.to_digit(10).unwrap() as u64;
                cur = Some(cur.unwrap_or(0).saturating_mul(10).saturating_add(d));
            }
            _ => return Err(()),
        }
    }
    flush(&mut pauli, &mut cur, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ps(t: &str) -> PauliString {
        PauliString::from_text(t).unwrap()
    }

    #[test]
    fn text_round_trips_and_sparse() {
        assert_eq!(ps("-XYZ").to_string(), "-XYZ");
        assert_eq!(ps("").to_string(), "+");
        assert_eq!(ps("-X2*Y6").to_string(), "-__X___Y");
        assert_eq!(ps("X6*Y6").to_string(), "+i______Z");
        assert_eq!(ps("i_x").to_string(), "+i_X");
        assert!(PauliString::from_text("XQ").is_err());
    }

    #[test]
    fn products_pick_up_phases() {
        assert_eq!(ps("X").mul(&ps("Y")).to_string(), "+iZ");
        assert_eq!(ps("Y").mul(&ps("X")).to_string(), "-iZ");
        assert_eq!(ps("X").mul(&ps("XX_")).to_string(), "+_X_");
        assert_eq!(ps("XXXX").mul(&ps("_XYZ")).to_string(), "+X_ZY");
        assert_eq!(ps("iX").tensor_power(2).to_string(), "-XX");
        assert_eq!(ps("iX").tensor_power(3).to_string(), "-iXXX");
    }

    #[test]
    fn products_agree_with_matrices_across_words() {
        // Long strings (two words) multiply as their matrices would, checked on a slice.
        let mut rng = crate::surface_code::Xorshift::new(7);
        for _ in 0..200 {
            let n = 3;
            let a = PauliString::from_fn(n, (rng.next_u64() & 3) as u8, |_| (rng.next_u64() & 3) as u8);
            let b = PauliString::from_fn(n, (rng.next_u64() & 3) as u8, |_| (rng.next_u64() & 3) as u8);
            let c = a.mul(&b);
            let (ma, mb, mc) = (a.to_unitary_matrix(true), b.to_unitary_matrix(true), c.to_unitary_matrix(true));
            let d = 1 << n;
            for i in 0..d {
                for j in 0..d {
                    let mut s = (0.0, 0.0);
                    for k in 0..d {
                        let (x, y) = (ma[i * d + k], mb[k * d + j]);
                        s.0 += x.0 * y.0 - x.1 * y.1;
                        s.1 += x.0 * y.1 + x.1 * y.0;
                    }
                    assert!((s.0 - mc[i * d + j].0).abs() < 1e-9 && (s.1 - mc[i * d + j].1).abs() < 1e-9);
                }
            }
            assert_eq!(PauliString::from_unitary_matrix(&mc, true, false).unwrap(), c);
            let big = PauliString::from_fn(130, 0, |k| ((k * 7) % 4) as u8);
            let big2 = PauliString::from_fn(130, 0, |k| ((k * 5 + 1) % 4) as u8);
            let mut slow = PauliString::new(130);
            let mut phase = 0u8;
            for k in 0..130 {
                let x = PauliString::from_fn(1, 0, |_| big.get(k)).mul(&PauliString::from_fn(1, 0, |_| big2.get(k)));
                phase = (phase + x.phase) & 3;
                slow.set(k, x.get(0));
            }
            slow.phase = phase;
            assert_eq!(big.mul(&big2), slow);
        }
    }
}
