//! CSS codes given by their check matrices, and a memory experiment for any of them.
//!
//! WHY THIS EXISTS
//! ---------------
//! The surface, colour and bivariate bicycle codes each have a hand-made syndrome circuit. A
//! code someone brings (a hypergraph product of two classical codes, a code from a paper's
//! table, a code under study) has none. This module takes H_X and H_Z, checks they commute,
//! finds paired logical operators, and writes a memory experiment that is correct for any
//! CSS code: every X check measured, then every Z check, each in layers no qubit is used
//! twice in, so no schedule question of hook errors or interleaving can make it wrong. It is
//! deeper than a tailored schedule, and its circuit distance can be below the code's where a
//! check's ancilla spreads an error; the tailored circuits remain for the codes that have them.
//!
//! Two families come built: the hypergraph product of any two classical codes (Tillich and
//! Zémor), and the triangular 6.6.6 colour code on Stim's own layout.

use crate::gf2::BitMatrix;

#[derive(Clone, Debug)]
pub struct CssCode {
    pub n: usize,
    /// Each check as the data qubits it acts on.
    pub hx: Vec<Vec<usize>>,
    pub hz: Vec<Vec<usize>>,
    /// Each check's colour (0, 1, 2) where the code is a colour code, the X and Z checks
    /// alike (one colour per plaquette).
    pub colors: Option<Vec<u8>>,
}

/// Rows as bit-packed words, for elimination.
fn packed(n: usize, rows: &[Vec<usize>]) -> Vec<Vec<u64>> {
    rows.iter()
        .map(|r| {
            let mut w = vec![0u64; n.div_ceil(64).max(1)];
            for &q in r {
                w[q / 64] ^= 1 << (q % 64);
            }
            w
        })
        .collect()
}

fn ones(n: usize, w: &[u64]) -> Vec<usize> {
    (0..n).filter(|&q| w[q / 64] >> (q % 64) & 1 == 1).collect()
}

/// An echelon basis kept as rows with their pivots: adding a row reduces it against the
/// basis, and keeps it when something is left.
struct Echelon {
    rows: Vec<(usize, Vec<u64>)>,
}

impl Echelon {
    fn new() -> Echelon {
        Echelon { rows: Vec::new() }
    }

    /// Whether `w` was independent of the basis (and is now in it).
    fn add(&mut self, mut w: Vec<u64>) -> bool {
        for (p, r) in &self.rows {
            if w[p / 64] >> (p % 64) & 1 == 1 {
                for (a, b) in w.iter_mut().zip(r) {
                    *a ^= b;
                }
            }
        }
        match w.iter().position(|&x| x != 0) {
            Some(i) => {
                let p = i * 64 + w[i].trailing_zeros() as usize;
                self.rows.push((p, w));
                true
            }
            None => false,
        }
    }
}

impl CssCode {
    /// The code of these checks on `n` data qubits: each check acts on data qubits below `n`,
    /// every X check commutes with every Z check, and at least one logical qubit is left.
    pub fn new(n: usize, hx: Vec<Vec<usize>>, hz: Vec<Vec<usize>>) -> Result<CssCode, String> {
        if n == 0 || n > 100_000 {
            return Err(format!("{n} data qubits: from 1 to 100,000"));
        }
        let clean = |rows: Vec<Vec<usize>>, kind: &str| -> Result<Vec<Vec<usize>>, String> {
            rows.into_iter()
                .enumerate()
                .map(|(i, mut r)| {
                    r.sort_unstable();
                    r.dedup();
                    match r.last() {
                        None => Err(format!("{kind} check {i} acts on no qubit")),
                        Some(&q) if q >= n => Err(format!("{kind} check {i} acts on qubit {q}, but there are {n}")),
                        _ => Ok(r),
                    }
                })
                .collect()
        };
        let (hx, hz) = (clean(hx, "X")?, clean(hz, "Z")?);
        let code = CssCode { n, hx, hz, colors: None };
        let zp = packed(n, &code.hz);
        for (i, x) in packed(n, &code.hx).iter().enumerate() {
            for (j, z) in zp.iter().enumerate() {
                if x.iter().zip(z).map(|(a, b)| (a & b).count_ones()).sum::<u32>() % 2 == 1 {
                    return Err(format!("X check {i} and Z check {j} anticommute"));
                }
            }
        }
        if code.k() == 0 {
            return Err("this code encodes no logical qubits".into());
        }
        Ok(code)
    }

    fn rank(&self, rows: &[Vec<usize>]) -> usize {
        let mut e = Echelon::new();
        packed(self.n, rows).into_iter().filter(|w| e.add(w.clone())).count()
    }

    /// Logical qubits: n − rank H_X − rank H_Z.
    pub fn k(&self) -> usize {
        self.n - self.rank(&self.hx) - self.rank(&self.hz)
    }

    /// Logical operators (X, Z), one per row, paired so that X_i and Z_j anticommute exactly
    /// when i = j.
    pub fn logicals(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let n = self.n;
        // Z logicals: in the kernel of H_X, outside the span of H_Z; X logicals likewise.
        let pick = |stabilizers: &[Vec<usize>], commute_with: &[Vec<usize>]| -> Vec<Vec<u64>> {
            let mut e = Echelon::new();
            for w in packed(n, stabilizers) {
                e.add(w);
            }
            let kernel = BitMatrix::from_rows(n, commute_with).kernel();
            (0..kernel.rows).map(|r| packed(n, &[kernel.row_ones(r)]).remove(0)).filter(|w| e.add(w.clone())).collect()
        };
        let lz = pick(&self.hz, &self.hx);
        let lx = pick(&self.hx, &self.hz);
        let k = lz.len();
        let dot = |a: &[u64], b: &[u64]| a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum::<u32>() % 2 == 1;
        // P = Lx Lzᵀ; replacing Lx by P⁻¹ Lx makes the pairing the identity.
        let mut p = BitMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                p.set(i, j, dot(&lx[i], &lz[j]));
            }
        }
        let inv = p.inverse().expect("logical X and Z operators pair nondegenerately");
        let lx: Vec<Vec<u64>> = (0..k)
            .map(|i| {
                let mut w = vec![0u64; lx[0].len()];
                for j in inv.row_ones(i) {
                    for (a, b) in w.iter_mut().zip(&lx[j]) {
                        *a ^= b;
                    }
                }
                w
            })
            .collect();
        (lx.iter().map(|w| ones(n, w)).collect(), lz.iter().map(|w| ones(n, w)).collect())
    }

    /// The hypergraph product of classical codes with parity checks `h1` (on `n1` bits) and
    /// `h2` (on `n2`): data qubits are bit pairs (n1 n2 of them) then check pairs (r1 r2),
    /// H_X = [H1 ⊗ I | I ⊗ H2ᵀ] and H_Z = [I ⊗ H2 | H1ᵀ ⊗ I].
    pub fn hypergraph_product(h1: &[Vec<usize>], n1: usize, h2: &[Vec<usize>], n2: usize) -> Result<CssCode, String> {
        for (h, nb, name) in [(h1, n1, "h1"), (h2, n2, "h2")] {
            if let Some(q) = h.iter().flatten().find(|&&q| q >= nb) {
                return Err(format!("{name} has a check on bit {q}, but there are {nb}"));
            }
        }
        let (r1, r2) = (h1.len(), h2.len());
        let bit = |a: usize, b: usize| a * n2 + b;
        let chk = |c1: usize, c2: usize| n1 * n2 + c1 * r2 + c2;
        let col = |h: &[Vec<usize>], nb: usize| {
            let mut t = vec![Vec::new(); nb];
            for (c, row) in h.iter().enumerate() {
                for &q in row {
                    t[q].push(c);
                }
            }
            t
        };
        let (t1, t2) = (col(h1, n1), col(h2, n2));
        // X checks indexed (check of h1, bit of h2); Z checks (bit of h1, check of h2).
        let mut hx = Vec::with_capacity(r1 * n2);
        for c1 in 0..r1 {
            for b in 0..n2 {
                let mut row: Vec<usize> = h1[c1].iter().map(|&a| bit(a, b)).collect();
                row.extend(t2[b].iter().map(|&c2| chk(c1, c2)));
                hx.push(row);
            }
        }
        let mut hz = Vec::with_capacity(n1 * r2);
        for a in 0..n1 {
            for c2 in 0..r2 {
                let mut row: Vec<usize> = h2[c2].iter().map(|&b| bit(a, b)).collect();
                row.extend(t1[a].iter().map(|&c1| chk(c1, c2)));
                hz.push(row);
            }
        }
        // Checks that are empty (a classical check of no bits) measure nothing.
        hx.retain(|r| !r.is_empty());
        hz.retain(|r| !r.is_empty());
        CssCode::new(n1 * n2 + r1 * r2, hx, hz)
    }

    /// The triangular 6.6.6 colour code of odd `distance`, on the layout of Stim's generated
    /// colour code: each plaquette is both an X and a Z check. [[7, 1, 3]] at distance 3 (the
    /// Steane code), [[19, 1, 5]] at 5.
    pub fn color_code(distance: usize) -> Result<CssCode, String> {
        if distance < 3 || distance.is_multiple_of(2) || distance > 101 {
            return Err(format!("distance {distance}: odd, from 3 to 101"));
        }
        let w = (distance + (distance - 1) / 2) as i64;
        let mut data = std::collections::BTreeMap::new();
        let mut plaquettes = Vec::new();
        for y in 0..w {
            for i in 0..w - y {
                let p = (y + 2 * i, y);
                if (i + 2 * y) % 3 == 2 {
                    plaquettes.push(p);
                } else {
                    let q = data.len();
                    data.insert(p, q);
                }
            }
        }
        let checks: Vec<Vec<usize>> = plaquettes
            .iter()
            .map(|&(x, y)| [(2, 0), (1, 1), (1, -1), (-2, 0), (-1, 1), (-1, -1)].iter().filter_map(|(dx, dy)| data.get(&(x + dx, y + dy)).copied()).collect())
            .collect();
        let mut code = CssCode::new(data.len(), checks.clone(), checks)?;
        // Plaquettes one or two rows apart can share a qubit, and three apart cannot, so the
        // row mod 3 colours them properly.
        code.colors = Some(plaquettes.iter().map(|&(_, y)| (y % 3) as u8).collect());
        Ok(code)
    }

    /// A memory experiment as Stim's text: the data prepared in the Z basis (or X), `rounds`
    /// rounds of syndrome extraction, the data read out. Each round measures every X check
    /// (ancilla in |+⟩, CNOTs from it, read in X) and then every Z check (ancilla in |0⟩,
    /// CNOTs onto it), in layers where no qubit is used twice. Noise, one strength `p`:
    /// depolarizing on the data at the start of each round and after every CNOT, and flips
    /// after every reset and before every measurement. Detectors compare each check with its
    /// last value (from the first round for the checks the preparation fixes) and with the
    /// final readout; the observables are the `k` logicals of the basis.
    pub fn memory(&self, rounds: usize, p: f64, x_basis: bool) -> Result<String, String> {
        self.memory_annotated(rounds, p, x_basis, false)
    }

    /// `memory`, and with `annotate` each detector given Chromobius's colour and basis as a 4th
    /// coordinate (the check's colour, plus 3 for a Z check). Colour codes only.
    pub fn memory_annotated(&self, rounds: usize, p: f64, x_basis: bool, annotate: bool) -> Result<String, String> {
        let tag = |x: bool, c: usize| -> String {
            match (&self.colors, annotate) {
                (Some(colors), true) => format!(", {}", colors[c] + if x { 0 } else { 3 }),
                _ => String::new(),
            }
        };
        if annotate && self.colors.is_none() {
            return Err("colour annotations are for colour codes (CssCode.color_code)".into());
        }
        crate::memory::probability(p)?;
        if rounds == 0 || rounds > 1_000_000 {
            return Err(format!("{rounds} rounds: from 1 to 1,000,000"));
        }
        let n = self.n;
        let (rx, rz) = (self.hx.len(), self.hz.len());
        let xa = |c: usize| (n + c) as u32;
        let za = |c: usize| (n + rx + c) as u32;
        let list = |v: &[u32]| v.iter().map(|q| q.to_string()).collect::<Vec<_>>().join(" ");
        let data: Vec<u32> = (0..n as u32).collect();
        let xs: Vec<u32> = (0..rx).map(xa).collect();
        let zs: Vec<u32> = (0..rz).map(za).collect();
        let noisy = p > 0.0;
        let mut lines: Vec<String> = Vec::new();

        // One round, as instructions.
        let mut round = Vec::new();
        round.push("TICK".into());
        if noisy {
            round.push(format!("DEPOLARIZE1({p}) {}", list(&data)));
        }
        for (x_checks, anc) in [(true, &xs), (false, &zs)] {
            let checks = if x_checks { &self.hx } else { &self.hz };
            if checks.is_empty() {
                continue;
            }
            let (reset, measure, flip) = if x_checks { ("RX", "MX", "Z_ERROR") } else { ("R", "M", "X_ERROR") };
            round.push(format!("{reset} {}", list(anc)));
            if noisy {
                round.push(format!("{flip}({p}) {}", list(anc)));
            }
            for layer in layers(checks, n) {
                let pairs: Vec<u32> = layer
                    .iter()
                    .flat_map(|&(c, q)| if x_checks { [anc[c], q as u32] } else { [q as u32, anc[c]] })
                    .collect();
                round.push("TICK".into());
                round.push(format!("CX {}", list(&pairs)));
                if noisy {
                    round.push(format!("DEPOLARIZE2({p}) {}", list(&pairs)));
                }
            }
            round.push("TICK".into());
            if noisy {
                round.push(format!("{flip}({p}) {}", list(anc)));
            }
            round.push(format!("{measure} {}", list(anc)));
        }
        let per = rx + rz;
        // Lookback of check (x, c) measured `back` rounds before the end of a round.
        let rec = |x: bool, c: usize, back: usize| if x { per - c + back * per } else { rz - c + back * per };
        let recs = |v: &[usize]| v.iter().map(|k| format!("rec[-{k}]")).collect::<Vec<_>>().join(" ");

        let (prep, prep_flip, final_m, final_flip) = if x_basis { ("RX", "Z_ERROR", "MX", "Z_ERROR") } else { ("R", "X_ERROR", "M", "X_ERROR") };
        lines.push(format!("{prep} {}", list(&data)));
        if noisy {
            lines.push(format!("{prep_flip}({p}) {}", list(&data)));
        }
        lines.extend(round.iter().cloned());
        let fixed = |x: bool| x == x_basis;
        for (x, count) in [(true, rx), (false, rz)] {
            if fixed(x) {
                for c in 0..count {
                    lines.push(format!("DETECTOR({c}, 0, {}{}) {}", u8::from(!x), tag(x, c), recs(&[rec(x, c, 0)])));
                }
            }
        }
        if rounds > 1 {
            let mut body = round.clone();
            body.push("SHIFT_COORDS(0, 1)".into());
            for (x, count) in [(true, rx), (false, rz)] {
                for c in 0..count {
                    body.push(format!("DETECTOR({c}, 0, {}{}) {}", u8::from(!x), tag(x, c), recs(&[rec(x, c, 0), rec(x, c, 1)])));
                }
            }
            if rounds == 2 {
                lines.extend(body);
            } else {
                lines.push(format!("REPEAT {} {{", rounds - 1));
                lines.extend(body.iter().map(|l| format!("    {l}")));
                lines.push("}".into());
            }
        }
        if noisy {
            lines.push(format!("{final_flip}({p}) {}", list(&data)));
        }
        lines.push(format!("{final_m} {}", list(&data)));
        let checks = if x_basis { &self.hx } else { &self.hz };
        for (c, row) in checks.iter().enumerate() {
            let mut r: Vec<usize> = row.iter().map(|&q| n - q).collect();
            r.push(n + rec(x_basis, c, 0));
            r.sort_unstable();
            lines.push(format!("DETECTOR({c}, 1, {}{}) {}", u8::from(!x_basis), tag(x_basis, c), recs(&r)));
        }
        let (lx, lz) = self.logicals();
        for (i, l) in (if x_basis { lx } else { lz }).iter().enumerate() {
            let mut r: Vec<usize> = l.iter().map(|&q| n - q).collect();
            r.sort_unstable();
            lines.push(format!("OBSERVABLE_INCLUDE({i}) {}", recs(&r)));
        }
        Ok(lines.join("\n") + "\n")
    }
}

/// The checks' CNOTs in layers where no check and no qubit appears twice: each (check, qubit)
/// edge given the first layer free at both ends.
fn layers(checks: &[Vec<usize>], n: usize) -> Vec<Vec<(usize, usize)>> {
    let mut out: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut busy_q: Vec<Vec<bool>> = Vec::new();
    let mut busy_c: Vec<Vec<bool>> = Vec::new();
    for (c, row) in checks.iter().enumerate() {
        for &q in row {
            let layer = (0..).find(|&l| l >= out.len() || !busy_q[l][q] && !busy_c[l][c]).expect("a free layer");
            if layer == out.len() {
                out.push(Vec::new());
                busy_q.push(vec![false; n]);
                busy_c.push(vec![false; checks.len()]);
            }
            out[layer].push((c, q));
            busy_q[layer][q] = true;
            busy_c[layer][c] = true;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repetition(d: usize) -> Vec<Vec<usize>> {
        (0..d - 1).map(|i| vec![i, i + 1]).collect()
    }

    #[test]
    fn hypergraph_products_of_repetition_codes_are_surface_codes() {
        for d in [2, 3, 5] {
            let c = CssCode::hypergraph_product(&repetition(d), d, &repetition(d), d).unwrap();
            assert_eq!((c.n, c.k()), (d * d + (d - 1) * (d - 1), 1));
        }
    }

    /// The colour code's plaquette colours are a proper 3-colouring: plaquettes sharing a qubit
    /// differ. The annotated memory carries them, plus 3 on the Z checks.
    #[test]
    fn colour_codes_are_properly_coloured() {
        for d in [3, 5, 7, 9] {
            let c = CssCode::color_code(d).unwrap();
            let colors = c.colors.as_ref().unwrap();
            assert_eq!(colors.len(), c.hx.len());
            for (a, ra) in c.hx.iter().enumerate() {
                for (b, rb) in c.hx.iter().enumerate().skip(a + 1) {
                    if ra.iter().any(|q| rb.contains(q)) {
                        assert_ne!(colors[a], colors[b], "d={d}: plaquettes {a} and {b}");
                    }
                }
            }
            let text = c.memory_annotated(3, 0.001, false, true).unwrap();
            assert!(text.lines().filter(|l| l.trim_start().starts_with("DETECTOR")).all(|l| l.split(')').next().unwrap().matches(',').count() == 3));
            assert!(text.contains(&format!("DETECTOR(0, 0, 1, {})", colors[0] + 3)));
            assert_eq!(c.memory_annotated(3, 0.001, false, false).unwrap(), c.memory(3, 0.001, false).unwrap());
        }
        let plain = CssCode::new(3, vec![vec![0, 1], vec![1, 2]], vec![]).unwrap();
        assert!(plain.memory_annotated(2, 0.001, true, true).is_err());
    }

    #[test]
    fn the_hamming_product_and_colour_codes() {
        let hamming = vec![vec![0, 2, 4, 6], vec![1, 2, 5, 6], vec![3, 4, 5, 6]];
        let c = CssCode::hypergraph_product(&hamming, 7, &hamming, 7).unwrap();
        assert_eq!((c.n, c.k()), (58, 16));
        for (d, n) in [(3, 7), (5, 19), (7, 37)] {
            let c = CssCode::color_code(d).unwrap();
            assert_eq!((c.n, c.k()), (n, 1));
        }
    }

    #[test]
    fn logicals_pair_and_commute_with_the_checks() {
        let hamming = vec![vec![0, 2, 4, 6], vec![1, 2, 5, 6], vec![3, 4, 5, 6]];
        let c = CssCode::hypergraph_product(&hamming, 7, &hamming, 7).unwrap();
        let (lx, lz) = c.logicals();
        let meet = |a: &[usize], b: &[usize]| a.iter().filter(|q| b.contains(q)).count() % 2;
        for i in 0..lx.len() {
            for j in 0..lz.len() {
                assert_eq!(meet(&lx[i], &lz[j]), usize::from(i == j));
            }
            assert!(c.hz.iter().all(|z| meet(z, &lx[i]) == 0));
            assert!(c.hx.iter().all(|x| meet(x, &lz[i]) == 0));
        }
    }

    #[test]
    fn bad_codes_are_refused() {
        assert!(CssCode::new(3, vec![vec![0, 1]], vec![vec![1, 2]]).is_err(), "anticommuting");
        assert!(CssCode::new(2, vec![vec![0, 1]], vec![vec![0, 1]]).is_err(), "no logical qubit");
        assert!(CssCode::new(2, vec![vec![0, 5]], vec![]).is_err(), "out of range");
        assert!(CssCode::color_code(4).is_err());
    }
}
