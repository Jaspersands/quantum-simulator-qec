//! BP+OSD: belief propagation, and where it does not converge, ordered-
//! statistics decoding (Fossorier and Lin; for quantum codes, Panteleev and
//! Kalachev, and Roffe et al.).
//!
//! WHY THIS EXISTS
//! ---------------
//! BP alone often fails on quantum codes: degenerate errors split its
//! beliefs, and it can oscillate forever. OSD turns its beliefs into an
//! answer anyway. It ranks the faults from most to least likely, takes the
//! most likely set of them that is linearly independent (an information set),
//! and solves the syndrome equation on those alone. That is OSD-0. Higher
//! orders also try flipping a few of the other, less likely faults and keep the
//! lightest consistent correction:
//! - OSD-E tries every combination of the first `order` of them;
//! - OSD-CS (combination sweep) tries each one alone and every pair among the
//!   first `order`.
//!
//! One elimination serves every candidate. After row-reducing [H | s] with the
//! columns in rank order, the pivot faults of a candidate that flips the
//! non-pivot set F are s′ ⊕ Σ_{f∈F} H′_f, where primes are the reduced forms.
//! A correction's weight is Σ ln(1/p) over the faults it contains, with p the
//! prior, as `ldpc` weighs them.

use crate::bp::{Bp, BpWork, Method, Schedule, SerialState};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OsdMethod {
    Osd0,
    Exhaustive(usize),
    CombinationSweep(usize),
}

pub struct BpOsd {
    pub bp: Bp,
    columns: Vec<Vec<u32>>,
    weight: Vec<f64>,
    pub method: Method,
    pub max_iter: usize,
    /// The method and order the candidates were set up with (`ldpc` sets them up once, when
    /// it is built).
    pub osd: OsdMethod,
    /// The order now: at 0, OSD-0 alone, whatever was set up (`ldpc`'s `osd_order`).
    pub order: usize,
    pub schedule: Schedule,
}

pub struct BpOsdWork {
    pub bp: BpWork,
    /// The serial schedule's order and generator, kept from decode to decode.
    pub serial: SerialState,
    /// The last OSD's order-0 solution (`ldpc`'s `osd0_decoding`).
    pub osd0: Vec<u8>,
    order: Vec<u32>,
    rows: Vec<u64>,
    words: usize,
    /// The correction: one byte per fault.
    pub correction: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpOsdOutcome {
    pub converged: bool,
    pub iterations: usize,
}

impl BpOsd {
    pub fn new(num_checks: usize, columns: Vec<Vec<u32>>, priors: &[f64], method: Method, max_iter: usize, osd: OsdMethod) -> Result<BpOsd, String> {
        // One matrix for BP and OSD alike: each column's checks sorted and
        // distinct, as Bp keeps them.
        let columns: Vec<Vec<u32>> = columns
            .into_iter()
            .map(|mut c| {
                c.sort_unstable();
                c.dedup();
                c
            })
            .collect();
        let bp = Bp::new(num_checks, &columns, priors)?;
        let weight = priors.iter().map(|&p| (1.0 / p).ln()).collect();
        let order = match osd {
            OsdMethod::Osd0 => 0,
            OsdMethod::Exhaustive(k) | OsdMethod::CombinationSweep(k) => k,
        };
        Ok(BpOsd { bp, columns, weight, method, max_iter, osd, order, schedule: Schedule::Parallel })
    }

    /// New priors, for BP and for OSD's weights.
    pub fn set_priors(&mut self, priors: &[f64]) -> Result<(), String> {
        self.bp.set_priors(priors)?;
        self.weight = priors.iter().map(|&p| (1.0 / p).ln()).collect();
        Ok(())
    }

    pub fn work(&self) -> BpOsdWork {
        let n = self.bp.num_vars;
        let words = (n + 1).div_ceil(64);
        BpOsdWork {
            bp: self.bp.work(),
            serial: SerialState::new(n),
            osd0: vec![0; n],
            order: (0..n as u32).collect(),
            rows: vec![0; self.bp.num_checks * words],
            words,
            correction: vec![0; n],
        }
    }

    /// Decode a syndrome (one byte per check); the correction is left in
    /// `w.correction`.
    pub fn decode(&self, syndrome: &[u8], w: &mut BpOsdWork) -> BpOsdOutcome {
        let out = self.bp.decode_scheduled(syndrome, self.method, self.max_iter, self.schedule, &mut w.serial, &mut w.bp);
        if out.converged {
            w.correction.copy_from_slice(&w.bp.hard);
            return BpOsdOutcome { converged: true, iterations: out.iterations };
        }
        self.osd(syndrome, w);
        BpOsdOutcome { converged: false, iterations: out.iterations }
    }

    fn osd(&self, syndrome: &[u8], w: &mut BpOsdWork) {
        let n = self.bp.num_vars;
        let m = self.bp.num_checks;
        let words = w.words;
        // Most likely flipped first: ascending posterior LLR, sorted as ldpc sorts it, with C's
        // qsort over (value, column) records in column order. Ties (min-sum repeats values,
        // and their order picks OSD-CS's candidates) end where this platform's C library
        // leaves them.
        let llr = &w.bp.llr;
        let mut records: Vec<(f64, u32)> = (0..n as u32).map(|i| (llr[i as usize], i)).collect();
        crate::cpp_random::CLibrary::NATIVE.qsort(&mut records, &mut |a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (slot, r) in w.order.iter_mut().zip(&records) {
            *slot = r.1;
        }
        // [H | s] with H's columns in that order; the syndrome is column n.
        w.rows.fill(0);
        for (pos, &v) in w.order.iter().enumerate() {
            for &c in &self.columns[v as usize] {
                w.rows[c as usize * words + pos / 64] |= 1 << (pos % 64);
            }
        }
        for (c, &s) in syndrome.iter().enumerate() {
            if s != 0 {
                w.rows[c * words + n / 64] |= 1 << (n % 64);
            }
        }
        // Row-reduce, pivoting on columns in rank order.
        let mut pivots: Vec<usize> = Vec::new();
        let mut r = 0;
        for col in 0..n {
            if r == m {
                break;
            }
            let (wi, bit) = (col / 64, 1u64 << (col % 64));
            let Some(p) = (r..m).find(|&i| w.rows[i * words + wi] & bit != 0) else { continue };
            if p != r {
                for k in 0..words {
                    w.rows.swap(p * words + k, r * words + k);
                }
            }
            for i in 0..m {
                if i != r && w.rows[i * words + wi] & bit != 0 {
                    for k in wi..words {
                        let x = w.rows[r * words + k];
                        w.rows[i * words + k] ^= x;
                    }
                }
            }
            pivots.push(col);
            r += 1;
        }
        let rank = pivots.len();
        let is_pivot = {
            let mut v = vec![false; n];
            for &p in &pivots {
                v[p] = true;
            }
            v
        };
        let non_pivots: Vec<usize> = (0..n).filter(|&c| !is_pivot[c]).collect();
        let get = |rows: &[u64], i: usize, col: usize| rows[i * words + col / 64] >> (col % 64) & 1 == 1;
        let weight_of = |pos: usize| self.weight[w.order[pos] as usize];

        // A candidate: the non-pivot positions flipped. Its pivot bits are the
        // reduced syndrome XOR the reduced columns of the flips; only its
        // weight is needed until it wins.
        let pivot_bit = |flips: &[usize], rows: &[u64], i: usize| flips.iter().fold(get(rows, i, n), |b, &f| b ^ get(rows, i, f));
        let weigh = |flips: &[usize], rows: &[u64]| -> f64 {
            let pivots_weight: f64 = (0..rank).filter(|&i| pivot_bit(flips, rows, i)).map(|i| weight_of(pivots[i])).sum();
            pivots_weight + flips.iter().map(|&f| weight_of(f)).sum::<f64>()
        };
        let mut best_w = weigh(&[], &w.rows);
        let mut best_flips: Vec<usize> = Vec::new();
        w.osd0.fill(0);
        for (i, &p) in pivots.iter().enumerate() {
            if pivot_bit(&[], &w.rows, i) {
                w.osd0[w.order[p] as usize] = 1;
            }
        }
        let mut consider = |flips: Vec<usize>| {
            let wt = weigh(&flips, &w.rows);
            if wt < best_w {
                best_w = wt;
                best_flips = flips;
            }
        };
        match if self.order == 0 { OsdMethod::Osd0 } else { self.osd } {
            OsdMethod::Osd0 => {}
            OsdMethod::Exhaustive(order) => {
                let k = order.min(non_pivots.len());
                for mask in 1u64..(1 << k) {
                    consider((0..k).filter(|&i| mask >> i & 1 == 1).map(|i| non_pivots[i]).collect());
                }
            }
            OsdMethod::CombinationSweep(order) => {
                for &f in &non_pivots {
                    consider(vec![f]);
                }
                let k = order.min(non_pivots.len());
                for i in 0..k {
                    for j in i + 1..k {
                        consider(vec![non_pivots[i], non_pivots[j]]);
                    }
                }
            }
        }
        let best_x: Vec<bool> = (0..rank).map(|i| pivot_bit(&best_flips, &w.rows, i)).collect();
        w.correction.fill(0);
        for (i, &p) in pivots.iter().enumerate() {
            if best_x[i] {
                w.correction[w.order[p] as usize] = 1;
            }
        }
        for &f in &best_flips {
            w.correction[w.order[f] as usize] ^= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_code::Xorshift;

    fn random_code(m: usize, n: usize, rng: &mut Xorshift) -> Vec<Vec<u32>> {
        (0..n)
            .map(|_| {
                let mut col: Vec<u32> = Vec::new();
                while col.len() < 3 {
                    let c = (rng.next_u64() % m as u64) as u32;
                    if !col.contains(&c) {
                        col.push(c);
                    }
                }
                col
            })
            .collect()
    }

    fn syndrome_of(columns: &[Vec<u32>], m: usize, e: &[u8]) -> Vec<u8> {
        let mut s = vec![0u8; m];
        for (v, &x) in e.iter().enumerate() {
            if x != 0 {
                for &c in &columns[v] {
                    s[c as usize] ^= 1;
                }
            }
        }
        s
    }

    /// Whatever BP does, the correction explains the syndrome, for every OSD
    /// method: OSD solves the syndrome equation exactly.
    #[test]
    fn every_correction_explains_its_syndrome() {
        let mut rng = Xorshift::new(11);
        let (m, n) = (30, 60);
        let columns = random_code(m, n, &mut rng);
        let priors = vec![0.05; n];
        for osd in [OsdMethod::Osd0, OsdMethod::Exhaustive(4), OsdMethod::CombinationSweep(5)] {
            // Few iterations, so OSD runs often.
            let dec = BpOsd::new(m, columns.clone(), &priors, Method::MinSum { scale: 0.625 }, 2, osd).unwrap();
            let mut w = dec.work();
            let mut osd_runs = 0;
            for _ in 0..200 {
                let e: Vec<u8> = (0..n).map(|_| u8::from(rng.next_f64() < 0.08)).collect();
                let s = syndrome_of(&columns, m, &e);
                let out = dec.decode(&s, &mut w);
                osd_runs += usize::from(!out.converged);
                assert_eq!(syndrome_of(&columns, m, &w.correction), s, "{osd:?}");
            }
            assert!(osd_runs > 50, "{osd:?}: OSD ran on {osd_runs} of 200");
        }
    }

    /// Higher orders never do worse than OSD-0 in weight, and exhaustive
    /// search over every non-pivot bit finds the true minimum weight.
    #[test]
    fn higher_orders_are_no_heavier_and_full_search_is_optimal() {
        let mut rng = Xorshift::new(12);
        let (m, n) = (8, 14);
        let columns = random_code(m, n, &mut rng);
        let priors: Vec<f64> = (0..n).map(|i| 0.02 + 0.01 * i as f64).collect();
        let weight = |e: &[u8]| e.iter().enumerate().filter(|x| *x.1 != 0).map(|(i, _)| (1.0 / priors[i]).ln()).sum::<f64>();
        let mk = |osd| BpOsd::new(m, columns.clone(), &priors, Method::ProductSum, 1, osd).unwrap();
        let (d0, dcs, dfull) = (mk(OsdMethod::Osd0), mk(OsdMethod::CombinationSweep(4)), mk(OsdMethod::Exhaustive(n)));
        let (mut w0, mut wcs, mut wfull) = (d0.work(), dcs.work(), dfull.work());
        for _ in 0..100 {
            let e: Vec<u8> = (0..n).map(|_| u8::from(rng.next_f64() < 0.15)).collect();
            let s = syndrome_of(&columns, m, &e);
            let (a, b, c) = (d0.decode(&s, &mut w0), dcs.decode(&s, &mut wcs), dfull.decode(&s, &mut wfull));
            if a.converged || b.converged || c.converged {
                continue;
            }
            let (x0, xcs, xfull) = (weight(&w0.correction), weight(&wcs.correction), weight(&wfull.correction));
            assert!(xcs <= x0 + 1e-12 && xfull <= xcs + 1e-12);
            // Brute force over every error with this syndrome.
            let mut best = f64::INFINITY;
            for pattern in 0u32..(1 << n) {
                let e: Vec<u8> = (0..n).map(|i| (pattern >> i & 1) as u8).collect();
                if syndrome_of(&columns, m, &e) == s {
                    best = best.min(weight(&e));
                }
            }
            assert!((xfull - best).abs() < 1e-9, "exhaustive OSD {xfull} vs brute force {best}");
        }
    }
}
