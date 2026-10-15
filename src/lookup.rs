//! Lookup-table decoders, as CUDA-Q QEC's `single_error_lut` and `multi_error_lut`
//! (`libs/qec/lib/decoders/lut.cpp`): every set of at most `depth` faults, by size and then in
//! lexicographic order of fault indices, entered under the syndrome it sets off, the first set
//! for each syndrome kept. Decoding is one lookup; a syndrome no set explains is not decoded.
//!
//! The error rates do not choose between sets: they give the probability reported for the set
//! found (the product of its faults' rates), as CUDA-Q's does.

use std::collections::HashMap;

/// A lookup table over a check matrix.
#[derive(Clone, Debug)]
pub struct LookupTable {
    num_rows: usize,
    num_faults: usize,
    table: HashMap<Vec<u64>, (Vec<u32>, f64)>,
}

/// C(n, k), saturating.
fn binomial(n: usize, k: usize) -> u128 {
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 0..k {
        r = r.saturating_mul((n - i) as u128) / (i as u128 + 1);
        if r > u64::MAX as u128 {
            return r;
        }
    }
    r
}

impl LookupTable {
    /// The table for the matrix with `num_rows` rows whose columns (faults) set off `columns`,
    /// of every set of `depth` or fewer faults. `rates` (one per fault, or none) give the
    /// probabilities reported.
    pub fn new(num_rows: usize, columns: &[Vec<u32>], depth: usize, rates: &[f64]) -> Result<LookupTable, String> {
        let n = columns.len();
        if depth < 1 {
            return Err("lut_error_depth must be >= 1".into());
        }
        if depth > n {
            return Err("lut_error_depth must be <= block_size".into());
        }
        if !rates.is_empty() {
            if rates.len() != n {
                return Err("error_rate_vec must be of size block_size".into());
            }
            if rates.iter().any(|&p| !(0.0..=1.0).contains(&p)) {
                return Err("error_rate_vec value is out of range [0, 1]".into());
            }
        }
        if binomial(n, depth) > 1_000_000_000 {
            return Err("lut_error_depth is too large for LUT decoder".into());
        }
        if let Some(&r) = columns.iter().flatten().find(|&&r| r as usize >= num_rows) {
            return Err(format!("row {r} is beyond the matrix's {num_rows} rows"));
        }
        let words = num_rows.div_ceil(64).max(1);
        let mut table = HashMap::new();
        for k in 1..=depth {
            // The k-subsets in lexicographic order (`std::prev_permutation` of 1…10…0).
            let mut idx: Vec<usize> = (0..k).collect();
            loop {
                let mut sig = vec![0u64; words];
                let mut p = 1.0;
                for &e in &idx {
                    for &r in &columns[e] {
                        sig[r as usize / 64] ^= 1 << (r % 64);
                    }
                    if e < rates.len() {
                        p *= rates[e];
                    }
                }
                table.entry(sig).or_insert_with(|| (idx.iter().map(|&e| e as u32).collect(), p));
                // Next subset.
                let mut i = k;
                while i > 0 && idx[i - 1] == n - k + i - 1 {
                    i -= 1;
                }
                if i == 0 {
                    break;
                }
                idx[i - 1] += 1;
                for j in i..k {
                    idx[j] = idx[j - 1] + 1;
                }
            }
        }
        Ok(LookupTable { num_rows, num_faults: n, table })
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_faults(&self) -> usize {
        self.num_faults
    }

    /// Syndromes entered.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// The faults (ascending) entered for the rows `fired`, and their probability.
    pub fn lookup(&self, fired: &[u32]) -> Option<(&[u32], f64)> {
        let mut sig = vec![0u64; self.num_rows.div_ceil(64).max(1)];
        for &r in fired {
            sig[r as usize / 64] ^= 1 << (r % 64);
        }
        self.table.get(&sig).map(|(e, p)| (e.as_slice(), *p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute force: the first set (by size, then lexicographic) with the syndrome.
    fn brute(num_rows: usize, columns: &[Vec<u32>], depth: usize, fired: &[u32]) -> Option<Vec<u32>> {
        let n = columns.len();
        let target: std::collections::BTreeSet<u32> = fired.iter().copied().collect();
        for k in 1..=depth {
            let mut sets: Vec<Vec<u32>> = Vec::new();
            for mask in 0u32..(1 << n) {
                if mask.count_ones() as usize == k {
                    sets.push((0..n as u32).filter(|i| mask >> i & 1 == 1).collect());
                }
            }
            sets.sort();
            for s in sets {
                let mut sy = std::collections::BTreeSet::new();
                for &e in &s {
                    for &r in &columns[e as usize] {
                        if !sy.remove(&r) {
                            sy.insert(r);
                        }
                    }
                }
                if sy == target {
                    return Some(s);
                }
            }
        }
        let _ = num_rows;
        None
    }

    #[test]
    fn matches_brute_force() {
        let mut state = 12345u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..40 {
            let (m, n) = (2 + (next() % 5) as usize, 3 + (next() % 7) as usize);
            let columns: Vec<Vec<u32>> = (0..n).map(|_| (0..m as u32).filter(|_| next() % 3 == 0).collect()).collect();
            for depth in 1..=n.min(3) {
                let t = LookupTable::new(m, &columns, depth, &[]).unwrap();
                for s in 0u32..(1 << m) {
                    let fired: Vec<u32> = (0..m as u32).filter(|i| s >> i & 1 == 1).collect();
                    let want = if fired.is_empty() { None } else { brute(m, &columns, depth, &fired) };
                    let got = if fired.is_empty() { None } else { t.lookup(&fired).map(|(e, _)| e.to_vec()) };
                    // (An empty syndrome may also be entered, by a set whose faults cancel.)
                    assert_eq!(got, want, "{columns:?} depth {depth} syndrome {fired:?}");
                }
            }
        }
    }

    #[test]
    fn checks_its_arguments() {
        let cols = vec![vec![0], vec![0, 1]];
        assert!(LookupTable::new(2, &cols, 0, &[]).is_err());
        assert!(LookupTable::new(2, &cols, 3, &[]).is_err());
        assert!(LookupTable::new(2, &cols, 1, &[0.1]).is_err());
        assert!(LookupTable::new(2, &cols, 1, &[0.1, 1.5]).is_err());
        let t = LookupTable::new(2, &cols, 2, &[0.1, 0.2]).unwrap();
        assert_eq!(t.lookup(&[1]), Some((&[0u32, 1][..], 0.1 * 0.2)));
    }
}
