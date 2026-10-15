//! Ambiguity Clustering (Wolanski and Barber, arXiv 2406.14527): a decoder for any decoding
//! problem given as parity checks on independent faults, built for qLDPC codes.
//!
//! WHY THIS EXISTS
//! ---------------
//! BP+OSD's Gaussian elimination is global: it reduces the whole matrix, and its search over
//! the non-pivot columns multiplies the ambiguities of separate parts of the syndrome. AC
//! eliminates only as far as the syndrome needs, and splits what it eliminated into blocks
//! (clusters) that are searched separately, so ambiguities add instead of multiply.
//!
//! After BP (whose answer is taken when it explains the syndrome):
//! 1. Initial solution. While some row with a 1 in the (transformed) syndrome is not a pivot
//!    row, pivot at the adjacent row-column pair whose column BP thinks likeliest. Each pivot
//!    is a 1×1 block; the syndrome now has its 1s on pivot rows only, so the pivot columns
//!    of the 1s explain it.
//! 2. Cluster formation. Add K more columns, each time the likeliest column not yet in the
//!    blocks that touches a row some pivot involved: a column with a 1 outside the blocks'
//!    rows is pivoted there and seeds a block; one inside them joins the blocks it touches,
//!    merging them.
//! 3. Cluster analysis. Faults outside the blocks are taken not to have happened, so each
//!    block is its own decoding problem. A block whose logical effect is a function of its
//!    syndrome (every row of L on its columns a combination of its rows) is unambiguous and
//!    read off; an ambiguous one is searched over its non-pivot columns, every choice of at
//!    most two, and each observable set to whichever value carries more probability.
//!    The blocks' effects add.
//!
//! Row operations only ever add an involved row (a pivot row, or one a pivot row was added
//! to) to a row that thereby becomes involved, so the involved rows span what the original
//! rows of the same indices span: a column touches an involved row now exactly when it did
//! in the original matrix, which is how stage 2 keeps its candidates in a heap. Columns in the
//! blocks stay zero outside the blocks' rows (a seed's row is zero on them), so later pivots
//! never disturb a block.

use crate::bp::{Bp, BpWork, Method, Schedule, SerialState};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const NONE: u32 = u32::MAX;

/// What one decode did, for the caller's statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AcOutcome {
    /// BP explained the syndrome by itself.
    pub converged: bool,
    /// No correction explains the syndrome (a row of it is a combination of no columns).
    pub failed: bool,
    pub blocks: usize,
    pub ambiguous_blocks: usize,
    /// The solutions searched in ambiguous blocks.
    pub solutions: usize,
}

pub struct AmbiguityClustering {
    pub bp: Bp,
    pub method: Method,
    pub max_iter: usize,
    pub schedule: Schedule,
    /// K: the columns stage 2 adds.
    pub extra_columns: usize,
    cols: Vec<Vec<u32>>,
    rows: Vec<Vec<u32>>,
    obs: Vec<u64>,
    /// ln p and ln(1 − p) per column.
    lp: Vec<f64>,
    lq: Vec<f64>,
}

/// One decode's state; one per thread.
pub struct AcWork {
    pub bp: BpWork,
    pub serial: SerialState,
    /// The likeliest solution found in each block (the initial solution in unambiguous ones):
    /// one byte per column.
    pub correction: Vec<u8>,
    /// The logical effect: per block the more probable value of each observable, summed.
    pub observables: u64,
    words: usize,
    /// Rows changed by a pivot, dense; the others are still the matrix's own.
    dense: Vec<u64>,
    live: Vec<bool>,
    live_rows: Vec<u32>,
    synd: Vec<u8>,
    pivot_col: Vec<u32>,
    pivot_row: Vec<u32>,
    in_c: Vec<bool>,
    c_row: Vec<bool>,
    involved: Vec<bool>,
    involved_rows: Vec<u32>,
    block: Vec<u32>,
    c_rows: Vec<u32>,
    extra: Vec<u32>,
    queued: Vec<bool>,
    queued_cols: Vec<u32>,
    best: Vec<u32>,
    pending: Vec<u32>,
    scratch: Vec<u32>,
    corrected: Vec<u32>,
}

/// A column by BP's belief that it happened: lower LLR first, then lower index.
#[derive(Clone, Copy, PartialEq)]
struct Likely(f64, u32);

impl Eq for Likely {}

impl Ord for Likely {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0).then(other.1.cmp(&self.1))
    }
}

impl PartialOrd for Likely {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let up = parent[parent[x as usize] as usize];
        parent[x as usize] = up;
        x = up;
    }
    x
}

impl AmbiguityClustering {
    /// `columns[j]` lists the checks fault `j` flips, `observables[j]` the observables it
    /// flips (as bits), `priors[j]` its probability; `extra_columns` is K.
    pub fn new(num_checks: usize, columns: Vec<Vec<u32>>, observables: Vec<u64>, priors: &[f64], method: Method, max_iter: usize, extra_columns: usize) -> Result<AmbiguityClustering, String> {
        let cols: Vec<Vec<u32>> = columns
            .into_iter()
            .map(|mut c| {
                c.sort_unstable();
                c.dedup();
                c
            })
            .collect();
        if observables.len() != cols.len() || priors.len() != cols.len() {
            return Err("one observable mask and one prior per column".into());
        }
        let bp = Bp::new(num_checks, &cols, priors)?;
        let mut rows = vec![Vec::new(); num_checks];
        for (j, c) in cols.iter().enumerate() {
            for &r in c {
                rows[r as usize].push(j as u32);
            }
        }
        let lp = priors.iter().map(|&p| p.ln()).collect();
        let lq = priors.iter().map(|&p| (-p).ln_1p()).collect();
        Ok(AmbiguityClustering { bp, method, max_iter, schedule: Schedule::Parallel, extra_columns, cols, rows, obs: observables, lp, lq })
    }

    pub fn num_columns(&self) -> usize {
        self.cols.len()
    }

    pub fn work(&self) -> AcWork {
        let (m, n) = (self.rows.len(), self.cols.len());
        let words = n.div_ceil(64).max(1);
        AcWork {
            bp: self.bp.work(),
            serial: SerialState::new(n),
            correction: vec![0; n],
            observables: 0,
            words,
            dense: vec![0; m * words],
            live: vec![false; m],
            live_rows: Vec::new(),
            synd: vec![0; m],
            pivot_col: vec![NONE; m],
            pivot_row: vec![NONE; n],
            in_c: vec![false; n],
            c_row: vec![false; m],
            involved: vec![false; m],
            involved_rows: Vec::new(),
            block: (0..m as u32).collect(),
            c_rows: Vec::new(),
            extra: Vec::new(),
            queued: vec![false; n],
            queued_cols: Vec::new(),
            best: vec![NONE; m],
            pending: Vec::new(),
            scratch: Vec::new(),
            corrected: Vec::new(),
        }
    }

    /// Decode a syndrome (one byte per check): the logical effect is left in
    /// `w.observables`, a correction in `w.correction`.
    pub fn decode(&self, syndrome: &[u8], w: &mut AcWork) -> AcOutcome {
        for &j in &w.corrected {
            w.correction[j as usize] = 0;
        }
        w.corrected.clear();
        w.observables = 0;
        let out = self.bp.decode_scheduled(syndrome, self.method, self.max_iter, self.schedule, &mut w.serial, &mut w.bp);
        if out.converged {
            for (j, &h) in w.bp.hard.iter().enumerate() {
                if h != 0 {
                    w.correction[j] = 1;
                    w.corrected.push(j as u32);
                    w.observables ^= self.obs[j];
                }
            }
            return AcOutcome { converged: true, ..AcOutcome::default() };
        }
        let mut outcome = AcOutcome::default();
        for (s, &x) in w.synd.iter_mut().zip(syndrome) {
            *s = u8::from(x != 0);
        }
        if self.initial_solution(w) {
            self.form_clusters(w);
            self.analyse(w, &mut outcome);
        } else {
            outcome.failed = true;
        }
        self.reset(w);
        outcome
    }

    fn llr(&self, w: &AcWork, j: u32) -> f64 {
        w.bp.llr[j as usize]
    }

    fn materialize(&self, w: &mut AcWork, r: u32) {
        let r = r as usize;
        if w.live[r] {
            return;
        }
        w.live[r] = true;
        w.live_rows.push(r as u32);
        let row = &mut w.dense[r * w.words..(r + 1) * w.words];
        row.fill(0);
        for &j in &self.rows[r] {
            row[j as usize / 64] |= 1 << (j % 64);
        }
    }

    /// The rows with a 1 in column `j` now, ascending.
    fn column(&self, w: &AcWork, j: u32, out: &mut Vec<u32>) {
        out.clear();
        for &r in &self.cols[j as usize] {
            if !w.live[r as usize] {
                out.push(r);
            }
        }
        let (word, bit) = (j as usize / 64, j % 64);
        for &r in &w.live_rows {
            if (w.dense[r as usize * w.words + word] >> bit) & 1 == 1 {
                out.push(r);
            }
        }
        out.sort_unstable();
    }

    /// Visit row `r`'s columns now.
    fn for_row(&self, w: &AcWork, r: u32, mut f: impl FnMut(u32)) {
        let r = r as usize;
        if !w.live[r] {
            for &j in &self.rows[r] {
                f(j);
            }
            return;
        }
        for (k, &word) in w.dense[r * w.words..(r + 1) * w.words].iter().enumerate() {
            let mut x = word;
            while x != 0 {
                f((k * 64) as u32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }

    fn involve(&self, w: &mut AcWork, r: u32) {
        if !w.involved[r as usize] {
            w.involved[r as usize] = true;
            w.involved_rows.push(r);
        }
    }

    /// Pivot at row `i`, column `j`: `i` added to every other row with a 1 in `j`.
    fn pivot(&self, w: &mut AcWork, i: u32, j: u32) {
        let mut rows = std::mem::take(&mut w.scratch);
        self.column(w, j, &mut rows);
        self.materialize(w, i);
        let words = w.words;
        for &l in &rows {
            if l == i {
                continue;
            }
            self.materialize(w, l);
            let (a, b) = (i as usize * words, l as usize * words);
            for k in 0..words {
                let x = w.dense[a + k];
                w.dense[b + k] ^= x;
            }
            w.synd[l as usize] ^= w.synd[i as usize];
            w.best[l as usize] = NONE;
            self.involve(w, l);
            if w.synd[l as usize] == 1 && w.pivot_col[l as usize] == NONE {
                w.pending.push(l);
            }
        }
        w.scratch = rows;
        w.pivot_col[i as usize] = j;
        w.pivot_row[j as usize] = i;
        w.in_c[j as usize] = true;
        w.c_row[i as usize] = true;
        w.c_rows.push(i);
        self.involve(w, i);
    }

    /// Stage 1: pivot until every row with a 1 in the syndrome is a pivot row. False when a
    /// row with a 1 has no columns left (no correction explains the syndrome).
    fn initial_solution(&self, w: &mut AcWork) -> bool {
        w.pending.clear();
        for r in 0..w.synd.len() {
            if w.synd[r] == 1 {
                w.pending.push(r as u32);
            }
        }
        loop {
            // The rows still to reduce, each with its likeliest column.
            let mut pending = std::mem::take(&mut w.pending);
            pending.sort_unstable();
            pending.dedup();
            pending.retain(|&r| w.synd[r as usize] == 1 && w.pivot_col[r as usize] == NONE);
            let mut choice: Option<(Likely, u32)> = None;
            for &r in &pending {
                if w.best[r as usize] == NONE {
                    let mut best: Option<Likely> = None;
                    self.for_row(w, r, |j| {
                        let c = Likely(self.llr(w, j), j);
                        if best.is_none_or(|b| c > b) {
                            best = Some(c);
                        }
                    });
                    match best {
                        Some(b) => w.best[r as usize] = b.1,
                        None => {
                            w.pending = pending;
                            return false;
                        }
                    }
                }
                let c = Likely(self.llr(w, w.best[r as usize]), w.best[r as usize]);
                if choice.is_none_or(|(b, _)| c > b) {
                    choice = Some((c, r));
                }
            }
            w.pending = pending;
            let Some((Likely(_, j), i)) = choice else { return true };
            self.pivot(w, i, j);
        }
    }

    fn enqueue_row(&self, w: &mut AcWork, heap: &mut BinaryHeap<Likely>, r: u32) {
        for &j in &self.rows[r as usize] {
            if !w.queued[j as usize] {
                w.queued[j as usize] = true;
                w.queued_cols.push(j);
                heap.push(Likely(self.llr(w, j), j));
            }
        }
    }

    /// Stage 2: add K columns to the blocks.
    fn form_clusters(&self, w: &mut AcWork) {
        let mut heap = BinaryHeap::new();
        let mut seen = 0;
        let mut rows = Vec::new();
        for _ in 0..self.extra_columns {
            while seen < w.involved_rows.len() {
                let r = w.involved_rows[seen];
                seen += 1;
                self.enqueue_row(w, &mut heap, r);
            }
            let j = loop {
                match heap.pop() {
                    Some(Likely(_, j)) if !w.in_c[j as usize] => break Some(j),
                    Some(_) => continue,
                    None => break None,
                }
            };
            let Some(j) = j else { return };
            self.column(w, j, &mut rows);
            if let Some(&i) = rows.iter().find(|&&r| !w.c_row[r as usize]) {
                // Seed a block.
                self.pivot(w, i, j);
            } else {
                // Grow the blocks it touches, merging them.
                w.in_c[j as usize] = true;
                w.extra.push(j);
                if let Some((&first, rest)) = rows.split_first() {
                    let root = find(&mut w.block, first);
                    for &r in rest {
                        let other = find(&mut w.block, r);
                        if other != root {
                            w.block[other as usize] = root;
                        }
                    }
                }
            }
        }
    }

    fn set(&self, w: &mut AcWork, j: u32) {
        w.correction[j as usize] = 1;
        w.corrected.push(j);
    }

    /// Stage 3: each block's logical effect, and a solution.
    fn analyse(&self, w: &mut AcWork, outcome: &mut AcOutcome) {
        // Blocks: their rows, and their non-pivot columns with each one's rows.
        let c_rows = std::mem::take(&mut w.c_rows);
        let mut members: Vec<(u32, u32)> = c_rows.iter().map(|&r| (find(&mut w.block, r), r)).collect();
        members.sort_unstable();
        let mut extras: Vec<(u32, u32, Vec<u32>)> = Vec::with_capacity(w.extra.len());
        let mut rows = Vec::new();
        for k in 0..w.extra.len() {
            let j = w.extra[k];
            self.column(w, j, &mut rows);
            let root = rows.first().map_or(NONE, |&r| find(&mut w.block, r));
            extras.push((root, j, rows.clone()));
        }
        extras.sort_unstable_by_key(|e| (e.0, e.1));
        let mut e = 0;
        let mut start = 0;
        while start < members.len() {
            let root = members[start].0;
            let mut end = start;
            while end < members.len() && members[end].0 == root {
                end += 1;
            }
            while e < extras.len() && extras[e].0 < root {
                e += 1;
            }
            let mut e_end = e;
            while e_end < extras.len() && extras[e_end].0 == root {
                e_end += 1;
            }
            let block_rows: Vec<u32> = members[start..end].iter().map(|m| m.1).collect();
            self.analyse_block(w, &block_rows, &extras[e..e_end], outcome);
            outcome.blocks += 1;
            start = end;
            e = e_end;
        }
        w.c_rows = c_rows;
    }

    fn analyse_block(&self, w: &mut AcWork, block_rows: &[u32], extras: &[(u32, u32, Vec<u32>)], outcome: &mut AcOutcome) {
        let pc = |r: u32| w.pivot_col[r as usize];
        // The initial solution: the pivot columns of the syndrome's 1s.
        let mut base_obs = 0u64;
        for &r in block_rows {
            if w.synd[r as usize] == 1 {
                base_obs ^= self.obs[pc(r) as usize];
            }
        }
        // Unambiguous when L on each non-pivot column is the combination of its rows' pivot
        // columns' L.
        let ambiguous = extras.iter().any(|(_, j, rs)| rs.iter().fold(self.obs[*j as usize], |a, &r| a ^ self.obs[pc(r) as usize]) != 0);
        if !ambiguous {
            let chosen: Vec<u32> = block_rows.iter().filter(|&&r| w.synd[r as usize] == 1).map(|&r| pc(r)).collect();
            w.observables ^= base_obs;
            for j in chosen {
                self.set(w, j);
            }
            return;
        }
        outcome.ambiguous_blocks += 1;
        // Each solution's probability relative to the initial one: choosing a non-pivot column
        // flips its rows' pivot columns.
        let flip = |r: u32| {
            let j = pc(r) as usize;
            let d = self.lp[j] - self.lq[j];
            if w.synd[r as usize] == 1 { -d } else { d }
        };
        let single: Vec<(f64, u64)> = extras
            .iter()
            .map(|(_, j, rs)| {
                let j = *j as usize;
                let delta = (self.lp[j] - self.lq[j]) + rs.iter().map(|&r| flip(r)).sum::<f64>();
                let o = rs.iter().fold(self.obs[j], |a, &r| a ^ self.obs[pc(r) as usize]);
                (delta, o)
            })
            .collect();
        let nobs = 64 - single.iter().fold(base_obs, |a, s| a | s.1 | base_obs).leading_zeros() as usize;
        let mut flipped = vec![0.0f64; nobs];
        let mut total = 0.0f64;
        let mut best = (0.0f64, NONE, NONE);
        let mut add = |delta: f64, o: u64, a: u32, b: u32, flipped: &mut [f64], total: &mut f64| {
            let p = delta.exp();
            *total += p;
            let o = o ^ base_obs;
            for (k, f) in flipped.iter_mut().enumerate() {
                if (o >> k) & 1 == 1 {
                    *f += p;
                }
            }
            if delta > best.0 {
                best = (delta, a, b);
            }
        };
        add(0.0, 0, NONE, NONE, &mut flipped, &mut total);
        let mut count = 1;
        for (a, &(da, oa)) in single.iter().enumerate() {
            add(da, oa, a as u32, NONE, &mut flipped, &mut total);
            count += 1;
            for (b, &(db, ob)) in single.iter().enumerate().skip(a + 1) {
                // Rows both columns touch are flipped twice: back to the initial solution.
                let (ra, rb) = (&extras[a].2, &extras[b].2);
                let (mut x, mut y, mut both) = (0, 0, 0.0);
                while x < ra.len() && y < rb.len() {
                    match ra[x].cmp(&rb[y]) {
                        Ordering::Less => x += 1,
                        Ordering::Greater => y += 1,
                        Ordering::Equal => {
                            both += flip(ra[x]);
                            x += 1;
                            y += 1;
                        }
                    }
                }
                add(da + db - 2.0 * both, oa ^ ob, a as u32, b as u32, &mut flipped, &mut total);
                count += 1;
            }
        }
        outcome.solutions += count;
        let mut effect = 0u64;
        for (k, &f) in flipped.iter().enumerate() {
            if f > total - f {
                effect |= 1 << k;
            }
        }
        // The likeliest solution found, as the block's correction.
        let mut chosen: Vec<u32> = Vec::new();
        let mut row_flip: Vec<u32> = Vec::new();
        for c in [best.1, best.2] {
            if c != NONE {
                let (_, j, rs) = &extras[c as usize];
                chosen.push(*j);
                row_flip.extend(rs);
            }
        }
        row_flip.sort_unstable();
        for &r in block_rows {
            let n = row_flip.partition_point(|&x| x < r);
            let times = row_flip[n..].iter().take_while(|&&x| x == r).count();
            if (w.synd[r as usize] as usize + times) % 2 == 1 {
                chosen.push(pc(r));
            }
        }
        w.observables ^= effect;
        for j in chosen {
            self.set(w, j);
        }
    }

    fn reset(&self, w: &mut AcWork) {
        for &r in &w.live_rows {
            w.live[r as usize] = false;
        }
        w.live_rows.clear();
        for &r in &w.c_rows {
            w.c_row[r as usize] = false;
            let j = w.pivot_col[r as usize];
            w.pivot_col[r as usize] = NONE;
            w.pivot_row[j as usize] = NONE;
            w.in_c[j as usize] = false;
        }
        w.c_rows.clear();
        for &j in &w.extra {
            w.in_c[j as usize] = false;
        }
        w.extra.clear();
        for &r in &w.involved_rows {
            w.involved[r as usize] = false;
            w.block[r as usize] = r;
            w.best[r as usize] = NONE;
        }
        w.involved_rows.clear();
        for &j in &w.queued_cols {
            w.queued[j as usize] = false;
        }
        w.queued_cols.clear();
        w.pending.clear();
        w.synd.fill(0);
        w.best.fill(NONE);
    }
}

#[cfg(test)]
impl AmbiguityClustering {
    /// After stage 2: every block column is zero outside the blocks' rows, each block column
    /// touches one block, and every pivot column is reduced.
    fn check_blocks(&self, w: &mut AcWork) {
        let mut rows = Vec::new();
        for j in 0..self.cols.len() as u32 {
            if !w.in_c[j as usize] {
                continue;
            }
            self.column(w, j, &mut rows);
            assert!(rows.iter().all(|&r| w.c_row[r as usize]), "column {j} leaves the blocks");
            let roots: Vec<u32> = rows.iter().map(|&r| find(&mut w.block, r)).collect();
            assert!(roots.windows(2).all(|p| p[0] == p[1]), "column {j} spans blocks");
            if w.pivot_row[j as usize] != NONE {
                assert_eq!(rows, vec![w.pivot_row[j as usize]], "pivot column {j} is not reduced");
            }
        }
        for r in 0..w.synd.len() {
            if w.synd[r] == 1 {
                assert!(w.c_row[r], "row {r} of the syndrome is outside the blocks");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_code::Xorshift;

    fn decoder(columns: Vec<Vec<u32>>, obs: Vec<u64>, p: f64, m: usize, k: usize) -> AmbiguityClustering {
        let priors = vec![p; columns.len()];
        AmbiguityClustering::new(m, columns, obs, &priors, Method::ProductSum, 9, k).unwrap()
    }

    fn syndrome_of(dec: &AmbiguityClustering, correction: &[u8], m: usize) -> Vec<u8> {
        let mut s = vec![0u8; m];
        for (j, &c) in correction.iter().enumerate() {
            if c != 0 {
                for &r in &dec.cols[j] {
                    s[r as usize] ^= 1;
                }
            }
        }
        s
    }

    #[test]
    fn corrections_explain_syndromes_and_blocks_stay_blocks() {
        // Random sparse codes: each column 2 to 4 checks, observables on a few columns.
        let mut rng = Xorshift::new(7);
        let (mut staged, mut ambiguous) = (0, 0);
        for trial in 0..200 {
            let m = 6 + (rng.next_u64() % 20) as usize;
            let n = m + 4 + (rng.next_u64() % 30) as usize;
            let columns: Vec<Vec<u32>> = (0..n)
                .map(|_| {
                    let w = 2 + (rng.next_u64() % 3) as usize;
                    (0..w).map(|_| (rng.next_u64() % m as u64) as u32).collect()
                })
                .collect();
            let obs: Vec<u64> = (0..n).map(|_| rng.next_u64() & 3).collect();
            let k = (rng.next_u64() % (n as u64 + 1)) as usize;
            let dec = decoder(columns, obs, 0.08, m, k);
            let mut w = dec.work();
            for _ in 0..20 {
                let e: Vec<u8> = (0..n).map(|_| u8::from(rng.next_u64() % 10 == 0)).collect();
                let s = syndrome_of(&dec, &e, m);
                // Stages 1 and 2 by hand, to look at the blocks.
                let bp = dec.bp.decode(&s, dec.method, dec.max_iter, &mut w.bp);
                if !bp.converged {
                    w.synd.copy_from_slice(&s);
                    assert!(dec.initial_solution(&mut w), "trial {trial}: a consistent syndrome failed");
                    dec.form_clusters(&mut w);
                    dec.check_blocks(&mut w);
                    dec.reset(&mut w);
                }
                let out = dec.decode(&s, &mut w);
                assert!(!out.failed);
                staged += usize::from(!out.converged);
                ambiguous += out.ambiguous_blocks;
                assert_eq!(syndrome_of(&dec, &w.correction, m), s, "trial {trial}");
                let from_correction = w.correction.iter().enumerate().filter(|(_, &c)| c != 0).fold(0, |a, (j, _)| a ^ dec.obs[j]);
                if out.ambiguous_blocks == 0 {
                    assert_eq!(w.observables, from_correction, "trial {trial}: an unambiguous effect differs from its correction's");
                }
            }
        }
        assert!(staged > 500 && ambiguous > 100, "AC ran on {staged} shots, {ambiguous} ambiguous blocks");
    }

    #[test]
    fn ring_with_every_column_is_maximum_likelihood() {
        // A ring of n bits (n odd, so no ties), checks between neighbours, the observable on
        // bit 0: with every column in the blocks, the one non-pivot column makes the search
        // exhaustive, so AC is maximum likelihood.
        for n in [5usize, 7, 9] {
            let columns: Vec<Vec<u32>> = (0..n as u32).map(|j| vec![j, (j + 1) % n as u32]).collect();
            let obs: Vec<u64> = (0..n).map(|j| u64::from(j == 0)).collect();
            let p = 0.1;
            let dec = decoder(columns, obs.clone(), p, n, n);
            let mut w = dec.work();
            for bits in 0u32..(1 << n) {
                let e: Vec<u8> = (0..n).map(|j| ((bits >> j) & 1) as u8).collect();
                let s = syndrome_of(&dec, &e, n);
                // Brute force over every error with this syndrome.
                let (mut flip, mut keep) = (0.0, 0.0);
                for f in 0u32..(1 << n) {
                    let x: Vec<u8> = (0..n).map(|j| ((f >> j) & 1) as u8).collect();
                    if syndrome_of(&dec, &x, n) != s {
                        continue;
                    }
                    let k = x.iter().filter(|&&b| b == 1).count() as i32;
                    let prob = p.powi(k) * (1.0 - p).powi(n as i32 - k);
                    if x[0] == 1 { flip += prob } else { keep += prob }
                }
                dec.decode(&s, &mut w);
                assert_eq!(w.observables, u64::from(flip > keep), "n {n} error {bits:b}");
            }
        }
    }
}
