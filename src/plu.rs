//! Dense GF(2) elimination, column by column, as `ldpc`'s `gf2dense::PluDecomposition`.
//!
//! WHY THIS EXISTS
//! ---------------
//! Localized statistics decoding (`lsd`) grows clusters one fault at a time and, after each
//! step, asks whether a cluster's syndrome is in the image of its columns; when it is, the
//! cluster is solved by forward and back substitution on the factors built so far. Which of a
//! cluster's columns become pivots decides which of its many solutions is returned, so to give
//! `ldpc`'s answers this is `ldpc`'s elimination step for step, not merely an elimination:
//!
//! - each incoming column has every earlier row swap and elimination applied to it in order,
//!   and its pivot is its first nonzero at or below the rank;
//! - columns are added to the matrix as the cluster grows (`add_column`) and eliminated from
//!   where the last check stopped (`rref_with_y_image_check`), which stops as soon as the
//!   syndrome is in the image, leaving later columns uneliminated;
//! - its quirks are kept: an empty column is not added at all, and a check with no new columns
//!   answers from the syndrome vector as it stood after the last elimination (rows that vector
//!   never covered, where ldpc reads out of bounds, are read from the syndrome).

/// The factorisation of a matrix held column by column (`csc[j]` lists column `j`'s rows).
#[derive(Clone, Debug, Default)]
pub(crate) struct Plu {
    csc: Vec<Vec<u32>>,
    y_check: Vec<u8>,
    l: Vec<Vec<usize>>,
    u: Vec<Vec<usize>>,
    pub(crate) matrix_rank: usize,
    pub(crate) cols_eliminated: usize,
    pub(crate) row_count: usize,
    pub(crate) col_count: usize,
    rows: Vec<usize>,
    swap_rows: Vec<usize>,
    elimination_rows: Vec<Vec<usize>>,
    pub(crate) pivot_cols: Vec<usize>,
    pub(crate) not_pivot_cols: Vec<usize>,
    lu_constructed: bool,
}

impl Plu {
    pub(crate) fn new(row_count: usize, col_count: usize, csc: Vec<Vec<u32>>) -> Plu {
        Plu { csc, row_count, col_count, ..Plu::default() }
    }

    pub(crate) fn reset(&mut self) {
        self.matrix_rank = 0;
        self.cols_eliminated = 0;
        self.rows.clear();
        self.swap_rows.clear();
        self.pivot_cols.clear();
        self.not_pivot_cols.clear();
        self.y_check.clear();
        self.l.clear();
        self.elimination_rows.clear();
        self.u.clear();
        self.lu_constructed = false;
    }

    /// The full factorisation, stopping once the rank is as large as it can be.
    #[cfg(test)]
    pub(crate) fn rref(&mut self) {
        self.reset();
        self.rows.extend(0..self.row_count);
        let max_rank = self.row_count.min(self.col_count);
        self.l.resize(self.row_count, Vec::new());
        for col in 0..self.col_count {
            self.eliminate_column(col);
            if self.matrix_rank == max_rank {
                break;
            }
        }
        self.lu_constructed = true;
    }

    /// Eliminate column `col` with every earlier operation; whether it is a pivot.
    pub(crate) fn eliminate_column(&mut self, col: usize) -> bool {
        let mut rr = vec![0u8; self.row_count];
        self.cols_eliminated = col + 1;
        for &r in &self.csc[col] {
            rr[r as usize] = 1;
        }
        for i in 0..self.matrix_rank {
            rr.swap(i, self.swap_rows[i]);
            if rr[i] == 1 {
                for &r in &self.elimination_rows[i] {
                    rr[r] ^= 1;
                }
            }
        }
        let rank = self.matrix_rank;
        let Some(p) = (rank..self.row_count).find(|&i| rr[i] == 1) else {
            self.not_pivot_cols.push(col);
            return false;
        };
        self.swap_rows.push(p);
        self.pivot_cols.push(col);
        rr.swap(rank, p);
        self.rows.swap(rank, p);
        self.elimination_rows.push(Vec::new());
        self.l.swap(rank, p);
        self.l[rank].push(rank);
        for i in rank + 1..self.row_count {
            if rr[i] == 1 {
                self.elimination_rows[rank].push(i);
                self.l[i].push(rank);
            }
        }
        self.u.push(Vec::new());
        for i in 0..=rank {
            if rr[i] == 1 {
                self.u[i].push(col);
            }
        }
        self.matrix_rank += 1;
        true
    }

    /// Solve `A x = y` with the factors built so far: forward substitution through L, then back
    /// substitution through U onto the pivot columns. `y` has one entry per row.
    pub(crate) fn lu_solve(&self, y: &[u8]) -> Vec<u8> {
        assert_eq!(y.len(), self.row_count, "one syndrome bit per row");
        assert!(self.lu_constructed, "factorise before solving");
        let mut x = vec![0u8; self.col_count];
        let mut b = vec![0u8; self.matrix_rank];
        for row in 0..self.matrix_rank {
            let sum = self.l[row].iter().fold(0u8, |s, &c| s ^ b[c]);
            b[row] = sum ^ y[self.rows[row]];
        }
        for row in (0..self.matrix_rank).rev() {
            let sum = self.u[row].iter().fold(0u8, |s, &c| s ^ x[c]);
            x[self.pivot_cols[row]] = sum ^ b[row];
        }
        x
    }

    /// Eliminate from column `start` on until `y` is in the image of the eliminated columns;
    /// whether it is. Starting from column 0 starts over.
    pub(crate) fn rref_with_y_image_check(&mut self, y: &[u8], start: usize) -> bool {
        if start == self.col_count {
            // ldpc reads its check vector as the last elimination left it; rows it never
            // covered (no column eliminated yet, where ldpc reads past its end) come from `y`.
            return (self.matrix_rank..self.row_count).all(|i| self.y_check.get(i).copied().unwrap_or(y[i]) == 0);
        }
        if start == 0 {
            self.reset();
            self.y_check = y.to_vec();
            self.rows.extend(0..self.row_count);
        }
        let previous = self.y_check.len();
        if previous < self.row_count {
            self.y_check.resize(self.row_count, 0);
            self.y_check[previous..self.row_count].copy_from_slice(&y[previous..self.row_count]);
        }
        if y[..self.row_count].iter().all(|&b| b == 0) {
            self.lu_constructed = true;
            return true;
        }
        let max_rank = self.row_count.min(self.col_count);
        if self.l.len() != self.row_count {
            self.l.resize(self.row_count, Vec::new());
        }
        let mut in_image = false;
        for col in start..self.col_count {
            if self.matrix_rank == max_rank {
                in_image = true;
                break;
            }
            if self.eliminate_column(col) {
                let r = self.matrix_rank - 1;
                self.y_check.swap(r, self.swap_rows[r]);
                if self.y_check[r] == 1 {
                    for &row in &self.elimination_rows[r] {
                        self.y_check[row] ^= 1;
                    }
                }
                in_image = (self.matrix_rank..self.row_count).all(|i| self.y_check[i] == 0);
            }
            if in_image {
                break;
            }
        }
        self.lu_constructed = true;
        in_image
    }

    /// Append a column not yet eliminated, growing the row count to cover it. An empty column
    /// is not added (as in ldpc).
    pub(crate) fn add_column(&mut self, col: &[u32]) {
        let Some(&max) = col.iter().max() else { return };
        self.csc.push(col.to_vec());
        let max = max as usize;
        if max >= self.row_count {
            self.rows.extend(self.row_count..=max);
            self.row_count = max + 1;
        }
        self.col_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_code::Xorshift;

    fn random(m: usize, n: usize, rng: &mut Xorshift) -> Vec<Vec<u32>> {
        (0..n).map(|_| (0..m as u32).filter(|_| rng.next_f64() < 0.3).collect()).collect()
    }

    fn apply(csc: &[Vec<u32>], x: &[u8], m: usize) -> Vec<u8> {
        let mut y = vec![0u8; m];
        for (j, col) in csc.iter().enumerate() {
            if x[j] == 1 {
                for &r in col {
                    y[r as usize] ^= 1;
                }
            }
        }
        y
    }

    fn bits(mask: u32, n: usize) -> Vec<u8> {
        (0..n).map(|j| (mask >> j & 1) as u8).collect()
    }

    /// lu_solve after a full rref solves every syndrome in the image, using pivot columns only.
    #[test]
    fn full_rref_solves_the_image() {
        let mut rng = Xorshift::new(3);
        for _ in 0..300 {
            let (m, n) = (1 + (rng.next_u64() % 12) as usize, 1 + (rng.next_u64() % 16) as usize);
            let csc = random(m, n, &mut rng);
            let x: Vec<u8> = (0..n).map(|_| u8::from(rng.next_f64() < 0.4)).collect();
            let y = apply(&csc, &x, m);
            let mut p = Plu::new(m, n, csc.clone());
            p.rref();
            let s = p.lu_solve(&y);
            assert_eq!(apply(&csc, &s, m), y);
            assert!(s.iter().enumerate().all(|(j, &b)| b == 0 || p.pivot_cols.contains(&j)));
        }
    }

    /// The image check agrees with brute force, and the solution it leaves explains the syndrome.
    #[test]
    fn image_check_is_exact() {
        let mut rng = Xorshift::new(4);
        for _ in 0..300 {
            let (m, n) = (1 + (rng.next_u64() % 8) as usize, 1 + (rng.next_u64() % 8) as usize);
            let csc = random(m, n, &mut rng);
            let y: Vec<u8> = (0..m).map(|_| u8::from(rng.next_f64() < 0.5)).collect();
            let in_image = (0u32..1 << n).any(|mask| apply(&csc, &bits(mask, n), m) == y);
            let mut p = Plu::new(m, n, csc.clone());
            assert_eq!(p.rref_with_y_image_check(&y, 0), in_image, "{csc:?} {y:?}");
            if in_image {
                assert_eq!(apply(&csc, &p.lu_solve(&y), m), y);
            }
        }
    }

    /// Columns added one at a time, the check resumed from where it stopped each time: a yes is
    /// always right and its solution explains the syndrome. (A no can be wrong, as in ldpc: a
    /// resumed check that eliminates only a non-pivot column answers no even if the syndrome was
    /// already in the image. LSD keeps that behaviour, so this test does not forbid it.)
    #[test]
    fn the_check_resumes() {
        let mut rng = Xorshift::new(5);
        for _ in 0..300 {
            let (m, n) = (1 + (rng.next_u64() % 8) as usize, 1 + (rng.next_u64() % 10) as usize);
            let csc: Vec<Vec<u32>> = random(m, n, &mut rng).into_iter().filter(|c| !c.is_empty()).collect();
            if csc.is_empty() {
                continue;
            }
            let rows = 1 + csc.iter().flatten().copied().max().unwrap() as usize;
            let y: Vec<u8> = (0..rows).map(|_| u8::from(rng.next_f64() < 0.5)).collect();
            let mut whole = Plu::new(rows, csc.len(), csc.clone());
            let want = whole.rref_with_y_image_check(&y, 0);
            let mut grown = Plu::new(rows, 0, Vec::new());
            let mut got = false;
            for col in &csc {
                grown.add_column(col);
                let start = grown.cols_eliminated;
                got = grown.rref_with_y_image_check(&y, start);
            }
            if got {
                assert!(want);
                assert_eq!(apply(&csc, &grown.lu_solve(&y), rows), y);
            }
        }
    }
}
