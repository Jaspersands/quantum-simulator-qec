//! Linear algebra over GF(2), on bit-packed rows.
//!
//! Codes that are not matchable need it twice: once to find a code's logical
//! operators (kernels, and what a kernel adds beyond a rowspace), and once per
//! shot in ordered-statistics decoding, which solves a syndrome equation by
//! Gaussian elimination on the columns BP trusts most.

/// A dense GF(2) matrix, rows packed 64 columns to a word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitMatrix {
    pub rows: usize,
    pub cols: usize,
    words: usize,
    data: Vec<u64>,
}

impl BitMatrix {
    pub fn zeros(rows: usize, cols: usize) -> BitMatrix {
        let words = cols.div_ceil(64).max(1);
        BitMatrix {
            rows,
            cols,
            words,
            data: vec![0; rows * words],
        }
    }

    pub fn identity(n: usize) -> BitMatrix {
        let mut m = BitMatrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, true);
        }
        m
    }

    /// From rows given as lists of the columns set in each.
    pub fn from_rows(cols: usize, rows: &[Vec<usize>]) -> BitMatrix {
        let mut m = BitMatrix::zeros(rows.len(), cols);
        for (r, row) in rows.iter().enumerate() {
            for &c in row {
                m.flip(r, c);
            }
        }
        m
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> bool {
        self.data[r * self.words + c / 64] >> (c % 64) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: bool) {
        let w = &mut self.data[r * self.words + c / 64];
        if v {
            *w |= 1 << (c % 64);
        } else {
            *w &= !(1 << (c % 64));
        }
    }

    #[inline]
    pub fn flip(&mut self, r: usize, c: usize) {
        self.data[r * self.words + c / 64] ^= 1 << (c % 64);
    }

    pub fn row(&self, r: usize) -> &[u64] {
        &self.data[r * self.words..(r + 1) * self.words]
    }

    /// The columns set in row `r`.
    pub fn row_ones(&self, r: usize) -> Vec<usize> {
        (0..self.cols).filter(|&c| self.get(r, c)).collect()
    }

    /// Row `dst` ^= row `src`.
    #[inline]
    pub fn xor_row(&mut self, dst: usize, src: usize) {
        if dst == src {
            return;
        }
        let w = self.words;
        let (a, b) = if dst < src {
            let (lo, hi) = self.data.split_at_mut(src * w);
            (&mut lo[dst * w..dst * w + w], &hi[..w])
        } else {
            let (lo, hi) = self.data.split_at_mut(dst * w);
            (&mut hi[..w], &lo[src * w..src * w + w])
        };
        for (x, y) in a.iter_mut().zip(b) {
            *x ^= y;
        }
    }

    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for k in 0..self.words {
            self.data.swap(a * self.words + k, b * self.words + k);
        }
    }

    pub fn transpose(&self) -> BitMatrix {
        let mut t = BitMatrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.get(r, c) {
                    t.set(c, r, true);
                }
            }
        }
        t
    }

    /// Stack `other`'s rows under this one's.
    pub fn stack(&self, other: &BitMatrix) -> BitMatrix {
        assert_eq!(self.cols, other.cols);
        let mut m = self.clone();
        m.rows += other.rows;
        m.data.extend_from_slice(&other.data);
        m
    }

    pub fn mul(&self, other: &BitMatrix) -> BitMatrix {
        assert_eq!(self.cols, other.rows);
        let mut out = BitMatrix::zeros(self.rows, other.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                if self.get(r, k) {
                    let w = other.words;
                    for j in 0..w {
                        out.data[r * out.words + j] ^= other.data[k * w + j];
                    }
                }
            }
        }
        out
    }

    /// This matrix times a vector given as one bit per column.
    pub fn mul_vec(&self, v: &[u8]) -> Vec<u8> {
        assert_eq!(v.len(), self.cols);
        (0..self.rows)
            .map(|r| {
                (0..self.cols)
                    .filter(|&c| v[c] != 0 && self.get(r, c))
                    .count() as u8
                    & 1
            })
            .collect()
    }

    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|&w| w == 0)
    }

    /// Reduced row echelon form in place; returns the pivot columns, in order.
    pub fn row_reduce(&mut self) -> Vec<usize> {
        let mut pivots = Vec::new();
        let mut r = 0;
        for c in 0..self.cols {
            if r == self.rows {
                break;
            }
            let Some(p) = (r..self.rows).find(|&i| self.get(i, c)) else {
                continue;
            };
            self.swap_rows(r, p);
            for i in 0..self.rows {
                if i != r && self.get(i, c) {
                    self.xor_row(i, r);
                }
            }
            pivots.push(c);
            r += 1;
        }
        pivots
    }

    pub fn rank(&self) -> usize {
        self.clone().row_reduce().len()
    }

    /// A basis of the null space {v : M v = 0}, one vector per row.
    pub fn kernel(&self) -> BitMatrix {
        let mut m = self.clone();
        let pivots = m.row_reduce();
        let free: Vec<usize> = (0..self.cols).filter(|c| !pivots.contains(c)).collect();
        let mut k = BitMatrix::zeros(free.len(), self.cols);
        for (i, &f) in free.iter().enumerate() {
            k.set(i, f, true);
            for (r, &p) in pivots.iter().enumerate() {
                if m.get(r, f) {
                    k.set(i, p, true);
                }
            }
        }
        k
    }

    /// The inverse of a square invertible matrix, or None.
    pub fn inverse(&self) -> Option<BitMatrix> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut aug = BitMatrix::zeros(n, 2 * n);
        for r in 0..n {
            for c in 0..n {
                if self.get(r, c) {
                    aug.set(r, c, true);
                }
            }
            aug.set(r, n + r, true);
        }
        let pivots = aug.row_reduce();
        if pivots.len() < n || pivots[n - 1] >= n {
            return None;
        }
        let mut inv = BitMatrix::zeros(n, n);
        for r in 0..n {
            for c in 0..n {
                if aug.get(r, n + c) {
                    inv.set(r, c, true);
                }
            }
        }
        Some(inv)
    }
}

/// Rows of `candidates` that lie outside the rowspace of `base`, chosen
/// greedily so each adds one to the rank: a basis of their span modulo it.
pub fn complement_rows(base: &BitMatrix, candidates: &BitMatrix) -> BitMatrix {
    let mut acc = base.clone();
    let mut rank = acc.rank();
    let mut chosen = Vec::new();
    for r in 0..candidates.rows {
        let mut one = BitMatrix::zeros(1, candidates.cols);
        for c in candidates.row_ones(r) {
            one.set(0, c, true);
        }
        let next = acc.stack(&one);
        let nr = next.rank();
        if nr > rank {
            acc = next;
            rank = nr;
            chosen.push(candidates.row_ones(r));
        }
    }
    BitMatrix::from_rows(candidates.cols, &chosen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_code::Xorshift;

    fn random(rows: usize, cols: usize, rng: &mut Xorshift) -> BitMatrix {
        let mut m = BitMatrix::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                if rng.next_f64() < 0.3 {
                    m.set(r, c, true);
                }
            }
        }
        m
    }

    #[test]
    fn the_kernel_is_annihilated_and_has_the_right_dimension() {
        let mut rng = Xorshift::new(1);
        for (rows, cols) in [(5, 9), (12, 70), (30, 130), (64, 64)] {
            let m = random(rows, cols, &mut rng);
            let k = m.kernel();
            assert_eq!(k.rows, cols - m.rank());
            assert!(m.mul(&k.transpose()).is_zero());
            assert_eq!(k.rank(), k.rows, "the kernel's rows are independent");
        }
    }

    #[test]
    fn an_inverse_is_an_inverse() {
        let mut rng = Xorshift::new(2);
        let mut found = 0;
        while found < 5 {
            let m = random(12, 12, &mut rng);
            if let Some(inv) = m.inverse() {
                assert_eq!(m.mul(&inv), BitMatrix::identity(12));
                found += 1;
            } else {
                assert!(m.rank() < 12);
            }
        }
    }

    #[test]
    fn row_reduction_keeps_the_rowspace() {
        let mut rng = Xorshift::new(3);
        let m = random(20, 100, &mut rng);
        let mut r = m.clone();
        let pivots = r.row_reduce();
        assert_eq!(pivots.len(), m.rank());
        assert_eq!(m.stack(&r).rank(), m.rank());
    }
}
