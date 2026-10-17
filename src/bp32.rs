//! Min-sum belief propagation in single precision, the parallel schedule, written so a GPU can
//! run the same operations in the same order (`gpu/bp.wgsl`) and give the same bits. This CPU
//! executor is the reference the GPU is held to; `bp` (f64, product-sum and min-sum, every
//! schedule) stays the decoder of record.
//!
//! The arithmetic is `bp`'s min-sum step for step — prefix and suffix minima along each check,
//! prefix and suffix sums down each column, a variable set when its LLR is at most zero, a stop
//! as soon as the decision reproduces the syndrome — in f32, with the adaptive scale 1 − 2⁻ⁱ
//! of iteration i tabulated beforehand so both executors read the same value.

use crate::bp::Bp;

/// A parity-check matrix as a Tanner graph in f32: the edge numbering of `Bp`.
#[derive(Clone)]
pub struct Bp32 {
    pub num_checks: usize,
    pub num_vars: usize,
    pub var_start: Vec<u32>,
    pub edge_check: Vec<u32>,
    pub edge_var: Vec<u32>,
    pub chk_start: Vec<u32>,
    pub chk_edges: Vec<u32>,
    pub prior_llr: Vec<f32>,
}

/// How a batch decoded: hard decisions as b8 rows (one row per shot, bit v variable v),
/// one byte per shot set where BP converged, and the iterations each shot took.
pub struct Bp32Outcome {
    pub hard: Vec<u8>,
    pub converged: Vec<u8>,
    pub iterations: Vec<u32>,
}

/// The scale of each iteration 1..=max_iter: `scale` itself, or ldpc's adaptive 1 − 2⁻ⁱ when
/// `scale` is zero, in f32.
pub fn scales(scale: f32, max_iter: usize) -> Vec<f32> {
    (1..=max_iter).map(|it| if scale == 0.0 { 1.0f32 - 0.5f32.powi(it as i32) } else { scale }).collect()
}

impl Bp32 {
    pub fn from_bp(bp: &Bp) -> Bp32 {
        let (var_start, edge_check, edge_var, chk_start, chk_edges) = bp.graph();
        Bp32 {
            num_checks: bp.num_checks,
            num_vars: bp.num_vars,
            var_start: var_start.to_vec(),
            edge_check: edge_check.to_vec(),
            edge_var: edge_var.to_vec(),
            chk_start: chk_start.to_vec(),
            chk_edges: chk_edges.to_vec(),
            prior_llr: bp.prior_llr().iter().map(|&p| p as f32).collect(),
        }
    }

    pub fn num_edges(&self) -> usize {
        self.edge_check.len()
    }

    /// One shot: `syndrome` one byte per check, `scales` per iteration; the hard decision into
    /// `hard` (one byte per variable), (converged, iterations) back. `b2c`, `c2b` and `candidate`
    /// are scratch of the right sizes.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_one(&self, syndrome: &[u8], scales: &[f32], b2c: &mut [f32], c2b: &mut [f32], llr: &mut [f32], hard: &mut [u8], candidate: &mut [u8]) -> (bool, u32) {
        for v in 0..self.num_vars {
            for e in self.var_start[v] as usize..self.var_start[v + 1] as usize {
                b2c[e] = self.prior_llr[v];
            }
        }
        for (it, &scale) in scales.iter().enumerate() {
            // Checks: prefix and suffix minima of |b2c| along the row, the sign the parity of the
            // negative messages and the syndrome, the minimum over no messages f32::MAX.
            for c in 0..self.num_checks {
                let row = &self.chk_edges[self.chk_start[c] as usize..self.chk_start[c + 1] as usize];
                let mut total_sign = u32::from(syndrome[c] != 0);
                let mut temp = f32::MAX;
                for &e in row {
                    let e = e as usize;
                    c2b[e] = temp;
                    if b2c[e] <= 0.0 {
                        total_sign += 1;
                    }
                    temp = temp.min(b2c[e].abs());
                }
                temp = f32::MAX;
                for &e in row.iter().rev() {
                    let e = e as usize;
                    let mut sign = total_sign;
                    if b2c[e] <= 0.0 {
                        sign += 1;
                    }
                    if temp < c2b[e] {
                        c2b[e] = temp;
                    }
                    let s = if sign % 2 == 0 { 1.0f32 } else { -1.0f32 };
                    c2b[e] *= s * scale;
                    temp = temp.min(b2c[e].abs());
                }
            }
            // Variables: posteriors and the decision, each edge primed by the prefix sum.
            for v in 0..self.num_vars {
                let mut temp = self.prior_llr[v];
                for e in self.var_start[v] as usize..self.var_start[v + 1] as usize {
                    b2c[e] = temp;
                    temp += c2b[e];
                }
                llr[v] = temp;
                hard[v] = u8::from(temp <= 0.0);
            }
            candidate.fill(0);
            for (e, &c) in self.edge_check.iter().enumerate() {
                candidate[c as usize] ^= hard[self.edge_var[e] as usize];
            }
            if candidate.iter().zip(syndrome).all(|(&a, &b)| a == u8::from(b != 0)) {
                return (true, it as u32 + 1);
            }
            for v in 0..self.num_vars {
                let mut temp = 0.0f32;
                for e in (self.var_start[v] as usize..self.var_start[v + 1] as usize).rev() {
                    b2c[e] += temp;
                    temp += c2b[e];
                }
            }
        }
        (false, scales.len() as u32)
    }

    /// `shots` syndromes (one byte per check each, back to back) decoded `threads` wide.
    pub fn decode_batch(&self, syndromes: &[u8], shots: usize, scale: f32, max_iter: usize, threads: usize) -> Bp32Outcome {
        let sc = scales(scale, max_iter);
        let stride = self.num_vars.div_ceil(8);
        let nc = self.num_checks;
        let parts = crate::parallel::parallel(shots, threads, |range| {
            let (mut b2c, mut c2b) = (vec![0f32; self.num_edges()], vec![0f32; self.num_edges()]);
            let (mut llr, mut hard, mut cand) = (vec![0f32; self.num_vars], vec![0u8; self.num_vars], vec![0u8; nc]);
            let mut rows = Vec::with_capacity(range.len() * stride);
            let mut conv = Vec::with_capacity(range.len());
            let mut iters = Vec::with_capacity(range.len());
            for s in range {
                let (ok, it) = self.decode_one(&syndromes[s * nc..(s + 1) * nc], &sc, &mut b2c, &mut c2b, &mut llr, &mut hard, &mut cand);
                let start = rows.len();
                rows.resize(start + stride, 0);
                for (v, &h) in hard.iter().enumerate() {
                    if h != 0 {
                        rows[start + v / 8] |= 1 << (v % 8);
                    }
                }
                conv.push(u8::from(ok));
                iters.push(it);
            }
            vec![(rows, conv, iters)]
        });
        let mut out = Bp32Outcome { hard: Vec::with_capacity(shots * stride), converged: Vec::with_capacity(shots), iterations: Vec::with_capacity(shots) };
        for (rows, conv, iters) in parts {
            out.hard.extend(rows);
            out.converged.extend(conv);
            out.iterations.extend(iters);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bp::Method;

    fn repetition(n: usize) -> (usize, Vec<Vec<u32>>, Vec<f64>) {
        // n bits, n − 1 checks on neighbours.
        let columns: Vec<Vec<u32>> = (0..n).map(|v| (0..n - 1).filter(|&c| c == v || c + 1 == v).map(|c| c as u32).collect()).collect();
        (n - 1, columns, vec![0.1; n])
    }

    #[test]
    fn f32_min_sum_agrees_with_f64_on_a_repetition_code() {
        let (m, columns, priors) = repetition(7);
        let bp = Bp::new(m, &columns, &priors).unwrap();
        let bp32 = Bp32::from_bp(&bp);
        let mut w = bp.work();
        let mut agree = 0;
        let mut total = 0;
        for s in 0..64u32 {
            let syndrome: Vec<u8> = (0..6).map(|c| ((s >> c) & 1) as u8).collect();
            let out = bp.decode(&syndrome, Method::MinSum { scale: 0.0 }, 10, &mut w);
            let o32 = bp32.decode_batch(&syndrome, 1, 0.0, 10, 1);
            let hard32: Vec<u8> = (0..7).map(|v| (o32.hard[v / 8] >> (v % 8)) & 1).collect();
            total += 1;
            if hard32 == w.hard && (o32.converged[0] != 0) == out.converged {
                agree += 1;
            }
        }
        assert!(agree >= total - 2, "{agree} of {total} syndromes decode the same in f32 and f64");
    }

    #[test]
    fn batches_equal_single_shots_on_any_threads() {
        let (m, columns, priors) = repetition(9);
        let bp32 = Bp32::from_bp(&Bp::new(m, &columns, &priors).unwrap());
        let syndromes: Vec<u8> = (0..200u32).flat_map(|s| (0..8).map(move |c| (s.wrapping_mul(2654435761) >> c) as u8 & 1)).collect();
        let a = bp32.decode_batch(&syndromes, 200, 0.625, 12, 1);
        let b = bp32.decode_batch(&syndromes, 200, 0.625, 12, 7);
        assert_eq!(a.hard, b.hard);
        assert_eq!(a.converged, b.converged);
        assert_eq!(a.iterations, b.iterations);
        assert_eq!(scales(0.0, 3), vec![0.5, 0.75, 0.875]);
    }
}
