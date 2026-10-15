//! Exact maximum-likelihood decoding by contraction, the quantity CUDA-Q QEC's
//! `tensor_network_decoder` contracts its Tanner-graph network for: for a check matrix, a
//! logical observable, independent error priors and a soft syndrome (each check's probability
//! of having fired),
//!
//!   mass_b = Σ_e Π_j P(e_j) · Π_i (v_i if check i's parity under e is 1, else 1 − v_i)
//!            · [the logical's parity under e is b],
//!
//! and the posterior that the logical flipped is mass_1 / (mass_0 + mass_1).
//!
//! The sum is contracted error by error, keeping a table over the parities of the checks still
//! open (touched by some processed error and not yet by all of theirs) and the logical's parity;
//! a check is closed by weighting its parity once its last error is in. Its cost is
//! exponential in the widest frontier, which a round-ordered matrix keeps small.

/// The widest frontier contracted (a table of 2^(frontier + 1) numbers).
pub const MAX_FRONTIER: usize = 28;

/// (mass_0, mass_1) for the matrix whose errors (columns, taken in `order`) set off `columns`,
/// the errors in the logical, their priors and each check's probability of having fired.
pub fn masses(num_checks: usize, columns: &[Vec<u32>], logical: &[bool], priors: &[f64], fired: &[f64], order: &[usize]) -> Result<(f64, f64), String> {
    let n = columns.len();
    if logical.len() != n || priors.len() != n || fired.len() != num_checks {
        return Err("the logical, priors and syndrome must match the matrix".into());
    }
    let mut last = vec![usize::MAX; num_checks];
    for (k, &e) in order.iter().enumerate() {
        for &c in &columns[e] {
            last[c as usize] = k;
        }
    }
    // Bit 0 holds the logical's parity; bit b ≥ 1 the parity of the check active[b − 1].
    let mut state = vec![1.0, 0.0];
    let mut active: Vec<u32> = Vec::new();
    let mut bit_of = vec![usize::MAX; num_checks];
    for (k, &e) in order.iter().enumerate() {
        for &c in &columns[e] {
            if bit_of[c as usize] == usize::MAX {
                if active.len() + 1 > MAX_FRONTIER {
                    return Err(format!("the contraction's frontier exceeds {MAX_FRONTIER} checks; order the matrix's columns by round (sort_pcm_columns)"));
                }
                active.push(c);
                bit_of[c as usize] = active.len();
                let len = state.len();
                state.resize(2 * len, 0.0);
            }
        }
        let mut mask = usize::from(logical[e]);
        for &c in &columns[e] {
            mask ^= 1 << bit_of[c as usize];
        }
        let p = priors[e];
        if mask == 0 {
            // (An error flipping nothing only rescales.)
            for x in state.iter_mut() {
                *x *= (1.0 - p) + p;
            }
        } else {
            let old = state.clone();
            for x in 0..state.len() {
                state[x] = (1.0 - p) * old[x] + p * old[x ^ mask];
            }
        }
        // Close the checks this error is the last of.
        for &c in &columns[e] {
            if last[c as usize] != k || bit_of[c as usize] == usize::MAX {
                continue;
            }
            let b = bit_of[c as usize];
            let top = active.len();
            if b != top {
                for x in 0..state.len() {
                    if (x >> b & 1) == 1 && (x >> top & 1) == 0 {
                        state.swap(x, x ^ (1 << b) ^ (1 << top));
                    }
                }
                let moved = active[top - 1];
                active[b - 1] = moved;
                bit_of[moved as usize] = b;
            }
            active.pop();
            bit_of[c as usize] = usize::MAX;
            let (w0, w1) = (1.0 - fired[c as usize], fired[c as usize]);
            let half = state.len() / 2;
            let mut out = vec![0.0; half];
            for (x, o) in out.iter_mut().enumerate() {
                *o = w0 * state[x] + w1 * state[x + half];
            }
            state = out;
        }
    }
    // Checks no error reaches have parity 0.
    let mut scale = 1.0;
    for (c, &l) in last.iter().enumerate() {
        if l == usize::MAX {
            scale *= 1.0 - fired[c];
        }
    }
    Ok((state[0] * scale, state[1] * scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(num_checks: usize, columns: &[Vec<u32>], logical: &[bool], priors: &[f64], fired: &[f64]) -> (f64, f64) {
        let n = columns.len();
        let mut m = [0.0, 0.0];
        for e in 0u32..(1 << n) {
            let mut w = 1.0;
            let mut parity = vec![false; num_checks];
            let mut l = false;
            for j in 0..n {
                let on = e >> j & 1 == 1;
                w *= if on { priors[j] } else { 1.0 - priors[j] };
                if on {
                    for &c in &columns[j] {
                        parity[c as usize] ^= true;
                    }
                    l ^= logical[j];
                }
            }
            for (i, &par) in parity.iter().enumerate() {
                w *= if par { fired[i] } else { 1.0 - fired[i] };
            }
            m[usize::from(l)] += w;
        }
        (m[0], m[1])
    }

    #[test]
    fn equals_brute_force() {
        let mut s = 99u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..60 {
            let (m, n) = (1 + (next() % 6) as usize, 1 + (next() % 10) as usize);
            let columns: Vec<Vec<u32>> = (0..n).map(|_| (0..m as u32).filter(|_| next() % 3 == 0).collect()).collect();
            let logical: Vec<bool> = (0..n).map(|_| next() % 2 == 0).collect();
            let priors: Vec<f64> = (0..n).map(|_| (next() % 100) as f64 / 250.0).collect();
            let fired: Vec<f64> = (0..m).map(|_| [0.0, 1.0, 0.3][(next() % 3) as usize]).collect();
            let mut order: Vec<usize> = (0..n).collect();
            order.rotate_left((next() as usize) % n);
            let (a0, a1) = masses(m, &columns, &logical, &priors, &fired, &order).unwrap();
            let (b0, b1) = brute(m, &columns, &logical, &priors, &fired);
            assert!((a0 - b0).abs() <= 1e-12 * (1.0 + b0.abs()) && (a1 - b1).abs() <= 1e-12 * (1.0 + b1.abs()), "{a0} {b0} {a1} {b1}");
        }
    }
}
