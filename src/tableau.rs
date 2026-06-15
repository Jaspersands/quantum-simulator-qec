use rand::Rng;

pub struct Tableau {
    pub n: usize,
    words_per_row: usize,
    // x and z are packed arrays of u64 of size (2*N + 1) * words_per_row
    x: Vec<u64>,
    z: Vec<u64>,
    // r is phase bits (0 for +1, 1 for -1) of size 2*N + 1
    r: Vec<u8>,
}

#[inline]
fn g(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
    match (x1, z1) {
        (false, false) => 0,
        (true, true) => (z2 as i32) - (x2 as i32),
        (true, false) => (z2 as i32) * (2 * (x2 as i32) - 1),
        (false, true) => (x2 as i32) * (1 - 2 * (z2 as i32)),
    }
}

impl Tableau {
    pub fn new(n: usize) -> Self {
        let words_per_row = (n + 63) / 64;
        let num_rows = 2 * n + 1; // 2n rows + 1 scratch row
        let mut x = vec![0; num_rows * words_per_row];
        let mut z = vec![0; num_rows * words_per_row];
        let r = vec![0; num_rows];

        // Initialize destabilizers to X_i
        for i in 0..n {
            let idx = i * words_per_row + (i / 64);
            x[idx] |= 1 << (i % 64);
        }

        // Initialize stabilizers to Z_i
        for i in 0..n {
            let row = n + i;
            let idx = row * words_per_row + (i / 64);
            z[idx] |= 1 << (i % 64);
        }

        Tableau {
            n,
            words_per_row,
            x,
            z,
            r,
        }
    }

    #[inline]
    pub fn get_x(&self, row: usize, col: usize) -> bool {
        let idx = row * self.words_per_row + (col / 64);
        ((self.x[idx] >> (col % 64)) & 1) != 0
    }

    #[inline]
    pub fn set_x(&mut self, row: usize, col: usize, val: bool) {
        let idx = row * self.words_per_row + (col / 64);
        let shift = col % 64;
        if val {
            self.x[idx] |= 1 << shift;
        } else {
            self.x[idx] &= !(1 << shift);
        }
    }

    #[inline]
    pub fn get_z(&self, row: usize, col: usize) -> bool {
        let idx = row * self.words_per_row + (col / 64);
        ((self.z[idx] >> (col % 64)) & 1) != 0
    }

    #[inline]
    pub fn set_z(&mut self, row: usize, col: usize, val: bool) {
        let idx = row * self.words_per_row + (col / 64);
        let shift = col % 64;
        if val {
            self.z[idx] |= 1 << shift;
        } else {
            self.z[idx] &= !(1 << shift);
        }
    }

    #[inline]
    pub fn get_r(&self, row: usize) -> u8 {
        self.r[row]
    }

    #[inline]
    pub fn set_r(&mut self, row: usize, val: u8) {
        self.r[row] = val;
    }

    pub fn rowsum(&mut self, i: usize, j: usize) {
        let mut sum_g = 0;
        for k in 0..self.n {
            let x1 = self.get_x(i, k);
            let z1 = self.get_z(i, k);
            let x2 = self.get_x(j, k);
            let z2 = self.get_z(j, k);
            sum_g += g(x1, z1, x2, z2);
        }

        let sum = (2 * (self.r[i] as i32) + 2 * (self.r[j] as i32) + sum_g).rem_euclid(4);
        assert!(
            sum == 0 || sum == 2,
            "Rowsum phase sum must be 0 or 2 mod 4, got {}. Rows: i={}, j={}",
            sum,
            i,
            j
        );

        self.r[i] = if sum == 0 { 0 } else { 1 };

        let w = self.words_per_row;
        for w_idx in 0..w {
            self.x[i * w + w_idx] ^= self.x[j * w + w_idx];
            self.z[i * w + w_idx] ^= self.z[j * w + w_idx];
        }
    }

    pub fn apply_h(&mut self, qubit: usize) {
        for i in 0..(2 * self.n) {
            let x_val = self.get_x(i, qubit);
            let z_val = self.get_z(i, qubit);
            self.set_r(i, self.get_r(i) ^ (x_val as u8 & z_val as u8));
            self.set_x(i, qubit, z_val);
            self.set_z(i, qubit, x_val);
        }
    }

    pub fn apply_s(&mut self, qubit: usize) {
        for i in 0..(2 * self.n) {
            let x_val = self.get_x(i, qubit);
            let z_val = self.get_z(i, qubit);
            self.set_r(i, self.get_r(i) ^ (x_val as u8 & z_val as u8));
            self.set_z(i, qubit, z_val ^ x_val);
        }
    }

    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        for i in 0..(2 * self.n) {
            let xc = self.get_x(i, control);
            let zc = self.get_z(i, control);
            let xt = self.get_x(i, target);
            let zt = self.get_z(i, target);

            let phase_flip = xc as u8 & zt as u8 & (xt as u8 ^ zc as u8 ^ 1);
            self.set_r(i, self.get_r(i) ^ phase_flip);

            self.set_x(i, target, xt ^ xc);
            self.set_z(i, control, zc ^ zt);
        }
    }

    pub fn measure(&mut self, qubit: usize) -> u8 {
        // Search for stabilizer row i in [n, 2n-1] such that x_{i, qubit} = 1
        let mut p_prime: Option<usize> = None;
        for i in self.n..(2 * self.n) {
            if self.get_x(i, qubit) {
                p_prime = Some(i);
                break;
            }
        }

        match p_prime {
            Some(p) => {
                // Case 1: Random measurement outcome
                for i in 0..(2 * self.n) {
                    if i != p && i != (p - self.n) && self.get_x(i, qubit) {
                        self.rowsum(i, p);
                    }
                }

                // Copy row p to row p-n
                let dest_row = p - self.n;
                let w = self.words_per_row;
                for w_idx in 0..w {
                    self.x[dest_row * w + w_idx] = self.x[p * w + w_idx];
                    self.z[dest_row * w + w_idx] = self.z[p * w + w_idx];
                }
                self.r[dest_row] = self.r[p];

                // Set row p to Z_qubit
                for col in 0..self.n {
                    self.set_x(p, col, false);
                    self.set_z(p, col, col == qubit);
                }

                // Choose a random outcome 0 or 1
                let outcome = if rand::thread_rng().gen::<bool>() { 1 } else { 0 };
                self.r[p] = outcome;
                outcome
            }
            None => {
                // Case 2: Deterministic measurement outcome
                let scratch_row = 2 * self.n;
                // Zero out the scratch row
                let w = self.words_per_row;
                for w_idx in 0..w {
                    self.x[scratch_row * w + w_idx] = 0;
                    self.z[scratch_row * w + w_idx] = 0;
                }
                self.r[scratch_row] = 0;

                for i in 0..self.n {
                    if self.get_x(i, qubit) {
                        self.rowsum(scratch_row, i + self.n);
                    }
                }

                self.r[scratch_row]
            }
        }
    }
}
