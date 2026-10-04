use super::{Error, Result};

/// A table of bits, one row per shot, bit-packed as Stim packs them: each row a whole number of
/// bytes, bit `k` of a row in byte `k / 8` at position `k % 8` (numpy's
/// `packbits(..., bitorder="little")`, Stim's `b8` format).
///
/// ```
/// use stabilizer_qec::BitTable;
///
/// let mut t = BitTable::zeros(2, 10);
/// t.set(1, 9, true);
/// assert!(t.get(1, 9) && !t.get(0, 9));
/// assert_eq!(t.ones(1), vec![9]);
/// assert_eq!(t.row_bytes(1), &[0, 0b10]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BitTable {
    num_rows: usize,
    num_bits: usize,
    data: Vec<u8>,
}

impl BitTable {
    /// `num_rows` rows of `num_bits` zero bits.
    pub fn zeros(num_rows: usize, num_bits: usize) -> BitTable {
        BitTable {
            num_rows,
            num_bits,
            data: vec![0; num_rows * num_bits.div_ceil(8)],
        }
    }

    /// A table from its packed bytes: `num_rows` rows of ⌈num_bits / 8⌉ bytes each.
    pub fn from_packed(num_rows: usize, num_bits: usize, data: Vec<u8>) -> Result<BitTable> {
        let stride = num_bits.div_ceil(8);
        if stride.checked_mul(num_rows) != Some(data.len()) {
            return Err(Error::new(format!(
                "{} bytes is not {num_rows} rows of {num_bits} bits",
                data.len()
            )));
        }
        Ok(BitTable {
            num_rows,
            num_bits,
            data,
        })
    }

    /// A table from rows of bools, each `num_bits` long.
    pub fn from_rows<R: AsRef<[bool]>>(num_bits: usize, rows: &[R]) -> Result<BitTable> {
        let mut t = BitTable::zeros(rows.len(), num_bits);
        for (i, row) in rows.iter().enumerate() {
            let row = row.as_ref();
            if row.len() != num_bits {
                return Err(Error::new(format!(
                    "row {i} has {} bits, not {num_bits}",
                    row.len()
                )));
            }
            for (k, &b) in row.iter().enumerate() {
                if b {
                    t.set(i, k, true);
                }
            }
        }
        Ok(t)
    }

    /// The number of rows (shots).
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The number of bits in each row.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    fn stride(&self) -> usize {
        self.num_bits.div_ceil(8)
    }

    /// Row `row`'s packed bytes.
    pub fn row_bytes(&self, row: usize) -> &[u8] {
        let s = self.stride();
        &self.data[row * s..(row + 1) * s]
    }

    /// Bit `bit` of row `row`.
    pub fn get(&self, row: usize, bit: usize) -> bool {
        assert!(
            row < self.num_rows && bit < self.num_bits,
            "bit ({row}, {bit}) is outside a {}×{} table",
            self.num_rows,
            self.num_bits
        );
        (self.data[row * self.stride() + bit / 8] >> (bit % 8)) & 1 == 1
    }

    /// Set bit `bit` of row `row`.
    pub fn set(&mut self, row: usize, bit: usize, value: bool) {
        assert!(
            row < self.num_rows && bit < self.num_bits,
            "bit ({row}, {bit}) is outside a {}×{} table",
            self.num_rows,
            self.num_bits
        );
        let i = row * self.stride() + bit / 8;
        if value {
            self.data[i] |= 1 << (bit % 8);
        } else {
            self.data[i] &= !(1 << (bit % 8));
        }
    }

    /// The indices of row `row`'s set bits, in order: for detection events, the detectors that
    /// fired.
    pub fn ones(&self, row: usize) -> Vec<u32> {
        let mut out = Vec::new();
        crate::shots::defects_from_b8(self.row_bytes(row), self.num_bits, &mut out);
        out
    }

    /// Row `row` as bools.
    pub fn row(&self, row: usize) -> Vec<bool> {
        (0..self.num_bits).map(|k| self.get(row, k)).collect()
    }

    /// All the packed bytes, row after row.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// The packed bytes, taken.
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}
