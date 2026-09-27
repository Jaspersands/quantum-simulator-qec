//! Stim's shot formats: `01` (one line of '0'/'1' per shot) and `b8` (each shot
//! packed into ceil(n/8) bytes, bit i at byte i/8, bit i%8, little-endian
//! within each byte). Google's published detection events use these.

pub fn pack_row(bits: &[bool], out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + bits.len().div_ceil(8), 0);
    for (i, &b) in bits.iter().enumerate() {
        if b {
            out[start + i / 8] |= 1 << (i % 8);
        }
    }
}

pub fn write_b8(shots: &[Vec<bool>], num_bits: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(shots.len() * num_bits.div_ceil(8));
    for s in shots {
        debug_assert_eq!(s.len(), num_bits);
        pack_row(s, &mut out);
    }
    out
}

pub fn read_b8(bytes: &[u8], num_bits: usize) -> Result<Vec<Vec<bool>>, String> {
    let stride = num_bits.div_ceil(8);
    if stride == 0 || bytes.len() % stride != 0 {
        return Err(format!("{} bytes is not a whole number of {num_bits}-bit shots", bytes.len()));
    }
    Ok(bytes
        .chunks(stride)
        .map(|row| (0..num_bits).map(|i| (row[i / 8] >> (i % 8)) & 1 == 1).collect())
        .collect())
}

/// Append the indices of `row`'s set bits below `num_bits`, ascending: a
/// b8 shot's detection events. Eight bytes at a time; bits past `num_bits`
/// are ignored.
pub fn defects_from_b8(row: &[u8], num_bits: usize, out: &mut Vec<u32>) {
    let bytes = num_bits.div_ceil(8).min(row.len());
    let mut base = 0usize;
    for chunk in row[..bytes].chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let mut word = u64::from_le_bytes(buf);
        let left = num_bits - base;
        if left < 64 {
            word &= (1u64 << left) - 1;
        }
        while word != 0 {
            out.push((base + word.trailing_zeros() as usize) as u32);
            word &= word - 1;
        }
        base += 64;
    }
}

pub fn write_01(shots: &[Vec<bool>]) -> String {
    let mut s = String::with_capacity(shots.iter().map(|r| r.len() + 1).sum());
    for row in shots {
        s.extend(row.iter().map(|&b| if b { '1' } else { '0' }));
        s.push('\n');
    }
    s
}

pub fn read_01(text: &str, num_bits: usize) -> Result<Vec<Vec<bool>>, String> {
    text.lines()
        .filter(|l| !l.is_empty())
        .enumerate()
        .map(|(i, line)| {
            if line.len() != num_bits {
                return Err(format!("shot {i} has {} bits, expected {num_bits}", line.len()));
            }
            line.chars()
                .map(|c| match c {
                    '0' => Ok(false),
                    '1' => Ok(true),
                    _ => Err(format!("shot {i}: unexpected character '{c}'")),
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Vec<bool>> {
        vec![
            vec![true, false, false, false, false, false, false, false, true, true],
            vec![false; 10],
            vec![true; 10],
        ]
    }

    #[test]
    fn b8_round_trips_and_is_little_endian_within_bytes() {
        let bytes = write_b8(&sample(), 10);
        assert_eq!(bytes.len(), 6);
        assert_eq!(bytes[0], 0b0000_0001);
        assert_eq!(bytes[1], 0b0000_0011);
        assert_eq!(read_b8(&bytes, 10).unwrap(), sample());
    }

    #[test]
    fn zero_one_round_trips() {
        let text = write_01(&sample());
        assert_eq!(text.lines().next().unwrap(), "1000000011");
        assert_eq!(read_01(&text, 10).unwrap(), sample());
    }

    #[test]
    fn malformed_input_is_an_error() {
        assert!(read_b8(&[0u8; 5], 10).is_err());
        assert!(read_01("101\n", 10).is_err());
        assert!(read_01("10x0000000\n", 10).is_err());
    }
    #[test]
    fn defects_from_b8_equals_the_bitwise_scan() {
        let mut rng = crate::surface_code::Xorshift::new(9);
        for trial in 0..500 {
            let n = 1 + (rng.next_u64() % 300) as usize;
            let mut row: Vec<u8> = (0..n.div_ceil(8)).map(|_| rng.next_u64() as u8).collect();
            if trial % 2 == 0 {
                // Stray bits past num_bits must be ignored.
                if let Some(last) = row.last_mut() {
                    *last |= 0x80;
                }
            }
            let slow: Vec<u32> = (0..n).filter(|&i| (row[i / 8] >> (i % 8)) & 1 == 1).map(|i| i as u32).collect();
            let mut fast = vec![7u32];
            defects_from_b8(&row, n, &mut fast);
            assert_eq!(&fast[1..], &slow[..], "n = {n}");
        }
    }
}
