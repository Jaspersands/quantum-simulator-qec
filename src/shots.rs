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
}
