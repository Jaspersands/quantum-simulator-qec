//! What MWPF takes from other crates, written here so that this crate keeps no dependencies:
//! `num_traits`' numeric traits, the few `itertools` adaptors and macros MWPF calls, `hashbrown`'s
//! map and set (here ordered, so that every run iterates alike), `lnexp`'s `ln_1p_exp`,
//! `prettytable`'s tables (as MWPF draws its parity matrices) and a random seed for the calls
//! that ask for one. The lock, JSON, xoshiro and priority queue are Fusion Blossom's
//! (`crate::fusion::shim`), which MWPF shares.

pub(crate) use crate::fusion::shim::{parking_lot, priority_queue, rand_xoshiro, serde_json};

/// `num_traits`' traits as MWPF uses them, for its `OrderedFloat` weights and for `f64`.
pub mod num_traits {
    pub trait Zero: Sized {
        fn zero() -> Self;
        fn is_zero(&self) -> bool;
        fn set_zero(&mut self) {
            *self = Self::zero();
        }
    }
    pub trait One: Sized {
        fn one() -> Self;
        fn is_one(&self) -> bool;
    }
    pub trait Signed: Sized {
        fn abs(&self) -> Self;
        fn abs_sub(&self, other: &Self) -> Self;
        fn signum(&self) -> Self;
        fn is_positive(&self) -> bool;
        fn is_negative(&self) -> bool;
    }
    pub trait FromPrimitive: Sized {
        fn from_i64(n: i64) -> Option<Self>;
        fn from_u64(n: u64) -> Option<Self>;
        fn from_f64(n: f64) -> Option<Self>;
        fn from_usize(n: usize) -> Option<Self> {
            Self::from_u64(n as u64)
        }
        fn from_i32(n: i32) -> Option<Self> {
            Self::from_i64(n as i64)
        }
    }
    pub trait ToPrimitive {
        fn to_i64(&self) -> Option<i64>;
        fn to_u64(&self) -> Option<u64>;
        fn to_f64(&self) -> Option<f64>;
        fn to_usize(&self) -> Option<usize> {
            self.to_u64().map(|x| x as usize)
        }
    }

    impl Zero for f64 {
        fn zero() -> Self {
            0.0
        }
        fn is_zero(&self) -> bool {
            *self == 0.0
        }
    }
    impl One for f64 {
        fn one() -> Self {
            1.0
        }
        fn is_one(&self) -> bool {
            *self == 1.0
        }
    }
    impl Signed for f64 {
        fn abs(&self) -> Self {
            f64::abs(*self)
        }
        fn abs_sub(&self, other: &Self) -> Self {
            if *self <= *other {
                0.0
            } else {
                *self - *other
            }
        }
        fn signum(&self) -> Self {
            f64::signum(*self)
        }
        fn is_positive(&self) -> bool {
            *self > 0.0
        }
        fn is_negative(&self) -> bool {
            *self < 0.0
        }
    }
    impl ToPrimitive for f64 {
        fn to_i64(&self) -> Option<i64> {
            if self.is_finite() && *self > -9223372036854777856.0 && *self < 9223372036854775808.0 {
                Some(*self as i64)
            } else {
                None
            }
        }
        fn to_u64(&self) -> Option<u64> {
            if self.is_finite() && *self > -1.0 && *self < 18446744073709551616.0 {
                Some(*self as u64)
            } else {
                None
            }
        }
        fn to_f64(&self) -> Option<f64> {
            Some(*self)
        }
    }
    impl ToPrimitive for i64 {
        fn to_i64(&self) -> Option<i64> {
            Some(*self)
        }
        fn to_u64(&self) -> Option<u64> {
            u64::try_from(*self).ok()
        }
        fn to_f64(&self) -> Option<f64> {
            Some(*self as f64)
        }
    }
}

/// `itertools`' adaptors as MWPF uses them: `unique` (first occurrences, in order), `sorted`,
/// `join`, and the `izip!` macro.
pub mod itertools {
    pub trait Itertools: Iterator + Sized {
        fn unique(self) -> std::vec::IntoIter<Self::Item>
        where
            Self::Item: Clone + Eq + std::hash::Hash,
        {
            let mut seen = std::collections::HashSet::new();
            let kept: Vec<Self::Item> = self.filter(|x| seen.insert(x.clone())).collect();
            kept.into_iter()
        }
        fn sorted(self) -> std::vec::IntoIter<Self::Item>
        where
            Self::Item: Ord,
        {
            let mut v: Vec<Self::Item> = self.collect();
            v.sort();
            v.into_iter()
        }
        fn join(self, sep: &str) -> String
        where
            Self::Item: std::fmt::Display,
        {
            let mut out = String::new();
            for (k, x) in self.enumerate() {
                if k > 0 {
                    out.push_str(sep);
                }
                out.push_str(&x.to_string());
            }
            out
        }
    }
    impl<I: Iterator> Itertools for I {}
}

/// `itertools::izip!`: iterators zipped into flat tuples.
macro_rules! izip {
    ($a:expr, $b:expr $(,)?) => {
        ::std::iter::Iterator::zip(::std::iter::IntoIterator::into_iter($a), $b)
    };
    ($a:expr, $b:expr, $c:expr $(,)?) => {
        ::std::iter::Iterator::map(
            ::std::iter::Iterator::zip(::std::iter::Iterator::zip(::std::iter::IntoIterator::into_iter($a), $b), $c),
            |((a, b), c)| (a, b, c),
        )
    };
    ($a:expr, $b:expr, $c:expr, $d:expr $(,)?) => {
        ::std::iter::Iterator::map(
            ::std::iter::Iterator::zip(
                ::std::iter::Iterator::zip(::std::iter::Iterator::zip(::std::iter::IntoIterator::into_iter($a), $b), $c),
                $d,
            ),
            |(((a, b), c), d)| (a, b, c, d),
        )
    };
}
pub(crate) use izip;

/// `hashbrown`'s map and set. MWPF's wheels iterate them in hash order, which `hashbrown` seeds
/// at random; here they are ordered, so that every run iterates alike.
pub mod hashbrown {
    pub type HashMap<K, V> = std::collections::BTreeMap<K, V>;
    pub type HashSet<T> = std::collections::BTreeSet<T>;
    pub mod hash_map {
        pub use std::collections::btree_map::Entry;
    }
}

/// `lnexp::LnExp::ln_1p_exp`: `ln(1 + exp(x))` by Maechler's cases, as `lnexp` 0.2 does.
pub mod lnexp {
    pub trait LnExp {
        fn ln_1p_exp(&self) -> Self;
    }
    impl LnExp for f64 {
        fn ln_1p_exp(&self) -> f64 {
            use std::f64::consts::LN_2;
            const PRECISION: f64 = f64::MANTISSA_DIGITS as f64;
            const LN_OF_P_MUL_LN_2: f64 = 3.6037789929704576;
            const X0: f64 = -PRECISION * LN_2;
            const X1: f64 = (PRECISION - 1.0) * LN_2 / 2.0;
            const X2: f64 = {
                let p_ln2 = PRECISION * LN_2;
                p_ln2 + LN_OF_P_MUL_LN_2 * (1.0 / p_ln2 - 1.0)
            };
            if *self <= X0 {
                self.exp()
            } else if *self <= X1 {
                self.exp().ln_1p()
            } else if *self <= X2 {
                *self + (-*self).exp()
            } else {
                *self
            }
        }
    }
}

/// A random 64-bit seed, for the calls that draw one when none is given (`thread_rng().gen()`).
pub fn random_u64() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    if let Ok(elapsed) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        hasher.write_u128(elapsed.as_nanos());
    }
    hasher.finish()
}

/// `prettytable`'s tables as MWPF draws its parity matrices: cells of one or more lines, aligned
/// left or right, between the separators of a [`format::TableFormat`].
pub mod prettytable {
    use std::fmt;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Alignment {
        Left,
        Center,
        Right,
    }

    #[derive(Clone, Debug)]
    pub struct Cell {
        content: Vec<String>,
        align: Alignment,
    }

    impl Cell {
        pub fn new(text: &str) -> Cell {
            Cell { content: text.split('\n').map(|s| s.to_string()).collect(), align: Alignment::Left }
        }
        /// `prettytable`'s style letters; only the alignment (`l`, `c`, `r`) changes the text.
        pub fn style_spec(mut self, spec: &str) -> Cell {
            let mut chars = spec.chars();
            while let Some(c) = chars.next() {
                match c {
                    'F' | 'B' => {
                        chars.next();
                    }
                    'l' => self.align = Alignment::Left,
                    'c' => self.align = Alignment::Center,
                    'r' => self.align = Alignment::Right,
                    _ => {}
                }
            }
            self
        }
        pub fn get_content(&self) -> String {
            self.content.join("\n")
        }
        fn width(&self) -> usize {
            self.content.iter().map(|s| s.chars().count()).max().unwrap_or(0)
        }
        fn height(&self) -> usize {
            self.content.len()
        }
        fn line(&self, k: usize, width: usize) -> String {
            let text = self.content.get(k).map(|s| s.as_str()).unwrap_or("");
            let fill = width.saturating_sub(text.chars().count());
            match self.align {
                Alignment::Left => format!("{text}{}", " ".repeat(fill)),
                Alignment::Right => format!("{}{text}", " ".repeat(fill)),
                Alignment::Center => {
                    let left = fill / 2;
                    format!("{}{text}{}", " ".repeat(left), " ".repeat(fill - left))
                }
            }
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct Row {
        cells: Vec<Cell>,
    }

    impl Row {
        pub fn empty() -> Row {
            Row { cells: vec![] }
        }
        pub fn new(cells: Vec<Cell>) -> Row {
            Row { cells }
        }
        pub fn add_cell(&mut self, cell: Cell) {
            self.cells.push(cell);
        }
        pub fn insert_cell(&mut self, index: usize, cell: Cell) {
            let index = index.min(self.cells.len());
            self.cells.insert(index, cell);
        }
        pub fn set_cell(&mut self, cell: Cell, column: usize) -> Result<(), &'static str> {
            match self.cells.get_mut(column) {
                Some(c) => {
                    *c = cell;
                    Ok(())
                }
                None => Err("Cannot find cell"),
            }
        }
        pub fn get_cell(&self, column: usize) -> Option<&Cell> {
            self.cells.get(column)
        }
        pub fn len(&self) -> usize {
            self.cells.len()
        }
        pub fn is_empty(&self) -> bool {
            self.cells.is_empty()
        }
        pub fn iter(&self) -> std::slice::Iter<'_, Cell> {
            self.cells.iter()
        }
    }

    impl<'a> IntoIterator for &'a Row {
        type Item = &'a Cell;
        type IntoIter = std::slice::Iter<'a, Cell>;
        fn into_iter(self) -> Self::IntoIter {
            self.cells.iter()
        }
    }

    pub mod format {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum LinePosition {
            Top,
            Title,
            Intern,
            Bottom,
        }

        #[derive(Clone, Copy, Debug)]
        pub struct LineSeparator {
            pub(crate) line: char,
            pub(crate) junction: char,
            pub(crate) ljunction: char,
            pub(crate) rjunction: char,
        }

        impl LineSeparator {
            pub fn new(line: char, junction: char, ljunction: char, rjunction: char) -> LineSeparator {
                LineSeparator { line, junction, ljunction, rjunction }
            }
        }

        #[derive(Clone, Copy, Debug, Default)]
        pub struct TableFormat {
            pub(crate) csep: Option<char>,
            pub(crate) lborder: Option<char>,
            pub(crate) rborder: Option<char>,
            pub(crate) top: Option<LineSeparator>,
            pub(crate) title: Option<LineSeparator>,
            pub(crate) intern: Option<LineSeparator>,
            pub(crate) bottom: Option<LineSeparator>,
            pub(crate) pad_left: usize,
            pub(crate) pad_right: usize,
        }

        impl TableFormat {
            pub fn new() -> TableFormat {
                TableFormat::default()
            }
            pub fn padding(&mut self, left: usize, right: usize) {
                self.pad_left = left;
                self.pad_right = right;
            }
            pub fn column_separator(&mut self, separator: char) {
                self.csep = Some(separator);
            }
            pub fn borders(&mut self, border: char) {
                self.lborder = Some(border);
                self.rborder = Some(border);
            }
            pub fn separators(&mut self, positions: &[LinePosition], separator: LineSeparator) {
                for position in positions {
                    match position {
                        LinePosition::Top => self.top = Some(separator),
                        LinePosition::Title => self.title = Some(separator),
                        LinePosition::Intern => self.intern = Some(separator),
                        LinePosition::Bottom => self.bottom = Some(separator),
                    }
                }
            }
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct Table {
        format: format::TableFormat,
        titles: Option<Row>,
        rows: Vec<Row>,
    }

    impl Table {
        pub fn new() -> Table {
            Table::default()
        }
        pub fn set_format(&mut self, format: format::TableFormat) {
            self.format = format;
        }
        pub fn set_titles(&mut self, titles: Row) {
            self.titles = Some(titles);
        }
        pub fn add_row(&mut self, row: Row) {
            self.rows.push(row);
        }
        pub fn printstd(&self) {
            print!("{self}");
        }

        fn widths(&self) -> Vec<usize> {
            let mut widths: Vec<usize> = vec![];
            for row in self.titles.iter().chain(self.rows.iter()) {
                for (k, cell) in row.cells.iter().enumerate() {
                    if widths.len() <= k {
                        widths.push(0);
                    }
                    widths[k] = widths[k].max(cell.width());
                }
            }
            widths
        }

        fn separator(&self, out: &mut String, sep: &Option<format::LineSeparator>, widths: &[usize]) {
            let Some(sep) = sep else { return };
            let f = &self.format;
            if f.lborder.is_some() {
                out.push(sep.ljunction);
            }
            for (k, w) in widths.iter().enumerate() {
                if k > 0 && f.csep.is_some() {
                    out.push(sep.junction);
                }
                for _ in 0..(w + f.pad_left + f.pad_right) {
                    out.push(sep.line);
                }
            }
            if f.rborder.is_some() {
                out.push(sep.rjunction);
            }
            out.push('\n');
        }

        fn row(&self, out: &mut String, row: &Row, widths: &[usize]) {
            let f = &self.format;
            let height = row.cells.iter().map(|c| c.height()).max().unwrap_or(1).max(1);
            let blank = Cell::new("");
            for line in 0..height {
                if let Some(b) = f.lborder {
                    out.push(b);
                }
                for (k, w) in widths.iter().enumerate() {
                    if k > 0 {
                        if let Some(c) = f.csep {
                            out.push(c);
                        }
                    }
                    out.push_str(&" ".repeat(f.pad_left));
                    out.push_str(&row.cells.get(k).unwrap_or(&blank).line(line, *w));
                    out.push_str(&" ".repeat(f.pad_right));
                }
                if let Some(b) = f.rborder {
                    out.push(b);
                }
                out.push('\n');
            }
        }
    }

    impl fmt::Display for Table {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let widths = self.widths();
            let mut out = String::new();
            self.separator(&mut out, &self.format.top, &widths);
            if let Some(titles) = &self.titles {
                self.row(&mut out, titles, &widths);
                self.separator(&mut out, &self.format.title, &widths);
            }
            for (k, row) in self.rows.iter().enumerate() {
                if k > 0 {
                    self.separator(&mut out, &self.format.intern, &widths);
                }
                self.row(&mut out, row, &widths);
            }
            self.separator(&mut out, &self.format.bottom, &widths);
            f.write_str(&out)
        }
    }
}

/// What MWPF uses of `rand` 0.8 (`Rng` on any generator, as `rand` draws `bool`s and integers;
/// `thread_rng` is a Xoshiro256** seeded per call).
pub mod rand {
    pub use super::rand_xoshiro::rand_core::{RngCore, SeedableRng};

    pub trait Gen {
        fn gen_from<R: RngCore + ?Sized>(rng: &mut R) -> Self;
    }
    impl Gen for bool {
        fn gen_from<R: RngCore + ?Sized>(rng: &mut R) -> Self {
            (rng.next_u32() as i32) < 0
        }
    }
    impl Gen for u64 {
        fn gen_from<R: RngCore + ?Sized>(rng: &mut R) -> Self {
            rng.next_u64()
        }
    }
    impl Gen for usize {
        fn gen_from<R: RngCore + ?Sized>(rng: &mut R) -> Self {
            rng.next_u64() as usize
        }
    }
    impl Gen for f64 {
        fn gen_from<R: RngCore + ?Sized>(rng: &mut R) -> Self {
            (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        }
    }

    pub trait Rng: RngCore {
        fn gen<T: Gen>(&mut self) -> T {
            T::gen_from(self)
        }
        fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
            let span = (range.end - range.start) as u64;
            assert!(span > 0, "cannot sample empty range");
            let zone = (span << span.leading_zeros()).wrapping_sub(1);
            loop {
                let m = (self.next_u64() as u128) * (span as u128);
                if (m as u64) <= zone {
                    return range.start + (m >> 64) as usize;
                }
            }
        }
    }
    impl<R: RngCore + ?Sized> Rng for R {}

    pub type ThreadRng = super::rand_xoshiro::Xoshiro256StarStar;

    pub fn thread_rng() -> ThreadRng {
        ThreadRng::seed_from_u64(super::random_u64())
    }

    pub mod prelude {
        pub use super::{thread_rng, Rng, RngCore, SeedableRng};
    }
}

/// gzip (RFC 1952) for `HTMLExport`'s compressed visualizer data (`flate2`'s `GzEncoder` and
/// `GzDecoder`): this writes stored deflate blocks, and reads any deflate stream.
pub mod gzip {
    fn crc32(data: &[u8]) -> u32 {
        let mut crc = !0u32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
            }
        }
        !crc
    }

    pub fn compress(data: &[u8]) -> Vec<u8> {
        // flate2's header: no name, no mtime, OS byte 255
        let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255];
        let mut chunks = data.chunks(65535).peekable();
        if chunks.peek().is_none() {
            out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
        }
        while let Some(chunk) = chunks.next() {
            out.push(if chunks.peek().is_none() { 1 } else { 0 });
            let n = chunk.len() as u16;
            out.extend_from_slice(&n.to_le_bytes());
            out.extend_from_slice(&(!n).to_le_bytes());
            out.extend_from_slice(chunk);
        }
        out.extend_from_slice(&crc32(data).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out
    }

    struct Bits<'a> {
        data: &'a [u8],
        pos: usize,
        bit: u32,
        nbits: u32,
    }

    impl Bits<'_> {
        fn need(&mut self, n: u32) -> Result<(), String> {
            while self.nbits < n {
                let b = *self.data.get(self.pos).ok_or("unexpected end of deflate stream")?;
                self.pos += 1;
                self.bit |= (b as u32) << self.nbits;
                self.nbits += 8;
            }
            Ok(())
        }
        fn bits(&mut self, n: u32) -> Result<u32, String> {
            if n == 0 {
                return Ok(0);
            }
            self.need(n)?;
            let v = self.bit & ((1u32 << n) - 1);
            self.bit >>= n;
            self.nbits -= n;
            Ok(v)
        }
        fn align(&mut self) {
            self.bit = 0;
            self.nbits = 0;
        }
    }

    /// A canonical Huffman code: counts per length and the symbols in code order.
    struct Huffman {
        counts: [u16; 16],
        symbols: Vec<u16>,
    }

    impl Huffman {
        fn new(lengths: &[u8]) -> Self {
            let mut counts = [0u16; 16];
            for &l in lengths {
                counts[l as usize] += 1;
            }
            counts[0] = 0;
            let mut offs = [0u16; 16];
            for l in 1..16 {
                offs[l] = offs[l - 1] + counts[l - 1];
            }
            let mut symbols = vec![0u16; lengths.len()];
            for (sym, &l) in lengths.iter().enumerate() {
                if l != 0 {
                    symbols[offs[l as usize] as usize] = sym as u16;
                    offs[l as usize] += 1;
                }
            }
            Huffman { counts, symbols }
        }
        fn decode(&self, bits: &mut Bits) -> Result<u16, String> {
            let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
            for len in 1..16 {
                code |= bits.bits(1)? as i32;
                let count = self.counts[len] as i32;
                if code - count < first {
                    return Ok(self.symbols[(index + (code - first)) as usize]);
                }
                index += count;
                first += count;
                first <<= 1;
                code <<= 1;
            }
            Err("invalid Huffman code".into())
        }
    }

    const LBASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
    const LEXT: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
    const DBASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
    const DEXT: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

    fn inflate(bits: &mut Bits, out: &mut Vec<u8>) -> Result<(), String> {
        loop {
            let last = bits.bits(1)?;
            match bits.bits(2)? {
                0 => {
                    bits.align();
                    let d = bits.data;
                    let p = bits.pos;
                    if p + 4 > d.len() {
                        return Err("unexpected end of deflate stream".into());
                    }
                    let n = u16::from_le_bytes([d[p], d[p + 1]]) as usize;
                    let nn = u16::from_le_bytes([d[p + 2], d[p + 3]]) as usize;
                    if n != !nn & 0xffff {
                        return Err("corrupt stored block".into());
                    }
                    let block = d.get(p + 4..p + 4 + n).ok_or("unexpected end of deflate stream")?;
                    out.extend_from_slice(block);
                    bits.pos = p + 4 + n;
                }
                kind @ (1 | 2) => {
                    let (lit, dist) = if kind == 1 {
                        let mut l = [8u8; 288];
                        l[144..256].fill(9);
                        l[256..280].fill(7);
                        (Huffman::new(&l), Huffman::new(&[5u8; 30]))
                    } else {
                        let hlit = bits.bits(5)? as usize + 257;
                        let hdist = bits.bits(5)? as usize + 1;
                        let hclen = bits.bits(4)? as usize + 4;
                        const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                        let mut cl = [0u8; 19];
                        for &o in ORDER.iter().take(hclen) {
                            cl[o] = bits.bits(3)? as u8;
                        }
                        let clh = Huffman::new(&cl);
                        let mut lengths = vec![0u8; hlit + hdist];
                        let mut k = 0;
                        while k < hlit + hdist {
                            let sym = clh.decode(bits)?;
                            let (val, rep) = match sym {
                                0..=15 => (sym as u8, 1),
                                16 => (*lengths.get(k.wrapping_sub(1)).ok_or("repeat with no previous length")?, 3 + bits.bits(2)? as usize),
                                17 => (0, 3 + bits.bits(3)? as usize),
                                _ => (0, 11 + bits.bits(7)? as usize),
                            };
                            if k + rep > hlit + hdist {
                                return Err("too many code lengths".into());
                            }
                            lengths[k..k + rep].fill(val);
                            k += rep;
                        }
                        (Huffman::new(&lengths[..hlit]), Huffman::new(&lengths[hlit..]))
                    };
                    loop {
                        let sym = lit.decode(bits)? as usize;
                        if sym < 256 {
                            out.push(sym as u8);
                        } else if sym == 256 {
                            break;
                        } else {
                            let s = sym - 257;
                            if s >= 29 {
                                return Err("invalid length symbol".into());
                            }
                            let len = LBASE[s] as usize + bits.bits(LEXT[s] as u32)? as usize;
                            let ds = dist.decode(bits)? as usize;
                            if ds >= 30 {
                                return Err("invalid distance symbol".into());
                            }
                            let d = DBASE[ds] as usize + bits.bits(DEXT[ds] as u32)? as usize;
                            if d > out.len() {
                                return Err("distance too far back".into());
                            }
                            let start = out.len() - d;
                            for i in 0..len {
                                out.push(out[start + i]);
                            }
                        }
                    }
                }
                _ => return Err("invalid block type".into()),
            }
            if last == 1 {
                return Ok(());
            }
        }
    }

    pub fn decompress(data: &[u8]) -> Result<Vec<u8>, String> {
        if data.len() < 18 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 8 {
            return Err("invalid gzip header".into());
        }
        let flags = data[3];
        let mut p = 10;
        if flags & 4 != 0 {
            let n = u16::from_le_bytes([data[p], data[p + 1]]) as usize;
            p += 2 + n;
        }
        for flag in [8u8, 16] {
            if flags & flag != 0 {
                while *data.get(p).ok_or("invalid gzip header")? != 0 {
                    p += 1;
                }
                p += 1;
            }
        }
        if flags & 2 != 0 {
            p += 2;
        }
        let mut bits = Bits { data, pos: p, bit: 0, nbits: 0 };
        let mut out = Vec::new();
        inflate(&mut bits, &mut out)?;
        let p = bits.pos;
        let trailer = data.get(p..p + 8).ok_or("missing gzip trailer")?;
        if u32::from_le_bytes(trailer[..4].try_into().unwrap()) != crc32(&out) {
            return Err("corrupt gzip stream does not have a matching checksum".into());
        }
        Ok(out)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn gzip_round_trips() {
            for n in [0usize, 1, 100, 65535, 65536, 200_000] {
                let data: Vec<u8> = (0..n).map(|k| (k * 7 % 251) as u8).collect();
                assert_eq!(decompress(&compress(&data)).unwrap(), data);
            }
        }

        #[test]
        fn gzip_reads_fixed_and_dynamic_huffman() {
            // zlib's level-6 deflate of "hello world" (fixed Huffman)
            let fixed = [0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x28, 0xcf, 0x2f, 0xca, 0x49, 0x01, 0, 0x85, 0x11, 0x4a, 0x0d, 0x0b, 0, 0, 0];
            assert_eq!(decompress(&fixed).unwrap(), b"hello world");
            // zlib's level-9 deflate of 3000 skewed letters (dynamic Huffman)
            let hex = "1f8b08000000000000ffed8dc10900410c026b5509db7f07978165df5780af302a194d323947e38c8eb41c2d7b736d40bfec7b2f9b2b38b0c9451f7ab313bbb0333bb10b3bbf3ff5d65b6fbdf5d65b6fbdf5d65bef2fef071f940fddb80b0000";
            let dynamic: Vec<u8> = (0..hex.len()).step_by(2).map(|k| u8::from_str_radix(&hex[k..k + 2], 16).unwrap()).collect();
            let expected: Vec<u8> = (0..3000usize).map(|k| b"abcdefgh"[if k % 5 != 0 { (k * k + 3 * k) % 8 } else { (k * 7) % 3 }]).collect();
            assert_eq!(decompress(&dynamic).unwrap(), expected);
        }
    }
}

/// Standard base64 (`base64`'s `BASE64_STANDARD`).
pub mod base64 {
    pub fn encode(data: &[u8]) -> String {
        crate::clifford::gltf::base64(data)
    }

    pub fn decode(text: &[u8]) -> Result<Vec<u8>, String> {
        fn val(c: u8) -> Result<u32, String> {
            Ok(match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                _ => return Err(format!("Invalid symbol {c}")),
            } as u32)
        }
        let trimmed = text.strip_suffix(b"==").or_else(|| text.strip_suffix(b"=")).unwrap_or(text);
        let mut out = Vec::with_capacity(trimmed.len() * 3 / 4);
        for chunk in trimmed.chunks(4) {
            let mut acc = 0u32;
            for &c in chunk {
                acc = acc << 6 | val(c)?;
            }
            match chunk.len() {
                4 => out.extend_from_slice(&[(acc >> 16) as u8, (acc >> 8) as u8, acc as u8]),
                3 => out.extend_from_slice(&[(acc >> 10) as u8, (acc >> 2) as u8]),
                2 => out.push((acc >> 4) as u8),
                _ => return Err("Invalid input length".into()),
            }
        }
        Ok(out)
    }
}

/// CBOR as `ciborium` 0.2 writes and reads `serde` values (`BenchmarkSuite`'s files): integers
/// in their shortest form, a float in the narrowest of f16, f32 and f64 that holds it exactly,
/// `None` as null, tuples and vectors as arrays, structs as maps keyed by field name.
pub mod cbor {
    #[derive(Clone, Debug, PartialEq)]
    pub enum Value {
        Null,
        Bool(bool),
        UInt(u64),
        NegInt(u64),
        Float(f64),
        Text(String),
        Array(Vec<Value>),
        Map(Vec<(Value, Value)>),
    }

    fn head(out: &mut Vec<u8>, major: u8, n: u64) {
        let m = major << 5;
        if n < 24 {
            out.push(m | n as u8);
        } else if n <= 0xff {
            out.extend_from_slice(&[m | 24, n as u8]);
        } else if n <= 0xffff {
            out.push(m | 25);
            out.extend_from_slice(&(n as u16).to_be_bytes());
        } else if n <= 0xffff_ffff {
            out.push(m | 26);
            out.extend_from_slice(&(n as u32).to_be_bytes());
        } else {
            out.push(m | 27);
            out.extend_from_slice(&n.to_be_bytes());
        }
    }

    /// `half`'s `f16::from_f64`, when converting back gives the same bits.
    fn exact_f16(x: f64) -> Option<u16> {
        let bits = x.to_bits();
        let sign = ((bits >> 48) & 0x8000) as u16;
        let exp = ((bits >> 52) & 0x7ff) as i32;
        let man = bits & ((1u64 << 52) - 1);
        if exp == 0x7ff {
            if man == 0 {
                return Some(sign | 0x7c00);
            }
            // NaN: half keeps the top ten mantissa bits and sets the quiet bit
            let m16 = (man >> 42) as u16 | 0x200;
            return if (m16 as u64) << 42 == man { Some(sign | 0x7c00 | m16) } else { None };
        }
        if exp == 0 && man == 0 {
            return Some(sign);
        }
        if exp == 0 {
            return None; // f64 subnormals are far below f16's range
        }
        let e = exp - 1023;
        if (-14..=15).contains(&e) {
            if man & ((1u64 << 42) - 1) != 0 {
                return None;
            }
            return Some(sign | (((e + 15) as u16) << 10) | (man >> 42) as u16);
        }
        if (-24..-14).contains(&e) {
            // subnormal f16: value = k * 2^-24
            let full = man | (1u64 << 52);
            let shift = 52 - (e + 24);
            if full & ((1u64 << shift) - 1) != 0 {
                return None;
            }
            return Some(sign | (full >> shift) as u16);
        }
        None
    }

    pub fn write(value: &Value, out: &mut Vec<u8>) {
        match value {
            Value::Null => out.push(0xf6),
            Value::Bool(b) => out.push(if *b { 0xf5 } else { 0xf4 }),
            Value::UInt(n) => head(out, 0, *n),
            Value::NegInt(n) => head(out, 1, *n),
            Value::Text(s) => {
                head(out, 3, s.len() as u64);
                out.extend_from_slice(s.as_bytes());
            }
            Value::Array(items) => {
                head(out, 4, items.len() as u64);
                for item in items {
                    write(item, out);
                }
            }
            Value::Map(entries) => {
                head(out, 5, entries.len() as u64);
                for (k, v) in entries {
                    write(k, out);
                    write(v, out);
                }
            }
            Value::Float(x) => {
                if let Some(h) = exact_f16(*x) {
                    out.push(0xf9);
                    out.extend_from_slice(&h.to_be_bytes());
                } else if (*x as f32) as f64 == *x || (x.is_nan() && ((*x as f32) as f64).to_bits() == x.to_bits()) {
                    out.push(0xfa);
                    out.extend_from_slice(&(*x as f32).to_be_bytes());
                } else {
                    out.push(0xfb);
                    out.extend_from_slice(&x.to_be_bytes());
                }
            }
        }
    }

    /// A value as `serde` hands it to `ciborium`.
    pub trait ToCbor {
        fn to_cbor(&self) -> Value;
    }
    /// A value as `serde` takes it from `ciborium`.
    pub trait FromCbor: Sized {
        fn from_cbor(value: &Value) -> Result<Self, String>;
    }
    impl ToCbor for usize {
        fn to_cbor(&self) -> Value {
            Value::UInt(*self as u64)
        }
    }
    impl FromCbor for usize {
        fn from_cbor(value: &Value) -> Result<Self, String> {
            match value {
                Value::UInt(n) => usize::try_from(*n).map_err(|e| e.to_string()),
                other => Err(format!("invalid type: {other:?}, expected usize")),
            }
        }
    }
    impl ToCbor for f64 {
        fn to_cbor(&self) -> Value {
            Value::Float(*self)
        }
    }
    impl FromCbor for f64 {
        fn from_cbor(value: &Value) -> Result<Self, String> {
            match value {
                Value::Float(x) => Ok(*x),
                Value::UInt(n) => Ok(*n as f64),
                Value::NegInt(n) => Ok(-1.0 - *n as f64),
                other => Err(format!("invalid type: {other:?}, expected f64")),
            }
        }
    }
    impl<T: ToCbor> ToCbor for Vec<T> {
        fn to_cbor(&self) -> Value {
            Value::Array(self.iter().map(ToCbor::to_cbor).collect())
        }
    }
    impl<T: FromCbor> FromCbor for Vec<T> {
        fn from_cbor(value: &Value) -> Result<Self, String> {
            match value {
                Value::Array(items) => items.iter().map(T::from_cbor).collect(),
                other => Err(format!("invalid type: {other:?}, expected a sequence")),
            }
        }
    }
    impl<T: ToCbor> ToCbor for Option<T> {
        fn to_cbor(&self) -> Value {
            self.as_ref().map_or(Value::Null, ToCbor::to_cbor)
        }
    }
    impl<T: FromCbor> FromCbor for Option<T> {
        fn from_cbor(value: &Value) -> Result<Self, String> {
            match value {
                Value::Null => Ok(None),
                other => T::from_cbor(other).map(Some),
            }
        }
    }
    impl<A: ToCbor, B: ToCbor> ToCbor for (A, B) {
        fn to_cbor(&self) -> Value {
            Value::Array(vec![self.0.to_cbor(), self.1.to_cbor()])
        }
    }
    impl<A: FromCbor, B: FromCbor> FromCbor for (A, B) {
        fn from_cbor(value: &Value) -> Result<Self, String> {
            match value {
                Value::Array(items) if items.len() == 2 => Ok((A::from_cbor(&items[0])?, B::from_cbor(&items[1])?)),
                other => Err(format!("invalid type: {other:?}, expected a tuple of size 2")),
            }
        }
    }
    impl<A: ToCbor, B: ToCbor, C: ToCbor> ToCbor for (A, B, C) {
        fn to_cbor(&self) -> Value {
            Value::Array(vec![self.0.to_cbor(), self.1.to_cbor(), self.2.to_cbor()])
        }
    }
    impl<A: FromCbor, B: FromCbor, C: FromCbor> FromCbor for (A, B, C) {
        fn from_cbor(value: &Value) -> Result<Self, String> {
            match value {
                Value::Array(items) if items.len() == 3 => Ok((A::from_cbor(&items[0])?, B::from_cbor(&items[1])?, C::from_cbor(&items[2])?)),
                other => Err(format!("invalid type: {other:?}, expected a tuple of size 3")),
            }
        }
    }

    impl<A: ToCbor, B: ToCbor, C: ToCbor, D: ToCbor> ToCbor for (A, B, C, D) {
        fn to_cbor(&self) -> Value {
            Value::Array(vec![self.0.to_cbor(), self.1.to_cbor(), self.2.to_cbor(), self.3.to_cbor()])
        }
    }

    /// A struct's fields, as `serde`'s derive writes them.
    pub fn fields(fields: &[(&str, Value)]) -> Value {
        Value::Map(fields.iter().map(|(k, v)| (Value::Text(k.to_string()), v.clone())).collect())
    }

    /// A struct's field by name (`serde`'s derive ignores unknown fields).
    pub fn field<T: FromCbor>(value: &Value, name: &str) -> Result<T, String> {
        match value {
            Value::Map(entries) => match entries.iter().find(|(k, _)| matches!(k, Value::Text(t) if t == name)) {
                Some((_, v)) => T::from_cbor(v),
                None => Err(format!("missing field `{name}`")),
            },
            other => Err(format!("invalid type: {other:?}, expected a map")),
        }
    }

    pub fn to_vec(value: &impl ToCbor) -> Vec<u8> {
        let mut out = Vec::new();
        write(&value.to_cbor(), &mut out);
        out
    }

    pub fn from_slice<T: FromCbor>(data: &[u8]) -> Result<T, String> {
        let mut pos = 0;
        T::from_cbor(&read(data, &mut pos)?)
    }

    fn f16_to_f64(h: u16) -> f64 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((h >> 10) & 0x1f) as i32;
        let man = (h & 0x3ff) as f64;
        sign * match exp {
            0 => man * 2f64.powi(-24),
            31 => {
                if man == 0.0 {
                    f64::INFINITY
                } else {
                    f64::NAN
                }
            }
            _ => (1.0 + man / 1024.0) * 2f64.powi(exp - 15),
        }
    }

    pub fn read(data: &[u8], pos: &mut usize) -> Result<Value, String> {
        let eof = || "unexpected end of CBOR data".to_string();
        let b = *data.get(*pos).ok_or_else(eof)?;
        *pos += 1;
        let major = b >> 5;
        let minor = b & 31;
        let mut arg = |n: usize| -> Result<u64, String> {
            let bytes = data.get(*pos..*pos + n).ok_or_else(eof)?;
            *pos += n;
            Ok(bytes.iter().fold(0u64, |a, &x| a << 8 | x as u64))
        };
        if major == 7 {
            return Ok(match minor {
                20 => Value::Bool(false),
                21 => Value::Bool(true),
                22 | 23 => Value::Null,
                25 => Value::Float(f16_to_f64(arg(2)? as u16)),
                26 => Value::Float(f32::from_bits(arg(4)? as u32) as f64),
                27 => Value::Float(f64::from_bits(arg(8)?)),
                _ => return Err(format!("unsupported CBOR simple value {minor}")),
            });
        }
        let n = match minor {
            0..=23 => minor as u64,
            24 => arg(1)?,
            25 => arg(2)?,
            26 => arg(4)?,
            27 => arg(8)?,
            _ => return Err("indefinite-length CBOR items are not supported".into()),
        };
        Ok(match major {
            0 => Value::UInt(n),
            1 => Value::NegInt(n),
            3 => {
                let bytes = data.get(*pos..*pos + n as usize).ok_or_else(eof)?;
                *pos += n as usize;
                Value::Text(String::from_utf8(bytes.to_vec()).map_err(|e| e.to_string())?)
            }
            4 => Value::Array((0..n).map(|_| read(data, pos)).collect::<Result<_, _>>()?),
            5 => Value::Map((0..n).map(|_| Ok((read(data, pos)?, read(data, pos)?))).collect::<Result<_, String>>()?),
            6 => read(data, pos)?, // tags are ignored
            _ => return Err(format!("unsupported CBOR major type {major}")),
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn bytes(v: Value) -> Vec<u8> {
            let mut out = vec![];
            write(&v, &mut out);
            out
        }

        #[test]
        fn floats_take_the_narrowest_exact_width() {
            assert_eq!(bytes(Value::Float(1.0)), [0xf9, 0x3c, 0x00]);
            assert_eq!(bytes(Value::Float(1.5)), [0xf9, 0x3e, 0x00]);
            assert_eq!(bytes(Value::Float(65504.0)), [0xf9, 0x7b, 0xff]);
            assert_eq!(bytes(Value::Float(5.960464477539063e-8)), [0xf9, 0x00, 0x01]);
            assert_eq!(bytes(Value::Float(100000.0)), [0xfa, 0x47, 0xc3, 0x50, 0x00]);
            assert_eq!(bytes(Value::Float(1.1)), [0xfb, 0x3f, 0xf1, 0x99, 0x99, 0x99, 0x99, 0x99, 0x9a]);
            assert_eq!(bytes(Value::Float(f64::INFINITY)), [0xf9, 0x7c, 0x00]);
            for x in [0.0, -2.5, 1e-7, 3.0e38, 6.1e-5, 1.0 / 3.0] {
                let b = bytes(Value::Float(x));
                assert_eq!(read(&b, &mut 0).unwrap(), Value::Float(x));
            }
        }

        #[test]
        fn integers_take_the_shortest_head() {
            assert_eq!(bytes(Value::UInt(23)), [0x17]);
            assert_eq!(bytes(Value::UInt(24)), [0x18, 24]);
            assert_eq!(bytes(Value::UInt(65536)).len(), 5);
        }
    }
}

/// The crates' names as MWPF's files use them.
pub mod prelude {
    pub(crate) use super::{base64, cbor, gzip, hashbrown, rand, itertools, lnexp, num_traits, parking_lot, prettytable, priority_queue, rand_xoshiro, serde_json};
    pub(crate) use crate::fusion::shim::json;
    pub(crate) use super::izip;
    #[cfg(feature = "python")]
    pub(crate) use pyo3::prelude::*;
}

