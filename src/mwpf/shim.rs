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
/// `GzDecoder`): this writes what `GzEncoder` writes at `Compression::default()`, byte for byte
/// (miniz_oxide's deflate, below), and reads any deflate stream.
pub mod gzip {
    /// `miniz_oxide` 0.8.0's deflate compressor (`deflate::core`), which `flate2` 1.0.35 (MWPF's
    /// `GzEncoder`) runs: at `Compression::default()`, level 6, lazy parsing with 128 probes, raw
    /// deflate. The parse, the block splits and the Huffman tables are miniz_oxide's, line for
    /// line, so the bytes are MWPF's; what is left out is what level 6 never runs (the level-1
    /// fast path, the zlib wrapper, RLE, filtered and forced raw or static blocks) and the
    /// streaming buffers, as the whole input is compressed at once and ends the stream (which,
    /// for its non-flushing writes, gives the bytes flate2's do). miniz_oxide is Copyright (c)
    /// 2017 Frommi, under the MIT license (or Zlib, or Apache-2.0).
    mod deflate {
        use std::cmp;

        /// The number of probes into the hash chains at level 6, as `NUM_PROBES[6]`.
        const LEVEL6_PROBES: u32 = 128;
        const MAX_SUPPORTED_HUFF_CODESIZE: usize = 32;
        const MAX_HUFF_SYMBOLS: usize = 288;
        const MAX_HUFF_TABLES: usize = 3;
        const MAX_HUFF_SYMBOLS_0: usize = 288;
        const MAX_HUFF_SYMBOLS_1: usize = 32;
        const MAX_HUFF_SYMBOLS_2: usize = 19;
        const LZ_DICT_SIZE: usize = 32_768;
        const LZ_DICT_SIZE_MASK: usize = (LZ_DICT_SIZE as u32 - 1) as usize;
        const MIN_MATCH_LEN: u8 = 3;
        const MAX_MATCH_LEN: usize = 258;
        const LZ_CODE_BUF_SIZE: usize = 64 * 1024;
        const LZ_DICT_FULL_SIZE: usize = LZ_DICT_SIZE + MAX_MATCH_LEN - 1 + 1;
        const LZ_HASH_BITS: i32 = 15;
        const LZ_HASH_SHIFT: i32 = (LZ_HASH_BITS + 2) / 3;
        const LZ_HASH_SIZE: usize = 1 << LZ_HASH_BITS;
        const HUFFMAN_LENGTH_ORDER: [u8; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
        const LITLEN_TABLE: usize = 0;
        const DIST_TABLE: usize = 1;
        const HUFF_CODES_TABLE: usize = 2;

        #[rustfmt::skip]
        const LEN_SYM: [u16; 256] = [
            257, 258, 259, 260, 261, 262, 263, 264, 265, 265, 266, 266, 267, 267, 268, 268, 269, 269, 269, 269, 270, 270, 270, 270, 271, 271, 271, 271, 272, 272, 272, 272,
            273, 273, 273, 273, 273, 273, 273, 273, 274, 274, 274, 274, 274, 274, 274, 274, 275, 275, 275, 275, 275, 275, 275, 275, 276, 276, 276, 276, 276, 276, 276, 276,
            277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278,
            279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280,
            281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281,
            282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282,
            283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283,
            284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 285,
        ];
        #[rustfmt::skip]
        const LEN_EXTRA: [u8; 256] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0,
        ];
        #[rustfmt::skip]
        const SMALL_DIST_SYM: [u8; 512] = [
            0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
            10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
            17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
            17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
            17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
        ];
        #[rustfmt::skip]
        const SMALL_DIST_EXTRA: [u8; 512] = [
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        ];
        #[rustfmt::skip]
        const LARGE_DIST_SYM: [u8; 128] = [
            0, 0, 18, 19, 20, 20, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25,
            26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
            28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
            29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
        ];
        #[rustfmt::skip]
        const LARGE_DIST_EXTRA: [u8; 128] = [
            0, 0, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        ];
        #[rustfmt::skip]
        const BITMASKS: [u32; 17] = [
            0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF, 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF,
        ];

        type Result<T> = core::result::Result<T, ()>;

        fn update_hash(current_hash: u16, byte: u8) -> u16 {
            ((current_hash << LZ_HASH_SHIFT) ^ u16::from(byte)) & (LZ_HASH_SIZE as u16 - 1)
        }

        fn memset<T: Copy>(slice: &mut [T], val: T) {
            for x in slice {
                *x = val
            }
        }

        const fn read_u16_le(slice: &[u8], pos: usize) -> u16 {
            slice[pos] as u16 | ((slice[pos + 1] as u16) << 8)
        }

        #[derive(Copy, Clone)]
        struct SymFreq {
            key: u16,
            sym_index: u16,
        }

        /// The output, a growing buffer in place of miniz_oxide's fixed one (`inner_pos` is its
        /// length); a block that is rewritten (as a stored or a static block) is truncated back.
        struct OutputBufferOxide {
            inner: Vec<u8>,
            bit_buffer: u32,
            bits_in: u32,
        }

        struct SavedOutputBufferOxide {
            pos: usize,
            bit_buffer: u32,
            bits_in: u32,
        }

        impl OutputBufferOxide {
            fn put_bits(&mut self, bits: u32, len: u32) {
                assert!(bits <= ((1u32 << len) - 1u32));
                self.bit_buffer |= bits << self.bits_in;
                self.bits_in += len;
                while self.bits_in >= 8 {
                    self.inner.push(self.bit_buffer as u8);
                    self.bit_buffer >>= 8;
                    self.bits_in -= 8;
                }
            }

            fn save(&self) -> SavedOutputBufferOxide {
                SavedOutputBufferOxide { pos: self.inner.len(), bit_buffer: self.bit_buffer, bits_in: self.bits_in }
            }

            fn load(&mut self, saved: SavedOutputBufferOxide) {
                self.inner.truncate(saved.pos);
                self.bit_buffer = saved.bit_buffer;
                self.bits_in = saved.bits_in;
            }

            fn pad_to_bytes(&mut self) {
                if self.bits_in != 0 {
                    let len = 8 - self.bits_in;
                    self.put_bits(0, len);
                }
            }
        }

        struct BitBuffer {
            bit_buffer: u64,
            bits_in: u32,
        }

        impl BitBuffer {
            fn put_fast(&mut self, bits: u64, len: u32) {
                self.bit_buffer |= bits << self.bits_in;
                self.bits_in += len;
            }

            fn flush(&mut self, output: &mut OutputBufferOxide) {
                let bytes = u64::to_le_bytes(self.bit_buffer);
                output.inner.extend_from_slice(&bytes[..(self.bits_in >> 3) as usize]);
                self.bit_buffer >>= self.bits_in & !7;
                self.bits_in &= 7;
            }
        }

        struct HuffmanOxide {
            count: [[u16; MAX_HUFF_SYMBOLS]; MAX_HUFF_TABLES],
            codes: [[u16; MAX_HUFF_SYMBOLS]; MAX_HUFF_TABLES],
            code_sizes: [[u8; MAX_HUFF_SYMBOLS]; MAX_HUFF_TABLES],
        }

        struct Rle {
            z_count: u32,
            repeat_count: u32,
            prev_code_size: u8,
        }

        impl Rle {
            fn prev_code_size(&mut self, packed_code_sizes: &mut [u8], packed_pos: &mut usize, h: &mut HuffmanOxide) -> Result<()> {
                let mut write = |buf| write(buf, packed_code_sizes, packed_pos);
                let counts = &mut h.count[HUFF_CODES_TABLE];
                if self.repeat_count != 0 {
                    if self.repeat_count < 3 {
                        counts[self.prev_code_size as usize] = counts[self.prev_code_size as usize].wrapping_add(self.repeat_count as u16);
                        let code = self.prev_code_size;
                        write(&[code, code, code][..self.repeat_count as usize])?;
                    } else {
                        counts[16] = counts[16].wrapping_add(1);
                        write(&[16, (self.repeat_count - 3) as u8][..])?;
                    }
                    self.repeat_count = 0;
                }
                Ok(())
            }

            fn zero_code_size(&mut self, packed_code_sizes: &mut [u8], packed_pos: &mut usize, h: &mut HuffmanOxide) -> Result<()> {
                let mut write = |buf| write(buf, packed_code_sizes, packed_pos);
                let counts = &mut h.count[HUFF_CODES_TABLE];
                if self.z_count != 0 {
                    if self.z_count < 3 {
                        counts[0] = counts[0].wrapping_add(self.z_count as u16);
                        write(&[0, 0, 0][..self.z_count as usize])?;
                    } else if self.z_count <= 10 {
                        counts[17] = counts[17].wrapping_add(1);
                        write(&[17, (self.z_count - 3) as u8][..])?;
                    } else {
                        counts[18] = counts[18].wrapping_add(1);
                        write(&[18, (self.z_count - 11) as u8][..])?;
                    }
                    self.z_count = 0;
                }
                Ok(())
            }
        }

        fn write(src: &[u8], dst: &mut [u8], dst_pos: &mut usize) -> Result<()> {
            match dst.get_mut(*dst_pos..*dst_pos + src.len()) {
                Some(s) => s.copy_from_slice(src),
                None => return Err(()),
            }
            *dst_pos += src.len();
            Ok(())
        }

        impl Default for HuffmanOxide {
            fn default() -> Self {
                HuffmanOxide { count: [[0; MAX_HUFF_SYMBOLS]; MAX_HUFF_TABLES], codes: [[0; MAX_HUFF_SYMBOLS]; MAX_HUFF_TABLES], code_sizes: [[0; MAX_HUFF_SYMBOLS]; MAX_HUFF_TABLES] }
            }
        }

        impl HuffmanOxide {
            fn radix_sort_symbols<'a>(symbols0: &'a mut [SymFreq], symbols1: &'a mut [SymFreq]) -> &'a mut [SymFreq] {
                let mut hist = [[0; 256]; 2];
                for freq in symbols0.iter() {
                    hist[0][(freq.key & 0xFF) as usize] += 1;
                    hist[1][((freq.key >> 8) & 0xFF) as usize] += 1;
                }
                let mut n_passes = 2;
                if symbols0.len() == hist[1][0] {
                    n_passes -= 1;
                }
                let mut current_symbols = symbols0;
                let mut new_symbols = symbols1;
                for (pass, hist_item) in hist.iter().enumerate().take(n_passes) {
                    let mut offsets = [0; 256];
                    let mut offset = 0;
                    for i in 0..256 {
                        offsets[i] = offset;
                        offset += hist_item[i];
                    }
                    for sym in current_symbols.iter() {
                        let j = ((sym.key >> (pass * 8)) & 0xFF) as usize;
                        new_symbols[offsets[j]] = *sym;
                        offsets[j] += 1;
                    }
                    std::mem::swap(&mut current_symbols, &mut new_symbols);
                }
                current_symbols
            }

            fn calculate_minimum_redundancy(symbols: &mut [SymFreq]) {
                match symbols.len() {
                    0 => (),
                    1 => symbols[0].key = 1,
                    n => {
                        symbols[0].key += symbols[1].key;
                        let mut root = 0;
                        let mut leaf = 2;
                        for next in 1..n - 1 {
                            if (leaf >= n) || (symbols[root].key < symbols[leaf].key) {
                                symbols[next].key = symbols[root].key;
                                symbols[root].key = next as u16;
                                root += 1;
                            } else {
                                symbols[next].key = symbols[leaf].key;
                                leaf += 1;
                            }
                            if (leaf >= n) || (root < next && symbols[root].key < symbols[leaf].key) {
                                symbols[next].key = symbols[next].key.wrapping_add(symbols[root].key);
                                symbols[root].key = next as u16;
                                root += 1;
                            } else {
                                symbols[next].key = symbols[next].key.wrapping_add(symbols[leaf].key);
                                leaf += 1;
                            }
                        }
                        symbols[n - 2].key = 0;
                        for next in (0..n - 2).rev() {
                            symbols[next].key = symbols[symbols[next].key as usize].key + 1;
                        }
                        let mut avbl = 1;
                        let mut used = 0;
                        let mut dpth = 0;
                        let mut root = (n - 2) as i32;
                        let mut next = (n - 1) as i32;
                        while avbl > 0 {
                            while (root >= 0) && (symbols[root as usize].key == dpth) {
                                used += 1;
                                root -= 1;
                            }
                            while avbl > used {
                                symbols[next as usize].key = dpth;
                                next -= 1;
                                avbl -= 1;
                            }
                            avbl = 2 * used;
                            dpth += 1;
                            used = 0;
                        }
                    }
                }
            }

            fn enforce_max_code_size(num_codes: &mut [i32], code_list_len: usize, max_code_size: usize) {
                if code_list_len <= 1 {
                    return;
                }
                num_codes[max_code_size] += num_codes[max_code_size + 1..].iter().sum::<i32>();
                let total = num_codes[1..=max_code_size].iter().rev().enumerate().fold(0u32, |total, (i, &x)| total + ((x as u32) << i));
                for _ in (1 << max_code_size)..total {
                    num_codes[max_code_size] -= 1;
                    for i in (1..max_code_size).rev() {
                        if num_codes[i] != 0 {
                            num_codes[i] -= 1;
                            num_codes[i + 1] += 2;
                            break;
                        }
                    }
                }
            }

            fn optimize_table(&mut self, table_num: usize, table_len: usize, code_size_limit: usize, static_table: bool) {
                let mut num_codes = [0i32; MAX_SUPPORTED_HUFF_CODESIZE + 1];
                let mut next_code = [0u32; MAX_SUPPORTED_HUFF_CODESIZE + 1];
                if static_table {
                    for &code_size in &self.code_sizes[table_num][..table_len] {
                        num_codes[code_size as usize] += 1;
                    }
                } else {
                    let mut symbols0 = [SymFreq { key: 0, sym_index: 0 }; MAX_HUFF_SYMBOLS];
                    let mut symbols1 = [SymFreq { key: 0, sym_index: 0 }; MAX_HUFF_SYMBOLS];
                    let mut num_used_symbols = 0;
                    for i in 0..table_len {
                        if self.count[table_num][i] != 0 {
                            symbols0[num_used_symbols] = SymFreq { key: self.count[table_num][i], sym_index: i as u16 };
                            num_used_symbols += 1;
                        }
                    }
                    let symbols = Self::radix_sort_symbols(&mut symbols0[..num_used_symbols], &mut symbols1[..num_used_symbols]);
                    Self::calculate_minimum_redundancy(symbols);
                    for symbol in symbols.iter() {
                        num_codes[symbol.key as usize] += 1;
                    }
                    Self::enforce_max_code_size(&mut num_codes, num_used_symbols, code_size_limit);
                    memset(&mut self.code_sizes[table_num][..], 0);
                    memset(&mut self.codes[table_num][..], 0);
                    let mut last = num_used_symbols;
                    for (i, &num_item) in num_codes.iter().enumerate().take(code_size_limit + 1).skip(1) {
                        let first = last - num_item as usize;
                        for symbol in &symbols[first..last] {
                            self.code_sizes[table_num][symbol.sym_index as usize] = i as u8;
                        }
                        last = first;
                    }
                }
                let mut j = 0;
                next_code[1] = 0;
                for i in 2..=code_size_limit {
                    j = (j + num_codes[i - 1]) << 1;
                    next_code[i] = j as u32;
                }
                for (&code_size, huff_code) in self.code_sizes[table_num].iter().take(table_len).zip(self.codes[table_num].iter_mut().take(table_len)) {
                    if code_size == 0 {
                        continue;
                    }
                    let mut code = next_code[code_size as usize];
                    next_code[code_size as usize] += 1;
                    let mut rev_code = 0;
                    for _ in 0..code_size {
                        rev_code = (rev_code << 1) | (code & 1);
                        code >>= 1;
                    }
                    *huff_code = rev_code as u16;
                }
            }

            fn start_static_block(&mut self, output: &mut OutputBufferOxide) {
                memset(&mut self.code_sizes[LITLEN_TABLE][0..144], 8);
                memset(&mut self.code_sizes[LITLEN_TABLE][144..256], 9);
                memset(&mut self.code_sizes[LITLEN_TABLE][256..280], 7);
                memset(&mut self.code_sizes[LITLEN_TABLE][280..288], 8);
                memset(&mut self.code_sizes[DIST_TABLE][..32], 5);
                self.optimize_table(LITLEN_TABLE, 288, 15, true);
                self.optimize_table(DIST_TABLE, 32, 15, true);
                output.put_bits(0b01, 2)
            }

            fn start_dynamic_block(&mut self, output: &mut OutputBufferOxide) -> Result<()> {
                // There will always be one, and only one end of block code.
                self.count[0][256] = 1;
                self.optimize_table(0, MAX_HUFF_SYMBOLS_0, 15, false);
                self.optimize_table(1, MAX_HUFF_SYMBOLS_1, 15, false);
                let num_lit_codes = 286 - self.code_sizes[0][257..286].iter().rev().take_while(|&x| *x == 0).count();
                let num_dist_codes = 30 - self.code_sizes[1][1..30].iter().rev().take_while(|&x| *x == 0).count();
                let mut code_sizes_to_pack = [0u8; MAX_HUFF_SYMBOLS_0 + MAX_HUFF_SYMBOLS_1];
                let mut packed_code_sizes = [0u8; MAX_HUFF_SYMBOLS_0 + MAX_HUFF_SYMBOLS_1];
                let total_code_sizes_to_pack = num_lit_codes + num_dist_codes;
                code_sizes_to_pack[..num_lit_codes].copy_from_slice(&self.code_sizes[0][..num_lit_codes]);
                code_sizes_to_pack[num_lit_codes..total_code_sizes_to_pack].copy_from_slice(&self.code_sizes[1][..num_dist_codes]);
                let mut rle = Rle { z_count: 0, repeat_count: 0, prev_code_size: 0xFF };
                memset(&mut self.count[HUFF_CODES_TABLE][..MAX_HUFF_SYMBOLS_2], 0);
                let mut packed_pos = 0;
                for &code_size in &code_sizes_to_pack[..total_code_sizes_to_pack] {
                    if code_size == 0 {
                        rle.prev_code_size(&mut packed_code_sizes, &mut packed_pos, self)?;
                        rle.z_count += 1;
                        if rle.z_count == 138 {
                            rle.zero_code_size(&mut packed_code_sizes, &mut packed_pos, self)?;
                        }
                    } else {
                        rle.zero_code_size(&mut packed_code_sizes, &mut packed_pos, self)?;
                        if code_size != rle.prev_code_size {
                            rle.prev_code_size(&mut packed_code_sizes, &mut packed_pos, self)?;
                            self.count[HUFF_CODES_TABLE][code_size as usize] = self.count[HUFF_CODES_TABLE][code_size as usize].wrapping_add(1);
                            write(&[code_size], &mut packed_code_sizes, &mut packed_pos)?;
                        } else {
                            rle.repeat_count += 1;
                            if rle.repeat_count == 6 {
                                rle.prev_code_size(&mut packed_code_sizes, &mut packed_pos, self)?;
                            }
                        }
                    }
                    rle.prev_code_size = code_size;
                }
                if rle.repeat_count != 0 {
                    rle.prev_code_size(&mut packed_code_sizes, &mut packed_pos, self)?;
                } else {
                    rle.zero_code_size(&mut packed_code_sizes, &mut packed_pos, self)?;
                }
                self.optimize_table(2, MAX_HUFF_SYMBOLS_2, 7, false);
                output.put_bits(2, 2);
                output.put_bits((num_lit_codes - 257) as u32, 5);
                output.put_bits((num_dist_codes - 1) as u32, 5);
                let mut num_bit_lengths = 18 - HUFFMAN_LENGTH_ORDER.iter().rev().take_while(|&swizzle| self.code_sizes[HUFF_CODES_TABLE][*swizzle as usize] == 0).count();
                num_bit_lengths = cmp::max(4, num_bit_lengths + 1);
                output.put_bits(num_bit_lengths as u32 - 4, 4);
                for &swizzle in &HUFFMAN_LENGTH_ORDER[..num_bit_lengths] {
                    output.put_bits(u32::from(self.code_sizes[HUFF_CODES_TABLE][swizzle as usize]), 3);
                }
                let mut packed_code_size_index = 0;
                while packed_code_size_index < packed_pos {
                    let code = packed_code_sizes[packed_code_size_index] as usize;
                    packed_code_size_index += 1;
                    assert!(code < MAX_HUFF_SYMBOLS_2);
                    output.put_bits(u32::from(self.codes[HUFF_CODES_TABLE][code]), u32::from(self.code_sizes[HUFF_CODES_TABLE][code]));
                    if code >= 16 {
                        output.put_bits(u32::from(packed_code_sizes[packed_code_size_index]), [2, 3, 7][code - 16]);
                        packed_code_size_index += 1;
                    }
                }
                Ok(())
            }
        }

        struct DictOxide {
            /// The maximum number of checks in the hash chain, for the initial and the lazy match.
            max_probes: [u32; 2],
            dict: Vec<u8>,
            next: Vec<u16>,
            hash: Vec<u16>,
            code_buf_dict_pos: usize,
            lookahead_size: usize,
            lookahead_pos: usize,
            size: usize,
        }

        impl DictOxide {
            fn new(flags: u32) -> Self {
                DictOxide {
                    max_probes: [1 + ((flags & 0xFFF) + 2) / 3, 1 + (((flags & 0xFFF) >> 2) + 2) / 3],
                    dict: vec![0; LZ_DICT_FULL_SIZE],
                    next: vec![0; LZ_DICT_SIZE],
                    hash: vec![0; LZ_DICT_SIZE],
                    code_buf_dict_pos: 0,
                    lookahead_size: 0,
                    lookahead_pos: 0,
                    size: 0,
                }
            }

            fn read_unaligned_u64(&self, pos: usize) -> u64 {
                let pos = pos & LZ_DICT_SIZE_MASK;
                let bytes: [u8; 8] = self.dict[pos..pos + 8].try_into().unwrap();
                u64::from_le_bytes(bytes)
            }

            fn read_as_u16(&self, pos: usize) -> u16 {
                read_u16_le(&self.dict[..], pos)
            }

            /// A match for the data at `lookahead_pos` longer than `match_len`, as (distance,
            /// length); the inputs back if there is none.
            fn find_match(&self, lookahead_pos: usize, max_dist: usize, max_match_len: u32, mut match_dist: u32, mut match_len: u32) -> (u32, u32) {
                let max_match_len = cmp::min(MAX_MATCH_LEN as u32, max_match_len);
                match_len = cmp::max(match_len, 1);
                let pos = lookahead_pos & LZ_DICT_SIZE_MASK;
                let mut probe_pos = pos;
                let mut num_probes_left = self.max_probes[(match_len >= 32) as usize];
                if max_match_len <= match_len {
                    return (match_dist, match_len);
                }
                let mut c01: u16 = self.read_as_u16(pos + match_len as usize - 1);
                let s01: u16 = self.read_as_u16(pos);
                'outer: loop {
                    let mut dist;
                    'found: loop {
                        num_probes_left -= 1;
                        if num_probes_left == 0 {
                            return (match_dist, match_len);
                        }
                        for _ in 0..3 {
                            let next_probe_pos = self.next[probe_pos] as usize;
                            dist = (lookahead_pos - next_probe_pos) & 0xFFFF;
                            if next_probe_pos == 0 || dist > max_dist {
                                return (match_dist, match_len);
                            }
                            probe_pos = next_probe_pos & LZ_DICT_SIZE_MASK;
                            if self.read_as_u16(probe_pos + match_len as usize - 1) == c01 {
                                break 'found;
                            }
                        }
                    }
                    if dist == 0 {
                        return (match_dist, match_len);
                    }
                    if self.read_as_u16(probe_pos) != s01 {
                        continue;
                    }
                    let mut p = pos + 2;
                    let mut q = probe_pos + 2;
                    for _ in 0..32 {
                        let p_data: u64 = self.read_unaligned_u64(p);
                        let q_data: u64 = self.read_unaligned_u64(q);
                        let xor_data = p_data ^ q_data;
                        if xor_data == 0 {
                            p += 8;
                            q += 8;
                        } else {
                            let trailing = xor_data.trailing_zeros();
                            let probe_len = p - pos + (trailing as usize >> 3);
                            if probe_len > match_len as usize {
                                match_dist = dist as u32;
                                match_len = cmp::min(max_match_len, probe_len as u32);
                                if match_len == max_match_len {
                                    return (match_dist, match_len);
                                }
                                c01 = self.read_as_u16(pos + match_len as usize - 1)
                            }
                            continue 'outer;
                        }
                    }
                    return (dist as u32, cmp::min(max_match_len, MAX_MATCH_LEN as u32));
                }
            }
        }

        struct LZOxide {
            codes: Vec<u8>,
            code_position: usize,
            flag_position: usize,
            total_bytes: u32,
            num_flags_left: u32,
        }

        impl LZOxide {
            fn new() -> Self {
                LZOxide { codes: vec![0; LZ_CODE_BUF_SIZE], code_position: 1, flag_position: 0, total_bytes: 0, num_flags_left: 8 }
            }

            fn write_code(&mut self, val: u8) {
                self.codes[usize::from(self.code_position as u16)] = val;
                self.code_position += 1;
            }

            fn init_flag(&mut self) {
                if self.num_flags_left == 8 {
                    *self.get_flag() = 0;
                    self.code_position -= 1;
                } else {
                    *self.get_flag() >>= self.num_flags_left;
                }
            }

            fn get_flag(&mut self) -> &mut u8 {
                &mut self.codes[usize::from(self.flag_position as u16)]
            }

            fn plant_flag(&mut self) {
                self.flag_position = self.code_position;
                self.code_position += 1;
            }

            fn consume_flag(&mut self) {
                self.num_flags_left -= 1;
                if self.num_flags_left == 0 {
                    self.num_flags_left = 8;
                    self.plant_flag();
                }
            }
        }

        struct CompressorOxide {
            lz: LZOxide,
            huff: Box<HuffmanOxide>,
            dict: DictOxide,
            output: OutputBufferOxide,
        }

        fn compress_lz_codes(huff: &HuffmanOxide, output: &mut OutputBufferOxide, lz_code_buf: &[u8]) -> Result<bool> {
            let mut flags = 1;
            let mut bb = BitBuffer { bit_buffer: u64::from(output.bit_buffer), bits_in: output.bits_in };
            let mut i: usize = 0;
            while i < lz_code_buf.len() {
                if flags == 1 {
                    flags = u32::from(lz_code_buf[i]) | 0x100;
                    i += 1;
                }
                if flags & 1 == 1 {
                    // a length code
                    flags >>= 1;
                    let sym;
                    let num_extra_bits;
                    let match_len = lz_code_buf[i] as usize;
                    let match_dist = read_u16_le(lz_code_buf, i + 1);
                    i += 3;
                    bb.put_fast(u64::from(huff.codes[0][LEN_SYM[match_len] as usize]), u32::from(huff.code_sizes[0][LEN_SYM[match_len] as usize]));
                    bb.put_fast(match_len as u64 & u64::from(BITMASKS[LEN_EXTRA[match_len] as usize]), u32::from(LEN_EXTRA[match_len]));
                    if match_dist < 512 {
                        sym = SMALL_DIST_SYM[match_dist as usize] as usize;
                        num_extra_bits = SMALL_DIST_EXTRA[match_dist as usize] as usize;
                    } else {
                        sym = LARGE_DIST_SYM[(match_dist >> 8) as usize] as usize;
                        num_extra_bits = LARGE_DIST_EXTRA[(match_dist >> 8) as usize] as usize;
                    }
                    bb.put_fast(u64::from(huff.codes[1][sym]), u32::from(huff.code_sizes[1][sym]));
                    bb.put_fast(u64::from(match_dist) & u64::from(BITMASKS[num_extra_bits]), num_extra_bits as u32);
                } else {
                    // literals
                    for _ in 0..3 {
                        flags >>= 1;
                        let lit = lz_code_buf[i];
                        i += 1;
                        bb.put_fast(u64::from(huff.codes[0][lit as usize]), u32::from(huff.code_sizes[0][lit as usize]));
                        if flags & 1 == 1 || i >= lz_code_buf.len() {
                            break;
                        }
                    }
                }
                bb.flush(output);
            }
            output.bits_in = 0;
            output.bit_buffer = 0;
            while bb.bits_in != 0 {
                let n = cmp::min(bb.bits_in, 16);
                output.put_bits(bb.bit_buffer as u32 & BITMASKS[n as usize], n);
                bb.bit_buffer >>= n;
                bb.bits_in -= n;
            }
            // the end of block symbol
            output.put_bits(u32::from(huff.codes[0][256]), u32::from(huff.code_sizes[0][256]));
            Ok(true)
        }

        fn compress_block(huff: &mut HuffmanOxide, output: &mut OutputBufferOxide, lz: &LZOxide, static_block: bool) -> Result<bool> {
            if static_block {
                huff.start_static_block(output);
            } else {
                huff.start_dynamic_block(output)?;
            }
            compress_lz_codes(huff, output, &lz.codes[..lz.code_position])
        }

        fn flush_block(d: &mut CompressorOxide, finish: bool) -> Result<()> {
            let output = &mut d.output;
            d.lz.init_flag();
            // the block header
            output.put_bits(finish as u32, 1);
            let saved_buffer = output.save();
            let use_static = d.lz.total_bytes < 48;
            let comp_success = compress_block(&mut d.huff, output, &d.lz, use_static)?;
            // A stored block where the compressed one would take as much space as the data.
            let expanded = (d.lz.total_bytes > 32) && (output.inner.len() - saved_buffer.pos + 1 >= (d.lz.total_bytes as usize)) && (d.dict.lookahead_pos - d.dict.code_buf_dict_pos <= d.dict.size);
            if expanded {
                output.load(saved_buffer);
                output.put_bits(0, 2);
                output.pad_to_bytes();
                output.put_bits(d.lz.total_bytes & 0xFFFF, 16);
                output.put_bits(!d.lz.total_bytes & 0xFFFF, 16);
                for i in 0..d.lz.total_bytes {
                    let pos = (d.dict.code_buf_dict_pos + i as usize) & LZ_DICT_SIZE_MASK;
                    output.put_bits(u32::from(d.dict.dict[pos]), 8);
                }
            } else if !comp_success {
                output.load(saved_buffer);
                compress_block(&mut d.huff, output, &d.lz, true)?;
            }
            if finish {
                output.pad_to_bytes();
            }
            memset(&mut d.huff.count[0][..MAX_HUFF_SYMBOLS_0], 0);
            memset(&mut d.huff.count[1][..MAX_HUFF_SYMBOLS_1], 0);
            d.lz.code_position = 1;
            d.lz.flag_position = 0;
            d.lz.num_flags_left = 8;
            d.dict.code_buf_dict_pos += d.lz.total_bytes as usize;
            d.lz.total_bytes = 0;
            Ok(())
        }

        fn record_literal(h: &mut HuffmanOxide, lz: &mut LZOxide, lit: u8) {
            lz.total_bytes += 1;
            lz.write_code(lit);
            *lz.get_flag() >>= 1;
            lz.consume_flag();
            h.count[0][lit as usize] += 1;
        }

        fn record_match(h: &mut HuffmanOxide, lz: &mut LZOxide, mut match_len: u32, mut match_dist: u32) {
            lz.total_bytes += match_len;
            match_dist -= 1;
            match_len -= u32::from(MIN_MATCH_LEN);
            lz.write_code(match_len as u8);
            lz.write_code(match_dist as u8);
            lz.write_code((match_dist >> 8) as u8);
            *lz.get_flag() >>= 1;
            *lz.get_flag() |= 0x80;
            lz.consume_flag();
            let symbol = if match_dist < 512 { SMALL_DIST_SYM[match_dist as usize] } else { LARGE_DIST_SYM[((match_dist >> 8) & 127) as usize] } as usize;
            h.count[1][symbol] += 1;
            h.count[0][LEN_SYM[usize::from(match_len as u8)] as usize] += 1;
        }

        /// `compress_normal` with the whole input and `TDEFLFlush::Finish`: miniz_oxide's lazy
        /// parse, flushing a block when its code buffer fills.
        fn compress_normal(d: &mut CompressorOxide, in_buf: &[u8]) -> Result<()> {
            let mut src_pos = 0;
            let mut lookahead_size = d.dict.lookahead_size;
            let mut lookahead_pos = d.dict.lookahead_pos;
            let mut saved_lit = 0u8;
            let mut saved_match_dist = 0u32;
            let mut saved_match_len = 0u32;
            while src_pos < in_buf.len() || lookahead_size != 0 {
                let src_buf_left = in_buf.len() - src_pos;
                let num_bytes_to_process = cmp::min(src_buf_left, MAX_MATCH_LEN - lookahead_size);
                if lookahead_size + d.dict.size >= usize::from(MIN_MATCH_LEN) - 1 && num_bytes_to_process > 0 {
                    let dictb = &mut d.dict;
                    let mut dst_pos = (lookahead_pos + lookahead_size) & LZ_DICT_SIZE_MASK;
                    let mut ins_pos = lookahead_pos + lookahead_size - 2;
                    // the hash starts from the first two bytes
                    let mut hash = update_hash(u16::from(dictb.dict[ins_pos & LZ_DICT_SIZE_MASK]), dictb.dict[(ins_pos + 1) & LZ_DICT_SIZE_MASK]);
                    lookahead_size += num_bytes_to_process;
                    for &c in &in_buf[src_pos..src_pos + num_bytes_to_process] {
                        dictb.dict[dst_pos] = c;
                        if dst_pos < MAX_MATCH_LEN - 1 {
                            dictb.dict[LZ_DICT_SIZE + dst_pos] = c;
                        }
                        hash = update_hash(hash, c);
                        dictb.next[ins_pos & LZ_DICT_SIZE_MASK] = dictb.hash[hash as usize];
                        dictb.hash[hash as usize] = ins_pos as u16;
                        dst_pos = (dst_pos + 1) & LZ_DICT_SIZE_MASK;
                        ins_pos += 1;
                    }
                    src_pos += num_bytes_to_process;
                } else {
                    let dictb = &mut d.dict;
                    for &c in &in_buf[src_pos..src_pos + num_bytes_to_process] {
                        let dst_pos = (lookahead_pos + lookahead_size) & LZ_DICT_SIZE_MASK;
                        dictb.dict[dst_pos] = c;
                        if dst_pos < MAX_MATCH_LEN - 1 {
                            dictb.dict[LZ_DICT_SIZE + dst_pos] = c;
                        }
                        lookahead_size += 1;
                        if lookahead_size + dictb.size >= usize::from(MIN_MATCH_LEN) {
                            let ins_pos = lookahead_pos + lookahead_size - 3;
                            let hash = ((u32::from(dictb.dict[ins_pos & LZ_DICT_SIZE_MASK]) << (LZ_HASH_SHIFT * 2)) ^ ((u32::from(dictb.dict[(ins_pos + 1) & LZ_DICT_SIZE_MASK]) << LZ_HASH_SHIFT) ^ u32::from(c))) & (LZ_HASH_SIZE as u32 - 1);
                            dictb.next[ins_pos & LZ_DICT_SIZE_MASK] = dictb.hash[hash as usize];
                            dictb.hash[hash as usize] = ins_pos as u16;
                        }
                    }
                    src_pos += num_bytes_to_process;
                }
                d.dict.size = cmp::min(LZ_DICT_SIZE - lookahead_size, d.dict.size);
                let mut len_to_move = 1;
                let mut cur_match_len = if saved_match_len != 0 { saved_match_len } else { u32::from(MIN_MATCH_LEN) - 1 };
                let cur_pos = lookahead_pos & LZ_DICT_SIZE_MASK;
                // a match for the bytes at the current position
                let (mut cur_match_dist, found_len) = d.dict.find_match(lookahead_pos, d.dict.size, lookahead_size as u32, 0, cur_match_len);
                cur_match_len = found_len;
                let far_and_small = cur_match_len == u32::from(MIN_MATCH_LEN) && cur_match_dist >= 8 * 1024;
                if far_and_small || cur_pos == cur_match_dist as usize {
                    cur_match_dist = 0;
                    cur_match_len = 0;
                }
                if saved_match_len != 0 {
                    if cur_match_len > saved_match_len {
                        record_literal(&mut d.huff, &mut d.lz, saved_lit);
                        if cur_match_len >= 128 {
                            record_match(&mut d.huff, &mut d.lz, cur_match_len, cur_match_dist);
                            saved_match_len = 0;
                            len_to_move = cur_match_len as usize;
                        } else {
                            saved_lit = d.dict.dict[cur_pos];
                            saved_match_dist = cur_match_dist;
                            saved_match_len = cur_match_len;
                        }
                    } else {
                        record_match(&mut d.huff, &mut d.lz, saved_match_len, saved_match_dist);
                        len_to_move = (saved_match_len - 1) as usize;
                        saved_match_len = 0;
                    }
                } else if cur_match_dist == 0 {
                    record_literal(&mut d.huff, &mut d.lz, d.dict.dict[cmp::min(cur_pos, d.dict.dict.len() - 1)]);
                } else if cur_match_len >= 128 {
                    record_match(&mut d.huff, &mut d.lz, cur_match_len, cur_match_dist);
                    len_to_move = cur_match_len as usize;
                } else {
                    // lazy matching: look for a better match at the next byte
                    saved_lit = d.dict.dict[cmp::min(cur_pos, d.dict.dict.len() - 1)];
                    saved_match_dist = cur_match_dist;
                    saved_match_len = cur_match_len;
                }
                lookahead_pos += len_to_move;
                assert!(lookahead_size >= len_to_move);
                lookahead_size -= len_to_move;
                d.dict.size = cmp::min(d.dict.size + len_to_move, LZ_DICT_SIZE);
                let lz_buf_tight = d.lz.code_position > LZ_CODE_BUF_SIZE - 8;
                let fat = ((d.lz.code_position * 115) >> 7) >= d.lz.total_bytes as usize;
                let fat_or_raw = (d.lz.total_bytes > 31 * 1024) && fat;
                if lz_buf_tight || fat_or_raw {
                    // flush_block reads these
                    d.dict.lookahead_size = lookahead_size;
                    d.dict.lookahead_pos = lookahead_pos;
                    flush_block(d, false)?;
                }
            }
            d.dict.lookahead_size = lookahead_size;
            d.dict.lookahead_pos = lookahead_pos;
            Ok(())
        }

        /// `data` as a raw deflate stream, the bytes miniz_oxide 0.8.0 writes at level 6.
        pub fn deflate(data: &[u8]) -> Vec<u8> {
            let mut d = CompressorOxide {
                lz: LZOxide::new(),
                huff: Box::default(),
                dict: DictOxide::new(LEVEL6_PROBES),
                output: OutputBufferOxide { inner: Vec::with_capacity(data.len() / 2 + 64), bit_buffer: 0, bits_in: 0 },
            };
            compress_normal(&mut d, data).expect("deflate code lengths fit their buffer");
            flush_block(&mut d, true).expect("deflate code lengths fit their buffer");
            d.output.inner
        }
    }

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
        // flate2's header: no name, no mtime, no extra flags at level 6, OS byte 255
        let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255];
        out.extend_from_slice(&deflate::deflate(data));
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
        fn gzip_writes_mwpfs_bytes() {
            // MWPF 0.2.12's HTMLExport.compress_content("hello world " * 10), base64-decoded
            let hex = "1f8b08000000000000ffbdc6b10900000803c155329c428a87808debbb85ddb921da0c253ffa002cbc940178000000";
            let expected: Vec<u8> = (0..hex.len()).step_by(2).map(|k| u8::from_str_radix(&hex[k..k + 2], 16).unwrap()).collect();
            assert_eq!(compress("hello world ".repeat(10).as_bytes()), expected);
            // an empty stream is one static block holding only its end
            assert_eq!(compress(b""), [0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
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

