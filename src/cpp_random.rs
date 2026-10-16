//! C++'s random engines and `std::shuffle`, as each standard library implements them.
//!
//! WHY THIS EXISTS
//! ---------------
//! `ldpc`'s random serial BP schedule shuffles its order with `std::shuffle` over an
//! `std::mt19937`, and its soft-information decoder with `std::default_random_engine`. The
//! engines are fixed by the C++ standard; `std::shuffle`, `uniform_int_distribution` and
//! `default_random_engine` are not, and the three standard libraries differ. Reproducing a
//! schedule therefore means reproducing the library the reference was compiled against:
//! libc++ (macOS), libstdc++ (Linux) or Microsoft's (Windows). `Library::NATIVE` is the one
//! `ldpc`'s wheels for this platform use.

/// A uniform random bit generator: `next` in `[MIN, MAX]`.
pub trait Urbg {
    const MIN: u64;
    const MAX: u64;
    fn next(&mut self) -> u64;
}

/// `std::mt19937`.
#[derive(Clone)]
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Mt19937 {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253u32.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { state, index: 624 }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                let mut v = self.state[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.state[i] = v;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

impl Urbg for Mt19937 {
    const MIN: u64 = 0;
    const MAX: u64 = u32::MAX as u64;
    fn next(&mut self) -> u64 {
        u64::from(self.next_u32())
    }
}

/// `std::linear_congruential_engine<uint_fast32_t, A, 0, 2147483647>`: `minstd_rand0` (A =
/// 16807) and `minstd_rand` (A = 48271).
#[derive(Clone)]
pub struct MinStd<const A: u64> {
    x: u64,
}

impl<const A: u64> MinStd<A> {
    pub fn new(seed: u64) -> Self {
        let x = seed % 2_147_483_647;
        MinStd { x: if x == 0 { 1 } else { x } }
    }
}

impl<const A: u64> Urbg for MinStd<A> {
    const MIN: u64 = 1;
    const MAX: u64 = 2_147_483_646;
    fn next(&mut self) -> u64 {
        self.x = self.x * A % 2_147_483_647;
        self.x
    }
}

/// A C++ standard library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Library {
    LibCxx,
    LibStdCxx,
    Msvc,
}

impl Library {
    /// The library `ldpc`'s wheels for this platform are built against.
    pub const NATIVE: Library = if cfg!(any(target_os = "macos", target_os = "ios")) {
        Library::LibCxx
    } else if cfg!(target_os = "windows") {
        Library::Msvc
    } else {
        Library::LibStdCxx
    };

    /// `std::uniform_int_distribution<T>{a, b}(g)` for an unsigned 64-bit `T` (as `shuffle`
    /// uses it).
    pub fn uniform<G: Urbg>(self, g: &mut G, a: u64, b: u64) -> u64 {
        match self {
            Library::LibCxx => libcxx_uniform(g, a, b),
            Library::LibStdCxx => libstdcxx_uniform(g, a, b),
            Library::Msvc => a + msvc_rng_from_urng(g, b - a + 1),
        }
    }

    /// `std::shuffle(v.begin(), v.end(), g)`.
    pub fn shuffle<T, G: Urbg>(self, v: &mut [T], g: &mut G) {
        let n = v.len();
        match self {
            Library::LibCxx => {
                if n > 1 {
                    let mut d = n as u64 - 1;
                    for first in 0..n - 1 {
                        let i = libcxx_uniform(g, 0, d) as usize;
                        if i != 0 {
                            v.swap(first, first + i);
                        }
                        d -= 1;
                    }
                }
            }
            Library::LibStdCxx => {
                if n == 0 {
                    return;
                }
                let urngrange = G::MAX - G::MIN;
                let urange = n as u64;
                if urngrange / urange >= urange {
                    // Two swap positions from each draw.
                    let mut i = 1;
                    if urange % 2 == 0 {
                        let j = libstdcxx_uniform(g, 0, 1) as usize;
                        v.swap(i, j);
                        i += 1;
                    }
                    while i != n {
                        let swap_range = i as u64 + 1;
                        let x = libstdcxx_uniform(g, 0, swap_range * (swap_range + 1) - 1);
                        let (p0, p1) = ((x / (swap_range + 1)) as usize, (x % (swap_range + 1)) as usize);
                        v.swap(i, p0);
                        i += 1;
                        v.swap(i, p1);
                        i += 1;
                    }
                    return;
                }
                for i in 1..n {
                    let j = libstdcxx_uniform(g, 0, i as u64) as usize;
                    v.swap(i, j);
                }
            }
            Library::Msvc => {
                for target in 1..n {
                    let off = msvc_rng_from_urng(g, target as u64 + 1) as usize;
                    if off != target {
                        v.swap(target, off);
                    }
                }
            }
        }
    }
}

/// `std::mt19937_64`, whose outputs the C++ standard fixes.
#[derive(Clone)]
pub struct Mt19937_64 {
    state: [u64; 312],
    index: usize,
}

impl Mt19937_64 {
    pub fn new(seed: u64) -> Mt19937_64 {
        let mut state = [0u64; 312];
        state[0] = seed;
        for i in 1..312 {
            state[i] = 6364136223846793005u64.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 62)).wrapping_add(i as u64);
        }
        Mt19937_64 { state, index: 312 }
    }

    pub fn next_u64(&mut self) -> u64 {
        const UPPER: u64 = 0xFFFF_FFFF_8000_0000;
        const LOWER: u64 = 0x7FFF_FFFF;
        if self.index >= 312 {
            for i in 0..312 {
                let x = (self.state[i] & UPPER) | (self.state[(i + 1) % 312] & LOWER);
                let mut xa = x >> 1;
                if x & 1 == 1 {
                    xa ^= 0xB502_6F5A_A966_19E9;
                }
                self.state[i] = self.state[(i + 156) % 312] ^ xa;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= (y >> 29) & 0x5555_5555_5555_5555;
        y ^= (y << 17) & 0x71D6_7FFF_EDA6_0000;
        y ^= (y << 37) & 0xFFF7_EEE0_0000_0000;
        y ^ (y >> 43)
    }
}

impl Urbg for Mt19937_64 {
    const MIN: u64 = 0;
    const MAX: u64 = u64::MAX;
    fn next(&mut self) -> u64 {
        self.next_u64()
    }
}

/// `a·b + c` as a C++ build for this machine computes `c + a * b` or `a * b + c` written in one
/// expression: contracted into a fused multiply-add on aarch64 (clang and gcc do there), not
/// on x86-64.
pub fn contracted(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(target_arch = "aarch64") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

impl Library {
    /// `std::generate_canonical<double, 53>(g)`: a draw in [0, 1) from as many engine outputs
    /// as cover 53 bits.
    pub fn canonical<G: Urbg>(self, g: &mut G) -> f64 {
        let range = (G::MAX - G::MIN) as f64 + 1.0; // 2^32 or 2^64
        let bits = range.log2() as u32;
        let k = 53u32.div_ceil(bits).max(1);
        match self {
            Library::LibCxx => {
                let mut sp = (g.next() - G::MIN) as f64;
                let mut base = range;
                for _ in 1..k {
                    sp += (g.next() - G::MIN) as f64 * base;
                    base *= range;
                }
                sp / base
            }
            Library::LibStdCxx | Library::Msvc => {
                let (mut sum, mut tmp) = (0.0f64, 1.0f64);
                for _ in 0..k {
                    sum += (g.next() - G::MIN) as f64 * tmp;
                    tmp *= range;
                }
                let r = sum / tmp;
                if r >= 1.0 {
                    1.0f64.next_down()
                } else {
                    r
                }
            }
        }
    }
}

/// `std::normal_distribution<double>(mean 0, deviation 1)`: Marsaglia's polar method, which
/// makes two values at a time; libc++ returns the first and keeps the second, libstdc++ (and
/// Microsoft's) the reverse.
#[derive(Clone, Debug, Default)]
pub struct Normal {
    saved: Option<f64>,
}

impl Normal {
    pub fn sample<G: Urbg>(&mut self, lib: Library, g: &mut G) -> f64 {
        if let Some(x) = self.saved.take() {
            return x;
        }
        let (u, v, s) = loop {
            let u = 2.0 * lib.canonical(g) - 1.0;
            let v = 2.0 * lib.canonical(g) - 1.0;
            let s = contracted(u, u, v * v);
            if !(s > 1.0 || s == 0.0) {
                break (u, v, s);
            }
        };
        let f = (-2.0 * s.ln() / s).sqrt();
        match lib {
            Library::LibCxx => {
                self.saved = Some(v * f);
                u * f
            }
            Library::LibStdCxx | Library::Msvc => {
                self.saved = Some(u * f);
                v * f
            }
        }
    }
}

impl Library {
    /// `std::sort(v.begin(), v.end(), less)` as this library implements it: unstable, so
    /// equal elements end in an order only the implementation fixes.
    pub fn sort_by<T: Copy>(self, v: &mut [T], less: &mut impl FnMut(&T, &T) -> bool) {
        match self {
            Library::LibCxx => {
                if v.len() > 1 {
                    let depth = 2 * (usize::BITS - 1 - v.len().leading_zeros()) as usize;
                    libcxx_introsort(v, 0, v.len(), less, depth, true);
                }
            }
            Library::LibStdCxx => {
                if v.len() > 1 {
                    let depth = 2 * (usize::BITS - 1 - v.len().leading_zeros()) as usize;
                    libstdcxx_introsort_loop(v, 0, v.len(), depth, less);
                    libstdcxx_final_insertion_sort(v, 0, v.len(), less);
                }
            }
            Library::Msvc => {
                let n = v.len() as isize;
                msvc_sort(v, 0, v.len(), n, less);
            }
        }
    }
}

// -- libc++'s sort (pattern-defeating introsort) ------------------------------------------------

fn lc_sort3<T: Copy>(v: &mut [T], x: usize, y: usize, z: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    if !c(&v[y], &v[x]) {
        if !c(&v[z], &v[y]) {
            return;
        }
        v.swap(y, z);
        if c(&v[y], &v[x]) {
            v.swap(x, y);
        }
        return;
    }
    if c(&v[z], &v[y]) {
        v.swap(x, z);
        return;
    }
    v.swap(x, y);
    if c(&v[z], &v[y]) {
        v.swap(y, z);
    }
}

fn lc_sort4<T: Copy>(v: &mut [T], x1: usize, x2: usize, x3: usize, x4: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    lc_sort3(v, x1, x2, x3, c);
    if c(&v[x4], &v[x3]) {
        v.swap(x3, x4);
        if c(&v[x3], &v[x2]) {
            v.swap(x2, x3);
            if c(&v[x2], &v[x1]) {
                v.swap(x1, x2);
            }
        }
    }
}

fn lc_sort5<T: Copy>(v: &mut [T], x: [usize; 5], c: &mut impl FnMut(&T, &T) -> bool) {
    lc_sort4(v, x[0], x[1], x[2], x[3], c);
    if c(&v[x[4]], &v[x[3]]) {
        v.swap(x[3], x[4]);
        if c(&v[x[3]], &v[x[2]]) {
            v.swap(x[2], x[3]);
            if c(&v[x[2]], &v[x[1]]) {
                v.swap(x[1], x[2]);
                if c(&v[x[1]], &v[x[0]]) {
                    v.swap(x[0], x[1]);
                }
            }
        }
    }
}

fn lc_insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    if first == last {
        return;
    }
    for i in first + 1..last {
        let mut j = i - 1;
        if c(&v[i], &v[j]) {
            let t = v[i];
            let mut k = j;
            j = i;
            loop {
                v[j] = v[k];
                j = k;
                if j == first {
                    break;
                }
                k -= 1;
                if !c(&t, &v[k]) {
                    break;
                }
            }
            v[j] = t;
        }
    }
}

fn lc_insertion_sort_unguarded<T: Copy>(v: &mut [T], first: usize, last: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    if first == last {
        return;
    }
    for i in first + 1..last {
        let mut j = i - 1;
        if c(&v[i], &v[j]) {
            let t = v[i];
            let mut k = j;
            j = i;
            loop {
                v[j] = v[k];
                j = k;
                k -= 1;
                if !c(&t, &v[k]) {
                    break;
                }
            }
            v[j] = t;
        }
    }
}

fn lc_insertion_sort_incomplete<T: Copy>(v: &mut [T], first: usize, last: usize, c: &mut impl FnMut(&T, &T) -> bool) -> bool {
    match last - first {
        0 | 1 => return true,
        2 => {
            if c(&v[last - 1], &v[first]) {
                v.swap(first, last - 1);
            }
            return true;
        }
        3 => {
            lc_sort3(v, first, first + 1, last - 1, c);
            return true;
        }
        4 => {
            lc_sort4(v, first, first + 1, first + 2, last - 1, c);
            return true;
        }
        5 => {
            lc_sort5(v, [first, first + 1, first + 2, first + 3, last - 1], c);
            return true;
        }
        _ => {}
    }
    let mut j = first + 2;
    lc_sort3(v, first, first + 1, j, c);
    let mut count = 0;
    let mut i = j + 1;
    while i != last {
        if c(&v[i], &v[j]) {
            let t = v[i];
            let mut k = j;
            j = i;
            loop {
                v[j] = v[k];
                j = k;
                if j == first {
                    break;
                }
                k -= 1;
                if !c(&t, &v[k]) {
                    break;
                }
            }
            v[j] = t;
            count += 1;
            if count == 8 {
                return i + 1 == last;
            }
        }
        j = i;
        i += 1;
    }
    true
}

fn lc_partition_with_equals_on_right<T: Copy>(v: &mut [T], begin: usize, end: usize, c: &mut impl FnMut(&T, &T) -> bool) -> (usize, bool) {
    let (mut first, mut last) = (begin, end);
    let pivot = v[first];
    loop {
        first += 1;
        if !c(&v[first], &pivot) {
            break;
        }
    }
    if begin == first - 1 {
        while first < last {
            last -= 1;
            if c(&v[last], &pivot) {
                break;
            }
        }
    } else {
        loop {
            last -= 1;
            if c(&v[last], &pivot) {
                break;
            }
        }
    }
    let already = first >= last;
    while first < last {
        v.swap(first, last);
        loop {
            first += 1;
            if !c(&v[first], &pivot) {
                break;
            }
        }
        loop {
            last -= 1;
            if c(&v[last], &pivot) {
                break;
            }
        }
    }
    let pivot_pos = first - 1;
    if begin != pivot_pos {
        v[begin] = v[pivot_pos];
    }
    v[pivot_pos] = pivot;
    (pivot_pos, already)
}

fn lc_partition_with_equals_on_left<T: Copy>(v: &mut [T], begin: usize, end: usize, c: &mut impl FnMut(&T, &T) -> bool) -> usize {
    let (mut first, mut last) = (begin, end);
    let pivot = v[first];
    if c(&pivot, &v[last - 1]) {
        loop {
            first += 1;
            if c(&pivot, &v[first]) {
                break;
            }
        }
    } else {
        loop {
            first += 1;
            if !(first < last && !c(&pivot, &v[first])) {
                break;
            }
        }
    }
    if first < last {
        loop {
            last -= 1;
            if !c(&pivot, &v[last]) {
                break;
            }
        }
    }
    while first < last {
        v.swap(first, last);
        loop {
            first += 1;
            if c(&pivot, &v[first]) {
                break;
            }
        }
        loop {
            last -= 1;
            if !c(&pivot, &v[last]) {
                break;
            }
        }
    }
    let pivot_pos = first - 1;
    if begin != pivot_pos {
        v[begin] = v[pivot_pos];
    }
    v[pivot_pos] = pivot;
    first
}

fn lc_sift_down<T: Copy>(v: &mut [T], first: usize, c: &mut impl FnMut(&T, &T) -> bool, len: usize, mut start: usize) {
    let mut child = start - first;
    if len < 2 || (len - 2) / 2 < child {
        return;
    }
    child = 2 * child + 1;
    let mut ci = first + child;
    if child + 1 < len && c(&v[ci], &v[ci + 1]) {
        ci += 1;
        child += 1;
    }
    if c(&v[ci], &v[start]) {
        return;
    }
    let top = v[start];
    loop {
        v[start] = v[ci];
        start = ci;
        if (len - 2) / 2 < child {
            break;
        }
        child = 2 * child + 1;
        ci = first + child;
        if child + 1 < len && c(&v[ci], &v[ci + 1]) {
            ci += 1;
            child += 1;
        }
        if c(&v[ci], &top) {
            break;
        }
    }
    v[start] = top;
}

fn lc_heap_sort<T: Copy>(v: &mut [T], first: usize, last: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    let n = last - first;
    if n > 1 {
        let mut start = (n - 2) / 2;
        loop {
            lc_sift_down(v, first, c, n, first + start);
            if start == 0 {
                break;
            }
            start -= 1;
        }
    }
    // sort_heap: pop with Floyd's sift-down, then sift up.
    let mut len = n;
    let mut end = last;
    while len > 1 {
        let top = v[first];
        let mut hole = first;
        let mut ci = first;
        let mut child = 0usize;
        loop {
            ci += child + 1;
            child = 2 * child + 1;
            if child + 1 < len && c(&v[ci], &v[ci + 1]) {
                ci += 1;
                child += 1;
            }
            v[hole] = v[ci];
            hole = ci;
            if child > (len - 2) / 2 {
                break;
            }
        }
        end -= 1;
        if hole == end {
            v[hole] = top;
        } else {
            v[hole] = v[end];
            hole += 1;
            v[end] = top;
            let mut l = hole - first;
            if l > 1 {
                l = (l - 2) / 2;
                let mut ptr = first + l;
                let mut last_i = hole - 1;
                if c(&v[ptr], &v[last_i]) {
                    let t = v[last_i];
                    loop {
                        v[last_i] = v[ptr];
                        last_i = ptr;
                        if l == 0 {
                            break;
                        }
                        l = (l - 1) / 2;
                        ptr = first + l;
                        if !c(&v[ptr], &t) {
                            break;
                        }
                    }
                    v[last_i] = t;
                }
            }
        }
        len -= 1;
    }
}

fn libcxx_introsort<T: Copy>(v: &mut [T], mut first: usize, mut last: usize, c: &mut impl FnMut(&T, &T) -> bool, mut depth: usize, mut leftmost: bool) {
    const LIMIT: usize = 24;
    const NINTHER: usize = 128;
    loop {
        let len = last - first;
        match len {
            0 | 1 => return,
            2 => {
                if c(&v[last - 1], &v[first]) {
                    v.swap(first, last - 1);
                }
                return;
            }
            3 => {
                lc_sort3(v, first, first + 1, last - 1, c);
                return;
            }
            4 => {
                lc_sort4(v, first, first + 1, first + 2, last - 1, c);
                return;
            }
            5 => {
                lc_sort5(v, [first, first + 1, first + 2, first + 3, last - 1], c);
                return;
            }
            _ => {}
        }
        if len < LIMIT {
            if leftmost {
                lc_insertion_sort(v, first, last, c);
            } else {
                lc_insertion_sort_unguarded(v, first, last, c);
            }
            return;
        }
        if depth == 0 {
            lc_heap_sort(v, first, last, c);
            return;
        }
        depth -= 1;
        let half = len / 2;
        if len > NINTHER {
            lc_sort3(v, first, first + half, last - 1, c);
            lc_sort3(v, first + 1, first + (half - 1), last - 2, c);
            lc_sort3(v, first + 2, first + (half + 1), last - 3, c);
            lc_sort3(v, first + (half - 1), first + half, first + (half + 1), c);
            v.swap(first, first + half);
        } else {
            lc_sort3(v, first + half, first, last - 1, c);
        }
        if !leftmost && !c(&v[first - 1], &v[first]) {
            first = lc_partition_with_equals_on_left(v, first, last, c);
            continue;
        }
        let (mut i, already) = lc_partition_with_equals_on_right(v, first, last, c);
        if already {
            let fs = lc_insertion_sort_incomplete(v, first, i, c);
            if lc_insertion_sort_incomplete(v, i + 1, last, c) {
                if fs {
                    return;
                }
                last = i;
                continue;
            } else if fs {
                i += 1;
                first = i;
                continue;
            }
        }
        libcxx_introsort(v, first, i, c, depth, leftmost);
        leftmost = false;
        first = i + 1;
    }
}

// -- Microsoft's sort (introsort: insertion sort to 32, ninther above 40, a three-way partition,
// heap sort past 1.5 log2 n divisions), from microsoft/STL (Apache-2.0 WITH LLVM-exception) ----

fn ms_insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    if first == last {
        return;
    }
    let mut mid = first;
    loop {
        mid += 1;
        if mid == last {
            return;
        }
        let val = v[mid];
        if c(&val, &v[first]) {
            v.copy_within(first..mid, first + 1);
            v[first] = val;
        } else {
            let mut hole = mid;
            let mut prev = hole - 1;
            while c(&val, &v[prev]) {
                v[hole] = v[prev];
                hole = prev;
                prev -= 1;
            }
            v[hole] = val;
        }
    }
}

fn ms_med3<T: Copy>(v: &mut [T], first: usize, mid: usize, last: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    if c(&v[mid], &v[first]) {
        v.swap(mid, first);
    }
    if c(&v[last], &v[mid]) {
        v.swap(last, mid);
        if c(&v[mid], &v[first]) {
            v.swap(mid, first);
        }
    }
}

/// `_Guess_median_unchecked`; `last` is inclusive.
fn ms_guess_median<T: Copy>(v: &mut [T], first: usize, mid: usize, last: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    let count = last - first;
    if 40 < count {
        let step = (count + 1) >> 3;
        let two_step = step << 1;
        ms_med3(v, first, first + step, first + two_step, c);
        ms_med3(v, mid - step, mid, mid + step, c);
        ms_med3(v, last - two_step, last - step, last, c);
        ms_med3(v, first + step, mid, last - step, c);
    } else {
        ms_med3(v, first, mid, last, c);
    }
}

/// `_Partition_by_pivot_unchecked`: (start, end) of the run equal to the pivot.
fn ms_partition<T: Copy>(v: &mut [T], first: usize, mut pfirst: usize, last: usize, c: &mut impl FnMut(&T, &T) -> bool) -> (usize, usize) {
    let mut plast = pfirst + 1;
    while first < pfirst && !c(&v[pfirst - 1], &v[pfirst]) && !c(&v[pfirst], &v[pfirst - 1]) {
        pfirst -= 1;
    }
    while plast < last && !c(&v[plast], &v[pfirst]) && !c(&v[pfirst], &v[plast]) {
        plast += 1;
    }
    let mut gfirst = plast;
    let mut glast = pfirst;
    loop {
        while gfirst < last {
            if c(&v[pfirst], &v[gfirst]) {
            } else if c(&v[gfirst], &v[pfirst]) {
                break;
            } else {
                if plast != gfirst {
                    v.swap(plast, gfirst);
                }
                plast += 1;
            }
            gfirst += 1;
        }
        while first < glast {
            let gp = glast - 1;
            if c(&v[gp], &v[pfirst]) {
            } else if c(&v[pfirst], &v[gp]) {
                break;
            } else {
                pfirst -= 1;
                if pfirst != gp {
                    v.swap(pfirst, gp);
                }
            }
            glast -= 1;
        }
        if glast == first && gfirst == last {
            return (pfirst, plast);
        }
        if glast == first {
            if plast != gfirst {
                v.swap(pfirst, plast);
            }
            plast += 1;
            v.swap(pfirst, gfirst);
            pfirst += 1;
            gfirst += 1;
        } else if gfirst == last {
            glast -= 1;
            pfirst -= 1;
            if glast != pfirst {
                v.swap(glast, pfirst);
            }
            plast -= 1;
            v.swap(pfirst, plast);
        } else {
            glast -= 1;
            v.swap(gfirst, glast);
            gfirst += 1;
        }
    }
}

pub(crate) fn ms_push_heap_by_index<T: Copy>(v: &mut [T], base: usize, mut hole: isize, top: isize, val: T, c: &mut impl FnMut(&T, &T) -> bool) {
    while top < hole {
        let idx = (hole - 1) >> 1;
        if !c(&v[base + idx as usize], &val) {
            break;
        }
        v[base + hole as usize] = v[base + idx as usize];
        hole = idx;
    }
    v[base + hole as usize] = val;
}

pub(crate) fn ms_pop_heap_hole_by_index<T: Copy>(v: &mut [T], base: usize, mut hole: isize, bottom: isize, val: T, c: &mut impl FnMut(&T, &T) -> bool) {
    let top = hole;
    let mut idx = hole;
    let max_non_leaf = (bottom - 1) >> 1;
    while idx < max_non_leaf {
        idx = 2 * idx + 2;
        if c(&v[base + idx as usize], &v[base + idx as usize - 1]) {
            idx -= 1;
        }
        v[base + hole as usize] = v[base + idx as usize];
        hole = idx;
    }
    if idx == max_non_leaf && bottom % 2 == 0 {
        v[base + hole as usize] = v[base + bottom as usize - 1];
        hole = bottom - 1;
    }
    ms_push_heap_by_index(v, base, hole, top, val, c);
}

fn ms_heap_sort<T: Copy>(v: &mut [T], first: usize, mut last: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    let bottom = (last - first) as isize;
    let mut hole = bottom >> 1;
    while hole > 0 {
        hole -= 1;
        let val = v[first + hole as usize];
        ms_pop_heap_hole_by_index(v, first, hole, bottom, val, c);
    }
    while last - first >= 2 {
        let l = last - 1;
        let val = v[l];
        v[l] = v[first];
        ms_pop_heap_hole_by_index(v, first, 0, (l - first) as isize, val, c);
        last -= 1;
    }
}

fn msvc_sort<T: Copy>(v: &mut [T], mut first: usize, mut last: usize, mut ideal: isize, c: &mut impl FnMut(&T, &T) -> bool) {
    loop {
        if last - first <= 32 {
            ms_insertion_sort(v, first, last, c);
            return;
        }
        if ideal <= 0 {
            ms_heap_sort(v, first, last, c);
            return;
        }
        let mid = first + ((last - first) >> 1);
        ms_guess_median(v, first, mid, last - 1, c);
        let (mf, ms) = ms_partition(v, first, mid, last, c);
        ideal = (ideal >> 1) + (ideal >> 2);
        if mf - first < last - ms {
            msvc_sort(v, first, mf, ideal, c);
            first = ms;
        } else {
            msvc_sort(v, ms, last, ideal, c);
            last = mf;
        }
    }
}

// -- libstdc++'s sort (introsort, then a final insertion sort) ----------------------------------

fn ls_unguarded_linear_insert<T: Copy>(v: &mut [T], mut last: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    let val = v[last];
    let mut next = last - 1;
    while c(&val, &v[next]) {
        v[last] = v[next];
        last = next;
        next -= 1;
    }
    v[last] = val;
}

fn ls_insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    if first == last {
        return;
    }
    for i in first + 1..last {
        if c(&v[i], &v[first]) {
            let val = v[i];
            v.copy_within(first..i, first + 1);
            v[first] = val;
        } else {
            ls_unguarded_linear_insert(v, i, c);
        }
    }
}

fn libstdcxx_final_insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    if last - first > 16 {
        ls_insertion_sort(v, first, first + 16, c);
        for i in first + 16..last {
            ls_unguarded_linear_insert(v, i, c);
        }
    } else {
        ls_insertion_sort(v, first, last, c);
    }
}

fn ls_adjust_heap<T: Copy>(v: &mut [T], first: usize, mut hole: usize, len: usize, value: T, c: &mut impl FnMut(&T, &T) -> bool) {
    let top = hole;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if c(&v[first + second], &v[first + second - 1]) {
            second -= 1;
        }
        v[first + hole] = v[first + second];
        hole = second;
    }
    if len & 1 == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[first + hole] = v[first + second - 1];
        hole = second - 1;
    }
    let mut parent = hole.wrapping_sub(1) / 2;
    while hole > top && c(&v[first + parent], &value) {
        v[first + hole] = v[first + parent];
        hole = parent;
        parent = hole.wrapping_sub(1) / 2;
    }
    v[first + hole] = value;
}

fn ls_heap_sort<T: Copy>(v: &mut [T], first: usize, last: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    let len = last - first;
    if len >= 2 {
        let mut parent = (len - 2) / 2;
        loop {
            let value = v[first + parent];
            ls_adjust_heap(v, first, parent, len, value, c);
            if parent == 0 {
                break;
            }
            parent -= 1;
        }
    }
    let mut end = last;
    while end - first > 1 {
        end -= 1;
        let value = v[end];
        v[end] = v[first];
        ls_adjust_heap(v, first, 0, end - first, value, c);
    }
}

fn libstdcxx_introsort_loop<T: Copy>(v: &mut [T], first: usize, mut last: usize, mut depth: usize, c: &mut impl FnMut(&T, &T) -> bool) {
    while last - first > 16 {
        if depth == 0 {
            ls_heap_sort(v, first, last, c);
            return;
        }
        depth -= 1;
        // The median of first+1, the middle and last-1 moved to first, then an unguarded
        // partition around it.
        let (a, b, cc) = (first + 1, first + (last - first) / 2, last - 1);
        let m = if c(&v[a], &v[b]) {
            if c(&v[b], &v[cc]) {
                b
            } else if c(&v[a], &v[cc]) {
                cc
            } else {
                a
            }
        } else if c(&v[a], &v[cc]) {
            a
        } else if c(&v[b], &v[cc]) {
            cc
        } else {
            b
        };
        v.swap(first, m);
        let (mut lo, mut hi) = (first + 1, last);
        let cut = loop {
            while c(&v[lo], &v[first]) {
                lo += 1;
            }
            hi -= 1;
            while c(&v[first], &v[hi]) {
                hi -= 1;
            }
            if lo >= hi {
                break lo;
            }
            v.swap(lo, hi);
            lo += 1;
        };
        libstdcxx_introsort_loop(v, cut, last, depth, c);
        last = cut;
    }
}

/// C's `qsort` as each C library implements it, with a three-way comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CLibrary {
    /// Apple's libc (FreeBSD's Bentley-McIlroy quicksort).
    Bsd,
    /// glibc (a stable merge sort, as in every release but 2.37 and 2.38).
    Glibc,
    /// Microsoft's CRT (its median-of-three quicksort, `msvc_qsort`).
    Msvc,
}

impl CLibrary {
    pub const NATIVE: CLibrary = if cfg!(any(target_os = "macos", target_os = "ios", target_os = "freebsd")) {
        CLibrary::Bsd
    } else if cfg!(target_os = "windows") {
        CLibrary::Msvc
    } else {
        CLibrary::Glibc
    };

    pub fn qsort<T: Copy>(self, v: &mut [T], cmp: &mut impl FnMut(&T, &T) -> std::cmp::Ordering) {
        match self {
            CLibrary::Bsd => bsd_qsort(v, cmp),
            CLibrary::Glibc => v.sort_by(|a, b| cmp(a, b)),
            CLibrary::Msvc => msvc_qsort(v, cmp),
        }
    }
}

/// Microsoft's CRT `qsort` (ucrt/stdlib/qsort.cpp, as in every CRT since Visual C++ 6): a
/// quicksort on an explicit stack, its pivot the median of the first, middle and last elements
/// and followed as it is swapped, ranges of eight or fewer sorted by selecting the largest, and
/// the smaller part split first. Unstable, so its order among equal elements is its own.
fn msvc_qsort<T: Copy>(v: &mut [T], cmp: &mut impl FnMut(&T, &T) -> std::cmp::Ordering) {
    use std::cmp::Ordering::{Equal, Greater};
    const CUTOFF: usize = 8;
    if v.len() < 2 {
        return;
    }
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let (mut lo, mut hi) = (0usize, v.len() - 1);
    loop {
        let size = hi - lo + 1;
        if size <= CUTOFF {
            // shortsort: move the largest of lo..=hi to hi, then shrink.
            let mut h = hi;
            while h > lo {
                let mut max = lo;
                for p in lo + 1..=h {
                    if cmp(&v[p], &v[max]) == Greater {
                        max = p;
                    }
                }
                v.swap(max, h);
                h -= 1;
            }
        } else {
            let mut mid = lo + size / 2;
            if cmp(&v[lo], &v[mid]) == Greater {
                v.swap(lo, mid);
            }
            if cmp(&v[lo], &v[hi]) == Greater {
                v.swap(lo, hi);
            }
            if cmp(&v[mid], &v[hi]) == Greater {
                v.swap(mid, hi);
            }
            // Positions as signed offsets: higuy can step to one below lo.
            let (mut loguy, mut higuy) = (lo as isize, hi as isize);
            let (lo_i, hi_i) = (lo as isize, hi as isize);
            loop {
                if mid as isize > loguy {
                    loop {
                        loguy += 1;
                        if !(loguy < mid as isize && cmp(&v[loguy as usize], &v[mid]) != Greater) {
                            break;
                        }
                    }
                }
                if mid as isize <= loguy {
                    loop {
                        loguy += 1;
                        if !(loguy <= hi_i && cmp(&v[loguy as usize], &v[mid]) != Greater) {
                            break;
                        }
                    }
                }
                loop {
                    higuy -= 1;
                    if !(higuy > mid as isize && cmp(&v[higuy as usize], &v[mid]) == Greater) {
                        break;
                    }
                }
                if higuy < loguy {
                    break;
                }
                v.swap(loguy as usize, higuy as usize);
                if mid as isize == higuy {
                    mid = loguy as usize;
                }
            }
            higuy += 1;
            if (mid as isize) < higuy {
                loop {
                    higuy -= 1;
                    if !(higuy > mid as isize && cmp(&v[higuy as usize], &v[mid]) == Equal) {
                        break;
                    }
                }
            }
            if mid as isize >= higuy {
                loop {
                    higuy -= 1;
                    if !(higuy > lo_i && cmp(&v[higuy as usize], &v[mid]) == Equal) {
                        break;
                    }
                }
            }
            if higuy - lo_i >= hi_i - loguy {
                if lo_i < higuy {
                    stack.push((lo, higuy as usize));
                }
                if loguy < hi_i {
                    lo = loguy as usize;
                    continue;
                }
            } else {
                if loguy < hi_i {
                    stack.push((loguy as usize, hi));
                }
                if lo_i < higuy {
                    hi = higuy as usize;
                    continue;
                }
            }
        }
        match stack.pop() {
            Some((l, h)) => (lo, hi) = (l, h),
            None => return,
        }
    }
}

fn bsd_med3<T: Copy>(v: &[T], a: usize, b: usize, c: usize, cmp: &mut impl FnMut(&T, &T) -> std::cmp::Ordering) -> usize {
    use std::cmp::Ordering::{Greater, Less};
    if cmp(&v[a], &v[b]) == Less {
        if cmp(&v[b], &v[c]) == Less {
            b
        } else if cmp(&v[a], &v[c]) == Less {
            c
        } else {
            a
        }
    } else if cmp(&v[b], &v[c]) == Greater {
        b
    } else if cmp(&v[a], &v[c]) == Less {
        a
    } else {
        c
    }
}

fn bsd_insertion<T: Copy>(v: &mut [T], cmp: &mut impl FnMut(&T, &T) -> std::cmp::Ordering) {
    for pm in 1..v.len() {
        let mut pl = pm;
        while pl > 0 && cmp(&v[pl - 1], &v[pl]) == std::cmp::Ordering::Greater {
            v.swap(pl, pl - 1);
            pl -= 1;
        }
    }
}

fn bsd_vecswap<T: Copy>(v: &mut [T], a: usize, b: usize, n: usize) {
    for i in 0..n {
        v.swap(a + i, b + i);
    }
}

fn bsd_qsort<T: Copy>(mut v: &mut [T], cmp: &mut impl FnMut(&T, &T) -> std::cmp::Ordering) {
    use std::cmp::Ordering::{Equal, Greater, Less};
    loop {
        let n = v.len();
        let mut swapped = false;
        if n <= 7 {
            bsd_insertion(v, cmp);
            return;
        }
        let mut pm = n / 2;
        if n > 7 {
            let mut pl = 0;
            let mut pn = n - 1;
            if n > 40 {
                let d = n / 8;
                pl = bsd_med3(v, pl, pl + d, pl + 2 * d, cmp);
                pm = bsd_med3(v, pm - d, pm, pm + d, cmp);
                pn = bsd_med3(v, pn - 2 * d, pn - d, pn, cmp);
            }
            pm = bsd_med3(v, pl, pm, pn, cmp);
        }
        v.swap(0, pm);
        let (mut pa, mut pb) = (1usize, 1usize);
        let (mut pc, mut pd) = (n as isize - 1, n as isize - 1);
        loop {
            while pb as isize <= pc {
                let r = cmp(&v[pb], &v[0]);
                if r == Greater {
                    break;
                }
                if r == Equal {
                    swapped = true;
                    v.swap(pa, pb);
                    pa += 1;
                }
                pb += 1;
            }
            while pb as isize <= pc {
                let r = cmp(&v[pc as usize], &v[0]);
                if r == Less {
                    break;
                }
                if r == Equal {
                    swapped = true;
                    v.swap(pc as usize, pd as usize);
                    pd -= 1;
                }
                pc -= 1;
            }
            if pb as isize > pc {
                break;
            }
            v.swap(pb, pc as usize);
            swapped = true;
            pb += 1;
            pc -= 1;
        }
        let _ = swapped;
        let pn = n;
        let d1 = pa.min(pb - pa);
        bsd_vecswap(v, 0, pb - d1, d1);
        let d1 = ((pd - pc) as usize).min(pn - pd as usize - 1);
        bsd_vecswap(v, pb, pn - d1, d1);
        let (d1, d2) = (pb - pa, (pd - pc) as usize);
        if d1 <= d2 {
            if d1 > 1 {
                bsd_qsort(&mut v[..d1], cmp);
            }
            if d2 > 1 {
                v = &mut v[pn - d2..];
                continue;
            }
        } else {
            if d2 > 1 {
                bsd_qsort(&mut v[pn - d2..], cmp);
            }
            if d1 > 1 {
                v = &mut v[..d1];
                continue;
            }
        }
        return;
    }
}

/// libc++'s `uniform_int_distribution` over 64 bits: the fewest bits that cover the range
/// (from `__independent_bits_engine`), drawn until in range.
fn libcxx_uniform<G: Urbg>(g: &mut G, a: u64, b: u64) -> u64 {
    let rp = b.wrapping_sub(a).wrapping_add(1);
    if rp == 1 {
        return a;
    }
    let dt = 64u32;
    let w = if rp == 0 { dt } else {
        let mut w = dt - rp.leading_zeros() - 1;
        if rp & (u64::MAX >> (dt - w)) != 0 {
            w += 1;
        }
        w
    };
    if G::MAX - G::MIN == u64::MAX {
        // An engine filling the word (mt19937_64): libc++'s __independent_bits_engine takes
        // one draw's low w bits.
        let mask = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
        loop {
            let u = g.next() & mask;
            if rp == 0 {
                return u;
            }
            if u < rp {
                return u + a;
            }
        }
    }
    let mut e = IndependentBits::new(w, G::MAX - G::MIN + 1);
    if rp == 0 {
        return e.eval(g);
    }
    loop {
        let u = e.eval(g);
        if u < rp {
            return u + a;
        }
    }
}

/// libc++'s `__independent_bits_engine<G, uint64_t>`.
struct IndependentBits {
    n: u64,
    n0: u64,
    w0: u32,
    y0: u128,
    y1: u128,
    mask0: u64,
    mask1: u64,
}

impl IndependentBits {
    fn new(w: u32, rp: u64) -> IndependentBits {
        // rp, the engine's range (≤ 2^32 here, so it fits u64; 2^32 for mt19937).
        let rp = u128::from(rp);
        let m = 127 - rp.leading_zeros(); // floor(log2(rp))
        let wdt = 64u32; // the working type's digits
        let edt = 64u32; // the engine result type's digits, as libc++ sees uint_fast32_t
        let w = u64::from(w);
        let mut n = w / u64::from(m) + u64::from(w % u64::from(m) != 0);
        let mut w0 = (w / n) as u32;
        let shifted = |w0: u32| if w0 < wdt { (rp >> w0) << w0 } else { 0 };
        let mut y0 = shifted(w0);
        if rp - y0 > y0 / u128::from(n) {
            n += 1;
            w0 = (w / n) as u32;
            y0 = shifted(w0);
        }
        let n0 = n - w % n;
        let y1 = if w0 < wdt - 1 { (rp >> (w0 + 1)) << (w0 + 1) } else { 0 };
        let mask0 = if w0 > 0 { u64::MAX >> (edt - w0) } else { 0 };
        let mask1 = if w0 < edt - 1 { u64::MAX >> (edt - (w0 + 1)) } else { u64::MAX };
        IndependentBits { n, n0, w0, y0, y1, mask0, mask1 }
    }

    fn eval<G: Urbg>(&mut self, g: &mut G) -> u64 {
        let mut sp: u64 = 0;
        for _ in 0..self.n0 {
            let u = loop {
                let u = g.next() - G::MIN;
                if u128::from(u) < self.y0 {
                    break u;
                }
            };
            sp = if self.w0 < 64 { sp << self.w0 } else { 0 };
            sp = sp.wrapping_add(u & self.mask0);
        }
        for _ in self.n0..self.n {
            let u = loop {
                let u = g.next() - G::MIN;
                if u128::from(u) < self.y1 {
                    break u;
                }
            };
            sp = if self.w0 < 63 { sp << (self.w0 + 1) } else { 0 };
            sp = sp.wrapping_add(u & self.mask1);
        }
        sp
    }
}

/// libstdc++'s `uniform_int_distribution<unsigned long>`: Lemire's nearly divisionless
/// method for a 32-bit engine, otherwise scaling down with rejection.
fn libstdcxx_uniform<G: Urbg>(g: &mut G, a: u64, b: u64) -> u64 {
    let urngrange = G::MAX - G::MIN;
    let urange = b - a;
    let ret = if urngrange > urange {
        let uerange = urange + 1;
        if urngrange == u64::MAX {
            // A 64-bit engine: Lemire's method with a 128-bit product.
            let mut product = u128::from(g.next()) * u128::from(uerange);
            let mut low = product as u64;
            if low < uerange {
                let threshold = uerange.wrapping_neg() % uerange;
                while low < threshold {
                    product = u128::from(g.next()) * u128::from(uerange);
                    low = product as u64;
                }
            }
            (product >> 64) as u64
        } else if urngrange == u64::from(u32::MAX) {
            let range = uerange as u32;
            let mut product = (g.next() - G::MIN) * u64::from(range);
            let mut low = product as u32;
            if low < range {
                let threshold = range.wrapping_neg() % range;
                while low < threshold {
                    product = (g.next() - G::MIN) * u64::from(range);
                    low = product as u32;
                }
            }
            product >> 32
        } else {
            let scaling = urngrange / uerange;
            let past = uerange * scaling;
            loop {
                let r = g.next() - G::MIN;
                if r < past {
                    break r / scaling;
                }
            }
        }
    } else if urngrange < urange {
        // Upscaling: several draws (not reached by shuffles of fewer than 2^32 elements).
        let uerngrange = urngrange + 1;
        loop {
            let high = uerngrange.wrapping_mul(libstdcxx_uniform(g, 0, urange / uerngrange));
            let r = high.wrapping_add(g.next() - G::MIN);
            if r <= urange && r >= high {
                break r;
            }
        }
    } else {
        g.next() - G::MIN
    };
    ret + a
}

/// Microsoft's `_Rng_from_urng`: a value in `[0, index)` from as many draws as cover it, kept
/// when unbiased.
fn msvc_rng_from_urng<G: Urbg>(g: &mut G, index: u64) -> u64 {
    let range = G::MAX - G::MIN;
    let bits = 64 - range.leading_zeros();
    let bmask: u64 = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    loop {
        let mut ret: u64 = 0;
        let mut mask: u64 = 0;
        while mask < index - 1 {
            ret = (ret << (bits - 1)) << 1;
            ret |= loop {
                let e = g.next() - G::MIN;
                if e <= bmask {
                    break e;
                }
            };
            mask = (mask << (bits - 1)) << 1;
            mask |= bmask;
        }
        if ret / index < mask / index || mask % index == index - 1 {
            return ret % index;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_is_the_standards() {
        // The standard fixes the 10000th output of a default-seeded mt19937.
        let mut g = Mt19937::new(5489);
        let mut x = 0;
        for _ in 0..10000 {
            x = g.next_u32();
        }
        assert_eq!(x, 4_123_659_995);
    }

    #[test]
    fn minstd_is_the_standards() {
        // And the 10000th of minstd_rand0 and minstd_rand.
        let mut a = MinStd::<16807>::new(1);
        let mut b = MinStd::<48271>::new(1);
        let (mut x, mut y) = (0, 0);
        for _ in 0..10000 {
            x = a.next();
            y = b.next();
        }
        assert_eq!((x, y), (1_043_618_065, 399_268_537));
    }

    #[test]
    fn shuffles_are_permutations() {
        for lib in [Library::LibCxx, Library::LibStdCxx, Library::Msvc] {
            for n in [0usize, 1, 2, 5, 64, 1000] {
                let mut v: Vec<usize> = (0..n).collect();
                let mut g = Mt19937::new(7);
                lib.shuffle(&mut v, &mut g);
                let mut s = v.clone();
                s.sort_unstable();
                assert_eq!(s, (0..n).collect::<Vec<_>>(), "{lib:?} {n}");
                for k in 0..20 {
                    let x = lib.uniform(&mut g, 3, 3 + k);
                    assert!((3..=3 + k).contains(&x));
                }
            }
        }
    }

    fn digest(v: &[usize]) -> i64 {
        v.iter().fold(0i64, |s, &x| (s * 31 + x as i64 + 1) % 1_000_000_007)
    }

    /// libc++'s shuffles, from a program built with Apple clang (tools/cpp_shuffle.cpp):
    /// mt19937(12345) shuffled twice, and default_random_engine (minstd_rand) seeded 777.
    #[test]
    fn libcxx_shuffles_are_libcxxs() {
        let want = [(1, 1, 1), (2, 63, 33), (5, 1_969_245, 3_756_585), (17, 559_408_892, 154_546_822), (64, 56_691_284, 166_027_507), (1000, 973_856_518, 651_726_378)];
        for (n, mt, dre) in want {
            let mut v: Vec<usize> = (0..n).collect();
            let mut g = Mt19937::new(12345);
            Library::LibCxx.shuffle(&mut v, &mut g);
            Library::LibCxx.shuffle(&mut v, &mut g);
            assert_eq!(digest(&v), mt, "mt19937 {n}");
            let mut u: Vec<usize> = (0..n).collect();
            let mut e = MinStd::<48271>::new(777);
            Library::LibCxx.shuffle(&mut u, &mut e);
            assert_eq!(digest(&u), dre, "minstd_rand {n}");
        }
    }

    /// libc++'s std::sort, from a program built with Apple clang (tools/cpp_sort.cpp): indices
    /// sorted by keys with many ties, descending, as ldpc's serial-relative schedule sorts.
    #[test]
    fn libcxx_sort_is_libcxxs() {
        let want: &[(usize, u64, i64)] = &[(3, 2, 2016), (3, 5, 2916), (3, 1000, 2916), (5, 2, 3844095), (5, 5, 4650345), (5, 1000, 3844065), (7, 2, 528260253), (7, 5, 613196960), (7, 1000, 612964267), (20, 2, 974634535), (20, 5, 642667742), (20, 1000, 313955370), (23, 2, 648736553), (23, 5, 199529229), (23, 1000, 176139502), (24, 2, 859197234), (24, 5, 61425341), (24, 1000, 326226836), (25, 2, 685806848), (25, 5, 665896286), (25, 1000, 934517784), (26, 2, 117852091), (26, 5, 178799643), (26, 1000, 960623249), (29, 2, 110896679), (29, 5, 973794251), (29, 1000, 461361499), (40, 2, 757271371), (40, 5, 685882196), (40, 1000, 529974324), (64, 2, 613908677), (64, 5, 880045805), (64, 1000, 341513197), (100, 2, 786493785), (100, 5, 632929811), (100, 1000, 509678525), (129, 2, 289556983), (129, 5, 692260614), (129, 1000, 34254592), (200, 2, 517281601), (200, 5, 275816752), (200, 1000, 649780900), (1000, 2, 362869866), (1000, 5, 720686258), (1000, 1000, 495611864), (5000, 2, 887144046), (5000, 5, 571596480), (5000, 1000, 975127627)];
        let mut x: u64 = 88_172_645_463_325_252;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for &(n, distinct, digest_want) in want {
            let key: Vec<f64> = (0..n).map(|_| (next() % distinct) as f64).collect();
            for _ in 0..n {
                next();
            }
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (next() % (i as u64 + 1)) as usize;
                idx.swap(i, j);
            }
            Library::LibCxx.sort_by(&mut idx, &mut |&a, &b| key[a] > key[b]);
            assert_eq!(digest(&idx), digest_want, "n {n}, {distinct} distinct keys");
        }
    }

    /// Microsoft's std::sort, from its source transcribed into portable C++
    /// (tools/cpp_msvc_sort.cpp), on the inputs of `libcxx_sort_is_libcxxs` and sizes around
    /// its insertion-sort (32) and ninther (40) thresholds.
    #[test]
    fn msvc_sort_is_microsofts() {
        let want: &[(usize, u64, i64)] = &[(3, 2, 2016), (3, 5, 2916), (3, 1000, 2916), (5, 2, 3844095), (5, 5, 4650345), (5, 1000, 3844065), (7, 2, 528260253), (7, 5, 613196960), (7, 1000, 612964267), (20, 2, 974634535), (20, 5, 642667742), (20, 1000, 313955370), (23, 2, 648736553), (23, 5, 199529229), (23, 1000, 176139502), (24, 2, 859197234), (24, 5, 714897753), (24, 1000, 701998544), (25, 2, 127452841), (25, 5, 691739525), (25, 1000, 934517784), (26, 2, 956478251), (26, 5, 144296326), (26, 1000, 960623249), (29, 2, 249389996), (29, 5, 613453087), (29, 1000, 174842090), (32, 2, 347634080), (32, 5, 667501341), (32, 1000, 801679711), (33, 2, 456144199), (33, 5, 631549787), (33, 1000, 575725597), (40, 2, 606230099), (40, 5, 636844812), (40, 1000, 830173047), (41, 2, 53634722), (41, 5, 931760663), (41, 1000, 896217091), (42, 2, 906528613), (42, 5, 163848745), (42, 1000, 280594710), (64, 2, 117337538), (64, 5, 427870190), (64, 1000, 484363923), (100, 2, 979501599), (100, 5, 881542083), (100, 1000, 991379470), (129, 2, 81077470), (129, 5, 371856770), (129, 1000, 22625200), (200, 2, 36647318), (200, 5, 554132272), (200, 1000, 153158743), (1000, 2, 691949902), (1000, 5, 757115253), (1000, 1000, 873469589), (5000, 2, 560773987), (5000, 5, 742651431), (5000, 1000, 274235655)];
        let mut x: u64 = 88_172_645_463_325_252;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for &(n, distinct, digest_want) in want {
            let key: Vec<f64> = (0..n).map(|_| (next() % distinct) as f64).collect();
            for _ in 0..n {
                next();
            }
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (next() % (i as u64 + 1)) as usize;
                idx.swap(i, j);
            }
            Library::Msvc.sort_by(&mut idx, &mut |&a, &b| key[a] > key[b]);
            assert_eq!(digest(&idx), digest_want, "n {n}, {distinct} distinct keys");
        }
        // Its heap sort, alone.
        let heap: &[(usize, u64, i64)] = &[(33, 2, 577647698), (33, 1000, 90761099), (100, 2, 712385183), (100, 1000, 844692272), (1000, 2, 511701770), (1000, 1000, 546014897)];
        for &(n, distinct, digest_want) in heap {
            let key: Vec<f64> = (0..n).map(|_| (next() % distinct) as f64).collect();
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (next() % (i as u64 + 1)) as usize;
                idx.swap(i, j);
            }
            msvc_sort(&mut idx, 0, n, 0, &mut |&a, &b| key[a] > key[b]);
            assert!(idx.windows(2).all(|w| key[w[0]] >= key[w[1]]));
            assert_eq!(digest(&idx), digest_want, "heap sort, n {n}, {distinct} distinct keys");
        }
    }

    /// Microsoft's CRT qsort sorts (it is checked against Microsoft's own where it runs, on
    /// Windows, through ldpc's OSD in tests/test_panqec_differential.py), and its order among
    /// equal values is not the input's.
    #[test]
    fn msvc_qsort_sorts() {
        let mut x: u64 = 88_172_645_463_325_252;
        let mut reordered = false;
        for n in [0usize, 1, 2, 3, 8, 9, 10, 40, 41, 500, 3000] {
            for distinct in [1u64, 2, 5, 1000] {
                let mut o: Vec<(f64, usize)> = (0..n)
                    .map(|i| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        ((x % distinct) as f64, i)
                    })
                    .collect();
                let mut stable = o.clone();
                stable.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                CLibrary::Msvc.qsort(&mut o, &mut |a, b| a.0.partial_cmp(&b.0).unwrap());
                assert!(o.windows(2).all(|w| w[0].0 <= w[1].0), "n {n}");
                let mut idx: Vec<usize> = o.iter().map(|r| r.1).collect();
                reordered |= idx != stable.iter().map(|r| r.1).collect::<Vec<_>>();
                idx.sort_unstable();
                assert!(idx.iter().copied().eq(0..n));
            }
        }
        assert!(reordered);
    }

    /// macOS's qsort, from a C program (tools/c_qsort.c): (value, index) records sorted by
    /// value with many ties, as ldpc's OSD sorts its columns.
    #[test]
    fn bsd_qsort_is_macoss() {
        let want: &[(usize, u64, i64)] = &[(3, 2, 1056), (3, 5, 1956), (3, 1000, 1056), (6, 2, 91460421), (6, 5, 86991771), (6, 1000, 62057661), (7, 2, 863902194), (7, 5, 63183281), (7, 1000, 8547581), (8, 2, 977882932), (8, 5, 749050492), (8, 1000, 969224837), (12, 2, 668580880), (12, 5, 138405752), (12, 1000, 172400152), (20, 2, 290932487), (20, 5, 813348366), (20, 1000, 165245755), (40, 2, 797948561), (40, 5, 497885979), (40, 1000, 183295677), (41, 2, 357531570), (41, 5, 864570903), (41, 1000, 239857558), (64, 2, 902392530), (64, 5, 894239459), (64, 1000, 882779166), (100, 2, 30802800), (100, 5, 213627627), (100, 1000, 391530246), (500, 2, 398289829), (500, 5, 390319272), (500, 1000, 311859471), (3000, 2, 484595915), (3000, 5, 212323273), (3000, 1000, 976278602)];
        let mut x: u64 = 88_172_645_463_325_252;
        for &(n, distinct, digest_want) in want {
            let mut o: Vec<(f64, usize)> = (0..n)
                .map(|i| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    ((x % distinct) as f64, i)
                })
                .collect();
            CLibrary::Bsd.qsort(&mut o, &mut |a, b| a.0.partial_cmp(&b.0).unwrap());
            let idx: Vec<usize> = o.iter().map(|r| r.1).collect();
            assert_eq!(digest(&idx), digest_want, "n {n}, {distinct} distinct values");
        }
    }
}
