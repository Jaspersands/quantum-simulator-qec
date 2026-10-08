//! A set of `u32` that iterates in the order `tsl::robin_set<int>` does.
//!
//! WHY THIS EXISTS
//! ---------------
//! `ldpc`'s localized statistics decoder keeps each cluster's bits, checks and candidates in
//! `tsl::robin_set<int>` (Tessil's robin-map, vendored in ldpc), and the order it walks them
//! decides things that change the answer: the order a merged cluster's columns enter the
//! surviving cluster's matrix, and so which columns become pivots. To return `ldpc`'s
//! corrections, LSD (`lsd`) walks its sets in that same order, which this reproduces. It is a
//! pure function of the insertions and removals, because `std::hash<int>` is the identity in
//! both libc++ and libstdc++:
//!
//! - open addressing over a power-of-two array of buckets, a key's ideal bucket `key & mask`;
//! - none allocated until the first insert; then 2 buckets, doubling whenever an insert finds
//!   the set holding half as many keys as buckets (max load factor 0.5);
//! - robin-hood insertion: a key probing past a bucket whose occupant sits closer to its own
//!   ideal bucket takes that bucket, and the occupant moves on;
//! - removal by backward shift: later keys of the run move back one bucket;
//! - growth reinserts the keys in the old array's bucket order;
//! - `clear` empties the buckets but keeps the array (the minimum load factor is 0).
//!
//! Iteration is bucket order. Robin-map's other rules (rehash after 8192 probes) cannot
//! trigger at the sizes LSD reaches and are left out.

const EMPTY: i16 = -1;

#[derive(Clone, Debug, Default)]
pub(crate) struct RobinSet {
    /// (distance from the ideal bucket or `EMPTY`, key).
    buckets: Vec<(i16, u32)>,
    len: usize,
}

impl RobinSet {
    pub(crate) fn new() -> RobinSet {
        RobinSet::default()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    fn mask(&self) -> usize {
        self.buckets.len().saturating_sub(1)
    }

    fn dist(&self, i: usize) -> i16 {
        self.buckets.get(i).map_or(EMPTY, |b| b.0)
    }

    fn next(&self, i: usize) -> usize {
        (i + 1) & self.mask()
    }

    fn find(&self, key: u32) -> Option<usize> {
        let mut i = key as usize & self.mask();
        let mut d = 0i16;
        while d <= self.dist(i) {
            if self.buckets[i].1 == key {
                return Some(i);
            }
            i = self.next(i);
            d += 1;
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, key: u32) -> bool {
        self.find(key).is_some()
    }

    /// Insert `key`; whether it was new.
    pub(crate) fn insert(&mut self, key: u32) -> bool {
        let mut i = key as usize & self.mask();
        let mut d = 0i16;
        while d <= self.dist(i) {
            if self.buckets[i].1 == key {
                return false;
            }
            i = self.next(i);
            d += 1;
        }
        if self.len >= self.buckets.len() / 2 {
            self.rehash(2 * self.buckets.len().max(1));
            i = key as usize & self.mask();
            d = 0;
            while d <= self.dist(i) {
                i = self.next(i);
                d += 1;
            }
        }
        self.place(i, d, key);
        self.len += 1;
        true
    }

    /// Put `key`, `d` from its ideal bucket, at bucket `i` (empty, or holding a key closer to
    /// its own), carrying displaced keys on.
    fn place(&mut self, mut i: usize, mut d: i16, mut key: u32) {
        loop {
            let (bd, bk) = self.buckets[i];
            if bd == EMPTY {
                self.buckets[i] = (d, key);
                return;
            }
            if d > bd {
                self.buckets[i] = (d, key);
                (d, key) = (bd, bk);
            }
            i = self.next(i);
            d += 1;
        }
    }

    fn rehash(&mut self, count: usize) {
        let old = std::mem::replace(&mut self.buckets, vec![(EMPTY, 0); count]);
        for (bd, key) in old {
            if bd != EMPTY {
                // robin-map's insert_value_on_rehash: the same robin-hood walk from distance 0.
                let i = key as usize & self.mask();
                self.place(i, 0, key);
            }
        }
    }

    /// Remove `key`; whether it was there.
    pub(crate) fn remove(&mut self, key: u32) -> bool {
        let Some(mut prev) = self.find(key) else { return false };
        self.buckets[prev].0 = EMPTY;
        self.len -= 1;
        let mut i = self.next(prev);
        while self.buckets[i].0 > 0 {
            self.buckets[prev] = (self.buckets[i].0 - 1, self.buckets[i].1);
            self.buckets[i].0 = EMPTY;
            prev = i;
            i = self.next(i);
        }
        true
    }

    /// Empty the set, keeping its buckets.
    pub(crate) fn clear(&mut self) {
        for b in &mut self.buckets {
            b.0 = EMPTY;
        }
        self.len = 0;
    }

    /// The keys in robin-map's iteration order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.buckets.iter().filter(|b| b.0 != EMPTY).map(|b| b.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_code::Xorshift;
    use std::collections::BTreeSet;

    /// The set holds what a BTreeSet holds through any sequence of inserts, removals and
    /// clears, and every key sits no further from its ideal bucket than its run allows.
    #[test]
    fn it_is_a_set() {
        let mut rng = Xorshift::new(9);
        let (mut a, mut b) = (RobinSet::new(), BTreeSet::new());
        for step in 0..20_000 {
            let key = (rng.next_u64() % 200) as u32;
            match rng.next_u64() % 10 {
                0..=5 => assert_eq!(a.insert(key), b.insert(key)),
                6..=8 => assert_eq!(a.remove(key), b.remove(&key)),
                _ if step % 997 == 0 => {
                    a.clear();
                    b.clear();
                }
                _ => assert_eq!(a.contains(key), b.contains(&key)),
            }
            assert_eq!(a.len(), b.len());
            let mut keys: Vec<u32> = a.iter().collect();
            keys.sort_unstable();
            assert_eq!(keys, b.iter().copied().collect::<Vec<_>>());
            assert!(a.buckets.iter().enumerate().all(|(i, &(d, k))| d == EMPTY || (k as usize + d as usize) & a.mask() == i));
        }
    }

    /// Worked by hand. 5, 1, 9: 5 goes in at 2 buckets (5 & 1 = 1); 1 grows the set to 4
    /// buckets (5 → 1, then 1 → 1 is taken, so 1 → 2); 9 grows it to 8 (5 → 5, 1 → 1, then
    /// 9 → 1 is taken by 1 at distance 0, so 9 → 2). Order: 1, 9, 5.
    #[test]
    fn its_order_is_robin_maps() {
        let mut s = RobinSet::new();
        for k in [5, 1, 9] {
            s.insert(k);
        }
        assert_eq!(s.buckets.len(), 8);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![1, 9, 5]);
        // Removing 1 shifts 9 back into its ideal bucket.
        s.remove(1);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![9, 5]);
        assert_eq!(s.buckets[1], (0, 9));
        // A cleared set keeps its 8 buckets: 17 then lands at 17 & 7 = 1, before 3.
        s.clear();
        s.insert(3);
        s.insert(17);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![17, 3]);
    }
}
