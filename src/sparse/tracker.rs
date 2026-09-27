//! The event queue: reminders to look at a node or a region at a given time.
//!
//! Only the earliest reminder per entity is queued, and a reminder made stale
//! by a change of growth rate is not removed: it fires, the flooder recomputes,
//! and finds nothing due. That is the paper's tracker, and it keeps the queue
//! small.
//!
//! Times only increase: a reminder is never set before the last one taken. So
//! the queue is a radix heap. Bucket b ≥ 1 holds reminders whose time first
//! differs from the last time taken in bit b − 1, and taking the next time
//! empties only the lowest non-empty bucket into the ones below it. Pushes cost
//! O(1), and each reminder moves down at most 64 times. Reminders at the
//! current time wait in a small binary heap ordered by item, so they come out in
//! exactly the order `BinaryHeap<Reverse<(i64, Item)>>` gave, and a decode
//! takes the same steps as before.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Item {
    Node(u32),
    Region(u32),
}

/// The flooder's queue of reminders.
pub(crate) type Tracker = RadixHeap<Item>;

/// A min-queue of (time, item) for times that never go backwards: every push
/// is at or after the last time popped. It pops in increasing (time, item)
/// order, exactly as `BinaryHeap<Reverse<(i64, T)>>`. The flooder's reminders
/// use it, and so does the shortest-path search that traces a matching's
/// paths.
pub(crate) struct RadixHeap<T> {
    /// The last time taken; every queued time is at least this.
    last: i64,
    /// Entries at time `last`, least item first.
    current: BinaryHeap<Reverse<T>>,
    /// Bucket b holds times whose highest bit differing from `last` is b − 1.
    buckets: Vec<Vec<(i64, T)>>,
    /// Which buckets hold anything, one bit per bucket.
    occupied: u64,
}

impl<T: Ord + Copy> Default for RadixHeap<T> {
    fn default() -> Self {
        RadixHeap { last: 0, current: BinaryHeap::new(), buckets: (0..64).map(|_| Vec::new()).collect(), occupied: 0 }
    }
}

impl<T: Ord + Copy> RadixHeap<T> {
    fn bucket(&self, t: i64) -> usize {
        64 - ((t ^ self.last) as u64).leading_zeros() as usize
    }

    pub fn clear(&mut self) {
        self.last = 0;
        self.current.clear();
        while self.occupied != 0 {
            let b = self.occupied.trailing_zeros() as usize;
            self.buckets[b].clear();
            self.occupied &= self.occupied - 1;
        }
    }

    pub fn push(&mut self, t: i64, item: T) {
        debug_assert!(t >= self.last, "a reminder at {t} before the last time taken, {}", self.last);
        if t == self.last {
            self.current.push(Reverse(item));
        } else {
            let b = self.bucket(t);
            self.buckets[b].push((t, item));
            self.occupied |= 1 << b;
        }
    }

    pub fn pop(&mut self) -> Option<(i64, T)> {
        if self.current.is_empty() {
            if self.occupied == 0 {
                return None;
            }
            // The lowest non-empty bucket holds the next time; empty it below.
            let b = self.occupied.trailing_zeros() as usize;
            self.occupied &= !(1 << b);
            let mut moving = std::mem::take(&mut self.buckets[b]);
            self.last = moving.iter().map(|x| x.0).min().expect("an occupied bucket is not empty");
            for &(t, item) in &moving {
                if t == self.last {
                    self.current.push(Reverse(item));
                } else {
                    let nb = self.bucket(t);
                    self.buckets[nb].push((t, item));
                    self.occupied |= 1 << nb;
                }
            }
            moving.clear();
            self.buckets[b] = moving;
        }
        self.current.pop().map(|Reverse(item)| (self.last, item))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_code::Xorshift;

    /// Monotone use, as the flooder's: every push is at or after the last pop.
    #[test]
    fn pops_in_exactly_the_binary_heaps_order() {
        let mut rng = Xorshift::new(11);
        for trial in 0..300 {
            let mut radix = Tracker::default();
            let mut heap = std::collections::BinaryHeap::new();
            let mut now = 0i64;
            for _ in 0..2000 {
                if rng.next_u64() % 3 != 0 || heap.is_empty() {
                    // Bursts at the current time, and jumps of every scale.
                    let dt = match rng.next_u64() % 4 {
                        0 => 0,
                        1 => (rng.next_u64() % 8) as i64,
                        2 => (rng.next_u64() % 100_000) as i64,
                        _ => (rng.next_u64() % (1 << 40)) as i64,
                    };
                    let item = if rng.next_u64() % 2 == 0 {
                        Item::Node((rng.next_u64() % 50) as u32)
                    } else {
                        Item::Region((rng.next_u64() % 50) as u32)
                    };
                    radix.push(now + dt, item);
                    heap.push(std::cmp::Reverse((now + dt, item)));
                } else {
                    let a = radix.pop();
                    let b = heap.pop().map(|std::cmp::Reverse(x)| x);
                    assert_eq!(a, b, "trial {trial}");
                    now = a.unwrap().0;
                }
            }
            while let Some(std::cmp::Reverse(x)) = heap.pop() {
                assert_eq!(radix.pop(), Some(x), "trial {trial}, draining");
            }
            assert_eq!(radix.pop(), None);
        }
    }

    #[test]
    fn clear_empties_and_restarts_time() {
        let mut t = Tracker::default();
        t.push(5, Item::Node(1));
        t.push(9, Item::Region(2));
        assert_eq!(t.pop(), Some((5, Item::Node(1))));
        t.clear();
        assert_eq!(t.pop(), None);
        t.push(0, Item::Node(3));
        assert_eq!(t.pop(), Some((0, Item::Node(3))));
    }
}
