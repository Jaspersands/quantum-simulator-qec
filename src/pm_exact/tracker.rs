// Copyright 2022 PyMatching Contributors
// Licensed under the Apache License, Version 2.0 (see src/pm_exact/LICENSE). Modified: transliterated
// to Rust for this crate.

//! PyMatching's timekeeping: `Varying` (`flooder_matcher_interop/varying.inl`), `cyclic<uint32_t>`
//! (`tracker/cyclic.h`), the radix heap of flood-check events (`tracker/radix_heap_queue.h`), with
//! its last-in-first-out ties, and the `QueuedEventTracker` (`tracker/queued_event_tracker.h`)
//! that keeps at most one live event per node or region.

/// A radius that varies with time: the low two bits are its slope (1 growing, 2 shrinking, 0
/// frozen), the rest its y intercept (`data >> 2`, arithmetic). PyMatching's `VaryingCT`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Varying(pub i64);

impl Varying {
    #[inline]
    pub fn get_distance_at_time(self, time: i64) -> i64 {
        if self.0 & 1 != 0 {
            (self.0 >> 2) + time
        } else if self.0 & 2 != 0 {
            (self.0 >> 2) - time
        } else {
            self.0 >> 2
        }
    }

    #[inline]
    pub fn y_intercept(self) -> i64 {
        self.0 >> 2
    }

    #[inline]
    pub fn is_growing(self) -> bool {
        self.0 & 1 != 0
    }

    #[inline]
    pub fn is_shrinking(self) -> bool {
        self.0 & 2 != 0
    }

    #[inline]
    pub fn is_frozen(self) -> bool {
        self.0 & 3 == 0
    }

    #[inline]
    pub fn time_of_x_intercept_for_shrinking(self) -> i64 {
        self.0 >> 2
    }

    #[inline]
    pub fn then_growing_at_time(self, t: i64) -> Varying {
        let d = self.get_distance_at_time(t);
        Varying(((d - t) << 2) + 1)
    }

    #[inline]
    pub fn then_shrinking_at_time(self, t: i64) -> Varying {
        let d = self.get_distance_at_time(t);
        Varying(((d + t) << 2) + 2)
    }

    #[inline]
    pub fn then_frozen_at_time(self, t: i64) -> Varying {
        Varying(self.get_distance_at_time(t) << 2)
    }

    #[inline]
    pub fn growing_varying_with_zero_distance_at_time(t: i64) -> Varying {
        Varying(((-t) << 2) + 1)
    }

    /// `Varying + offset`: the y intercept moved by `offset`.
    #[inline]
    pub fn plus(self, offset: i64) -> Varying {
        Varying(self.0 + (offset << 2))
    }
}

/// `cyclic<uint32_t>`'s `a > b`.
#[inline]
pub fn cyclic_gt(a: u32, b: u32) -> bool {
    a.wrapping_sub(b).wrapping_sub(1) < (1u32 << 31) - 1
}

/// `cyclic<uint32_t>`'s `a < b`.
#[inline]
pub fn cyclic_lt(a: u32, b: u32) -> bool {
    cyclic_gt(b, a)
}

/// `cyclic<uint32_t>::widen_from_nearby_reference`.
#[inline]
pub fn widen_from_nearby_reference(value: u32, reference: i64) -> i64 {
    let cref = reference as u32;
    let mut result = reference + value.wrapping_sub(cref) as i64;
    if cyclic_gt(cref, value) {
        result -= 1i64 << 32;
    }
    result
}

/// A flood-check event: a time (cyclic) and what to look at. `item`'s high bit marks a region
/// (`LOOK_AT_SHRINKING_REGION`); otherwise it is a node (`LOOK_AT_NODE`, or `LOOK_AT_SEARCH_NODE`
/// in the search flooder's queue). PyMatching's `NO_FLOOD_CHECK_EVENT` is `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloodEvent {
    pub time: u32,
    pub item: u32,
}

pub const REGION_BIT: u32 = 1 << 31;

impl FloodEvent {
    #[inline]
    pub fn node(node: u32, time: u32) -> FloodEvent {
        FloodEvent { time, item: node }
    }

    #[inline]
    pub fn region(region: u32, time: u32) -> FloodEvent {
        FloodEvent { time, item: region | REGION_BIT }
    }

    #[inline]
    pub fn is_region(self) -> bool {
        self.item & REGION_BIT != 0
    }

    #[inline]
    pub fn index(self) -> u32 {
        self.item & !REGION_BIT
    }
}

/// `radix_heap_queue<false>`: 33 buckets by the highest bit where an event's time differs from
/// the present; bucket 0 is popped from the back, so equal times come out last in, first out.
#[derive(Clone, Debug)]
pub struct RadixHeapQueue {
    buckets: [Vec<FloodEvent>; 33],
    pub cur_time: i64,
    num_enqueued: usize,
}

impl Default for RadixHeapQueue {
    fn default() -> Self {
        RadixHeapQueue { buckets: std::array::from_fn(|_| Vec::new()), cur_time: 0, num_enqueued: 0 }
    }
}

impl RadixHeapQueue {
    #[inline]
    pub fn len(&self) -> usize {
        self.num_enqueued
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_enqueued == 0
    }

    #[inline]
    fn bucket_for(&self, time: u32) -> usize {
        (32 - (time ^ self.cur_time as u32).leading_zeros()) as usize
    }

    #[inline]
    pub fn enqueue(&mut self, ev: FloodEvent) {
        let b = self.bucket_for(ev.time);
        self.buckets[b].push(ev);
        self.num_enqueued += 1;
    }

    pub fn dequeue(&mut self) -> Option<FloodEvent> {
        if self.num_enqueued == 0 {
            return None;
        }
        if self.buckets[0].is_empty() {
            let mut b = 1;
            while self.buckets[b].is_empty() {
                b += 1;
            }
            if b == 1 {
                self.buckets.swap(0, 1);
                self.cur_time += 1;
            } else {
                let mut source = std::mem::take(&mut self.buckets[b]);
                // The plain minimum of the 32-bit times, as PyMatching takes it.
                let mut min_time = source[0].time;
                for e in &source[1..] {
                    min_time = min_time.min(e.time);
                }
                self.cur_time = widen_from_nearby_reference(min_time, self.cur_time);
                for &e in &source {
                    let t = self.bucket_for(e.time);
                    self.buckets[t].push(e);
                }
                source.clear();
                self.buckets[b] = source;
            }
        }
        self.num_enqueued -= 1;
        self.buckets[0].pop()
    }

    /// The events in the queue in time order (ties in no particular order), for drawing.
    pub fn to_vec(&self) -> Vec<FloodEvent> {
        let mut v: Vec<FloodEvent> = self.buckets[..32].iter().flatten().copied().collect();
        let now = self.cur_time as u32;
        v.sort_by_key(|e| e.time.wrapping_sub(now));
        v
    }

    pub fn clear(&mut self) {
        for b in self.buckets.iter_mut() {
            b.clear();
        }
        self.num_enqueued = 0;
    }

    pub fn reset(&mut self) {
        self.clear();
        self.cur_time = 0;
    }
}

/// `QueuedEventTracker`: the one event a node or region wants looked at, and the one it has in
/// the queue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueuedEventTracker {
    pub desired_time: u32,
    pub queued_time: u32,
    pub has_desired_time: bool,
    pub has_queued_time: bool,
}

impl QueuedEventTracker {
    #[inline]
    pub fn clear(&mut self) {
        *self = QueuedEventTracker::default();
    }

    #[inline]
    pub fn set_desired_event(&mut self, ev: FloodEvent, queue: &mut RadixHeapQueue) {
        self.has_desired_time = true;
        self.desired_time = ev.time;
        if !self.has_queued_time || cyclic_gt(self.queued_time, ev.time) {
            self.queued_time = ev.time;
            self.has_queued_time = true;
            queue.enqueue(ev);
        }
    }

    #[inline]
    pub fn set_no_desired_event(&mut self) {
        self.has_desired_time = false;
    }

    /// Whether to process a dequeued event (false: discard it), requeueing it at the desired
    /// time if that moved later.
    #[inline]
    pub fn dequeue_decision(&mut self, mut ev: FloodEvent, queue: &mut RadixHeapQueue) -> bool {
        if !self.has_queued_time || ev.time != self.queued_time {
            return false;
        }
        self.has_queued_time = false;
        if !self.has_desired_time {
            return false;
        }
        if ev.time != self.desired_time {
            self.has_queued_time = true;
            self.queued_time = self.desired_time;
            ev.time = self.desired_time;
            queue.enqueue(ev);
            return false;
        }
        self.has_desired_time = false;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varying_follows_pymatching() {
        let v = Varying::growing_varying_with_zero_distance_at_time(10);
        assert!(v.is_growing());
        assert_eq!(v.get_distance_at_time(15), 5);
        let f = v.then_frozen_at_time(15);
        assert!(f.is_frozen());
        assert_eq!(f.y_intercept(), 5);
        let s = f.then_shrinking_at_time(20);
        assert!(s.is_shrinking());
        assert_eq!(s.get_distance_at_time(22), 3);
        assert_eq!(s.time_of_x_intercept_for_shrinking(), 25);
        assert_eq!(Varying(1).plus(-3).get_distance_at_time(4), 1);
    }

    #[test]
    fn cyclic_comparisons_and_widening() {
        assert!(cyclic_gt(5, 3) && !cyclic_gt(3, 5) && !cyclic_gt(3, 3));
        assert!(cyclic_gt(1, u32::MAX));
        assert!(cyclic_lt(u32::MAX, 1));
        assert_eq!(widen_from_nearby_reference(7, 5), 7);
        assert_eq!(widen_from_nearby_reference(3, 5), 3);
        assert_eq!(widen_from_nearby_reference(1, (1i64 << 32) - 2), (1i64 << 32) + 1);
    }

    #[test]
    fn queue_pops_in_time_order_with_lifo_ties() {
        let mut q = RadixHeapQueue::default();
        for (t, k) in [(5u32, 0u32), (2, 1), (5, 2), (9, 3), (2, 4), (0, 5)] {
            q.enqueue(FloodEvent::node(k, t));
        }
        let mut got = Vec::new();
        while let Some(e) = q.dequeue() {
            got.push((e.time, e.item, q.cur_time));
        }
        assert_eq!(got, vec![(0, 5, 0), (2, 4, 2), (2, 1, 2), (5, 2, 5), (5, 0, 5), (9, 3, 9)]);
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn tracker_keeps_one_live_event() {
        let mut q = RadixHeapQueue::default();
        let mut t = QueuedEventTracker::default();
        t.set_desired_event(FloodEvent::node(0, 10), &mut q);
        t.set_desired_event(FloodEvent::node(0, 4), &mut q);
        assert_eq!(q.len(), 2);
        let e = q.dequeue().unwrap();
        assert_eq!(e.time, 4);
        assert!(t.dequeue_decision(e, &mut q));
        let e = q.dequeue().unwrap();
        assert!(!t.dequeue_decision(e, &mut q));
        // A desired time later than the queued one requeues instead.
        t.set_desired_event(FloodEvent::node(0, 12), &mut q);
        t.set_desired_event(FloodEvent::node(0, 15), &mut q);
        let e = q.dequeue().unwrap();
        assert_eq!(e.time, 12);
        assert!(!t.dequeue_decision(e, &mut q));
        let e = q.dequeue().unwrap();
        assert_eq!(e.time, 15);
        assert!(t.dequeue_decision(e, &mut q));
    }
}
