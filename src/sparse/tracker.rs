//! The event queue: reminders to look at a node or a region at a given time.
//!
//! Only the earliest reminder per entity is queued, and a reminder made stale
//! by a change of growth rate is not removed: it fires, the flooder recomputes,
//! and finds nothing due. That is the paper's tracker, and it keeps the queue
//! small.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Item {
    Node(u32),
    Region(u32),
}

#[derive(Default)]
pub(crate) struct Tracker {
    heap: BinaryHeap<Reverse<(i64, Item)>>,
}

impl Tracker {
    pub fn clear(&mut self) {
        self.heap.clear();
    }
    pub fn push(&mut self, t: i64, item: Item) {
        self.heap.push(Reverse((t, item)));
    }
    pub fn pop(&mut self) -> Option<(i64, Item)> {
        self.heap.pop().map(|Reverse(x)| x)
    }
}
