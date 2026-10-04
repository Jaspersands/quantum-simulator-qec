//! Work over shots cut into one contiguous range per thread. Used by the Python bindings; the
//! WebAssembly build has one thread and does not call it.
#![cfg_attr(not(feature = "python"), allow(dead_code))]

use std::ops::Range;

/// `threads = 0` is every core; never more threads than units of work.
pub fn resolve_threads(threads: usize, units: usize) -> usize {
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads
    };
    threads.clamp(1, units.max(1))
}

/// `work` over `0..n`, one contiguous range per thread, the results in order.
pub fn parallel<T: Send>(
    n: usize,
    threads: usize,
    work: impl Fn(Range<usize>) -> Vec<T> + Sync,
) -> Vec<T> {
    let threads = resolve_threads(threads, n);
    let chunk = n.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let work = &work;
        let handles: Vec<_> = (0..threads)
            .map(|t| scope.spawn(move || work((t * chunk).min(n)..((t + 1) * chunk).min(n))))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a worker thread panicked"))
            .collect()
    })
}
