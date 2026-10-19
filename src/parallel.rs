//! Work over shots shared among threads: one contiguous range per thread, or one shot at a
//! time to whichever thread is free (`parallel_dynamic`). Used by the Python bindings; the
//! WebAssembly build has one thread and does not call it.
#![cfg_attr(not(feature = "python"), allow(dead_code))]

use std::ops::Range;

/// `threads = 0` is every core; never more threads than units of work.
pub fn resolve_threads(threads: usize, units: usize) -> usize {
    let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads };
    threads.clamp(1, units.max(1))
}

/// `work` over `0..n`, one contiguous range per thread, the results in order.
pub fn parallel<T: Send>(n: usize, threads: usize, work: impl Fn(Range<usize>) -> Vec<T> + Sync) -> Vec<T> {
    let threads = resolve_threads(threads, n);
    let chunk = n.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let work = &work;
        let handles: Vec<_> =
            (0..threads).map(|t| scope.spawn(move || work((t * chunk).min(n)..((t + 1) * chunk).min(n)))).collect();
        handles.into_iter().flat_map(|h| h.join().expect("a worker thread panicked")).collect()
    })
}

/// `work(state, i)` for each `i` in `0..n`, each thread taking the next `i` as it finishes one
/// (for work whose cost varies from item to item) with its own state, `init()` made once per
/// thread; the results in order.
pub fn parallel_dynamic<S, T: Send>(n: usize, threads: usize, init: impl Fn() -> S + Sync, work: impl Fn(&mut S, usize) -> T + Sync) -> Vec<T> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    if n == 0 {
        return Vec::new();
    }
    let threads = resolve_threads(threads, n);
    let next = AtomicUsize::new(0);
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    std::thread::scope(|scope| {
        let (init, work, next) = (&init, &work, &next);
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(move || {
                    let mut state = init();
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break done;
                        }
                        done.push((i, work(&mut state, i)));
                    }
                })
            })
            .collect();
        for h in handles {
            for (i, t) in h.join().expect("a worker thread panicked") {
                out[i] = Some(t);
            }
        }
    });
    out.into_iter().map(|t| t.expect("every item is done")).collect()
}
