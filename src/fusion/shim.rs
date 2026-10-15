//! What Fusion Blossom takes from other crates, written here so that this crate keeps no
//! dependencies: `parking_lot`'s lock (recursive reads), `serde_json`'s value and `json!`
//! macro (snapshots and configurations), `rand_xoshiro`'s generator (its example codes'
//! errors), `rayon`'s pool (units solved on threads), `weak_table`'s pointer-keyed tables,
//! `priority_queue` and `nonzero`. Each keeps the names and behaviour Fusion Blossom uses.

#![allow(dead_code)]

/// `parking_lot::RwLock`: a reader-writer lock whose reads may nest on one thread
/// (`read_recursive`) whatever writers wait, as Fusion Blossom's pointers need.
pub mod parking_lot {
    use std::cell::UnsafeCell;
    use std::marker::PhantomData;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const WRITER: usize = 1 << (usize::BITS - 1);

    /// The raw lock's type, named in guards' types as `lock_api` names it.
    pub struct RawRwLock;

    pub struct RwLock<T: ?Sized> {
        state: AtomicUsize,
        data: UnsafeCell<T>,
    }

    unsafe impl<T: ?Sized + Send> Send for RwLock<T> {}
    unsafe impl<T: ?Sized + Send + Sync> Sync for RwLock<T> {}

    fn backoff(spins: &mut u32) {
        if *spins < 64 {
            std::hint::spin_loop();
            *spins += 1;
        } else {
            std::thread::yield_now();
        }
    }

    impl<T> RwLock<T> {
        pub const fn new(value: T) -> Self {
            RwLock { state: AtomicUsize::new(0), data: UnsafeCell::new(value) }
        }

        pub fn into_inner(self) -> T {
            self.data.into_inner()
        }
    }

    impl<T: ?Sized> RwLock<T> {
        /// A read that waits only for a writer holding the lock, never for one waiting, so
        /// it may nest inside another read on the same thread.
        pub fn read_recursive(&self) -> lock_api::RwLockReadGuard<'_, RawRwLock, T> {
            let mut spins = 0;
            loop {
                let s = self.state.load(Ordering::Relaxed);
                if s & WRITER == 0 && self.state.compare_exchange_weak(s, s + 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                    return lock_api::RwLockReadGuard { lock: self, _raw: PhantomData };
                }
                backoff(&mut spins);
            }
        }

        pub fn read(&self) -> lock_api::RwLockReadGuard<'_, RawRwLock, T> {
            self.read_recursive()
        }

        pub fn write(&self) -> lock_api::RwLockWriteGuard<'_, RawRwLock, T> {
            let mut spins = 0;
            loop {
                if self.state.compare_exchange_weak(0, WRITER, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                    return lock_api::RwLockWriteGuard { lock: self, _raw: PhantomData };
                }
                backoff(&mut spins);
            }
        }

        pub fn get_mut(&mut self) -> &mut T {
            self.data.get_mut()
        }
    }

    impl<T: ?Sized + std::fmt::Debug> std::fmt::Debug for RwLock<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RwLock").field("data", &&*self.read_recursive()).finish()
        }
    }

    pub mod lock_api {
        use super::{RwLock, WRITER};
        use std::marker::PhantomData;
        use std::sync::atomic::Ordering;

        pub struct RwLockReadGuard<'a, R, T: ?Sized> {
            pub(super) lock: &'a RwLock<T>,
            pub(super) _raw: PhantomData<R>,
        }

        pub struct RwLockWriteGuard<'a, R, T: ?Sized> {
            pub(super) lock: &'a RwLock<T>,
            pub(super) _raw: PhantomData<R>,
        }

        impl<R, T: ?Sized> std::ops::Deref for RwLockReadGuard<'_, R, T> {
            type Target = T;
            fn deref(&self) -> &T {
                unsafe { &*self.lock.data.get() }
            }
        }

        impl<R, T: ?Sized> Drop for RwLockReadGuard<'_, R, T> {
            fn drop(&mut self) {
                self.lock.state.fetch_sub(1, Ordering::Release);
            }
        }

        impl<R, T: ?Sized> std::ops::Deref for RwLockWriteGuard<'_, R, T> {
            type Target = T;
            fn deref(&self) -> &T {
                unsafe { &*self.lock.data.get() }
            }
        }

        impl<R, T: ?Sized> std::ops::DerefMut for RwLockWriteGuard<'_, R, T> {
            fn deref_mut(&mut self) -> &mut T {
                unsafe { &mut *self.lock.data.get() }
            }
        }

        impl<R, T: ?Sized> Drop for RwLockWriteGuard<'_, R, T> {
            fn drop(&mut self) {
                debug_assert_eq!(self.lock.state.load(Ordering::Relaxed), WRITER);
                self.lock.state.store(0, Ordering::Release);
            }
        }

        impl<R, T: ?Sized + std::fmt::Debug> std::fmt::Debug for RwLockReadGuard<'_, R, T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                (**self).fmt(f)
            }
        }

        impl<R, T: ?Sized + std::fmt::Debug> std::fmt::Debug for RwLockWriteGuard<'_, R, T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                (**self).fmt(f)
            }
        }
    }
}

/// `rand_xoshiro`'s xoshiro256** and its seeding by SplitMix64, as `rand_xoshiro` 0.6 does.
pub mod rand_xoshiro {
    pub mod rand_core {
        pub trait RngCore {
            fn next_u32(&mut self) -> u32;
            fn next_u64(&mut self) -> u64;
        }

        pub trait SeedableRng: Sized {
            fn seed_from_u64(seed: u64) -> Self;
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Xoshiro256StarStar {
        s: [u64; 4],
    }

    fn splitmix64(x: &mut u64) -> u64 {
        *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    impl rand_core::SeedableRng for Xoshiro256StarStar {
        fn seed_from_u64(seed: u64) -> Self {
            let mut x = seed;
            let s = [splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x)];
            if s == [0; 4] {
                return Self::seed_from_u64(0);
            }
            Xoshiro256StarStar { s }
        }
    }

    impl rand_core::RngCore for Xoshiro256StarStar {
        fn next_u32(&mut self) -> u32 {
            (self.next_u64() >> 32) as u32
        }

        fn next_u64(&mut self) -> u64 {
            let s = &mut self.s;
            let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
            let t = s[1] << 17;
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = s[3].rotate_left(45);
            result
        }
    }
}

/// `rayon`'s pool as Fusion Blossom uses it. Fusion's units are scheduled by `scope_fifo`,
/// which here runs its tasks on the pool's threads in the order they were issued (so a fusion
/// unit, issued after its children, waits only on tasks already running); `join` and the
/// parallel iterators run in order on the calling thread, which `rayon` permits.
pub mod rayon {
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    thread_local! {
        static THREAD_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
    }

    /// The index of the pool thread running this, if a pool thread runs it.
    pub fn current_thread_index() -> Option<usize> {
        THREAD_INDEX.with(|i| i.get())
    }

    pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let ra = a();
        (ra, b())
    }

    pub struct Scope<'scope> {
        _marker: std::marker::PhantomData<&'scope ()>,
    }

    impl<'scope> Scope<'scope> {
        pub fn spawn<F: FnOnce(&Scope<'scope>) + Send + 'scope>(&self, f: F) {
            f(self)
        }
    }

    pub fn scope<'scope, OP, R>(op: OP) -> R
    where
        OP: FnOnce(&Scope<'scope>) -> R,
    {
        op(&Scope { _marker: std::marker::PhantomData })
    }

    type Task<'scope> = Box<dyn FnOnce(&ScopeFifo<'scope>) + Send + 'scope>;

    pub struct ScopeFifo<'scope> {
        tasks: RefCell<Vec<Task<'scope>>>,
    }

    impl<'scope> ScopeFifo<'scope> {
        pub fn spawn_fifo<F: FnOnce(&ScopeFifo<'scope>) + Send + 'scope>(&self, f: F) {
            self.tasks.borrow_mut().push(Box::new(f));
        }
    }

    #[derive(Debug)]
    pub struct ThreadPool {
        threads: usize,
    }

    impl ThreadPool {
        pub fn current_num_threads(&self) -> usize {
            self.threads
        }

        pub fn scope<'scope, OP, R>(&self, op: OP) -> R
        where
            OP: FnOnce(&Scope<'scope>) -> R,
        {
            op(&Scope { _marker: std::marker::PhantomData })
        }

        pub fn install<OP: FnOnce() -> R, R>(&self, op: OP) -> R {
            op()
        }

        /// Runs `op`, then every task it spawned, on the pool's threads in the order spawned.
        pub fn scope_fifo<'scope, OP, R>(&self, op: OP) -> R
        where
            OP: FnOnce(&ScopeFifo<'scope>) -> R,
        {
            let scope = ScopeFifo { tasks: RefCell::new(Vec::new()) };
            let out = op(&scope);
            let tasks = scope.tasks.into_inner();
            if tasks.is_empty() {
                return out;
            }
            let workers = self.threads.max(1).min(tasks.len());
            let queue: Mutex<VecDeque<Task<'scope>>> = Mutex::new(tasks.into());
            std::thread::scope(|s| {
                for index in 0..workers {
                    let queue = &queue;
                    s.spawn(move || {
                        THREAD_INDEX.with(|i| i.set(Some(index)));
                        loop {
                            let task = queue.lock().unwrap().pop_front();
                            match task {
                                Some(task) => {
                                    let inner = ScopeFifo { tasks: RefCell::new(Vec::new()) };
                                    task(&inner);
                                    assert!(inner.tasks.into_inner().is_empty(), "a fusion task spawns no tasks");
                                }
                                None => break,
                            }
                        }
                    });
                }
            });
            out
        }
    }

    #[derive(Default)]
    pub struct ThreadPoolBuilder {
        threads: usize,
    }

    impl ThreadPoolBuilder {
        pub fn new() -> Self {
            ThreadPoolBuilder { threads: 0 }
        }

        pub fn num_threads(mut self, threads: usize) -> Self {
            self.threads = threads;
            self
        }

        /// Pinning threads to cores is not done here; the handler is not called.
        pub fn start_handler<H: Fn(usize) + Send + Sync + 'static>(self, _handler: H) -> Self {
            self
        }

        pub fn build(self) -> Result<ThreadPool, String> {
            let threads = if self.threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { self.threads };
            Ok(ThreadPool { threads })
        }
    }

    pub mod prelude {
        /// `par_iter()`: here the ordinary iterator.
        pub trait IntoParallelRefIterator<'a> {
            type Iter: Iterator;
            fn par_iter(&'a self) -> Self::Iter;
        }

        impl<'a, T: 'a> IntoParallelRefIterator<'a> for [T] {
            type Iter = std::slice::Iter<'a, T>;
            fn par_iter(&'a self) -> Self::Iter {
                self.iter()
            }
        }

        impl<'a, T: 'a> IntoParallelRefIterator<'a> for Vec<T> {
            type Iter = std::slice::Iter<'a, T>;
            fn par_iter(&'a self) -> Self::Iter {
                self.iter()
            }
        }

        /// `into_par_iter()`: here the ordinary iterator.
        pub trait IntoParallelIterator {
            type Iter: Iterator;
            fn into_par_iter(self) -> Self::Iter;
        }

        impl<I: IntoIterator> IntoParallelIterator for I {
            type Iter = I::IntoIter;
            fn into_par_iter(self) -> Self::Iter {
                self.into_iter()
            }
        }

        /// `collect_into_vec`: here collecting in order.
        pub trait CollectIntoVec<T> {
            fn collect_into_vec(self, target: &mut Vec<T>);
        }

        impl<T, I: Iterator<Item = T>> CollectIntoVec<T> for I {
            fn collect_into_vec(self, target: &mut Vec<T>) {
                target.clear();
                target.extend(self);
            }
        }

        /// `par_extend`: here `extend`.
        pub trait ParallelExtend<T> {
            fn par_extend<I: IntoIterator<Item = T>>(&mut self, iter: I);
        }

        impl<T> ParallelExtend<T> for Vec<T> {
            fn par_extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
                self.extend(iter)
            }
        }
    }
}

/// `weak_table`'s pointer-keyed tables, for membership and lookup (Fusion Blossom never
/// iterates them).
pub mod weak_table {
    use std::collections::HashMap;
    use std::ops::Deref;

    pub mod traits {
        pub trait WeakElement {
            type Strong;
            fn new(view: &Self::Strong) -> Self;
            fn view(&self) -> Option<Self::Strong>;
            fn clone(view: &Self::Strong) -> Self::Strong;
        }
    }

    use traits::WeakElement;

    fn address<S: Deref>(strong: &S) -> usize {
        (&**strong) as *const S::Target as *const u8 as usize
    }

    /// Whether the weak element still points at `strong` (an address can be reused once the
    /// element it named is gone).
    fn alive<W: WeakElement>(weak: &W, strong: &W::Strong) -> bool
    where
        W::Strong: Deref,
    {
        weak.view().is_some_and(|v| address(&v) == address(strong))
    }

    pub struct PtrWeakHashSet<W: WeakElement>(HashMap<usize, W>);

    impl<W: WeakElement> Default for PtrWeakHashSet<W> {
        fn default() -> Self {
            PtrWeakHashSet(HashMap::new())
        }
    }

    impl<W: WeakElement> std::fmt::Debug for PtrWeakHashSet<W> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "PtrWeakHashSet({} entries)", self.0.len())
        }
    }

    impl<W: WeakElement> PtrWeakHashSet<W>
    where
        W::Strong: Deref,
    {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn insert(&mut self, strong: W::Strong) -> bool {
            let key = address(&strong);
            let fresh = !self.0.get(&key).is_some_and(|w| alive(w, &strong));
            self.0.insert(key, W::new(&strong));
            fresh
        }

        pub fn contains(&self, strong: &W::Strong) -> bool {
            self.0.get(&address(strong)).is_some_and(|w| alive(w, strong))
        }

        pub fn remove(&mut self, strong: &W::Strong) -> bool {
            let key = address(strong);
            let present = self.contains(strong);
            self.0.remove(&key);
            present
        }

        pub fn clear(&mut self) {
            self.0.clear()
        }
    }

    pub struct PtrWeakKeyHashMap<W: WeakElement, V>(HashMap<usize, (W, V)>);

    impl<W: WeakElement, V> Default for PtrWeakKeyHashMap<W, V> {
        fn default() -> Self {
            PtrWeakKeyHashMap(HashMap::new())
        }
    }

    impl<W: WeakElement, V> std::fmt::Debug for PtrWeakKeyHashMap<W, V> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "PtrWeakKeyHashMap({} entries)", self.0.len())
        }
    }

    impl<W: WeakElement, V> PtrWeakKeyHashMap<W, V>
    where
        W::Strong: Deref,
    {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn insert(&mut self, key: W::Strong, value: V) -> Option<V> {
            let k = address(&key);
            let old = self.0.remove(&k).and_then(|(w, v)| alive(&w, &key).then_some(v));
            self.0.insert(k, (W::new(&key), value));
            old
        }

        pub fn get(&self, key: &W::Strong) -> Option<&V> {
            self.0.get(&address(key)).and_then(|(w, v)| alive(w, key).then_some(v))
        }

        pub fn contains_key(&self, key: &W::Strong) -> bool {
            self.get(key).is_some()
        }

        pub fn remove(&mut self, key: &W::Strong) -> Option<V> {
            let k = address(key);
            match self.0.get(&k) {
                Some((w, _)) if alive(w, key) => self.0.remove(&k).map(|(_, v)| v),
                _ => None,
            }
        }

        pub fn clear(&mut self) {
            self.0.clear()
        }
    }
}

/// `priority_queue::PriorityQueue`: items with a priority, popped highest first, a pushed
/// item's priority replaced if it is already there.
pub mod priority_queue {
    use std::collections::HashMap;
    use std::hash::Hash;

    pub struct PriorityQueue<I, P> {
        items: Vec<(I, P)>,
        index: HashMap<I, usize>,
    }

    impl<I: Hash + Eq + Clone, P: Ord> Default for PriorityQueue<I, P> {
        fn default() -> Self {
            PriorityQueue { items: Vec::new(), index: HashMap::new() }
        }
    }

    impl<I: Hash + Eq + Clone, P: Ord> PriorityQueue<I, P> {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn len(&self) -> usize {
            self.items.len()
        }

        pub fn is_empty(&self) -> bool {
            self.items.is_empty()
        }

        fn swap(&mut self, a: usize, b: usize) {
            self.items.swap(a, b);
            self.index.insert(self.items[a].0.clone(), a);
            self.index.insert(self.items[b].0.clone(), b);
        }

        fn up(&mut self, mut k: usize) {
            while k > 0 {
                let parent = (k - 1) / 2;
                if self.items[k].1 > self.items[parent].1 {
                    self.swap(k, parent);
                    k = parent;
                } else {
                    break;
                }
            }
        }

        fn down(&mut self, mut k: usize) {
            loop {
                let (l, r) = (2 * k + 1, 2 * k + 2);
                let mut top = k;
                if l < self.items.len() && self.items[l].1 > self.items[top].1 {
                    top = l;
                }
                if r < self.items.len() && self.items[r].1 > self.items[top].1 {
                    top = r;
                }
                if top == k {
                    break;
                }
                self.swap(k, top);
                k = top;
            }
        }

        /// Insert `item`, or change its priority if it is there; the old priority, if any.
        pub fn push(&mut self, item: I, priority: P) -> Option<P> {
            if let Some(&k) = self.index.get(&item) {
                let old = std::mem::replace(&mut self.items[k].1, priority);
                self.up(k);
                let k = self.index[&item];
                self.down(k);
                return Some(old);
            }
            self.items.push((item.clone(), priority));
            let k = self.items.len() - 1;
            self.index.insert(item, k);
            self.up(k);
            None
        }

        pub fn change_priority(&mut self, item: &I, priority: P) -> Option<P> {
            if self.index.contains_key(item) {
                self.push(item.clone(), priority)
            } else {
                None
            }
        }

        pub fn get_priority(&self, item: &I) -> Option<&P> {
            self.index.get(item).map(|&k| &self.items[k].1)
        }

        pub fn peek(&self) -> Option<(&I, &P)> {
            self.items.first().map(|(i, p)| (i, p))
        }

        pub fn pop(&mut self) -> Option<(I, P)> {
            if self.items.is_empty() {
                return None;
            }
            let last = self.items.len() - 1;
            self.swap(0, last);
            let (item, priority) = self.items.pop().expect("nonempty");
            self.index.remove(&item);
            if !self.items.is_empty() {
                self.down(0);
            }
            Some((item, priority))
        }
    }
}

/// `serde_json`'s value, map, `json!` macro and text, for Fusion Blossom's snapshots and
/// configurations. Objects keep their keys sorted, as `serde_json`'s default map does.
pub mod serde_json {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;

    #[derive(Clone, Debug, PartialEq)]
    pub enum Number {
        PosInt(u64),
        NegInt(i64),
        Float(f64),
    }

    #[derive(Clone, Debug, PartialEq, Default)]
    pub enum Value {
        #[default]
        Null,
        Bool(bool),
        Number(Number),
        String(String),
        Array(Vec<Value>),
        Object(Map<String, Value>),
    }

    pub type Map<K, V> = BTreeMap<K, V>;

    pub type Error = String;

    impl Value {
        pub fn as_object(&self) -> Option<&Map<String, Value>> {
            match self {
                Value::Object(m) => Some(m),
                _ => None,
            }
        }
        pub fn as_object_mut(&mut self) -> Option<&mut Map<String, Value>> {
            match self {
                Value::Object(m) => Some(m),
                _ => None,
            }
        }
        pub fn as_array(&self) -> Option<&Vec<Value>> {
            match self {
                Value::Array(a) => Some(a),
                _ => None,
            }
        }
        pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value>> {
            match self {
                Value::Array(a) => Some(a),
                _ => None,
            }
        }
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Value::String(s) => Some(s),
                _ => None,
            }
        }
        pub fn as_bool(&self) -> Option<bool> {
            match self {
                Value::Bool(b) => Some(*b),
                _ => None,
            }
        }
        pub fn as_u64(&self) -> Option<u64> {
            match self {
                Value::Number(Number::PosInt(u)) => Some(*u),
                _ => None,
            }
        }
        pub fn as_i64(&self) -> Option<i64> {
            match self {
                Value::Number(Number::PosInt(u)) => i64::try_from(*u).ok(),
                Value::Number(Number::NegInt(i)) => Some(*i),
                _ => None,
            }
        }
        pub fn as_f64(&self) -> Option<f64> {
            match self {
                Value::Number(Number::PosInt(u)) => Some(*u as f64),
                Value::Number(Number::NegInt(i)) => Some(*i as f64),
                Value::Number(Number::Float(f)) => Some(*f),
                _ => None,
            }
        }
        pub fn is_null(&self) -> bool {
            matches!(self, Value::Null)
        }
        pub fn get(&self, key: &str) -> Option<&Value> {
            self.as_object().and_then(|m| m.get(key))
        }
    }

    impl std::ops::Index<&str> for Value {
        type Output = Value;
        fn index(&self, key: &str) -> &Value {
            static NULL: Value = Value::Null;
            self.get(key).unwrap_or(&NULL)
        }
    }

    impl std::ops::IndexMut<&str> for Value {
        fn index_mut(&mut self, key: &str) -> &mut Value {
            if self.is_null() {
                *self = Value::Object(Map::new());
            }
            match self {
                Value::Object(m) => m.entry(key.to_string()).or_insert(Value::Null),
                _ => panic!("cannot index a non-object JSON value with \"{key}\""),
            }
        }
    }

    impl std::fmt::Display for Value {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut out = String::new();
            write_value(self, &mut out);
            f.write_str(&out)
        }
    }

    fn write_string(s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 => {
                    let _ = write!(out, "\\u{:04x}", c as u32);
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }

    fn write_value(v: &Value, out: &mut String) {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(Number::PosInt(u)) => {
                let _ = write!(out, "{u}");
            }
            Value::Number(Number::NegInt(i)) => {
                let _ = write!(out, "{i}");
            }
            Value::Number(Number::Float(x)) => {
                if x.is_finite() {
                    let _ = write!(out, "{x:?}");
                } else {
                    out.push_str("null");
                }
            }
            Value::String(s) => write_string(s, out),
            Value::Array(a) => {
                out.push('[');
                for (k, x) in a.iter().enumerate() {
                    if k > 0 {
                        out.push(',');
                    }
                    write_value(x, out);
                }
                out.push(']');
            }
            Value::Object(m) => {
                out.push('{');
                for (k, (key, x)) in m.iter().enumerate() {
                    if k > 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    write_value(x, out);
                }
                out.push('}');
            }
        }
    }

    /// A type written as a JSON value (`serde::Serialize` for what Fusion Blossom writes).
    pub trait ToJson {
        fn to_json(&self) -> Value;
        /// The text `serde_json::to_string` writes: a struct's fields in their declared order
        /// (`to_json`, like `serde_json::to_value`, sorts an object's keys).
        fn write_text(&self, out: &mut String) {
            let _ = write!(out, "{}", self.to_json());
        }
    }

    fn write_seq<'a, T: ToJson + 'a>(items: impl Iterator<Item = &'a T>, out: &mut String) {
        out.push('[');
        for (k, item) in items.enumerate() {
            if k > 0 {
                out.push(',');
            }
            item.write_text(out);
        }
        out.push(']');
    }

    /// A struct's fields as `serde`'s derive writes them, in order.
    pub fn write_fields(fields: &[(&str, &dyn ToJson)], out: &mut String) {
        out.push('{');
        for (k, (name, value)) in fields.iter().enumerate() {
            if k > 0 {
                out.push(',');
            }
            let _ = write!(out, "\"{name}\":");
            value.write_text(out);
        }
        out.push('}');
    }

    /// A type read from a JSON value (`serde::Deserialize` for what Fusion Blossom reads).
    pub trait FromJson: Sized {
        fn from_json(value: &Value) -> Result<Self, Error>;
    }

    pub fn to_value<T: ToJson + ?Sized>(value: &T) -> Result<Value, Error> {
        Ok(value.to_json())
    }

    pub fn from_value<T: FromJson>(value: Value) -> Result<T, Error> {
        T::from_json(&value)
    }

    pub fn to_string<T: ToJson + ?Sized>(value: &T) -> Result<String, Error> {
        let mut out = String::new();
        value.write_text(&mut out);
        Ok(out)
    }

    pub fn to_writer<W: std::io::Write, T: ToJson + ?Sized>(mut writer: W, value: &T) -> Result<(), Error> {
        writer.write_all(to_string(value)?.as_bytes()).map_err(|e| e.to_string())
    }

    pub fn from_str<T: FromJson>(text: &str) -> Result<T, Error> {
        let mut p = Parser { s: text.as_bytes(), k: 0 };
        let v = p.value()?;
        p.ws();
        if p.k != p.s.len() {
            return Err(format!("trailing characters at {}", p.k));
        }
        T::from_json(&v)
    }

    struct Parser<'a> {
        s: &'a [u8],
        k: usize,
    }

    impl Parser<'_> {
        fn ws(&mut self) {
            while self.k < self.s.len() && matches!(self.s[self.k], b' ' | b'\n' | b'\r' | b'\t') {
                self.k += 1;
            }
        }
        fn eat(&mut self, c: u8) -> Result<(), Error> {
            self.ws();
            if self.s.get(self.k) == Some(&c) {
                self.k += 1;
                Ok(())
            } else {
                Err(format!("expected '{}' at {}", c as char, self.k))
            }
        }
        fn word(&mut self, w: &str, v: Value) -> Result<Value, Error> {
            if self.s[self.k..].starts_with(w.as_bytes()) {
                self.k += w.len();
                Ok(v)
            } else {
                Err(format!("unexpected text at {}", self.k))
            }
        }
        fn value(&mut self) -> Result<Value, Error> {
            self.ws();
            match self.s.get(self.k) {
                None => Err("unexpected end of JSON".into()),
                Some(b'n') => self.word("null", Value::Null),
                Some(b't') => self.word("true", Value::Bool(true)),
                Some(b'f') => self.word("false", Value::Bool(false)),
                Some(b'"') => Ok(Value::String(self.string()?)),
                Some(b'[') => {
                    self.k += 1;
                    let mut a = Vec::new();
                    self.ws();
                    if self.s.get(self.k) == Some(&b']') {
                        self.k += 1;
                        return Ok(Value::Array(a));
                    }
                    loop {
                        a.push(self.value()?);
                        self.ws();
                        match self.s.get(self.k) {
                            Some(b',') => self.k += 1,
                            Some(b']') => {
                                self.k += 1;
                                return Ok(Value::Array(a));
                            }
                            _ => return Err(format!("expected ',' or ']' at {}", self.k)),
                        }
                    }
                }
                Some(b'{') => {
                    self.k += 1;
                    let mut m = Map::new();
                    self.ws();
                    if self.s.get(self.k) == Some(&b'}') {
                        self.k += 1;
                        return Ok(Value::Object(m));
                    }
                    loop {
                        self.ws();
                        let key = self.string()?;
                        self.eat(b':')?;
                        let v = self.value()?;
                        m.insert(key, v);
                        self.ws();
                        match self.s.get(self.k) {
                            Some(b',') => self.k += 1,
                            Some(b'}') => {
                                self.k += 1;
                                return Ok(Value::Object(m));
                            }
                            _ => return Err(format!("expected ',' or '}}' at {}", self.k)),
                        }
                    }
                }
                Some(_) => self.number(),
            }
        }
        fn string(&mut self) -> Result<String, Error> {
            if self.s.get(self.k) != Some(&b'"') {
                return Err(format!("expected a string at {}", self.k));
            }
            self.k += 1;
            let mut out = Vec::new();
            loop {
                match self.s.get(self.k) {
                    None => return Err("unterminated string".into()),
                    Some(b'"') => {
                        self.k += 1;
                        return String::from_utf8(out).map_err(|e| e.to_string());
                    }
                    Some(b'\\') => {
                        let c = *self.s.get(self.k + 1).ok_or("unterminated escape")?;
                        self.k += 2;
                        match c {
                            b'"' => out.push(b'"'),
                            b'\\' => out.push(b'\\'),
                            b'/' => out.push(b'/'),
                            b'b' => out.push(8),
                            b'f' => out.push(12),
                            b'n' => out.push(b'\n'),
                            b'r' => out.push(b'\r'),
                            b't' => out.push(b'\t'),
                            b'u' => {
                                let hex = std::str::from_utf8(self.s.get(self.k..self.k + 4).ok_or("short \\u escape")?).map_err(|e| e.to_string())?;
                                let mut code = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                                self.k += 4;
                                if (0xD800..0xDC00).contains(&code) && self.s.get(self.k..self.k + 2) == Some(b"\\u") {
                                    let hex = std::str::from_utf8(self.s.get(self.k + 2..self.k + 6).ok_or("short \\u escape")?).map_err(|e| e.to_string())?;
                                    let low = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                                    self.k += 6;
                                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                                }
                                let ch = char::from_u32(code).ok_or("invalid \\u escape")?;
                                let mut buf = [0u8; 4];
                                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            }
                            _ => return Err(format!("invalid escape at {}", self.k)),
                        }
                    }
                    Some(&c) => {
                        out.push(c);
                        self.k += 1;
                    }
                }
            }
        }
        fn number(&mut self) -> Result<Value, Error> {
            let start = self.k;
            while self.k < self.s.len() && matches!(self.s[self.k], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                self.k += 1;
            }
            let text = std::str::from_utf8(&self.s[start..self.k]).map_err(|e| e.to_string())?;
            if text.is_empty() {
                return Err(format!("unexpected character at {start}"));
            }
            if let Ok(u) = text.parse::<u64>() {
                return Ok(Value::Number(Number::PosInt(u)));
            }
            if let Ok(i) = text.parse::<i64>() {
                return Ok(Value::Number(Number::NegInt(i)));
            }
            text.parse::<f64>().map(|x| Value::Number(Number::Float(x))).map_err(|_| format!("invalid number '{text}'"))
        }
    }

    // What `json!` and the snapshots write.
    impl ToJson for Value {
        fn to_json(&self) -> Value {
            self.clone()
        }
    }
    impl<T: ToJson + ?Sized> ToJson for &T {
        fn to_json(&self) -> Value {
            (**self).to_json()
        }
        fn write_text(&self, out: &mut String) {
            (**self).write_text(out)
        }
    }
    impl ToJson for bool {
        fn to_json(&self) -> Value {
            Value::Bool(*self)
        }
    }
    macro_rules! unsigned {
        ($($t:ty),*) => {$(
            impl ToJson for $t {
                fn to_json(&self) -> Value {
                    Value::Number(Number::PosInt(*self as u64))
                }
            }
        )*};
    }
    macro_rules! signed {
        ($($t:ty),*) => {$(
            impl ToJson for $t {
                fn to_json(&self) -> Value {
                    if *self >= 0 {
                        Value::Number(Number::PosInt(*self as u64))
                    } else {
                        Value::Number(Number::NegInt(*self as i64))
                    }
                }
            }
        )*};
    }
    unsigned!(u8, u16, u32, u64, usize);
    signed!(i8, i16, i32, i64, isize);
    impl ToJson for f64 {
        fn to_json(&self) -> Value {
            if self.is_finite() {
                Value::Number(Number::Float(*self))
            } else {
                Value::Null
            }
        }
    }
    impl ToJson for f32 {
        fn to_json(&self) -> Value {
            (*self as f64).to_json()
        }
    }
    impl ToJson for str {
        fn to_json(&self) -> Value {
            Value::String(self.to_string())
        }
    }
    impl ToJson for String {
        fn to_json(&self) -> Value {
            Value::String(self.clone())
        }
    }
    impl<T: ToJson> ToJson for Option<T> {
        fn to_json(&self) -> Value {
            self.as_ref().map_or(Value::Null, |v| v.to_json())
        }
        fn write_text(&self, out: &mut String) {
            match self {
                Some(v) => v.write_text(out),
                None => out.push_str("null"),
            }
        }
    }
    impl<T: ToJson> ToJson for [T] {
        fn to_json(&self) -> Value {
            Value::Array(self.iter().map(|v| v.to_json()).collect())
        }
        fn write_text(&self, out: &mut String) {
            write_seq(self.iter(), out)
        }
    }
    impl<T: ToJson> ToJson for Vec<T> {
        fn to_json(&self) -> Value {
            self.as_slice().to_json()
        }
        fn write_text(&self, out: &mut String) {
            self.as_slice().write_text(out)
        }
    }
    impl<T: ToJson, const N: usize> ToJson for [T; N] {
        fn to_json(&self) -> Value {
            self.as_slice().to_json()
        }
        fn write_text(&self, out: &mut String) {
            self.as_slice().write_text(out)
        }
    }
    impl<A: ToJson, B: ToJson> ToJson for (A, B) {
        fn to_json(&self) -> Value {
            Value::Array(vec![self.0.to_json(), self.1.to_json()])
        }
        fn write_text(&self, out: &mut String) {
            write_fields_seq(&[&self.0, &self.1], out)
        }
    }
    impl<A: ToJson, B: ToJson, C: ToJson> ToJson for (A, B, C) {
        fn to_json(&self) -> Value {
            Value::Array(vec![self.0.to_json(), self.1.to_json(), self.2.to_json()])
        }
        fn write_text(&self, out: &mut String) {
            write_fields_seq(&[&self.0, &self.1, &self.2], out)
        }
    }

    fn write_fields_seq(items: &[&dyn ToJson], out: &mut String) {
        out.push('[');
        for (k, item) in items.iter().enumerate() {
            if k > 0 {
                out.push(',');
            }
            item.write_text(out);
        }
        out.push(']');
    }
    impl<K: std::fmt::Display, V: ToJson> ToJson for BTreeMap<K, V> {
        fn to_json(&self) -> Value {
            Value::Object(self.iter().map(|(k, v)| (k.to_string(), v.to_json())).collect())
        }
    }
    impl<T: ToJson> ToJson for std::collections::BTreeSet<T> {
        fn to_json(&self) -> Value {
            Value::Array(self.iter().map(|v| v.to_json()).collect())
        }
        fn write_text(&self, out: &mut String) {
            write_seq(self.iter(), out)
        }
    }

    // What the configurations read.
    impl FromJson for Value {
        fn from_json(value: &Value) -> Result<Self, Error> {
            Ok(value.clone())
        }
    }
    impl FromJson for bool {
        fn from_json(value: &Value) -> Result<Self, Error> {
            value.as_bool().ok_or_else(|| format!("expected a boolean, got {value}"))
        }
    }
    macro_rules! from_integer {
        ($($t:ty),*) => {$(
            impl FromJson for $t {
                fn from_json(value: &Value) -> Result<Self, Error> {
                    let n = match value {
                        Value::Number(Number::PosInt(u)) => <$t>::try_from(*u).ok(),
                        Value::Number(Number::NegInt(i)) => <$t>::try_from(*i).ok(),
                        _ => None,
                    };
                    n.ok_or_else(|| format!("expected an integer ({}), got {value}", stringify!($t)))
                }
            }
        )*};
    }
    from_integer!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);
    impl FromJson for f64 {
        fn from_json(value: &Value) -> Result<Self, Error> {
            value.as_f64().ok_or_else(|| format!("expected a number, got {value}"))
        }
    }
    impl FromJson for String {
        fn from_json(value: &Value) -> Result<Self, Error> {
            value.as_str().map(str::to_string).ok_or_else(|| format!("expected a string, got {value}"))
        }
    }
    impl<T: FromJson> FromJson for Option<T> {
        fn from_json(value: &Value) -> Result<Self, Error> {
            if value.is_null() {
                Ok(None)
            } else {
                T::from_json(value).map(Some)
            }
        }
    }
    impl<T: FromJson> FromJson for Vec<T> {
        fn from_json(value: &Value) -> Result<Self, Error> {
            value.as_array().ok_or_else(|| format!("expected an array, got {value}"))?.iter().map(T::from_json).collect()
        }
    }
    impl<A: FromJson, B: FromJson> FromJson for (A, B) {
        fn from_json(value: &Value) -> Result<Self, Error> {
            match value.as_array().map(Vec::as_slice) {
                Some([a, b]) => Ok((A::from_json(a)?, B::from_json(b)?)),
                _ => Err(format!("expected a pair, got {value}")),
            }
        }
    }
    impl<A: FromJson, B: FromJson, C: FromJson> FromJson for (A, B, C) {
        fn from_json(value: &Value) -> Result<Self, Error> {
            match value.as_array().map(Vec::as_slice) {
                Some([a, b, c]) => Ok((A::from_json(a)?, B::from_json(b)?, C::from_json(c)?)),
                _ => Err(format!("expected a triple, got {value}")),
            }
        }
    }

    /// Reads the fields of a JSON object into a struct: `field(name, default)` takes a field,
    /// or its default where absent, and `finish` refuses fields it did not take (serde's
    /// `deny_unknown_fields`) when `strict`.
    pub struct Fields<'a> {
        object: &'a Map<String, Value>,
        taken: Vec<&'static str>,
        what: &'static str,
    }

    impl<'a> Fields<'a> {
        pub fn of(value: &'a Value, what: &'static str) -> Result<Self, Error> {
            let object = value.as_object().ok_or_else(|| format!("{what}: expected an object, got {value}"))?;
            Ok(Fields { object, taken: Vec::new(), what })
        }

        pub fn field<T: FromJson>(&mut self, name: &'static str, default: Option<fn() -> T>) -> Result<T, Error> {
            self.taken.push(name);
            match (self.object.get(name), default) {
                (Some(v), _) => T::from_json(v).map_err(|e| format!("{}.{name}: {e}", self.what)),
                (None, Some(d)) => Ok(d()),
                (None, None) => Err(format!("{}: missing field `{name}`", self.what)),
            }
        }

        /// A field that may be absent (an `Option` field, which serde defaults to `None`).
        pub fn optional<T: FromJson>(&mut self, name: &'static str) -> Result<Option<T>, Error> {
            self.taken.push(name);
            match self.object.get(name) {
                Some(v) => Option::<T>::from_json(v).map_err(|e| format!("{}.{name}: {e}", self.what)),
                None => Ok(None),
            }
        }

        pub fn finish(self, strict: bool) -> Result<(), Error> {
            if strict {
                if let Some(k) = self.object.keys().find(|k| !self.taken.contains(&k.as_str())) {
                    return Err(format!("{}: unknown field `{k}`, expected one of {:?}", self.what, self.taken));
                }
            }
            Ok(())
        }
    }
}

/// `serde_json::json!`, after `serde_json`'s own macro: JSON written in Rust, its values any
/// expression of a `ToJson` type and its keys any expression giving a string.
macro_rules! json {
    ($($json:tt)+) => {
        json_internal!($($json)+)
    };
}

macro_rules! json_internal {
    (@array [$($elems:expr,)*]) => {
        vec![$($elems,)*]
    };
    (@array [$($elems:expr),*]) => {
        vec![$($elems),*]
    };
    (@array [$($elems:expr,)*] null $($rest:tt)*) => {
        json_internal!(@array [$($elems,)* json_internal!(null)] $($rest)*)
    };
    (@array [$($elems:expr,)*] true $($rest:tt)*) => {
        json_internal!(@array [$($elems,)* json_internal!(true)] $($rest)*)
    };
    (@array [$($elems:expr,)*] false $($rest:tt)*) => {
        json_internal!(@array [$($elems,)* json_internal!(false)] $($rest)*)
    };
    (@array [$($elems:expr,)*] [$($array:tt)*] $($rest:tt)*) => {
        json_internal!(@array [$($elems,)* json_internal!([$($array)*])] $($rest)*)
    };
    (@array [$($elems:expr,)*] {$($map:tt)*} $($rest:tt)*) => {
        json_internal!(@array [$($elems,)* json_internal!({$($map)*})] $($rest)*)
    };
    (@array [$($elems:expr,)*] $next:expr, $($rest:tt)*) => {
        json_internal!(@array [$($elems,)* json_internal!($next),] $($rest)*)
    };
    (@array [$($elems:expr,)*] $last:expr) => {
        json_internal!(@array [$($elems,)* json_internal!($last)])
    };
    (@array [$($elems:expr),*] , $($rest:tt)*) => {
        json_internal!(@array [$($elems,)*] $($rest)*)
    };

    (@object $object:ident () () ()) => {};
    (@object $object:ident [$($key:tt)+] ($value:expr) , $($rest:tt)*) => {
        let _ = $object.insert(::std::string::String::from($($key)+), $value);
        json_internal!(@object $object () ($($rest)*) ($($rest)*));
    };
    (@object $object:ident [$($key:tt)+] ($value:expr)) => {
        let _ = $object.insert(::std::string::String::from($($key)+), $value);
    };
    (@object $object:ident ($($key:tt)+) (: null $($rest:tt)*) $copy:tt) => {
        json_internal!(@object $object [$($key)+] (json_internal!(null)) $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: true $($rest:tt)*) $copy:tt) => {
        json_internal!(@object $object [$($key)+] (json_internal!(true)) $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: false $($rest:tt)*) $copy:tt) => {
        json_internal!(@object $object [$($key)+] (json_internal!(false)) $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: [$($array:tt)*] $($rest:tt)*) $copy:tt) => {
        json_internal!(@object $object [$($key)+] (json_internal!([$($array)*])) $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: {$($map:tt)*} $($rest:tt)*) $copy:tt) => {
        json_internal!(@object $object [$($key)+] (json_internal!({$($map)*})) $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: $value:expr , $($rest:tt)*) $copy:tt) => {
        json_internal!(@object $object [$($key)+] (json_internal!($value)) , $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: $value:expr) $copy:tt) => {
        json_internal!(@object $object [$($key)+] (json_internal!($value)));
    };
    (@object $object:ident () (($key:expr) : $($rest:tt)*) $copy:tt) => {
        json_internal!(@object $object ($key) (: $($rest)*) (: $($rest)*));
    };
    (@object $object:ident ($($key:tt)*) ($tt:tt $($rest:tt)*) $copy:tt) => {
        json_internal!(@object $object ($($key)* $tt) ($($rest)*) ($($rest)*));
    };

    (null) => {
        $crate::fusion::shim::serde_json::Value::Null
    };
    (true) => {
        $crate::fusion::shim::serde_json::Value::Bool(true)
    };
    (false) => {
        $crate::fusion::shim::serde_json::Value::Bool(false)
    };
    ([]) => {
        $crate::fusion::shim::serde_json::Value::Array(vec![])
    };
    ([ $($tt:tt)+ ]) => {
        $crate::fusion::shim::serde_json::Value::Array(json_internal!(@array [] $($tt)+))
    };
    ({}) => {
        $crate::fusion::shim::serde_json::Value::Object($crate::fusion::shim::serde_json::Map::new())
    };
    ({ $($tt:tt)+ }) => {
        $crate::fusion::shim::serde_json::Value::Object({
            let mut object = $crate::fusion::shim::serde_json::Map::new();
            json_internal!(@object object () ($($tt)+) ($($tt)+));
            object
        })
    };
    ($other:expr) => {
        $crate::fusion::shim::serde_json::ToJson::to_json(&$other)
    };
}

/// `nonzero::nonzero!`: a nonzero constant.
macro_rules! nz {
    ($e:expr) => {
        ::std::num::NonZeroUsize::new($e as usize).expect("nonzero")
    };
}
