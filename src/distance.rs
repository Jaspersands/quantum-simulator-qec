//! Circuit distance: the fewest faults that together flip an observable and set off no detector.
//!
//! WHY THIS EXISTS
//! ---------------
//! A code's distance is a property of its checks; a circuit's is smaller wherever one fault in
//! the syndrome circuit does the work of several on the data. It is the number that decides how
//! fast the logical error rate falls, and the one worth stating for every circuit the package
//! builds. Stim finds it two ways, both here: over the graph-like pieces of a decomposed model
//! (`shortest_graphlike_error`, exact for what it searches), and by a bounded search over the
//! model's faults themselves, hyperedges included (`search_for_undetectable_logical_errors`).

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};

use crate::dem::Dem;

/// A fault as its detectors and observables.
pub type Fault = (Vec<u32>, u64);

/// The smallest set of the model's graph-like faults that flips an observable and no detector,
/// found as Stim's `shortest_graphlike_error` finds it, so that it is Stim's own answer, fault
/// for fault (`stim/search/graphlike/algo.cc`):
///
/// - Each fault of nonzero probability with at most two detectors is an edge between them (or
///   from its one detector to the boundary, which is not a node), each node's edges kept in the
///   order they first appear, without repeats. With `ignore_ungraphlike`, a decomposed fault
///   (`^`-separated) or one of more detectors is left out; without it, each piece of a
///   decomposed fault is an edge, and a piece of three or more detectors is refused. A fault (or
///   piece) flipping only observables is a logical error of one fault by itself.
/// - One breadth-first search over states (a moving detection event, a held one, the
///   observables flipped so far), started from every edge that flips an observable, in node
///   order: each step moves the moving event along an edge, a state is visited once whichever
///   event moves, and when the moving event reaches the boundary the held one moves next. The
///   first state with no event left and an observable flipped is the answer; its faults are
///   sorted as Stim sorts them (by their targets, observables after detectors).
pub fn shortest_graphlike(dem: &Dem, ignore_ungraphlike: bool) -> Result<Vec<Fault>, String> {
    const NONE: u32 = u32::MAX;
    let n = dem.num_detectors;
    let mut adj: Vec<Vec<(u32, u64)>> = vec![Vec::new(); n];
    let add = |adj: &mut Vec<Vec<(u32, u64)>>, src: u32, dst: u32, obs: u64| {
        let edges = &mut adj[src as usize];
        if !edges.contains(&(dst, obs)) {
            edges.push((dst, obs));
        }
    };
    let mut distance_1 = 0u64;
    for m in dem.mechanisms.iter().filter(|m| m.p != 0.0) {
        let decomposed = m.pieces.len() > 1;
        if decomposed && ignore_ungraphlike {
            continue;
        }
        let whole = [(m.detectors.as_slice(), m.observables)];
        let split: Vec<(&[u32], u64)> = if decomposed { m.pieces.iter().map(|p| (p.detectors.as_slice(), p.observables)).collect() } else { Vec::new() };
        for &(dets, obs) in if decomposed { split.as_slice() } else { whole.as_slice() } {
            match *dets {
                [] => {
                    if distance_1 == 0 && obs != 0 {
                        distance_1 = obs;
                    }
                }
                [a] => add(&mut adj, a, NONE, obs),
                [a, b] => {
                    add(&mut adj, a, b, obs);
                    add(&mut adj, b, a, obs);
                }
                _ if ignore_ungraphlike => {}
                _ => return Err(format!("a fault sets off {} detectors: not graph-like (decompose the model, or ignore such faults)", dets.len())),
            }
        }
    }
    if distance_1 != 0 {
        return Ok(vec![(Vec::new(), distance_1)]);
    }
    // Each visited state keeps the state it was reached from.
    let empty: State = (NONE, NONE, 0);
    let mut back = Back::default();
    back.insert(empty, empty);
    let mut queue: VecDeque<State> = VecDeque::new();
    for (node1, edges) in adj.iter().enumerate() {
        for &(node2, obs) in edges {
            if (node1 as u32) < node2 && obs != 0 {
                let start = (node1 as u32, node2, obs);
                queue.push_back(start);
                back.entry(key(start)).or_insert(empty);
            }
        }
    }
    while let Some(cur) = queue.pop_front() {
        for &(opp, obs) in &adj[cur.0 as usize] {
            let next = (opp, cur.1, obs ^ cur.2);
            match back.entry(key(next)) {
                Entry::Occupied(_) => continue,
                Entry::Vacant(v) => {
                    v.insert(cur);
                }
            }
            if next.0 == next.1 {
                return Ok(backtrack(&back, next));
            }
            // One event resolved at the boundary: move the other.
            queue.push_back(if next.0 == NONE { (next.1, next.0, next.2) } else { next });
        }
    }
    Err(no_logical_error(dem, "Failed to find any graphlike logical errors.", Some(adj.iter().any(|e| !e.is_empty()))))
}

/// Stim's message when a search finds nothing, with its warnings for a model that makes the
/// search vacuous (`stim/search/{graphlike,hyper}/algo.cc`); `graphlike_edges` is whether the
/// graph-like search had any edge.
fn no_logical_error(dem: &Dem, head: &str, graphlike_edges: Option<bool>) -> String {
    let mut msg = head.to_string();
    if dem.num_observables == 0 {
        msg += "\n    WARNING: NO OBSERVABLES. The circuit or detector error model didn't define any observables, making it vacuously impossible to find a logical error.";
    }
    if dem.num_detectors == 0 {
        msg += "\n    WARNING: NO DETECTORS. The circuit or detector error model didn't define any detectors.";
    }
    if dem.mechanisms.is_empty() {
        msg += "\n    WARNING: NO ERRORS. The circuit or detector error model didn't include any errors, making it vacuously impossible to find a logical error.";
    } else if graphlike_edges == Some(false) {
        msg += "\n    WARNING: NO GRAPHLIKE ERRORS. Although the circuit or detector error model does define some errors, none of them are graphlike (i.e. have at most two detection events), making it vacuously impossible to find a graphlike logical error.";
    }
    msg
}

/// A state of the graph-like search: (the moving detection event, the held one, the observables
/// flipped so far), `u32::MAX` for an event gone (or at the boundary).
type State = (u32, u32, u64);

/// Each visited state (by `key`) and the state it was reached from.
type Back = HashMap<State, State, BuildHasherDefault<Mix>>;

/// A state as visited: an unordered pair of events, both events gone being one state.
fn key((a, b, m): State) -> State {
    if a < b {
        (a, b, m)
    } else if a > b {
        (b, a, m)
    } else {
        (u32::MAX, u32::MAX, m)
    }
}

/// The faults along the search's path to `last`, each the difference between a state and the one
/// before it, sorted by their targets as Stim sorts its model's instructions.
fn backtrack(back: &Back, last: State) -> Vec<Fault> {
    let mut out: Vec<Fault> = Vec::new();
    let mut cur = last;
    loop {
        let prev = back[&key(cur)];
        let mut nodes = [cur.0, cur.1, prev.0, prev.1, u32::MAX];
        nodes.sort_unstable();
        let mut dets = Vec::new();
        let mut k = 0;
        while k < 4 {
            if nodes[k] == nodes[k + 1] {
                k += 2;
            } else {
                dets.push(nodes[k]);
                k += 1;
            }
        }
        out.push((dets, cur.2 ^ prev.2));
        if prev.0 == prev.1 {
            break;
        }
        cur = prev;
    }
    out.sort_by_key(targets);
    out
}

/// A fast hash for the search's states (the default hasher's resistance to chosen keys is not
/// needed for these, and costs most of the search's time).
#[derive(Default)]
struct Mix(u64);

impl Hasher for Mix {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u32(&mut self, x: u32) {
        self.write_u64(u64::from(x));
    }

    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// Stim's `search_for_undetectable_logical_errors`, step for step as Stim takes it, so that it
/// is Stim's own answer, fault for fault (`stim/search/hyper/{graph,algo}.cc`):
///
/// - Limits of two and two are the graph-like search's (`shortest_graphlike` ignoring larger
///   faults), as in Stim.
/// - Each fault of nonzero probability with at most `max_degree` detectors is an edge at each of
///   its detectors, in the model's order (repeats left out: they reach no new set). One with no
///   detector and an observable flipped is a logical error by itself (the last such, as Stim).
/// - The search starts from each edge that flips an observable, taken at its lowest detector in
///   detector order, and steps by adding an edge at the set's lowest detector, never growing the
///   set past `max_symptoms` (nor at all with `no_increase`); each set (with its observables) is
///   visited once, and the empty set with no observable counts as visited from the start.
/// - The first empty set reached (necessarily with an observable flipped) ends the search; its
///   faults, sorted by their targets (detectors, then observables) with pairs of one fault
///   cancelled, are the answer: the fewest faults this search can reach.
///
/// Stim keeps every set it visits for as long as it runs. Here they are kept in columns of at
/// most `max_memory_bytes` (see `Store`, and [`default_memory_budget`]); a search that fills
/// them ends with Stim's message for one that finds nothing, then the budget and the sets
/// visited.
pub fn search_undetectable(dem: &Dem, max_symptoms: usize, max_degree: usize, no_increase: bool, max_memory_bytes: usize) -> Result<Vec<Fault>, String> {
    if max_degree == 2 && max_symptoms == 2 {
        return shortest_graphlike(dem, true);
    }
    let mut edges: Vec<Fault> = Vec::new();
    let mut seen = HashSet::new();
    let mut at: Vec<Vec<u32>> = vec![Vec::new(); dem.num_detectors];
    let mut distance_1 = 0u64;
    for m in dem.mechanisms.iter().filter(|m| m.p != 0.0) {
        if m.detectors.len() > max_degree {
            continue;
        }
        if m.detectors.is_empty() {
            if m.observables != 0 {
                distance_1 = m.observables;
            }
            continue;
        }
        if seen.insert((m.detectors.clone(), m.observables)) {
            let k = u32::try_from(edges.len()).map_err(|_| "too many faults for the search".to_string())?;
            for &d in &m.detectors {
                at[d as usize].push(k);
            }
            edges.push((m.detectors.clone(), m.observables));
        }
    }
    if distance_1 != 0 {
        return Ok(vec![(Vec::new(), distance_1)]);
    }
    let full = |why: String| no_logical_error(dem, &format!("Failed to find any logical errors. {why}"), None);
    let mut store = Store::new(max_memory_bytes);
    for (node, ks) in at.iter().enumerate() {
        for &k in ks {
            let (dets, obs) = &edges[k as usize];
            if *obs != 0 && dets[0] as usize == node && dets.len() <= max_symptoms {
                store.visit(dets, *obs, 0, k).map_err(full)?;
            }
        }
    }
    // The sets are visited in the order they were stored: breadth first.
    let (mut cur, mut next) = (Vec::new(), Vec::new());
    let mut s = 1;
    while s < store.len() {
        cur.clear();
        cur.extend_from_slice(store.dets(s));
        let obs = store.obs[s];
        for &k in &at[cur[0] as usize] {
            xor_into(&cur, &edges[k as usize].0, &mut next);
            if next.len() > max_symptoms || (no_increase && next.len() > cur.len()) {
                continue;
            }
            let Some(id) = store.visit(&next, obs ^ edges[k as usize].1, s as u32, k).map_err(full)? else { continue };
            if next.is_empty() {
                let mut out = Vec::new();
                let mut t = id as usize;
                while t != 0 {
                    out.push(edges[store.edge[t] as usize].clone());
                    t = store.parent[t] as usize;
                }
                out.sort_by_key(targets);
                let mut i = 0;
                while i + 1 < out.len() {
                    if out[i] == out[i + 1] {
                        out.drain(i..i + 2);
                    } else {
                        i += 1;
                    }
                }
                return Ok(out);
            }
        }
        s += 1;
    }
    Err(no_logical_error(dem, "Failed to find any logical errors.", None))
}

/// The search's visited sets, in columns: set `s` has the detectors `pool[start[s]..start[s + 1]]`
/// and the observables `obs[s]`, and was reached from set `parent[s]` by adding edge `edge[s]`
/// (`u32::MAX` for the empty set, set 0, and the starts' parent). `table` holds each set's id at
/// the slot its contents hash to (open addressing, probed linearly, at most three quarters full).
/// Its columns and table, as allocated, stay within `budget` bytes, even while one grows (or
/// within the few that hold the empty set, if more).
struct Store {
    pool: Vec<u32>,
    start: Vec<u64>,
    obs: Vec<u64>,
    parent: Vec<u32>,
    edge: Vec<u32>,
    table: Vec<u32>,
    budget: usize,
}

const NONE: u32 = u32::MAX;

impl Store {
    /// A store holding the empty set with no observable.
    fn new(budget: usize) -> Store {
        let mut store = Store { pool: Vec::new(), start: vec![0, 0], obs: vec![0], parent: vec![NONE], edge: vec![NONE], table: vec![NONE; 16], budget };
        let (slot, _) = store.find(&[], 0);
        store.table[slot] = 0;
        store
    }

    /// How many sets it holds.
    fn len(&self) -> usize {
        self.obs.len()
    }

    /// Set `s`'s detectors.
    fn dets(&self, s: usize) -> &[u32] {
        &self.pool[self.start[s] as usize..self.start[s + 1] as usize]
    }

    /// The bytes its columns and table hold.
    fn bytes(&self) -> usize {
        4 * (self.pool.capacity() + self.parent.capacity() + self.edge.capacity() + self.table.len()) + 8 * (self.start.capacity() + self.obs.capacity())
    }

    /// The table slot holding the set with these contents (true), or the empty one where it
    /// would go (false).
    fn find(&self, dets: &[u32], obs: u64) -> (usize, bool) {
        let mut h = Mix::default();
        for &d in dets {
            h.write_u32(d);
        }
        h.write_u64(obs);
        // The table's length is a power of two; Mix's high bits are its best mixed.
        let mask = self.table.len() - 1;
        let mut slot = (h.finish() >> (64 - self.table.len().trailing_zeros())) as usize;
        loop {
            match self.table[slot] {
                NONE => return (slot, false),
                s if self.obs[s as usize] == obs && self.dets(s as usize) == dets => return (slot, true),
                _ => slot = (slot + 1) & mask,
            }
        }
    }

    /// The set's id if it is new (it is then stored), None if it was visited before; Err (the
    /// words for Stim's message) if the budget has no room for it.
    fn visit(&mut self, dets: &[u32], obs: u64, parent: u32, edge: u32) -> Result<Option<u32>, String> {
        let (mut slot, seen) = self.find(dets, obs);
        if seen {
            return Ok(None);
        }
        let n = self.len();
        let Some(id) = u32::try_from(n).ok().filter(|&id| id != NONE) else {
            return Err(format!("(the search's 32-bit ids ran out after {n} states)"));
        };
        if 4 * (n + 1) > 3 * self.table.len() {
            if !self.grow_table() {
                return Err(format!("(memory budget of {} bytes reached after {n} states)", self.budget));
            }
            slot = self.find(dets, obs).0;
        }
        let (mut total, budget) = (self.bytes(), self.budget);
        let room = reserve(&mut self.pool, dets.len(), &mut total, budget)
            && reserve(&mut self.start, 1, &mut total, budget)
            && reserve(&mut self.obs, 1, &mut total, budget)
            && reserve(&mut self.parent, 1, &mut total, budget)
            && reserve(&mut self.edge, 1, &mut total, budget);
        if !room {
            return Err(format!("(memory budget of {} bytes reached after {n} states)", self.budget));
        }
        self.pool.extend_from_slice(dets);
        self.start.push(self.pool.len() as u64);
        self.obs.push(obs);
        self.parent.push(parent);
        self.edge.push(edge);
        self.table[slot] = id;
        Ok(Some(id))
    }

    /// The table at twice its length, every set placed again, if the budget has room for it
    /// (the old table is let go first).
    fn grow_table(&mut self) -> bool {
        let len = 2 * self.table.len();
        if self.bytes() - 4 * self.table.len() + 4 * len > self.budget {
            return false;
        }
        self.table = Vec::new();
        self.table = vec![NONE; len];
        for s in 0..self.len() {
            let (slot, _) = self.find(self.dets(s), self.obs[s]);
            self.table[slot] = s as u32;
        }
        true
    }
}

/// Room in `v` for `extra` more, `total` being the store's bytes (updated): if it is full, its
/// capacity doubled, or raised as far as the budget allows; false if that is too little. A
/// column that grows is copied, so its old allocation counts until the copy is made: the store
/// holds at most `budget` bytes even while it grows.
fn reserve<T>(v: &mut Vec<T>, extra: usize, total: &mut usize, budget: usize) -> bool {
    let need = v.len() + extra;
    if need <= v.capacity() {
        return true;
    }
    let size = std::mem::size_of::<T>();
    let cap = (2 * v.capacity()).max(need).min(budget.saturating_sub(*total) / size);
    if cap < need {
        return false;
    }
    let old = size * v.capacity();
    v.reserve_exact(cap - v.len());
    *total = *total - old + size * v.capacity();
    true
}

/// The budget of `search_undetectable`'s store when none is given: half the machine's physical
/// memory, or no limit (as in Stim) where the platform does not say how much it has.
pub fn default_memory_budget() -> usize {
    physical_memory().map_or(usize::MAX, |bytes| usize::try_from(bytes / 2).unwrap_or(usize::MAX))
}

/// The machine's physical memory in bytes: `hw.memsize` on macOS and iOS.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn physical_memory() -> Option<u64> {
    extern "C" {
        fn sysctlbyname(name: *const std::ffi::c_char, oldp: *mut std::ffi::c_void, oldlenp: *mut usize, newp: *mut std::ffi::c_void, newlen: usize) -> std::ffi::c_int;
    }
    let mut bytes = 0u64;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: hw.memsize is a 64-bit integer, written into `bytes`, whose size `len` gives.
    let ok = unsafe { sysctlbyname(c"hw.memsize".as_ptr(), (&mut bytes as *mut u64).cast(), &mut len, std::ptr::null_mut(), 0) } == 0;
    (ok && len == std::mem::size_of::<u64>()).then_some(bytes)
}

/// The machine's physical memory in bytes: `MemTotal` in `/proc/meminfo` on Linux and Android.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn physical_memory() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib: u64 = text.lines().find_map(|line| line.strip_prefix("MemTotal:"))?.trim().strip_suffix("kB")?.trim().parse().ok()?;
    kib.checked_mul(1024)
}

/// The machine's physical memory in bytes: `GlobalMemoryStatusEx`'s `ullTotalPhys` on Windows.
#[cfg(windows)]
fn physical_memory() -> Option<u64> {
    // MEMORYSTATUSEX, of which only the total is read.
    #[repr(C)]
    #[allow(dead_code)]
    struct MemoryStatus {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatus) -> i32;
    }
    let mut status = MemoryStatus { length: std::mem::size_of::<MemoryStatus>() as u32, memory_load: 0, total_phys: 0, avail_phys: 0, total_page_file: 0, avail_page_file: 0, total_virtual: 0, avail_virtual: 0, avail_extended_virtual: 0 };
    // SAFETY: `status` is a MEMORYSTATUSEX with its length set, as the call requires.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;
    ok.then_some(status.total_phys)
}

/// Elsewhere (WebAssembly among them) the physical memory is not known.
#[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", windows)))]
fn physical_memory() -> Option<u64> {
    None
}

/// `out` = the symmetric difference of two sorted sets.
fn xor_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

/// A fault's targets as Stim orders them: detectors, then observables after every detector.
fn targets((dets, obs): &Fault) -> Vec<u64> {
    let mut t: Vec<u64> = dets.iter().map(|&d| u64::from(d)).collect();
    t.extend((0..64).filter(|k| obs >> k & 1 == 1).map(|k| 1u64 << 63 | k));
    t
}

/// The faults as a model's text, each `error(1)`, as Stim writes the result.
pub fn to_dem_text(faults: &[Fault]) -> String {
    let mut s = String::new();
    for (dets, obs) in faults {
        let mut t: Vec<String> = dets.iter().map(|d| format!("D{d}")).collect();
        t.extend((0..64).filter(|k| obs >> k & 1 == 1).map(|k| format!("L{k}")));
        s += &format!("error(1) {}\n", t.join(" "));
    }
    s
}

/// The faults as terms for `explain`: detectors, then `OBS | k`.
pub fn as_terms(faults: &[Fault]) -> Vec<Vec<u64>> {
    faults
        .iter()
        .map(|(dets, obs)| {
            let mut t: Vec<u64> = dets.iter().map(|&d| u64::from(d)).collect();
            t.extend((0..64).filter(|k| obs >> k & 1 == 1).map(|k| crate::dem_program::OBS | k));
            t
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repetition(d: usize) -> Dem {
        // A line of d data faults: D0 ... D(d−2) between them, the last flipping L0.
        let mut text = String::from("error(0.1) D0\n");
        for k in 0..d - 2 {
            text += &format!("error(0.1) D{k} D{}\n", k + 1);
        }
        text += &format!("error(0.1) D{} L0\n", d - 2);
        Dem::parse(&text).unwrap()
    }

    #[test]
    fn a_repetition_code_has_its_distance() {
        for d in [2, 3, 5, 9] {
            let found = shortest_graphlike(&repetition(d), true).unwrap();
            assert_eq!(found.len(), d);
        }
    }

    #[test]
    fn parallel_edges_make_a_short_cycle() {
        let dem = Dem::parse("error(0.1) D0 D1\nerror(0.1) D0 D1 L0\nerror(0.1) D0\nerror(0.1) D1 D2\nerror(0.1) D2 L0").unwrap();
        assert_eq!(shortest_graphlike(&dem, true).unwrap().len(), 2);
    }

    #[test]
    fn no_logical_and_hyperedges() {
        assert!(shortest_graphlike(&Dem::parse("error(0.1) D0 D1\nerror(0.1) D1").unwrap(), true).is_err());
        assert!(shortest_graphlike(&Dem::parse("error(0.1) D0 D1 D2 L0").unwrap(), false).is_err());
        assert_eq!(shortest_graphlike(&Dem::parse("error(0.1) L0").unwrap(), true).unwrap().len(), 1);
        // Decomposed faults count only when hyperedges are not ignored (as in Stim).
        let decomposed = Dem::parse("error(0.1) D5 D6 ^ D0\nerror(0.1) D7 D8 ^ D0 L0").unwrap();
        assert!(shortest_graphlike(&decomposed, true).is_err());
        assert_eq!(shortest_graphlike(&decomposed, false).unwrap().len(), 2);
    }

    #[test]
    fn the_store_keeps_within_its_budget() {
        // Sets of three detectors stored until the budget is full: the columns and table never
        // hold more than it, and every set is found again. A column (or the table) stops growing
        // when its new allocation does not fit beside the store, so the store fills more than
        // half the budget (but for one set's detectors).
        for budget in [1_000, 4_096, 100_000, 1 << 20, 3 << 20] {
            let mut store = Store::new(budget);
            let mut n = 0;
            let why = loop {
                match store.visit(&[n, n + 1, n + 2], u64::from(n % 3), 0, n) {
                    Ok(id) => assert_eq!(id, Some(n + 1)),
                    Err(why) => break why,
                }
                assert!(store.bytes() <= budget);
                n += 1;
            };
            assert_eq!(why, format!("(memory budget of {budget} bytes reached after {} states)", store.len()));
            assert!(2 * store.bytes() + 12 > budget, "{budget}: {} bytes", store.bytes());
            for k in 0..n {
                assert_eq!(store.visit(&[k, k + 1, k + 2], u64::from(k % 3), 0, 0), Ok(None));
                assert_eq!((store.dets(k as usize + 1), store.obs[k as usize + 1], store.edge[k as usize + 1]), (&[k, k + 1, k + 2][..], u64::from(k % 3), k));
            }
        }
    }

    #[test]
    fn a_full_store_ends_the_search_with_stims_message() {
        // A line of 30 faults: the search stores 31 sets to find it, and a budget that holds
        // fewer ends it. 400 bytes hold the empty set and seven more of one detector each (28
        // bytes a set, beside the 64-byte table and the columns' spare room): the observables'
        // column, at 64 bytes, cannot then grow, as its copy would not fit beside the store.
        let dem = repetition(30);
        assert_eq!(search_undetectable(&dem, 4, 4, false, 100_000).unwrap().len(), 30);
        assert_eq!(search_undetectable(&dem, 4, 4, false, usize::MAX).unwrap().len(), 30);
        assert_eq!(search_undetectable(&dem, 4, 4, false, 400).unwrap_err(), "Failed to find any logical errors. (memory budget of 400 bytes reached after 8 states)");
        // A model with nothing to search keeps Stim's message and warnings, whatever the budget.
        let no_observables = Dem::parse("error(0.1) D0 D1\nerror(0.1) D1").unwrap();
        assert_eq!(search_undetectable(&no_observables, 4, 4, false, 0).unwrap_err(), no_logical_error(&no_observables, "Failed to find any logical errors.", None));
    }

    #[test]
    fn the_default_budget_is_half_the_machines_memory() {
        let tool = if cfg!(target_os = "macos") {
            Some(("sysctl", vec!["-n", "hw.memsize"]))
        } else if cfg!(target_os = "linux") {
            Some(("getconf", vec!["_PHYS_PAGES"]))
        } else {
            None
        };
        let Some((program, args)) = tool else { return };
        let read = |program: &str, args: &[&str]| -> u64 { String::from_utf8(std::process::Command::new(program).args(args).output().unwrap().stdout).unwrap().trim().parse().unwrap() };
        let mut bytes = read(program, &args);
        if cfg!(target_os = "linux") {
            bytes *= read("getconf", &["PAGESIZE"]);
        }
        assert_eq!(physical_memory(), Some(bytes));
        assert_eq!(default_memory_budget() as u64, bytes / 2);
    }
}
