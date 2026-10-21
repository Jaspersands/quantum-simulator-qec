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
/// most `max_memory_bytes` (see `Store`, and [`default_memory_budget`]): a set of k detectors
/// takes 24 + 4k bytes and a share of the table, which the room the columns keep for growing
/// spreads over 40 to 100 bytes of budget (50 to 80 for sets of two to four detectors). A
/// search that fills them (or that the allocator refuses) ends with Stim's message for one that
/// finds nothing, then the budget and the sets visited. A larger budget never stores fewer
/// sets, so never fails where a smaller one finds the faults; the answer is returned when it is
/// reached, not stored. Limits of two and two take the graph-like search, which keeps no such
/// sets: the budget does not bound it, as nothing bounds it in Stim.
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
            let (dets, flips) = &edges[k as usize];
            xor_into(&cur, dets, &mut next);
            if next.len() > max_symptoms || (no_increase && next.len() > cur.len()) {
                continue;
            }
            if next.is_empty() && obs ^ flips != 0 {
                // The first empty set with an observable (it cannot have been visited, as the
                // search ends here): the answer, edge k and the edges that reached set s.
                let mut out = vec![edges[k as usize].clone()];
                let mut t = s;
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
            store.visit(&next, obs ^ flips, s as u32, k).map_err(full)?;
        }
        s += 1;
    }
    Err(no_logical_error(dem, "Failed to find any logical errors.", None))
}

/// The search's visited sets, in columns: set `s` has the detectors `pool[start[s]..start[s + 1]]`
/// and the observables `obs[s]`, and was reached from set `parent[s]` by adding edge `edge[s]`
/// (both `u32::MAX` for set 0, the empty set; a start's parent is set 0). `table` holds each set's
/// id at the slot its contents hash to (open addressing, probed linearly, at most three quarters
/// full). Its columns and table, as allocated, stay within `budget` bytes, even while one grows
/// (or within the few that hold the empty set, if more).
///
/// A set that needs room has every column's capacity and the table's length decided before any
/// grows (`plan`): the least they need must fit, and then each column that must grow doubles or,
/// taken in turn with the rest at the least they need, grows as far as the budget allows. So no
/// column's growth takes the room another needs for the same set, and a larger budget does not
/// store fewer sets (the tests sweep budgets byte by byte for it).
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

/// The bytes of an entry in each column: `pool`, `start`, `obs`, `parent`, `edge`.
const SIZES: [usize; 5] = [4, 8, 8, 4, 4];

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

    /// The columns' capacities, in the order of `SIZES`.
    fn caps(&self) -> [usize; 5] {
        [self.pool.capacity(), self.start.capacity(), self.obs.capacity(), self.parent.capacity(), self.edge.capacity()]
    }

    /// The bytes its columns and table hold.
    fn bytes(&self) -> usize {
        4 * self.table.len() + self.caps().iter().zip(SIZES).map(|(cap, size)| cap * size).sum::<usize>()
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
    /// words for Stim's message) if the budget has no room for it, or the allocator refuses it.
    fn visit(&mut self, dets: &[u32], obs: u64, parent: u32, edge: u32) -> Result<Option<u32>, String> {
        let (n, budget) = (self.len(), self.budget);
        let full = move || format!("(memory budget of {budget} bytes reached after {n} states)");
        if self.table.is_empty() {
            // Its table was let go to grow, and the allocator refused the new one.
            return Err(full());
        }
        let (mut slot, seen) = self.find(dets, obs);
        if seen {
            return Ok(None);
        }
        let Some(id) = u32::try_from(n).ok().filter(|&id| id != NONE) else {
            return Err(format!("(the search's 32-bit ids ran out after {n} states)"));
        };
        let table = if 4 * (n + 1) > 3 * self.table.len() { 2 * self.table.len() } else { self.table.len() };
        let need = [self.pool.len() + dets.len(), n + 2, n + 1, n + 1, n + 1];
        if table > self.table.len() || need.iter().zip(self.caps()).any(|(&need, cap)| need > cap) {
            let rehash = table > self.table.len();
            let Some(caps) = self.plan(table, &need) else { return Err(full()) };
            if !self.grow(table, &caps) {
                return Err(full());
            }
            if rehash {
                slot = self.find(dets, obs).0;
            }
        }
        self.pool.extend_from_slice(dets);
        self.start.push(self.pool.len() as u64);
        self.obs.push(obs);
        self.parent.push(parent);
        self.edge.push(edge);
        self.table[slot] = id;
        Ok(Some(id))
    }

    /// The columns' capacities for `need` entries in each and a table of `table` slots, decided
    /// together before any grows: None if the least they need does not fit the budget. Each
    /// column that must grow doubles, or, in turn, with the columns after it at the least they
    /// need, grows as far as the budget allows.
    fn plan(&self, table: usize, need: &[usize; 5]) -> Option<[usize; 5]> {
        let caps = self.caps();
        let mut plan: [usize; 5] = std::array::from_fn(|i| caps[i].max(need[i]));
        if self.peak(table, &plan) > self.budget {
            return None;
        }
        for i in (0..5).filter(|&i| need[i] > caps[i]) {
            // The largest capacity that fits, up to twice the column's: `peak` grows with it.
            let (mut lo, mut hi) = (need[i], need[i].max(caps[i].saturating_mul(2)));
            while lo < hi {
                plan[i] = lo + (hi - lo).div_ceil(2);
                if self.peak(table, &plan) <= self.budget {
                    lo = plan[i];
                } else {
                    hi = plan[i] - 1;
                }
            }
            plan[i] = lo;
        }
        Some(plan)
    }

    /// The most bytes the store holds while `grow` takes it to a table of `table` slots and
    /// columns of `caps`: the table first, the old one let go before the new one is made; then
    /// each column in turn, copied, so that its old allocation counts until the copy is made.
    fn peak(&self, table: usize, caps: &[usize; 5]) -> usize {
        let mut total = (self.bytes() - 4 * self.table.len()).saturating_add(table.saturating_mul(4));
        let mut peak = total;
        for ((&old, &new), size) in self.caps().iter().zip(caps).zip(SIZES) {
            if new > old {
                peak = peak.max(total.saturating_add(new.saturating_mul(size)));
                total = total.saturating_add((new - old).saturating_mul(size));
            }
        }
        peak
    }

    /// The table at `table` slots, every set placed again, then the columns at `caps`, as
    /// `peak` reckons; false if the allocator refuses. The store then keeps what has grown,
    /// and without a table (refused after the old one was let go) it refuses every set.
    fn grow(&mut self, table: usize, caps: &[usize; 5]) -> bool {
        if table > self.table.len() {
            self.table = Vec::new();
            let mut slots = Vec::new();
            if !try_capacity(&mut slots, table) {
                return false;
            }
            slots.resize(table, NONE);
            self.table = slots;
            for s in 0..self.len() {
                let (slot, _) = self.find(self.dets(s), self.obs[s]);
                self.table[slot] = s as u32;
            }
        }
        try_capacity(&mut self.pool, caps[0]) && try_capacity(&mut self.start, caps[1]) && try_capacity(&mut self.obs, caps[2]) && try_capacity(&mut self.parent, caps[3]) && try_capacity(&mut self.edge, caps[4])
    }
}

/// Room in `v` for `cap` entries in all (exactly that many, if it grows); false, where
/// `reserve_exact` would abort the process, if the allocator refuses it.
fn try_capacity<T>(v: &mut Vec<T>, cap: usize) -> bool {
    if cap <= v.capacity() {
        return true;
    }
    #[cfg(test)]
    if cap.saturating_mul(std::mem::size_of::<T>()) > tests::ALLOCATION_LIMIT.with(std::cell::Cell::get) {
        return false;
    }
    v.try_reserve_exact(cap - v.len()).is_ok()
}

/// The budget of `search_undetectable`'s store when none is given: half the machine's physical
/// memory (on Linux, half the cgroup's memory limit if that is less), or no limit (as in Stim)
/// where the platform does not say how much it has.
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

/// The machine's physical memory in bytes on Linux and Android: `MemTotal` in `/proc/meminfo`,
/// or the memory limit of the cgroups the process runs in if that is less (a container's or a
/// batch job's, where `MemTotal` is the host's).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn physical_memory() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib: u64 = text.lines().find_map(|line| line.strip_prefix("MemTotal:"))?.trim().strip_suffix("kB")?.trim().parse().ok()?;
    let total = kib.checked_mul(1024)?;
    let cgroups = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    Some(cgroup_limit(std::path::Path::new("/sys/fs/cgroup"), &cgroups).map_or(total, |limit| limit.min(total)))
}

/// The least memory limit on the cgroups that hold the process, the cgroup file systems being
/// mounted at `root` (`/sys/fs/cgroup`) and `cgroups` being `/proc/self/cgroup`: cgroup v2's
/// `memory.max` and v1's `memory/memory.limit_in_bytes`, at the root (a container's, the root of
/// its cgroup namespace) and in each cgroup on the process's path below it. A file that is
/// missing, or not a number (v2's `max`), sets no limit; neither does a path out of the root.
#[cfg_attr(not(any(target_os = "linux", target_os = "android")), allow(dead_code))]
fn cgroup_limit(root: &std::path::Path, cgroups: &str) -> Option<u64> {
    let read = |file: std::path::PathBuf| std::fs::read_to_string(file).ok()?.trim().parse::<u64>().ok();
    // A line is `id:controllers:path`: v2's has id 0 and no controllers; v1's memory one lists memory.
    let path = |v2: bool| {
        cgroups.lines().find_map(|line| {
            let mut fields = line.splitn(3, ':');
            let (id, controllers, path) = (fields.next()?, fields.next()?, fields.next()?);
            let found = if v2 { id == "0" && controllers.is_empty() } else { controllers.split(',').any(|c| c == "memory") };
            found.then_some(path)
        })
    };
    let mut limits = Vec::new();
    for (mut here, file, path) in [(root.to_path_buf(), "memory.max", path(true)), (root.join("memory"), "memory.limit_in_bytes", path(false))] {
        limits.extend(read(here.join(file)));
        let parts: Vec<&str> = path.unwrap_or("").split('/').filter(|part| !part.is_empty() && *part != ".").collect();
        if !parts.contains(&"..") {
            for part in parts {
                here.push(part);
                limits.extend(read(here.join(file)));
            }
        }
    }
    limits.into_iter().min()
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

    thread_local! {
        /// The most bytes `try_capacity` lets a column or table take in this thread's test,
        /// refusing more as an allocator out of memory would.
        pub(super) static ALLOCATION_LIMIT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
    }

    fn repetition(d: usize) -> Dem {
        // A line of d data faults: D0 ... D(d−2) between them, the last flipping L0.
        let mut text = String::from("error(0.1) D0\n");
        for k in 0..d - 2 {
            text += &format!("error(0.1) D{k} D{}\n", k + 1);
        }
        text += &format!("error(0.1) D{} L0\n", d - 2);
        Dem::parse(&text).unwrap()
    }

    fn hypergraph() -> Dem {
        // Twelve detectors in a line, a fault on each one, pair and triple of neighbours, every
        // third flipping L0: the search at (6, 3) stores fifty sets before it finds three
        // faults.
        let mut text = String::new();
        for k in 0..12 {
            for width in 1..=3 {
                if k + width <= 12 {
                    let dets: Vec<String> = (k..k + width).map(|d| format!("D{d}")).collect();
                    text += &format!("error(0.1) {}{}\n", dets.join(" "), if (k + width) % 3 == 0 { " L0" } else { "" });
                }
            }
        }
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
        // A line of 30 faults: the search stores 30 sets to find it (the empty set and one of
        // each detector; the answer is not stored), and a budget that holds fewer ends it. 400
        // bytes hold the empty set and seven more of one detector each (28 bytes a set, beside
        // the 64-byte table and the columns' spare room): the observables' column, at 64 bytes,
        // cannot then grow, as its copy would not fit beside the store.
        let dem = repetition(30);
        assert_eq!(search_undetectable(&dem, 4, 4, false, 100_000).unwrap().len(), 30);
        assert_eq!(search_undetectable(&dem, 4, 4, false, usize::MAX).unwrap().len(), 30);
        assert_eq!(search_undetectable(&dem, 4, 4, false, 400).unwrap_err(), "Failed to find any logical errors. (memory budget of 400 bytes reached after 8 states)");
        // A model with nothing to search keeps Stim's message and warnings, whatever the budget.
        let no_observables = Dem::parse("error(0.1) D0 D1\nerror(0.1) D1").unwrap();
        assert_eq!(search_undetectable(&no_observables, 4, 4, false, 0).unwrap_err(), no_logical_error(&no_observables, "Failed to find any logical errors.", None));
    }

    #[test]
    fn the_answer_takes_no_room() {
        // The search returns the empty set it reaches with an observable flipped, without storing
        // it: a budget that holds the sets before it finds the faults. Here those are the empty
        // set and {D0} with L0; storing the answer too, 164 bytes failed "after 2 states".
        let dem = Dem::parse("error(0.1) D0 L0\nerror(0.1) D0").unwrap();
        let least = (0..).find(|&budget| Store::new(budget).visit(&[0], 1, 0, 0).is_ok()).unwrap();
        assert_eq!(search_undetectable(&dem, 4, 4, false, least - 1).unwrap_err(), format!("Failed to find any logical errors. (memory budget of {} bytes reached after 1 states)", least - 1));
        for budget in least..least + 1_000 {
            assert_eq!(search_undetectable(&dem, 4, 4, false, budget), Ok(vec![(vec![0], 0), (vec![0], 1)]), "{budget}");
        }
    }

    #[test]
    fn a_larger_budget_never_stores_fewer_sets() {
        // Sets stored until the budget is full, at every budget to 8,000 bytes and in windows of
        // consecutive budgets above: a larger budget stores at least as many. Grown one after
        // another, a column's doubling could take the room the next needed for the same set, so
        // that 132 bytes stored a set fewer than 131. The sets' sizes are fixed (one to six), or
        // drawn from one to eight, or mostly one and now and then forty.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut draw = move |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n) as u32
        };
        let mut sequences: Vec<Vec<u32>> = (1..=6).map(|k| vec![k; 3_000]).collect();
        sequences.push((0..3_000).map(|_| 1 + draw(8)).collect());
        sequences.push((0..3_000).map(|_| if draw(30) == 0 { 40 } else { 1 }).collect());
        let budgets: Vec<usize> = (0..8_000).chain([20_000, 50_000, 100_000].into_iter().flat_map(|b| b..b + 200)).collect();
        let mut dets = Vec::new();
        for sizes in &sequences {
            let mut most = 0;
            for &budget in &budgets {
                let mut store = Store::new(budget);
                let mut first = 0;
                for (n, &k) in sizes.iter().enumerate() {
                    dets.clear();
                    dets.extend(first..first + k);
                    if store.visit(&dets, n as u64, 0, 0).is_err() {
                        break;
                    }
                    first += k;
                }
                assert!(store.len() >= most, "sizes {:?}...: {budget} bytes store {} sets, fewer than a smaller budget's {most}", &sizes[..4], store.len());
                most = store.len();
            }
            assert!(most < sizes.len(), "the largest budget runs out too");
        }
    }

    #[test]
    fn a_larger_budget_never_finds_less() {
        // Searches at every budget to well past the least that finds the faults: from there on
        // each finds them (the same faults), and below it each fails having stored no fewer sets
        // than at a smaller budget.
        let cases = [(Dem::parse("error(0.1) D0 L0\nerror(0.1) D0").unwrap(), 4, 4), (repetition(30), 4, 4), (hypergraph(), 6, 3)];
        for (dem, max_symptoms, max_degree) in &cases {
            let want = search_undetectable(dem, *max_symptoms, *max_degree, false, usize::MAX).unwrap();
            let (mut least, mut states) = (None, 0);
            for budget in 0..8_000 {
                match search_undetectable(dem, *max_symptoms, *max_degree, false, budget) {
                    Ok(faults) => {
                        assert_eq!(faults, want);
                        least.get_or_insert(budget);
                    }
                    Err(why) => {
                        assert_eq!(least, None, "{budget} bytes: {why}");
                        let stored: usize = why.rsplit("after ").next().unwrap().strip_suffix(" states)").unwrap().parse().unwrap();
                        assert!(stored >= states, "{budget} bytes: {why}, but a smaller budget stored {states}");
                        states = stored;
                    }
                }
            }
            assert!(least.is_some_and(|least| least < 4_000), "{least:?}");
        }
    }

    #[test]
    fn a_refused_allocation_ends_the_search_with_the_budget_message() {
        // An allocator out of memory refuses a column's growth (here past 127 bytes: the starts'
        // column, from 8 offsets to 16, for the seventh set after the empty one): the search
        // fails with the budget message, where `reserve_exact` would abort the process.
        let limit = |bytes: usize| ALLOCATION_LIMIT.with(|limit| limit.set(bytes));
        limit(127);
        let why = search_undetectable(&repetition(30), 4, 4, false, usize::MAX).unwrap_err();
        assert_eq!(why, format!("Failed to find any logical errors. (memory budget of {} bytes reached after 7 states)", usize::MAX));
        // The store keeps the sets it holds, and once the allocator has room, takes more.
        let mut store = Store::new(usize::MAX);
        let why = (0..).find_map(|n| store.visit(&[n], 0, 0, n).err()).unwrap();
        assert_eq!(why, format!("(memory budget of {} bytes reached after 7 states)", usize::MAX));
        limit(usize::MAX);
        for n in 0..6 {
            assert_eq!(store.visit(&[n], 0, 0, n), Ok(None));
        }
        assert_eq!(store.visit(&[6], 0, 0, 6), Ok(Some(7)));
        // The table refused after the old one was let go: the store refuses every set after.
        limit(4 * 16);
        assert!(!store.grow(32, &store.caps()));
        assert!(store.table.is_empty());
        limit(usize::MAX);
        for n in [0, 7] {
            assert_eq!(store.visit(&[n], 0, 0, n), Err(format!("(memory budget of {} bytes reached after 8 states)", usize::MAX)));
        }
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
            // Or the cgroups' limit, if less (`cgroup_limit` is checked on a made-up tree below).
            let cgroups = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
            bytes = cgroup_limit(std::path::Path::new("/sys/fs/cgroup"), &cgroups).map_or(bytes, |limit| limit.min(bytes));
        }
        assert_eq!(physical_memory(), Some(bytes));
        assert_eq!(default_memory_budget() as u64, bytes / 2);
    }

    #[test]
    fn a_cgroups_memory_limit_is_read_at_the_root_and_on_the_path() {
        // A made-up /sys/fs/cgroup: v2's memory.max and v1's memory.limit_in_bytes at the root
        // and on the process's path; the least number is the limit. "max", a missing or garbled
        // file, and a path out of the root set none.
        let root = std::env::temp_dir().join(format!("stabilizer_qec_cgroups_{}", std::process::id()));
        let write = |path: &str, text: &str| {
            let file = root.join(path);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, text).unwrap();
        };
        assert_eq!(cgroup_limit(&root, "0::/job/step\n"), None);
        write("memory.max", "max\n");
        write("job/memory.max", "max\n");
        write("job/step/memory.max", "2147483648\n");
        write("other/memory.max", "1000\n");
        write("memory/memory.limit_in_bytes", "9223372036854771712\n");
        write("memory/slurm/memory.limit_in_bytes", "4294967296\n");
        write("memory/slurm/task/memory.limit_in_bytes", "lots\n");
        // No path: the roots' files alone (v2's says max; v1's is its "no limit", a number).
        assert_eq!(cgroup_limit(&root, ""), Some(9223372036854771712));
        // A container's own limit, at its namespace's root.
        write("memory.max", "3221225472\n");
        assert_eq!(cgroup_limit(&root, "0::/\n"), Some(3221225472));
        write("memory.max", "max\n");
        // v2: the job's step sets 2 GiB below an unlimited job.
        assert_eq!(cgroup_limit(&root, "0::/job/step\n"), Some(2147483648));
        // v1 beside v2: the memory controller's line (with others in it), 4 GiB.
        assert_eq!(cgroup_limit(&root, "7:cpu,memory:/slurm/task\n0::/job\n"), Some(4294967296));
        assert_eq!(cgroup_limit(&root, "7:cpu,memory:/slurm/task\n0::/job/step\n"), Some(2147483648));
        // A path out of the namespace's root (another cgroup's) reads the root alone.
        assert_eq!(cgroup_limit(&root, "0::/../other\n"), Some(9223372036854771712));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
