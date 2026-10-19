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
use std::rc::Rc;

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

/// The most search states `search_undetectable` visits before giving up.
const MAX_STATES: usize = 20_000_000;

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
pub fn search_undetectable(dem: &Dem, max_symptoms: usize, max_degree: usize, no_increase: bool) -> Result<Vec<Fault>, String> {
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
    // Each visited set: its detectors (shared with the index), observables, the set it was reached
    // from and the edge added (u32::MAX for the empty set and the starts' parent).
    const NONE: u32 = u32::MAX;
    let mut states: Vec<(Rc<[u32]>, u64, u32, u32)> = vec![(Rc::from(Vec::new()), 0, NONE, NONE)];
    let mut index: HashSet<SetKey, BuildHasherDefault<Mix>> = HashSet::default();
    index.insert(SetKey(states[0].0.clone(), 0));
    let mut queue = VecDeque::new();
    let mut visit = |states: &mut Vec<(Rc<[u32]>, u64, u32, u32)>, dets: &[u32], obs: u64, parent: u32, k: u32| -> Result<Option<u32>, String> {
        let key = SetKey(Rc::from(dets), obs);
        if index.contains(&key) {
            return Ok(None);
        }
        if states.len() >= MAX_STATES {
            return Err(format!("the search visited {MAX_STATES} sets of detectors without finding an undetectable logical error; lower its limits"));
        }
        let id = states.len() as u32;
        states.push((key.0.clone(), obs, parent, k));
        index.insert(key);
        Ok(Some(id))
    };
    for (node, ks) in at.iter().enumerate() {
        for &k in ks {
            let (dets, obs) = &edges[k as usize];
            if *obs != 0 && dets[0] as usize == node && dets.len() <= max_symptoms {
                if let Some(id) = visit(&mut states, dets, *obs, 0, k)? {
                    queue.push_back(id);
                }
            }
        }
    }
    let mut next = Vec::new();
    while let Some(s) = queue.pop_front() {
        let (cur, obs) = (states[s as usize].0.clone(), states[s as usize].1);
        for &k in &at[cur[0] as usize] {
            xor_into(&cur, &edges[k as usize].0, &mut next);
            if next.len() > max_symptoms || (no_increase && next.len() > cur.len()) {
                continue;
            }
            let Some(id) = visit(&mut states, &next, obs ^ edges[k as usize].1, s, k)? else { continue };
            if next.is_empty() {
                let mut out = Vec::new();
                let mut t = id;
                while t != 0 {
                    out.push(edges[states[t as usize].3 as usize].clone());
                    t = states[t as usize].2;
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
            queue.push_back(id);
        }
    }
    Err(no_logical_error(dem, "Failed to find any logical errors.", None))
}

/// A visited set of the search: its detectors and observables, hashed word by word.
#[derive(PartialEq, Eq)]
struct SetKey(Rc<[u32]>, u64);

impl std::hash::Hash for SetKey {
    fn hash<H: Hasher>(&self, h: &mut H) {
        for &d in self.0.iter() {
            h.write_u32(d);
        }
        h.write_u64(self.1);
    }
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
}
