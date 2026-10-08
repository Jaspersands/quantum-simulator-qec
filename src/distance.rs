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
    Err("there is no undetectable logical error among the model's graph-like faults".into())
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
    let targets = |(dets, obs): &Fault| -> Vec<u64> {
        let mut t: Vec<u64> = dets.iter().map(|&d| u64::from(d)).collect();
        t.extend((0..64).filter(|k| obs >> k & 1 == 1).map(|k| 1u64 << 63 | k));
        t
    };
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

/// Stim's `search_for_undetectable_logical_errors`: a breadth-first search over sets of fired
/// detectors (and flipped observables). Each fault of at most `max_degree` detectors is an edge
/// (the whole fault, hyperedges included); a search starts from each, and a step adds a fault
/// touching the set's lowest detector, never growing the set past `max_symptoms` (nor at all
/// with `no_increase`). The first set with no detector and an observable flipped is the
/// answer: the fewest faults this search can reach.
pub fn search_undetectable(dem: &Dem, max_symptoms: usize, max_degree: usize, no_increase: bool) -> Result<Vec<Fault>, String> {
    let mut edges: Vec<Fault> = Vec::new();
    let mut seen = HashSet::new();
    for m in &dem.mechanisms {
        if m.detectors.is_empty() {
            if m.observables != 0 {
                return Ok(vec![(Vec::new(), m.observables)]);
            }
            continue;
        }
        if m.detectors.len() <= max_degree && seen.insert((m.detectors.clone(), m.observables)) {
            edges.push((m.detectors.clone(), m.observables));
        }
    }
    let mut touching: HashMap<u32, Vec<usize>> = HashMap::new();
    for (k, (dets, _)) in edges.iter().enumerate() {
        for &d in dets {
            touching.entry(d).or_default().push(k);
        }
    }
    let xor = |a: &[u32], b: &[u32]| -> Vec<u32> {
        let (mut i, mut j, mut out) = (0, 0, Vec::with_capacity(a.len() + b.len()));
        while i < a.len() || j < b.len() {
            match (a.get(i), b.get(j)) {
                (Some(x), Some(y)) if x == y => {
                    i += 1;
                    j += 1;
                }
                (Some(x), Some(y)) if x < y => {
                    out.push(*x);
                    i += 1;
                }
                (Some(_), Some(y)) => {
                    out.push(*y);
                    j += 1;
                }
                (Some(x), None) => {
                    out.push(*x);
                    i += 1;
                }
                (None, Some(y)) => {
                    out.push(*y);
                    j += 1;
                }
                (None, None) => unreachable!(),
            }
        }
        out
    };
    // States as (detectors, observables), each with the state it came from and the fault added.
    let mut states: Vec<(Vec<u32>, u64, usize, usize)> = Vec::new();
    let mut index: HashMap<(Vec<u32>, u64), usize> = HashMap::new();
    let mut queue = VecDeque::new();
    for (k, (dets, obs)) in edges.iter().enumerate() {
        if dets.len() > max_symptoms {
            continue;
        }
        if let std::collections::hash_map::Entry::Vacant(slot) = index.entry((dets.clone(), *obs)) {
            slot.insert(states.len());
            states.push((dets.clone(), *obs, usize::MAX, k));
            queue.push_back(states.len() - 1);
        }
    }
    let path = |states: &Vec<(Vec<u32>, u64, usize, usize)>, mut s: usize| -> Vec<Fault> {
        let mut out = Vec::new();
        while s != usize::MAX {
            out.push(edges[states[s].3].clone());
            s = states[s].2;
        }
        out.sort();
        out
    };
    while let Some(s) = queue.pop_front() {
        let (dets, obs) = (states[s].0.clone(), states[s].1);
        if dets.is_empty() {
            if obs != 0 {
                return Ok(path(&states, s));
            }
            continue;
        }
        for &k in touching.get(&dets[0]).map(Vec::as_slice).unwrap_or(&[]) {
            let next = xor(&dets, &edges[k].0);
            if next.len() > max_symptoms || (no_increase && next.len() > dets.len()) {
                continue;
            }
            let key = (next, obs ^ edges[k].1);
            if index.contains_key(&key) {
                continue;
            }
            if states.len() >= MAX_STATES {
                return Err(format!("the search visited {MAX_STATES} sets of detectors without finding an undetectable logical error; lower its limits"));
            }
            index.insert(key.clone(), states.len());
            states.push((key.0, key.1, s, k));
            queue.push_back(states.len() - 1);
        }
    }
    Err("the search found no undetectable logical error within its limits".into())
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
