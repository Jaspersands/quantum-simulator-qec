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

use std::collections::{HashMap, HashSet, VecDeque};

use crate::dem::Dem;

/// A fault as its detectors and observables.
pub type Fault = (Vec<u32>, u64);

/// The smallest set of the model's graph-like faults that flips an observable and no detector,
/// as Stim's `shortest_graphlike_error` finds it: each fault of at most two detectors an edge
/// between them (or to the boundary), and the shortest closed walk whose edges flip an
/// observable. With `ignore_ungraphlike`, faults that are not graph-like as written (more
/// detectors, or decomposed into `^`-separated pieces) are left out, as Stim leaves them;
/// without it, a decomposed fault's pieces are edges, and a piece of three or more detectors is
/// refused.
pub fn shortest_graphlike(dem: &Dem, ignore_ungraphlike: bool) -> Result<Vec<Fault>, String> {
    let boundary = dem.num_detectors as u32;
    let mut edges: Vec<(u32, u32, u64)> = Vec::new();
    let mut seen = HashSet::new();
    for m in &dem.mechanisms {
        // A fault flipping only observables is a logical error by itself.
        if m.detectors.is_empty() && m.observables != 0 && m.pieces.len() <= 1 {
            return Ok(vec![(Vec::new(), m.observables)]);
        }
        let pieces: Vec<Fault> = if m.pieces.len() <= 1 {
            vec![(m.detectors.clone(), m.observables)]
        } else if ignore_ungraphlike {
            continue;
        } else {
            m.pieces.iter().map(|p| (p.detectors.clone(), p.observables)).collect()
        };
        for (dets, obs) in pieces {
            let (a, b) = match dets.as_slice() {
                [] => continue,
                [a] => (*a, boundary),
                [a, b] => (*a.min(b), *a.max(b)),
                _ if ignore_ungraphlike => continue,
                _ => return Err(format!("a fault sets off {} detectors: not graph-like (decompose the model, or ignore such faults)", dets.len())),
            };
            if seen.insert((a, b, obs)) {
                edges.push((a, b, obs));
            }
        }
    }
    let n = dem.num_detectors + 1;
    let mut adj: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n];
    for (k, &(a, b, _)) in edges.iter().enumerate() {
        adj[a as usize].push((b, k));
        adj[b as usize].push((a, k));
    }
    // From every node: a breadth-first tree labelled by the observables its paths flip. An edge
    // between two reached nodes whose labels it does not reconcile closes a walk that flips an
    // observable and no detector; the shortest over all starts is a shortest such cycle.
    let mut best: Option<(usize, Vec<usize>)> = None;
    let mut dist = vec![u32::MAX; n];
    let mut label = vec![0u64; n];
    let mut parent = vec![usize::MAX; n];
    let mut touched: Vec<usize> = Vec::new();
    for s in 0..n {
        if adj[s].is_empty() {
            continue;
        }
        for &v in &touched {
            dist[v] = u32::MAX;
            parent[v] = usize::MAX;
        }
        touched.clear();
        dist[s] = 0;
        label[s] = 0;
        touched.push(s);
        let mut queue = VecDeque::from([s]);
        while let Some(u) = queue.pop_front() {
            let du = dist[u] as usize;
            if let Some((len, _)) = &best {
                // A walk closed from here is at least 2·du long.
                if 2 * du >= *len {
                    break;
                }
            }
            for &(w, e) in &adj[u] {
                let w = w as usize;
                let through = label[u] ^ edges[e].2;
                if dist[w] == u32::MAX {
                    dist[w] = du as u32 + 1;
                    label[w] = through;
                    parent[w] = e;
                    touched.push(w);
                    queue.push_back(w);
                } else if e != parent[u] && through != label[w] {
                    let len = du + dist[w] as usize + 1;
                    if best.as_ref().is_none_or(|(b, _)| len < *b) {
                        // The walk's edges: both tree paths and this edge, shared edges cancelled.
                        let mut count: HashMap<usize, u32> = HashMap::new();
                        *count.entry(e).or_default() += 1;
                        for mut v in [u, w] {
                            while v != s {
                                let pe = parent[v];
                                *count.entry(pe).or_default() += 1;
                                let (a, b, _) = edges[pe];
                                v = if a as usize == v { b as usize } else { a as usize };
                            }
                        }
                        let odd: Vec<usize> = count.into_iter().filter(|&(_, c)| c % 2 == 1).map(|(k, _)| k).collect();
                        best = Some((odd.len(), odd));
                    }
                }
            }
        }
    }
    let (_, chosen) = best.ok_or("there is no undetectable logical error among the model's graph-like faults")?;
    let mut out: Vec<Fault> = chosen
        .into_iter()
        .map(|k| {
            let (a, b, obs) = edges[k];
            let dets = if b == boundary { vec![a] } else { vec![a, b] };
            (dets, obs)
        })
        .collect();
    out.sort();
    Ok(out)
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
