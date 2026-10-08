//! The tables that lift a Möbius matching back to the code: each detector's representative
//! basic fault (`choose_rgb_reps`), the charge graph (how pairs of faults move charge), and the
//! drag graph (what dragging charge between neighbouring detectors flips). Chromobius's
//! `graph/choose_rgb_reps.cc`, `charge_graph.cc` and `drag_graph.cc`.

use std::collections::{BTreeMap, BTreeSet};

use super::{next_charge, weight, Charge, ColorBasis, Key, BOUNDARY, NEUTRAL};

/// A basic fault by colour: its red, green and blue detectors (`BOUNDARY` where it has none),
/// the observables it flips and its net charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RgbEdge {
    pub nodes: [u32; 3],
    pub obs: u64,
    pub charge: Charge,
}

impl RgbEdge {
    const EMPTY: RgbEdge = RgbEdge { nodes: [BOUNDARY; 3], obs: 0, charge: NEUTRAL };

    pub fn weight(&self) -> usize {
        self.nodes.iter().filter(|&&n| n != BOUNDARY).count()
    }

    pub fn node(&self, c: Charge) -> u32 {
        if c == NEUTRAL {
            BOUNDARY
        } else {
            self.nodes[c as usize - 1]
        }
    }
}

/// `choose_rgb_reps_from_atomic_errors`: for each detector, the heaviest basic fault of
/// distinct colours through it; a detector with none borrows a same-coloured neighbour's,
/// moved onto itself along the fault joining them.
pub(crate) fn choose_rgb_reps(atomic: &BTreeMap<Key, u64>, colors: &[ColorBasis]) -> Vec<RgbEdge> {
    let mut result = vec![RgbEdge::EMPTY; colors.len()];
    for (err, &obs) in atomic {
        let mut rep = RgbEdge { obs, ..RgbEdge::EMPTY };
        let mut w = 0;
        for &n in err.iter().filter(|&&n| n != BOUNDARY) {
            let c = colors[n as usize].color;
            rep.nodes[c as usize - 1] = n;
            rep.charge ^= c;
            w += 1;
        }
        if rep.weight() != w {
            continue;
        }
        for &n in err.iter().filter(|&&n| n != BOUNDARY) {
            if w > result[n as usize].weight() {
                result[n as usize] = rep;
            }
        }
    }
    for (e, &obs) in atomic {
        if weight(e) != 2 {
            continue;
        }
        let (a, b) = (e[0] as usize, e[1] as usize);
        let (c1, c2) = (colors[a].color, colors[b].color);
        if c1 != c2 {
            continue;
        }
        let (w1, w2) = (result[a].weight(), result[b].weight());
        if w1 == 0 && w2 > 0 {
            result[a] = result[b];
            result[a].nodes[c1 as usize - 1] = e[0];
            result[a].obs ^= obs;
        }
        if w2 == 0 && w1 > 0 {
            result[b] = result[a];
            result[b].nodes[c2 as usize - 1] = e[1];
            result[b].obs ^= obs;
        }
    }
    result
}

/// Which observables moving charge from one detector to a neighbour flips, by neighbour
/// (`BOUNDARY` for the boundary); each detector its own neighbour, flipping none.
pub(crate) struct ChargeGraph {
    pub nodes: Vec<BTreeMap<u32, u64>>,
}

impl ChargeGraph {
    fn add_edge(&mut self, a: u32, b: u32, obs: u64) {
        if a != BOUNDARY {
            self.nodes[a as usize].insert(b, obs);
        }
        if b != BOUNDARY {
            self.nodes[b as usize].insert(a, obs);
        }
    }

    /// `ChargeGraph::from_atomic_errors`: the pairs and singles among the basic faults, then
    /// the pair or single left by any two faults through a detector, one a triplet.
    pub fn from_atomic(atomic: &BTreeMap<Key, u64>, num_nodes: usize) -> ChargeGraph {
        let mut g = ChargeGraph { nodes: (0..num_nodes).map(|k| BTreeMap::from([(k as u32, 0u64)])).collect() };
        for (err, &obs) in atomic {
            if err[2] == BOUNDARY {
                g.add_edge(err[0], err[1], obs);
            }
        }
        let mut by_node: BTreeMap<u32, Vec<Key>> = BTreeMap::new();
        for err in atomic.keys() {
            for &n in err.iter().filter(|&&n| n != BOUNDARY) {
                by_node.entry(n).or_default().push(*err);
            }
        }
        for faults in by_node.values() {
            for (k1, e1) in faults.iter().enumerate() {
                for e2 in &faults[k1 + 1..] {
                    if weight(e1) < 3 && weight(e2) < 3 {
                        continue;
                    }
                    let mut buf: Vec<u32> = e1.iter().chain(e2.iter()).copied().collect();
                    buf.sort_unstable();
                    let mut kept: Vec<u32> = Vec::new();
                    for x in buf {
                        if kept.last() == Some(&x) {
                            kept.pop();
                        } else {
                            kept.push(x);
                        }
                    }
                    let (a, b) = match kept.as_slice() {
                        [a] => (*a, BOUNDARY),
                        [a, b] | [a, b, BOUNDARY] => (*a, *b),
                        _ => continue,
                    };
                    g.add_edge(a, b, atomic[e1] ^ atomic[e2]);
                }
            }
        }
        g
    }

    /// `BfsSearcher::find_shortest_path_obs_flip` with its depth limit of 2: the observables a
    /// path of one or two steps from `src` to `dst` flips, and whether paths of that length
    /// disagree (Chromobius returns whichever its hash order meets first).
    fn shortest_path_flip(&self, src: u32, dst: u32) -> Option<(u64, bool)> {
        if src == dst {
            return Some((0, false));
        }
        let near = &self.nodes[src as usize];
        if let Some(&f) = near.get(&dst) {
            return Some((f, false));
        }
        let mut found: Option<u64> = None;
        let mut ambiguous = false;
        for (&m, &f1) in near {
            if m == BOUNDARY {
                continue;
            }
            if let Some(&f2) = self.nodes[m as usize].get(&dst) {
                let f = f1 ^ f2;
                ambiguous |= found.is_some_and(|g| g != f);
                found.get_or_insert(f);
            }
        }
        found.map(|f| (f, ambiguous))
    }
}

/// (from, to, charge carried from, charge arriving): the observables dragging charge flips,
/// and whether Chromobius's search could have found another value.
pub(crate) struct DragGraph {
    pub map: BTreeMap<(u32, u32, Charge, Charge), (u64, bool)>,
}

impl DragGraph {
    fn add(&mut self, n1: u32, n2: u32, c1: Charge, c2: Charge, flip: u64, ambiguous: bool) {
        self.map.insert((n1, n2, c1, c2), (flip, ambiguous));
        self.map.insert((n2, n1, c2, c1), (flip, ambiguous));
    }

    /// `DragGraph::from_charge_graph_paths_for_sub_edges_of_atomic_errors`.
    pub fn new(charge: &ChargeGraph, atomic: &BTreeMap<Key, u64>, reps: &[RgbEdge], colors: &[ColorBasis]) -> DragGraph {
        let mut drag = DragGraph { map: BTreeMap::new() };
        let mut decomposed: BTreeSet<(u32, u32)> = BTreeSet::new();
        let pair = |a: u32, b: u32| (a.min(b), a.max(b));
        let dump = |drag: &mut DragGraph, a: u32, b: u32, ab_obs: u64| {
            if reps[a as usize].weight() != 3 {
                return;
            }
            let (ca, cb) = (colors[a as usize].color, colors[b as usize].color);
            let c = ca ^ cb;
            if c == NEUTRAL {
                return;
            }
            let r1 = charge.shortest_path_flip(reps[a as usize].node(ca), a);
            let r2 = charge.shortest_path_flip(reps[a as usize].node(cb), b);
            if let (Some((f1, a1)), Some((f2, a2))) = (r1, r2) {
                drag.add(a, b, c, NEUTRAL, f1 ^ f2 ^ reps[a as usize].obs ^ ab_obs, a1 || a2);
            }
        };
        for (err, &obs) in atomic {
            match weight(err) {
                3 => {
                    let [a, b, c] = *err;
                    decomposed.insert(pair(a, b));
                    decomposed.insert(pair(a, c));
                    decomposed.insert(pair(b, c));
                }
                2 => {
                    let (a, b) = (err[0], err[1]);
                    let (ca, cb) = (colors[a as usize].color, colors[b as usize].color);
                    let p = charge.nodes[a as usize][&b];
                    drag.add(a, b, ca, cb, p, false);
                    drag.add(a, b, NEUTRAL, NEUTRAL, 0, false);
                    dump(&mut drag, a, b, obs);
                    dump(&mut drag, b, a, obs);
                    decomposed.insert(pair(a, b));
                }
                1 => {
                    let n = err[0];
                    let c = colors[n as usize].color;
                    drag.add(n, n, c, NEUTRAL, obs, false);
                    drag.add(n, n, NEUTRAL, NEUTRAL, 0, false);
                    let r = reps[n as usize];
                    if r.weight() == 3 {
                        let c1 = next_charge(c);
                        let c2 = next_charge(c1);
                        drag.add(n, n, c1, c2, r.obs ^ obs, false);
                    }
                }
                _ => {}
            }
        }
        for &(n1, n2) in &decomposed {
            let (r1, r2) = (reps[n1 as usize], reps[n2 as usize]);
            for c in 1..4 {
                let (a, b) = (r1.node(c), r2.node(c));
                if a != BOUNDARY && b != BOUNDARY {
                    if let Some((f, amb)) = charge.shortest_path_flip(a, b) {
                        drag.add(n1, n2, c, c, f, amb);
                    }
                }
            }
            drag.add(n1, n2, NEUTRAL, NEUTRAL, 0, false);
        }
        drag
    }
}
