//! The spacetime kernel of a large circuit, locally: its low-weight generators, found by
//! hashing, and per shot the classes of the faults that fired, summed over products of the
//! generators those faults touch.
//!
//! A set of locations is in the kernel when the gauges can match what it flips, which the gauge
//! solver's reduction tests exactly: reduce each location's flips by the gauges and the residues
//! of a kernel element sum to zero. The generators sought are
//!
//! - single locations whose residue is zero (a rotation the state does not see);
//! - pairs with equal residues (two places the same fault can sit, say either side of a gate,
//!   or the two qubits of a weight-two stabiliser);
//! - three and four locations whose residues sum to zero, built from pairs of locations that
//!   share a detector (a stabiliser's worth of faults in one time step).
//!
//! A shot's weight is then a product over clusters of touched generators (those holding a
//! fault that fired, joined when they share a location) of |1 + Σ r|² / (1 + Σ |r|²) over the
//! products of up to `order` of the cluster's generators, times each untouched generator's
//! factor on its own. The estimate is self-normalised.

use std::collections::{BTreeSet, HashMap};

use super::{Element, GaugeSolver, Program, Shot, Slot};
use crate::statevec::C;

/// Options for the coherent sampler's kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoherentOptions {
    /// The most generators multiplied together in a shot's class sum.
    pub order: usize,
    /// The most generators in a cluster before products are cut to pairs.
    pub max_cluster: usize,
    /// Whether to look for generators of three and four locations.
    pub quads: bool,
    /// How many steps of shared locations a shot's clusters reach past the generators its
    /// faults touch.
    pub hops: usize,
}

impl Default for CoherentOptions {
    fn default() -> Self {
        CoherentOptions { order: 3, max_cluster: 12, quads: true, hops: 1 }
    }
}

/// The kernel's local generators, and each location's generators.
pub struct Kernel {
    pub generators: Vec<Element>,
    /// Groups of locations that flip exactly the same outcomes: the same fault in different
    /// places. Their class sums have a closed form.
    pub groups: Vec<Group>,
    /// Each location's group's first member (itself if in none).
    rep_of: Vec<usize>,
    /// A random word per location: a set's key is the XOR of its locations'.
    zobrist: Vec<u64>,
    by_location: Vec<Vec<usize>>,
    options: CoherentOptions,
}

/// Locations that flip exactly the same outcomes, with each one's operator relative to the
/// first's: toggling an even set T of them has ratio Π_{i∈T} τ_i, τ_i the location's own toggle
/// factor times `mu[i]`.
#[derive(Clone, Debug)]
pub struct Group {
    pub members: Vec<usize>,
    pub mu: Vec<C>,
}

/// A residue as a hashable key.
type Key = Vec<u64>;

fn xor(a: &[u64], b: &[u64]) -> Key {
    a.iter().zip(b).map(|(x, y)| x ^ y).collect()
}

impl Kernel {
    pub fn new(p: &Program, solver: &GaugeSolver, options: CoherentOptions) -> Kernel {
        let nl = p.locations.len();
        let words = p.slots.len().div_ceil(64);
        let residue: Vec<Key> = (0..nl)
            .map(|l| {
                let mut v = vec![0u64; words];
                for &s in &solver.flips[l] {
                    v[s / 64] ^= 1 << (s % 64);
                }
                solver.reduce(&mut v);
                v
            })
            .collect();
        let zero = vec![0u64; words];
        // Groups: equal flips, operators equal up to a phase that composes consistently.
        let mut by_flips: HashMap<&Vec<usize>, Vec<usize>> = HashMap::new();
        for l in 0..nl {
            if !solver.flips[l].is_empty() {
                by_flips.entry(&solver.flips[l]).or_default().push(l);
            }
        }
        let mut groups: Vec<Group> = Vec::new();
        let mut is_rep = vec![true; nl];
        let phase = |e: &Element| {
            let v = [C::ONE, C::I, C::new(-1.0, 0.0), C::new(0.0, -1.0)][e.k as usize % 4];
            if e.offset { C::ZERO - v } else { v }
        };
        let mut lists: Vec<&Vec<usize>> = by_flips.values().filter(|v| v.len() > 1).collect();
        lists.sort();
        for list in lists {
            // Runs of the list with no coherent location between that anticommutes with their
            // operator: such a location's fault would change the run's proposal.
            let mut runs: Vec<Vec<usize>> = Vec::new();
            let mut blockers: Vec<usize> = Vec::new();
            for &l in list.iter() {
                let start_new = match runs.last() {
                    None => true,
                    Some(run) => blockers.iter().any(|&b| b > run[0] && b < l),
                };
                if start_new {
                    runs.push(vec![l]);
                    blockers = p.anticommuting_after(l);
                } else {
                    runs.last_mut().unwrap().push(l);
                }
            }
            for run in runs {
                let rep = run[0];
                let mut members = vec![rep];
                let mut mu = vec![C::ONE];
                for &l in &run[1..] {
                    let Some(e) = p.element(&[rep, l], solver) else { continue };
                    if !e.z.is_empty() {
                        continue;
                    }
                    let m = phase(&e);
                    // Consistent with every member taken so far.
                    let ok = members.iter().zip(&mu).skip(1).all(|(&j, &mj)| {
                        let pair = if j < l { [j, l] } else { [l, j] };
                        p.element(&pair, solver).is_some_and(|e2| e2.z.is_empty() && (phase(&e2) - mj * m).norm2() < 1e-12)
                    });
                    if ok {
                        members.push(l);
                        mu.push(m);
                    }
                }
                if members.len() > 1 {
                    for &l in &members[1..] {
                        is_rep[l] = false;
                    }
                    groups.push(Group { members, mu });
                }
            }
        }
        let mut candidates: BTreeSet<Vec<usize>> = BTreeSet::new();
        // Singles, and pairs of equal residue (a chain through each group).
        let mut by_residue: HashMap<&Key, Vec<usize>> = HashMap::new();
        for (l, r) in residue.iter().enumerate() {
            if !is_rep[l] {
                continue;
            }
            if *r == zero {
                candidates.insert(vec![l]);
            } else {
                by_residue.entry(r).or_default().push(l);
            }
        }
        for group in by_residue.values() {
            for w in group.windows(2) {
                candidates.insert(vec![w[0], w[1]]);
            }
        }
        if options.quads {
            // Pairs of locations sharing a detector (or observable), keyed by their residues' sum.
            let record_of: Vec<Option<usize>> = p.slots.iter().map(|s| if let Slot::Record(k) = s { Some(*k) } else { None }).collect();
            let mut checks_of_record: Vec<Vec<usize>> = vec![Vec::new(); p.num_records];
            for (d, recs) in p.detectors.iter().chain(&p.observables).enumerate() {
                for &r in recs {
                    checks_of_record[r].push(d);
                }
            }
            let nchecks = p.detectors.len() + p.observables.len();
            let mut at_check: Vec<Vec<usize>> = vec![Vec::new(); nchecks];
            for l in 0..nl {
                let mut fired: Vec<usize> = Vec::new();
                for &s in &solver.flips[l] {
                    if let Some(k) = record_of[s] {
                        fired.extend(&checks_of_record[k]);
                    }
                }
                fired.sort_unstable();
                let mut odd = Vec::new();
                for d in fired {
                    if odd.last() == Some(&d) {
                        odd.pop();
                    } else {
                        odd.push(d);
                    }
                }
                if is_rep[l] {
                    for d in odd {
                        at_check[d].push(l);
                    }
                }
            }
            let mut pairs: HashMap<Key, Vec<(usize, usize)>> = HashMap::new();
            let mut seen: BTreeSet<(usize, usize)> = BTreeSet::new();
            for ls in &at_check {
                if ls.len() > 64 {
                    continue;
                }
                for (i, &a) in ls.iter().enumerate() {
                    for &b in &ls[i + 1..] {
                        let (a, b) = (a.min(b), a.max(b));
                        if residue[a] != residue[b] && seen.insert((a, b)) {
                            pairs.entry(xor(&residue[a], &residue[b])).or_default().push((a, b));
                        }
                    }
                }
            }
            for (key, list) in &pairs {
                // Triples: a pair summing to a single location's residue.
                if let Some(ls) = by_residue.get(key) {
                    for &(a, b) in list.iter().take(8) {
                        for &c in ls.iter().take(8) {
                            if c != a && c != b {
                                let mut g = vec![a, b, c];
                                g.sort_unstable();
                                candidates.insert(g);
                            }
                        }
                    }
                }
                // Quads: two disjoint pairs with the same sum.
                for (i, &(a, b)) in list.iter().enumerate().take(16) {
                    for &(c, d) in list[i + 1..].iter().take(16) {
                        if a != c && a != d && b != c && b != d {
                            let mut g = vec![a, b, c, d];
                            g.sort_unstable();
                            candidates.insert(g);
                        }
                    }
                }
            }
        }
        // Smallest first, each kept only if independent of those kept (over GF(2), by locations).
        let mut candidates: Vec<Vec<usize>> = candidates.into_iter().collect();
        candidates.sort_by_key(|g| g.len());
        let lw = nl.div_ceil(64);
        let mut basis: Vec<(usize, Vec<u64>)> = Vec::new();
        let mut generators: Vec<Element> = Vec::new();
        for g in candidates {
            let mut v = vec![0u64; lw];
            for &l in &g {
                v[l / 64] ^= 1 << (l % 64);
            }
            for (pivot, row) in &basis {
                if v[pivot / 64] >> (pivot % 64) & 1 == 1 {
                    v.iter_mut().zip(row).for_each(|(a, b)| *a ^= b);
                }
            }
            let Some(w) = v.iter().position(|&w| w != 0) else { continue };
            let Some(e) = p.element(&g, solver) else { continue };
            let pivot = w * 64 + v[w].trailing_zeros() as usize;
            for (_, row) in basis.iter_mut() {
                if row[pivot / 64] >> (pivot % 64) & 1 == 1 {
                    row.iter_mut().zip(&v).for_each(|(a, b)| *a ^= b);
                }
            }
            basis.push((pivot, v));
            generators.push(e);
        }
        let mut by_location = vec![Vec::new(); nl];
        for (i, g) in generators.iter().enumerate() {
            for &l in &g.members {
                by_location[l].push(i);
            }
        }
        let mut rep_of: Vec<usize> = (0..nl).collect();
        for g in &groups {
            for &l in &g.members {
                rep_of[l] = g.members[0];
            }
        }
        let zobrist = (0..nl as u64).map(|l| crate::batch_sampler::splitmix64(l.wrapping_add(0x9e37_79b9_7f4a_7c15))).collect();
        Kernel { generators, groups, rep_of, zobrist, by_location, options }
    }

    /// The whole kernel's dimension: locations less the rank of their residues.
    pub fn dimension(&self, p: &Program, solver: &GaugeSolver) -> usize {
        let words = p.slots.len().div_ceil(64);
        let mut rows: Vec<(usize, Vec<u64>)> = Vec::new();
        for l in 0..p.locations.len() {
            let mut v = vec![0u64; words];
            for &s in &solver.flips[l] {
                v[s / 64] ^= 1 << (s % 64);
            }
            solver.reduce(&mut v);
            for (pivot, r) in &rows {
                if v[pivot / 64] >> (pivot % 64) & 1 == 1 {
                    v.iter_mut().zip(r).for_each(|(a, b)| *a ^= b);
                }
            }
            if let Some(w) = v.iter().position(|&w| w != 0) {
                rows.push((w * 64 + v[w].trailing_zeros() as usize, v));
            }
        }
        p.locations.len() - rows.len()
    }

    /// The dimension the generators and groups span.
    pub fn local_span(&self) -> usize {
        let n = self.rep_of.len();
        let lw = n.div_ceil(64);
        let mut rows: Vec<(usize, Vec<u64>)> = Vec::new();
        let vecs = self.generators.iter().map(|g| g.members.clone()).chain(self.groups.iter().flat_map(|g| g.members[1..].iter().map(move |&l| vec![g.members[0], l])));
        for members in vecs {
            let mut v = vec![0u64; lw];
            for &l in &members {
                v[l / 64] ^= 1 << (l % 64);
            }
            for (pivot, r) in &rows {
                if v[pivot / 64] >> (pivot % 64) & 1 == 1 {
                    v.iter_mut().zip(r).for_each(|(a, b)| *a ^= b);
                }
            }
            if let Some(w) = v.iter().position(|&w| w != 0) {
                rows.push((w * 64 + v[w].trailing_zeros() as usize, v));
            }
        }
        rows.len()
    }

    /// The groups, as `Program::set_merge` takes them.
    pub fn merge_groups(&self) -> Vec<(Vec<usize>, Vec<C>)> {
        self.groups.iter().map(|g| (g.members.clone(), g.mu.clone())).collect()
    }

    /// A shot's weight, by clusters of the generators its faults touch. `cache` holds the
    /// products already carried through the circuit (by their locations).
    pub fn weight(&self, p: &Program, shot: &Shot, solver: &GaugeSolver, cache: &mut HashMap<u64, Option<Element>>) -> f64 {
        let mut touched: Vec<usize> = Vec::new();
        for (l, &f) in shot.fired.iter().enumerate() {
            if f {
                touched.extend(&self.by_location[self.rep_of[l]]);
            }
        }
        touched.sort_unstable();
        touched.dedup();
        // With their neighbours (generators sharing a location), whose products with them
        // move the faults on.
        for _ in 0..self.options.hops {
            let neighbours: Vec<usize> = touched.iter().flat_map(|&g| self.generators[g].members.iter().flat_map(|&l| self.by_location[l].iter().copied())).collect();
            touched.extend(neighbours);
            touched.sort_unstable();
            touched.dedup();
        }
        // Each group as one location: its sums over the even and the odd sets of its members to
        // toggle, in closed form. The even sum is the group's own factor; in any element holding
        // the group's first member, odd over even takes that member's place.
        let mut w = 1.0;
        let mut eff: Vec<Option<(C, f64)>> = vec![None; self.rep_of.len()];
        for g in &self.groups {
            let (mut plus, mut minus, mut nplus, mut nminus) = (C::ONE, C::ONE, 1.0, 1.0);
            for (&l, &mu) in g.members.iter().zip(&g.mu) {
                let tau = p.toggle(shot, l) * mu;
                plus = plus * (C::ONE + tau);
                minus = minus * (C::ONE - tau);
                nplus *= 1.0 + tau.norm2();
                nminus *= 1.0 - tau.norm2();
            }
            let (even, odd) = ((plus + minus).scale(0.5), (plus - minus).scale(0.5));
            // A class that cancels exactly (even = 0) has weight 0 unless an element moves it
            // to the odd sum; a tiny stand-in keeps that limit finite.
            let even = if even.norm2() < 1e-200 { C::new(1e-100, 0.0) } else { even };
            let (mut neven, mut nodd) = ((nplus + nminus) / 2.0, (nplus - nminus) / 2.0);
            if !p.merge.is_empty() {
                // Drawn as one, the group's parity came from the proposal, not the twirl: the
                // twirl's class sum over the drawn configuration (neven) becomes ρ over q, the
                // proposal's probability of the drawn parity over the twirl's of the drawn
                // configuration, and nodd/neven the proposal's odds of the other parity.
                let taus: Vec<C> = g.members.iter().zip(&g.mu).map(|(&l, &mu)| p.unfired_toggle(l, mu, shot.anti[l])).collect();
                let odd_p = Program::odd_proposal(&taus);
                let last = *g.members.last().unwrap();
                let fired = shot.fired[last];
                let cos2: f64 = g.members.iter().map(|&l| p.locations[l].theta.cos().powi(2)).product();
                let q_drawn = if fired { cos2 * p.locations[last].theta.tan().powi(2) } else { cos2 };
                let (rho, rho_other) = if fired { (odd_p, 1.0 - odd_p) } else { (1.0 - odd_p, odd_p) };
                neven = rho / q_drawn;
                nodd = rho_other / q_drawn;
            }
            w *= even.norm2() / neven;
            let inv = even.conj().scale(1.0 / even.norm2());
            eff[g.members[0]] = Some((odd * inv, nodd / neven));
        }
        // An element's ratio and its squared size, groups standing in for their first members.
        let term = |e: &Element| -> (C, f64) {
            let (mut r, mut n) = (p.prefactor(shot, e), 1.0);
            for &l in &e.members {
                match eff[l] {
                    Some((c, m)) => {
                        r = r * c;
                        n *= m;
                    }
                    None => {
                        let t = p.toggle(shot, l);
                        r = r * t;
                        n *= t.norm2();
                    }
                }
            }
            (r, n)
        };
        // Generators no fault touches, each on its own: |1 + r|² / (1 + |r|²).
        let mut t = touched.iter().peekable();
        for (g, e) in self.generators.iter().enumerate() {
            if t.peek() == Some(&&g) {
                t.next();
                continue;
            }
            let (r, n) = term(e);
            w *= (C::ONE + r).norm2() / (1.0 + n);
        }
        if touched.is_empty() {
            return w;
        }
        // Clusters: touched generators joined when they share a location.
        let n = touched.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        let mut owner: HashMap<usize, usize> = HashMap::new();
        for (i, &g) in touched.iter().enumerate() {
            for &l in &self.generators[g].members {
                if let Some(&j) = owner.get(&l) {
                    let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                    parent[a] = b;
                } else {
                    owner.insert(l, i);
                }
            }
        }
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(touched[i]);
        }
        for gens in clusters.values() {
            let order = if gens.len() > self.options.max_cluster { self.options.order.min(2) } else { self.options.order };
            // Generators of the cluster sharing a location.
            let n = gens.len();
            let mut at: HashMap<usize, Vec<usize>> = HashMap::new();
            for (i, &g) in gens.iter().enumerate() {
                for &l in &self.generators[g].members {
                    at.entry(l).or_default().push(i);
                }
            }
            let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
            for list in at.values() {
                for &a in list {
                    for &b in list {
                        if a != b {
                            adj[a].push(b);
                        }
                    }
                }
            }
            for a in adj.iter_mut() {
                a.sort_unstable();
                a.dedup();
            }
            let (mut sum, mut norm) = (C::ONE, 1.0);
            let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
            // Every connected set of up to `order` generators (each sharing a location with
            // another), by Wernicke's enumeration; each resulting element once.
            let mut visit = |subset: &[usize], seen: &mut std::collections::HashSet<u64>| {
                let mut members: Vec<usize> = Vec::new();
                for &i in subset {
                    members = symmetric_difference(&members, &self.generators[gens[i]].members);
                }
                if members.is_empty() {
                    return;
                }
                let key = members.iter().fold(0u64, |h, &l| h ^ self.zobrist[l]);
                if !seen.insert(key) {
                    return;
                }
                let e = if subset.len() == 1 {
                    Some(self.generators[gens[subset[0]]].clone())
                } else {
                    cache.entry(key).or_insert_with(|| p.element(&members, solver)).clone()
                };
                if let Some(e) = e {
                    let (r, m) = term(&e);
                    sum = sum + r;
                    norm += m;
                }
            };
            for v in 0..n {
                let ext: Vec<usize> = adj[v].iter().copied().filter(|&u| u > v).collect();
                let mut stack: Vec<(Vec<usize>, Vec<usize>)> = vec![(vec![v], ext)];
                while let Some((sub, mut ext)) = stack.pop() {
                    visit(&sub, &mut seen);
                    if sub.len() == order {
                        continue;
                    }
                    while let Some(w) = ext.pop() {
                        let mut next = ext.clone();
                        for &u in &adj[w] {
                            if u > v && !sub.contains(&u) && !next.contains(&u) && !sub.iter().any(|&s| adj[s].binary_search(&u).is_ok()) {
                                next.push(u);
                            }
                        }
                        let mut sub2 = sub.clone();
                        sub2.push(w);
                        stack.push((sub2, next));
                    }
                }
            }
            w *= sum.norm2() / norm;
        }
        w
    }
}

fn symmetric_difference(a: &[usize], b: &[usize]) -> Vec<usize> {
    let (mut i, mut j, mut out) = (0, 0, Vec::with_capacity(a.len() + b.len()));
    while i < a.len() || j < b.len() {
        if j == b.len() || (i < a.len() && a[i] < b[j]) {
            out.push(a[i]);
            i += 1;
        } else if i == a.len() || b[j] < a[i] {
            out.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;

    /// The truncated kernel's exact distribution against the state vector's: the bias, with no
    /// sampling noise.
    fn bias(text: &str, options: CoherentOptions) -> f64 {
        let c = Circuit::parse(text).unwrap();
        let p = Program::new(&c).unwrap();
        let solver = GaugeSolver::new(&p);
        let kernel = Kernel::new(&p, &solver, options);
        let mut cache = HashMap::new();
        let ours = p.exact_distribution_by(&mut |shot| kernel.weight(&p, shot, &solver, &mut cache), true, 1 << 22).unwrap();
        let total: f64 = ours.values().sum();
        let sv = crate::statevec::Program::new(&c).unwrap();
        let reference = sv.reference_parities().to_vec();
        let nd = sv.num_detectors();
        let mut theirs = std::collections::BTreeMap::new();
        for ((dets, obs), q) in sv.distribution(1 << 22).unwrap() {
            let d: Vec<bool> = dets.iter().zip(&reference).map(|(a, b)| a ^ b).collect();
            let o = (0..sv.num_observables()).fold(0u64, |acc, k| acc | (((obs >> k & 1 == 1) ^ reference[nd + k]) as u64) << k);
            *theirs.entry((d, o)).or_insert(0.0) += q;
        }
        let keys: BTreeSet<_> = ours.keys().chain(theirs.keys()).cloned().collect();
        keys.iter().map(|k| (ours.get(k).copied().unwrap_or(0.0) / total - theirs.get(k).copied().unwrap_or(0.0)).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn local_generators_are_found() {
        // Z before and after a CZ on its control is one fault in two places.
        let p = Program::new(&Circuit::parse("RX 0\nR 1\nI_ERROR[R_Z(theta=0.1)] 0\nCZ 0 1\nI_ERROR[R_Z(theta=0.1)] 0\nMX 0\nM 1").unwrap()).unwrap();
        let k = Kernel::new(&p, &GaugeSolver::new(&p), CoherentOptions::default());
        assert!(k.groups.iter().any(|g| g.members == vec![0, 1]), "{:?}", k.groups);
    }

    #[test]
    fn small_circuits_are_exact_or_nearly() {
        let rep = "R 0 1 2\nI_ERROR[R_X(theta=0.2)] 0 2\nCX 0 1\nCX 2 1\nMR 1\nI_ERROR[R_X(theta=0.15)] 0 2\nCX 0 1\nCX 2 1\nMR 1\n\
                   DETECTOR rec[-2]\nDETECTOR rec[-1] rec[-2]\nM 0 2\nDETECTOR rec[-1] rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]";
        let b = bias(rep, CoherentOptions::default());
        assert!(b < 1e-3, "{b}");
        let two = "RX 0\nR 1\nI_ERROR[R_Z(theta=0.3)] 0\nCZ 0 1\nI_ERROR[R_Z(theta=0.2)] 0\nMX 0\nM 1\nDETECTOR rec[-2]";
        assert!(bias(two, CoherentOptions::default()) < 1e-10);
    }
}

#[cfg(test)]
mod survey {
    use super::*;
    use crate::circuit::Circuit;

    /// The local kernel with merged groups, exhaustively, over random small circuits: as exact
    /// as the whole kernel.
    #[test]
    fn exact_over_random_circuits() {
        let mut rng = crate::surface_code::Xorshift::new(2026);
        let ops1 = ["H", "S", "X", "SQRT_X", "I_ERROR[R_X(theta=0.3)]", "I_ERROR[R_Y(theta=-0.25)]", "I_ERROR[R_Z(theta=0.4)]", "X_ERROR(0.1)", "DEPOLARIZE1(0.1)", "R", "RX", "M", "MX", "MR"];
        let ops2 = ["CX", "CZ", "SWAP", "II_ERROR[R_ZZ(theta=0.35)]"];
        let mut worst: f64 = 0.0;
        for _ in 0..80 {
            let n = 1 + (rng.next_u64() % 3) as usize;
            let mut lines = Vec::new();
            let (mut rot, mut rand, mut noise, mut measured) = (0, 0, 0, 0usize);
            while lines.len() < 12 {
                let two = n > 1 && rng.next_u64().is_multiple_of(3);
                let op = if two { ops2[(rng.next_u64() % ops2.len() as u64) as usize] } else { ops1[(rng.next_u64() % ops1.len() as u64) as usize] };
                if op.contains("theta") { if rot == 4 { continue; } rot += 1; }
                if op.contains("ERROR(") || op.contains("DEPOL") { if noise == 2 { continue; } noise += 1; }
                if op.starts_with('M') || (op.starts_with('R') && !op.starts_with("R_")) { if rand == 4 { continue; } rand += 1; measured += op.starts_with('M') as usize; }
                let a = (rng.next_u64() % n as u64) as usize;
                if two { let b = (a + 1 + (rng.next_u64() % (n as u64 - 1)) as usize) % n; lines.push(format!("{op} {a} {b}")); } else { lines.push(format!("{op} {a}")); }
            }
            lines.push(format!("M {}", (0..n).map(|q| q.to_string()).collect::<Vec<_>>().join(" ")));
            let total = measured + n;
            for _ in 0..2 { lines.push(format!("DETECTOR rec[-{}]", 1 + rng.next_u64() % total as u64)); }
            lines.push(format!("OBSERVABLE_INCLUDE(0) rec[-{}]", 1 + rng.next_u64() % total as u64));
            let text = lines.join("\n");
            let c = Circuit::parse(&text).unwrap();
            let p = Program::new(&c).unwrap();
            let solver = GaugeSolver::new(&p);
            let kernel = Kernel::new(&p, &solver, CoherentOptions::default());
            let mut cache = HashMap::new();
            let mut pm = p.clone();
            pm.set_merge(&kernel.merge_groups());
            let ours = pm.exact_distribution_by(&mut |shot| kernel.weight(&pm, shot, &solver, &mut cache), true, 1 << 22).unwrap();
            let full = p.exact_distribution(&p.full_kernel().unwrap(), true, 1 << 22).unwrap();
            let total_w: f64 = ours.values().sum();
            let b = full.iter().map(|(k, v)| (ours.get(k).copied().unwrap_or(0.0) / total_w - v).abs()).fold(0.0, f64::max);
            if b > worst {
                worst = b;
                println!("bias {b:.2e} (kernel {} generators, full {})\n{text}\n", kernel.generators.len(), p.full_kernel().unwrap().len());
            }
        }
        assert!(worst < 1e-9, "worst {worst:.2e}");
    }
}

#[cfg(test)]
mod merged {
    use super::*;
    use crate::circuit::Circuit;

    fn compare(text: &str) {
        let c = Circuit::parse(text).unwrap();
        let p = Program::new(&c).unwrap();
        let solver = GaugeSolver::new(&p);
        let kernel = Kernel::new(&p, &solver, CoherentOptions::default());
        let mut pm = p.clone();
        pm.set_merge(&kernel.merge_groups());
        let mut cache = HashMap::new();
        let ours = pm.exact_distribution_by(&mut |shot| kernel.weight(&pm, shot, &solver, &mut cache), true, 1 << 22).unwrap();
        let full = p.exact_distribution(&p.full_kernel().unwrap(), true, 1 << 22).unwrap();
        assert!(!kernel.groups.is_empty());
        for (k, v) in &full {
            assert!((ours.get(k).copied().unwrap_or(0.0) - v).abs() < 1e-12, "{text}\n{k:?}");
        }
    }

    #[test]
    fn merged_groups_are_exact() {
        compare("RX 0\nR 1\nI_ERROR[R_Z(theta=0.3)] 0\nCZ 0 1\nI_ERROR[R_Z(theta=0.2)] 0\nMX 0\nM 1\nDETECTOR rec[-2]");
        compare("RX 0\nI_ERROR[R_Z(theta=0.3)] 0\nZ_ERROR(0.2) 0\nI_ERROR[R_Z(theta=0.2)] 0\nMX 0\nDETECTOR rec[-1]");
    }
}

#[cfg(test)]
mod diagnose {
    use super::*;
    use crate::circuit::Circuit;

    /// Per-shot weights of the local kernel against the whole kernel's, on a circuit from
    /// `COH_CIRCUIT` (run with `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn local_against_full() {
        let c = Circuit::parse(&std::fs::read_to_string(std::env::var("COH_CIRCUIT").unwrap()).unwrap()).unwrap();
        let p = Program::new(&c).unwrap();
        let solver = GaugeSolver::new(&p);
        let order = std::env::var("COH_ORDER").map_or(3, |v| v.parse().unwrap());
        let k = Kernel::new(&p, &solver, CoherentOptions { order, ..CoherentOptions::default() });
        println!("kernel dimension {} local span {}", k.dimension(&p, &solver), k.local_span());
        if p.locations.len() > 20 {
            return;
        }
        let full = p.full_kernel().unwrap();
        println!("locations {} full kernel {} generators {} groups {:?}", p.locations.len(), full.len(), k.generators.len(), k.groups.iter().map(|g| g.members.clone()).collect::<Vec<_>>());
        for g in &k.generators {
            println!("  gen {:?}", g.members);
        }
        // Full kernel dimension and whether the local generators and groups span it.
        let nl = p.locations.len();
        let mut basis: Vec<u64> = Vec::new();
        let add = |v: u64, basis: &mut Vec<u64>| {
            let mut v = v;
            for &b in basis.iter() {
                v = v.min(v ^ b);
            }
            if v != 0 {
                basis.push(v);
                basis.sort_unstable_by(|a, b| b.cmp(a));
                true
            } else {
                false
            }
        };
        for g in &k.generators {
            add(g.members.iter().fold(0u64, |a, &l| a | 1 << l), &mut basis);
        }
        for g in &k.groups {
            for &l in &g.members[1..] {
                add((1u64 << g.members[0]) | (1 << l), &mut basis);
            }
        }
        let local_dim = basis.len();
        let mut missing = Vec::new();
        for e in &full {
            let v = e.members.iter().fold(0u64, |a, &l| a | 1 << l);
            if add(v, &mut basis) {
                missing.push(e.members.clone());
            }
        }
        println!("local span dim {local_dim}, full dim {}, missing {:?}", basis.len(), missing);
        let _ = nl;
        let mut cache = HashMap::new();
        let (mut worst, mut sum_l, mut sum_f) = (0.0f64, 0.0, 0.0);
        for seed in 0..20000u64 {
            let mut rng = crate::surface_code::Xorshift::new(seed);
            let shot = p.run_shot(&mut super::super::shot::Stream(&mut rng));
            let wl = k.weight(&p, &shot, &solver, &mut cache);
            let wf = p.weight(&shot, &full);
            sum_l += wl;
            sum_f += wf;
            if (wl - wf).abs() > worst {
                worst = (wl - wf).abs();
                let fired: Vec<usize> = (0..nl).filter(|&l| shot.fired[l]).collect();
                println!("seed {seed}: local {wl:.4} full {wf:.4} fired {fired:?}");
            }
        }
        println!("mean local {:.4} full {:.4}", sum_l / 20000.0, sum_f / 20000.0);
    }
}
