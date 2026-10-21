//! A shot's whole class sum by contraction: the sum over every product of the kernel's
//! generators, exact at any number of them, at a cost exponential only in how many interact.
//!
//! A product of the generators is c ∈ GF(2)^k (c_i = 1 for those in it). It holds location l
//! when an odd number of the generators holding l are in it, and its ratio (`Program::ratio`,
//! the groups' first members standing for their groups as in `Kernel::weight`) is
//!
//! ```text
//!     r(c) = i^{q(c)} · Π_i s_i^{c_i} · Π_l f_l(⊕_{i∋l} c_i)
//! ```
//!
//! - f_l(0) = 1 and f_l(1) is l's toggle;
//! - s_i = (−1)^{m′·z_i}, the shot's noiseless-branch outcomes on the unread slots generator i
//!   ends on (the slots of a product are the sum of its generators');
//! - the product's phase i^k (−1)^{offset} is a quadratic form over Z4,
//!   q(c) = Σ a_i c_i + 2 Σ_{i<j} b_ij c_i c_j, with a_i = k_i + 2·offset_i and b_ij from the
//!   product of the two. Only operators that meet can couple, so b_ij is zero for a pair whose
//!   operators, carried through the circuit, never act on one qubit at one time.
//!
//! The squared size Σ |r|² is the same sum with |f_l|² (a group's odd-over-even size) and no
//! phase. Both sums are contracted by eliminating the c_i one at a time (bucket elimination)
//! along a min-fill order of the graph whose edges are the generators sharing a location and the
//! pairs with b_ij = 1. Each variable's table spans it and its neighbours when it goes, 2^{w+1}
//! entries at width w, so a shot costs O(k·2^w). The plan (the order, each table's variables,
//! where each factor and table goes) is built once; a shot fills the factors and eliminates.
//! Past `max_entries` entries in all (at most `MAX_ENTRIES`), or with a table of more than 32
//! variables, the plan is not built and the kernel truncates instead.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::kernel::{symmetric_difference, Kernel};
use super::{reset_of, Element, GaugeSolver, Program, Step};
use crate::statevec::C;

/// One shot's factors, as the class sums read them.
pub struct ShotFactors {
    /// Each location's toggle ratio (for a group's first member, the group's odd sum over its
    /// even sum).
    pub toggle: Vec<C>,
    /// Each location's squared size (for a group's first member, the odd sum's over the even's).
    pub size: Vec<f64>,
    /// Each generator's sign from the shot's noiseless branch: true for (−1)^{m′·z_i} = −1.
    pub sign: Vec<bool>,
}

/// A factor of the class sum: 1 when an even number of the generators of its scope are in the
/// product, else the product of its locations' toggles. The factor of a single generator also
/// carries that generator's phase i^{a_i} and sign.
struct Factor {
    scope: Vec<usize>,
    locations: Vec<usize>,
    generator: Option<(usize, u8)>,
}

impl Factor {
    /// Its value at odd parity in the sum and in the squared size.
    fn value(&self, shot: &ShotFactors) -> (C, f64) {
        let (mut c, mut s) = (C::ONE, 1.0);
        for &l in &self.locations {
            c = c * shot.toggle[l];
            s *= shot.size[l];
        }
        if let Some((g, a)) = self.generator {
            c = c * [C::ONE, C::I, C::new(-1.0, 0.0), C::new(0.0, -1.0)][a as usize];
            if shot.sign[g] {
                c = C::ZERO - c;
            }
        }
        (c, s)
    }
}

/// An earlier table read by a later elimination: where it is, and its index for each of the
/// later table's, by 8 bits of that index at a time (each entry the bits it contributes).
struct Gather {
    offset: usize,
    chunks: Vec<[u32; 256]>,
}

impl Gather {
    fn index(&self, u: usize) -> usize {
        self.offset + self.chunks.iter().enumerate().fold(0, |acc, (c, t)| acc | t[(u >> (8 * c)) & 0xff] as usize)
    }
}

/// One variable summed out: the product of its bucket's factors and tables over `bits`
/// variables (it at bit 0, the rest after), summed over its two values into a table of the rest.
struct Elimination {
    bits: usize,
    /// The bucket's factors, with the bits of their scope.
    factors: Vec<(usize, u32)>,
    /// Where the bucket's pairs with b_ij = 1 make the sum's product negative (one bit per
    /// entry; empty without such pairs).
    negate: Vec<u64>,
    tables: Vec<Gather>,
    /// Where the result goes.
    out: usize,
}

/// A shot's class sums by contraction, planned once for a kernel.
pub struct ContractionPlan {
    factors: Vec<Factor>,
    steps: Vec<Elimination>,
    /// The results with no variable left, whose product is the sum.
    scalars: Vec<usize>,
    /// Room for every result.
    arena: usize,
    /// The largest table.
    largest: usize,
    width: usize,
    entries: usize,
    /// Each generator's a_i, and the pairs with b_ij = 1: the quadratic form, for checking (in
    /// debug builds and tests).
    #[cfg_attr(not(any(test, debug_assertions)), allow(dead_code))]
    phases: Vec<u8>,
    #[cfg_attr(not(any(test, debug_assertions)), allow(dead_code))]
    couplings: Vec<(usize, usize)>,
}

/// The largest budget: a larger `max_entries` is this one. A shot's tables then hold 2^31
/// entries at most, and none spans more than 30 variables besides the one it sums out.
pub const MAX_ENTRIES: usize = 1 << 31;

/// A table spans its variable and `degree` others: past 31 of them its index no longer fits the
/// 32-bit masks a plan reads it with, so a plan refuses a degree of 32 or more at any budget.
const MAX_DEGREE: usize = 32;

/// Why a complete kernel has no contraction plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Its tables would hold more than the budget's entries in all (or one would span 32
    /// variables or more besides its own): the kernel truncates, and a larger budget may plan it.
    Budget,
    /// Its generators' products are not what a kernel's are: a product of two did not carry
    /// through the circuit, or their phases are not a quadratic form. Neither can happen to a
    /// complete kernel's generators (their products are kernel elements, and a product's phase
    /// differs from the sum of its factors' by even cross terms only), so this is a bug: debug
    /// builds panic, release builds truncate and say so.
    Internal(&'static str),
}

/// An element's phase i^k (−1)^offset as a power of i.
fn phase(e: &Element) -> u8 {
    (e.k + if e.offset { 2 } else { 0 }) % 4
}

impl ContractionPlan {
    /// The plan for the kernel's generators (the kernel complete, so their products are the
    /// whole kernel up to the groups), or None when its tables would hold more than
    /// `max_entries` entries in all (0 turns contraction off; past `MAX_ENTRIES` it is
    /// `MAX_ENTRIES`). `try_build` says why.
    pub fn build(p: &Program, solver: &GaugeSolver, kernel: &Kernel, max_entries: usize) -> Option<ContractionPlan> {
        ContractionPlan::try_build(p, solver, kernel, max_entries).ok()
    }

    /// The plan, or why there is none: the budget's doing ([`Refusal::Budget`]) or a bug
    /// ([`Refusal::Internal`], which panics in debug builds).
    pub fn try_build(p: &Program, solver: &GaugeSolver, kernel: &Kernel, max_entries: usize) -> Result<ContractionPlan, Refusal> {
        let generators = &kernel.generators;
        let k = generators.len();
        let max_entries = max_entries.min(MAX_ENTRIES);
        // Every variable's table has two entries at least.
        if max_entries == 0 || k > max_entries / 2 {
            return Err(Refusal::Budget);
        }
        // The factors: a single generator's (its phase, its sign, the locations it alone
        // holds), then one per other set of generators that hold a location in common.
        let mut scope_of: Vec<Vec<usize>> = vec![Vec::new(); p.locations.len()];
        for (i, g) in generators.iter().enumerate() {
            for &l in &g.members {
                scope_of[l].push(i);
            }
        }
        let mut by_scope: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
        for (l, scope) in scope_of.iter().enumerate() {
            if !scope.is_empty() {
                by_scope.entry(scope.clone()).or_default().push(l);
            }
        }
        let phases: Vec<u8> = generators.iter().map(phase).collect();
        let mut factors: Vec<Factor> = (0..k).map(|i| Factor { scope: vec![i], locations: by_scope.remove(&vec![i]).unwrap_or_default(), generator: Some((i, phases[i])) }).collect();
        factors.extend(by_scope.into_iter().map(|(scope, locations)| Factor { scope, locations, generator: None }));
        let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); k];
        for f in &factors {
            for &a in &f.scope {
                adj[a].extend(f.scope.iter().copied().filter(|&b| b != a));
            }
        }
        // A cheap refusal first, before the couplings cost a product of each pair that meets:
        // the order of the shared locations' graph alone, which the couplings only add edges
        // to. Min-fill is a heuristic, though, not monotone under added edges, so this order
        // can (rarely) be wider than the whole graph's: a kernel the whole graph's order would
        // fit is then refused, conservatively (it truncates, with the warning; no weight is
        // wrong).
        min_fill(adj.clone(), max_entries).ok_or(Refusal::Budget)?;
        let pairs = meeting_pairs(p, solver, generators);
        debug_assert!(pairs.is_some(), "a generator of a complete kernel is not in the kernel");
        let pairs = pairs.ok_or(Refusal::Internal("a generator's gauges do not solve: it is not in the kernel"))?;
        let mut couplings = Vec::new();
        for (i, j) in pairs {
            let e = p.element(&symmetric_difference(&generators[i].members, &generators[j].members), solver);
            debug_assert!(e.is_some(), "the product of generators {i} and {j} is not in the kernel");
            let e = e.ok_or(Refusal::Internal("the product of two generators does not carry through the circuit"))?;
            let cross = (4 + phase(&e) - (phases[i] + phases[j]) % 4) % 4;
            debug_assert!(cross % 2 == 0, "generators {i} and {j}: their product's phase is not a quadratic form's (an odd cross term)");
            match cross {
                0 => {}
                2 => {
                    couplings.push((i, j));
                    adj[i].insert(j);
                    adj[j].insert(i);
                }
                _ => return Err(Refusal::Internal("the products' phases are not a quadratic form (an odd cross term)")),
            }
        }
        let order = min_fill(adj, max_entries).ok_or(Refusal::Budget)?;
        let plan = ContractionPlan::eliminate(factors, couplings, &order, phases);
        #[cfg(debug_assertions)]
        assert!(plan.factor_form_holds(p, solver, generators, 64, 0x5eed), "the class sum's factor form disagrees with Program::element");
        Ok(plan)
    }

    /// The elimination along `order`: each factor and coupling in the bucket of its first
    /// variable to go, each table in the bucket of its first remaining one.
    fn eliminate(factors: Vec<Factor>, couplings: Vec<(usize, usize)>, order: &[usize], phases: Vec<u8>) -> ContractionPlan {
        enum Input {
            Factor(usize),
            Pair(usize, usize),
            Table(usize),
        }
        let k = order.len();
        let mut pos = vec![0usize; k];
        for (e, &v) in order.iter().enumerate() {
            pos[v] = e;
        }
        let first = |scope: &[usize]| scope.iter().map(|&v| pos[v]).min().expect("a scope has variables");
        let mut buckets: Vec<Vec<Input>> = (0..k).map(|_| Vec::new()).collect();
        for (f, factor) in factors.iter().enumerate() {
            buckets[first(&factor.scope)].push(Input::Factor(f));
        }
        for &(i, j) in &couplings {
            buckets[first(&[i, j])].push(Input::Pair(i, j));
        }
        let mut rests: Vec<Vec<usize>> = Vec::with_capacity(k);
        let mut outs: Vec<usize> = Vec::with_capacity(k);
        let (mut steps, mut scalars, mut arena, mut largest, mut width, mut entries) = (Vec::with_capacity(k), Vec::new(), 0usize, 1usize, 0usize, 0usize);
        for e in 0..k {
            let v = order[e];
            let bucket = std::mem::take(&mut buckets[e]);
            let mut vars: BTreeSet<usize> = BTreeSet::new();
            for input in &bucket {
                match *input {
                    Input::Factor(f) => vars.extend(factors[f].scope.iter().copied()),
                    Input::Pair(i, j) => vars.extend([i, j]),
                    Input::Table(s) => vars.extend(rests[s].iter().copied()),
                }
            }
            vars.remove(&v);
            let rest: Vec<usize> = vars.into_iter().collect();
            let bit = |x: usize| if x == v { 0 } else { 1 + rest.binary_search(&x).expect("in the table") };
            let mask = |scope: &[usize]| scope.iter().fold(0u32, |m, &x| m | 1 << bit(x));
            let bits = 1 + rest.len();
            let size = 1usize << bits;
            let mut step = Elimination { bits, factors: Vec::new(), negate: Vec::new(), tables: Vec::new(), out: arena };
            for input in &bucket {
                match *input {
                    Input::Factor(f) => step.factors.push((f, mask(&factors[f].scope))),
                    Input::Pair(i, j) => {
                        let m = mask(&[i, j]) as usize;
                        if step.negate.is_empty() {
                            step.negate = vec![0u64; size.div_ceil(64)];
                        }
                        for u in (0..size).filter(|u| u & m == m) {
                            step.negate[u / 64] ^= 1 << (u % 64);
                        }
                    }
                    Input::Table(s) => {
                        let places: Vec<usize> = rests[s].iter().map(|&x| bit(x)).collect();
                        let chunks = (0..bits.div_ceil(8))
                            .map(|c| {
                                let mut t = [0u32; 256];
                                for (x, entry) in t.iter_mut().enumerate() {
                                    *entry = places.iter().enumerate().filter(|&(_, &pl)| pl / 8 == c && (x >> (pl % 8)) & 1 == 1).fold(0, |a, (b, _)| a | 1 << b);
                                }
                                t
                            })
                            .collect();
                        step.tables.push(Gather { offset: outs[s], chunks });
                    }
                }
            }
            arena += size / 2;
            largest = largest.max(size);
            width = width.max(rest.len());
            entries += size;
            if rest.is_empty() {
                scalars.push(step.out);
            } else {
                buckets[first(&rest)].push(Input::Table(e));
            }
            outs.push(step.out);
            rests.push(rest);
            steps.push(step);
        }
        ContractionPlan { factors, steps, scalars, arena, largest, width, entries, phases, couplings }
    }

    /// The most variables a table holds besides the one it sums out.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The entries of every table, summed: a shot's work.
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// The whole class's sums for a shot, the empty product's 1 included: (Σ r, Σ |r|²) over
    /// every product of the generators.
    pub fn class_sums(&self, shot: &ShotFactors) -> (C, f64) {
        let values: Vec<(C, f64)> = self.factors.iter().map(|f| f.value(shot)).collect();
        let (mut sums, mut sizes) = (vec![C::ZERO; self.arena], vec![0.0; self.arena]);
        let (mut t, mut n) = (vec![C::ONE; self.largest], vec![1.0; self.largest]);
        for step in &self.steps {
            let size = 1usize << step.bits;
            let (t, n) = (&mut t[..size], &mut n[..size]);
            t.fill(C::ONE);
            n.fill(1.0);
            for &(f, mask) in &step.factors {
                let (c, s) = values[f];
                for u in 0..size {
                    if (u as u32 & mask).count_ones() & 1 == 1 {
                        t[u] = t[u] * c;
                        n[u] *= s;
                    }
                }
            }
            for g in &step.tables {
                for u in 0..size {
                    let j = g.index(u);
                    t[u] = t[u] * sums[j];
                    n[u] *= sizes[j];
                }
            }
            if !step.negate.is_empty() {
                for (u, x) in t.iter_mut().enumerate() {
                    if step.negate[u / 64] >> (u % 64) & 1 == 1 {
                        *x = C::ZERO - *x;
                    }
                }
            }
            for u in 0..size / 2 {
                sums[step.out + u] = t[2 * u] + t[2 * u + 1];
                sizes[step.out + u] = n[2 * u] + n[2 * u + 1];
            }
        }
        self.scalars.iter().fold((C::ONE, 1.0), |(c, s), &j| (c * sums[j], s * sizes[j]))
    }

    /// Whether `count` random products of the generators (seeded) have the phase and the unread
    /// slots `Program::element` gives them: q(c) = Σ a_i + 2 Σ b_ij over the product, its slots
    /// the sum of the generators'.
    #[cfg(any(test, debug_assertions))]
    pub(crate) fn factor_form_holds(&self, p: &Program, solver: &GaugeSolver, generators: &[Element], count: usize, seed: u64) -> bool {
        let k = generators.len();
        if k == 0 {
            return true;
        }
        let coupled: HashSet<(usize, usize)> = self.couplings.iter().copied().collect();
        let mut rng = crate::surface_code::Xorshift::new(seed);
        (0..count).all(|_| {
            let size = 1 + (rng.next_u64() % k.min(8) as u64) as usize;
            let mut set: Vec<usize> = (0..size).map(|_| (rng.next_u64() % k as u64) as usize).collect();
            set.sort_unstable();
            set.dedup();
            let (mut members, mut z) = (Vec::new(), Vec::new());
            let mut q = 0usize;
            for (a, &i) in set.iter().enumerate() {
                members = symmetric_difference(&members, &generators[i].members);
                z = symmetric_difference(&z, &generators[i].z);
                q += self.phases[i] as usize + 2 * set[a + 1..].iter().filter(|&&j| coupled.contains(&(i, j))).count();
            }
            p.element(&members, solver).is_some_and(|e| phase(&e) as usize == q % 4 && e.z == z)
        })
    }
}

/// A min-fill elimination order of the graph `adj` (ties to fewer neighbours, then the lower
/// vertex), or None once its tables, 2^{1 + neighbours} entries each, pass `max_entries`, or a
/// vertex has `MAX_DEGREE` neighbours or more when it goes, whatever the budget.
fn min_fill(mut adj: Vec<BTreeSet<usize>>, max_entries: usize) -> Option<Vec<usize>> {
    let fill = |adj: &[BTreeSet<usize>], v: usize| {
        let nb: Vec<usize> = adj[v].iter().copied().collect();
        (0..nb.len()).map(|a| nb[a + 1..].iter().filter(|&&b| !adj[nb[a]].contains(&b)).count()).sum::<usize>()
    };
    let n = adj.len();
    let mut key: Vec<(usize, usize)> = (0..n).map(|v| (fill(&adj, v), adj[v].len())).collect();
    let mut queue: BTreeSet<(usize, usize, usize)> = (0..n).map(|v| (key[v].0, key[v].1, v)).collect();
    let (mut order, mut entries) = (Vec::with_capacity(n), 0usize);
    while let Some((_, degree, v)) = queue.pop_first() {
        if degree >= MAX_DEGREE {
            return None;
        }
        // Checked: on a 32-bit target a table of 2^32 entries does not fit a usize.
        entries = entries.checked_add(1usize.checked_shl(degree as u32 + 1)?)?;
        if entries > max_entries {
            return None;
        }
        let nb: Vec<usize> = adj[v].iter().copied().collect();
        // The fill of v's neighbours and of theirs can change.
        let mut affected: BTreeSet<usize> = nb.iter().copied().collect();
        for &a in &nb {
            affected.extend(adj[a].iter().copied());
        }
        affected.remove(&v);
        for &x in &affected {
            queue.remove(&(key[x].0, key[x].1, x));
        }
        for &a in &nb {
            adj[a].remove(&v);
            adj[a].extend(nb.iter().copied().filter(|&b| b != a));
        }
        adj[v].clear();
        order.push(v);
        for &x in &affected {
            key[x] = (fill(&adj, x), adj[x].len());
            queue.insert((key[x].0, key[x].1, x));
        }
    }
    Some(order)
}

/// The pairs of generators whose operators meet: carried through the circuit (each one's
/// locations and its gauges apart), acting on one qubit at one time, or sharing a location. No
/// other pair's product has a phase other than the sum of theirs. None if a generator is not in
/// the kernel.
fn meeting_pairs(p: &Program, solver: &GaugeSolver, generators: &[Element]) -> Option<Vec<(usize, usize)>> {
    let k = generators.len();
    let mut by_location: Vec<Vec<usize>> = vec![Vec::new(); p.locations.len()];
    let mut by_gauge: Vec<Vec<usize>> = vec![Vec::new(); p.gauges.len()];
    for (i, g) in generators.iter().enumerate() {
        let mut flipped = vec![0u64; p.slots.len().div_ceil(64)];
        for &l in &g.members {
            by_location[l].push(i);
            for &s in &solver.flips[l] {
                flipped[s / 64] ^= 1 << (s % 64);
            }
        }
        for j in solver.solve(&flipped)? {
            by_gauge[j].push(i);
        }
    }
    let mut f = Footprints { words: k.div_ceil(64), ops: vec![0; p.num_qubits * 4 * k.div_ceil(64)], last: vec![Vec::new(); p.num_qubits], pairs: HashSet::new() };
    for gens in &by_location {
        for (a, &i) in gens.iter().enumerate() {
            f.pairs.extend(gens[a + 1..].iter().map(|&j| (i, j)));
        }
    }
    for q in 0..p.num_qubits {
        f.insert(q, p.gauges[q].pauli, 2, &by_gauge[q]);
        f.record(q);
    }
    for (s, step) in p.steps.iter().enumerate() {
        match *step {
            Step::H(q) => {
                f.apply(q, q, |x, z, _, _| std::mem::swap(x, z));
            }
            Step::S(q) => f.apply(q, q, |x, z, _, _| *z ^= *x),
            Step::Cx(c, t) => {
                f.apply(c, t, |xc, zc, xt, zt| {
                    *xt ^= *xc;
                    *zc ^= *zt;
                });
                f.record(c);
                f.record(t);
            }
            Step::Cz(a, b) => {
                f.apply(a, b, |xa, za, xb, zb| {
                    *zb ^= *xa;
                    *za ^= *xb;
                });
                f.record(a);
                f.record(b);
            }
            Step::Measure { q, .. } | Step::Reset { q, .. } => {
                if let Some(g) = p.gauge_at[s][0] {
                    f.insert(q, p.gauges[g].pauli, 2, &by_gauge[g]);
                    f.record(q);
                }
                if reset_of(step) {
                    f.clear(q);
                }
                if let Some(g) = p.gauge_at[s][1] {
                    f.insert(q, p.gauges[g].pauli, 2, &by_gauge[g]);
                }
                f.record(q);
            }
            Step::Coherent(loc) if !by_location[loc].is_empty() => {
                for &(q, c) in &p.locations[loc].paulis {
                    f.insert(q, c, 0, &by_location[loc]);
                    f.record(q);
                }
            }
            _ => {}
        }
    }
    let mut pairs: Vec<(usize, usize)> = f.pairs.into_iter().collect();
    pairs.sort_unstable();
    Some(pairs)
}

/// Every generator's operator carried through the circuit at once, one bit per generator: on
/// each qubit four rows (its locations' X and Z parts, its gauges' X and Z parts), and the pairs
/// found on one qubit at one time.
struct Footprints {
    words: usize,
    ops: Vec<u64>,
    /// Each qubit's generators when last recorded.
    last: Vec<Vec<u64>>,
    pairs: HashSet<(usize, usize)>,
}

impl Footprints {
    fn row(&self, q: usize, r: usize) -> std::ops::Range<usize> {
        let start = (q * 4 + r) * self.words;
        start..start + self.words
    }

    /// The Pauli `c` on qubit q into the location rows (`at` 0) or the gauge rows (`at` 2) of
    /// the generators `gens`.
    fn insert(&mut self, q: usize, c: u8, at: usize, gens: &[usize]) {
        for &i in gens {
            for (r, bit) in [(at, c & 1), (at + 1, c & 2)] {
                if bit != 0 {
                    let start = self.row(q, r).start;
                    self.ops[start + i / 64] ^= 1 << (i % 64);
                }
            }
        }
    }

    /// A gate on qubits a and b (the same for a one-qubit gate) applied to both parts' rows, word
    /// by word: `gate(x_a, z_a, x_b, z_b)`.
    fn apply(&mut self, a: usize, b: usize, gate: impl Fn(&mut u64, &mut u64, &mut u64, &mut u64)) {
        for part in [0, 2] {
            for w in 0..self.words {
                let (xa, za, xb, zb) = (self.row(a, part).start + w, self.row(a, part + 1).start + w, self.row(b, part).start + w, self.row(b, part + 1).start + w);
                let (mut v, mut u, mut s, mut t) = (self.ops[xa], self.ops[za], self.ops[xb], self.ops[zb]);
                gate(&mut v, &mut u, &mut s, &mut t);
                if a != b {
                    self.ops[xb] = s;
                    self.ops[zb] = t;
                }
                self.ops[xa] = v;
                self.ops[za] = u;
            }
        }
    }

    fn clear(&mut self, q: usize) {
        let start = self.row(q, 0).start;
        self.ops[start..start + 4 * self.words].fill(0);
    }

    /// The generators on qubit q now: every pair of them meets.
    fn record(&mut self, q: usize) {
        let here: Vec<u64> = (0..self.words).map(|w| (0..4).fold(0, |a, r| a | self.ops[self.row(q, r).start + w])).collect();
        if here == self.last[q] {
            return;
        }
        let gens: Vec<usize> = (0..self.words * 64).filter(|&i| here[i / 64] >> (i % 64) & 1 == 1).collect();
        for (a, &i) in gens.iter().enumerate() {
            self.pairs.extend(gens[a + 1..].iter().map(|&j| (i, j)));
        }
        self.last[q] = here;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::coherent::kernel::{CoherentOptions, MAX_CONTRACTION};
    use crate::surface_code::Xorshift;
    use std::collections::HashMap;

    fn kernel(text: &str, options: CoherentOptions) -> (Program, GaugeSolver, Kernel) {
        let p = Program::new(&Circuit::parse(text).unwrap()).unwrap();
        let solver = GaugeSolver::new(&p);
        let k = Kernel::new(&p, &solver, options);
        (p, solver, k)
    }

    /// A random circuit of `rotations` single-location rotations (X, Y, Z and ZZ) among S, H,
    /// CX and CZ on a few qubits, up to `measured` of them measured midway, all at the end.
    fn random_circuit(rng: &mut crate::surface_code::Xorshift, qubits: usize, rotations: usize, measured: usize) -> String {
        let mut pick = |n: usize| (rng.next_u64() % n as u64) as usize;
        let mut lines = vec![format!("R {}", (0..qubits).map(|q| q.to_string()).collect::<Vec<_>>().join(" "))];
        for q in 0..qubits {
            if pick(2) == 0 {
                lines.push(format!("H {q}"));
            }
        }
        let (mut placed, mut records) = (0, 0usize);
        while placed < rotations {
            let a = pick(qubits);
            let b = (a + 1 + pick(qubits - 1)) % qubits;
            let theta = [0.13, -0.21, 0.27, 0.34, -0.4][pick(5)];
            match pick(12) {
                0 => lines.push(format!("H {a}")),
                1 => lines.push(format!("S {a}")),
                2 => lines.push(format!("CX {a} {b}")),
                3 => lines.push(format!("CZ {a} {b}")),
                4 if records < measured => {
                    lines.push(format!("{} {a}", ["MR", "MX", "M"][pick(3)]));
                    records += 1;
                }
                5 => {
                    lines.push(format!("II_ERROR[R_ZZ(theta={theta})] {a} {b}"));
                    placed += 1;
                }
                _ => {
                    lines.push(format!("I_ERROR[R_{}(theta={theta})] {a}", ["X", "Y", "Z"][pick(3)]));
                    placed += 1;
                }
            }
        }
        lines.push(format!("M {}", (0..qubits).map(|q| q.to_string()).collect::<Vec<_>>().join(" ")));
        let total = records + qubits;
        for _ in 0..2 {
            lines.push(format!("DETECTOR rec[-{}]", 1 + pick(total)));
        }
        lines.push(format!("OBSERVABLE_INCLUDE(0) rec[-{}]", 1 + pick(total)));
        lines.join("\n")
    }

    /// A complete kernel's plan (the default budget's), asserting that a kernel refused one
    /// needs it: refused by the budget, and needing more entries than the budget along the order
    /// a plan with no budget takes. A refusal within the budget (by the early pass on the shared
    /// locations alone, or by a bug) fails.
    fn plan_within_budget<'a>(p: &Program, solver: &GaugeSolver, kernel: &'a Kernel, text: &str) -> Option<&'a ContractionPlan> {
        if kernel.complete && kernel.plan().is_none() {
            assert_eq!(kernel.contraction_refusal(), Some(Refusal::Budget), "{text}");
            let unbounded = ContractionPlan::try_build(p, solver, kernel, usize::MAX).unwrap_or_else(|why| panic!("no plan at any budget ({why:?})\n{text}"));
            assert!(unbounded.entries() > MAX_CONTRACTION, "refused within the budget: the plan needs {} entries\n{text}", unbounded.entries());
        }
        kernel.plan()
    }

    /// Generated circuits whose kernels have `k` generators, each with a contraction plan:
    /// (text, program, solver, kernel), the first `count` the generator gives for each k, those
    /// with couplings (b_ij ≠ 0) and, with `groups` and fewer generators than rotations, groups.
    /// Every complete kernel met with k generators has a plan, or needs more than the budget.
    /// The trials take their number of qubits from `qubits` in turn.
    fn circuits(ks: std::ops::RangeInclusive<usize>, (qubits, rotations, measured): (&[usize], usize, usize), count: usize, groups: bool) -> Vec<(String, Program, GaugeSolver, Kernel)> {
        let mut rng = crate::surface_code::Xorshift::new(2026);
        let mut found: BTreeMap<usize, usize> = BTreeMap::new();
        let mut out = Vec::new();
        for trial in 0..20_000 {
            if ks.clone().all(|k| found.get(&k).copied().unwrap_or(0) >= count) {
                break;
            }
            let text = random_circuit(&mut rng, qubits[trial % qubits.len()], rotations, measured);
            let (p, solver, kernel) = kernel(&text, CoherentOptions::default());
            let k = kernel.generators.len();
            if !ks.contains(&k) || found.get(&k).copied().unwrap_or(0) >= count || !kernel.complete {
                continue;
            }
            let Some(plan) = plan_within_budget(&p, &solver, &kernel, &text) else { continue };
            if plan.couplings.is_empty() || (groups && k < rotations && kernel.groups.is_empty()) {
                continue;
            }
            *found.entry(k).or_insert(0) += 1;
            out.push((text, p, solver, kernel));
        }
        assert!(ks.clone().all(|k| found.get(&k) == Some(&count)), "{found:?}");
        out
    }

    /// The groups' own sums for a shot: Π even and Π of the even sets' squared sizes, which
    /// the whole kernel's sums hold besides the class sums the contraction gives.
    fn group_sums(p: &Program, kernel: &Kernel, shot: &super::super::Shot) -> (C, f64) {
        let (mut even, mut size) = (C::ONE, 1.0);
        for g in &kernel.groups {
            let (mut plus, mut minus, mut nplus, mut nminus) = (C::ONE, C::ONE, 1.0, 1.0);
            for (&l, &mu) in g.members.iter().zip(&g.mu) {
                let tau = p.toggle(shot, l) * mu;
                plus = plus * (C::ONE + tau);
                minus = minus * (C::ONE - tau);
                nplus *= 1.0 + tau.norm2();
                nminus *= 1.0 - tau.norm2();
            }
            even = even * (plus + minus).scale(0.5);
            size *= (nplus + nminus) / 2.0;
        }
        (even, size)
    }

    /// The against_whole harness (`kernel::nonlocal`), with the whole kernel's sums taken
    /// apart: over `shots` shots of `run` (the program, or it with the groups merged), the
    /// largest relative difference between the contraction's class sums, times the groups'
    /// own, and the whole kernel's 1 + Σ r and 1 + Σ |r|² (Program::ratio's factors, each
    /// location's toggle computed once a shot), and between the kernel's weight and the whole
    /// kernel's. The first shots' whole weights are Program::weight's itself.
    fn against_whole_parts(p: &Program, solver: &GaugeSolver, kernel: &Kernel, full: &[Element], run: &Program, shots: u64) -> f64 {
        let plan = kernel.plan().unwrap();
        let mut cache = HashMap::new();
        let mut worst = 0.0f64;
        for seed in 0..shots {
            let mut rng = crate::surface_code::Xorshift::new(seed);
            let shot = run.run_shot(&mut super::super::shot::Stream(&mut rng));
            let toggles: Vec<C> = (0..p.locations.len()).map(|l| p.toggle(&shot, l)).collect();
            let (mut sum, mut size) = (C::ONE, 1.0);
            for e in full {
                let r = e.members.iter().fold(p.prefactor(&shot, e), |acc, &l| acc * toggles[l]);
                sum = sum + r;
                size += r.norm2();
            }
            let whole = sum.norm2() / size;
            if seed < 3 {
                assert!((whole - p.weight(&shot, full)).abs() <= 1e-12 * whole.max(1.0));
            }
            let (s, n) = plan.class_sums(&kernel.shot_factors(p, &shot).1);
            let (even, neven) = group_sums(p, kernel, &shot);
            let w = kernel.weight(p, &shot, solver, &mut cache);
            worst = worst.max((s * even - sum).norm2().sqrt() / size.sqrt()).max((n * neven - size).abs() / size).max((w - whole).abs() / whole.max(1.0));
        }
        worst
    }

    /// Every shot's class sums are the whole kernel's, from thirteen generators to twenty, the
    /// circuits mixing X, Y and Z rotations, ZZ crosstalk, S, H, CX and CZ, with couplings
    /// (b_ij ≠ 0) and groups, on shots drawn with the groups merged and without.
    #[test]
    fn equals_whole_kernel_k13_to_20() {
        for (text, p, solver, kernel) in circuits(13..=20, (&[4, 5], 20, 3), 1, true) {
            let plan = kernel.plan().unwrap();
            assert!(plan.factor_form_holds(&p, &solver, &kernel.generators, 256, 1), "{text}");
            let full = p.full_kernel().unwrap();
            let mut merged = p.clone();
            merged.set_merge(&kernel.merge_groups());
            // The whole kernel's sums cost a pass over its elements (up to 2^20) a shot.
            let shots = (15_000_000 / full.len() as u64).clamp(20, 300);
            for run in [&p, &merged] {
                let worst = against_whole_parts(&p, &solver, &kernel, &full, run, shots);
                assert!(worst < 1e-10, "k = {}, width {}\n{text}\nworst {worst:.3e}", kernel.generators.len(), plan.width());
            }
        }
    }

    /// Random small circuits past twelve generators, contracted: each outcome's weighted
    /// probability is the state vector's (fifteen rotations on two qubits, measured at the end:
    /// 2^21 branches to sum, the frame's six coins and the rotations').
    #[test]
    fn matches_state_vector_random_small() {
        for (text, _, _, kernel) in circuits(13..=13, (&[2], 15, 0), 2, false) {
            assert!(kernel.is_whole());
            let b = crate::coherent::kernel::tests::bias(&text, CoherentOptions::default());
            assert!(b < 1e-10, "{text}\nbias {b:.3e}");
        }
    }

    /// A budget of 0 builds no plan, and the sampler then truncates as 2.1 did: on the d = 3
    /// surface memory over three rounds (fourteen generators) the weights of seed 9's first
    /// 20,000 shots are 2.1's (its sum and first weights, recorded from the 2.1.0 wheel).
    #[test]
    fn budget_zero_falls_back() {
        let text = crate::coherent::kernel::nonlocal::surface(3, 3, 0.25);
        let (p, solver, k) = kernel(&text, CoherentOptions::default());
        assert_eq!((k.generators.len(), k.contraction_width()), (14, Some(3)));
        assert!(ContractionPlan::build(&p, &solver, &k, 0).is_none());
        let options = CoherentOptions { max_contraction: 0, ..CoherentOptions::default() };
        let (_, _, k0) = kernel(&text, options);
        assert!(k0.plan().is_none() && k0.contraction_width().is_none() && !k0.is_whole());
        assert_eq!((k.contraction_refusal(), k0.contraction_refusal()), (None, Some(Refusal::Budget)));
        let s = crate::coherent::sampler::CoherentSampler::new(&Circuit::parse(&text).unwrap(), options).unwrap();
        let w = s.sample_seeded(9, 0, 20_000, 1).weights;
        let sum: f64 = w.iter().sum();
        assert!((sum - 20213.884915594033).abs() < 1e-9, "{sum}");
        for (a, b) in w.iter().zip([1.0974372803964274, 2.255544504808714, 1.0509262125690255, 0.19754302237313576, 2.077504971231143, 2.4156493000859047]) {
            assert!((a - b).abs() < 1e-12, "{a} {b}");
        }
        // With the plan, the same shots' weights are the whole class's: they differ.
        let exact = crate::coherent::sampler::CoherentSampler::new(&Circuit::parse(&text).unwrap(), CoherentOptions::default()).unwrap();
        assert!(exact.kernel.is_whole());
        assert!((exact.sample_seeded(9, 0, 20_000, 1).weights.iter().sum::<f64>() - sum).abs() > 1.0);
    }

    /// Past the budget nothing is built, however wide: the plan never allocates 2^width.
    #[test]
    fn wide_kernels_are_refused_quickly() {
        let text = crate::coherent::kernel::nonlocal::surface(3, 3, 0.25);
        let (p, solver, k) = kernel(&text, CoherentOptions::default());
        let plan = k.plan().unwrap();
        assert_eq!((plan.width(), ContractionPlan::build(&p, &solver, &k, plan.entries()).is_some()), (3, true));
        assert!(ContractionPlan::build(&p, &solver, &k, plan.entries() - 1).is_none());
    }

    /// However large the budget, no table spans more than 32 variables (past that its index
    /// overflows the plan's 32-bit masks) or holds more than 2^31 entries: the order refuses a
    /// variable of 32 neighbours or more at any budget, and a budget past `MAX_ENTRIES` is
    /// `MAX_ENTRIES`. Forty rounds of the d = 3 surface memory have 39 non-local generators,
    /// each the first round's logical with another round's, all sharing the first round's
    /// rotations: a clique of 39, refused at usize::MAX before any pair is multiplied.
    #[test]
    fn huge_budgets_never_plan_32_wide_tables() {
        let clique = |n: usize| -> Vec<BTreeSet<usize>> { (0..n).map(|v| (0..n).filter(|&u| u != v).collect()).collect() };
        // 32 variables: 31 neighbours at the first, 2^33 − 2 entries in all.
        assert_eq!(min_fill(clique(32), usize::MAX).map(|order| order.len()), Some(32));
        assert_eq!(min_fill(clique(32), (1 << 33) - 3), None);
        assert_eq!(min_fill(clique(33), usize::MAX), None);
        let (p, solver, k) = kernel(&crate::coherent::kernel::nonlocal::surface(3, 40, 0.25), CoherentOptions::default());
        assert!(k.complete && k.num_nonlocal == 39 && k.plan().is_none(), "{} {}", k.complete, k.num_nonlocal);
        assert_eq!(k.contraction_refusal(), Some(Refusal::Budget));
        assert_eq!(ContractionPlan::try_build(&p, &solver, &k, usize::MAX).err(), Some(Refusal::Budget));
        let wide = CoherentOptions { max_contraction: usize::MAX, ..CoherentOptions::default() };
        let (_, _, k) = kernel(&crate::coherent::kernel::nonlocal::surface(3, 40, 0.25), wide);
        assert!(k.complete && k.contraction_width().is_none() && k.contraction_refusal() == Some(Refusal::Budget));
        // A narrow kernel plans the same at any budget past its own.
        let (p, solver, k) = kernel(&crate::coherent::kernel::nonlocal::surface(3, 3, 0.25), CoherentOptions::default());
        let plan = ContractionPlan::build(&p, &solver, &k, usize::MAX).unwrap();
        assert_eq!((plan.width(), plan.entries()), (k.plan().unwrap().width(), k.plan().unwrap().entries()));
    }

    /// A refusal that is not the budget's is a bug, never advice to raise the budget: with one
    /// generator's phase off by one (its products' phases then no quadratic form), debug builds
    /// panic, and release builds refuse with `Refusal::Internal`.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "not a quadratic form"))]
    fn odd_cross_terms_are_a_bug() {
        let (p, solver, mut k) = kernel(&crate::coherent::kernel::nonlocal::surface(3, 3, 0.25), CoherentOptions::default());
        let (i, _) = meeting_pairs(&p, &solver, &k.generators).unwrap()[0];
        assert_eq!(k.contraction_refusal(), None);
        k.generators[i].k = (k.generators[i].k + 1) % 4;
        let why = ContractionPlan::try_build(&p, &solver, &k, MAX_CONTRACTION).err();
        assert!(matches!(why, Some(Refusal::Internal(what)) if what.contains("quadratic form")), "{why:?}");
    }

    /// A random circuit for the brute force: rotations about X, Y and Z, about XX, YY and ZZ, and
    /// about mixed two- and three-qubit Pauli products (R_PAULI), among H, S, S_DAG, SQRT_X, CX,
    /// CZ, CY and SWAP, resets (R, RX) and up to `measured` mid-circuit measurements (MR, MX, M,
    /// MRX, MY), every qubit measured at the end.
    fn random_circuit_wide(rng: &mut Xorshift, qubits: usize, rotations: usize, measured: usize) -> String {
        let mut pick = |n: usize| (rng.next_u64() % n as u64) as usize;
        let mut lines = vec![format!("R {}", (0..qubits).map(|q| q.to_string()).collect::<Vec<_>>().join(" "))];
        for q in 0..qubits {
            if pick(2) == 0 {
                lines.push(format!("H {q}"));
            }
        }
        let (mut placed, mut records) = (0, 0usize);
        while placed < rotations {
            let a = pick(qubits);
            let b = (a + 1 + pick(qubits - 1)) % qubits;
            let c = (b + 1 + pick(qubits - 1)) % qubits;
            let theta = [0.13, -0.21, 0.27, 0.34, -0.4, 0.17][pick(6)];
            match pick(22) {
                0 => lines.push(format!("H {a}")),
                1 => lines.push(format!("S {a}")),
                2 => lines.push(format!("S_DAG {a}")),
                3 => lines.push(format!("SQRT_X {a}")),
                4 => lines.push(format!("CX {a} {b}")),
                5 => lines.push(format!("CZ {a} {b}")),
                6 => lines.push(format!("CY {a} {b}")),
                7 => lines.push(format!("SWAP {a} {b}")),
                8 if records < measured => {
                    lines.push(format!("{} {a}", ["MR", "MX", "M", "MRX", "MY"][pick(5)]));
                    records += 1;
                }
                9 => lines.push(format!("{} {a}", ["R", "RX"][pick(2)])),
                10 => {
                    lines.push(format!("II_ERROR[R_{p}{p}(theta={theta})] {a} {b}", p = ["X", "Y", "Z"][pick(3)]));
                    placed += 1;
                }
                11 if qubits >= 3 && a != c => {
                    let letters: String = (0..3).map(|_| ['X', 'Y', 'Z'][pick(3)]).collect();
                    lines.push(format!("I_ERROR[R_PAULI(theta={theta}, pauli={letters})] {a} {b} {c}"));
                    placed += 1;
                }
                12 => {
                    let letters: String = (0..2).map(|_| ['X', 'Y', 'Z'][pick(3)]).collect();
                    lines.push(format!("I_ERROR[R_PAULI(theta={theta}, pauli={letters})] {a} {b}"));
                    placed += 1;
                }
                _ => {
                    lines.push(format!("I_ERROR[R_{}(theta={theta})] {a}", ["X", "Y", "Z"][pick(3)]));
                    placed += 1;
                }
            }
        }
        lines.push(format!("M {}", (0..qubits).map(|q| q.to_string()).collect::<Vec<_>>().join(" ")));
        let total = records + qubits;
        for _ in 0..2 {
            lines.push(format!("DETECTOR rec[-{}]", 1 + pick(total)));
        }
        lines.push(format!("OBSERVABLE_INCLUDE(0) rec[-{}]", 1 + pick(total)));
        lines.join("\n")
    }

    /// Every nonzero product of the generators, each listed once (in Gray-code order), carried
    /// through the circuit by `Program::element`.
    fn every_product(p: &Program, solver: &GaugeSolver, generators: &[Element]) -> Vec<Element> {
        let mut out = Vec::with_capacity((1 << generators.len()) - 1);
        let mut members: Vec<usize> = Vec::new();
        for i in 1u64..(1u64 << generators.len()) {
            members = symmetric_difference(&members, &generators[i.trailing_zeros() as usize].members);
            out.push(p.element(&members, solver).expect("a product of kernel generators is in the kernel"));
        }
        out
    }

    /// A shot's class sums by brute force, over `products` (every nonzero product of the
    /// generators) and the empty one, from the shot's factors as the contraction reads them:
    /// (Σ r, Σ |r|², Σ |r|).
    fn sums_of_every_product(products: &[Element], p: &Program, f: &ShotFactors, shot: &crate::coherent::Shot) -> (C, f64, f64) {
        let (mut sum, mut norm, mut l1) = (C::ONE, 1.0, 1.0f64);
        for e in products {
            let (mut r, mut n) = (p.prefactor(shot, e), 1.0);
            for &l in &e.members {
                r = r * f.toggle[l];
                n *= f.size[l];
            }
            sum = sum + r;
            norm += n;
            l1 += r.norm2().sqrt();
        }
        (sum, norm, l1)
    }

    /// Every shot's class sums are the sums over all 2^k products of the generators, each
    /// listed by `Program::element` and summed from the shot's factors: k from 13 to 16, eight
    /// circuits each on 3–6 qubits with 18–29 rotations (some past `full_kernel`'s 20 locations),
    /// mixing every rotation and gate `random_circuit_wide` draws, with couplings and groups, on
    /// 60 shots drawn plain and 60 with the groups merged. The difference is measured against
    /// Σ |r|: a class whose group's even sum cancels has toggles near 1e100, which both sums
    /// carry to the last bit. Every complete kernel met with 13 to 16 generators has a plan, or
    /// needs more than the budget.
    #[test]
    fn class_sums_equal_every_product_k13_to_16() {
        let mut rng = Xorshift::new(0xbeef_2022);
        let mut found: BTreeMap<usize, usize> = BTreeMap::new();
        let (mut complete, mut refused, mut coupled, mut grouped, mut wide, mut worst) = (0, 0, 0, 0, 0, 0.0f64);
        let mut texts = Vec::new();
        for trial in 0..60_000 {
            if (13..=16).all(|k| found.get(&k).copied().unwrap_or(0) >= 8) {
                break;
            }
            let text = random_circuit_wide(&mut rng, 3 + trial % 4, 18 + trial % 12, 4);
            let (p, solver, kernel) = kernel(&text, CoherentOptions::default());
            let k = kernel.generators.len();
            if !(13..=16).contains(&k) || !kernel.complete {
                continue;
            }
            complete += 1;
            let Some(plan) = plan_within_budget(&p, &solver, &kernel, &text) else {
                refused += 1;
                continue;
            };
            if found.get(&k).copied().unwrap_or(0) >= 8 {
                continue;
            }
            *found.entry(k).or_insert(0) += 1;
            coupled += usize::from(!plan.couplings.is_empty());
            grouped += usize::from(!kernel.groups.is_empty());
            wide += usize::from(p.locations.len() > 20);
            let products = every_product(&p, &solver, &kernel.generators);
            let mut merged = p.clone();
            merged.set_merge(&kernel.merge_groups());
            for run in [&p, &merged] {
                for seed in 0..60u64 {
                    let mut r = Xorshift::new(seed * 7919 + 1);
                    let shot = run.run_shot(&mut super::super::shot::Stream(&mut r));
                    let (_, f) = kernel.shot_factors(&p, &shot);
                    let ((s, n), (bs, bn, l1)) = (plan.class_sums(&f), sums_of_every_product(&products, &p, &f, &shot));
                    let e = ((s - bs).norm2().sqrt() / l1).max((n - bn).abs() / bn);
                    assert!(e < 1e-11, "k = {k}, width {}: error {e:e}\n{text}", plan.width());
                    worst = worst.max(e);
                }
            }
            texts.push(text);
        }
        eprintln!("k 13 to 16: {found:?}; complete {complete}, refused {refused}; with couplings {coupled}, groups {grouped}, past 20 locations {wide}; worst {worst:e}");
        assert!((13..=16).all(|k| found.get(&k) == Some(&8)), "{found:?}");
        assert!(coupled >= 24 && grouped >= 16 && wide >= 4, "couplings {coupled}, groups {grouped}, past 20 locations {wide}");
        let has = |what: &dyn Fn(&str) -> bool| texts.iter().any(|t| t.lines().any(what));
        for gate in ["S_DAG ", "SQRT_X ", "CY ", "SWAP ", "RX ", "MRX ", "MY ", "II_ERROR[R_XX", "II_ERROR[R_YY", "II_ERROR[R_ZZ", "I_ERROR[R_Y("] {
            assert!(has(&|l: &str| l.starts_with(gate)), "no {gate}");
        }
        for targets in [2, 3] {
            assert!(has(&|l: &str| l.starts_with("I_ERROR[R_PAULI") && l.rsplit(']').next().unwrap().split_whitespace().count() == targets), "no {targets}-qubit R_PAULI");
        }
    }

    /// Plans are deterministic: two built for one kernel give the same class sums, bit for bit.
    #[test]
    fn plans_are_deterministic() {
        let text = crate::coherent::kernel::nonlocal::surface(3, 3, 0.25);
        let (p, _, k1) = kernel(&text, CoherentOptions::default());
        let (_, _, k2) = kernel(&text, CoherentOptions::default());
        for seed in 1..200u64 {
            let shot = p.run_shot(&mut super::super::shot::Stream(&mut Xorshift::new(seed)));
            let a = k1.plan().unwrap().class_sums(&k1.shot_factors(&p, &shot).1);
            let b = k2.plan().unwrap().class_sums(&k2.shot_factors(&p, &shot).1);
            assert_eq!((a.0.re.to_bits(), a.0.im.to_bits(), a.1.to_bits()), (b.0.re.to_bits(), b.0.im.to_bits(), b.1.to_bits()), "seed {seed}");
        }
    }
}
