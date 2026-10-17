//! The joint sampler: Clifford+T circuits with coherent errors on every gate. The non-Clifford
//! *gates* (`I[T]`, `I[R_Z(theta=…)]`, `I[U3(…)]`) are exact terms of the stabilizer-rank
//! engine; the coherent *errors* (`I_ERROR[R_Z(theta=…)]`, `II_ERROR[R_ZZ(…)]`) are drawn as
//! their Pauli twirl and the shot is reweighted, as the coherent sampler reweights it, by the
//! interference the twirl leaves out.
//!
//! Expanding the errors alone, the circuit is U = Σ_S a_S U_S over error sets S, where U_S has
//! the Pauli P_g inserted at each g ∈ S and keeps every gate, T gates included, and
//! a_S = Π_{g∈S}(−i sin θ_g) Π_{g∉S} cos θ_g. For a record m (and the unread outcomes u of
//! its resets) let |Ψ_S⟩ be the term engine's final state along that trajectory with S's
//! Paulis in its frame: P(m, u) = ‖Σ_S a_S |Ψ_S⟩‖² = Σ_{S,S′} a_S a*_{S′} ⟨Ψ_{S′}|Ψ_S⟩. A shot
//! draws S as the twirl (|a_S|²) and (m, u) from the exact terms (‖Ψ_S‖²), and its weight puts
//! back the cross terms: with ρ_G = a_{S⊕G}/a_S,
//!
//! ```text
//!     w = 1 + Σ_{G ≠ ∅} ρ_G ⟨Ψ_{S⊕G}|Ψ_S⟩ / ‖Ψ_S‖²,
//! ```
//!
//! each pair counted once, from its likelier side (so the weight stays real and the estimate
//! unbiased for every pair whose difference G is among the sets summed). Two error sets whose
//! states overlap differ by an element of the spacetime kernel projected onto the error
//! locations — every non-Clifford gate is a location for the kernel's search, so that the
//! interference running through the T gates is found — and the sets summed are those
//! projections: the kernel's generators each on its own, products of up to `order` of the
//! generators a fired error touches (per cluster), and products within each group of locations
//! with equal flips (the same fault in different places, a T gate's among them), as the
//! coherent sampler truncates; with the whole kernel, every projection, and the identity is
//! exact (`exact_distribution`, checked against the state vector). Unlike the coherent
//! sampler's class form, the sum is not a product over clusters: through a T gate, members of
//! different clusters overlap. Each member's state is a re-run of the term engine with the
//! member's Paulis inserted and the shot's outcomes forced, brought to the shot's frame so
//! that overlaps are dot products over shared keys.

use std::collections::{BTreeMap, HashMap};

use super::kernel::{CoherentOptions, Kernel};
use super::{Element, GaugeSolver, Program as KernelProgram};
use crate::circuit::Circuit;
use crate::stabrank::{Key, Memo, Odometer, Program, Recording, Replay, Run, Source, Stream};
use crate::statevec::C;
use crate::surface_code::Xorshift;

pub struct JointSampler {
    /// The term engine, the coherent errors as twirled locations.
    pub program: Program,
    /// The kernel's circuit: every rotation, exact gate or error, a location.
    pub kernel_program: KernelProgram,
    pub kernel: Kernel,
    solver: GaugeSolver,
    /// Each twirled location's index among the kernel's locations.
    loc_of_twirl: Vec<usize>,
    /// Each kernel location's twirled index, if it is an error.
    twirl_of_loc: Vec<Option<usize>>,
    /// The kernel's generators, and every pair within each group of equal-flip locations.
    generators: Vec<Element>,
    /// Each group's pair generators (indices into `generators`).
    group_gens: Vec<Vec<usize>>,
    by_location: Vec<Vec<usize>>,
    options: CoherentOptions,
    /// With the whole kernel (small circuits): every element, one class.
    whole: Option<Vec<Element>>,
}

impl JointSampler {
    pub fn new(circuit: &Circuit, options: CoherentOptions) -> Result<JointSampler, String> {
        let expanded = crate::stabrank::expand_clifford_rotations(circuit)?;
        let program = Program::with_twirl(&expanded)?;
        let kernel_program = KernelProgram::new(&expanded)?;
        let solver = GaugeSolver::new(&kernel_program);
        let kernel = Kernel::new(&kernel_program, &solver, options.clone());
        let mut loc_of_twirl = Vec::new();
        let mut twirl_of_loc = vec![None; kernel_program.locations.len()];
        for (l, &exact) in kernel_program.exact.iter().enumerate() {
            if !exact {
                twirl_of_loc[l] = Some(loc_of_twirl.len());
                loc_of_twirl.push(l);
            }
        }
        if loc_of_twirl.len() != program.num_twirl() {
            return Err("the kernel's error locations do not match the term engine's".into());
        }
        let mut generators = kernel.generators.clone();
        let mut group_gens = Vec::new();
        for g in &kernel.groups {
            let mut gens = Vec::new();
            for (i, &a) in g.members.iter().enumerate() {
                for &b in &g.members[i + 1..] {
                    if let Some(e) = kernel_program.element(&[a, b], &solver) {
                        gens.push(generators.len());
                        generators.push(e);
                    }
                }
            }
            group_gens.push(gens);
        }
        let mut by_location = vec![Vec::new(); kernel_program.locations.len()];
        for (i, g) in generators.iter().enumerate() {
            for &l in &g.members {
                by_location[l].push(i);
            }
        }
        Ok(JointSampler { program, kernel_program, kernel, solver, loc_of_twirl, twirl_of_loc, generators, group_gens, by_location, options, whole: None })
    }

    /// The same, weighing every shot with the whole kernel: for small circuits, the exact
    /// identity.
    pub fn with_whole_kernel(circuit: &Circuit) -> Result<JointSampler, String> {
        let mut s = JointSampler::new(circuit, CoherentOptions::default())?;
        s.whole = Some(s.kernel_program.full_kernel()?);
        Ok(s)
    }

    pub fn num_detectors(&self) -> usize {
        self.program.num_detectors()
    }

    pub fn num_observables(&self) -> usize {
        self.program.num_observables()
    }

    /// The coherent errors.
    pub fn num_locations(&self) -> usize {
        self.loc_of_twirl.len()
    }

    /// The non-Clifford gates, run exactly.
    pub fn num_exact(&self) -> usize {
        self.kernel_program.exact.iter().filter(|&&e| e).count()
    }

    pub fn num_generators(&self) -> usize {
        self.generators.len()
    }

    /// The error sets G summed for a shot whose errors `fired`: distinct, sorted lists of
    /// twirled locations, never empty. The projections of every untouched generator, of the
    /// products of up to `order` touched generators per cluster, and of the products within
    /// each group; with the whole kernel, of every element.
    fn members(&self, fired: &[bool], cache: &mut HashMap<u64, Option<Element>>) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = Vec::new();
        let add = |elements: Vec<Element>, out: &mut Vec<Vec<usize>>| {
            for e in elements {
                let g: Vec<usize> = e.members.iter().filter_map(|&l| self.twirl_of_loc[l]).collect();
                if !g.is_empty() && !out.contains(&g) {
                    out.push(g);
                }
            }
        };
        if let Some(all) = &self.whole {
            add(all.clone(), &mut out);
            return out;
        }
        let mut fired_locs: Vec<bool> = vec![false; self.kernel_program.locations.len()];
        for (t, &f) in fired.iter().enumerate() {
            if f {
                fired_locs[self.loc_of_twirl[t]] = true;
            }
        }
        let touched = self.kernel.touched(&self.generators, &self.by_location, &fired_locs, self.options.hops);
        let mut t = touched.iter().peekable();
        for (g, e) in self.generators.iter().enumerate() {
            if t.peek() == Some(&&g) {
                t.next();
                continue;
            }
            add(vec![e.clone()], &mut out);
        }
        let mut element_of = |members: &[usize]| self.kernel_program.element(members, &self.solver);
        for cluster in self.kernel.clusters(&self.generators, &touched) {
            let elements = self.kernel.products(&self.generators, &cluster, &self.options, cache, &mut element_of);
            add(elements, &mut out);
        }
        for gens in &self.group_gens {
            if gens.len() > 1 {
                let elements = self.kernel.products(&self.generators, gens, &self.options, cache, &mut element_of);
                add(elements, &mut out);
            }
        }
        out
    }

    /// a_{S⊕G} / a_S for the error set G: −i tan θ to add a fault, i / tan θ to take one away.
    fn rho(&self, fired: &[bool], g: &[usize]) -> C {
        g.iter().fold(C::ONE, |acc, &t| {
            let theta = self.kernel_program.locations[self.loc_of_twirl[t]].theta;
            let v = if fired[t] { C::new(0.0, 1.0 / theta.tan()) } else { C::new(0.0, -theta.tan()) };
            acc * v
        })
    }

    /// A member's final state in the shot's frame, as its terms (keys to amplitudes, the
    /// trajectory's probability in their norm), from a re-run with the shot's draws replayed
    /// and the member's errors inserted. Zero (empty) where the shot's outcomes are impossible
    /// for it.
    fn member(&self, base: &Run, fired: &[bool], memo: &mut Memo) -> HashMap<Key, C> {
        let mut src = Replay::new(&base.trace, fired);
        let run = self.program.run_full(&mut src, memo);
        if run.dead {
            return HashMap::new();
        }
        self.program.in_frame(&run, &base.frame)
    }

    /// One shot: its records, and its weight.
    pub(crate) fn shot(&self, src: &mut dyn Source, memo: &mut Memo, cache: &mut HashMap<u64, Option<Element>>) -> (Vec<bool>, f64) {
        let mut rec = Recording::new(src, self.program.num_twirl());
        let base = self.program.run_full(&mut rec, memo);
        let fired = rec.fired.clone();
        let mut base_run = base;
        base_run.trace = rec.into_trace();
        let base_terms = self.program.in_frame(&base_run, &base_run.frame);
        let base_norm: f64 = base_terms.values().map(|c| c.norm2()).sum();
        if base_norm <= 0.0 {
            return (base_run.records, 1.0);
        }
        let mut w = 1.0;
        for g in self.members(&fired, cache) {
            let mut member_fired = fired.clone();
            for &t in &g {
                member_fired[t] = !member_fired[t];
            }
            let rho = self.rho(&fired, &g);
            let terms = self.member(&base_run, &member_fired, memo);
            if terms.is_empty() {
                continue;
            }
            // The pair (S, S ⊕ G) counted from its likelier side, both of its orders at once
            // (2 Re): the twirl's probability of the member relative to the shot's is
            // |ρ|² ‖Ψ_member‖² / ‖Ψ_S‖²; a near tie goes to the smaller firing pattern, which
            // both sides agree on.
            let member_norm: f64 = terms.values().map(|c| c.norm2()).sum();
            let relative = rho.norm2() * member_norm / base_norm;
            let likelier = if (relative - 1.0).abs() < 1e-9 { fired < member_fired } else { relative < 1.0 };
            if !likelier {
                continue;
            }
            let overlap = terms.iter().fold(C::ZERO, |acc, (k, c)| acc + c.conj() * base_terms.get(k).copied().unwrap_or(C::ZERO));
            // a_S a*_{S⊕G} ⟨Ψ_{S⊕G}|Ψ_S⟩ over |a_S|² ‖Ψ_S‖² is ρ* times the overlap.
            w += 2.0 * (rho.conj() * overlap).re / base_norm;
        }
        (base_run.records, w)
    }

    /// Shots `first..first + shots`, each from its own stream: detection events and observable
    /// flips as b8 rows, and each shot's weight.
    pub fn sample_seeded(&self, seed: u64, first: u64, shots: usize, threads: usize) -> super::sampler::CoherentBatch {
        let p = &self.program;
        let (nd, no) = (p.num_detectors(), p.num_observables());
        let (dw, ow) = (nd.div_ceil(8), no.div_ceil(8));
        let rows = crate::parallel::parallel(shots, threads, |range| {
            let mut memo = Memo::default();
            let mut cache = HashMap::new();
            range
                .map(|s| {
                    let mut rng = Xorshift::new(crate::batch_sampler::batch_seed(seed, first + s as u64));
                    let (records, w) = self.shot(&mut Stream(&mut rng), &mut memo, &mut cache);
                    (p.outcome(&records), w)
                })
                .collect()
        });
        let mut batch = super::sampler::CoherentBatch { detectors: vec![0u8; shots * dw], observables: vec![0u8; shots * ow], weights: Vec::with_capacity(shots) };
        for (s, ((dets, obs), w)) in rows.into_iter().enumerate() {
            for (k, &b) in dets.iter().enumerate() {
                if b {
                    batch.detectors[s * dw + k / 8] |= 1 << (k % 8);
                }
            }
            for k in 0..no {
                if obs >> k & 1 == 1 {
                    batch.observables[s * ow + k / 8] |= 1 << (k % 8);
                }
            }
            batch.weights.push(w);
        }
        batch
    }
}

impl JointSampler {
    /// Every outcome's probability by the weighted twirl, summed over every branch (noise
    /// draws, firings, outcomes): with the whole kernel, the exact distribution; with the
    /// truncated one, its bias made visible. At most `max_branches`.
    pub fn exact_distribution(&self, max_branches: usize) -> Result<BTreeMap<(Vec<bool>, u64), f64>, String> {
        let mut out = BTreeMap::new();
        let mut memo = Memo::default();
        let mut cache = HashMap::new();
        let mut odo = Odometer::default();
        let mut branches = 0usize;
        loop {
            odo.restart();
            let (records, w) = self.shot(&mut odo, &mut memo, &mut cache);
            if odo.p > 1e-15 {
                *out.entry(self.program.outcome(&records)).or_insert(0.0) += odo.p * w;
            }
            branches += 1;
            if branches > max_branches {
                return Err(format!("more than {max_branches} branches"));
            }
            if !odo.advance() {
                return Ok(out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truth(c: &Circuit) -> BTreeMap<(Vec<bool>, u64), f64> {
        crate::statevec::Program::new(c).unwrap().distribution(1 << 22).unwrap()
    }

    fn close(a: &BTreeMap<(Vec<bool>, u64), f64>, b: &BTreeMap<(Vec<bool>, u64), f64>, text: &str) {
        let keys: std::collections::BTreeSet<_> = a.keys().chain(b.keys()).collect();
        for k in keys {
            let (x, y) = (a.get(k).copied().unwrap_or(0.0), b.get(k).copied().unwrap_or(0.0));
            assert!((x - y).abs() < 1e-9, "{text}\n{k:?}: {x} vs {y}");
        }
    }

    fn check(text: &str) {
        let c = Circuit::parse(text).unwrap();
        let s = JointSampler::with_whole_kernel(&c).unwrap();
        close(&s.exact_distribution(1 << 22).unwrap(), &truth(&c), text);
    }

    #[test]
    fn one_error_before_a_t_gate() {
        check("R 0\nH 0\nI_ERROR[R_X(theta=0.3)] 0\nI[T] 0\nH 0\nM 0\nDETECTOR rec[-1]");
        check("R 0\nH 0\nI_ERROR[R_Z(theta=0.3)] 0\nI[T] 0\nI_ERROR[R_X(theta=0.2)] 0\nH 0\nM 0\nDETECTOR rec[-1]");
    }

    #[test]
    fn errors_either_side_of_t_gates_on_two_qubits() {
        check("R 0 1\nH 0\nI_ERROR[R_Y(theta=0.25)] 0\nCX 0 1\nI[T] 1\nII_ERROR[R_ZZ(theta=0.2)] 0 1\nI[T_DAG] 0\nH 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]");
    }

    #[test]
    fn with_noise_resets_and_mid_circuit_measurements() {
        check("R 0 1\nH 0\nI_ERROR[R_X(theta=0.3)] 0\nDEPOLARIZE1(0.1) 1\nCX 0 1\nI[T] 0\nMR 1\nI_ERROR[R_Z(theta=0.2)] 1\nCX 0 1\nI[U3(theta=0.5, phi=0.2, lambda=1.1)] 1\nH 0\nM(0.05) 0 1\nDETECTOR rec[-1] rec[-3]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]");
    }

    #[test]
    fn random_circuits() {
        let mut rng = Xorshift::new(99);
        for case in 0..40 {
            let n = 1 + case % 3;
            let text = crate::stabrank::tests::random_circuit(&mut rng, n, 5 + case % 5, true);
            let c = Circuit::parse(&text).unwrap();
            // The kernel's program takes no feedback; the whole kernel is for a few locations.
            let Ok(p) = KernelProgram::new(&c) else { continue };
            if p.locations.len() > 12 {
                continue;
            }
            check(&text);
        }
    }

    #[test]
    fn without_gates_the_weights_are_the_coherent_samplers() {
        // No non-Clifford gate: every member is Pauli-related and the class weight is 1.9's.
        let text = "R 0 1\nH 0\nI_ERROR[R_Z(theta=0.3)] 0\nCX 0 1\nI_ERROR[R_X(theta=0.2)] 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]";
        check(text);
        let c = Circuit::parse(text).unwrap();
        let s = JointSampler::new(&c, CoherentOptions::default()).unwrap();
        assert_eq!(s.num_exact(), 0);
        let b = s.sample_seeded(1, 0, 200, 1);
        assert_eq!(b.weights.len(), 200);
        assert!(b.weights.iter().all(|w| w.is_finite() && *w >= 0.0));
    }
}
