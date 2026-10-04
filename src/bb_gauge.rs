//! The gauging measurement of a logical operator on a bivariate bicycle code
//! (Williamson and Yoder, arXiv:2410.02213), mono-layer, as Cross, He, Rall
//! and Yoder build it for the gross code (arXiv:2407.18393, Section 3).
//!
//! WHY THIS EXISTS
//! ---------------
//! Automorphisms move logical information around; a computer also has to
//! measure it. To measure a logical X̄ with support V₀: every Z check that
//! touches V₀ (the set C₀) meets it in an even number of qubits, so C₀ is a
//! hypergraph on V₀, a graph where every check meets it in two. One new
//! qubit per edge, prepared in |0⟩. One Gauss-law X check per vertex: its
//! data qubit and its edges' qubits, so their product is X̄ (every edge has
//! an even number of ends). Each Z check of C₀ gains its own edge qubit and
//! so commutes with them. Flux Z checks, one per independent cycle (a set of
//! edges with no boundary), fix the new qubits' gauge. Measuring the
//! Gauss-law checks for a few cycles measures X̄, and every original check
//! goes on being measured. Cross et al. give no edge lists, but the
//! construction is an algorithm, so it is built here and checked.

use crate::bb::BbCode;
use crate::gf2::BitMatrix;

/// The ancilla system that measures one logical X operator.
#[derive(Clone, Debug)]
pub struct Gauging {
    /// V₀: the operator's data qubits, sorted.
    pub support: Vec<usize>,
    /// C₀: the Z checks touching V₀, sorted. Edge qubit i belongs to check `edges[i]`.
    pub edges: Vec<usize>,
    /// Edges added for expansion, between two vertices (indices into
    /// `support`), belonging to no check: edge qubits `edges.len()` onwards.
    pub extra: Vec<(usize, usize)>,
    /// Per edge (the checks' then the added ones), its ends: indices into `support`.
    pub incidence: Vec<Vec<usize>>,
    /// Per vertex, its edges: indices into `edges`.
    pub gauss: Vec<Vec<usize>>,
    /// The flux checks: a minimum-weight basis of the cycles, as edge indices, lightest first.
    pub flux: Vec<Vec<usize>>,
}

impl Gauging {
    pub fn new(code: &BbCode, support: &[usize]) -> Result<Gauging, String> {
        let hz = code.hz();
        let mut support = support.to_vec();
        support.sort_unstable();
        support.dedup();
        let edges: Vec<usize> = (0..hz.rows)
            .filter(|&c| hz.row_ones(c).iter().any(|q| support.contains(q)))
            .collect();
        let incidence: Vec<Vec<usize>> = edges
            .iter()
            .map(|&c| {
                hz.row_ones(c)
                    .iter()
                    .filter_map(|q| support.iter().position(|v| v == q))
                    .collect()
            })
            .collect();
        if let Some(i) = incidence.iter().position(|e| e.len() % 2 == 1) {
            return Err(format!(
                "Z check {} meets the operator in an odd number of qubits: it is not a logical X",
                edges[i]
            ));
        }
        let mut g = Gauging {
            support,
            edges,
            extra: Vec::new(),
            incidence,
            gauss: Vec::new(),
            flux: Vec::new(),
        };
        g.finish()?;
        Ok(g)
    }

    /// The minimal system plus edges, added until every set U of at most half
    /// the vertices has |δU| ≥ |U| (a Cheeger constant of at least 1,
    /// Williamson and Yoder's condition for the merged code to keep the
    /// code's distance). Each round takes the worst set and joins a vertex in
    /// it to one outside, the pair of least total degree not already joined.
    pub fn expanded(code: &BbCode, support: &[usize]) -> Result<Gauging, String> {
        let mut g = Gauging::new(code, support)?;
        for _ in 0..64 {
            let (b, u) = g.cheeger();
            if b >= u.len() {
                g.finish()?;
                return Ok(g);
            }
            let degree = |v: usize| g.gauss[v].len();
            let joined = |x: usize, y: usize| {
                g.incidence
                    .iter()
                    .any(|e| e.len() == 2 && e.contains(&x) && e.contains(&y))
            };
            let outside: Vec<usize> = (0..g.support.len()).filter(|v| !u.contains(v)).collect();
            let pair = u
                .iter()
                .flat_map(|&x| outside.iter().map(move |&y| (x, y)))
                .filter(|&(x, y)| !joined(x, y))
                .min_by_key(|&(x, y)| (degree(x) + degree(y), x, y))
                .ok_or("no pair left to join across the worst cut")?;
            g.extra.push(pair);
            g.incidence.push(vec![pair.0, pair.1]);
            g.gauss = g.vertex_edges();
        }
        Err("64 added edges did not reach a Cheeger constant of 1".into())
    }

    /// Per-vertex edges and the flux checks, from `incidence`.
    fn finish(&mut self) -> Result<(), String> {
        self.gauss = self.vertex_edges();
        self.flux = self.min_weight_cycles()?;
        Ok(())
    }

    fn vertex_edges(&self) -> Vec<Vec<usize>> {
        (0..self.support.len())
            .map(|v| {
                (0..self.num_edges())
                    .filter(|&e| self.incidence[e].contains(&v))
                    .collect()
            })
            .collect()
    }

    /// Edge qubits: one per check touching the operator, one per added edge.
    pub fn num_edges(&self) -> usize {
        self.incidence.len()
    }

    /// The worst cut, exactly: over every set U of at most half the vertices,
    /// the one minimising |δU| / |U| (the smallest such U on ties), where δU
    /// is the edges with an odd number of ends in U. For a graph that is the
    /// edges leaving U, and |δU| ≥ |U| throughout is Williamson and Yoder's
    /// Cheeger condition; for hyperedges (the product operator's) it is its
    /// generalisation, the count that decides how a logical's weight changes. The Gauss-law checks on
    /// U multiply to X on U and on δU, so a logical containing U can trade
    /// |U| data qubits for |δU| edge qubits. Returns (|δU|, U).
    pub fn cheeger(&self) -> (usize, Vec<usize>) {
        let n = self.support.len();
        assert!(n <= 26, "{n} vertices are too many to enumerate");
        let masks: Vec<u32> = self
            .incidence
            .iter()
            .map(|e| e.iter().fold(0u32, |m, &v| m | 1 << v))
            .collect();
        let (mut best_b, mut best_u) = (usize::MAX, 0u32);
        for u in 1u32..(1 << n) {
            let k = u.count_ones() as usize;
            if 2 * k > n {
                continue;
            }
            let b = masks
                .iter()
                .filter(|&&m| (m & u).count_ones() % 2 == 1)
                .count();
            let (bk, kk) = (
                best_b.saturating_mul(k),
                b * best_u.count_ones().max(1) as usize,
            );
            if best_b == usize::MAX || kk < bk || (kk == bk && k < best_u.count_ones() as usize) {
                best_b = b;
                best_u = u;
            }
        }
        (best_b, (0..n).filter(|&v| best_u >> v & 1 == 1).collect())
    }

    /// F: the Z checks of C₀ restricted to V₀, edges × vertices.
    pub fn restricted(&self) -> BitMatrix {
        BitMatrix::from_rows(self.support.len(), &self.incidence)
    }

    /// A light basis of the cycle space {s : sᵀF = 0}, lightest first. Up to
    /// 2²⁰ vectors, every vector of the space, lightest first, kept when
    /// independent of those kept: the greedy basis of a linear matroid is a
    /// minimum-weight one. Beyond that, a basis reduced greedily (a vector is
    /// replaced by its sum with another while that is lighter), which is
    /// light but not proven minimal.
    fn min_weight_cycles(&self) -> Result<Vec<Vec<usize>>, String> {
        let space = self.restricted().transpose().kernel();
        let ne = self.num_edges();
        if space.rows > 20 {
            let mut basis: Vec<Vec<bool>> = (0..space.rows)
                .map(|r| (0..ne).map(|e| space.get(r, e)).collect())
                .collect();
            let weight = |v: &[bool]| v.iter().filter(|&&x| x).count();
            loop {
                let mut improved = false;
                for i in 0..basis.len() {
                    for j in 0..basis.len() {
                        if i == j {
                            continue;
                        }
                        let sum: Vec<bool> =
                            basis[i].iter().zip(&basis[j]).map(|(a, b)| a ^ b).collect();
                        if weight(&sum) < weight(&basis[i]) {
                            basis[i] = sum;
                            improved = true;
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
            let mut out: Vec<Vec<usize>> = basis
                .iter()
                .map(|v| (0..ne).filter(|&e| v[e]).collect())
                .collect();
            out.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
            return Ok(out);
        }
        let mut all: Vec<Vec<usize>> = (1u32..(1 << space.rows))
            .map(|mask| {
                let mut on = vec![false; ne];
                for r in (0..space.rows).filter(|r| mask >> r & 1 == 1) {
                    for e in space.row_ones(r) {
                        on[e] ^= true;
                    }
                }
                (0..ne).filter(|&e| on[e]).collect()
            })
            .collect();
        all.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        let mut kept: Vec<Vec<usize>> = Vec::new();
        for cycle in all {
            if kept.len() == space.rows {
                break;
            }
            kept.push(cycle);
            if BitMatrix::from_rows(ne, &kept).rank() < kept.len() {
                kept.pop();
            }
        }
        Ok(kept)
    }

    /// The deformed code (H_X', H_Z') on the data and then the edge qubits
    /// (edge qubit i is data index num_data + i). H_X': the code's X checks,
    /// then a Gauss-law check per vertex. H_Z': the code's Z checks, those of
    /// C₀ with their edge qubit, then the flux checks.
    pub fn deformed(&self, code: &BbCode) -> (BitMatrix, BitMatrix) {
        let (hx, hz, n) = (code.hx(), code.hz(), code.num_data());
        let cols = n + self.num_edges();
        let mut x: Vec<Vec<usize>> = (0..hx.rows).map(|r| hx.row_ones(r)).collect();
        for (v, es) in self.gauss.iter().enumerate() {
            x.push(
                std::iter::once(self.support[v])
                    .chain(es.iter().map(|e| n + e))
                    .collect(),
            );
        }
        let mut z: Vec<Vec<usize>> = (0..hz.rows).map(|r| hz.row_ones(r)).collect();
        for (i, &c) in self.edges.iter().enumerate() {
            z[c].push(n + i);
        }
        z.extend(
            self.flux
                .iter()
                .map(|cycle| cycle.iter().map(|e| n + e).collect()),
        );
        (
            BitMatrix::from_rows(cols, &x),
            BitMatrix::from_rows(cols, &z),
        )
    }

    /// Edges whose ends, counted mod 2, are exactly `z`'s qubits in V₀: a
    /// solution s of Fᵀs = z|V₀. A Z operator on the data that commutes with
    /// the operator passes through the merge as itself times Z on these
    /// edges, which then commutes with every Gauss-law check.
    pub fn edges_for(&self, z: &[usize]) -> Result<Vec<usize>, String> {
        let ne = self.num_edges();
        let rows: Vec<Vec<usize>> = (0..self.support.len())
            .map(|v| {
                let mut row = self.gauss[v].clone();
                if z.contains(&self.support[v]) {
                    row.push(ne);
                }
                row
            })
            .collect();
        let mut m = BitMatrix::from_rows(ne + 1, &rows);
        let pivots = m.row_reduce();
        if pivots.contains(&ne) {
            return Err(
                "no set of edges has that boundary: the operator does not commute with it".into(),
            );
        }
        let mut s: Vec<usize> = pivots
            .iter()
            .enumerate()
            .filter(|&(r, _)| m.get(r, ne))
            .map(|(_, &p)| p)
            .collect();
        s.sort_unstable();
        Ok(s)
    }

    /// Ancilla qubits added: one per edge, per Gauss-law check, per flux check.
    pub fn ancillas(&self) -> usize {
        self.num_edges() + self.support.len() + self.flux.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bb::gross_operator;

    fn gauged(name: &str) -> (BbCode, Gauging, Vec<usize>) {
        let code = BbCode::gross();
        let l = gross_operator(name).unwrap();
        let g = Gauging::new(&code, &l).unwrap();
        (code, g, l)
    }

    /// The deformed code's checks commute, the Gauss-law checks multiply to
    /// the operator (each edge qubit an even number of times), the flux
    /// checks span the cycles, and it encodes 11 logical qubits: L is now a
    /// stabilizer.
    #[test]
    fn the_deformed_code_measures_the_operator() {
        for name in ["f", "gh", "f+gh"] {
            let (code, g, l) = gauged(name);
            let (hx, hz) = g.deformed(&code);
            assert!(
                hx.mul(&hz.transpose()).is_zero(),
                "{name}: the checks commute"
            );
            let n = hx.cols;
            assert_eq!(
                n - hx.rank() - hz.rank(),
                11,
                "{name}: 11 logical qubits while merged"
            );
            let mut product = vec![false; n];
            for r in code.half()..code.half() + g.support.len() {
                for q in hx.row_ones(r) {
                    product[q] ^= true;
                }
            }
            let got: Vec<usize> = (0..n).filter(|&q| product[q]).collect();
            assert_eq!(got, l, "{name}: the Gauss-law checks multiply to L");
            assert_eq!(
                g.flux.len(),
                g.num_edges() - g.restricted().rank(),
                "{name}: a flux check per independent cycle"
            );
        }
    }

    /// X(f, 0) and X(g, h) each give a 3-regular graph on 12 vertices: 18
    /// edges, 7 independent cycles, 37 ancilla qubits. The product's support
    /// has 22 qubits and some checks meet it in four: a hypergraph.
    #[test]
    fn the_ancilla_systems_have_the_expected_sizes() {
        for (name, edges, flux) in [("f", 18, 7), ("gh", 18, 7), ("f+gh", 31, 10)] {
            let (_, g, _) = gauged(name);
            assert_eq!((g.edges.len(), g.flux.len()), (edges, flux), "{name}");
            assert_eq!(g.ancillas(), edges + g.support.len() + flux);
        }
        let (_, g, _) = gauged("f");
        assert!(
            g.gauss.iter().all(|es| es.len() == 3),
            "every vertex of X(f, 0)'s graph has degree 3"
        );
    }

    /// X(f, 0)'s graph is two clusters of six joined by two edges: its
    /// Cheeger constant is 1/3, and multiplying by the Gauss-law checks on one
    /// cluster trades six data qubits for two edge qubits, which is how the
    /// merged code's X distance falls to 8. X(g, h)'s is 2/3. The expanded
    /// systems add edges until every set of at most half the vertices has at
    /// least as many boundary edges as vertices (Williamson and Yoder's
    /// condition), and remain valid gaugings.
    #[test]
    fn expansion_brings_the_cheeger_constant_to_one() {
        let code = BbCode::gross();
        for (name, minimal) in [("f", (2, 6)), ("gh", (4, 6))] {
            let l = gross_operator(name).unwrap();
            let g = Gauging::new(&code, &l).unwrap();
            let (b, u) = g.cheeger();
            assert_eq!(
                (b, u.len()),
                minimal,
                "{name}: the worst cut of the minimal system"
            );
            let x = Gauging::expanded(&code, &l).unwrap();
            let (b, u) = x.cheeger();
            eprintln!(
                "{name}: {} extra edges, worst cut {b}/{}",
                x.extra.len(),
                u.len()
            );
            assert!(
                b >= u.len() && !x.extra.is_empty(),
                "{name}: expanded to a Cheeger constant of at least 1"
            );
            let (hx, hz) = x.deformed(&code);
            assert!(
                hx.mul(&hz.transpose()).is_zero(),
                "{name}: the expanded checks commute"
            );
            assert_eq!(
                hx.cols - hx.rank() - hz.rank(),
                11,
                "{name}: 11 logical qubits while merged"
            );
            assert_eq!(x.ancillas(), x.num_edges() + x.support.len() + x.flux.len());
            assert_eq!(
                x.flux.len(),
                x.num_edges() - x.restricted().rank(),
                "{name}: a flux check per independent cycle"
            );
        }
    }

    /// The flux basis is the lightest one, lightest first.
    #[test]
    fn the_flux_basis_is_light() {
        let (_, g, _) = gauged("f");
        let w: Vec<usize> = g.flux.iter().map(Vec::len).collect();
        eprintln!("flux weights, f: {w:?}");
        assert!(
            w.windows(2).all(|p| p[0] <= p[1]),
            "sorted by weight: {w:?}"
        );
        assert!(*w.last().unwrap() <= 8, "cycles of length at most 8: {w:?}");
    }

    /// A Z logical passes through the merge along an edge path whose boundary
    /// is its overlap with L's support, which exists exactly when it
    /// commutes with L.
    #[test]
    fn z_logicals_route_through_the_edges() {
        let (code, g, l) = gauged("f");
        let (_, lz) = code.logicals();
        for j in 0..lz.rows {
            let z = lz.row_ones(j);
            let overlap = z.iter().filter(|q| l.contains(q)).count();
            let got = g.edges_for(&z);
            assert_eq!(
                got.is_ok(),
                overlap % 2 == 0,
                "Z logical {j}: overlap {overlap}"
            );
            if let Ok(s) = got {
                // Each vertex meets the path as often as z meets it, mod 2.
                for (v, es) in g.gauss.iter().enumerate() {
                    let meets = es.iter().filter(|e| s.contains(e)).count() % 2 == 1;
                    assert_eq!(
                        meets,
                        z.contains(&g.support[v]),
                        "Z logical {j}, vertex {v}"
                    );
                }
            }
        }
    }
}
