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
    /// Per edge, its ends: indices into `support`.
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
        let edges: Vec<usize> = (0..hz.rows).filter(|&c| hz.row_ones(c).iter().any(|q| support.contains(q))).collect();
        let incidence: Vec<Vec<usize>> = edges
            .iter()
            .map(|&c| hz.row_ones(c).iter().filter_map(|q| support.iter().position(|v| v == q)).collect())
            .collect();
        if let Some(i) = incidence.iter().position(|e| e.len() % 2 == 1) {
            return Err(format!("Z check {} meets the operator in an odd number of qubits: it is not a logical X", edges[i]));
        }
        let gauss = (0..support.len()).map(|v| (0..edges.len()).filter(|&e| incidence[e].contains(&v)).collect()).collect();
        let mut g = Gauging { support, edges, incidence, gauss, flux: Vec::new() };
        g.flux = g.min_weight_cycles()?;
        Ok(g)
    }

    /// F: the Z checks of C₀ restricted to V₀, edges × vertices.
    pub fn restricted(&self) -> BitMatrix {
        BitMatrix::from_rows(self.support.len(), &self.incidence)
    }

    /// A minimum-weight basis of the cycle space {s : sᵀF = 0}: every vector
    /// of the space, lightest first, kept when independent of those kept.
    /// The greedy basis of a linear matroid is a minimum-weight one.
    fn min_weight_cycles(&self) -> Result<Vec<Vec<usize>>, String> {
        let space = self.restricted().transpose().kernel();
        if space.rows > 16 {
            return Err(format!("{} independent cycles are too many to enumerate", space.rows));
        }
        let ne = self.edges.len();
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
        let cols = n + self.edges.len();
        let mut x: Vec<Vec<usize>> = (0..hx.rows).map(|r| hx.row_ones(r)).collect();
        for (v, es) in self.gauss.iter().enumerate() {
            x.push(std::iter::once(self.support[v]).chain(es.iter().map(|e| n + e)).collect());
        }
        let mut z: Vec<Vec<usize>> = (0..hz.rows).map(|r| hz.row_ones(r)).collect();
        for (i, &c) in self.edges.iter().enumerate() {
            z[c].push(n + i);
        }
        z.extend(self.flux.iter().map(|cycle| cycle.iter().map(|e| n + e).collect()));
        (BitMatrix::from_rows(cols, &x), BitMatrix::from_rows(cols, &z))
    }

    /// Edges whose ends, counted mod 2, are exactly `z`'s qubits in V₀: a
    /// solution s of Fᵀs = z|V₀. A Z operator on the data that commutes with
    /// the operator passes through the merge as itself times Z on these
    /// edges, which then commutes with every Gauss-law check.
    pub fn edges_for(&self, z: &[usize]) -> Result<Vec<usize>, String> {
        let ne = self.edges.len();
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
            return Err("no set of edges has that boundary: the operator does not commute with it".into());
        }
        let mut s: Vec<usize> = pivots.iter().enumerate().filter(|&(r, _)| m.get(r, ne)).map(|(_, &p)| p).collect();
        s.sort_unstable();
        Ok(s)
    }

    /// Ancilla qubits added: one per edge, per Gauss-law check, per flux check.
    pub fn ancillas(&self) -> usize {
        self.edges.len() + self.support.len() + self.flux.len()
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
            assert!(hx.mul(&hz.transpose()).is_zero(), "{name}: the checks commute");
            let n = hx.cols;
            assert_eq!(n - hx.rank() - hz.rank(), 11, "{name}: 11 logical qubits while merged");
            let mut product = vec![false; n];
            for r in code.half()..code.half() + g.support.len() {
                for q in hx.row_ones(r) {
                    product[q] ^= true;
                }
            }
            let got: Vec<usize> = (0..n).filter(|&q| product[q]).collect();
            assert_eq!(got, l, "{name}: the Gauss-law checks multiply to L");
            assert_eq!(g.flux.len(), g.edges.len() - g.restricted().rank(), "{name}: a flux check per independent cycle");
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
        assert!(g.gauss.iter().all(|es| es.len() == 3), "every vertex of X(f, 0)'s graph has degree 3");
    }

    /// The flux basis is the lightest one, lightest first.
    #[test]
    fn the_flux_basis_is_light() {
        let (_, g, _) = gauged("f");
        let w: Vec<usize> = g.flux.iter().map(Vec::len).collect();
        eprintln!("flux weights, f: {w:?}");
        assert!(w.windows(2).all(|p| p[0] <= p[1]), "sorted by weight: {w:?}");
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
            assert_eq!(got.is_ok(), overlap % 2 == 0, "Z logical {j}: overlap {overlap}");
            if let Ok(s) = got {
                // Each vertex meets the path as often as z meets it, mod 2.
                for (v, es) in g.gauss.iter().enumerate() {
                    let meets = es.iter().filter(|e| s.contains(e)).count() % 2 == 1;
                    assert_eq!(meets, z.contains(&g.support[v]), "Z logical {j}, vertex {v}");
                }
            }
        }
    }
}
