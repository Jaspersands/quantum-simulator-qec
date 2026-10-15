// Ported from Fusion Blossom 0.2.13 (Yue Wu; MIT; see src/fusion/LICENSE and
// THIRD_PARTY_NOTICES.md): its default features, its dependencies replaced by src/fusion/shim.rs.

//! Fusion Blossom: exact minimum-weight perfect matching by a primal-dual blossom algorithm
//! on the decoding graph itself, serial or partitioned into units solved on threads and
//! fused along their boundaries (Wu and Zhong, arXiv 2305.08307). Python's
//! `stabilizer_qec.fusion_blossom` is its Python API.

#![allow(clippy::all, dead_code, unused_imports, unused_variables, unused_mut, unused_macros)]

#[macro_use]
pub(crate) mod shim;

pub(crate) use shim::{parking_lot, priority_queue, rand_xoshiro, rayon, serde_json, weak_table};

mod json_impls;
pub mod complete_graph;
pub mod dual_module;
pub mod dual_module_parallel;
pub mod dual_module_serial;
pub mod example_codes;
pub mod example_partition;
pub mod mwpm_solver;
pub mod pointers;
pub mod primal_module;
pub mod primal_module_parallel;
pub mod primal_module_serial;
pub mod util;
pub mod visualize;

/// Blossom V's interface, served by this crate's exact blossom (`crate::blossom`): Fusion
/// Blossom checks itself against Blossom V and offers it as a solver.
pub mod blossom_v {
    pub fn safe_minimum_weight_perfect_matching(node_num: usize, weighted_edges: &[(usize, usize, u32)]) -> Vec<usize> {
        let edges: Vec<(usize, usize, i64)> = weighted_edges.iter().map(|&(i, j, w)| (i, j, w as i64)).collect();
        crate::blossom::mwpm_on_edges(node_num, &edges).expect("a perfect matching")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn blossom_v_test_1() {
            // cargo test blossom_v_test_1 -- --nocapture
            let node_num = 4;
            let edges: Vec<(usize, usize, u32)> = vec![(0, 1, 100), (2, 3, 110), (0, 2, 500), (1, 3, 300)];
            let output = safe_minimum_weight_perfect_matching(node_num, &edges);
            assert_eq!(output, vec![1, 0, 3, 2]);
        }
    }
}

use complete_graph::*;
use util::*;

#[allow(clippy::unnecessary_cast)]
pub fn fusion_mwpm(initializer: &SolverInitializer, syndrome_pattern: &SyndromePattern) -> Vec<VertexIndex> {
    // sanity check
    assert!(initializer.vertex_num > 1, "at least one vertex required");
    let max_safe_weight = ((Weight::MAX as usize) / initializer.vertex_num as usize) as Weight;
    for (i, j, weight) in initializer.weighted_edges.iter() {
        if weight > &max_safe_weight {
            panic!(
                "edge {}-{} has weight {} > max safe weight {}, it may cause fusion blossom to overflow",
                i, j, weight, max_safe_weight
            );
        }
    }
    // by default use serial implementation fusion blossom
    mwpm_solver::LegacySolverSerial::mwpm_solve(initializer, syndrome_pattern)
}

/// fall back to use blossom V library to solve MWPM (install blossom V required)
#[allow(clippy::unnecessary_cast)]
pub fn blossom_v_mwpm(initializer: &SolverInitializer, defect_vertices: &[VertexIndex]) -> Vec<VertexIndex> {
    // this feature will be automatically enabled if you install blossom V source code, see README.md for more information
    if false {
        panic!("need blossom V library, see README.md")
    }
    // sanity check
    assert!(initializer.vertex_num > 1, "at least one vertex required");
    let max_safe_weight = ((i32::MAX as usize) / initializer.vertex_num as usize) as Weight;
    for (i, j, weight) in initializer.weighted_edges.iter() {
        if weight > &max_safe_weight {
            panic!(
                "edge {}-{} has weight {} > max safe weight {}, it may cause blossom V library to overflow",
                i, j, weight, max_safe_weight
            );
        }
    }
    let mut complete_graph = CompleteGraph::new(initializer.vertex_num, &initializer.weighted_edges);
    blossom_v_mwpm_reuse(&mut complete_graph, initializer, defect_vertices)
}

#[allow(clippy::unnecessary_cast)]
pub fn blossom_v_mwpm_reuse(
    complete_graph: &mut CompleteGraph,
    initializer: &SolverInitializer,
    defect_vertices: &[VertexIndex],
) -> Vec<VertexIndex> {
    // first collect virtual vertices and real vertices
    let mut is_virtual: Vec<bool> = (0..initializer.vertex_num).map(|_| false).collect();
    let mut is_defect: Vec<bool> = (0..initializer.vertex_num).map(|_| false).collect();
    for &virtual_vertex in initializer.virtual_vertices.iter() {
        assert!(virtual_vertex < initializer.vertex_num, "invalid input");
        assert!(!is_virtual[virtual_vertex as usize], "same virtual vertex appears twice");
        is_virtual[virtual_vertex as usize] = true;
    }
    let mut mapping_to_defect_vertices: Vec<usize> = (0..initializer.vertex_num).map(|_| usize::MAX).collect();
    for (i, &defect_vertex) in defect_vertices.iter().enumerate() {
        assert!(defect_vertex < initializer.vertex_num, "invalid input");
        assert!(!is_virtual[defect_vertex as usize], "syndrome vertex cannot be virtual");
        assert!(!is_defect[defect_vertex as usize], "same syndrome vertex appears twice");
        is_defect[defect_vertex as usize] = true;
        mapping_to_defect_vertices[defect_vertex as usize] = i;
    }
    // for each real vertex, add a corresponding virtual vertex to be matched
    let defect_num = defect_vertices.len();
    let legacy_vertex_num = defect_num * 2;
    let mut legacy_weighted_edges = Vec::<(usize, usize, u32)>::new();
    let mut boundaries = Vec::<Option<(VertexIndex, Weight)>>::new();
    for (i, &defect_vertex) in defect_vertices.iter().enumerate() {
        let complete_graph_edges = complete_graph.all_edges(defect_vertex);
        let mut boundary: Option<(VertexIndex, Weight)> = None;
        for (&peer, &(_, weight)) in complete_graph_edges.iter() {
            if is_virtual[peer as usize] && (boundary.is_none() || weight < boundary.as_ref().unwrap().1) {
                boundary = Some((peer, weight));
            }
        }
        if let Some((_, weight)) = boundary {
            // connect this real vertex to it's corresponding virtual vertex
            legacy_weighted_edges.push((i, i + defect_num, weight as u32));
        }
        boundaries.push(boundary); // save for later resolve legacy matchings
        for (&peer, &(_, weight)) in complete_graph_edges.iter() {
            if is_defect[peer as usize] {
                let j = mapping_to_defect_vertices[peer as usize];
                if i < j {
                    // remove duplicated edges
                    legacy_weighted_edges.push((i, j, weight as u32));
                    // println!{"edge {} {} {} ", i, j, weight};
                }
            }
        }
        for j in (i + 1)..defect_num {
            // virtual boundaries are always fully connected with weight 0
            legacy_weighted_edges.push((i + defect_num, j + defect_num, 0));
        }
    }
    // run blossom V to get matchings
    // println!("[debug] legacy_vertex_num: {:?}", legacy_vertex_num);
    // println!("[debug] legacy_weighted_edges: {:?}", legacy_weighted_edges);
    let matchings = blossom_v::safe_minimum_weight_perfect_matching(legacy_vertex_num, &legacy_weighted_edges);
    let mut mwpm_result = Vec::new();
    for i in 0..defect_num {
        let j = matchings[i];
        if j < defect_num {
            // match to a real vertex
            mwpm_result.push(defect_vertices[j]);
        } else {
            assert_eq!(
                j,
                i + defect_num,
                "if not matched to another real vertex, it must match to it's corresponding virtual vertex"
            );
            mwpm_result.push(
                boundaries[i]
                    .as_ref()
                    .expect("boundary must exist if match to virtual vertex")
                    .0,
            );
        }
    }
    mwpm_result
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DetailedMatching {
    /// must be a real vertex
    pub a: DefectIndex,
    /// might be a virtual vertex, but if it's a real vertex, then b > a stands
    pub b: DefectIndex,
    /// every vertex in between this pair, in the order `a -> path[0].0 -> path[1].0 -> .... -> path[-1].0` and it's guaranteed that path[-1].0 = b; might be empty if a and b are adjacent
    pub path: Vec<(DefectIndex, Weight)>,
    /// the overall weight of this path
    pub weight: Weight,
}

/// compute detailed matching information, note that the output will not include duplicated matched pairs
#[allow(clippy::unnecessary_cast)]
pub fn detailed_matching(
    initializer: &SolverInitializer,
    defect_vertices: &[DefectIndex],
    mwpm_result: &[DefectIndex],
) -> Vec<DetailedMatching> {
    let defect_num = defect_vertices.len();
    let mut is_defect: Vec<bool> = (0..initializer.vertex_num).map(|_| false).collect();
    for &defect_vertex in defect_vertices.iter() {
        assert!(defect_vertex < initializer.vertex_num, "invalid input");
        assert!(!is_defect[defect_vertex as usize], "same syndrome vertex appears twice");
        is_defect[defect_vertex as usize] = true;
    }
    assert_eq!(defect_num, mwpm_result.len(), "invalid mwpm result");
    let mut complete_graph = complete_graph::CompleteGraph::new(initializer.vertex_num, &initializer.weighted_edges);
    let mut details = Vec::new();
    for i in 0..defect_num {
        let a = defect_vertices[i];
        let b = mwpm_result[i];
        if !is_defect[b as usize] || a < b {
            let (path, weight) = complete_graph.get_path(a, b);
            let detail = DetailedMatching { a, b, path, weight };
            details.push(detail);
        }
    }
    details
}

