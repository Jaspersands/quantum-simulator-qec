// Ported from MWPF 0.2.12 (Yue Wu; MIT; see src/mwpf/LICENSE and THIRD_PARTY_NOTICES.md): the
// features its Python wheels are built with (`f64_weight`, `embed_visualizer`), its dependencies
// replaced by src/mwpf/shim.rs, and its LP solver HiGHS called by src/mwpf/lp.rs (HiGHS itself is
// in src/highs/, built with the `highs` feature; without it, lp.rs's own simplex).

//! MWPF: hypergraph minimum-weight parity factor decoding, a primal-dual algorithm over
//! invalid subgraphs that finds a parity factor and a lower bound on the optimum's weight
//! (Wu et al., arXiv 2508.04969), with union-find and single-hair relaxers as its
//! plugins. Python's `stabilizer_qec.mwpf` is its Python API.

#![allow(clippy::all, dead_code, unused_imports, unused_variables, unused_mut, unused_macros, unused_assignments, unknown_lints, mismatched_lifetime_syntaxes)]
#![deny(unconditional_recursion)]
#![allow(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links, rustdoc::bare_urls, rustdoc::invalid_html_tags, rustdoc::invalid_rust_codeblocks)]

#[macro_use]
pub(crate) mod shim;

pub(crate) use crate::fusion::shim::json;
pub(crate) use shim::{hashbrown, itertools, lnexp, num_traits, parking_lot, prettytable, priority_queue, rand_xoshiro, serde_json};

/// MWPF's version, which its snapshots, visualizer files and command line report (in place of
/// `CARGO_PKG_VERSION`, this crate's own)
pub const MWPF_VERSION: &str = "0.2.12";

pub mod bp;
pub mod cli;
mod cli_help;
pub mod cluster;
pub mod decoding_hypergraph;
pub mod dual_module;
pub mod dual_module_pq;
pub mod example_codes;
pub mod html_export;
pub mod invalid_subgraph;
mod json_impls;
pub mod lp;
pub mod matrix;
pub mod model_hypergraph;
pub mod mwpf_solver;
pub mod ordered_float;
pub mod plugin;
pub mod plugin_single_hair;
pub mod plugin_union_find;
pub mod pointers;
pub mod primal_module;
pub mod primal_module_serial;
pub mod primal_module_union_find;
#[cfg(feature = "python")]
pub mod python_signal_checker;
pub mod relaxer;
pub mod relaxer_forest;
pub mod relaxer_optimizer;
pub mod union_find;
pub mod util;
#[cfg(feature = "python")]
pub mod util_py;
pub mod visualize;

/// MWPF's command line, as `mwpf.run_cli(['mwpf', ...])` runs it: `parameters[0]` names the
/// binary, and help, a version or an error exits the process (as clap does).
#[cfg_attr(feature = "python", pyo3::pyfunction)]
pub fn run_cli(parameters: Vec<String>) {
    cli::Cli::parse_from(parameters).run();
}

/// `stabilizer_qec._core.mwpf`: MWPF's Python module (its `mwpf.mwpf`).
#[cfg(feature = "python")]
pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    let py = m.py();
    let mwpf = PyModule::new(py, "mwpf")?;
    util::register(&mwpf)?;
    visualize::register(&mwpf)?;
    example_codes::register(&mwpf)?;
    mwpf_solver::register(&mwpf)?;
    html_export::register(&mwpf)?;
    util_py::register(&mwpf)?;
    mwpf.add_function(wrap_pyfunction!(run_cli, &mwpf)?)?;
    m.add_submodule(&mwpf)?;
    Ok(())
}
