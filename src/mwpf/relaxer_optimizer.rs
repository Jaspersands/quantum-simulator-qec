//! Relaxer Optimizer
//!
//! It's possible that two (or more) positive relaxers are bouncing, and the actual growth
//! is exponentially smaller but never reaching the optimal. In this case, we need
//! this module to optimize the positive relaxers. This only takes effect when such bouncing
//! is detected, and remains minimum in all other cases to avoid reduce time complexity.
//!

#[allow(unused_imports)]
use crate::mwpf::shim::prelude::*;
use crate::mwpf::invalid_subgraph::*;
use crate::mwpf::relaxer::*;
use crate::mwpf::util::*;

use std::sync::Arc;

use num_traits::{Signed, Zero};





#[derive(Default, Debug)]
pub enum OptimizerResult {
    #[default]
    Init,
    Optimized,     // normal
    EarlyReturned, // early return when the result is positive
    Skipped,       // when the `should_optimize` check returns false
}

impl OptimizerResult {
    pub fn or(&mut self, other: Self) {
        match self {
            OptimizerResult::EarlyReturned => {}
            _ => match other {
                OptimizerResult::Init => {}
                OptimizerResult::EarlyReturned => {
                    *self = OptimizerResult::EarlyReturned;
                }
                OptimizerResult::Skipped => {
                    *self = OptimizerResult::Skipped;
                }
                _ => {}
            },
        }
    }
}


#[derive(Default)]
pub struct RelaxerOptimizer {
    /// the set of existing relaxers
    relaxers: FastIterSet<Relaxer>,
}


#[derive(Default)]
pub struct ConstraintLine {
    pub lhs: Vec<(Rational, String)>,
    pub rhs: Rational,
}

impl RelaxerOptimizer {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ConstraintLine {
    pub fn new() -> Self {
        Self::default()
    }
}

fn rational_to_str(value: &Rational) -> String {
    format!("{}/{}", value.numer(), value.denom())
}

impl std::fmt::Display for ConstraintLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lhs_str_vec: Vec<String> = self
            .lhs
            .iter()
            .enumerate()
            .map(|(idx, (coefficient, var))| {
                let mut coefficient_str = rational_to_str(coefficient) + var;
                if idx > 0 && !coefficient_str.starts_with('-') {
                    coefficient_str = "+".to_string() + &coefficient_str;
                }
                coefficient_str
            })
            .collect();
        f.write_str(&(lhs_str_vec.join(" ") + " <= " + &rational_to_str(&self.rhs)))
    }
}

impl RelaxerOptimizer {
    /// moves all relaxer from other to here, when merging clusters
    pub fn append(&mut self, other: &mut RelaxerOptimizer) {
        self.relaxers.append(&mut other.relaxers);
    }

    pub fn insert(&mut self, relaxer: Relaxer) {
        self.relaxers.insert(relaxer);
    }

    pub fn should_optimize(&self, relaxer: &Relaxer) -> bool {
        // avoid calling optimizer on simple growing relaxer
        if relaxer.get_direction().len() < 2 {
            return false;
        }
        // self.relaxers.contains(relaxer)
        true
    }


    // the same method, but with f64 weight
    pub fn optimize(
        &mut self,
        relaxer: Relaxer,
        edge_slacks: FastIterMap<EdgeIndex, Rational>,
        mut dual_variables: FastIterMap<Arc<InvalidSubgraph>, Rational>,
    ) -> (Relaxer, bool) {
        use crate::mwpf::lp::{HighsModelStatus, RowProblem, Sense};
        use num_traits::ToPrimitive;

        use crate::mwpf::ordered_float::OrderedFloat;

        for invalid_subgraph in relaxer.get_direction().keys() {
            if !dual_variables.contains_key(invalid_subgraph) {
                dual_variables.insert(invalid_subgraph.clone(), OrderedFloat::zero());
            }
        }

        let mut model = RowProblem::default().optimise(Sense::Maximise);
        model.set_option("parallel", "off");
        model.set_option("threads", 1);

        let mut x_vars = vec![];
        let mut y_vars = vec![];
        let mut invalid_subgraphs = Vec::with_capacity(dual_variables.len());
        let mut edge_contributor: FastIterMap<EdgeIndex, Vec<usize>> =
            edge_slacks.keys().map(|&edge_index| (edge_index, vec![])).collect();

        for (var_index, (invalid_subgraph, dual_variable)) in dual_variables.iter().enumerate() {
            // constraint of the dual variable >= 0
            let x = model.add_col(1.0, 0.0.., []);
            let y = model.add_col(-1.0, 0.0.., []);
            x_vars.push(x);
            y_vars.push(y);

            // constraint for xs ys <= dual_variable
            model.add_row(
                ..dual_variable.to_f64().unwrap(),
                [(x_vars[var_index], -1.0), (y_vars[var_index], 1.0)],
            );
            invalid_subgraphs.push(invalid_subgraph.clone());

            for &edge_index in invalid_subgraph.hair.iter() {
                edge_contributor.get_mut(&edge_index).unwrap().push(var_index);
            }
        }

        for (&edge_index, ref slack) in edge_slacks.iter() {
            let mut row_entries = vec![];
            for &var_index in edge_contributor[&edge_index].iter() {
                row_entries.push((x_vars[var_index], 1.0));
                row_entries.push((y_vars[var_index], -1.0));
            }

            // constraint of edge: sum(y_S) <= weight
            model.add_row(..=slack.to_f64().unwrap(), row_entries);
        }

        let solved = model.solve();

        let mut direction: FastIterMap<Arc<InvalidSubgraph>, OrderedFloat> = FastIterMap::new();
        if solved.status() == HighsModelStatus::Optimal {
            let solution = solved.get_solution();

            // calculate the objective function
            let mut res = OrderedFloat::new(0.0);
            let cols = solution.columns();
            for i in 0..x_vars.len() {
                res += OrderedFloat::new(cols[2 * i] - cols[2 * i + 1]);
            }

            // check positivity of the objective
            if !(res.is_positive()) {
                return (relaxer, true);
            }

            for (var_index, invalid_subgraph) in invalid_subgraphs.iter().enumerate() {
                let overall_growth = cols[2 * var_index] - cols[2 * var_index + 1];
                if !overall_growth.is_zero() {
                    direction.insert(invalid_subgraph.clone(), Rational::from(overall_growth));
                }
            }
        } else {
            println!("solved status: {:?}", solved.status());
            unreachable!();
        }

        self.relaxers.insert(relaxer);
        (Relaxer::new(direction), false)
    }

}

#[cfg(test)]
pub mod tests {
    // MWPF's tests here exercise the HiGHS library; this is its first example on src/mwpf/lp.rs
    #[test]
    fn lp_highs_simple() {
        use crate::mwpf::lp::{HighsModelStatus, RowProblem, Sense};
        // max x + 2y + z subject to 3x + y <= 6, y + 2z <= 7
        let mut model = RowProblem::default().optimise(Sense::Maximise);
        let x = model.add_col(1., (0.).., []);
        let y = model.add_col(2., (0.).., []);
        let z = model.add_col(1., (0.).., []);
        model.add_row(..=6., [(x, 3.), (y, 1.)]);
        model.add_row(..=7., [(y, 1.), (z, 2.)]);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        // the expected solution is x=0 y=6 z=0.5
        assert_eq!(solved.get_solution().columns(), &[0., 6., 0.5]);
    }
}
