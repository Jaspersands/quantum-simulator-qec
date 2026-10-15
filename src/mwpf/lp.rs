//! The linear programs MWPF's relaxer optimizer solves: the part of the `highs` crate's interface
//! it uses (a row problem of nonnegative columns and rows bounded above).
//!
//! With this crate's `highs` feature (on in the Python wheels) they are solved by HiGHS 1.8.0
//! (src/highs/, built by build.rs), called as the `highs` crate calls it, so that the optimum
//! returned is MWPF's: MWPF's programs often have many optimal vertices, and the one taken steers
//! its relaxers, and so the subgraphs, bounds and running time that follow.
//!
//! Without it they are solved by a two-phase simplex on a dense dictionary (Chvátal's), Dantzig's
//! rule for the entering column and Bland's after degenerate pivots, with the right-hand sides
//! perturbed for the ratio test so that round-off cannot make it cycle. Its optimum has HiGHS's
//! value, but where several vertices are optimal it may be another, and so may what MWPF does next.

use std::ops::{Bound, RangeBounds};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sense {
    Maximise,
    Minimise,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HighsModelStatus {
    Optimal,
    Infeasible,
    Unbounded,
    /// any of HiGHS's other statuses, by its number
    Other(i32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Col(usize);

#[derive(Clone, Default, Debug)]
pub struct RowProblem {
    costs: Vec<f64>,
    rows: Vec<(f64, Vec<(usize, f64)>)>,
    /// the calls as made, which HiGHS replays in their order
    calls: Vec<Call>,
}

/// An option's value, as the `highs` crate passes each type.
#[derive(Clone, Debug)]
pub enum OptionValue {
    Bool(bool),
    Int(i32),
    Double(f64),
    String(String),
}

impl From<bool> for OptionValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}
impl From<i32> for OptionValue {
    fn from(value: i32) -> Self {
        Self::Int(value)
    }
}
impl From<f64> for OptionValue {
    fn from(value: f64) -> Self {
        Self::Double(value)
    }
}
impl From<&str> for OptionValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

#[derive(Clone, Debug)]
enum Call {
    Option(String, OptionValue),
    Col(f64),
    Row(f64, Vec<(usize, f64)>),
}

fn lower_of(bounds: &impl RangeBounds<f64>) -> f64 {
    match bounds.start_bound() {
        Bound::Included(&x) | Bound::Excluded(&x) => x,
        Bound::Unbounded => f64::NEG_INFINITY,
    }
}

fn upper_of(bounds: &impl RangeBounds<f64>) -> f64 {
    match bounds.end_bound() {
        Bound::Included(&x) | Bound::Excluded(&x) => x,
        Bound::Unbounded => f64::INFINITY,
    }
}

impl RowProblem {
    /// A column `cost * x`, `x` within `bounds` (MWPF's are all `0..`).
    pub fn add_col<const N: usize>(&mut self, cost: f64, bounds: impl RangeBounds<f64>, _entries: [(usize, f64); N]) -> Col {
        assert!(lower_of(&bounds) == 0.0 && upper_of(&bounds) == f64::INFINITY, "only columns x >= 0 are supported");
        self.costs.push(cost);
        self.calls.push(Call::Col(cost));
        Col(self.costs.len() - 1)
    }

    /// A row `sum(entries) <= upper` (MWPF's rows are bounded above only).
    pub fn add_row(&mut self, bounds: impl RangeBounds<f64>, entries: impl IntoIterator<Item = (Col, f64)>) {
        assert!(lower_of(&bounds) == f64::NEG_INFINITY, "only rows bounded above are supported");
        let entries: Vec<(Col, f64)> = entries.into_iter().collect();
        self.calls.push(Call::Row(upper_of(&bounds), entries.iter().map(|(col, value)| (col.0, *value)).collect()));
        let mut merged: Vec<(usize, f64)> = vec![];
        for (col, value) in entries {
            match merged.iter_mut().find(|(c, _)| *c == col.0) {
                Some(entry) => entry.1 += value,
                None => merged.push((col.0, value)),
            }
        }
        self.rows.push((upper_of(&bounds), merged));
    }

    pub fn optimise(self, sense: Sense) -> Model {
        Model { problem: self, sense }
    }
}

pub struct Model {
    problem: RowProblem,
    sense: Sense,
}

impl Model {
    pub fn add_col<const N: usize>(&mut self, cost: f64, bounds: impl RangeBounds<f64>, entries: [(usize, f64); N]) -> Col {
        self.problem.add_col(cost, bounds, entries)
    }

    pub fn add_row(&mut self, bounds: impl RangeBounds<f64>, entries: impl IntoIterator<Item = (Col, f64)>) {
        self.problem.add_row(bounds, entries)
    }

    /// HiGHS's options (MWPF turns off its parallelism); with HiGHS they are set in order with the
    /// columns and rows, and none changes the simplex here.
    pub fn set_option<V: Into<OptionValue>>(&mut self, name: &str, value: V) {
        self.problem.calls.push(Call::Option(name.to_string(), value.into()));
    }

    /// Solved by HiGHS, as MWPF's wheels are, when this crate is built with its `highs` feature.
    #[cfg(feature = "highs")]
    pub fn solve(self) -> SolvedModel {
        highs::solve(&self.problem.calls, self.sense)
    }

    /// Solved by the simplex below.
    #[cfg(not(feature = "highs"))]
    pub fn solve(self) -> SolvedModel {
        let n = self.problem.costs.len();
        let sign = if self.sense == Sense::Maximise { 1.0 } else { -1.0 };
        let costs: Vec<f64> = self.problem.costs.iter().map(|c| sign * c).collect();
        let mut a = vec![vec![0.0; n]; self.problem.rows.len()];
        let mut b = vec![0.0; self.problem.rows.len()];
        for (i, (upper, entries)) in self.problem.rows.iter().enumerate() {
            b[i] = *upper;
            for &(j, value) in entries {
                a[i][j] = value;
            }
        }
        let (status, columns) = simplex_max(&costs, &a, &b);
        SolvedModel { status, columns }
    }
}

pub struct SolvedModel {
    status: HighsModelStatus,
    columns: Vec<f64>,
}

impl SolvedModel {
    pub fn status(&self) -> HighsModelStatus {
        self.status
    }
    pub fn get_solution(&self) -> Solution {
        Solution { columns: self.columns.clone() }
    }
}

pub struct Solution {
    columns: Vec<f64>,
}

impl Solution {
    pub fn columns(&self) -> &[f64] {
        &self.columns
    }
}

/// HiGHS 1.8.0 (src/highs/, built by build.rs), called as the `highs` crate 1.6.1 that MWPF
/// uses calls it: a quiet instance given an empty problem and the sense, then MWPF's options,
/// columns (`Highs_addCol`) and rows (`Highs_addRow`) in the order MWPF made them, and a run.
#[cfg(feature = "highs")]
mod highs {
    use super::{Call, HighsModelStatus, OptionValue, Sense, SolvedModel};
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};

    type HighsInt = c_int;

    extern "C" {
        fn Highs_create() -> *mut c_void;
        fn Highs_destroy(highs: *mut c_void);
        fn Highs_passLp(highs: *mut c_void, num_col: HighsInt, num_row: HighsInt, num_nz: HighsInt, a_format: HighsInt, sense: HighsInt, offset: f64, col_cost: *const f64, col_lower: *const f64, col_upper: *const f64, row_lower: *const f64, row_upper: *const f64, a_start: *const HighsInt, a_index: *const HighsInt, a_value: *const f64) -> HighsInt;
        fn Highs_changeObjectiveSense(highs: *mut c_void, sense: HighsInt) -> HighsInt;
        fn Highs_setBoolOptionValue(highs: *mut c_void, option: *const c_char, value: HighsInt) -> HighsInt;
        fn Highs_setIntOptionValue(highs: *mut c_void, option: *const c_char, value: HighsInt) -> HighsInt;
        fn Highs_setDoubleOptionValue(highs: *mut c_void, option: *const c_char, value: f64) -> HighsInt;
        fn Highs_setStringOptionValue(highs: *mut c_void, option: *const c_char, value: *const c_char) -> HighsInt;
        fn Highs_addCol(highs: *mut c_void, cost: f64, lower: f64, upper: f64, num_new_nz: HighsInt, index: *const HighsInt, value: *const f64) -> HighsInt;
        fn Highs_addRow(highs: *mut c_void, lower: f64, upper: f64, num_new_nz: HighsInt, index: *const HighsInt, value: *const f64) -> HighsInt;
        fn Highs_run(highs: *mut c_void) -> HighsInt;
        fn Highs_getModelStatus(highs: *const c_void) -> HighsInt;
        fn Highs_getNumCols(highs: *const c_void) -> HighsInt;
        fn Highs_getNumRows(highs: *const c_void) -> HighsInt;
        fn Highs_getSolution(highs: *const c_void, col_value: *mut f64, col_dual: *mut f64, row_value: *mut f64, row_dual: *mut f64) -> HighsInt;
    }

    const STATUS_ERROR: HighsInt = -1;
    const MATRIX_FORMAT_COLUMN_WISE: HighsInt = 1;
    const OBJECTIVE_SENSE_MINIMIZE: HighsInt = 1;
    const OBJECTIVE_SENSE_MAXIMIZE: HighsInt = -1;

    /// Frees the instance however the solve ends.
    struct Instance(*mut c_void);
    impl Drop for Instance {
        fn drop(&mut self) {
            unsafe { Highs_destroy(self.0) }
        }
    }

    fn check(status: HighsInt, call: &str) {
        // the `highs` crate panics on an error and lets warnings pass
        assert!(status != STATUS_ERROR, "HiGHS error: {call} returned an error status");
    }

    fn set_option(highs: *mut c_void, name: &str, value: &OptionValue) {
        let name = CString::new(name).expect("invalid option name");
        let status = unsafe {
            match value {
                OptionValue::Bool(value) => Highs_setBoolOptionValue(highs, name.as_ptr(), *value as HighsInt),
                OptionValue::Int(value) => Highs_setIntOptionValue(highs, name.as_ptr(), *value),
                OptionValue::Double(value) => Highs_setDoubleOptionValue(highs, name.as_ptr(), *value),
                OptionValue::String(value) => {
                    let value = CString::new(value.as_str()).expect("invalid highs option value");
                    Highs_setStringOptionValue(highs, name.as_ptr(), value.as_ptr())
                }
            }
        };
        check(status, "Highs_setOptionValue");
    }

    pub(super) fn solve(calls: &[Call], sense: Sense) -> SolvedModel {
        let instance = Instance(unsafe { Highs_create() });
        let highs = instance.0;
        set_option(highs, "output_flag", &OptionValue::Bool(false));
        set_option(highs, "log_to_console", &OptionValue::Bool(false));
        let start: [HighsInt; 1] = [0];
        let none: [f64; 0] = [];
        let no_index: [HighsInt; 0] = [];
        check(
            unsafe { Highs_passLp(highs, 0, 0, 0, MATRIX_FORMAT_COLUMN_WISE, OBJECTIVE_SENSE_MINIMIZE, 0.0, none.as_ptr(), none.as_ptr(), none.as_ptr(), none.as_ptr(), none.as_ptr(), start.as_ptr(), no_index.as_ptr(), none.as_ptr()) },
            "Highs_passLp",
        );
        let sense = if sense == Sense::Maximise { OBJECTIVE_SENSE_MAXIMIZE } else { OBJECTIVE_SENSE_MINIMIZE };
        assert_eq!(unsafe { Highs_changeObjectiveSense(highs, sense) }, 0, "changeObjectiveSense failed");
        for call in calls {
            match call {
                Call::Option(name, value) => set_option(highs, name, value),
                Call::Col(cost) => check(unsafe { Highs_addCol(highs, *cost, 0.0, f64::INFINITY, 0, no_index.as_ptr(), none.as_ptr()) }, "Highs_addCol"),
                Call::Row(upper, entries) => {
                    let index: Vec<HighsInt> = entries.iter().map(|&(col, _)| col as HighsInt).collect();
                    let value: Vec<f64> = entries.iter().map(|&(_, value)| value).collect();
                    check(unsafe { Highs_addRow(highs, f64::NEG_INFINITY, *upper, index.len() as HighsInt, index.as_ptr(), value.as_ptr()) }, "Highs_addRow");
                }
            }
        }
        check(unsafe { Highs_run(highs) }, "Highs_run");
        let status = match unsafe { Highs_getModelStatus(highs) } {
            7 => HighsModelStatus::Optimal,
            8 => HighsModelStatus::Infeasible,
            10 => HighsModelStatus::Unbounded,
            other => HighsModelStatus::Other(other),
        };
        let cols = unsafe { Highs_getNumCols(highs) } as usize;
        let rows = unsafe { Highs_getNumRows(highs) } as usize;
        let (mut columns, mut col_dual, mut row_value, mut row_dual) = (vec![0.0; cols], vec![0.0; cols], vec![0.0; rows], vec![0.0; rows]);
        unsafe { Highs_getSolution(highs, columns.as_mut_ptr(), col_dual.as_mut_ptr(), row_value.as_mut_ptr(), row_dual.as_mut_ptr()) };
        SolvedModel { status, columns }
    }
}

const EPS: f64 = 1e-9;
/// Ratios this close are ties, broken by the smaller basic variable.
const TIE: f64 = 1e-12;
/// The right-hand sides' perturbation: row `i` is relaxed by `PERTURBATION * (1 + u_i)`, `u_i` in
/// [0, 1) and distinct, for the ratio test only. MWPF's programs are highly degenerate (many rows
/// tight at once, right-hand sides that are zero up to round-off) and, without it, ties decided
/// by round-off make the simplex cycle; the solution is read from the unperturbed right-hand sides.
const PERTURBATION: f64 = 1e-9;
/// Degenerate pivots in a row after which Bland's rule takes over.
const DEGENERATE_LIMIT: usize = 50;
/// Pivots per row and column after which the simplex stops (a safeguard that MWPF's programs
/// have not reached): phase two then returns its current point, feasible if not optimal.
const PIVOTS_PER_DIMENSION: usize = 50;

/// A dictionary: basic variables `x_B = rhs - a x_N`, objective `z = value + c x_N`.
struct Dictionary {
    a: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    /// the perturbation of `rhs`, transformed with it
    perturbation: Vec<f64>,
    c: Vec<f64>,
    value: f64,
    basic: Vec<usize>,
    nonbasic: Vec<usize>,
}

enum Outcome {
    Optimal,
    Unbounded,
    /// stopped after the pivot limit
    Stopped,
}

impl Dictionary {
    fn pivot(&mut self, row: usize, col: usize) {
        let p = self.a[row][col];
        let width = self.nonbasic.len();
        // the entering variable's row: x_j = (rhs_r - sum_{k != j} a_rk x_k - x_leaving) / a_rj
        self.rhs[row] /= p;
        self.perturbation[row] /= p;
        for k in 0..width {
            if k != col {
                self.a[row][k] /= p;
            }
        }
        self.a[row][col] = 1.0 / p;
        for i in 0..self.a.len() {
            if i == row {
                continue;
            }
            let f = self.a[i][col];
            if f == 0.0 {
                continue;
            }
            self.rhs[i] -= f * self.rhs[row];
            self.perturbation[i] -= f * self.perturbation[row];
            for k in 0..width {
                if k != col {
                    self.a[i][k] -= f * self.a[row][k];
                }
            }
            self.a[i][col] = -f * self.a[row][col];
        }
        let f = self.c[col];
        if f != 0.0 {
            self.value += f * self.rhs[row];
            for k in 0..width {
                if k != col {
                    self.c[k] -= f * self.a[row][k];
                }
            }
            self.c[col] = -f * self.a[row][col];
        }
        std::mem::swap(&mut self.basic[row], &mut self.nonbasic[col]);
    }

    fn run(&mut self) -> Outcome {
        let mut degenerate = 0;
        let limit = PIVOTS_PER_DIMENSION * (self.a.len() + self.nonbasic.len());
        for _ in 0..limit {
            let bland = degenerate >= DEGENERATE_LIMIT;
            let mut entering: Option<usize> = None;
            for k in 0..self.nonbasic.len() {
                if self.c[k] > EPS {
                    entering = match entering {
                        None => Some(k),
                        Some(e) if bland && self.nonbasic[k] < self.nonbasic[e] => Some(k),
                        Some(e) if !bland && self.c[k] > self.c[e] => Some(k),
                        keep => keep,
                    };
                }
            }
            let Some(col) = entering else { return Outcome::Optimal };
            let mut leaving: Option<(usize, f64)> = None;
            for i in 0..self.a.len() {
                if self.a[i][col] > EPS {
                    let ratio = (self.rhs[i] + self.perturbation[i]).max(0.0) / self.a[i][col];
                    leaving = match leaving {
                        None => Some((i, ratio)),
                        Some((_, best)) if ratio < best - TIE => Some((i, ratio)),
                        Some((r, best)) if ratio <= best + TIE && self.basic[i] < self.basic[r] => Some((i, ratio.min(best))),
                        keep => keep,
                    };
                }
            }
            let Some((row, ratio)) = leaving else { return Outcome::Unbounded };
            degenerate = if ratio <= TIE { degenerate + 1 } else { 0 };
            self.pivot(row, col);
        }
        Outcome::Stopped
    }
}

/// maximise `c x` subject to `a x <= b`, `x >= 0`: the status and, if optimal, `x`.
pub fn simplex_max(c: &[f64], a: &[Vec<f64>], b: &[f64]) -> (HighsModelStatus, Vec<f64>) {
    let n = c.len();
    let m = b.len();
    // distinct perturbations from a Weyl sequence (golden-ratio steps), as fractions of [0, 1)
    let perturbation = (1..=m as u64).map(|i| PERTURBATION * (1.0 + (i.wrapping_mul(0x9E3779B97F4A7C15) >> 11) as f64 / (1u64 << 53) as f64)).collect();
    let mut d = Dictionary {
        a: a.to_vec(),
        rhs: b.to_vec(),
        perturbation,
        c: c.to_vec(),
        value: 0.0,
        basic: (n..n + m).collect(),
        nonbasic: (0..n).collect(),
    };
    // phase one, when the origin is infeasible: maximise -x0 with x0 >= 0 added to every row
    let worst = (0..m).min_by(|&i, &j| d.rhs[i].partial_cmp(&d.rhs[j]).unwrap());
    if let Some(worst) = worst.filter(|&i| d.rhs[i] < -EPS) {
        let x0 = n + m;
        for row in d.a.iter_mut() {
            row.push(-1.0);
        }
        let original_c = std::mem::replace(&mut d.c, vec![0.0; n + 1]);
        d.c[n] = -1.0;
        d.nonbasic.push(x0);
        d.pivot(worst, n);
        if matches!(d.run(), Outcome::Stopped) || d.value < -EPS {
            return (HighsModelStatus::Infeasible, vec![]);
        }
        // x0 leaves the basis if it is there (at zero), then its column goes
        if let Some(row) = d.basic.iter().position(|&v| v == x0) {
            let col = (0..d.nonbasic.len())
                .max_by(|&p, &q| d.a[row][p].abs().partial_cmp(&d.a[row][q].abs()).unwrap())
                .expect("a column to pivot on");
            d.pivot(row, col);
        }
        let col = d.nonbasic.iter().position(|&v| v == x0).expect("x0 is nonbasic");
        for row in d.a.iter_mut() {
            row.remove(col);
        }
        d.nonbasic.remove(col);
        // the objective in the current nonbasic variables
        d.c = vec![0.0; d.nonbasic.len()];
        d.value = 0.0;
        for (k, &v) in d.nonbasic.iter().enumerate() {
            if v < n {
                d.c[k] += original_c[v];
            }
        }
        for (i, &v) in d.basic.iter().enumerate() {
            if v < n && original_c[v] != 0.0 {
                d.value += original_c[v] * d.rhs[i];
                for k in 0..d.nonbasic.len() {
                    d.c[k] -= original_c[v] * d.a[i][k];
                }
            }
        }
    }
    match d.run() {
        Outcome::Unbounded => (HighsModelStatus::Unbounded, vec![]),
        Outcome::Optimal | Outcome::Stopped => {
            let mut x = vec![0.0; n];
            for (i, &v) in d.basic.iter().enumerate() {
                if v < n {
                    x[v] = d.rhs[i].max(0.0);
                }
            }
            (HighsModelStatus::Optimal, x)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force(c: &[f64], a: &[Vec<f64>], b: &[f64]) -> Option<f64> {
        // every vertex: n tight constraints among the m rows and n bounds
        let n = c.len();
        let m = b.len();
        let mut rows: Vec<(Vec<f64>, f64)> = a.iter().cloned().zip(b.iter().cloned()).collect();
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = -1.0;
            rows.push((e, 0.0));
        }
        let mut best: Option<f64> = None;
        let total = rows.len();
        let mut pick = vec![0usize; n];
        fn choose(k: usize, start: usize, total: usize, pick: &mut Vec<usize>, f: &mut dyn FnMut(&[usize])) {
            if k == pick.len() {
                f(pick);
                return;
            }
            for s in start..total {
                pick[k] = s;
                choose(k + 1, s + 1, total, pick, f);
            }
        }
        choose(0, 0, total, &mut pick, &mut |sel: &[usize]| {
            let mut mat: Vec<Vec<f64>> = sel.iter().map(|&s| rows[s].0.clone()).collect();
            let mut rhs: Vec<f64> = sel.iter().map(|&s| rows[s].1).collect();
            for col in 0..n {
                let p = (col..n).max_by(|&x, &y| mat[x][col].abs().partial_cmp(&mat[y][col].abs()).unwrap()).unwrap();
                if mat[p][col].abs() < 1e-12 {
                    return;
                }
                mat.swap(col, p);
                rhs.swap(col, p);
                for r in 0..n {
                    if r != col {
                        let f = mat[r][col] / mat[col][col];
                        for k in 0..n {
                            mat[r][k] -= f * mat[col][k];
                        }
                        rhs[r] -= f * rhs[col];
                    }
                }
            }
            let x: Vec<f64> = (0..n).map(|k| rhs[k] / mat[k][k]).collect();
            if x.iter().any(|&v| v < -1e-9) {
                return;
            }
            for i in 0..m {
                let lhs: f64 = (0..n).map(|j| a[i][j] * x[j]).sum();
                if lhs > b[i] + 1e-9 {
                    return;
                }
            }
            let value: f64 = (0..n).map(|j| c[j] * x[j]).sum();
            best = Some(best.map_or(value, |v: f64| v.max(value)));
        });
        best
    }

    #[test]
    fn lp_against_vertex_enumeration() {
        use crate::fusion::shim::rand_xoshiro::rand_core::{RngCore, SeedableRng};
        let mut rng = crate::fusion::shim::rand_xoshiro::Xoshiro256StarStar::seed_from_u64(7);
        let mut uniform = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        for trial in 0..400 {
            let n = 1 + (uniform() * 4.0) as usize;
            let m = 1 + (uniform() * 5.0) as usize;
            let c: Vec<f64> = (0..n).map(|_| (uniform() * 6.0).floor() - 2.0).collect();
            let a: Vec<Vec<f64>> = (0..m).map(|_| (0..n).map(|_| (uniform() * 5.0).floor() - 2.0).collect()).collect();
            let b: Vec<f64> = (0..m).map(|_| (uniform() * 8.0).floor() - if trial % 3 == 0 { 3.0 } else { 0.0 }).collect();
            let (status, x) = simplex_max(&c, &a, &b);
            let best = brute_force(&c, &a, &b);
            match status {
                HighsModelStatus::Optimal => {
                    let value: f64 = (0..n).map(|j| c[j] * x[j]).sum();
                    for i in 0..m {
                        let lhs: f64 = (0..n).map(|j| a[i][j] * x[j]).sum();
                        assert!(lhs <= b[i] + 1e-7, "trial {trial}: row {i} violated");
                    }
                    assert!((value - best.expect("feasible")).abs() < 1e-7, "trial {trial}: {value} vs {best:?}");
                }
                HighsModelStatus::Infeasible => assert!(best.is_none(), "trial {trial}: feasible, {best:?}"),
                HighsModelStatus::Unbounded => {
                    // unbounded: a feasible point exists, and enumeration (of vertices) may find one
                    assert!(c.iter().any(|&v| v > 0.0));
                }
                HighsModelStatus::Other(status) => panic!("trial {trial}: status {status}"),
            }
        }
    }

    #[test]
    fn lp_mwpf_shape() {
        // maximise x0 - y0 + x1 - y1: x0 - y0 <= dual ... as MWPF writes it
        let mut problem = RowProblem::default();
        let x0 = problem.add_col(1.0, 0.0.., []);
        let y0 = problem.add_col(-1.0, 0.0.., []);
        let x1 = problem.add_col(1.0, 0.0.., []);
        let y1 = problem.add_col(-1.0, 0.0.., []);
        problem.add_row(..1.0, [(x0, -1.0), (y0, 1.0)]);
        problem.add_row(..0.0, [(x1, -1.0), (y1, 1.0)]);
        problem.add_row(..=2.0, [(x0, 1.0), (y0, -1.0), (x1, 1.0), (y1, -1.0)]);
        problem.add_row(..=0.5, [(x1, 1.0), (y1, -1.0)]);
        let mut model = problem.optimise(Sense::Maximise);
        model.set_option("threads", 1);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        let cols = solved.get_solution().columns().to_vec();
        let total = cols[0] - cols[1] + cols[2] - cols[3];
        assert!((total - 2.0).abs() < 1e-12);
    }
}

