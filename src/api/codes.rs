use super::{probability, Circuit, Error, Result};
use crate::memory::{CodeKind, NoiseModel};

/// A measurement or preparation basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Basis {
    /// The X basis (|+⟩, |−⟩).
    X,
    /// The Z basis (|0⟩, |1⟩).
    Z,
}

impl Basis {
    pub(crate) fn engine(self) -> crate::circuit::Basis {
        match self {
            Basis::X => crate::circuit::Basis::X,
            Basis::Z => crate::circuit::Basis::Z,
        }
    }
}

/// A surface code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SurfaceCode {
    /// The rotated surface code: d² data qubits and d² − 1 checks.
    Rotated,
    /// The XZZX surface code, which tolerates biased noise.
    Xzzx,
}

/// Circuit noise.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Noise {
    /// SD6, the standard circuit-level model at strength `p` (Gidney, Newman, Fowler and
    /// Broughton 2021): two-qubit depolarizing after every CNOT, single-qubit after every gate and
    /// on every idle qubit, flips on every reset and measurement; as Stim's generated circuits.
    Sd6 {
        /// The strength.
        p: f64,
    },
    /// The engine's own model at strength `p`, with bias `eta` = p_Z / (p_X + p_Y).
    Biased {
        /// The strength.
        p: f64,
        /// The bias (0.5 is unbiased).
        eta: f64,
    },
}

/// A surface-code memory experiment: a patch of distance `distance` (2 to 11) prepared in
/// `basis`, held for `rounds` rounds of syndrome extraction (at most 10,000), and read out;
/// detectors on every check and one observable.
///
/// ```
/// use stabilizer_qec::{memory_circuit, Basis, Noise, SurfaceCode};
///
/// let c = memory_circuit(SurfaceCode::Rotated, 3, 3, Noise::Sd6 { p: 0.001 }, Basis::Z)?;
/// assert_eq!((c.num_detectors(), c.num_observables()), (24, 1));
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
pub fn memory_circuit(code: SurfaceCode, distance: usize, rounds: usize, noise: Noise, basis: Basis) -> Result<Circuit> {
    let kind = match code {
        SurfaceCode::Rotated => CodeKind::Rotated,
        SurfaceCode::Xzzx => CodeKind::Xzzx,
    };
    let noise = match noise {
        Noise::Sd6 { p } => NoiseModel::Sd6 { p },
        Noise::Biased { p, eta } => NoiseModel::Current { p, eta },
    };
    Circuit::from_engine(crate::memory::generate(kind, distance, rounds, noise, basis.engine())?)
}

/// A weight-12 logical X of the gross code whose measurement is built (see
/// [`BivariateBicycleCode::logical_measurement_circuit`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GrossOperator {
    /// Bravyi et al.'s X(f, 0).
    F,
    /// X(g, h).
    Gh,
    /// Their product, the joint measurement a CNOT needs.
    FTimesGh,
}

/// One of IBM's bivariate bicycle codes (Bravyi et al., Nature 627, 778, 2024), with the paper's
/// depth-8 syndrome cycle.
///
/// ```
/// use stabilizer_qec::{Basis, BivariateBicycleCode};
///
/// let gross = BivariateBicycleCode::gross();
/// assert_eq!((gross.n(), gross.k()), (144, 12));
/// let c = gross.memory_circuit(2, 0.003, Basis::Z)?;
/// assert_eq!(c.num_observables(), 12);
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct BivariateBicycleCode {
    inner: crate::bb::BbCode,
    gross: bool,
}

impl BivariateBicycleCode {
    /// The gross code, [[144, 12, 12]].
    pub fn gross() -> BivariateBicycleCode {
        BivariateBicycleCode { inner: crate::bb::BbCode::gross(), gross: true }
    }

    /// [[72, 12, 6]].
    pub fn bb72() -> BivariateBicycleCode {
        BivariateBicycleCode { inner: crate::bb::BbCode::bb72(), gross: false }
    }

    /// Data qubits.
    pub fn n(&self) -> usize {
        2 * self.inner.l * self.inner.m
    }

    /// Logical qubits.
    pub fn k(&self) -> usize {
        self.inner.logicals().0.rows
    }

    /// (H_X, H_Z): each check as the data qubits it acts on.
    pub fn check_matrices(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let rows = |m: &crate::gf2::BitMatrix| (0..m.rows).map(|r| m.row_ones(r)).collect();
        (rows(&self.inner.hx()), rows(&self.inner.hz()))
    }

    /// (logical X, logical Z): `k` operators each, as the data qubits they act on, paired so that
    /// X_i and Z_j anticommute exactly when i = j.
    pub fn logicals(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let (lx, lz) = self.inner.logicals();
        let rows = |m: &crate::gf2::BitMatrix| (0..m.rows).map(|r| m.row_ones(r)).collect();
        (rows(&lx), rows(&lz))
    }

    /// A memory: the data prepared in `basis`, `cycles` syndrome cycles (at most 10,000) under
    /// circuit noise `p`, the data read out. The observables are the `k` logicals of that basis.
    pub fn memory_circuit(&self, cycles: usize, p: f64, basis: Basis) -> Result<Circuit> {
        crate::bb::check_cycles(cycles, cycles)?;
        probability(p, "p")?;
        let c = match basis {
            Basis::Z => crate::circuit::Circuit::parse(&self.inner.memory_z(cycles, p))?,
            Basis::X => crate::bb_circuit::memory(&self.inner, basis.engine(), cycles, p),
        };
        Circuit::from_engine(c)
    }

    /// The gauging measurement of a gross-code logical X (Cross, He, Rall and Yoder's
    /// construction): `pre` memory cycles, `merged` cycles of the deformed code, `post` memory
    /// cycles, the data read in `basis`. `expanded` adds edges until every set of at most half
    /// the vertices has as many edges leaving it as vertices, which keeps the distance.
    #[allow(clippy::too_many_arguments)]
    pub fn logical_measurement_circuit(
        &self,
        operator: GrossOperator,
        basis: Basis,
        pre: usize,
        merged: usize,
        post: usize,
        p: f64,
        expanded: bool,
    ) -> Result<Circuit> {
        if !self.gross {
            return Err(Error::new("logical measurements are built for the gross code only"));
        }
        crate::bb::check_cycles(pre.saturating_add(merged).saturating_add(post).max(1), merged)?;
        probability(p, "p")?;
        let name = match operator {
            GrossOperator::F => "f",
            GrossOperator::Gh => "gh",
            GrossOperator::FTimesGh => "f+gh",
        };
        let support = crate::bb::gross_operator(name)?;
        let g = if expanded {
            crate::bb_gauge::Gauging::expanded(&self.inner, &support)
        } else {
            crate::bb_gauge::Gauging::new(&self.inner, &support)
        }?;
        Circuit::from_engine(crate::bb_circuit::logical_measurement(&self.inner, &g, basis.engine(), pre, merged, post, p)?)
    }
}
