use super::{probability, Circuit, Error, Result, Window, WindowMode, WindowOptions};
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

impl GrossOperator {
    fn name(self) -> &'static str {
        match self {
            GrossOperator::F => "f",
            GrossOperator::Gh => "gh",
            GrossOperator::FTimesGh => "f+gh",
        }
    }
}

/// An automorphism of a bivariate bicycle code: the shift x^a y^b of both halves of the data,
/// with or without the ZX-duality, and what it does to the logical qubits.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Automorphism {
    /// (a, b): the shift x^a y^b.
    pub shift: (usize, usize),
    /// Whether it composes the shift with Bravyi et al.'s ZX-duality.
    pub dual: bool,
    /// Its action on the 2k logical basis elements (the X logicals, then the Z logicals, as
    /// [`BivariateBicycleCode::logicals`] gives them): `action[i]` holds the basis elements
    /// whose sum, over GF(2), is the image of element `i`.
    pub action: Vec<Vec<usize>>,
}

/// The gauging ancilla system that measures a gross-code logical X (see
/// [`BivariateBicycleCode::gauging`]).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Gauging {
    /// The operator's data qubits: the graph's vertices.
    pub support: Vec<usize>,
    /// The Z checks touching the support, each an edge qubit.
    pub edges: Vec<usize>,
    /// The edges added for expansion, as pairs of vertices.
    pub extra_edges: Vec<(usize, usize)>,
    /// Each edge's two ends, as indices into `support`.
    pub incidence: Vec<Vec<usize>>,
    /// Each vertex's edges: its Gauss-law check.
    pub gauss: Vec<Vec<usize>>,
    /// The flux checks, as edge indices.
    pub flux: Vec<Vec<usize>>,
    /// The merged code's X checks, over the data qubits then the edge qubits.
    pub hx: Vec<Vec<usize>>,
    /// Its Z checks, likewise.
    pub hz: Vec<Vec<usize>>,
    /// The merged cycle's depth.
    pub ticks: usize,
    /// The sparsest cut: (edges leaving it, vertices in it).
    pub worst_cut: (usize, usize),
}

/// A streamed memory's outcome (see [`stream_memory`]).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct StreamResult {
    /// Shots whose observable was decoded wrongly.
    pub failures: usize,
    /// Shots run: the shots asked for, rounded up to a multiple of 64.
    pub shots: usize,
    /// For the first stream of each batch of 64, each window's decode time in seconds.
    pub window_seconds: Vec<Vec<f64>>,
    /// The windows.
    pub windows: Vec<Window>,
    /// The wall time.
    pub seconds: f64,
}

/// A memory too long to model whole, sampled round by round and window-decoded as it streams:
/// an SD6 memory at strength `p` of `rounds` rounds (a million is fine), Z basis, its windows'
/// graphs built once from a short template. `shots` is rounded up to a multiple of 64; the
/// same `seed` gives the same failures on any number of `threads` (0 is every core).
///
/// ```
/// use stabilizer_qec::{stream_memory, SurfaceCode, WindowMode, WindowOptions};
///
/// let r = stream_memory(SurfaceCode::Rotated, 3, 200, 0.001, WindowOptions::new(2, 2, WindowMode::Parallel), 64, 1, 1)?;
/// assert_eq!(r.shots, 64);
/// assert!(r.failures < 64 && !r.windows.is_empty());
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn stream_memory(
    code: SurfaceCode,
    distance: usize,
    rounds: usize,
    p: f64,
    windows: WindowOptions,
    shots: usize,
    seed: u64,
    threads: usize,
) -> Result<StreamResult> {
    probability(p, "p")?;
    if rounds == 0 {
        return Err(Error::new("rounds must be at least 1"));
    }
    let kind = match code {
        SurfaceCode::Rotated => CodeKind::Rotated,
        SurfaceCode::Xzzx => CodeKind::Xzzx,
    };
    let (commit, buffer, mode, correlations) = windows.parts();
    let mode = match mode {
        WindowMode::Sliding => crate::window::Mode::Sliding,
        WindowMode::Parallel => crate::window::Mode::Parallel,
    };
    let batches = shots.max(1).div_ceil(64);
    let out = crate::batch::stream_shots(kind, distance, p, rounds, commit, buffer, mode, correlations, batches, seed, threads)?;
    stream_result(out)
}

/// Any circuit with a loop, sampled round by round and window-decoded as it streams, its
/// windows' graphs from a template of its folded error model: the middle windows of a short
/// version of the loop serve every window of the long one. As [`stream_memory`], for circuits
/// of your own (Stim's generated ones among them): their detectors need a time coordinate.
///
/// ```
/// use stabilizer_qec::{stream_circuit, Circuit, WindowMode, WindowOptions};
///
/// // A distance-3 repetition code, 500 rounds; each detector's last coordinate is its round.
/// let c: Circuit = "R 0 1 2 3 4\nMR 3 4\nREPEAT 500 {\n X_ERROR(0.002) 0 1 2\n CX 0 3 1 3 1 4 2 4\n MR 3 4\n \
///     DETECTOR(0, 1) rec[-2] rec[-4]\n DETECTOR(1, 1) rec[-1] rec[-3]\n SHIFT_COORDS(0, 1)\n}\nM 0 1 2\n\
///     DETECTOR(0, 1) rec[-5] rec[-3] rec[-2]\nDETECTOR(1, 1) rec[-4] rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]".parse()?;
/// let r = stream_circuit(&c, WindowOptions::new(4, 4, WindowMode::Parallel), 128, 7, 1)?;
/// assert_eq!(r.shots, 128);
/// assert!(r.failures < 8);
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
pub fn stream_circuit(circuit: &Circuit, windows: WindowOptions, shots: usize, seed: u64, threads: usize) -> Result<StreamResult> {
    let (commit, buffer, mode, correlations) = windows.parts();
    let mode = match mode {
        WindowMode::Sliding => crate::window::Mode::Sliding,
        WindowMode::Parallel => crate::window::Mode::Parallel,
    };
    let batches = shots.max(1).div_ceil(64);
    let out = crate::batch::stream_circuit(&circuit.inner, commit, buffer, mode, correlations, batches, seed, threads)?;
    stream_result(out)
}

fn stream_result(out: crate::batch::StreamOutcome) -> Result<StreamResult> {
    if out.unexplained > 0 {
        return Err(Error::new(format!(
            "internal error in stabilizer_qec: window commits left {} defects unexplained; please report it",
            out.unexplained
        )));
    }
    let windows: Vec<Window> = out.windows.into_iter().map(Window::from_info).collect();
    let per = windows.len().max(1);
    Ok(StreamResult {
        failures: out.failures,
        shots: out.shots,
        window_seconds: out.times.chunks(per).map(<[f64]>::to_vec).collect(),
        windows,
        seconds: out.seconds,
    })
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
        BivariateBicycleCode::wrap(crate::bb::BbCode::gross())
    }

    /// [[72, 12, 6]].
    pub fn bb72() -> BivariateBicycleCode {
        BivariateBicycleCode::wrap(crate::bb::BbCode::bb72())
    }

    /// [[90, 8, 10]]: A = x⁹ + y + y², B = 1 + x² + x⁷ on a 15 × 3 torus.
    pub fn bb90() -> BivariateBicycleCode {
        BivariateBicycleCode::wrap(crate::bb::BbCode::bb90())
    }

    /// [[108, 8, 10]].
    pub fn bb108() -> BivariateBicycleCode {
        BivariateBicycleCode::wrap(crate::bb::BbCode::bb108())
    }

    /// [[288, 12, 18]]: A = x³ + y² + y⁷, B = y³ + x + x² on a 12 × 12 torus.
    pub fn bb288() -> BivariateBicycleCode {
        BivariateBicycleCode::wrap(crate::bb::BbCode::bb288())
    }

    /// One of the codes of Bravyi et al.'s Table 3 by its number of data qubits: `"72"`,
    /// `"90"`, `"108"`, `"144"` (or `"gross"`) or `"288"`.
    pub fn named(name: &str) -> Result<BivariateBicycleCode> {
        Ok(BivariateBicycleCode::wrap(crate::bb::BbCode::named(name)?))
    }

    /// The code of A = Σ x^i y^j over the monomials `(i, j)` of `a`, and B over `b`, on an
    /// `l` × `m` torus: H_X = [A | B], H_Z = [Bᵀ | Aᵀ]. Each polynomial's three monomials must
    /// differ, and the code must encode a qubit.
    ///
    /// ```
    /// use stabilizer_qec::BivariateBicycleCode;
    ///
    /// let code = BivariateBicycleCode::from_polynomials(9, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])?;
    /// assert_eq!((code.n(), code.k()), (108, 8));
    /// # Ok::<(), stabilizer_qec::Error>(())
    /// ```
    pub fn from_polynomials(l: usize, m: usize, a: [(usize, usize); 3], b: [(usize, usize); 3]) -> Result<BivariateBicycleCode> {
        Ok(BivariateBicycleCode::wrap(crate::bb::BbCode::new(l, m, a, b)?))
    }

    fn wrap(inner: crate::bb::BbCode) -> BivariateBicycleCode {
        let gross = inner == crate::bb::BbCode::gross();
        BivariateBicycleCode { inner, gross }
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

    /// Every automorphism the code's shifts give: x^a y^b for each `a < l`, `b < m`, with and
    /// without Bravyi et al.'s ZX-duality, and its action on the logical qubits.
    ///
    /// ```
    /// use stabilizer_qec::BivariateBicycleCode;
    ///
    /// let autos = BivariateBicycleCode::gross().automorphisms()?;
    /// assert_eq!(autos.len(), 2 * 12 * 6);
    /// let identity = &autos[0];
    /// assert_eq!((identity.shift, identity.dual), ((0, 0), false));
    /// assert!(identity.action.iter().enumerate().all(|(i, row)| row == &vec![i]));
    /// # Ok::<(), stabilizer_qec::Error>(())
    /// ```
    pub fn automorphisms(&self) -> Result<Vec<Automorphism>> {
        use crate::bb_auto::Automorphism as Engine;
        let mut out = Vec::with_capacity(2 * self.inner.l * self.inner.m);
        for a in 0..self.inner.l {
            for b in 0..self.inner.m {
                for dual in [false, true] {
                    let m = self.inner.logical_action(Engine { shift: (a, b), dual })?;
                    out.push(Automorphism { shift: (a, b), dual, action: (0..m.rows).map(|r| m.row_ones(r)).collect() });
                }
            }
        }
        Ok(out)
    }

    /// The ancilla system whose merged code measures a gross-code logical X (Williamson and
    /// Yoder's gauging construction, as Cross, He, Rall and Yoder build it). `expanded` adds
    /// edges until every set of at most half the vertices has as many edges leaving it as
    /// vertices, which keeps the distance while it is measured.
    ///
    /// ```
    /// use stabilizer_qec::{BivariateBicycleCode, GrossOperator};
    ///
    /// let g = BivariateBicycleCode::gross().gauging(GrossOperator::F, false)?;
    /// assert_eq!(g.support.len(), 12);
    /// assert_eq!(g.hx[0].len() > 0, true);
    /// # Ok::<(), stabilizer_qec::Error>(())
    /// ```
    pub fn gauging(&self, operator: GrossOperator, expanded: bool) -> Result<Gauging> {
        if !self.gross {
            return Err(Error::new("logical measurements are built for the gross code only"));
        }
        let support = crate::bb::gross_operator(operator.name())?;
        let g = if expanded {
            crate::bb_gauge::Gauging::expanded(&self.inner, &support)
        } else {
            crate::bb_gauge::Gauging::new(&self.inner, &support)
        }?;
        let (hx, hz) = g.deformed(&self.inner);
        let rows = |m: &crate::gf2::BitMatrix| (0..m.rows).map(|r| m.row_ones(r)).collect();
        let ticks = crate::bb_circuit::merged_cycle(&self.inner, &g).ticks.len();
        let (boundary, side) = g.cheeger();
        Ok(Gauging {
            support: g.support.clone(),
            edges: g.edges.clone(),
            extra_edges: g.extra.clone(),
            incidence: g.incidence.clone(),
            gauss: g.gauss.clone(),
            flux: g.flux.clone(),
            hx: rows(&hx),
            hz: rows(&hz),
            ticks,
            worst_cut: (boundary, side.len()),
        })
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
        let support = crate::bb::gross_operator(operator.name())?;
        let g = if expanded {
            crate::bb_gauge::Gauging::expanded(&self.inner, &support)
        } else {
            crate::bb_gauge::Gauging::new(&self.inner, &support)
        }?;
        Circuit::from_engine(crate::bb_circuit::logical_measurement(&self.inner, &g, basis.engine(), pre, merged, post, p)?)
    }
}
