//! Lattice surgery on rotated surface-code patches (Horsman, Fowler, Devitt and Van Meter
//! 2012), each operation one circuit with SD6 noise at `p`: patches prepared, held, merged across
//! a seam for `merged` rounds, split, held and read out. Distances are odd, at least 3.
//!
//! ```
//! use stabilizer_qec::{lattice_surgery, Basis};
//!
//! let zz = lattice_surgery::zz_measurement(3, 3, 0.001, Basis::Z)?;
//! assert_eq!(zz.num_observables(), 3); // the outcome Z₁Z₂, then Z₁ and Z₂
//! # Ok::<(), stabilizer_qec::Error>(())
//! ```

use super::{probability, Basis, Circuit, Result};

/// Z⊗Z on two patches side by side, held `distance` rounds before and after. `Basis::Z`: both
/// in |0⟩, observables the outcome Z₁Z₂, Z₁ and Z₂; `Basis::X`: both in |+⟩, observable X₁X₂.
pub fn zz_measurement(distance: usize, merged: usize, p: f64, basis: Basis) -> Result<Circuit> {
    zz_measurement_held(distance, distance, merged, distance, p, basis)
}

/// Z⊗Z with `pre` rounds apart before the merge and `post` after it.
pub fn zz_measurement_held(distance: usize, pre: usize, merged: usize, post: usize, p: f64, basis: Basis) -> Result<Circuit> {
    probability(p, "p")?;
    Circuit::from_engine(crate::surgery::zz_circuit(distance, pre, merged, post, p, basis.engine())?)
}

/// X⊗X, the mirror of [`zz_measurement`]: two patches one above the other. `Basis::X`: the
/// outcome and each patch's X; `Basis::Z`: Z₁Z₂.
pub fn xx_measurement(distance: usize, merged: usize, p: f64, basis: Basis) -> Result<Circuit> {
    probability(p, "p")?;
    Circuit::from_engine(crate::surgery::xx_circuit(distance, merged, p, basis.engine())?)
}

/// A logical CNOT: control, an ancilla in |+⟩ and target in an L; Z_C Z_A, then X_A X_T, then
/// the ancilla read in Z. `inputs` `Basis::Z`: |0⟩|0⟩, observables Z_C and Z_T with its frame;
/// `Basis::X`: |+⟩|+⟩, observables X_T and X_C X_T with its frame.
pub fn cnot(distance: usize, merged: usize, p: f64, inputs: Basis) -> Result<Circuit> {
    probability(p, "p")?;
    Circuit::from_engine(crate::surgery::cnot_circuit(distance, merged, p, inputs.engine())?)
}

/// `k` Z⊗Z measurements in a row on two patches in |0⟩|0⟩, each of `merged` rounds and followed
/// by a round apart. Observables: each outcome, then Z₁ and Z₂.
pub fn repeated_zz(distance: usize, k: usize, merged: usize, p: f64) -> Result<Circuit> {
    probability(p, "p")?;
    Circuit::from_engine(crate::surgery::repeated_circuit(distance, k, merged, p)?)
}

/// `n` patches (2 to 32) in a row, all in |0⟩, merged at once: each neighbouring pair's Z⊗Z.
/// Observables: each seam's outcome, then each patch's Z.
pub fn line(distance: usize, n: usize, merged: usize, p: f64) -> Result<Circuit> {
    probability(p, "p")?;
    Circuit::from_engine(crate::surgery::line_circuit(distance, n, merged, p)?)
}
