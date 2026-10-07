//! The crate's stable API: everything re-exported at the crate root from here.
//!
//! It mirrors the Python package's (`stabilizer-qec`), in Rust's idiom: circuits and error models
//! that parse from Stim's text and print back to it, a sampler and a converter producing
//! bit-packed shot tables, decoders that are `Send + Sync` and decode batches across threads, and
//! generators for the circuits the project studies. Every fallible call returns [`Error`].

#![warn(missing_docs)]

mod bits;
mod circuits;
mod codes;
mod decoders;
pub mod lattice_surgery;

pub use bits::BitTable;
pub use circuits::{Circuit, CircuitErrorLocation, DemOptions, DemSampler, DemSamples, DetectorErrorModel, DetectorSampler, DiagramKind, ExplainedError, GeneratedNoise, MeasurementConverter, Pauli, Samples, Target, TargetWithCoords};
pub use codes::{memory_circuit, stream_circuit, stream_memory, Automorphism, Basis, BivariateBicycleCode, CssCode, Gauging, GrossOperator, Noise, StreamResult, SurfaceCode};
pub use decoders::{
    BeliefMatching, BpDecoder, BpMethod, BpOptions, BpOsd, BpOsdDecoder, BpOsdOutcome, BpOutcome, Matching, OsdMethod,
    Prediction, UnionFind, Window, WindowMatching, WindowMode, WindowOptions,
};

/// Why a call failed: the input was not something the engine can take, with a message saying
/// what and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl Error {
    pub(crate) fn new(message: impl Into<String>) -> Error {
        Error { message: message.into() }
    }

    /// What went wrong, for a person to read.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(message: String) -> Error {
        Error { message }
    }
}

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Every probability argument: a number in [0, 1].
pub(crate) fn probability(p: f64, name: &str) -> Result<f64> {
    if (0.0..=1.0).contains(&p) {
        Ok(p)
    } else {
        Err(Error::new(format!("{name} = {p} is outside [0, 1]")))
    }
}
