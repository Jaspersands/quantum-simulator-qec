//! Python bindings for the cross-check harness (`tools/xcheck.py`).
//!
//! Shots cross the boundary as bytes in Stim's b8 layout (numpy's
//! `packbits(..., bitorder="little")`), and predictions come back as
//! little-endian u64 and f64 arrays, so no numpy crate is needed on this side.

use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::circuit::{Basis, Circuit};
use crate::dem::Dem;
use crate::dem_decoder::DemDecoder;
use crate::frame_sampler::FrameSampler;
use crate::memory::{generate, CodeKind, NoiseModel};
use crate::shots::{pack_row, read_b8, write_01};
use crate::surface_code::Xorshift;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

#[pyfunction]
#[pyo3(signature = (code, d, rounds, noise, p, eta=0.5, basis="z"))]
fn generate_circuit(code: &str, d: usize, rounds: usize, noise: &str, p: f64, eta: f64, basis: &str) -> PyResult<String> {
    let kind = match code {
        "rotated" => CodeKind::Rotated,
        "xzzx" => CodeKind::Xzzx,
        _ => return Err(err(format!("unknown code '{code}'"))),
    };
    let noise = match noise {
        "current" => NoiseModel::Current { p, eta },
        "sd6" => NoiseModel::Sd6 { p },
        _ => return Err(err(format!("unknown noise model '{noise}'"))),
    };
    let basis = match basis {
        "z" => Basis::Z,
        "x" => Basis::X,
        _ => return Err(err(format!("unknown basis '{basis}'"))),
    };
    Ok(generate(kind, d, rounds, noise, basis).map_err(err)?.to_stim())
}

#[pyfunction]
#[pyo3(signature = (circuit_text, decompose=false))]
fn dem_from_circuit(circuit_text: &str, decompose: bool) -> PyResult<String> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    Ok(Dem::from_circuit(&c).map_err(err)?.to_stim(decompose))
}

type Decoded<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize, f64);

fn decode_packed<'py>(py: Python<'py>, dem: &Dem, packed: &[u8], num_shots: usize) -> PyResult<Decoded<'py>> {
    let decoder = DemDecoder::new(dem).map_err(err)?;
    let nd = dem.num_detectors;
    let stride = nd.div_ceil(8);
    if packed.len() != stride * num_shots {
        return Err(err(format!("{} bytes is not {num_shots} shots of {nd} detectors", packed.len())));
    }
    let mut preds = Vec::with_capacity(8 * num_shots);
    let mut weights = Vec::with_capacity(8 * num_shots);
    let mut errors = 0usize;
    let mut defects = Vec::new();
    let start = Instant::now();
    for s in 0..num_shots {
        defects.clear();
        let row = &packed[s * stride..(s + 1) * stride];
        for i in 0..nd {
            if (row[i / 8] >> (i % 8)) & 1 == 1 {
                defects.push(i as u32);
            }
        }
        match decoder.decode(&defects) {
            Ok(pred) => {
                preds.extend_from_slice(&pred.observables.to_le_bytes());
                weights.extend_from_slice(&pred.weight.to_le_bytes());
            }
            Err(_) => {
                errors += 1;
                preds.extend_from_slice(&u64::MAX.to_le_bytes());
                weights.extend_from_slice(&f64::NAN.to_le_bytes());
            }
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    Ok((PyBytes::new_bound(py, &preds), PyBytes::new_bound(py, &weights), errors, seconds))
}

/// Decode with a model someone else wrote, pieces and all (Stim's decomposed DEM).
#[pyfunction]
fn decode_b8<'py>(py: Python<'py>, dem_text: &str, packed: &[u8], num_shots: usize) -> PyResult<Decoded<'py>> {
    let dem = Dem::parse(dem_text).map_err(err)?;
    decode_packed(py, &dem, packed, num_shots)
}

/// Decode with this engine's own model of the circuit, and its own decomposition.
#[pyfunction]
fn decode_b8_own<'py>(py: Python<'py>, circuit_text: &str, packed: &[u8], num_shots: usize) -> PyResult<Decoded<'py>> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    let dem = Dem::from_circuit(&c).map_err(err)?;
    decode_packed(py, &dem, packed, num_shots)
}

#[pyfunction]
fn sample_b8<'py>(
    py: Python<'py>,
    circuit_text: &str,
    num_shots: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    let sampler = FrameSampler::new(&c).map_err(err)?;
    let mut rng = Xorshift::new(seed);
    let no = sampler.num_observables();
    let mut dets = Vec::new();
    let mut obs = Vec::new();
    for _ in 0..num_shots {
        let shot = sampler.sample(&mut rng);
        pack_row(&shot.detectors, &mut dets);
        let bits: Vec<bool> = (0..no).map(|k| (shot.observables >> k) & 1 == 1).collect();
        pack_row(&bits, &mut obs);
    }
    Ok((PyBytes::new_bound(py, &dets), PyBytes::new_bound(py, &obs)))
}

#[pyfunction]
fn b8_to_01(packed: &[u8], num_bits: usize) -> PyResult<String> {
    Ok(write_01(&read_b8(packed, num_bits).map_err(err)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(dem_from_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(decode_b8, m)?)?;
    m.add_function(wrap_pyfunction!(decode_b8_own, m)?)?;
    m.add_function(wrap_pyfunction!(sample_b8, m)?)?;
    m.add_function(wrap_pyfunction!(b8_to_01, m)?)?;
    Ok(())
}
