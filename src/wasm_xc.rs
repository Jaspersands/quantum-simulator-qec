//! WebAssembly surface for the general path: the live Stim comparison in
//! section 10 and the SD6 option in sections 07 and 09.
//!
//! Text crosses the boundary through one byte buffer. JS asks for room with
//! `wasm_text_buf`, writes UTF-8 into it and calls an export; the export writes
//! its reply (circuit text or a JSON summary) back into the same buffer and
//! returns its length. One buffer, single-threaded, the same pattern as the
//! matching cost matrix.

use std::fmt::Write as _;

use crate::circuit::{Basis, Circuit};
use crate::dem::{compare, Dem};
use crate::dem_decoder::DemDecoder;
use crate::batch_sampler::BatchSampler;
use crate::memory::{generate, CodeKind, NoiseModel};

static mut TEXT: Vec<u8> = Vec::new();
static mut SLOT: Option<Circuit> = None;
static mut DECODE_ERRORS: usize = 0;
/// (code, d, rounds, noise, p bits, eta bits) and the compiled sampler and decoder.
#[allow(clippy::type_complexity)]
static mut CACHE: Option<((usize, usize, usize, usize, u64, u64), BatchSampler, DemDecoder)> = None;

/// Tests that drive the exports share one text buffer, so they take turns.
#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn text() -> &'static mut Vec<u8> {
    unsafe { &mut *std::ptr::addr_of_mut!(TEXT) }
}

pub(crate) fn reply(s: &str) -> usize {
    let t = text();
    t.clear();
    t.extend_from_slice(s.as_bytes());
    t.len()
}

pub(crate) fn json_error(e: &str) -> usize {
    let escaped = e.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ");
    reply(&format!("{{\"ok\":false,\"error\":\"{escaped}\"}}"))
}

fn noise_model(noise: usize, p: f64, eta: f64) -> NoiseModel {
    if noise == 1 {
        NoiseModel::Sd6 { p }
    } else {
        NoiseModel::Current { p, eta }
    }
}

fn code_kind(code: usize) -> CodeKind {
    if code == 1 {
        CodeKind::Xzzx
    } else {
        CodeKind::Rotated
    }
}

#[no_mangle]
pub extern "C" fn wasm_text_buf(len: usize) -> *mut u8 {
    let t = text();
    t.clear();
    t.resize(len, 0);
    t.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn wasm_text_ptr() -> *const u8 {
    text().as_ptr()
}

/// Generate a memory circuit into the slot and reply with its Stim text.
#[no_mangle]
pub extern "C" fn wasm_xc_generate(code: usize, d: usize, rounds: usize, noise: usize, p: f64, eta: f64, basis: usize) -> usize {
    let basis = if basis == 1 { Basis::X } else { Basis::Z };
    match generate(code_kind(code), d, rounds, noise_model(noise, p, eta), basis) {
        Ok(c) => {
            let s = c.to_stim();
            unsafe { *std::ptr::addr_of_mut!(SLOT) = Some(c) };
            reply(&s)
        }
        Err(e) => reply(&format!("ERROR: {e}")),
    }
}

/// Parse the buffer as a circuit into the slot.
#[no_mangle]
pub extern "C" fn wasm_xc_load_circuit() -> usize {
    let src = String::from_utf8_lossy(text()).into_owned();
    match Circuit::parse(&src) {
        Ok(c) => {
            unsafe { *std::ptr::addr_of_mut!(SLOT) = Some(c) };
            reply("{\"ok\":true}")
        }
        Err(e) => json_error(&e),
    }
}

/// Parse the buffer as Stim's DEM, build ours from the slot, and compare.
#[no_mangle]
pub extern "C" fn wasm_xc_compare() -> usize {
    let src = String::from_utf8_lossy(text()).into_owned();
    let theirs = match Dem::parse(&src) {
        Ok(d) => d,
        Err(e) => return json_error(&format!("reading Stim's model: {e}")),
    };
    let circuit = match unsafe { (*std::ptr::addr_of!(SLOT)).as_ref() } {
        Some(c) => c,
        None => return json_error("no circuit loaded"),
    };
    let ours = match Dem::from_circuit(circuit) {
        Ok(d) => d,
        Err(e) => return json_error(&e),
    };
    let c = compare(&ours, &theirs, 1e-9);
    let mut s = String::new();
    let _ = write!(
        s,
        "{{\"ok\":true,\"detectors\":{},\"ours\":{},\"theirs\":{},\"missing\":{},\"extra\":{},\"differing\":{},\"maxRel\":{:e}}}",
        ours.num_detectors, c.ours, c.theirs, c.missing, c.extra, c.differing, c.max_rel
    );
    reply(&s)
}

/// Sample `runs` shots of a memory-Z experiment on the general path, 64 at a
/// time with the bit-parallel sampler, and decode
/// them (or only sample, when `decode` is 0, so JS can time the two apart).
/// Returns the failure rate; a shot that fails to decode counts as a failure
/// and is also counted in `wasm_xc_decode_errors`.
#[no_mangle]
pub extern "C" fn wasm_xc_run(
    code: usize,
    d: usize,
    rounds: usize,
    noise: usize,
    p: f64,
    eta: f64,
    runs: usize,
    decode: u32,
) -> f64 {
    let key = (code, d, rounds, noise, p.to_bits(), eta.to_bits());
    let cache = unsafe { &mut *std::ptr::addr_of_mut!(CACHE) };
    if cache.as_ref().map(|c| c.0) != Some(key) {
        let built = generate(code_kind(code), d, rounds, noise_model(noise, p, eta), Basis::Z).and_then(|c| {
            let dem = Dem::from_circuit(&c)?;
            Ok((key, BatchSampler::new(&c)?, DemDecoder::new(&dem)?))
        });
        match built {
            Ok(b) => *cache = Some(b),
            Err(_) => {
                unsafe { *std::ptr::addr_of_mut!(DECODE_ERRORS) = runs };
                return f64::NAN;
            }
        }
    }
    let (_, sampler, decoder) = cache.as_ref().unwrap();
    let mut rng = crate::surface_code::next_shot_rng();
    let (mut failures, mut errors) = (0usize, 0usize);
    // 64 shots a batch, the last one partial; only the lanes asked for are decoded.
    let mut done = 0usize;
    while done < runs {
        let batch = sampler.sample(&mut rng);
        let lanes = (runs - done).min(64);
        done += lanes;
        if decode == 0 {
            continue;
        }
        let defects = batch.all_defects();
        for (lane, shot) in defects.iter().enumerate().take(lanes) {
            match decoder.decode(shot) {
                Ok(pred) => failures += ((pred.observables ^ batch.lane_observables(lane)) & 1) as usize,
                Err(_) => {
                    errors += 1;
                    failures += 1;
                }
            }
        }
    }
    unsafe { *std::ptr::addr_of_mut!(DECODE_ERRORS) = errors };
    if runs == 0 {
        0.0
    } else {
        failures as f64 / runs as f64
    }
}

#[no_mangle]
pub extern "C" fn wasm_xc_decode_errors() -> usize {
    unsafe { *std::ptr::addr_of!(DECODE_ERRORS) }
}
