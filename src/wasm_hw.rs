//! WebAssembly surface for section 11: Google's raw Willow readouts turned into
//! detection events and decoded in the reader's browser.
//!
//! Text (circuits, error models, JSON replies) crosses through `wasm_xc`'s text
//! buffer. Binary rows (measurements, sweep bits, detection events,
//! predictions) cross through a second buffer here, in Stim's b8 layout. Two
//! decoder slots hold the two priors decoded side by side.

use crate::circuit::Circuit;
use crate::dem::Dem;
use crate::dem_decoder::DemDecoder;
use crate::m2d::M2d;
use crate::wasm_xc::{json_error, reply, text};

static mut BYTES: Vec<u8> = Vec::new();
static mut DECODERS: [Option<DemDecoder>; 2] = [None, None];

fn bytes() -> &'static mut Vec<u8> {
    unsafe { &mut *std::ptr::addr_of_mut!(BYTES) }
}

fn decoders() -> &'static mut [Option<DemDecoder>; 2] {
    unsafe { &mut *std::ptr::addr_of_mut!(DECODERS) }
}

fn text_string() -> String {
    String::from_utf8_lossy(text()).into_owned()
}

#[no_mangle]
pub extern "C" fn wasm_bytes_buf(len: usize) -> *mut u8 {
    let b = bytes();
    b.clear();
    b.resize(len, 0);
    b.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn wasm_bytes_ptr() -> *const u8 {
    bytes().as_ptr()
}

/// The text buffer holds the ideal circuit; the byte buffer, `num_shots`
/// measurement rows followed by as many sweep-bit rows. On success the byte
/// buffer holds the detection-event rows followed by the observable rows, and
/// the reply says how many bytes each takes.
#[no_mangle]
pub extern "C" fn wasm_hw_m2d(num_shots: usize) -> usize {
    let circuit = match Circuit::parse(&text_string()) {
        Ok(c) => c,
        Err(e) => return json_error(&e),
    };
    let m = match M2d::new(&circuit) {
        Ok(m) => m,
        Err(e) => return json_error(&e),
    };
    let b = bytes();
    let split = m.num_measurements.div_ceil(8) * num_shots;
    if split > b.len() {
        return json_error("fewer bytes than the measurements need");
    }
    let (dets, obs) = match m.convert_b8(&b[..split], &b[split..], num_shots) {
        Ok(x) => x,
        Err(e) => return json_error(&e),
    };
    b.clear();
    b.extend_from_slice(&dets);
    b.extend_from_slice(&obs);
    reply(&format!(
        "{{\"ok\":true,\"detectors\":{},\"observables\":{},\"detBytes\":{},\"obsBytes\":{}}}",
        m.num_detectors,
        m.num_observables,
        dets.len(),
        obs.len()
    ))
}

fn install(slot: usize, dem: Result<Dem, String>) -> usize {
    if slot >= 2 {
        return json_error("there are two decoder slots");
    }
    let dem = match dem {
        Ok(d) => d,
        Err(e) => return json_error(&e),
    };
    match DemDecoder::new(&dem) {
        Ok(dec) => {
            let reply_text = format!(
                "{{\"ok\":true,\"detectors\":{},\"mechanisms\":{},\"rules\":{}}}",
                dem.num_detectors,
                dem.mechanisms.len(),
                dec.correlations().num_rules()
            );
            decoders()[slot] = Some(dec);
            reply(&reply_text)
        }
        Err(e) => json_error(&e),
    }
}

/// Decoder `slot` from our own model of the noisy circuit in the text buffer.
#[no_mangle]
pub extern "C" fn wasm_hw_model_circuit(slot: usize) -> usize {
    install(slot, Circuit::parse(&text_string()).and_then(|c| Dem::from_circuit(&c)))
}

/// Decoder `slot` from the detector error model text in the text buffer.
#[no_mangle]
pub extern "C" fn wasm_hw_model_dem(slot: usize) -> usize {
    install(slot, Dem::parse(&text_string()))
}

/// Decode the detection-event rows in the byte buffer with decoder `slot`,
/// plainly or with correlated matching. The byte buffer then holds one byte per
/// shot: observable 0's predicted flip, or 255 where the shot could not be
/// decoded, which the reply counts.
#[no_mangle]
pub extern "C" fn wasm_hw_decode(slot: usize, num_shots: usize, correlated: u32) -> usize {
    let Some(dec) = decoders().get(slot).and_then(|d| d.as_ref()) else {
        return json_error("no model in that slot");
    };
    let nd = dec.num_detectors();
    let stride = nd.div_ceil(8);
    let b = bytes();
    if b.len() != stride * num_shots {
        return json_error(&format!("{} bytes is not {num_shots} shots of {nd} detectors", b.len()));
    }
    let mut out = Vec::with_capacity(num_shots);
    let mut defects = Vec::new();
    let mut errors = 0usize;
    for s in 0..num_shots {
        let row = &b[s * stride..(s + 1) * stride];
        defects.clear();
        defects.extend((0..nd).filter(|&i| (row[i / 8] >> (i % 8)) & 1 == 1).map(|i| i as u32));
        let result = if correlated != 0 { dec.decode_correlated(&defects) } else { dec.decode(&defects) };
        out.push(match result {
            Ok(p) => (p.observables & 1) as u8,
            Err(_) => {
                errors += 1;
                255
            }
        });
    }
    *b = out;
    reply(&format!("{{\"ok\":true,\"errors\":{errors}}}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_text(s: &str) {
        let t = text();
        t.clear();
        t.extend_from_slice(s.as_bytes());
    }

    fn set_bytes(v: &[u8]) {
        let b = bytes();
        b.clear();
        b.extend_from_slice(v);
    }

    fn reply_text(len: usize) -> String {
        String::from_utf8(text()[..len].to_vec()).unwrap()
    }

    /// The whole path the live panel takes, on a two-qubit circuit: raw
    /// readouts to detection events, a model, and predictions.
    #[test]
    fn readouts_to_predictions() {
        let _turn = crate::wasm_xc::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_text("R 0 1\nCX sweep[0] 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]\n");
        // Shots: meas (1, 0) with the sweep bit set; meas (1, 1) with it clear.
        set_bytes(&[0b01, 0b11, 1, 0]);
        let r = reply_text(wasm_hw_m2d(2));
        assert!(r.contains("\"ok\":true") && r.contains("\"detBytes\":2") && r.contains("\"obsBytes\":2"), "{r}");
        // The second shot read qubit 0 as 1 with its sweep bit clear: D0 and
        // the observable both flipped.
        assert_eq!(bytes().as_slice(), &[0b00, 0b01, 0, 1]);

        set_text("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n");
        assert!(reply_text(wasm_hw_model_dem(0)).contains("\"ok\":true"));
        set_bytes(&[0b01, 0b00]);
        let r = reply_text(wasm_hw_decode(0, 2, 0));
        assert_eq!(r, "{\"ok\":true,\"errors\":0}");
        assert_eq!(bytes().as_slice(), &[1, 0]);

        set_bytes(&[0]);
        assert!(reply_text(wasm_hw_decode(0, 2, 1)).contains("\"ok\":false"));
        assert!(reply_text(wasm_hw_decode(1, 1, 0)).contains("no model"));
    }
}
