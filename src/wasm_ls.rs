//! WebAssembly surface for section 14: lattice surgery sampled and decoded in
//! the reader's browser.

use crate::batch_sampler::BatchSampler;
use crate::circuit::Basis;
use crate::dem::Dem;
use crate::dem_decoder::DemDecoder;
use crate::surface_code::next_shot_rng;
use crate::surgery::Surgery;
use crate::wasm_xc::{json_error, reply};

struct Setup {
    sampler: BatchSampler,
    decoder: DemDecoder,
    correlated: bool,
}

static mut SETUP: Option<Setup> = None;

/// Build Z-basis lattice surgery at distance `d` with `merged` merged rounds
/// (d rounds before and after), SD6 noise `p`, and its matcher.
#[no_mangle]
pub extern "C" fn wasm_ls_setup(d: usize, merged: usize, p: f64, correlated: u32) -> usize {
    let built = (|| -> Result<(Setup, usize), String> {
        let circuit = Surgery { d, pre: d, merged, post: d, basis: Basis::Z, p }.circuit()?;
        let dem = Dem::from_circuit(&circuit)?;
        let sampler = BatchSampler::new(&circuit)?;
        let decoder = DemDecoder::new(&dem)?;
        Ok((Setup { sampler, decoder, correlated: correlated != 0 }, dem.num_detectors))
    })();
    match built {
        Ok((s, detectors)) => {
            unsafe { *std::ptr::addr_of_mut!(SETUP) = Some(s) };
            reply(&format!("{{\"ok\":true,\"detectors\":{detectors}}}"))
        }
        Err(e) => json_error(&e),
    }
}

/// Sample 64 shots and decode them. Replies with the shots that failed at
/// all, on the merge outcome (L0), and on either patch (L1, L2).
#[no_mangle]
pub extern "C" fn wasm_ls_run() -> usize {
    let Some(s) = (unsafe { (*std::ptr::addr_of_mut!(SETUP)).as_mut() }) else {
        return json_error("no surgery set up");
    };
    let mut rng = next_shot_rng();
    let batch = s.sampler.sample(&mut rng);
    let (mut any, mut outcome, mut patches) = (0usize, 0usize, 0usize);
    for lane in 0..64 {
        let defects = batch.lane_defects(lane);
        let got = if s.correlated { s.decoder.decode_correlated(&defects) } else { s.decoder.decode(&defects) };
        let wrong = match got {
            Ok(p) => (p.observables ^ batch.lane_observables(lane)) & 0b111,
            Err(_) => 0b111,
        };
        any += usize::from(wrong != 0);
        outcome += usize::from(wrong & 1 != 0);
        patches += usize::from(wrong & 0b110 != 0);
    }
    reply(&format!("{{\"ok\":true,\"shots\":64,\"any\":{any},\"outcome\":{outcome},\"patches\":{patches}}}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm_xc::text;

    #[test]
    fn a_batch_decodes() {
        let _turn = crate::wasm_xc::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let len = wasm_ls_setup(3, 3, 0.001, 1);
        let r = String::from_utf8(text()[..len].to_vec()).unwrap();
        assert!(r.contains("\"ok\":true"), "{r}");
        let len = wasm_ls_run();
        let r = String::from_utf8(text()[..len].to_vec()).unwrap();
        assert!(r.contains("\"shots\":64"), "{r}");
        let len = wasm_ls_setup(3, 1, 0.001, 1);
        let r = String::from_utf8(text()[..len].to_vec()).unwrap();
        assert!(r.contains("undetectable"), "one merged round is refused: {r}");
    }
}
