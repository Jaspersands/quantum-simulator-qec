//! WebAssembly surface for section 14: lattice surgery sampled and decoded in
//! the reader's browser.

use crate::batch_sampler::BatchSampler;
use crate::circuit::Basis;
use crate::dem::Dem;
use crate::dem_decoder::DemDecoder;
use crate::surface_code::next_shot_rng;
use crate::surgery::{cnot, Surgery};
use crate::wasm_xc::{json_error, reply};

struct Setup {
    sampler: BatchSampler,
    decoder: DemDecoder,
    correlated: bool,
    observables: usize,
}

static mut SETUP: Option<Setup> = None;

/// Build a lattice-surgery program at distance `d` with `merged` merged
/// rounds, SD6 noise `p`, and its matcher. `kind` 0: Z⊗Z on two patches in
/// |0⟩ (d rounds before and after); 1: the logical CNOT on |0⟩|0⟩; 2: the
/// CNOT on |+⟩|+⟩.
#[no_mangle]
pub extern "C" fn wasm_ls_setup(kind: u32, d: usize, merged: usize, p: f64, correlated: u32) -> usize {
    let built = (|| -> Result<(Setup, usize), String> {
        let circuit = match kind {
            0 => Surgery { d, pre: d, merged, post: d, basis: Basis::Z, p }.circuit()?,
            1 => cnot(d, merged, p, Basis::Z).circuit()?,
            2 => cnot(d, merged, p, Basis::X).circuit()?,
            other => return Err(format!("no lattice-surgery program of kind {other}")),
        };
        let dem = Dem::from_circuit(&circuit)?;
        dem.refuse_undetectable()?;
        let sampler = BatchSampler::new(&circuit)?;
        let decoder = DemDecoder::new(&dem)?;
        let observables = dem.num_observables;
        Ok((Setup { sampler, decoder, correlated: correlated != 0, observables }, dem.num_detectors))
    })();
    match built {
        Ok((s, detectors)) => {
            let observables = s.observables;
            unsafe { *std::ptr::addr_of_mut!(SETUP) = Some(s) };
            reply(&format!("{{\"ok\":true,\"detectors\":{detectors},\"observables\":{observables}}}"))
        }
        Err(e) => json_error(&e),
    }
}

/// Sample 64 shots and decode them. Replies with the shots that failed at
/// all, on each observable (`wrong`), and, for the Z⊗Z experiment's two
/// questions, on the merge outcome (L0) and on either patch (L1, L2).
#[no_mangle]
pub extern "C" fn wasm_ls_run() -> usize {
    let Some(s) = (unsafe { (*std::ptr::addr_of_mut!(SETUP)).as_mut() }) else {
        return json_error("no surgery set up");
    };
    let mut rng = next_shot_rng();
    let batch = s.sampler.sample(&mut rng);
    let mask = (1u64 << s.observables) - 1;
    let (mut any, mut outcome, mut patches) = (0usize, 0usize, 0usize);
    let mut each = vec![0usize; s.observables];
    for lane in 0..64 {
        let defects = batch.lane_defects(lane);
        let got = if s.correlated { s.decoder.decode_correlated(&defects) } else { s.decoder.decode(&defects) };
        let wrong = match got {
            Ok(p) => (p.observables ^ batch.lane_observables(lane)) & mask,
            Err(_) => mask,
        };
        any += usize::from(wrong != 0);
        outcome += usize::from(wrong & 1 != 0);
        patches += usize::from(wrong & !1 != 0);
        for (i, e) in each.iter_mut().enumerate() {
            *e += usize::from((wrong >> i) & 1 != 0);
        }
    }
    let wrong: Vec<String> = each.iter().map(|x| x.to_string()).collect();
    reply(&format!(
        "{{\"ok\":true,\"shots\":64,\"any\":{any},\"outcome\":{outcome},\"patches\":{patches},\"wrong\":[{}]}}",
        wrong.join(",")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm_xc::text;

    #[test]
    fn a_batch_decodes() {
        let _turn = crate::wasm_xc::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let len = wasm_ls_setup(0, 3, 3, 0.001, 1);
        let r = String::from_utf8(text()[..len].to_vec()).unwrap();
        assert!(r.contains("\"ok\":true"), "{r}");
        let len = wasm_ls_run();
        let r = String::from_utf8(text()[..len].to_vec()).unwrap();
        assert!(r.contains("\"shots\":64"), "{r}");
        let len = wasm_ls_setup(0, 3, 1, 0.001, 1);
        let r = String::from_utf8(text()[..len].to_vec()).unwrap();
        assert!(r.contains("undetectable"), "one merged round is refused: {r}");
        // The CNOT, both input pairs: two observables each.
        for kind in [1, 2] {
            let len = wasm_ls_setup(kind, 3, 3, 0.001, 1);
            let r = String::from_utf8(text()[..len].to_vec()).unwrap();
            assert!(r.contains("\"observables\":2"), "{r}");
            let len = wasm_ls_run();
            let r = String::from_utf8(text()[..len].to_vec()).unwrap();
            assert!(r.contains("\"shots\":64") && r.contains("\"wrong\":["), "{r}");
        }
    }
}
