//! WebAssembly surface for section 13: a bivariate bicycle code's memory,
//! sampled and decoded by BP+OSD in the reader's browser.
//!
//! The page times each call; `wasm_bb_run` samples 64 shots with the batch
//! sampler and decodes them, so its time is mostly the decoder's.

use crate::batch_sampler::BatchSampler;
use crate::bb::BbCode;
use crate::bp::Method;
use crate::circuit::Circuit;
use crate::dem::Dem;
use crate::osd::{BpOsd, BpOsdWork, OsdMethod};
use crate::surface_code::next_shot_rng;
use crate::wasm_xc::{json_error, reply};

struct Setup {
    sampler: BatchSampler,
    decoder: BpOsd,
    work: BpOsdWork,
    observables: Vec<u64>,
    num_detectors: usize,
}

static mut SETUP: Option<Setup> = None;

/// Build `code` (0: [[72, 12, 6]]; 1: the gross code [[144, 12, 12]]) as the
/// paper's Z-basis memory of `cycles` cycles at circuit noise `p`, its
/// undecomposed error model, and a BP+OSD-CS(7) decoder with min-sum BP
/// (adaptive scaling) for up to `max_iter` iterations.
#[no_mangle]
pub extern "C" fn wasm_bb_setup(code: u32, cycles: usize, p: f64, max_iter: usize) -> usize {
    let code = if code == 1 {
        BbCode::gross()
    } else {
        BbCode::bb72()
    };
    let built = (|| -> Result<Setup, String> {
        let circuit = Circuit::parse(&code.memory_z(cycles, p))?;
        let dem = Dem::from_circuit_undecomposed(&circuit)?;
        let sampler = BatchSampler::new(&circuit)?;
        let columns: Vec<Vec<u32>> = dem.mechanisms.iter().map(|m| m.detectors.clone()).collect();
        let priors: Vec<f64> = dem.mechanisms.iter().map(|m| m.p).collect();
        let observables = dem.mechanisms.iter().map(|m| m.observables).collect();
        let decoder = BpOsd::new(
            dem.num_detectors,
            columns,
            &priors,
            Method::MinSum { scale: 0.0 },
            max_iter,
            OsdMethod::CombinationSweep(7),
        )?;
        let work = decoder.work();
        Ok(Setup {
            sampler,
            decoder,
            work,
            observables,
            num_detectors: dem.num_detectors,
        })
    })();
    match built {
        Ok(s) => {
            let text = format!(
                "{{\"ok\":true,\"detectors\":{},\"faults\":{},\"qubits\":{}}}",
                s.num_detectors,
                s.observables.len(),
                4 * code.half()
            );
            unsafe { *std::ptr::addr_of_mut!(SETUP) = Some(s) };
            reply(&text)
        }
        Err(e) => json_error(&e),
    }
}

/// Sample 64 shots and decode each. Replies with the shots on which any of
/// the 12 logical qubits was predicted wrongly, and on how many BP converged
/// without OSD.
#[no_mangle]
pub extern "C" fn wasm_bb_run() -> usize {
    let Some(s) = (unsafe { (*std::ptr::addr_of_mut!(SETUP)).as_mut() }) else {
        return json_error("no code set up");
    };
    let mut rng = next_shot_rng();
    let batch = s.sampler.sample(&mut rng);
    let mut syndrome = vec![0u8; s.num_detectors];
    let (mut failures, mut converged) = (0usize, 0usize);
    for lane in 0..64 {
        syndrome.fill(0);
        for d in batch.lane_defects(lane) {
            syndrome[d as usize] = 1;
        }
        let out = s.decoder.decode(&syndrome, &mut s.work);
        converged += usize::from(out.converged);
        let predicted = s
            .work
            .correction
            .iter()
            .enumerate()
            .filter(|x| *x.1 != 0)
            .fold(0u64, |a, (v, _)| a ^ s.observables[v]);
        failures += usize::from(predicted != batch.lane_observables(lane));
    }
    reply(&format!(
        "{{\"ok\":true,\"shots\":64,\"failures\":{failures},\"converged\":{converged}}}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm_xc::text;

    fn reply_text(len: usize) -> String {
        String::from_utf8(text()[..len].to_vec()).unwrap()
    }

    #[test]
    fn a_batch_decodes() {
        let _turn = crate::wasm_xc::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let r = reply_text(wasm_bb_setup(0, 2, 0.002, 200));
        assert!(
            r.contains("\"ok\":true") && r.contains("\"qubits\":144"),
            "{r}"
        );
        let r = reply_text(wasm_bb_run());
        assert!(r.contains("\"shots\":64"), "{r}");
        let failures: usize = r
            .split("\"failures\":")
            .nth(1)
            .unwrap()
            .split(',')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(failures < 16, "{r}");
    }
}
