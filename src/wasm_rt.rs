//! WebAssembly surface for section 12: window decoding of simulated memory
//! streams in the reader's browser, beside global decoding of the same streams.
//!
//! The engine has no clock of its own (it imports nothing), so the page times
//! each call. `wasm_rt_windows` samples the streams as they are decoded, as a
//! real stream arrives, with its scratch space made once at setup; the second
//! draw of the same shots, kept whole for global decoding, is made in
//! `wasm_rt_global`, outside the window decoder's time.

use crate::batch_sampler::BatchSampler;
use crate::circuit::Basis;
use crate::dem::Dem;
use crate::dem_decoder::DemDecoder;
use crate::memory::{generate_repeat, CodeKind};
use crate::sparse::Scratch;
use crate::stream::{run_stream, StreamDecoder, StreamPlan};
use crate::surface_code::{next_shot_rng, Xorshift};
use crate::wasm_xc::{json_error, reply};
use crate::window::Mode;

struct Setup {
    stream: StreamDecoder,
    plan: StreamPlan,
    sampler: BatchSampler,
    global: DemDecoder,
    correlated: bool,
    scratches: Vec<Scratch>,
    /// The last batch's generator, before it drew, and its window predictions:
    /// global decoding draws the same shots again.
    last: Option<(Xorshift, [u64; 64])>,
}

static mut SETUP: Option<Setup> = None;

fn setup() -> Option<&'static mut Setup> {
    unsafe { (*std::ptr::addr_of_mut!(SETUP)).as_mut() }
}

/// Build the window decoder, its plan, a sampler and a global decoder for a
/// rotated SD6 memory. Replies with the windows and layers.
#[no_mangle]
pub extern "C" fn wasm_rt_setup(
    d: usize,
    rounds: usize,
    p: f64,
    commit: usize,
    buffer: usize,
    parallel: u32,
    correlated: u32,
) -> usize {
    let mode = if parallel != 0 {
        Mode::Parallel
    } else {
        Mode::Sliding
    };
    let built = (|| -> Result<Setup, String> {
        let stream = StreamDecoder::new(
            CodeKind::Rotated,
            d,
            p,
            Basis::Z,
            commit,
            buffer,
            mode,
            rounds,
        )?;
        let plan = stream.plan()?;
        let circuit = generate_repeat(CodeKind::Rotated, d, rounds, p, Basis::Z)?;
        let sampler = BatchSampler::new(&circuit)?;
        let global = DemDecoder::new(&Dem::from_circuit(&circuit)?)?;
        let scratches = stream.scratches();
        Ok(Setup {
            stream,
            plan,
            sampler,
            global,
            correlated: correlated != 0,
            scratches,
            last: None,
        })
    })();
    match built {
        Ok(s) => {
            let text = format!(
                "{{\"ok\":true,\"windows\":{},\"layers\":{},\"template\":{},\"detectors\":{}}}",
                s.plan.specs.len(),
                s.plan.layers,
                s.stream.template_layers,
                s.sampler.num_detectors
            );
            unsafe { *std::ptr::addr_of_mut!(SETUP) = Some(s) };
            reply(&text)
        }
        Err(e) => json_error(&e),
    }
}

/// Sample 64 streams and window-decode them as they stream. Replies with the
/// failures and the defects left unexplained.
#[no_mangle]
pub extern "C" fn wasm_rt_windows() -> usize {
    let Some(s) = setup() else {
        return json_error("no stream set up");
    };
    let mut rng = next_shot_rng();
    let before = rng.clone();
    match run_stream(
        &s.stream,
        &s.plan,
        &s.sampler,
        &mut rng,
        s.correlated,
        &mut s.scratches,
        None,
    ) {
        Ok((pred, truth, unexplained, _)) => {
            let failures = (0..64)
                .filter(|&l| ((pred[l] ^ (truth >> l)) & 1) == 1)
                .count();
            s.last = Some((before, pred));
            reply(&format!(
                "{{\"ok\":true,\"failures\":{failures},\"unexplained\":{unexplained}}}"
            ))
        }
        Err(e) => json_error(&format!("{e:?}")),
    }
}

/// Decode the last batch's 64 streams globally, all rounds at once. Replies
/// with the global failures and on how many streams the two decoders agree.
#[no_mangle]
pub extern "C" fn wasm_rt_global() -> usize {
    let Some(s) = setup() else {
        return json_error("no stream set up");
    };
    let Some((before, pred)) = s.last.as_ref() else {
        return json_error("no batch decoded yet");
    };
    // The same draws the streams were made from, this time kept whole.
    let batch = s.sampler.sample(&mut before.clone());
    let (mut failures, mut agree) = (0usize, 0usize);
    let defects = batch.all_defects();
    for (lane, shot) in defects.iter().enumerate() {
        let got = if s.correlated {
            s.global.decode_correlated(shot)
        } else {
            s.global.decode(shot)
        };
        match got {
            Ok(p) => {
                failures += ((p.observables ^ batch.lane_observables(lane)) & 1) as usize;
                agree += usize::from((p.observables ^ pred[lane]) & 1 == 0);
            }
            Err(e) => return json_error(&format!("{e:?}")),
        }
    }
    reply(&format!(
        "{{\"ok\":true,\"failures\":{failures},\"agree\":{agree}}}"
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
    fn a_small_stream_decodes_like_the_global_decoder() {
        let _turn = crate::wasm_xc::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let r = reply_text(wasm_rt_setup(3, 30, 0.006, 3, 3, 1, 0));
        assert!(r.contains("\"ok\":true"), "{r}");
        let w = reply_text(wasm_rt_windows());
        assert!(w.contains("\"unexplained\":0"), "{w}");
        let g = reply_text(wasm_rt_global());
        assert!(g.contains("\"ok\":true"), "{g}");
        let agree: usize = g
            .split("\"agree\":")
            .nth(1)
            .unwrap()
            .trim_end_matches('}')
            .parse()
            .unwrap();
        assert!(agree >= 60, "{g}");
    }
}
