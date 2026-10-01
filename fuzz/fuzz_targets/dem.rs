//! Any bytes as detector-error-model text, through the parser and every decoder.
//! Run: cargo +nightly fuzz run dem
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    stabilizer_qec::fuzzing::exercise_dem(&String::from_utf8_lossy(data));
});
