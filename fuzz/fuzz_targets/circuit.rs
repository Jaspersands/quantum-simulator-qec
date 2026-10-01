//! Any bytes as circuit text, through every stage that reads a circuit.
//! Run: cargo +nightly fuzz run circuit
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    stabilizer_qec::fuzzing::exercise_circuit(&String::from_utf8_lossy(data));
});
