//! Circuit texts shared by the tests of several modules.

/// Every construct the parser supports, in a circuit that is also a valid
/// (deterministic) two-qubit parity check.
pub const SAMPLE: &str = "\
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
QUBIT_COORDS(2, 0) 2
R 0 1 2
X_ERROR(0.001) 0 1 2
TICK
CX 0 1
DEPOLARIZE2(0.001) 0 1
CX 2 1
DEPOLARIZE2(0.001) 2 1
X_ERROR(0.001) 1
MR 1
DETECTOR(1, 0, 0) rec[-1]
REPEAT 2 {
    TICK
    DEPOLARIZE1(0.001) 0 2
    CX 0 1 2 1
    MR(0.001) 1
    SHIFT_COORDS(0, 0, 1)
    DETECTOR(1, 0, 0) rec[-1] rec[-2]
}
M 0 2
DETECTOR(1, 0, 1) rec[-1] rec[-2] rec[-3]
OBSERVABLE_INCLUDE(0) rec[-1]
";

/// A distance-3 repetition code (data 0, 2, 4; ancillas 1, 3), two rounds of
/// circuit noise at 0.01, then a transversal readout. Distance 3 against X
/// errors, so every single fault must be corrected.
pub const REP3: &str = "\
R 0 1 2 3 4
TICK
CX 0 1 2 3
DEPOLARIZE2(0.01) 0 1 2 3
TICK
CX 2 1 4 3
DEPOLARIZE2(0.01) 2 1 4 3
TICK
X_ERROR(0.01) 1 3
MR 1 3
DETECTOR(1, 0, 0) rec[-2]
DETECTOR(3, 0, 0) rec[-1]
TICK
DEPOLARIZE1(0.01) 0 2 4
CX 0 1 2 3
DEPOLARIZE2(0.01) 0 1 2 3
TICK
CX 2 1 4 3
DEPOLARIZE2(0.01) 2 1 4 3
TICK
X_ERROR(0.01) 1 3
MR 1 3
DETECTOR(1, 0, 1) rec[-2] rec[-4]
DETECTOR(3, 0, 1) rec[-1] rec[-3]
TICK
X_ERROR(0.01) 0 2 4
M 0 2 4
DETECTOR(1, 0, 2) rec[-3] rec[-2] rec[-5]
DETECTOR(3, 0, 2) rec[-2] rec[-1] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-1]
";
