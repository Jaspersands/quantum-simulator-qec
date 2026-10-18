// The lane program (src/lanes.rs) run on a GPU: one invocation per word of 32 shots, the
// same opcodes, the same integer draws in the same order as the CPU executor, so the two give
// the same bits. The program is interpreted from `code`; REPEAT blocks run on a small loop
// stack. Every word owns a slice of the frame, ring, detector, observable and record buffers.

struct Params {
    num_words: u32,
    num_qubits: u32,
    ring_mask: u32,
    num_detectors: u32,
    num_observables: u32,
    num_measurements: u32,
    keep_measurements: u32,
    code_len: u32,
}

@group(0) @binding(0) var<storage, read> code: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
// Four words of xoshiro128** state per word.
@group(0) @binding(2) var<storage, read> seeds: array<u32>;
// x frames then z frames per word: frames[word * 2 * nq + q] and frames[word * 2 * nq + nq + q].
@group(0) @binding(3) var<storage, read_write> frames: array<u32>;
@group(0) @binding(4) var<storage, read_write> ring: array<u32>;
// num_words * num_detectors words, then one DONE word per word that ran to its end.
@group(0) @binding(5) var<storage, read_write> dets: array<u32>;
@group(0) @binding(6) var<storage, read_write> obs: array<u32>;
@group(0) @binding(7) var<storage, read_write> meas: array<u32>;

const DONE: u32 = 0xC0DEu;
const OP_RESET: u32 = 1u;
const OP_H: u32 = 2u;
const OP_S: u32 = 3u;
const OP_CX: u32 = 4u;
const OP_CZ: u32 = 5u;
const OP_MEASURE: u32 = 6u;
const OP_PAULI_ERROR: u32 = 7u;
const OP_DEPOLARIZE1: u32 = 8u;
const OP_DEPOLARIZE2: u32 = 9u;
const OP_PAULI_CHANNEL1: u32 = 10u;
const OP_DETECTOR: u32 = 11u;
const OP_OBSERVABLE: u32 = 12u;
const OP_REPEAT: u32 = 13u;
const OP_CORRELATED: u32 = 14u;
const OP_PAULI_CHANNEL2: u32 = 15u;
const OP_PAD: u32 = 16u;
const OP_FEEDBACK: u32 = 17u;
const OP_HERALDED: u32 = 18u;

const PROB_NEVER: u32 = 0u;
const PROB_ALWAYS: u32 = 1u;

const STACK: u32 = 16u;

// xoshiro128** state, as `lanes::Rng32`.
var<private> s0: u32;
var<private> s1: u32;
var<private> s2: u32;
var<private> s3: u32;
var<private> xbase: u32;
var<private> zbase: u32;
var<private> rbase: u32;
var<private> rmask: u32;
var<private> m: u32;
var<private> chain: u32;
var<private> det_base: u32;
var<private> det: u32;
var<private> obs_base: u32;
var<private> meas_base: u32;
var<private> keep: bool;

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn next() -> u32 {
    let result = rotl(s1 * 5u, 7u) * 9u;
    let t = s1 << 9u;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotl(s3, 11u);
    return result;
}

fn bernoulli(kind: u32, thr: u32) -> u32 {
    if kind == PROB_NEVER {
        return 0u;
    }
    if kind == PROB_ALWAYS {
        return 0xFFFFFFFFu;
    }
    var w = 0u;
    for (var lane = 0u; lane < 32u; lane++) {
        if next() <= thr {
            w |= 1u << lane;
        }
    }
    return w;
}

fn reset(basis: u32, q: u32) {
    if basis == 0u {
        frames[xbase + q] = 0u;
        frames[zbase + q] = next();
    } else {
        frames[zbase + q] = 0u;
        frames[xbase + q] = next();
    }
}

fn pauli_on_lane(q: u32, lane: u32, pauli: u32) {
    let bit = 1u << lane;
    if (pauli & 1u) != 0u {
        frames[xbase + q] ^= bit;
    }
    if (pauli & 2u) != 0u {
        frames[zbase + q] ^= bit;
    }
}

fn pauli_word(q: u32, pauli: u32, w: u32) {
    if (pauli & 1u) != 0u {
        frames[xbase + q] ^= w;
    }
    if (pauli & 2u) != 0u {
        frames[zbase + q] ^= w;
    }
}

fn record(w: u32) {
    ring[rbase + (m & rmask)] = w;
    if keep {
        meas[meas_base + m] = w;
    }
    m += 1u;
}

fn lookback(k: u32) -> u32 {
    return ring[rbase + ((m - k) & rmask)];
}

fn lowest(w: u32) -> u32 {
    return countTrailingZeros(w);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.x;
    if word >= params.num_words {
        return;
    }
    let nq = params.num_qubits;
    s0 = seeds[4u * word];
    s1 = seeds[4u * word + 1u];
    s2 = seeds[4u * word + 2u];
    s3 = seeds[4u * word + 3u];
    xbase = word * 2u * nq;
    zbase = xbase + nq;
    rmask = params.ring_mask;
    rbase = word * (rmask + 1u);
    m = 0u;
    chain = 0u;
    det_base = word * params.num_detectors;
    det = 0u;
    obs_base = word * params.num_observables;
    meas_base = word * params.num_measurements;
    keep = params.keep_measurements != 0u;

    // Every qubit starts in |0>, to which Z is invisible: the x frame 0, the z frame random.
    for (var q = 0u; q < nq; q++) {
        frames[xbase + q] = 0u;
        frames[zbase + q] = next();
    }
    for (var i = 0u; i < params.num_observables; i++) {
        obs[obs_base + i] = 0u;
    }

    // The loop stack: where a REPEAT body starts, where it ends, how many runs remain.
    var stack_start: array<u32, 16>;
    var stack_end: array<u32, 16>;
    var stack_left: array<u32, 16>;
    var depth = 0u;

    var pc = 0u;
    let end = params.code_len;
    loop {
        // A finished loop body: run it again, or pop it.
        while depth > 0u && pc == stack_end[depth - 1u] {
            if stack_left[depth - 1u] > 1u {
                stack_left[depth - 1u] -= 1u;
                pc = stack_start[depth - 1u];
            } else {
                depth -= 1u;
            }
        }
        if pc >= end {
            break;
        }
        let op = code[pc];
        pc += 1u;
        switch op {
            case OP_RESET: {
                let basis = code[pc];
                let n = code[pc + 1u];
                pc += 2u;
                for (var k = 0u; k < n; k++) {
                    reset(basis, code[pc + k]);
                }
                pc += n;
            }
            case OP_H: {
                let n = code[pc];
                pc += 1u;
                for (var k = 0u; k < n; k++) {
                    let q = code[pc + k];
                    let x = frames[xbase + q];
                    frames[xbase + q] = frames[zbase + q];
                    frames[zbase + q] = x;
                }
                pc += n;
            }
            case OP_S: {
                let n = code[pc];
                pc += 1u;
                for (var k = 0u; k < n; k++) {
                    let q = code[pc + k];
                    frames[zbase + q] ^= frames[xbase + q];
                }
                pc += n;
            }
            case OP_CX: {
                let n = code[pc];
                pc += 1u;
                for (var k = 0u; k < n; k++) {
                    let c = code[pc + 2u * k];
                    let t = code[pc + 2u * k + 1u];
                    frames[xbase + t] ^= frames[xbase + c];
                    frames[zbase + c] ^= frames[zbase + t];
                }
                pc += 2u * n;
            }
            case OP_CZ: {
                let n = code[pc];
                pc += 1u;
                for (var k = 0u; k < n; k++) {
                    let a = code[pc + 2u * k];
                    let b = code[pc + 2u * k + 1u];
                    frames[zbase + b] ^= frames[xbase + a];
                    frames[zbase + a] ^= frames[xbase + b];
                }
                pc += 2u * n;
            }
            case OP_MEASURE: {
                let basis = code[pc];
                let do_reset = code[pc + 1u] != 0u;
                let kind = code[pc + 2u];
                let thr = code[pc + 3u];
                let n = code[pc + 4u];
                pc += 5u;
                for (var k = 0u; k < n; k++) {
                    let q = code[pc + k];
                    var rec = 0u;
                    if basis == 0u {
                        rec = frames[xbase + q];
                    } else {
                        rec = frames[zbase + q];
                    }
                    rec ^= bernoulli(kind, thr);
                    record(rec);
                    if basis == 0u {
                        frames[zbase + q] = next();
                    } else {
                        frames[xbase + q] = next();
                    }
                    if do_reset {
                        reset(basis, q);
                    }
                }
                pc += n;
            }
            case OP_PAULI_ERROR: {
                let pauli = code[pc];
                let kind = code[pc + 1u];
                let thr = code[pc + 2u];
                let n = code[pc + 3u];
                pc += 4u;
                for (var k = 0u; k < n; k++) {
                    let w = bernoulli(kind, thr);
                    pauli_word(code[pc + k], pauli, w);
                }
                pc += n;
            }
            case OP_DEPOLARIZE1: {
                let kind = code[pc];
                let thr = code[pc + 1u];
                let n = code[pc + 2u];
                pc += 3u;
                for (var k = 0u; k < n; k++) {
                    let q = code[pc + k];
                    var w = bernoulli(kind, thr);
                    while w != 0u {
                        let lane = lowest(w);
                        w &= w - 1u;
                        let pauli = 1u + next() % 3u;
                        pauli_on_lane(q, lane, pauli);
                    }
                }
                pc += n;
            }
            case OP_DEPOLARIZE2: {
                let kind = code[pc];
                let thr = code[pc + 1u];
                let n = code[pc + 2u];
                pc += 3u;
                for (var k = 0u; k < n; k++) {
                    let a = code[pc + 2u * k];
                    let b = code[pc + 2u * k + 1u];
                    var w = bernoulli(kind, thr);
                    while w != 0u {
                        let lane = lowest(w);
                        w &= w - 1u;
                        let r = 1u + next() % 15u;
                        pauli_on_lane(a, lane, r & 3u);
                        pauli_on_lane(b, lane, r >> 2u);
                    }
                }
                pc += 2u * n;
            }
            case OP_PAULI_CHANNEL1: {
                let kind = code[pc];
                let thr = code[pc + 1u];
                let cx = code[pc + 2u];
                let cxy = code[pc + 3u];
                let n = code[pc + 4u];
                pc += 5u;
                for (var k = 0u; k < n; k++) {
                    let q = code[pc + k];
                    var w = bernoulli(kind, thr);
                    while w != 0u {
                        let lane = lowest(w);
                        w &= w - 1u;
                        let u = next();
                        var pauli = 2u;
                        if u < cx {
                            pauli = 1u;
                        } else if u < cxy {
                            pauli = 3u;
                        }
                        pauli_on_lane(q, lane, pauli);
                    }
                }
                pc += n;
            }
            case OP_DETECTOR: {
                let n = code[pc];
                pc += 1u;
                var w = 0u;
                for (var k = 0u; k < n; k++) {
                    w ^= lookback(code[pc + k]);
                }
                dets[det_base + det] = w;
                det += 1u;
                pc += n;
            }
            case OP_OBSERVABLE: {
                let i = code[pc];
                let n = code[pc + 1u];
                pc += 2u;
                var w = 0u;
                for (var k = 0u; k < n; k++) {
                    w ^= lookback(code[pc + k]);
                }
                pc += n;
                let mm = code[pc];
                pc += 1u;
                for (var k = 0u; k < mm; k++) {
                    let q = code[pc + 2u * k];
                    let p = code[pc + 2u * k + 1u];
                    if (p & 1u) != 0u {
                        w ^= frames[zbase + q];
                    }
                    if (p & 2u) != 0u {
                        w ^= frames[xbase + q];
                    }
                }
                pc += 2u * mm;
                obs[obs_base + i] ^= w;
            }
            case OP_REPEAT: {
                let count = code[pc];
                let len = code[pc + 1u];
                pc += 2u;
                if count == 0u || len == 0u || depth >= STACK {
                    pc += len;
                } else {
                    stack_start[depth] = pc;
                    stack_end[depth] = pc + len;
                    stack_left[depth] = count;
                    depth += 1u;
                }
            }
            case OP_CORRELATED: {
                let kind = code[pc];
                let thr = code[pc + 1u];
                let chained = code[pc + 2u] != 0u;
                let n = code[pc + 3u];
                pc += 4u;
                var w = bernoulli(kind, thr);
                if chained {
                    w &= ~chain;
                    chain |= w;
                } else {
                    chain = w;
                }
                for (var k = 0u; k < n; k++) {
                    pauli_word(code[pc + 2u * k], code[pc + 2u * k + 1u], w);
                }
                pc += 2u * n;
            }
            case OP_PAULI_CHANNEL2: {
                let kind = code[pc];
                let thr = code[pc + 1u];
                let cuts = pc + 2u;
                pc += 17u;
                let n = code[pc];
                pc += 1u;
                for (var k = 0u; k < n; k++) {
                    let a = code[pc + 2u * k];
                    let b = code[pc + 2u * k + 1u];
                    var w = bernoulli(kind, thr);
                    while w != 0u {
                        let lane = lowest(w);
                        w &= w - 1u;
                        let u = next();
                        var cas = 15u;
                        for (var j = 0u; j < 15u; j++) {
                            if u < code[cuts + j] {
                                cas = j + 1u;
                                break;
                            }
                        }
                        pauli_on_lane(a, lane, pauli_code(cas / 4u));
                        pauli_on_lane(b, lane, pauli_code(cas % 4u));
                    }
                }
                pc += 2u * n;
            }
            case OP_PAD: {
                let kind = code[pc];
                let thr = code[pc + 1u];
                let count = code[pc + 2u];
                pc += 3u;
                for (var k = 0u; k < count; k++) {
                    record(bernoulli(kind, thr));
                }
            }
            case OP_FEEDBACK: {
                let pauli = code[pc];
                let k = code[pc + 1u];
                let q = code[pc + 2u];
                pc += 3u;
                pauli_word(q, pauli, lookback(k));
            }
            case OP_HERALDED: {
                let kind = code[pc];
                let thr = code[pc + 1u];
                let c0 = code[pc + 2u];
                let c1 = code[pc + 3u];
                let c2 = code[pc + 4u];
                let n = code[pc + 5u];
                pc += 6u;
                for (var k = 0u; k < n; k++) {
                    let q = code[pc + k];
                    let w = bernoulli(kind, thr);
                    record(w);
                    var lanes = w;
                    while lanes != 0u {
                        let lane = lowest(lanes);
                        lanes &= lanes - 1u;
                        let u = next();
                        var pauli = 2u;
                        if u < c0 {
                            pauli = 0u;
                        } else if u < c1 {
                            pauli = 1u;
                        } else if u < c2 {
                            pauli = 3u;
                        }
                        pauli_on_lane(q, lane, pauli);
                    }
                }
                pc += n;
            }
            default: {
                // An unknown opcode: stop this word (the CPU reference would have refused it).
                pc = end;
            }
        }
    }
    // The word ran to its end: a dispatch the driver cut short leaves this unwritten.
    dets[params.num_words * params.num_detectors + word] = DONE;
}

// Stim's case order I, X, Y, Z as the frame's bits (x = 1, z = 2).
fn pauli_code(k: u32) -> u32 {
    switch k {
        case 0u: { return 0u; }
        case 1u: { return 1u; }
        case 2u: { return 3u; }
        default: { return 2u; }
    }
}
