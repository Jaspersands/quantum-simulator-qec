// Min-sum belief propagation in f32, the parallel schedule, one invocation per shot: the same
// operations in the same order as src/bp32.rs, so the two give the same bits. The graph is
// shared (var_start, edge_check, edge_var, chk_start, chk_edges, back to back in `graph`), the
// priors and the per-iteration scales too; every shot owns a slice of the messages, the
// posteriors and the work buffer.

struct Params {
    num_shots: u32,
    num_checks: u32,
    num_vars: u32,
    num_edges: u32,
    max_iter: u32,
    // Offsets into `graph`.
    off_var_start: u32,
    off_edge_check: u32,
    off_edge_var: u32,
    off_chk_start: u32,
    off_chk_edges: u32,
    // Offset of the scales in `floats` (the priors come first).
    off_scales: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> graph: array<u32>;
@group(0) @binding(2) var<storage, read> floats: array<f32>;
// One u32 per (shot, check): 1 where the check fired.
@group(0) @binding(3) var<storage, read> syndromes: array<u32>;
// Per shot: b2c over the edges, then c2b over the edges.
@group(0) @binding(4) var<storage, read_write> messages: array<f32>;
@group(0) @binding(5) var<storage, read_write> llr: array<f32>;
// Per shot: the hard decision per variable, the candidate syndrome per check, then
// (converged, iterations).
@group(0) @binding(6) var<storage, read_write> work: array<u32>;

const F32_MAX: f32 = 3.4028234663852886e38;

fn var_start(v: u32) -> u32 {
    return graph[params.off_var_start + v];
}
fn edge_check(e: u32) -> u32 {
    return graph[params.off_edge_check + e];
}
fn edge_var(e: u32) -> u32 {
    return graph[params.off_edge_var + e];
}
fn chk_start(c: u32) -> u32 {
    return graph[params.off_chk_start + c];
}
fn chk_edge(k: u32) -> u32 {
    return graph[params.off_chk_edges + k];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let shot = id.x;
    if shot >= params.num_shots {
        return;
    }
    let nv = params.num_vars;
    let nc = params.num_checks;
    let ne = params.num_edges;
    let b2c = shot * 2u * ne;
    let c2b = b2c + ne;
    let lbase = shot * nv;
    let hard = shot * (nv + nc + 3u);
    let cand = hard + nv;
    let mbase = cand + nc;
    let sbase = shot * nc;

    for (var v = 0u; v < nv; v++) {
        let p = floats[v];
        for (var e = var_start(v); e < var_start(v + 1u); e++) {
            messages[b2c + e] = p;
        }
    }
    var converged = 0u;
    var iterations = params.max_iter;
    for (var it = 0u; it < params.max_iter; it++) {
        let scale = floats[params.off_scales + it];
        // Checks.
        for (var c = 0u; c < nc; c++) {
            let start = chk_start(c);
            let end = chk_start(c + 1u);
            var total_sign = syndromes[sbase + c];
            var temp = F32_MAX;
            for (var k = start; k < end; k++) {
                let e = chk_edge(k);
                messages[c2b + e] = temp;
                let m = messages[b2c + e];
                if m <= 0.0 {
                    total_sign += 1u;
                }
                temp = min(temp, abs(m));
            }
            temp = F32_MAX;
            for (var k = end; k > start; k--) {
                let e = chk_edge(k - 1u);
                let m = messages[b2c + e];
                var sign = total_sign;
                if m <= 0.0 {
                    sign += 1u;
                }
                if temp < messages[c2b + e] {
                    messages[c2b + e] = temp;
                }
                var s = 1.0;
                if (sign % 2u) != 0u {
                    s = -1.0;
                }
                messages[c2b + e] = messages[c2b + e] * (s * scale);
                temp = min(temp, abs(m));
            }
        }
        // Variables: posteriors and the decision, each edge primed by the prefix sum.
        for (var v = 0u; v < nv; v++) {
            var temp = floats[v];
            for (var e = var_start(v); e < var_start(v + 1u); e++) {
                messages[b2c + e] = temp;
                temp = temp + messages[c2b + e];
            }
            llr[lbase + v] = temp;
            if temp <= 0.0 {
                work[hard + v] = 1u;
            } else {
                work[hard + v] = 0u;
            }
        }
        for (var c = 0u; c < nc; c++) {
            work[cand + c] = 0u;
        }
        for (var e = 0u; e < ne; e++) {
            work[cand + edge_check(e)] ^= work[hard + edge_var(e)];
        }
        var ok = true;
        for (var c = 0u; c < nc; c++) {
            if work[cand + c] != syndromes[sbase + c] {
                ok = false;
                break;
            }
        }
        if ok {
            converged = 1u;
            iterations = it + 1u;
            break;
        }
        for (var v = 0u; v < nv; v++) {
            var temp = 0.0;
            for (var e = var_start(v + 1u); e > var_start(v); e--) {
                messages[b2c + e - 1u] = messages[b2c + e - 1u] + temp;
                temp = temp + messages[c2b + e - 1u];
            }
        }
    }
    work[mbase] = converged;
    work[mbase + 1u] = iterations;
    // The shot ran to its end: a dispatch the driver dropped leaves this unwritten.
    work[mbase + 2u] = 0xC0DEu;
}
