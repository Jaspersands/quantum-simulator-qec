"""Record ldpc's mbp.hpp, built from tools/mbp_reference.cpp, on the cases of tests/data/mbp_reference.json."""
import json, subprocess, sys, numpy as np
harness, out_path = sys.argv[1], sys.argv[2]
def run_ref(m, n, entries, ch, al, beta, gamma, method, max_iter, syndromes):
    lines = [f"{m} {n} {len(entries)}"] + [f"{r} {c} {v}" for r, c, v in entries]
    lines += [" ".join(repr(float(x)) for x in row) for row in ch] + [" ".join(repr(float(x)) for x in row) for row in al]
    lines += [f"{beta!r} {gamma!r} {method} {max_iter} {len(syndromes)}"] + [" ".join(map(str, s)) for s in syndromes]
    out = subprocess.run([harness], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout.split("\n")
    res = []
    for t in range(len(syndromes)):
        blk = out[5 * t: 5 * t + 5]
        res.append(dict(decoding=list(map(int, blk[0].split())), converge=bool(int(blk[1].split()[0])), iterations=int(blk[1].split()[1]), llr=[[float(x) for x in blk[2 + w].split()] for w in range(3)]))
    return res
def hgp(rng):
    a = (rng.random((int(rng.integers(2, 4)), int(rng.integers(3, 5)))) < 0.5).astype(np.uint8)
    b = (rng.random((int(rng.integers(2, 4)), int(rng.integers(3, 5)))) < 0.5).astype(np.uint8)
    ma, na = a.shape; mb, nb = b.shape
    hx = np.hstack([np.kron(a, np.eye(nb, dtype=np.uint8)), np.kron(np.eye(ma, dtype=np.uint8), b.T)]) % 2
    hz = np.hstack([np.kron(np.eye(na, dtype=np.uint8), b), np.kron(a.T, np.eye(mb, dtype=np.uint8))]) % 2
    return hx, hz
rng = np.random.default_rng(2026)
cases = []
while len(cases) < 16:
    gf4 = len(cases) % 2 == 1
    if gf4:
        m, n = int(rng.integers(3, 8)), int(rng.integers(5, 14))
        h = np.where(rng.random((m, n)) < 0.35, rng.integers(1, 4, (m, n)), 0)
        if (h != 0).sum(0).min() == 0: continue
        entries = [(int(r), int(c), int(h[r, c])) for r, c in zip(*np.nonzero(h))]
        case = dict(Hgf4=h.tolist())
    else:
        hx, hz = hgp(rng)
        if hx.sum() == 0 or hz.sum() == 0: continue
        m, n = hx.shape[0] + hz.shape[0], hx.shape[1]
        entries = [(int(r), int(c), 3) for r, c in zip(*np.nonzero(hz))] + [(int(r) + hz.shape[0], int(c), 1) for r, c in zip(*np.nonzero(hx))]
        case = dict(HX_CSS=hx.tolist(), HZ_CSS=hz.tolist())
    method = len(cases) // 2 % 2
    p = float(rng.uniform(0.02, 0.1)); bias = [float(x) for x in rng.uniform(0.2, 1, 3)]
    alpha = float(rng.choice([1.0, 0.8, 0.65, 1.2])); beta = float(rng.choice([0.0, 0.3])); gamma = float(rng.choice([1.0, 0.625])); max_iter = int(rng.integers(2, 10))
    nb = np.asarray(bias) / np.sum(np.asarray(bias))  # as ldpc's Python normalises it
    ch = [[float(nb[w] * p)] * n for w in range(3)]
    al = [[alpha] * n for _ in range(3)]
    syndromes = []
    for _ in range(3):
        e = np.where(rng.random(n) < 0.12, rng.integers(1, 4, n), 0)
        s = [sum(1 for (r, c, v) in entries if r == i and e[c] != 0 and e[c] != v) % 2 for i in range(m)]
        syndromes.append(s)
    case.update(error_rate=p, xyz_bias=bias, alpha_parameter=alpha, beta_parameter=beta, gamma_parameter=gamma, bp_method=method, max_iter=max_iter, syndromes=syndromes, expected=run_ref(m, n, entries, ch, al, beta, gamma, method, max_iter, syndromes))
    cases.append(case)
json.dump(cases, open(out_path, "w"))
print(len(cases), sum(r["converge"] for c in cases for r in c["expected"]), "converged of", 3 * len(cases))
