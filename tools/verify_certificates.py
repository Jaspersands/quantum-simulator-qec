"""Re-derive conformance certificates from first principles, importing nothing from this package.

A certificate (stabilizer_qec.conformance.Certificate, as JSON, one per line or a JSON list)
claims that two corrections of one syndrome both explain it and that one weighs more, so the
heavier is not a minimum-weight correction (Shen and Zhong, arXiv 2609.07035, Lemma 5.3). This
program parses the certificate's model text itself, checks both corrections are in range,
unrepeated and flip exactly the syndrome's detectors, recomputes their weights
Σ ln((1 − p)/p), and checks the claimed order. It prints one line per certificate and exits
non-zero if any fails.

    python tools/verify_certificates.py certificates.jsonl
"""

import json
import math
import sys


def faults_of(model):
    out = []
    for line in model.splitlines():
        line = line.split("#")[0].strip()
        if not line.startswith("error("):
            continue
        p = float(line[len("error("):line.index(")")])
        dets = [int(t[1:]) for t in line[line.index(")") + 1:].split() if t.startswith("D")]
        out.append((p, dets))
    return out


def weight(faults, correction, fired):
    if len(set(correction)) != len(correction) or any(not 0 <= j < len(faults) for j in correction):
        return None
    flips = set()
    for j in correction:
        flips ^= set(faults[j][1])
    if flips != fired:
        return None
    return sum(math.log((1 - faults[j][0]) / faults[j][0]) for j in correction)


def verify(cert):
    faults = faults_of(cert["model"])
    fired = set(cert["syndrome"])
    a = weight(faults, list(cert["lighter"]), fired)
    b = weight(faults, list(cert["heavier"]), fired)
    if a is None or b is None:
        return False, "a correction does not explain the syndrome"
    if not b - a > 1e-9 * max(1.0, abs(b)):
        return False, f"the weights {a} and {b} are not in the claimed order"
    if abs(a - cert["lighter_weight"]) > 1e-6 * max(1.0, abs(a)) or abs(b - cert["heavier_weight"]) > 1e-6 * max(1.0, abs(b)):
        return False, "the claimed weights are not the corrections' weights"
    return True, f"{cert['subject']} on {cert['instance']} ({cert['relation']}): {b:.6g} > {a:.6g}"


def main(paths):
    certs = []
    for path in paths:
        text = open(path).read().strip()
        certs += json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    bad = 0
    for cert in certs:
        ok, why = verify(cert)
        bad += not ok
        print(("ok   " if ok else "FAIL ") + why)
    print(f"{len(certs) - bad} verified, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
