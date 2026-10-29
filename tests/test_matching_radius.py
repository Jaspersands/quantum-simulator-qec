"""Long uniform paths must finish even when scaled radius offsets exceed i32.

This is a deliberate divergence from the reference: PyMatching 2.4.0's
``DetectorNode::compute_wrapped_radius`` returns ``int32_t``, so on these chains (131 nodes or more,
at weight 1.5 or 16,777,214.5) its decoder never finishes, while 130 nodes decode. The port's radius
offsets are i64 (src/pm_exact/flooder.rs), and the CHANGELOG lists the divergence."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("nodes", [130, 131, 300, 1000])
@pytest.mark.parametrize("weight", [1.5, 16_777_214.5])
def test_uniform_chain_large_radius(nodes, weight):
    # Isolate the decoder: a radius regression used to loop forever, including after SIGINT.
    code = '''
import sys
import numpy as np
import stabilizer_qec as sq
n, weight = int(sys.argv[1]), float(sys.argv[2])
m = sq.Matching()
expected = np.zeros(5, dtype=np.uint8)
for i in range(n - 1):
    m.add_edge(i, i + 1, fault_ids={i % 5}, weight=weight)
    expected[i % 5] ^= 1
syndrome = np.zeros(n, dtype=np.uint8)
syndrome[0] = syndrome[-1] = 1
prediction, total = m.decode(syndrome, return_weight=True)
assert np.array_equal(prediction, expected), (prediction, expected)
assert total == (n - 1) * weight, total
syndrome[-1] = 0
try:
    m.decode(syndrome)
except ValueError as error:
    assert "perfect matching" in str(error).lower(), error
else:
    raise AssertionError("odd syndrome without boundary was accepted")
syndrome[-1] = 1
prediction, total = m.decode(syndrome, return_weight=True)
assert np.array_equal(prediction, expected)
assert total == (n - 1) * weight
'''
    result = subprocess.run([sys.executable, "-c", code, str(nodes), str(weight)],
                            env=os.environ.copy(), capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
