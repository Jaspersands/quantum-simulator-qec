# modified for stabilizer_qec: its Stim and ldpc are this package's, and quits is stabilizer_qec.quits.
import numpy as np

PCM = __import__('os').path.join(__import__('os').path.dirname(__file__), 'parity_check_matrices')

from stabilizer_qec.quits.qldpc_code import HgpCode


def test_cardinalNSmerge_builds_ns_group_and_depth():
    h = np.loadtxt(
        PCM + "/n=12_dv=3_dc=4_dist=6.txt",
        dtype=int,
    )
    code = HgpCode(h, h)

    code.build_circuit(
        strategy="cardinalNSmerge",
        num_rounds=0,
        basis="Z",
        seed=1,
    )

    assert hasattr(code, "colored_edges_NS")
    assert set(code.num_colors.keys()) == {"E", "NS", "W"}

    flattened_ns_edges = [
        edge
        for color in sorted(code.colored_edges_NS.keys())
        for edge in code.colored_edges_NS[color]
    ]
    assert len(flattened_ns_edges) == len(code.edges_N) + len(code.edges_S)
    assert code.depth == code.num_colors["E"] + code.num_colors["NS"] + code.num_colors["W"]
