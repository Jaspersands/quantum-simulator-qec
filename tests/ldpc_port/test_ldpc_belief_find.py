# ldpc's own test (Copyright (c) 2022 Joschka Roffe, MIT License; see THIRD_PARTY_NOTICES.md),
# from its repository's python_test/, run against stabilizer_qec.ldpc: imports modified.
import pytest
from stabilizer_qec.ldpc.belief_find_decoder import BeliefFindDecoder
from stabilizer_qec.ldpc.codes import hamming_code, rep_code, ring_code


def test_peeling_input():
    pcm = rep_code(3)
    BeliefFindDecoder(pcm, error_rate=0.1, uf_method="peeling")

    pcm = ring_code(3)
    BeliefFindDecoder(pcm, error_rate=0.1, uf_method="peeling")

    pcm = hamming_code(3)

    with pytest.raises(ValueError):
        BeliefFindDecoder(pcm, error_rate=0.1, uf_method="peeling")


if __name__ == "__main__":
    test_peeling_input()
