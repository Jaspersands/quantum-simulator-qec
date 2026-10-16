# (c) Copyright Riverlane 2020-2025.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
import pytest

from stabilizer_qec.deltakit.circuit._parse_stim import group_targets


@pytest.mark.parametrize(
    ("qubit_indices", "expected_groups"),
    [
        ([0, 1, 2, 3], [{0, 1, 2, 3}]),
        ([0, 1, 2, 2], [{0, 1, 2}, {2}]),
        ([0, 1, 2, 1, 2, 2], [{0, 1, 2}, {1, 2}, {2}]),
    ],
)
def test_grouping_single_duplicated_item_returns_two_sets(
    qubit_indices, expected_groups
):
    assert group_targets(qubit_indices) == expected_groups
