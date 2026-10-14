"""Editing a detector error model keeps its numbers exact (its printed text rounds them to Stim's
16 digits, which earlier edits went through)."""

import stabilizer_qec as sq


def test_numbers_survive_edits():
    p = 0.050887600767594354  # a merged probability: all 17 digits matter
    dem = sq.DetectorErrorModel()
    dem.append("error", [p], [sq.target_relative_detector_id(3)])
    dem.append("detector", [1e-20, 2.5], [sq.target_relative_detector_id(0)])
    assert dem[0].args_copy() == [p] and dem[1].args_copy() == [1e-20, 2.5]
    assert str(dem) == "error(0.05088760076759435) D3\ndetector(9.999999999999999e-21, 2.5) D0"
    dem.append("logical_observable", [], [sq.target_logical_observable_id(0)])
    assert dem[0].args_copy() == [p]
    assert dem.copy()[0].args_copy() == [p] and dem.flattened()[0].args_copy() == [p]
