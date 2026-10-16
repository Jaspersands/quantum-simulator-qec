# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from __future__ import annotations

import numpy as np
import pytest

from stabilizer_qec.deltakit.explorer.data import write_binary_data_to_file
from stabilizer_qec.deltakit.explorer.enums._api_enums import DataFormat
from stabilizer_qec.deltakit.explorer.types._types import Measurements


@pytest.mark.parametrize("destination", [DataFormat.B8, DataFormat.CSV, DataFormat.F01])
def test_converter_file_serialisation(destination, tmp_path):
    data = [[1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
    path = tmp_path / f"file.{destination.value}"
    write_binary_data_to_file(
        data=data,
        destination_format=destination,
        result_file=path,
    )
    np.allclose(Measurements(path, destination, 4).as_numpy(), data)


def test_converter_file_serialisation_raises(tmp_path):
    data = [[1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
    path = tmp_path / "file.unknown"
    with pytest.raises(NotImplementedError):
        write_binary_data_to_file(
            data=data,
            destination_format="UNKNOWN",
            result_file=path,
        )
