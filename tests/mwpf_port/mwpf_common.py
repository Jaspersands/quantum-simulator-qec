# MWPF 0.2.12's tests/python/common.py, for stabilizer_qec.mwpf: the package imported as `mwpf`, and a
# temporary folder for the HTML files the tests write.
import os, math, pytest, sys, tempfile

import stabilizer_qec.mwpf as mwpf

OUT_DIR = tempfile.mkdtemp(prefix="mwpf_port_")


def circle_positions(n: int):
    positions = []
    for i in range(n):
        positions.append(
            mwpf.VisualizePosition(
                0.5 + 0.5 * math.cos(2 * math.pi * i / n),
                0.5 + 0.5 * math.sin(2 * math.pi * i / n),
                0,
            )
        )
    return positions
