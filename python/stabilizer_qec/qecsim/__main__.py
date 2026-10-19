# Ported from qecsim 1.0b9 (qecsim/qecsim; David K. Tuckett, BSD 3-Clause; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: qecsim is stabilizer_qec.qecsim, and its command line `stabilizer-qec qecsim`.
"""
This module exposes the qecsim command line interface (CLI) via module script.
"""

from stabilizer_qec.qecsim.cli import cli

if __name__ == '__main__':
    cli(prog_name="python -m stabilizer_qec.qecsim")
