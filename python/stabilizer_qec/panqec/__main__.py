# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec; its
# command line is `stabilizer-qec panqec`, which the cluster scripts it writes call.
"""
Command line interface.
"""

from multiprocessing import freeze_support
from .cli import cli


if __name__ == '__main__':
    freeze_support()
    cli(prog_name="python -m stabilizer_qec.panqec")
