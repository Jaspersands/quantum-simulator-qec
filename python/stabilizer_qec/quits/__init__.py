"""QUITS package.

Known difference: QUITS' ``examples/circuit_distance_search.py`` (Stim's
``Circuit.search_for_undetectable_logical_errors`` on a BPC [[72,8,8]] cardinal circuit, limits
6, 6) finds the same four errors on this package's Stim as on Stim, but the search, which is this
package's and not QUITS', takes about three times Stim's memory (456 MB peak against 139 MB) and
five times its time (8.7 s against 1.6 s; measured on 2026-10-19, macOS arm64).
"""

# modified for stabilizer_qec: QUITS 1.2.0 (MIT, Kang, Lin, Yao and Gökduman), ported whole on
# this package's Stim and ldpc; its version is the one ported.
__version__ = "1.2.0"

from .api import *
