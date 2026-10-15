# MWPF 0.2.12's tests/python/test_solver_config_backward_compatibility.py, run against stabilizer_qec.mwpf: imports changed, and files
# written to a temporary folder.
from mwpf_common import *


def test_flatten_config():
    code = mwpf.CodeCapacityColorCode(d=5, p=0.01)
    config = {
        "timeout": 6.0,
        "cluster_node_limit": 3,
    }
    mwpf.Solver(code.get_initializer(), config)


def test_legacy_config():
    code = mwpf.CodeCapacityColorCode(d=5, p=0.01)
    config = {
        "primal": {
            "timeout": 6.0,
            "cluster_node_limit": 3,
        }
    }
    mwpf.Solver(code.get_initializer(), config)
