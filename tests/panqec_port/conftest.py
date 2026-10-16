# modified for stabilizer_qec: PanQEC's `--runslow` option (which pytest takes only from the
# rootdir's conftest) is the environment variable PANQEC_RUNSLOW, and applies to its tests only.
import os
import pathlib

import pytest

_HERE = pathlib.Path(__file__).parent


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PANQEC_RUNSLOW"):
        # PANQEC_RUNSLOW set: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="needs PANQEC_RUNSLOW=1 to run")
    for item in items:
        if "slow" in item.keywords and _HERE in pathlib.Path(str(item.fspath)).parents:
            item.add_marker(skip_slow)
