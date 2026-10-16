# (c) Copyright Riverlane 2020-2025.
# modified for stabilizer_qec: the port is no distribution of its own; its version is deltakit-circuit's.
from stabilizer_qec.deltakit import circuit


def test_package_version_is_expected_version():
    # To ensure no false positives test this in tox rather than Poetry.
    assert circuit.__version__ == "0.9.2"
