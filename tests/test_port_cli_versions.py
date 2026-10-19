"""The ported command lines report their versions under the upstream program names, as PanQEC's and
qecsim's own do (``panqec, version 0.1.7``): the import rename once made them
``stabilizer_qec.panqec`` and ``stabilizer_qec.qecsim``."""

import pytest

click_testing = pytest.importorskip("click.testing")


@pytest.mark.parametrize("module, name", [("stabilizer_qec.panqec.cli", "panqec"), ("stabilizer_qec.qecsim.cli", "qecsim")])
def test_version_names_the_upstream_program(module, name):
    try:
        cli_module = __import__(module, fromlist=["cli"])
    except ImportError as e:  # the panqec or qecsim extra is not installed
        pytest.skip(str(e))
    version = __import__(module.rsplit(".", 1)[0], fromlist=["__version__"]).__version__
    result = click_testing.CliRunner().invoke(cli_module.cli, ["--version"])
    assert result.exit_code == 0
    assert result.output == f"{name}, version {version}\n"
