# modified for stabilizer_qec: its ldpc and PyMatching are this package's, and panqec is stabilizer_qec.panqec.
from importlib import reload
import pytest
import os
from stabilizer_qec import panqec as panqec
from stabilizer_qec.panqec.config import PANQEC_DIR, PANQEC_DARK_THEME


def test_output_dir_exists():
    assert os.path.exists(PANQEC_DIR)


def test_error_raised_if_panqec_dir_does_not_exist(monkeypatch):
    monkeypatch.setenv('PANQEC_DIR', '/fake/path/here/')
    with pytest.raises(FileNotFoundError):
        reload(panqec.config)


def test_dark_theme_known():
    assert type(PANQEC_DARK_THEME) is bool


def test_default_output_dir_is_named_for_the_user(monkeypatch, tmp_path):
    from stabilizer_qec.panqec import config
    monkeypatch.setattr(config.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(config.getpass, 'getuser', lambda: 'alice')
    assert config._default_panqec_dir() == os.path.join(str(tmp_path), 'panqec-alice')
    monkeypatch.setattr(config.getpass, 'getuser', lambda: 'bob')
    assert config._default_panqec_dir() == os.path.join(str(tmp_path), 'panqec-bob')

    def no_user_name():
        raise OSError('no login name')
    monkeypatch.setattr(config.getpass, 'getuser', no_user_name)
    assert config._default_panqec_dir() == os.path.join(str(tmp_path), 'panqec')
