# Ported from qecsim 1.0b9 (qecsim/qecsim; David K. Tuckett, BSD 3-Clause; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: qecsim is stabilizer_qec.qecsim.
import pytest

from stabilizer_qec.qecsim import util


def test_init_logging_fall_through(monkeypatch):
    # logging fall-through
    monkeypatch.setenv('QECSIM_LOGGING_CFG', 'blah.ini')
    util.init_logging()  # no error raised


def test_chunker_iterator():
    # test with iterator of iterables
    my_length = 10
    my_iterator = ((i, i) for i in range(my_length))
    my_chunk_len = 3
    count = 0
    for chunk in util.chunker(my_iterator, my_chunk_len):
        chunk = list(chunk)
        print(chunk)
        assert len(chunk) <= my_chunk_len, 'Invalid chunk length'
        expected_chunk = [(i, i) for i in range(count * my_chunk_len, min((count + 1) * my_chunk_len, my_length))]
        assert chunk == expected_chunk
        count += 1


def test_chunker_iterable():
    # test with iterable of iterables
    my_length = 10
    my_iterable = [(i, i) for i in range(my_length)]
    my_chunk_len = 3
    count = 0
    for chunk in util.chunker(my_iterable, my_chunk_len):
        chunk = list(chunk)
        print(chunk)
        assert len(chunk) <= my_chunk_len, 'Invalid chunk length'
        expected_chunk = [(i, i) for i in range(count * my_chunk_len, min((count + 1) * my_chunk_len, my_length))]
        assert chunk == expected_chunk
        count += 1


def _hostile_dir(directory):
    # A directory someone else supplied: a logging config whose handler arguments run code, and a
    # "Blossom V library" that is not one.
    (directory / 'logging_qecsim.ini').write_text(
        '[loggers]\nkeys=root\n[handlers]\nkeys=h\n[formatters]\nkeys=\n'
        '[logger_root]\nhandlers=h\n'
        "[handler_h]\nclass=StreamHandler\nargs=(__import__('pathlib').Path('MARKER').write_text('ran') and sys.stderr,)\n")
    (directory / 'clib').mkdir()
    (directory / 'clib' / 'libpypm.so').write_bytes(b'not a library')


def _set_home(monkeypatch, home):
    # Move ~ to home. Setting HOME is not enough: on Windows Path.home() reads USERPROFILE and ignores HOME, so the
    # runner's real ~/.qecsim would be searched.
    from pathlib import Path

    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: cls(home)))
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))


@pytest.mark.parametrize('qecsim_cfg', [None, ''])
def test_working_directory_config_not_loaded(tmp_path, monkeypatch, qecsim_cfg):
    # stabilizer_qec: the configuration comes from $QECSIM_CFG or ~/.qecsim only, never from the
    # working directory (qecsim's ./logging_qecsim.ini ran the code in its handler arguments, and
    # its ./clib/libpypm.so was loaded in place of the matcher). An empty QECSIM_CFG, which qecsim
    # took as the working directory and which a CI env entry for an undefined variable gives, counts
    # as unset.
    import logging
    from pathlib import Path

    work, home = tmp_path / 'work', tmp_path / 'home'
    work.mkdir()
    home.mkdir()
    _hostile_dir(work)
    if qecsim_cfg is None:
        monkeypatch.delenv('QECSIM_CFG', raising=False)
    else:
        monkeypatch.setenv('QECSIM_CFG', qecsim_cfg)
    _set_home(monkeypatch, home)
    monkeypatch.chdir(work)
    root = logging.getLogger()
    handlers = list(root.handlers)
    try:
        util.init_logging()
    finally:
        for h in [h for h in root.handlers if h not in handlers]:
            root.removeHandler(h)
    assert not (work / 'MARKER').exists()
    for name in ('logging_qecsim.ini', Path('clib', 'libpypm.so')):
        with pytest.raises(FileNotFoundError):
            util._resolve_path(name)


def test_home_and_env_config_resolve(tmp_path, monkeypatch):
    # stabilizer_qec: ~/.qecsim, and $QECSIM_CFG when it is set, still resolve.
    home, cfg = tmp_path / 'home', tmp_path / 'cfg'
    (home / '.qecsim').mkdir(parents=True)
    cfg.mkdir()
    (home / '.qecsim' / 'logging_qecsim.ini').write_text('')
    (cfg / 'logging_qecsim.ini').write_text('')
    _set_home(monkeypatch, home)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('QECSIM_CFG', raising=False)
    assert util._resolve_path('logging_qecsim.ini') == home / '.qecsim' / 'logging_qecsim.ini'
    monkeypatch.setenv('QECSIM_CFG', '')
    assert util._resolve_path('logging_qecsim.ini') == home / '.qecsim' / 'logging_qecsim.ini'
    monkeypatch.setenv('QECSIM_CFG', str(cfg))
    assert util._resolve_path('logging_qecsim.ini') == cfg / 'logging_qecsim.ini'
    # A relative QECSIM_CFG is read relative to the working directory, as in qecsim (SECURITY.md says to give an
    # absolute one).
    monkeypatch.setenv('QECSIM_CFG', 'cfg')
    assert util._resolve_path('logging_qecsim.ini').resolve() == (cfg / 'logging_qecsim.ini').resolve()
