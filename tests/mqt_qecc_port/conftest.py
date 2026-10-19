"""Fixtures MQT QECC's tests take from pytest plugins: ``script_runner`` (pytest-console-scripts),
running MQT's command line in this process under the name its tests give it, and ``mocker``
(pytest-mock) where that plugin is not installed."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import sys
import traceback
from dataclasses import dataclass
from unittest import mock

import pytest

# MQT QECC's console scripts (its `[project.scripts]`), by the names its tests run them as.
_SCRIPTS = {"mqt.qecc.cc-decoder": "stabilizer_qec.mqt_qecc.cc_decoder.cli:cli"}


@dataclass
class _Result:
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


class _ScriptRunner:
    """pytest-console-scripts' runner in its in-process mode, for MQT QECC's scripts."""

    def run(self, command: list[str], *args: str) -> _Result:
        name, *argv = [*command, *args]
        module, _, func = _SCRIPTS[name].partition(":")
        main = getattr(importlib.import_module(module), func)
        out, err = io.StringIO(), io.StringIO()
        code = 0
        saved = sys.argv
        sys.argv = [name, *argv]
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    main()
                except SystemExit as e:
                    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
                except Exception:  # noqa: BLE001 - reported as the script's failure, as a process's would be
                    traceback.print_exc()
                    code = 1
        finally:
            sys.argv = saved
        return _Result(code, out.getvalue(), err.getvalue())


@pytest.fixture
def script_runner() -> _ScriptRunner:
    return _ScriptRunner()


if importlib.util.find_spec("pytest_mock") is None:

    class _Mocker:
        def __init__(self) -> None:
            self._patches: list = []

        def patch(self, target: str, *args, **kwargs):  # noqa: ANN002, ANN003
            p = mock.patch(target, *args, **kwargs)
            self._patches.append(p)
            return p.start()

        def stopall(self) -> None:
            for p in reversed(self._patches):
                p.stop()
            self._patches.clear()

    @pytest.fixture
    def mocker():  # noqa: ANN201
        m = _Mocker()
        yield m
        m.stopall()


def pytest_configure(config: pytest.Config) -> None:
    # pytest-xdist's mark, which MQT's tests use to keep tests that share files on one worker.
    config.addinivalue_line("markers", "xdist_group(name): run these tests on one pytest-xdist worker")
