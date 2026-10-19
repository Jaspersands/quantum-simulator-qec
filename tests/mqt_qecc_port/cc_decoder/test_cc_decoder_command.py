"""MQT QECC's ``mqt.qecc.cc-decoder`` script, run as ``stabilizer-qec mqt.qecc.cc-decoder`` and as
``python -m stabilizer_qec.mqt_qecc.cc_decoder``. (This package's tests, not MQT's; MQT's own run the
script through ``script_runner``, see ``test_cc_decoder.py``.)"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys

from stabilizer_qec import _cli
from stabilizer_qec.mqt_qecc.cc_decoder import HexagonalColorCode

from .test_utils import check_and_load_json

# MQT's result file name (its lattice type as Python formats the enum)
_RESULT = f"./code={HexagonalColorCode(distance=3).lattice_type},distance=3,p=0.1,solver=z3.json"


def _main(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = _cli.main(["mqt.qecc.cc-decoder", *argv])
    return code, out.getvalue(), err.getvalue()


def test_command_runs_the_decoder(tmp_path) -> None:
    code, _, err = _main("3", "0.1", "--nr_sims", "2", "--results_dir", str(tmp_path))
    assert code == 0, err
    result = check_and_load_json(_RESULT, str(tmp_path))
    assert result["distance"] == 3
    assert result["p"] == 0.1


def test_command_help_is_the_scripts() -> None:
    code, out, _ = _main("--help")
    assert code == 0
    assert out.startswith("usage: stabilizer-qec mqt.qecc.cc-decoder ")
    assert "--nr_sims" in out and "distance" in out and "--solver" in out


def test_command_usage_error_exits_2() -> None:
    code, _, err = _main()
    assert code == 2
    assert "the following arguments are required: distance, error_rate" in err


def test_command_is_listed() -> None:
    assert "mqt.qecc.cc-decoder" in _cli.help_for("commands")


def test_python_dash_m(tmp_path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "stabilizer_qec.mqt_qecc.cc_decoder", "3", "0.1", "--nr_sims", "1", "--results_dir", str(tmp_path)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=tmp_path,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / _RESULT).exists()
    proc = subprocess.run([sys.executable, "-m", "stabilizer_qec.mqt_qecc.cc_decoder", "--help"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert proc.stdout.startswith("usage: python -m stabilizer_qec.mqt_qecc.cc_decoder ")
