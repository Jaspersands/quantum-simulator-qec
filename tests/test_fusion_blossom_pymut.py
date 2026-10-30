"""fb.PyMut refuses misuse with RuntimeError, not with an assert (which python -O removes), and a
field that is None counts as entered."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import stabilizer_qec.fusion_blossom as fb


class Obj:
    pass


def obj(value):
    o = Obj()
    o.field = value
    return o


def test_double_enter_raises():
    m = fb.PyMut(obj([1]), "field")
    with pytest.raises(RuntimeError, match="enter twice"):
        with m:
            m.__enter__()


def test_double_enter_of_a_none_field_raises():
    m = fb.PyMut(obj(None), "field")
    with pytest.raises(RuntimeError, match="enter twice"):
        with m:
            m.__enter__()


def test_exit_without_enter_raises_and_keeps_the_field():
    o = obj([9])
    with pytest.raises(RuntimeError):
        fb.PyMut(o, "field").__exit__(None, None, None)
    assert o.field == [9]


def test_guards_hold_under_python_O():
    code = textwrap.dedent('''
        import stabilizer_qec.fusion_blossom as fb

        class Obj:
            pass

        def fail(message):
            raise SystemExit(message)

        o = Obj()
        o.field = [1]
        m = fb.PyMut(o, "field")
        try:
            with m:
                m.__enter__()
        except RuntimeError:
            pass
        else:
            fail("double enter allowed")
        o.field = [9]
        try:
            fb.PyMut(o, "field").__exit__(None, None, None)
        except RuntimeError:
            pass
        else:
            fail("exit without enter allowed")
        if o.field != [9]:
            fail("exit without enter wrote the field")
    ''')
    subprocess.run([sys.executable, "-O", "-c", code], check=True, capture_output=True)
