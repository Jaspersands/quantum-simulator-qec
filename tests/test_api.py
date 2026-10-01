"""The package's surface: what is public, what is deprecated, and how errors are raised."""

from __future__ import annotations

import warnings

import pytest

import stabilizer_qec as sq


def test_public_names_resolve():
    for name in sq.__all__:
        assert getattr(sq, name) is not None
    assert sq.__version__.count(".") >= 2


@pytest.mark.parametrize("name", sorted(sq._DEPRECATED))
def test_deprecated_names_warn_and_still_work(name):
    with pytest.warns(DeprecationWarning, match=name):
        obj = getattr(sq, name)
    assert obj is getattr(sq._core, name)


def test_deprecated_function_still_runs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        text = sq.generate_circuit("rotated", 3, 3, "sd6", 0.001)
    assert sq.Circuit(text).num_detectors == 24


def test_unknown_names_are_attribute_errors():
    with pytest.raises(AttributeError):
        sq.no_such_thing  # noqa: B018


def test_panics_become_runtime_errors():
    from stabilizer_qec._util import call

    class PanicException(BaseException):
        pass

    def boom():
        raise PanicException("index out of bounds")

    with pytest.raises(RuntimeError, match="internal error"):
        call(boom)


def test_huge_integers_are_value_errors():
    with pytest.raises(ValueError):
        sq.surgery.line(3, n=2**70, merged=3, p=0.001)
    with pytest.raises(ValueError):
        sq.memory_circuit(distance=2**64, rounds=3, p=0.001)


def test_huge_and_odd_numbers_are_value_or_type_errors():
    import math

    for p in (10**400, math.nan, -math.inf):
        with pytest.raises(ValueError):
            sq.memory_circuit(distance=3, rounds=3, p=p)
    with pytest.raises(ValueError):
        sq.memory_circuit(distance=3, rounds=3, p=0.001, noise="current", eta=10**400)
    with pytest.raises(TypeError):
        sq.memory_circuit(distance=3, rounds=3, p="0.001")
    with pytest.raises(ValueError):
        sq.BpDecoder([[1, 1]], error_rate=1.5)
