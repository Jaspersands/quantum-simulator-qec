"""The package's surface: what is public, what is deprecated, and how errors are raised."""

from __future__ import annotations

import importlib
import sys

import pytest

import stabilizer_qec as sq


def test_public_names_resolve():
    for name in sq.__all__:
        assert getattr(sq, name) is not None
    assert sq.__version__.count(".") >= 2


@pytest.mark.parametrize("name", sorted(sq._REMOVED))
def test_removed_names_say_what_replaced_them(name):
    with pytest.raises(AttributeError, match="removed in 1.0; use"):
        getattr(sq, name)
    assert getattr(sq._core, name) is not None  # still in the extension, for the tools


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


def test_circuit_and_model_reads_do_not_block_edits_on_another_thread():
    import threading

    from stabilizer_qec import _core

    circuit = _core.Circuit("X_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    model = _core.Dem("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.2) D1")
    reads = [lambda: circuit.reference_sample(), lambda: circuit.shortest_graphlike(True, False), lambda: model.shortest_graphlike(True)]
    for read in reads:
        stop = threading.Event()
        errors = []

        def loop(read=read):
            while not stop.is_set():
                try:
                    read()
                except Exception as e:
                    errors.append(e)
                    return

        worker = threading.Thread(target=loop)
        worker.start()
        try:
            for _ in range(200):
                circuit.append_text("X 0")
                model.append_text("error(0.1) D0 L0")
        finally:
            stop.set()
            worker.join()
        assert errors == []

def test_deltakit_names_the_python_it_needs(monkeypatch):
    """Before Python 3.11, importing the Deltakit package says which Python it needs, instead of
    failing inside Deltakit's code with a typing or syntax error."""
    for name in [m for m in sys.modules if m == "stabilizer_qec.deltakit" or m.startswith("stabilizer_qec.deltakit.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "version_info", (3, 10, 15, "final", 0))
    with pytest.raises(ImportError, match=r"needs Python 3\.11 or later.*Python 3\.10$"):
        importlib.import_module("stabilizer_qec.deltakit.circuit")
