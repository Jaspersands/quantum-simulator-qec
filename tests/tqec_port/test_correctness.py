"""Correctness of compilation isolation, metadata and serialization."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
import pickle

import pytest

from stabilizer_qec import tqec


def move():
    return tqec.gallery.move_rotation(tqec.Basis.Z)


def test_concurrent_compilations_are_isolated(monkeypatch):
    from stabilizer_qec.tqec._compile import compile as module

    # Controlled overlap: the first compilation pauses immediately after reset;
    # the second starts while it is paused. Both then generate concurrently.
    original = module._start_from_fresh_caches
    reset_entered, second_started = Event(), Event()
    calls = [0]
    def reset():
        original()
        calls[0] += 1
        if calls[0] == 1:
            reset_entered.set()
            assert second_started.wait(10)
    monkeypatch.setattr(module, "_start_from_fresh_caches", reset)
    barrier = Barrier(2)
    def worker(second=False):
        if second:
            assert reset_entered.wait(10)
            second_started.set()
        compiled = tqec.compile_block_graph(move(), convention="fixed_boundary")
        barrier.wait(30)
        return str(compiled.generate_stim_circuit(1))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(worker)
        second = pool.submit(worker, True)
        texts = [first.result(60), second.result(60)]
    expected = str(tqec.compile_block_graph(move(), convention="fixed_boundary").generate_stim_circuit(1))
    assert texts == [expected, expected]


def test_repeat_generation_keeps_compilation_local_state():
    first = tqec.compile_block_graph(move(), convention="fixed_boundary")
    control = tqec.compile_block_graph(move(), convention="fixed_boundary")
    for k in (1, 1, 2):
        assert str(first.generate_stim_circuit(k)) == str(control.generate_stim_circuit(k))
