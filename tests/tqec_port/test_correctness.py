"""Correctness of compilation isolation, metadata and serialization."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, local

import pytest

from stabilizer_qec import tqec


def move():
    return tqec.gallery.move_rotation(tqec.Basis.Z)


def test_concurrent_compilations_are_isolated(monkeypatch):
    from stabilizer_qec.tqec._compile import compile as module

    # Observe real contention, rather than a worker merely announcing its intent
    # to compile. The first reset waits for the second acquisition to fail while
    # the first transaction owns the underlying lock.
    original_lock = module._COMPILATION_LOCK
    reset_entered, contended = Event(), Event()
    ownership = local()

    class ObservedLock:
        def __enter__(self):
            if not original_lock.acquire(blocking=False):
                contended.set()
                assert original_lock.acquire(timeout=10), "compilation lock acquisition timed out"
            ownership.active = True
            return self

        def __exit__(self, *exc):
            ownership.active = False
            original_lock.release()

    monkeypatch.setattr(module, "_COMPILATION_LOCK", ObservedLock())
    original_reset = module._start_from_fresh_caches
    reset_calls, builder_calls = [], []

    def reset():
        # Signal even on a broken no-lock route so both workers can fail without
        # spending their timeouts waiting on a peer that already failed.
        reset_entered.set()
        assert getattr(ownership, "active", False), "cache reset ran outside the compilation transaction"
        original_reset()
        reset_calls.append(True)
        if len(reset_calls) == 1:
            assert contended.wait(10), "second compilation never contended for the held transaction"

    monkeypatch.setattr(module, "_start_from_fresh_caches", reset)
    for builder in (module.FixedBulkCubeBuilder, module.FixedBulkPipeBuilder, module.FixedBoundaryCubeBuilder, module.FixedBoundaryPipeBuilder):
        original_build = builder.__call__

        def build(self, *args, _original=original_build, _name=builder.__name__, **kwargs):
            assert getattr(ownership, "active", False), "builder ran outside the compilation transaction"
            builder_calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(builder, "__call__", build)

    barrier = Barrier(2)

    def worker(second=False):
        if second:
            assert reset_entered.wait(10)
        compiled = tqec.compile_block_graph(move(), convention="fixed_boundary")
        barrier.wait(30)
        return str(compiled.generate_stim_circuit(1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(worker)
        second = pool.submit(worker, True)
        texts = [first.result(60), second.result(60)]
    assert contended.is_set()
    assert len(reset_calls) == 2
    assert set(builder_calls) == {"FixedBoundaryCubeBuilder", "FixedBoundaryPipeBuilder"}
    expected = str(tqec.compile_block_graph(move(), convention="fixed_boundary").generate_stim_circuit(1))
    assert texts == [expected, expected]


def test_repeat_generation_keeps_compilation_local_state():
    first = tqec.compile_block_graph(move(), convention="fixed_boundary")
    control = tqec.compile_block_graph(move(), convention="fixed_boundary")
    for k in (1, 1, 2):
        assert str(first.generate_stim_circuit(k)) == str(control.generate_stim_circuit(k))


@pytest.mark.parametrize("engine", [False, True])
def test_observable_iterator_matches_list_metadata(engine):
    graph = tqec.gallery.memory(tqec.Basis.Z)
    surfaces = graph.find_correlation_surfaces()
    def compile(observables):
        return tqec.compile_block_graph(graph, 3, observables=observables) if engine else tqec.compile_block_graph(graph, observables=observables)
    control = compile(surfaces)
    generated = compile(iter(surfaces))
    if engine:
        assert generated.observables == control.observables
        generated, control = generated.computation, control.computation
    assert generated.num_observables == control.num_observables == len(surfaces)
    assert generated.stabilizers == control.stabilizers
    actual = generated.generate_stim_circuit(1)
    assert actual.num_observables == generated.num_observables
    assert str(actual) == str(control.generate_stim_circuit(1))
