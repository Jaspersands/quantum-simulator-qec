"""Real-time decoding (`stabilizer_qec.realtime`): asynchronous decodes and the queue equal
`decode_batch` on every decoder, tickets come back in order, statistics count what ran; the
decoder server over stdin/stdout (a subprocess) and over TCP (a thread) decode as the local
decoder does; CUDA-Q QEC's host functions."""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import realtime
from stabilizer_qec.realtime import AsyncResult, DecoderClient, DecoderQueue, DecoderStats, Protocol, make_decoder, serve


@pytest.fixture(scope="module")
def memory():
    c = sq.memory_circuit(distance=5, rounds=5, p=0.004)
    dets, obs = c.compile_detector_sampler(seed=7).sample(2000, separate_observables=True)
    return c, dets, obs


@pytest.mark.parametrize("name", ["matching", "correlated_matching", "union_find", "bposd", "bplsd", "relay_bp", "local_clustering", "search", "belief_matching"])
def test_queue_equals_decode_batch(memory, name):
    c, dets, obs = memory
    dec = make_decoder(name, circuit=c)
    # Batch by batch, as the queue decodes: Relay-BP seeds a shot's memory strengths from its
    # index in its batch, so its answers depend on the batching; the others' do not.
    want = np.concatenate([dec.decode_batch(dets[i:i + 500]) for i in range(0, 2000, 500)])
    if name != "relay_bp":
        assert np.array_equal(want, dec.decode_batch(dets))
    with DecoderQueue(dec, workers=2) as q:
        tickets = [q.enqueue_syndromes(dets[i:i + 500]) for i in range(0, 2000, 500)]
        got = np.concatenate([q.get_corrections(t) for t in tickets])
        assert q.pending() == 0
        stats = q.stats()
    assert np.array_equal(got, want)
    assert (stats.decoded, stats.batches, stats.pending) == (2000, 4, 0) and stats.queue_depth_max >= 1
    assert 0 < stats.mean_service_s <= stats.mean_latency_s and stats.p50_latency_s <= stats.p99_latency_s <= stats.max_latency_s and stats.throughput_per_s > 0
    assert (dec.decode_batch(dets) != obs).any(axis=1).mean() < 0.5


def test_decode_async_and_order(memory):
    c, dets, obs = memory
    m = sq.Matching(c.detector_error_model(decompose_errors=True))
    r = m.decode_async(dets[:100])
    assert isinstance(r, AsyncResult) and r.shots == 100 and r.wait(30) and r.ready()
    assert np.array_equal(r.get(), m.decode_batch(dets[:100])) and r.latency is not None and r.service is not None and 0 <= r.service <= r.latency
    one = m.decode_async(dets[0]).get()
    assert one.shape == (1, m.num_observables) and np.array_equal(one[0], m.decode(dets[0]))
    with DecoderQueue(m) as q:
        for i in range(5):
            q.enqueue_syndromes(dets[i * 10:(i + 1) * 10])
        assert q.pending() == 5
        for i in range(5):
            assert np.array_equal(q.get_corrections(), m.decode_batch(dets[i * 10:(i + 1) * 10]))
        with pytest.raises(LookupError):
            q.get_corrections()
        t = q.enqueue_syndromes(dets[:3])
        q.get_corrections(t)
        with pytest.raises(LookupError):
            q.get_corrections(t)


def test_queue_reset_and_timeouts(memory):
    c, dets, obs = memory
    gate = threading.Event()

    def slow(shots, **kw):
        gate.wait(10)
        return np.zeros((len(shots), 1), dtype=np.uint8)

    with DecoderQueue(decode=slow, name="slow") as q:
        first = q.enqueue_syndromes(dets[:5])
        second = q.enqueue_syndromes(dets[:5])
        r = q.submit(dets[:2])
        assert not r.ready() and not r.wait(0.05)
        with pytest.raises(TimeoutError):
            q.get_corrections(second, timeout=0.05)
        q.reset()
        assert q.pending() == 0 and q.stats().batches == 0
        with pytest.raises(LookupError):
            q.get_corrections(first)
        gate.set()
        assert r.wait(10)
    with pytest.raises(TypeError):
        DecoderQueue(object())
    with pytest.raises(ValueError):
        DecoderQueue(decode=slow, workers=0)
    q = DecoderQueue(decode=slow)
    q.close()
    with pytest.raises(RuntimeError):
        q.enqueue_syndromes(dets[:1])


def test_protocol_frames():
    frame = Protocol.pack(Protocol.DECODE, b"\x01\x02", shots=2, flags=3)
    assert len(frame) == Protocol.HEADER.size + 2
    import io

    kind, flags, shots, payload = Protocol.read(io.BytesIO(frame))
    assert (kind, flags, shots, payload) == (Protocol.DECODE, 3, 2, b"\x01\x02")
    with pytest.raises(EOFError):
        Protocol.read(io.BytesIO(b""))
    with pytest.raises(ValueError):
        Protocol.read(io.BytesIO(b"XXXX" + frame[4:]))
    with pytest.raises(EOFError):
        Protocol.read(io.BytesIO(frame[:-1]))


def test_tcp_server(memory):
    c, dets, obs = memory
    dem = c.detector_error_model(decompose_errors=True)
    m = sq.Matching(dem)
    bound = {}
    started = threading.Event()

    def ready(server):
        bound["server"] = server
        started.set()

    thread = threading.Thread(target=serve, args=(m,), kwargs=dict(tcp="127.0.0.1:0", workers=2, ready=ready), daemon=True)
    thread.start()
    assert started.wait(10)
    host, port = bound["server"].server_address[:2]
    try:
        with DecoderClient.connect(host, port, timeout=10) as client:
            assert (client.num_detectors, client.num_observables, client.decoder_name, client.workers) == (dem.num_detectors, dem.num_observables, "Matching", 2)
            assert np.array_equal(client.decode_batch(dets), m.decode_batch(dets))
            assert np.array_equal(client.decode(dets[3]), m.decode(dets[3]))
            packed = np.packbits(dets[:7].astype(np.uint8), axis=1, bitorder="little")
            assert np.array_equal(client.decode_batch(packed, bit_packed_shots=True, bit_packed_predictions=True), np.packbits(m.decode_batch(dets[:7]).astype(np.uint8), axis=1, bitorder="little"))
            stats = client.stats()
            assert isinstance(stats, DecoderStats) and stats.decoded == 2000 + 1 + 7 and stats.batches == 3
            client.reset()
            assert client.stats().batches == 0
            with pytest.raises(ValueError, match="refused"):  # a malformed frame: the server answers ERROR
                client._ask(Protocol.DECODE, b"\x00", 2)
            with pytest.raises(ValueError, match="refused"):
                client._ask(Protocol.HELLO)
            assert np.array_equal(client.decode_batch(dets[:4]), m.decode_batch(dets[:4]))  # the connection survived
        with DecoderClient.connect(host, port, timeout=10) as second:  # a new connection after BYE
            assert np.array_equal(second.decode_batch(dets[:2]), m.decode_batch(dets[:2]))
    finally:
        bound["server"].shutdown()
        thread.join(10)


def test_stdio_server_subprocess(memory, tmp_path):
    c, dets, obs = memory
    dem = c.detector_error_model(decompose_errors=True)
    path = tmp_path / "m.dem"
    path.write_text(str(dem))
    with DecoderClient.spawn(str(path), decoder="union_find") as client:
        assert client.decoder_name == "UnionFind" and client.num_detectors == dem.num_detectors
        assert np.array_equal(client.decode_batch(dets[:300]), sq.UnionFind(dem).decode_batch(dets[:300]))
        assert client.stats().decoded == 300
    circ = tmp_path / "c.stim"
    circ.write_text(str(c))
    with DecoderClient.spawn(circuit=str(circ), decoder="matching", threads=1) as client:
        assert np.array_equal(client.decode_batch(dets[:300]), sq.Matching(dem).decode_batch(dets[:300]))
    with pytest.raises(TypeError):
        DecoderClient.spawn()


def test_serve_cli_arguments(memory, tmp_path, capsys):
    c, dets, obs = memory
    path = tmp_path / "m.dem"
    path.write_text(str(c.detector_error_model(decompose_errors=True)))
    assert sq.main(command_line_args=["serve", "--dem", str(path)]) == 1  # neither --tcp nor --stdio
    assert "one of the two" in capsys.readouterr().err
    assert sq.main(command_line_args=["serve", "--stdio"]) == 1
    assert "--dem or --circuit" in capsys.readouterr().err
    with pytest.raises(ValueError):
        serve(sq.Matching(c.detector_error_model(decompose_errors=True)))
    with pytest.raises(ValueError):
        make_decoder("nope", circuit=c)


def test_cudaq_qec_host_api(memory):
    import stabilizer_qec.cudaq_qec as qec

    c, dets, obs = memory
    m = sq.Matching(c.detector_error_model(decompose_errors=True))
    ids = qec.configure_decoders({"surface": m})
    assert ids == ["surface"]
    t = qec.enqueue_syndromes("surface", dets[:50])
    assert np.array_equal(qec.get_corrections("surface", t), m.decode_batch(dets[:50]))
    qec.enqueue_syndromes("surface", dets[50:60])
    assert np.array_equal(qec.get_corrections("surface"), m.decode_batch(dets[50:60]))
    assert qec.decoder_stats("surface").decoded == 60
    qec.reset_decoder("surface")
    assert qec.decoder_stats("surface").decoded == 0
    with pytest.raises(KeyError):
        qec.enqueue_syndromes("other", dets[:1])
    # CUDA-Q QEC's own Decoder objects queue too, through their decode_batch.
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    lut = qec.get_decoder("single_error_lut", H)
    ids = qec.configure_decoders([lut])
    t = qec.enqueue_syndromes(ids[0], np.array([[1, 0], [1, 1]], dtype=np.uint8))
    got = qec.get_corrections(ids[0], t)
    assert np.array_equal(np.asarray([np.asarray(r.result) for r in got]), np.asarray([np.asarray(r.result) for r in lut.decode_batch(np.array([[1, 0], [1, 1]], dtype=np.uint8))]))


def test_async_result_repr_and_latency_grow_with_queue(memory):
    c, dets, obs = memory
    m = sq.Matching(c.detector_error_model(decompose_errors=True))
    with DecoderQueue(m, workers=1) as q:
        results = [q.submit(dets) for _ in range(6)]
        for r in results:
            r.get(60)
        lat = [r.latency for r in results]
        assert "done" in repr(results[-1]) and lat[-1] >= lat[0] and q.stats().queue_depth_max >= 2
