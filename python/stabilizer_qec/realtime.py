"""Real-time decoding: decodes that run in the background, a decoder queue with its statistics,
and a decoder server over TCP or over stdin/stdout, with its client — CUDA-Q QEC's
`decode_async`, `enqueue_syndromes`, `get_corrections` and `reset_decoder` on this package's
decoders.

Every DEM decoder has ``decode_async(shots)``: the batch runs on a worker thread (the GIL
released inside the engine) and comes back as an `AsyncResult`. A `DecoderQueue` wraps a decoder
with a pool of workers, numbered tickets, corrections handed back in order and `DecoderStats`
(latency from enqueue to result, service time, throughput, queue depth). `serve` answers
`DecoderClient` over a framed protocol, the same on a TCP socket and on a subprocess's pipes.

>>> import numpy as np
>>> import stabilizer_qec as sq
>>> from stabilizer_qec.realtime import DecoderQueue
>>> c = sq.memory_circuit(distance=3, rounds=3, p=0.01)
>>> dets, obs = c.compile_detector_sampler(seed=1).sample(100, separate_observables=True)
>>> m = sq.Matching(c.detector_error_model(decompose_errors=True))
>>> with DecoderQueue(m) as q:
...     tickets = [q.enqueue_syndromes(dets[i:i + 25]) for i in range(0, 100, 25)]
...     out = np.concatenate([q.get_corrections() for _ in tickets])
...     stats = q.stats()
>>> bool((out == m.decode_batch(dets)).all()), stats.batches, stats.decoded
(True, 4, 100)
"""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
import weakref
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Union

import numpy as np

__all__ = ["AsyncResult", "DecoderQueue", "DecoderStats", "DecoderClient", "serve", "make_decoder", "DECODERS", "Protocol"]

# ---------------------------------------------------------------------------------------------
# Asynchronous decodes and the queue
# ---------------------------------------------------------------------------------------------


class AsyncResult:
    """A decode running in the background. ``ready()`` says whether it has finished, ``wait``
    blocks up to a timeout, ``get`` blocks and returns the predictions (what ``decode_batch``
    returns). ``latency`` is the seconds from submission to completion, ``service`` the seconds
    the decode itself took, both once it is done."""

    __slots__ = ("_future", "_ticket", "_submitted", "_started", "_finished", "_shots")

    def __init__(self, future: "concurrent.futures.Future", ticket: int, submitted: float, shots: int) -> None:
        self._future, self._ticket, self._submitted, self._shots = future, ticket, submitted, shots
        self._started: Optional[float] = None
        self._finished: Optional[float] = None

    @property
    def ticket(self) -> int:
        return self._ticket

    @property
    def shots(self) -> int:
        return self._shots

    def ready(self) -> bool:
        return self._future.done()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """True once the decode has finished (or was dropped by a reset), False if ``timeout``
        seconds passed first."""
        try:
            self._future.exception(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return False
        except concurrent.futures.CancelledError:
            return True
        return True

    def get(self, timeout: Optional[float] = None) -> Any:
        """The predictions; raises what the decode raised, or ``RuntimeError`` for a batch a
        reset dropped."""
        try:
            return self._future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"the decode of ticket {self._ticket} is not finished after {timeout} s") from None
        except concurrent.futures.CancelledError:
            raise RuntimeError(f"the decode of ticket {self._ticket} was dropped by a reset") from None

    result = get

    @property
    def latency(self) -> Optional[float]:
        return None if self._finished is None else self._finished - self._submitted

    @property
    def service(self) -> Optional[float]:
        return None if self._finished is None or self._started is None else self._finished - self._started

    def __repr__(self) -> str:
        state = "done" if self.ready() else "pending"
        return f"AsyncResult(ticket={self._ticket}, shots={self._shots}, {state})"


class DecoderStats(NamedTuple):
    """What a `DecoderQueue` has done: shots and batches decoded, the latency (enqueue to
    result) mean, median, 99th percentile and maximum in seconds, the mean service time (the
    decode alone), shots per second over the span from the first enqueue to the last result,
    the deepest the queue got, and how many batches are still pending."""

    decoded: int
    batches: int
    mean_latency_s: float
    p50_latency_s: float
    p99_latency_s: float
    max_latency_s: float
    mean_service_s: float
    throughput_per_s: float
    queue_depth_max: int
    pending: int

    def asdict(self) -> dict:
        return dict(self._asdict())


def _shot_count(shots: Any) -> int:
    a = np.asarray(shots)
    return 1 if a.ndim <= 1 else int(a.shape[0])


class DecoderQueue:
    """A decoder behind a pool of ``workers`` threads. ``enqueue_syndromes`` takes a batch of
    syndromes (any shape ``decode_batch`` takes, or one syndrome) and returns a ticket;
    ``get_corrections`` returns the predictions of the oldest batch not yet collected, or of a
    given ticket, waiting for it; ``reset`` drops what has not started and clears the
    statistics; ``stats`` is a `DecoderStats`. ``decode_batch``'s keyword arguments (``threads``,
    ``bit_packed_shots``, heralds, ...) pass through ``enqueue_syndromes``. The decoder is any
    object with ``decode_batch`` (this package's decoders, CUDA-Q QEC's, a function of your own
    given as ``decode``).

    One worker decodes batches in order of arrival; more workers decode several at once, each
    batch still whole. Latency counts the wait in the queue, so a queue fed faster than it
    decodes shows it."""

    def __init__(self, decoder: Any = None, *, workers: int = 1, decode: Optional[Callable[..., Any]] = None, name: Optional[str] = None) -> None:
        if decode is None:
            if decoder is None or not hasattr(decoder, "decode_batch"):
                raise TypeError("DecoderQueue needs a decoder with decode_batch, or a decode function")
            decode = decoder.decode_batch
        if workers < 1:
            raise ValueError(f"workers must be at least 1, not {workers}")
        self.decoder = decoder
        self.name = name if name is not None else (type(decoder).__name__ if decoder is not None else getattr(decode, "__name__", "decode"))
        self._decode = decode
        self._workers = workers
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stabilizer-qec-decode")
        self._lock = threading.Lock()
        self._next_ticket = 0
        self._results: Dict[int, AsyncResult] = {}
        self._uncollected: List[int] = []
        self._latencies: List[float] = []
        self._services: List[float] = []
        self._decoded = 0
        self._first_enqueue: Optional[float] = None
        self._last_finish: Optional[float] = None
        self._in_flight = 0
        self._depth_max = 0
        self._closed = False
        self._epoch = 0  # bumped by reset: batches of an earlier epoch leave no statistics

    # -- the API --

    def enqueue_syndromes(self, shots: Any, **kwargs: Any) -> int:
        """Queues a batch; returns its ticket."""
        if self._closed:
            raise RuntimeError("the queue is closed")
        if isinstance(shots, np.ndarray) and shots.ndim == 1:
            shots = shots.reshape(1, -1)  # one syndrome, as decode_async takes it
        n = _shot_count(shots)
        submitted = time.perf_counter()
        with self._lock:
            ticket = self._next_ticket
            self._next_ticket += 1
            epoch = self._epoch
            if self._first_enqueue is None:
                self._first_enqueue = submitted
            self._in_flight += 1
            self._depth_max = max(self._depth_max, self._in_flight)
        holder: Dict[str, Any] = {}

        def run() -> Any:
            holder["started"] = time.perf_counter()
            return self._decode(shots, **kwargs)

        future = self._pool.submit(run)
        result = AsyncResult(future, ticket, submitted, n)
        with self._lock:
            self._results[ticket] = result
            self._uncollected.append(ticket)

        def done(f: "concurrent.futures.Future") -> None:
            finished = time.perf_counter()
            result._started = holder.get("started", submitted)
            result._finished = finished
            with self._lock:
                self._in_flight -= 1
                if epoch == self._epoch and not f.cancelled() and f.exception() is None:
                    self._latencies.append(finished - submitted)
                    self._services.append(finished - result._started)
                    self._decoded += n
                    self._last_finish = finished

        future.add_done_callback(done)
        return ticket

    def submit(self, shots: Any, **kwargs: Any) -> AsyncResult:
        """``enqueue_syndromes`` as an `AsyncResult` (what ``decode_async`` returns); the result
        belongs to the caller and is not kept by the queue."""
        ticket = self.enqueue_syndromes(shots, **kwargs)
        with self._lock:
            self._uncollected.remove(ticket)
            return self._results.pop(ticket)

    def get_corrections(self, ticket: Optional[int] = None, timeout: Optional[float] = None) -> Any:
        """The predictions of ``ticket``, or of the oldest batch not yet collected; waits for it.
        A batch is handed back once."""
        with self._lock:
            if ticket is None:
                if not self._uncollected:
                    raise LookupError("no batch is waiting to be collected")
                ticket = self._uncollected[0]
            result = self._results.get(ticket)
        if result is None:
            raise LookupError(f"ticket {ticket} is unknown, collected already, or dropped by reset")
        # Wait first: a batch whose wait timed out is still there for the next call.
        out = result.get(timeout)
        with self._lock:
            if self._results.get(ticket) is result:
                del self._results[ticket]
                if ticket in self._uncollected:
                    self._uncollected.remove(ticket)
        return out

    def pending(self) -> int:
        """Batches enqueued and not yet collected."""
        with self._lock:
            return len(self._uncollected)

    def reset(self) -> None:
        """Drops every batch that has not started, forgets collected and uncollected results,
        and clears the statistics; batches already running finish unseen."""
        with self._lock:
            self._epoch += 1
            results = list(self._results.values())
            self._results.clear()
            self._uncollected.clear()
            self._latencies.clear()
            self._services.clear()
            self._decoded = 0
            self._first_enqueue = None
            self._last_finish = None
            self._depth_max = 0
        for r in results:
            r._future.cancel()

    def stats(self) -> DecoderStats:
        with self._lock:
            lat = sorted(self._latencies)
            n = len(lat)
            span = (self._last_finish - self._first_enqueue) if (self._first_enqueue is not None and self._last_finish is not None) else 0.0
            pct = lambda q: float(np.percentile(lat, 100.0 * q)) if n else 0.0  # noqa: E731  (numpy's linear interpolation: the median is the median)
            return DecoderStats(self._decoded, n, (sum(lat) / n) if n else 0.0, pct(0.5), pct(0.99), lat[-1] if n else 0.0, (sum(self._services) / n) if n else 0.0, (self._decoded / span) if span > 0 else 0.0, self._depth_max, len(self._uncollected))

    @property
    def workers(self) -> int:
        return self._workers

    def close(self, wait: bool = True) -> None:
        self._closed = True
        self._pool.shutdown(wait=wait)

    def __enter__(self) -> "DecoderQueue":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"DecoderQueue({self.name}, workers={self._workers}, pending={self.pending()})"


_shared: Dict[int, "tuple[weakref.ref, DecoderQueue]"] = {}
_shared_lock = threading.Lock()


def shared_queue(decoder: Any, workers: Optional[int] = None) -> DecoderQueue:
    """The queue ``decode_async`` uses for a decoder: one per decoder object, made on first use
    with ``workers`` threads (the machine's cores by default), closed and forgotten when the
    decoder is garbage-collected."""
    key = id(decoder)
    with _shared_lock:
        entry = _shared.get(key)
        if entry is not None and entry[0]() is decoder:
            return entry[1]
        import os

        def forget(k=key):
            with _shared_lock:
                gone = _shared.pop(k, None)
            if gone is not None:
                gone[1].close(wait=False)

        # The queue must not keep the decoder alive: it calls decode_batch through a weak method
        # and is closed when the decoder goes (a decoder without weak references is held for
        # the process's life).
        try:
            ref = weakref.ref(decoder, lambda _r: forget())
        except TypeError:
            strong = decoder
            ref = lambda: strong  # noqa: E731

        def decode(shots, **kwargs):
            target = ref()
            if target is None:
                raise RuntimeError("the decoder behind this queue was garbage-collected")
            return target.decode_batch(shots, **kwargs)

        q = DecoderQueue(decode=decode, workers=workers or max(1, os.cpu_count() or 1), name=type(decoder).__name__)
        _shared[key] = (ref, q)
        return q


def decode_async(decoder: Any, shots: Any, **kwargs: Any) -> AsyncResult:
    """``decoder.decode_batch(shots, **kwargs)`` in the background: an `AsyncResult`."""
    return shared_queue(decoder).submit(shots, **kwargs)


# ---------------------------------------------------------------------------------------------
# Decoders by name (the CLI's, the server's)
# ---------------------------------------------------------------------------------------------


def _registry() -> Dict[str, Callable[[Any], Any]]:
    from ._decoders import AmbiguityClustering, BeliefMatching, CollisionClustering, LocalClustering, BpLsd, BpOsd, ColorMatching, Matching, RelayBp, SearchDecoder, UnionFind
    from ._matching import _CorrelatedMatching

    return {
        "matching": Matching,
        "correlated_matching": _CorrelatedMatching,
        "belief_matching": BeliefMatching,
        "union_find": UnionFind,
        "bposd": BpOsd,
        "bplsd": BpLsd,
        "relay_bp": RelayBp,
        "color_matching": ColorMatching,
        "search": SearchDecoder,
        "ambiguity_clustering": AmbiguityClustering,
        "collision_clustering": CollisionClustering,
        "local_clustering": LocalClustering,
    }


DECODERS = ("matching", "correlated_matching", "belief_matching", "union_find", "bposd", "bplsd", "relay_bp", "color_matching", "search", "ambiguity_clustering", "collision_clustering", "local_clustering")
_GRAPH = {"matching", "correlated_matching", "belief_matching", "union_find", "color_matching", "ambiguity_clustering", "collision_clustering", "local_clustering"}


def make_decoder(name: str, model: Any = None, *, circuit: Any = None, **options: Any) -> Any:
    """A decoder by its CLI name on a detector error model, or on a circuit's model (decomposed
    for the graph decoders, as they need it)."""
    if name not in DECODERS:
        raise ValueError(f"unknown decoder {name!r}; one of {', '.join(DECODERS)}")
    if circuit is not None:
        from ._circuit import Circuit

        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        model = circuit.detector_error_model(decompose_errors=name in _GRAPH)
    if model is None:
        raise TypeError("make_decoder needs a model or a circuit")
    return _registry()[name](model, **options)


# ---------------------------------------------------------------------------------------------
# The wire protocol
# ---------------------------------------------------------------------------------------------


class Protocol:
    """Frames: a 20-byte header ``!4sBBHIQ`` — magic ``SQRT``, version 1, kind, flags, the shot
    count, the payload length — then the payload. Kinds: ``HELLO`` (server to client on connect,
    JSON with ``num_detectors``, ``num_observables``, ``decoder``), ``DECODE`` (b8 syndromes, one
    row per shot, the shot count in the header) answered by ``CORRECTIONS`` (b8 predictions),
    ``STATS`` answered by ``STATS_REPLY`` (JSON), ``RESET`` answered by ``OK``, ``BYE``, and
    ``ERROR`` (JSON ``{"error": ...}``) for a request the server refuses, after which the
    connection goes on."""

    MAGIC = b"SQRT"
    VERSION = 1
    HEADER = struct.Struct("!4sBBHIQ")
    HELLO, DECODE, CORRECTIONS, STATS, STATS_REPLY, RESET, OK, BYE, ERROR = range(1, 10)
    NAMES = {1: "HELLO", 2: "DECODE", 3: "CORRECTIONS", 4: "STATS", 5: "STATS_REPLY", 6: "RESET", 7: "OK", 8: "BYE", 9: "ERROR"}

    @classmethod
    def pack(cls, kind: int, payload: bytes = b"", shots: int = 0, flags: int = 0) -> bytes:
        return cls.HEADER.pack(cls.MAGIC, cls.VERSION, kind, flags, shots, len(payload)) + payload

    MAX_FRAME = 256 << 20

    @classmethod
    def read_header(cls, stream: Any) -> "tuple[int, int, int, int]":
        """(kind, flags, shots, payload length) from a readable; ``EOFError`` at a clean close."""
        head = _read_exactly(stream, cls.HEADER.size)
        if not head:
            raise EOFError
        if len(head) != cls.HEADER.size:
            raise EOFError("the connection closed inside a header")
        magic, version, kind, flags, shots, length = cls.HEADER.unpack(head)
        if magic != cls.MAGIC:
            raise ValueError(f"not a stabilizer-qec frame (magic {magic!r})")
        if version != cls.VERSION:
            raise ValueError(f"protocol version {version} is not {cls.VERSION}")
        return kind, flags, shots, length

    @classmethod
    def read_payload(cls, stream: Any, length: int) -> bytes:
        payload = _read_exactly(stream, length) if length else b""
        if length and len(payload) != length:
            raise EOFError("the connection closed inside a frame")
        return payload

    @classmethod
    def read(cls, stream: Any, max_length: Optional[int] = None) -> "tuple[int, int, int, bytes]":
        """(kind, flags, shots, payload) from a readable; ``EOFError`` at a clean close,
        ``ValueError`` for a frame longer than ``max_length`` (`MAX_FRAME` by default), refused
        before its payload is read."""
        kind, flags, shots, length = cls.read_header(stream)
        limit = cls.MAX_FRAME if max_length is None else max_length
        if length > limit:
            raise ValueError(f"frame too long ({length} bytes; at most {limit})")
        return kind, flags, shots, cls.read_payload(stream, length)


def _with_timeout(fn: Callable[[], Any], timeout: Optional[float], what: str) -> Any:
    """``fn()`` on a helper thread, ``TimeoutError`` if it takes longer than ``timeout``."""
    if timeout is None:
        return fn()
    box: Dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as ex:  # re-raised on the caller's thread
            box["error"] = ex

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"{what} did not arrive within {timeout} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _read_exactly(stream: Any, n: int) -> bytes:
    if n == 0:
        return b""
    parts = []
    got = 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            break
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def _b8_stride(bits: int) -> int:
    return (bits + 7) // 8


# ---------------------------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------------------------


class _Session:
    """One connection: frames in, frames out, the decoder behind a queue."""

    def __init__(self, queue: DecoderQueue, num_detectors: int, num_observables: int, threads: int, max_shots: int = 1 << 20) -> None:
        self.queue, self.nd, self.no, self.threads, self.max_shots = queue, num_detectors, num_observables, threads, max_shots

    def hello(self) -> bytes:
        return Protocol.pack(Protocol.HELLO, json.dumps({"num_detectors": self.nd, "num_observables": self.no, "decoder": self.queue.name, "workers": self.queue.workers, "max_shots": self.max_shots}).encode())

    def handle(self, kind: int, flags: int, shots: int, payload: bytes) -> "Optional[bytes]":
        """The reply frame, or None for BYE."""
        try:
            if kind == Protocol.DECODE:
                stride = _b8_stride(self.nd)
                if shots < 0 or len(payload) != shots * stride:
                    raise ValueError(f"DECODE of {shots} shots needs {shots * stride} bytes of b8 syndromes, got {len(payload)}")
                if shots == 0:
                    return Protocol.pack(Protocol.CORRECTIONS, b"", 0)
                rows = np.frombuffer(payload, dtype=np.uint8).reshape(shots, stride)
                takes = _params(self.queue._decode)
                kwargs: Dict[str, Any] = {}
                if "bit_packed_shots" in takes or "**" in takes:
                    kwargs["bit_packed_shots"] = True
                else:  # a decoder of plain rows (CUDA-Q QEC's, say)
                    rows = np.unpackbits(rows, axis=1, bitorder="little")[:, : self.nd]
                if self.threads and ("threads" in takes or "**" in takes):
                    kwargs["threads"] = self.threads
                ticket = self.queue.enqueue_syndromes(rows, **kwargs)
                out = self.queue.get_corrections(ticket)
                if hasattr(out, "predictions"):  # a result object rather than an array
                    out = out.predictions
                out = np.asarray(out)
                if out.ndim != 2 or out.shape[0] != shots:
                    raise ValueError(f"the decoder returned shape {out.shape} for {shots} shots")
                # One bit per observable, or already b8-packed rows.
                packed = np.packbits(out.astype(np.uint8), axis=1, bitorder="little") if out.shape[1] == self.no else out.astype(np.uint8)
                if packed.shape[1] != _b8_stride(self.no):
                    raise ValueError(f"the decoder returned {out.shape[1]} columns for {self.no} observables")
                return Protocol.pack(Protocol.CORRECTIONS, packed.tobytes(), shots)
            if kind == Protocol.STATS:
                return Protocol.pack(Protocol.STATS_REPLY, json.dumps(self.queue.stats().asdict()).encode())
            if kind == Protocol.RESET:
                self.queue.reset()
                return Protocol.pack(Protocol.OK)
            if kind == Protocol.BYE:
                return None
            raise ValueError(f"unexpected frame kind {Protocol.NAMES.get(kind, kind)}")
        except Exception as ex:  # the client hears what went wrong; the connection stays
            return Protocol.pack(Protocol.ERROR, json.dumps({"error": f"{type(ex).__name__}: {ex}"}).encode())

    def run(self, reader: Any, writer: Any) -> None:
        try:
            writer.write(self.hello())
            writer.flush()
            while True:
                try:
                    kind, flags, shots, length = Protocol.read_header(reader)
                    # Refuse what cannot be right before buffering it: a DECODE's payload is
                    # exactly its shots' b8 rows, and no frame is longer than the cap.
                    stride = _b8_stride(self.nd)
                    if kind == Protocol.DECODE and (shots > self.max_shots or length != shots * stride):
                        raise ValueError(f"DECODE of {shots} shots needs {shots * stride} bytes of b8 syndromes (at most {self.max_shots} shots), got {length}")
                    if length > Protocol.MAX_FRAME:
                        raise ValueError(f"frame too long ({length} bytes; at most {Protocol.MAX_FRAME})")
                    payload = Protocol.read_payload(reader, length)
                except EOFError:
                    return
                except ValueError as ex:
                    writer.write(Protocol.pack(Protocol.ERROR, json.dumps({"error": str(ex)}).encode()))
                    writer.flush()
                    return
                reply = self.handle(kind, flags, shots, payload)
                if reply is None:
                    return
                writer.write(reply)
                writer.flush()
        except OSError:  # the connection died or timed out: nothing to answer
            return


def _params(fn: Callable[..., Any]) -> set:
    """The parameter names a callable takes; a ``**kwargs`` is every name (the set holds "**")."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return set()
    names = set(params)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        names.add("**")
    return names


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    # Clients connecting at once queue here until accepted (socketserver's default of 5 reset
    # most of a burst of 16).
    request_queue_size = max(128, int(getattr(socket, "SOMAXCONN", 128)))


def serve(decoder: Any, *, tcp: Optional[str] = None, stdio: bool = False, workers: int = 1, threads: int = 0, num_detectors: Optional[int] = None, num_observables: Optional[int] = None, ready: Optional[Callable[[Any], None]] = None, timeout: Optional[float] = 300.0, max_shots: int = 1 << 20) -> None:
    """Serves ``decoder`` (anything with ``decode_batch``, ``num_detectors`` and
    ``num_observables``, or those given) to clients: over ``tcp="host:port"`` (port 0 picks one;
    ``ready(server)`` is called with the bound server, whose ``server_address`` tells which) until
    the server is shut down, or over this process's stdin and stdout (``stdio=True``) until the
    client says BYE or closes the pipe. ``workers`` decode batches at once; ``threads`` is each
    batch's thread count for decoders that take one. A TCP connection idle for ``timeout`` seconds
    (or one that sends part of a frame and stops) is dropped; a DECODE of more than ``max_shots``
    shots, or whose length is not its shots' b8 rows, is refused before it is read."""
    nd = int(num_detectors if num_detectors is not None else decoder.num_detectors)
    no = int(num_observables if num_observables is not None else decoder.num_observables)
    queue = DecoderQueue(decoder, workers=workers)
    session = _Session(queue, nd, no, threads, max_shots)
    if stdio == bool(tcp):
        raise ValueError("serve over tcp='host:port' or stdio=True, one of the two")
    try:
        if stdio:
            session.run(sys.stdin.buffer, sys.stdout.buffer)
            return
        host, _, port = tcp.rpartition(":")
        if not host or not port:
            raise ValueError(f"tcp must be 'host:port', not {tcp!r}")

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                session.run(self.rfile, self.wfile)

        Handler.timeout = timeout

        with _Server((host, int(port)), Handler) as server:
            if ready is not None:
                ready(server)
            server.serve_forever()
    finally:
        queue.close(wait=False)


# ---------------------------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------------------------


class DecoderClient:
    """A decoder served elsewhere. ``DecoderClient.connect(host, port)`` reaches a TCP server;
    ``DecoderClient.spawn(...)`` starts ``stabilizer-qec serve --stdio`` as a subprocess on a
    model or circuit file and talks over its pipes. ``decode_batch`` and ``decode`` are the
    decoders'; ``stats`` and ``reset`` are the server's queue's; ``close`` says BYE."""

    def __init__(self, reader: Any, writer: Any, *, process: Optional[subprocess.Popen] = None, sock: Optional[socket.socket] = None, timeout: Optional[float] = 30.0) -> None:
        self._reader, self._writer, self._process, self._sock = reader, writer, process, sock
        self._lock = threading.Lock()
        try:
            kind, _, _, payload = _with_timeout(lambda: Protocol.read(reader, 1 << 20), timeout, "the decoder server's HELLO")
            if kind != Protocol.HELLO:
                raise ValueError(f"expected HELLO, got {Protocol.NAMES.get(kind, kind)}")
            hello = json.loads(payload.decode())
        except BaseException:
            self._abandon()
            raise
        self.num_detectors = int(hello["num_detectors"])
        self.num_observables = int(hello["num_observables"])
        self.decoder_name = str(hello.get("decoder", ""))
        self.workers = int(hello.get("workers", 1))
        #: The most shots the server takes in one DECODE; ``decode_batch`` sends larger batches in pieces.
        self.max_shots = int(hello.get("max_shots", 1 << 20))

    def _abandon(self) -> None:
        # The peer first: a reader thread may still be blocked in a read holding the buffer's
        # lock, and closing the file would wait for it; shutting the socket (or killing the
        # child) ends that read.
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for f in (getattr(self, "_writer", None), getattr(self, "_reader", None)):
            try:
                if f is not None:
                    f.close()
            except OSError:
                pass
        if self._sock is not None:
            self._sock.close()

    @classmethod
    def connect(cls, host: str, port: int, timeout: Optional[float] = 30.0) -> "DecoderClient":
        """Reaches a TCP server; ``timeout`` covers the connection and its HELLO."""
        sock = socket.create_connection((host, int(port)), timeout=timeout)
        sock.settimeout(None)
        return cls(sock.makefile("rb"), sock.makefile("wb"), sock=sock, timeout=timeout)

    @classmethod
    def spawn(cls, dem: Optional[str] = None, *, circuit: Optional[str] = None, decoder: str = "matching", workers: int = 1, threads: int = 0, python: Optional[str] = None, timeout: Optional[float] = 60.0) -> "DecoderClient":
        """Starts ``stabilizer-qec serve --stdio`` as a subprocess on a model or circuit file and
        talks over its pipes. A child that does not say HELLO within ``timeout`` seconds (or dies
        first, its error on stderr) is killed and the error raised here."""
        if (dem is None) == (circuit is None):
            raise TypeError("spawn needs a dem path or a circuit path, one of the two")
        argv = [python or sys.executable, "-m", "stabilizer_qec._cli", "serve", "--stdio", "--decoder", decoder, "--workers", str(workers), "--threads", str(threads)]
        argv += ["--dem", dem] if dem is not None else ["--circuit", circuit]
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            return cls(process.stdout, process.stdin, process=process, timeout=timeout)
        except EOFError:
            code = process.poll()
            raise RuntimeError(f"the decoder server exited (code {code}) before saying HELLO; its error is on stderr") from None

    def _ask(self, kind: int, payload: bytes = b"", shots: int = 0) -> "tuple[int, int, bytes]":
        with self._lock:
            self._writer.write(Protocol.pack(kind, payload, shots))
            self._writer.flush()
            k, _, n, body = Protocol.read(self._reader)
        if k == Protocol.ERROR:
            raise ValueError("the decoder server refused: " + json.loads(body.decode()).get("error", "?"))
        return k, n, body

    def decode_batch(self, shots: Any, *, bit_packed_shots: bool = False, bit_packed_predictions: bool = False) -> np.ndarray:
        from ._util import rows_to_b8, b8_to_rows

        packed, n = rows_to_b8(shots, self.num_detectors, bit_packed_shots, "shots")
        stride = _b8_stride(self.num_detectors)
        limit = max(1, self.max_shots)
        if n <= limit:
            kind, m, body = self._ask(Protocol.DECODE, packed, n)
            if kind != Protocol.CORRECTIONS or m != n:
                raise ValueError(f"expected CORRECTIONS for {n} shots, got {Protocol.NAMES.get(kind, kind)} for {m}")
            return b8_to_rows(body, n, self.num_observables, bit_packed_predictions)
        # More than the server takes at once: in pieces (the server would refuse the frame and
        # close the connection while the payload is still on its way).
        pieces = []
        for start in range(0, n, limit):
            count = min(limit, n - start)
            kind, m, body = self._ask(Protocol.DECODE, packed[start * stride:(start + count) * stride], count)
            if kind != Protocol.CORRECTIONS or m != count:
                raise ValueError(f"expected CORRECTIONS for {count} shots, got {Protocol.NAMES.get(kind, kind)} for {m}")
            pieces.append(body)
        return b8_to_rows(b"".join(pieces), n, self.num_observables, bit_packed_predictions)

    def decode(self, syndrome: Any) -> np.ndarray:
        s = np.asarray(syndrome)
        if s.ndim != 1:
            raise ValueError(f"a syndrome is 1-dimensional, not {s.ndim}-dimensional; use decode_batch for many")
        return self.decode_batch(s.reshape(1, -1))[0]

    def stats(self) -> DecoderStats:
        kind, _, body = self._ask(Protocol.STATS)
        return DecoderStats(**json.loads(body.decode()))

    def reset(self) -> None:
        self._ask(Protocol.RESET)

    def close(self) -> None:
        try:
            with self._lock:
                self._writer.write(Protocol.pack(Protocol.BYE))
                self._writer.flush()
        except (OSError, ValueError):
            pass
        for f in (self._writer, self._reader):
            try:
                f.close()
            except OSError:
                pass
        if self._sock is not None:
            self._sock.close()
        if self._process is not None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def __enter__(self) -> "DecoderClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"DecoderClient({self.decoder_name}, {self.num_detectors} detectors, {self.num_observables} observables)"
