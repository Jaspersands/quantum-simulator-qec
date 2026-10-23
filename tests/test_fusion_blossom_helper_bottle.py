"""Fusion Blossom's ``helper.bottle`` and ``helper.serve``: ``stabilizer_qec.fusion_blossom.helper.bottle`` is the
Bottle 0.14-dev that fusion_blossom 0.2.13 embeds (its ``src/bottle.py``), unchanged, and ``helper.serve`` serves the
visualizer website on it as fusion_blossom's does. The server's answers (status, headers and body) equal
fusion_blossom's, as recorded in ``tests/fusion_blossom_port/helper_server.json`` (``python
tests/test_fusion_blossom_helper_bottle.py`` records it with fusion_blossom 0.2.13 installed), and side by side with
fusion_blossom where it is installed. Deliberate resolved-path, malformed-UTF-8 and refused-Range safety
differences are checked independently in test_local_server_boundaries.py; the recorded corpus stays immutable."""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys
import threading
import time
from wsgiref.simple_server import WSGIServer

import pytest

from conftest import reference

import stabilizer_qec.fusion_blossom as fb

FIXTURE = pathlib.Path(__file__).parent / "fusion_blossom_port" / "helper_server.json"
# fusion_blossom 0.2.13's src/bottle.py (Bottle 0.14-dev, 4,909 lines).
BOTTLE_SHA256 = "909046a12ed686e3b4d1db43c78dfed103dc16e0698e20f341f8202f307ea383"
# The data folder served under /data/, and the requests: the website's files, data files (one in a
# folder, and a range of one), a missing data file and website file, paths out of the data folder,
# and the methods Bottle refuses. Each is sent with one Host header, as error pages quote the URL.
DATA = {"visualizer.json": '{"format":"fusion_blossom","version":"0.2.13","snapshots":[]}\n', "sub/a.txt": "hello\n"}
PATHS = ["/", "/index.html", "/index.js", "/icon.svg", "/package.json", "/?filename=visualizer.json", "//index.js",
         "/data/visualizer.json", "/data/visualizer.json?x=1", "/data/sub/a.txt", "/data/missing.json", "/missing.js",
         "/data/../x", "/data/%2e%2e/x", "/data//etc/passwd", "/sub/../index.js", "/data/", "/data"]
CASES = [(method, path, {}) for path in PATHS for method in ("GET", "HEAD", "POST")]
CASES.append(("GET", "/data/visualizer.json", {"Range": "bytes=2-9"}))
HOST = "localhost:51665"
# Headers whose values name the moment, the Python, or the data file's place and time on disk.
VOLATILE = ("Date", "Server", "Last-Modified", "Etag")


def write_data(folder):
    for name, text in DATA.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(text.encode())


def running(server, run_server):
    """``run_server`` in a thread, once it has made the server's ``srv``."""
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not hasattr(server, "srv"):
        assert thread.is_alive(), "the server thread died"
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.01)
    return thread


def start(module, folder):
    """``module.helper.serve`` on a port the system picks, running."""
    server, run_server = module.helper.serve("localhost", 0, str(folder), return_server=True)
    return server, running(server, run_server)


def stop(server, thread):
    server.srv.shutdown()
    thread.join()
    server.srv.server_close()


def answers(port):
    out = []
    for method, path, headers in CASES:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            conn.request(method, path, headers={"Host": HOST, **headers})
            r = conn.getresponse()
            out.append((r.status, r.reason, r.getheaders(), r.read()))
        finally:
            conn.close()
    return out


def summary(answer, volatile=VOLATILE):
    """An answer as recorded: volatile headers' values left out, a long body by its sha256."""
    status, reason, headers, body = answer
    case = {"status": status, "reason": reason, "headers": [[k, None if k in volatile else v] for k, v in headers]}
    if len(body) < 4096:
        case["body"] = body.decode()
    else:
        case["body_sha256"] = hashlib.sha256(body).hexdigest()
    return case


def test_bottle_is_verbatim():
    from stabilizer_qec.fusion_blossom.helper import bottle

    assert fb.helper.bottle is bottle and "bottle" in dir(fb.helper)
    # .gitattributes requires LF for this path; the raw upstream hash intentionally rejects CRLF.
    assert hashlib.sha256(pathlib.Path(bottle.__file__).read_bytes()).hexdigest() == BOTTLE_SHA256
    assert (bottle.__version__, bottle.__name__) == ("0.14-dev", "stabilizer_qec.fusion_blossom.helper.bottle")
    # fusion_blossom also installs its copies as the modules `bottle` and `helper`; this package does not.
    assert sys.modules.get("bottle") is not bottle and sys.modules.get("helper") is not fb.helper


def test_bottle_is_imported_with_the_package():
    # fusion_blossom runs its Bottle when the package is imported (lib.rs:59-63), and so does this package's
    # fusion_blossom, with no warning; ``import stabilizer_qec`` alone imports neither.
    code = ("import sys, warnings, stabilizer_qec; assert 'stabilizer_qec.fusion_blossom.helper.bottle' not in sys.modules; "
            "warnings.simplefilter('error'); import stabilizer_qec.fusion_blossom as fb; "
            "print('bottle' in vars(fb.helper), fb.helper.bottle is sys.modules['stabilizer_qec.fusion_blossom.helper.bottle'])")
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(fb.__file__).parent.parent.parent)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert (out.returncode, out.stdout) == (0, "True True\n"), out.stderr


@pytest.mark.parametrize("package", ["stabilizer_qec.", ""], ids=["ours", "sinter"])
def test_sinter_imports_fusion_blossom_when_it_decodes(package):
    # sinter imports fusion_blossom inside its decoder's methods (and for annotations only under TYPE_CHECKING), so
    # ``import sinter`` loads neither fusion_blossom nor its Bottle and appends no import hook; nor does
    # ``import stabilizer_qec.sinter``, whose decoder then imports this package's fusion_blossom, and Bottle, to decode.
    if not package and importlib.util.find_spec("sinter") is None:
        pytest.skip("sinter is not installed")
    code = (f"import sys; hooks = list(sys.meta_path); import {package}sinter as sinter; "
            f"print(sorted(m for m in sys.modules if m.split('.')[-1] == 'bottle' or (m + '.').startswith('{package}fusion_blossom.')), "
            "[type(h).__name__ for h in sys.meta_path if h not in hooks])")
    if package:
        code += ("; import numpy as np, stabilizer_qec as stim; "
                 "decoder = sinter.BUILT_IN_DECODERS['fusion_blossom'].compile_decoder_for_dem(dem=stim.DetectorErrorModel('error(0.1) D0 L0')); "
                 "print(decoder.decode_shots_bit_packed(bit_packed_detection_event_data=np.array([[1], [0]], dtype=np.uint8)).tolist(), "
                 "'stabilizer_qec.fusion_blossom.helper.bottle' in sys.modules)")
    else:
        # Real stim 1.16.0 on CPython 3.13 aborts at interpreter shutdown under PYTHONDEVMODE=1 (`python -X dev -c
        # 'import stim'` exits 134: "_PyMem_DebugRawFree: bad ID"), after sinter's answer is printed. Leave once stdout
        # is flushed, so that crash cannot hide a correct answer. An exception before this point still exits non-zero
        # (1, or -6 when that crash follows the traceback). Our arm keeps a full shutdown, so a crash in ours fails it.
        code += "; sys.stdout.flush(); import os; os._exit(0)"
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(fb.__file__).parent.parent.parent)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert (out.returncode, out.stdout) == (0, "[] []\n" + ("[[1], [0]] True\n" if package else "")), out.stderr


def test_running_fails_at_once_when_the_server_thread_dies(tmp_path):
    """A server that cannot bind ends its thread, and ``running`` fails then, not at its deadline."""
    server, run_server = fb.helper.serve("localhost", 65536, str(tmp_path), return_server=True)
    errors = []

    def run_server_keeping_its_error():
        try:
            run_server()
        except OverflowError as e:  # bind(): port must be 0-65535
            errors.append(e)

    with pytest.raises(AssertionError, match="the server thread died"):
        running(server, run_server_keeping_its_error)
    assert len(errors) == 1 and not hasattr(server, "srv")


def test_return_server_is_bottle_wsgiref(tmp_path):
    server, run_server = fb.helper.serve("localhost", 0, str(tmp_path), return_server=True)
    assert type(server) is fb.helper.bottle.WSGIRefServer
    assert (server.host, server.port, server.options, server.quiet, repr(server)) == ("localhost", 0, {}, False, "WSGIRefServer()")
    assert not hasattr(server, "srv")  # run_server makes it
    thread = running(server, run_server)
    assert type(server.srv) is WSGIServer
    stop(server, thread)
    assert server.quiet is True and server.port == server.srv.server_port > 0


@pytest.mark.parametrize("which", ["ours", "fusion_blossom"])
def test_later_serve_changes_what_servers_serve(tmp_path, which):
    """The routes go on Bottle's default app, so a running server serves the data folder of the
    latest ``serve`` call."""
    module = fb if which == "ours" else reference("fusion_blossom")
    for name in "AB":
        (tmp_path / name).mkdir()
        (tmp_path / name / "a.txt").write_bytes(name.encode())
    server, thread = start(module, tmp_path / "A")

    def get():
        conn = http.client.HTTPConnection("127.0.0.1", server.srv.server_port, timeout=30)
        try:
            conn.request("GET", "/data/a.txt")
            return conn.getresponse().read()
        finally:
            conn.close()

    try:
        before = get()
        module.helper.serve("localhost", 0, str(tmp_path / "B"), return_server=True)
        after = get()
    finally:
        stop(server, thread)
    assert (before, after) == (b"A", b"B")


def test_serve_responses_equal_recorded(tmp_path):
    recorded = json.loads(FIXTURE.read_text())
    assert (recorded["host"], recorded["data"], [c["request"] for c in recorded["cases"]]) == (HOST, DATA, [list(c) for c in CASES])
    write_data(tmp_path)
    server, thread = start(fb, tmp_path)
    try:
        ours = answers(server.srv.server_port)
    finally:
        stop(server, thread)
    for case, answer in zip(recorded["cases"], ours):
        assert summary(answer) == {k: v for k, v in case.items() if k != "request"}, case["request"]


def test_serve_responses_equal_reference(tmp_path):
    ref = reference("fusion_blossom")
    write_data(tmp_path)
    started = []
    try:
        for module in (ref, fb):
            started.append(start(module, tmp_path))
        theirs, ours = (answers(server.srv.server_port) for server, _ in started)
    finally:
        for server, thread in started:
            stop(server, thread)
    for case, a, b in zip(CASES, theirs, ours):
        assert summary(a, ("Date",)) == summary(b, ("Date",)) and a[3] == b[3], case


if __name__ == "__main__":
    # Record the fixture from fusion_blossom.
    import importlib.metadata
    import tempfile

    import fusion_blossom

    with tempfile.TemporaryDirectory() as folder:
        write_data(pathlib.Path(folder))
        server, thread = start(fusion_blossom, folder)
        try:
            theirs = answers(server.srv.server_port)
        finally:
            stop(server, thread)
    source = f"fusion_blossom {importlib.metadata.version('fusion_blossom')} on {platform.python_implementation()} {platform.python_version()}, {platform.system()}"
    cases = [json.dumps({"request": list(case), **summary(answer)}) for case, answer in zip(CASES, theirs)]
    head = json.dumps({"source": source, "host": HOST, "data": DATA})[:-1]
    FIXTURE.write_text(head + ', "cases": [\n' + ",\n".join(cases) + "\n]}\n")
    print(f"{FIXTURE}: {len(cases)} answers from {source}")
