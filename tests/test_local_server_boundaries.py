"""Synthetic loopback-only server regressions; no browser, user service or external requests."""
import gc
import http.client
import socket
import threading
import time
import warnings
from contextlib import contextmanager

import pytest

from stabilizer_qec.fusion_blossom import helper
from stabilizer_qec.estimate.qualtran import ui


@contextmanager
def fusion_server(folder):
    server, run = helper.serve('127.0.0.1', 0, str(folder), return_server=True)
    errors = []

    def worker():
        try:
            run()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3
    while not hasattr(server, 'srv') and thread.is_alive() and time.monotonic() < deadline:
        thread.join(.01)
    assert not errors and hasattr(server, 'srv')
    try:
        yield server.port
    finally:
        server.srv.shutdown()
        thread.join(3)
        server.srv.server_close()
        assert not thread.is_alive() and not errors


@contextmanager
def page_server(monkeypatch):
    # Avoid loading optional assets and computation: exercise the real transport with tiny inputs.
    monkeypatch.setattr(ui, '_plotly_js', lambda: None)
    server = ui._page_server('127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        thread.join(3)
        server.server_close()
        assert not thread.is_alive()


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        reply = conn.getresponse()
        return reply.status, dict(reply.getheaders()), reply.read()
    finally:
        conn.close()


def raw(port, target, method=b'GET', headers=b''):
    with socket.create_connection(('127.0.0.1', port), timeout=3) as conn:
        conn.sendall(method + b' ' + target + b' HTTP/1.0\r\nHost: localhost\r\n' + headers + b'\r\n')
        conn.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b''.join(chunks)


def test_fusion_symlinks_unicode_and_cwd(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    root.mkdir()
    (root / 'sub').mkdir()
    (root / 'sub' / 'café.txt').write_text('ordinary Unicode', encoding='utf8')
    (tmp_path / 'outside.txt').write_text('synthetic outside', encoding='utf8')
    (root / 'internal').symlink_to(root / 'sub', target_is_directory=True)
    (root / 'escape').symlink_to(tmp_path / 'outside.txt')
    monkeypatch.chdir(tmp_path)
    with fusion_server('data') as port:
        monkeypatch.chdir(root)
        for path in ('/data/sub/caf%C3%A9.txt', '/data/internal/caf%C3%A9.txt'):
            assert request(port, 'GET', path)[::2] == (200, b'ordinary Unicode')
        for method in ('GET', 'HEAD'):
            status, _, body = request(port, method, '/data/escape')
            assert status == 403 and b'synthetic outside' not in body


def test_fusion_invalid_utf8_has_bounded_error(tmp_path, capsys):
    with fusion_server(tmp_path) as port:
        for target in (b'/data/%ff', b'/data/%c0%af', b'/missing/\xff'):
            reply = raw(port, target)
            assert reply.startswith(b'HTTP/1.0 400 ') and len(reply) < 1024
            assert b'400 Bad Request\n' in reply
        assert request(port, 'GET', '/data/missing')[0] == 404
    assert 'Traceback' not in capsys.readouterr().err


def test_fusion_ranges_close_handles_and_keep_conditionals(tmp_path):
    (tmp_path / 'a.txt').write_bytes(b'abcdef')
    gc.collect()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always', ResourceWarning)
        with fusion_server(tmp_path) as port:
            assert request(port, 'GET', '/data/a.txt', headers={'Range': 'bytes=20-30'})[0] == 416
            status, headers, body = request(port, 'GET', '/data/a.txt', headers={'Range': 'bytes=1-3'})
            assert (status, headers['Content-Range'], body) == (206, 'bytes 1-3/6', b'bcd')
            status, headers, body = request(port, 'HEAD', '/data/a.txt', headers={'Range': 'bytes=1-3'})
            assert (status, headers['Content-Length'], body) == (206, '3', b'')
            status, headers, _ = request(port, 'GET', '/data/a.txt')
            assert request(port, 'GET', '/data/a.txt', headers={'Range': 'bytes=20-30', 'If-None-Match': headers['Etag']})[0] == 304
        gc.collect()
    assert not [w for w in recorded if issubclass(w.category, ResourceWarning)]


def test_visualizer_failed_bind_and_input_cleanup(tmp_path, monkeypatch):
    # Hold our own synthetic loopback port. Failed bind must raise its OSError, without input/browser.
    with socket.socket() as held:
        held.bind(('127.0.0.1', 0))
        held.listen(1)
        monkeypatch.setattr('builtins.input', lambda: pytest.fail('prompted after failed bind'))
        monkeypatch.setattr(helper.webbrowser, 'open', lambda *args: pytest.fail('opened a browser'))
        with pytest.raises(OSError):
            helper.open_visualizer('synthetic.json', host='127.0.0.1', port=held.getsockname()[1], data_folder=str(tmp_path))
    started = []
    original = helper.serve

    def tracked(*args, **kwargs):
        result = original(*args, **kwargs)
        started.append(result[0])
        return result

    monkeypatch.setattr(helper, 'serve', tracked)
    monkeypatch.setattr('builtins.input', lambda: '')
    helper.open_visualizer('synthetic.json', host='127.0.0.1', port=0, data_folder=str(tmp_path), open_browser=False)
    assert started[-1].srv.socket.fileno() == -1
    monkeypatch.setattr('builtins.input', lambda: (_ for _ in ()).throw(EOFError('synthetic input ended')))
    with pytest.raises(EOFError):
        helper.open_visualizer('synthetic.json', host='127.0.0.1', port=0, data_folder=str(tmp_path), open_browser=False)
    assert started[-1].srv.socket.fileno() == -1


def test_visualizer_no_initialized_server(monkeypatch):
    class Uninitialized:
        options = {}
    monkeypatch.setattr(helper, 'serve', lambda *args, **kwargs: (Uninitialized(), lambda: None))
    with pytest.raises(RuntimeError, match='did not initialize'):
        helper.open_visualizer('synthetic.json', open_browser=False)



def test_visualizer_failure_after_server_created_before_loop(monkeypatch):
    from types import SimpleNamespace
    closed = []
    server = SimpleNamespace(options={})

    def run():
        server.srv = SimpleNamespace(server_close=lambda: closed.append(True), shutdown=lambda: pytest.fail('shutdown before serving'))
        raise OSError('synthetic pre-loop failure')

    monkeypatch.setattr(helper, 'serve', lambda *args, **kwargs: (server, run))
    with pytest.raises(OSError, match='synthetic pre-loop failure'):
        helper.open_visualizer('synthetic.json', open_browser=False)
    assert closed == [True]

def test_page_length_and_media_types(monkeypatch, capsys):
    with page_server(monkeypatch) as port:
        for length, status in ((None, 411), ('-1', 400), ('nonnumeric', 400), ('0' * 5000, 400), ('9' * 5000, 413), (str(ui._MAX_BODY + 1), 413)):
            header = b'' if length is None else ('Content-Length: ' + length + '\r\n').encode()
            reply = raw(port, b'/update', b'POST', header)
            assert reply.startswith(('HTTP/1.0 ' + str(status) + ' ').encode()) and len(reply) < 1024
        for kind in (None, 'text/plain', 'application/x-www-form-urlencoded', 'text/json'):
            headers = {} if kind is None else {'Content-Type': kind}
            assert request(port, 'POST', '/update', b'[]', headers)[0] == 415
        for kind in ('application/json', 'Application/JSON; charset=utf-8', 'application/problem+json; charset=utf-8'):
            assert request(port, 'POST', '/update', b'[null,null,null,[],null,null,null,null]', {'Content-Type': kind})[::2] == (204, b'')
        assert request(port, 'POST', '/update', b'[]', {'Content-Type': 'application/json'})[0] == 400
    assert 'Traceback' not in capsys.readouterr().err


def test_page_routes_methods_head_and_bad_targets(monkeypatch, capsys):
    with page_server(monkeypatch) as port:
        for method in ('GET', 'HEAD', 'POST', 'OPTIONS', 'PUT', 'DELETE', 'PATCH', 'TRACE', 'SYNTHETIC'):
            assert raw(port, b'http://[/', method.encode()).startswith(b'HTTP/1.0 400 ')
            assert request(port, method, '/absent')[0] == 404
        for path in ('/', '/layout.json', '/dependencies.json'):
            get = request(port, 'GET', path)
            head = request(port, 'HEAD', path)
            assert get[0] == head[0] == 200 and head[2] == b''
            assert get[1]['Content-Length'] == head[1]['Content-Length']
            for method in ('POST', 'PUT', 'DELETE', 'PATCH', 'TRACE', 'SYNTHETIC'):
                status, headers, _ = request(port, method, path)
                assert status == 405 and set(headers['Allow'].split(', ')) == {'GET', 'HEAD', 'OPTIONS'}
            status, headers, body = request(port, 'OPTIONS', path)
            assert status == 200 and not body and 'HEAD' in headers['Allow']
        for method in ('GET', 'HEAD', 'PUT', 'DELETE', 'PATCH', 'SYNTHETIC'):
            status, headers, body = request(port, method, '/update')
            assert status == 405 and set(headers['Allow'].split(', ')) == {'POST', 'OPTIONS'}
            if method == 'HEAD':
                assert not body
        assert request(port, 'OPTIONS', '/update')[::2] == (200, b'')
        assert request(port, 'GET', 'http://example.invalid/')[0] == 200  # parsing only, no fetch
    assert 'Traceback' not in capsys.readouterr().err


def test_dash_port_forwarding_is_explicit(monkeypatch):
    from types import SimpleNamespace
    seen = []
    monkeypatch.setattr(ui, 'make_app', lambda: SimpleNamespace(run=lambda **kwargs: seen.append(kwargs)))
    ui.serve(port=0, dash=True)
    assert seen == [{'host': '127.0.0.1', 'port': 0}]
    assert 'Dash 4.4.1 treats zero' in ui.serve.__doc__
