"""Qualtran's Dash app (qualtran/surface_code/ui.py, Qualtran 0.7.0; Copyright 2024 Google LLC, Apache-2.0):
`ui.make_app()` and `ui.app` where dash is installed, and `ui.serve()`'s standard-library page where it is not.
The app's title, layout and callbacks are recorded from Qualtran's own app (with dash 4.4.1) by
tools/qualtran_ui_fixture.py in tests/estimate_port/qualtran_layout.json."""

from __future__ import annotations

import base64
import http.client
import json
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

from stabilizer_qec.estimate.qualtran import ui

GOLDEN = json.loads((pathlib.Path(__file__).parent / "qualtran_layout.json").read_text())
# The fields of Dash's callback list that make the callback's signature (the others are Dash's own options,
# whose set changes between Dash versions).
SIGNATURE = ("output", "inputs", "state", "prevent_initial_call", "clientside_function")
ALGORITHM = ["qubits", "Measurements", "Ts", "Toffolis", "Rotations", "Rotation Depth"]
GIDNEY = (1e-4, 1e-3, ui._GIDNEY_FOWLER_MODEL, [10, 10, 10, 10, 10, 10], "GidneyFowler", "GidneyFowlerCCZ", 1, "BeverlandEtAlRotationCost")
BEVERLAND = (3e-4, 5e-3, ui._BEVERLAND_MODEL, [250, 77, 10**7, 10**6, 300, 40], "BeverlandEtAl", "FifteenToOne933", 3, "SevenDigitsOfPrecisionConstantCost")
# numpy's global seed the standard-library server is started with (Beverland's runtime plot draws from it).
SEED = 7
PAGE = pathlib.Path(ui.__file__).parent / "_ui_static" / "ui.html"
# The page's plotly.js tag (cdn.plot.ly's plotly.js 3.7.0, pinned by the hash of plotly 6.9.0's copy), and the
# tag the server puts in its place where plotly is installed.
CDN_TAG = b'<script id="plotly-js" src="https://cdn.plot.ly/plotly-3.7.0.min.js" integrity="sha384-l4G4qURPwALv583BKlynU/4LiUx0rk9jcTX58Aq+Jc+jgWb52zeH2geZkSS3UPPs" crossorigin="anonymous" charset="utf-8"></script>'
LOCAL_TAG = b'<script id="plotly-js" src="plotly.min.js" charset="utf-8"></script>'
HIDE_PLOTLY = 'import sys\nsys.modules["plotly"] = None  # as if plotly were not installed\n'


def _installed_plotly_js():
    """The installed plotly package's plotly.js (the copy Dash serves), or None without plotly."""
    try:
        import plotly
    except ImportError:
        return None
    return (pathlib.Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_bytes()


def _page(local_plotly):
    """The page the standard-library server sends: ui.html, its plotly.js the installed plotly's or cdn.plot.ly's."""
    page = PAGE.read_bytes()
    assert page.count(CDN_TAG) == 1
    return page.replace(CDN_TAG, LOCAL_TAG) if local_plotly else page


def _encode(obj):
    return obj.to_plotly_json()


def _walk(component):
    """Each component of a layout (as ``to_plotly_json()`` dicts), depth first."""
    if hasattr(component, "to_plotly_json"):
        component = component.to_plotly_json()
    if isinstance(component, dict) and {"namespace", "type", "props"} <= set(component):
        yield component
        yield from _walk(component["props"].get("children"))
    elif isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)


def _as_json(value):
    """``value`` as JSON gives it back (numpy's numbers and arrays as Python's, tuples as lists)."""

    def plain(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))

    return json.loads(json.dumps(value, default=plain))


def _debase(x):
    """plotly's typed arrays ({"dtype", "bdata"}) as lists."""
    if isinstance(x, dict) and {"dtype", "bdata"} <= set(x):
        return np.frombuffer(base64.b64decode(x["bdata"]), dtype=np.dtype(x["dtype"])).tolist()
    if isinstance(x, dict):
        return {k: _debase(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_debase(v) for v in x]
    return x


def test_mirror_layout_equals_qualtran():
    """Without dash, the layout ``serve()`` sends is Qualtran's: the same 92 components, ids and props."""
    layout = ui._layout()
    assert json.dumps(layout.to_plotly_json(), default=_encode, sort_keys=True) == json.dumps(GOLDEN["layout"], sort_keys=True)
    components = list(_walk(layout))
    assert len(components) == 92
    ids = [c["props"]["id"] for c in components if "id" in c["props"]]
    assert ids[:4] == ["physical_error_rate", "error_budget", "estimation_model", {"type": "algorithm", "property": "qubits"}]
    assert ids[-3:] == ["qubit-pie-chart", "runtime-container", "runtime-plot"] and len(ids) == 22
    # Qualtran's builders, by its names
    assert [c.to_plotly_json()["type"] for c in ui.input_components()][:3] == ["Table", "P", "RadioItems"]
    assert len(ui.algorithm_summary_components()) == 2 and len(ui.create_ouputs()) == 1
    assert [i.component_id_str() for i in ui._ALL_INPUTS] == [i["id"] for i in GOLDEN["callbacks"][0]["inputs"]]


def test_layout_json_equals_qualtran():
    """`make_app()` is Qualtran's app: its title and its layout's ``to_plotly_json()``."""
    pytest.importorskip("dash")
    app = ui.make_app()
    assert app.title == GOLDEN["title"] == "Qualtran Resource Estimation"
    assert json.dumps(app.layout.to_plotly_json(), default=_encode, sort_keys=True) == json.dumps(GOLDEN["layout"], sort_keys=True)
    assert ui.app is ui.app and ui.app.title == GOLDEN["title"]


def test_ci_pins_the_goldens_dash():
    """CI installs the dash this file's goldens were recorded with (a later dash may serialize components
    differently), and the `dash` extra's floor is that version."""
    root = pathlib.Path(__file__).parents[2]
    ci = root / ".github" / "workflows" / "ci.yml"
    if not ci.exists():
        pytest.skip("needs the repository's .github/workflows/ci.yml")
    assert GOLDEN["dash"] == "4.4.1"
    loops = [line for line in ci.read_text().splitlines() if line.lstrip().startswith("for p in ") and " plotly " in line]
    assert len(loops) == 1 and re.findall(r"(?<=\s)dash\S*", loops[0]) == ["dash==" + GOLDEN["dash"]]
    assert re.findall(r"^dash = .*$", (root / "pyproject.toml").read_text(), re.M) == ['dash = ["dash>=' + GOLDEN["dash"] + '"]']


@pytest.mark.parametrize("server", ["standard library", "dash"])
def test_callback_signature(server):
    """The one callback: its nine outputs and eight inputs (the algorithm's six by the pattern
    ``{'type': 'algorithm', 'property': ALL}``), as Dash lists Qualtran's."""
    want = GOLDEN["callbacks"]
    assert len(want) == 1 and want[0]["output"].count("...") == 8 and len(want[0]["inputs"]) == 8
    assert want[0]["inputs"][3]["id"] == '{"property":["ALL"],"type":"algorithm"}'
    if server == "dash":
        pytest.importorskip("dash")
        have = ui.make_app()._callback_list
    else:
        have = ui._dependencies()
    assert [{k: c[k] for k in SIGNATURE} for c in have] == [{k: c[k] for k in SIGNATURE} for c in want]


def _dash_payload(args):
    """Dash's request to ``/_dash-update-component`` for `update`'s arguments ``args``."""
    callback = GOLDEN["callbacks"][0]
    outputs = [dict(zip(("id", "property"), o.rsplit(".", 1))) for o in callback["output"][2:-2].split("...")]
    inputs = []
    for dep, value in zip(callback["inputs"], args):
        if dep["id"].startswith("{"):
            inputs.append([{"id": {"type": "algorithm", "property": p}, "property": "value", "value": v} for p, v in zip(ALGORITHM, value)])
        else:
            inputs.append({"id": dep["id"], "property": dep["property"], "value": value})
    return {"output": callback["output"], "outputs": outputs, "inputs": inputs, "changedPropIds": [], "state": []}


def _dash_outputs(response):
    """The nine output values of Dash's response, in the callback's order."""
    body = response.get_json()
    assert body["multi"] is True
    order = [o.rsplit(".", 1) for o in GOLDEN["callbacks"][0]["output"][2:-2].split("...")]
    return [_debase(body["response"][i][p]) for i, p in order]


def test_dash_app_round_trip():
    """Through Dash's own server: the outputs are `update`'s (its figures as plotly figures, which add plotly's
    template as Qualtran's do), a missing input gives Dash's 204, and an error Dash's 500."""
    pytest.importorskip("dash")
    import plotly.io

    client = ui.make_app().server.test_client()
    for args in (GIDNEY, BEVERLAND):
        np.random.seed(5)
        response = client.post("/_dash-update-component", json=_dash_payload(args))
        assert response.status_code == 200
        np.random.seed(5)
        out = ui.update(*args)
        want = [json.loads(plotly.io.to_json(o.to_plotly())) if isinstance(o, ui.Figure) else o for o in out]
        assert _dash_outputs(response) == _debase(_as_json(want))
        assert "template" in want[0]["layout"]
    assert client.post("/_dash-update-component", json=_dash_payload(GIDNEY[:1] + (None,) + GIDNEY[2:])).status_code == 204
    algorithm = [10, None, 10, 10, 10, 10]
    assert client.post("/_dash-update-component", json=_dash_payload(GIDNEY[:3] + (algorithm,) + GIDNEY[4:])).status_code == 204


def test_dash_app_against_qualtran():
    """Where Qualtran's app runs, ours answers Dash's requests as it does: the same layout, dependencies and
    callback responses (plotly's typed arrays read as lists; numpy's global seed set before each request)."""
    pytest.importorskip("dash")
    try:
        from qualtran.surface_code import ui as their_ui
    except ImportError:
        pytest.skip("Qualtran's ui needs qualtran, dash, plotly and pandas")
    ours, theirs = ui.make_app().server.test_client(), their_ui.app.server.test_client()
    assert ours.get("/_dash-layout").get_json() == theirs.get("/_dash-layout").get_json()
    assert ours.get("/_dash-dependencies").get_json() == theirs.get("/_dash-dependencies").get_json()
    for args in (GIDNEY, BEVERLAND, BEVERLAND[:6] + (1, "BeverlandEtAlRotationCost")):
        responses = []
        for client in (ours, theirs):
            np.random.seed(11)
            responses.append(client.post("/_dash-update-component", json=_dash_payload(args)))
        assert [r.status_code for r in responses] == [200, 200]
        assert _dash_outputs(responses[0]) == _dash_outputs(responses[1]), args


def test_update_json_is_strict():
    """The standard-library server writes `update`'s outputs as JSON that browsers read: numpy's values as
    Python's and a number that is not finite as null, as plotly's encoder (which writes Dash's responses) does."""
    value = (ui.Figure([{"x": np.array([1.5, np.nan]), "y": [np.int64(2), float("inf")], "s": np.str_("a")}]), {"display": "none"}, np.float32(-np.inf), [np.bool_(True)])
    want = [{"data": [{"x": [1.5, None], "y": [2, None], "s": "a"}], "layout": {}}, {"display": "none"}, None, [True]]
    assert ui._strict_json(value) == want
    json.dumps(ui._strict_json(value), allow_nan=False)
    try:
        from plotly.io.json import to_json_plotly
    except ImportError:
        return
    assert json.loads(to_json_plotly(value)) == want


HIDE_DASH = """
import sys
from stabilizer_qec.estimate.qualtran import ui
assert "dash" not in sys.modules and "plotly" not in sys.modules, "ui imported dash or plotly"
sys.modules["dash"] = None  # as if dash were not installed
for get in (lambda: ui.app, ui.make_app, lambda: ui.serve(dash=True)):
    try:
        get()
    except ImportError as ex:
        print(ex)
    else:
        raise SystemExit("no ImportError")
for probe in (lambda: hasattr(ui, "app"), lambda: getattr(ui, "app", None)):  # as the module's docstring says
    try:
        probe()
    except ImportError:
        print("ImportError")
assert ui.update(*{args!r})[4] == "47.5"


class Served(Exception):
    pass


def page_server(host, port):
    raise Served(host, port)


ui._page_server = page_server  # serve() without dash serves the standard library's page
try:
    ui.serve()
except Served as ex:
    print("page", ex.args)
"""


def test_import_without_dash():
    """`ui` imports without importing dash (or plotly), and its estimates run; the app (``ui.app``,
    `make_app`, ``serve(dash=True)``) raises an ImportError naming dash (so do ``hasattr`` and ``getattr``
    with a default on ``ui.app``, as the docstring says), and ``serve()`` serves the standard library's page
    and says nothing on stderr."""
    code = HIDE_DASH.format(args=GIDNEY)
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert lines == ["make_app needs dash: pip install dash (serve() runs the app without it)"] * 3 + ["ImportError"] * 2 + ["page ('127.0.0.1', 8050)"]
    assert run.stderr == ""
    assert "ImportError" in ui.__doc__ and 'hasattr(ui, "app")' in ui.__doc__


BROKEN_DASH = """
import sys
sys.path.insert(0, {path!r})  # a dash that is installed but does not import
from stabilizer_qec.estimate.qualtran import ui
for get in (lambda: ui.app, ui.make_app, lambda: ui.serve(dash=True)):
    try:
        get()
    except ImportError as ex:
        print(ex)
    else:
        raise SystemExit("no ImportError")


class Served(Exception):
    pass


def page_server(host, port):
    raise Served(host, port)


ui._page_server = page_server
try:
    ui.serve()
except Served as ex:
    print("page", ex.args)
"""


def test_broken_dash(tmp_path):
    """Where dash is installed but does not import (a Flask or Werkzeug it does not work with, say), the app
    raises an ImportError that says so (not "pip install dash"), and ``serve()`` serves the standard library's
    page, saying why on stderr in one line."""
    (tmp_path / "dash").mkdir()
    (tmp_path / "dash" / "__init__.py").write_text("raise ImportError(\"cannot import name 'url_quote' from 'werkzeug.urls'\")\n")
    run = subprocess.run([sys.executable, "-c", BROKEN_DASH.format(path=str(tmp_path))], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    why = "make_app needs dash, which is installed but does not import (ImportError: cannot import name 'url_quote' from 'werkzeug.urls')"
    assert run.stdout.splitlines() == [why] * 3 + ["page ('127.0.0.1', 8050)"]
    assert run.stderr == why + "; serve() runs the app on the standard library's page instead\n"


MAIN = """
import runpy, socketserver, sys
if {hide_dash}:
    sys.modules["dash"] = None  # as if dash were not installed
else:
    import dash
    dash.Dash.run = lambda self, *args, **kwargs: print("Dash.run", args, sorted(kwargs.items()))
# The page's server binds nothing, and Ctrl-C comes as it starts serving.
socketserver.TCPServer.server_bind = socketserver.TCPServer.server_activate = lambda self: None


def interrupt(self, poll_interval=0.5):
    raise KeyboardInterrupt


socketserver.BaseServer.serve_forever = interrupt
runpy.run_module("stabilizer_qec.estimate.qualtran.ui", run_name="__main__", alter_sys=True)  # python -m
"""


@pytest.mark.parametrize("with_dash", [False, True], ids=["page", "dash"])
def test_run_as_main(with_dash):
    """``python -m stabilizer_qec.estimate.qualtran.ui`` runs `serve()` with its defaults, as Qualtran's
    ``python -m qualtran.surface_code.ui`` runs its app: Dash's ``app.run`` where dash is installed, else the
    standard library's page (until Ctrl-C), with nothing on stderr."""
    if with_dash:
        pytest.importorskip("dash")
    run = subprocess.run([sys.executable, "-c", MAIN.format(hide_dash=not with_dash)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    want = "Dash.run () [('host', '127.0.0.1'), ('port', 8050)]" if with_dash else "Qualtran Resource Estimation is running on http://127.0.0.1:8050/"
    assert (run.stdout, run.stderr) == (want + "\n", "")


class _Served:
    """``ui.serve(port=0, dash=False)`` in a process of its own (after ``prelude``, and with numpy's global seed
    `SEED`), and the URL it prints."""

    def __init__(self, stderr_path, prelude=""):
        self.stderr_path = stderr_path
        self._stderr = open(stderr_path, "w")
        code = f"import numpy as np\nfrom stabilizer_qec.estimate.qualtran import ui\n{prelude}\nnp.random.seed({SEED})\nui.serve(port=0, dash=False)\n"
        self.proc = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE, stderr=self._stderr, text=True)
        lines: queue.Queue = queue.Queue()

        def read():
            for line in self.proc.stdout:
                lines.put(line)
            lines.put(None)  # the process closed its output: it ended

        threading.Thread(target=read, daemon=True).start()
        try:
            first = lines.get(timeout=120)
        except queue.Empty:
            first = None
        if first is None:
            self.close()
            raise RuntimeError("serve printed no URL:\n" + stderr_path.read_text())
        self.first_line = first.rstrip("\n")
        self.url = self.first_line.rsplit(" ", 1)[-1]

    def request(self, path, body=None, method=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        except urllib.error.HTTPError as ex:
            return ex.code, ex.headers.get("Content-Type"), ex.read()

    def status(self, method, path, headers):
        """The status of a request with these headers and no body."""
        return self.answer(method, path, headers)[0]

    def answer(self, method, path, headers):
        """The status and body of the answer to a request with these headers and no body."""
        host, port = urllib.parse.urlsplit(self.url).netloc.rsplit(":", 1)
        conn = http.client.HTTPConnection(host, int(port), timeout=60)
        try:
            conn.putrequest(method, path)
            for name, value in headers.items():
                conn.putheader(name, value)
            conn.endheaders()
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=60)
        self.proc.stdout.close()
        self._stderr.close()


@pytest.fixture
def served(tmp_path):
    server = _Served(tmp_path / "stderr.txt")
    yield server
    server.close()


def test_serve_fallback_round_trip(served):
    """``serve(dash=False)``: the standard library's server sends the page, Qualtran's layout and callback, and
    answers a POST of `update`'s arguments to ``/update`` with its outputs as JSON (204 where `update` raises
    `PreventUpdate`)."""
    assert served.first_line.startswith("Qualtran Resource Estimation is running on http://127.0.0.1:") and served.url.endswith("/")
    status, kind, page = served.request("")
    assert (status, kind) == (200, "text/html; charset=utf-8")
    local_plotly = _installed_plotly_js() is not None
    assert page == _page(local_plotly)
    assert (LOCAL_TAG if local_plotly else CDN_TAG) in page and b"layout.json" in page and b"dependencies.json" in page
    status, kind, layout = served.request("layout.json")
    assert (status, kind) == (200, "application/json")
    assert layout.decode() == json.dumps(GOLDEN["layout"], sort_keys=True)
    status, kind, deps = served.request("dependencies.json")
    assert (status, kind) == (200, "application/json")
    assert [{k: c[k] for k in SIGNATURE} for c in json.loads(deps)] == [{k: c[k] for k in SIGNATURE} for c in GOLDEN["callbacks"]]

    status, kind, body = served.request("update", list(GIDNEY))
    assert (status, kind) == (200, "application/json")
    assert json.loads(body) == _as_json(ui.update(*GIDNEY))
    # Beverland's runtime plot draws its factory counts from numpy's global random state, which the server was
    # started with seeded by SEED; Gidney-Fowler's draws nothing, so this is the server's first draw, as here.
    status, kind, body = served.request("update", list(BEVERLAND))
    np.random.seed(SEED)
    want = _as_json(ui.update(*BEVERLAND))
    assert (status, kind) == (200, "application/json")
    assert json.loads(body) == want and len(want[2]["data"][0]["x"]) == 5

    for missing in (GIDNEY[:1] + (None,) + GIDNEY[2:], GIDNEY[:3] + ([10, 10, None, 10, 10, 10],) + GIDNEY[4:]):
        assert served.request("update", list(missing)) == (204, None, b"")


def test_serve_fallback_refusals(served):
    """What the standard-library server refuses: an unknown path (404), a body that is not JSON (nested past
    Python's recursion limit too) or not `update`'s eight arguments (400), a POST without a Content-Length
    (411), with one that is not a number or is negative (400) or over a megabyte (413), a method a path does not
    take (405), and an error in `update` (500, its traceback on the server's stderr)."""
    assert served.request("missing")[0] == 404
    assert served.request("update", b"{not json")[0] == 400
    assert served.request("update", b"[" * 200000)[0] == 400
    assert served.status("POST", "/update", {}) == 411
    assert served.status("POST", "/update", {"Content-Length": "-1"}) == 400
    # 411 only where there is no Content-Length; one that is not a number (as Werkzeug, Dash's server, reads
    # one: ASCII digits, no "+" or "_") is a 400.
    assert served.answer("POST", "/update", {}) == (411, b"411 Length Required\n")
    assert served.answer("POST", "/update", {"Content-Length": "-1"}) == (400, b"400 Bad Request: the Content-Length is negative\n")
    for length in ("abc", "", "1_0", "+5", "1.5", "0x10", "\u00b2"):  # the last: a superscript 2, a digit to str.isdigit()
        assert served.answer("POST", "/update", {"Content-Length": length}) == (400, b"400 Bad Request: the Content-Length is not a number\n"), length
    assert served.status("POST", "/update", {"Content-Length": str((1 << 20) + 1)}) == 413
    assert served.request("update", {"args": list(GIDNEY)})[0] == 400
    assert served.request("update", list(GIDNEY)[:7])[0] == 400
    assert served.request("layout.json", list(GIDNEY))[0] == 405 and served.request("update")[0] == 405
    status, _, body = served.request("update", list(GIDNEY[:4]) + ["NoSuchScheme"] + list(GIDNEY[5:]))
    assert status == 500 and b"NoSuchScheme" not in body
    assert "KeyError: 'NoSuchScheme'" in served.stderr_path.read_text()
    assert served.request("update", list(GIDNEY))[0] == 200  # the server still runs


UNENCODABLE = """
_update = ui.update


def update(*args):  # an output JSON cannot write, where the factory count is 2
    out = _update(*args)
    return out[:4] + (object(),) + out[5:] if args[6] == 2 else out


ui.update = update
"""


def test_serve_fallback_unencodable_answer(tmp_path):
    """An answer the standard-library server cannot write as JSON is a 500 (its traceback on the server's
    stderr, as an error in `update` is), not a dropped connection, and the server keeps running."""
    server = _Served(tmp_path / "stderr.txt", prelude=UNENCODABLE)
    try:
        assert server.request("update", list(GIDNEY[:6]) + [2] + list(GIDNEY[7:])) == (500, "text/plain; charset=utf-8", b"500 Internal Server Error\n")
        log = server.stderr_path.read_text()
        assert "Exception on /update [POST]" in log and "TypeError: Object of type object is not JSON serializable" in log
        status, kind, body = server.request("update", list(GIDNEY))
        assert (status, kind) == (200, "application/json") and json.loads(body) == _as_json(ui.update(*GIDNEY))
    finally:
        server.close()


@pytest.mark.parametrize("plotly", ["installed", "hidden"])
def test_serve_fallback_plotly_js(tmp_path, plotly):
    """The page's plotly.js: where plotly is installed, the package's own copy (``plotly/package_data/
    plotly.min.js``, the one Dash serves by default) at ``/plotly.min.js``, in place of cdn.plot.ly's; without
    plotly, cdn.plot.ly's plotly.js 3.7.0 pinned by its hash, and no ``/plotly.min.js``."""
    js = _installed_plotly_js()
    if plotly == "installed" and js is None:
        pytest.skip("needs plotly")
    server = _Served(tmp_path / "stderr.txt", prelude=HIDE_PLOTLY if plotly == "hidden" else "")
    try:
        status, kind, page = server.request("")
        assert (status, kind) == (200, "text/html; charset=utf-8")
        if plotly == "installed":
            assert page == _page(True) and CDN_TAG not in page
            assert server.request("plotly.min.js") == (200, "text/javascript; charset=utf-8", js)
            assert server.request("plotly.min.js", b"[]")[0] == 405
        else:
            assert page == _page(False) == PAGE.read_bytes()
            assert server.request("plotly.min.js")[0] == 404
        assert server.stderr_path.read_text().count("Exception") == 0
    finally:
        server.close()


def _node():
    """node, where it is installed and its fetch (which the page's smoke test uses) is global and prints no warning;
    else None. fetch is global from node 18. Node 18.0-18.12 and 19.0 print "ExperimentalWarning: The Fetch API is
    an experimental feature" on stderr at its first call, which the test's empty-stderr check would take for the
    page's own output; node dropped that warning in 18.13.0 and 19.1.0 (nodejs/node#45287)."""
    node = shutil.which("node")
    if node is None:
        return None
    version = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=60).stdout
    match = re.match(r"v(\d+)\.(\d+)\.", version)
    if match is None:
        return None
    major_minor = (int(match.group(1)), int(match.group(2)))
    return node if major_minor >= (18, 13) and major_minor != (19, 0) else None


@pytest.mark.parametrize(("version", "used"), [("v16.20.1", False), ("v18.0.0", False), ("v18.12.0", False), ("v18.13.0", True), ("v18.17.0", True), ("v19.0.1", False), ("v19.1.0", True), ("v20.11.1", True), ("v24.18.0", True), ("", False)])
def test_node_gate(monkeypatch, version, used):
    """The node smoke test runs only under a node whose fetch prints no ExperimentalWarning (18.13 on, but not
    19.0), and skips where node prints no version."""
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, version + "\n" if version else "", "")

    monkeypatch.setattr(shutil, "which", lambda name: "/opt/node/bin/node" if name == "node" else None)
    monkeypatch.setattr(subprocess, "run", run)
    assert _node() == ("/opt/node/bin/node" if used else None)
    assert calls == [["/opt/node/bin/node", "--version"]]
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert _node() is None and len(calls) == 1


def _initial_args():
    """`update`'s arguments as the layout gives them at first: each input's value, the pattern's in layout order."""
    components = [c for c in _walk(GOLDEN["layout"]) if "id" in c["props"]]
    args = []
    for dep in GOLDEN["callbacks"][0]["inputs"]:
        if dep["id"].startswith("{"):
            pattern = json.loads(dep["id"])
            args.append([c["props"][dep["property"]] for c in components if isinstance(c["props"]["id"], dict) and c["props"]["id"]["type"] == pattern["type"]])
        else:
            args.append(next(c["props"][dep["property"]] for c in components if c["props"]["id"] == dep["id"]))
    return args


def _shown(out):
    """`update`'s outputs as the page shows them: the figures it draws, the styles it sets, the text and the
    read-only inputs' values."""
    out = _as_json(out)
    return {"qubit-pie-chart": out[0], "runtime-container": out[1], "runtime-plot": out[2], "magic-name": "".join(out[3]), "total_magic": str(out[4]), "magic-factory-container": out[5], "min_factory_count": str(out[6]), "duration-container": out[7], "duration": str(out[8])}


def test_page_script_in_node(served, tmp_path):
    """The page's script (``_ui_static/ui.html``), run by node against the standard-library server in a small DOM
    with a stand-in for plotly.js (tests/estimate_port/qualtran_ui_page.js): it draws every component of the
    layout, posts the inputs' first values once on load as Dash does, and shows `update`'s outputs. A number
    input sends its value on Enter or on leaving it, null where the value is out of range (a 204, which keeps the
    outputs), and nothing where the number did not change; of two requests, the later one's answer is shown even
    when the earlier's comes last; the title says "Updating..." while a request runs; a graph that is shown is
    resized; the page writes nothing to the console."""
    node = _node()
    if node is None:
        pytest.skip("needs node 18.13 or later, other than 19.0, whose fetch prints no ExperimentalWarning")
    page = served.request("")[2]
    (tmp_path / "page.html").write_bytes(page)
    qubits = json.dumps({"type": "algorithm", "property": "qubits"}, sort_keys=True, separators=(",", ":"))
    actions = [
        {"do": "load"},
        {"do": "type", "id": "error_budget", "text": "2", "commit": "enter"},
        {"do": "type", "id": "error_budget", "text": "0.01", "commit": "blur"},
        {"do": "type", "id": "error_budget", "text": "0.01", "commit": "enter"},
        {"do": "race", "first": {"do": "type", "id": qubits, "text": "100", "commit": "enter"}, "second": {"do": "type", "id": qubits, "text": "200", "commit": "enter"}},
        {"do": "radio", "id": "estimation_model", "value": ui._BEVERLAND_MODEL},
    ]
    (tmp_path / "actions.json").write_text(json.dumps(actions))
    harness = pathlib.Path(__file__).parent / "qualtran_ui_page.js"
    run = subprocess.run([node, str(harness), str(tmp_path / "page.html"), served.url, str(tmp_path / "actions.json")], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0 and run.stderr == "", run.stderr
    got = json.loads(run.stdout)

    ids = [c["props"]["id"] for c in _walk(GOLDEN["layout"]) if "id" in c["props"]]
    assert got["ids"] == [json.dumps(i, sort_keys=True, separators=(",", ":")) if isinstance(i, dict) else i for i in ids] and len(ids) == 22
    assert got["notes"] == []  # every component drawn

    first = _initial_args()
    assert first == [1e-4, 1e-3, ui._GIDNEY_FOWLER_MODEL, [10] * 6, "BeverlandEtAl", "FifteenToOne733", 1, "BeverlandEtAlRotationCost"]
    over = first[:1] + [None] + first[2:]  # an error budget of 2, over the input's max of 1
    budget = first[:1] + [0.01] + first[2:]
    hundred, two_hundred = (budget[:3] + [[n] + budget[3][1:]] + budget[4:] for n in (100, 200))
    beverland = two_hundred[:2] + [ui._BEVERLAND_MODEL] + two_hundred[3:]
    np.random.seed(SEED)  # the server's first draw from numpy's global random state: Gidney-Fowler's draw none
    want_beverland = _shown(ui.update(*beverland))
    want = [
        ([(first, 200)], _shown(ui.update(*first)), ["qubit-pie-chart"]),
        ([(over, 204)], _shown(ui.update(*first)), []),
        ([(budget, 200)], _shown(ui.update(*budget)), ["qubit-pie-chart"]),
        ([], _shown(ui.update(*budget)), []),
        ([(hundred, 200), (two_hundred, 200)], _shown(ui.update(*two_hundred)), ["qubit-pie-chart"]),
        ([(beverland, 200)], want_beverland, ["qubit-pie-chart", "runtime-plot"]),
    ]
    assert len(got["steps"]) == len(want)
    for step, (posts, outputs, resized) in zip(got["steps"], want):
        assert step["posts"] == [{"args": args, "status": status, "title": "Updating..."} for args, status in posts]
        assert step["outputs"] == outputs
        assert (step["resized"], step["title"], step["logged"]) == (resized, "Qualtran Resource Estimation", [])
    assert want_beverland["runtime-container"] == {"display": "block"} and want_beverland["duration-container"] == {"display": "none"}
