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
    `make_app`, ``serve(dash=True)``) raises an ImportError naming dash, and ``serve()`` serves the
    standard library's page."""
    code = HIDE_DASH.format(args=GIDNEY)
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert lines == ["make_app needs dash: pip install dash (serve() runs the app without it)"] * 3 + ["page ('127.0.0.1', 8050)"]


class _Served:
    """``ui.serve(port=0, dash=False)`` in a process of its own, and the URL it prints."""

    def __init__(self, stderr_path):
        self.stderr_path = stderr_path
        self._stderr = open(stderr_path, "w")
        code = "from stabilizer_qec.estimate.qualtran import ui\nui.serve(port=0, dash=False)\n"
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
        host, port = urllib.parse.urlsplit(self.url).netloc.rsplit(":", 1)
        conn = http.client.HTTPConnection(host, int(port), timeout=60)
        try:
            conn.putrequest(method, path)
            for name, value in headers.items():
                conn.putheader(name, value)
            conn.endheaders()
            return conn.getresponse().status
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
    assert page == (pathlib.Path(ui.__file__).parent / "_ui_static" / "ui.html").read_bytes()
    assert b"https://cdn.plot.ly/plotly-3.7.0.min.js" in page and b"layout.json" in page and b"dependencies.json" in page
    status, kind, layout = served.request("layout.json")
    assert (status, kind) == (200, "application/json")
    assert layout.decode() == json.dumps(GOLDEN["layout"], sort_keys=True)
    status, kind, deps = served.request("dependencies.json")
    assert (status, kind) == (200, "application/json")
    assert [{k: c[k] for k in SIGNATURE} for c in json.loads(deps)] == [{k: c[k] for k in SIGNATURE} for c in GOLDEN["callbacks"]]

    status, kind, body = served.request("update", list(GIDNEY))
    assert (status, kind) == (200, "application/json")
    assert json.loads(body) == _as_json(ui.update(*GIDNEY))
    # Beverland's runtime plot draws its factory counts from numpy's global random state (the server's own);
    # its first point, the minimum factory count, does not depend on it.
    status, _, body = served.request("update", list(BEVERLAND))
    out, want = json.loads(body), _as_json(ui.update(*BEVERLAND))
    assert status == 200 and out[:2] + out[3:] == want[:2] + want[3:]
    trace, want_trace = out[2]["data"][0], want[2]["data"][0]
    assert (trace["x"][0], trace["y"][0], trace["text"][0]) == (want_trace["x"][0], want_trace["y"][0], want_trace["text"][0])
    assert out[2]["layout"] == want[2]["layout"]

    for missing in (GIDNEY[:1] + (None,) + GIDNEY[2:], GIDNEY[:3] + ([10, 10, None, 10, 10, 10],) + GIDNEY[4:]):
        assert served.request("update", list(missing)) == (204, None, b"")


def test_serve_fallback_refusals(served):
    """What the standard-library server refuses: an unknown path (404), a body that is not JSON (nested past
    Python's recursion limit too) or not `update`'s eight arguments (400), a POST without a usable
    Content-Length (411, 400) or over a megabyte (413), a method a path does not take (405), and an error in
    `update` (500, its traceback on the server's stderr)."""
    assert served.request("missing")[0] == 404
    assert served.request("update", b"{not json")[0] == 400
    assert served.request("update", b"[" * 200000)[0] == 400
    assert served.status("POST", "/update", {}) == 411
    assert served.status("POST", "/update", {"Content-Length": "-1"}) == 400
    assert served.status("POST", "/update", {"Content-Length": str((1 << 20) + 1)}) == 413
    assert served.request("update", {"args": list(GIDNEY)})[0] == 400
    assert served.request("update", list(GIDNEY)[:7])[0] == 400
    assert served.request("layout.json", list(GIDNEY))[0] == 405 and served.request("update")[0] == 405
    status, _, body = served.request("update", list(GIDNEY[:4]) + ["NoSuchScheme"] + list(GIDNEY[5:]))
    assert status == 500 and b"NoSuchScheme" not in body
    assert "KeyError: 'NoSuchScheme'" in served.stderr_path.read_text()
    assert served.request("update", list(GIDNEY))[0] == 200  # the server still runs
