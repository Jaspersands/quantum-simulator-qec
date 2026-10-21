"""The QDK's report of an estimate as `stabilizer_qec.estimate.azure.EstimatorResult` draws it, against
tests/estimate_port/qdk_reports.json.gz (recorded by tools/qdk_report_fixtures.py from qdk 1.33.1 and
qsharp_widgets 1.33.1): for forty results (25 recorded estimates, a failed one, batches with failed, frontier
and differing items) the HTML of ``_repr_html_()``, ``summary``, ``[:]`` and other slices and the diagrams, with
what the QDK prints and the exceptions it raises, without markdown and with markdown and python-markdown-math;
and the qsharp_widgets charts (the SpaceChart donut, the estimates overview's scatter chart, their colors, axis
ticks and numbers as JavaScript writes them) against the widgets' own drawing, element by element."""

from __future__ import annotations

import copy
import gzip
import json
import math
import pathlib
import pickle
import re
import sys
import warnings
import xml.etree.ElementTree as ET

import pytest

from stabilizer_qec.estimate import azure as az
from stabilizer_qec.estimate import azure_report as rep

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import qdk_report_fixtures as fx  # noqa: E402

DATA = json.loads(gzip.decompress((pathlib.Path(__file__).parent / "qdk_reports.json.gz").read_bytes()))
RESULTS = {name: DATA[name] for name in ("qdk_estimates", "qdk_review2")}
RENDERINGS = DATA["renderings"]
CASES = {case["name"]: case for case in RENDERINGS["cases"]}
PLAIN = fx.unpack(RENDERINGS["plain"], RENDERINGS["lines"])
MARKDOWN = fx.unpack(RENDERINGS["markdown"], RENDERINGS["lines"])
WIDGET = DATA["widget"]
DRAWN = WIDGET["drawn"]

try:
    import qdk.estimator as qdk_estimator
except ImportError:  # pragma: no cover
    qdk_estimator = None

needs_qdk = pytest.mark.skipif(qdk_estimator is None, reason="qdk is not installed")
# matplotlib 3.9 (the last for Python 3.9) warns on import under pyparsing 3.3: not this package's warning.
pytestmark = pytest.mark.filterwarnings("ignore:'enablePackrat' deprecated")


def _markdown_as_recorded() -> bool:
    if not rep.has_markdown:
        return False
    import importlib.metadata

    import markdown

    return {"markdown": markdown.__version__, "python-markdown-math": importlib.metadata.version("python-markdown-math")} == RENDERINGS["markdown_versions"]


def test_report_fixture_is_the_tools_cases():
    assert RENDERINGS["qdk"] == fx.QDK_VERSION and RENDERINGS["cases"] == json.loads(json.dumps(fx.REPORT_CASES)) and len(CASES) == 40
    assert set(PLAIN) == set(MARKDOWN) == set(CASES) and sum(len(steps) for steps in PLAIN.values()) >= 160
    assert WIDGET["space"] == json.loads(json.dumps(fx.widget_cases(RESULTS)["space"])) and WIDGET["scatter"] == fx.WIDGET_SCATTERS and WIDGET["numbers"] == fx.WIDGET_NUMBERS


@pytest.mark.parametrize("name", sorted(CASES))
def test_report_against_qdk_recorded(name, monkeypatch):
    """Without markdown: every step's HTML (or exception, class and message) and what it printed is the QDK's."""
    monkeypatch.setattr(rep, "has_markdown", False)
    assert fx.render_case(CASES[name], RESULTS, az.EstimatorResult, az.EstimatorError) == PLAIN[name]


@pytest.mark.skipif(not _markdown_as_recorded(), reason="markdown 3.11.2 and python-markdown-math 0.9 (the recorded versions) are not installed")
@pytest.mark.parametrize("name", sorted(CASES))
def test_report_with_markdown_against_qdk_recorded(name):
    """With markdown and python-markdown-math: the explanations and assumptions converted, as the QDK's."""
    assert fx.render_case(CASES[name], RESULTS, az.EstimatorResult, az.EstimatorError, diagrams=False) == MARKDOWN[name]


def _counts():
    return az.LogicalCounts(num_qubits=10, t_count=100, rotation_count=5, rotation_depth=3)


def _qdk_batch_value(data: dict, path: str) -> str:
    """The QDK's side-by-side value: the path's keys followed while each is there, else "N/A"."""
    value = data
    for key in path.split("/"):
        if key not in value:
            return "N/A"
        value = value[key]
    return str(value)


def test_batch_frontier_item_is_na():
    """The QDK's side-by-side table follows each entry's path in each item as it is: a frontier item has no
    top-level counts, so its column is "N/A" there (in the text and in the HTML), where its own report reads its
    first entry."""
    batch = _counts().estimate([{"qubitParams": {"name": "qubit_gate_ns_e3"}}, {"estimateType": "frontier"}])
    entries = [e for g in batch[0]["reportData"]["groups"] for e in g["entries"]]
    expected = {e["label"]: [_qdk_batch_value(batch.data(i), e["path"]) for i in (0, 1)] for e in entries}
    assert sum(v[1] == "N/A" for v in expected.values()) >= 10 and sum(v[1] != "N/A" for v in expected.values()) >= 5
    for text in (str(batch[:]), str(batch.summary), str(batch[0:2])):
        header, *lines = text.splitlines()
        col0 = header.index("0", 6)
        rows = {line[2:col0].rstrip(): line.rstrip() for line in lines if line.startswith("  ")}
        assert set(expected) <= set(rows)
        for label, (first, second) in expected.items():
            assert rows[label][col0:].startswith(first) and rows[label].endswith("  " + second), label
    html_rows = re.findall(r'<td style="font-weight: bold; vertical-align: top; white-space: nowrap">(.*?)</td>\s*<td style="vertical-align: top; white-space: nowrap">(.*?)</td>\s*<td style="vertical-align: top; white-space: nowrap">(.*?)</td>', batch[:]._repr_html_())
    html = {label: [a, b] for label, a, b in html_rows}
    assert {label: html.get(label) for label in expected} == expected
    frontier = batch[1]
    assert "N/A" not in str(frontier.summary) and "N/A" not in frontier.summary.content and frontier["physicalCountsFormatted"]["runtime"] in frontier._repr_html_()


def test_summary_and_slices_are_text_and_html(monkeypatch):
    monkeypatch.setattr(rep, "has_markdown", False)  # the assumptions are listed only with markdown
    counts = _counts()
    one = counts.estimate({"qubitParams": {"name": "qubit_gate_ns_e4"}})
    assert isinstance(one.summary, str) and one.summary.splitlines()[0] == "Physical resource estimates"
    assert one.summary._repr_html_() == one.summary.content and one.summary.content.startswith("\n            <style>\n                .aqre-tooltip {")
    assert one._repr_html_().startswith("\n                <details open>") and one._repr_html_().endswith("<details><summary style=\"display:list-item\"><strong>Assumptions</strong></summary><ul></ul></details>")
    assert one.summary is one.summary
    batch = counts.estimate([{}] * 7)
    assert batch.summary.startswith("(the first 5 of 7 items") and batch.summary._repr_html_() == batch._repr_html_()
    assert batch._repr_html_().startswith("<p><b>Info:</b> <i>The overview table is cut off after 5 items. If you want to see all items, suffix the result variable with <code>[:]</code></i></p>")
    assert "<thead><tr><th>Item</th>" + "".join(f"<th>{i}</th>" for i in range(7)) + "</tr></thead>" in batch[:].content and "<thead><tr><th>Item</th><th>2</th><th>3</th></tr></thead>" in batch[2:4]._repr_html_()
    with pytest.raises(ValueError, match="^Cannot pass slice to '__getitem__' for non-batching job$"):
        _ = one[:]


def test_none_succeeded_prints_when_drawn(capsys):
    """The QDK prints "None of the jobs succeeded" when it draws a batch's table with no succeeded item (at
    construction, and for each slice) and draws nothing; here when the HTML is drawn, once per table."""
    batch = _counts().estimate([{"constraints": {"maxDuration": "1 ns"}}] * 2)
    assert batch.summary == "None of the jobs succeeded" and capsys.readouterr().out == ""
    assert batch._repr_html_() == "" and batch._repr_html_() == "" and capsys.readouterr().out == "None of the jobs succeeded\n"
    sliced = batch[0:1]
    assert sliced.content == sliced.content == "" and capsys.readouterr().out == "None of the jobs succeeded\n"


def test_attributes_only_where_the_qdk_has_them():
    """As the QDK's: a succeeded result has `summary` and `diagram`, a failed one neither, a batch `plot` (and
    the text summary of its overview here), a single result no `plot`."""
    batch = _counts().estimate([{}, {"constraints": {"maxDuration": "1 ns"}}])
    failed = batch[1]
    for obj, name in ((failed, "summary"), (failed, "diagram"), (batch, "diagram"), (batch[0], "plot"), (failed, "plot")):
        with pytest.raises(AttributeError, match=f"^'EstimatorResult' object has no attribute '{name}'$"):
            getattr(obj, name)
        assert not hasattr(obj, name)
    with pytest.raises(az.EstimatorError, match="MaxDurationTooSmall"):
        failed._repr_html_()
    assert callable(batch.plot) and hasattr(batch[0], "summary") and hasattr(batch[0], "diagram")


def test_results_and_drawings_pickle():
    """A result pickles (and copies) after its tables are drawn, and draws them again as the original does; a
    table pickles with its text and HTML, a chart with its SVG and HTML."""
    counts = _counts()
    one, batch = counts.estimate({"estimateType": "frontier"}), counts.estimate([{}, {"constraints": {"maxDuration": "1 ns"}}])
    drawn = (one._repr_html_(), str(one.summary), one.summary.content, one.diagram.space.content, batch._repr_html_(), str(batch.summary), batch.summary.content)
    for again, batch_again in ((pickle.loads(pickle.dumps(one)), pickle.loads(pickle.dumps(batch))), (copy.deepcopy(one), copy.deepcopy(batch)), (copy.copy(one), copy.copy(batch))):
        assert again == one and batch_again == batch and batch_again[1].error.code == batch[1].error.code
        assert (again._repr_html_(), str(again.summary), again.summary.content, again.diagram.space.content, batch_again._repr_html_(), str(batch_again.summary), batch_again.summary.content) == drawn
    for table in (one.summary, batch.summary, batch[:]):
        table_again = pickle.loads(pickle.dumps(table))
        assert type(table_again) is rep.Report and table_again == table and table_again.content == table.content == table_again._repr_html_()
    chart = one.space_chart()
    chart_again = pickle.loads(pickle.dumps(chart))
    assert type(chart_again) is rep.Chart and chart_again == chart and chart_again._repr_html_() == chart._repr_html_()


def test_result_pickled_by_2_1_draws():
    """A result pickled by 2.1, which had no drawn tables to keep (its state lacks them), draws when unpickled
    (and copied), as one pickled here does: a single result, a frontier, and a batch with a failed item."""
    counts = _counts()
    results = [counts.estimate({}), counts.estimate({"estimateType": "frontier"}), counts.estimate([{}, {"constraints": {"maxDuration": "1 ns"}}])]
    drawn = [(r._repr_html_(), str(r.summary), r.summary.content) for r in results]

    def pickled_by_2_1(result) -> bytes:
        """The result as 2.1 made it (no drawn tables, its items' neither) pickled as 2.1 pickles it: 2.1 has no
        __getstate__, so its state is its __dict__."""
        result = copy.deepcopy(result)
        for r in [result] + ([] if result._is_simple else list(dict.values(result))):
            del r.__dict__["_drawn"]
        with pytest.MonkeyPatch.context() as mp:
            mp.delattr(az.EstimatorResult, "__getstate__")
            data = pickle.dumps(result)
        assert b"_drawn" not in data and b"_is_simple" in data
        return data

    for result, (html, text, content) in zip(results, drawn):
        again = pickle.loads(pickled_by_2_1(result))
        assert again == result and (again._repr_html_(), str(again.summary), again.summary.content) == (html, text, content)
        assert hasattr(again, "summary") and set(again.__dict__) == set(result.__dict__)
        assert copy.copy(again).summary.content == copy.deepcopy(again).summary.content == content
    assert pickle.loads(pickled_by_2_1(results[0])).diagram.space.content == results[0].diagram.space.content
    batch = pickle.loads(pickled_by_2_1(results[2]))
    assert batch[0]._drawn == batch[1]._drawn == {} and batch[1].error.code == "Qdk.Qsc.Estimates.MaxDurationTooSmall" and str(batch[0].summary) == str(results[2][0].summary)


def test_diagram():
    one = _counts().estimate({"estimateType": "frontier"})
    diagram = one.diagram
    data = dict(one.data())
    data.pop("reportData")
    assert diagram.data_json == json.dumps(data).replace(" ", "") and "reportData" in one.data()
    assert diagram.vis_lib == "https://cdn-aquavisualization-prod.azureedge.net/resource-estimation/index.js"
    assert diagram.space._repr_html_() == f"\n            <script src={diagram.vis_lib}></script>\n            <re-space-diagram data={diagram.data_json}></re-space-diagram>"
    assert diagram.time.content == f"\n            <script src={diagram.vis_lib}></script>\n            <re-time-diagram data={diagram.data_json}></re-time-diagram>"


@pytest.fixture
def pyplot():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def _figure(plt) -> dict:
    fig = plt.gcf()
    ax = fig.axes[0]
    return {
        "size": list(fig.get_size_inches()),
        "labels": (ax.get_xlabel(), ax.get_ylabel()),
        "scales": (ax.get_xscale(), ax.get_yscale()),
        "points": [c.get_offsets().tolist() for c in ax.collections],
        "markers": [c.get_paths()[0].vertices.round(6).tolist() for c in ax.collections],
        "legend": [t.get_text() for t in ax.get_legend().get_texts()],
        "xticks": list(ax.get_xticks()),
        "xticklabels": [(t.get_text(), t.get_rotation()) for t in ax.get_xticklabels()],
    }


def _plot(result, **kwargs) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*non-interactive.*cannot be shown", category=UserWarning)
        result.plot(**kwargs)


def test_plot(pyplot):
    batch = _counts().estimate([{"qubitParams": {"name": n}} for n in ("qubit_gate_ns_e3", "qubit_gate_us_e3", "qubit_maj_ns_e4", "qubit_maj_ns_e6", "qubit_gate_ns_e4")])
    _plot(batch, labels=["a", "b"])
    fig = _figure(pyplot)
    assert fig["size"] == [15.0, 8.0] and fig["labels"] == ("Runtime", "Physical qubits") and fig["scales"] == ("log", "log")
    assert fig["points"] == [[[batch.data(i)["physicalCounts"]["runtime"], batch.data(i)["physicalCounts"]["physicalQubits"]]] for i in range(5)]
    assert fig["legend"] == ["a", "b", "2", "3", "4"] and fig["xticklabels"][:3] == [("1 ns", 90.0), ("1 µs", 90.0), ("1 ms", 90.0)]
    assert len(set(map(str, fig["markers"][:4]))) == 4 and fig["markers"][4] == fig["markers"][0]


def test_plot_needs_matplotlib(monkeypatch):
    batch = _counts().estimate([{}, {}])
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"^Missing optional 'matplotlib' dependency. To install run: pip install matplotlib$"):
        batch.plot()


@needs_qdk
def test_plot_against_qdk_live(pyplot):
    params = [{"qubitParams": {"name": "qubit_gate_ns_e3"}}, {"qubitParams": {"name": "qubit_maj_ns_e6"}, "qecScheme": {"name": "floquet_code"}}, {"errorBudget": 0.1}]
    counts = {"numQubits": 10, "tCount": 100, "rotationCount": 5, "rotationDepth": 3}
    figures = []
    for result in (qdk_estimator.LogicalCounts(counts).estimate(params), az.LogicalCounts.fromdict(counts).estimate(params)):
        _plot(result, labels=["first"])
        figures.append(_figure(pyplot))
        pyplot.close("all")
    assert figures[0] == figures[1]


# ---------------------------------------------------------------------------------------------
# The charts, against the qsharp_widgets drawing
# ---------------------------------------------------------------------------------------------

_VAR = re.compile(r"^var\(--[\w-]+, (.+)\)$")
_NUMBER = re.compile(r"^-?(\d+(\.\d*)?|\.\d+)(e[-+]?\d+)?$")


def _material(key: str):
    return fx.materialize(key, RESULTS)


def _as_tree(element) -> dict:
    """An ElementTree element as the widget harness writes a node: tag (no namespace), attributes, children
    (its text first)."""
    children = ([element.text] if element.text else []) + [_as_tree(c) for c in element]
    return {"tag": element.tag.split("}")[-1], "attrs": {k.split("}")[-1]: v for k, v in element.attrib.items()}, "children": children}


def _joined(children: list) -> list:
    """Adjacent strings joined, as the DOM's text is."""
    out: list = []
    for c in children:
        if isinstance(c, str) and out and isinstance(out[-1], str):
            out[-1] += c
        else:
            out.append(c)
    return out


# How many units in the last place a drawn number may differ from the widget's. V8's sin and cos (fdlibm's) are
# not the C library's: on macOS, where the widget's drawings were recorded (node v24.18.0), 20 of the donuts' path
# strings differ from the port's, in 32 coordinates, 20 by 1 ulp and 12 by 2; nothing else differs. The C
# libraries of Linux and Windows are within an ulp of the true sin and cos too, as fdlibm is (not measured here).
ULPS = 4


def _same_value(theirs: str, ours: str) -> bool:
    """Equal strings, or the same numbers to within `ULPS` units in the last place (no absolute tolerance, so a
    coordinate near zero is held as tightly as any other)."""
    if theirs == ours:
        return True
    a, b = re.split(r"[ ,]", theirs), re.split(r"[ ,]", ours)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        if not (_NUMBER.match(x) and _NUMBER.match(y)) or not _within_ulps(float(x), float(y)):
            return False
    return True


def _within_ulps(x: float, y: float) -> bool:
    return abs(x - y) <= ULPS * math.ulp(max(abs(x), abs(y)))


def test_chart_tolerance_is_ulps():
    """The tolerance is `ULPS` units in the last place of the numbers compared, however small they are: one more
    fails."""
    for x in (430.0, 116.37900316405418, 1.5e-20, 1e-300):
        near, far = x + ULPS * math.ulp(x), x + (ULPS + 1) * math.ulp(x)
        assert _same_value(repr(x), repr(near)) and _same_value(f"M {x!r} 2", f"M {near!r} 2") and _same_value(repr(-near), repr(-x))
        assert not _same_value(repr(x), repr(far)) and not _same_value(f"M {x!r} 2", f"M {far!r} 2")
    assert _same_value("430", "430.00000000000006") and not _same_value("430", "430.0000000000003")
    # Zero is held to the smallest subnormals (5e-324 is 1 ulp from it), where abs_tol=1e-9 let 1e-12 through.
    assert not _same_value("0", "1e-12") and not _same_value("0", "1e-300") and _same_value("0", "5e-324") and _same_value("0", "-0")
    assert not _same_value("M 1 2", "L 1 2") and not _same_value("1 2", "1 2 3")


# What the port adds to an element so that the chart draws alone as the widget's stylesheet draws it.
ADDED = {"xmlns", "fill", "stroke", "stroke-width", "r", "text-anchor", "dominant-baseline", "font-size", "font-family", "transform", "style"}


def _compare(theirs, ours, where: str = "") -> None:
    if isinstance(theirs, str):
        assert theirs == ours, where
        return
    assert ours["tag"] == theirs["tag"], where
    where = f"{where}/{theirs['tag']}"
    for key, value in theirs["attrs"].items():
        resolved = _VAR.sub(r"\1", value) if key in ("fill", "stroke") else value
        assert key in ours["attrs"] and _same_value(resolved, ours["attrs"][key]), (where, key, value, ours["attrs"].get(key))
    assert set(ours["attrs"]) - set(theirs["attrs"]) <= ADDED, (where, set(ours["attrs"]) - set(theirs["attrs"]))
    a, b = _joined(theirs["children"]), _joined(ours["children"])
    assert len(a) == len(b), (where, a, b)
    for x, y in zip(a, b):
        _compare(x, y, where)


def _without_watermark(node):
    """The widget's tree less its watermark ("Created with Microsoft Quantum Resource Estimator"), which the
    port does not draw: these charts are not made by Microsoft's estimator."""
    if isinstance(node, str):
        return node
    return dict(node, children=[_without_watermark(c) for c in node["children"] if isinstance(c, str) or c["attrs"].get("class") != "qs-scatterChart-watermark"])


def test_number_strings_are_javascripts():
    for text, (string, locale, rounded) in zip(WIDGET["numbers"], DRAWN["numbers"]):
        x = float(text)
        r = rep._js_round(x)
        assert (rep._js_string(x), rep._js_locale(x), "-0" if r == 0 and math.copysign(1, r) < 0 else rep._js_string(r)) == (string, locale, rounded), text


def test_color_map_and_ticks_are_the_widgets():
    assert [rep.color_map(n) for n in WIDGET["colors"]] == DRAWN["colors"]
    assert [rep.time_ticks(a, b) for a, b in WIDGET["time_ticks"]] == DRAWN["time_ticks"]
    assert [rep.integer_ticks(a, b) for a, b in WIDGET["integer_ticks"]] == DRAWN["integer_ticks"]
    # Where the widget never returns (a lower bound below 1: its first tick, 10^⌊log₁₀ min⌋ rounded, is 0, and
    # 0 × 10 stays 0), the port starts at 1.
    assert rep.integer_ticks(0.5, 2) == [{"value": 1, "label": "1"}] and rep.integer_ticks(0.5, 0.8) == []
    # NaN bounds give no ticks, as in the widget; an infinite upper bound (where the widget never returns either)
    # stops at the last power of ten below it.
    assert rep.integer_ticks(math.nan, 5) == [] and rep.time_ticks(math.nan, 5) == [] and len(rep.integer_ticks(1, math.inf)) == 309


def test_run_names_are_the_widgets():
    for keys, drawn in zip(WIDGET["scatter"], DRAWN["scatter"]):
        assert rep.run_names([_material(k) for k in keys]) == drawn["run_names"], keys


@pytest.mark.parametrize("index", range(len(WIDGET["space"])), ids=[f"{k.split('/')[-1]}@{i}" for k, i in WIDGET["space"]])
def test_space_chart_against_widget(index):
    """The SpaceChart (qsharp_widgets' donut of the physical qubits, the algorithm's against the T factories',
    and its side table) of each single result, a frontier's at several entries (one past its end reads the
    first, as the widget does): the widget's elements, attributes and text."""
    key, entry = WIDGET["space"][index]
    chart = az.EstimatorResult(_material(key)).space_chart(entry)
    (theirs,) = DRAWN["space"][index]
    _compare(theirs, _as_tree(ET.fromstring(chart._repr_html_())))
    _compare(theirs["children"][0], _as_tree(ET.fromstring(str(chart))))


@pytest.mark.parametrize("index", range(len(WIDGET["scatter"])))
def test_scatter_chart_against_widget(index):
    """The estimates overview's scatter chart (physical qubits against runtime, both logarithmic) of results
    together, a frontier's entries joined by a line: the widget's elements, attributes and text, and its colors."""
    keys = WIDGET["scatter"][index]
    drawn = DRAWN["scatter"][index]
    datas = [_material(k) for k in keys]
    result = az.EstimatorResult(datas[0] if len(datas) == 1 else datas)
    chart = result.scatter_chart()
    (div,) = drawn["tree"]
    _compare(_without_watermark(div["children"][0]), _as_tree(ET.fromstring(str(chart))))
    assert rep.color_map(len(keys)) == drawn["colors"]
    legend = ET.fromstring(chart._repr_html_())
    assert [td.text or "" for td in legend.iter("td") if td.get("data-colid") == "0"] == drawn["run_names"]
    rows = [[td.text for td in tr.iter("td") if td.get("data-colid")] for tr in legend.iter("tr")][1:]
    firsts = [fx.materialize(k, RESULTS) for k in keys]
    firsts = [d["frontierEntries"][0] if "frontierEntries" in d else d for d in firsts]
    assert [r[1:] for r in rows] == [[d["physicalCountsFormatted"]["physicalQubitsForTfactoriesPercentage"], f"{d['physicalCounts']['physicalQubits']:,}", d["physicalCountsFormatted"]["runtime"], f"{d['physicalCounts']['rqops']:,}"] for d in firsts]


def test_space_chart_svg_wellformed():
    """Every chart is an SVG document (and its HTML well-formed XML) that xml.etree reads, with the SVG namespace,
    a size, and only finite numbers or NaN where the widget draws NaN."""
    charts = [az.EstimatorResult(_material(key)).space_chart(entry) for key, entry in WIDGET["space"]]
    charts += [az.EstimatorResult([_material(k) for k in keys] if len(keys) > 1 else _material(keys[0])).scatter_chart() for keys in WIDGET["scatter"]]
    for chart in charts:
        root = ET.fromstring(str(chart))
        assert root.tag == "{http://www.w3.org/2000/svg}svg" and root.get("width") and root.get("viewBox") and chart._repr_svg_() == str(chart)
        ET.fromstring(chart._repr_html_())
        assert "Infinity" not in str(chart) and "undefined" not in str(chart)
    assert len(charts) == len(WIDGET["space"]) + len(WIDGET["scatter"])


def _strokes(chart) -> list:
    """Each drawn point's color, in order (a frontier's entries one after another)."""
    return [c.get("stroke") for c in ET.fromstring(str(chart)).iter("{http://www.w3.org/2000/svg}circle")]


def _table_names(chart) -> list:
    return [td.text or "" for td in ET.fromstring(chart._repr_html_()).iter("td") if td.get("data-colid") == "0"]


def test_charts_of_results():
    """A batch's space chart is its item's; a failed result raises its error; the scatter of a batch draws its
    succeeded items (the widget draws none of them when one has failed). As in the widget, colors and run names
    count the estimates, here every item of the batch, and each item keeps its color, the color map's too: a
    failed item's entry is dropped with it."""
    counts = _counts()
    batch = counts.estimate([{}, {"constraints": {"maxDuration": "1 ns"}}, {"estimateType": "frontier"}])
    frontier = len(batch[2]["frontierEntries"])
    with pytest.raises(ValueError, match=r"^Cannot draw the space chart of a batching job: draw an item's \(result\[i\]\.space_chart\(\)\)$"):
        batch.space_chart()
    with pytest.raises(az.EstimatorError):
        batch[1].space_chart()
    with pytest.raises(az.EstimatorError):
        batch[1].scatter_chart()
    assert str(batch[2].space_chart(1)) != str(batch[2].space_chart()) == str(batch[2].space_chart(0))
    three = rep.color_map(3)
    assert frontier > 1 and _strokes(batch.scatter_chart()) == [three[0]] + [three[2]] * frontier
    assert _strokes(batch.scatter_chart(colors=["red", "green", "blue"])) == ["red"] + ["blue"] * frontier
    assert _strokes(batch.scatter_chart(colors=three)) == _strokes(batch.scatter_chart())
    # A list of one per succeeded item does not count the failed one: ignored, with the widget's warning.
    for colors in (["red"], ["red", "blue"], ["red", "green", "blue", "black"]):
        with pytest.warns(UserWarning, match="^The number of colors does not match the number of estimates. Ignoring provided colors.$"):
            assert _strokes(batch.scatter_chart(colors=colors)) == _strokes(batch.scatter_chart())
    for names in (["a"], ["mine", "yours"]):
        with pytest.warns(UserWarning, match="^The number of runNames does not match the number of estimates. Ignoring provided runNames.$"):
            assert _table_names(batch.scatter_chart(run_names=names)) == rep.run_names([batch.data(0), batch.data(2)])
    assert _table_names(batch.scatter_chart(run_names=["mine", "failed", "yours"])) == ["mine", "yours"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # matching counts warn of nothing; an empty list is no list, as in the widget
        batch.scatter_chart(colors=["red", "green", "blue"], run_names=["mine", "failed", "yours"])
        assert _strokes(batch.scatter_chart(colors=[], run_names=[])) == _strokes(batch.scatter_chart())
    # A single result is one estimate.
    assert _strokes(batch[0].scatter_chart(colors=["red"])) == ["red"] and _table_names(batch[0].scatter_chart(run_names=["one"])) == ["one"]
    with pytest.warns(UserWarning, match="^The number of colors does not match"):
        batch[2].scatter_chart(colors=["red"] * frontier)
    # With no failed item, the colors are the color map's of the batch, as the widget draws them.
    whole = counts.estimate([{}, {"errorBudget": 0.01}, {"qubitParams": {"name": "qubit_maj_ns_e6"}, "qecScheme": {"name": "floquet_code"}}])
    assert _strokes(whole.scatter_chart()) == three and _strokes(whole.scatter_chart(colors=["a", "b", "c"])) == ["a", "b", "c"]
