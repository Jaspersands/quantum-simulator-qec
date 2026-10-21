# Ported from the QDK (microsoft/qdk 1.33.1; Copyright (c) Microsoft Corporation, MIT License; see LICENSE-qdk in
# this directory): qdk/estimator/_estimator.py's report (`HTMLWrapper`, `EstimatorResult`'s HTML tables and its
# `_plot`, `EstimatorResultDiagram`) and the charts of the QDK's notebook widgets as qsharp_widgets 1.33.1 bundles
# them (source/npm/qsharp/ux: spaceChart.tsx, scatterChart.tsx, estimatesOverview.tsx, resultsTable.tsx,
# colormap.ts and src/utils.ts). Modified for stabilizer_qec: written for Python 3.9; a result's tables are drawn
# when first asked for (the QDK draws them when the result is made); the charts are SVG text drawn here (the
# widgets draw them with preact in the browser), with JavaScript's arithmetic and number strings, the widgets'
# stylesheet written on the elements (its colors' fallbacks, the points' size, the axis title's rotation), and
# without the widget's watermark; the overview's table is static (no sorting, menus or selection).
"""The QDK's report of an Azure estimate, for `stabilizer_qec.estimate.azure.EstimatorResult`: the HTML tables
(``_repr_html_()``, ``summary``, a batch's side-by-side ``[:]``) beside their text, the diagrams' HTML
(``diagram.space``, ``diagram.time``), a batch's matplotlib plot (``plot()``), and the charts of the QDK's
notebook widgets as SVG: the space chart (the physical qubits of the algorithm against the T factories', a
donut with a side table) and the estimates overview's scatter chart (physical qubits against runtime, both
logarithmic, a frontier's entries joined by a line). Each is what the QDK and qsharp_widgets draw
(tests/estimate_port/qdk_reports.json.gz). With markdown and python-markdown-math installed the explanations
and assumptions are converted, as the QDK converts them.

>>> from stabilizer_qec.estimate.azure import LogicalCounts
>>> r = LogicalCounts(num_qubits=12, t_count=500).estimate()
>>> r.summary.splitlines()[0], r["physicalCountsFormatted"]["runtime"]
('Physical resource estimates', '3 millisecs')
>>> r.space_chart().count("<path")  # the donut: the algorithm's qubits and the T factories'
2
>>> frontier = LogicalCounts(num_qubits=12, t_count=500).estimate({"estimateType": "frontier"})
>>> chart = frontier.scatter_chart()
>>> chart.startswith('<svg xmlns="http://www.w3.org/2000/svg"'), chart.count("<circle"), len(frontier["frontierEntries"])
(True, 13, 13)
"""

from __future__ import annotations

import json
import math
import warnings
from decimal import ROUND_HALF_UP, Context, Decimal
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    # Both markdown and mdx_math (from python-markdown-math) must be present for the QDK's markdown rendering; if
    # either is missing, the explanations are written as they are.
    import markdown  # type: ignore[import-untyped]
    import mdx_math  # type: ignore[import-not-found]  # noqa: F401

    has_markdown = True
except ImportError:
    has_markdown = False


def _get_markdown() -> Optional[Any]:
    """Returns new Markdown object if markdown is available, None otherwise."""
    if has_markdown:
        return markdown.Markdown(extensions=["mdx_math"])
    return None


def _is_succeeded(data: Dict[str, Any]) -> bool:
    return "status" in data and data["status"] == "success"


class HTMLWrapper:
    """
    Simple HTML wrapper to expose _repr_html_ for Jupyter clients.
    """

    def __init__(self, content: str):
        self.content = content

    def _repr_html_(self):
        return self.content


class Report(str):
    """A table as text (the string itself) and as the QDK's HTML (`content`, and ``_repr_html_`` for notebooks),
    drawn when first asked for. The QDK draws a batch's table when the result or the slice is made, and prints
    "None of the jobs succeeded" there when no item in it succeeded; here that is printed when it is drawn."""

    def __new__(cls, text: str, draw: Callable[[], str]) -> "Report":
        report = super().__new__(cls, text)
        report._draw = draw
        report._html = None
        return report

    @property
    def content(self) -> str:
        if self._html is None:
            self._html = self._draw()
        return self._html

    def _repr_html_(self) -> str:
        return self.content

    def __reduce__(self):
        # Pickled (or copied) with its HTML, drawn now if it was not yet.
        return (_drawn_report, (str(self), self.content))


def _drawn_report(text: str, html: str) -> Report:
    return Report(text, lambda: html)


# ---------------------------------------------------------------------------------------------
# The QDK's HTML tables (qdk/estimator/_estimator.py: EstimatorResult._item_result_table,
# _item_result_summary_table, _batch_result_table and the batch overview of its __init__)
# ---------------------------------------------------------------------------------------------


def item_result_table(result) -> str:
    """The QDK's ``_repr_html_`` of one result: each report group (open where always visible) with its entries'
    label, value and explained description, then the assumptions (listed only with markdown). A path's key the
    result lacks at a level is read in its first frontier entry."""
    html = ""

    md = _get_markdown()
    for group in result["reportData"]["groups"]:
        html += f"""
                <details {"open" if group["alwaysVisible"] else ""}>
                    <summary style="display:list-item">
                        <strong>{group["title"]}</strong>
                    </summary>
                    <table>"""
        for entry in group["entries"]:
            val = result
            for key in entry["path"].split("/"):
                if key not in val and "frontierEntries" in val:
                    val = val["frontierEntries"][0]
                val = val[key]
            if md is not None:
                explanation = md.convert(entry["explanation"])
            else:
                explanation = entry["explanation"]
            html += f"""
                    <tr>
                        <td style="font-weight: bold; vertical-align: top; white-space: nowrap">{entry["label"]}</td>
                        <td style="vertical-align: top; white-space: nowrap">{val}</td>
                        <td style="text-align: left">
                            <strong>{entry["description"]}</strong>
                            <hr style="margin-top: 2px; margin-bottom: 0px; border-top: solid 1px black" />
                            {explanation}
                        </td>
                    </tr>
                """
        html += "</table></details>"

    html += '<details><summary style="display:list-item"><strong>Assumptions</strong></summary><ul>'
    if md is not None:
        for assumption in result["reportData"]["assumptions"]:
            html += f"<li>{md.convert(assumption)}</li>"
    html += "</ul></details>"

    return html


def item_result_summary_table(result) -> str:
    """The QDK's ``summary`` of one result: the groups' entries as label (its explanation a tooltip), value and
    description, under the tooltips' style, then the assumptions (listed only with markdown)."""
    html = """
            <style>
                .aqre-tooltip {
                    position: relative;
                    border-bottom: 1px dotted black;
                }

                .aqre-tooltip .aqre-tooltiptext {
                    font-weight: normal;
                    visibility: hidden;
                    width: 600px;
                    background-color: #e0e0e0;
                    color: black;
                    text-align: center;
                    border-radius: 6px;
                    padding: 5px 5px;
                    position: absolute;
                    z-index: 1;
                    top: 150%;
                    left: 50%;
                    margin-left: -200px;
                    border: solid 1px black;
                }

                .aqre-tooltip .aqre-tooltiptext::after {
                    content: "";
                    position: absolute;
                    bottom: 100%;
                    left: 50%;
                    margin-left: -5px;
                    border-width: 5px;
                    border-style: solid;
                    border-color: transparent transparent black transparent;
                }

                .aqre-tooltip:hover .aqre-tooltiptext {
                    visibility: visible;
                }
            </style>"""

    md = _get_markdown()
    for group in result["reportData"]["groups"]:
        html += f"""
                <details {"open" if group["alwaysVisible"] else ""}>
                    <summary style="display:list-item">
                        <strong>{group["title"]}</strong>
                    </summary>
                    <table>"""
        for entry in group["entries"]:
            val = result
            for key in entry["path"].split("/"):
                val = val[key]
            if md is not None:
                explanation = md.convert(entry["explanation"])
            else:
                explanation = entry["explanation"]
            html += f"""
                    <tr class="aqre-tooltip">
                        <td style="font-weight: bold"><span class="aqre-tooltiptext">{explanation}</span>{entry["label"]}</td>
                        <td>{val}</td>
                        <td style="text-align: left">{entry["description"]}</td>
                    </tr>
                """
        html += "</table></details>"

    html += "<details><summary style='display:list-item'><strong>Assumptions</strong></summary><ul>"
    if md is not None:
        for assumption in result["reportData"]["assumptions"]:
            html += f"<li>{md.convert(assumption)}</li>"
    html += "</ul></details>"

    return html


def batch_result_table(result, indices: Iterable[int]) -> str:
    """The QDK's side-by-side table of a batch's items `indices`: the groups of the first succeeded item, each
    label once (from every succeeded item's group at that position), and each item's value, its path's keys
    followed as they are ("N/A" where one is missing, so for a frontier item's counts, and for a failed item).
    With no succeeded item the QDK prints "None of the jobs succeeded" and draws nothing. The assumptions (with
    markdown) are item 0's: a failed item 0 raises KeyError, as in the QDK."""
    indices = list(indices)
    succeeded_item_indices = [i for i in indices if _is_succeeded(result[i])]
    if len(succeeded_item_indices) == 0:
        print("None of the jobs succeeded")
        return ""

    first_succeeded_item_index = succeeded_item_indices[0]

    html = ""

    md = _get_markdown()

    item_headers = "".join(f"<th>{i}</th>" for i in indices)

    for group_index, group in enumerate(result[first_succeeded_item_index]["reportData"]["groups"]):
        html += f"""
                <details {"open" if group["alwaysVisible"] else ""}>
                    <summary style="display:list-item">
                        <strong>{group["title"]}</strong>
                    </summary>
                    <table>
                        <thead><tr><th>Item</th>{item_headers}</tr></thead>"""

        visited_entries = set()

        for entry in [entry for index in succeeded_item_indices for entry in result[index]["reportData"]["groups"][group_index]["entries"]]:
            label = entry["label"]
            if label in visited_entries:
                continue
            visited_entries.add(label)

            html += f"""
                    <tr>
                        <td style="font-weight: bold; vertical-align: top; white-space: nowrap">{label}</td>
                """

            for index in indices:
                val = result[index]
                if index in succeeded_item_indices:
                    for key in entry["path"].split("/"):
                        if key in val:
                            val = val[key]
                        else:
                            val = "N/A"
                            break
                else:
                    val = "N/A"
                html += f"""
                            <td style="vertical-align: top; white-space: nowrap">{val}</td>
                    """

            html += """
                    </tr>
                """
        html += "</table></details>"

    html += '<details><summary style="display:list-item"><strong>Assumptions</strong></summary><ul>'
    if md is not None:
        for assumption in result[0]["reportData"]["assumptions"]:
            html += f"<li>{md.convert(assumption)}</li>"
    html += "</ul></details>"

    return html


def batch_overview(result) -> str:
    """The QDK's ``_repr_html_`` of a batch: its first `MAX_DEFAULT_ITEMS_IN_TABLE` (5) items side by side, after
    a note when there are more."""
    num_items = len(result)
    html = ""
    if num_items > result.MAX_DEFAULT_ITEMS_IN_TABLE:
        html += f"<p><b>Info:</b> <i>The overview table is cut off after {result.MAX_DEFAULT_ITEMS_IN_TABLE} items. If you want to see all items, suffix the result variable with <code>[:]</code></i></p>"
        num_items = result.MAX_DEFAULT_ITEMS_IN_TABLE
    html += batch_result_table(result, range(num_items))
    return html


# ---------------------------------------------------------------------------------------------
# The same tables as text
# ---------------------------------------------------------------------------------------------


def _lookup(result, path: str):
    """A single report's value: the path's keys followed into the result, into a frontier's first entry where a
    level lacks one (as the QDK's tables of one result read it); "N/A" where the path ends."""
    value = result
    for key in path.split("/"):
        if isinstance(value, dict) and key not in value and "frontierEntries" in value:
            value = value["frontierEntries"][0]
        if not isinstance(value, dict) or key not in value:
            return "N/A"
        value = value[key]
    return value


def _batch_value(item, path: str):
    """A side-by-side value, as the QDK's batch table reads it: the path's keys followed while each is there
    (no frontier entry is read), else "N/A"."""
    value = item
    for key in path.split("/"):
        if not isinstance(value, dict) or key not in value:
            return "N/A"
        value = value[key]
    return value


def text_table(header: Optional[List[str]], sections: List[Tuple[str, List[List[str]]]]) -> str:
    """Columns padded to their widest cell, each section under its title."""
    rows = [r for _, rs in sections for r in rs] + ([header] if header else [])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def line(r):
        return ("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths))).rstrip()

    out = [line(header)] if header else []
    for title, rs in sections:
        out.append(title)
        out.extend(line(r) for r in rs)
    return "\n".join(out)


def item_summary_text(result) -> str:
    """One result's summary as text: each group's entries as label, value and description."""
    groups = result["reportData"]["groups"]
    return text_table(None, [(g["title"], [[e["label"], str(_lookup(result, e["path"])), e["description"]] for e in g["entries"]]) for g in groups])


def batch_text(result, indices: Iterable[int]) -> str:
    """The side-by-side table as text: the rows of the first succeeded item's groups, each label once, with every
    item's value (as the QDK's HTML: "N/A" for a failed item or a path an item lacks at its level)."""
    items = {i: dict.__getitem__(result, i) for i in indices}
    succeeded = [i for i, item in items.items() if _is_succeeded(item)]
    if not succeeded:
        return "None of the jobs succeeded"
    sections = []
    for group_index, group in enumerate(items[succeeded[0]]["reportData"]["groups"]):
        rows, seen = [], set()
        for i in succeeded:
            for entry in items[i]["reportData"]["groups"][group_index]["entries"]:
                if entry["label"] in seen:
                    continue
                seen.add(entry["label"])
                rows.append([entry["label"]] + [str(_batch_value(item, entry["path"])) if j in succeeded else "N/A" for j, item in items.items()])
        sections.append((group["title"], rows))
    return text_table(["Item"] + [str(i) for i in items], sections)


# ---------------------------------------------------------------------------------------------
# The QDK's diagrams and plot (qdk/estimator/_estimator.py: EstimatorResultDiagram, EstimatorResult._plot)
# ---------------------------------------------------------------------------------------------


class EstimatorResultDiagram:
    """The QDK's space and time diagrams of one result: HTML that loads the QDK's visualization script (from the
    QDK's CDN) with the result (less its report) as compact JSON."""

    def __init__(self, data: Dict[str, Any]):
        data.pop("reportData")
        self.data_json = json.dumps(data).replace(" ", "")
        self.vis_lib = "https://cdn-aquavisualization-prod.azureedge.net/resource-estimation/index.js"
        self.space = HTMLWrapper(self._space_diagram())
        self.time = HTMLWrapper(self._time_diagram())

    def _space_diagram(self):
        html = f"""
            <script src={self.vis_lib}></script>
            <re-space-diagram data={self.data_json}></re-space-diagram>"""
        return html

    def _time_diagram(self):
        html = f"""
            <script src={self.vis_lib}></script>
            <re-time-diagram data={self.data_json}></re-time-diagram>"""
        return html


def plot(result, **kwargs: Any) -> None:
    """
    Plots all result items in a space time plot, where the x-axis shows
    total runtime, and the y-axis shows total number of physical qubits.
    Both axes are in log-scale.

    :param **kwargs: Common options:

        - ``labels`` (list): List of labels for the legend. Defaults to ``[]``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Missing optional 'matplotlib' dependency. To install run: " "pip install matplotlib")

    labels = kwargs.pop("labels", [])

    [xs, ys] = zip(
        *[
            (
                result.data(i)["physicalCounts"]["runtime"],
                result.data(i)["physicalCounts"]["physicalQubits"],
            )
            for i in range(len(result))
        ]
    )

    _ = plt.figure(figsize=(15, 8))

    plt.ylabel("Physical qubits")
    plt.xlabel("Runtime")
    plt.loglog()
    for i, (x, y) in enumerate(zip(xs, ys)):
        if isinstance(labels, list) and i < len(labels):
            label = labels[i]
        else:
            label = str(i)
        plt.scatter(x=[x], y=[y], label=label, marker="os+x"[i % 4])

    nsec = 1
    usec = 1e3 * nsec
    msec = 1e3 * usec
    sec = 1e3 * msec
    min = 60 * sec
    hour = 60 * min
    day = 24 * hour
    week = 7 * day
    month = 31 * day
    year = 365 * month
    decade = 10 * year
    century = 10 * decade

    time_units = [nsec, usec, msec, sec, min, hour, day, week, month, year, decade, century]
    time_labels = ["1 ns", "1 µs", "1 ms", "1 s", "1 min", "1 hour", "1 day", "1 week", "1 month", "1 year", "1 decade", "1 century"]

    cutoff = next((i for i, x in enumerate(time_units) if x > max(xs)), len(time_units) - 1) + 1

    plt.xticks(time_units[0:cutoff], time_labels[0:cutoff], rotation=90)
    plt.legend(loc="upper left")
    plt.show()


# ---------------------------------------------------------------------------------------------
# JavaScript's numbers, for the widgets' charts
# ---------------------------------------------------------------------------------------------


def _shortest(x: float) -> Tuple[str, int]:
    """The shortest digits that read back as a positive x (Python's repr finds them, as JavaScript does) and
    their exponent: x = digits × 10^exponent."""
    t = Decimal(repr(x)).normalize().as_tuple()
    return "".join(map(str, t.digits)), int(t.exponent)


def _js_string(x: float) -> str:
    """JavaScript's ``String(x)`` of a number: the shortest digits, positional from 1e-6 below 1e21 and
    exponential outside (``1e+21``, ``1.23e-18``); ``NaN``, ``Infinity``; a negative zero ``0``."""
    x = float(x)
    if x != x:
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0:
        return "0"
    digits, exponent = _shortest(abs(x))
    k, n = len(digits), exponent + len(digits)
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * -n + digits
    else:
        body = digits[0] + ("." + digits[1:] if k > 1 else "") + "e" + ("+" if n >= 1 else "-") + str(abs(n - 1))
    return ("-" if x < 0 else "") + body


_WIDE = Context(prec=1000)


def _js_locale(x: float) -> str:
    """JavaScript's ``x.toLocaleString()`` (``Intl.NumberFormat().format``) in en-US: the shortest digits rounded
    half up to three decimals, the thousands grouped with commas; ``NaN``, ``∞``, and ``-0`` for a negative
    number that rounds to zero."""
    x = float(x)
    if x != x:
        return "NaN"
    sign = "-" if math.copysign(1.0, x) < 0 else ""
    if math.isinf(x):
        return sign + "∞"
    digits, exponent = _shortest(abs(x)) if x != 0 else ("0", 0)
    fixed = f"{Decimal(digits).scaleb(exponent, _WIDE).quantize(Decimal('0.001'), ROUND_HALF_UP, _WIDE):f}"
    whole, _, fraction = fixed.partition(".")
    fraction = fraction.rstrip("0")
    return sign + f"{int(whole):,}" + ("." + fraction if fraction else "")


def _js_round(x: float) -> float:
    """JavaScript's ``Math.round``: the nearest integer, a half up (toward +∞); a negative zero kept."""
    x = float(x)
    if not math.isfinite(x):
        return x
    r = math.floor(x)
    if x - r >= 0.5:
        r += 1
    return -0.0 if r == 0 and math.copysign(1.0, x) < 0 else float(r)


def _js_value_string(value: Any) -> str:
    """JavaScript's ``String(value)`` of a JSON value (or `_UNDEFINED`)."""
    if value is _UNDEFINED:
        return "undefined"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _js_string(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ",".join("" if v is None or v is _UNDEFINED else _js_value_string(v) for v in value)
    return "[object Object]"


def _div(a: float, b: float) -> float:
    """JavaScript's division: by zero an infinity with the operands' signs, 0/0 NaN."""
    a, b = float(a), float(b)
    if b == 0.0:
        if a == 0.0 or a != a:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def _log(x: float) -> float:
    """``Math.log``: -∞ at 0, NaN below."""
    x = float(x)
    return math.nan if x != x or x < 0 else -math.inf if x == 0 else math.log(x)


def _log10(x: float) -> float:
    x = float(x)
    return math.nan if x != x or x < 0 else -math.inf if x == 0 else math.log10(x)


def _floor(x: float) -> float:
    return x if not math.isfinite(x) else float(math.floor(x))


def _ceil(x: float) -> float:
    return x if not math.isfinite(x) else float(math.ceil(x))


def _pow10(k: float) -> float:
    """``10 ** k``."""
    if k != k:
        return math.nan
    try:
        return 10.0**k
    except OverflowError:
        return math.inf


# ---------------------------------------------------------------------------------------------
# The widgets' axis ticks and colors (src/utils.ts: CreateTimeTicks, CreateIntegerTicks; colormap.ts)
# ---------------------------------------------------------------------------------------------

_TIME_TICKS = [(1.0, "1 nanosecond", "nanoseconds"), (1e3, "1 microsecond", "microseconds"), (1e6, "1 millisecond", "milliseconds"), (1e9, "1 second", "seconds"), (6e10, "1 minute", "minutes"), (36e11, "1 hour", "hours"), (864e11, "1 day", "days"), (6048e11, "1 week", "weeks"), (2592e12, "1 month", "months"), (31536e12, "1 year", "years"), (31536e13, "1 decade", "decades"), (31536e14, "1 century", "centuries")]


def time_ticks(lo: float, hi: float) -> List[Dict[str, Any]]:
    """The widgets' ticks of a runtime axis from `lo` to `hi` ns: the units (a nanosecond to a century) within
    it, else multiples of the unit below it (``30 seconds``), past a century of centuries."""
    ticks = _TIME_TICKS
    last = len(ticks) - 1
    if hi < lo or lo <= 0:
        return []
    left = 0
    while left < len(ticks) and ticks[left][0] <= lo:
        left += 1
    right = left
    if left > 0:
        left -= 1
    if right >= len(ticks):
        right = last
    while right < last and ticks[right][0] <= hi:
        right += 1
    out = [{"value": ticks[i][0], "label": ticks[i][1]} for i in range(left, right + 1) if ticks[i][0] >= lo and ticks[i][0] <= hi]
    if out:
        return out
    if left < last:
        coeff = _pow10(_floor(_log10(ticks[left + 1][0] / ticks[left][0])))
        while True:
            i = 1
            while True:
                val = ticks[left][0] * i * coeff
                if val >= lo and val <= hi:
                    out.append({"value": val, "label": _js_string(i * coeff) + " " + ticks[left][2]})
                i += 1
                if not ((i > 2 or len(out) < 1) and val < ticks[left + 1][0]):
                    break
            coeff /= 10
            if not (len(out) < 1 and coeff >= 1):
                break
    else:
        unit = last
        while True:
            coeff = _pow10(_floor(_log10(hi / ticks[unit][0])))
            while True:
                i = 1
                while True:
                    val = ticks[unit][0] * i * coeff
                    if val >= lo and val <= hi:
                        out.append({"value": val, "label": _js_string(i * coeff) + " " + ticks[unit][2]})
                    i += 1
                    if not ((i > 2 or len(out) < 1) and val < hi):
                        break
                coeff /= 10
                if not (len(out) < 1 and coeff >= 1):
                    break
            unit -= 1
            if not (len(out) < 1 and unit >= 0):
                break
    return out


def integer_ticks(lo: float, hi: float) -> List[Dict[str, Any]]:
    """The widgets' ticks of a count axis from `lo` to `hi`: the powers of ten within it, else the multiples of
    the largest power of ten below it that has some. Where the widget never returns (`lo` below 1: its first
    power, 10^⌊log₁₀ lo⌋ rounded, is 0, and ten times 0 is 0), the powers start at 1."""
    if hi < lo or lo <= 0:
        return []
    first = _js_round(_pow10(_floor(_log10(lo))))
    end = _js_round(_pow10(_ceil(_log10(hi))))
    if first == 0:
        first = 1.0
    out = []
    i = first
    while i <= end and not math.isinf(i):
        if i >= lo and i <= hi:
            out.append({"value": i, "label": _js_locale(i)})
        i *= 10
    if not out:
        step = first
        while step >= 1:
            i = _ceil(lo / step) * step
            while i <= hi:
                out.append({"value": i, "label": _js_locale(i)})
                i += step
            if out:
                break
            step /= 10
    return out


def _hue_to_rgb(p: float, q: float, t: float) -> float:
    if t < 0:
        t += 1
    if t > 1:
        t -= 1
    if t < 0.16666666666666666:
        return p + (q - p) * 6 * t
    if t < 0.5:
        return q
    if t < 0.6666666666666666:
        return p + (q - p) * (0.6666666666666666 - t) * 6
    return p


def _hsl_to_rgb(h: float, s: float, light: float) -> List[int]:
    if s == 0:
        r = g = b = light
    else:
        q = light * (1 + s) if light < 0.5 else light + s - light * s
        p = 2 * light - q
        r, g, b = _hue_to_rgb(p, q, h + 1 / 3), _hue_to_rgb(p, q, h), _hue_to_rgb(p, q, h - 1 / 3)
    return [int(_js_round(r * 255)), int(_js_round(g * 255)), int(_js_round(b * 255))]


def color_map(count: int) -> List[str]:
    """The widgets' colors of `count` estimates: hues by the golden ratio, saturations by √2 and lightnesses by
    e, each the fractional part of the index times it, as hex."""
    colors = []
    for i in range(count):
        h = math.fmod(i * 0.618033988749895, 1.0)
        s = (1 - math.fmod(i * 1.4142135623730951, 1.0)) * 0.5 + 0.5
        light = math.fmod(i * 2.718281828459045, 1.0) * 0.3 + 0.35
        r, g, b = _hsl_to_rgb(h, s, light)
        colors.append(f"#{(r << 16) | (g << 8) | b:06X}")
    return colors


# ---------------------------------------------------------------------------------------------
# The widgets' run names and estimates (estimatesOverview.tsx: createRunNames, reDataToRowScatter, reDataToRow;
# data.ts: CreateSingleEstimateResult)
# ---------------------------------------------------------------------------------------------

_UNDEFINED = object()


def _get(value: Any, *keys: str) -> Any:
    """A JavaScript property chain: undefined where a key is missing."""
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return _UNDEFINED
        value = value[key]
    return value


def _set_key(value: Any) -> Any:
    """A value as a JavaScript Set compares it (SameValueZero): strings and numbers by value, NaN equal to
    itself, objects by identity."""
    if value is _UNDEFINED or value is None:
        return (repr(value),)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        f = float(value)
        return ("nan",) if f != f else ("number", f + 0.0)
    if isinstance(value, str):
        return ("string", value)
    return ("object", id(value))


def run_names(estimates: Sequence[Dict[str, Any]]) -> List[str]:
    """The widgets' names of estimates shown together: one alone is named by its shared run name, qubit or
    scheme; several by the fields that differ among them (shared run name, qubit, scheme, error budget), with
    the index added when names repeat."""
    if len(estimates) == 1:
        for value in (_get(estimates[0], "jobParams", "sharedRunName"), _get(estimates[0], "jobParams", "qubitParams", "name"), _get(estimates[0], "jobParams", "qecScheme", "name")):
            if value is not None and value is not _UNDEFINED:
                return [value if isinstance(value, str) else _js_value_string(value)]
        return ["estimate"]
    fields: List[List[Any]] = [[] for _ in estimates]
    for read in (lambda d: _get(d, "jobParams", "sharedRunName"), lambda d: _get(d, "jobParams", "qubitParams", "name"), lambda d: _get(d, "jobParams", "qecScheme", "name"), lambda d: _js_value_string(_get(d, "jobParams", "errorBudget"))):
        values = [read(d) for d in estimates]
        if len({_set_key(v) for v in values}) > 1:
            for field, value in zip(fields, values):
                field.append(value)
    names = [", ".join("" if v is None or v is _UNDEFINED else _js_value_string(v) for v in field) for field in fields]
    if len(set(names)) != len(names):
        return [f"{name} ({index})" for index, name in enumerate(names)]
    return names


def _single_estimate(data: Dict[str, Any], index: Optional[int]) -> Dict[str, Any]:
    """The widgets' estimate of a result at `index`: a frontier's entry (its first where `index` is out of
    range) with the result's parameters and counts; any other result itself."""
    entries = data.get("frontierEntries")
    if index is None or entries is None or len(entries) == 0:
        return data
    if index < 0 or index >= len(entries):
        index = 0
    entry = entries[index]
    return {"status": data.get("status"), "jobParams": data.get("jobParams"), "physicalCounts": entry.get("physicalCounts"), "physicalCountsFormatted": entry.get("physicalCountsFormatted"), "logicalQubit": entry.get("logicalQubit"), "tfactory": entry.get("tfactory"), "errorBudget": entry.get("errorBudget"), "logicalCounts": data.get("logicalCounts")}


def _plot_item(entry: Dict[str, Any]) -> Tuple[float, float, str]:
    label = _js_value_string(entry["physicalCountsFormatted"]["runtime"]) + ", physical qubits: " + _js_value_string(entry["physicalCountsFormatted"]["physicalQubits"]) + ", code distance: " + _js_value_string(entry["logicalQubit"]["codeDistance"])
    return float(entry["physicalCounts"]["runtime"]), float(entry["physicalCounts"]["physicalQubits"]), label


def _scatter_items(data: Dict[str, Any]) -> List[Tuple[float, float, str]]:
    """A result's points: its estimate, or a frontier's entries."""
    entries = data.get("frontierEntries")
    if entries is None or len(entries) == 0:
        return [_plot_item(data)]
    return [_plot_item(e) for e in entries]


# ---------------------------------------------------------------------------------------------
# The charts as SVG (spaceChart.tsx: SpaceChart; scatterChart.tsx: ScatterChart; the overview's table)
# ---------------------------------------------------------------------------------------------

SVG_NS = "http://www.w3.org/2000/svg"
# The stylesheet's (qsharp-ux.css) colors outside VS Code and Jupyter's themes: the charts' fallbacks.
_ORANGE, _BLUE = "orange", "blue"
_MAIN_COLOR, _BORDER_COLOR, _ROW_BACKGROUND = "#202020", "#768f9c", "#f2f2f2"
_MAX_VALUE, _MIN_VALUE = 1.7976931348623157e308, 5e-324  # Number.MAX_VALUE, Number.MIN_VALUE (the least positive)


class Chart(str):
    """A chart: the SVG document (the string itself, and ``_repr_svg_``) and, for notebooks, its HTML
    (``_repr_html_``) with what the widget shows beside it."""

    def __new__(cls, svg: str, html: str) -> "Chart":
        chart = super().__new__(cls, svg)
        chart._html = html
        return chart

    def _repr_svg_(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        return self._html

    def __reduce__(self):
        return (Chart, (str(self), self._html))


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _el(tag: str, attrs: Sequence[Tuple[str, Any]], *children: str) -> str:
    """An element: attributes as the DOM gets them (a number as JavaScript writes it), children already
    written."""
    written = "".join(f' {k}="{_escape(v if isinstance(v, str) else _js_string(v))}"' for k, v in attrs)
    return f"<{tag}{written}>{''.join(children)}</{tag}>"


def _text(value: Any) -> str:
    return _escape(_js_value_string(value))


def _pie_segment(x: float, y: float, radius: float, start: float, end: float, inner: float) -> str:
    """The widget's donut segment from angle `start` to `end` (degrees, clockwise from the right) between the
    radii."""
    large_arc = "0" if end - start <= 180 else "1"
    start_x = x + radius * math.cos(math.pi * start / 180)
    start_y = y + radius * math.sin(math.pi * start / 180)
    end_x = x + radius * math.cos(math.pi * end / 180)
    end_y = y + radius * math.sin(math.pi * end / 180)
    inner_start_x = x + inner * math.cos(math.pi * start / 180)
    inner_start_y = y + inner * math.sin(math.pi * start / 180)
    inner_end_x = x + inner * math.cos(math.pi * end / 180)
    inner_end_y = y + inner * math.sin(math.pi * end / 180)
    js = _js_string
    return f"M {js(start_x)} {js(start_y)} A {js(radius)} {js(radius)} 0 {large_arc} 1 {js(end_x)} {js(end_y)} " + f"L {js(inner_end_x)} {js(inner_end_y)} A {js(inner)} {js(inner)} 0 {large_arc} 0 {js(inner_start_x)} {js(inner_start_y)} Z"


def space_chart(data: Dict[str, Any], index: Optional[int] = None) -> Chart:
    """qsharp_widgets' SpaceChart of a result: a donut of its physical qubits, the algorithm's (orange) against
    the T factories' (blue), the total in the middle; in the HTML, beside it, the table of the qubits (in total,
    per T factory and their copies, the algorithm's logical and physical ones, per logical qubit). A frontier's
    entry `index` (the first where it is None or out of range)."""
    est = _single_estimate(data, 0 if index is None else index)
    breakdown = est["physicalCounts"]["breakdown"]
    algorithm = float(breakdown["physicalQubitsForAlgorithm"])
    factories = float(breakdown["physicalQubitsForTfactories"])
    raw_angle = 360 * _div(algorithm, algorithm + factories)
    angle = 359.9 if raw_angle >= 360 else 0.1 if raw_angle <= 0 else raw_angle
    per_factory = _js_round(_div(factories, float(breakdown["numTfactories"])))
    svg_attrs = [("class", "qs-widget-spaceChart"), ("width", "400"), ("height", "400"), ("viewBox", "50 0 450 450"), ("id", "pieChart"), ("fill", _MAIN_COLOR), ("font-family", "sans-serif")]
    children = [
        _el("path", [("d", _pie_segment(250, 185, 180, 0, angle, 120)), ("fill", _ORANGE), ("stroke", "white")]),
        _el("path", [("d", _pie_segment(250, 185, 180, angle, 360, 120)), ("fill", _BLUE), ("stroke", "white")]),
        _el("text", [("x", "250"), ("y", "180"), ("text-anchor", "middle"), ("font-size", "16")], "Total physical qubits"),
        _el("text", [("x", "250"), ("y", "220"), ("text-anchor", "middle"), ("font-size", "32")], _text(est["physicalCountsFormatted"]["physicalQubits"])),
        _el("rect", [("x", "125"), ("y", "400"), ("width", "25"), ("height", "25"), ("fill", _ORANGE), ("stroke", "white"), ("stroke-width", "1")]),
        _el("text", [("x", "155"), ("y", "408"), ("text-anchor", "start"), ("font-size", "12")], "Algorithm qubits"),
        _el("text", [("x", "155"), ("y", "425"), ("text-anchor", "start"), ("font-size", "16")], _escape(_js_locale(algorithm))),
        _el("rect", [("x", "275"), ("y", "400"), ("width", "25"), ("height", "25"), ("fill", _BLUE), ("stroke", "white"), ("stroke-width", "1")]),
        _el("text", [("x", "305"), ("y", "408"), ("text-anchor", "start"), ("font-size", "12")], "T factory qubits"),
        _el("text", [("x", "305"), ("y", "425"), ("text-anchor", "start"), ("font-size", "16")], _escape(_js_locale(factories))),
    ]
    svg = _el("svg", [("xmlns", SVG_NS)] + svg_attrs, *children)
    header_style = f"font-size: 13px; background-color: {_ROW_BACKGROUND}; border-top: 0.5px gray solid; border-bottom: 0.5px gray solid; padding: 10px;"

    def header(title: str) -> str:
        return _el("div", [("class", "spaceReportHeader"), ("style", header_style)], title)

    def row(label: str, value: float) -> str:
        return _el("div", [("class", "spaceReportRow"), ("style", "display: flex; font-size: 12px; padding: 10px 24px;")], _el("div", [("class", "spaceDetailText"), ("style", "width: 200px;")], label), _el("div", [], _escape(_js_locale(value))))

    report = _el(
        "div",
        [("class", "spaceReport"), ("style", "display: flex; flex-direction: column;")],
        header("Physical resource estimates"),
        row("Total physical qubits", est["physicalCounts"]["physicalQubits"]),
        header("T factory parameters"),
        row("Physical T factory qubits", breakdown["physicalQubitsForTfactories"]),
        header("Resource estimation breakdown"),
        row("T factory copies", breakdown["numTfactories"]),
        row("Physical qubits per T factory", per_factory),
        row("Physical algorithmic qubits", algorithm),
        row("Logical algorithmic qubits", breakdown["algorithmicLogicalQubits"]),
        header("Logical qubit parameters"),
        row("Physical qubits", est["logicalQubit"]["physicalQubits"]),
    )
    html = _el("div", [("style", "display: flex; flex-wrap: wrap; margin-top: 8px;")], svg, report)
    return Chart(svg, html)


# The overview table's columns the widget shows at first (estimatesOverview.tsx's initialColumns), by their index.
_SHOWN = [0, 10, 13, 11, 12]
_COLUMNS = ["Run name", "T factory fraction", "Physical qubits", "Runtime", "rQOPS"]


def _row_cells(data: Dict[str, Any], name: str) -> List[str]:
    """The overview table's shown cells of an estimate (reDataToRow, as getCellStr writes them): the run name,
    the T factory fraction and the runtime in words, the physical qubits and rQOPS as locale strings."""
    est = _single_estimate(data, 0)

    def cell(value: Any) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return _js_locale(value)
        return "" if value is None or value == "" else _js_value_string(value)

    return [cell(name), cell(est["physicalCountsFormatted"]["physicalQubitsForTfactoriesPercentage"]), cell(est["physicalCounts"]["physicalQubits"]), cell(est["physicalCountsFormatted"]["runtime"]), cell(est["physicalCounts"]["rqops"])]


def scatter_chart(estimates: Sequence[Dict[str, Any]], colors: Optional[Sequence[str]] = None, names: Optional[Sequence[str]] = None, kept: Optional[Sequence[int]] = None) -> Chart:
    """The estimates overview's scatter chart of results drawn together: each result's physical qubits against
    its runtime (a frontier's entries joined by a line), both axes logarithmic with the widgets' ticks, a color
    each (the widgets' color map, or `colors` when there are as many as `estimates`). The HTML has the
    overview's table above it: each result's color and run name (`names` when there are as many), T factory
    fraction, physical qubits, runtime and rQOPS. `kept`, where given, are the indices of the estimates drawn (a
    batch's succeeded items: the widget cannot draw a failed one). As in the widget, `colors` and `names` are
    counted against all of `estimates`, and the colors, the color map's too, are given out over all of them;
    those drawn keep theirs. The run names made up where `names` are not given are made up from those drawn (a
    failed item has no parameters to name)."""
    datas = list(estimates)
    if colors is not None and len(colors) > 0 and len(colors) != len(datas):
        warnings.warn("The number of colors does not match the number of estimates. Ignoring provided colors.", stacklevel=3)
    if names is not None and len(names) > 0 and len(names) != len(datas):
        warnings.warn("The number of runNames does not match the number of estimates. Ignoring provided runNames.", stacklevel=3)
    colormap = list(colors) if colors is not None and len(colors) == len(datas) else color_map(len(datas))
    given_names = list(names) if names is not None and len(names) == len(datas) else None
    if kept is not None:
        datas, colormap = [datas[i] for i in kept], [colormap[i] for i in kept]
        given_names = None if given_names is None else [given_names[i] for i in kept]
    labels = run_names(datas) if given_names is None else given_names
    series = [(colormap[i], _scatter_items(d)) for i, d in enumerate(datas)]

    min_x, max_x, min_y, max_y = _MAX_VALUE, _MIN_VALUE, _MAX_VALUE, _MIN_VALUE  # as the widget starts them
    for _, items in series:
        for x, y, _ in items:
            min_x, max_x, min_y, max_y = _js_min(min_x, x), _js_max(max_x, x), _js_min(min_y, y), _js_max(max_y, y)
    range_x, range_y = (min_x / 2, max_x * 2), (min_y / 2, max_y * 2)
    x_ticks, y_ticks = time_ticks(*range_x), integer_ticks(*range_y)

    title_width, tick_caption_width, tick_length, axis_width = 20, 100, 5, 1
    left_margin = title_width + tick_caption_width + tick_length + axis_width
    right_margin = 160
    title_height, tick_caption_height = 20, 16
    y_margin = title_height + tick_caption_height + tick_length + axis_width
    svg_width, width_padding, svg_height, height_padding = 960, 50, 480, 10
    plot_width = svg_width - left_margin - right_margin
    plot_height = svg_height - y_margin
    view_box = f"{-left_margin} {-plot_height - height_padding} {svg_width + width_padding} {svg_height + height_padding}"
    text_padding, text_y_padding = 5, 4

    def position(value: float, rng: Tuple[float, float]) -> float:
        return _div(_log(value) - _log(rng[0]), _log(rng[1]) - _log(rng[0]))

    def to_x(value: float) -> float:
        return position(value, range_x) * plot_width

    def to_y(value: float) -> float:
        return -position(value, range_y) * plot_height

    axis = [("class", "qs-scatterChart-axis")]
    tick_line = [("class", "qs-scatterChart-tick-line"), ("stroke", _BORDER_COLOR)]
    children = [_el("line", axis + [("x1", "0"), ("y1", "0"), ("x2", plot_width), ("y2", "0"), ("stroke", _BORDER_COLOR)])]
    for tick in x_ticks:
        x = to_x(tick["value"])
        children.append(_el("line", [("y1", "1"), ("y2", tick_length), ("x1", x), ("x2", x)] + tick_line))
        children.append(_el("text", [("y", tick_length + tick_caption_height), ("x", x), ("class", "qs-scatterChart-x-tick-text"), ("text-anchor", "middle"), ("fill", _MAIN_COLOR)], _escape(tick["label"])))
    children.append(_el("line", axis + [("x1", "0"), ("y1", "0"), ("x2", "0"), ("y2", -svg_height), ("stroke", _BORDER_COLOR)]))
    for tick in y_ticks:
        y = to_y(tick["value"])
        children.append(_el("line", [("x1", "0"), ("x2", -tick_length), ("y1", y), ("y2", y)] + tick_line))
        children.append(_el("text", [("x", -tick_length - text_padding), ("y", y + text_y_padding), ("class", "qs-scatterChart-y-tick-text"), ("text-anchor", "end"), ("fill", _MAIN_COLOR)], _escape(tick["label"])))
    title = [("text-anchor", "middle"), ("font-size", "16"), ("fill", _MAIN_COLOR)]
    children.append(_el("text", [("x", plot_width / 2), ("y", y_margin), ("class", "qs-scatterChart-x-axisTitle")] + title, "Runtime (logarithmic)"))
    # The stylesheet writes this title vertically (writing-mode vertical-lr) and turns it about the origin
    # (rotate(180deg)), to the left of the axis, reading upward: the same as these two rotations.
    title_x, title_y = left_margin - title_height, plot_height / 2
    children.append(_el("text", [("x", title_x), ("y", title_y), ("class", "qs-scatterChart-y-axisTitle")] + title + [("dominant-baseline", "central"), ("transform", f"rotate(180) rotate(90 {_js_string(title_x)} {_js_string(title_y)})")], "Physical qubits (logarithmic)"))
    points = []
    for s, (color, items) in enumerate(series):
        centers = [(to_x(x), to_y(y)) for x, y, _ in items]
        for p, (x, y, label) in enumerate(items):
            points.append(_el("circle", [("data-index", f"[{s},{p}]"), ("data-label", label), ("class", "qs-scatterChart-point qs-scatterChart-hover"), ("cx", centers[p][0]), ("cy", centers[p][1]), ("stroke", color), ("r", "2"), ("stroke-width", "4")]))
        points.append(_el("polyline", [("points", " ".join(f"{_js_string(cx)},{_js_string(cy)}" for cx, cy in centers)), ("stroke", color), ("fill", "none")]))
    children.append(_el("g", [], *points))
    svg = _el("svg", [("xmlns", SVG_NS), ("style", "margin-top: 12px"), ("viewBox", view_box), ("width", svg_width), ("height", svg_height), ("font-family", "sans-serif")], *children)

    head = _el("tr", [], _el("th", []), *[_el("th", [("data-colid", str(i))], _escape(c)) for i, c in zip(_SHOWN, _COLUMNS)])
    rows = []
    for d, (color, _), name in zip(datas, series, labels):
        cells = _row_cells(d, name)
        icon = _el("svg", [("width", "16"), ("height", "16"), ("style", "position: relative;")], _el("path", [("stroke-width", "1.5"), ("stroke", color), ("stroke-linecap", "round"), ("d", "M4,5 h8 M4,8 h8 M4,11 h8")]))
        rows.append(_el("tr", [("data-rowid", cells[0])], _el("td", [], icon), *[_el("td", [("data-colid", str(i))], _escape(c)) for i, c in zip(_SHOWN, cells)]))
    table = _el("table", [("class", "qs-resultsTable-sortedTable")], _el("thead", [], head), _el("tbody", [], *rows))
    html = _el("div", [("class", "qs-estimatesOverview")], table, _el("div", [("style", "position: relative")], svg))
    return Chart(svg, html)


def _js_min(a: float, b: float) -> float:
    """``Math.min``: NaN if either is."""
    if a != a or b != b:
        return math.nan
    return b if b < a or (b == a == 0 and math.copysign(1.0, b) < 0) else a


def _js_max(a: float, b: float) -> float:
    if a != a or b != b:
        return math.nan
    return b if b > a or (b == a == 0 and math.copysign(1.0, a) < 0) else a
