# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5, src/tsim/utils/diagram.py; Apache License 2.0;
# see LICENSE in the tsim directory), modified for stabilizer_qec: the SVG is this package's timeline
# (which labels the tagged gates itself, so Tsim's placeholder rewriting and its lxml and pyzx imports
# are not needed), and the diagram is this package's Diagram; the HTML wrappers are Tsim's.
"""SVG diagram wrappers for Tsim's ``Circuit.diagram``: the raw SVG as ``str()``, and a notebook
view of a given width and height, zoomable with Ctrl/Cmd + wheel."""

from __future__ import annotations

import re
import uuid
from typing import Any

from stabilizer_qec._circuit import Diagram as _Diagram


class Diagram(_Diagram):
    """A timeline or time-slice SVG, as Tsim's diagram: ``str()`` is the raw SVG (for an
    ``.svg`` file), and a notebook shows the HTML view of the given size."""

    __slots__ = ("_html",)

    def __init__(self, svg: str, html: str, type: str = "timeline-svg") -> None:
        super().__init__(svg, type, "svg")
        self._html = html

    def __str__(self) -> str:
        """Return the raw SVG string."""
        return self._text

    def _repr_html_(self) -> Any:
        """Return HTML representation for Jupyter notebook display."""
        return self._html

    def _repr_svg_(self) -> Any:
        return None


def _viewbox_size(svg: str) -> tuple[float, float] | None:
    """Return (width, height) parsed from an SVG viewBox attribute."""
    m = re.search(r'viewBox="[^"]*\s([\d.]+)\s+([\d.]+)"', svg)
    if m is None:
        return None
    w, h = map(float, m.groups())
    return w, h


def _width_from_viewbox(svg: str, height: float) -> float | None:
    """Compute width from an SVG viewBox while preserving aspect ratio."""
    size = _viewbox_size(svg)
    if size is None:
        return None
    w, h = size
    if h == 0:
        return None
    return float(height) / h * w


def wrap_svg(
    svg: str,
    *,
    width: float | None = None,
    height: float | None = None,
) -> str:
    """Wrap an SVG string in a container div.

    Args:
        svg: Raw SVG markup.
        width: Width of the container in pixels.
        height: Height of the container in pixels (unused, kept for API
            symmetry with :func:`wrap_svg_zoomable`).

    """
    if width is None:
        return f"""
        <div style="background: white">
        {svg}
        </div>
        """

    return f"""
    <div style="overflow-x: scroll; background: white; width: fit-content;">
    <div style="width: {width}px">
    {svg}
    </div>
    </div>
    """


def wrap_svg_zoomable(
    svg: str, *, width: float | None = None, height: float = 700
) -> str:
    """Wrap an SVG in a zoomable, scrollable container.

    The container has the given pixel dimensions.  When *width* is ``None``
    the container fills the available width.  Users can pan by scrolling
    and zoom with pinch-zoom (trackpad) or Ctrl/Cmd + wheel.  Zoom is
    anchored at the cursor position.

    Args:
        svg: Raw SVG markup.
        width: Pixel width of the container, or ``None`` for full width.
        height: Pixel height of the container.

    """
    size = _viewbox_size(svg)
    if size is None or size[1] == 0:
        nat_w, nat_h = 800.0, 200.0
    else:
        nat_w, nat_h = size

    # Stim SVGs only have a viewBox, no explicit width/height. Inside an
    # inline-block container they collapse, so force the SVG to render at its
    # natural pixel size.
    sized_svg = re.sub(
        r"<svg\b",
        f'<svg width="{nat_w}" height="{nat_h}"',
        svg,
        count=1,
    )

    # Initial scale fits the SVG into the container.
    scale_h = height / nat_h if nat_h > 0 else 1.0
    if width is not None and nat_w > 0:
        initial_scale = min(scale_h, width / nat_w)
    else:
        initial_scale = scale_h
    init_w = nat_w * initial_scale
    init_h = nat_h * initial_scale

    width_style = f"width:{width}px" if width is not None else "width:100%"
    uid = uuid.uuid4().hex[:12]
    return f"""
<div data-tsim-zoom="{uid}" style="{width_style}; height:{height}px; overflow:auto; background:white; border:1px solid #eee; position:relative;">
  <div style="display:inline-block; overflow:hidden; width:{init_w}px; height:{init_h}px;">
    <div style="transform-origin:0 0; display:block; width:{nat_w}px; height:{nat_h}px; transform:scale({initial_scale});">
      {sized_svg}
    </div>
  </div>
</div>
<script>
(function() {{
  var wrap = document.querySelector('[data-tsim-zoom="{uid}"]');
  if (!wrap || wrap.dataset.tsimZoomInit) return;
  wrap.dataset.tsimZoomInit = "1";
  var size = wrap.firstElementChild;
  var xform = size.firstElementChild;
  var natW = {nat_w};
  var natH = {nat_h};
  var scale = {initial_scale};
  var cw = wrap.clientWidth;
  if (cw > 0 && natW > 0) {{
    scale = cw / natW;
  }}
  function apply() {{
    xform.style.transform = 'scale(' + scale + ')';
    size.style.width = (natW * scale) + 'px';
    size.style.height = (natH * scale) + 'px';
  }}
  apply();
  wrap.addEventListener('wheel', function(e) {{
    if (e.ctrlKey || e.metaKey) {{
      e.preventDefault();
      var rect = wrap.getBoundingClientRect();
      var mx = e.clientX - rect.left + wrap.scrollLeft;
      var my = e.clientY - rect.top + wrap.scrollTop;
      var factor = Math.exp(-e.deltaY * 0.01);
      var newScale = Math.min(Math.max(0.02, scale * factor), 40);
      var ratio = newScale / scale;
      scale = newScale;
      apply();
      wrap.scrollLeft = mx * ratio - (e.clientX - rect.left);
      wrap.scrollTop = my * ratio - (e.clientY - rect.top);
    }}
  }}, {{ passive: false }});
}})();
</script>
"""


def render_svg(
    svg: str,
    type: str,
    *,
    width: float | None = None,
    height: float | None = None,
    zoomable: bool = True,
) -> Diagram:
    """Tsim's ``render_svg`` on an SVG already drawn: the missing dimension from the SVG's
    aspect ratio, and the HTML view (zoomable or not)."""
    # Compute the missing dimension from the SVG viewBox aspect ratio.
    if width is None and height is not None:
        width = _width_from_viewbox(svg, height)
    elif height is None and width is not None:
        size = _viewbox_size(svg)
        if size is not None and size[0] != 0:
            height = width / size[0] * size[1]

    if zoomable:
        html = wrap_svg_zoomable(
            svg, width=width, height=height if height is not None else 700
        )
    else:
        html = wrap_svg(svg, width=width, height=height)
    return Diagram(svg, html, type)
