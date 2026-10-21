"""Qualtran 0.7.0's Dash app (``qualtran.surface_code.ui.app``; Copyright 2024 Google LLC, Apache-2.0),
recorded for tests/estimate_port/test_qualtran_ui_app.py: its title, its layout as Dash's
``to_plotly_json()`` gives it, and its callbacks as Dash lists them (``app._callback_list``, which Dash
serves at ``/_dash-dependencies``).

    python -I tools/qualtran_ui_fixture.py           # with qualtran 0.7.0 and dash 4.4.1
    python -I tools/qualtran_ui_fixture.py --check   # fail if the file differs

Writes tests/estimate_port/qualtran_layout.json. ``json.dumps(fixture["layout"], sort_keys=True)`` is the
layout's JSON with its keys sorted, as the test compares it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "estimate_port" / "qualtran_layout.json"


def build() -> str:
    import importlib.metadata

    from qualtran.surface_code import ui

    def encode(obj):
        return obj.to_plotly_json()

    layout = json.loads(json.dumps(ui.app.layout.to_plotly_json(), default=encode))
    record = {"qualtran": importlib.metadata.version("qualtran"), "dash": importlib.metadata.version("dash"), "title": ui.app.title, "layout": layout, "callbacks": ui.app._callback_list}
    return json.dumps(record, indent=1, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the recorded file differs")
    args = parser.parse_args()
    text = build()
    if args.check:
        if OUT.read_text() != text:
            print(f"{OUT} differs from Qualtran's app", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
