"""Build the API reference and guide, served at qcompiler.jaspersands.com/api/, from the
installed package's docstrings (pdoc).

    python tools/api_docs.py     # -> api/

Run it with the package built from this checkout installed, and pdoc (pip install pdoc).
The guide is the package's own docstring; tests/test_docs.py runs its examples.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "api"

ICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E"
    "%3Crect width='16' height='16' rx='3' fill='%231c1d1f'/%3E"
    "%3Ccircle cx='5' cy='5' r='1.6' fill='%23f4f2ec'/%3E%3Ccircle cx='11' cy='5' r='1.6' fill='%23f4f2ec'/%3E"
    "%3Ccircle cx='5' cy='11' r='1.6' fill='%23f4f2ec'/%3E%3Ccircle cx='11' cy='11' r='1.6' fill='%23c0392b'/%3E%3C/svg%3E"
)


def scrub(out: Path, package_dir: Path) -> None:
    """Replace the machine's own paths in the rendered pages: a default value computed at import
    time (a bundled file, a per-user cache) is shown with the directory it resolved to."""
    pairs = [(str(package_dir.parent), "site-packages"), (str(package_dir.resolve().parent), "site-packages"),
             (str(Path.home()), "~")]
    for page in out.rglob("*"):
        if page.suffix in (".html", ".js"):
            text = page.read_text(encoding="utf-8")
            new = text
            for old, mark in pairs:
                new = new.replace(old, mark)
            if new != text:
                page.write_text(new, encoding="utf-8")


def main() -> None:
    import stabilizer_qec

    if OUT.exists():
        shutil.rmtree(OUT)
    subprocess.run(
        [
            # The sinter adapter is named too: it stays out of __all__, which a star import
            # would otherwise make require sinter.
            sys.executable, "-m", "pdoc", "stabilizer_qec", "stabilizer_qec.sinter",
            # 2.0's subpackages written here (the whole ports of other tools' packages keep
            # those tools' documentation).
            "stabilizer_qec.estimate", "stabilizer_qec.tqec", "stabilizer_qec.loom", "stabilizer_qec.cudaq_qec",
            "stabilizer_qec.realtime", "stabilizer_qec.tsim", "stabilizer_qec.analysis",
            "--output-directory", str(OUT),
            "--docformat", "markdown",
            "--favicon", ICON,
            "--logo-link", "https://qcompiler.jaspersands.com",
            "--footer-text", f"stabilizer-qec {stabilizer_qec.__version__}",
            "--no-show-source",
            "--template-directory", str(ROOT / "tools" / "api_theme"),
        ],
        check=True,
    )
    scrub(OUT, Path(stabilizer_qec.__file__).parent)
    print(f"wrote {OUT.relative_to(ROOT)}/ for stabilizer-qec {stabilizer_qec.__version__}")


if __name__ == "__main__":
    main()
