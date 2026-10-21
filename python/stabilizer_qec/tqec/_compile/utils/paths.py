# Ported from tqec (tqec/tqec 0.2.0, src/tqec/utils/paths.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec: the directory is not created, platformdirs (which tqec requires) is
# used when it is installed, and the gallery paths are left out.
"""tqec's default detector database path, the default of every ``database_path`` this package
takes for one of tqec's signatures. This port keeps its detector database in memory: the path is
never read or written."""

from __future__ import annotations

import os
import posixpath
import sys
from pathlib import Path
from typing import Final


def _user_data_path(appname: str) -> Path:
    """``platformdirs.user_data_path(appname=appname)``, from which tqec takes its path:
    platformdirs' own when it is installed (tqec requires it, so with tqec installed this is
    tqec's path), else platformdirs' rule for the platform without its overrides:
    ``%LOCALAPPDATA%\\<appname>\\<appname>`` on Windows, else ``$XDG_DATA_HOME/<appname>`` when
    that is absolute, else ``~/Library/Application Support/<appname>`` on macOS and
    ``~/.local/share/<appname>`` elsewhere."""
    try:
        from platformdirs import user_data_path

        return user_data_path(appname=appname)
    except Exception:  # not installed, or no home directory: the path is never read, so the import goes on
        pass
    if sys.platform == "win32":
        return Path(os.path.normpath(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")), appname, appname)
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    if posixpath.isabs(xdg):
        return Path(xdg, appname)
    return Path(os.path.expanduser("~/Library/Application Support" if sys.platform == "darwin" else "~/.local/share"), appname)


USER_DATA_PATH: Final[Path] = _user_data_path("TQEC")


# Get the detector database path.
# This path can be provided through the TQEC_DETECTOR_DATABASE_PATH environment variable, or it
# defaults to a user-controlled directory.
def _get_database_path() -> Path:
    env_db_path = os.getenv("TQEC_DETECTOR_DATABASE_PATH")
    if env_db_path is not None:
        return Path(env_db_path)
    else:
        return USER_DATA_PATH / "detector_database.pkl"


DEFAULT_DETECTOR_DATABASE_PATH: Final[Path] = _get_database_path()
