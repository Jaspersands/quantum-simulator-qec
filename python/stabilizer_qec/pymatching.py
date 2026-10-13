"""PyMatching's module under this package: ``import stabilizer_qec.pymatching as pymatching``
gives ``Matching`` (this package's, which is PyMatching 2.4's API), ``set_seed``,
``randomize`` and ``rand_float``; ``python -m stabilizer_qec.pymatching`` is its command line
(``predict``, ``count_mistakes``)."""

from __future__ import annotations

from . import _matching as matching
from ._matching import Matching, rand_float, randomize, set_seed

__all__ = ["Matching", "cli", "matching", "rand_float", "randomize", "set_seed"]


def cli(*, command_line_args: list) -> int:
    """PyMatching's ``cli``: ``predict`` or ``count_mistakes`` with PyMatching's flags."""
    from ._cli import main

    args = list(command_line_args)
    if not args or args[0] not in ("predict", "count_mistakes"):
        raise ValueError(
            "Unrecognized command. Available commands are:\n"
            "    pymatching predict --dem file [--in file] [--out file] [--in_format 01|b8|...] [--out_format 01|b8|...] [--in_includes_appended_observables]\n"
            "    pymatching count_mistakes --dem file [--in file] [--out file] [--in_format 01|b8|...] [--out_format 01|B8|...] [--in_includes_appended_observables] [--obs_in] [--obs_in_format]"
        )
    return main(command_line_args=args)


if __name__ == "__main__":
    import sys

    sys.exit(cli(command_line_args=sys.argv[1:]))

