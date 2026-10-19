# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
# Ported from MQT QECC 2.0.0 (munich-quantum-toolkit/qecc; Chair for Design Automation, TUM; MIT; see
# THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, qecsim and bposd are this
# package's, and mqt.qecc is stabilizer_qec.mqt_qecc.

"""Command line interface for the CC decoder."""

from __future__ import annotations

import argparse

from . import decoder
from .comparison import tn_decoder


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Run the CC decoder as cli.

    (stabilizer_qec: ``argv`` (default ``sys.argv[1:]``) and ``prog`` (default from ``sys.argv[0]``)
    let ``stabilizer-qec mqt.qecc.cc-decoder`` and ``python -m stabilizer_qec.mqt_qecc.cc_decoder``
    run it, as MQT QECC's ``mqt.qecc.cc-decoder`` script does.)
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("distance", type=int, help="the distance of the code")
    parser.add_argument("error_rate", type=float, help="the error rate")
    parser.add_argument(
        "--type",
        type=str,
        default="hexagon",
        help="type of the code lattice (hexagon or square_octagon). Default: hexagon",
    )
    parser.add_argument(
        "--nr_sims",
        type=int,
        default=10000,
        help="the number of simulations to run. Default: 10000",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="the directory to save the results to. Default: ./results",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="maxsat",
        help="the decoder to use (maxsat or tn). Default: maxsat",
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="z3",
        help="maxsat solver to use (path to a executable). Default: z3",
    )

    args = parser.parse_args(argv)

    if args.decoder == "maxsat":
        decoder.run(
            args.type,
            args.distance,
            args.error_rate,
            args.nr_sims,
            args.results_dir,
            args.solver,
        )
    elif args.decoder == "tn":
        tn_decoder.run(args.distance, args.error_rate, args.nr_sims, args.results_dir)
    else:
        msg = f"Unknown decoder {args.decoder}. Please choose either maxsat or tn."
        raise ValueError(msg)
