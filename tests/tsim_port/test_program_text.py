# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5, test/unit/utils/test_program_text.py;
# Apache License 2.0), modified for stabilizer_qec: tsim is stabilizer_qec.tsim, and the tests of
# Tsim's Circuit class (test_circuit_scientific_notation, test_circuit_str_scientific_notation_roundtrip, test_circuit_eq_roundtrip_scientific_notation, test_circuit_parse_error_shows_snippet) left out.
import re

import pytest

from stabilizer_qec.tsim.utils.program_text import (
    shorthand_to_stim,
    stim_to_shorthand,
)


def test_shorthand_to_stim_t_and_t_dag():
    text = "T 0 1\nT_DAG 2"
    expected = "S[T] 0 1\nS_DAG[T] 2"
    assert shorthand_to_stim(text) == expected


def test_shorthand_to_stim_tagged_t_and_t_dag():
    text = "T[phase] 0\nT_DAG[phase] 1"
    expected = "S[T:phase] 0\nS_DAG[T:phase] 1"
    assert shorthand_to_stim(text) == expected


def test_shorthand_to_stim_tpp_and_tpp_dag():
    text = "TPP X0*Y1\nTPP_DAG Z0"
    expected = "SPP[T] X0*Y1\nSPP_DAG[T] Z0"
    assert shorthand_to_stim(text) == expected


def test_stim_to_shorthand_tpp_and_tpp_dag():
    text = "SPP[T] X0*Y1\nSPP_DAG[T] Z0"
    expected = "TPP X0*Y1\nTPP_DAG Z0"
    assert stim_to_shorthand(text) == expected


def test_shorthand_tpp_roundtrip():
    text = "TPP X0*Z1\nTPP_DAG !Y0*Y1"
    assert stim_to_shorthand(shorthand_to_stim(text)) == text


def test_shorthand_to_stim_rotations():
    text = "R_X(0.25) 0\nR_Y(-0.5) 1\nR_Z(0.3) 2"
    expected = (
        "I[R_X(theta=0.25*pi)] 0\nI[R_Y(theta=-0.5*pi)] 1\nI[R_Z(theta=0.3*pi)] 2"
    )
    assert shorthand_to_stim(text) == expected


def test_shorthand_to_stim_u3():
    text = "U3(0.3, 0.24, 0.49) 0"
    expected = "I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 0"
    assert shorthand_to_stim(text) == expected


def test_stim_to_shorthand_t_and_t_dag():
    text = "S[T] 0 1\nS_DAG[T] 2"
    expected = "T 0 1\nT_DAG 2"
    assert stim_to_shorthand(text) == expected


def test_stim_to_shorthand_tagged_t_and_t_dag():
    text = "S[T:phase] 0\nS_DAG[T:phase] 1"
    expected = "T[phase] 0\nT_DAG[phase] 1"
    assert stim_to_shorthand(text) == expected


def test_stim_to_shorthand_rotations_and_u3():
    text = (
        "I[R_X(theta=0.25*pi)] 0\n"
        "I[R_Y(theta=-0.5*pi)] 1\n"
        "I[R_Z(theta=0.3*pi)] 2\n"
        "I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 3"
    )
    expected = "R_X(0.25) 0\nR_Y(-0.5) 1\nR_Z(0.3) 2\nU3(0.3, 0.24, 0.49) 3"
    assert stim_to_shorthand(text) == expected


def test_shorthand_roundtrip():
    text = "T 0\nR_X(0.5) 1\nU3(0.1, 0.2, 0.3) 2"
    assert stim_to_shorthand(shorthand_to_stim(text)) == text


def test_shorthand_scientific_notation():
    result = shorthand_to_stim("R_Z(4e-4) 0")
    assert "I[R_Z(theta=0.0004*pi)]" in result


def test_shorthand_scientific_notation_u3():
    result = shorthand_to_stim("U3(1e-2, 2.5e1, 3e-3) 0")
    assert "I[U3(" in result


def test_stim_to_shorthand_rotation_scientific_notation():
    text = "I[R_Z(theta=1e-07*pi)] 0\nI[R_X(theta=-2.5e+02*pi)] 1"
    expected = "R_Z(1e-07) 0\nR_X(-2.5e+02) 1"
    assert stim_to_shorthand(text) == expected


def test_stim_to_shorthand_u3_scientific_notation():
    text = "I[U3(theta=1e-07*pi, phi=2e-07*pi, lambda=3e-07*pi)] 0"
    expected = "U3(1e-07, 2e-07, 3e-07) 0"
    assert stim_to_shorthand(text) == expected