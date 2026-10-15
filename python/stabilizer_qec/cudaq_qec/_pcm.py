# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec/lib/pcm_utils.cpp and
# python/bindings/py_decoder.cpp; Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates,
# Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec.
"""CUDA-Q QEC's parity-check-matrix tools: sorting columns by the rounds they touch, slicing
rounds, extending, shuffling, simplifying and random matrices. Random matrices and shuffles
draw from ``std::mt19937_64`` through libstdc++'s distributions, as CUDA-Q's builds (Linux only)
do, on every platform."""

from __future__ import annotations

import functools
import secrets
from typing import Any, List, Optional, Tuple

import numpy as np

from .. import _core

_MAX32 = 2**32 - 1


def _dense(H: Any, name: str) -> np.ndarray:
    a = np.asarray(H)
    if a.ndim != 2:
        raise RuntimeError(f"{name}: PCM must be a 2D tensor")
    return (a != 0).astype(np.uint8)


def _cols(a: np.ndarray) -> List[List[int]]:
    return [np.flatnonzero(a[:, j]).tolist() for j in range(a.shape[1])]


def _round_order(cols: List[List[int]], S: int, B: Optional[int]) -> List[int]:
    """``get_sorted_pcm_column_indices``: columns by first round, last round (when rounds are
    given), then by their rows taken from both ends inward; empty columns first."""
    use_rounds = B is not None or S > 0
    if B is None:
        round_of = (lambda r: r // S) if S > 0 else (lambda r: 0)
    else:
        round_of = lambda r: 0 if r < B else 1 + (r - B) // S  # noqa: E731

    def less(a: int, b: int) -> bool:
        av, bv = cols[a], cols[b]
        if not av and bv:
            return True
        if av and not bv:
            return False
        if not av and not bv:
            return a < b
        ah, at, bh, bt = 0, len(av) - 1, 0, len(bv) - 1
        if use_rounds:
            af, al, bf, bl = round_of(av[ah]), round_of(av[at]), round_of(bv[bh]), round_of(bv[bt])
            if af != bf:
                return af < bf
            if al != bl:
                return al < bl
        while True:
            if av[ah] != bv[bh]:
                return av[ah] < bv[bh]
            if ah == at and bh != bt:
                return True
            if ah != at and bh == bt:
                return False
            if ah == at and bh == bt:
                return a < b
            if av[at] != bv[bt]:
                return av[at] < bv[bt]
            ah += 1
            bh += 1
            if ah == at and bh != bt:
                return True
            if ah != at and bh == bt:
                return False
            if ah == at and bh == bt:
                return a < b
            at -= 1
            bt -= 1

    def cmp(a: int, b: int) -> int:
        return -1 if less(a, b) else (1 if less(b, a) else 0)

    return sorted(range(len(cols)), key=functools.cmp_to_key(cmp))


def get_sorted_pcm_column_indices(H: Any, num_syndromes_per_round: int = 0) -> List[int]:
    """The column order that sorts ``H`` (by round, with ``num_syndromes_per_round``)."""
    return _round_order(_cols(_dense(H, "get_sorted_pcm_column_indices")), int(num_syndromes_per_round), None)


def pcm_is_sorted(H: Any, num_syndromes_per_round: int = 0) -> bool:
    """Whether ``H``'s columns are already in sorted order."""
    order = get_sorted_pcm_column_indices(H, num_syndromes_per_round)
    return all(c == k for k, c in enumerate(order))


def _reorder_dense(a: np.ndarray, order: List[int], row_begin: int = 0, row_end: int = _MAX32) -> np.ndarray:
    if row_begin > row_end:
        raise ValueError("reorder_pcm_columns: row_begin must be less than or equal to row_end")
    m, n = a.shape
    if any(c >= n or c < 0 for c in order):
        raise ValueError("reorder_pcm_columns: column_order contains a column index that is greater than the number of columns in PCM")
    if m == 0:
        if row_begin != 0:
            raise ValueError("reorder_pcm_columns: row_begin is greater than the number of rows in PCM")
        return np.zeros((0, len(order)), dtype=np.uint8)
    if row_begin >= m:
        raise ValueError("reorder_pcm_columns: row_begin is greater than the number of rows in PCM")
    end = min(row_end, m - 1)
    return np.ascontiguousarray(a[row_begin : end + 1][:, list(order)] if order else np.zeros((end - row_begin + 1, 0), dtype=np.uint8))


def reorder_pcm_columns(H: Any, column_order: Any) -> Any:
    """``H`` with its columns taken in ``column_order`` (a scipy CSC matrix for a scipy input)."""
    order = [int(c) for c in column_order]
    if hasattr(H, "tocsr"):
        import scipy.sparse

        a = (H.toarray() != 0).astype(np.uint8)
        return scipy.sparse.csc_matrix(_reorder_dense(a, order) if a.shape[0] else np.zeros((0, len(order)), dtype=np.uint8), dtype=np.uint8)
    return _reorder_dense(_dense(H, "reorder_pcm_columns"), order)


def sort_pcm_columns(H: Any, num_syndromes_per_round: int = 0) -> np.ndarray:
    """``H`` with its columns sorted."""
    a = _dense(H, "sort_pcm_columns")
    return _reorder_dense(a, get_sorted_pcm_column_indices(a, num_syndromes_per_round))


def dump_pcm(H: Any) -> None:
    """Print ``H`` as rows of ``1`` and ``.``, then a blank line."""
    a = np.asarray(H)
    if a.ndim == 1:
        print("".join("1" if x > 0 else "." for x in a))
    else:
        for row in a:
            print("".join("1" if x > 0 else "." for x in row))
    print()


def _seed(seed: int) -> int:
    if seed < 0 or seed > _MAX32:
        raise TypeError("seed must fit an unsigned 32-bit integer")
    return int(seed) if seed else secrets.randbits(32)  # (std::random_device's 32 bits)


def generate_random_pcm(n_rounds: int, n_errs_per_round: int, n_syndromes_per_round: int, weight: int, seed: int = 0) -> np.ndarray:
    """A random round-structured check matrix: in each round, each error flips ``weight``
    detectors of its round (or, by a coin, of its round and the next), drawn from
    ``std::mt19937_64(seed)`` (a random seed for 0)."""
    cells = _core.cudaq_random_pcm(int(n_rounds), int(n_errs_per_round), int(n_syndromes_per_round), int(weight), _seed(int(seed)))
    pcm = np.zeros((n_rounds * n_syndromes_per_round, n_rounds * n_errs_per_round), dtype=np.uint8)
    for r, c in cells:
        pcm[r, c] = 1
    return pcm


def shuffle_pcm_columns(H: Any, seed: int = 0) -> Any:
    """``H`` with its columns shuffled by ``std::shuffle`` with ``std::mt19937_64(seed)``."""
    n = H.shape[1]
    return reorder_pcm_columns(H, _core.cpp_shuffled_order(int(n), _seed(int(seed))))


def _layout(S: int, B: int, num_rows: int) -> Tuple[bool, int, Any, Any]:
    boundary = B != 0 and B != S
    if not boundary:
        rounds = num_rows // S
    elif num_rows >= 2 * B:
        rounds = (num_rows - 2 * B) // S + 2
    else:
        rounds = 0

    def start(r: int) -> int:
        if not boundary:
            return r * S
        if r == 0:
            return 0
        if r >= rounds:
            return num_rows
        return B + (r - 1) * S

    def row_round(row: int) -> int:
        if not boundary:
            return row // S
        if row < B:
            return 0
        if row >= num_rows - B:
            return rounds - 1
        return 1 + (row - B) // S

    return boundary, rounds, start, row_round


def get_pcm_for_rounds(H: Any, num_syndromes_per_round: int, start_round: int, end_round: int, straddle_start_round: bool = False, straddle_end_round: bool = False, num_boundary_syndromes: int = 0) -> Tuple[np.ndarray, int, int]:
    """The rows of rounds ``start_round`` to ``end_round`` and the columns inside them (or
    straddling the first or last, if asked): (the sub-matrix, its first column, its last)."""
    a = _dense(H, "get_pcm_for_rounds")
    S, B = int(num_syndromes_per_round), int(num_boundary_syndromes)
    m = a.shape[0]
    if S == 0:
        raise ValueError("get_pcm_for_rounds: num_syndromes_per_round must be greater than 0")
    if B > S:
        raise ValueError("get_pcm_for_rounds: num_boundary_syndromes must be <= num_syndromes_per_round")
    boundary, rounds, start, row_round = _layout(S, B, m)
    if boundary and (m < 2 * B or (m - 2 * B) % S != 0):
        raise ValueError("get_pcm_for_rounds: number of PCM rows is inconsistent with the given num_syndromes_per_round and num_boundary_syndromes")
    if end_round >= rounds:
        raise ValueError("get_pcm_for_rounds: end_round is greater than the last round index")
    first_row = start(start_round)
    if first_row >= m:
        raise ValueError("get_pcm_for_rounds: first_row_to_keep is greater than the number of rows in PCM")
    last_row = start(end_round + 1) - 1
    keep: List[int] = []
    for c, col in enumerate(_cols(a)):
        if not col:
            continue
        fr, lr = row_round(col[0]), row_round(col[-1])
        if lr < start_round or fr > end_round:
            continue
        inside = fr >= start_round and lr <= end_round
        if inside or (straddle_start_round and fr <= start_round <= lr) or (straddle_end_round and fr <= end_round <= lr):
            keep.append(c)
    first, last = (keep[0], keep[-1]) if keep else (_MAX32, 0)
    return _reorder_dense(a, keep, first_row, last_row), first, last


def pcm_extend_to_n_rounds(H: Any, num_syndromes_per_round: int, n_rounds: int) -> Tuple[np.ndarray, List[int]]:
    """A sorted round-structured matrix extended to ``n_rounds`` by repeating its repeating
    round: (the matrix, the original column each new column copies)."""
    a = _dense(H, "pcm_extend_to_n_rounds")
    S = int(num_syndromes_per_round)
    orig = a.shape[0] // S
    if orig > n_rounds:
        raise ValueError("extend_pcm_to_n_rounds: original number of rounds must be less than or equal to n_rounds")
    if not pcm_is_sorted(a, S):
        raise ValueError("extend_pcm_to_n_rounds: input PCM is not sorted")
    num_orig_non_zero_cols = get_pcm_for_rounds(a, S, 0, orig - 1, True, True)[0].shape[1]
    prior, pf, pl = get_pcm_for_rounds(a, S, 0, 0, True, True)
    first_match, last_match = _MAX32, 0
    firsts, lasts = [pf], [pl]
    for w in range(1, orig):
        sub, f, l_ = get_pcm_for_rounds(a, S, w, w, True, True)
        firsts.append(f)
        lasts.append(l_)
        if sub.shape == prior.shape and np.array_equal(sub, prior):
            first_match, last_match = min(first_match, w), max(last_match, w)
        elif first_match != _MAX32:
            break
        prior = sub
    if first_match == _MAX32:
        raise ValueError("extend_pcm_to_n_rounds: PCM round analysis determined that consecutive rounds never matched each other, so we don't know how to insert more rounds")
    overlap = lasts[first_match - 1] + 1 - firsts[first_match]
    per_round = prior.shape[1] - overlap
    added = n_rounds - orig
    new = np.zeros((a.shape[0] + S * added, num_orig_non_zero_cols + per_round * added), dtype=np.uint8)
    cols: List[int] = []
    sub, _f, last_column = get_pcm_for_rounds(a, S, 0, last_match - 1, True, True)
    new[: sub.shape[0], : sub.shape[1]] = sub
    cols.extend(range(0, last_column + 1))
    rows_done, cols_done = sub.shape[0], sub.shape[1]
    sub, first_column, last_column = get_pcm_for_rounds(a, S, first_match, first_match, True, True)
    for _ in range(added):
        c0 = cols_done - overlap
        new[rows_done : rows_done + sub.shape[0], c0 : c0 + sub.shape[1]] = sub
        cols.extend(range(first_column + overlap, last_column + 1))
        rows_done += S
        cols_done += per_round
    sub, first_column, last_column = get_pcm_for_rounds(a, S, last_match, orig - 1, True, True)
    c0 = new.shape[1] - (a.shape[1] - first_column)
    new[rows_done : rows_done + sub.shape[0], c0 : c0 + sub.shape[1]] = sub
    cols.extend(range(first_column + overlap, last_column + 1))
    if len(cols) != new.shape[1]:
        raise RuntimeError(f"extend_pcm_to_n_rounds: column_list.size() [value: {len(cols)}] != new_pcm.shape()[1] [value: {new.shape[1]}]")
    return new, cols


def simplify_pcm(H: Any, weights: Any, num_syndromes_per_round: int) -> Tuple[np.ndarray, np.ndarray]:
    """``H`` sorted, with empty columns and columns of weight 0 dropped and equal adjacent
    columns merged (their probabilities composed as independent errors)."""
    a = _dense(H, "simplify_pcm")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    cols = _cols(a)
    new_cols: List[List[int]] = []
    new_w: List[float] = []
    for c in _round_order(cols, int(num_syndromes_per_round), None):
        if not cols[c] or w[c] == 0:
            continue
        if new_cols and new_cols[-1] == cols[c]:
            p, q = new_w[-1], w[c]
            new_w[-1] = p + q - 2.0 * p * q
        else:
            new_cols.append(cols[c])
            new_w.append(float(w[c]))
    out = np.zeros((a.shape[0], len(new_cols)), dtype=np.uint8)
    for j, col in enumerate(new_cols):
        out[col, j] = 1
    return out, np.array(new_w, dtype=np.float64)


def pcm_to_sparse_vec(pcm: Any) -> List[int]:
    """Each row's columns, each row ended by -1."""
    a = (pcm.toarray() if hasattr(pcm, "toarray") else np.asarray(pcm)) != 0
    out: List[int] = []
    for row in a:
        out.extend(int(c) for c in np.flatnonzero(row))
        out.append(-1)
    return out
