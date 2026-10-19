"""The surgery engine's program writer (``src/surgery.rs``: ``Layout``, ``Writer`` and
``Program::circuit``) in Python, for the programs the engine refuses because two patches share
a tile at different times (a block graph whose tile is used again after its first cube's patch
is read out: two CNOTs in a row reuse the ancilla's tile).

It is the engine's code line for line, with two changes that only matter when a tile is shared:
each patch keeps its own final readout (the engine keeps one per data qubit, which a later patch
on the tile would overwrite), and a patch prepared on a tile forgets the checks the tile's last
patch measured there, so its first round starts afresh (the engine compares a check with the
round before, which on a tile measured out and prepared again at once would compare two patches).
On every program the engine takes the text is the engine's, instruction for instruction; the
tests hold the two to that."""

from __future__ import annotations

from typing import Optional, Sequence

_X_ORDER = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_Z_ORDER = ((-1, -1), (1, -1), (-1, 1), (1, 1))

_MAX_ROUNDS = 10_000
_MAX_VOLUME = 1 << 24
_MAX_QUBITS = 1 << 24


class _Check:
    __slots__ = ("pos", "x_type", "anc", "at_step", "_support")

    def __init__(self, pos: tuple[int, int], x_type: bool, anc: int, at_step: list[Optional[int]]) -> None:
        self.pos = pos
        self.x_type = x_type
        self.anc = anc
        self.at_step = at_step
        self._support = sorted(q for q in at_step if q is not None)

    def support(self) -> list[int]:
        return self._support


class _Box:
    __slots__ = ("x0", "x1", "y0", "y1")

    def __init__(self, x0: int, x1: int, y0: int, y1: int) -> None:
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def union(self, other: "_Box") -> "_Box":
        return _Box(min(self.x0, other.x0), max(self.x1, other.x1), min(self.y0, other.y0), max(self.y1, other.y1))


class _Layout:
    """A grid of ``cols`` × ``rows`` tiles, 2d + 2 apart, data first, row by row, then every
    check position, row by row."""

    def __init__(self, d: int, cols: int, rows: int) -> None:
        self.d = d
        w, h = cols * (2 * d + 2) - 2, rows * (2 * d + 2) - 2
        self.positions: list[tuple[int, int]] = []
        self.index: dict[tuple[int, int], int] = {}
        for y in range(1, h, 2):
            for x in range(1, w, 2):
                self.index[(x, y)] = len(self.positions)
                self.positions.append((x, y))
        for y in range(0, h + 1, 2):
            for x in range(0, w + 1, 2):
                self.index[(x, y)] = len(self.positions)
                self.positions.append((x, y))

    def tile(self, tile: tuple[int, int]) -> _Box:
        i, j = tile
        x0, y0 = 1 + i * (2 * self.d + 2), 1 + j * (2 * self.d + 2)
        return _Box(x0, x0 + 2 * self.d - 2, y0, y0 + 2 * self.d - 2)

    def data(self, b: _Box) -> list[int]:
        return sorted(self.index[(x, y)] for y in range(b.y0, b.y1 + 1, 2) for x in range(b.x0, b.x1 + 1, 2))

    def code(self, b: _Box) -> list[_Check]:
        """The rotated code on box ``b``: X-type boundaries left and right, Z-type top and bottom."""
        checks = []
        for y in range(b.y0 - 1, b.y1 + 2, 2):
            for x in range(b.x0 - 1, b.x1 + 2, 2):
                x_type = ((x + y) // 2) % 2 == 1
                allowed = (b.y0 < y < b.y1) if x_type else (b.x0 < x < b.x1)
                if not allowed:
                    continue
                order = _X_ORDER if x_type else _Z_ORDER
                at_step: list[Optional[int]] = [None] * 4
                for k, (dx, dy) in enumerate(order):
                    if b.x0 <= x + dx <= b.x1 and b.y0 <= y + dy <= b.y1:
                        at_step[k] = self.index[(x + dx, y + dy)]
                if sum(q is not None for q in at_step) >= 2:
                    checks.append(_Check((x, y), x_type, self.index[(x, y)], at_step))
        return checks


def _join(qubits) -> str:
    return " ".join(str(q) for q in qubits)


def _pairs(pairs) -> str:
    return " ".join(f"{a} {b}" for a, b in pairs)


class _Writer:
    def __init__(self, p: float) -> None:
        self.lines: list[str] = []
        self.m = 0
        self.p = p
        self.pr = repr(float(p))
        # pos -> (rec, support, round)
        self.last: dict[tuple[int, int], tuple[int, list[int], int]] = {}
        self.fresh: dict[int, str] = {}
        self.measured: dict[int, tuple[int, str]] = {}
        self.round = 0

    def lookback(self, rec: int) -> int:
        return self.m - rec

    def noise(self, line: str) -> None:
        if self.p > 0.0:
            self.lines.append(line)

    def depol1(self, qubits: list[int]) -> None:
        if qubits:
            self.noise(f"DEPOLARIZE1({self.pr}) {_join(qubits)}")

    def _flip(self, basis: str, qubits: list[int]) -> None:
        self.noise(f"{'X_ERROR' if basis == 'z' else 'Z_ERROR'}({self.pr}) {_join(qubits)}")

    def prepare(self, basis: str, qubits: list[int]) -> None:
        if not qubits:
            return
        self.lines.append(f"{'RX' if basis == 'x' else 'R'} {_join(qubits)}")
        self._flip(basis, qubits)
        for q in qubits:
            self.fresh[q] = basis

    def measure(self, basis: str, qubits: list[int]) -> list[int]:
        """Measure data qubits; their records in the order given."""
        self._flip(basis, qubits)
        self.lines.append(f"{'MX' if basis == 'x' else 'M'} {_join(qubits)}")
        recs = list(range(self.m, self.m + len(qubits)))
        self.m += len(qubits)
        for q, r in zip(qubits, recs):
            self.measured[q] = (r, basis)
        return recs

    def run_round(self, checks: list[_Check], data: list[int]) -> list[int]:
        """One round of syndrome extraction on ``checks``, the SD6 way; each check's record."""
        self.round += 1
        anc = [c.anc for c in checks]
        had = [c.anc for c in checks if c.x_type]
        active = data + anc

        def idle(busy) -> list[int]:
            busy = set(busy)
            return [q for q in active if q not in busy]

        self.lines.append("TICK")
        self.lines.append(f"R {_join(anc)}")
        self.noise(f"X_ERROR({self.pr}) {_join(anc)}")
        self.depol1([q for q in data if q not in self.fresh])
        for pass_ in range(2):
            self.lines.append("TICK")
            if had:
                self.lines.append(f"H {_join(had)}")
                self.depol1(had)
            self.depol1(idle(had))
            if pass_ == 1:
                break
            for step in range(4):
                self.lines.append("TICK")
                pairs = []
                for ch in checks:
                    q = ch.at_step[step]
                    if q is not None:
                        pairs.append((ch.anc, q) if ch.x_type else (q, ch.anc))
                if pairs:
                    self.lines.append(f"CX {_pairs(pairs)}")
                    self.noise(f"DEPOLARIZE2({self.pr}) {_pairs(pairs)}")
                self.depol1(idle(q for pair in pairs for q in pair))
        self.lines.append("TICK")
        self.noise(f"X_ERROR({self.pr}) {_join(anc)}")
        self.lines.append(f"M {_join(anc)}")
        recs = list(range(self.m, self.m + len(anc)))
        self.m += len(anc)
        self.depol1(list(data))

        for ch, rec in zip(checks, recs):
            basis = "x" if ch.x_type else "z"
            support = ch.support()
            targets = [rec]
            prev = self.last.get(ch.pos)
            if prev is not None and prev[2] + 1 != self.round:
                prev = None
            if prev is not None:
                prev_rec, prev_support, _ = prev
                targets.append(prev_rec)

                def read_since(q: int) -> Optional[int]:
                    got = self.measured.get(q)
                    if got is not None and got[1] == basis and got[0] > prev_rec:
                        return got[0]
                    return None

                ok = True
                for q in support:
                    if q not in self.fresh:
                        continue
                    ok &= self.fresh[q] == basis
                    if q in prev_support:
                        r = read_since(q)
                        if r is None:
                            ok = False
                        else:
                            targets.append(r)
                ok &= all(q in self.fresh for q in support if q not in prev_support)
                for q in prev_support:
                    if q in support:
                        continue
                    r = read_since(q)
                    if r is None:
                        ok = False
                    else:
                        targets.append(r)
                deterministic = ok
            else:
                deterministic = all(self.fresh.get(q) == basis for q in support)
            if deterministic:
                self.lines.append(f"DETECTOR({ch.pos[0]}, {ch.pos[1]}, {self.round}) " + " ".join(f"rec[-{self.lookback(r)}]" for r in targets))
            self.last[ch.pos] = (rec, support, self.round)
        self.fresh.clear()
        return recs


def _qubits_for(d: int, tiles: int) -> int:
    return tiles * 2 * (d + 1) ** 2


def _check_volume(qubits: int, rounds: int) -> None:
    if qubits * rounds > _MAX_VOLUME:
        raise ValueError(f"about {qubits} qubits over {rounds} rounds: more than the {_MAX_VOLUME} qubit-rounds a program is written out for")


def _step_rounds(steps: Sequence[tuple]) -> int:
    return sum(s[1] if s[0] == "rounds" else 1 for s in steps)


def program_text(d: int, p: float, tiles: Sequence[tuple[int, int]], steps: Sequence[tuple], observables: Sequence[Sequence[tuple]]) -> str:
    """The program as Stim text, as ``_core.surgery_program`` writes it; patches may share a tile
    while their lifetimes do not overlap."""
    tiles = [tuple(t) for t in tiles]
    n = len(tiles)
    rounds = _step_rounds(steps)
    # The binding's size check (``check_size``), then the program's own.
    qubits = _qubits_for(d, n)
    if qubits > _MAX_QUBITS:
        raise ValueError(f"{n} patches at distance {d} need more qubits than Stim can index")
    if rounds > _MAX_ROUNDS:
        raise ValueError(f"{rounds} rounds and steps: at most {_MAX_ROUNDS} are written out")
    _check_volume(qubits, rounds + 2 * d)
    if d < 3 or d % 2 == 0:
        raise ValueError(f"lattice surgery needs an odd distance of at least 3, not {d}")
    if not tiles or any(i < 0 or j < 0 for i, j in tiles):
        raise ValueError("a program needs patches on tiles of non-negative column and row")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"the noise strength {p} is outside [0, 1]")
    cols = max(t[0] for t in tiles) + 1
    rows = max(t[1] for t in tiles) + 1
    grid_qubits = _qubits_for(d, cols * rows)
    if grid_qubits > _MAX_QUBITS:
        raise ValueError(f"the program's grid of tiles at distance {d} needs more qubits than Stim can index")
    if rounds > _MAX_ROUNDS:
        raise ValueError(f"the program runs {rounds} rounds and steps, more than the {_MAX_ROUNDS} written out")
    _check_volume(grid_qubits, rounds)

    layout = _Layout(d, cols, rows)
    boxes = [layout.tile(t) for t in tiles]
    codes = [layout.code(b) for b in boxes]
    datas = [layout.data(b) for b in boxes]
    data_sets = [set(x) for x in datas]
    on_tile: dict[tuple[int, int], list[int]] = {}
    for q, t in enumerate(tiles):
        on_tile.setdefault(t, []).append(q)

    w = _Writer(p)
    for q, (x, y) in enumerate(layout.positions):
        w.lines.append(f"QUBIT_COORDS({x}, {y}) {q}")
    UNPREPARED, LIVE, DONE = 0, 1, 2
    life = [UNPREPARED] * n
    in_merge: list[Optional[int]] = [None] * n
    # Each merge: patches, horizontal, code, data, seam, new_checks, outcomes, seam_recs.
    merges: list[dict] = []
    active: list[int] = []
    final_basis: list[Optional[str]] = [None] * n
    final_rec: list[dict[int, int]] = [{} for _ in range(n)]

    def check(q: int) -> None:
        if not 0 <= q < n:
            raise ValueError(f"there is no patch {q}")

    def listed(patches: Sequence[int]) -> None:
        if not patches:
            raise ValueError("a step names no patch")
        for i, q in enumerate(patches):
            check(q)
            if q in patches[:i]:
                raise ValueError(f"patch {q} is named twice in one step")

    for step in steps:
        kind = step[0]
        if kind == "prepare":
            patches, basis = list(step[1]), step[2]
            listed(patches)
            for q in patches:
                if life[q] != UNPREPARED:
                    raise ValueError(f"patch {q} is prepared twice")
                for other in on_tile[tiles[q]]:
                    if other != q and life[other] == LIVE:
                        raise ValueError(f"patch {q} is prepared on tile {tiles[q]}, which patch {other} still holds")
                life[q] = LIVE
                # A patch on a tile used before starts afresh (no-op on a tile's first patch).
                for ch in codes[q]:
                    w.last.pop(ch.pos, None)
            w.prepare(basis, [x for q in patches for x in datas[q]])
        elif kind == "rounds":
            k = step[1]
            units = [(q, None) for q in range(n) if life[q] == LIVE and in_merge[q] is None]
            units += [(merges[m]["patches"][0], m) for m in active]
            units.sort(key=lambda u: u[0])
            if not units:
                raise ValueError("rounds with no live patch")
            checks: list[_Check] = []
            data: list[int] = []
            offsets = []
            for q, m in units:
                offsets.append(len(checks))
                if m is None:
                    checks.extend(codes[q])
                    data.extend(datas[q])
                else:
                    checks.extend(merges[m]["code"])
                    data.extend(merges[m]["data"])
            for _ in range(k):
                recs = w.run_round(checks, data)
                for (_, m), off in zip(units, offsets):
                    if m is not None and merges[m]["outcomes"] is None:
                        merges[m]["outcomes"] = [[recs[off + i] for i in seam] for seam in merges[m]["new_checks"]]
        elif kind == "merge":
            patches = list(step[1])
            listed(patches)
            if len(patches) < 2:
                raise ValueError("a merge needs at least two patches")
            for q in patches:
                if life[q] != LIVE or in_merge[q] is not None:
                    raise ValueError(f"patch {q} is not live and alone, so it cannot merge")
            t = [tiles[q] for q in patches]
            horizontal = all(b == (a[0] + 1, a[1]) for a, b in zip(t, t[1:]))
            vertical = all(b == (a[0], a[1] + 1) for a, b in zip(t, t[1:]))
            if not horizontal and not vertical:
                raise ValueError(f"patches {patches} are not a line of neighbouring tiles, in order")
            bbox = boxes[patches[0]]
            for q in patches[1:]:
                bbox = bbox.union(boxes[q])
            code = layout.code(bbox)
            data = layout.data(bbox)
            seam = [x for x in data if not any(x in data_sets[q] for q in patches)]
            own = {pc.pos for q in patches for pc in codes[q]}

            def across(c: _Check) -> int:
                return c.pos[0] if horizontal else c.pos[1]

            def span(q: int) -> tuple[int, int]:
                return (boxes[q].x0, boxes[q].x1) if horizontal else (boxes[q].y0, boxes[q].y1)

            new_checks = []
            for a, b in zip(patches, patches[1:]):
                after, before = span(a)[1], span(b)[0]
                new_checks.append([i for i, c in enumerate(code) if c.x_type != horizontal and after < across(c) < before and c.pos not in own])
            w.lines.append("TICK")
            w.prepare("x" if horizontal else "z", seam)
            m = len(merges)
            for q in patches:
                in_merge[q] = m
            merges.append({"patches": patches, "horizontal": horizontal, "code": code, "data": data, "seam": seam, "new_checks": new_checks, "outcomes": None, "seam_recs": None})
            active.append(m)
        elif kind == "split":
            if not active:
                raise ValueError("a split with nothing merged")
            w.lines.append("TICK")
            for m in active:
                merges[m]["seam_recs"] = w.measure("x" if merges[m]["horizontal"] else "z", merges[m]["seam"])
                for q in merges[m]["patches"]:
                    in_merge[q] = None
            active = []
        elif kind == "measure":
            patches, basis = list(step[1]), step[2]
            listed(patches)
            for q in patches:
                if life[q] != LIVE or in_merge[q] is not None:
                    raise ValueError(f"patch {q} is not live and alone, so it cannot be measured")
            w.lines.append("TICK")
            data = [x for q in patches for x in datas[q]]
            recs = w.measure(basis, data)
            readout = dict(zip(data, recs))
            want_x = basis == "x"
            for q in patches:
                life[q] = DONE
                final_basis[q] = basis
                final_rec[q] = {x: readout[x] for x in datas[q]}
                for ch in codes[q]:
                    if ch.x_type != want_x:
                        continue
                    # Only a check measured in the last round predicts the readout.
                    last = w.last.get(ch.pos)
                    if last is None or last[2] != w.round:
                        continue
                    last_rec, last_support, _ = last
                    support = ch.support()
                    targets = [final_rec[q][x] for x in support]
                    targets.append(last_rec)
                    # Qubits its last measurement read and its patch does not: seam qubits,
                    # read at a split in the same basis.
                    support_set = set(support)
                    for lost in last_support:
                        if lost in support_set:
                            continue
                        got = w.measured.get(lost)
                        if got is not None and got[1] == basis and got[0] > last_rec:
                            targets.append(got[0])
                        else:
                            raise ValueError(f"the check at {ch.pos} lost a qubit not measured in its basis")
                    w.lines.append(f"DETECTOR({ch.pos[0]}, {ch.pos[1]}, {w.round + 1}) " + " ".join(f"rec[-{w.lookback(r)}]" for r in targets))
        else:
            raise ValueError(f"unknown surgery step {kind!r}")

    for index, terms in enumerate(observables):
        recs: list[int] = []
        for term in terms:
            name = term[0]
            if name == "logical":
                _, patch, basis, line = term
                check(patch)
                if final_basis[patch] != basis:
                    raise ValueError(f"patch {patch}'s logical {basis.upper()} needs it measured in {basis.upper()}")
                if line >= d:
                    raise ValueError(f"a patch has {d} lines, not {line + 1}")
                b = boxes[patch]
                at = 2 * line
                if basis == "z":
                    qs = [layout.index[(b.x0 + at, y)] for y in range(b.y0, b.y1 + 1, 2)]
                else:
                    qs = [layout.index[(x, b.y0 + at)] for x in range(b.x0, b.x1 + 1, 2)]
                recs.extend(w.lookback(final_rec[patch][q]) for q in qs)
            elif name == "outcome":
                _, merge, seam = term
                if not 0 <= merge < len(merges):
                    raise ValueError(f"there is no merge {merge}")
                out = merges[merge]["outcomes"]
                if out is None:
                    raise ValueError(f"merge {merge} had no rounds")
                if not 0 <= seam < len(out):
                    raise ValueError(f"merge {merge} has no seam {seam}")
                recs.extend(w.lookback(r) for r in out[seam])
            elif name == "seam":
                _, merge, patch, line = term
                check(patch)
                if line >= d:
                    raise ValueError(f"a patch has {d} lines, not {line + 1}")
                if not 0 <= merge < len(merges):
                    raise ValueError(f"there is no merge {merge}")
                ms = merges[merge]
                if patch not in ms["patches"]:
                    raise ValueError(f"patch {patch} is not in merge {merge}")
                if ms["seam_recs"] is None:
                    raise ValueError(f"merge {merge} was never split")
                b = boxes[patch]
                at = 2 * line
                for q, r in zip(ms["seam"], ms["seam_recs"]):
                    x, y = layout.positions[q]
                    if (ms["horizontal"] and y == b.y0 + at) or (not ms["horizontal"] and x == b.x0 + at):
                        recs.append(w.lookback(r))
            else:
                raise ValueError(f"unknown surgery term {name!r}")
        w.lines.append(f"OBSERVABLE_INCLUDE({index}) " + " ".join(f"rec[-{k}]" for k in recs))
    return "\n".join(w.lines) + "\n"
