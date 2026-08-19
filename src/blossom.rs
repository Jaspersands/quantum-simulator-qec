//! Exact minimum-weight perfect matching (Edmonds' blossom), with hard ceilings.
//!
//! # Why this is written defensively
//!
//! The algorithm is intricate — odd cycles are contracted into "blossoms",
//! searched over, and expanded again — and an earlier attempt at it had a loop
//! in `add_blossom` that appended to a growing vector forever when the
//! contracted structure was inconsistent. It consumed a hundred gigabytes before
//! anyone stopped it. macOS enforces neither `ulimit -v` nor `ulimit -d`, so
//! there is no outside guard rail to fall back on.
//!
//! Hence: every array is sized from `n` at construction and never grows, every
//! loop carries an explicit bound, and any violation sets `failed` and unwinds to
//! `None`. Total memory is O(n²) words, fixed before the search starts. A bug in
//! here can produce a wrong answer — the oracle test exists to catch that — but
//! it cannot exhaust memory, because there is nothing left to allocate.
//!
//! `None` is not a failure the caller has to handle specially: the decoder keeps
//! its existing matching, which is never worse than Union-Find.

use std::collections::VecDeque;

const INF: i64 = i64::MAX / 4;

/// Ceiling on vertices. 2·defects at d = 11 stays far below this; beyond it the
/// O(n³) search would be too slow to want anyway.
pub const MAX_VERTICES: usize = 512;

struct Blossom {
    n: usize,
    n_x: usize,
    eu: Vec<Vec<usize>>,
    ev: Vec<Vec<usize>>,
    ew: Vec<Vec<i64>>,
    lab: Vec<i64>,
    mate: Vec<usize>,
    slack: Vec<usize>,
    st: Vec<usize>,
    pa: Vec<usize>,
    flower_from: Vec<Vec<usize>>,
    s: Vec<i32>,
    vis: Vec<usize>,
    /// Contents of each blossom. Capacity is fixed; `push_flower` refuses to grow.
    flower: Vec<Vec<usize>>,
    q: VecDeque<usize>,
    stamp: usize,
    /// Set by any guard. Once true every operation is a no-op and the solve
    /// returns None.
    failed: bool,
    /// Which guard tripped first — for diagnosing an implementation that bails
    /// more often than it should.
    why: &'static str,
    /// Total work done, against a global ceiling — a backstop for a spin that
    /// individual loop bounds happen not to catch.
    work: usize,
    work_cap: usize,
}

impl Blossom {
    fn new(n: usize, weights: &[Vec<i64>]) -> Self {
        let m = n * 2 + 1;
        let mut b = Blossom {
            n,
            n_x: n,
            eu: vec![vec![0; m]; m],
            ev: vec![vec![0; m]; m],
            ew: vec![vec![0; m]; m],
            lab: vec![0; m],
            mate: vec![0; m],
            slack: vec![0; m],
            st: vec![0; m],
            pa: vec![0; m],
            flower_from: vec![vec![0; n + 1]; m],
            s: vec![0; m],
            vis: vec![0; m],
            // A blossom is an odd cycle in the *contracted* graph, so it can hold
            // up to n_x = 2n nodes — not n. Sizing it at n was an
            // under-allocation that tripped the capacity guard on 7% of
            // instances.
            flower: (0..m).map(|_| Vec::with_capacity(2 * n + 4)).collect(),
            q: VecDeque::with_capacity(m * 4),
            stamp: 0,
            failed: false,
            why: "",
            work: 0,
            work_cap: 4000 * (n + 2) * (n + 2),
        };
        for u in 1..=n {
            for v in 1..=n {
                b.eu[u][v] = u;
                b.ev[u][v] = v;
                b.ew[u][v] = if u == v { 0 } else { weights[u - 1][v - 1] };
            }
        }
        b
    }

    fn fail(&mut self, why: &'static str) {
        if !self.failed {
            self.failed = true;
            self.why = why;
        }
    }

    /// Charge one unit of work; false once the budget is gone.
    #[inline]
    fn tick(&mut self) -> bool {
        self.work += 1;
        if self.work > self.work_cap {
            self.fail("work budget");
        }
        !self.failed
    }

    /// Append to a blossom without ever reallocating.
    fn push_flower(&mut self, b: usize, value: usize) {
        if self.flower[b].len() >= self.flower[b].capacity() {
            self.fail("flower capacity");
            return;
        }
        self.flower[b].push(value);
    }

    fn push_q(&mut self, x: usize) {
        if self.q.len() >= self.q.capacity() {
            self.fail("queue capacity");
            return;
        }
        self.q.push_back(x);
    }

    fn e_delta(&self, u: usize, v: usize) -> i64 {
        self.lab[self.eu[u][v]] + self.lab[self.ev[u][v]] - self.ew[self.eu[u][v]][self.ev[u][v]] * 2
    }

    fn update_slack(&mut self, u: usize, x: usize) {
        if self.slack[x] == 0 || self.e_delta(u, x) < self.e_delta(self.slack[x], x) {
            self.slack[x] = u;
        }
    }

    fn set_slack(&mut self, x: usize) {
        self.slack[x] = 0;
        for u in 1..=self.n {
            if self.ew[u][x] > 0 && self.st[u] != x && self.s[self.st[u]] == 0 {
                self.update_slack(u, x);
            }
        }
    }

    fn q_push(&mut self, x: usize) {
        if self.failed {
            return;
        }
        if x <= self.n {
            self.push_q(x);
        } else {
            for i in 0..self.flower[x].len() {
                if !self.tick() {
                    return;
                }
                let y = self.flower[x][i];
                self.q_push(y);
            }
        }
    }

    fn set_st(&mut self, x: usize, b: usize) {
        if self.failed {
            return;
        }
        self.st[x] = b;
        if x > self.n {
            for i in 0..self.flower[x].len() {
                if !self.tick() {
                    return;
                }
                let y = self.flower[x][i];
                self.set_st(y, b);
            }
        }
    }

    fn get_pr(&mut self, b: usize, xr: usize) -> usize {
        match self.flower[b].iter().position(|&y| y == xr) {
            Some(pr) => {
                if pr % 2 == 1 {
                    let len = self.flower[b].len();
                    self.flower[b][1..len].reverse();
                    len - pr
                } else {
                    pr
                }
            }
            None => {
                self.fail("get_pr: not in blossom");
                0
            }
        }
    }

    fn set_match(&mut self, u: usize, v: usize) {
        if self.failed || !self.tick() {
            return;
        }
        self.mate[u] = self.ev[u][v];
        if u > self.n {
            let xr = self.flower_from[u][self.eu[u][v]];
            if xr == 0 {
                self.fail("set_match: no flower_from");
                return;
            }
            let pr = self.get_pr(u, xr);
            if self.failed || pr > self.flower[u].len() {
                self.fail("set_match: bad pr");
                return;
            }
            for i in 0..pr {
                let a = self.flower[u][i];
                let b = self.flower[u][i ^ 1];
                self.set_match(a, b);
            }
            self.set_match(xr, v);
            self.flower[u].rotate_left(pr);
        }
    }

    fn augment(&mut self, mut u: usize, mut v: usize) {
        for _ in 0..=(8 * self.n + 32) {
            if self.failed {
                return;
            }
            let xnv = self.st[self.mate[u]];
            self.set_match(u, v);
            if xnv == 0 {
                return;
            }
            let pa_x = self.st[self.pa[xnv]];
            self.set_match(xnv, pa_x);
            u = pa_x;
            v = xnv;
        }
        self.fail("augment: too many steps");
    }

    fn get_lca(&mut self, mut u: usize, mut v: usize) -> usize {
        self.stamp += 1;
        let t = self.stamp;
        for _ in 0..=(4 * self.n + 8) {
            if u == 0 && v == 0 {
                return 0;
            }
            if u != 0 {
                if self.vis[u] == t {
                    return u;
                }
                self.vis[u] = t;
                u = self.st[self.mate[u]];
                if u != 0 {
                    u = self.st[self.pa[u]];
                }
            }
            std::mem::swap(&mut u, &mut v);
        }
        self.fail("get_lca: too many steps");
        0
    }

    fn add_blossom(&mut self, u: usize, lca: usize, v: usize) {
        let mut b = self.n + 1;
        while b <= self.n_x && self.st[b] != 0 {
            b += 1;
        }
        if b > self.n_x {
            self.n_x += 1;
            b = self.n_x;
        }
        if b >= self.st.len() {
            self.fail("add_blossom: out of blossom slots");
            return;
        }
        self.lab[b] = 0;
        self.s[b] = 0;
        self.mate[b] = self.mate[lca];
        self.flower[b].clear();
        self.push_flower(b, lca);

        // Walking up from u to the common ancestor visits at most n nodes; more
        // than that means the alternating tree is inconsistent. This is the loop
        // that once ran away.
        let mut x = u;
        let mut steps = 0usize;
        while x != lca {
            steps += 1;
            if steps > 2 * self.n + 4 || self.failed {
                self.fail("add_blossom: walk from u");
                return;
            }
            let y = self.st[self.mate[x]];
            self.push_flower(b, x);
            self.push_flower(b, y);
            let p = self.st[self.pa[y]];
            self.q_push(y);
            x = p;
        }
        let len = self.flower[b].len();
        if len > 1 {
            self.flower[b][1..len].reverse();
        }

        let mut x = v;
        let mut steps = 0usize;
        while x != lca {
            steps += 1;
            if steps > 2 * self.n + 4 || self.failed {
                self.fail("add_blossom: walk from v");
                return;
            }
            let y = self.st[self.mate[x]];
            self.push_flower(b, x);
            self.push_flower(b, y);
            let p = self.st[self.pa[y]];
            self.q_push(y);
            x = p;
        }
        if self.failed {
            return;
        }

        self.set_st(b, b);
        for x in 1..=self.n_x {
            self.ew[b][x] = 0;
            self.ew[x][b] = 0;
        }
        for x in 1..=self.n {
            self.flower_from[b][x] = 0;
        }
        for i in 0..self.flower[b].len() {
            let xs = self.flower[b][i];
            for x in 1..=self.n_x {
                if self.ew[b][x] == 0 || self.e_delta(xs, x) < self.e_delta(b, x) {
                    self.eu[b][x] = self.eu[xs][x];
                    self.ev[b][x] = self.ev[xs][x];
                    self.ew[b][x] = self.ew[xs][x];
                    self.eu[x][b] = self.eu[x][xs];
                    self.ev[x][b] = self.ev[x][xs];
                    self.ew[x][b] = self.ew[x][xs];
                }
            }
            for x in 1..=self.n {
                if self.flower_from[xs][x] != 0 {
                    self.flower_from[b][x] = xs;
                }
            }
        }
        self.set_slack(b);
    }

    fn expand_blossom(&mut self, b: usize) {
        for i in 0..self.flower[b].len() {
            let y = self.flower[b][i];
            self.set_st(y, y);
        }
        let anchor = self.eu[b][self.pa[b]];
        let xr = self.flower_from[b][anchor];
        if xr == 0 {
            self.fail("expand: no flower_from");
            return;
        }
        let pr = self.get_pr(b, xr);
        if self.failed || pr >= self.flower[b].len() {
            self.fail("expand: bad pr");
            return;
        }
        let mut i = 0;
        while i + 1 < pr + 1 && i + 1 < self.flower[b].len() && i < pr {
            let xs = self.flower[b][i];
            let xns = self.flower[b][i + 1];
            self.pa[xs] = self.eu[xns][xs];
            self.s[xs] = 1;
            self.s[xns] = 0;
            self.slack[xs] = 0;
            self.set_slack(xns);
            self.q_push(xns);
            i += 2;
        }
        self.s[xr] = 1;
        self.pa[xr] = self.pa[b];
        for i in (pr + 1)..self.flower[b].len() {
            let xs = self.flower[b][i];
            self.s[xs] = -1;
            self.set_slack(xs);
        }
        self.st[b] = 0;
    }

    /// `a` and `b` index the edge as the caller found it — possibly blossoms.
    /// The parent pointer must record the *original* vertex on `a`'s side of
    /// that specific edge, so the lookup has to happen before either index is
    /// replaced by the blossom containing it. Shadowing them first, as an
    /// earlier version did, stored the endpoint from the other side of a
    /// different edge, and the alternating tree it built could not be walked
    /// back — which surfaced as `augment` and `add_blossom` failing to
    /// terminate on roughly half of all instances.
    fn on_found_edge(&mut self, a: usize, b: usize) -> bool {
        let u = self.st[a];
        let v = self.st[b];
        if self.s[v] == -1 {
            self.pa[v] = self.eu[a][b];
            self.s[v] = 1;
            let nu = self.st[self.mate[v]];
            self.slack[v] = 0;
            self.slack[nu] = 0;
            self.s[nu] = 0;
            self.q_push(nu);
        } else if self.s[v] == 0 {
            let lca = self.get_lca(u, v);
            if self.failed {
                return false;
            }
            if lca == 0 {
                self.augment(u, v);
                self.augment(v, u);
                return !self.failed;
            }
            self.add_blossom(u, lca, v);
        }
        false
    }

    fn matching(&mut self) -> bool {
        if self.failed {
            return false;
        }
        for x in 0..=self.n_x {
            self.s[x] = -1;
            self.slack[x] = 0;
        }
        self.q.clear();
        for x in 1..=self.n_x {
            if self.st[x] == x && self.mate[x] == 0 {
                self.pa[x] = 0;
                self.s[x] = 0;
                self.q_push(x);
            }
        }
        if self.q.is_empty() || self.failed {
            return false;
        }

        for _ in 0..=(4 * self.n + 16) {
            while let Some(u) = self.q.pop_front() {
                if !self.tick() {
                    return false;
                }
                if self.s[self.st[u]] == 1 {
                    continue;
                }
                for v in 1..=self.n {
                    if self.ew[u][v] > 0 && self.st[u] != self.st[v] {
                        if self.e_delta(u, v) == 0 {
                            if self.on_found_edge(u, v) {
                                return true;
                            }
                            if self.failed {
                                return false;
                            }
                        } else {
                            let sv = self.st[v];
                            self.update_slack(u, sv);
                        }
                    }
                }
            }

            let mut d = INF;
            for b in (self.n + 1)..=self.n_x {
                if self.st[b] == b && self.s[b] == 1 {
                    d = d.min(self.lab[b] / 2);
                }
            }
            for x in 1..=self.n_x {
                if self.st[x] == x && self.slack[x] != 0 {
                    if self.s[x] == -1 {
                        d = d.min(self.e_delta(self.slack[x], x));
                    } else if self.s[x] == 0 {
                        d = d.min(self.e_delta(self.slack[x], x) / 2);
                    }
                }
            }
            if d >= INF {
                return false;
            }
            for u in 1..=self.n {
                match self.s[self.st[u]] {
                    0 => {
                        if self.lab[u] <= d {
                            return false;
                        }
                        self.lab[u] -= d;
                    }
                    1 => self.lab[u] += d,
                    _ => {}
                }
            }
            for b in (self.n + 1)..=self.n_x {
                if self.st[b] == b {
                    match self.s[b] {
                        0 => self.lab[b] += d * 2,
                        1 => self.lab[b] -= d * 2,
                        _ => {}
                    }
                }
            }
            self.q.clear();
            for x in 1..=self.n_x {
                if self.st[x] == x
                    && self.slack[x] != 0
                    && self.st[self.slack[x]] != x
                    && self.e_delta(self.slack[x], x) == 0
                {
                    let sl = self.slack[x];
                    if self.on_found_edge(sl, x) {
                        return true;
                    }
                    if self.failed {
                        return false;
                    }
                }
            }
            for b in (self.n + 1)..=self.n_x {
                if self.st[b] == b && self.s[b] == 1 && self.lab[b] == 0 {
                    self.expand_blossom(b);
                    if self.failed {
                        return false;
                    }
                }
            }
        }
        self.fail("matching: too many passes");
        false
    }

    fn solve(&mut self) -> Option<Vec<usize>> {
        for x in 0..=self.n {
            self.mate[x] = 0;
            self.st[x] = x;
        }
        let mut w_max = 0i64;
        for u in 1..=self.n {
            for v in 1..=self.n {
                self.flower_from[u][v] = if u == v { u } else { 0 };
                w_max = w_max.max(self.ew[u][v]);
            }
        }
        for u in 1..=self.n {
            self.lab[u] = w_max;
        }
        let mut rounds = 0usize;
        while self.matching() {
            rounds += 1;
            if rounds > self.n {
                self.fail("solve: too many augmentations");
                break;
            }
        }
        if self.failed {
            return None;
        }
        // Only a genuine perfect matching is worth returning.
        for u in 1..=self.n {
            let m = self.mate[u];
            if m == 0 || m > self.n || m == u || self.mate[m] != u {
                self.fail("incomplete matching");
                return None;
            }
        }
        Some((1..=self.n).map(|u| self.mate[u] - 1).collect())
    }
}

/// Minimum-weight perfect matching on a complete graph of `n` vertices.
///
/// `cost[i][j]` is the weight of pairing `i` with `j`; `n` must be even. Returns
/// the partner of each vertex, or `None` if any internal ceiling was reached —
/// in which case the caller should keep whatever matching it already had.
pub fn min_weight_perfect_matching(n: usize, cost: &[Vec<i64>]) -> Option<Vec<usize>> {
    if n == 0 {
        return Some(Vec::new());
    }
    if !n.is_multiple_of(2) || n > MAX_VERTICES {
        return None;
    }
    // Maximise (C - w) instead of minimising w: every perfect matching uses the
    // same number of edges, so the orderings are exact mirrors.
    let mut top = 0i64;
    for row in cost.iter().take(n) {
        for &w in row.iter().take(n) {
            if w > top {
                top = w;
            }
        }
    }
    let flipped: Vec<Vec<i64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 0 } else { top - cost[i][j] + 1 }).collect())
        .collect();

    solve_with_reason(n, &flipped).0
}

/// Same, but reporting which ceiling stopped it. For tests and diagnosis.
pub fn min_weight_perfect_matching_diagnostic(
    n: usize,
    cost: &[Vec<i64>],
) -> (Option<Vec<usize>>, &'static str) {
    if n == 0 {
        return (Some(Vec::new()), "");
    }
    if !n.is_multiple_of(2) || n > MAX_VERTICES {
        return (None, "unsupported size");
    }
    let mut top = 0i64;
    for row in cost.iter().take(n) {
        for &w in row.iter().take(n) {
            if w > top {
                top = w;
            }
        }
    }
    let flipped: Vec<Vec<i64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 0 } else { top - cost[i][j] + 1 }).collect())
        .collect();
    solve_with_reason(n, &flipped)
}

fn solve_with_reason(n: usize, flipped: &[Vec<i64>]) -> (Option<Vec<usize>>, &'static str) {
    let mut b = Blossom::new(n, flipped);
    let out = b.solve();
    (out, b.why)
}
