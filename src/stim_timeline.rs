//! The text timeline exactly as Stim draws it (`diagram("timeline-text")`), worked out from
//! Stim's output: operations placed moment by moment (an operation moves to the next moment
//! when any wire it spans is taken), labelled as Stim labels them, each column as wide as its
//! widest label; a `TICK` group of more than one moment boxed above and below; a loop drawn once
//! between `|` columns headed `/REP n`, its records, detectors and coordinates written in terms
//! of `iter`.

use std::collections::BTreeMap;

use crate::circuit::{fmt_args, instr_line, split_instruction, Circuit, Instr};

/// Stim's number format in diagrams: whole numbers in full, others shortest.
fn num(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        fmt_args(&[x])
    }
}

fn args_text(a: &[f64]) -> String {
    if a.is_empty() {
        String::new()
    } else {
        format!("({})", a.iter().map(|&x| num(x)).collect::<Vec<_>>().join(","))
    }
}

/// An index in a loop drawn once: its value in the first pass, and its step per pass of each
/// enclosing loop (outermost first).
#[derive(Clone)]
struct Sym {
    base: i64,
    steps: Vec<i64>,
}

impl Sym {
    fn text(&self) -> String {
        let mut s = self.base.to_string();
        for (k, &step) in self.steps.iter().enumerate() {
            if step == 0 {
                continue;
            }
            let it = if k == 0 { "iter".to_string() } else { format!("iter{}", k + 1) };
            if step == 1 {
                s += &format!("+{it}");
            } else {
                s += &format!("+{it}*{step}");
            }
        }
        s
    }

    fn symbolic(&self) -> bool {
        self.steps.iter().any(|&s| s != 0)
    }
}

/// A coordinate in a loop drawn once: its first pass's value and its shift per pass.
fn coord_text(base: f64, steps: &[f64]) -> String {
    let mut s = num(base);
    for (k, &step) in steps.iter().enumerate() {
        if step == 0.0 {
            continue;
        }
        let it = if k == 0 { "iter".to_string() } else { format!("iter{}", k + 1) };
        if step == 1.0 {
            s += &format!("+{it}");
        } else {
            s += &format!("+{it}*{}", num(step));
        }
    }
    s
}

#[derive(Default)]
struct Grid {
    /// (column, row) → label; rows: 0 the top border, 1 + 2q qubit q, 2 + 2q between q and q + 1.
    cells: BTreeMap<(usize, usize), String>,
    /// Vertical lines: (column, first row, last row), drawn through the rows between.
    lines: Vec<(usize, usize, usize)>,
    /// Boxes over moment groups: (first column, last column).
    boxes: Vec<(usize, usize)>,
    /// Loop columns: (start column, end column).
    loops: Vec<(usize, usize)>,
}

struct Drawer {
    grid: Grid,
    qubits: usize,
    moment: usize,
    used: Vec<bool>,
    tick_start: usize,
    has_ticks: bool,
    /// The qubit of each measurement (first pass of each loop), by record index.
    rec_qubit: Vec<u32>,
    /// Records, detectors and coordinate shifts before the current point, as symbols.
    rec: Sym,
    det: Sym,
    shift: Vec<(f64, Vec<f64>)>,
    /// Steps of the enclosing loops (records, detectors, coordinate shifts per pass).
    depth: usize,
    /// Each qubit's coordinates, as last declared.
    coords: BTreeMap<u32, Vec<f64>>,
}

impl Drawer {
    fn row(q: u32) -> usize {
        1 + 2 * q as usize
    }

    fn any_used(&self) -> bool {
        self.used.iter().any(|&u| u)
    }

    fn next_moment(&mut self) {
        self.moment += 1;
        self.used.iter_mut().for_each(|u| *u = false);
    }

    /// Place labels on qubits, joined by a line from the lowest to the highest when `joined`.
    fn place(&mut self, cells: Vec<(u32, String)>, joined: bool) {
        if cells.is_empty() {
            return;
        }
        let lo = cells.iter().map(|c| c.0).min().unwrap() as usize;
        let hi = cells.iter().map(|c| c.0).max().unwrap() as usize;
        if hi >= self.used.len() {
            self.used.resize(hi + 1, false);
            self.qubits = self.qubits.max(hi + 1);
        }
        let span: Vec<usize> = if joined { (lo..=hi).collect() } else { cells.iter().map(|c| c.0 as usize).collect() };
        if span.iter().any(|&q| self.used[q]) {
            self.next_moment();
        }
        for &q in &span {
            self.used[q] = true;
        }
        for (q, label) in cells {
            self.grid.cells.insert((self.moment, Drawer::row(q)), label);
        }
        if joined && hi > lo {
            self.grid.lines.push((self.moment, Drawer::row(lo as u32), Drawer::row(hi as u32)));
        }
    }

    fn close_group(&mut self) {
        let end = if self.any_used() { self.moment } else { self.moment.saturating_sub(1) };
        if self.has_ticks && end > self.tick_start && self.moment >= self.tick_start {
            self.grid.boxes.push((self.tick_start, end));
        }
    }

    /// A `TICK` always moves on a moment, an empty one included.
    fn tick(&mut self) {
        self.close_group();
        self.next_moment();
        self.tick_start = self.moment;
    }

    fn record(&mut self, q: u32) -> String {
        let r = self.rec.clone();
        self.rec.base += 1;
        if (r.base as usize) < (1 << 22) {
            if self.rec_qubit.len() <= r.base as usize {
                self.rec_qubit.resize(r.base as usize + 1, u32::MAX);
            }
            self.rec_qubit[r.base as usize] = q;
        }
        format!("rec[{}]", r.text())
    }

    /// The record `k` back, as a symbol, and the qubit it measured.
    fn lookback(&self, k: u32) -> (Sym, u32) {
        let s = Sym { base: self.rec.base - i64::from(k), steps: self.rec.steps.clone() };
        let q = usize::try_from(s.base).ok().and_then(|i| self.rec_qubit.get(i)).copied().unwrap_or(u32::MAX);
        (s, q)
    }

    fn walk(&mut self, instrs: &[Instr]) {
        for ins in instrs {
            self.op(ins);
        }
    }

    fn op(&mut self, ins: &Instr) {
        match ins {
            Instr::Tick => return self.tick(),
            Instr::Repeat { count, body, .. } => return self.repeat(*count, body),
            Instr::ShiftCoords(c) => {
                for (k, &v) in c.iter().enumerate() {
                    if self.shift.len() <= k {
                        self.shift.push((0.0, vec![0.0; self.depth]));
                    }
                    self.shift[k].0 += v;
                }
                return;
            }
            _ => {}
        }
        let line = instr_line(ins);
        let Ok((name, _tag, a, tokens)) = split_instruction(&line) else { return };
        self.text_op(&name, &a, &tokens, ins);
    }

    fn text_op(&mut self, name: &str, a: &[f64], tokens: &[&str], ins: &Instr) {
        let head = format!("{name}{}", args_text(a));
        let qubit = |t: &str| t.trim_start_matches('!').parse::<u32>().ok();
        match name {
            "DETECTOR" => {
                // Drawn on the qubit at its coordinates, else on its lowest measured qubit.
                let mut refs = Vec::new();
                let mut q = u32::MAX;
                for t in tokens {
                    if let Some(k) = t.strip_prefix("rec[-").and_then(|r| r.strip_suffix(']')).and_then(|r| r.parse::<u32>().ok()) {
                        let (s, mq) = self.lookback(k);
                        q = q.min(mq);
                        refs.push(format!("rec[{}]", s.text()));
                    }
                }
                let at: Vec<f64> = a.iter().enumerate().map(|(k, &v)| v + self.shift.get(k).map_or(0.0, |s| s.0)).collect();
                if let Some(&cq) = self.coords.iter().filter(|(_, c)| !c.is_empty() && c.len() <= at.len() && at[..c.len()] == c[..]).map(|(q, _)| q).min() {
                    q = cq;
                }
                let d = self.det.clone();
                self.det.base += 1;
                let coords = if a.is_empty() {
                    String::new()
                } else {
                    let parts: Vec<String> = a
                        .iter()
                        .enumerate()
                        .map(|(k, &v)| {
                            let (base, steps) = self.shift.get(k).cloned().unwrap_or((0.0, vec![0.0; self.depth]));
                            coord_text(v + base, &steps)
                        })
                        .collect();
                    format!("({})", parts.join(","))
                };
                let id = if d.symbolic() { format!("D[{}]", d.text()) } else { format!("D{}", d.text()) };
                let rhs = if refs.is_empty() { "1".to_string() } else { refs.join("*") };
                self.place(vec![(if q == u32::MAX { 0 } else { q }, format!("DETECTOR{coords}:{id}={rhs}"))], false);
            }
            "OBSERVABLE_INCLUDE" => {
                let index = a.first().copied().unwrap_or(0.0) as u64;
                let mut refs = Vec::new();
                let mut paulis = Vec::new();
                let (mut earliest, mut q) = (i64::MAX, u32::MAX);
                for t in tokens {
                    if let Some(k) = t.strip_prefix("rec[-").and_then(|r| r.strip_suffix(']')).and_then(|r| r.parse::<u32>().ok()) {
                        let (s, mq) = self.lookback(k);
                        if s.base < earliest {
                            (earliest, q) = (s.base, mq);
                        }
                        refs.push(format!("rec[{}]", s.text()));
                    } else {
                        let t = t.trim_start_matches('!');
                        if let Ok(pq) = t[1..].parse::<u32>() {
                            paulis.push((pq, format!("L{index}*={}", &t[..1])));
                        }
                    }
                }
                if !paulis.is_empty() {
                    self.place(paulis, true);
                } else {
                    let rhs = refs.join("*");
                    self.place(vec![(if q == u32::MAX { 0 } else { q }, format!("OBSERVABLE_INCLUDE:L{index}*={rhs}"))], false);
                }
            }
            "MPP" | "SPP" | "SPP_DAG" => {
                for product in tokens {
                    let factors: Vec<(u32, char)> = product
                        .split('*')
                        .filter_map(|f| {
                            let f = f.trim_start_matches('!');
                            Some((f.get(1..)?.parse().ok()?, f.chars().next()?))
                        })
                        .collect();
                    let rec = if name == "MPP" { factors.first().map(|&(q, _)| format!(":{}", self.record(q))).unwrap_or_default() } else { String::new() };
                    let cells = factors.into_iter().map(|(q, p)| (q, format!("{name}[{p}]{}{rec}", args_text(a)))).collect();
                    self.place(cells, true);
                }
            }
            "E" | "ELSE_CORRELATED_ERROR" => {
                // A correlated error opens a moment of its own (others may then join it).
                if self.any_used() {
                    self.next_moment();
                }
                let cells = tokens
                    .iter()
                    .filter_map(|t| Some((t.get(1..)?.parse::<u32>().ok()?, format!("{name}[{}]{}", &t[..1], args_text(a)))))
                    .collect();
                self.place(cells, true);
            }
            "MPAD" => {
                for t in tokens {
                    if let Ok(v) = t.parse::<u32>() {
                        let r = self.record(v);
                        self.place(vec![(v, format!("MPAD:{r}"))], false);
                    }
                }
            }
            "QUBIT_COORDS" => {
                for t in tokens {
                    if let Some(q) = qubit(t) {
                        let at: Vec<f64> = a.iter().enumerate().map(|(k, &v)| v + self.shift.get(k).map_or(0.0, |s| s.0)).collect();
                        self.coords.insert(q, at);
                        self.place(vec![(q, format!("QUBIT_COORDS{}", args_text(a)))], false);
                    }
                }
            }
            _ => {
                let pairs = crate::gates::find(name).map(|g| g.arity == 2).unwrap_or(false)
                    || ["MXX", "MYY", "MZZ", "DEPOLARIZE2", "PAULI_CHANNEL_2", "II", "II_ERROR", "CX", "CY", "CZ", "SWAP"].contains(&name);
                let measures = name.starts_with('M') || name.starts_with("HERALDED");
                if pairs {
                    for pair in tokens.chunks(2) {
                        self.pair(name, &head, a, pair, measures);
                    }
                } else {
                    for t in tokens {
                        let Some(q) = qubit(t) else { continue };
                        let label = if measures { format!("{head}:{}", self.record(q)) } else { head.clone() };
                        self.place(vec![(q, label)], false);
                    }
                }
                let _ = ins;
            }
        }
    }

    fn pair(&mut self, name: &str, head: &str, a: &[f64], pair: &[&str], measures: bool) {
        let [t0, t1] = pair else { return };
        let control = |t: &str| -> Option<String> {
            if let Some(k) = t.strip_prefix("rec[-").and_then(|r| r.strip_suffix(']')).and_then(|r| r.parse::<u32>().ok()) {
                Some(format!("rec[{}]", self.lookback(k).0.text()))
            } else {
                t.strip_prefix("sweep[").and_then(|r| r.strip_suffix(']')).map(|k| format!("sweep[{k}]"))
            }
        };
        let qubit = |t: &str| t.trim_start_matches('!').parse::<u32>().ok();
        // Classically controlled: the Pauli on the qubit, by the bit.
        if let (Some(c), Some(q)) = (control(t0), qubit(t1)) {
            let p = match name {
                "CX" | "XCX" | "YCX" => "X",
                "CY" | "XCY" | "YCY" => "Y",
                _ => "Z",
            };
            return self.place(vec![(q, format!("{p}^{c}"))], false);
        }
        if let (Some(q), Some(c)) = (qubit(t0), control(t1)) {
            // `XCZ q rec`: the bit controls the first letter's Pauli on the qubit.
            let p = match name {
                "XCZ" => "X",
                "YCZ" => "Y",
                _ => "Z",
            };
            return self.place(vec![(q, format!("{p}^{c}"))], false);
        }
        let (Some(a0), Some(a1)) = (qubit(t0), qubit(t1)) else { return };
        let (l0, l1) = match name {
            "CX" => ("@".to_string(), "X".to_string()),
            "CY" => ("@".into(), "Y".into()),
            "CZ" => ("@".into(), "@".into()),
            "XCX" => ("X".into(), "X".into()),
            "XCY" => ("X".into(), "Y".into()),
            "XCZ" => ("X".into(), "@".into()),
            "YCX" => ("Y".into(), "X".into()),
            "YCY" => ("Y".into(), "Y".into()),
            "YCZ" => ("Y".into(), "@".into()),
            "CXSWAP" => ("ZSWAP".into(), "XSWAP".into()),
            "SWAPCX" => ("XSWAP".into(), "ZSWAP".into()),
            "CZSWAP" => ("ZSWAP".into(), "ZSWAP".into()),
            "PAULI_CHANNEL_2" => (format!("{name}[0]{}", args_text(a)), format!("{name}[1]{}", args_text(a))),
            _ if measures => (format!("{head}:{}", self.record(a0)), head.to_string()),
            _ => (head.to_string(), head.to_string()),
        };
        self.place(vec![(a0, l0), (a1, l1)], true);
    }

    fn repeat(&mut self, count: u64, body: &[Instr]) {
        // The group before the loop ends, and the loop's opening column is a moment of its own.
        self.close_group();
        if self.any_used() {
            self.next_moment();
        }
        let start = self.moment;
        self.grid.cells.insert((start, 0), format!("/REP {count}"));
        self.next_moment();
        self.tick_start = self.moment;
        let (r0, d0) = (self.rec.base, self.det.base);
        let s0: Vec<f64> = self.shift.iter().map(|s| s.0).collect();
        // Every symbol gains a step for this loop, found after a dry pass of the body.
        let (per_rec, per_det, per_shift) = self.passes(body);
        self.depth += 1;
        self.rec.steps.push(per_rec);
        self.det.steps.push(per_det);
        for (k, s) in self.shift.iter_mut().enumerate() {
            s.1.push(per_shift.get(k).copied().unwrap_or(0.0));
        }
        while self.shift.len() < per_shift.len() {
            let mut steps = vec![0.0; self.depth];
            steps[self.depth - 1] = per_shift[self.shift.len()];
            self.shift.push((0.0, steps));
        }
        self.walk(body);
        self.close_group();
        if self.any_used() {
            self.next_moment();
        }
        let end = self.moment;
        self.grid.loops.push((start, end));
        self.next_moment();
        self.tick_start = self.moment;
        // After the loop: every pass counted, the symbols' steps for it dropped.
        self.depth -= 1;
        self.rec.steps.pop();
        self.det.steps.pop();
        self.rec.base = r0 + per_rec * count as i64;
        self.det.base = d0 + per_det * count as i64;
        for (k, s) in self.shift.iter_mut().enumerate() {
            s.1.pop();
            s.0 = s0.get(k).copied().unwrap_or(0.0) + per_shift.get(k).copied().unwrap_or(0.0) * count as f64;
        }
    }

    /// Records, detectors and coordinate shifts in one pass of a body (loops counted in full).
    fn passes(&self, body: &[Instr]) -> (i64, i64, Vec<f64>) {
        let counts = crate::batch_sampler::Counts::of(body).unwrap_or_default();
        let mut shift = Vec::new();
        fn add(instrs: &[Instr], times: f64, shift: &mut Vec<f64>) {
            for i in instrs {
                match i {
                    Instr::ShiftCoords(c) => {
                        for (k, &v) in c.iter().enumerate() {
                            if shift.len() <= k {
                                shift.resize(k + 1, 0.0);
                            }
                            shift[k] += v * times;
                        }
                    }
                    Instr::Repeat { count, body, .. } => add(body, times * *count as f64, shift),
                    _ => {}
                }
            }
        }
        add(body, 1.0, &mut shift);
        (counts.measurements as i64, counts.detectors as i64, shift)
    }

    fn render(mut self) -> String {
        self.close_group();
        // Rows: the top border, qubit q at 1 + 2q with a gap row after each but the last, and
        // the bottom border.
        let rows = 2 * self.qubits.max(1);
        let cols = self.grid.cells.keys().map(|k| k.0 + 1).chain(self.grid.loops.iter().map(|l| l.1 + 1)).chain(self.grid.boxes.iter().map(|b| b.1 + 1)).max().unwrap_or(0);
        let mut width = vec![1usize; cols];
        for ((c, _), label) in &self.grid.cells {
            width[*c] = width[*c].max(label.chars().count());
        }
        let labels: Vec<String> = (0..self.qubits).map(|q| format!("q{q}:")).collect();
        let lw = labels.iter().map(|l| l.len()).max().unwrap_or(0);
        let mut x = vec![lw + 2; cols + 1];
        for c in 0..cols {
            x[c + 1] = x[c] + width[c] + 1;
        }
        let total = x[cols];
        let mut grid: Vec<Vec<char>> = (0..=rows).map(|r| vec![if r % 2 == 1 { '-' } else { ' ' }; total]).collect();
        // Qubit rows: the label, then wire.
        for (q, l) in labels.iter().enumerate() {
            let r = 1 + 2 * q;
            let pad = lw - l.len();
            for (i, ch) in format!("{}{l} ", " ".repeat(pad)).chars().enumerate() {
                grid[r][i] = ch;
            }
        }
        let put = |grid: &mut Vec<Vec<char>>, r: usize, at: usize, s: &str| {
            for (i, ch) in s.chars().enumerate() {
                if at + i < grid[r].len() {
                    grid[r][at + i] = ch;
                }
            }
        };
        for &(c, r0, r1) in &self.grid.lines {
            for r in r0 + 1..r1 {
                grid[r][x[c]] = '|';
            }
        }
        for &(s, e) in &self.grid.loops {
            for r in 1..rows {
                grid[r][x[s]] = '|';
                grid[r][x[e]] = '|';
            }
            grid[rows][x[s]] = '\\';
            grid[0][x[e]] = '\\';
            grid[rows][x[e]] = '/';
        }
        for &(s, e) in &self.grid.boxes {
            let (a, b) = (x[s], x[e] + width[e] - 1);
            for i in a..=b {
                grid[0][i] = '-';
                grid[rows][i] = '-';
            }
            grid[0][a] = '/';
            grid[0][b] = '\\';
            grid[rows][a] = '\\';
            grid[rows][b] = '/';
        }
        for ((c, r), label) in &self.grid.cells {
            put(&mut grid, *r, x[*c], label);
        }
        let mut lines: Vec<String> = grid.into_iter().map(|row| row.into_iter().collect::<String>().trim_end().to_string()).collect();
        let border = !self.grid.boxes.is_empty() || !self.grid.loops.is_empty();
        if !border {
            lines.remove(rows);
            lines.remove(0);
        }
        lines.join("\n") + "\n"
    }
}

/// The circuit's timeline as Stim's text.
pub fn timeline_text(circuit: &Circuit) -> Result<String, String> {
    let counts = crate::batch_sampler::Counts::of(&circuit.instrs)?;
    if counts.qubits > 4096 {
        return Err(format!("{} qubits: a timeline draws at most 4096", counts.qubits));
    }
    let has_ticks = circuit.to_stim().lines().any(|l| l.trim_start().starts_with("TICK"));
    let mut d = Drawer {
        grid: Grid::default(),
        qubits: counts.qubits,
        moment: 0,
        used: vec![false; counts.qubits.max(1)],
        tick_start: 0,
        has_ticks,
        rec_qubit: Vec::new(),
        rec: Sym { base: 0, steps: Vec::new() },
        det: Sym { base: 0, steps: Vec::new() },
        shift: Vec::new(),
        depth: 0,
        coords: BTreeMap::new(),
    };
    d.walk(&circuit.instrs);
    Ok(d.render())
}
