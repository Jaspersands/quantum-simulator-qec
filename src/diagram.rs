//! Pictures of a circuit and its model, after Stim's `diagram`: a timeline of its gates (as
//! text or SVG), the detectors' stabilizers at a moment (a detector slice), and the matching
//! graph.
//!
//! A timeline places each operation in the first column where every wire it crosses is free,
//! starts a new group of columns at each `TICK`, and draws a `REPEAT` block once, boxed with its
//! count; measurements are numbered by their record (the first pass of a loop), and detectors
//! and observables are written where they are declared. A detector slice walks the circuit
//! backwards from its end to the moment asked for, carrying for each qubit the detectors an X
//! error there would flip and those a Z error would (the error model's walk, `dem_build`): a
//! detector a Z error flips has an X component on that qubit, and one an X error flips a Z
//! component, which is the stabilizer the detector compares at that moment.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::circuit::{fmt_args, split_instruction, Basis, Circuit, Control, Instr};

/* -- Timeline layout ------------------------------------------------------- */

/// One operation in the timeline: its column, and its label on each wire it touches; `joined`
/// draws a line through the wires between them.
struct Op {
    col: usize,
    cells: Vec<(u32, String)>,
    joined: bool,
}

struct Layout {
    ops: Vec<Op>,
    cols: usize,
    qubits: usize,
    /// Column spans of each TICK's group, and of each loop with its count and nesting depth.
    ticks: Vec<(usize, usize)>,
    loops: Vec<(usize, usize, u64, usize)>,
}

struct Builder {
    layout: Layout,
    next: Vec<usize>,
    tick_start: usize,
    /// The qubit each measurement record read (the first pass of a loop), u32::MAX for none.
    rec_qubit: Vec<u32>,
    /// Records counted through loops, and detectors.
    records: u64,
    detectors: u64,
    /// Loops already walked, as (first record, records per pass, records in all), for finding
    /// the qubit of a record made in a pass not drawn.
    folds: Vec<(u64, u64, u64)>,
    depth: usize,
}

fn args(name: &str, a: &[f64]) -> String {
    if a.is_empty() {
        name.to_string()
    } else {
        format!("{name}({})", fmt_args(a))
    }
}

impl Builder {
    fn new(qubits: usize) -> Builder {
        Builder {
            layout: Layout {
                ops: Vec::new(),
                cols: 0,
                qubits,
                ticks: Vec::new(),
                loops: Vec::new(),
            },
            next: vec![0; qubits.max(1)],
            tick_start: 0,
            rec_qubit: Vec::new(),
            records: 0,
            detectors: 0,
            folds: Vec::new(),
            depth: 0,
        }
    }

    fn align(&mut self) -> usize {
        let m = self.next.iter().copied().max().unwrap_or(0);
        self.next.iter_mut().for_each(|n| *n = m);
        m
    }

    fn place(&mut self, cells: Vec<(u32, String)>, joined: bool) {
        if cells.is_empty() {
            return;
        }
        let qs: Vec<usize> = cells.iter().map(|c| c.0 as usize).collect();
        let (lo, hi) = (*qs.iter().min().unwrap(), *qs.iter().max().unwrap());
        let span: Vec<usize> = if joined {
            (lo..=hi).collect()
        } else {
            qs.clone()
        };
        let col = span.iter().map(|&q| self.next[q]).max().unwrap_or(0);
        for &q in &span {
            self.next[q] = col + 1;
        }
        self.layout.cols = self.layout.cols.max(col + 1);
        self.layout.ops.push(Op { col, cells, joined });
    }

    fn record(&mut self, q: u32) -> String {
        let r = self.records;
        self.records += 1;
        if self.rec_qubit.len() < (1 << 20) {
            self.rec_qubit.push(q);
        }
        format!("rec[{r}]")
    }

    /// The qubit measured by absolute record `r`, through loops drawn once.
    fn qubit_of(&self, mut r: u64) -> u32 {
        for &(first, per, all) in self.folds.iter().rev() {
            if per > 0 && r >= first + per && r < first + all {
                r = first + (r - first) % per;
            } else if r >= first + all {
                r -= all - per;
            }
        }
        self.rec_qubit.get(r as usize).copied().unwrap_or(u32::MAX)
    }

    fn lookback(&self, k: u32) -> Option<u64> {
        self.records.checked_sub(u64::from(k))
    }

    fn walk(&mut self, instrs: &[Instr], tag: &str) {
        for ins in instrs {
            self.op(ins, tag);
        }
    }

    fn op(&mut self, ins: &Instr, tag: &str) {
        let t = |name: &str| {
            if tag.is_empty() {
                name.to_string()
            } else {
                format!("{name}[{tag}]")
            }
        };
        let each = |qs: &[u32], label: &str| {
            qs.iter()
                .map(|&q| (q, label.to_string()))
                .collect::<Vec<_>>()
        };
        match ins {
            Instr::Reset { basis, qubits } => {
                let name = t(if *basis == Basis::X { "RX" } else { "R" });
                for &q in qubits {
                    self.place(vec![(q, name.clone())], false);
                }
            }
            Instr::H(q) => q.iter().for_each(|&q| self.place(vec![(q, t("H"))], false)),
            Instr::S(q) => q.iter().for_each(|&q| self.place(vec![(q, t("S"))], false)),
            Instr::Pauli { pauli, qubits } => {
                let name = t(["I", "X", "Z", "Y"][*pauli as usize]);
                qubits
                    .iter()
                    .for_each(|&q| self.place(vec![(q, name.clone())], false));
            }
            Instr::Cx(pairs) => pairs
                .iter()
                .for_each(|&(c, x)| self.place(vec![(c, "@".into()), (x, "X".into())], true)),
            Instr::Cz(pairs) => pairs
                .iter()
                .for_each(|&(a, b)| self.place(vec![(a, "@".into()), (b, "@".into())], true)),
            Instr::SweepX(pairs) => pairs
                .iter()
                .for_each(|&(k, q)| self.place(vec![(q, format!("X^sweep[{k}]"))], false)),
            Instr::Measure {
                basis,
                reset,
                flip,
                qubits,
            } => {
                let name = match (reset, basis) {
                    (false, Basis::Z) => "M",
                    (false, Basis::X) => "MX",
                    (true, Basis::Z) => "MR",
                    (true, Basis::X) => "MRX",
                };
                let name = if *flip > 0.0 {
                    args(&t(name), &[*flip])
                } else {
                    t(name)
                };
                for &q in qubits {
                    let r = self.record(q);
                    self.place(vec![(q, format!("{name}:{r}"))], false);
                }
            }
            Instr::PauliError { pauli, p, qubits } => {
                let name = args(
                    &t(["", "X_ERROR", "Z_ERROR", "Y_ERROR"][*pauli as usize]),
                    &[*p],
                );
                qubits
                    .iter()
                    .for_each(|&q| self.place(vec![(q, name.clone())], false));
            }
            Instr::Depolarize1 { p, qubits } => {
                let name = args(&t("DEPOLARIZE1"), &[*p]);
                qubits
                    .iter()
                    .for_each(|&q| self.place(vec![(q, name.clone())], false));
            }
            Instr::Depolarize2 { p, pairs } => {
                let name = args(&t("DEPOLARIZE2"), &[*p]);
                pairs
                    .iter()
                    .for_each(|&(a, b)| self.place(each(&[a, b], &name), true));
            }
            Instr::PauliChannel1 { px, py, pz, qubits } => {
                let name = args(&t("PAULI_CHANNEL_1"), &[*px, *py, *pz]);
                qubits
                    .iter()
                    .for_each(|&q| self.place(vec![(q, name.clone())], false));
            }
            Instr::PauliChannel2 { probs, pairs } => {
                let name = args(&t("PAULI_CHANNEL_2"), probs);
                pairs
                    .iter()
                    .for_each(|&(a, b)| self.place(each(&[a, b], &name), true));
            }
            Instr::Correlated { p, paulis, chained } => {
                let name = args(
                    &t(if *chained {
                        "ELSE_CORRELATED_ERROR"
                    } else {
                        "E"
                    }),
                    &[*p],
                );
                let cells = paulis
                    .iter()
                    .map(|&(q, pa)| (q, format!("{name}:{}", ["I", "X", "Z", "Y"][pa as usize])))
                    .collect();
                self.place(cells, true);
            }
            Instr::Pad { values, .. } => {
                for _ in values {
                    self.record(u32::MAX);
                }
            }
            Instr::Feedback {
                pauli,
                control,
                qubit,
            } => {
                let p = ["I", "X", "Z", "Y"][*pauli as usize];
                let c = match control {
                    Control::Rec(k) => self
                        .lookback(*k)
                        .map_or_else(|| format!("rec[-{k}]"), |r| format!("rec[{r}]")),
                    Control::Sweep(k) => format!("sweep[{k}]"),
                };
                self.place(vec![(*qubit, format!("{p}^{c}"))], false);
            }
            Instr::Heralded {
                erase,
                args: a,
                qubits,
                ..
            } => {
                let name = args(
                    &t(if *erase {
                        "HERALDED_ERASE"
                    } else {
                        "HERALDED_PAULI_CHANNEL_1"
                    }),
                    a,
                );
                for &q in qubits {
                    let r = self.record(q);
                    self.place(vec![(q, format!("{name}:{r}"))], false);
                }
            }
            Instr::Detector { coords, recs } => {
                let refs: Vec<u64> = recs.iter().filter_map(|&k| self.lookback(k)).collect();
                let q = refs
                    .iter()
                    .map(|&r| self.qubit_of(r))
                    .filter(|&q| q != u32::MAX)
                    .min()
                    .unwrap_or(0);
                let d = self.detectors;
                self.detectors += 1;
                let rhs: Vec<String> = refs.iter().map(|r| format!("rec[{r}]")).collect();
                let label = format!(
                    "{}:D{d}={}",
                    args(&t("DETECTOR"), coords),
                    if rhs.is_empty() {
                        "1".into()
                    } else {
                        rhs.join("*")
                    }
                );
                self.place(vec![(q, label)], false);
            }
            Instr::Observable {
                index,
                recs,
                paulis,
            } => {
                let refs: Vec<u64> = recs.iter().filter_map(|&k| self.lookback(k)).collect();
                let mut terms: Vec<String> = refs.iter().map(|r| format!("rec[{r}]")).collect();
                terms.extend(paulis.iter().map(|&(q, p, inv)| {
                    format!(
                        "{}{}{q}",
                        if inv { "!" } else { "" },
                        ["I", "X", "Z", "Y"][p as usize]
                    )
                }));
                let q = refs
                    .iter()
                    .map(|&r| self.qubit_of(r))
                    .chain(paulis.iter().map(|&(q, ..)| q))
                    .filter(|&q| q != u32::MAX)
                    .min()
                    .unwrap_or(0);
                self.place(
                    vec![(
                        q,
                        format!("{}:L{index}*={}", t("OBSERVABLE_INCLUDE"), terms.join("*")),
                    )],
                    false,
                );
            }
            Instr::QubitCoords { coords, qubits } => {
                let name = args(&t("QUBIT_COORDS"), coords);
                qubits
                    .iter()
                    .for_each(|&q| self.place(vec![(q, name.clone())], false));
            }
            Instr::ShiftCoords(_) => {}
            Instr::Tick => {
                let m = self.align();
                if m > self.tick_start {
                    self.layout.ticks.push((self.tick_start, m));
                }
                self.tick_start = m;
            }
            Instr::Repeat { count, body, .. } => {
                let start = self.align();
                self.tick_start = start;
                let first = self.records;
                let d0 = self.detectors;
                self.depth += 1;
                self.walk(body, tag);
                self.depth -= 1;
                let end = self.align();
                if end > self.tick_start {
                    self.layout.ticks.push((self.tick_start, end));
                }
                self.tick_start = end;
                let per = self.records - first;
                self.records = first + per * count;
                self.detectors = d0 + (self.detectors - d0) * count;
                self.folds.push((first, per, per * count));
                self.layout
                    .loops
                    .push((start, end.max(start + 1), *count, self.depth));
                self.layout.cols = self.layout.cols.max(end.max(start + 1));
                self.next.iter_mut().for_each(|n| *n = end.max(start + 1));
            }
            Instr::Gate {
                line,
                body,
                tag: own,
            } => {
                if !own.is_empty()
                    && body.len() == 1
                    && !matches!(body[0], Instr::Gate { .. })
                    && !line_is_gate(line)
                {
                    // A tagged instruction: drawn as itself, with its tag.
                    self.op(&body[0], own);
                } else {
                    self.gate(line, body);
                }
            }
        }
    }

    /// A gate the engine runs decomposed, drawn as written.
    fn gate(&mut self, line: &str, body: &[Instr]) {
        let Ok((name, tag, a, tokens)) = split_instruction(line) else {
            return;
        };
        let base = args(
            &if tag.is_empty() {
                name.clone()
            } else {
                format!("{name}[{tag}]")
            },
            &a,
        );
        let measures = body
            .iter()
            .filter(|i| matches!(i, Instr::Measure { .. }))
            .count();
        let pauli_products = tokens
            .iter()
            .any(|t| t.contains('*') || t.trim_start_matches('!').starts_with(['X', 'Y', 'Z']));
        if pauli_products {
            for tok in &tokens {
                let factors: Vec<(u32, String)> = tok
                    .split('*')
                    .filter_map(|f| {
                        let f = f.trim_start_matches('!');
                        let q = f.get(1..)?.parse::<u32>().ok()?;
                        Some((q, f[..1].to_string()))
                    })
                    .collect();
                let rec = if measures > 0 {
                    factors
                        .first()
                        .map(|&(q, _)| format!(":{}", self.record(q)))
                } else {
                    None
                };
                let cells = factors
                    .into_iter()
                    .map(|(q, p)| (q, format!("{base}:{p}{}", rec.clone().unwrap_or_default())))
                    .collect();
                self.place(cells, true);
            }
            return;
        }
        let qubits: Vec<(u32, bool)> = tokens
            .iter()
            .filter_map(|t| {
                let inv = t.starts_with('!');
                t.trim_start_matches('!')
                    .parse::<u32>()
                    .ok()
                    .map(|q| (q, inv))
            })
            .collect();
        let classical: Vec<&&str> = tokens
            .iter()
            .filter(|t| t.starts_with("rec[") || t.starts_with("sweep["))
            .collect();
        if !classical.is_empty() {
            // Classically controlled gates: the engine's feedback, drawn as such.
            self.walk(body, "");
            return;
        }
        let arity = crate::gates::find(&name).map_or_else(
            || {
                if ["MXX", "MYY", "MZZ", "II", "II_ERROR"].contains(&name.as_str()) {
                    2
                } else {
                    1
                }
            },
            |g| g.arity as usize,
        );
        let per_record = measures > 0 && measures < qubits.len();
        for group in qubits.chunks(arity.max(1)) {
            let rec = if measures > 0 && (per_record || arity == 1) {
                Some(self.record(group[0].0))
            } else {
                None
            };
            let cells = group
                .iter()
                .map(|&(q, inv)| {
                    (
                        q,
                        format!(
                            "{}{base}{}",
                            if inv { "!" } else { "" },
                            rec.as_ref().map_or(String::new(), |r| format!(":{r}"))
                        ),
                    )
                })
                .collect();
            self.place(cells, arity > 1);
        }
    }
}

/// Whether a gate's line names one of the engine's own decomposed gates (not a tagged native
/// instruction).
fn line_is_gate(line: &str) -> bool {
    split_instruction(line).is_ok_and(|(name, ..)| {
        crate::gates::find(&name).is_some()
            || [
                "MY", "MRY", "RY", "MPP", "MXX", "MYY", "MZZ", "SPP", "SPP_DAG", "II", "I_ERROR",
                "II_ERROR",
            ]
            .contains(&name.as_str())
    })
}

fn layout(circuit: &Circuit) -> Result<Layout, String> {
    let counts = crate::batch_sampler::Counts::of(&circuit.instrs)?;
    if counts.qubits > 4096 {
        return Err(format!(
            "{} qubits: a timeline draws at most 4096",
            counts.qubits
        ));
    }
    let mut b = Builder::new(counts.qubits);
    b.walk(&circuit.instrs, "");
    let m = b.align();
    if m > b.tick_start {
        b.layout.ticks.push((b.tick_start, m));
    }
    b.layout.cols = b.layout.cols.max(m);
    Ok(b.layout)
}

/// The timeline as text: a row per qubit, `@`/`X` and `|` for controlled gates, `/---\` over
/// each TICK's group and `/REP n` over each loop.
pub fn timeline_text(circuit: &Circuit) -> Result<String, String> {
    crate::stim_timeline::timeline_text(circuit)
}

/* -- SVG -------------------------------------------------------------------- */

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// Each picture paints its own background, so its dark-mode colours never land on a light page.
const STYLE: &str = "<style>text{font-family:ui-monospace,Menlo,Consolas,monospace;fill:#1c1d1f}.bg{fill:#fff}\
.wire{stroke:#9a9a9a;stroke-width:1}.box{fill:#fff;stroke:#1c1d1f;stroke-width:1}.dot{fill:#1c1d1f}.edge{stroke:#8a8a8a}\
.meas{fill:#f4f2ec;stroke:#1c1d1f}.noise{fill:#fbe9e7;stroke:#c0392b}.ann{fill:#eef3fb;stroke:#2c5f9e}\
.join{stroke:#1c1d1f;stroke-width:1.5}.tick{fill:#f3f3f1}.loop{fill:none;stroke:#2c5f9e;stroke-dasharray:4 3}.dim{fill:#8a8a8a}\
@media (prefers-color-scheme:dark){text{fill:#e8e6e1}.bg{fill:#161616}.box{fill:#1c1d1f;stroke:#e8e6e1}.dot{fill:#e8e6e1}\
.meas{fill:#2a2926;stroke:#e8e6e1}.noise{fill:#3a2422}.ann{fill:#1f2a38}.join{stroke:#e8e6e1}.tick{fill:#222}.edge{stroke:#777}}</style>";

/// An SVG's opening: its size, the style sheet and the background.
fn open_svg(w: f64, h: f64, font: u32) -> String {
    format!("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w:.0} {h:.0}\" width=\"{w:.0}\" height=\"{h:.0}\" font-size=\"{font}\">{STYLE}<rect class=\"bg\" width=\"100%\" height=\"100%\"/>")
}

/// The timeline as an SVG picture.
pub fn timeline_svg(circuit: &Circuit) -> Result<String, String> {
    let l = layout(circuit)?;
    let nq = l.qubits.max(1);
    let mut width = vec![1usize; l.cols];
    for op in &l.ops {
        for (_, s) in &op.cells {
            width[op.col] = width[op.col].max(s.chars().count());
        }
    }
    let colw: Vec<f64> = width.iter().map(|&w| 7.2 * w as f64 + 18.0).collect();
    let mut x0 = vec![60.0f64; l.cols + 1];
    for c in 0..l.cols {
        x0[c + 1] = x0[c] + colw[c] + 8.0;
    }
    let (row, top) = (
        44.0,
        34.0 + 14.0 * l.loops.iter().map(|x| x.3).max().unwrap_or(0) as f64,
    );
    let y = |q: u32| top + 20.0 + row * q as f64;
    let w = x0[l.cols] + 20.0;
    let h = y(nq as u32 - 1) + 40.0;
    let mut s = open_svg(w, h, 12);
    for &(a, b) in &l.ticks {
        let _ = write!(s, "<rect class=\"tick\" x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"4\"/>", x0[a] - 3.0, top, x0[b] - x0[a] - 2.0, h - top - 10.0);
    }
    for &(a, b, n, depth) in &l.loops {
        let yy = 8.0 + 14.0 * (depth as f64 - 1.0);
        let _ = write!(
            s,
            "<rect class=\"loop\" x=\"{:.1}\" y=\"{yy:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"6\"/><text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\">REPEAT {n}</text>",
            x0[a] - 5.0,
            x0[b.min(l.cols)] - x0[a] + 2.0,
            h - yy - 4.0,
            x0[a] + 2.0,
            yy + 11.0
        );
    }
    for q in 0..nq as u32 {
        let _ = write!(s, "<line class=\"wire\" x1=\"50\" y1=\"{0:.1}\" x2=\"{1:.1}\" y2=\"{0:.1}\"/><text x=\"8\" y=\"{2:.1}\">q{q}</text>", y(q), w - 10.0, y(q) + 4.0);
    }
    for op in &l.ops {
        let cx = x0[op.col] + colw[op.col] / 2.0;
        if op.joined && op.cells.len() > 1 {
            let lo = op.cells.iter().map(|c| c.0).min().unwrap();
            let hi = op.cells.iter().map(|c| c.0).max().unwrap();
            let _ = write!(
                s,
                "<line class=\"join\" x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\"/>",
                y(lo),
                y(hi)
            );
        }
        for (q, label) in &op.cells {
            let yq = y(*q);
            if label == "@" {
                let _ = write!(
                    s,
                    "<circle class=\"dot\" cx=\"{cx:.1}\" cy=\"{yq:.1}\" r=\"5\"/>"
                );
                continue;
            }
            if label == "X" && op.joined {
                let _ = write!(
                    s,
                    "<circle cx=\"{cx:.1}\" cy=\"{yq:.1}\" r=\"10\" class=\"box\"/><line class=\"join\" x1=\"{:.1}\" y1=\"{yq:.1}\" x2=\"{:.1}\" y2=\"{yq:.1}\"/><line class=\"join\" x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\"/>",
                    cx - 10.0,
                    cx + 10.0,
                    yq - 10.0,
                    yq + 10.0
                );
                continue;
            }
            let class = if label.contains("rec[")
                && (label.starts_with('M')
                    || label.starts_with("!M")
                    || label.starts_with("HERALDED"))
            {
                "box meas"
            } else if label.contains("ERROR")
                || label.starts_with("DEPOLARIZE")
                || label.starts_with("PAULI_CHANNEL")
                || label.starts_with('E')
            {
                "box noise"
            } else if label.starts_with("DETECTOR")
                || label.starts_with("OBSERVABLE")
                || label.starts_with("QUBIT_COORDS")
            {
                "box ann"
            } else {
                "box"
            };
            let bw = 7.2 * label.chars().count() as f64 + 12.0;
            let _ = write!(
                s,
                "<rect class=\"{class}\" x=\"{:.1}\" y=\"{:.1}\" width=\"{bw:.1}\" height=\"20\" rx=\"3\"/><text x=\"{cx:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
                cx - bw / 2.0,
                yq - 10.0,
                yq + 4.0,
                esc(label)
            );
        }
    }
    s.push_str("</svg>");
    Ok(s)
}

/* -- Detector slices ------------------------------------------------------- */

/// A detector slice: each detector's (or observable's) name, and its Pauli on each qubit.
pub type Slice = Vec<(String, Vec<(u32, char)>)>;

/// What each detector (and observable, as `L`k) compares at the moment after `tick` TICKs:
/// per detector, the Pauli on each qubit of its support. Detectors with nothing to compare at
/// that moment (made later, or already read) are left out. Walks the circuit unrolled, so the
/// circuit must fit the unrolling limit.
pub fn detector_slice(circuit: &Circuit, tick: u64) -> Result<Slice, String> {
    let res = circuit.resolve()?;
    let nd = res.detectors.len();
    // Target ids: detectors, then observables at nd + k.
    let mut rec_sym: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); res.num_measurements];
    for (d, recs) in res.detectors.iter().enumerate() {
        for &m in recs {
            toggle(&mut rec_sym[m], d);
        }
    }
    for (o, recs) in res.observables.iter().enumerate() {
        for &m in recs {
            toggle(&mut rec_sym[m], nd + o);
        }
    }
    let total_ticks = res
        .instrs
        .iter()
        .filter(|i| matches!(i, Instr::Tick))
        .count() as u64;
    if tick > total_ticks {
        return Err(format!(
            "the circuit has {total_ticks} TICKs; there is no moment after {tick}"
        ));
    }
    let nq = res.num_qubits;
    let (mut sx, mut sz) = (vec![BTreeSet::new(); nq], vec![BTreeSet::new(); nq]);
    let mut m = res.num_measurements;
    let mut ticks_after = total_ticks;
    for ins in res.instrs.iter().rev() {
        if ticks_after == tick && matches!(ins, Instr::Tick) {
            break;
        }
        match ins {
            Instr::Tick => ticks_after -= 1,
            Instr::Reset { qubits, .. } => {
                for &q in qubits.iter().rev() {
                    sx[q as usize].clear();
                    sz[q as usize].clear();
                }
            }
            Instr::Measure {
                basis,
                reset,
                qubits,
                ..
            } => {
                for &q in qubits.iter().rev() {
                    m -= 1;
                    let q = q as usize;
                    if *reset {
                        sx[q].clear();
                        sz[q].clear();
                    }
                    let r = rec_sym[m].clone();
                    match basis {
                        Basis::Z => xor_all(&mut sx[q], &r),
                        Basis::X => xor_all(&mut sz[q], &r),
                    }
                }
            }
            Instr::Pad { values, .. } => m -= values.len(),
            Instr::Heralded { qubits, .. } => m -= qubits.len(),
            Instr::H(qs) => qs
                .iter()
                .rev()
                .for_each(|&q| std::mem::swap(&mut sx[q as usize], &mut sz[q as usize])),
            Instr::S(qs) => {
                for &q in qs.iter().rev() {
                    let z = sz[q as usize].clone();
                    xor_all(&mut sx[q as usize], &z);
                }
            }
            Instr::Cx(pairs) => {
                for &(c, t) in pairs.iter().rev() {
                    let (xt, zc) = (sx[t as usize].clone(), sz[c as usize].clone());
                    xor_all(&mut sx[c as usize], &xt);
                    xor_all(&mut sz[t as usize], &zc);
                }
            }
            Instr::Cz(pairs) => {
                for &(a, b) in pairs.iter().rev() {
                    let (za, zb) = (sz[a as usize].clone(), sz[b as usize].clone());
                    xor_all(&mut sx[a as usize], &zb);
                    xor_all(&mut sx[b as usize], &za);
                }
            }
            Instr::Feedback {
                pauli,
                control: Control::Rec(k),
                qubit,
            } => {
                let q = *qubit as usize;
                let mut sym = BTreeSet::new();
                if pauli & 1 != 0 {
                    xor_all(&mut sym, &sx[q]);
                }
                if pauli & 2 != 0 {
                    xor_all(&mut sym, &sz[q]);
                }
                xor_all(&mut rec_sym[m - *k as usize], &sym);
            }
            Instr::Observable { index, paulis, .. } => {
                for &(q, p, _) in paulis {
                    if p & 1 != 0 {
                        toggle(&mut sz[q as usize], nd + *index as usize);
                    }
                    if p & 2 != 0 {
                        toggle(&mut sx[q as usize], nd + *index as usize);
                    }
                }
            }
            _ => {}
        }
    }
    let mut by_target: std::collections::BTreeMap<usize, Vec<(u32, char)>> =
        std::collections::BTreeMap::new();
    for q in 0..nq {
        let ids: BTreeSet<usize> = sx[q].union(&sz[q]).copied().collect();
        for id in ids {
            // An X error flips it: it has a Z component here; a Z error: an X component.
            let p = match (sx[q].contains(&id), sz[q].contains(&id)) {
                (true, true) => 'Y',
                (true, false) => 'Z',
                _ => 'X',
            };
            by_target.entry(id).or_default().push((q as u32, p));
        }
    }
    Ok(by_target
        .into_iter()
        .map(|(id, v)| {
            (
                if id < nd {
                    format!("D{id}")
                } else {
                    format!("L{}", id - nd)
                },
                v,
            )
        })
        .collect())
}

fn toggle(s: &mut BTreeSet<usize>, x: usize) {
    if !s.remove(&x) {
        s.insert(x);
    }
}

fn xor_all(s: &mut BTreeSet<usize>, other: &BTreeSet<usize>) {
    for &x in other {
        toggle(s, x);
    }
}

/// The detector slice as text: a line per detector, its Paulis.
pub fn detslice_text(circuit: &Circuit, tick: u64) -> Result<String, String> {
    let mut out = String::new();
    for (name, support) in detector_slice(circuit, tick)? {
        let terms: Vec<String> = support.iter().map(|(q, p)| format!("{p}{q}")).collect();
        let _ = writeln!(out, "{name}: {}", terms.join(" "));
    }
    Ok(out)
}

/// The qubits' positions (their first two `QUBIT_COORDS`), or a row when they have none.
/// Where to draw each qubit: at its first two coordinates; a qubit the circuit uses but gives
/// no coordinates goes in a row under the rest; a qubit it never uses is not drawn.
fn qubit_positions(circuit: &Circuit, nq: usize) -> Vec<Option<(f64, f64)>> {
    let mut pos: Vec<Option<(f64, f64)>> = vec![None; nq];
    for ins in &circuit.flattened().instrs {
        let body: Vec<&Instr> = match ins {
            Instr::Gate { body, .. } => body.iter().collect(),
            other => vec![other],
        };
        for i in body {
            if let Instr::QubitCoords { coords, qubits } = i {
                for &q in qubits {
                    if (q as usize) < nq && coords.len() >= 2 {
                        pos[q as usize] = Some((coords[0], coords[1]));
                    }
                }
            }
        }
    }
    let mut used = vec![true; nq];
    if let Ok(l) = layout(circuit) {
        used = vec![false; nq];
        for (q, _) in l.ops.iter().flat_map(|op| &op.cells) {
            used[*q as usize] = true;
        }
    }
    let placed: Vec<(f64, f64)> = pos.iter().flatten().copied().collect();
    let minx = placed.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let (minx, below) = if placed.is_empty() {
        (0.0, 0.0)
    } else {
        (
            minx,
            placed.iter().map(|p| p.1).fold(f64::MIN, f64::max) + 1.5,
        )
    };
    let mut next = 0.0;
    for q in 0..nq {
        if pos[q].is_none() && used[q] {
            pos[q] = Some((minx + next, below));
            next += 1.0;
        }
    }
    pos
}

/// The detector slice as an SVG picture: the qubits where their coordinates put them, and each
/// detector's support drawn as a shape over them, coloured by its Pauli (X red, Z blue, Y
/// green).
pub fn detslice_svg(circuit: &Circuit, tick: u64) -> Result<String, String> {
    let slice = detector_slice(circuit, tick)?;
    let nq = crate::batch_sampler::Counts::of(&circuit.instrs)?.qubits;
    let pos = qubit_positions(circuit, nq);
    let (minx, maxx) = pos
        .iter()
        .flatten()
        .fold((0.0f64, 0.0f64), |(a, b), p| (a.min(p.0), b.max(p.0)));
    let (miny, maxy) = pos
        .iter()
        .flatten()
        .fold((0.0f64, 0.0f64), |(a, b), p| (a.min(p.1), b.max(p.1)));
    let scale = 48.0;
    let px = |p: (f64, f64)| (40.0 + (p.0 - minx) * scale, 40.0 + (p.1 - miny) * scale);
    let (w, h) = (
        80.0 + (maxx - minx) * scale,
        80.0 + (maxy - miny) * scale + 18.0,
    );
    let mut s = open_svg(w, h, 10);
    let colour = |support: &[(u32, char)]| {
        let has = |c: char| support.iter().any(|x| x.1 == c);
        match (has('X'), has('Z'), has('Y')) {
            (true, false, false) => "#c0392b",
            (false, true, false) => "#2c5f9e",
            (false, false, true) => "#2e8b57",
            _ => "#8e6bbf",
        }
    };
    for (name, support) in &slice {
        let pts: Vec<(f64, f64)> = support
            .iter()
            .map(|&(q, _)| px(pos[q as usize].unwrap_or_default()))
            .collect();
        let c = colour(support);
        let title = format!(
            "<title>{name}: {}</title>",
            support
                .iter()
                .map(|(q, p)| format!("{p}{q}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        match pts.len() {
            1 => {
                let _ = write!(s, "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"12\" fill=\"{c}\" fill-opacity=\"0.35\" stroke=\"{c}\">{title}</circle>", pts[0].0, pts[0].1);
            }
            2 => {
                let _ = write!(
                    s,
                    "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{c}\" stroke-width=\"10\" stroke-opacity=\"0.4\" stroke-linecap=\"round\">{title}</line>",
                    pts[0].0, pts[0].1, pts[1].0, pts[1].1
                );
            }
            _ => {
                let hull = convex_hull(&pts);
                let path: Vec<String> = hull
                    .iter()
                    .map(|p| format!("{:.1},{:.1}", p.0, p.1))
                    .collect();
                let _ = write!(s, "<polygon points=\"{}\" fill=\"{c}\" fill-opacity=\"0.35\" stroke=\"{c}\">{title}</polygon>", path.join(" "));
            }
        }
    }
    for (q, p) in pos.iter().enumerate() {
        let Some(p) = *p else { continue };
        let (x, y) = px(p);
        let _ = write!(s, "<circle class=\"dot\" cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"3\"/><text class=\"dim\" x=\"{:.1}\" y=\"{:.1}\">{q}</text>", x + 4.0, y - 4.0);
    }
    let _ = write!(
        s,
        "<text x=\"8\" y=\"{:.1}\" font-size=\"11\">after TICK {tick}: {} detectors</text></svg>",
        h - 6.0,
        slice.len()
    );
    Ok(s)
}

fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut p: Vec<(f64, f64)> = points.to_vec();
    p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &q in &p {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], q) <= 0.0 {
            lower.pop();
        }
        lower.push(q);
    }
    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &q in p.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], q) <= 0.0 {
            upper.pop();
        }
        upper.push(q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/* -- Matching graph ---------------------------------------------------------- */

/// A decomposed model's matching graph as an SVG picture: each detector where its coordinates
/// put it (space, with time drawn as a slant), each graph-like fault an edge, edges to the
/// boundary as stubs, and edges that flip an observable drawn heavier.
pub fn matchgraph_svg(dem: &crate::dem::Dem) -> Result<String, String> {
    let (edges, _) = crate::dem_decoder::merged_edges(dem)?;
    let nd = dem.num_detectors;
    // Space as a floor seen from above at a slant, time stacking the floors downward: each
    // round's layer its own band, and the edges between rounds short verticals between bands.
    let spread = dem
        .detector_coords
        .iter()
        .filter_map(|c| c.get(1))
        .fold((f64::MAX, f64::MIN), |(a, b), &y| (a.min(y), b.max(y)));
    let depth = if spread.1 >= spread.0 {
        0.45 * (spread.1 - spread.0) + 1.5
    } else {
        1.0
    };
    let coord = |d: usize| -> (f64, f64) {
        let c = dem
            .detector_coords
            .get(d)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        match c {
            [x, y, t, ..] => (x + 0.5 * y, 0.45 * y + depth * t),
            [x, t] => (*x, *t),
            [x] => (*x, 0.0),
            [] => (d as f64, 0.0),
        }
    };
    let pts: Vec<(f64, f64)> = (0..nd).map(coord).collect();
    let (minx, maxx) = pts
        .iter()
        .fold((0.0f64, 0.0f64), |(a, b), p| (a.min(p.0), b.max(p.0)));
    let (miny, maxy) = pts
        .iter()
        .fold((0.0f64, 0.0f64), |(a, b), p| (a.min(p.1), b.max(p.1)));
    let scale = (900.0 / (maxx - minx).max(maxy - miny).max(1.0)).min(56.0);
    let px = |p: (f64, f64)| (30.0 + (p.0 - minx) * scale, 30.0 + (p.1 - miny) * scale);
    let (w, h) = (60.0 + (maxx - minx) * scale, 60.0 + (maxy - miny) * scale);
    let mut s = open_svg(w, h, 9);
    for &(u, v, p, obs) in &edges {
        let a = px(pts[u as usize]);
        let (b, stub) = if (v as usize) < nd {
            (px(pts[v as usize]), false)
        } else {
            ((a.0 - 0.4 * scale, a.1 - 0.4 * scale), true)
        };
        let colour = if obs != 0 {
            " stroke=\"#c0392b\""
        } else {
            " class=\"edge\""
        };
        let width = if obs != 0 { 1.8 } else { 1.0 };
        let dash = if stub {
            " stroke-dasharray=\"3 2\""
        } else {
            ""
        };
        let _ = write!(
            s,
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"{colour} stroke-width=\"{width}\"{dash}><title>D{u}–{} p={p:.3e}</title></line>",
            a.0,
            a.1,
            b.0,
            b.1,
            if stub { "boundary".to_string() } else { format!("D{v}") }
        );
    }
    for (d, &p) in pts.iter().enumerate() {
        let (x, y) = px(p);
        let _ = write!(s, "<circle class=\"dot\" cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"3\"><title>D{d}</title></circle>");
    }
    s.push_str("</svg>");
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bell_pair_timeline() {
        let c = Circuit::parse("R 0 1\nH 0\nTICK\nCX 0 1\nTICK\nM 0 1\nDETECTOR rec[-1] rec[-2]")
            .unwrap();
        let t = timeline_text(&c).unwrap();
        assert!(
            t.contains("q0: -R-H-@-M:rec[0]-DETECTOR:D0=rec[1]*rec[0]-"),
            "{t}"
        );
        assert!(t.contains("q1: -R---X-M:rec[1]-"), "{t}");
        assert!(timeline_svg(&c).unwrap().starts_with("<svg"));
    }

    #[test]
    fn loops_are_drawn_once_and_numbered_through() {
        let c = Circuit::parse("R 0 1\nREPEAT 3 {\n CX 0 1\n MR 1\n DETECTOR rec[-1]\n}\nM 0\nDETECTOR rec[-1] rec[-2]").unwrap();
        let t = timeline_text(&c).unwrap();
        assert!(t.contains("REP 3"), "{t}");
        // After three passes: record 3, and detector 3, read the last pass's record 2 (qubit 1).
        assert!(
            t.contains("M:rec[3]") && t.contains("DETECTOR:D3=rec[3]*rec[2]"),
            "{t}"
        );
    }

    #[test]
    fn slices_of_a_repetition_code() {
        let c = Circuit::parse("R 0 1 2\nTICK\nCX 0 1\nTICK\nCX 2 1\nTICK\nMR 1\nDETECTOR rec[-1]")
            .unwrap();
        // Before the first CNOT, D0 compares Z0 Z1 Z2: the parity the ancilla will collect, and
        // the ancilla's own |0⟩.
        assert_eq!(
            detector_slice(&c, 1).unwrap(),
            vec![("D0".to_string(), vec![(0, 'Z'), (1, 'Z'), (2, 'Z')])]
        );
        assert_eq!(
            detector_slice(&c, 3).unwrap(),
            vec![("D0".to_string(), vec![(1, 'Z')])]
        );
        assert!(detector_slice(&c, 4).is_err());
    }
}
