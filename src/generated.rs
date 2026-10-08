//! Stim's generated memory experiments (`stim.Circuit.generated`), written exactly as Stim
//! writes them: the repetition code, the rotated and unrotated surface codes in either basis,
//! and the colour code's XYZ memory, with Stim's four noise parameters. The same qubits, the
//! same coordinates, the same gate order, the same detectors and observable, so a circuit made
//! here prints as Stim's character for character, and results from either engine compare.

use std::collections::{BTreeMap, BTreeSet};

/// Stim's four noise strengths; a zero leaves its channel out, as Stim does.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Noise {
    pub after_clifford_depolarization: f64,
    pub before_round_data_depolarization: f64,
    pub before_measure_flip_probability: f64,
    pub after_reset_flip_probability: f64,
}

/// The tasks Stim generates, by its names.
pub const TASKS: [&str; 6] = [
    "repetition_code:memory",
    "surface_code:rotated_memory_x",
    "surface_code:rotated_memory_z",
    "surface_code:unrotated_memory_x",
    "surface_code:unrotated_memory_z",
    "color_code:memory_xyz",
];

/// Stim's circuit text for `task` (one of `TASKS`), at `distance` over `rounds` rounds.
pub fn generate(task: &str, distance: u32, rounds: u64, noise: &Noise) -> Result<String, String> {
    for (name, p) in [
        ("after_clifford_depolarization", noise.after_clifford_depolarization),
        ("before_round_data_depolarization", noise.before_round_data_depolarization),
        ("before_measure_flip_probability", noise.before_measure_flip_probability),
        ("after_reset_flip_probability", noise.after_reset_flip_probability),
    ] {
        if !(0.0..=1.0).contains(&p) {
            return Err(format!("{name} must be a probability in [0, 1], not {p}"));
        }
    }
    if rounds < 1 {
        return Err("Need rounds >= 1.".into());
    }
    if distance < 2 {
        return Err("Need distance >= 2.".into());
    }
    if distance > 1001 {
        return Err(format!("distance {distance}: at most 1001"));
    }
    let g = Gen { noise: *noise };
    let lines = match task {
        "repetition_code:memory" => g.repetition(distance, rounds),
        "surface_code:rotated_memory_x" => g.rotated(distance, rounds, true),
        "surface_code:rotated_memory_z" => g.rotated(distance, rounds, false),
        "surface_code:unrotated_memory_x" => g.unrotated(distance, rounds, true),
        "surface_code:unrotated_memory_z" => g.unrotated(distance, rounds, false),
        "color_code:memory_xyz" => {
            if rounds < 2 {
                return Err("Need rounds >= 2.".into());
            }
            if distance.is_multiple_of(2) {
                return Err("Need an odd distance.".into());
            }
            g.color(distance, rounds)
        }
        other => return Err(format!("unknown code task '{other}': one of {}", TASKS.join(", "))),
    };
    Ok(lines.join("\n") + "\n")
}

/// A generated colour-code circuit flattened, each detector `(x, y, t)` given Chromobius's
/// colour and basis annotation as a 4th coordinate: `(y + t) mod 3`, the X basis with the
/// colour turning with the round, as Chromobius's authors annotate Stim's `memory_xyz`
/// circuits (clorco's `make_mxyz_color_code_from_stim_gen`). The circuit is flattened because
/// the annotation changes from round to round.
pub fn generate_annotated(task: &str, distance: u32, rounds: u64, noise: &Noise) -> Result<String, String> {
    if task != "color_code:memory_xyz" {
        return Err(format!("colour annotations are for the colour code (color_code:memory_xyz), not '{task}'"));
    }
    let text = generate(task, distance, rounds, noise)?;
    let mut flat = crate::circuit::Circuit::parse(&text)?.flattened();
    for ins in &mut flat.instrs {
        if let crate::circuit::Instr::Detector { coords, .. } = ins {
            let (y, t) = (coords[1], coords[2]);
            coords.push((y + t).rem_euclid(3.0));
        }
    }
    Ok(flat.to_stim())
}

/// The comment block `stim gen` writes before a generated circuit: its parameters, a picture of
/// its qubits (each cell padded to the widest label, rows from the top) and a legend.
pub fn header(task: &str, distance: u32, rounds: u64, noise: &Noise) -> Result<String, String> {
    generate(task, distance, rounds, noise)?;
    let (code, name) = task.split_once(':').unwrap_or((task, ""));
    let g = |x: f64| crate::dem_program::fmt_g(x, 6);
    let mut s = format!("# Generated {code} circuit.\n# task: {name}\n# rounds: {rounds}\n# distance: {distance}\n");
    s += &format!("# before_round_data_depolarization: {}\n", g(noise.before_round_data_depolarization));
    s += &format!("# before_measure_flip_probability: {}\n", g(noise.before_measure_flip_probability));
    s += &format!("# after_reset_flip_probability: {}\n", g(noise.after_reset_flip_probability));
    s += &format!("# after_clifford_depolarization: {}\n", g(noise.after_clifford_depolarization));
    let d = i64::from(distance);
    // (x, y, label) for every qubit, in the layout's own coordinates.
    let mut cells: Vec<(i64, i64, String)> = Vec::new();
    let legend: &[&str] = match task {
        "repetition_code:memory" => {
            for q in 0..2 * d - 1 {
                let kind = if q == 0 { 'L' } else if q % 2 == 1 { 'Z' } else { 'd' };
                cells.push((q, 0, format!("{kind}{q}")));
            }
            &["#     Z# = measurement qubit"]
        }
        "surface_code:rotated_memory_x" | "surface_code:rotated_memory_z" => {
            let memory_x = task.ends_with('x');
            let index = |x: i64, y: i64| x + (y / 2) * (2 * d + 1);
            for x in 0..d {
                for y in 0..d {
                    let observable = if memory_x { x == 0 } else { y == 0 };
                    let (cx, cy) = (2 * x + 1, 2 * y + 1);
                    cells.push((cx, cy, format!("{}{}", if observable { 'L' } else { 'd' }, index(cx, cy))));
                }
            }
            for x in 0..=d {
                for y in 0..=d {
                    let parity = x % 2 != y % 2;
                    if (x == 0 || x == d) && parity || (y == 0 || y == d) && !parity {
                        continue;
                    }
                    cells.push((2 * x, 2 * y, format!("{}{}", if parity { 'X' } else { 'Z' }, index(2 * x, 2 * y))));
                }
            }
            &["#     X# = measurement qubit (X stabilizer)", "#     Z# = measurement qubit (Z stabilizer)"]
        }
        "surface_code:unrotated_memory_x" | "surface_code:unrotated_memory_z" => {
            let memory_x = task.ends_with('x');
            let w = 2 * d - 1;
            for x in 0..w {
                for y in 0..w {
                    let kind = match (x % 2, y % 2) {
                        (a, b) if a == b => {
                            let observable = if memory_x { x == 0 } else { y == 0 };
                            if observable {
                                'L'
                            } else {
                                'd'
                            }
                        }
                        (1, _) => 'X',
                        _ => 'Z',
                    };
                    cells.push((x, y, format!("{kind}{}", x + y * w)));
                }
            }
            &["#     X# = measurement qubit (X stabilizer)", "#     Z# = measurement qubit (Z stabilizer)"]
        }
        _ => {
            let w = d + (d - 1) / 2;
            let mut q = 0;
            for y in 0..w {
                for i in 0..w - y {
                    let kind = if (i + 2 * y) % 3 == 2 {
                        ['G', 'B', 'R'][(y % 3) as usize]
                    } else if y == 0 {
                        'L'
                    } else {
                        'd'
                    };
                    cells.push((y + 2 * i, y, format!("{kind}{q}")));
                    q += 1;
                }
            }
            &["#     R# = measurement qubit (red hex)", "#     G# = measurement qubit (green hex)", "#     B# = measurement qubit (blue hex)"]
        }
    };
    let width = cells.iter().map(|c| c.2.len()).max().unwrap_or(0);
    let rows = cells.iter().map(|c| c.1).max().unwrap_or(0) + 1;
    let mut grid: Vec<Vec<String>> = vec![Vec::new(); rows as usize];
    for (x, y, label) in cells {
        let row = &mut grid[y as usize];
        if row.len() <= x as usize {
            row.resize(x as usize + 1, String::new());
        }
        row[x as usize] = label;
    }
    s += "# layout:\n";
    for row in grid.iter().rev() {
        s.push('#');
        for entry in row {
            s += &format!(" {entry:<width$}");
        }
        s.push('\n');
    }
    s += "# Legend:\n#     d# = data qubit\n#     L# = data qubit with logical observable crossing\n";
    for line in legend {
        s += line;
        s.push('\n');
    }
    Ok(s)
}

fn list(targets: &[u32]) -> String {
    targets.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ")
}

fn op(name: &str, targets: &[u32]) -> String {
    format!("{name} {}", list(targets))
}

fn op_p(name: &str, p: f64, targets: &[u32]) -> String {
    format!("{name}({p}) {}", list(targets))
}

fn coords(c: &[f64]) -> String {
    c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
}

/// Appends an instruction as Stim's `safe_append` does: joined to the one before when it is the
/// same gate with the same arguments.
fn emit(c: &mut Vec<String>, line: String) {
    if let (Some(last), Some((head, targets))) = (c.last_mut(), line.split_once(' ')) {
        let annotation = ["DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "REPEAT"].iter().any(|n| head.starts_with(n));
        if !annotation && last.split_once(' ').map(|x| x.0) == Some(head) {
            last.push(' ');
            last.push_str(targets);
            return;
        }
    }
    c.push(line);
}

/// An annotation with coordinates and lookbacks (`rec[-k]`).
fn detector(at: &[f64], recs: &[usize]) -> String {
    let r: Vec<String> = recs.iter().map(|k| format!("rec[-{k}]")).collect();
    format!("DETECTOR({}) {}", coords(at), r.join(" "))
}

/// Stim's `Circuit * n`: nothing for 0, the body itself for 1, a loop otherwise.
fn repeat(body: &[String], n: u64) -> Vec<String> {
    match n {
        0 => Vec::new(),
        1 => body.to_vec(),
        _ => {
            let mut out = vec![format!("REPEAT {n} {{")];
            out.extend(body.iter().map(|l| format!("    {l}")));
            out.push("}".into());
            out
        }
    }
}

struct Gen {
    noise: Noise,
}

impl Gen {
    fn begin_round(&self, c: &mut Vec<String>, data: &[u32]) {
        c.push("TICK".into());
        if self.noise.before_round_data_depolarization > 0.0 {
            emit(c, op_p("DEPOLARIZE1", self.noise.before_round_data_depolarization, data));
        }
    }

    fn unitary(&self, c: &mut Vec<String>, name: &str, targets: &[u32], two: bool) {
        emit(c, op(name, targets));
        if self.noise.after_clifford_depolarization > 0.0 {
            emit(c, op_p(if two { "DEPOLARIZE2" } else { "DEPOLARIZE1" }, self.noise.after_clifford_depolarization, targets));
        }
    }

    fn flip(basis: char) -> &'static str {
        if basis == 'X' {
            "Z_ERROR"
        } else {
            "X_ERROR"
        }
    }

    fn named(prefix: &str, basis: char) -> String {
        if basis == 'Z' {
            prefix.to_string()
        } else {
            format!("{prefix}{basis}")
        }
    }

    fn reset(&self, c: &mut Vec<String>, targets: &[u32], basis: char) {
        emit(c, op(&Self::named("R", basis), targets));
        if self.noise.after_reset_flip_probability > 0.0 {
            emit(c, op_p(Self::flip(basis), self.noise.after_reset_flip_probability, targets));
        }
    }

    fn measure(&self, c: &mut Vec<String>, targets: &[u32], basis: char) {
        if self.noise.before_measure_flip_probability > 0.0 {
            emit(c, op_p(Self::flip(basis), self.noise.before_measure_flip_probability, targets));
        }
        emit(c, op(&Self::named("M", basis), targets));
    }

    fn measure_reset(&self, c: &mut Vec<String>, targets: &[u32]) {
        if self.noise.before_measure_flip_probability > 0.0 {
            emit(c, op_p("X_ERROR", self.noise.before_measure_flip_probability, targets));
        }
        emit(c, op("MR", targets));
        if self.noise.after_reset_flip_probability > 0.0 {
            emit(c, op_p("X_ERROR", self.noise.after_reset_flip_probability, targets));
        }
    }

    fn repetition(&self, d: u32, rounds: u64) -> Vec<String> {
        let n = 2 * d - 1;
        let data: Vec<u32> = (0..n).step_by(2).collect();
        let meas: Vec<u32> = (1..n).step_by(2).collect();
        let first: Vec<u32> = meas.iter().flat_map(|&m| [m - 1, m]).collect();
        let second: Vec<u32> = meas.iter().flat_map(|&m| [m + 1, m]).collect();
        let mut cycle = Vec::new();
        self.begin_round(&mut cycle, &data);
        self.unitary(&mut cycle, "CX", &first, true);
        cycle.push("TICK".into());
        self.unitary(&mut cycle, "CX", &second, true);
        cycle.push("TICK".into());
        self.measure_reset(&mut cycle, &meas);
        let nm = meas.len();
        let mut out = Vec::new();
        self.reset(&mut out, &(0..n).collect::<Vec<_>>(), 'Z');
        out.extend(cycle.iter().cloned());
        for (k, &m) in meas.iter().enumerate() {
            out.push(detector(&[m as f64, 0.0], &[nm - k]));
        }
        let mut body = cycle;
        body.push("SHIFT_COORDS(0, 1)".into());
        for (k, &m) in meas.iter().enumerate() {
            body.push(detector(&[m as f64, 0.0], &[nm - k, 2 * nm - k]));
        }
        out.extend(repeat(&body, rounds - 1));
        self.measure(&mut out, &data, 'Z');
        let nd = data.len();
        for (k, &m) in meas.iter().enumerate() {
            // Data qubits m − 1 and m + 1 are data k and k + 1.
            out.push(detector(&[m as f64, 1.0], &[nd - k - 1, nd - k, nd + nm - k]));
        }
        out.push("OBSERVABLE_INCLUDE(0) rec[-1]".into());
        out
    }

    fn rotated(&self, d: u32, rounds: u64, memory_x: bool) -> Vec<String> {
        let d = d as i64;
        let mut data = BTreeSet::new();
        let (mut xm, mut zm) = (BTreeSet::new(), BTreeSet::new());
        for x in 0..d {
            for y in 0..d {
                data.insert((2 * x + 1, 2 * y + 1));
            }
        }
        for x in 0..=d {
            for y in 0..=d {
                let parity = x % 2 != y % 2;
                if (x == 0 || x == d) && parity || (y == 0 || y == d) && !parity {
                    continue;
                }
                if parity {
                    xm.insert((2 * x, 2 * y));
                } else {
                    zm.insert((2 * x, 2 * y));
                }
            }
        }
        let index = |(x, y): (i64, i64)| (x + (y / 2) * (2 * d + 1)) as u32;
        let x_obs: Vec<(i64, i64)> = (0..d).map(|y| (1, 2 * y + 1)).collect();
        let z_obs: Vec<(i64, i64)> = (0..d).map(|x| (2 * x + 1, 1)).collect();
        let surface = Surface {
            data,
            xm,
            zm,
            x_order: [(1, 1), (-1, 1), (1, -1), (-1, -1)],
            z_order: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            observable: if memory_x { x_obs } else { z_obs },
            memory_x,
        };
        self.surface(&surface, &index, rounds)
    }

    fn unrotated(&self, d: u32, rounds: u64, memory_x: bool) -> Vec<String> {
        let w = 2 * d as i64 - 1;
        let mut data = BTreeSet::new();
        let (mut xm, mut zm) = (BTreeSet::new(), BTreeSet::new());
        for x in 0..w {
            for y in 0..w {
                match (x % 2, y % 2) {
                    (a, b) if a == b => data.insert((x, y)),
                    (1, _) => xm.insert((x, y)),
                    _ => zm.insert((x, y)),
                };
            }
        }
        let index = |(x, y): (i64, i64)| (x + y * w) as u32;
        let x_obs: Vec<(i64, i64)> = (0..d as i64).map(|y| (0, 2 * y)).collect();
        let z_obs: Vec<(i64, i64)> = (0..d as i64).map(|x| (2 * x, 0)).collect();
        let order = [(1, 0), (0, 1), (0, -1), (-1, 0)];
        let surface = Surface { data, xm, zm, x_order: order, z_order: order, observable: if memory_x { x_obs } else { z_obs }, memory_x };
        self.surface(&surface, &index, rounds)
    }

    /// Stim's `_finish_surface_code_circuit`, shared by both layouts.
    fn surface(&self, s: &Surface, index: &dyn Fn((i64, i64)) -> u32, rounds: u64) -> Vec<String> {
        let mut q2p: BTreeMap<u32, (i64, i64)> = BTreeMap::new();
        for &p in s.data.iter().chain(&s.xm).chain(&s.zm) {
            q2p.insert(index(p), p);
        }
        let sorted = |set: &BTreeSet<(i64, i64)>| {
            let mut v: Vec<u32> = set.iter().map(|&p| index(p)).collect();
            v.sort_unstable();
            v
        };
        let data = sorted(&s.data);
        let xq = sorted(&s.xm);
        let mut meas: Vec<u32> = xq.iter().copied().chain(sorted(&s.zm)).collect();
        meas.sort_unstable();
        let data_order: BTreeMap<(i64, i64), usize> = data.iter().enumerate().map(|(k, q)| (q2p[q], k)).collect();
        let meas_order: BTreeMap<(i64, i64), usize> = meas.iter().enumerate().map(|(k, q)| (q2p[q], k)).collect();
        let (nd, nm) = (data.len(), meas.len());
        let add = |a: (i64, i64), b: (i64, i64)| (a.0 + b.0, a.1 + b.1);

        let mut cycle = Vec::new();
        self.begin_round(&mut cycle, &data);
        self.unitary(&mut cycle, "H", &xq, false);
        for k in 0..4 {
            let mut targets = Vec::new();
            for &m in &s.xm {
                let q = add(m, s.x_order[k]);
                if s.data.contains(&q) {
                    targets.extend([index(m), index(q)]);
                }
            }
            for &m in &s.zm {
                let q = add(m, s.z_order[k]);
                if s.data.contains(&q) {
                    targets.extend([index(q), index(m)]);
                }
            }
            cycle.push("TICK".into());
            self.unitary(&mut cycle, "CX", &targets, true);
        }
        cycle.push("TICK".into());
        self.unitary(&mut cycle, "H", &xq, false);
        cycle.push("TICK".into());
        self.measure_reset(&mut cycle, &meas);

        let chosen = if s.memory_x { &s.xm } else { &s.zm };
        let basis = if s.memory_x { 'X' } else { 'Z' };
        let mut out = Vec::new();
        for (q, p) in &q2p {
            out.push(format!("QUBIT_COORDS({}) {q}", coords(&[p.0 as f64, p.1 as f64])));
        }
        self.reset(&mut out, &data, basis);
        self.reset(&mut out, &meas, 'Z');
        out.extend(cycle.iter().cloned());
        for &m in chosen {
            out.push(detector(&[m.0 as f64, m.1 as f64, 0.0], &[nm - meas_order[&m]]));
        }
        let mut body = cycle;
        body.push("SHIFT_COORDS(0, 0, 1)".into());
        for q in &meas {
            let m = q2p[q];
            let k = nm - meas_order[&m];
            body.push(detector(&[m.0 as f64, m.1 as f64, 0.0], &[k, k + nm]));
        }
        out.extend(repeat(&body, rounds - 1));
        self.measure(&mut out, &data, basis);
        for &m in chosen {
            let mut recs: Vec<usize> = s.z_order.iter().map(|&o| add(m, o)).filter(|q| s.data.contains(q)).map(|q| nd - data_order[&q]).collect();
            recs.push(nd + nm - meas_order[&m]);
            recs.sort_unstable();
            out.push(detector(&[m.0 as f64, m.1 as f64, 1.0], &recs));
        }
        let mut obs: Vec<usize> = s.observable.iter().map(|q| nd - data_order[q]).collect();
        obs.sort_unstable();
        out.push(format!("OBSERVABLE_INCLUDE(0) {}", obs.iter().map(|k| format!("rec[-{k}]")).collect::<Vec<_>>().join(" ")));
        out
    }

    /// The colour code on a triangle of hexagons, every check measured in turn in the Z, X and
    /// Y bases by rotating the data with `C_XYZ` each round. Coordinates are kept doubled in x
    /// (the rows are offset by halves).
    fn color(&self, d: u32, rounds: u64) -> Vec<String> {
        let w = (d + (d - 1) / 2) as i64;
        let mut q2p: Vec<(i64, i64)> = Vec::new();
        let (mut data, mut meas) = (BTreeMap::new(), BTreeMap::new());
        for y in 0..w {
            for i in 0..w - y {
                let p = (y + 2 * i, y);
                let q = q2p.len() as u32;
                if (i + 2 * y) % 3 == 2 {
                    meas.insert(p, q);
                } else {
                    data.insert(p, q);
                }
                q2p.push(p);
            }
        }
        let mut dq: Vec<u32> = data.values().copied().collect();
        dq.sort_unstable();
        let mut mq: Vec<u32> = meas.values().copied().collect();
        mq.sort_unstable();
        let (nd, nm) = (dq.len(), mq.len());
        let data_order: BTreeMap<u32, usize> = dq.iter().enumerate().map(|(k, &q)| (q, k)).collect();
        let meas_order: BTreeMap<u32, usize> = mq.iter().enumerate().map(|(k, &q)| (q, k)).collect();
        const ORDER: [(i64, i64); 6] = [(2, 0), (1, 1), (1, -1), (-2, 0), (-1, 1), (-1, -1)];
        let at = |p: (i64, i64), t: f64| [p.0 as f64 / 2.0, p.1 as f64, t];

        let mut cycle = Vec::new();
        self.begin_round(&mut cycle, &dq);
        self.unitary(&mut cycle, "C_XYZ", &dq, false);
        for o in ORDER {
            let mut targets = Vec::new();
            for (&m, &mqi) in &meas {
                if let Some(&q) = data.get(&(m.0 + o.0, m.1 + o.1)) {
                    targets.extend([q, mqi]);
                }
            }
            cycle.push("TICK".into());
            self.unitary(&mut cycle, "CX", &targets, true);
        }
        cycle.push("TICK".into());
        self.measure_reset(&mut cycle, &mq);

        let mut out = Vec::new();
        for (q, p) in q2p.iter().enumerate() {
            out.push(format!("QUBIT_COORDS({}) {q}", coords(&at(*p, 0.0)[..2])));
        }
        self.reset(&mut out, &(0..q2p.len() as u32).collect::<Vec<_>>(), 'Z');
        out.extend(repeat(&cycle, 2));
        for &q in &mq {
            let k = nm - meas_order[&q];
            out.push(detector(&at(q2p[q as usize], 0.0), &[k, k + nm]));
        }
        let mut body = cycle;
        body.push("SHIFT_COORDS(0, 0, 1)".into());
        for &q in &mq {
            let k = nm - meas_order[&q];
            body.push(detector(&at(q2p[q as usize], 0.0), &[k, k + nm, k + 2 * nm]));
        }
        out.extend(repeat(&body, rounds - 2));
        let basis = ['Z', 'X', 'Y'][(rounds % 3) as usize];
        self.measure(&mut out, &dq, basis);
        for &q in &mq {
            let m = q2p[q as usize];
            let mut recs: Vec<usize> = ORDER.iter().filter_map(|o| data.get(&(m.0 + o.0, m.1 + o.1))).map(|d| nd - data_order[d]).collect();
            let k = nd + nm - meas_order[&q];
            if basis != 'X' {
                recs.push(k);
            }
            if basis != 'Z' {
                recs.push(k + nm);
            }
            recs.sort_unstable();
            out.push(detector(&at(m, 1.0), &recs));
        }
        let mut obs: Vec<usize> = data.iter().filter(|(p, _)| p.1 == 0).map(|(_, q)| nd - data_order[q]).collect();
        obs.sort_unstable();
        out.push(format!("OBSERVABLE_INCLUDE(0) {}", obs.iter().map(|k| format!("rec[-{k}]")).collect::<Vec<_>>().join(" ")));
        out
    }
}

struct Surface {
    data: BTreeSet<(i64, i64)>,
    xm: BTreeSet<(i64, i64)>,
    zm: BTreeSet<(i64, i64)>,
    x_order: [(i64, i64); 4],
    z_order: [(i64, i64); 4],
    observable: Vec<(i64, i64)>,
    memory_x: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_parses_and_has_its_detectors() {
        let noise = Noise { after_clifford_depolarization: 0.001, ..Noise::default() };
        for task in TASKS {
            let text = generate(task, 3, 3, &noise).unwrap();
            let c = crate::circuit::Circuit::parse(&text).unwrap();
            assert_eq!(c.to_stim(), text, "{task}");
        }
    }

    #[test]
    fn bad_parameters_are_refused() {
        let n = Noise::default();
        assert!(generate("color_code:memory_xyz", 3, 1, &n).is_err());
        assert!(generate("color_code:memory_xyz", 4, 3, &n).is_err());
        assert!(generate("surface_code:rotated_memory_z", 1, 3, &n).is_err());
        assert!(generate("surface_code:nope", 3, 3, &n).is_err());
        assert!(generate("repetition_code:memory", 3, 3, &Noise { after_reset_flip_probability: 2.0, ..n }).is_err());
    }
}
