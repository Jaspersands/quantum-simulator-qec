//! A detector error model as Stim writes one: errors, detector and observable declarations,
//! `shift_detectors`, and `repeat` blocks, kept as a program rather than unrolled.
//!
//! WHY THIS EXISTS
//! ---------------
//! A long memory's model repeats itself round after round. Stim folds the repetition into a
//! `repeat` block, so a d = 11 memory of 10,000 rounds is a few hundred lines; unrolled it is
//! millions of faults. This holds the folded form, reads and prints it, and unrolls it (into
//! `Dem`, the flat model the decoders take) only when a decoder needs it.

use std::fmt::Write as _;

use crate::circuit::{split_instruction, strip_comment};
use crate::dem::{xor_prob, Dem, Mechanism, Piece};

/// An observable's target: its index with the top bit set, so that within a sorted list of
/// targets the detectors come first, as in Stim.
pub const OBS: u64 = 1 << 63;

/// The largest detector index a flat model may name.
pub const MAX_DETECTOR: u64 = (1 << 24) - 1;

/// The most errors and declarations a model may unroll into.
pub const MAX_UNROLLED_LINES: u64 = 1 << 24;

#[derive(Clone, Debug, PartialEq)]
pub enum DemInstr {
    /// `error[tag](p) D0 D1 ^ D2 L0`: each piece's targets sorted (detectors relative to the
    /// current shift), pieces in order.
    Error { p: f64, pieces: Vec<Vec<u64>>, tag: String },
    /// `detector[tag](coords) D3`: detectors, relative.
    Detector { coords: Vec<f64>, targets: Vec<u64>, tag: String },
    /// `logical_observable[tag] L1`.
    Observable { targets: Vec<u32>, tag: String },
    /// `shift_detectors[tag](coords) n`.
    Shift { coords: Vec<f64>, by: u64, tag: String },
    /// `repeat[tag] count { body }`.
    Repeat { count: u64, body: Vec<DemInstr>, tag: String },
}

/// A model as a program.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DemProgram {
    pub instrs: Vec<DemInstr>,
}

/// What a program holds, counted through its loops without unrolling them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// One more than the largest detector index named, saturating.
    pub num_detectors: u64,
    /// One more than the largest observable index named.
    pub num_observables: usize,
    /// Error instructions, counted through loops, saturating.
    pub num_errors: u64,
}

/// A body's counts: its largest detector index plus one relative to its start (0 for none),
/// how far it shifts, its errors, and its observables.
#[derive(Clone, Copy, Default)]
struct BlockStats {
    end: u64,
    shift: u64,
    errors: u64,
    observables: usize,
}

fn block_stats(instrs: &[DemInstr]) -> BlockStats {
    let mut s = BlockStats::default();
    for ins in instrs {
        match ins {
            DemInstr::Error { pieces, .. } => {
                s.errors = s.errors.saturating_add(1);
                for t in pieces.iter().flatten() {
                    if t & OBS == 0 {
                        s.end = s.end.max(s.shift.saturating_add(t + 1));
                    } else {
                        s.observables = s.observables.max((t & !OBS) as usize + 1);
                    }
                }
            }
            DemInstr::Detector { targets, .. } => {
                for t in targets {
                    s.end = s.end.max(s.shift.saturating_add(t + 1));
                }
            }
            DemInstr::Observable { targets, .. } => {
                for &t in targets {
                    s.observables = s.observables.max(t as usize + 1);
                }
            }
            DemInstr::Shift { by, .. } => s.shift = s.shift.saturating_add(*by),
            DemInstr::Repeat { count, body, .. } => {
                if *count == 0 {
                    continue;
                }
                let b = block_stats(body);
                if b.end > 0 {
                    let last = s.shift.saturating_add(b.shift.saturating_mul(count - 1));
                    s.end = s.end.max(last.saturating_add(b.end));
                }
                s.shift = s.shift.saturating_add(b.shift.saturating_mul(*count));
                s.errors = s.errors.saturating_add(b.errors.saturating_mul(*count));
                s.observables = s.observables.max(b.observables);
            }
        }
    }
    s
}

/// The significant digits Stim prints a model's numbers to on this machine: its stream's
/// precision is `numeric_limits<long double>::digits10 + 1`, and `long double` is a double on
/// Windows and Apple silicon (16), x87's 80-bit extended on other x86-64 (19), and IEEE quad
/// on other ARM64 (34).
pub const STIM_PRECISION: usize = if cfg!(all(target_arch = "x86_64", not(target_os = "windows"))) {
    19
} else if cfg!(all(target_arch = "aarch64", not(any(target_os = "windows", target_vendor = "apple")))) {
    34
} else {
    16
};

/// A number as Stim prints one in a model on this machine (`fmt_g` at `STIM_PRECISION`).
pub fn fmt_g16(x: f64) -> String {
    fmt_g(x, STIM_PRECISION)
}

/// C++'s default stream format at precision `digits` (`%.{digits}g`): that many significant
/// digits, trailing zeros dropped, and an exponent below 10⁻⁴ or from 10^digits.
pub fn fmt_g(x: f64, digits: usize) -> String {
    if x == 0.0 || !x.is_finite() {
        return if x == 0.0 { "0".into() } else { format!("{x}") };
    }
    let sci = format!("{x:.*e}", digits - 1);
    let (mantissa, exp) = sci.split_once('e').expect("an exponent");
    let exp: i32 = exp.parse().expect("an integer exponent");
    let trim = |t: &str| -> String {
        if t.contains('.') {
            t.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            t.to_string()
        }
    };
    if exp < -4 || exp >= digits as i32 {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa), exp.abs())
    } else {
        trim(&format!("{x:.*}", (digits as i32 - 1 - exp) as usize))
    }
}

fn g16_args(args: &[f64]) -> String {
    args.iter().map(|&v| fmt_g16(v)).collect::<Vec<_>>().join(", ")
}

fn tagged(tag: &str) -> String {
    if tag.is_empty() {
        String::new()
    } else {
        format!("[{tag}]")
    }
}

fn with_args(name: &str, tag: &str, args: &[f64]) -> String {
    let mut s = format!("{name}{}", tagged(tag));
    if !args.is_empty() {
        let _ = write!(s, "({})", g16_args(args));
    }
    s
}

fn push_target(s: &mut String, t: u64) {
    if t & OBS == 0 {
        let _ = write!(s, " D{t}");
    } else {
        let _ = write!(s, " L{}", t & !OBS);
    }
}

/// XOR `t` into a sorted target list.
fn toggle(v: &mut Vec<u64>, t: u64) {
    match v.binary_search(&t) {
        Ok(i) => {
            v.remove(i);
        }
        Err(i) => v.insert(i, t),
    }
}

impl DemProgram {
    pub fn stats(&self) -> Stats {
        let b = block_stats(&self.instrs);
        Stats { num_detectors: b.end, num_observables: b.observables, num_errors: b.errors }
    }

    /// Stim's text.
    pub fn to_stim(&self) -> String {
        let mut s = String::new();
        emit(&self.instrs, "", &mut s);
        s
    }

    pub fn parse(text: &str) -> Result<DemProgram, String> {
        let lines: Vec<&str> = text.lines().collect();
        let mut pos = 0usize;
        let instrs = parse_block(&lines, &mut pos, false)?;
        Ok(DemProgram { instrs })
    }

    /// The program unrolled: no `repeat` and no `shift_detectors`, detectors absolute and
    /// coordinates shifted, in the program's order. Refuses past `MAX_UNROLLED_LINES` errors and
    /// declarations, or past detector `MAX_DETECTOR`.
    pub fn flattened(&self) -> Result<DemProgram, String> {
        let mut out = Vec::new();
        let mut st = Unroll { offset: 0, coords: Vec::new(), budget: MAX_UNROLLED_LINES };
        unroll(&self.instrs, &mut st, &mut |ins| out.push(ins))?;
        Ok(DemProgram { instrs: out })
    }

    /// Every detector's time (its last coordinate, shifts applied), by detector, walking the
    /// declarations alone: a model of millions of faults costs only its detectors. Refuses a
    /// detector without coordinates, or more than `MAX_DETECTOR` + 1 detectors.
    pub fn detector_times(&self) -> Result<Vec<f64>, String> {
        let n = self.stats().num_detectors;
        if n > MAX_DETECTOR + 1 {
            return Err(format!("the model has {n} detectors, past the largest index, {MAX_DETECTOR}"));
        }
        let mut times = vec![f64::NAN; n as usize];
        fn walk(instrs: &[DemInstr], offset: &mut u64, coords: &mut Vec<f64>, times: &mut [f64]) -> Result<(), String> {
            for ins in instrs {
                match ins {
                    DemInstr::Detector { coords: c, targets, .. } => {
                        let k = c.len().checked_sub(1).ok_or("a detector has no coordinates; windows need its time")?;
                        let t = c[k] + coords.get(k).copied().unwrap_or(0.0);
                        for &d in targets {
                            times[(d + *offset) as usize] = t;
                        }
                    }
                    DemInstr::Shift { coords: c, by, .. } => {
                        if coords.len() < c.len() {
                            coords.resize(c.len(), 0.0);
                        }
                        for (a, b) in coords.iter_mut().zip(c) {
                            *a += b;
                        }
                        *offset += by;
                    }
                    DemInstr::Repeat { count, body, .. } => {
                        // A body that declares nothing and shifts nothing changes nothing.
                        if body.iter().any(|i| !matches!(i, DemInstr::Error { .. } | DemInstr::Observable { .. })) {
                            for _ in 0..*count {
                                walk(body, offset, coords, times)?;
                            }
                        }
                    }
                    DemInstr::Error { .. } | DemInstr::Observable { .. } => {}
                }
            }
            Ok(())
        }
        walk(&self.instrs, &mut 0, &mut Vec::new(), &mut times)?;
        if let Some(d) = times.iter().position(|t| t.is_nan()) {
            return Err(format!("detector D{d} is not declared with coordinates; windows need its time"));
        }
        Ok(times)
    }

    /// The flat model the decoders take, in the program's order.
    pub fn to_dem(&self) -> Result<Dem, String> {
        let stats = self.stats();
        if stats.num_detectors > MAX_DETECTOR + 1 {
            return Err(format!("the model names detector {}, past the largest index, {MAX_DETECTOR}", stats.num_detectors - 1));
        }
        if stats.num_observables > 64 {
            return Err(format!(
                "the model has {} observables; this engine's decoders and model sampler take at most 64 (the text, counts, circuit sampling and measurement conversion take any number)",
                stats.num_observables
            ));
        }
        let mut dem = Dem { num_detectors: stats.num_detectors as usize, num_observables: stats.num_observables, ..Dem::default() };
        dem.detector_coords = vec![Vec::new(); dem.num_detectors];
        let mut st = Unroll { offset: 0, coords: Vec::new(), budget: MAX_UNROLLED_LINES };
        unroll(&self.instrs, &mut st, &mut |ins| match ins {
            DemInstr::Error { p, pieces, tag } => dem.mechanisms.push(mechanism(p, &pieces, tag)),
            DemInstr::Detector { coords, targets, tag } => {
                for d in targets {
                    let d = d as usize;
                    dem.detector_coords[d] = coords.clone();
                    if !tag.is_empty() {
                        if dem.detector_tags.len() <= d {
                            dem.detector_tags.resize(d + 1, String::new());
                        }
                        dem.detector_tags[d] = tag.clone();
                    }
                }
            }
            DemInstr::Observable { targets, tag } => {
                for l in targets {
                    let l = l as usize;
                    if !tag.is_empty() {
                        if dem.observable_tags.len() <= l {
                            dem.observable_tags.resize(l + 1, String::new());
                        }
                        dem.observable_tags[l] = tag.clone();
                    }
                }
            }
            DemInstr::Shift { .. } | DemInstr::Repeat { .. } => {}
        })?;
        Ok(dem)
    }

    /// The flat model of a model this engine built: `to_dem`, then faults that are the same in
    /// every way merged and all sorted, as the decoders have always been given it.
    pub fn to_dem_merged(&self) -> Result<Dem, String> {
        let mut dem = self.to_dem()?;
        let key = |m: &Mechanism| (m.detectors.clone(), m.observables, m.tag.clone(), m.pieces.clone());
        dem.mechanisms.sort_by(|a, b| {
            a.detectors
                .cmp(&b.detectors)
                .then(a.observables.cmp(&b.observables))
                .then(a.tag.cmp(&b.tag))
                .then(a.pieces.cmp(&b.pieces))
        });
        let mut merged: Vec<Mechanism> = Vec::with_capacity(dem.mechanisms.len());
        for m in dem.mechanisms {
            match merged.last_mut() {
                Some(last) if key(last) == key(&m) => last.p = xor_prob(last.p, m.p),
                _ => merged.push(m),
            }
        }
        dem.mechanisms = merged;
        Ok(dem)
    }
}

/// A flat model's mechanism from an error's absolute pieces, as the `.dem` reader has always
/// made one: the pieces' XOR, and the pieces kept when there are several or the one is
/// graph-like.
fn mechanism(p: f64, pieces: &[Vec<u64>], tag: String) -> Mechanism {
    let mut all = Vec::new();
    let mut kept = Vec::new();
    for piece in pieces {
        let (dets, obs) = split(piece);
        kept.push(Piece { detectors: dets, observables: obs });
        for &t in piece {
            toggle(&mut all, t);
        }
    }
    let (detectors, observables) = split(&all);
    let pieces = if kept.len() > 1 {
        kept.into_iter().filter(|p| !p.detectors.is_empty() || p.observables != 0).collect()
    } else if (1..=2).contains(&detectors.len()) {
        vec![Piece { detectors: detectors.clone(), observables }]
    } else {
        Vec::new()
    };
    Mechanism { p, detectors, observables, pieces, tag }
}

/// A sorted target list as detectors (`u32`) and an observable mask.
pub fn split(targets: &[u64]) -> (Vec<u32>, u64) {
    let mut dets = Vec::new();
    let mut obs = 0u64;
    for &t in targets {
        if t & OBS == 0 {
            dets.push(t as u32);
        } else {
            obs ^= 1u64 << (t & !OBS);
        }
    }
    (dets, obs)
}

struct Unroll {
    offset: u64,
    coords: Vec<f64>,
    budget: u64,
}

fn unroll(instrs: &[DemInstr], st: &mut Unroll, out: &mut dyn FnMut(DemInstr)) -> Result<(), String> {
    let absolute = |t: u64, offset: u64| -> Result<u64, String> {
        if t & OBS != 0 {
            return Ok(t);
        }
        t.checked_add(offset)
            .filter(|&d| d <= MAX_DETECTOR)
            .ok_or_else(|| format!("the model names a detector past the largest index, {MAX_DETECTOR}"))
    };
    let spend = |st: &mut Unroll| -> Result<(), String> {
        if st.budget == 0 {
            return Err(format!("the model unrolls past {MAX_UNROLLED_LINES} errors and declarations"));
        }
        st.budget -= 1;
        Ok(())
    };
    for ins in instrs {
        match ins {
            DemInstr::Error { p, pieces, tag } => {
                spend(st)?;
                let mut abs = Vec::with_capacity(pieces.len());
                for piece in pieces {
                    let mut v = piece.iter().map(|&t| absolute(t, st.offset)).collect::<Result<Vec<_>, _>>()?;
                    v.sort_unstable();
                    abs.push(v);
                }
                out(DemInstr::Error { p: *p, pieces: abs, tag: tag.clone() });
            }
            DemInstr::Detector { coords, targets, tag } => {
                spend(st)?;
                let coords = coords.iter().enumerate().map(|(i, v)| v + st.coords.get(i).copied().unwrap_or(0.0)).collect();
                let targets = targets.iter().map(|&t| absolute(t, st.offset)).collect::<Result<Vec<_>, _>>()?;
                out(DemInstr::Detector { coords, targets, tag: tag.clone() });
            }
            DemInstr::Observable { .. } => {
                spend(st)?;
                out(ins.clone());
            }
            DemInstr::Shift { coords, by, .. } => {
                if st.coords.len() < coords.len() {
                    st.coords.resize(coords.len(), 0.0);
                }
                for (a, b) in st.coords.iter_mut().zip(coords) {
                    *a += b;
                }
                st.offset = st.offset.checked_add(*by).filter(|&o| o <= MAX_DETECTOR + 1).ok_or_else(|| {
                    format!("shift_detectors moves past detector {MAX_DETECTOR}, the largest index")
                })?;
            }
            DemInstr::Repeat { count, body, .. } => {
                for _ in 0..*count {
                    // Every pass costs at least one unit, so an empty body cannot spin forever.
                    spend(st)?;
                    unroll(body, st, out)?;
                }
            }
        }
    }
    Ok(())
}

fn emit(instrs: &[DemInstr], indent: &str, s: &mut String) {
    for ins in instrs {
        match ins {
            DemInstr::Error { p, pieces, tag } => {
                let _ = write!(s, "{indent}{}", with_args("error", tag, &[*p]));
                for (k, piece) in pieces.iter().enumerate() {
                    if k > 0 {
                        s.push_str(" ^");
                    }
                    for &t in piece {
                        push_target(s, t);
                    }
                }
            }
            DemInstr::Detector { coords, targets, tag } => {
                let _ = write!(s, "{indent}{}", with_args("detector", tag, coords));
                for &t in targets {
                    push_target(s, t);
                }
            }
            DemInstr::Observable { targets, tag } => {
                let _ = write!(s, "{indent}logical_observable{}", tagged(tag));
                for &t in targets {
                    let _ = write!(s, " L{t}");
                }
            }
            DemInstr::Shift { coords, by, tag } => {
                let _ = write!(s, "{indent}{} {by}", with_args("shift_detectors", tag, coords));
            }
            DemInstr::Repeat { count, body, tag } => {
                let _ = writeln!(s, "{indent}repeat{} {count} {{", tagged(tag));
                emit(body, &format!("{indent}    "), s);
                let _ = write!(s, "{indent}}}");
            }
        }
        s.push('\n');
    }
}

fn parse_block(lines: &[&str], pos: &mut usize, nested: bool) -> Result<Vec<DemInstr>, String> {
    let mut out = Vec::new();
    while *pos < lines.len() {
        let lineno = *pos + 1;
        let line = strip_comment(lines[*pos]).trim();
        *pos += 1;
        if line.is_empty() {
            continue;
        }
        if line == "}" {
            if nested {
                return Ok(out);
            }
            return Err(format!("line {lineno}: unmatched '}}'"));
        }
        if let Some(rest) = line.strip_suffix('{') {
            let (name, tag, args, parts) = split_instruction(rest).map_err(|e| format!("line {lineno}: {e}"))?;
            if name != "REPEAT" {
                return Err(format!("line {lineno}: only repeat opens a block"));
            }
            let count: u64 = match (args.is_empty(), parts.as_slice()) {
                (true, [n]) => n.parse().ok(),
                _ => None,
            }
            .ok_or_else(|| format!("line {lineno}: repeat needs a count"))?;
            let body = parse_block(lines, pos, true)?;
            out.push(DemInstr::Repeat { count, body, tag: tag.to_string() });
            continue;
        }
        let (name, tag, args, tokens) = split_instruction(line).map_err(|e| format!("line {lineno}: {e}"))?;
        let tag = tag.to_string();
        let bad = |t: &str| format!("line {lineno}: bad target '{t}'");
        let detector = |t: &str| -> Result<u64, String> {
            t.strip_prefix('D').and_then(|d| d.parse::<u64>().ok()).filter(|&d| d <= MAX_DETECTOR).ok_or_else(|| bad(t))
        };
        let observable = |t: &str| -> Result<u32, String> {
            let l = t.strip_prefix('L').and_then(|l| l.parse::<u32>().ok()).ok_or_else(|| bad(t))?;
            Ok(l)
        };
        out.push(match name.as_str() {
            "ERROR" => {
                if args.len() != 1 || !(0.0..=1.0).contains(&args[0]) {
                    return Err(format!("line {lineno}: error takes one probability"));
                }
                let mut pieces = vec![Vec::new()];
                for t in &tokens {
                    if *t == "^" {
                        pieces.push(Vec::new());
                    } else if t.starts_with('D') {
                        toggle(pieces.last_mut().unwrap(), detector(t)?);
                    } else if t.starts_with('L') {
                        toggle(pieces.last_mut().unwrap(), OBS | u64::from(observable(t)?));
                    } else {
                        return Err(bad(t));
                    }
                }
                DemInstr::Error { p: args[0], pieces, tag }
            }
            "DETECTOR" => {
                let targets = tokens.iter().map(|t| detector(t)).collect::<Result<Vec<_>, _>>()?;
                DemInstr::Detector { coords: args, targets, tag }
            }
            "LOGICAL_OBSERVABLE" => {
                let targets = tokens.iter().map(|t| observable(t)).collect::<Result<Vec<_>, _>>()?;
                DemInstr::Observable { targets, tag }
            }
            "SHIFT_DETECTORS" => {
                let by = match tokens.as_slice() {
                    [] => 0,
                    [n] => n.parse::<u64>().ok().filter(|&n| n <= MAX_DETECTOR + 1).ok_or_else(|| bad(n))?,
                    _ => return Err(format!("line {lineno}: shift_detectors takes one count")),
                };
                DemInstr::Shift { coords: args, by, tag }
            }
            _ => return Err(format!("line {lineno}: unsupported instruction '{name}'")),
        });
    }
    if nested {
        return Err("unterminated repeat block".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::fmt_g;

    #[test]
    fn numbers_print_as_stims_do() {
        // At 19 digits, as Stim prints on x86-64 Linux, and 34, as on ARM64 Linux.
        assert_eq!(fmt_g(0.0025961611285238335, 19), "0.002596161128523833544");
        assert_eq!(fmt_g(0.5, 34), "0.5");
        assert_eq!(fmt_g(0.1, 19), "0.1000000000000000056");
        let fmt_g16 = |x| fmt_g(x, 16);
        for (x, want) in [
            (0.002596161128523834, "0.002596161128523834"),
            (6.669779853440971e-05, "6.669779853440971e-05"),
            (0.001, "0.001"),
            (0.5, "0.5"),
            (1.0, "1"),
            (2.0, "2"),
            (-1.5, "-1.5"),
            (0.0, "0"),
            (1e-300, "1e-300"),
            (123456789.0, "123456789"),
            (1e16, "1e+16"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
        ] {
            assert_eq!(fmt_g16(x), want, "{x}");
        }
    }
}
