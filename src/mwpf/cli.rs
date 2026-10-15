//! MWPF's command line (`stabilizer_qec.mwpf.run_cli`, MWPF's `mwpf.run_cli`).
//!
//! MWPF parses it with clap 4.5's derive; this file parses it the same way without clap: the
//! help and version text (stored as clap prints it, in `cli_help.rs`), the errors (clap's
//! messages, usage lines and "did you mean" tips, with its rules for which arguments a usage
//! line names), and the exit codes (0 after help or a version, 2 after an error), printed
//! without colour.

#[allow(unused_imports)]
use crate::mwpf::shim::prelude::*;
use super::cli_help::*;
use crate::mwpf::bp::bp::BpSparse;
use crate::mwpf::example_codes::*;
use crate::mwpf::matrix::*;
use crate::mwpf::mwpf_solver::*;
use crate::mwpf::shim::rand::RngCore;
use crate::mwpf::shim::rand::{thread_rng, Rng, SeedableRng};
use crate::mwpf::util::*;
use crate::mwpf::visualize::*;
use num_traits::FromPrimitive;
use std::collections::BTreeMap;
use std::io::Write as _;
use std::sync::Arc;

const TEST_EACH_ROUNDS: usize = 100;
const VERSION: &str = super::MWPF_VERSION;

/// rand 0.8's `SmallRng` on 64-bit targets: Xoshiro256++, seeded by `SeedableRng`'s default
/// `seed_from_u64` (PCG32 filling the seed; `SmallRng` does not take Xoshiro256++'s SplitMix64).
struct SmallRng {
    s: [u64; 4],
}

impl SmallRng {
    fn seed_from_u64(mut state: u64) -> Self {
        let mut s = [0u64; 4];
        for x in s.iter_mut() {
            let mut halves = [0u32; 2];
            for half in halves.iter_mut() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(11634580027462260723);
                let xorshifted = (((state >> 18) ^ state) >> 27) as u32;
                *half = xorshifted.rotate_right((state >> 59) as u32);
            }
            *x = halves[0] as u64 | (halves[1] as u64) << 32;
        }
        if s == [0; 4] {
            // Xoshiro256++'s `from_seed` of zeros: its own `seed_from_u64(0)` (SplitMix64)
            let mut state = 0u64;
            for x in s.iter_mut() {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                *x = z ^ (z >> 31);
            }
        }
        SmallRng { s }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
}

// ---------------------------------------------------------------------------------------------
// The commands and their arguments, as MWPF's derive declares them
// ---------------------------------------------------------------------------------------------

const CODE_TYPES: &[&str] = &[
    "code-capacity-repetition-code",
    "code-capacity-planar-code",
    "code-capacity-color-code",
    "code-capacity-tailored-code",
    "error-pattern-reader",
    "qec-playground-code",
];
const SOLVER_TYPES: &[&str] = &["union-find", "single-hair", "joint-single-hair", "error-pattern-logger"];
const VERIFIERS: &[&str] = &["none", "fusion-serial", "actual-error", "strict-actual-error"];
const MATRIX_TYPES: &[&str] = &["echelon-tail-tight", "echelon-tight", "echelon"];

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Help,
    Version,
    Flag,
    Usize,
    U64,
    F64,
    Str,
    Json,
    Choice(&'static [&'static str]),
}

struct ArgSpec {
    id: &'static str,
    short: Option<char>,
    long: Option<&'static str>,
    /// the value's name (`<D>`, `--pe <PE>`)
    name: &'static str,
    /// a positional argument's place (from 1)
    index: Option<usize>,
    required: bool,
    kind: Kind,
}

const fn opt(id: &'static str, short: Option<char>, long: &'static str, name: &'static str, kind: Kind) -> ArgSpec {
    ArgSpec { id, short, long: Some(long), name, index: None, required: false, kind }
}

const fn flag(id: &'static str, short: Option<char>, long: &'static str) -> ArgSpec {
    ArgSpec { id, short, long: Some(long), name: "", index: None, required: false, kind: Kind::Flag }
}

const HELP: ArgSpec = ArgSpec { id: "help", short: Some('h'), long: Some("help"), name: "", index: None, required: false, kind: Kind::Help };
const VERSION_ARG: ArgSpec = ArgSpec { id: "version", short: Some('V'), long: Some("version"), name: "", index: None, required: false, kind: Kind::Version };

const BENCHMARK_ARGS: &[ArgSpec] = &[
    ArgSpec { id: "d", short: None, long: None, name: "D", index: Some(1), required: true, kind: Kind::Usize },
    ArgSpec { id: "p", short: None, long: None, name: "P", index: Some(2), required: true, kind: Kind::F64 },
    opt("pe", Some('e'), "pe", "PE", Kind::F64),
    opt("noisy_measurements", Some('n'), "noisy-measurements", "NOISY_MEASUREMENTS", Kind::Usize),
    opt("code_type", Some('c'), "code-type", "CODE_TYPE", Kind::Choice(CODE_TYPES)),
    opt("code_config", None, "code-config", "CODE_CONFIG", Kind::Json),
    flag("enable_visualizer", None, "enable-visualizer"),
    opt("visualizer_json_filepath", None, "visualizer-json-filepath", "VISUALIZER_JSON_FILEPATH", Kind::Str),
    opt("visualizer_html_filepath", None, "visualizer-html-filepath", "VISUALIZER_HTML_FILEPATH", Kind::Str),
    flag("print_syndrome_pattern", None, "print-syndrome-pattern"),
    flag("print_error_pattern", None, "print-error-pattern"),
    opt("verifier", None, "verifier", "VERIFIER", Kind::Choice(VERIFIERS)),
    opt("total_rounds", Some('r'), "total-rounds", "TOTAL_ROUNDS", Kind::Usize),
    opt("solver_type", Some('p'), "solver-type", "SOLVER_TYPE", Kind::Choice(SOLVER_TYPES)),
    opt("solver_config", None, "solver-config", "SOLVER_CONFIG", Kind::Json),
    opt("pb_message", None, "pb-message", "PB_MESSAGE", Kind::Str),
    flag("use_deterministic_seed", None, "use-deterministic-seed"),
    opt("benchmark_profiler_output", None, "benchmark-profiler-output", "BENCHMARK_PROFILER_OUTPUT", Kind::Str),
    opt("starting_iteration", None, "starting-iteration", "STARTING_ITERATION", Kind::Usize),
    opt("apply_deterministic_seed", None, "apply-deterministic-seed", "APPLY_DETERMINISTIC_SEED", Kind::U64),
    opt("single_seed", None, "single-seed", "SINGLE_SEED", Kind::U64),
    flag("use_bp", None, "use-bp"),
    opt("bp_application_ratio", None, "bp-application-ratio", "BP_APPLICATION_RATIO", Kind::F64),
    HELP,
    VERSION_ARG,
];

const MATRIX_SPEED_ARGS: &[ArgSpec] = &[
    opt("matrix_type", Some('c'), "matrix-type", "MATRIX_TYPE", Kind::Choice(MATRIX_TYPES)),
    opt("width", None, "width", "WIDTH", Kind::Usize),
    opt("height", None, "height", "HEIGHT", Kind::Usize),
    opt("one_density", None, "one-density", "ONE_DENSITY", Kind::F64),
    opt("total_rounds", Some('r'), "total-rounds", "TOTAL_ROUNDS", Kind::Usize),
    opt("deterministic_seed", None, "deterministic-seed", "DETERMINISTIC_SEED", Kind::U64),
    HELP,
    VERSION_ARG,
];

const DECODER_SPEED_ARGS: &[ArgSpec] = &[
    ArgSpec { id: "file_path", short: Some('f'), long: Some("file-path"), name: "FILE_PATH", index: None, required: true, kind: Kind::Str },
    opt("solver_type", Some('p'), "solver-type", "SOLVER_TYPE", Kind::Choice(SOLVER_TYPES)),
    opt("solver_config", None, "solver-config", "SOLVER_CONFIG", Kind::Json),
    flag("use_bp", None, "use-bp"),
    opt("bp_application_ratio", None, "bp-application-ratio", "BP_APPLICATION_RATIO", Kind::F64),
    opt("bp_max_iter", None, "bp-max-iter", "BP_MAX_ITER", Kind::Usize),
    HELP,
    VERSION_ARG,
];

const CODE_CAPACITY_ARGS: &[ArgSpec] = &[
    flag("print_command", Some('c'), "print-command"),
    flag("enable_visualizer", Some('v'), "enable-visualizer"),
    flag("use_strict", Some('u'), "use-strict"),
    flag("print_syndrome_pattern", Some('s'), "print-syndrome-pattern"),
    opt("solver_type", Some('p'), "solver-type", "SOLVER_TYPE", Kind::Choice(SOLVER_TYPES)),
    opt("solver_config", None, "solver-config", "SOLVER_CONFIG", Kind::Json),
    HELP,
    VERSION_ARG,
];

const HELP_VERSION_ARGS: &[ArgSpec] = &[HELP, VERSION_ARG];

const HELP_COMMAND_ARGS: &[ArgSpec] =
    &[ArgSpec { id: "subcommand", short: None, long: None, name: "COMMAND", index: Some(1), required: false, kind: Kind::Str }];

struct Cmd {
    /// the words after `mwpf` that name it
    path: &'static [&'static str],
    args: &'static [ArgSpec],
    subcommands: &'static [&'static str],
    /// clap's derive puts a struct's arguments in an `ArgGroup` named after it
    group: bool,
    /// its usage line when no arguments are named, after the binary's name
    usage: &'static str,
    short_help: &'static str,
    long_help: &'static str,
}

const ROOT_SUBCOMMANDS: &[&str] = &["benchmark", "matrix-speed", "decoder-speed", "custom-test", "custom-test2", "test", "help"];
const TEST_SUBCOMMANDS: &[&str] = &["common", "code-capacity", "help"];

const COMMANDS: &[Cmd] = &[
    Cmd { path: &[], args: HELP_VERSION_ARGS, subcommands: ROOT_SUBCOMMANDS, group: false, usage: "<COMMAND>", short_help: ROOT_SHORT, long_help: ROOT_LONG },
    Cmd { path: &["benchmark"], args: BENCHMARK_ARGS, subcommands: &[], group: true, usage: "[OPTIONS] <D> <P>", short_help: BENCHMARK_SHORT, long_help: BENCHMARK_LONG },
    Cmd { path: &["matrix-speed"], args: MATRIX_SPEED_ARGS, subcommands: &[], group: true, usage: "[OPTIONS]", short_help: MATRIX_SPEED_SHORT, long_help: MATRIX_SPEED_LONG },
    Cmd { path: &["decoder-speed"], args: DECODER_SPEED_ARGS, subcommands: &[], group: true, usage: "[OPTIONS] --file-path <FILE_PATH>", short_help: DECODER_SPEED_SHORT, long_help: DECODER_SPEED_LONG },
    Cmd { path: &["custom-test"], args: HELP_VERSION_ARGS, subcommands: &[], group: false, usage: "", short_help: CUSTOM_TEST_SHORT, long_help: CUSTOM_TEST_LONG },
    Cmd { path: &["custom-test2"], args: HELP_VERSION_ARGS, subcommands: &[], group: false, usage: "", short_help: CUSTOM_TEST2_SHORT, long_help: CUSTOM_TEST2_LONG },
    Cmd { path: &["test"], args: HELP_VERSION_ARGS, subcommands: TEST_SUBCOMMANDS, group: false, usage: "<COMMAND>", short_help: TEST_SHORT, long_help: TEST_LONG },
    Cmd { path: &["test", "common"], args: HELP_VERSION_ARGS, subcommands: &[], group: false, usage: "", short_help: TEST_COMMON_SHORT, long_help: TEST_COMMON_LONG },
    Cmd { path: &["test", "code-capacity"], args: CODE_CAPACITY_ARGS, subcommands: &[], group: true, usage: "[OPTIONS]", short_help: TEST_CODE_CAPACITY_SHORT, long_help: TEST_CODE_CAPACITY_LONG },
    Cmd { path: &["help"], args: HELP_COMMAND_ARGS, subcommands: &[], group: false, usage: "[COMMAND]...", short_help: HELP_HELP, long_help: HELP_HELP },
    Cmd { path: &["test", "help"], args: HELP_COMMAND_ARGS, subcommands: &[], group: false, usage: "[COMMAND]...", short_help: HELP_HELP, long_help: HELP_HELP },
];

fn command(path: &[&str]) -> &'static Cmd {
    COMMANDS.iter().find(|c| c.path == path).expect("a command of MWPF's command line")
}

impl ArgSpec {
    fn takes_value(&self) -> bool {
        !matches!(self.kind, Kind::Help | Kind::Version | Kind::Flag)
    }

    /// clap's `Display` for an argument, as its errors name it
    fn display(&self) -> String {
        match (self.index, self.long, self.short) {
            (Some(_), _, _) => format!("<{}>", self.name),
            (None, Some(long), _) if self.takes_value() => format!("--{long} <{}>", self.name),
            (None, Some(long), _) => format!("--{long}"),
            (None, None, Some(short)) => format!("-{short}"),
            _ => unreachable!(),
        }
    }

    fn stylized(&self, required: bool) -> String {
        match self.index {
            Some(_) if !required => format!("[{}]", self.name),
            _ => self.display(),
        }
    }

    fn possible_values(&self) -> &'static [&'static str] {
        match self.kind {
            Kind::Choice(values) => values,
            _ => &[],
        }
    }
}

// ---------------------------------------------------------------------------------------------
// clap's parser, for these commands
// ---------------------------------------------------------------------------------------------

/// How clap stops: text for stdout (help, versions; exit 0) or stderr (errors; exit 2).
struct Exit {
    text: String,
    stderr: bool,
    code: i32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Ident {
    Short,
    Long,
    Index,
}

/// an argument (by its place in `Cmd::args`) or the command's group
#[derive(Clone, Copy, PartialEq, Debug)]
enum Id {
    Arg(usize),
    Group,
}

struct Pending {
    arg: usize,
    ident: Ident,
    values: Vec<String>,
}

#[derive(Default)]
struct Matches {
    values: BTreeMap<&'static str, String>,
    subcommand: Option<(&'static str, Box<Matches>)>,
}

impl Matches {
    fn get(&self, id: &str) -> Option<&str> {
        self.values.get(id).map(|s| s.as_str())
    }
    fn flag(&self, id: &str) -> bool {
        self.values.contains_key(id)
    }
    fn usize_or(&self, id: &str, default: usize) -> usize {
        self.get(id).map_or(default, |s| s.parse().unwrap())
    }
    fn u64(&self, id: &str) -> Option<u64> {
        self.get(id).map(|s| s.parse().unwrap())
    }
    fn f64(&self, id: &str) -> Option<f64> {
        self.get(id).map(|s| s.parse().unwrap())
    }
    fn json(&self, id: &str) -> serde_json::Value {
        self.get(id).map_or_else(|| json!({}), |s| serde_json::parse(s).unwrap())
    }
}

/// strsim's Jaro similarity, over characters
fn jaro(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() && b.is_empty() {
        return 1.0;
    } else if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let search_range = (a.len().max(b.len()) / 2).saturating_sub(1);
    let mut a_flags = vec![false; a.len()];
    let mut b_flags = vec![false; b.len()];
    let mut matches = 0usize;
    for (i, ca) in a.iter().enumerate() {
        let min_bound = i.saturating_sub(search_range);
        let max_bound = b.len().min(i + search_range + 1);
        for j in min_bound..max_bound {
            if *ca == b[j] && !b_flags[j] {
                a_flags[i] = true;
                b_flags[j] = true;
                matches += 1;
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }
    let matched_b: Vec<char> = b.iter().zip(&b_flags).filter(|(_, f)| **f).map(|(c, _)| *c).collect();
    let matched_a = a.iter().zip(&a_flags).filter(|(_, f)| **f).map(|(c, _)| *c);
    let transpositions = matched_a.zip(matched_b).filter(|(x, y)| x != y).count() / 2;
    let m = matches as f64;
    (m / a.len() as f64 + m / b.len() as f64 + (matches - transpositions) as f64 / m) / 3.0
}

/// clap's suggestions: the candidates more than 0.7 similar, least similar first
fn did_you_mean<'a>(value: &str, candidates: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut found: Vec<(f64, String)> =
        candidates.into_iter().map(|c| (jaro(value, c), c.to_string())).filter(|(confidence, _)| *confidence > 0.7).collect();
    found.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    found.into_iter().map(|(_, c)| c).collect()
}

struct Parser<'a> {
    cmd: &'static Cmd,
    /// the binary's name, from the first argument (as clap takes it)
    bin: Option<&'a str>,
    /// explicitly given arguments (and the group, once one of its members is), in order
    matched: Vec<(Id, String)>,
    pending: Option<Pending>,
}

enum Parsed {
    ValuesDone,
    Opt(usize),
    NoArg,
}

impl<'a> Parser<'a> {
    fn new(cmd: &'static Cmd, bin: Option<&'a str>) -> Self {
        Parser { cmd, bin, matched: vec![], pending: None }
    }

    fn arg(&self, k: usize) -> &'static ArgSpec {
        &self.cmd.args[k]
    }

    fn in_group(&self, k: usize) -> bool {
        self.cmd.group && !matches!(self.arg(k).kind, Kind::Help | Kind::Version)
    }

    fn has_positionals(&self) -> bool {
        self.cmd.args.iter().any(|a| a.index.is_some())
    }

    /// the name usage lines start with
    fn usage_name(&self) -> String {
        let mut words: Vec<&str> = self.bin.into_iter().collect();
        words.extend(self.cmd.path.iter());
        if words.is_empty() {
            "mwpf".into()
        } else {
            words.join(" ")
        }
    }

    fn help_text(&self, long: bool) -> String {
        let text = if long { self.cmd.long_help } else { self.cmd.short_help };
        // the help command's text is stored once, as 's
        let mut written = String::from("Usage: mwpf");
        let words: &[&str] = if self.cmd.path.last() == Some(&"help") { &["help"] } else { self.cmd.path };
        for word in words {
            written.push(' ');
            written.push_str(word);
        }
        text.replacen(&written, &format!("Usage: {}", self.usage_name()), 1)
    }

    fn format_group(&self) -> String {
        let members: Vec<String> = (0..self.cmd.args.len())
            .filter(|&k| self.in_group(k))
            .map(|k| {
                let a = self.arg(k);
                if a.index.is_some() {
                    a.name.to_string()
                } else {
                    a.display()
                }
            })
            .collect();
        format!("<{}>", members.join("|"))
    }

    /// clap's usage line: the plain one when `used` is empty, else one naming the required
    /// arguments and those used
    fn usage(&self, used: &[Id]) -> String {
        let name = self.usage_name();
        if used.is_empty() {
            return if self.cmd.usage.is_empty() { name } else { format!("{name} {}", self.cmd.usage) };
        }
        let required: Vec<Id> = (0..self.cmd.args.len()).filter(|&k| self.arg(k).required).map(Id::Arg).collect();
        let all: Vec<Id> = required.iter().chain(used.iter()).copied().collect();
        let group = all.contains(&Id::Group);
        let mut opts: Vec<String> = vec![];
        let mut positionals: BTreeMap<usize, String> = BTreeMap::new();
        for id in all.iter() {
            if let Id::Arg(k) = *id {
                if group && self.in_group(k) {
                    continue;
                }
                let a = self.arg(k);
                match a.index {
                    Some(index) => {
                        positionals.insert(index, a.stylized(true));
                    }
                    None => {
                        let s = a.stylized(true);
                        if !opts.contains(&s) {
                            opts.push(s);
                        }
                    }
                }
            }
        }
        for (k, a) in self.cmd.args.iter().enumerate() {
            if let Some(index) = a.index {
                if !(group && self.in_group(k)) {
                    positionals.entry(index).or_insert_with(|| a.stylized(false));
                }
            }
        }
        let mut s = format!("{name} ");
        for o in opts {
            s += &o;
            s.push(' ');
        }
        if group {
            s += &self.format_group();
            s.push(' ');
        }
        for p in positionals.values() {
            s += p;
            s.push(' ');
        }
        if !self.cmd.subcommands.is_empty() {
            s += "<COMMAND>";
        }
        s.trim_end().to_string()
    }

    fn error(&self, body: String, usage: Option<String>) -> Exit {
        let mut text = format!("error: {body}");
        if let Some(usage) = usage {
            text += &format!("\n\nUsage: {usage}");
        }
        // the help command has no `--help` (nor commands to say `help` of)
        if self.cmd.args.iter().any(|a| a.kind == Kind::Help) {
            text += "\n\nFor more information, try '--help'.\n";
        } else {
            text.push('\n');
        }
        Exit { text, stderr: true, code: 2 }
    }

    fn unknown_argument(&self, arg: &str, did_you_mean: Option<(String, Option<String>)>, trailing_tip: bool, usage: String) -> Exit {
        let mut body = format!("unexpected argument '{arg}' found");
        let mut tips = vec![];
        if trailing_tip {
            tips.push(format!("to pass '{arg}' as a value, use '-- {arg}'"));
        }
        let mut suggested = false;
        match did_you_mean {
            Some((flag, None)) => {
                body += &format!("\n\n  tip: a similar argument exists: '{flag}'");
                suggested = true;
            }
            Some((flag, Some(sub))) => tips.push(format!("'{sub} {flag}' exists")),
            None => {}
        }
        if !tips.is_empty() && !suggested {
            body.push('\n');
        }
        for tip in tips {
            body += &format!("\n  tip: {tip}");
        }
        self.error(body, Some(usage))
    }

    fn unrecognized_subcommand(&self, name: &str) -> Exit {
        self.error(format!("unrecognized subcommand '{name}'"), Some(self.usage(&[])))
    }

    fn invalid_value(&self, arg: &ArgSpec, bad: &str) -> Exit {
        let possible = arg.possible_values();
        let mut body = if bad.is_empty() {
            format!("a value is required for '{}' but none was supplied", arg.display())
        } else {
            format!("invalid value '{bad}' for '{}'", arg.display())
        };
        if !possible.is_empty() {
            body += &format!("\n  [possible values: {}]", possible.join(", "));
        }
        if !bad.is_empty() {
            if let Some(similar) = did_you_mean(bad, possible.iter().copied()).pop() {
                body += &format!("\n\n  tip: a similar value exists: '{similar}'");
            }
        }
        self.error(body, None)
    }

    fn used_ids(&self, keep: impl Fn(Id) -> bool) -> Vec<Id> {
        self.matched.iter().map(|(id, _)| *id).filter(|id| keep(*id)).collect()
    }

    fn start_arg(&mut self, k: usize, value: String) {
        self.matched.push((Id::Arg(k), value));
        if self.in_group(k) && !self.matched.iter().any(|(id, _)| *id == Id::Group) {
            self.matched.push((Id::Group, String::new()));
        }
    }

    fn check_value(&self, k: usize, raw: &str) -> Result<(), Exit> {
        let a = self.arg(k);
        let failure = match a.kind {
            Kind::Usize => raw.parse::<usize>().err().map(|e| e.to_string()),
            Kind::U64 => raw.parse::<u64>().err().map(|e| e.to_string()),
            Kind::F64 => raw.parse::<f64>().err().map(|e| e.to_string()),
            Kind::Json => {
                return match serde_json::parse(raw) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(self.error(format!("invalid value 'should be like {{\"a\":1}}, parse error: {e}' for '{}'", a.display()), None)),
                }
            }
            Kind::Choice(values) => {
                return if values.contains(&raw) { Ok(()) } else { Err(self.invalid_value(a, raw)) };
            }
            _ => None,
        };
        match failure {
            Some(reason) => Err(self.error(format!("invalid value '{raw}' for '{}': {reason}", a.display()), None)),
            None => Ok(()),
        }
    }

    fn resolve_pending(&mut self) -> Result<(), Exit> {
        match self.pending.take() {
            Some(p) => self.react(p.arg, p.ident, p.values),
            None => Ok(()),
        }
    }

    fn react(&mut self, k: usize, ident: Ident, values: Vec<String>) -> Result<(), Exit> {
        self.resolve_pending()?;
        let a = self.arg(k);
        if a.takes_value() && values.is_empty() {
            return Err(self.invalid_value(a, ""));
        }
        match a.kind {
            Kind::Help => Err(Exit { text: self.help_text(ident != Ident::Short), stderr: false, code: 0 }),
            Kind::Version => {
                let mut name = String::from("mwpf");
                for word in self.cmd.path {
                    name += "-";
                    name += word;
                }
                Err(Exit { text: format!("{name} {VERSION}\n"), stderr: false, code: 0 })
            }
            _ => {
                if let Some(at) = self.matched.iter().position(|(id, _)| *id == Id::Arg(k)) {
                    self.matched.remove(at);
                    return Err(self.error(format!("the argument '{}' cannot be used multiple times", a.display()), Some(self.usage(&[]))));
                }
                let value = values.into_iter().next().unwrap_or_else(|| "true".into());
                self.start_arg(k, value.clone());
                self.check_value(k, &value)
            }
        }
    }

    fn parse_opt_value(&mut self, k: usize, ident: Ident, attached: Option<String>) -> Result<Parsed, Exit> {
        match attached {
            Some(v) => {
                self.react(k, ident, vec![v])?;
                Ok(Parsed::ValuesDone)
            }
            None => {
                self.resolve_pending()?;
                self.pending = Some(Pending { arg: k, ident, values: vec![] });
                Ok(Parsed::Opt(k))
            }
        }
    }

    fn did_you_mean_error(&mut self, arg: &str, remaining: &[String], trailing: bool) -> Exit {
        let longs: Vec<&str> = self.cmd.args.iter().filter_map(|a| a.long).collect();
        let suggestion = match did_you_mean(arg, longs).pop() {
            Some(found) => Some((found, None)),
            None => self
                .cmd
                .subcommands
                .iter()
                .filter_map(|sub| {
                    let mut path: Vec<&str> = self.cmd.path.to_vec();
                    path.push(sub);
                    let sc = command(&path);
                    let found = did_you_mean(arg, sc.args.iter().filter_map(|a| a.long)).pop()?;
                    let score = remaining.iter().position(|r| r == sub)?;
                    Some((score, (found, Some(sub.to_string()))))
                })
                .min_by_key(|(score, _)| *score)
                .map(|(_, s)| s),
        };
        if let Some((name, None)) = &suggestion {
            let k = self.cmd.args.iter().position(|a| a.long == Some(name.as_str())).unwrap();
            self.start_arg(k, String::new());
        }
        let suggestion = suggestion.map(|(flag, sub)| (format!("--{flag}"), sub));
        // clap 4.5.30 keeps the group here (later versions drop it)
        let used = self.used_ids(|_| true);
        let trailing_tip = suggestion.is_none() && !trailing && self.has_positionals();
        self.unknown_argument(&format!("--{arg}"), suggestion, trailing_tip, self.usage(&used))
    }

    fn help_subcommand(&self, names: &[String]) -> Exit {
        let mut path: Vec<&'static str> = self.cmd.path.to_vec();
        for name in names {
            let sc = command(&path);
            match sc.subcommands.iter().find(|s| *s == name) {
                Some(s) => path.push(s),
                None => return Parser::new(sc, self.bin).unrecognized_subcommand(name),
            }
        }
        Exit { text: Parser::new(command(&path), self.bin).help_text(true), stderr: false, code: 0 }
    }

    fn parse(mut self, args: &[String]) -> Result<Matches, Exit> {
        let mut state = Parsed::ValuesDone;
        let mut pos_counter = 1;
        let mut trailing = false;
        let mut subcommand: Option<(&'static str, usize)> = None;
        let positional_count = self.cmd.args.iter().filter(|a| a.index.is_some()).count();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            i += 1;
            if !trailing {
                if !matches!(state, Parsed::Opt(_)) {
                    if let Some(sub) = self.cmd.subcommands.iter().find(|s| *s == a) {
                        if *sub == "help" {
                            return Err(self.help_subcommand(&args[i..]));
                        }
                        subcommand = Some((sub, i));
                        break;
                    }
                }
                if a == "--" {
                    trailing = true;
                    continue;
                } else if let Some(long) = a.strip_prefix("--") {
                    let (name, value) = match long.split_once('=') {
                        Some((n, v)) => (n, Some(v.to_string())),
                        None => (long, None),
                    };
                    match self.cmd.args.iter().position(|x| x.long == Some(name)) {
                        Some(k) => {
                            let spec = self.arg(k);
                            if spec.takes_value() {
                                state = self.parse_opt_value(k, Ident::Long, value)?;
                            } else if let Some(rest) = value {
                                let mut used = self.used_ids(|id| match id {
                                    Id::Arg(j) => !self.arg(j).required,
                                    Id::Group => true,
                                });
                                used.push(Id::Arg(k));
                                let _ = self.resolve_pending();
                                let usage = self.usage(&used);
                                return Err(self.error(format!("unexpected value '{rest}' for '{}' found; no more were expected", spec.display()), Some(usage)));
                            } else {
                                self.react(k, Ident::Long, vec![])?;
                                state = Parsed::ValuesDone;
                            }
                            continue;
                        }
                        None => {
                            let _ = self.resolve_pending();
                            return Err(self.did_you_mean_error(name, &args[i..], trailing));
                        }
                    }
                } else if a.len() > 1 && a.starts_with('-') {
                    let flags: Vec<char> = a[1..].chars().collect();
                    let mut result = Parsed::NoArg;
                    let mut c_at = 0;
                    let mut offset = 1;
                    let mut unknown = None;
                    while c_at < flags.len() {
                        let c = flags[c_at];
                        offset += c.len_utf8();
                        c_at += 1;
                        let Some(k) = self.cmd.args.iter().position(|x| x.short == Some(c)) else {
                            unknown = Some(c);
                            break;
                        };
                        if !self.arg(k).takes_value() {
                            self.react(k, Ident::Short, vec![])?;
                            result = Parsed::ValuesDone;
                            continue;
                        }
                        let rest = &a[offset..];
                        let attached = if rest.is_empty() { None } else { Some(rest.strip_prefix('=').unwrap_or(rest).to_string()) };
                        result = self.parse_opt_value(k, Ident::Short, attached)?;
                        break;
                    }
                    if let Some(c) = unknown {
                        let _ = self.resolve_pending();
                        let trailing_tip = !trailing && self.has_positionals();
                        return Err(self.unknown_argument(&format!("-{c}"), None, trailing_tip, self.usage(&[])));
                    }
                    if !matches!(result, Parsed::NoArg) {
                        state = result;
                        continue;
                    }
                }
                if let Parsed::Opt(k) = state {
                    self.pending.as_mut().expect("an option waiting for its value").values.push(a.clone());
                    let _ = k;
                    state = Parsed::ValuesDone;
                    continue;
                }
            }
            if pos_counter <= positional_count {
                let k = self.cmd.args.iter().position(|x| x.index == Some(pos_counter)).unwrap();
                self.resolve_pending()?;
                self.pending = Some(Pending { arg: k, ident: Ident::Index, values: vec![a.clone()] });
                pos_counter += 1;
                state = Parsed::ValuesDone;
            } else {
                let _ = self.resolve_pending();
                return Err(self.match_arg_error(a, trailing));
            }
        }
        let mut matches = Matches::default();
        if let Some((sub, at)) = subcommand {
            let mut path: Vec<&str> = self.cmd.path.to_vec();
            path.push(sub);
            let sub_matches = Parser::new(command(&path), self.bin).parse(&args[at..])?;
            matches.subcommand = Some((sub, Box::new(sub_matches)));
        }
        self.resolve_pending()?;
        if matches.subcommand.is_none() && !self.cmd.subcommands.is_empty() && self.matched.is_empty() {
            // `arg_required_else_help`
            return Err(Exit { text: self.help_text(false), stderr: true, code: 2 });
        }
        let missing: Vec<usize> =
            (0..self.cmd.args.len()).filter(|&k| self.arg(k).required && !self.matched.iter().any(|(id, _)| *id == Id::Arg(k))).collect();
        if !missing.is_empty() {
            let mut opts = vec![];
            let mut positionals = BTreeMap::new();
            for &k in missing.iter() {
                let a = self.arg(k);
                match a.index {
                    Some(index) => {
                        positionals.insert(index, a.stylized(true));
                    }
                    None => opts.push(a.stylized(true)),
                }
            }
            let mut body = String::from("the following required arguments were not provided:");
            for name in opts.iter().chain(positionals.values()) {
                body += &format!("\n  {name}");
            }
            let mut used = self.used_ids(|id| id != Id::Group);
            used.extend(missing.iter().map(|&k| Id::Arg(k)));
            return Err(self.error(body, Some(self.usage(&used))));
        }
        for (id, value) in self.matched {
            if let Id::Arg(k) = id {
                matches.values.insert(self.cmd.args[k].id, value);
            }
        }
        Ok(matches)
    }

    fn match_arg_error(&self, a: &str, trailing: bool) -> Exit {
        if trailing && self.cmd.subcommands.contains(&a) {
            let body = format!("unexpected argument '{a}' found\n\n  tip: subcommand '{a}' exists; to use it, remove the '--' before it");
            return self.error(body, Some(self.usage(&[])));
        }
        let looks_like_flag = a.starts_with('-') && a != "-" && !(a.starts_with("--") && a.len() == 2);
        let trailing_tip = !trailing && self.has_positionals() && looks_like_flag;
        if !self.cmd.subcommands.is_empty() {
            let candidates = did_you_mean(a, self.cmd.subcommands.iter().copied());
            if !candidates.is_empty() {
                let mut body = format!("unrecognized subcommand '{a}'\n\n  tip: ");
                if candidates.len() == 1 {
                    body += "a similar subcommand exists: ";
                } else {
                    body += "some similar subcommands exist: ";
                }
                body += &candidates.iter().map(|c| format!("'{c}'")).collect::<Vec<_>>().join(", ");
                if trailing_tip {
                    body += &format!("\n  tip: to pass '{a}' as a value, use '{} -- {a}'", self.bin.unwrap_or("mwpf"));
                }
                return self.error(body, Some(self.usage(&[])));
            }
            if !self.has_positionals() {
                return self.unrecognized_subcommand(a);
            }
        }
        self.unknown_argument(a, None, trailing_tip, self.usage(&[]))
    }
}

/// Parses MWPF's command line (`parameters[0]` naming the binary), printing help, a version or
/// an error and exiting the process as clap does when it stops there.
fn parse_command_line(parameters: &[String]) -> Matches {
    let bin = parameters.first().and_then(|p| std::path::Path::new(p).file_name()).and_then(|f| f.to_str());
    let rest = if parameters.is_empty() { &parameters[..] } else { &parameters[1..] };
    match Parser::new(command(&[]), bin).parse(rest) {
        Ok(matches) => matches,
        Err(exit) => {
            if exit.stderr {
                let mut err = std::io::stderr().lock();
                let _ = err.write_all(exit.text.as_bytes());
                let _ = err.flush();
            } else {
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(exit.text.as_bytes());
                let _ = out.flush();
            }
            let _ = std::io::stdout().lock().flush();
            std::process::exit(exit.code)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The commands' parameters
// ---------------------------------------------------------------------------------------------

/// note that these code type is only for example, to test and demonstrate the correctness of the algorithm, but not for real QEC simulation;
/// for real simulation, please refer to <https://github.com/yuewuo/QEC-Playground>
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum ExampleCodeType {
    /// quantum repetition code with perfect stabilizer measurement
    CodeCapacityRepetitionCode,
    /// planar code with perfect stabilizer measurement, only one type of stabilizer's decoding graph (thus normal graph)
    CodeCapacityPlanarCode,
    /// color code with perfect stabilizer measurement
    CodeCapacityColorCode,
    /// tailored surface code, which is essentially rotated planar code with depolarizing noise model
    CodeCapacityTailoredCode,
    /// read from error pattern file, generated using option `--solver-type error-pattern-logger`
    ErrorPatternReader,
    /// QEC-Playground's codes (MWPF's wheels build them on the `qecp` crate, which this package
    /// does not include)
    QECPlaygroundCode,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum SolverType {
    /// the solver from Union-Find decoder
    UnionFind,
    /// the single-hair solver
    SingleHair,
    /// joint single-hair solver
    JointSingleHair,
    /// log error into a file for later fetch
    ErrorPatternLogger,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Verifier {
    /// disable verifier
    None,
    /// use the serial version of fusion blossom library to verify
    FusionSerial,
    /// if the actual error has smaller weight than the lower bound of the weight range then fail
    ActualError,
    /// if the actual error has smaller weight than the solved subgraph then fail
    StrictActualError,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum MatrixSpeedClass {
    EchelonTailTight,
    EchelonTight,
    Echelon,
}

/// a value-enum's variant from its name on the command line
fn variant<T: Copy>(names: &[&str], variants: &[T], name: &str) -> T {
    variants[names.iter().position(|n| *n == name).expect("a parsed choice")]
}

impl ExampleCodeType {
    fn from_name(name: &str) -> Self {
        use ExampleCodeType::*;
        variant(CODE_TYPES, &[CodeCapacityRepetitionCode, CodeCapacityPlanarCode, CodeCapacityColorCode, CodeCapacityTailoredCode, ErrorPatternReader, QECPlaygroundCode], name)
    }
}

impl SolverType {
    fn from_name(name: &str) -> Self {
        use SolverType::*;
        variant(SOLVER_TYPES, &[UnionFind, SingleHair, JointSingleHair, ErrorPatternLogger], name)
    }
    /// its name on the command line (serde's kebab-case name)
    pub fn name(&self) -> &'static str {
        SOLVER_TYPES[*self as usize]
    }
}

impl Verifier {
    fn from_name(name: &str) -> Self {
        use Verifier::*;
        variant(VERIFIERS, &[None, FusionSerial, ActualError, StrictActualError], name)
    }
}

impl MatrixSpeedClass {
    fn from_name(name: &str) -> Self {
        use MatrixSpeedClass::*;
        variant(MATRIX_TYPES, &[EchelonTailTight, EchelonTight, Echelon], name)
    }
}

#[derive(Clone)]
pub struct DecoderSpeedParameters {
    file_path: String,
    solver_type: SolverType,
    solver_config: serde_json::Value,
    use_bp: bool,
    bp_application_ratio: Option<f64>,
    bp_max_iter: Option<usize>,
}

#[derive(Clone)]
pub struct BenchmarkParameters {
    d: VertexNum,
    p: f64,
    pe: f64,
    noisy_measurements: VertexNum,
    code_type: ExampleCodeType,
    code_config: serde_json::Value,
    enable_visualizer: bool,
    visualizer_json_filepath: Option<String>,
    visualizer_html_filepath: Option<String>,
    print_syndrome_pattern: bool,
    print_error_pattern: bool,
    verifier: Verifier,
    total_rounds: usize,
    solver_type: SolverType,
    solver_config: serde_json::Value,
    pb_message: String,
    use_deterministic_seed: bool,
    benchmark_profiler_output: Option<String>,
    starting_iteration: usize,
    apply_deterministic_seed: Option<u64>,
    single_seed: Option<u64>,
    use_bp: bool,
    bp_application_ratio: Option<f64>,
}

#[derive(Clone, Debug)]
pub enum TestCommands {
    /// test common cases
    Common,
    /// test various codes using code capacity noise model
    CodeCapacity {
        print_command: bool,
        enable_visualizer: bool,
        use_strict: bool,
        print_syndrome_pattern: bool,
        solver_type: SolverType,
        solver_config: serde_json::Value,
    },
}

#[derive(Clone)]
pub struct MatrixSpeedParameters {
    matrix_type: MatrixSpeedClass,
    width: usize,
    height: usize,
    one_density: f64,
    total_rounds: usize,
    deterministic_seed: Option<u64>,
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    Benchmark(BenchmarkParameters),
    MatrixSpeed(MatrixSpeedParameters),
    DecoderSpeed(DecoderSpeedParameters),
    CustomTest,
    CustomTest2,
    Test { command: TestCommands },
}

#[derive(Clone)]
pub struct Cli {
    command: Commands,
}

impl Cli {
    /// `clap::Parser::parse_from`: exits the process after help, a version or an error
    pub fn parse_from<I: IntoIterator<Item = T>, T: Into<String>>(parameters: I) -> Self {
        let parameters: Vec<String> = parameters.into_iter().map(Into::into).collect();
        let matches = parse_command_line(&parameters);
        let (name, m) = matches.subcommand.expect("a subcommand");
        let command = match name {
            "benchmark" => Commands::Benchmark(BenchmarkParameters {
                d: m.usize_or("d", 0),
                p: m.f64("p").unwrap(),
                pe: m.f64("pe").unwrap_or(0.),
                noisy_measurements: m.usize_or("noisy_measurements", 0),
                code_type: ExampleCodeType::from_name(m.get("code_type").unwrap_or("code-capacity-tailored-code")),
                code_config: m.json("code_config"),
                enable_visualizer: m.flag("enable_visualizer"),
                visualizer_json_filepath: m.get("visualizer_json_filepath").map(String::from),
                visualizer_html_filepath: m.get("visualizer_html_filepath").map(String::from),
                print_syndrome_pattern: m.flag("print_syndrome_pattern"),
                print_error_pattern: m.flag("print_error_pattern"),
                verifier: Verifier::from_name(m.get("verifier").unwrap_or("actual-error")),
                total_rounds: m.usize_or("total_rounds", 1000),
                solver_type: SolverType::from_name(m.get("solver_type").unwrap_or("union-find")),
                solver_config: m.json("solver_config"),
                pb_message: m.get("pb_message").unwrap_or("").to_string(),
                use_deterministic_seed: m.flag("use_deterministic_seed"),
                benchmark_profiler_output: m.get("benchmark_profiler_output").map(String::from),
                starting_iteration: m.usize_or("starting_iteration", 0),
                apply_deterministic_seed: m.u64("apply_deterministic_seed"),
                single_seed: m.u64("single_seed"),
                use_bp: m.flag("use_bp"),
                bp_application_ratio: m.f64("bp_application_ratio"),
            }),
            "matrix-speed" => Commands::MatrixSpeed(MatrixSpeedParameters {
                matrix_type: MatrixSpeedClass::from_name(m.get("matrix_type").unwrap_or("echelon-tail-tight")),
                width: m.usize_or("width", 50),
                height: m.usize_or("height", 50),
                one_density: m.f64("one_density").unwrap_or(0.1),
                total_rounds: m.usize_or("total_rounds", 100000),
                deterministic_seed: m.u64("deterministic_seed"),
            }),
            "decoder-speed" => Commands::DecoderSpeed(DecoderSpeedParameters {
                file_path: m.get("file_path").unwrap().to_string(),
                solver_type: SolverType::from_name(m.get("solver_type").unwrap_or("joint-single-hair")),
                solver_config: m.json("solver_config"),
                use_bp: m.flag("use_bp"),
                bp_application_ratio: m.f64("bp_application_ratio"),
                bp_max_iter: m.get("bp_max_iter").map(|s| s.parse().unwrap()),
            }),
            "custom-test" => Commands::CustomTest,
            "custom-test2" => Commands::CustomTest2,
            "test" => {
                let (name, t) = m.subcommand.expect("a test");
                Commands::Test {
                    command: match name {
                        "common" => TestCommands::Common,
                        _ => TestCommands::CodeCapacity {
                            print_command: t.flag("print_command"),
                            enable_visualizer: t.flag("enable_visualizer"),
                            use_strict: t.flag("use_strict"),
                            print_syndrome_pattern: t.flag("print_syndrome_pattern"),
                            solver_type: SolverType::from_name(t.get("solver_type").unwrap_or("union-find")),
                            solver_config: t.json("solver_config"),
                        },
                    },
                }
            }
            _ => unreachable!(),
        };
        Cli { command }
    }
}

impl MatrixSpeedClass {
    pub fn run(&self, parameters: MatrixSpeedParameters, samples: Vec<Vec<(Vec<usize>, bool)>>) {
        match *self {
            MatrixSpeedClass::EchelonTailTight => {
                let mut matrix = Echelon::<Tail<Tight<BasicMatrix>>>::new();
                for edge_index in 0..parameters.width {
                    matrix.add_tight_variable(edge_index);
                }
                Self::run_on_matrix_interface(&matrix, samples)
            }
            MatrixSpeedClass::EchelonTight => {
                let mut matrix = Echelon::<Tight<BasicMatrix>>::new();
                for edge_index in 0..parameters.width {
                    matrix.add_tight_variable(edge_index);
                }
                Self::run_on_matrix_interface(&matrix, samples)
            }
            MatrixSpeedClass::Echelon => {
                let mut matrix = Echelon::<BasicMatrix>::new();
                for edge_index in 0..parameters.width {
                    matrix.add_variable(edge_index);
                }
                Self::run_on_matrix_interface(&matrix, samples)
            }
        }
    }

    pub fn run_on_matrix_interface<M: MatrixView + Clone>(matrix: &M, samples: Vec<Vec<(Vec<usize>, bool)>>) {
        for parity_checks in samples.iter() {
            let mut matrix = matrix.clone();
            for (vertex_index, (incident_edges, parity)) in parity_checks.iter().enumerate() {
                matrix.add_constraint(vertex_index, incident_edges, *parity);
            }
            // for a MatrixView, visiting the columns and rows is sufficient to update its internal state
            matrix.columns();
            matrix.rows();
        }
    }
}

impl Cli {
    pub fn run(self) {
        match self.command {
            Commands::Benchmark(BenchmarkParameters {
                d,
                p,
                pe,
                noisy_measurements,
                code_type,
                enable_visualizer,
                visualizer_json_filepath,
                visualizer_html_filepath,
                verifier,
                total_rounds,
                solver_type,
                    pb_message: _,
                solver_config,
                code_config,
                use_deterministic_seed,
                benchmark_profiler_output,
                print_syndrome_pattern,
                starting_iteration,
                print_error_pattern,
                apply_deterministic_seed,
                single_seed,
                use_bp,
                bp_application_ratio,
            }) => {
                // whether to disable progress bar, useful when running jobs in background
                let mut code: Box<dyn ExampleCode> = code_type.build(d, p, noisy_measurements, code_config);

                // setting up the BP decoder
                let mut pcm = BpSparse::new(code.vertex_num(), code.edge_num(), 0);
                for (col_index, edge) in code.edges().iter().enumerate() {
                    for &row_index in edge.vertices.iter() {
                        pcm.insert_entry(row_index, col_index);
                    }
                }

                if pe != 0. {
                    code.set_erasure_probability(pe);
                }
                // create initializer and solver
                let initializer = Arc::new(code.get_initializer());
                let mut solver = solver_type.build(&initializer, &*code, solver_config.clone());
                if use_bp {
                    solver = match SolverBPWrapper::new(solver.solver_base(), 1, bp_application_ratio.unwrap_or(0.1))
                        .solver
                        .inner
                    {
                        SolverEnum::SolverSerialUnionFind(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverSerialSingleHair(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverSerialJointSingleHair(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverErrorPatternLogger(_) => panic!("not supported"),
                    };
                }
                let mut result_verifier = verifier.build(&initializer);
                // prepare progress bar display

                // single seed mode, intended only execute a single failing round
                if let Some(seed) = single_seed {
                    let (syndrome_pattern, error_pattern) = code.generate_random_errors(seed);

                    if print_syndrome_pattern {
                        println!("syndrome_pattern: {:?}", syndrome_pattern);
                    }
                    if print_error_pattern {
                        println!("error_pattern: {:?}", error_pattern);
                    }
                    let mut visualizer = None;
                    if enable_visualizer {
                        let new_visualizer =
                            Visualizer::new(visualizer_json_filepath.clone(), code.get_positions(), true).unwrap();
                        visualizer = Some(new_visualizer);
                    }

                    solver.solve_visualizer(syndrome_pattern.clone(), visualizer.as_mut());
                    result_verifier.verify(&mut solver, &syndrome_pattern, &error_pattern, visualizer.as_mut(), seed);
                    if let Some(html_path) = &visualizer_html_filepath {
                        if let Some(visualizer) = visualizer.as_mut() {
                            visualizer.save_html(html_path);
                        }
                    }
                    solver.clear();

                    return;
                }

                let mut benchmark_profiler = BenchmarkProfiler::new(noisy_measurements, benchmark_profiler_output);
                thread_rng().gen::<u64>();
                let mut seed = match apply_deterministic_seed {
                    Some(seed) => seed,
                    None => thread_rng().gen::<u64>(),
                };
                let mut rng = SmallRng::seed_from_u64(seed);
                for round in (starting_iteration as u64)..(total_rounds as u64) {
                    seed = if use_deterministic_seed { round } else { rng.next_u64() };
                    let (syndrome_pattern, error_pattern) = code.generate_random_errors(seed);

                    if print_syndrome_pattern {
                        println!("syndrome_pattern: {:?}", syndrome_pattern);
                    }
                    if print_error_pattern {
                        println!("error_pattern: {:?}", error_pattern);
                    }
                    // create a new visualizer each round
                    let mut visualizer = None;
                    if enable_visualizer {
                        let new_visualizer =
                            Visualizer::new(visualizer_json_filepath.clone(), code.get_positions(), true).unwrap();
                        visualizer = Some(new_visualizer);
                    }
                    benchmark_profiler.begin(&syndrome_pattern, &error_pattern);
                    solver.solve_visualizer(syndrome_pattern.clone(), visualizer.as_mut());
                    benchmark_profiler.event("decoded".to_string());
                    result_verifier.verify(&mut solver, &syndrome_pattern, &error_pattern, visualizer.as_mut(), seed);
                    benchmark_profiler.event("verified".to_string());
                    solver.clear(); // also count the clear operation

                    benchmark_profiler.end(Some(&*solver));

                    if solver.get_tuning_time().is_some() {
                        solver.clear_tuning_time();
                    }
                    if let Some(html_path) = &visualizer_html_filepath {
                        if let Some(visualizer) = visualizer.as_mut() {
                            visualizer.save_html(html_path);
                        }
                    }
                }

                // printing the total round time and total tuning time for benchmark purpose
                eprintln!("total resolve time {:?}", benchmark_profiler.sum_round_time);
                eprintln!("total tuning time {:?}", benchmark_profiler.sum_tuning_time);
            }
            Commands::MatrixSpeed(parameters) => {
                let MatrixSpeedParameters {
                    matrix_type,
                    width,
                    height,
                    one_density,
                    total_rounds,
                    deterministic_seed,
                } = parameters.clone();
                // fist generate the parity samples
                let mut samples = Vec::with_capacity(total_rounds);
                let deterministic_seed = deterministic_seed.unwrap_or_else(|| crate::mwpf::shim::rand::thread_rng().gen());
                let mut rng = DeterministicRng::seed_from_u64(deterministic_seed);
                for _ in 0..total_rounds {
                    let mut parity_checks: Vec<(Vec<usize>, bool)> = Vec::with_capacity(height);
                    for _ in 0..height {
                        parity_checks.push((
                            (0..width).filter(|_| rng.next_f64() < one_density).collect(),
                            rng.next_f64() < one_density,
                        ))
                    }
                    samples.push(parity_checks);
                }
                // call the matrix operation
                matrix_type.run(parameters, samples);
            }
            Commands::Test { command } => match command {
                TestCommands::Common => {
                    println!("[Common Test] Union-Find on Code Capacity Noise");
                    execute_in_cli(["".to_owned(), "test".to_owned(), "code-capacity".to_owned()].iter(), true);
                }
                TestCommands::CodeCapacity {
                    print_command,
                    enable_visualizer,
                    use_strict,
                    print_syndrome_pattern,
                    solver_type,
                    solver_config,
                } => {
                    let mut parameters = vec![];
                    let code_types = ["repetition", "planar", "tailored", "color"];

                    for code_type in code_types.iter() {
                        for p in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.499] {
                            for d in [3, 7, 11, 15, 19] {
                                parameters.push(vec![
                                    format!("{d}"),
                                    format!("{p}"),
                                    format!("--code-type"),
                                    format!("code-capacity-{code_type}-code"),
                                    format!("--pb-message"),
                                    format!("{code_type} {d} {p}"),
                                ]);
                            }
                        }
                    }

                    let command_head = vec![format!(""), format!("benchmark")];
                    let mut command_tail = vec!["--total-rounds".to_string(), format!("{TEST_EACH_ROUNDS}")];
                    command_tail.append(&mut vec![
                        format!("--verifier"),
                        if use_strict {
                            "strict-actual-error".to_string()
                        } else {
                            "actual-error".to_string()
                        },
                    ]);
                    if enable_visualizer {
                        command_tail.append(&mut vec![format!("--enable-visualizer")]);
                    }
                    if print_syndrome_pattern {
                        command_tail.append(&mut vec![format!("--print-syndrome-pattern")]);
                    }
                    command_tail.append(&mut vec![
                        format!("--solver-type"),
                        solver_type.name().to_string(),
                        format!("--solver-config"),
                        serde_json::to_string(&solver_config).unwrap(),
                    ]);
                    for parameter in parameters.iter() {
                        execute_in_cli(
                            command_head.iter().chain(parameter.iter()).chain(command_tail.iter()),
                            print_command,
                        );
                    }
                }
            },
            Commands::DecoderSpeed(DecoderSpeedParameters {
                file_path,
                solver_type,
                solver_config,
                use_bp,
                bp_application_ratio,
                bp_max_iter,
            }) => {
                if !file_path.ends_with("cbor") {
                    eprintln!(
                        "only support cbor file, the file path \"{file_path}\" does not end with cbor, operations may fail"
                    );
                }
                let BenchmarkSuite {
                    initializer,
                    syndrome_patterns,
                } = BenchmarkSuite::from_cbor(&file_path);

                // time construction time
                let start = std::time::Instant::now();
                let initializer = Arc::new(initializer);
                let mut decoder = solver_type.bench_build(&initializer, solver_config);
                if use_bp {
                    decoder = match SolverBPWrapper::new(
                        decoder.solver_base(),
                        bp_max_iter.unwrap_or(1),
                        bp_application_ratio.unwrap_or(0.1),
                    )
                    .solver
                    .inner
                    {
                        SolverEnum::SolverSerialUnionFind(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverSerialSingleHair(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverSerialJointSingleHair(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverErrorPatternLogger(_) => panic!("not supported"),
                    };
                }
                let construction_time = start.elapsed();
                eprintln!("construction time {:?}", construction_time);

                // time solving time
                let start = std::time::Instant::now();
                for syndrome_pattern in syndrome_patterns.into_iter() {
                    decoder.solve(syndrome_pattern);
                    decoder.clear();
                }
                let solving_time = start.elapsed();
                eprintln!("solving time {:?}", solving_time);
            }
            Commands::CustomTest => {
                // custom test: ported from test_override_weights_decoding
                let vertex_num = 3;
                let edge1 = HyperEdge {
                    vertices: vec![0, 1],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let edge2 = HyperEdge {
                    vertices: vec![1, 2],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let edge3 = HyperEdge {
                    vertices: vec![2, 0],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let initializer = SolverInitializer::new(vertex_num, vec![edge1, edge2, edge3]);
                let initializer = Arc::new(initializer);
                let solver: SolverSerialJointSingleHair = SolverSerialJointSingleHair::new(&initializer, json!({}));
                let mut solver = SolverBPWrapper::new(solver.solver_base(), 1, 0.625);

                // // force set wieghts to [0, 0, 100]
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![0.0.into(), 0.0.into(), 100.0.into()],
                //     1.0.into(),
                //     None,
                // ));

                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(bound.upper, 0.0);

                // // mix weights of ratio = 0.8
                // // 0.8 * [0, 0, 100] + 0.2 * [100, 100, 100] = [20, 20, 100]
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![0.0.into(), 0.0.into(), 100.0.into()],
                //     0.8.into(),
                //     None,
                // ));
                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(bound.upper, 40.0);

                // set to negative weight
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![(-20.0).into(), 10.0.into(), 100.0.into()],
                //     1.0.into(),
                //     None,
                // ));
                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.upper, (-10.0));

                solver.solve(SyndromePattern::new_vertices(vec![0, 2]));
                let (subgraph, bound) = solver.subgraph_range();
                println!("{:?}", subgraph);
                println!("({}, {})", bound.lower, bound.upper);
                assert_eq!(bound.lower, bound.upper);
                assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![2]);
                assert_eq!(bound.upper, Weight::from_float(100.0).unwrap());
            }
            Commands::CustomTest2 => {
                // custom test: ported from test_override_weights_decoding
                let vertex_num = 3;
                let edge1 = HyperEdge {
                    vertices: vec![0, 1],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let edge2 = HyperEdge {
                    vertices: vec![1, 2],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let edge3 = HyperEdge {
                    vertices: vec![2, 0],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let initializer = SolverInitializer::new(vertex_num, vec![edge1, edge2, edge3]);
                let initializer = Arc::new(initializer);
                let mut solver: SolverSerialJointSingleHair = SolverSerialJointSingleHair::new(&initializer, json!({}));
                // let mut solver = SolverBPWrapper::new(solver.solver_base(), 1, 0.625);

                // // force set wieghts to [0, 0, 100]
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![0.0.into(), 0.0.into(), 100.0.into()],
                //     1.0.into(),
                //     None,
                // ));

                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(bound.upper, 0.0);

                // // mix weights of ratio = 0.8
                // // 0.8 * [0, 0, 100] + 0.2 * [100, 100, 100] = [20, 20, 100]
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![0.0.into(), 0.0.into(), 100.0.into()],
                //     0.8.into(),
                //     None,
                // ));
                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(bound.upper, 40.0);

                // set to negative weight
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![(-20.0).into(), 10.0.into(), 100.0.into()],
                //     1.0.into(),
                //     None,
                // ));
                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.upper, (-10.0));

                solver.solve(SyndromePattern::new_vertices(vec![0, 2]));
                let (subgraph, bound) = solver.subgraph_range();
                println!("{:?}", subgraph);
                println!("({}, {})", bound.lower, bound.upper);
                assert_eq!(bound.lower, bound.upper);
                assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![2]);
                assert_eq!(bound.upper, Weight::from_float(100.0).unwrap());
            }
        }
    }
}

pub fn execute_in_cli<'a>(iter: impl Iterator<Item = &'a String> + Clone, print_command: bool) {
    if print_command {
        print!("[command]");
        for word in iter.clone() {
            if word.contains(char::is_whitespace) {
                print!("'{word}' ")
            } else {
                print!("{word} ")
            }
        }
        println!();
    }
    Cli::parse_from(iter.cloned()).run();
}

/// `more_asserts::assert_le!`
#[track_caller]
fn assert_le<T: PartialOrd + std::fmt::Debug>(left: T, right: T, message: std::fmt::Arguments) {
    if !(left <= right) {
        panic!("assertion failed: `(left <= right)`\n  left: `{left:?}`,\n right: `{right:?}`: {message}");
    }
}

impl ExampleCodeType {
    fn build(
        &self,
        d: VertexNum,
        p: f64,
        _noisy_measurements: VertexNum,
        mut code_config: serde_json::Value,
    ) -> Box<dyn ExampleCode> {
        match self {
            Self::CodeCapacityRepetitionCode => {
                assert_eq!(code_config, json!({}), "config not supported");
                Box::new(CodeCapacityRepetitionCode::new(d, p))
            }
            Self::CodeCapacityPlanarCode => {
                assert_eq!(code_config, json!({}), "config not supported");
                Box::new(CodeCapacityPlanarCode::new(d, p))
            }
            Self::CodeCapacityTailoredCode => {
                let mut pxy = 0.; // default to infinite bias
                let config = code_config.as_object_mut().expect("config must be JSON object");
                if let Some(value) = config.remove("pxy") {
                    pxy = value.as_f64().expect("code_count number");
                }
                Box::new(CodeCapacityTailoredCode::new(d, pxy, p))
            }
            Self::CodeCapacityColorCode => {
                assert_eq!(code_config, json!({}), "config not supported");
                Box::new(CodeCapacityColorCode::new(d, p))
            }
            Self::ErrorPatternReader => Box::new(ErrorPatternReader::new(code_config)),
            Self::QECPlaygroundCode => panic!("QEC-Playground's codes are not part of stabilizer_qec.mwpf (MWPF's wheels build them on the qecp crate)"),
        }
    }
}

impl SolverType {
    pub fn build(
        &self,
        initializer: &Arc<SolverInitializer>,
        code: &dyn ExampleCode,
        solver_config: serde_json::Value,
    ) -> Box<dyn SolverTrait> {
        match self {
            Self::UnionFind => Box::new(SolverSerialUnionFind::new(initializer, solver_config)),
            Self::SingleHair => Box::new(SolverSerialSingleHair::new(initializer, solver_config)),
            Self::JointSingleHair => Box::new(SolverSerialJointSingleHair::new(initializer, solver_config)),
            Self::ErrorPatternLogger => Box::new(SolverErrorPatternLogger::new(initializer, code, solver_config)),
        }
    }

    pub fn bench_build(
        &self,
        initializer: &Arc<SolverInitializer>,
        solver_config: serde_json::Value,
    ) -> Box<dyn SolverTrait> {
        match self {
            Self::UnionFind => Box::new(SolverSerialUnionFind::new(initializer, solver_config)),
            Self::SingleHair => Box::new(SolverSerialSingleHair::new(initializer, solver_config)),
            Self::JointSingleHair => Box::new(SolverSerialJointSingleHair::new(initializer, solver_config)),
            Self::ErrorPatternLogger => panic!("error pattern logger does not support decoder (stable) speed benchmark"),
        }
    }
}

impl Verifier {
    fn build(&self, initializer: &SolverInitializer) -> Box<dyn ResultVerifier> {
        match self {
            Self::None => Box::new(VerifierNone {}),
            Self::FusionSerial => Box::new(VerifierFusionSerial {
                initializer: initializer.clone(),
            }),
            Self::ActualError => Box::new(VerifierActualError {
                initializer: initializer.clone(),
                is_strict: false,
            }),
            Self::StrictActualError => Box::new(VerifierActualError {
                initializer: initializer.clone(),
                is_strict: true,
            }),
        }
    }
}

trait ResultVerifier {
    fn verify(
        &mut self,
        solver: &mut Box<dyn SolverTrait>,
        syndrome_pattern: &SyndromePattern,
        error_pattern: &Subgraph,
        visualizer: Option<&mut Visualizer>,
        seed: u64,
    );
}

struct VerifierNone {}

impl ResultVerifier for VerifierNone {
    fn verify(
        &mut self,
        _solver: &mut Box<dyn SolverTrait>,
        _syndrome_pattern: &SyndromePattern,
        _error_pattern: &Subgraph,
        _visualizer: Option<&mut Visualizer>,
        _seed: u64,
    ) {
    }
}

struct VerifierFusionSerial {
    pub initializer: SolverInitializer,
}

impl ResultVerifier for VerifierFusionSerial {
    fn verify(
        &mut self,
        _solver: &mut Box<dyn SolverTrait>,
        _syndrome_pattern: &SyndromePattern,
        _error_pattern: &Subgraph,
        _visualizer: Option<&mut Visualizer>,
        _seed: u64,
    ) {
        println!("{}", self.initializer.vertex_num);
        unimplemented!()
    }
}

struct VerifierActualError {
    initializer: SolverInitializer,
    pub is_strict: bool,
}

impl ResultVerifier for VerifierActualError {
    fn verify(
        &mut self,
        solver: &mut Box<dyn SolverTrait>,
        syndrome_pattern: &SyndromePattern,
        error_pattern: &Subgraph,
        visualizer: Option<&mut Visualizer>,
        seed: u64,
    ) {
        if !syndrome_pattern.erasures.is_empty()
            || !syndrome_pattern.heralds.is_empty()
            || syndrome_pattern.override_weights.is_some()
        {
            unimplemented!()
        }
        let actual_weight = if error_pattern.is_empty() && !syndrome_pattern.defect_vertices.is_empty() {
            // error pattern is not generated by the simulator
            Rational::from_usize(usize::MAX).unwrap()
        } else {
            Rational::from(
                self.initializer
                    .get_subgraph_total_weight(&OutputSubgraph::new(error_pattern.clone(), Default::default())),
            )
        };
        let (subgraph, weight_range) = solver.subgraph_range_visualizer(visualizer);

        // solver.print_clusters();
        // assert!(
        //     self.initializer
        //         .matches_subgraph_syndrome(&subgraph, &syndrome_pattern.defect_vertices),
        //     "bug: the result subgraph does not match the syndrome || the seed is {seed:?}"
        // );
        assert_le(
            &weight_range.lower,
            &actual_weight,
            format_args!("bug: the lower bound of weight range is larger than the actual weight || the seed is {seed:?}"),
        );
        if self.is_strict {
            let subgraph_weight = Rational::from(self.initializer.get_subgraph_total_weight(&subgraph));
            assert_le(&subgraph_weight, &actual_weight, format_args!("it's not a minimum-weight parity subgraph: the actual error pattern has smaller weight, range: {weight_range:?}"));
            assert_eq!(
                weight_range.lower, weight_range.upper,
                "the weight range must be optimal: lower = upper || the seed is {seed:?}"
            );
        }
    }
}
