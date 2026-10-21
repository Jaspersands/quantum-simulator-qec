// Copyright 2022 PyMatching Contributors
// Licensed under the Apache License, Version 2.0 (see src/pm_exact/LICENSE). Modified: transliterated
// to Rust for this crate.

//! PyMatching's `animate` drawing (`diagram/mwpm_diagram.cc`): the detectors' coordinates
//! projected to the page (`pick_coords_for_drawing_from_dem`), the decoder's state as an SVG
//! (`write_decoder_state_as_svg`), and the run drawn frame by frame
//! (`write_animated_decoding_svg_frames`), each frame handed to a sink.
//!
//! The drawing is PyMatching's line for line: the graph's edges grey (boundary edges dashed),
//! detection events red, each region's area traced as a polygon per detection event it grew from
//! (red growing, green matched, dark green matched to the boundary, blue shrinking), pending
//! collisions orange, alternating-tree edges black (dashed to children), blossom cycles blue,
//! matches thick black, and the event about to be handled highlighted. Numbers are `float`s as
//! C++'s stream prints them (`%g`, 6 digits), computed as PyMatching's macOS arm64 wheel computes
//! them, which its recorded frames pin down: `a * b + c * d` in one expression has its left
//! product fused with the sum (clang's contraction) on aarch64, in `lerp_pos` and in the boundary
//! direction's length, and is rounded twice on x86-64, where C++ builds have no fused
//! multiply-add by default; polygon corners are sorted by angle with the platform's `std::sort`
//! (`cpp_random::Library::NATIVE`, whose order among equal angles is its own) and `atan2f`. (A
//! newer clang, Apple clang 21, builds PyMatching's program without the fusion, and some corners
//! then differ in their sixth digit.)
//!
//! PyMatching draws regions (and their tree nodes) in heap-address order
//! (`std::set<GraphFillRegion *>`); here they are drawn in arena-slot order. A frame's lines are
//! PyMatching's, in another order.
//!
//! Where the queue runs dry before every detection event's region is frozen (no perfect
//! matching), PyMatching's loop hands the empty event to the flooder, which throws `Unknown
//! tentative event type.`; so does this, after the frames before it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use super::flooder::MatcherEvent;
use super::graph::NONE;
use super::tracker::Varying;
use super::{Mwpm, MwpmEvent};
use crate::cpp_random::Library;

/// What PyMatching's animation throws when its queue runs dry first.
pub const UNKNOWN_TENTATIVE_EVENT: &str = "Unknown tentative event type.";

/// A point on the page: PyMatching's `std::pair<float, float>`.
pub type Point = (f32, f32);

/// `a * b + c` written in one C++ expression (`c` another product or a sum's other term), as
/// PyMatching's builds compute it: one rounding on aarch64, two on x86-64
/// (`cpp_random::contracted`, in `float`).
#[inline]
fn contracted32(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(target_arch = "aarch64") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// `std::min(a, b)`: `b < a ? b : a`.
#[inline]
fn cpp_min(a: f32, b: f32) -> f32 {
    if b < a {
        b
    } else {
        a
    }
}

/// `std::max(a, b)`: `a < b ? b : a`.
#[inline]
fn cpp_max(a: f32, b: f32) -> f32 {
    if a < b {
        b
    } else {
        a
    }
}

/// `ostream << float`: the value as a double at the stream's precision, 6 (`%g`).
fn put(out: &mut String, x: f32) {
    let x = x as f64;
    if x.is_nan() {
        out.push_str("nan");
    } else if x == 0.0 && x.is_sign_negative() {
        out.push_str("-0");
    } else {
        out.push_str(&crate::dem_program::fmt_g(x, 6));
    }
}

/// `pick_coords_for_drawing_from_dem`: each detector's first two coordinates (its index, or
/// its one coordinate, and 0 where it has fewer), the rest projected onto them; a point 5 units
/// out from the average for its boundary edge; all shifted to start at 0, scaled by
/// `pixels_per_unit_length`, and offset by 4. `detector_coordinates` is the model's
/// `get_detector_coordinates()`, detector by detector (empty for none).
pub fn pick_coords_for_drawing(detector_coordinates: &[Vec<f64>], pixels_per_unit_length: f32) -> (Vec<Point>, Vec<Point>) {
    let n = detector_coordinates.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut coords: Vec<Point> = Vec::with_capacity(n);
    for (k, cs) in detector_coordinates.iter().enumerate() {
        if cs.is_empty() {
            coords.push((k as f32, 0.0));
            continue;
        }
        let mut c = if cs.len() == 1 { (cs[0] as f32, 0.0) } else { (cs[0] as f32, cs[1] as f32) };
        // An arbitrary orthographic projection of the other axes (in double, then stored).
        let mut s = 1.0f64;
        for (d, &v) in cs.iter().enumerate().skip(2) {
            s *= 0.66;
            c.0 = crate::cpp_random::contracted(v, s, c.0 as f64) as f32;
            c.1 = (c.1 as f64 + v * s / (d + 1) as f64) as f32;
        }
        coords.push(c);
    }
    let (mut avg_x, mut avg_y) = (0f32, 0f32);
    for c in &coords {
        avg_x += c.0;
        avg_y += c.1;
    }
    avg_x /= n as f32;
    avg_y /= n as f32;
    let mut boundary: Vec<Point> = Vec::with_capacity(n);
    for c in &coords {
        let (mut dx, mut dy) = (c.0 - avg_x, c.1 - avg_y);
        let mut d = contracted32(dx, dx, dy * dy).sqrt();
        if (d as f64) < 1e-5 {
            dx = 11.0;
            dy = 7.0;
            d = contracted32(dx, dx, dy * dy).sqrt();
        }
        dx /= d;
        dy /= d;
        dx *= 5.0;
        dy *= 5.0;
        boundary.push((c.0 + dx, c.1 + dy));
    }
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    for k in 0..n {
        min_x = cpp_min(min_x, cpp_min(coords[k].0, boundary[k].0));
        min_y = cpp_min(min_y, cpp_min(coords[k].1, boundary[k].1));
    }
    for k in 0..n {
        coords[k].0 -= min_x;
        boundary[k].0 -= min_x;
        coords[k].1 -= min_y;
        boundary[k].1 -= min_y;
    }
    for k in 0..n {
        coords[k].0 *= pixels_per_unit_length;
        coords[k].1 *= pixels_per_unit_length;
        boundary[k].0 *= pixels_per_unit_length;
        boundary[k].1 *= pixels_per_unit_length;
        coords[k].0 += 4.0;
        coords[k].1 += 4.0;
        boundary[k].0 += 4.0;
        boundary[k].1 += 4.0;
    }
    (coords, boundary)
}

/// `StateHelper`: one frame's drawing.
struct State<'a> {
    m: &'a Mwpm,
    t: i64,
    coords: &'a [Point],
    boundary_coords: &'a [Point],
    out: &'a mut String,
}

impl State<'_> {
    fn neighbor_coords(&self, src: u32, dst: u32) -> Point {
        if dst != NONE {
            self.coords[dst as usize]
        } else {
            self.boundary_coords[src as usize]
        }
    }

    fn lerp_pos(&self, src: u32, local_neighbor_index: usize, mut ratio: f32) -> Point {
        if ratio > 1.0 {
            ratio = 1.0;
        }
        let c = self.coords[src as usize];
        let nc = self.neighbor_coords(src, self.m.graph().neighbors(src)[local_neighbor_index]);
        (contracted32(nc.0, ratio, c.0 * (1.0 - ratio)), contracted32(nc.1, ratio, c.1 * (1.0 - ratio)))
    }

    /// `approximate_region_polygons`: per detection event the region grew from (in node order),
    /// the corners where its area ends, deduplicated and sorted by angle about their average.
    fn approximate_region_polygons(&self, region: u32) -> Vec<Vec<Point>> {
        let m = self.m;
        let g = m.graph();
        let mut points_by_source: BTreeMap<i64, Vec<Point>> = BTreeMap::new();
        for n in m.total_area(region) {
            let src = m.node(n).reached_from_source;
            let perimeter = points_by_source.entry(if src == NONE { -1 } else { src as i64 }).or_default();
            if m.compute_local_radius_at_time_bounded_by_region(n, self.t, region) == 0 {
                perimeter.push(self.coords[n as usize]);
                continue;
            }
            for (nk, &w) in g.neighbor_weights(n).iter().enumerate() {
                let Some(r) = m.compute_stitch_radius_at_time_bounded_by_region_towards_neighbor(n, self.t, region, nk) else { continue };
                let ratio = if w == 0 { 0.0 } else { r / w as f32 };
                perimeter.push(self.lerp_pos(n, nk, ratio));
            }
        }
        let mut result = Vec::new();
        for (_, mut p) in points_by_source {
            // A `std::set<std::pair<float, float>>`: in order, each point once.
            p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            p.dedup();
            if p.is_empty() {
                continue;
            }
            let (mut cx, mut cy) = (0f32, 0f32);
            for e in &p {
                cx += e.0;
                cy += e.1;
            }
            cx /= p.len() as f32;
            cy /= p.len() as f32;
            Library::NATIVE.sort_by(&mut p, &mut |a: &Point, b: &Point| (a.1 - cy).atan2(a.0 - cx) < (b.1 - cy).atan2(b.0 - cx));
            result.push(p);
        }
        result
    }

    fn fill_region(&mut self, region: u32, fill: &str, stroke: &str) {
        for poly in self.approximate_region_polygons(region) {
            self.out.push_str(" <path d=\"");
            for (k, p) in poly.iter().enumerate() {
                self.out.push(if k == 0 { 'M' } else { 'L' });
                put(self.out, p.0);
                self.out.push(' ');
                put(self.out, p.1);
                self.out.push(' ');
            }
            let _ = writeln!(self.out, " z\" stroke=\"{stroke}\" fill=\"{fill}\"/>");
        }
    }

    fn line(&mut self, a: Point, b: Point, attrs: &str) {
        self.out.push_str(" <line x1=\"");
        put(self.out, a.0);
        self.out.push_str("\" x2=\"");
        put(self.out, b.0);
        self.out.push_str("\" y1=\"");
        put(self.out, a.1);
        self.out.push_str("\" y2=\"");
        put(self.out, b.1);
        self.out.push_str("\" ");
        self.out.push_str(attrs);
        self.out.push_str("/>\n");
    }

    fn circle(&mut self, c: Point, attrs: &str) {
        self.out.push_str(" <circle cx=\"");
        put(self.out, c.0);
        self.out.push_str("\" cy=\"");
        put(self.out, c.1);
        self.out.push_str("\" r=\"3\" ");
        self.out.push_str(attrs);
        self.out.push_str("/>\n");
    }

    fn draw_detector_graph_edges(&mut self) {
        let g = self.m.graph();
        for k in 0..g.num_nodes as u32 {
            for &n2 in g.neighbors(k) {
                let nc = self.neighbor_coords(k, n2);
                if n2 != NONE {
                    if n2 > k {
                        self.line(self.coords[k as usize], nc, "stroke=\"#CCC\"");
                    }
                } else {
                    self.line(self.coords[k as usize], nc, "stroke-dasharray=\"4 2\" stroke=\"#FAA\"");
                }
            }
        }
    }

    fn draw_unexcited_detector_nodes(&mut self) {
        for k in 0..self.m.graph().num_nodes as u32 {
            if self.m.node(k).reached_from_source != k {
                self.circle(self.coords[k as usize], "stroke=\"none\" fill=\"#CCC\"");
            }
        }
    }

    /// `find_all_regions`: the regions that reached a node, and their blossom ancestors.
    fn find_all_regions(&self) -> BTreeSet<u32> {
        let mut regions = BTreeSet::new();
        for k in 0..self.m.graph().num_nodes as u32 {
            let mut r = self.m.node(k).region_that_arrived;
            while r != NONE {
                regions.insert(r);
                r = self.m.region(r).blossom_parent;
            }
        }
        regions
    }

    fn draw_detection_events(&mut self) {
        for k in 0..self.m.graph().num_nodes as u32 {
            if self.m.node(k).reached_from_source == k {
                self.circle(self.coords[k as usize], "stroke=\"none\" fill=\"red\"");
            }
        }
    }

    fn draw_pending_collisions(&mut self) {
        let (m, t) = (self.m, self.t);
        let g = m.graph();
        for k in 0..g.num_nodes as u32 {
            let n = m.node(k);
            if n.region_that_arrived_top == NONE {
                continue;
            }
            let Some((i, time)) = m.next_event_at_node(k) else { continue };
            if time != t {
                continue;
            }
            let r1 = m.local_radius(k);
            let other = g.neighbors(k)[i];
            let r2 = if other == NONE { Varying(0) } else { m.local_radius(other) };
            // `colliding_with`
            if ((r1.0 | r2.0) & 3) != 1 {
                continue;
            }
            let w = g.neighbor_weights(k)[i];
            if r1.get_distance_at_time(t) + r2.get_distance_at_time(t) != w as i64 {
                continue;
            }
            if other != NONE && (m.node(other).region_that_arrived_top == NONE || m.node(other).region_that_arrived_top == n.region_that_arrived_top) {
                continue;
            }
            let col = self.lerp_pos(k, i, r1.get_distance_at_time(t) as f32 / cpp_max(1.0, w as f32));
            self.circle(col, "stroke=\"black\" fill=\"orange\"");
        }
    }

    fn draw_match_edges(&mut self, regions: &BTreeSet<u32>) {
        for &r in regions {
            let reg = self.m.region(r);
            if reg.blossom_parent_top == r && reg.radius.is_frozen() {
                let e = reg.matched.edge;
                if e.loc_from == NONE {
                    continue; // (a frozen top region is always matched)
                }
                let to = if e.loc_to == NONE { self.boundary_coords[e.loc_from as usize] } else { self.coords[e.loc_to as usize] };
                self.line(self.coords[e.loc_from as usize], to, "stroke-width=\"4\" stroke=\"#000\"");
            }
        }
    }

    fn draw_alt_tree_edges(&mut self, regions: &BTreeSet<u32>) {
        let tree_nodes: BTreeSet<u32> = regions.iter().map(|&r| self.m.region(r).alt_tree_node).filter(|&a| a != NONE).collect();
        for a in tree_nodes {
            let tn = self.m.alt_tree_node(a);
            if tn.inner_region != NONE {
                let e = tn.inner_to_outer_edge;
                self.line(self.coords[e.loc_from as usize], self.coords[e.loc_to as usize], "stroke=\"#000\"");
            }
            for child in &tn.children {
                let e = child.edge;
                self.line(self.coords[e.loc_from as usize], self.coords[e.loc_to as usize], "stroke-dasharray=\"4 2\" stroke=\"#000\"");
            }
        }
    }

    fn draw_blossom_edges(&mut self, regions: &BTreeSet<u32>) {
        for &r in regions {
            for c in &self.m.region(r).blossom_children {
                self.line(self.coords[c.edge.loc_from as usize], self.coords[c.edge.loc_to as usize], "stroke-width=\"4\" stroke=\"#22F\"");
            }
        }
    }

    /// `span`: the page's size, the farthest point rounded up to a multiple of 4, plus 4.
    fn span(&self) -> (usize, usize) {
        let (mut width, mut height) = (0f32, 0f32);
        for c in self.coords.iter().chain(self.boundary_coords) {
            width = cpp_max(width, c.0);
            height = cpp_max(height, c.1);
        }
        ((width / 4.0).ceil() as usize * 4 + 4, (height / 4.0).ceil() as usize * 4 + 4)
    }
}

/// `write_decoder_state_as_svg`: the decoder's state at its clock, with `focused_event` (an
/// event about to be handled) highlighted.
pub fn write_decoder_state_as_svg(coords: &[Point], boundary_coords: &[Point], m: &Mwpm, focused_event: Option<MatcherEvent>, out: &mut String) {
    let mut s = State { m, t: m.cur_time(), coords, boundary_coords, out };
    let (w, h) = s.span();
    let _ = writeln!(s.out, "<svg width=\"{w}\" height=\"{h}\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\">");
    let _ = writeln!(s.out, " <path d=\"M 0 0 L {w} 0 L {w} {h} L 0 {h} Z\" fill=\"white\" stroke=\"#CCC\"/>");
    s.draw_detector_graph_edges();
    s.draw_unexcited_detector_nodes();
    let regions = s.find_all_regions();
    for &r in &regions {
        let top = m.region(m.region(r).blossom_parent_top);
        let color = if top.radius.is_growing() {
            "#FF000080"
        } else if top.radius.is_frozen() {
            if top.matched.edge.loc_to == NONE {
                "#00200080"
            } else {
                "#00FF0080"
            }
        } else {
            "#0000FF80"
        };
        s.fill_region(r, color, "none");
    }
    s.draw_detection_events();
    s.draw_pending_collisions();
    s.draw_alt_tree_edges(&regions);
    s.draw_blossom_edges(&regions);
    s.draw_match_edges(&regions);
    match focused_event {
        Some(MatcherEvent::RegionHitRegion { edge, .. }) => {
            let (a, b) = (s.coords[edge.loc_from as usize], s.coords[edge.loc_to as usize]);
            s.line(a, b, "stroke-width=\"15\" stroke=\"#000\"");
            s.line(a, b, "stroke-width=\"12\" stroke=\"#F80\"");
        }
        Some(MatcherEvent::RegionHitBoundary { region, .. }) => s.fill_region(region, "#80800080", "#000"),
        Some(MatcherEvent::BlossomShatter { blossom_region, .. }) => s.fill_region(blossom_region, "#FF800080", "#FF8000"),
        None => {}
    }
    s.out.push_str("</svg>\n");
}

/// The frame counts and pacing of `write_animated_decoding_svg_frames`.
#[derive(Clone, Copy, Debug)]
pub struct FrameSchedule {
    /// Frames spent on each event the matcher handles (0: none, and no pause).
    pub frames_per_mwpm_event: usize,
    /// Frames of the detection events alone, before anything grows.
    pub hold_frames_at_start: usize,
    /// Frames of the final state.
    pub hold_frames_at_end: usize,
    /// The growth between in-between frames (0: none, except one after each event's frames).
    pub max_growth_between_frames: usize,
}

/// Why `write_animated_decoding_svg_frames` stopped before the end.
#[derive(Debug)]
pub enum AnimationError<E> {
    /// The sink refused a frame.
    Frame(E),
    /// The queue ran dry first (`UNKNOWN_TENTATIVE_EVENT`).
    UnknownTentativeEvent,
}

/// `write_animated_decoding_svg_frames`: a region at each detection event, then the run step by
/// step, each frame handed to `emit` as drawn: the held start frames; an in-between frame each
/// time the clock passes the next multiple of `max_growth_between_frames` after the last
/// drawn event (drawn at that time); `frames_per_mwpm_event` frames before the matcher handles
/// each event the flooder finds; and once every detection event's region is frozen, the held end
/// frames. `m` must be fresh.
pub fn write_animated_decoding_svg_frames<E>(
    m: &mut Mwpm,
    coords: &[Point],
    boundary_coords: &[Point],
    detection_events: &[u64],
    schedule: FrameSchedule,
    mut emit: impl FnMut(&str) -> Result<(), E>,
) -> Result<(), AnimationError<E>> {
    m.begin_animation(detection_events);
    let mut out = String::new();
    let mut frame = |m: &Mwpm, ev: Option<MatcherEvent>| -> Result<(), AnimationError<E>> {
        out.clear();
        write_decoder_state_as_svg(coords, boundary_coords, m, ev, &mut out);
        emit(&out).map_err(AnimationError::Frame)
    };
    for _ in 0..schedule.hold_frames_at_start {
        frame(&*m, None)?;
    }
    let growth = schedule.max_growth_between_frames as i64;
    let mut next_draw_time = growth;
    let mut just_drew_interesting_event = false;
    while !m.all_frozen(detection_events) {
        // Advance time (a dry queue leaves the clock where it is).
        let dequeued = m.step();
        // In-between growth frames.
        while (just_drew_interesting_event || growth != 0) && m.cur_time() > next_draw_time {
            let t = m.cur_time();
            m.set_cur_time(next_draw_time);
            let drawn = frame(&*m, None);
            m.set_cur_time(t);
            drawn?;
            next_draw_time += growth;
            just_drew_interesting_event = false;
        }
        if dequeued.is_none() {
            // PyMatching's NO_FLOOD_CHECK_EVENT passes `dequeue_decision` and reaches the default
            // case of `process_tentative_event_returning_mwpm_event`.
            return Err(AnimationError::UnknownTentativeEvent);
        }
        // Was it interesting?
        let Some(MwpmEvent::Found(ev)) = m.step() else { continue };
        for _ in 0..schedule.frames_per_mwpm_event {
            frame(&*m, Some(ev))?;
            next_draw_time = m.cur_time();
            just_drew_interesting_event = true;
        }
        // The state after it.
        m.step();
    }
    for _ in 0..schedule.hold_frames_at_end {
        frame(&*m, None)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pm_exact::{Graph, NeighbourOrder};
    use crate::user_graph::{Merge, UserGraph};

    #[test]
    fn coordinates() {
        // No coordinates (its index), one, two, and four (projected); one detector at the average.
        let (c, b) = pick_coords_for_drawing(&[vec![], vec![2.0], vec![1.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]], 10.0);
        assert_eq!(c.len(), 4);
        assert_eq!(b.len(), 4);
        let all = c.iter().chain(&b);
        assert!(all.clone().all(|p| p.0 >= 4.0 && p.1 >= 4.0));
        assert!(all.clone().any(|p| p.0 == 4.0) && all.clone().any(|p| p.1 == 4.0));
        let (c, b) = pick_coords_for_drawing(&[vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]], 1.0);
        // The middle detector is at the average: its boundary is 5 units towards (11, 7).
        let (dx, dy) = (b[1].0 - c[1].0, b[1].1 - c[1].1);
        assert!(((dx * dx + dy * dy).sqrt() - 5.0).abs() < 1e-5 && (dy / dx - 7.0 / 11.0).abs() < 1e-5);
        assert_eq!(pick_coords_for_drawing(&[], 4.0), (vec![], vec![]));
    }

    #[test]
    fn numbers_print_as_cpp_streams_do() {
        let mut s = String::new();
        for x in [4.0f32, 12.5, 184.76953, 1234567.0, 0.0001, 0.00001, -0.0] {
            put(&mut s, x);
            s.push(' ');
        }
        assert_eq!(s, "4 12.5 184.77 1.23457e+06 0.0001 1e-05 -0 ");
    }

    fn line(n: usize) -> Graph {
        let mut g = UserGraph::new(n, 0);
        g.add_boundary_edge(0, &[], 1.0, -1.0, Merge::Disallow).unwrap();
        for k in 0..n - 1 {
            g.add_edge(k, k + 1, &[], 1.0 + k as f64, -1.0, Merge::Disallow).unwrap();
        }
        Graph::from_user_graph_with(&g, NeighbourOrder::V2_4, 1000)
    }

    fn frames(g: Graph, dets: &[u64], schedule: FrameSchedule) -> (Vec<String>, Result<(), AnimationError<()>>) {
        let n = g.num_nodes;
        let coords: Vec<Vec<f64>> = (0..n).map(|k| vec![k as f64, 0.0]).collect();
        let (c, b) = pick_coords_for_drawing(&coords, 20.0);
        let mut m = Mwpm::new(g);
        let mut out = Vec::new();
        let r = write_animated_decoding_svg_frames(&mut m, &c, &b, dets, schedule, |s| {
            out.push(s.to_string());
            Ok(())
        });
        (out, r)
    }

    #[test]
    fn a_run_drawn() {
        let schedule = FrameSchedule { frames_per_mwpm_event: 2, hold_frames_at_start: 3, hold_frames_at_end: 4, max_growth_between_frames: 50 };
        let (out, r) = frames(line(6), &[1, 4], schedule);
        assert!(r.is_ok());
        assert!(out.len() > 3 + 4 + 2);
        assert!(out.iter().all(|f| f.starts_with("<svg width=") && f.ends_with("</svg>\n")));
        // The start shows the detection events only; the end, the matches.
        assert_eq!(out[0].matches("fill=\"red\"").count(), 2);
        assert!(!out[0].contains("<path d=\"M") || out[0].matches("<path").count() == 1 + 2);
        assert!(out.last().unwrap().contains("stroke-width=\"4\" stroke=\"#000\""));
        assert!(out.iter().any(|f| f.contains("stroke=\"#F80\"")));
        // No detection events: the held frames alone.
        let (out, r) = frames(line(3), &[], schedule);
        assert!(r.is_ok() && out.len() == 7);
    }

    #[test]
    fn a_dry_queue_is_refused_after_its_frames() {
        // One detection event and no boundary: the queue runs dry before the region freezes.
        // (Integral weights are kept, doubled: the region reaches node 1 at time 2, node 2 at 6.)
        let mut g = UserGraph::new(3, 0);
        g.add_edge(0, 1, &[], 1.0, -1.0, Merge::Disallow).unwrap();
        g.add_edge(1, 2, &[], 2.0, -1.0, Merge::Disallow).unwrap();
        let schedule = FrameSchedule { frames_per_mwpm_event: 1, hold_frames_at_start: 1, hold_frames_at_end: 1, max_growth_between_frames: 1 };
        let (out, r) = frames(Graph::from_user_graph_with(&g, NeighbourOrder::V2_4, 1000), &[0], schedule);
        assert!(matches!(r, Err(AnimationError::UnknownTentativeEvent)));
        // The start frame, the growth at times 1 to 5, and no end frames.
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn a_refused_frame_stops_the_run() {
        let schedule = FrameSchedule { frames_per_mwpm_event: 1, hold_frames_at_start: 2, hold_frames_at_end: 1, max_growth_between_frames: 10 };
        let g = line(4);
        let (c, b) = pick_coords_for_drawing(&[vec![], vec![], vec![], vec![]], 8.0);
        let mut m = Mwpm::new(g);
        let mut seen = 0;
        let r = write_animated_decoding_svg_frames(&mut m, &c, &b, &[0, 3], schedule, |_| {
            seen += 1;
            if seen == 2 {
                Err("no room")
            } else {
                Ok(())
            }
        });
        assert!(matches!(r, Err(AnimationError::Frame("no room"))));
        assert_eq!(seen, 2);
    }
}
