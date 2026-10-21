// Copyright 2022 PyMatching Contributors
// Licensed under the Apache License, Version 2.0 (see src/pm_exact/LICENSE). Modified: transliterated
// to Rust for this crate.

//! The matcher: the alternating tree (`matcher/alternating_tree.{h,cc}`) and `Mwpm`'s event
//! handling and blossom shattering (`matcher/mwpm.cc`), in PyMatching's order. Recursions that
//! can run as deep as a tree are loops here, visiting in the same order.

use super::flooder::{CompressedEdge, MatcherEvent, RegionEdge};
use super::graph::NONE;
use super::Mwpm;

/// `AltTreeEdge`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AltTreeEdge {
    pub alt_tree_node: u32,
    pub edge: CompressedEdge,
}

impl AltTreeEdge {
    pub const EMPTY: AltTreeEdge = AltTreeEdge { alt_tree_node: NONE, edge: CompressedEdge::EMPTY };
}

impl Default for AltTreeEdge {
    fn default() -> Self {
        AltTreeEdge::EMPTY
    }
}

/// `AltTreeNode`: a shrinking inner region (`NONE` at the root) and the growing outer region it
/// is matched to, with the tree edges.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AltTreeNode {
    pub inner_region: u32,
    pub outer_region: u32,
    pub inner_to_outer_edge: CompressedEdge,
    pub parent: AltTreeEdge,
    pub children: Vec<AltTreeEdge>,
    pub visited: bool,
}

/// `MatchingResult`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatchingResult {
    pub obs_mask: u64,
    pub weight: i64,
}

/// `unstable_erase`: the first child on `node` swapped out for the last.
#[inline]
fn unstable_erase(children: &mut Vec<AltTreeEdge>, node: u32) {
    if let Some(k) = children.iter().position(|c| c.alt_tree_node == node) {
        children.swap_remove(k);
    }
}

impl Mwpm {
    /// A new `AltTreeNode(inner, outer, edge)` (inner may be `NONE`), pointed to by its regions.
    pub(crate) fn alloc_alt_tree_node(&mut self, inner: u32, outer: u32, inner_to_outer_edge: CompressedEdge) -> u32 {
        let a = self.alt.alloc(AltTreeNode::default);
        let n = &mut self.alt.items[a as usize];
        n.inner_region = inner;
        n.outer_region = outer;
        n.inner_to_outer_edge = inner_to_outer_edge;
        n.parent = AltTreeEdge::EMPTY;
        n.children.clear();
        n.visited = false;
        if inner != NONE {
            self.regions.items[inner as usize].alt_tree_node = a;
        }
        self.regions.items[outer as usize].alt_tree_node = a;
        a
    }

    /// `AltTreeNode::add_child`.
    #[inline]
    pub(crate) fn add_child(&mut self, parent: u32, child: AltTreeEdge) {
        self.alt.items[parent as usize].children.push(child);
        self.alt.items[child.alt_tree_node as usize].parent = AltTreeEdge { alt_tree_node: parent, edge: child.edge.reversed() };
    }

    /// `AltTreeNode::become_root`: the rotations run from the root down, as PyMatching's
    /// recursion unwinds. As there, the moved inner regions keep their old tree pointers until
    /// the tree is shattered.
    pub(crate) fn become_root(&mut self, node: u32) {
        let mut path = std::mem::take(&mut self.stack);
        path.clear();
        let mut cur = node;
        path.push(cur);
        while self.alt.items[cur as usize].parent.alt_tree_node != NONE {
            cur = self.alt.items[cur as usize].parent.alt_tree_node;
            path.push(cur);
        }
        for j in (0..path.len() - 1).rev() {
            let (n, p) = (path[j], path[j + 1]);
            let inner = self.alt.items[n as usize].inner_region;
            let parent_edge = self.alt.items[n as usize].parent.edge;
            let pn = &mut self.alt.items[p as usize];
            pn.inner_region = inner;
            pn.inner_to_outer_edge = parent_edge;
            unstable_erase(&mut pn.children, n);
            let nn = &mut self.alt.items[n as usize];
            nn.inner_region = NONE;
            nn.parent = AltTreeEdge::EMPTY;
            let e = nn.inner_to_outer_edge.reversed();
            self.add_child(n, AltTreeEdge { alt_tree_node: p, edge: e });
            self.alt.items[n as usize].inner_to_outer_edge = CompressedEdge::EMPTY;
        }
        self.stack = path;
    }

    /// `AltTreeNode::most_recent_common_ancestor` (`NONE`: different trees).
    pub(crate) fn most_recent_common_ancestor(&mut self, this: u32, other: u32) -> u32 {
        let alt = &mut self.alt.items;
        let mut this_current = this;
        alt[this_current as usize].visited = true;
        let mut other_current = other;
        alt[other_current as usize].visited = true;
        let common_ancestor;
        loop {
            let this_parent = alt[this_current as usize].parent.alt_tree_node;
            let other_parent = alt[other_current as usize].parent.alt_tree_node;
            if this_parent != NONE || other_parent != NONE {
                if this_parent != NONE {
                    this_current = this_parent;
                    if alt[this_current as usize].visited {
                        common_ancestor = this_current;
                        break;
                    }
                    alt[this_current as usize].visited = true;
                }
                if other_parent != NONE {
                    other_current = other_parent;
                    if alt[other_current as usize].visited {
                        common_ancestor = other_current;
                        break;
                    }
                    alt[other_current as usize].visited = true;
                }
            } else {
                return NONE;
            }
        }
        alt[common_ancestor as usize].visited = false;
        let mut p = alt[common_ancestor as usize].parent.alt_tree_node;
        while p != NONE && alt[p as usize].visited {
            alt[p as usize].visited = false;
            p = alt[p as usize].parent.alt_tree_node;
        }
        common_ancestor
    }

    /// `AltTreeNode::prune_upward_path_stopping_before`: (orphan edges, the path's region edges).
    pub(crate) fn prune_upward_path_stopping_before(&mut self, node: u32, prune_parent: u32, back: bool) -> (Vec<AltTreeEdge>, Vec<RegionEdge>) {
        let mut orphan_edges: Vec<AltTreeEdge> = Vec::new();
        let mut path: Vec<RegionEdge> = Vec::new();
        let mut cur = node;
        while cur != prune_parent {
            let c = &mut self.alt.items[cur as usize];
            if orphan_edges.is_empty() {
                orphan_edges = std::mem::take(&mut c.children);
            } else {
                orphan_edges.append(&mut c.children);
            }
            let (inner, outer, i2o, parent) = (c.inner_region, c.outer_region, c.inner_to_outer_edge, c.parent);
            if back {
                path.push(RegionEdge { region: inner, edge: i2o });
                path.push(RegionEdge { region: self.alt.items[parent.alt_tree_node as usize].outer_region, edge: parent.edge.reversed() });
            } else {
                path.push(RegionEdge { region: outer, edge: i2o.reversed() });
                path.push(RegionEdge { region: inner, edge: parent.edge });
            }
            unstable_erase(&mut self.alt.items[parent.alt_tree_node as usize].children, cur);
            self.regions.items[outer as usize].alt_tree_node = NONE;
            self.regions.items[inner as usize].alt_tree_node = NONE;
            self.alt.del(cur);
            cur = parent.alt_tree_node;
        }
        (orphan_edges, path)
    }

    pub(crate) fn make_child(&mut self, parent: u32, inner: u32, outer: u32, inner_to_outer_edge: CompressedEdge, edge: CompressedEdge) -> u32 {
        let child = self.alloc_alt_tree_node(inner, outer, inner_to_outer_edge);
        self.add_child(parent, AltTreeEdge { alt_tree_node: child, edge });
        child
    }

    /// `shatter_descendants_into_matches_and_freeze`: children first, in order, then the node.
    pub(crate) fn shatter_descendants_into_matches_and_freeze(&mut self, root: u32) {
        let mut stack: Vec<(u32, usize)> = vec![(root, 0)];
        while let Some(top) = stack.last_mut() {
            let (n, i) = *top;
            if i < self.alt.items[n as usize].children.len() {
                top.1 += 1;
                let c = self.alt.items[n as usize].children[i].alt_tree_node;
                stack.push((c, 0));
                continue;
            }
            stack.pop();
            let (inner, outer, i2o) = {
                let a = &self.alt.items[n as usize];
                (a.inner_region, a.outer_region, a.inner_to_outer_edge)
            };
            if inner != NONE {
                self.alt.items[n as usize].parent = AltTreeEdge::EMPTY;
                self.add_match(inner, outer, i2o);
                self.set_region_frozen(inner);
                self.set_region_frozen(outer);
                self.regions.items[inner as usize].alt_tree_node = NONE;
                self.regions.items[outer as usize].alt_tree_node = NONE;
            }
            if outer != NONE {
                self.regions.items[outer as usize].alt_tree_node = NONE;
            }
            self.alt.del(n);
        }
    }

    fn handle_tree_hitting_boundary(&mut self, region: u32, edge: CompressedEdge) {
        let node = self.regions.items[region as usize].alt_tree_node;
        self.become_root(node);
        self.shatter_descendants_into_matches_and_freeze(node);
        self.regions.items[region as usize].matched = super::flooder::Match { region: NONE, edge };
        self.set_region_frozen(region);
    }

    fn handle_tree_hitting_boundary_match(&mut self, unmatched: u32, matched: u32, edge: CompressedEdge) {
        let node = self.regions.items[unmatched as usize].alt_tree_node;
        self.add_match(unmatched, matched, edge);
        self.set_region_frozen(unmatched);
        self.become_root(node);
        self.shatter_descendants_into_matches_and_freeze(node);
    }

    fn handle_tree_hitting_other_tree(&mut self, region1: u32, region2: u32, edge: CompressedEdge) {
        let alt1 = self.regions.items[region1 as usize].alt_tree_node;
        let alt2 = self.regions.items[region2 as usize].alt_tree_node;
        self.become_root(alt1);
        self.become_root(alt2);
        self.shatter_descendants_into_matches_and_freeze(alt1);
        self.shatter_descendants_into_matches_and_freeze(alt2);
        self.add_match(region1, region2, edge);
        self.set_region_frozen(region1);
        self.set_region_frozen(region2);
    }

    fn handle_tree_hitting_match(&mut self, unmatched: u32, matched: u32, edge: CompressedEdge) {
        let alt = self.regions.items[unmatched as usize].alt_tree_node;
        let m = self.regions.items[matched as usize].matched;
        self.make_child(alt, matched, m.region, m.edge, edge);
        let other = m.region;
        self.regions.items[other as usize].matched = super::flooder::Match::EMPTY;
        self.regions.items[matched as usize].matched = super::flooder::Match::EMPTY;
        self.set_region_shrinking(matched);
        self.set_region_growing(other);
    }

    fn handle_tree_hitting_self(&mut self, region1: u32, region2: u32, edge: CompressedEdge, common_ancestor: u32) {
        let alt1 = self.regions.items[region1 as usize].alt_tree_node;
        let alt2 = self.regions.items[region2 as usize].alt_tree_node;
        let (orphans1, path1) = self.prune_upward_path_stopping_before(alt1, common_ancestor, true);
        let (orphans2, path2) = self.prune_upward_path_stopping_before(alt2, common_ancestor, false);
        let mut cycle = path2;
        cycle.reserve(path1.len() + 1);
        cycle.extend(path1.iter().rev().copied());
        cycle.push(RegionEdge { region: region1, edge });
        let old_outer = self.alt.items[common_ancestor as usize].outer_region;
        self.regions.items[old_outer as usize].alt_tree_node = NONE;
        let blossom = self.create_blossom(cycle);
        self.alt.items[common_ancestor as usize].outer_region = blossom;
        self.regions.items[blossom as usize].alt_tree_node = common_ancestor;
        for c in orphans1.into_iter().chain(orphans2) {
            self.add_child(common_ancestor, c);
        }
    }

    fn handle_blossom_shattering(&mut self, blossom: u32, in_parent_region: u32, in_child_region: u32) {
        for k in 0..self.regions.items[blossom as usize].blossom_children.len() {
            let c = self.regions.items[blossom as usize].blossom_children[k].region;
            self.clear_blossom_parent(c);
        }
        let cycle = std::mem::take(&mut self.regions.items[blossom as usize].blossom_children);
        let blossom_alt = self.regions.items[blossom as usize].alt_tree_node;
        let bsize = cycle.len();
        let (mut parent_idx, mut child_idx) = (0usize, 0usize);
        for (i, re) in cycle.iter().enumerate() {
            if re.region == in_parent_region {
                parent_idx = i;
            }
            if re.region == in_child_region {
                child_idx = i;
            }
        }
        let gap = (child_idx + bsize - parent_idx) % bsize;
        let mut current = self.alt.items[blossom_alt as usize].parent.alt_tree_node;
        unstable_erase(&mut self.alt.items[current as usize].children, blossom_alt);
        let mut child_edge = self.alt.items[blossom_alt as usize].parent.edge.reversed();
        let (evens_start, evens_end);
        if gap % 2 == 0 {
            evens_start = child_idx + 1;
            evens_end = child_idx + bsize - gap;
            let mut i = parent_idx;
            while i < parent_idx + gap {
                current = self.make_child(current, cycle[i % bsize].region, cycle[(i + 1) % bsize].region, cycle[i % bsize].edge, child_edge);
                child_edge = cycle[(i + 1) % bsize].edge;
                let (inner, outer) = (self.alt.items[current as usize].inner_region, self.alt.items[current as usize].outer_region);
                self.set_region_shrinking(inner);
                self.set_region_growing(outer);
                i += 2;
            }
        } else {
            evens_start = parent_idx + 1;
            evens_end = parent_idx + gap;
            let mut i = 0;
            while i < bsize - gap {
                let k1 = (parent_idx + bsize - i) % bsize;
                let k2 = (parent_idx + bsize - i - 1) % bsize;
                let k3 = (parent_idx + bsize - i - 2) % bsize;
                current = self.make_child(current, cycle[k1].region, cycle[k2].region, cycle[k2].edge.reversed(), child_edge);
                child_edge = cycle[k3].edge.reversed();
                let (inner, outer) = (self.alt.items[current as usize].inner_region, self.alt.items[current as usize].outer_region);
                self.set_region_shrinking(inner);
                self.set_region_growing(outer);
                i += 2;
            }
        }
        let mut j = evens_start;
        while j < evens_end {
            let (k1, k2) = (j % bsize, (j + 1) % bsize);
            self.add_match(cycle[k1].region, cycle[k2].region, cycle[k1].edge);
            self.reschedule_total_area(cycle[k1].region);
            self.reschedule_total_area(cycle[k2].region);
            j += 2;
        }
        let inner = cycle[child_idx].region;
        self.alt.items[blossom_alt as usize].inner_region = inner;
        self.set_region_shrinking(inner);
        self.regions.items[inner as usize].alt_tree_node = blossom_alt;
        self.add_child(current, AltTreeEdge { alt_tree_node: blossom_alt, edge: child_edge });
        self.regions.del(blossom);
    }

    fn handle_region_hit_region(&mut self, region1: u32, region2: u32, edge: CompressedEdge) {
        let alt1 = self.regions.items[region1 as usize].alt_tree_node;
        let alt2 = self.regions.items[region2 as usize].alt_tree_node;
        if alt1 != NONE && alt2 != NONE {
            let ca = self.most_recent_common_ancestor(alt1, alt2);
            if ca == NONE {
                self.handle_tree_hitting_other_tree(region1, region2, edge);
            } else {
                self.handle_tree_hitting_self(region1, region2, edge, ca);
            }
        } else if alt1 != NONE {
            if self.regions.items[region2 as usize].matched.region != NONE {
                self.handle_tree_hitting_match(region1, region2, edge);
            } else {
                self.handle_tree_hitting_boundary_match(region1, region2, edge);
            }
        } else if self.regions.items[region1 as usize].matched.region != NONE {
            self.handle_tree_hitting_match(region2, region1, edge.reversed());
        } else {
            self.handle_tree_hitting_boundary_match(region2, region1, edge.reversed());
        }
    }

    /// `Mwpm::process_event`.
    pub(crate) fn process_event(&mut self, event: MatcherEvent) {
        match event {
            MatcherEvent::RegionHitRegion { region1, region2, edge } => self.handle_region_hit_region(region1, region2, edge),
            MatcherEvent::RegionHitBoundary { region, edge } => self.handle_tree_hitting_boundary(region, edge),
            MatcherEvent::BlossomShatter { blossom_region, in_parent_region, in_child_region } => self.handle_blossom_shattering(blossom_region, in_parent_region, in_child_region),
        }
    }

    /// `pair_and_shatter_subblossoms_and_extract_matches` and `..._match_edges` in one: the
    /// blossom's radius counted, its children paired around the cycle from the one its match
    /// leaves from, each pair shattered.
    fn pair_and_shatter_subblossoms(&mut self, region: u32, res: &mut MatchingResult, edges: bool) -> u32 {
        for k in 0..self.regions.items[region as usize].blossom_children.len() {
            let c = self.regions.items[region as usize].blossom_children[k].region;
            self.clear_blossom_parent_ignoring_wrapped_radius(c);
        }
        let m = self.regions.items[region as usize].matched;
        let sub = self.nodes[m.edge.loc_from as usize].region_that_arrived_top;
        self.regions.items[sub as usize].matched = m;
        if m.region != NONE {
            self.regions.items[m.region as usize].matched.region = sub;
        }
        res.weight += self.regions.items[region as usize].radius.y_intercept();
        let children = std::mem::take(&mut self.regions.items[region as usize].blossom_children);
        let num_children = children.len();
        let index = children.iter().position(|e| e.region == sub).unwrap_or(num_children);
        let mut i = 0;
        while i < num_children - 1 {
            let re1 = children[(index + i + 1) % num_children];
            let re2 = children[(index + i + 2) % num_children];
            self.add_match(re1.region, re2.region, re1.edge);
            self.shatter_blossom_and_extract(re1.region, res, edges);
            i += 2;
        }
        self.regions.del(region);
        sub
    }

    /// `shatter_blossom_and_extract_matches` (into `res`) and, with `edges`,
    /// `shatter_blossom_and_extract_match_edges` (onto `match_edges`).
    pub(crate) fn shatter_blossom_and_extract(&mut self, mut region: u32, res: &mut MatchingResult, edges: bool) {
        self.cleanup_shell_area(region);
        let m = self.regions.items[region as usize].matched;
        if m.region != NONE {
            self.cleanup_shell_area(m.region);
            if self.regions.items[region as usize].blossom_children.is_empty() && self.regions.items[m.region as usize].blossom_children.is_empty() {
                res.obs_mask ^= m.edge.obs_mask;
                res.weight += self.regions.items[region as usize].radius.y_intercept() + self.regions.items[m.region as usize].radius.y_intercept();
                if edges {
                    self.match_edges.push(m.edge);
                }
                self.regions.del(m.region);
                self.regions.del(region);
                return;
            }
        } else if self.regions.items[region as usize].blossom_children.is_empty() {
            res.obs_mask ^= m.edge.obs_mask;
            res.weight += self.regions.items[region as usize].radius.y_intercept();
            if edges {
                self.match_edges.push(m.edge);
            }
            self.regions.del(region);
            return;
        }
        if !self.regions.items[region as usize].blossom_children.is_empty() {
            region = self.pair_and_shatter_subblossoms(region, res, edges);
        }
        let m = self.regions.items[region as usize].matched.region;
        if m != NONE && !self.regions.items[m as usize].blossom_children.is_empty() {
            self.pair_and_shatter_subblossoms(m, res, edges);
        }
        self.shatter_blossom_and_extract(region, res, edges);
    }

    /// `Mwpm::create_detection_event`.
    pub(crate) fn create_detection_event(&mut self, n: u32) {
        let region = self.alloc_region();
        let a = self.alloc_alt_tree_node(NONE, region, CompressedEdge::EMPTY);
        self.regions.items[region as usize].alt_tree_node = a;
        self.do_region_created_at_empty_detector_node(region, n);
    }
}
