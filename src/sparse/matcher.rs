//! What an event means: the seven cases of the paper, over alternating trees
//! of regions.

use super::state::{AltNode, CEdge, Radius, Region, BOUNDARY, NONE, NO_TIME};
use super::Solver;

impl<'a> Solver<'a> {
    fn set_match(&mut self, r: u32, partner: u32, e: CEdge) {
        self.s.regions[r as usize].matched = Some((partner, e));
        if partner != BOUNDARY {
            self.s.regions[partner as usize].matched = Some((r, e.rev()));
        }
    }

    /// `n` and its ancestors up to the root, into `out`.
    fn path_to_root(&self, mut n: u32, out: &mut Vec<u32>) {
        out.clear();
        out.push(n);
        while self.s.alt[n as usize].parent != NONE {
            n = self.s.alt[n as usize].parent;
            out.push(n);
        }
    }

    fn root_of(&self, mut n: u32) -> u32 {
        while self.s.alt[n as usize].parent != NONE {
            n = self.s.alt[n as usize].parent;
        }
        n
    }

    /// Index of the blossom child of `b` that holds `defect`.
    pub(crate) fn child_index(&self, b: u32, children: &[(u32, CEdge)], defect: u32) -> usize {
        let mut r = self.s.nodes[defect as usize].own;
        while self.s.regions[r as usize].blossom_parent != b {
            r = self.s.regions[r as usize].blossom_parent;
        }
        children.iter().position(|c| c.0 == r).expect("the defect lies in a child of the blossom")
    }

    /// Two top regions touch; `e` runs from `r1` to `r2`.
    pub(crate) fn on_collide(&mut self, r1: u32, r2: u32, e: CEdge) {
        let n1 = self.s.regions[r1 as usize].tree;
        let n2 = self.s.regions[r2 as usize].tree;
        if n1 == NONE {
            assert!(n2 != NONE, "a collision needs a growing region");
            return self.on_collide(r2, r1, e.rev());
        }
        if n2 == NONE {
            let (partner, _) = self.s.regions[r2 as usize].matched.expect("a region outside every tree is matched");
            if partner == BOUNDARY {
                self.take_from_boundary(n1, r2, e)
            } else {
                self.grow_tree(n1, r2, e)
            }
        } else if self.root_of(n1) == self.root_of(n2) {
            self.form_blossom(n1, n2, e)
        } else {
            self.augment(n1, n2, e)
        }
    }

    /// (f) A growing region reaches the boundary.
    pub(crate) fn on_boundary(&mut self, r: u32, e: CEdge) {
        let n = self.s.regions[r as usize].tree;
        assert!(n != NONE, "only a growing region reaches the boundary");
        self.set_match(r, BOUNDARY, e);
        self.dissolve(n);
    }

    /// (a) A growing region reaches a matched pair; both join its tree.
    fn grow_tree(&mut self, n: u32, m: u32, e: CEdge) {
        let (partner, me) = self.s.regions[m as usize].matched.take().expect("matched");
        self.s.regions[partner as usize].matched = None;
        let children = self.s.spare_u32();
        let child = self.new_alt(AltNode {
            inner: m,
            outer: partner,
            inner_to_outer: me,
            parent: n,
            parent_edge: e,
            children,
            alive: true,
            mark: 0,
        });
        self.s.alt[n as usize].children.push(child);
        self.s.regions[m as usize].tree = child;
        self.s.regions[partner as usize].tree = child;
        self.set_slope(m, -1);
        self.set_slope(partner, 1);
        self.reschedule(m);
        self.reschedule(partner);
    }

    /// (g) A growing region reaches a region matched to the boundary. It takes
    /// that region as its partner, and its tree dissolves.
    fn take_from_boundary(&mut self, n: u32, m: u32, e: CEdge) {
        let r = self.s.alt[n as usize].outer;
        self.set_match(r, m, e);
        self.dissolve(n);
    }

    /// (b) Two trees meet: augment along both paths to their roots.
    fn augment(&mut self, n1: u32, n2: u32, e: CEdge) {
        let (r1, r2) = (self.s.alt[n1 as usize].outer, self.s.alt[n2 as usize].outer);
        self.set_match(r1, r2, e);
        self.dissolve(n1);
        self.dissolve(n2);
    }

    /// Turn the tree holding `n` into matched pairs. `n`'s outer region has just
    /// been matched outside the tree. Along the path to the root each inner
    /// region matches its parent's outer region; every other node's pair matches
    /// along its own edge.
    fn dissolve(&mut self, n: u32) {
        let mut path = std::mem::take(&mut self.s.path_a);
        self.path_to_root(n, &mut path);
        let on_path = self.s.next_stamp();
        for &m in &path {
            let (parent, inner, pe) = {
                let a = &mut self.s.alt[m as usize];
                a.mark = on_path;
                (a.parent, a.inner, a.parent_edge)
            };
            if parent != NONE {
                let pouter = self.s.alt[parent as usize].outer;
                self.set_match(inner, pouter, pe.rev());
            }
        }
        let root = *path.last().expect("path has a root");
        self.s.path_a = path;
        let mut stack = std::mem::take(&mut self.s.stack);
        let mut regions = std::mem::take(&mut self.s.touched_regions);
        stack.clear();
        regions.clear();
        stack.push(root);
        while let Some(x) = stack.pop() {
            // Every node of the tree dies here, so its children can be moved out.
            let a = &mut self.s.alt[x as usize];
            let (inner, outer, io, off_path) = (a.inner, a.outer, a.inner_to_outer, a.mark != on_path);
            a.alive = false;
            stack.append(&mut a.children);
            if inner != NONE {
                if off_path {
                    self.set_match(inner, outer, io);
                }
                regions.push(inner);
            }
            regions.push(outer);
        }
        self.s.stack = stack;
        for &r in &regions {
            self.s.regions[r as usize].tree = NONE;
            self.set_slope(r, 0);
        }
        for &r in &regions {
            self.reschedule(r);
        }
        self.s.touched_regions = regions;
    }

    /// (c) Two growing regions of one tree meet. The cycle through their common
    /// ancestor becomes a blossom, which takes the ancestor's outer place.
    pub(crate) fn form_blossom(&mut self, n1: u32, n2: u32, e: CEdge) {
        let mut path1 = std::mem::take(&mut self.s.path_a);
        let mut path2 = std::mem::take(&mut self.s.path_b);
        self.path_to_root(n1, &mut path1);
        self.path_to_root(n2, &mut path2);
        // The common ancestor: the first node of path2 that path1 holds.
        let in_path1 = self.s.next_stamp();
        for &x in &path1 {
            self.s.alt[x as usize].mark = in_path1;
        }
        let a = *path2.iter().find(|&&x| self.s.alt[x as usize].mark == in_path1).expect("same tree");
        let p1 = &path1[..path1.iter().position(|&x| x == a).expect("a is on path1")];
        let p2 = &path2[..path2.iter().position(|&x| x == a).expect("a is on path2")];

        let mut cycle = self.s.spare_cycles.pop().unwrap_or_default();
        let mut cur = self.s.alt[a as usize].outer;
        for &m in p1.iter().rev() {
            let (pe, inner, io, outer) = {
                let am = &self.s.alt[m as usize];
                (am.parent_edge, am.inner, am.inner_to_outer, am.outer)
            };
            cycle.push((cur, pe));
            cycle.push((inner, io));
            cur = outer;
        }
        cycle.push((cur, e));
        for &m in p2 {
            let (outer, io, inner, pe) = {
                let am = &self.s.alt[m as usize];
                (am.outer, am.inner_to_outer, am.inner, am.parent_edge)
            };
            cycle.push((outer, io.rev()));
            cycle.push((inner, pe.rev()));
        }

        let len = cycle.len();
        let shell = self.s.spare_u32();
        let b = self.new_region(Region {
            radius: Radius { y0: -self.s.now, slope: 1 },
            blossom_parent: NONE,
            children: cycle,
            shell,
            tree: a,
            matched: None,
            queued: NO_TIME,
            dead: false,
        });
        for i in 0..len {
            let c = self.s.regions[b as usize].children[i].0;
            self.enclose(c, b);
        }

        // The cycle's tree nodes die; their children that are not themselves
        // on the cycle move to the ancestor, after its own remaining children.
        let on = self.s.next_stamp();
        for &m in p1.iter().chain(p2.iter()) {
            self.s.alt[m as usize].mark = on;
        }
        let mut orphans = std::mem::take(&mut self.s.stack);
        orphans.clear();
        for &m in p1.iter().chain(p2.iter()) {
            let children = std::mem::take(&mut self.s.alt[m as usize].children);
            orphans.extend(children.iter().copied().filter(|&c| self.s.alt[c as usize].mark != on));
            self.s.alt[m as usize].alive = false;
            self.s.keep_u32(children);
        }
        let mut kept = std::mem::take(&mut self.s.alt[a as usize].children);
        kept.retain(|&c| self.s.alt[c as usize].mark != on);
        for &o in &orphans {
            self.s.alt[o as usize].parent = a;
        }
        kept.extend_from_slice(&orphans);
        self.s.alt[a as usize].children = kept;
        self.s.stack = orphans;
        self.s.path_a = path1;
        self.s.path_b = path2;
        self.s.alt[a as usize].outer = b;
        self.reschedule(b);
    }

    /// Put top region `c` inside blossom `b`, which starts at radius zero now,
    /// keeping every node's local radius continuous.
    fn enclose(&mut self, c: u32, b: u32) {
        let now = self.s.now;
        let nodes = self.nodes_under(c);
        for &v in &nodes {
            let l = self.local_radius(v);
            let n = &mut self.s.nodes[v as usize];
            n.top = b;
            n.wrapped = l;
        }
        self.done_with_nodes(nodes);
        let reg = &mut self.s.regions[c as usize];
        let y = reg.radius.at(now);
        self.s.growing -= u32::from(reg.radius.slope > 0);
        reg.radius = Radius { y0: y, slope: 0 };
        reg.blossom_parent = b;
        reg.tree = NONE;
        reg.matched = None;
    }

    /// (d) A shrinking blossom reaches radius zero and comes apart. The even
    /// path around its cycle, between the children its two tree edges attach
    /// to, rejoins the tree; the rest of the cycle pairs off.
    pub(crate) fn shatter(&mut self, b: u32) {
        let n = self.s.regions[b as usize].tree;
        let (p, pe, iot) = {
            let a = &self.s.alt[n as usize];
            (a.parent, a.parent_edge, a.inner_to_outer)
        };
        let children = self.s.regions[b as usize].children.clone();
        let k = children.len();
        let i_in = self.child_index(b, &children, pe.b);
        let i_out = self.child_index(b, &children, iot.a);

        let now = self.s.now;
        for &(c, _) in &children {
            let yc = self.s.regions[c as usize].radius.at(now);
            let nodes = self.nodes_under(c);
            for &v in &nodes {
                let l = self.local_radius(v);
                let nd = &mut self.s.nodes[v as usize];
                nd.top = c;
                nd.wrapped = l - yc;
            }
            self.done_with_nodes(nodes);
        }
        for &(c, _) in &children {
            self.s.regions[c as usize].blossom_parent = NONE;
        }
        self.s.regions[b as usize].dead = true;
        self.s.regions[b as usize].children.clear();

        let forward = ((i_out + k - i_in) % k).is_multiple_of(2);
        let mut path = vec![i_in];
        let mut edges = Vec::new();
        let mut j = i_in;
        while j != i_out {
            if forward {
                edges.push(children[j].1);
                j = (j + 1) % k;
            } else {
                let prev = (j + k - 1) % k;
                edges.push(children[prev].1.rev());
                j = prev;
            }
            path.push(j);
        }
        let mut on_path = vec![false; k];
        for &i in &path {
            on_path[i] = true;
        }

        let mut changed = Vec::new();
        let mut j = if forward { (i_out + 1) % k } else { (i_in + 1) % k };
        while !on_path[j] {
            let next = (j + 1) % k;
            let (c1, c2, ce) = (children[j].0, children[next].0, children[j].1);
            self.set_match(c1, c2, ce);
            self.set_slope(c1, 0);
            self.set_slope(c2, 0);
            changed.push(c1);
            changed.push(c2);
            j = (next + 1) % k;
        }

        let q: Vec<u32> = path.iter().map(|&i| children[i].0).collect();
        let (mut parent, mut pedge, mut first) = (p, pe, NONE);
        let mut i = 0;
        while i + 1 < q.len() {
            let children = self.s.spare_u32();
            let node = self.new_alt(AltNode {
                inner: q[i],
                outer: q[i + 1],
                inner_to_outer: edges[i],
                parent,
                parent_edge: pedge,
                children,
                alive: true,
                mark: 0,
            });
            if parent == p {
                first = node;
            } else {
                self.s.alt[parent as usize].children.push(node);
            }
            self.s.regions[q[i] as usize].tree = node;
            self.s.regions[q[i + 1] as usize].tree = node;
            self.set_slope(q[i], -1);
            self.set_slope(q[i + 1], 1);
            changed.push(q[i]);
            changed.push(q[i + 1]);
            parent = node;
            pedge = edges[i + 1];
            i += 2;
        }
        let last = *q.last().expect("the path has an end");
        {
            let a = &mut self.s.alt[n as usize];
            a.inner = last;
            a.parent = parent;
            a.parent_edge = pedge;
        }
        if parent != p {
            self.s.alt[parent as usize].children.push(n);
        }
        if first != NONE {
            let pc = &mut self.s.alt[p as usize].children;
            let slot = pc.iter().position(|&c| c == n).expect("n is a child of p");
            pc[slot] = first;
        }
        self.s.regions[last as usize].tree = n;
        self.set_slope(last, -1);
        changed.push(last);
        for r in changed {
            self.reschedule(r);
        }
    }

    /// (e) A trivial shrinking region reaches radius zero. Its tree parent and
    /// child now touch through it, and the three form a blossom.
    pub(crate) fn implode(&mut self, r: u32) {
        let n = self.s.regions[r as usize].tree;
        let (p, pe, iot) = {
            let a = &self.s.alt[n as usize];
            (a.parent, a.parent_edge, a.inner_to_outer)
        };
        let e = CEdge { a: pe.a, b: iot.b, obs: pe.obs ^ iot.obs };
        self.form_blossom(p, n, e);
    }
}
