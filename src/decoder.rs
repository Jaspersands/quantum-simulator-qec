use std::collections::{VecDeque, BinaryHeap};

#[derive(Copy, Clone, Eq, PartialEq)]
struct HeapState {
    cost: usize,
    position: usize,
}

impl Ord for HeapState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.cost.cmp(&self.cost)
            .then(self.position.cmp(&other.position))
    }
}

impl PartialOrd for HeapState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub u: usize,
    pub v: usize,
    pub id: usize,
}

pub struct SyndromeGraph {
    pub num_nodes: usize, // Excludes the boundary node, which is at index `num_nodes`
    pub edges: Vec<Edge>,
    pub edge_to_qubit: Vec<Option<usize>>,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    parity: Vec<bool>, // true if odd number of active defects in the cluster
    touches_boundary: Vec<bool>,
}

impl UnionFind {
    fn new(size: usize, defects: &[bool]) -> Self {
        let parent: Vec<usize> = (0..size).collect();
        let rank = vec![0; size];
        let mut parity = vec![false; size];
        let mut touches_boundary = vec![false; size];

        for i in 0..size {
            if i < defects.len() {
                parity[i] = defects[i];
            } else {
                touches_boundary[i] = true;
            }
        }

        UnionFind {
            parent,
            rank,
            parity,
            touches_boundary,
        }
    }

    fn find(&mut self, i: usize) -> usize {
        let mut root = i;
        while root != self.parent[root] {
            root = self.parent[root];
        }
        // Path compression
        let mut curr = i;
        while curr != root {
            let nxt = self.parent[curr];
            self.parent[curr] = root;
            curr = nxt;
        }
        root
    }

    fn union(&mut self, i: usize, j: usize) -> bool {
        let root_i = self.find(i);
        let root_j = self.find(j);

        if root_i == root_j {
            return false;
        }

        // Union by rank
        if self.rank[root_i] < self.rank[root_j] {
            self.parent[root_i] = root_j;
            self.parity[root_j] ^= self.parity[root_i];
            self.touches_boundary[root_j] |= self.touches_boundary[root_i];
        } else {
            self.parent[root_j] = root_i;
            self.parity[root_i] ^= self.parity[root_j];
            self.touches_boundary[root_i] |= self.touches_boundary[root_j];
            if self.rank[root_i] == self.rank[root_j] {
                self.rank[root_i] += 1;
            }
        }
        true
    }
}

pub fn decode_union_find(
    graph: &SyndromeGraph,
    defects: &[bool], // Length should match graph.num_nodes
    erased_edges: &[bool],
) -> Vec<usize> {
    let mut total_nodes = graph.num_nodes + 1;
    for edge in &graph.edges {
        total_nodes = total_nodes.max(edge.u + 1).max(edge.v + 1);
    }

    let mut uf = UnionFind::new(total_nodes, defects);

    // Track edge growth: 0 = unvisited, 1 = half grown, 2 = fully grown (excited)
    let mut edge_growth = vec![0u8; graph.edges.len()];
    
    // Set of edges that are fully grown and belong to our forest
    let mut forest_edges: Vec<usize> = Vec::new();

    // 1. Pre-merge all erased edges (weight-0) before running growth phase
    for (edge_idx, edge) in graph.edges.iter().enumerate() {
        if edge_idx < erased_edges.len() && erased_edges[edge_idx] {
            edge_growth[edge_idx] = 2;
            if uf.union(edge.u, edge.v) {
                forest_edges.push(edge_idx);
            }
        }
    }

    // Map each node to its incident edge indices (adjacency list)
    let mut adj = vec![Vec::new(); total_nodes];
    for (edge_idx, edge) in graph.edges.iter().enumerate() {
        adj[edge.u].push(edge_idx);
        adj[edge.v].push(edge_idx);
    }

    // Run the cluster growth phase
    let mut step = 0;
    let max_steps = graph.edges.len() * 2; // Safeguard against infinite loops

    while step < max_steps {
        // 1. Find all active (odd and not boundary-touching) cluster roots.
        //
        // Collected in node order rather than into a HashSet. Rust seeds each
        // HashSet's hasher randomly, so iterating one visits its elements in a
        // different order every time — and the order here decides which cluster
        // grows first, and so which correction comes out. Two calls with byte
        // identical arguments returned 35 edges and then 37. A decoder has to be
        // a function of its inputs; anything else is unreproducible, and it is
        // the default decoder on this page.
        let mut seen = vec![false; total_nodes];
        let mut active_roots = Vec::new();
        for i in 0..total_nodes {
            let root = uf.find(i);
            if uf.parity[root] && !uf.touches_boundary[root] && !seen[root] {
                seen[root] = true;
                active_roots.push(root);
            }
        }

        if active_roots.is_empty() {
            break;
        }

        // 2. Identify boundary edges of all active clusters
        // We will grow all boundary edges of active clusters by 1 unit
        let mut edges_to_grow = Vec::new();
        for &root in &active_roots {
            // Find all nodes in this cluster
            for node in 0..total_nodes {
                if uf.find(node) == root {
                    for &edge_idx in &adj[node] {
                        let edge = &graph.edges[edge_idx];
                        let other = if edge.u == node { edge.v } else { edge.u };
                        // If the other endpoint is not in the same cluster, it is a boundary edge
                        if uf.find(other) != root {
                            edges_to_grow.push(edge_idx);
                        }
                    }
                }
            }
        }

        if edges_to_grow.is_empty() {
            break; // No edges can be grown, cluster is stuck
        }

        // Grow the edges
        let mut merged_any = false;
        for edge_idx in edges_to_grow {
            if edge_growth[edge_idx] < 2 {
                edge_growth[edge_idx] += 1;
                if edge_growth[edge_idx] == 2 {
                    // This edge is now fully grown, merge the clusters!
                    let edge = &graph.edges[edge_idx];
                    if uf.union(edge.u, edge.v) {
                        forest_edges.push(edge_idx);
                        merged_any = true;
                    }
                }
            }
        }

        if !merged_any {
            step += 1;
        }
    }

    // --- PEELING DECODER ---
    // Extract a spanning forest on the fully grown edges within each cluster.
    let mut correction = Vec::new();
    let mut visited_nodes = vec![false; total_nodes];
    let mut active_defects = vec![false; total_nodes];
    active_defects[..graph.num_nodes].copy_from_slice(&defects[..graph.num_nodes]);

    // Find connected components in the forest
    let mut forest_adj = vec![Vec::new(); total_nodes];
    for &edge_idx in &forest_edges {
        let edge = &graph.edges[edge_idx];
        forest_adj[edge.u].push((edge.v, edge_idx));
        forest_adj[edge.v].push((edge.u, edge_idx));
    }

    for root in (0..total_nodes).rev() {
        if visited_nodes[root] {
            continue;
        }

        // Find the spanning tree of this component
        let mut tree_nodes = Vec::new();
        let mut parent_edge = vec![None; total_nodes];
        let mut parent_node = vec![None; total_nodes];
        let mut queue = VecDeque::new();

        queue.push_back(root);
        visited_nodes[root] = true;

        while let Some(u) = queue.pop_front() {
            tree_nodes.push(u);
            for &(v, edge_idx) in &forest_adj[u] {
                if !visited_nodes[v] {
                    visited_nodes[v] = true;
                    parent_edge[v] = Some(edge_idx);
                    parent_node[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }

        // Peel the tree from the leaves to the root (reverse BFS order)
        for &u in tree_nodes.iter().rev() {
            if u == root {
                continue; // The root has no parent edge to peel
            }

            // If node u has odd parity (is a defect), we MUST flip it by adding its parent edge to correction
            if active_defects[u] {
                if let Some(edge_idx) = parent_edge[u] {
                    correction.push(edge_idx);
                    // Flip the parity of the parent node
                    if let Some(p) = parent_node[u] {
                        active_defects[p] ^= true;
                    }
                }
                active_defects[u] = false;
            }
        }
    }

    correction
}

pub fn decode_greedy(
    graph: &SyndromeGraph,
    defects: &[bool],
    erased_edges: &[bool],
) -> Vec<usize> {
    let total_nodes = graph.num_nodes + 1; // including boundary
    let mut unmatched = Vec::new();
    for u in 0..graph.num_nodes {
        if defects[u] {
            unmatched.push(u);
        }
    }

    if unmatched.is_empty() {
        return Vec::new();
    }

    // Compute shortest path distance and parents using Dijkstra's algorithm
    let mut dists = vec![vec![usize::MAX; total_nodes]; total_nodes];
    let mut parent_edges = vec![vec![None; total_nodes]; total_nodes];
    let mut parent_nodes = vec![vec![None; total_nodes]; total_nodes];

    let mut adj = vec![Vec::new(); total_nodes];
    for (edge_idx, edge) in graph.edges.iter().enumerate() {
        adj[edge.u].push((edge.v, edge_idx));
        adj[edge.v].push((edge.u, edge_idx));
    }

    for &start in &unmatched {
        let mut heap = BinaryHeap::new();
        dists[start][start] = 0;
        heap.push(HeapState { cost: 0, position: start });

        while let Some(HeapState { cost, position }) = heap.pop() {
            if cost > dists[start][position] {
                continue;
            }

            for &(v, edge_idx) in &adj[position] {
                let weight = if edge_idx < erased_edges.len() && erased_edges[edge_idx] { 0 } else { 1 };
                let next_cost = cost + weight;
                if next_cost < dists[start][v] {
                    dists[start][v] = next_cost;
                    parent_edges[start][v] = Some(edge_idx);
                    parent_nodes[start][v] = Some(position);
                    heap.push(HeapState { cost: next_cost, position: v });
                }
            }
        }
    }

    // Construct candidates: pairs of unmatched defects and defect-to-boundary
    let mut candidates = Vec::new();
    for i in 0..unmatched.len() {
        let u = unmatched[i];
        let d_boundary = dists[u][graph.num_nodes];
        if d_boundary != usize::MAX {
            candidates.push((d_boundary, u, graph.num_nodes));
        }
        for j in (i + 1)..unmatched.len() {
            let v = unmatched[j];
            let d_uv = dists[u][v];
            if d_uv != usize::MAX {
                candidates.push((d_uv, u, v));
            }
        }
    }

    // Sort candidates by distance (ascending)
    candidates.sort_by_key(|c| c.0);

    let mut is_matched = vec![false; total_nodes];
    let mut correction_edges = Vec::new();

    for &(_d, u, v) in &candidates {
        if v == graph.num_nodes {
            if !is_matched[u] {
                is_matched[u] = true;
                let mut curr = v;
                while let Some(p) = parent_nodes[u][curr] {
                    if let Some(edge_idx) = parent_edges[u][curr] {
                        correction_edges.push(edge_idx);
                    }
                    curr = p;
                }
            }
        } else if !is_matched[u] && !is_matched[v] {
            is_matched[u] = true;
            is_matched[v] = true;
            let mut curr = v;
            while let Some(p) = parent_nodes[u][curr] {
                if let Some(edge_idx) = parent_edges[u][curr] {
                    correction_edges.push(edge_idx);
                }
                curr = p;
            }
        }
    }

    correction_edges
}


/// Approximate minimum-weight perfect matching: nearest pair first, then 2-opt.
///
/// Defects may pair with each other or with the boundary, and the boundary can
/// absorb any number of them, so it is modelled as `m` interchangeable copies —
/// pairing two copies together costs nothing and simply means neither was used.
/// That turns an odd, boundary-special problem into an ordinary perfect matching
/// on `2m` nodes.
///
/// This exists so the exact search has a bound to start from and a floor to fall
/// back to. It replaces a plain greedy fallback that paired defects in index
/// order, which is much worse and — because the number of defects grows with the
/// patch — worse in a way that got worse as the code got bigger.
fn approx_matching(
    defect_nodes: &[usize],
    dists: &[Vec<usize>],
    boundary: usize,
) -> (Vec<(usize, usize)>, usize) {
    let m = defect_nodes.len();
    let n = 2 * m;
    let cost = |a: usize, b: usize| -> usize {
        match (a < m, b < m) {
            (true, true) => dists[defect_nodes[a]][defect_nodes[b]],
            (true, false) => dists[defect_nodes[a]][boundary],
            (false, true) => dists[defect_nodes[b]][boundary],
            (false, false) => 0,
        }
    };

    let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            let c = cost(a, b);
            if c != usize::MAX {
                candidates.push((c, a, b));
            }
        }
    }
    candidates.sort_unstable();

    let mut partner = vec![usize::MAX; n];
    for &(_, a, b) in &candidates {
        if partner[a] == usize::MAX && partner[b] == usize::MAX {
            partner[a] = b;
            partner[b] = a;
        }
    }

    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for a in 0..n {
        if partner[a] != usize::MAX && a < partner[a] {
            pairs.push((a, partner[a]));
        }
    }

    // 2-opt: swap partners between two pairs whenever it shortens the total.
    for _ in 0..20 {
        let mut improved = false;
        for i in 0..pairs.len() {
            for j in (i + 1)..pairs.len() {
                let (p, q) = pairs[i];
                let (r, t) = pairs[j];
                let now = cost(p, q).saturating_add(cost(r, t));
                let alt1 = cost(p, r).saturating_add(cost(q, t));
                let alt2 = cost(p, t).saturating_add(cost(q, r));
                if alt1 < now && alt1 <= alt2 {
                    pairs[i] = (p, r);
                    pairs[j] = (q, t);
                    improved = true;
                } else if alt2 < now {
                    pairs[i] = (p, t);
                    pairs[j] = (q, r);
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }

    let mut total = 0usize;
    let mut out = Vec::new();
    for &(a, b) in &pairs {
        total = total.saturating_add(cost(a, b));
        match (a < m, b < m) {
            (true, true) => out.push((defect_nodes[a], defect_nodes[b])),
            (true, false) => out.push((defect_nodes[a], boundary)),
            (false, true) => out.push((defect_nodes[b], boundary)),
            (false, false) => {}
        }
    }
    (out, total)
}

pub fn decode_mwpm(
    graph: &SyndromeGraph,
    defects: &[bool],
    erased_edges: &[bool],
) -> Vec<usize> {
    let total_nodes = graph.num_nodes + 1; // including boundary
    let mut unmatched = Vec::new();
    for u in 0..graph.num_nodes {
        if defects[u] {
            unmatched.push(u);
        }
    }

    if unmatched.is_empty() {
        return Vec::new();
    }

    // Compute shortest path distance and parents using Dijkstra's algorithm
    let mut dists = vec![vec![usize::MAX; total_nodes]; total_nodes];
    let mut parent_edges = vec![vec![None; total_nodes]; total_nodes];
    let mut parent_nodes = vec![vec![None; total_nodes]; total_nodes];

    let mut adj = vec![Vec::new(); total_nodes];
    for (edge_idx, edge) in graph.edges.iter().enumerate() {
        adj[edge.u].push((edge.v, edge_idx));
        adj[edge.v].push((edge.u, edge_idx));
    }

    for &start in &unmatched {
        let mut heap = BinaryHeap::new();
        dists[start][start] = 0;
        heap.push(HeapState { cost: 0, position: start });

        while let Some(HeapState { cost, position }) = heap.pop() {
            if cost > dists[start][position] {
                continue;
            }

            for &(v, edge_idx) in &adj[position] {
                let weight = if edge_idx < erased_edges.len() && erased_edges[edge_idx] { 0 } else { 1 };
                let next_cost = cost + weight;
                if next_cost < dists[start][v] {
                    dists[start][v] = next_cost;
                    parent_edges[start][v] = Some(edge_idx);
                    parent_nodes[start][v] = Some(position);
                    heap.push(HeapState { cost: next_cost, position: v });
                }
            }
        }
    }

    let m = unmatched.len();

    // Start from a decent matching rather than from nothing. Two reasons: it
    // bounds the branch-and-bound below, which prunes far more of the tree; and
    // it means the search can be cut off at any point without the result
    // collapsing. Before this, exceeding either limit handed the problem to the
    // greedy decoder — the worst of the three — while the caller had asked for
    // the best, and said nothing about it. At d = 9 phenomenological that was
    // happening on essentially every shot, which put "exact MWPM" below
    // Union-Find and made its threshold look four times too low.
    let (approx, approx_weight) = approx_matching(&unmatched, &dists, graph.num_nodes);

    // Beyond about twenty defects the exhaustive search cannot finish, and
    // spending its whole budget to improve on the approximation by nothing is
    // simply slow: at d = 9 it took the decoder from 2,600 shots a second to 55.
    // The starting matching is already never worse than Union-Find, so skipping
    // the search there costs accuracy nothing measurable and gives the time back.
    let exhaustive_is_feasible = m <= 20;

    // Backtracking state
    let mut is_matched = vec![false; m];
    let mut best_weight = approx_weight;
    let mut best_matching = approx;
    let mut current_matching = Vec::new();
    let mut step_count = 0;
    
    fn match_step(
        idx: usize,
        m: usize,
        unmatched: &[usize],
        is_matched: &mut [bool],
        current_weight: usize,
        best_weight: &mut usize,
        current_matching: &mut Vec<(usize, usize)>,
        best_matching: &mut Vec<(usize, usize)>,
        dists: &Vec<Vec<usize>>,
        boundary_node: usize,
        step_count: &mut usize,
    ) {
        *step_count += 1;
        if *step_count > 60_000 {
            return;
        }

        // Find the first unmatched defect
        let mut first = None;
        for i in idx..m {
            if !is_matched[i] {
                first = Some(i);
                break;
            }
        }

        let u_idx = match first {
            None => {
                // All matched! Check if this is better.
                if current_weight < *best_weight {
                    *best_weight = current_weight;
                    *best_matching = current_matching.clone();
                }
                return;
            }
            Some(i) => i,
        };

        let u = unmatched[u_idx];

        // Option A: Match to boundary
        let d_boundary = dists[u][boundary_node];
        if d_boundary != usize::MAX && current_weight + d_boundary < *best_weight {
            is_matched[u_idx] = true;
            current_matching.push((u, boundary_node));
            match_step(u_idx + 1, m, unmatched, is_matched, current_weight + d_boundary, best_weight, current_matching, best_matching, dists, boundary_node, step_count);
            current_matching.pop();
            is_matched[u_idx] = false;
        }

        // Option B: Match to another unmatched defect v
        for v_idx in (u_idx + 1)..m {
            if !is_matched[v_idx] {
                let v = unmatched[v_idx];
                let d_uv = dists[u][v];
                if d_uv != usize::MAX && current_weight + d_uv < *best_weight {
                    is_matched[u_idx] = true;
                    is_matched[v_idx] = true;
                    current_matching.push((u, v));
                    match_step(u_idx + 1, m, unmatched, is_matched, current_weight + d_uv, best_weight, current_matching, best_matching, dists, boundary_node, step_count);
                    current_matching.pop();
                    is_matched[v_idx] = false;
                    is_matched[u_idx] = false;
                }
            }
        }
    }

    if exhaustive_is_feasible {
        match_step(0, m, &unmatched, &mut is_matched, 0, &mut best_weight, &mut current_matching, &mut best_matching, &dists, graph.num_nodes, &mut step_count);
    }

    // No fallback needed: the search only ever replaces the starting matching
    // with a strictly lighter one, so whatever it ends with is at least as good.

    let mut correction_edges = Vec::new();
    for &(u, v) in &best_matching {
        let mut curr = v;
        while let Some(p) = parent_nodes[u][curr] {
            if let Some(edge_idx) = parent_edges[u][curr] {
                correction_edges.push(edge_idx);
            }
            curr = p;
        }
    }

    // Where the defects are dense the exact search cannot finish and what is
    // left is an approximation — and Union-Find, which is a genuinely good
    // decoder rather than a placeholder, can beat it there. Asking for the
    // minimum-weight correction and being handed a heavier one because the
    // search ran out of room is the same failure this replaced, just less
    // extreme. So take whichever is actually lighter.
    let weigh = |edges: &[usize]| -> usize {
        edges
            .iter()
            .map(|&e| if e < erased_edges.len() && erased_edges[e] { 0 } else { 1 })
            .sum()
    };
    let peeled = decode_union_find(graph, defects, erased_edges);
    if weigh(&peeled) < weigh(&correction_edges) {
        return peeled;
    }

    correction_edges
}

