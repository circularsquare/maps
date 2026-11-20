//! Line paths, stretches, routes and path search.

use super::*;

impl Network {
    /// The constructed station or edge set is complete for a line: it can carry trains.
    pub fn line_constructed(&self, l: u32) -> bool {
        if let Some(r) = self.line_bus.get(l as usize).and_then(|b| b.as_ref()) {
            return r.built;
        }
        if let Some(r) = self.line_ferry.get(l as usize).and_then(|f| f.as_ref()) {
            return r.built && self.line_stops(l).iter().all(|&s| self.node_ok(s) && self.node_built[s as usize]);
        }
        self.line_path(l).iter().all(|&p| self.edge_ok(path_edge(p)) && self.edge_built[path_edge(p) as usize])
            && self.back_only_edges(l).iter().all(|&e| self.edge_ok(e) && self.edge_built[e as usize])
            && self.line_stops(l).iter().all(|&s| self.node_ok(s) && self.node_built[s as usize])
    }

    /// T-3474: the part of a rail line's plan its trains run. A constructed line runs all of it; one
    /// that is not runs the longest stretch of consecutive constructed stops joined by constructed
    /// track with a constructed way back (ties to more stops, then the earlier). None while no two
    /// stops are joined so (the whole plan is previewed), and for buses and ferries. A built arc of
    /// a loop is an open line, run both ways.
    pub fn line_stretch(&self, l: u32) -> Option<Stretch> {
        let i = l as usize;
        if !self.line_ok(l) || self.line_bus[i].is_some() || self.line_ferry[i].is_some() {
            return None;
        }
        let (stops, path) = (self.line_stops(l), self.line_path(l));
        if stops.len() < 2 || path.is_empty() {
            return None;
        }
        if self.line_constructed(l) {
            return Some(Stretch { stop0: 0, stop1: stops.len() as u32, path0: 0, path1: path.len() as u32 });
        }
        let built_stop = |s: u32| self.node_ok(s) && self.node_built[s as usize];
        let built_edge = |e: u32| self.edge_ok(e) && self.edge_built[e as usize];
        if stops.iter().filter(|&&s| built_stop(s)).count() < 2 {
            return None;
        }
        let nodes: Vec<u32> = std::iter::once(self.path_nodes(path[0]).0).chain(path.iter().map(|&p| self.path_nodes(p).1)).collect();
        let ones = self.line_ones(l);
        let (pos, _) = self.match_run(&nodes, stops, &self.stop_reversals(path), ones, path)?;
        let mut best: Option<(Stretch, f64, usize)> = None;
        for a in 0..stops.len() {
            if !built_stop(stops[a]) {
                continue;
            }
            let (mut end, mut len) = (pos[a], 0.0);
            for b in a + 1..stops.len() {
                let next = pos[b];
                // a stop passed twice (a balloon, a loop back past its first stop) ends the stretch
                if !built_stop(stops[b]) || next <= end || !path[end..next].iter().all(|&p| built_edge(path_edge(p))) {
                    break;
                }
                len += path[end..next].iter().map(|&p| self.edge_len[path_edge(p) as usize]).sum::<f64>();
                end = next;
                let (ss, sp) = (&stops[a..=b], &path[pos[a]..end]);
                let once = self.runs_once(l, ss, sp);
                if !once && !self.back_run_of(ss, sp, ones).is_some_and(|(_, back)| back.iter().all(|&p| built_edge(path_edge(p)))) {
                    continue;
                }
                let n = b - a + 1;
                if best.map_or(true, |(_, been, m)| len > been + 1e-9 || ((len - been).abs() <= 1e-9 && n > m)) {
                    best = Some((Stretch { stop0: a as u32, stop1: b as u32 + 1, path0: pos[a] as u32, path1: end as u32 }, len, n));
                }
            }
        }
        best.map(|b| b.0)
    }

    /// T-3474: a line runs part of its plan only: its stretch is not the whole of it.
    pub fn stretch_is_part(&self, l: u32, s: &Stretch) -> bool {
        s.stop0 != 0 || s.stop1 as usize != self.line_stops(l).len() || s.path0 != 0 || s.path1 as usize != self.line_path(l).len()
    }

    /// The edges only a rail line's return run takes, round one-way track such as a flyover's other
    /// leg (T-3422): they must be constructed for it to run, as its path's are.
    pub fn back_only_edges(&self, l: u32) -> Vec<u32> {
        let path = self.line_path(l);
        if self.line_runs_once(l) || path.iter().all(|&p| self.dir_ok(p ^ 1)) {
            return vec![];
        }
        let Some(back) = self.back_path(l) else { return vec![] };
        let mut out: Vec<u32> = back.iter().map(|&p| path_edge(p)).filter(|e| !path.iter().any(|&p| path_edge(p) == *e)).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The point of an edge nearest to (x, y): chainage and distance, m.
    pub fn project(&self, e: u32, x: f64, y: f64) -> (f64, f64) {
        geom::project(self.edge_pieces(e), x, y)
    }

    // ---------------------------------------------------------------- routing

    /// Whether a path entry (edge, direction) may run: a lane of the piece runs its way (not
    /// against one-way single track, T-3094).
    pub fn dir_ok(&self, p: u32) -> bool {
        self.lane_of(p).is_some()
    }

    /// A rail line whose trains run its path once round, one way: a loop set to run one way (T-2981),
    /// or one whose track takes it round only one way (`runs_once`).
    pub fn line_runs_once(&self, l: u32) -> bool {
        self.runs_once(l, self.line_stops(l), self.line_path(l))
    }

    /// Whether line `l` run over `stops` and `path` (its plan or the stretch it runs) is a loop that
    /// runs once round, one way: set so (T-2981), or set both ways where its return run cannot go
    /// round the other way as run 0 goes round (T-3835): one-way track sends it round again, or back
    /// at a station partway, which no player means by a loop.
    pub fn runs_once(&self, l: u32, stops: &[u32], path: &[u32]) -> bool {
        self.loop_kind(stops, path) != LoopKind::None && (self.line_one_way.get(l as usize).copied().unwrap_or(false) || !self.back_laps(stops, path, self.line_ones(l)))
    }

    /// T-3835: a loop's return run goes round once as run 0 does: its path run backward runs, or the
    /// way back `back_run_of` finds turns back no more often than run 0's path and passes the first
    /// stop no more often (no second lap).
    pub fn back_laps(&self, stops: &[u32], path: &[u32], ones: &[(u32, u8)]) -> bool {
        if path.iter().all(|&p| self.dir_ok(p ^ 1)) {
            return true;
        }
        let Some((_, back)) = self.back_run_of(stops, path, ones) else { return false };
        let first = stops[0];
        let passes = |p: &[u32]| p.iter().filter(|&&e| self.path_nodes(e).1 == first).count();
        self.path_reversals(&back).len() <= self.path_reversals(path).len() && passes(&back) <= passes(path)
    }

    /// A loop set to run one way (T-2981), over `stops` and `path`, whatever its track allows.
    pub(super) fn set_once(&self, l: u32, stops: &[u32], path: &[u32]) -> bool {
        self.line_one_way.get(l as usize).copied().unwrap_or(false) && self.loop_kind(stops, path) != LoopKind::None
    }

    /// The path of a line's return run (T-3094): its path run backward, or where that runs against a
    /// one-way track, the route back through its run 1 stops (`back_run_of`, T-3348). None when
    /// there is no way back.
    pub fn back_path_of(&self, stops: &[u32], path: &[u32], ones: &[(u32, u8)]) -> Option<Vec<u32>> {
        self.back_run_of(stops, path, ones).map(|r| r.1)
    }

    pub fn back_path(&self, l: u32) -> Option<Vec<u32>> {
        self.back_path_of(self.line_stops(l), self.line_path(l), self.line_ones(l))
    }

    /// Rail lines whose return run takes a way of its own, round one-way track: their tracks are not
    /// all on `lines_on_edge`, so an edit rebuilds them whatever it touches.
    pub fn back_lines(&self) -> Vec<u32> {
        // T-3348: a run may stop at a station beside a stop whose platform is not on its way, which
        // may leave the path, and an edit anywhere near may add or take one
        let beside = |l: u32| self.line_stops(l).iter().any(|&s| self.node_ok(s) && self.way_limited(s));
        if (0..self.edge_count() as u32).all(|e| self.edge_way(e) == 0) && !self.node_one.iter().any(|&o| o != 0) {
            return vec![];
        }
        (0..self.line_count() as u32)
            .filter(|&l| {
                self.line_ok(l)
                    && self.line_bus.get(l as usize).map_or(true, |b| b.is_none())
                    && self.line_ferry.get(l as usize).map_or(true, |f| f.is_none())
                    && (self.line_path(l).iter().any(|&p| !self.dir_ok(p ^ 1)) || beside(l))
                    // (a loop run one way by its track stays: track built anywhere may give it a way back, T-3835)
                    && !(self.set_once(l, self.line_stops(l), self.line_path(l)) && self.stretch_set_once(l))
            })
            .collect()
    }

    /// T-3474: `line_runs_once` for the part of the plan its trains run: a loop's built arc runs
    /// both ways.
    pub fn stretch_runs_once(&self, l: u32) -> bool {
        match self.line_stretch(l) {
            Some(s) => self.runs_once(l, s.stops(self.line_stops(l)), s.path(self.line_path(l))),
            None => self.line_runs_once(l),
        }
    }

    /// `stretch_runs_once` by the line's setting alone.
    pub(super) fn stretch_set_once(&self, l: u32) -> bool {
        match self.line_stretch(l) {
            Some(s) => self.set_once(l, s.stops(self.line_stops(l)), s.path(self.line_path(l))),
            None => self.set_once(l, self.line_stops(l), self.line_path(l)),
        }
    }

    /// Fastest path from `from` to `to` over valid moves, by speed-limit running time. `arrived`
    /// is the side the train came in on at `from`, or None at a line's first stop. It leaves on
    /// the other side, or turns back there at `REVERSAL_SEARCH_S` extra; on the way it may turn
    /// back at a track end at the same cost, a station there or not (T-797, T-3595: never at a
    /// through station it passes or a junction). Returns the path and the side it arrives on at `to`.
    pub fn find_path(&self, from: u32, to: u32, arrived: Option<u8>) -> Option<(Vec<u32>, u8)> {
        self.find_path_side(from, to, arrived, None)
    }

    /// `find_path` arriving at `to` on the side `want` when given (T-798: a loop's closing hop
    /// arrives so the train carries on round).
    pub fn find_path_side(&self, from: u32, to: u32, arrived: Option<u8>, want: Option<u8>) -> Option<(Vec<u32>, u8)> {
        let ne = self.edge_count();
        let mut dist = vec![f64::INFINITY; 2 * ne];
        let mut prev = vec![u32::MAX; 2 * ne];
        let mut heap = BinaryHeap::new();
        let p = &self.node_ports[from as usize];
        for (k, &c) in p.ends().iter().enumerate() {
            let turn = if arrived == Some(p.side[k]) { REVERSAL_SEARCH_S } else { 0.0 };
            let (e, which) = (c >> 1, c & 1);
            let st = e << 1 | which; // leaving from end a = direction 0
            if !self.dir_ok(st) {
                continue;
            }
            dist[st as usize] = self.edge_time[e as usize] + turn;
            heap.push(Key(dist[st as usize], st));
        }
        while let Some(Key(d, st)) = heap.pop() {
            if d > dist[st as usize] {
                continue;
            }
            let (node, arr, mut on) = self.moves_after(st);
            let p = &self.node_ports[node as usize];
            let arr_side = arr.map_or(0, |i| p.side[i]);
            if node == to && want.map_or(true, |w| w == arr_side) {
                let mut path = vec![st];
                let mut cur = st;
                while prev[cur as usize] != u32::MAX {
                    cur = prev[cur as usize];
                    path.push(cur);
                }
                path.reverse();
                return Some((path, arr_side));
            }
            // a train may turn back at a track end (T-797 B3, T-3595), the way it came
            let back = if self.turn_end(node) { arr.or(Some(0)) } else { None };
            if let Some(i) = back {
                on |= (self.dir_ok(p.end[i]) as u8) << i;
            }
            for k in ends_in(on) {
                let nst = p.end[k];
                let e = nst >> 1;
                let nd = d + self.edge_time[e as usize] + if back == Some(k) { REVERSAL_SEARCH_S } else { 0.0 };
                if nd < dist[nst as usize] {
                    dist[nst as usize] = nd;
                    prev[nst as usize] = st;
                    heap.push(Key(nd, nst));
                }
            }
        }
        None
    }

    /// Fastest path through a list of stops, hop by hop; it may turn back at a stop.
    pub fn route(&self, stops: &[u32], ones: &[(u32, u8)]) -> Option<Vec<u32>> {
        if stops.iter().any(|&s| !self.node_ok(s) || self.node_platform[s as usize] == 0) { return None; }
        self.route_hops(stops, ones).ok()
    }

    /// `route` over stops that are all stations, or the first hop with no path (T-2895: hop k runs
    /// from stop k to stop k + 1). Past a one-way station on the way back's track (T-3565,
    /// `route_legs`), and a one-way stop the player set (`ones`, T-3711) the other way.
    pub fn route_hops(&self, stops: &[u32], ones: &[(u32, u8)]) -> Result<Vec<u32>, usize> {
        if let Some(p) = self.route_legs(stops, ones) {
            return Ok(p);
        }
        self.route_hops_plain(stops)
    }

    /// `route_hops` through every stop.
    pub fn route_hops_plain(&self, stops: &[u32]) -> Result<Vec<u32>, usize> {
        let mut path: Vec<u32> = vec![];
        let mut side = None;
        let closes = stops.len() >= 3 && stops[0] == stops[stops.len() - 1];
        for (k, w) in stops.windows(2).enumerate() {
            let want = if closes && k + 2 == stops.len() { self.carry_on_side(&path) } else { None };
            let carry = want.and_then(|_| self.find_path_side(w[0], w[1], side, want));
            let plain = self.find_path_side(w[0], w[1], side, None);
            let (p, s) = match (carry, plain) {
                (Some(c), Some(p)) => if self.carry_worth(&c.0, &p.0) { c } else { p },
                (c, p) => c.or(p).ok_or(k)?,
            };
            path.extend(p);
            side = Some(s);
        }
        Ok(path)
    }

    /// Whether any track joins two nodes, whatever way a train would have to turn on it (T-2895).
    pub fn joined(&self, a: u32, b: u32) -> bool {
        let mut seen = vec![false; self.node_count()];
        let mut queue = vec![a];
        seen[a as usize] = true;
        while let Some(n) = queue.pop() {
            if n == b {
                return true;
            }
            for &c in self.node_ports[n as usize].ends() {
                let m = self.path_nodes(c).1;
                if !seen[m as usize] {
                    seen[m as usize] = true;
                    queue.push(m);
                }
            }
        }
        false
    }

    /// The side a path ending at its first node must arrive on to carry straight on into its first
    /// entry (T-798: a circle line).
    pub fn carry_on_side(&self, path: &[u32]) -> Option<u8> {
        let &p0 = path.first()?;
        self.port_side(path_edge(p0), path_dir(p0)).map(|s| 1 - s)
    }

    /// Whether a loop's closing hop that carries on round (`carry`) is taken over the plain one,
    /// which may turn back at the first stop (`plain`): when it is at most `CARRY_SLACK_S` slower,
    /// so a closing never runs a long way round the network to avoid a turn (T-3021).
    pub fn carry_worth(&self, carry: &[u32], plain: &[u32]) -> bool {
        let t = |p: &[u32]| p.iter().map(|&x| self.edge_time[path_edge(x) as usize]).sum::<f64>();
        t(carry) <= t(plain) + CARRY_SLACK_S
    }

    /// T-3021: the quickest way on from a path's last node, round a loop of track and back onto the
    /// path itself, never turning back, within `budget` s of track time: its entries and the index k
    /// into the path's node list where it rejoins. `back`: headed back toward the path's start (the
    /// train leaves node k over `path[k - 1]` the other way, k >= 1, the latest k where several
    /// fit); else carrying on along `path[k]` (a balloon's first pass, k <= `limit`, the earliest).
    pub fn loop_back(&self, path: &[u32], back: bool, limit: usize, budget: f64) -> Option<(Vec<u32>, usize)> {
        let &last = path.last()?;
        let mut nodes = vec![self.path_nodes(path[0]).0];
        nodes.extend(path.iter().map(|&p| self.path_nodes(p).1));
        self.port_side(path_edge(last), 1 - path_dir(last))?;
        // per node of the path, the rejoins there: (k, the entry the train carries on into)
        let mut want: std::collections::HashMap<u32, Vec<(usize, u32)>> = Default::default();
        for k in 0..nodes.len().min(limit + 1) {
            let entry = if back {
                if k == 0 { continue; }
                path[k - 1] ^ 1
            } else if k < path.len() {
                path[k]
            } else {
                continue;
            };
            want.entry(nodes[k]).or_default().push((k, entry));
        }
        let ne = self.edge_count();
        let mut dist = vec![f64::INFINITY; 2 * ne];
        let mut prev = vec![u32::MAX; 2 * ne];
        let mut heap = BinaryHeap::new();
        let (node, _, on) = self.moves_after(last);
        let p = &self.node_ports[node as usize];
        for i in ends_in(on) {
            let c = p.end[i];
            dist[c as usize] = self.edge_time[(c >> 1) as usize];
            heap.push(Key(dist[c as usize], c));
        }
        while let Some(Key(d, st)) = heap.pop() {
            if d > budget {
                return None;
            }
            if d > dist[st as usize] {
                continue;
            }
            let (node, _, on) = self.moves_after(st);
            let p = &self.node_ports[node as usize];
            if let Some(ks) = want.get(&node) {
                let goes = |x: &&(usize, u32)| p.find(path_edge(x.1), path_dir(x.1)).is_some_and(|j| on >> j & 1 != 0);
                let hit = if back { ks.iter().rev().find(goes) } else { ks.iter().find(goes) };
                if let Some(&(k, _)) = hit {
                    let mut out = vec![st];
                    let mut cur = st;
                    while prev[cur as usize] != u32::MAX {
                        cur = prev[cur as usize];
                        out.push(cur);
                    }
                    out.reverse();
                    return Some((out, k));
                }
            }
            for i in ends_in(on) {
                let c = p.end[i];
                let nd = d + self.edge_time[(c >> 1) as usize];
                if nd < dist[c as usize] {
                    dist[c as usize] = nd;
                    prev[c as usize] = st;
                    heap.push(Key(nd, c));
                }
            }
        }
        None
    }

    /// Where to split an edge at chainage `s` without changing its geometry (notes/T-008.md 4.1):
    /// the new node, the first part (keeps the edge's id) and the second part. A split inside an
    /// arc gets a new PI on each side with the arc's radius; the neighbouring PIs whose legs
    /// change keep their current radius as a set one.
    pub fn split_plan(&self, e: u32, s: f64) -> Result<(NodeData, EdgeData, EdgeData), Issue> {
        if !self.edge_ok(e) {
            return Err(Issue::NoSuchEdge { edge: e });
        }
        geom::with_ramps(self.edge_ramps(e), || self.split_plan_in(e, s))
    }

    pub(super) fn split_plan_in(&self, e: u32, s: f64) -> Result<(NodeData, EdgeData, EdgeData), Issue> {
        if self.edge_crossover[e as usize] {
            return Err(Issue::Crossover { edge: e });
        }
        let v = self.edge_vertices(e);
        let f = self.fit(&v);
        let mut s = s;
        if s < SPLIT_ROOM || s > f.len - SPLIT_ROOM {
            return Err(Issue::SplitPoint { edge: e });
        }
        // Snap to a piece boundary within 2 m, so new PIs are never a hair from the node.
        for p in &f.pieces {
            for b in [p.s0, p.s0 + p.len] {
                if (s - b).abs() < 2.0 {
                    s = b;
                }
            }
        }
        if geom::on_ramp(&f.vert, s - 1.0, s + 1.0) {
            return Err(Issue::SplitPoint { edge: e });
        }
        let z = geom::height_at(&f.vert, s);
        let level = cost::level_of(z);
        let j = geom::piece_at(&f.pieces, s);
        let piece = f.pieces[j];
        let (x, y, th) = piece.at(s - piece.s0);
        let node = NodeData { x: quant(x), y: quant(y), level, flying: false };
        let freeze = |vv: &mut Vec<Pi>, k: usize, src: usize| {
            if k > 0 && k < vv.len() - 1 && vv[k].radius == 0.0 && f.radius[src] > 0.0 {
                vv[k].radius = quant(f.radius[src]);
            }
        };
        let marker = Pi::new(node.x, node.y, 0.0, level);
        let (mut p1, mut p2): (Vec<Pi>, Vec<Pi>);
        let inside_arc = piece.k != 0.0 && s > piece.s0 + 1e-9 && s < piece.s0 + piece.len - 1e-9;
        if inside_arc {
            let m = f.piece_at_vertex[j] as usize;
            // `r` is the ground radius the new PIs are given; the tangents are plane geometry.
            let scaled = self.frame.scaled();
            let r = quant(piece.ground_radius());
            let rt = if scaled { piece.radius() } else { r };
            let (d1, d2) = if scaled {
                let d1 = piece.plane_d(s - piece.s0);
                (d1, piece.plen - d1)
            } else {
                (s - piece.s0, piece.s0 + piece.len - s)
            };
            let t1 = rt * (d1 * piece.k.abs() / 2.0).tan();
            let t2 = rt * (d2 * piece.k.abs() / 2.0).tan();
            let mut pi1 = Pi::new(quant(x - t1 * th.cos()), quant(y - t1 * th.sin()), r, v[m].level);
            let mut pi2 = Pi::new(quant(x + t2 * th.cos()), quant(y + t2 * th.sin()), r, v[m].level);
            // On the millimetre grid a new PI a few metres from the node turns by a slightly
            // different angle than the arc did, and at the arc's radius its curve would overrun
            // the leg to the node ("does not fit", T-258). Each takes the radius that ends its
            // curve at the node; that is the arc's to well under 1% unless the PI is within a few
            // metres of the node, and then the split goes to the arc's end instead.
            // the radius at `p` (between `a` and `b`) whose tangent is `leg` long
            let fits = |a: (f64, f64), p: &Pi, b: (f64, f64), leg: f64| -> f64 {
                let half = geom::wrap((b.1 - p.y).atan2(b.0 - p.x) - (p.y - a.1).atan2(p.x - a.0)).abs() / 2.0;
                if half < 1e-12 { f64::INFINITY } else { leg / half.tan() }
            };
            let to_node = |p: &Pi| (node.x - p.x).hypot(node.y - p.y);
            // In ground metres (plane radius over k at the PI; k is 1 unscaled).
            let r1 = fits((v[m - 1].x, v[m - 1].y), &pi1, (node.x, node.y), to_node(&pi1)) / self.k_at(pi1.y);
            let r2 = fits((node.x, node.y), &pi2, (v[m + 1].x, v[m + 1].y), to_node(&pi2)) / self.k_at(pi2.y);
            if r1.min(r2) < 0.99 * r {
                let end = if s - piece.s0 < piece.s0 + piece.len - s { piece.s0 } else { piece.s0 + piece.len };
                return self.split_plan(e, end);
            }
            pi1.radius = r.min((r1 * 0.99999 / QUANT).floor() * QUANT);
            pi2.radius = r.min((r2 * 0.99999 / QUANT).floor() * QUANT);
            p1 = v[..m].to_vec();
            p1.push(pi1);
            p1.push(marker);
            p2 = vec![marker, pi2];
            p2.extend_from_slice(&v[m + 1..]);
            freeze(&mut p1, m - 1, m - 1);
            freeze(&mut p2, 2, m + 1);
        } else {
            // On the straight of leg i (a boundary snap lands on a straight's leg too).
            let i = if piece.k == 0.0 {
                f.piece_at_vertex[j] as usize
            } else if (s - piece.s0).abs() < 1e-9 {
                f.piece_at_vertex[j] as usize - 1
            } else {
                f.piece_at_vertex[j] as usize
            };
            // A PI at the split point (a joint kept for its level, T-622) is the node now.
            let at_node = |p: &Pi| (p.x - node.x).hypot(p.y - node.y) < 0.01;
            let drop_prev = i > 0 && at_node(&v[i]);
            let drop_next = i + 1 < v.len() - 1 && at_node(&v[i + 1]);
            p1 = v[..i + !drop_prev as usize].to_vec();
            p1.push(marker);
            p2 = vec![marker];
            p2.extend_from_slice(&v[i + 1 + drop_next as usize..]);
            if !drop_prev {
                freeze(&mut p1, i, i);
            }
            if !drop_next {
                freeze(&mut p2, 1, i + 1);
            }
            // T-622: a ramp is centred between the two vertices of the leg, so cutting the leg
            // would move it. Where the ramp lies on one straight, a vertex at each end of it (on
            // the track, no turn) holds it where it was, on whichever side of the cut it is.
            if !drop_prev && !drop_next && v[i].level != v[i + 1].level {
                let (l0, l1) = (v[i].level, v[i + 1].level);
                // the ramp as laid (T-3027: its length is the edge's), one vertex pair per level change
                let m = (0..i).filter(|&k| v[k].level != v[k + 1].level).count();
                let (r0, r1) = match (f.vert.get(1 + 2 * m), f.vert.get(2 + 2 * m)) {
                    (Some(a), Some(b)) => (a[0] - 0.01, b[0] + 0.01),
                    _ => {
                        let half = geom::ramps().per_level * (l1 as f64 - l0 as f64).abs() / 2.0 + 0.01;
                        let c = (f.vert_s[i] + f.vert_s[i + 1]) / 2.0;
                        (c - half, c + half)
                    }
                };
                let (j0, j1) = (geom::piece_at(&f.pieces, r0), geom::piece_at(&f.pieces, r1));
                if r0 > 0.5 && r1 < f.len - 0.5 && j0 == j1 && f.pieces[j0].k == 0.0 {
                    let at = |r: f64, level: i8| {
                        let (x, y, _) = geom::pos_at(&f.pieces, r);
                        Pi::new(quant(x), quant(y), 0.0, level)
                    };
                    let ramp = [at(r0, l0), at(r1, l1)];
                    if r1 < s {
                        p1.splice(p1.len() - 1..p1.len() - 1, ramp);
                    } else if r0 > s {
                        p2.splice(1..1, ramp);
                    }
                }
            }
        }
        // T-622: two curves back to back (a joined pair of tangent arcs) have no straight between
        // them to spare, and the new node and PIs sit on the millimetre grid, which turns the
        // legs by a hair (a 1,796 m curve's tangent grew 24.413 to 24.426 m). A curve with a set
        // radius that no longer fits gives up 0.05% of it until it does.
        for part in [&mut p1, &mut p2] {
            for _ in 0..10 {
                let bad = self.fit(part).clamped;
                if bad.is_empty() {
                    break;
                }
                for j in bad {
                    part[j].radius = (part[j].radius * 0.9995 / QUANT).floor() * QUANT;
                }
            }
        }
        let i = e as usize;
        let (tracks, built, off, kind, sep, oneway) = (self.edge_tracks[i], self.edge_built[i], self.edge_off[i], self.edge_kind[i], self.edge_sep[i], self.edge_oneway[i]);
        // A T-079 edge's clicked points go with the half they lie on; one at the split is the
        // node now.
        let (mut t1, mut t2) = (vec![], vec![]);
        for p in &self.edge_thru[i] {
            let sp = self.project(e, p.x, p.y).0;
            if sp < s - 5.0 {
                t1.push(*p);
            } else if sp > s + 5.0 {
                t2.push(*p);
            }
        }
        let (join, limit, ramp) = (self.edge_join[i], self.edge_limit[i], self.edge_ramp[i]);
        let a = EdgeData { a: self.edge_a[i], b: u32::MAX, tracks, pis: p1[1..p1.len() - 1].to_vec(), built, off, thru: t1, crossover: false, kind, sep, day: self.edge_day[i], oneway, join: [join[0], 0], limit, ramp };
        let b = EdgeData { a: u32::MAX, b: self.edge_b[i], tracks, pis: p2[1..p2.len() - 1].to_vec(), built, off, thru: t2, crossover: false, kind, sep, day: self.edge_day[i], oneway, join: [0, join[1]], limit, ramp };
        Ok((node, a, b))
    }
}
