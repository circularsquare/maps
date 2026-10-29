//! Lines (stops, look, cars, trains, loops) and undo.

use super::*;

#[wasm_bindgen]
impl TrackApi {
    // ---- lines
    /// Add a line with a whole-line fleet at each demand level.
    #[allow(clippy::too_many_arguments)]
    pub fn set_line_with_trains(&mut self, id: u32, stops: &[u32], high: u32, medium: u32, low: u32, dwell_s: f32, turnaround_s: f32, cars: u8, colour: u32, name: String) -> u32 {
        let Some((stops, path)) = edited_stops(&self.w.net, stops) else {
            return self.refuse(Issue::LinePath { line: id });
        };
        self.run(Op::Line { id, data: Some(LineData { symbol: String::new(), trains: Some([high, medium, low]), auto: None, name, colour, tph: if stops.len() < 2 { [0.0; 3] } else { [6.0, 3.0, 1.0] }, stops, path, dwell_s, turnaround_s, cars, fast: false, bus: None, group: 0, group_name: String::new(), group_look: Default::default() }) })
    }
    /// Change a line's stops, keeping everything else; the path is found again.
    pub fn set_stops(&mut self, id: u32, stops: &[u32]) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        let Some((stops, path)) = edited_stops(&self.w.net, stops) else { return self.refuse(Issue::LinePath { line: id }) };
        d.stops = stops;
        d.path = path;
        self.run(Op::Line { id, data: Some(d) })
    }
    /// T-1791: take stop `i` out and keep the path: a stop between the ends leaves the track as it
    /// is (`set_stops` would find the route again, and could move a local onto faster track); an
    /// end stop's removal cuts the path back to the new end. Refused if a train turns back there
    /// (a reversal falls on a stop), for a loop's ends, and below two stops.
    pub fn drop_stop(&mut self, id: u32, i: u32) -> u32 {
        self.drop_stops(id, i as usize, i as usize)
    }
    /// T-1791: end the line at stop `i`: every stop past it toward the nearer end goes, and the path
    /// is cut there (the last stop's end when the two are level).
    pub fn end_line_at(&mut self, id: u32, i: u32) -> u32 {
        let Some(d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        let (i, n) = (i as usize, d.stops.len());
        if i >= n || n < 3 || i == 0 || i == n - 1 {
            return self.refuse(Issue::LinePath { line: id });
        }
        if i < n - 1 - i { self.drop_stops(id, 0, i - 1) } else { self.drop_stops(id, i + 1, n - 1) }
    }
    /// T-1791: make a station the line passes a stop, keeping the path.
    pub fn stop_passed(&mut self, id: u32, station: u32) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        let Some(stops) = stopped_at(&self.w.net, &d, station) else { return self.refuse(Issue::LinePath { line: id }) };
        d.stops = stops;
        self.run(Op::Line { id, data: Some(d) })
    }
    /// Stops `lo..=hi` out; `lo..=hi` is one interior stop, a start or an end of the line.
    fn drop_stops(&mut self, id: u32, lo: usize, hi: usize) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        let Some((stops, path)) = dropped(&self.w.net, &d, lo, hi) else { return self.refuse(Issue::LinePath { line: id }) };
        d.stops = stops;
        d.path = path;
        self.run(Op::Line { id, data: Some(d) })
    }
    /// Rename or recolour a line; its path stays.
    pub fn set_line_look(&mut self, id: u32, name: String, colour: u32) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        d.name = name;
        d.colour = colour;
        self.run(Op::Line { id, data: Some(d) })
    }
    /// Custom badge symbol; omitted look edits retain it.
    pub fn set_line_badge(&mut self, id: u32, name: String, colour: u32, symbol: String) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        d.name = name; d.colour = colour; d.symbol = symbol;
        self.run(Op::Line { id, data: Some(d) })
    }
    /// A line's train length in cars (T-733), 0 for the most its shortest platform fits. Kept as
    /// set; while a platform is shorter the trains run at what it fits (`build_line`).
    pub fn set_line_cars(&mut self, id: u32, cars: u8) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        d.cars = cars;
        self.run(Op::Line { id, data: Some(d) })
    }
    /// A copy has the source's stops and settings but no trains until the player sets them
    /// (T-1161): headways stay headways, at none.
    pub fn copy_line(&mut self, source: u32, id: u32, name: String) -> u32 {
        let Some(mut d) = self.w.net.line_data(source) else { return self.refuse(Issue::NoSuchLine { line: source }) };
        d.name = name;
        // a bus line's copy is a planned route, paid for on Construct (T-1464)
        if let Some(r) = d.bus.as_mut() {
            r.built = false;
        }
        d.trains = Some([0; 3]);
        d.tph = [0.0; 3];
        if d.auto.is_some() {
            d.auto = Some([0; 3]);
        }
        // T-1791: the copy joins its source's group, a new one with the source if it had none
        let mut ops = vec![];
        if d.group == 0 {
            d.group = self.new_line_group();
            let mut s = self.w.net.line_data(source).unwrap();
            s.group = d.group;
            ops.push(Op::Line { id: source, data: Some(s) });
        }
        ops.push(Op::Line { id, data: Some(d) });
        let r = self.run(if ops.len() == 1 { ops.pop().unwrap() } else { Op::Batch(ops) });
        if r == 0 { self.w.net.next_line_number = self.w.net.next_line_number.saturating_add(1); }
        r
    }

    // ---- line groups (T-1791): a grouping for the Lines tab; no service edit
    /// A group number no live line has.
    pub fn new_line_group(&self) -> u32 {
        let n = &self.w.net;
        (0..n.line_count()).filter(|&l| n.line_ok(l as u32)).map(|l| n.line_group[l]).max().unwrap_or(0) + 1
    }
    /// A line's group, 0 for none.
    pub fn line_group(&self, id: u32) -> u32 {
        if self.w.net.line_ok(id) { self.w.net.line_group[id as usize] } else { 0 }
    }
    /// A line's group's name as the player set it, empty for the default.
    pub fn line_group_name(&self, id: u32) -> String {
        if self.w.net.line_ok(id) { self.w.net.line_group_name[id as usize].clone() } else { String::new() }
    }
    /// Put `lines` in `group` (0: out of any group), one undoable edit. A line joining takes the
    /// group's name; a group left with one line stops being one.
    pub fn set_line_group(&mut self, lines: &[u32], group: u32) -> u32 {
        let n = &self.w.net;
        if let Some(&l) = lines.iter().find(|&&l| !n.line_ok(l)) {
            return self.refuse(Issue::NoSuchLine { line: l });
        }
        let live = || (0..n.line_count() as u32).filter(|&l| n.line_ok(l));
        let meta = |l: u32| (n.line_group_name[l as usize].clone(), n.line_group_look[l as usize].clone());
        let joined = if group == 0 { Default::default() } else { live().find(|&l| n.line_group[l as usize] == group && !lines.contains(&l)).map_or_else(Default::default, meta) };
        let mut after: Vec<(u32, u32, (String, GroupLook))> = live().map(|l| {
            if lines.contains(&l) { (l, group, joined.clone()) } else { (l, n.line_group[l as usize], meta(l)) }
        }).collect();
        // a group of one dissolves
        for k in 0..after.len() {
            let g = after[k].1;
            if g != 0 && after.iter().filter(|x| x.1 == g).count() == 1 {
                after[k] = (after[k].0, 0, Default::default());
            }
        }
        let ops: Vec<Op> = after.into_iter().filter(|(l, g, m)| n.line_group[*l as usize] != *g || meta(*l) != *m).map(|(l, g, (s, look))| {
            let mut d = n.line_data(l).unwrap();
            d.group = g;
            d.group_name = s;
            d.group_look = look;
            Op::Line { id: l, data: Some(d) }
        }).collect();
        if ops.is_empty() {
            return 0;
        }
        self.run(Op::Batch(ops))
    }
    /// Name a group (empty: its oldest line's name again), one undoable edit.
    pub fn set_line_group_name(&mut self, group: u32, name: String) -> u32 {
        self.edit_group(group, |d| d.group_name = name.clone())
    }
    /// T-1813: a group's bullet text (empty: from its name) and colour (negative: the colour most
    /// of its lines share), one undoable edit.
    pub fn set_line_group_look(&mut self, group: u32, symbol: String, colour: i32) -> u32 {
        let look = GroupLook { symbol, colour: u32::try_from(colour).ok().map(|c| c & 0xff_ffff) };
        self.edit_group(group, |d| d.group_look = look.clone())
    }
    /// A group's bullet text set by hand, empty for none.
    pub fn line_group_symbol(&self, id: u32) -> String {
        if self.w.net.line_ok(id) { self.w.net.line_group_look[id as usize].symbol.clone() } else { String::new() }
    }
    /// A group's bullet colour set by hand, -1 for none.
    pub fn line_group_colour(&self, id: u32) -> i32 {
        if self.w.net.line_ok(id) { self.w.net.line_group_look[id as usize].colour.map_or(-1, |c| c as i32) } else { -1 }
    }
    // ---- T-1813: the Lines and Stations tabs' hand order, a setting outside the undo history
    /// Set the hand order of lines (`kind` 0) or stations (1) to `ids`, first first; empty: none.
    pub fn set_list_order(&mut self, kind: u8, ids: &[u32]) {
        let n = &mut self.w.net;
        let len = if kind == 0 { n.line_count() } else { n.node_count() };
        let rank = if kind == 0 { &mut n.line_rank } else { &mut n.node_rank };
        rank.clear();
        rank.resize(len, 0);
        for (i, &id) in ids.iter().enumerate() {
            if (id as usize) < len && rank[id as usize] == 0 { rank[id as usize] = i as u32 + 1; }
        }
    }
    /// The hand order of lines (`kind` 0) or stations (1): the live ids with a place, first first.
    pub fn list_order(&self, kind: u8) -> Vec<u32> {
        let n = &self.w.net;
        let (rank, ok): (&[u32], &dyn Fn(u32) -> bool) = if kind == 0 { (&n.line_rank, &|l| n.line_ok(l)) } else { (&n.node_rank, &|k| n.node_ok(k)) };
        let mut v: Vec<(u32, u32)> = rank.iter().enumerate().filter(|&(id, &r)| r > 0 && ok(id as u32)).map(|(id, &r)| (r, id as u32)).collect();
        v.sort_unstable();
        v.into_iter().map(|x| x.1).collect()
    }
    pub fn next_line_name(&mut self) -> String {
        loop {
            let name = format!("Line {}", self.w.net.next_line_number);
            self.w.net.next_line_number = self.w.net.next_line_number.saturating_add(1);
            if !(0..self.w.net.line_count() as u32).any(|l| self.w.net.line_ok(l) && self.w.net.line_name[l as usize] == name) { return name; }
        }
    }
    /// Try every insertion slot using the same turn-valid route as set_stops.
    pub fn insert_stop(&self, line: u32, station: u32) -> Vec<f64> {
        best_insert(&self.w.net, line, station, &mut HopMemo::new()).map_or(vec![], |(stops, _)| stops.iter().map(|&s| s as f64).collect())
    }

    /// The stations a line passes without stopping (T-646): `[gap, node]` pairs in path order, gap
    /// i lying between stop i and stop i + 1. Only stations on the line's own path count, not
    /// those on a quad pair beside it; each once.
    pub fn passed_stations(&self, line: u32) -> Vec<f64> {
        let net = &self.w.net;
        if !net.line_ok(line) {
            return vec![];
        }
        let stops = net.line_stops(line);
        let path = net.line_path(line);
        let Some(&first) = path.first() else { return vec![] };
        let mut nodes = vec![net.end_node(first >> 1, first & 1)];
        nodes.extend(path.iter().map(|&p| net.end_node(p >> 1, 1 - (p & 1))));
        let (mut gap, mut out, mut seen) = (0usize, vec![], vec![]);
        for n in nodes {
            if gap + 1 < stops.len() && n == stops[gap + 1] {
                gap += 1;
            } else if net.is_station(n) && !stops.contains(&n) && !seen.contains(&n) && gap + 1 < stops.len() {
                seen.push(n);
                out.extend([gap as f64, n as f64]);
            }
        }
        out
    }

    /// How every station stands to a line while its stops are edited (T-233, T-237): per station
    /// node, `[node, tier, reversal]`, tier 0 a stop of the line, 1 adding it keeps the line's
    /// path (a station on it, or past an end on track the path can carry on over: the path is
    /// extended), 2 adding it changes the path, 3 it cannot be added, 4 adding it makes trains
    /// reverse at the station `reversal` where they did not (T-797; -1 for the other tiers), 5 the
    /// first stop, where a click closes the line into a loop, 6 a loop's first stop, where a click
    /// opens it (T-798). Worked out with `insert_stop`'s own search, so the markers say what a click does. Empty
    /// if the line is not there.
    pub fn stop_tiers(&self, line: u32) -> Vec<f64> {
        let net = &self.w.net;
        if !net.line_ok(line) {
            return vec![];
        }
        let stops = net.line_stops(line);
        let old = net.line_path(line);
        let old_turns = turn_nodes(net, old);
        let reach = reachable_from(net, stops);
        let mut memo = HopMemo::new();
        let mut out = vec![];
        for n in 0..net.node_count() as u32 {
            if !net.node_ok(n) || net.node_platform[n as usize] == 0 {
                continue;
            }
            let (tier, at) = if stops.first() == Some(&n) && closed(stops) {
                (6.0, -1.0)
            } else if stops.first() == Some(&n) && close_path(net, line, &mut memo).is_some() {
                (5.0, -1.0)
            } else if stops.contains(&n) {
                (0.0, -1.0)
            } else if !stops.is_empty() && !reach[n as usize] {
                (3.0, -1.0)
            } else {
                match best_insert(net, line, n, &mut memo) {
                    None => (3.0, -1.0),
                    Some((_, path)) => match turn_nodes(net, &path).into_iter().find(|t| !old_turns.contains(t)) {
                        Some(t) => (4.0, t as f64),
                        None if old.is_empty() || path.windows(old.len()).any(|w| w == old) => (1.0, -1.0),
                        None => (2.0, -1.0),
                    },
                }
            };
            out.extend([n as f64, tier, at]);
        }
        out
    }

    /// A line's loop (T-798): 0 none, 1 a circle, 2 a balloon.
    pub fn line_loop(&self, line: u32) -> u8 {
        match self.w.svc.lines.get(line as usize).filter(|l| l.ok).map(|l| l.kind) {
            Some(LoopKind::Circle) => 1,
            Some(LoopKind::Balloon) => 2,
            _ => 0,
        }
    }

    /// The stops where a line's trains reverse partway, as indices into its stops (T-797).
    pub fn line_reversals(&self, line: u32) -> Vec<f64> {
        let Some(ls) = self.w.svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![] };
        ls.runs[0].turn.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as f64).collect()
    }

    /// The bare track ends where a line's trains reverse between stops, as nodes in path order
    /// (T-797 B3).
    pub fn line_end_turns(&self, line: u32) -> Vec<f64> {
        let Some(ls) = self.w.svc.lines.get(line as usize).filter(|l| l.ok) else { return vec![] };
        let run = &ls.runs[0];
        run.ends.iter().map(|x| run.nodes[x.0 as usize] as f64).collect()
    }
    /// Remove a line. A constructed bus route goes for good, like constructed track (T-1464).
    pub fn remove_line(&mut self, id: u32) -> u32 {
        let mut op = Op::Line { id, data: None };
        // the last other line of its group leaves it too (T-1791)
        let n = &self.w.net;
        let g = self.line_group(id);
        let rest: Vec<u32> = (0..n.line_count() as u32).filter(|&l| g != 0 && l != id && n.line_ok(l) && n.line_group[l as usize] == g).collect();
        if let [other] = rest[..] {
            let mut d = n.line_data(other).unwrap();
            d.group = 0;
            op = Op::Batch(vec![op, Op::Line { id: other, data: Some(d) }]);
        }
        if self.w.net.line_bus.get(id as usize).and_then(|b| b.as_ref()).is_some_and(|r| r.built) {
            return self.run_charged(op);
        }
        self.run(op)
    }

    /// Allocate whole trains to the entire line, across both directions.
    pub fn set_train_counts(&mut self, id: u32, high: u32, medium: u32, low: u32) -> u32 {
        let Some(d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        if d.auto.is_some() {
            return self.run(Op::AutoTrains { id, auto: None, trains: Some([high, medium, low]) });
        }
        self.run(Op::TrainCounts { id, trains: Some([high, medium, low]), tph: d.tph })
    }

    /// T-625: the line's counts follow a headway in seconds per level (0 = no service): the
    /// fewest trains that keep it over the round trip, worked out again after every edit. Switching
    /// to headways keeps the trains the line has (T-752).
    pub fn set_auto_headway(&mut self, id: u32, high: u32, medium: u32, low: u32) -> u32 {
        if !self.w.net.line_ok(id) { return self.refuse(Issue::NoSuchLine { line: id }) }
        let auto = [high, medium, low];
        let svc = self.w.svc.lines.get(id as usize);
        let trip = svc.map_or([0.0; 3], |s| s.round_trip);
        let trains = match (self.w.net.line_auto[id as usize], self.w.net.line_trains[id as usize]) {
            (None, Some(t)) => t,
            (None, None) => svc.map_or([0; 3], |s| s.trains),
            _ => std::array::from_fn(|lev| auto_trains(trip[lev], auto[lev])),
        };
        self.run(Op::AutoTrains { id, auto: Some(auto), trains: Some(trains) })
    }

    /// T-625: back to counts set by hand, as they stand.
    pub fn clear_auto_headway(&mut self, id: u32) -> u32 {
        let Some(d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        self.run(Op::AutoTrains { id, auto: None, trains: d.trains })
    }

    /// The line's headways in seconds per level when its counts follow them, else empty (T-625).
    pub fn line_auto(&self, id: u32) -> Vec<f64> {
        if !self.w.net.line_ok(id) { return vec![]; }
        self.w.net.line_auto[id as usize].map_or(vec![], |a| a.iter().map(|&s| s as f64).collect())
    }

    // ---- undo
    pub fn can_undo(&self) -> bool {
        self.w.can_undo()
    }
    pub fn can_redo(&self) -> bool {
        self.w.can_redo()
    }
    pub fn undo(&mut self) -> u32 {
        self.step(false)
    }
    pub fn redo(&mut self) -> u32 {
        self.step(true)
    }
    pub(super) fn step(&mut self, redo: bool) -> u32 {
        let fleet = self.fleet;
        let at = self.edit_start();
        let before = self.w.svc.before_edit(&self.w.net, at);
        let cash = self.spendable();
        match self.w.step_paid(redo, cash, &|w| trains_to_buy(w, fleet)) {
            Some(r) => self.settle(r, &before),
            None => {
                self.money_short = None;
                self.issues.clear();
                1
            }
        }
    }

    /// The last refusal's problems as kinds the UI turns into plain words, one per line:
    /// the `Issue` variant (`GroundOverWater`, `Crossing`, `CrossingOverlap`, ...), or
    /// `Geom:<GeomIssue variant>`, or `Money:<need>:<have>` (US$M).
    pub fn issue_kinds(&self) -> String {
        if let Some((need, have)) = self.money_short {
            return format!("Money:{need:.1}:{have:.1}");
        }
        let mut v: Vec<String> = self.issues.iter().map(issue_kind).collect();
        v.dedup();
        v.join("\n")
    }

}

/// A stop repeated straight after itself counts once (T-1242): a loop's middle stop taken out
/// leaves a one-stop line, not a refusal.
fn distinct_stops(stops: &[u32]) -> Vec<u32> {
    let mut s = stops.to_vec();
    s.dedup();
    s
}

/// The stops a stop edit leaves and their path. Stops closed into a loop whose path only goes
/// out and back the same way (a lollipop's loop stops taken out) make no loop (T-1273): the line
/// ends where it turns, a terminus, rather than reversing partway there.
fn edited_stops(net: &Network, stops: &[u32]) -> Option<(Vec<u32>, Vec<u32>)> {
    let stops = distinct_stops(stops);
    let path = net.route(&stops)?;
    let n = path.len();
    if !closed(&stops) || n == 0 || (0..n).any(|i| path[i] != path[n - 1 - i] ^ 1) {
        return Some((stops, path));
    }
    let nodes: Vec<u32> = std::iter::once(net.path_nodes(path[0]).0).chain(path.iter().map(|&p| net.path_nodes(p).1)).collect();
    let at = super::super::net::match_stops(&nodes, &stops, &net.stop_reversals(&path))?;
    let kept: Vec<u32> = stops.iter().zip(&at).filter(|x| *x.1 <= n / 2).map(|x| *x.0).collect();
    let kept = distinct_stops(&kept);
    let path = net.route(&kept)?;
    Some((kept, path))
}

/// A path's node list, `path.len() + 1` nodes.
fn path_node_list(net: &Network, path: &[u32]) -> Vec<u32> {
    path.first().map_or(vec![], |&f| std::iter::once(net.path_nodes(f).0).chain(path.iter().map(|&p| net.path_nodes(p).1)).collect())
}

/// T-1791: a line's stops and path with stops `lo..=hi` out (one interior stop, or a run at the start
/// or the end), the path kept between the new first and last stop. None if a train reverses at a
/// stop that goes, for a loop's ends, below two stops, or for a bus route.
fn dropped(net: &Network, d: &LineData, lo: usize, hi: usize) -> Option<(Vec<u32>, Vec<u32>)> {
    let n = d.stops.len();
    if d.bus.is_some() || lo > hi || hi >= n || !(lo == hi || lo == 0 || hi == n - 1) || n - (hi - lo + 1) < 2 {
        return None;
    }
    if closed(&d.stops) && (lo == 0 || hi == n - 1 || n - (hi - lo + 1) < 3) {
        return None;
    }
    let at = super::super::net::match_stops(&path_node_list(net, &d.path), &d.stops, &net.stop_reversals(&d.path))?;
    let stops: Vec<u32> = d.stops[..lo].iter().chain(&d.stops[hi + 1..]).copied().collect();
    if stops.windows(2).any(|w| w[0] == w[1]) {
        return None;
    }
    let first = if lo == 0 { hi + 1 } else { 0 };
    let last = if hi == n - 1 { lo - 1 } else { n - 1 };
    let path = d.path[at[first]..at[last]].to_vec();
    super::super::net::match_stops(&path_node_list(net, &path), &stops, &net.stop_reversals(&path))?;
    Some((stops, path))
}

/// T-1791: a line's stops with a station its path passes put in, the path untouched. None if it is
/// a stop already, not on the path, or the path would turn back where no stop is.
fn stopped_at(net: &Network, d: &LineData, station: u32) -> Option<Vec<u32>> {
    if d.bus.is_some() || d.stops.contains(&station) || !net.is_station(station) {
        return None;
    }
    let nodes = path_node_list(net, &d.path);
    let must = net.stop_reversals(&d.path);
    let at = super::super::net::match_stops(&nodes, &d.stops, &must)?;
    nodes.iter().enumerate().filter(|x| *x.1 == station).find_map(|(k, _)| {
        let pos = at.iter().filter(|&&a| a < k).count();
        let mut stops = d.stops.clone();
        if pos == 0 || pos == stops.len() {
            return None;
        }
        stops.insert(pos, station);
        super::super::net::match_stops(&nodes, &stops, &must).map(|_| stops)
    })
}

impl TrackApi {
    /// Change every line of a group (T-1791, T-1813) in one undoable edit.
    fn edit_group(&mut self, group: u32, f: impl Fn(&mut LineData)) -> u32 {
        let n = &self.w.net;
        let ops: Vec<Op> = (0..n.line_count() as u32).filter(|&l| group != 0 && n.line_ok(l) && n.line_group[l as usize] == group).map(|l| {
            let mut d = n.line_data(l).unwrap();
            f(&mut d);
            Op::Line { id: l, data: Some(d) }
        }).collect();
        if ops.is_empty() {
            return self.refuse(Issue::NoSuchLine { line: u32::MAX });
        }
        self.run(Op::Batch(ops))
    }
}
