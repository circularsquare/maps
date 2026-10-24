//! Lines (stops, look, cars, trains, loops) and undo.

use super::*;

#[wasm_bindgen]
impl TrackApi {
    // ---- lines
    /// Add a line with a whole-line fleet at each demand level.
    #[allow(clippy::too_many_arguments)]
    pub fn set_line_with_trains(&mut self, id: u32, stops: &[u32], high: u32, medium: u32, low: u32, dwell_s: f32, turnaround_s: f32, cars: u8, colour: u32, name: String) -> u32 {
        let Some(path) = self.w.net.route(stops) else {
            return self.refuse(Issue::LinePath { line: id });
        };
        self.run(Op::Line { id, data: Some(LineData { symbol: String::new(), trains: Some([high, medium, low]), auto: None, name, colour, stops: stops.to_vec(), path, tph: if stops.len() < 2 { [0.0; 3] } else { [6.0, 3.0, 1.0] }, dwell_s, turnaround_s, cars, fast: false }) })
    }
    /// Change a line's stops, keeping everything else; the path is found again.
    pub fn set_stops(&mut self, id: u32, stops: &[u32]) -> u32 {
        let Some(mut d) = self.w.net.line_data(id) else { return self.refuse(Issue::NoSuchLine { line: id }) };
        let Some(path) = self.w.net.route(stops) else { return self.refuse(Issue::LinePath { line: id }) };
        d.stops = stops.to_vec();
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
    pub fn copy_line(&mut self, source: u32, id: u32, name: String) -> u32 {
        let Some(mut d) = self.w.net.line_data(source) else { return self.refuse(Issue::NoSuchLine { line: source }) };
        d.name = name;
        let r = self.run(Op::Line { id, data: Some(d) });
        if r == 0 { self.w.net.next_line_number = self.w.net.next_line_number.saturating_add(1); }
        r
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
    pub fn remove_line(&mut self, id: u32) -> u32 {
        self.run(Op::Line { id, data: None })
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
