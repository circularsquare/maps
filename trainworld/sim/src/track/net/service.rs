//! Service rows, train slots and the clock-shaped reads (hours, mix, running trains).

use super::*;

impl Network {
    /// The network's frame (origin and kind).
    pub fn frame(&self) -> Frame {
        self.frame
    }

    /// T-3542: the slot serving each period (`params::SLOTS`): each period's row's earliest period
    /// on the line with the most rows (`refresh_slots`, `PERIOD_SLOT_N`). With every line on three
    /// rows it is `PERIOD_SLOT_3`, and a game costs and figures what it did with three levels.
    pub fn period_slot(&self) -> [usize; 5] {
        self.period_slot
    }
    /// Sets `period_slot` and keeps it through `refresh_slots` (tests: every period its own slot).
    #[cfg(test)]
    pub(crate) fn set_period_slot(&mut self, p: [usize; 5]) {
        self.period_slot = p;
        self.slot_list = slot_list(p);
        self.slots_pinned = true;
    }
    /// T-3542: `period_slot` from the lines' rows as they stand; true when it changed (the slots a
    /// change newly splits or joins are seeded or cleared by `TrackWorld::reslot`).
    pub fn refresh_slots(&mut self) -> bool {
        let moved = self.slots_from_rows();
        self.mix_moved = self.refresh_mix();
        moved
    }
    pub(super) fn slots_from_rows(&mut self) -> bool {
        #[cfg(test)]
        if self.slots_pinned { return false; }
        let n = (0..self.line_count()).filter(|&l| self.line_ok(l as u32)).map(|l| self.rows(l)).max().unwrap_or(ROWS_MIN);
        let p = PERIOD_SLOT_N[n - ROWS_MIN];
        if p == self.period_slot { return false; }
        self.period_slot = p;
        self.slot_list = slot_list(p);
        true
    }
    /// T-3542: every line's `Mix` from its hours, rows and the slots; returns the lines whose mix
    /// changed.
    pub fn refresh_mix(&mut self) -> Vec<u32> {
        let n = self.line_count();
        self.line_mix.resize(n, None);
        let mut moved = vec![];
        for l in 0..n {
            let m = if self.line_ok(l as u32) { Mix::of(self.hours(l), self.rows(l), self.period_slot).map(Box::new) } else { None };
            if m != self.line_mix[l] {
                self.line_mix[l] = m;
                moved.push(l as u32);
            }
        }
        moved
    }
    /// T-3542: the hours line `l` splits its day at: its own, else the game's (bus and ferry lines
    /// too, T-3767); a quad twin its line's.
    pub fn hours(&self, l: usize) -> Cuts {
        let l = match self.line_twin.get(l) { Some(super::twin::Twin::Of(p)) => *p as usize, _ => l };
        // a line's own hours go with its own periods (a line following the game's periods follows its hours)
        if self.line_rows.get(l).copied().unwrap_or(0) == 0 {
            return self.hours_default;
        }
        match self.line_x.get(l).map_or([0; 5], |x| x.hours) { [0, 0, 0, 0, 0] => self.hours_default, c => c }
    }
    /// T-3542: the game's hours (`line` None) or line `line`'s own (all 0 for the game's); returns
    /// the old setting. Every line it moves is rescheduled.
    pub fn set_hours(&mut self, line: Option<u32>, cuts: Cuts) -> Result<Cuts, Issue> {
        let old = match line {
            None if cuts_ok(cuts) => std::mem::replace(&mut self.hours_default, cuts),
            Some(id) if self.line_ok(id) && (cuts == [0; 5] || cuts_ok(cuts)) => std::mem::replace(&mut self.line_x[id as usize].hours, cuts),
            Some(id) if !self.line_ok(id) => return Err(Issue::NoSuchLine { line: id }),
            _ => return Err(Issue::LineSchedule { line: line.unwrap_or(u32::MAX) }),
        };
        let moved: Vec<u32> = (0..self.line_count() as u32).filter(|&l| self.line_ok(l) && match line {
            None => true,
            Some(id) => l == id || self.line_twin_of(l) == super::twin::Twin::Of(id),
        }).collect();
        self.touch.schedules.extend(moved);
        Ok(old)
    }
    /// T-3542: line `l`'s `Mix`, None where its hours put one row alone in every slot, the one its
    /// periods have at the default hours.
    pub fn mix(&self, l: usize) -> Option<&Mix> {
        self.line_mix.get(l).and_then(|m| m.as_deref())
    }
    /// T-3542: each place's share of slot `s`'s hours on line `l`, where more than one place runs in
    /// it (None: `place(l, s)` alone).
    pub fn mixed(&self, l: usize, s: usize) -> Option<&[f64; 5]> {
        self.mix(l).filter(|m| m.mixed[s]).map(|m| &m.w[s])
    }
    /// T-3542: after `period_slot` changed from `old`: each slot now worked out that was not takes
    /// what its periods' slot had (rates both kinds of day, phase, split, pace, kept lines), so the
    /// service starts from where it was; each no longer worked out goes back to as made (0, a split
    /// of half, no kept lines), as the slots no period uses stay.
    pub fn seed_slots(&mut self, old: [usize; 5]) {
        let was = slot_list(old);
        let before = |s: usize| was.0[..was.1].contains(&s);
        let new: Vec<(usize, usize)> = (0..SLOTS).filter(|&s| self.slot_worked(s) && !before(s)).map(|s| (s, old[s])).collect();
        let gone: Vec<usize> = (0..SLOTS).filter(|&s| !self.slot_worked(s) && before(s)).collect();
        fn fix<T: Copy>(v: &mut [[T; SLOTS]], new: &[(usize, usize)], gone: &[usize], none: T) {
            for a in v {
                for &(s, from) in new { a[s] = a[from]; }
                for &s in gone { a[s] = none; }
            }
        }
        fix(&mut self.line_tph, &new, &gone, 0.0);
        fix(&mut self.line_tph_rest, &new, &gone, 0.0);
        fix(&mut self.line_phase, &new, &gone, 0.0);
        fix(&mut self.line_split, &new, &gone, 0.5);
        fix(&mut self.line_pace, &new, &gone, 0.0);
        fix(&mut self.line_phase_rest, &new, &gone, 0.0);
        fix(&mut self.line_split_rest, &new, &gone, 0.5);
        fix(&mut self.line_pace_rest, &new, &gone, 0.0);
        for kept in [&mut self.settle_kept, &mut self.settle_kept_rest] {
            kept.retain(|k| !gone.contains(&k.1));
            let add: Vec<_> = kept.iter().flat_map(|k| new.iter().filter(move |n| n.1 == k.1).map(move |n| (k.0, n.0, k.2, k.3, k.4))).collect();
            kept.extend(add);
        }
    }
    /// T-3542: the rows line `l` sets its trains in (3, 4 or 5): its own, else the game's
    /// (`rows_default`), bus and ferry lines too (T-3767); a quad twin its line's.
    pub fn rows(&self, l: usize) -> usize {
        let l = match self.line_twin.get(l) { Some(super::twin::Twin::Of(p)) => *p as usize, _ => l };
        (match self.line_rows.get(l).copied().unwrap_or(0) { 0 => self.rows_default, r => r } as usize).clamp(ROWS_MIN, ROWS_MAX)
    }
    /// The place (`params::places`) whose value line `l` runs in slot `s`; where its hours put more
    /// than one there (`mixed`), the one with the most of them.
    pub fn place(&self, l: usize, s: usize) -> usize {
        match self.mix(l) {
            Some(m) => m.main[s] as usize,
            None => PERIOD_PLACE[self.rows(l) - ROWS_MIN][s],
        }
    }
    /// Whether place `k` of line `l` runs any of slot `s`'s hours.
    pub fn place_in(&self, l: usize, s: usize, k: usize) -> bool {
        match self.mixed(l, s) {
            Some(w) => w[k] > 0.0,
            None => self.place(l, s) == k,
        }
    }
    /// Line `l`'s rows in the day's order: each its place and the slots worked out it covers (one
    /// or more: a row of three rows covers three in a game of five; a row whose hours reach into
    /// another period, T-3542, that period's slot too).
    pub fn row_slots(&self, l: usize) -> Vec<(usize, Vec<usize>)> {
        let n = self.rows(l) - ROWS_MIN;
        if self.mix(l).is_some() {
            return ROW_PLACES[n].iter().map(|&k| (k, self.slots().filter(|&s| self.place_in(l, s, k)).collect())).collect();
        }
        ROW_PLACES[n].iter().map(|&k| {
            let mut s: Vec<usize> = (0..5).filter(|&p| PERIOD_PLACE[n][p] == k).map(|p| self.period_slot[p]).collect();
            s.dedup();
            (k, s)
        }).collect()
    }
    /// Line `l`'s whole trains at its five places, None while it has none allocated.
    pub fn trains_of(&self, l: usize) -> Option<[u32; 5]> {
        let t = self.line_trains.get(l).copied().flatten()?;
        Some(places(t, self.line_x.get(l).map_or([0; 2], |x| x.trains)))
    }
    /// Line `l`'s headways at its five places, None for a line set by count.
    pub fn auto_of(&self, l: usize) -> Option<[u32; 5]> {
        let a = self.line_auto.get(l).copied().flatten()?;
        Some(places(a, self.line_x.get(l).map_or([0; 2], |x| x.auto)))
    }
    /// Line `l`'s whole trains in slot `s` (0 for none): its place's (`place`).
    pub fn trains_at(&self, l: usize, s: usize) -> u32 {
        self.trains_of(l).map_or(0, |t| t[self.place(l, s)])
    }
    /// T-3542: line `l`'s trains in slot `s` as its hours share it (`mixed`): each place's count
    /// times its share of the slot's hours; `trains_at` where one place runs it.
    pub fn trains_in(&self, l: usize, s: usize) -> f64 {
        match (self.mixed(l, s), self.trains_of(l)) {
            (Some(w), Some(t)) => (0..5).filter(|&k| w[k] > 0.0).map(|k| w[k] * t[k] as f64).sum(),
            _ => self.trains_at(l, s) as f64,
        }
    }
    /// T-3542: the most whole trains any of line `l`'s places runs in slot `s` (the fleet there).
    pub fn trains_most(&self, l: usize, s: usize) -> u32 {
        match (self.mixed(l, s), self.trains_of(l)) {
            (Some(w), Some(t)) => (0..5).filter(|&k| w[k] > 0.0).map(|k| t[k]).max().unwrap_or(0),
            _ => self.trains_at(l, s),
        }
    }
    /// T-3542: what a settle keeps a line by in slot `s` (`settle_kept`): its trains there, or where
    /// its hours mix places a hash of their counts, so another count there is another state.
    pub fn trains_key(&self, l: usize, s: usize) -> u32 {
        match (self.mixed(l, s), self.trains_of(l)) {
            (Some(w), Some(t)) => (0..5).filter(|&k| w[k] > 0.0).fold(0x811c9dc5u32, |h, k| (h ^ t[k] ^ (k as u32) << 28).wrapping_mul(0x01000193)),
            _ => self.trains_at(l, s),
        }
    }
    /// T-3542: line `l`'s fleet per slot: each slot's most trains of the places it runs there
    /// (`trains_most`), 0 in the slots not worked out.
    pub fn slot_trains(&self, l: usize) -> [u32; SLOTS] {
        match (self.mix(l), self.trains_of(l)) {
            (Some(_), Some(_)) => std::array::from_fn(|s| if self.slot_worked(s) { self.trains_most(l, s) } else { 0 }),
            (_, t) => self.places_in(l, t.unwrap_or([0; 5]), 0),
        }
    }
    /// Line `l`'s headway in slot `s`, None for a line set by count.
    pub fn auto_at(&self, l: usize, s: usize) -> Option<u32> {
        self.auto_of(l).map(|a| a[self.place(l, s)])
    }
    /// Sets line `l`'s trains from its five places.
    pub fn set_trains_of(&mut self, l: usize, t: Option<[u32; 5]>) {
        self.line_trains[l] = t.map(|t| unplace(t).0);
        if let Some(t) = t { self.line_x[l].trains = unplace(t).1; }
    }
    /// Sets line `l`'s headways from its five places.
    pub fn set_auto_of(&mut self, l: usize, a: Option<[u32; 5]>) {
        self.line_auto[l] = a.map(|a| unplace(a).0);
        if let Some(a) = a { self.line_x[l].auto = unplace(a).1; }
    }
    /// The slots worked out, in order: those serving some period. The others are never written
    /// or read (they keep their defaults).
    pub fn slots(&self) -> impl Iterator<Item = usize> + Clone {
        let (l, n) = self.slot_list;
        l.into_iter().take(n)
    }
    /// The slot a readout of three rows (high, medium, low) reads row `r` of line `l` from: a line
    /// of four or five rows shows its rush or morning, evening and night there (`ROW_SLOT`).
    pub fn row_slot(&self, _l: usize, r: usize) -> usize {
        ROW_SLOT[r]
    }
    /// A slot array of three rows' values (a save's, by the row each slot's periods have on a line
    /// of three rows) spread over the slots worked out, `none` in the others.
    pub fn rows_in<T: Copy>(&self, r: [T; ROWS], none: T) -> [T; SLOTS] {
        std::array::from_fn(|s| if self.slot_worked(s) { r[SLOT_ROW[s]] } else { none })
    }
    /// Line `l`'s five places (`params::places`) spread over the slots worked out, each slot from
    /// the place it runs, `none` in the others.
    pub fn places_in<T: Copy>(&self, l: usize, p: [T; 5], none: T) -> [T; SLOTS] {
        std::array::from_fn(|s| if self.slot_worked(s) { p[self.place(l, s)] } else { none })
    }
    /// Whether slot `s` is worked out (`slots`).
    pub fn slot_worked(&self, s: usize) -> bool {
        self.slot_list.0[..self.slot_list.1].contains(&s)
    }
    /// A slot array read back per row, each from the row's slot (`ROW_SLOT`).
    pub fn rows_of<T: Copy>(&self, s: [T; SLOTS]) -> [T; ROWS] {
        slots_to_rows(s)
    }
    /// A slot array back as line `l`'s five places: each the most of the slots worked out it runs in
    /// (a row tied across slots takes its busiest's), a place the line does not use (midday on a line
    /// of three rows) its first row's.
    pub fn places_most<T: Copy + PartialOrd>(&self, l: usize, s: [T; SLOTS]) -> [T; 5] {
        let mut out: [Option<T>; 5] = [None; 5];
        for k in self.slots() {
            if let Some(w) = self.mixed(l, k) {
                for p in (0..5).filter(|&p| w[p] > 0.0) {
                    if out[p].map_or(true, |v| s[k] > v) { out[p] = Some(s[k]); }
                }
                continue;
            }
            let o = &mut out[self.place(l, k)];
            if o.map_or(true, |v| s[k] > v) { *o = Some(s[k]); }
        }
        let first = out[0].unwrap_or(s[HIGH]);
        out.map(|v| v.unwrap_or(first))
    }
    /// Whether two of line `l`'s five places agree at every place its rows use (`row_slots`).
    pub fn same_places<T: Copy + PartialEq>(&self, l: usize, a: [T; 5], b: [T; 5]) -> bool {
        ROW_PLACES[self.rows(l) - ROWS_MIN].iter().all(|&k| a[k] == b[k])
    }
    /// A slot array's midday and afternoon slots (places 3 and 4), as `Extra` keeps them.
    pub fn extra_of<T: Copy>(&self, s: [T; SLOTS]) -> [T; 2] {
        [s[1], s[2]]
    }

    /// T-2607: the trains line `l` runs at level `lev` over a round trip of `trip_s`: its count, but
    /// on a line keeping a headway no more than the headway needs (a fraction), the time left over
    /// spent waiting at the ends, so the headway set is the headway run.
    pub fn running_trains(&self, l: usize, lev: usize, trip_s: f64) -> f64 {
        // T-3542: a slot whose hours several places run: each place's trains for its share of them
        if let Some(w) = self.mixed(l, lev) {
            return (0..5).filter(|&k| w[k] > 0.0).map(|k| w[k] * self.running_borrowed(l, lev, k, trip_s)).sum();
        }
        self.running_place(l, lev, self.trains_at(l, lev), self.auto_at(l, lev), trip_s)
    }
    /// T-3542: the trains an hour place `k` of line `l` runs in slot `s` where its hours mix places
    /// there (`mixed`): the slot's rate `tph` (the places' time-weighted mean, as settled) shared
    /// out by what each place's trains run over the round trip `trip_s`. `tph` itself elsewhere.
    pub fn place_tph(&self, l: usize, s: usize, k: usize, tph: f32, trip_s: f64) -> f32 {
        let (Some(w), Some(t)) = (self.mixed(l, s), self.trains_of(l)) else { return tph };
        let auto = self.auto_of(l);
        let run = |k: usize| self.running_borrowed(l, s, k, trip_s);
        let all: f64 = (0..5).filter(|&j| w[j] > 0.0).map(|j| w[j] * run(j)).sum();
        if !(all > 0.0 && w[k] > 0.0 && trip_s > 0.0) { return 0.0; }
        // where the slot runs what its trains would (no full track cut it, within the settle's
        // tolerance), each place its own: a headway place exactly its headway
        if (tph as f64 * trip_s / (3600.0 * all) - 1.0).abs() < 2e-4 {
            return match auto.map(|a| a[k]) {
                Some(a) if a > 0 && t[k] as f64 >= trip_s / a as f64 && run(k) * a as f64 >= trip_s * (1.0 - 1e-9) => (3600.0 / a as f64) as f32,
                _ => (run(k) * 3600.0 / trip_s) as f32,
            };
        }
        (tph as f64 * run(k) / all) as f32
    }
    /// T-3542: the trains place `k` of line `l` runs in slot `lev` over a round trip of `trip_s`
    /// (`running_place`), where its hours reach into a period other than its own no more than it
    /// runs in its own (the rate of a slot it runs alone, the least of them): the borrowed hour of a
    /// row on full track never leaves more often than the row does on that track.
    pub fn running_borrowed(&self, l: usize, lev: usize, k: usize, trip_s: f64) -> f64 {
        let (Some(t), auto) = (self.trains_of(l), self.auto_of(l)) else { return 0.0 };
        let run = self.running_place(l, lev, t[k], auto.map(|a| a[k]), trip_s);
        let own =self.slots().filter(|&s| s != lev && self.mixed(l, s).is_none() && self.place(l, s) == k).map(|s| self.line_tph[l][s] as f64).reduce(f64::min);
        match own {
            Some(r) if trip_s > 0.0 => run.min(r * trip_s / 3600.0),
            _ => run,
        }
    }
    /// T-3542: `running_trains` of one place: `n` trains, headway `auto`, in slot `lev`.
    pub fn running_place(&self, l: usize, lev: usize, n: u32, auto: Option<u32>, trip_s: f64) -> f64 {
        let n = n as f64;
        match auto {
            Some(a) if a > 0 && trip_s > 0.0 => n.min(trip_s / a as f64),
            _ => {
                // T-2973: a count line in step with others at a merge
                let p = self.line_pace_of(l as u32)[lev];
                if p > 0.0 && trip_s > 0.0 { n.min(trip_s / p as f64) } else { n }
            }
        }
    }

    /// T-3521: a line's trains an hour at level `lev` on kind of day `row` (`params::WEEKDAYS`,
    /// `WEEKEND`): the weekdays' `line_tph`, or the weekends' settled rate when there was one.
    pub fn line_tph_on(&self, l: usize, row: usize, lev: usize) -> f32 {
        if row == WEEKEND && self.rest_solved {
            self.line_tph_rest.get(l).map_or(0.0, |t| t[lev])
        } else {
            self.line_tph.get(l).map_or(0.0, |t| t[lev])
        }
    }

    /// T-3521: whether any line's weekend setting is other than the weekdays' trains at 100%.
    pub fn any_weekend(&self) -> bool {
        self.line_weekend.iter().any(|w| *w != Weekend::default())
    }
    /// T-1925: a line's departure phase per level, seconds (`line_phase`; 0 where none is set).
    pub fn line_phase_of(&self, l: u32) -> [f32; SLOTS] {
        self.line_phase.get(l as usize).copied().unwrap_or([0.0; SLOTS])
    }
    /// T-2973: the headway a count line keeps to run in step at a merge per level (`line_pace`).
    pub fn line_pace_of(&self, l: u32) -> [f32; SLOTS] {
        self.line_pace.get(l as usize).copied().unwrap_or([0.0; SLOTS])
    }
    /// T-2972: the share of a line's spare time stood at its last stop per level (`line_split`).
    pub fn line_split_of(&self, l: u32) -> [f32; SLOTS] {
        self.line_split.get(l as usize).copied().unwrap_or([0.5; SLOTS])
    }
    /// T-3735: `line_phase_of` on weekends (`line_phase_rest`, else the weekdays').
    pub fn line_phase_rest_of(&self, l: u32) -> [f32; SLOTS] {
        self.line_phase_rest.get(l as usize).copied().unwrap_or_else(|| self.line_phase_of(l))
    }
    /// T-3735: `line_split_of` on weekends (`line_split_rest`, else the weekdays').
    pub fn line_split_rest_of(&self, l: u32) -> [f32; SLOTS] {
        self.line_split_rest.get(l as usize).copied().unwrap_or_else(|| self.line_split_of(l))
    }
    /// T-3584: `line_pace_of` on weekends (`line_pace_rest`; the weekdays' before a weekend settle
    /// has paced them).
    pub fn line_pace_rest_of(&self, l: u32) -> [f32; SLOTS] {
        if self.line_pace_rest.is_empty() { self.line_pace_of(l) } else { self.line_pace_rest.get(l as usize).copied().unwrap_or([0.0; SLOTS]) }
    }
    /// T-3735: whether line `l` runs other rates on weekends than on weekdays at some slot, so its
    /// weekend departures are laid out on their own (`Service::lattice`, `TrackWorld::place_weekend`);
    /// every other line runs its weekday departures every day.
    pub fn runs_own_weekend(&self, l: usize) -> bool {
        self.rest_solved && l < self.line_count() && (0..SLOTS).any(|lev| self.line_tph_on(l, WEEKEND, lev) != self.line_tph[l][lev])
    }
    /// T-3521: a headway line's headways on weekends (`line_auto` while the weekends settle): the
    /// weekdays' at its percent (`weekend_headway`), none where its weekend counts are set by hand
    /// (they run as a count line's, bar the headways stepped to, T-3557). None for a count line.
    /// At the line's five places (`params::places`).
    pub fn weekend_auto(&self, l: usize) -> Option<[u32; 5]> {
        let a = self.auto_of(l)?;
        match self.line_weekend.get(l).copied().unwrap_or_default() {
            // a headway stepped to by hand keeps the cap on the trains set to run it (T-3557); a level
            // without one runs as a count line
            Weekend { counts: Some(_), headways, .. } => {
                let h = places(headways, self.line_x[l].weekend_headways);
                (!self.same_places(l, h, [0; 5])).then_some(h)
            }
            Weekend { share: 100, .. } => Some(a),
            Weekend { share: p, .. } => Some(a.map(|h| weekend_headway(h, p))),
        }
    }
    /// Seconds a line's local time runs ahead of the game clock (T-309): the region of its first
    /// stop's. 0 without a time zone table or stops.
    pub fn line_shift(&self, l: u32) -> f64 {
        if !self.time_zones.set || !self.line_ok(l) {
            return 0.0;
        }
        if let Some(p) = self.route_points(l) {
            return p.first().map_or(0.0, |p| self.line_shift_at(p[0], p[1]));
        }
        let Some(&s) = self.line_stops(l).first() else { return 0.0 };
        self.line_shift_at(self.node_x[s as usize], self.node_y[s as usize])
    }
    /// Seconds local time runs ahead of the game clock at a plane point (T-309).
    pub fn line_shift_at(&self, x: f64, y: f64) -> f64 {
        if !self.time_zones.set {
            return 0.0;
        }
        let (lon, lat) = self.frame.to_lonlat(x, y);
        self.time_zones.shift_at(lon, lat)
    }

    /// Plane metres per ground metre at plane `y`: `cosh(y / R)` in the Mercator frame, exactly 1
    /// in the others (frame.rs). Metre figures entering the plane are multiplied by it.
    #[inline]
    pub fn k_at(&self, y: f64) -> f64 {
        if self.frame.scaled() {
            super::frame::merc_k(y)
        } else {
            1.0
        }
    }
}
