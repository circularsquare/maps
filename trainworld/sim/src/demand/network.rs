//! The player's network as the demand kernel sees it: stations, lines as ordered stops, run time
//! per hop, headway per period, seats and crush load per train. Plus a hand-made New York-ish
//! network for the spike (25 lines, ~400 stations, radial, crosstown and orbital).

/// Morning peak, midday, evening peak, evening, night: the periods of `track::params::PERIOD_HOURS`,
/// the one definition schedules and demand share.
pub const PERIODS: usize = crate::track::params::PERIOD_HOURS.len();

#[derive(Clone)]
pub struct Network {
    /// Station position, metres from the pack origin.
    pub st_x: Vec<f32>,
    pub st_y: Vec<f32>,
    pub line_name: Vec<String>,
    /// CSR: stops of line l are `line_stops[line_off[l]..line_off[l+1]]`.
    pub line_off: Vec<u32>,
    pub line_stops: Vec<u32>,
    /// Minutes from stop k to stop k+1 (parallel to `line_stops`; 0 on the last stop).
    pub hop_min: Vec<f32>,
    /// Minutes from stop k to stop k-1, the reverse run (parallel to `line_stops`; 0 on the first
    /// stop). Empty: the reverse run takes `hop_min`'s times.
    pub hop_back: Vec<f32>,
    pub line_speed_kmh: Vec<f32>,
    pub line_headway_min: Vec<[f32; PERIODS]>,
    pub line_seats: Vec<f32>,
    pub line_crush: Vec<f32>,
}

#[derive(Clone, Copy)]
enum Kind {
    Subway,
    Light,
    Commuter,
    Orbital,
}

impl Kind {
    /// Stop spacing in km at distance r from the centre.
    fn spacing(self, r: f64) -> f64 {
        match self {
            Kind::Subway => {
                if r < 8.0 {
                    0.9
                } else {
                    1.4
                }
            }
            Kind::Light => 1.1,
            Kind::Commuter => 5.0,
            Kind::Orbital => 1.8,
        }
    }
    /// Average speed including dwell.
    fn speed(self) -> f32 {
        match self {
            Kind::Subway | Kind::Orbital => 35.0,
            Kind::Light => 25.0,
            Kind::Commuter => 55.0,
        }
    }
    fn headways(self) -> [f32; PERIODS] {
        match self {
            Kind::Subway => [4.0, 8.0, 4.0, 10.0, 20.0],
            Kind::Light => [7.0, 12.0, 7.0, 15.0, 30.0],
            Kind::Commuter => [12.0, 30.0, 12.0, 30.0, 60.0],
            Kind::Orbital => [6.0, 10.0, 6.0, 12.0, 20.0],
        }
    }
    /// (seats, crush load) per train. Subway: 10 cars x ~44 seats, ~160 at crush.
    fn capacity(self) -> (f32, f32) {
        match self {
            Kind::Subway => (440.0, 1600.0),
            Kind::Light => (150.0, 400.0),
            Kind::Commuter => (1100.0, 1800.0),
            Kind::Orbital => (300.0, 1100.0),
        }
    }
}

/// Waypoints in km from Times Square (the T-004 pack origin), from real places.
#[rustfmt::skip]
const LINES: &[(&str, Kind, &[(f64, f64)])] = &[
    ("Broadway-7 Av", Kind::Subway, &[(6.3, 13.6), (4.6, 8.5), (2.2, 3.0), (0.0, 0.0), (-2.0, -5.8), (-0.4, -7.3), (3.0, -10.0), (6.5, -11.5)]),
    ("Lexington", Kind::Subway, &[(12.0, 16.0), (8.0, 11.4), (4.0, 5.1), (0.7, -0.7), (-1.6, -5.5), (-0.4, -7.3), (4.0, -8.5), (8.0, -10.0)]),
    ("8 Av", Kind::Subway, &[(5.5, 11.0), (3.5, 6.0), (-0.3, -0.5), (-2.3, -4.5), (-0.6, -7.0), (5.0, -8.0), (11.0, -9.5), (19.4, -17.0)]),
    ("6 Av-Queens Blvd", Kind::Subway, &[(14.8, -6.4), (9.0, -2.5), (3.3, -1.2), (0.3, -0.3), (-1.2, -4.0), (-0.9, -8.0), (-0.5, -13.0), (0.3, -20.1)]),
    ("Broadway BMT", Kind::Subway, &[(5.5, 1.8), (3.3, -1.2), (0.1, 0.1), (-1.8, -5.0), (-1.2, -7.5), (-3.2, -13.8)]),
    ("Flushing", Kind::Subway, &[(13.1, 0.1), (6.5, -0.5), (3.3, -1.2), (0.6, -0.6), (-0.6, -0.2)]),
    ("Second Av-Bronx", Kind::Subway, &[(11.0, 16.0), (9.0, 9.0), (3.8, 4.2), (1.5, -1.5), (-1.0, -5.0)]),
    ("Canarsie", Kind::Subway, &[(-1.8, -2.6), (1.2, -2.4), (3.5, -6.0), (6.0, -8.5), (9.0, -11.0)]),
    ("Brooklyn-Queens", Kind::Subway, &[(-0.4, -7.3), (2.0, -4.5), (3.3, -1.2), (5.2, 2.4)]),
    ("PATH Newark", Kind::Subway, &[(-15.1, -2.7), (-10.5, -3.0), (-6.6, -2.8), (-3.8, -2.6), (-2.2, -5.6)]),
    ("Hudson light rail", Kind::Light, &[(-12.0, -11.0), (-8.5, -7.0), (-6.6, -3.5), (-4.5, 0.5), (-4.0, 6.0)]),
    ("Staten Island", Kind::Light, &[(-7.5, -12.8), (-13.0, -18.0), (-22.3, -27.4)]),
    ("Hudson", Kind::Commuter, &[(0.7, -0.7), (4.0, 5.1), (6.3, 13.6), (7.2, 19.8), (8.9, 48.0), (9.0, 70.0)]),
    ("Harlem", Kind::Commuter, &[(0.7, -0.7), (4.0, 5.1), (9.5, 14.0), (17.7, 30.6), (30.8, 70.0)]),
    ("New Haven", Kind::Commuter, &[(0.7, -0.7), (4.0, 5.1), (11.0, 13.0), (17.1, 17.0), (37.3, 32.1), (60.0, 42.0), (89.0, 60.0)]),
    ("LIRR Main", Kind::Commuter, &[(-0.7, -0.9), (3.3, -1.2), (8.0, -3.5), (14.8, -6.4), (29.0, -1.9), (38.5, 1.0), (74.2, 5.6)]),
    ("LIRR Babylon", Kind::Commuter, &[(-0.7, -0.9), (14.8, -6.4), (25.0, -9.0), (40.0, -8.5), (55.6, -6.4), (75.0, -2.0)]),
    ("LIRR Port Washington", Kind::Commuter, &[(-0.7, -0.9), (3.3, -1.2), (13.1, 0.1), (25.1, 7.9)]),
    ("NJ Northeast Corridor", Kind::Commuter, &[(-0.7, -0.9), (-7.6, 0.3), (-15.1, -2.7), (-19.4, -10.1), (-38.8, -29.0), (-64.8, -60.0)]),
    ("NJ Coast", Kind::Commuter, &[(-0.7, -0.9), (-7.6, 0.3), (-15.1, -2.7), (-19.4, -10.1), (-24.0, -26.0), (-10.0, -40.0), (-0.4, -51.0)]),
    ("NJ Morris", Kind::Commuter, &[(-0.7, -0.9), (-7.6, 0.3), (-15.1, -2.7), (-25.0, 1.0), (-41.7, 4.3), (-48.4, 13.6)]),
    ("NJ Bergen", Kind::Commuter, &[(-0.7, -0.9), (-7.6, 0.3), (-10.0, 10.0), (-15.6, 17.7), (-13.9, 39.0)]),
    ("Triboro", Kind::Orbital, &[(-3.2, -13.8), (2.0, -12.0), (6.0, -8.5), (10.0, -4.0), (9.5, 2.0), (8.0, 11.4), (6.3, 13.6)]),
    ("Outer ring", Kind::Orbital, &[(-19.4, -10.1), (-15.1, -2.7), (-12.0, 8.0), (-4.9, 14.2), (7.2, 19.8), (17.1, 17.0), (25.1, 7.9), (29.0, -1.9), (27.0, -12.0), (17.3, -12.5)]),
    ("Bronx-Queens", Kind::Orbital, &[(0.0, 13.0), (8.0, 11.4), (13.0, 6.0), (13.1, 0.1), (14.8, -6.4), (17.3, -12.5)]),
];

/// Stops closer than this to an existing station on another line become that station.
const MERGE_KM: f64 = 0.35;

pub fn synth_network() -> Network {
    let mut net = Network {
        st_x: vec![],
        st_y: vec![],
        line_name: vec![],
        line_off: vec![0],
        line_stops: vec![],
        hop_min: vec![],
        hop_back: vec![],
        line_speed_kmh: vec![],
        line_headway_min: vec![],
        line_seats: vec![],
        line_crush: vec![],
    };
    for &(name, kind, pts) in LINES {
        let mut stops: Vec<u32> = vec![];
        let place =|x: f64, y: f64, net: &mut Network, stops: &mut Vec<u32>| {
            let mut found = None;
            for s in 0..net.st_x.len() {
                let d = ((net.st_x[s] as f64 / 1000.0 - x).powi(2) + (net.st_y[s] as f64 / 1000.0 - y).powi(2)).sqrt();
                if d < MERGE_KM {
                    found = Some(s as u32);
                    break;
                }
            }
            let s = found.unwrap_or_else(|| {
                net.st_x.push((x * 1000.0) as f32);
                net.st_y.push((y * 1000.0) as f32);
                (net.st_x.len() - 1) as u32
            });
            if !stops.contains(&s) {
                stops.push(s);
            }
        };
        place(pts[0].0, pts[0].1, &mut net, &mut stops);
        for w in pts.windows(2) {
            let ((x0, y0), (x1, y1)) = (w[0], w[1]);
            let len = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
            let mid_r = (((x0 + x1) / 2.0).powi(2) + ((y0 + y1) / 2.0).powi(2)).sqrt();
            let k = (len / kind.spacing(mid_r)).round().max(1.0) as usize;
            for i in 1..k {
                let f = i as f64 / k as f64;
                let (x, y) = (x0 + f * (x1 - x0), y0 + f * (y1 - y0));
                // commuter lines run express through the inner area
                if matches!(kind, Kind::Commuter) && (x * x + y * y).sqrt() < 9.0 {
                    continue;
                }
                place(x, y, &mut net, &mut stops);
            }
            place(x1, y1, &mut net, &mut stops);
        }
        net.line_name.push(name.to_string());
        net.line_stops.extend_from_slice(&stops);
        net.hop_min.extend(std::iter::repeat(0.0).take(stops.len()));
        net.line_off.push(net.line_stops.len() as u32);
        net.line_speed_kmh.push(kind.speed());
        net.line_headway_min.push(kind.headways());
        let (seats, crush) = kind.capacity();
        net.line_seats.push(seats);
        net.line_crush.push(crush);
    }
    net.recompute_run_times();
    net
}

impl Network {
    pub fn n_stations(&self) -> usize {
        self.st_x.len()
    }
    pub fn n_lines(&self) -> usize {
        self.line_name.len()
    }
    pub fn stops(&self, l: usize) -> &[u32] {
        &self.line_stops[self.line_off[l] as usize..self.line_off[l + 1] as usize]
    }

    /// Run time per hop from straight-line distance x 1.15 (curves) at the line's average speed.
    pub fn recompute_run_times(&mut self) {
        for l in 0..self.n_lines() {
            let (a, b) = (self.line_off[l] as usize, self.line_off[l + 1] as usize);
            for k in a..b {
                self.hop_min[k] = if k + 1 < b {
                    let (s, t) = (self.line_stops[k] as usize, self.line_stops[k + 1] as usize);
                    let d = ((self.st_x[s] - self.st_x[t]).powi(2) + (self.st_y[s] - self.st_y[t]).powi(2)).sqrt();
                    d * 1.15 / 1000.0 / self.line_speed_kmh[l] * 60.0
                } else {
                    0.0
                };
            }
        }
    }

    /// The edit used for timing: move every station served only by line `l` by (dx, dy) metres.
    /// Returns how many stations moved.
    pub fn move_line(&mut self, l: usize, dx: f32, dy: f32) -> usize {
        let mut nlines = vec![0u32; self.n_stations()];
        for k in 0..self.n_lines() {
            for &s in self.stops(k) {
                nlines[s as usize] += 1;
            }
        }
        let mut moved = 0;
        let stops: Vec<u32> = self.stops(l).to_vec();
        for s in stops {
            if nlines[s as usize] == 1 {
                self.st_x[s as usize] += dx;
                self.st_y[s as usize] += dy;
                moved += 1;
            }
        }
        self.recompute_run_times();
        moved
    }
}
