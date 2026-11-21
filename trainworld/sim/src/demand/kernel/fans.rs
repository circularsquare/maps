use super::*;

// ---------------------------------------------------------------------------------------------
// The ferry fans (T-2196, T-3892; docs/demand-mode.md): a share of trips count each minute aboard
// a ferry at `FAN_RIDE_W` of a minute. The outings' shares are `everyday::FAN_SHARES`.

pub const FAN_RIDE_W: f32 = 0.1;
/// T-3892: the commutes' share of fans, half the outings' (Anita, 2026-11-20: "maybe half as much
/// for work or medical or airport").
pub const COMMUTE_FAN_SHARE: f32 = 0.075;
pub const NO_FAN: u16 = u16::MAX;

/// T-2196: what the ferry fans' rail costs, composed through the ports from the period's skims: a
/// fan's best way from station a to b is the ordinary one or a ride on one ferry, p to q, with the
/// ordinary ways to p and from q (`sk.r`, a change at each), each minute aboard at `FAN_RIDE_W`.
/// One ferry ride a trip, and the cheaper of the two ways (no spread between them).
pub struct FanCosts {
    /// each port's station
    pub ports: Vec<u32>,
    /// S x S: the fan's cost where `via` names a port, else unused
    pub cost: Vec<f32>,
    /// S x S: the port left the ferry at, `NO_FAN` where the ordinary way is as cheap
    pub via: Vec<u16>,
    /// S x ports: for a trip from a leaving the ferry at port j, the port boarded at
    pub first: Vec<u16>,
    /// ports x ports: the ride's boarding and alighting route nodes
    pub ride: Vec<(u32, u32)>,
    /// stations with any `via` from them (`row`) or to them (`col`)
    pub row: Vec<bool>,
    pub col: Vec<bool>,
}

impl FanCosts {
    pub fn new(g: &Graph, p: &Params, sk: &Skims, ferry: &[bool]) -> Option<FanCosts> {
        let (s, nr) = (g.n_st, g.rn_line.len());
        let is_ferry = |r: usize| ferry.get(g.rn_line[r] as usize).copied().unwrap_or(false);
        let mut port_of = vec![NO_FAN; s];
        let mut ports: Vec<u32> = vec![];
        for r in (0..nr).filter(|&r| is_ferry(r)) {
            let st = g.rn_station[r] as usize;
            if port_of[st] == NO_FAN {
                port_of[st] = ports.len() as u16;
                ports.push(st as u32);
            }
        }
        let np = ports.len();
        if np < 2 || np >= NO_FAN as usize {
            return None;
        }
        // the fan's cost from boarding at port i to leaving at port j, one ferry line, its
        // boarding as the search has it (the wait and the change's fixed cost, `rn_left`)
        let mut fr = vec![f32::INFINITY; np * np];
        let mut ride = vec![(NONE, NONE); np * np];
        for r in (0..nr).filter(|&r| is_ferry(r) && g.rn_tph[r] > 0.0) {
            let st = g.rn_station[r] as usize;
            let Some(board) = (g.adj_off[st] as usize..g.adj_off[st + 1] as usize).find(|&e| g.adj_to[e] as usize == s + r) else {
                continue;
            };
            let i = port_of[st] as usize;
            let mut c = g.adj_cost[board] + g.rn_left[r];
            let mut x = r;
            for _ in 0..nr {
                let node = s + x;
                let Some(e) = (g.adj_off[node] as usize..g.adj_off[node + 1] as usize).find(|&e| g.adj_seg[e] as usize == node) else {
                    break;
                };
                c += FAN_RIDE_W * (g.adj_cost[e] - g.rn_fare[x]) + g.rn_fare[x];
                x = g.adj_to[e] as usize - s;
                let j = port_of[g.rn_station[x] as usize] as usize;
                if j != i && c < fr[i * np + j] {
                    fr[i * np + j] = c;
                    ride[i * np + j] = (r as u32, x as u32);
                }
                if x == r {
                    break;
                }
            }
        }
        // per station a and port j, the best way to leave the ferry at j: the ordinary way to a
        // port i (none from i itself, whose boarding is the first and so has no change cost)
        let chg = p.change_min();
        let mut to_j = vec![f32::INFINITY; s * np];
        let mut first = vec![NO_FAN; s * np];
        for a in 0..s {
            for (i, &pi) in ports.iter().enumerate() {
                let to = if a == pi as usize { -chg } else { sk.r[a * s + pi as usize] };
                if !to.is_finite() {
                    continue;
                }
                for j in 0..np {
                    let v = to + fr[i * np + j];
                    if v < to_j[a * np + j] {
                        to_j[a * np + j] = v;
                        first[a * np + j] = i as u16;
                    }
                }
            }
        }
        let mut cost = vec![0f32; s * s];
        let mut via = vec![NO_FAN; s * s];
        let (mut row, mut col) = (vec![false; s], vec![false; s]);
        let mut js: Vec<(usize, usize, f32)> = vec![];
        for a in 0..s {
            js.clear();
            js.extend((0..np).filter(|&j| to_j[a * np + j].is_finite()).map(|j| (j, ports[j] as usize, to_j[a * np + j])));
            if js.is_empty() {
                continue;
            }
            for b in (0..s).filter(|&b| b != a) {
                let mut best = sk.r[a * s + b];
                if !best.is_finite() {
                    continue;
                }
                let mut bj = NO_FAN;
                for &(j, q, v) in &js {
                    let v = if b == q { v } else { v + chg + sk.r[q * s + b] };
                    if v < best {
                        best = v;
                        bj = j as u16;
                    }
                }
                if bj != NO_FAN {
                    cost[a * s + b] = best;
                    via[a * s + b] = bj;
                    row[a] = true;
                    col[b] = true;
                }
            }
        }
        if !row.contains(&true) {
            return None;
        }
        Some(FanCosts { ports, cost, via, first, ride, row, col })
    }
    pub fn np(&self) -> usize {
        self.ports.len()
    }
    /// A fan's trips `x` from station a to b by the ferry (`via` set): the ordinary rides to the
    /// port boarded and from the port left go into `od`, the ride itself into `rides` (per port
    /// pair: riders, those whose trip starts at the port boarded, those whose trip ends at the
    /// port left).
    #[inline]
    pub fn route(&self, s: usize, a: usize, b: usize, x: f32, od: &mut [f32], rides: &mut [[f32; 3]]) {
        let np = self.np();
        let j = self.via[a * s + b] as usize;
        let fi = self.first[a * np + j] as usize;
        let (pp, qq) = (self.ports[fi] as usize, self.ports[j] as usize);
        if a != pp {
            od[a * s + pp] += x;
        }
        if b != qq {
            od[qq * s + b] += x;
        }
        let r = &mut rides[fi * np + j];
        r[0] += x;
        if a == pp {
            r[1] += x;
        }
        if b == qq {
            r[2] += x;
        }
    }
}

/// T-3892: the commute's fans (`mode_choice_fans`): the fans' costs, and per station the stations
/// a fan reaches cheaper by a ferry from it, with that cost.
pub struct CommuteFans {
    pub fc: FanCosts,
    pub share: f32,
    pub off: Vec<u32>,
    pub to: Vec<u32>,
    pub cost: Vec<f32>,
}

impl CommuteFans {
    pub fn new(g: &Graph, p: &Params, sk: &Skims, ferry: &[bool]) -> Option<CommuteFans> {
        let mut fc = FanCosts::new(g, p, sk, ferry)?;
        let s = g.n_st;
        // the trip home rides the legs back, towards the port left and the home station: a pair
        // whose way back reaches a station no strategy was searched to goes the ordinary way
        let np = fc.np();
        let searched = |x: usize| sk.slot[x] != NONE;
        let (mut row, mut col) = (vec![false; s], vec![false; s]);
        for a in (0..s).filter(|&a| fc.row[a]) {
            for b in 0..s {
                let j = fc.via[a * s + b];
                if j == NO_FAN {
                    continue;
                }
                let fi = fc.first[a * np + j as usize] as usize;
                let (pp, qq) = (fc.ports[fi] as usize, fc.ports[j as usize] as usize);
                let back = fc.ride[j as usize * np + fi].0 != NONE || searched(pp);
                if (b == qq || searched(qq)) && (a == pp || searched(a)) && back {
                    row[a] = true;
                    col[b] = true;
                } else {
                    fc.via[a * s + b] = NO_FAN;
                }
            }
        }
        (fc.row, fc.col) = (row, col);
        if !fc.row.contains(&true) {
            return None;
        }
        let (mut off, mut to, mut cost) = (Vec::with_capacity(s + 1), vec![], vec![]);
        off.push(0);
        for a in 0..s {
            if fc.row[a] {
                for t in 0..s {
                    if fc.via[a * s + t] != NO_FAN {
                        to.push(t as u32);
                        cost.push(fc.cost[a * s + t]);
                    }
                }
            }
            off.push(to.len() as u32);
        }
        Some(CommuteFans { fc, share: COMMUTE_FAN_SHARE, off, to, cost })
    }
    /// Station a's stations reached cheaper by a ferry, and the fan's cost to each.
    #[inline]
    pub fn of(&self, a: usize) -> (&[u32], &[f32]) {
        let r = self.off[a] as usize..self.off[a + 1] as usize;
        (&self.to[r.clone()], &self.cost[r])
    }
    /// The fans' home -> work trips by the ferry (`ModeOut::od_fan`) as their rides to and from
    /// the ports, into `od`; their ferry rides by port pair, home -> work.
    pub fn route_all(&self, s: usize, od_fan: &[f32], od: &mut [f32]) -> Vec<[f32; 3]> {
        let np = self.fc.np();
        let mut rides = vec![[0f32; 3]; np * np];
        for a in (0..s).filter(|&a| self.fc.row[a]) {
            for &t in self.of(a).0 {
                let x = od_fan[a * s + t as usize];
                if x > 0.0 {
                    self.fc.route(s, a, t as usize, x, od, &mut rides);
                }
            }
        }
        rides
    }
    /// The rides both ways, as `DemandApi::pass` mixes the trips home (`fw` of each station pair's
    /// home -> work trips and `fh` of the reverse pair's): a return ride is the ride back from the
    /// port left to the port boarded, or where the ferry has none, the ordinary way between the
    /// two ports (into `od`, already mixed).
    pub fn both_ways(&self, s: usize, rides: &[[f32; 3]], fw: f32, fh: f32, od: &mut [f32]) -> Vec<[f32; 3]> {
        let np = self.fc.np();
        let mut out = vec![[0f32; 3]; np * np];
        for i in 0..np {
            for j in 0..np {
                let [x, from, to] = rides[i * np + j];
                if x == 0.0 {
                    continue;
                }
                let o = &mut out[i * np + j];
                o[0] += fw * x;
                o[1] += fw * from;
                o[2] += fw * to;
                if self.fc.ride[j * np + i].0 != NONE {
                    let o = &mut out[j * np + i];
                    o[0] += fh * x;
                    o[1] += fh * to;
                    o[2] += fh * from;
                } else {
                    let (p, q) = (self.fc.ports[i] as usize, self.fc.ports[j] as usize);
                    od[q * s + p] += fh * x;
                }
            }
        }
        out
    }
    /// The rides (`both_ways`) on the ferry's route nodes, the ports' entries and exits set right
    /// (the rides to and from the ports counted a trip each), as the outings' `add_fan_rides`.
    pub fn add_rides(&self, g: &Graph, rides: &[[f32; 3]], l: &mut Loads) {
        let s = g.n_st;
        let np = self.fc.np();
        for (ij, &[x, from, to]) in rides.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let (p, q) = (self.fc.ports[ij / np] as usize, self.fc.ports[ij % np] as usize);
            let (rb, ra) = (self.fc.ride[ij].0 as usize, self.fc.ride[ij].1 as usize);
            l.board[p] += x;
            l.rn_board[rb] += x;
            let mut r = rb;
            while r != ra {
                l.seg[r] += x;
                l.pax_min += (x * g.rn_ride[r]) as f64;
                r = g.rn_next[r] as usize - s;
            }
            l.alight[q] += x;
            l.rn_alight[ra] += x;
            l.entry[p] += from;
            l.exit[p] = (l.exit[p] - (x - from)).max(0.0);
            l.exit[q] += to;
            l.entry[q] = (l.entry[q] - (x - to)).max(0.0);
        }
    }
}
