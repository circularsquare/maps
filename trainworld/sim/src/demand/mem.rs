//! T-2496: what the demand populations hold on the heap, structure by structure (`mem_report` on
//! `DemandApi`, `EverydayApi` and `IntercityApi`; `bin/demand_mem.rs`). Counted from each vector's
//! capacity, so a figure is what the allocator holds for it; a structure shared through an `Arc` or
//! `Rc` is counted where the report says.

use super::kernel::{Graph, Gravity, Loads, Subzones, XferIndex, Zones};
use super::network::Network;
use super::pack::City;
use std::mem::size_of;

/// Bytes a vector's buffer takes (its capacity, not its length).
pub(crate) fn vb<T>(v: &Vec<T>) -> usize {
    v.capacity() * size_of::<T>()
}

/// Named byte counts, printed largest first.
#[derive(Default)]
pub(crate) struct Report(Vec<(String, usize)>);

impl Report {
    pub(crate) fn add(&mut self, name: impl Into<String>, bytes: usize) {
        self.0.push((name.into(), bytes));
    }
    pub(crate) fn total(&self) -> usize {
        self.0.iter().map(|r| r.1).sum()
    }
    /// One line per entry over 10 KB, "MB name", largest first, and the total.
    pub(crate) fn text(&self) -> String {
        let mut rows: Vec<&(String, usize)> = self.0.iter().filter(|r| r.1 >= 10_000).collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        let mb = |b: usize| b as f64 / 1048576.0;
        let mut s: String = rows.iter().map(|(n, b)| format!("{:>9.2} {n}\n", mb(*b))).collect();
        s.push_str(&format!("{:>9.2} total\n", mb(self.total())));
        s
    }
}

pub(crate) fn network(n: &Network) -> usize {
    vb(&n.st_x) + vb(&n.st_y) + vb(&n.line_name) + n.line_name.iter().map(|s| s.capacity()).sum::<usize>() + vb(&n.line_off) + vb(&n.line_stops) + vb(&n.hop_min) + vb(&n.hop_back)
        + vb(&n.line_speed_kmh) + vb(&n.line_headway_min) + vb(&n.line_seats) + vb(&n.line_rated) + vb(&n.line_crush) + vb(&n.st_group) + vb(&n.stop_x) + vb(&n.stop_y)
        + vb(&n.line_wrap) + vb(&n.span) + vb(&n.st_ups) + vb(&n.st_passage) + vb(&n.catches) + n.catches.iter().map(|c| vb(&c.pts)).sum::<usize>() + vb(&n.st_bus) + vb(&n.st_tram) + vb(&n.back_stops)
}

pub(crate) fn graph(g: &Graph) -> usize {
    vb(&g.rn_station) + vb(&g.rn_line) + vb(&g.rn_ride) + vb(&g.rn_fare) + vb(&g.rn_tph) + vb(&g.rn_left) + vb(&g.rn_span) + vb(&g.rn_bus) + vb(&g.rn_dx) + vb(&g.rn_dy)
        + vb(&g.st_spread) + vb(&g.st_wait) + vb(&g.st_bld) + vb(&g.rev_off) + vb(&g.rev_from) + vb(&g.rev_edge) + vb(&g.adj_off) + vb(&g.adj_to) + vb(&g.adj_cost) + vb(&g.adj_seg)
        + vb(&g.rn_next) + vb(&g.fare_curve.km) + vb(&g.fare_curve.min) + vb(&g.rn_km) + vb(&g.rn_fare_mul) + vb(&g.rn_kmw)
}

pub(crate) fn loads(l: &Loads) -> usize {
    vb(&l.seg) + vb(&l.board) + vb(&l.alight) + vb(&l.rn_board) + vb(&l.rn_alight) + vb(&l.entry) + vb(&l.exit) + vb(&l.rn_fare_x)
}

pub(crate) fn xfer(x: &XferIndex) -> usize {
    vb(&x.at)
}

pub(crate) fn city(c: &City, r: &mut Report, at: &str) {
    r.add(format!("{at}city cells (h3, x, y, pop, jobs, region)"), vb(&c.h3) + vb(&c.x) + vb(&c.y) + vb(&c.pop) + vb(&c.jobs) + vb(&c.region));
    if let Some(g) = &c.gravity {
        r.add(format!("{at}city pack gravity"), vb(&g.cell_zone) + vb(&g.gx) + vb(&g.gy) + vb(&g.zone_region) + vb(&g.row) + vb(&g.col) + vb(&g.home) + vb(&g.work) + vb(&g.cells9) + vb(&g.size));
    }
}

pub(crate) fn zones(z: &Zones) -> usize {
    vb(&z.cell_zone) + vb(&z.gx) + vb(&z.gy) + vb(&z.region) + vb(&z.pop) + vb(&z.jobs) + vb(&z.ncell) + vb(&z.size)
}

pub(crate) fn gravity(g: &Gravity, r: &mut Report, at: &str) {
    r.add(format!("{at}gravity row, col, decay table"), vb(&g.row) + vb(&g.col) + vb(&g.ftab));
    r.add(format!("{at}gravity fine walks (wf_*)"), vb(&g.wf_off) + vb(&g.wf_j) + vb(&g.wf_c) + vb(&g.wf_s));
}

pub(crate) fn subzones(s: &Subzones, r: &mut Report, at: &str) {
    r.add(format!("{at}subzones per subzone"), vb(&s.zone) + vb(&s.pop) + vb(&s.jobs) + vb(&s.cx) + vb(&s.cy) + vb(&s.home_park) + vb(&s.part));
    r.add(format!("{at}subzones access lists"), vb(&s.acc_off) + vb(&s.acc_mid) + vb(&s.acc_st) + vb(&s.acc_w) + vb(&s.acc_d) + vb(&s.acc_t));
    r.add(format!("{at}subzones homes, origins, destinations"), vb(&s.home_off) + vb(&s.home_zones) + vb(&s.zo_off) + vb(&s.zo) + vb(&s.zd_off) + vb(&s.zd));
    r.add(format!("{at}subzones cells (cell, mem, walk_c)"), vb(&s.cell) + vb(&s.mem_off) + vb(&s.mem) + vb(&s.walk_c) + vb(&s.st_part));
}
