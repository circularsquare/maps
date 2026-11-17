//! T-2496: the market's heap bytes by structure (`demand::mem`).

use super::api::Cols;
use super::*;
use crate::demand::mem::{self, vb, Report};

fn cols(c: &Cols) -> usize {
    match c {
        Cols::Short(v) => vb(v),
        Cols::Long(v) => vb(v),
    }
}
fn col(c: &Col) -> usize {
    match c {
        Col::F32(v) => vb(v),
        Col::Milli(v) => vb(v),
        Col::F64(v) => vb(v),
    }
}
fn ix(c: &Ix) -> usize {
    match c {
        Ix::U16(v) => vb(v),
        Ix::U32(v) => vb(v),
    }
}
fn nested<T>(v: &Vec<Vec<T>>) -> usize {
    vb(v) + v.iter().map(vb).sum::<usize>()
}
fn csr(p: &(Vec<u32>, Vec<u32>)) -> usize {
    vb(&p.0) + vb(&p.1)
}

impl Market {
    pub(super) fn mem_into(&self, r: &mut Report, at: &str) {
        r.add(format!("{at}places"), vb(&self.places) + self.places.iter().map(|p| p.zone.capacity() + vb(&p.air) + p.air.iter().map(|a| vb(&a.1)).sum::<usize>()).sum::<usize>());
        r.add(format!("{at}pair rules (rules.out)"), vb(&self.rules.out) + vb(&self.rules.slot) + vb(&self.rules.open));
        r.add(format!("{at}near (place pairs)"), self.near.capacity() * (size_of::<((u32, u32), f64)>() + 1));
        r.add(format!("{at}zones (Zone)"), vb(&self.zones));
        r.add(format!("{at}gid_place, comp_of_gid, cell_zone, site_key"), vb(&self.gid_place) + vb(&self.comp_of_gid) + vb(&self.cell_zone) + vb(&self.site_key.0));
        let mut gc = 0;
        for g in &self.groups {
            let mut sub = Report::default();
            mem::city(&g.city, &mut sub, "");
            gc += sub.total() + mem::zones(&g.kz) + vb(&g.kplace);
        }
        r.add(format!("{at}groups (each pack's sites: City, Zones)"), vb(&self.groups) + gc);
        r.add(format!("{at}commutes (+ by home, by work)"), vb(&self.commutes) + csr(&self.c_home) + csr(&self.c_work));
        r.add(format!("{at}everyday pairs (+ by home, by activity)"), vb(&self.everyday) + csr(&self.e_home) + csr(&self.e_act) + vb(&self.e_base));
        let cw = &self.cw;
        r.add(format!("{at}corridors: longer trips (has_trips, trips; shared with MarketPacks)"), vb(&cw.has_trips) + vb(&cw.trips));
        r.add(format!("{at}corridors: commutes, leaving_by"), vb(&cw.commute) + vb(&cw.leaving_by));
        r.add(format!("{at}drives (shared with MarketPacks)"), self.drives.heap_bytes());
    }
}

impl IntercityApi {
    pub(crate) fn mem_into(&self, r: &mut Report, at: &str) {
        self.m.mem_into(r, &format!("{at}market: "));
        r.add(format!("{at}served zone pairs (Pair)"), vb(&self.pairs));
        r.add(format!("{at}served zones, served_comp, want, win"), vb(&self.served) + vb(&self.served_at) + nested(&self.served_comp) + vb(&self.want) + vb(&self.win));
        r.add(format!("{at}access lists (access, _work, _c, _cw, _c_walk, _ew)"), nested(&self.access) + nested(&self.access_work) + nested(&self.access_c) + nested(&self.access_cw) + nested(&self.access_c_walk) + nested(&self.access_ew));
        r.add(format!("{at}commute pairs served (CPair)"), vb(&self.cpairs));
        r.add(format!("{at}everyday pairs served (EPairOn)"), vb(&self.epairs));
        if let Some(s) = &self.svc {
            r.add(format!("{at}service"), vb(&s.st_x) + vb(&s.st_y) + vb(&s.st_map) + vb(&s.line_off) + vb(&s.stops) + vb(&s.stop_xy) + vb(&s.times) + vb(&s.tph) + vb(&s.cars) + vb(&s.group) + vb(&s.span) + vb(&s.ups) + vb(&s.passages) + vb(&s.catches) + vb(&s.st_bus) + vb(&s.st_tram) + vb(&s.back));
        }
        let sp = &self.st_pat;
        r.add(format!("{at}station pattern (st_pat)"), vb(&sp.off) + cols(&sp.col) + vb(&sp.zoff) + cols(&sp.zst) + vb(&sp.zrow) + vb(&sp.zix));
        let (mut net, mut g, mut l, mut el, mut corr, mut pr, mut st, mut rest) = (0, 0, 0, 0, 0, 0, 0, 0);
        for p in self.per.iter().flatten() {
            net += mem::network(&p.net);
            g += mem::graph(&p.g);
            l += mem::loads(&p.avg) + mem::loads(&p.c_avg) + p.r_avg.as_ref().map_or(0, mem::loads);
            el += vb(&p.e_avg) + p.e_avg.iter().flatten().map(mem::loads).sum::<usize>();
            corr += vb(&p.corridors.k) + vb(&p.corridors.v) + vb(&p.c_corridors.k) + vb(&p.c_corridors.v);
            pr += vb(&p.pair_rail) + vb(&p.c_frac) + vb(&p.e_frac);
            st += vb(&p.st);
            rest += vb(&p.fac) + vb(&p.seen) + mem::xfer(&p.xi) + vb(&p.xfer);
        }
        r.add(format!("{at}periods: network copies"), net);
        r.add(format!("{at}periods: graphs"), g);
        r.add(format!("{at}periods: loads (avg, c_avg, r_avg)"), l);
        r.add(format!("{at}periods: everyday loads by purpose (e_avg)"), el);
        r.add(format!("{at}periods: corridors (Sparse)"), corr);
        r.add(format!("{at}periods: pair_rail, c_frac, e_frac"), pr);
        r.add(format!("{at}periods: station patterns (st)"), st);
        r.add(format!("{at}periods: crowding, transfers"), rest);
    }
    /// `mem_into` as text (`bin/demand_mem.rs`).
    pub fn mem_report(&self) -> String {
        let mut r = Report::default();
        self.mem_into(&mut r, "");
        r.text()
    }
}

impl MarketPacks {
    /// T-2496: what the held packs and the world's rows take (the corridors and drives are the
    /// composition's, counted there).
    pub fn mem_report(&self) -> String {
        let mut r = Report::default();
        let rows = self.world.rows.as_ref();
        r.add("world rows (Row)", rows.map_or(0, |v| vb(v) + v.iter().map(|w| w.zone.capacity() + w.country.capacity() + vb(&w.near) + vb(&w.commute) + vb(&w.air) + w.air.iter().map(|a| vb(&a.1)).sum::<usize>() + vb(&w.fg)).sum::<usize>()));
        r.add("world attractions", vb(&self.world.attractions));
        let mut z = 0;
        for p in self.zones.values() {
            z += p.zone.capacity() + p.country.capacity() + vb(&p.zones) + vb(&p.zone_ng) + vb(&p.zone_nt) + vb(&p.sites.x) + vb(&p.sites.y) + col(&p.sites.pop) + col(&p.sites.jobs) + ix(&p.sites.zone) + vb(&p.site_park) + col(&p.cell_pop) + vb(&p.zone_tw)
                + ix(&p.commute_home) + ix(&p.commute_place) + ix(&p.commute_work) + vb(&p.commute_trips) + ix(&p.everyday_home) + ix(&p.everyday_place) + ix(&p.everyday_zone) + vb(&p.everyday_kind) + vb(&p.everyday_trips) + vb(&p.everyday_walk) + vb(&p.airports) + nested(&p.air_direct);
        }
        r.add("market packs (ZoneIn)", z);
        r.text()
    }
}
