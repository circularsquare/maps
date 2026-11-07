//! Station buildings (T-2217, `track::building`, notes/T-2208.md 3): added, reshaped, renamed and
//! removed outside the undo history and paid at once, as passages are. Corners are plane x, y
//! pairs, flat; refusals are `Building*` issues (`issue_kinds`), or `Money`.

use super::*;
use crate::track::building::{self, Building};
use crate::track::upgrades;

fn corners_of(flat: &[f64]) -> Vec<[f64; 2]> {
    flat.chunks_exact(2).map(|p| [quant(p[0]), quant(p[1])]).collect()
}

#[wasm_bindgen]
impl TrackApi {
    fn building(&self, id: u32) -> Option<&Building> {
        self.w.net.buildings.get(id as usize).and_then(|b| b.as_ref())
    }

    /// Pay `price` US$M at once: false, with the money issue, when cash is short.
    fn pay_building(&mut self, price: f64) -> bool {
        let cash = self.spendable();
        if price > cash + 1e-6 {
            self.issues.clear();
            self.money_short = Some((price, cash));
            return false;
        }
        self.cash -= price;
        self.money_short = None;
        self.issues.clear();
        true
    }

    /// The name a building gets: `name` trimmed, or with none its first constructed member's.
    fn building_name_for(&self, name: &str, corners: &[[f64; 2]]) -> String {
        let name = name.trim();
        if name.is_empty() { building::default_name(&self.w.net, &building::members(&self.w.net, corners)) } else { name.to_string() }
    }

    /// Add a station building round `corners` (x, y pairs), named `name` (none: its first
    /// constructed member's, the part before any slash). Paid at once, US$2,000 a m²; not undoable.
    /// 0 done (`last_building`, `building_charge`); 1 refused.
    pub fn add_building(&mut self, corners: &[f64], name: String) -> u32 {
        self.building_last = (u32::MAX, 0.0);
        let c = corners_of(corners);
        if let Some(i) = building::refusal(&self.w.net, u32::MAX, &c) {
            return self.refuse(i);
        }
        let price = upgrades::building_price(building::area_m2(&self.w.net, &c));
        if !self.pay_building(price) {
            return 1;
        }
        let name = self.building_name_for(&name, &c);
        let id = self.w.net.buildings.len() as u32;
        self.w.net.buildings.push(Some(Building { name, corners: c }));
        self.building_last = (id, price);
        0
    }

    /// Give building `id` a new outline. Pays for the floor it did not cover before and refunds
    /// nothing for floor given up. 0 done (`building_charge`); 1 refused.
    pub fn reshape_building(&mut self, id: u32, corners: &[f64]) -> u32 {
        self.building_last = (u32::MAX, 0.0);
        let Some(old) = self.building(id).map(|b| b.corners.clone()) else { return self.refuse(Issue::NoSuchBuilding { building: id }) };
        let c = corners_of(corners);
        if let Some(i) = building::refusal(&self.w.net, id, &c) {
            return self.refuse(i);
        }
        let price = upgrades::building_price(building::added_m2(&self.w.net, &old, &c));
        if !self.pay_building(price) {
            return 1;
        }
        self.w.net.buildings[id as usize].as_mut().unwrap().corners = c;
        self.building_last = (u32::MAX, price);
        0
    }

    /// Rename building `id` (an empty name: its first constructed member's). 0 done; 1 no such building.
    pub fn rename_building(&mut self, id: u32, name: String) -> u32 {
        let Some(c) = self.building(id).map(|b| b.corners.clone()) else { return self.refuse(Issue::NoSuchBuilding { building: id }) };
        let name = self.building_name_for(&name, &c);
        self.w.net.buildings[id as usize].as_mut().unwrap().name = name;
        self.issues.clear();
        self.money_short = None;
        0
    }

    /// Remove building `id`; nothing is refunded. 0 done; 1 no such building.
    pub fn remove_building(&mut self, id: u32) -> u32 {
        if self.building(id).is_none() {
            return self.refuse(Issue::NoSuchBuilding { building: id });
        }
        self.w.net.buildings[id as usize] = None;
        self.issues.clear();
        self.money_short = None;
        0
    }

    /// What adding a building round `corners` (`id` `u32::MAX`), or reshaping building `id` to them,
    /// would be: [floor m², price US$M (for a reshape the floor added), stations whose platform
    /// touches it, constructed ones of them]. Sets the issues to the refusal it would meet, if any
    /// (not money: compare the price with cash). Changes nothing.
    pub fn building_quote(&mut self, id: u32, corners: &[f64]) -> Vec<f64> {
        let c = corners_of(corners);
        let net = &self.w.net;
        let area = building::area_m2(net, &c);
        let m2 = match self.building(id) {
            Some(b) => building::added_m2(net, &b.corners, &c),
            None => area,
        };
        let members = building::members(net, &c);
        let built = members.iter().filter(|&&m| net.node_built[m as usize]).count();
        let out = vec![area, upgrades::building_price(m2), members.len() as f64, built as f64];
        let refusal = if id != u32::MAX && self.building(id).is_none() { Some(Issue::NoSuchBuilding { building: id }) } else { building::refusal(net, id, &c) };
        self.money_short = None;
        self.issues = refusal.into_iter().collect();
        out
    }

    /// The buildings standing: per building its id, corner count `n`, then `n` x, y pairs.
    pub fn buildings_info(&self) -> Vec<f64> {
        let mut out = vec![];
        for (id, b) in self.w.net.buildings.iter().enumerate() {
            if let Some(b) = b {
                out.extend([id as f64, b.corners.len() as f64]);
                out.extend(b.corners.iter().flatten());
            }
        }
        out
    }
    /// Building `id`'s name ("" for none).
    pub fn building_name(&self, id: u32) -> String {
        self.building(id).map(|b| b.name.clone()).unwrap_or_default()
    }
    /// Building `id`'s floor, m² (0 for none).
    pub fn building_area(&self, id: u32) -> f64 {
        self.building(id).map_or(0.0, |b| building::area_m2(&self.w.net, &b.corners))
    }
    /// The stations whose platform touches building `id`, any level, constructed or not, by node id.
    pub fn building_members(&self, id: u32) -> Vec<u32> {
        self.building(id).map(|b| building::members(&self.w.net, &b.corners)).unwrap_or_default()
    }
    /// The building the last `add_building` made, `u32::MAX` if it was refused.
    pub fn last_building(&self) -> u32 {
        self.building_last.0
    }
    /// What the last `add_building` or `reshape_building` paid, US$M.
    pub fn building_charge(&self) -> f64 {
        self.building_last.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Double track along y = 0 from -2000 to 2000 (plane m, the legacy frame: ground m), with
    /// constructed 200 m stations at x = 0 ("Central / Main") and x = 1000, a blueprint one at -1000,
    /// and track along y = 60 with a constructed station at x = 150 (a level below).
    fn net() -> (TrackApi, [u32; 4]) {
        let mut a = TrackApi::new(-73.985, 40.758, true);
        a.set_cash(1e5);
        a.set_route_tangent(true);
        assert_eq!(a.add_route(&[0.0, 0.0, -2000.0, 0.0, 0.0, 0.0, 0.0, 2000.0, 0.0, 0.0], &[], 2, false), 0, "{}", a.issues());
        assert_eq!(a.add_route(&[0.0, 0.0, -2000.0, 60.0, -1.0, 0.0, 0.0, 2000.0, 60.0, -1.0], &[], 2, false), 0, "{}", a.issues());
        let on = |a: &TrackApi, x: f64, y: f64| {
            let n = &a.w.net;
            (0..n.edge_count() as u32)
                .find(|&e| {
                    let (p, q) = (n.edge_a[e as usize] as usize, n.edge_b[e as usize] as usize);
                    n.edge_ok(e) && (n.node_y[p] - y).abs() < 1.0 && n.node_x[p].min(n.node_x[q]) < x && n.node_x[p].max(n.node_x[q]) > x
                })
                .unwrap()
        };
        for (x, y, name) in [(0.0, 0.0, "Central / Main"), (1000.0, 0.0, "East"), (-1000.0, 0.0, "West"), (150.0, 60.0, "Low")] {
            let e = on(&a, x, y);
            assert_eq!(a.add_station(2, e, x, y, 200, name.into()), 0, "{}", a.issues());
        }
        let at = |a: &TrackApi, x: f64, y: f64| {
            let n = &a.w.net;
            (0..n.node_count() as u32).find(|&m| n.is_station(m) && (n.node_x[m as usize] - x).abs() < 1.0 && (n.node_y[m as usize] - y).abs() < 1.0).unwrap()
        };
        let st = [at(&a, 0.0, 0.0), at(&a, 1000.0, 0.0), at(&a, -1000.0, 0.0), at(&a, 150.0, 60.0)];
        assert_eq!(a.construct(&[], &[st[0], st[1], st[3]]), 0, "{}", a.issues());
        (a, st)
    }

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<f64> {
        vec![x0, y0, x1, y0, x1, y1, x0, y1]
    }

    #[test]
    fn a_building_is_paid_by_its_floor_and_takes_in_every_platform_it_touches() {
        let (mut a, st) = net();
        // 200 x 50 round Central: 10,000 m², US$20M; the station at y = 60 is 10 m off its edge
        let c = rect(-100.0, -25.0, 100.0, 25.0);
        let q = a.building_quote(u32::MAX, &c);
        assert_eq!(q, vec![10_000.0, 20.0, 1.0, 1.0], "{}", a.issue_kinds());
        let cash = a.cash();
        assert_eq!(a.add_building(&c, "".into()), 0, "{}", a.issue_kinds());
        assert!((cash - a.cash() - 20.0).abs() < 1e-9);
        assert_eq!((a.last_building(), a.building_charge()), (0, 20.0));
        assert_eq!(a.building_name(0), "Central", "the part before the slash");
        assert_eq!(a.building_members(0), vec![st[0]]);
        assert_eq!(a.buildings_info(), vec![0.0, 4.0, -100.0, -25.0, 100.0, -25.0, 100.0, 25.0, -100.0, 25.0]);
        // reaching to y = 53 takes in the low station: its platform is within 8 m (any level)
        let cash = a.cash();
        assert_eq!(a.reshape_building(0, &rect(-100.0, -25.0, 100.0, 53.0)), 0, "{}", a.issue_kinds());
        assert!((cash - a.cash() - 0.002 * 200.0 * 28.0).abs() < 1e-9, "only the floor added");
        assert_eq!(a.building_members(0), vec![st[0], st[3]]);
        // moved sideways: pays for the new floor only; made smaller: nothing back
        let cash = a.cash();
        assert_eq!(a.reshape_building(0, &rect(-50.0, -25.0, 150.0, 53.0)), 0, "{}", a.issue_kinds());
        assert!((cash - a.cash() - 0.002 * 50.0 * 78.0).abs() < 1e-6, "{}", cash - a.cash());
        let cash = a.cash();
        assert_eq!(a.reshape_building(0, &rect(-50.0, -25.0, 100.0, 53.0)), 0);
        assert_eq!(a.cash(), cash);
        assert_eq!(a.building_charge(), 0.0);
        assert_eq!(a.rename_building(0, "  Grand hall ".into()), 0);
        assert_eq!(a.building_name(0), "Grand hall");
        // the built value counts it
        let (total, built) = (a.w.net.total_cost(), a.w.net.built_cost());
        let floor = a.building_area(0);
        assert!((floor - 150.0 * 78.0).abs() < 1e-6);
        assert_eq!(a.remove_building(0), 0);
        assert!((a.w.net.total_cost() - (total - 0.002 * floor)).abs() < 1e-6 && (a.w.net.built_cost() - (built - 0.002 * floor)).abs() < 1e-6);
        assert!((cash - a.cash()).abs() < 1e-12, "nothing refunded");
        assert!(a.buildings_info().is_empty());
        assert_eq!(a.remove_building(0), 1);
        assert_eq!(a.issue_kinds(), "NoSuchBuilding");
    }

    #[test]
    fn each_refusal() {
        let (mut a, _) = net();
        let refused = |a: &mut TrackApi, c: &[f64]| {
            let cash = a.cash();
            if a.add_building(c, "x".into()) == 0 {
                return String::new();
            }
            assert_eq!(a.cash(), cash);
            assert_eq!(a.last_building(), u32::MAX);
            a.issue_kinds()
        };
        assert_eq!(refused(&mut a, &[-100.0, -25.0, 100.0, -25.0]), "BuildingCorners", "two corners");
        let many: Vec<f64> = (0..25).flat_map(|i| { let t = i as f64 / 25.0 * std::f64::consts::TAU; [100.0 * t.cos(), 100.0 * t.sin()] }).collect();
        assert_eq!(refused(&mut a, &many), "BuildingCorners", "25 corners");
        assert_eq!(refused(&mut a, &many[..48]), "", "24 are fine");
        assert_eq!(a.remove_building(a.last_building()), 0);
        assert_eq!(refused(&mut a, &[-100.0, -25.0, 100.0, -25.0, 100.0, -25.0005, -100.0, 25.0]), "BuildingCorners", "a corner placed twice");
        assert_eq!(refused(&mut a, &[-100.0, -25.0, 100.0, 25.0, 100.0, -25.0, -100.0, 25.0]), "BuildingEdgesCross");
        assert_eq!(refused(&mut a, &rect(-100.0, -25.0, 950.0, 25.0)), "BuildingTooWide", "1,051 m corner to corner");
        assert_eq!(refused(&mut a, &rect(-20.0, -20.0, 20.0, 4.0)), "BuildingTooSmall", "960 m²");
        assert_eq!(refused(&mut a, &rect(-1100.0, -25.0, -900.0, 25.0)), "BuildingNoStation", "a blueprint station only");
        assert_eq!(refused(&mut a, &rect(400.0, -25.0, 600.0, 25.0)), "BuildingNoStation", "no station at all");
        assert_eq!(refused(&mut a, &rect(-100.0, -25.0, 100.0, 25.0)), "");
        let hall = a.last_building();
        assert_eq!(refused(&mut a, &rect(50.0, -25.0, 250.0, 25.0)), "BuildingOverlap", "over the first");
        assert_eq!(refused(&mut a, &rect(100.0, -25.0, 250.0, 25.0)), "", "a shared wall is fine");
        a.set_cash(1.0);
        assert!(refused(&mut a, &rect(900.0, -25.0, 1100.0, 25.0)).starts_with("Money:20.0:1.0"));
        // a reshape meets the same rules, its own floor apart
        a.set_cash(1e5);
        assert_eq!(a.reshape_building(hall, &rect(-100.0, -25.0, 120.0, 25.0)), 1);
        assert_eq!(a.issue_kinds(), "BuildingOverlap");
        assert_eq!(a.reshape_building(hall, &rect(-110.0, -30.0, 100.0, 25.0)), 0, "{}", a.issue_kinds());
        assert_eq!(a.reshape_building(7, &rect(-110.0, -30.0, 100.0, 25.0)), 1);
        assert_eq!(a.issue_kinds(), "NoSuchBuilding");
        // the quote names the refusal without changing anything
        let q = a.building_quote(u32::MAX, &rect(-20.0, -20.0, 20.0, 4.0));
        assert_eq!((q[0], a.issue_kinds().as_str()), (960.0, "BuildingTooSmall"));
    }

    /// A station that goes leaves its building standing, empty; one drawn inside joins it; the
    /// buildings survive a save and load.
    #[test]
    fn buildings_outlive_their_stations_and_ride_in_the_save() {
        let (mut a, st) = net();
        assert_eq!(a.add_building(&rect(-100.0, -25.0, 100.0, 25.0), "Hall".into()), 0, "{}", a.issue_kinds());
        let bytes = a.save();
        assert_eq!(a.delete_pieces(&[], &[st[0]]), 0, "{}", a.issues());
        assert!(a.building_members(0).is_empty());
        assert_eq!(a.building_name(0), "Hall");
        let mut b = TrackApi::new(-73.985, 40.758, true);
        assert!(b.load(&bytes));
        assert_eq!(b.buildings_info(), vec![0.0, 4.0, -100.0, -25.0, 100.0, -25.0, 100.0, 25.0, -100.0, 25.0]);
        assert_eq!(b.building_name(0), "Hall");
        assert_eq!(b.building_members(0).len(), 1);
        assert_eq!(b.save(), bytes);
    }

    /// T-2683: a name the player typed is flagged; the flag survives a platform change, undo, a
    /// save and load, and the load's tidying of capitals; a name placed by the game is not flagged.
    #[test]
    fn a_typed_station_name_is_flagged_and_kept() {
        let (mut a, st) = net();
        assert_eq!(a.named_stations(), Vec::<u32>::new(), "placed names are the game's");
        assert_eq!(a.set_station_props(st[1], 200, "MY HOME STATION".into()), 0, "{}", a.issues());
        assert_eq!(a.named_stations(), vec![st[1]]);
        assert_eq!(a.set_station_props(st[1], 200, "MY HOME STATION".into()), 0);
        assert_eq!(a.set_station_props(st[0], 240, "Central / Main".into()), 0, "{}", a.issues());
        assert_eq!(a.named_stations(), vec![st[1]], "the same name with a new platform is no typing");
        let bytes = a.save();
        let mut b = TrackApi::new(-73.985, 40.758, true);
        assert!(b.load(&bytes));
        assert_eq!(b.named_stations(), vec![st[1]]);
        assert_eq!(b.node_name(st[1]), "MY HOME STATION", "load does not tidy it");
        assert_eq!(b.save(), bytes);
    }

    /// After T-2193: a save with ferries and buildings writes LFR2 then BLD1 and loads both; the
    /// ferry save from before the building reads back without buildings, to the bit.
    #[test]
    fn buildings_follow_the_ferries_in_the_save() {
        use crate::track::ferry::K_PORT;
        let (mut a, _) = net();
        let port = |a: &mut TrackApi, x: f64, name: &str| {
            assert_eq!(a.add_port(x, -3100.0, x, -3000.0, name.into()), 0, "{}", a.issues());
            a.last_port()
        };
        let (p, q) = (port(&mut a, 0.0, "A"), port(&mut a, 2000.0, "B"));
        let l = a.new_line_id();
        let route = [0.0, -3000.0, K_PORT as f64, 3.0, 2000.0, -3000.0, K_PORT as f64, 3.0];
        assert_eq!(a.set_ferry_line(l, &[p, q], &route, 900, 1200, 1800, 60.0, 300.0, 0x1f77b4, "Ferry 1".into()), 0, "{}", a.issues());
        let ferries = a.save();
        assert!(ferries.windows(4).any(|w| w == b"LFR2") && !ferries.windows(4).any(|w| w == b"BLD1"));
        assert_eq!(a.add_building(&rect(-100.0, -25.0, 100.0, 25.0), "Hall".into()), 0, "{}", a.issue_kinds());
        let both = a.save();
        assert_eq!(&both[..ferries.len()], &ferries[..], "BLD1 only follows what was there");
        assert_eq!(&both[ferries.len()..ferries.len() + 4], b"BLD1");
        assert_eq!(TrackApi::save_unread(&both), 0);
        let mut b = TrackApi::new(-73.985, 40.758, true);
        assert!(b.load(&both));
        assert!(b.is_ferry(l) && b.is_port(p) && b.is_port(q));
        assert_eq!(b.building_name(0), "Hall");
        assert_eq!(b.save(), both);
        let mut c = TrackApi::new(-73.985, 40.758, true);
        assert!(c.load(&ferries));
        assert!(c.is_ferry(l) && c.buildings_info().is_empty());
        assert_eq!(c.save(), ferries);
    }

    /// In the Mercator frame floor and reach are ground metres.
    #[test]
    fn floor_is_ground_metres_in_the_world_frame() {
        let mut a = TrackApi::new(-73.985, 40.758, true);
        assert!(a.set_empty_frame(2));
        let (x, y) = (-8_236_000.0, 4_975_000.0);
        let k = crate::track::frame::merc_k(y);
        let c: Vec<[f64; 2]> = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]].iter().map(|p| [quant(x + p[0] * k), quant(y + p[1] * k)]).collect();
        assert!((building::area_m2(&a.w.net, &c) - 5000.0).abs() < 0.5, "{}", building::area_m2(&a.w.net, &c));
        a.set_cash(1e5);
        let flat: Vec<f64> = c.iter().flatten().copied().collect();
        assert_eq!(a.add_building(&flat, "x".into()), 1);
        assert_eq!(a.issue_kinds(), "BuildingNoStation");
    }
}
