//! Local time per region (T-309, docs/world.md "Local time"). One game clock keeps world time,
//! counted in the registry's `clockUtcOffset` (UTC-5). Each line runs on the local time of the
//! region of its first stop: its service periods (`service.rs`, each `Lattice` carries its shift),
//! its boarding account's hours and its weekends (`boarding.rs`). A region's offset is a whole hour
//! of standard time from the registry, no daylight saving.
//!
//! The table comes from the clock worker (`TrackApi::set_time_zones`, app/src/game/localTime.ts
//! builds it); it is not saved. With no table every shift is 0: the game as it was before local
//! time, which every earlier test runs on.

/// A registered zone's centre and its whole-hour UTC offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Zone {
    pub lon: f64,
    pub lat: f64,
    pub utc_h: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimeZones {
    /// The game clock's UTC offset, hours.
    pub clock_utc_h: f64,
    /// A place is in the nearest zone whose centre is within this many metres (`cityWithin` in
    /// app/src/game/cities.ts); past it, its longitude's offset.
    pub reach_m: f64,
    pub zones: Vec<Zone>,
    /// Set from a table: without one every shift is 0.
    pub set: bool,
}

impl TimeZones {
    /// From the flat table: clock offset (h), reach (m), then per zone lon, lat, offset (h).
    pub fn from_table(v: &[f64]) -> TimeZones {
        if v.len() < 2 {
            return TimeZones::default();
        }
        let zones = v[2..].chunks_exact(3).map(|c| Zone { lon: c[0], lat: c[1], utc_h: c[2] }).collect();
        TimeZones { clock_utc_h: v[0], reach_m: v[1], zones, set: true }
    }

    /// Whole-hour UTC offset at (lon, lat): the nearest zone's within reach, else the longitude's
    /// nominal one. The same rule as `utcOffsetAt` (app/src/game/localTime.ts).
    pub fn utc_offset_at(&self, lon: f64, lat: f64) -> f64 {
        let cos = (lat * std::f64::consts::PI / 180.0).cos();
        let near = self
            .zones
            .iter()
            .min_by(|a, b| ((lon - a.lon) * cos).hypot(lat - a.lat).total_cmp(&((lon - b.lon) * cos).hypot(lat - b.lat)));
        if let Some(z) = near {
            let kx = 111_320.0 * cos;
            if ((lon - z.lon) * kx).hypot((lat - z.lat) * 111_320.0) <= self.reach_m {
                return z.utc_h;
            }
        }
        // JavaScript's Math.round: halves round up
        (lon / 15.0 + 0.5).floor()
    }

    /// Seconds local time at (lon, lat) runs ahead of the game clock; 0 without a table.
    pub fn shift_at(&self, lon: f64, lat: f64) -> f64 {
        if !self.set {
            return 0.0;
        }
        (self.utc_offset_at(lon, lat) - self.clock_utc_h) * 3600.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zones_then_longitude() {
        let t = TimeZones::from_table(&[-5.0, 120_000.0, -73.985, 40.758, -5.0, -79.3832, 43.6534, -5.0]);
        assert_eq!(t.shift_at(-74.0, 40.7), 0.0);
        assert_eq!(t.shift_at(-79.4, 43.7), 0.0);
        // London, in no registered zone: its longitude's offset, UTC+0, five hours ahead
        assert_eq!(t.shift_at(-0.1276, 51.5072), 5.0 * 3600.0);
        // Paris by longitude is UTC+0 too (it keeps UTC+1 once registered)
        assert_eq!(t.shift_at(2.35, 48.86), 5.0 * 3600.0);
        assert_eq!(TimeZones::default().shift_at(-0.1276, 51.5072), 0.0);
    }
}
