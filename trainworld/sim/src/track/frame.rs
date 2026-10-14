//! World frames (T-125): how the network's planar metres relate to longitude and latitude.
//!
//! Two frames exist and a network records which one it lives in.
//!
//! - **Equirect** is the original frame (SPEC 11): `x = R dlon cos(lat0)`, `y = R dlat` from an
//!   origin. Every save made before T-125 is in it and keeps it for life, so its geometry, costs
//!   and money are unchanged. Its east-west scale is `cos(lat0) / cos(lat)` of the truth: 1.2% off
//!   at Philadelphia, 2.7% at Boston or Washington, growing without bound.
//! - **Lcc** is the world frame of new games: a spherical Lambert conformal conic tangent at the
//!   origin's latitude. Conformal (circles stay circles, so arc radii and curve speeds are right)
//!   and its scale depends on latitude alone: `k = 1 + dlat^2 / 2` to second order, where `dlat` is
//!   the latitude distance from the origin in radians. That is 0.06% at 220 km north or south of
//!   the origin, 0.24% at 450 km and 0.4% at 555 km, at any longitude. Ground length is plane
//!   length divided by `k`.
//!
//! Near the origin the two frames agree to the first order, which is why Lcc keeps the origin and
//! the metre units of the legacy frame; they are different charts of the same ground.
//!
//! - **Merc** is the frame of new games since T-306 (docs/world.md): spherical Mercator over the
//!   whole Earth, `x = R lon`, `y = R atanh(sin lat)`, no origin. Conformal like Lcc, with scale
//!   `k = 1 / cos(lat) = cosh(y / R)` (1.32 at New York, 1.61 at London). Unlike the other two the
//!   track model does not treat its plane metres as ground metres: every piece of track carries
//!   its ground length (geom.rs), and metre figures are scaled by `k` where they enter the plane.
//!   So lengths, radii and costs are true anywhere between 85 S and 85 N.

use super::params::EARTH_R;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Equirect = 0,
    Lcc = 1,
    Merc = 2,
}

impl FrameKind {
    pub fn from_u8(v: u8) -> Option<FrameKind> {
        match v {
            0 => Some(FrameKind::Equirect),
            1 => Some(FrameKind::Lcc),
            2 => Some(FrameKind::Merc),
            _ => None,
        }
    }
}

/// Track is refused nearer the poles than this, degrees (docs/world.md).
pub const MERC_MAX_LAT: f64 = 85.0;

/// Mercator scale at plane `y`: plane metres per ground metre, `cosh(y / R)`.
#[inline]
pub fn merc_k(y: f64) -> f64 {
    (y / EARTH_R).cosh()
}

/// True if a Mercator plane point is inside the buildable world: within 85 degrees of the
/// equator and inside the antimeridian (the plane's east and west edges).
pub fn merc_inside(x: f64, y: f64) -> bool {
    let ymax = EARTH_R * MERC_MAX_LAT.to_radians().sin().atanh();
    x.abs() < PI * EARTH_R && y.abs() <= ymax
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub kind: FrameKind,
    pub lon0: f64,
    pub lat0: f64,
    // Lcc constants: cone constant, R F, and the radius of the origin parallel.
    n: f64,
    rf: f64,
    rho0: f64,
    cos_lat0: f64,
}

impl Frame {
    pub fn new(kind: FrameKind, lon0: f64, lat0: f64) -> Frame {
        if kind == FrameKind::Merc {
            // No origin: the plane is the whole Earth's.
            return Frame { kind, lon0: 0.0, lat0: 0.0, n: 0.0, rf: 0.0, rho0: 0.0, cos_lat0: 1.0 };
        }
        let phi0 = lat0.to_radians();
        let n = phi0.sin();
        let t0 = (PI / 4.0 + phi0 / 2.0).tan();
        let rf = EARTH_R * phi0.cos() * t0.powf(n) / n;
        Frame { kind, lon0, lat0, n, rf, rho0: EARTH_R * phi0.cos() / n, cos_lat0: phi0.cos() }
    }
    pub fn equirect(lon0: f64, lat0: f64) -> Frame {
        Frame::new(FrameKind::Equirect, lon0, lat0)
    }
    pub fn lcc(lon0: f64, lat0: f64) -> Frame {
        Frame::new(FrameKind::Lcc, lon0, lat0)
    }
    pub fn merc() -> Frame {
        Frame::new(FrameKind::Merc, 0.0, 0.0)
    }
    /// The same origin in another frame.
    pub fn with_kind(&self, kind: FrameKind) -> Frame {
        Frame::new(kind, self.lon0, self.lat0)
    }

    /// Plane metres to longitude and latitude, degrees.
    pub fn to_lonlat(&self, x: f64, y: f64) -> (f64, f64) {
        match self.kind {
            FrameKind::Equirect => {
                let lat0 = self.lat0.to_radians();
                (self.lon0 + (x / (EARTH_R * lat0.cos())).to_degrees(), self.lat0 + (y / EARTH_R).to_degrees())
            }
            FrameKind::Lcc => {
                let dy = self.rho0 - y;
                let rho = x.hypot(dy);
                let theta = x.atan2(dy);
                let lat = 2.0 * (self.rf / rho).powf(1.0 / self.n).atan() - PI / 2.0;
                (self.lon0 + (theta / self.n).to_degrees(), lat.to_degrees())
            }
            FrameKind::Merc => ((x / EARTH_R).to_degrees(), (y / EARTH_R).sinh().atan().to_degrees()),
        }
    }

    /// Longitude and latitude, degrees, to plane metres.
    pub fn from_lonlat(&self, lon: f64, lat: f64) -> (f64, f64) {
        match self.kind {
            FrameKind::Equirect => (
                EARTH_R * (lon - self.lon0).to_radians() * self.cos_lat0,
                EARTH_R * (lat - self.lat0).to_radians(),
            ),
            FrameKind::Lcc => {
                let rho = self.rf / (PI / 4.0 + lat.to_radians() / 2.0).tan().powf(self.n);
                let theta = self.n * wrap_lon(lon - self.lon0).to_radians();
                (rho * theta.sin(), self.rho0 - rho * theta.cos())
            }
            FrameKind::Merc => (EARTH_R * wrap_lon(lon).to_radians(), EARTH_R * lat.to_radians().sin().atanh()),
        }
    }

    /// How much longer a metre on the ground is drawn in the plane, east-west and north-south,
    /// at a latitude (degrees). Ground length = plane length / scale. Lcc is the same both ways.
    pub fn scale_at(&self, lat: f64) -> (f64, f64) {
        match self.kind {
            FrameKind::Equirect => (self.cos_lat0 / lat.to_radians().cos(), 1.0),
            FrameKind::Lcc => {
                let phi = lat.to_radians();
                let rho = self.rf / (PI / 4.0 + phi / 2.0).tan().powf(self.n);
                let k = self.n * rho / (EARTH_R * phi.cos());
                (k, k)
            }
            FrameKind::Merc => {
                let k = 1.0 / lat.to_radians().cos();
                (k, k)
            }
        }
    }

    /// True for the frame whose plane metres are not ground metres (Merc): its pieces of track
    /// carry ground lengths and its metre figures are scaled by `k` (geom.rs).
    pub fn scaled(&self) -> bool {
        self.kind == FrameKind::Merc
    }
}

fn wrap_lon(d: f64) -> f64 {
    let mut d = d % 360.0;
    if d > 180.0 {
        d -= 360.0;
    } else if d < -180.0 {
        d += 360.0;
    }
    d
}

/// Great-circle distance in metres on the sphere the frames use.
pub fn great_circle(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dl = (lon2 - lon1).to_radians();
    let a = ((p2 - p1) / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * EARTH_R * a.sqrt().asin()
}

#[cfg(test)]
mod tests {
    use super::*;

    const NYC: (f64, f64) = (-73.985, 40.758);

    #[test]
    fn origin_maps_to_zero_in_both_frames() {
        for f in [Frame::equirect(NYC.0, NYC.1), Frame::lcc(NYC.0, NYC.1)] {
            let (x, y) = f.from_lonlat(NYC.0, NYC.1);
            assert!(x.abs() < 1e-6 && y.abs() < 1e-6, "{f:?} {x} {y}");
            let (lon, lat) = f.to_lonlat(0.0, 0.0);
            assert!((lon - NYC.0).abs() < 1e-12 && (lat - NYC.1).abs() < 1e-12);
        }
    }

    #[test]
    fn round_trips_to_a_micrometre_across_the_extent() {
        for f in [Frame::equirect(NYC.0, NYC.1), Frame::lcc(NYC.0, NYC.1)] {
            for lat in [35.8, 38.9, 39.9526, 40.758, 42.36, 44.0, 45.8] {
                for lon in [-90.0, -80.0, -75.1636, -73.985, -71.06, -65.0] {
                    let (x, y) = f.from_lonlat(lon, lat);
                    let (lo, la) = f.to_lonlat(x, y);
                    let (x2, y2) = f.from_lonlat(lo, la);
                    assert!((x - x2).hypot(y - y2) < 1e-6, "{f:?} {lon} {lat}: {x} {y} -> {x2} {y2}");
                    assert!((lo - lon).abs() < 1e-9 && (la - lat).abs() < 1e-9, "{f:?} {lon} {lat}: {lo} {la}");
                }
            }
        }
    }

    #[test]
    fn equirect_is_the_legacy_formula() {
        // The formula every save before T-125 assumes (app/src/game/coords.ts, net.rs tile_of).
        let f = Frame::equirect(NYC.0, NYC.1);
        let (x, y) = f.from_lonlat(-75.1636, 39.9526);
        let rad = std::f64::consts::PI / 180.0;
        assert_eq!(x, 6371008.8 * (-75.1636 + 73.985) * rad * (40.758 * rad).cos());
        assert_eq!(y, 6371008.8 * (39.9526 - 40.758) * rad);
    }

    /// Values from PROJ 9 (pyproj 3.6.1, `+proj=lcc +lat_1=40.758 +lat_0=40.758 +lon_0=-73.985
    /// +R=6371008.8`), see notes/T-125.md for the script.
    #[test]
    fn lcc_matches_proj() {
        let f = Frame::lcc(NYC.0, NYC.1);
        for (lon, lat, x, y) in LCC_PROJ {
            let (px, py) = f.from_lonlat(*lon, *lat);
            assert!((px - x).abs() < 0.01 && (py - y).abs() < 0.01, "({lon},{lat}) {px} {py} vs {x} {y}");
        }
    }

    #[test]
    fn lcc_scale_is_one_plus_half_dlat_squared() {
        let f = Frame::lcc(NYC.0, NYC.1);
        assert!((f.scale_at(NYC.1).0 - 1.0).abs() < 1e-12);
        for dlat in [1.0f64, 2.0, 4.0, 5.0] {
            let k = f.scale_at(NYC.1 + dlat).0;
            let want = 1.0 + dlat.to_radians().powi(2) / 2.0;
            assert!((k - want).abs() < 1.2e-6 * dlat.powi(3), "dlat {dlat}: {k} vs {want}");
            let k = f.scale_at(NYC.1 - dlat).0;
            assert!((k - want).abs() < 1.2e-6 * dlat.powi(3), "dlat -{dlat}: {k} vs {want}");
        }
        // Measured on the ground: a 1 km step in the plane near Philadelphia, any direction.
        let (lon, lat) = (-75.1636, 39.9526);
        let (x, y) = f.from_lonlat(lon, lat);
        for th in [0.0f64, 0.7, 1.9, 3.0] {
            let (lo, la) = f.to_lonlat(x + 1000.0 * th.cos(), y + 1000.0 * th.sin());
            let ground = great_circle(lon, lat, lo, la);
            let k = f.scale_at(lat).0;
            assert!((1000.0 / ground - k).abs() < 2e-5, "{th}: {}", 1000.0 / ground);
        }
    }

    #[test]
    fn legacy_frame_is_off_by_the_cosine_ratio() {
        let f = Frame::equirect(NYC.0, NYC.1);
        let (kx, ky) = f.scale_at(39.9526);
        assert!((kx - 0.98813).abs() < 1e-4 && ky == 1.0, "{kx}");
    }

    /// (lon, lat, x, y) from PROJ.
    const LCC_PROJ: &[(f64, f64, f64, f64)] = &[
        (-73.985, 40.758, 0.0, 0.0),
        (-75.1636, 39.9526, -100470.111, -88884.804),
        (-74.7429, 40.2171, -64354.543, -59868.426),
        (-71.0589, 42.3601, 240472.637, 182178.277),
        (-77.0369, 38.9072, -264158.239, -201241.85),
        (-79.9959, 40.4406, -508301.948, -17879.352),
        (-87.6298, 41.8781, -1125352.859, 212217.906),
        (-73.5673, 45.5017, 32668.173, 528168.785),
        (-80.8431, 35.2271, -625136.579, -591508.916),
        (-65.0, 44.0, 718598.228, 397506.944),
    ];
}
