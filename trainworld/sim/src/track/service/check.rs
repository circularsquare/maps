//! T-4133: `Service::note_changes`' check mode, natively. `TW_TRAINS_CHECK=1` runs the full walk
//! (every train running at the edit moved again, as before T-4133) on a copy beside every bounded one and panics
//! where any line's eras differ, or its shifts differ in what trains they hold or by more than `TOL_S`;
//! `TW_TRAINS_CHECK=skip` also leaves the first moved line with trains running out of the bounded walk, so the
//! check must fail. `TW_TRAINS=full` walks every train (as before T-4133), to time the two in one build. `stats`
//! counts and times them.
//!
//! The full walk works a train out again from where it stood even when nothing about it changed, which can move
//! its shift by a rounding error (the bisection of `time_reaching`); the bounded walk leaves that train as it
//! was. Those differences are counted (`stats` [6], [7]), not refused.

/// Shifts the two walks may differ by, seconds: rounding, not a train put elsewhere.
pub const TOL_S: f64 = 1e-6;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::TOL_S;
    use crate::track::net::Network;
    use crate::track::service::{Before, Service};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;

    thread_local! {
        static INNER: Cell<bool> = const { Cell::new(false) };
    }
    /// before_edit calls, their ns, trains running at them; note_changes calls, their ns, trains walked;
    /// lines whose shifts differ from the full walk's by rounding, the most (f64 bits); calls checked
    static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

    fn var(k: &str) -> &'static str {
        static V: OnceLock<[String; 2]> = OnceLock::new();
        let v = V.get_or_init(|| ["TW_TRAINS_CHECK", "TW_TRAINS"].map(|k| std::env::var(k).unwrap_or_default()));
        if k == "TW_TRAINS_CHECK" { &v[0] } else { &v[1] }
    }
    fn inner() -> bool {
        INNER.with(|c| c.get())
    }
    pub fn full() -> bool {
        inner() || var("TW_TRAINS") == "full"
    }
    pub fn before(t0: std::time::Instant, trains: usize) {
        if inner() {
            return;
        }
        STATS[0].fetch_add(1, Relaxed);
        STATS[1].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        STATS[2].fetch_add(trains as u64, Relaxed);
    }
    /// The service as the edit left it, for the full walk to run on: only when checking.
    pub fn shadow(svc: &Service) -> Option<Box<Service>> {
        (!inner() && !var("TW_TRAINS_CHECK").is_empty()).then(|| Box::new(svc.clone()))
    }
    /// `TW_TRAINS_CHECK=skip`: the trains of the first line with one to move again left where they were.
    pub fn skip_line(skip: &mut u32, line: u32) -> bool {
        if inner() || var("TW_TRAINS_CHECK") != "skip" {
            return false;
        }
        if *skip == u32::MAX {
            *skip = line;
        }
        *skip == line
    }
    pub fn after(net: &Network, svc: &Service, shadow: Option<Box<Service>>, before: &Before, walked: usize, now: f64, t0: std::time::Instant) {
        if inner() {
            return;
        }
        STATS[3].fetch_add(1, Relaxed);
        STATS[4].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        STATS[5].fetch_add(walked as u64, Relaxed);
        let Some(mut all) = shadow else { return };
        STATS[8].fetch_add(1, Relaxed);
        INNER.with(|c| c.set(true));
        all.note_changes(net, before, now);
        INNER.with(|c| c.set(false));
        assert_eq!(svc.eras.len(), all.eras.len(), "trains check: era tables differ in length");
        let mut bad = vec![];
        for l in 0..svc.eras.len() {
            if svc.eras[l] != all.eras[l] {
                bad.push((l, "eras".to_string()));
                continue;
            }
            let sorted = |v: &[(u8, f64, f64)]| {
                let mut v = v.to_vec();
                v.sort_by(|a, b| (a.0, a.1).partial_cmp(&(b.0, b.1)).unwrap());
                v
            };
            let (a, b) = (sorted(svc.shifts.get(l).map_or(&[][..], |v| &v[..])), sorted(all.shifts.get(l).map_or(&[][..], |v| &v[..])));
            if a.len() != b.len() || a.iter().zip(&b).any(|(x, y)| (x.0, x.1) != (y.0, y.1)) {
                bad.push((l, format!("shifts hold other trains: {a:?} vs {b:?}")));
                continue;
            }
            let most = a.iter().zip(&b).map(|(x, y)| (x.2 - y.2).abs()).fold(0.0, f64::max);
            if most > TOL_S {
                bad.push((l, format!("a shift differs by {most:e} s")));
            } else if most > 0.0 {
                STATS[6].fetch_add(1, Relaxed);
                STATS[7].fetch_max(most.to_bits(), Relaxed);
            }
        }
        assert!(bad.is_empty(), "trains check: {} lines left otherwise than by the full walk: {:?}", bad.len(), &bad[..bad.len().min(6)]);
    }
    pub fn stats() -> [u64; 9] {
        std::array::from_fn(|i| STATS[i].load(Relaxed))
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::{after, before, full, shadow, skip_line, stats};

#[cfg(target_arch = "wasm32")]
pub fn full() -> bool {
    false
}
