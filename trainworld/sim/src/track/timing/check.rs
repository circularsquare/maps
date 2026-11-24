//! T-4132: `retime`'s check mode, natively. `TW_RETIME_CHECK=1` works every line out again from
//! scratch after each retime and panics where any line's variants, patterns or timing differ from
//! what the bounded one left; `TW_RETIME_CHECK=skip` also leaves the first cluster that moved
//! standing, so the check must fail; `order` compares with a full retime in the resources' index order
//! (as before T-4132) and counts the lines that differ. `TW_RETIME=full` works every cluster out again whenever one
//! moved (as before T-4132), to time the two in one build. `stats` counts and times them.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use crate::track::net::Network;
    use crate::track::service::Service;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;

    thread_local! {
        static INNER: Cell<bool> = const { Cell::new(false) };
    }
    /// calls, ns, clusters, clusters worked out, lines worked out, calls that worked out some
    /// clusters and kept others with variants, calls checked, lines timed otherwise in index order
    static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

    fn var(k: &str) -> &'static str {
        static V: OnceLock<[String; 2]> = OnceLock::new();
        let v = V.get_or_init(|| ["TW_RETIME_CHECK", "TW_RETIME"].map(|k| std::env::var(k).unwrap_or_default()));
        if k == "TW_RETIME_CHECK" { &v[0] } else { &v[1] }
    }
    fn inner() -> bool {
        INNER.with(|c| c.get())
    }
    /// `TW_RETIME_CHECK=order`: the full retime the check compares with reads the resources in
    /// their index order, as before T-4132, and the lines it times otherwise are counted, not refused.
    pub fn index_order() -> bool {
        inner() && var("TW_RETIME_CHECK") == "order"
    }
    pub fn full() -> bool {
        !inner() && var("TW_RETIME") == "full"
    }
    pub fn skip_one(stood: &mut [bool]) {
        if !inner() && var("TW_RETIME_CHECK") == "skip" {
            if let Some(s) = stood.iter_mut().find(|s| !**s) {
                *s = true;
            }
        }
    }
    pub fn after(net: &Network, svc: &Service, t0: std::time::Instant, clusters: usize, redone: usize, lines: usize, mixed: bool) {
        if inner() {
            return;
        }
        let add = |i: usize, x: u64| STATS[i].fetch_add(x, Relaxed);
        add(0, 1);
        add(1, t0.elapsed().as_nanos() as u64);
        add(2, clusters as u64);
        add(3, redone as u64);
        add(4, lines as u64);
        add(5, mixed as u64);
        if var("TW_RETIME_CHECK").is_empty() {
            return;
        }
        add(6, 1);
        let mut all = svc.clone();
        all.forget_timing();
        INNER.with(|c| c.set(true));
        crate::track::timing::retime(net, &mut all);
        INNER.with(|c| c.set(false));
        let bad: Vec<usize> = (0..svc.lines.len())
            .filter(|&l| {
                let (a, b) = (&svc.lines[l], &all.lines[l]);
                a.timed != b.timed || a.runs.iter().zip(&b.runs).any(|(x, y)| x.vars != y.vars || x.pattern != y.pattern)
            })
            .collect();
        if var("TW_RETIME_CHECK") == "order" {
            add(7, bad.len() as u64);
            for &l in &bad {
                let (a, b) = (&svc.lines[l], &all.lines[l]);
                for r in 0..a.runs.len() {
                    for lev in 0..a.runs[r].vars.len() {
                        let (x, y) = (&a.runs[r].vars[lev], &b.runs[r].vars[lev]);
                        if x != y || a.runs[r].pattern[lev] != b.runs[r].pattern[lev] {
                            let most = x.iter().zip(y).map(|(p, q)| (p.prof.duration - q.prof.duration).abs()).fold(0.0, f64::max);
                            let holds = x.iter().zip(y).map(|(p, q)| p.stop_holds.iter().zip(&q.stop_holds).map(|(s, t)| (s - t).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
                            let pat = a.runs[r].pattern[lev].iter().zip(&b.runs[r].pattern[lev]).filter(|(p, q)| p != q).count();
                            eprintln!("ORDER line {l} run {r} lev {lev}: vars {} vs {}, most duration diff {most:e} s, stop hold diff {holds:e} s, pattern entries differing {pat} of {}", x.len(), y.len(), a.runs[r].pattern[lev].len());
                        }
                    }
                }
            }
            return;
        }
        assert!(bad.is_empty(), "retime check: {} lines timed otherwise than by a full retime: {:?}", bad.len(), &bad[..bad.len().min(12)]);
    }
    pub fn stats() -> [u64; 8] {
        std::array::from_fn(|i| STATS[i].load(Relaxed))
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::{after, full, index_order, skip_one, stats};

#[cfg(target_arch = "wasm32")]
pub fn full() -> bool {
    false
}
#[cfg(target_arch = "wasm32")]
pub fn skip_one(_: &mut [bool]) {}
#[cfg(target_arch = "wasm32")]
pub fn index_order() -> bool {
    false
}
