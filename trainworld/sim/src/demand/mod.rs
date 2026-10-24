//! The demand model (SPEC 4.3, 4.4), everything flat structure-of-arrays. Where each part lives
//! is `docs/code-map.md`; the stages of one city and one period are:
//!
//! - `pack`: city pack format 1 loader (notes/T-004.md) and writer (synthetic packs, and the
//!   pipeline's gravity step `bin/pack_gravity.rs`).
//! - `synth`: a New York-sized synthetic city and a hand-made network.
//! - `kernel`: gravity between coarse zones, per-cell station access, access subzones,
//!   station-to-station search, mode choice over subzone pairs, assignment, crowding (MSA).
//! - `api`, `everyday` and `intercity`: the three populations the demand workers solve, commutes,
//!   everyday outings and the regional market.
//! - `bench`: runs the stages with timings and heap figures (`src/bin/demand_bench.rs`).
//!
//! Design and numbers: notes/T-005.md.

/// The counting allocator and the benchmark that reads it: native only, so the workers' wasm does
/// not count every allocation.
#[cfg(not(target_arch = "wasm32"))]
pub mod alloc;
/// T-026: the demand workers' API (`DemandApi`), in the app's wasm build.
pub mod api;
pub mod intercity;
pub mod everyday;
#[cfg(not(target_arch = "wasm32"))]
pub mod bench;
pub mod clock;
/// T-260: kinds of day (weekday, weekend), worked out from the weekday's solve.
pub mod days;
pub mod kernel;
/// T-036: four f32 lanes (wasm simd128, SSE2) for the commute pair loop.
pub mod lanes;
pub mod network;
pub mod pack;
/// T-275: routed walks (around water, over bridges) and drives (OSM roads) from the pack.
pub mod routes;
pub mod synth;
/// T-195: the trip explorer's query (one trip's route, costs and mode chances).
pub mod trip;
#[cfg(test)]
mod t720;
/// T-173: the everyday warm-up a demand worker runs at startup.
pub mod warm;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
