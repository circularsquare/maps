//! The track model (T-040): SPEC 6 and notes/T-008.md in code. Design notes, API and numbers:
//! notes/T-040.md.
//!
//! - `params`: every constant (speeds, levels, costs, capacity).
//! - `geom`: PIs to arcs and straights, heights and ramps, positions, sampling.
//! - `cost`: build costs and the water mask interface.
//! - `cross`: exact intersections and the 1 km grid.
//! - `net`: the network as flat arrays with stable ids, derived geometry, ports, crossings,
//!   validity, routing, split planning.
//! - `profile`: minimum-time run profiles as constant-acceleration phases, holds.
//! - `capacity`: resources, utilisation, Kingman delay, holds per line.
//! - `service`: per line and direction: runs, profiles per demand level, stop times, round trip,
//!   trains needed, the trips to draw.
//! - `save`: the network part of a save (varints, mm deltas).
//! - `wasm`: the wasm-bindgen API for the clock worker (a sketch).
//! - `bench`: a synthetic 2,000 km network and edit timings (`src/bin/track_bench.rs`).
//! - `world`: edit operations with inverses, undo/redo, dirty sets; what the clock worker owns.

#[cfg(any(not(target_arch = "wasm32"), feature = "track-bench"))]
pub mod bench;
pub mod capacity;
pub mod cost;
pub mod cross;
pub mod geom;
pub mod net;
pub mod params;
pub mod profile;
pub mod save;
pub mod service;
/// The clock worker's API; left out of the demand workers' module (feature `track-api`, T-051).
#[cfg(feature = "track-api")]
pub mod wasm;
pub mod world;

#[cfg(test)]
mod tests;
