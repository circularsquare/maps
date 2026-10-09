//! The local demand kernel (SPEC 4.3, 4.4): T-005 spike, gravity from the pack (T-019), access
//! logit and the other access leg (T-020). Readable, not final.
//!
//! Pipeline for one city and one period, everything flat structure-of-arrays:
//!
//! - `pack`: city pack format 1 loader (notes/T-004.md) and writer (synthetic packs, and the
//!   pipeline's gravity step `bin/pack_gravity.rs`).
//! - `synth`: a New York-sized synthetic city and a hand-made network.
//! - `kernel`: gravity between coarse zones, per-cell station access, access subzones,
//!   station-to-station search, mode choice over subzone pairs, assignment, crowding (MSA).
//! - `bench`: runs the stages with timings and heap figures; shared by the native binary
//!   (`src/bin/demand_bench.rs`) and the wasm export (`wasm.rs`).
//!
//! Design and numbers: notes/T-005.md.

pub mod alloc;
/// T-026: the demand workers' API (`DemandApi`), in the app's wasm build.
pub mod api;
pub mod bench;
pub mod clock;
pub mod kernel;
pub mod network;
pub mod pack;
pub mod synth;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
