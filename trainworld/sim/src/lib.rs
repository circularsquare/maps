//! Simulation core, compiled to WebAssembly and run in Web Workers: the track model for the
//! clock worker, and the demand kernel for the demand workers.

/// The demand kernel and the demand workers' API. Behind the `demand` feature; the app's demand
/// module enables it (`wasm:demand`), the clock worker's module leaves it out.
#[cfg(feature = "demand")]
pub mod demand;

/// T-040 track model: geometry, validity, costs, edits, run-time profiles, capacity. The clock
/// worker's API is `track::wasm::TrackApi`.
pub mod track;
