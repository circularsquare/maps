//! Simulation core, compiled to WebAssembly and run in Web Workers: the track model for the
//! clock worker, and the demand kernel for the demand workers.

/// The demand kernel and the demand workers' API. Behind the `demand` feature; the app's demand
/// module enables it (`wasm:demand`), the clock worker's module leaves it out.
#[cfg(feature = "demand")]
pub mod demand;

/// The wasm memory grown ahead of need, and the clock worker's module's allocator (T-2699).
#[cfg(target_arch = "wasm32")]
pub mod heap;

/// T-2589: the network diagram's carry solve (`diagram::CarrySolver`), for the diagram worker.
#[cfg(feature = "diagram")]
pub mod diagram;

/// T-040 track model: geometry, validity, costs, edits, run-time profiles, capacity. The clock
/// worker's API is `track::wasm::TrackApi`.
pub mod track;
