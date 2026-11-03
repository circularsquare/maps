//! The track model (T-040): SPEC 6 and notes/T-008.md in code. Design notes, API and numbers:
//! notes/T-040.md.
//!
//! - `params`: every constant (speeds, levels, costs, capacity).
//! - `geom`: PIs to arcs and straights, heights and ramps, positions, sampling.
//! - `frame`: the world frames, planar metres to and from longitude and latitude (T-125).
//! - `bridge`: a ground-level segment across water becomes a bridge with ramps on land (T-656).
//! - `building`: station buildings (T-2217): outlines round stations, their members and refusals.
//! - `bus`: bus lines (T-1464): a drawn route with no track, its stops and its trip.
//! - `ferry`: ferry lines (T-2193): ports, a route on water the app searched, the boats' trip.
//! - `cost`: build costs and the water mask interface.
//! - `cross`: exact intersections and the 1 km grid.
//! - `net`: the network as flat arrays with stable ids, derived geometry, ports, crossings,
//!   validity, routing, split planning.
//! - `profile`: minimum-time run profiles as constant-acceleration phases, holds.
//! - `capacity`: resources, utilisation, Kingman delay, holds per line.
//! - `overtake`: passing tracks (T-2231): where slower lines wait for faster ones to pass.
//! - `service`: per line and direction: runs, profiles per demand level, stop times, round trip,
//!   trains needed, the trips to draw.
//! - `streets`: the car's street speed by place, for a bus's speed (T-1577).
//! - `span`: each stop's service span for demand (T-842): the share of a period's riders with a
//!   train due within their tolerance.
//! - `boarding`: the boarding account (T-164): demand's riders per period turned into each
//!   train's load and each platform's crowd, a day at a time.
//! - `localtime`: each region's local time (T-309): the zone table and a place's shift.
//! - `names`: station and line names from a real network, tidied (T-1442); on load and in `realsave`.
//! - `save`: the network part of a save (varints, mm deltas).
//! - `trace`: a real railway's line as the corners the track tool places (T-727c).
//! - `upgrades`: station upgrades (T-1281): their bits, prices and turning rules.
//! - `wasm`: the wasm-bindgen API for the clock worker.
//! - `bench`: a synthetic 2,000 km network and edit timings (`src/bin/track_bench.rs`).
//! - `world`: edit operations with inverses, undo/redo, dirty sets; what the clock worker owns.

#[cfg(not(target_arch = "wasm32"))]
pub mod bench;
pub mod boarding;
pub mod bridge;
pub mod building;
pub mod bus;
pub mod capacity;
pub mod cost;
pub mod cross;
pub mod ferry;
pub mod frame;
pub mod geom;
pub mod localtime;
pub mod names;
pub mod net;
pub mod overtake;
pub mod params;
pub mod profile;
pub mod related;
pub mod roads;
pub mod save;
pub mod service;
pub mod spacing;
pub mod span;
pub mod streets;
pub mod trace;
pub mod upgrades;
/// The clock worker's API; left out of the demand workers' module (feature `track-api`, T-051).
#[cfg(feature = "track-api")]
pub mod wasm;
pub mod world;

#[cfg(test)]
mod frame_tests;
#[cfg(test)]
mod merc_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod transition_tests;
