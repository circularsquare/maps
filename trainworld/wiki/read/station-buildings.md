# Station buildings

Draw an outline around some stations and they become one building. Every station whose platform touches the outline belongs, at any level, built or not. Touching means inside it or within 8 m of an edge.

The stations join into one complex, so people can change between any two of them on foot. Each keeps its own place on the map and its own walkers.

**What riders feel:**

- A wait counts 0.8 of its minutes, the same as a [waiting room](waiting-and-changing.md). A waiting room on top adds nothing.
- The fixed bother of changing trains counts 50%.
- The walk between platforms still counts in full. So does a transfer fare.

Nothing changes for the walk from home to the platform: that's [walking to stations](walking-to-stations.md).

**What it costs.** US$2,000 a square metre of floor, paid at once. No [upkeep](upkeep.md). Redrawing it pays only for floor it didn't have before. Giving floor up, or knocking the building down, refunds nothing.

**The limits.** It takes 3 to 24 corners, no two more than 1 km apart, at least 1,000 square metres of floor, and at least one built station touching it. It can't overlap another building. Over water and track is fine. A station removed later leaves its building standing, empty.

Exact rule: `docs/track-and-service.md` "Stations" for the outline and price, `docs/demand-kernel.md` "Access and station-to-station times" for what riders feel. The same wait and change rules apply to commutes, everyday trips and trips between cities.
In the code: **main** `sim/src/track/building.rs` `members`, also `touches` `sim/src/track/wasm/buildings.rs` `sim/src/demand/kernel/subzones.rs` `wait_factor` `sim/src/demand/kernel/search.rs` `change_f` `app/src/game/buildings.ts` `BUILDING_TOUCH_M`
