// T-4257: a bus's speeds on each kind of road, from the sim (`bus_road_speeds`, sim/src/track/wasm/bus.rs), so the route
// tool's quickest road is the one the sim would time fastest. The track model's code is loaded on the page once (the
// same module `ghostLocal` uses).

let loading: Promise<[number, number, number] | null> | null = null;

/** km/h on local streets, on trunk roads and motorways, and on busways; null if the code could not be loaded */
export function busSpeeds(): Promise<[number, number, number] | null> {
  loading ??= import("../wasm/track/sim")
    .then(async (m) => {
      await m.default();
      const v = m.bus_road_speeds();
      return [v[0], v[1], v[2]] as [number, number, number];
    })
    .catch(() => null)
    .then((v) => {
      if (!v) loading = null; // a failed load is tried again by the next ask
      return v;
    });
  return loading;
}
