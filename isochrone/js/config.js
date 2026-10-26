// What the page sends with `init` and the searches and paint read. An object, since an
// imported binding can't be assigned from another module.
export const cfg = {
    bands: null, alpha: 190, maxHours: 1024,
    inv: null,                   // hours per km for each stored value, -1 for water
    canoeBase: 'data/canoe/',    // canoe data folder, relative to worker.js
    // the canoe model (docs/canoe.md 2, 3)
    K: 1000,                     // hours of weight per unit of hazard
    hazard: true,                // water edges carry hazard on a canoe start; off, weight = time
    // water hazard, per real hour on the water (docs/canoe.md 3, research/canoe.md 5)
    hz: {
        river: 1e-5, rapids: 1e-4,   // flat river; runnable rapids
        water: 2e-5,                 // sea or lake, anywhere
        // summer wave height (m, upper bound of each step) -> rate offshore
        waves: [[0.5, 0], [1, 2e-4], [1.5, 1e-3], [2, 3e-3], [3, 1e-2], [Infinity, 3e-2]],
        waveKm: 20,                  // the wave rate grows from 0 at the shore to full here
        cold: [[5, 3], [15, 2]],     // sea temperature below (C) -> wave rate multiplier
        supplyKm: [250, 600], supply: 0.05,  // supplies: 0 at the first distance, rising to this
        farKm: 20, exposure: 3,      // past farKm a paddling hour is this many real hours out
    },
    paddle: 5,                   // km/h on still water
    upMin: 1.5,                  // km/h, the slowest upstream paddling (lining)
    upShare: 0.6,                // upstream speed is paddle - upShare x river speed
    transfer: 0.25,              // hours to get in or out of the canoe
    carry: 2,                    // carrying the canoe takes this many times the walking time
    iceMax: 15,                  // % summer ice cover above which a sea cell has no canoe node
    hzStop: Math.log(100),       // survival under 1% stops a path
    sea8: true,                  // 8 moves only past 20 km from shore: 16 took ~1.0-1.1 s a click (T-022)
};

export function setConfig(msg) {
    cfg.bands = msg.bands; cfg.alpha = msg.alpha; cfg.maxHours = msg.maxHours;
    if (msg.canoeBase) cfg.canoeBase = msg.canoeBase;
    cfg.inv = new Float32Array(256).fill(-1);
    for (let v = 1; v < 256; v++) cfg.inv[v] = msg.meta.scale / v;
}
