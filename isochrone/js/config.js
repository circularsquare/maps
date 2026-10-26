// What the page sends with `init` and the searches and paint read. An object, since an
// imported binding can't be assigned from another module.
export const cfg = {
    bands: null, alpha: 190, maxHours: 1024,
    inv: null,                   // hours per km for each stored value, -1 for water
    canoeBase: 'data/canoe/',    // canoe data folder, relative to worker.js
    // the canoe model (docs/canoe.md 2, 3)
    K: 1000,                     // hours of weight per unit of hazard
    hazard: false,               // edges carry hazard (T-006); off, the search keeps weight = time
    paddle: 5,                   // km/h on still water
    upMin: 1.5,                  // km/h, the slowest upstream paddling (lining)
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
