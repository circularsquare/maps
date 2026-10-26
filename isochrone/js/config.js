// What the page sends with `init` and the searches and paint read. An object, since an
// imported binding can't be assigned from another module.
export const cfg = {
    bands: null, alpha: 190, maxHours: 1024,
    inv: null,                   // hours per km for each stored value, -1 for water
};

export function setConfig(msg) {
    cfg.bands = msg.bands; cfg.alpha = msg.alpha; cfg.maxHours = msg.maxHours;
    cfg.inv = new Float32Array(256).fill(-1);
    for (let v = 1; v < 256; v++) cfg.inv[v] = msg.meta.scale / v;
}
