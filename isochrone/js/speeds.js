// The speeds the model uses, for the page's speeds panel (sent with `ready`). Canoe figures are
// worked out from cfg, so the panel follows the model.
import { cfg } from './config.js';

// MAP's off-road walking speeds by MODIS land cover, km/h (research/friction.md 1): baked into
// the walking surface the page reads. Grouped as the panel shows them.
const WALK = [
    ['Roads, towns', [['every road', 5], ['towns', 5]]],
    ['Grass, savanna', [['grassland', 4.86], ['savanna', 4.86], ['woody savanna', 4.86]]],
    ['Shrubs, bare ground', [['open shrubland', 4.2], ['closed shrubland', 3], ['barren', 3]]],
    ['Forest', [['broadleaf, leaf-shedding', 4], ['needleleaf', 3.24], ['mixed', 3.24], ['rainforest', 1.62]]],
    ['Farmland', [['cropland mosaic', 3.24], ['cropland', 2.5]]],
    ['Wetland', [['wetland', 2]]],
    ['Snow and ice', [['snow and ice', 1.62]]],
    ['Ponds under 5 km²', [['wading', 1]]],
];
// MAP's slowing for height and slope (research/friction.md 1), as factors on the speeds above
const elevFactor = m => 1.016 * Math.exp(-0.0001072 * m);
const slopeFactor = deg => Math.min(6 * Math.exp(-3.5 * Math.abs(Math.tan(deg * Math.PI / 180) + 0.05)), 5) / 5;

// River flow speed v = 0.51 Q^0.11 m/s (tools/canoe/rivers.py, docs/canoe.md 2)
const riverV = q => 0.51 * Math.pow(q, 0.11);
const RIVERS = [['Small river', 1], ['Big river', 100], ['Huge river', 10000]];   // m³/s
const SEA_CURRENT = 1;   // m/s, a strong current: the top 0.2% of the drifter grid

export function speeds() {
    const p = cfg.paddle, up = v => Math.max(p - cfg.upShare * v, cfg.upMin);
    return {
        walk: WALK.map(([label, parts]) => ({ label, parts,
            min: Math.min(...parts.map(x => x[1])), max: Math.max(...parts.map(x => x[1])) })),
        elev: [1000, 3000, 5000].map(m => [m, elevFactor(m)]),
        slope: [5, 10, 20].map(d => [d, slopeFactor(d)]),
        carry: cfg.carry, paddle: p, transfer: cfg.transfer, upMin: cfg.upMin, upShare: cfg.upShare,
        rivers: RIVERS.map(([label, q]) => {
            const v = riverV(q) * 3.6;
            return { label, q, v, down: p + v, up: up(v) };
        }),
        sea: { c: SEA_CURRENT * 3.6, with: p + SEA_CURRENT * 3.6, against: p - SEA_CURRENT * 3.6 },
    };
}
