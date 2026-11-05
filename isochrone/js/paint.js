// Solved levels to RGBA pixels.
import { cfg } from './config.js';

// RGBA pixels, linear in Web Mercator y (how MapLibre stretches an image between its
// corners), one canvas column per cell column.
const MAX_PIXELS = 20e6;
// Canoe starts: water (sea, lakes, river water) is this far mixed towards white, so coasts show (T-063).
const WATER_MIX = 0.22;
const merc = lat => Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360));
export function paint(L, canoe) {
    const { W, H, dist, north, south } = L;
    const { bands, alpha } = cfg;
    const yTop = merc(north), yBot = merc(south);
    const lonRad = W * L.cellDeg * Math.PI / 180;
    const cw = W, ch0 = Math.max(1, Math.round(cw * (yTop - yBot) / lonRad));
    // Several canvas rows per cell row, so the hole the page cuts for a finer window (whole canvas
    // rows) lands within a fraction of a cell of the window's edge (T-037). Capped by pixel count.
    const sub = Math.max(1, Math.min(4, Math.floor(MAX_PIXELS / (cw * ch0))));
    const ch = Math.max(1, Math.min(8192, ch0 * sub));
    const px = new Uint8ClampedArray(cw * ch * 4);
    const info = canoe && L.cn ? L.cn.info : null;   // canoe starts: its low byte is the cell's water kind, 0 on land
    const edges = bands.map(b => b.h), cols = bands.map(b => b.color);
    const nb = edges.length;
    for (let y = 0; y < ch; y++) {
        const my = yTop - (y + 0.5) / ch * (yTop - yBot);
        const lat = (2 * Math.atan(Math.exp(my)) - Math.PI / 2) * 180 / Math.PI;
        // The cell row this canvas row falls in, so cells are square blocks (same rule as hover).
        const r = Math.min(H - 1, Math.max(0, Math.floor((north - lat) / L.cellDeg)));
        const o0 = r * W;
        for (let x = 0; x < cw; x++) {
            const h = dist[o0 + x];
            if (!(h <= edges[nb - 1])) continue;
            let k = 0;
            while (h > edges[k]) k++;
            const o = (y * cw + x) * 4, col = cols[k];
            if (info && (info[o0 + x] & 0xFF) !== 0) {
                px[o] = col[0] + (255 - col[0]) * WATER_MIX; px[o + 1] = col[1] + (255 - col[1]) * WATER_MIX;
                px[o + 2] = col[2] + (255 - col[2]) * WATER_MIX;
            } else { px[o] = col[0]; px[o + 1] = col[1]; px[o + 2] = col[2]; }
            px[o + 3] = alpha;
        }
    }
    return { px, cw, ch };
}
