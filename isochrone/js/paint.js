// Solved levels to RGBA pixels.
import { cfg } from './config.js';

// RGBA pixels, linear in Web Mercator y (how MapLibre stretches an image between its
// corners), one canvas column per cell column.
const merc = lat => Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360));
export function paint(L) {
    const { W, H, dist, north, south } = L;
    const { bands, alpha } = cfg;
    const yTop = merc(north), yBot = merc(south);
    const lonRad = W * L.cellDeg * Math.PI / 180;
    const cw = W, ch = Math.max(1, Math.min(8192, Math.round(cw * (yTop - yBot) / lonRad)));
    const px = new Uint8ClampedArray(cw * ch * 4);
    const edges = bands.map(b => b.h), cols = bands.map(b => b.color);
    const nb = edges.length;
    for (let y = 0; y < ch; y++) {
        const my = yTop - (y + 0.5) / ch * (yTop - yBot);
        const lat = (2 * Math.atan(Math.exp(my)) - Math.PI / 2) * 180 / Math.PI;
        const v = (north - lat) / L.cellDeg - 0.5;
        let r0 = Math.floor(v); const fy = v - r0;
        let r1 = r0 + 1;
        if (r0 < 0) r0 = 0; if (r1 > H - 1) r1 = H - 1; if (r0 > H - 1) r0 = H - 1;
        const o0 = r0 * W, o1 = r1 * W;
        // Blend the two rows when both cells were reached, else take the nearer.
        for (let x = 0; x < cw; x++) {
            const a = dist[o0 + x], c = dist[o1 + x];
            const h = (a !== Infinity && c !== Infinity) ? a * (1 - fy) + c * fy : (fy < 0.5 ? a : c);
            if (!(h <= edges[nb - 1])) continue;
            let k = 0;
            while (h > edges[k]) k++;
            const o = (y * cw + x) * 4, col = cols[k];
            px[o] = col[0]; px[o + 1] = col[1]; px[o + 2] = col[2]; px[o + 3] = alpha;
        }
    }
    return { px, cw, ch };
}
