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
    const info = canoe && L.cn ? L.cn.info : null;   // canoe starts: its low byte is the cell's water kind, 0 on land
    const edges = bands.map(b => b.h), cols = bands.map(b => b.color);
    const nb = edges.length;
    if (cfg.look === 'diag' || cfg.look === 'hex') return paintLook(L, info, ch0, yTop, yBot);
    const px = new Uint8ClampedArray(cw * ch * 4);
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

// Other cell looks (T-060), painting only: the grid stays square. Each canvas pixel takes the colour
// of one cell, chosen by the look. Canvas columns and rows are both XS per cell, so a pixel has a
// place inside its cell.
//   diag: where the two neighbours on a corner share a band that is not the cell's own, the corner
//         triangle (legs DIAG_CUT of a cell) takes that band: step corners read as 45 degree cuts.
//   hex:  pointy-top hexagons a cell across (HEX_SIZE), each taking the colour of the cell under its
//         centre. Their density is 1.155 per cell, so no cell is dropped and about one in seven is
//         shown twice.
const DIAG_CUT = 0.5;
const HEX_SIZE = 1;
const HEX_DY = 0.8660254;
function paintLook(L, info, ch0, yTop, yBot) {
    const { W, H, dist, north } = L;
    const { bands, alpha } = cfg;
    const edges = bands.map(b => b.h), nb = edges.length;
    const XS = Math.max(1, Math.min(4, Math.floor(Math.sqrt(MAX_PIXELS / (W * ch0)))));
    const cw = W * XS, ch = Math.min(8192, ch0 * XS);
    // each cell's band (255: not painted)
    const B = new Uint8Array(W * H).fill(255);
    for (let i = 0; i < B.length; i++) {
        const h = dist[i];
        if (!(h <= edges[nb - 1])) continue;
        let k = 0;
        while (h > edges[k]) k++;
        B[i] = k;
    }
    const px = new Uint8ClampedArray(cw * ch * 4);
    const hex = cfg.look === 'hex';
    const lat0 = L.cellDeg;
    for (let y = 0; y < ch; y++) {
        const my = yTop - (y + 0.5) / ch * (yTop - yBot);
        const lat = (2 * Math.atan(Math.exp(my)) - Math.PI / 2) * 180 / Math.PI;
        const v = Math.min(H - 1e-6, Math.max(0, (north - lat) / lat0));   // cell rows, fractional
        const r = Math.floor(v), fy = v - r;
        for (let x = 0; x < cw; x++) {
            const u = (x + 0.5) / XS, c = Math.min(W - 1, Math.floor(u)), fx = u - c;
            let cell = r * W + c;
            if (hex) {
                const s = HEX_SIZE, uu = u / s, vv = v / s;
                const j0 = Math.floor(vv / HEX_DY);
                let best = Infinity, bu = 0, bv = 0;
                for (let j = j0; j <= j0 + 1; j++) {
                    const off = (j & 1) ? 0.5 : 0;
                    const cu = Math.round(uu - off) + off, cv = j * HEX_DY;
                    const d = (uu - cu) * (uu - cu) + (vv - cv) * (vv - cv);
                    if (d < best) { best = d; bu = cu; bv = cv; }
                }
                const hc = Math.min(W - 1, Math.max(0, Math.floor(bu * s))), hr = Math.min(H - 1, Math.max(0, Math.floor(bv * s)));
                cell = hr * W + hc;
            } else {
                const b = B[cell];
                const n = r > 0 ? cell - W : -1, s = r < H - 1 ? cell + W : -1;
                const w = c > 0 ? cell - 1 : -1, e = c < W - 1 ? cell + 1 : -1;
                if (fy < DIAG_CUT) {
                    if (n >= 0 && B[n] !== b) {
                        if (w >= 0 && B[w] === B[n] && fx + fy < DIAG_CUT) cell = n;
                        else if (e >= 0 && B[e] === B[n] && (1 - fx) + fy < DIAG_CUT) cell = n;
                    }
                } else if (fy > 1 - DIAG_CUT) {
                    if (s >= 0 && B[s] !== b) {
                        if (w >= 0 && B[w] === B[s] && fx + (1 - fy) < DIAG_CUT) cell = s;
                        else if (e >= 0 && B[e] === B[s] && (1 - fx) + (1 - fy) < DIAG_CUT) cell = s;
                    }
                }
            }
            const k = B[cell];
            if (k === 255) continue;
            const o = (y * cw + x) * 4, col = bands[k].color;
            if (info && (info[cell] & 0xFF) !== 0) {
                px[o] = col[0] + (255 - col[0]) * WATER_MIX; px[o + 1] = col[1] + (255 - col[1]) * WATER_MIX;
                px[o + 2] = col[2] + (255 - col[2]) * WATER_MIX;
            } else { px[o] = col[0]; px[o + 1] = col[1]; px[o + 2] = col[2]; }
            px[o + 3] = alpha;
        }
    }
    return { px, cw, ch };
}
