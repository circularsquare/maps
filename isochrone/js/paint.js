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
    if (cfg.look === 'hex') return paintHex(L, info, ch0, yTop, yBot);
    if (cfg.look === 'diag') return paintLook(L, info, ch0, yTop, yBot);
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
//   hex:  the default. Pointy-top hexagons a cell across (HEX_SIZE), each taking the colour of the
//         cell under its centre. Their density is 1.155 per cell, so no cell is dropped and about
//         one in seven is shown twice.
const DIAG_CUT = 0.5;
const HEX_SIZE = 1;
const HEX_DY = 0.8660254;

// Hex: which cell each canvas pixel shows depends only on the grid's geometry, so it is worked out
// once per geometry (the last few kept: L10 repaints on every click with the same one) and a paint
// is then one lookup per pixel into the cells' packed colours, written as 32-bit words.
const hexMaps = new Map();
function hexMap(L, XS, cw, ch, yTop, yBot) {
    const { W, H, north, cellDeg } = L;
    const key = `${W},${H},${XS},${ch},${yTop},${yBot},${north},${cellDeg}`;
    let m = hexMaps.get(key);
    if (m) { hexMaps.delete(key); hexMaps.set(key, m); return m; }
    m = new Int32Array(cw * ch);
    const s = HEX_SIZE, inv = 1 / s;
    // per canvas column and for each of the two row parities (centres at whole or half units):
    // the nearest centre's column in cells and its squared distance along x
    const colP = [new Int32Array(cw), new Int32Array(cw)], dxP = [new Float64Array(cw), new Float64Array(cw)];
    for (let p = 0; p < 2; p++) {
        const off = p * 0.5;
        for (let x = 0; x < cw; x++) {
            const uu = (x + 0.5) / XS * inv, c = Math.floor(uu - off + 0.5) + off;
            dxP[p][x] = (uu - c) * (uu - c);
            colP[p][x] = Math.min(W - 1, Math.max(0, Math.floor(c * s)));
        }
    }
    for (let y = 0; y < ch; y++) {
        const my = yTop - (y + 0.5) / ch * (yTop - yBot);
        const lat = (2 * Math.atan(Math.exp(my)) - Math.PI / 2) * 180 / Math.PI;
        const vv = Math.min(H - 1e-6, Math.max(0, (north - lat) / cellDeg)) * inv;
        const j0 = Math.floor(vv / HEX_DY), j1 = j0 + 1;
        const p0 = j0 & 1, p1 = j1 & 1;
        const dx0 = dxP[p0], dx1 = dxP[p1], cl0 = colP[p0], cl1 = colP[p1];
        const cv0 = j0 * HEX_DY, cv1 = j1 * HEX_DY;
        const d0 = (vv - cv0) * (vv - cv0), d1 = (vv - cv1) * (vv - cv1);
        const r0 = Math.min(H - 1, Math.max(0, Math.floor(cv0 * s))) * W, r1 = Math.min(H - 1, Math.max(0, Math.floor(cv1 * s))) * W;
        const o = y * cw;
        for (let x = 0; x < cw; x++) {
            m[o + x] = dx1[x] + d1 < dx0[x] + d0 ? r1 + cl1[x] : r0 + cl0[x];
        }
    }
    hexMaps.set(key, m);
    if (hexMaps.size > 3) hexMaps.delete(hexMaps.keys().next().value);
    return m;
}

function paintHex(L, info, ch0, yTop, yBot) {
    const { W, H, dist } = L;
    const { bands, alpha } = cfg;
    const edges = bands.map(b => b.h), nb = edges.length;
    const XS = Math.max(1, Math.min(4, Math.floor(Math.sqrt(MAX_PIXELS / (W * ch0)))));
    const cw = W * XS, ch = Math.min(8192, ch0 * XS);
    // each band's packed colour, ABGR as a little-endian word (0: not painted)
    const pack = (r, g, b) => ((alpha << 24) | (Math.round(b) << 16) | (Math.round(g) << 8) | Math.round(r)) >>> 0;
    const land = bands.map(b => pack(b.color[0], b.color[1], b.color[2]));
    const water = bands.map(b => pack(b.color[0] + (255 - b.color[0]) * WATER_MIX, b.color[1] + (255 - b.color[1]) * WATER_MIX, b.color[2] + (255 - b.color[2]) * WATER_MIX));
    const C = new Uint32Array(W * H);
    const last = edges[nb - 1];
    let k = 0;   // neighbouring cells are mostly in the same band, so start from the last one's
    for (let i = 0; i < C.length; i++) {
        const h = dist[i];
        if (!(h <= last)) continue;
        while (k > 0 && h <= edges[k - 1]) k--;
        while (h > edges[k]) k++;
        C[i] = info && (info[i] & 0xFF) !== 0 ? water[k] : land[k];
    }
    const map = hexMap(L, XS, cw, ch, yTop, yBot);
    const buf = new ArrayBuffer(cw * ch * 4);
    const out = new Uint32Array(buf);
    for (let p = 0; p < out.length; p++) out[p] = C[map[p]];
    return { px: new Uint8ClampedArray(buf), cw, ch };
}

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
    const lat0 = L.cellDeg;
    for (let y = 0; y < ch; y++) {
        const my = yTop - (y + 0.5) / ch * (yTop - yBot);
        const lat = (2 * Math.atan(Math.exp(my)) - Math.PI / 2) * 180 / Math.PI;
        const v = Math.min(H - 1e-6, Math.max(0, (north - lat) / lat0));   // cell rows, fractional
        const r = Math.floor(v), fy = v - r;
        for (let x = 0; x < cw; x++) {
            const u = (x + 0.5) / XS, c = Math.min(W - 1, Math.floor(u)), fx = u - c;
            let cell = r * W + c;
            {
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
