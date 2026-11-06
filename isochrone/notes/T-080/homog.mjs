// Fine-grid crossing times of each coarse cell (upscaled, "effective" slowness).
// For a coarse cell of k x k fine cells: a search on the fine grid inside the cell plus m fine
// cells round it, from the fine column just west of the cell (time 0) to the cell's east column,
// and the same north to south. The search is the page's walking search (16 moves, a move costs
// its length times the mean hours per km of the cells it touches, knight moves need both
// cells between land). Returns hours per km per direction: min over the far edge ("best"
// crossing) and mean over the far edge's reached cells.
import { MDC, MDR } from '../../js/search.js';

const E0C = new Int32Array(16), E0R = new Int32Array(16), E1C = new Int32Array(16), E1R = new Int32Array(16);
{
    let k = 8;
    for (const sx of [1, -1]) for (const sy of [1, -1]) {
        E0C[k] = 0; E0R[k] = sy; E1C[k] = sx; E1R[k] = sy; k++;
        E0C[k] = sx; E0R[k] = 0; E1C[k] = sx; E1R[k] = sy; k++;
    }
}

// g: uint8 speeds (km/h x 40), W x H; rowLat(r) -> latitude of fine row r; cellDeg the fine cell.
export function crossings(g, W, H, k, m, north, cellDeg) {
    const KM = 111.195;
    const dyKm = cellDeg * KM, dxKm = new Float32Array(H);
    for (let r = 0; r < H; r++) dxKm[r] = cellDeg * KM * Math.cos((north - (r + 0.5) * cellDeg) * Math.PI / 180);
    const CW = W / k, CH = H / k;
    const out = { ewMin: new Float32Array(CW * CH).fill(Infinity), ewMean: new Float32Array(CW * CH).fill(Infinity),
                  nsMin: new Float32Array(CW * CH).fill(Infinity), nsMean: new Float32Array(CW * CH).fill(Infinity) };
    const B = k + 2 * m, NB = B * B;
    const dist = new Float64Array(NB), done = new Uint8Array(NB), heap = new Int32Array(NB * 16 + 16), inv = new Float64Array(NB);
    const hd = new Float64Array(NB * 16 + 16);
    const len = new Float64Array(B * 16);
    for (let R = 0; R < CH; R++) {
        const r0 = R * k - m;
        // move lengths by block row
        for (let br = 0; br < B; br++) {
            const r = Math.min(H - 1, Math.max(0, r0 + br));
            for (let q = 0; q < 16; q++) {
                const r2 = Math.min(H - 1, Math.max(0, r + MDR[q]));
                const dx = MDC[q] * (dxKm[r] + dxKm[r2]) * 0.5, dy = MDR[q] * dyKm;
                len[br * 16 + q] = Math.sqrt(dx * dx + dy * dy);
            }
        }
        for (let C = 0; C < CW; C++) {
            const c0 = C * k - m;
            let any = false;
            for (let br = 0; br < B; br++) for (let bc = 0; bc < B; bc++) {
                const r = r0 + br, c = c0 + bc;
                const v = r >= 0 && r < H && c >= 0 && c < W ? g[r * W + c] : 0;
                inv[br * B + bc] = v ? 40 / v : -1;
            }
            for (let i = m; i < m + k && !any; i++) for (let j = m; j < m + k; j++) if (inv[i * B + j] > 0) { any = true; break; }
            if (!any) continue;
            const ci = R * CW + C;
            for (const dir of [0, 1]) {
                // dir 0: west -> east; 1: north -> south. Source line just outside the cell.
                dist.fill(Infinity); done.fill(0);
                let hs = 0;
                const push = (n, d) => { // binary heap of (d, n) packed in two arrays
                    let i = hs++; heap[i] = n; hd[i] = d;
                    while (i > 0) { const p = (i - 1) >> 1; if (hd[p] <= hd[i]) break; swap(i, p); i = p; }
                };
                for (let t = 0; t < k; t++) {
                    const br = dir ? m - 1 : m + t, bc = dir ? m + t : m - 1;
                    if (br < 0 || bc < 0) continue;
                    const n = br * B + bc;
                    if (inv[n] > 0) { dist[n] = 0; push(n, 0); }
                }
                while (hs > 0) {
                    const u = heap[0], du = hd[0];
                    hs--; if (hs > 0) { heap[0] = heap[hs]; hd[0] = hd[hs]; sift(hs); }
                    if (done[u]) continue;
                    done[u] = 1;
                    const ur = (u / B) | 0, uc = u - ur * B, iu = inv[u];
                    for (let q = 0; q < 16; q++) {
                        const vr = ur + MDR[q], vc = uc + MDC[q];
                        if (vr < 0 || vr >= B || vc < 0 || vc >= B) continue;
                        const v = vr * B + vc;
                        if (done[v] || inv[v] < 0) continue;
                        let mean;
                        if (q < 8) mean = (iu + inv[v]) * 0.5;
                        else {
                            const a = inv[(ur + E0R[q]) * B + uc + E0C[q]], b = inv[(ur + E1R[q]) * B + uc + E1C[q]];
                            if (a < 0 || b < 0) continue;
                            mean = (iu + inv[v] + a + b) * 0.25;
                        }
                        const nd = du + len[ur * 16 + q] * mean;
                        if (nd < dist[v]) { dist[v] = nd; push(v, nd); }
                    }
                }
                let mn = Infinity, sum = 0, n = 0;
                for (let t = 0; t < k; t++) {
                    const br = dir ? m + k - 1 : m + t, bc = dir ? m + t : m + k - 1;
                    const d = dist[br * B + bc];
                    if (d < Infinity) { mn = Math.min(mn, d); sum += d; n++; }
                }
                const span = dir ? k * dyKm : k * dxKm[Math.min(H - 1, Math.max(0, R * k + (k >> 1)))];
                if (dir === 0) { out.ewMin[ci] = mn / span; out.ewMean[ci] = n ? sum / n / span : Infinity; }
                else { out.nsMin[ci] = mn / span; out.nsMean[ci] = n ? sum / n / span : Infinity; }
            }
        }
    }
    return out;
    function swap(i, j) { const a = heap[i]; heap[i] = heap[j]; heap[j] = a; const d = hd[i]; hd[i] = hd[j]; hd[j] = d; }
    function sift(n) {
        let i = 0;
        for (;;) {
            const l = 2 * i + 1; if (l >= n) break;
            let c = l; if (l + 1 < n && hd[l + 1] < hd[l]) c = l + 1;
            if (hd[c] >= hd[i]) break;
            swap(i, c); i = c;
        }
    }
}

// A coarse uint8 grid from crossing slowness s (hours per km) per direction, on the land cells of
// `base` (the build's grid: the land rule is kept). pick(ew, ns) -> hours per km or Infinity.
export function gridFrom(base, X, pick) {
    const out = new Uint8Array(base.length);
    for (let i = 0; i < base.length; i++) {
        if (!base[i]) continue;
        const s = pick(X, i);
        out[i] = s < Infinity && s > 0 ? Math.max(1, Math.min(255, Math.round(40 / s))) : base[i];
    }
    return out;
}
