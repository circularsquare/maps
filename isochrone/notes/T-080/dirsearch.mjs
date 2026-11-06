// A walking search over a coarse level whose cells carry two crossing slownesses (hours per km
// east-west and north-south): a move of (dx, dy) km through a cell costs sqrt((sx dx)^2 + (sy dy)^2),
// averaged over the cells the move touches as the page's search does. Start seed only.
import { MDC, MDR } from '../../js/search.js';

const E0C = new Int32Array(16), E0R = new Int32Array(16), E1C = new Int32Array(16), E1R = new Int32Array(16);
{
    let k = 8;
    for (const sx of [1, -1]) for (const sy of [1, -1]) {
        E0C[k] = 0; E0R[k] = sy; E1C[k] = sx; E1R[k] = sy; k++;
        E0C[k] = sx; E0R[k] = 0; E1C[k] = sx; E1R[k] = sy; k++;
    }
}

export function solveDir(L, sx, sy, start, maxHours = 1024) {
    const { W, H, dxKm, dyKm } = L;
    const N = W * H;
    const DX = new Float32Array(H * 16), DY = new Float32Array(H * 16);
    for (let r = 0; r < H; r++) for (let k = 0; k < 16; k++) {
        const r2 = Math.min(H - 1, Math.max(0, r + MDR[k]));
        DX[r * 16 + k] = Math.abs(MDC[k] * (dxKm[r] + dxKm[r2]) * 0.5); DY[r * 16 + k] = Math.abs(MDR[k] * dyKm);
    }
    const ok = (i) => sx[i] < Infinity || sy[i] < Infinity;
    const cost = (i, dx, dy) => {
        const a = sx[i] < Infinity ? sx[i] : sy[i], b = sy[i] < Infinity ? sy[i] : sx[i];
        return Math.sqrt(a * a * dx * dx + b * b * dy * dy);
    };
    const dist = new Float32Array(N).fill(Infinity);
    const done = new Uint8Array(N);
    let heapN = new Int32Array(1 << 20), heapD = new Float32Array(1 << 20), hs = 0;
    const push = (n, d) => {
        if (hs >= heapN.length) { const a = new Int32Array(heapN.length * 2); a.set(heapN); heapN = a; const b = new Float32Array(heapD.length * 2); b.set(heapD); heapD = b; }
        let i = hs++;
        while (i > 0) { const p = (i - 1) >> 1; if (heapD[p] <= d) break; heapN[i] = heapN[p]; heapD[i] = heapD[p]; i = p; }
        heapN[i] = n; heapD[i] = d;
    };
    const pop = () => {
        const n = heapN[0]; hs--;
        if (hs > 0) {
            const ln = heapN[hs], ld = heapD[hs];
            let i = 0;
            for (;;) {
                let c = 2 * i + 1; if (c >= hs) break;
                if (c + 1 < hs && heapD[c + 1] < heapD[c]) c++;
                if (heapD[c] >= ld) break;
                heapN[i] = heapN[c]; heapD[i] = heapD[c]; i = c;
            }
            heapN[i] = ln; heapD[i] = ld;
        }
        return n;
    };
    dist[start] = 0; push(start, 0);
    while (hs > 0) {
        const du0 = heapD[0], u = pop();
        if (done[u]) continue;
        done[u] = 1;
        const du = dist[u];
        if (du > maxHours) break;
        const r = (u / W) | 0, c = u - r * W;
        for (let k = 0; k < 16; k++) {
            const r2 = r + MDR[k], c2 = c + MDC[k];
            if (r2 < 0 || r2 >= H || c2 < 0 || c2 >= W) continue;
            const v = r2 * W + c2;
            if (done[v] || !ok(v)) continue;
            const dx = DX[r * 16 + k], dy = DY[r * 16 + k];
            let t;
            if (k < 8) t = (cost(u, dx, dy) + cost(v, dx, dy)) * 0.5;
            else {
                const i0 = (r + E0R[k]) * W + c + E0C[k], i1 = (r + E1R[k]) * W + c + E1C[k];
                if (!ok(i0) || !ok(i1)) continue;
                t = (cost(u, dx, dy) + cost(v, dx, dy) + cost(i0, dx, dy) + cost(i1, dx, dy)) * 0.25;
            }
            const nd = du + t;
            if (nd < dist[v]) { dist[v] = nd; push(v, nd); }
        }
    }
    return dist;
}
