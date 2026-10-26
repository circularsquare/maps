// Dijkstra with an indexed binary heap, 16 neighbours per cell.
import { cfg } from './config.js';

// 16 moves as flat tables: 8 neighbours, then 8 knight moves. A knight move also counts
// the speed of the two cells it passes between, which must be land too, so a one cell wide
// strait or river of water can't be hopped over. Those two cells always lie between the
// start and end rows and columns, so they need no bounds check of their own.
const MDC = new Int32Array(16), MDR = new Int32Array(16);
const E0C = new Int32Array(16), E0R = new Int32Array(16), E1C = new Int32Array(16), E1R = new Int32Array(16);
(function buildMoves() {
    const plain = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    let k = 0;
    for (const [dc, dr] of plain) { MDC[k] = dc; MDR[k] = dr; k++; }
    for (const sx of [1, -1]) for (const sy of [1, -1]) {
        MDC[k] = sx; MDR[k] = 2 * sy; E0C[k] = 0; E0R[k] = sy; E1C[k] = sx; E1R[k] = sy; k++;
        MDC[k] = 2 * sx; MDR[k] = sy; E0C[k] = sx; E0R[k] = 0; E1C[k] = sx; E1R[k] = sy; k++;
    }
})();

// Yield to the event loop so newer messages can land; much faster than setTimeout.
const chan = new MessageChannel();
const wakers = [];
chan.port1.onmessage = () => wakers.shift()();
const tick = () => new Promise(res => { wakers.push(res); chan.port2.postMessage(0); });

// Dijkstra from seeds (cell index -> hours). Returns dist, or null if alive() went false.
export async function solve(L, seedIdx, seedT, alive) {
    const { grid, W, H, wrap, dxKm, dyKm } = L;
    const { inv, maxHours } = cfg;
    const N = W * H;
    // length of each move from each row, km
    const stepKm = new Float32Array(H * 16);
    for (let r = 0; r < H; r++) for (let k = 0; k < 16; k++) {
        const r2 = r + MDR[k];
        if (r2 < 0 || r2 >= H) continue;
        const dx = MDC[k] * (dxKm[r] + dxKm[r2]) * 0.5, dy = MDR[k] * dyKm;
        stepKm[r * 16 + k] = Math.sqrt(dx * dx + dy * dy);
    }
    const dist = new Float32Array(N).fill(Infinity);
    const heap = new Int32Array(N);
    const pos = new Int32Array(N).fill(-1);
    let size = 0;

    function up(i) {
        const node = heap[i], d = dist[node];
        while (i > 0) {
            const p = (i - 1) >> 1;
            if (dist[heap[p]] <= d) break;
            heap[i] = heap[p]; pos[heap[i]] = i; i = p;
        }
        heap[i] = node; pos[node] = i;
    }
    function down(i) {
        const node = heap[i], d = dist[node];
        for (;;) {
            let c = 2 * i + 1;
            if (c >= size) break;
            if (c + 1 < size && dist[heap[c + 1]] < dist[heap[c]]) c++;
            if (dist[heap[c]] >= d) break;
            heap[i] = heap[c]; pos[heap[i]] = i; i = c;
        }
        heap[i] = node; pos[node] = i;
    }

    for (let k = 0; k < seedIdx.length; k++) {
        const v = seedIdx[k], t = seedT[k];
        if (!(t < dist[v])) continue;
        dist[v] = t;
        if (pos[v] === -1) { heap[size] = v; pos[v] = size; size++; up(size - 1); }
        else up(pos[v]);
    }

    let pops = 0, tLast = performance.now();
    while (size > 0) {
        if ((++pops & 0x3FFF) === 0 && performance.now() - tLast > 30) {
            await tick();
            if (!alive()) return null;
            tLast = performance.now();
        }
        const u = heap[0];
        pos[u] = -2; // settled
        size--;
        if (size > 0) { heap[0] = heap[size]; pos[heap[0]] = 0; down(0); }
        const du = dist[u];
        if (du > maxHours) break;
        const r = (u / W) | 0, c = u - r * W;
        const invU = inv[grid[u]];
        const sr = r * 16;
        for (let k = 0; k < 16; k++) {
            const r2 = r + MDR[k];
            if (r2 < 0 || r2 >= H) continue;
            let c2 = c + MDC[k];
            if (c2 < 0 || c2 >= W) { if (!wrap) continue; c2 = (c2 + W) % W; }
            const v = r2 * W + c2;
            if (pos[v] === -2) continue;
            const s = grid[v];
            if (!s) continue;
            let mean;
            if (k < 8) mean = (invU + inv[s]) * 0.5;
            else {
                let c0 = c + E0C[k], c1 = c + E1C[k];
                if (wrap) { c0 = (c0 + W) % W; c1 = (c1 + W) % W; }
                const e0 = grid[(r + E0R[k]) * W + c0], e1 = grid[(r + E1R[k]) * W + c1];
                if (!e0 || !e1) continue;
                mean = (invU + inv[s] + inv[e0] + inv[e1]) * 0.25;
            }
            const nd = du + stepKm[sr + k] * mean;
            if (nd < dist[v]) {
                dist[v] = nd;
                if (pos[v] === -1) { heap[size] = v; pos[v] = size; size++; up(size - 1); }
                else up(pos[v]);
            }
        }
    }
    return dist;
}
