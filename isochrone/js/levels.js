// A level is a block of cells with its own grid and, once solved, its dist. L10 is the whole
// world and wraps at 180 degrees; a window of L3 or L1 does not.

const KM_PER_DEG = 111.195;

// A level: a block of cells cellDeg wide, its top-left corner at (west, north).
export function makeLevel(grid, W, H, west, north, cellDeg, wrap) {
    const dxKm = new Float32Array(H);
    for (let r = 0; r < H; r++) {
        const lat = north - (r + 0.5) * cellDeg;
        dxKm[r] = cellDeg * KM_PER_DEG * Math.cos(lat * Math.PI / 180);
    }
    return { grid, W, H, west, north, south: north - H * cellDeg, east: west + W * cellDeg,
             cellDeg, wrap, dxKm, dyKm: cellDeg * KM_PER_DEG, dist: null };
}

// Nearest land cell to a point within R cells, or -1.
export function snap(L, lng, lat, R) {
    const { grid, W, H, wrap, cellDeg } = L;
    let dc0 = (lng - L.west) / cellDeg;
    if (wrap) dc0 = ((dc0 % W) + W) % W;
    const col = Math.floor(dc0), row = Math.floor((L.north - lat) / cellDeg);
    let best = -1, bestD = Infinity;
    for (let dr = -R; dr <= R; dr++) {
        const r = row + dr;
        if (r < 0 || r >= H) continue;
        for (let dc = -R; dc <= R; dc++) {
            let c = col + dc;
            if (wrap) c = (c + W) % W; else if (c < 0 || c >= W) continue;
            if (!grid[r * W + c]) continue;
            const d = dc * dc + dr * dr;
            if (d < bestD) { bestD = d; best = r * W + c; }
        }
    }
    return best;
}

// Time at a point from a solved level, blending the reached cells round it.
// undefined when the point is outside the level, Infinity when nothing near is reached.
export function sample(L, lng, lat) {
    const { W, H, cellDeg, wrap, dist } = L;
    let x = (lng - L.west) / cellDeg;
    if (wrap) x = ((x % W) + W) % W;
    else {
        x = (((lng - L.west) % 360) + 360) % 360 / cellDeg;
        if (x < 0.5 || x > W - 0.5) return undefined;
    }
    let y = (L.north - lat) / cellDeg;
    if (y < 0.5 || y > H - 0.5) { if (!wrap) return undefined; }
    x -= 0.5; y -= 0.5;
    const c0 = Math.floor(x), r0 = Math.floor(y), fx = x - c0, fy = y - r0;
    let sum = 0, wsum = 0;
    for (let j = 0; j < 2; j++) {
        const r = Math.min(H - 1, Math.max(0, r0 + j));
        const wy = j ? fy : 1 - fy;
        for (let i = 0; i < 2; i++) {
            let c = c0 + i;
            if (wrap) c = (c + W) % W; else c = Math.min(W - 1, Math.max(0, c));
            const d = dist[r * W + c];
            if (d === Infinity) continue;
            const w = wy * (i ? fx : 1 - fx) + 1e-6;
            sum += d * w; wsum += w;
        }
    }
    return wsum ? sum / wsum : Infinity;
}
