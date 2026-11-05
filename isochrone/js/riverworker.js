// Module worker for the river lines (T-045, T-048): fetches and parses the canoe river files and joins
// their steps into smoothed lines, so none of it holds up the page. Started by js/riverlines.js.
// Message in:  { id, urls: [url, ...], row0s, col0s, T, deg, minQ } (the world file has T = L10 width)
// Message out: { id, features, ms } (ms is the time spent here)

const CLOSED = new Set([2, 3, 4, 5, 6]);   // unrunnable, fall, dam, weir, reservoir (canoedata.js CLS)

async function fetchGz(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(url + ' ' + r.status);
    return new Uint8Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
}

// uint32 n, cell uint32[n], len uint16[n], dir uint8[n], v uint8[n] (0.01 m/s), cls uint8[n]
function parse(b) {
    const n = new DataView(b.buffer, b.byteOffset, 4).getUint32(0, true);
    const cell = new Uint32Array(b.slice(4, 4 + 4 * n).buffer);
    const o = 4 + 6 * n;
    return { n, cell, dir: b.subarray(o, o + n), v: b.subarray(o + n, o + 2 * n), cls: b.subarray(o + 2 * n, o + 3 * n) };
}
const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
// discharge by stored speed (0.01 m/s): v = 0.51 Q^0.11
const Q_OF = Float64Array.from({ length: 256 }, (_, v) => Math.pow(Math.max(v, 1) / 51, 1 / 0.11));

// Steps of one or more parsed files as lines. Each part: { rv, row0, col0, T (cells per row of
// the file's cell numbering) }; cells are numbered globally as row * COLS + col on one grid of
// `deg` degrees from 85N, 180W.
function lines(parts, deg, minQ) {
    const COLS = Math.round(360 / deg);
    const up = [], dn = [], q = [], shut = [];
    for (const { rv, row0, col0, T } of parts) {
        for (let i = 0; i < rv.n; i++) {
            const qq = Q_OF[rv.v[i]];
            if (qq < minQ) continue;
            const d = DIRS[rv.dir[i]];
            if (!d) continue;
            const lr = Math.floor(rv.cell[i] / T), r = row0 + lr, c = col0 + rv.cell[i] - lr * T;
            up.push(r * COLS + c);
            dn.push((r + d[1]) * COLS + ((c + d[0]) % COLS + COLS) % COLS);
            q.push(qq);
            shut.push(CLOSED.has(rv.cls[i]) ? 1 : 0);
        }
    }
    const n = up.length;
    // width class and closedness must match along a line
    const wq = q.map(x => Math.min(8, Math.floor(Math.log10(x) * 2)));
    const outOf = new Map(), inCount = new Map(), inStep = new Map();
    for (let s = 0; s < n; s++) {
        const o = outOf.get(up[s]);
        if (o) o.push(s); else outOf.set(up[s], [s]);
        inCount.set(dn[s], (inCount.get(dn[s]) || 0) + 1);
        inStep.set(dn[s], s);
    }
    const same = (a, b) => wq[a] === wq[b] && shut[a] === shut[b];
    const pt = k => {
        const r = Math.floor(k / COLS), c = k - r * COLS;
        return [-180 + (c + 0.5) * deg, 85 - (r + 0.5) * deg];
    };
    const used = new Uint8Array(n), features = [];
    const follow = s0 => {
        const coords = [pt(up[s0])];
        let s = s0, lq = 0, len = 0;
        for (;;) {
            used[s] = 1; lq += Math.log10(q[s]); len++;
            const p = pt(dn[s]), last = coords[coords.length - 1];
            if (p[0] - last[0] > 180) p[0] -= 360; else if (last[0] - p[0] > 180) p[0] += 360;
            coords.push(p);
            if (inCount.get(dn[s]) !== 1) break;
            const next = (outOf.get(dn[s]) || []).find(t => !used[t] && same(s, t));
            if (next === undefined) break;
            s = next;
        }
        features.push({ type: 'Feature', properties: { lq: lq / len, shut: shut[s0] },
                        geometry: { type: 'LineString', coordinates: smooth(coords) } });
    };
    for (let s = 0; s < n; s++) {
        if (used[s]) continue;
        const k = up[s], inS = inStep.get(k);
        if (inCount.get(k) === 1 && !used[inS] && same(inS, s)) continue;   // inside a line: its start comes first
        follow(s);
    }
    for (let s = 0; s < n; s++) if (!used[s]) follow(s);   // loops
    return features;
}

// One Chaikin pass, ends kept. Coordinates to 1e-4 degree (~10 m, far under a cell), which keeps
// the json the map stringifies on the page's thread short.
const rd = v => Math.round(v * 1e4) / 1e4;
function smooth(c) {
    if (c.length < 3) return c.map(p => [rd(p[0]), rd(p[1])]);
    const out = [[rd(c[0][0]), rd(c[0][1])]];
    for (let i = 0; i < c.length - 1; i++) {
        const [x0, y0] = c[i], [x1, y1] = c[i + 1];
        if (i > 0) out.push([rd(0.75 * x0 + 0.25 * x1), rd(0.75 * y0 + 0.25 * y1)]);
        if (i < c.length - 2) out.push([rd(0.25 * x0 + 0.75 * x1), rd(0.25 * y0 + 0.75 * y1)]);
    }
    out.push([rd(c[c.length - 1][0]), rd(c[c.length - 1][1])]);
    return out;
}

// Parsed files by url, oldest first.
const tiles = new Map();
const getFile = url => {
    let p = tiles.get(url);
    if (p) { tiles.delete(url); tiles.set(url, p); return p; }
    p = fetchGz(url).then(parse);
    p.catch(() => tiles.delete(url));
    tiles.set(url, p);
    while (tiles.size > 120) tiles.delete(tiles.keys().next().value);
    return p;
};

self.onmessage = async ({ data: { id, urls, row0s, col0s, Ts, deg, minQ } }) => {
    try {
        const got = await Promise.all(urls.map(getFile));
        const t0 = performance.now();
        const parts = got.map((rv, i) => ({ rv, row0: row0s[i], col0: col0s[i], T: Ts[i] }));
        const features = lines(parts, deg, minQ);
        // (a blob url for MapLibre to fetch was tried: its json parse then landed on the page's thread, slower)
        self.postMessage({ id, features, ms: Math.round(performance.now() - t0) });
    } catch (e) {
        self.postMessage({ id, error: e.message });
    }
};
