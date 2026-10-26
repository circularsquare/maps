// River lines for the page (T-045): the canoe's own river steps (data/canoe/rivers*, docs/canoe.md
// 4) drawn as thin lines, wider for more water. Whole world from L10 when zoomed out, else the
// L3 or L1 tiles round the view. Steps are joined into lines between confluences and smoothed
// once, so the cell grid does not show as a staircase.

const L10 = { W: 4320, deg: 1 / 12 };
// level by zoom, the smallest discharge drawn (m³/s) by zoom, and the most tiles a view may load
const TIER = z => z < 5 ? 10 : z < 7 ? 3 : 1;
const MIN_Q = z => z < 3 ? 1000 : z < 4 ? 300 : z < 6 ? 100 : z < 7 ? 30 : z < 8 ? 3 : 1;
const MAX_TILES = { 3: 40, 1: 20 };
const CLOSED = new Set([2, 3, 4, 5, 6]);   // unrunnable, fall, dam, weir, reservoir (canoedata.js CLS)
const riverLog = window.riverLog = [];     // { level, minQ, lines, ms } per rebuild, for checks

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

// One Chaikin pass, ends kept.
function smooth(c) {
    if (c.length < 3) return c;
    const out = [c[0]];
    for (let i = 0; i < c.length - 1; i++) {
        const [x0, y0] = c[i], [x1, y1] = c[i + 1];
        if (i > 0) out.push([0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1]);
        if (i < c.length - 2) out.push([0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1]);
    }
    out.push(c[c.length - 1]);
    return out;
}

// Adds the source and layer (before layer `before`) and keeps them to the view while shown.
export function riverLines(map, base, before) {
    map.addSource('rivers', { type: 'geojson', data: { type: 'FeatureCollection', features: [] }, tolerance: 0.3 });
    map.addLayer({ id: 'rivers', type: 'line', source: 'rivers', layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
            'line-color': '#5b9bd5',
            'line-opacity': ['case', ['==', ['get', 'shut'], 1], 0.3, 0.65],
            'line-width': ['interpolate', ['linear'], ['zoom'],
                3, ['interpolate', ['linear'], ['get', 'lq'], 0, 0.4, 4, 1.6],
                9, ['interpolate', ['linear'], ['get', 'lq'], 0, 1, 4, 3.5]],
        } }, before);
    let on = false, shownKey = '', seq = 0;
    let index = null, world = null;
    const tiles = new Map();   // url -> Promise of parsed steps, oldest first
    const getIndex = () => index || (index = fetch(base + 'index.json').then(r => r.json()).then(ix => {
        for (const k in ix.rivers) ix.rivers[k].set = new Set(ix.rivers[k].tiles);
        return ix;
    }).catch(e => { index = null; throw e; }));
    const getWorld = () => world || (world = fetchGz(base + 'rivers.bin.gz').then(parse).catch(e => { world = null; throw e; }));
    const getTile = url => {
        let p = tiles.get(url);
        if (p) { tiles.delete(url); tiles.set(url, p); return p; }
        p = fetchGz(url).then(parse);
        p.catch(() => tiles.delete(url));
        tiles.set(url, p);
        while (tiles.size > 120) tiles.delete(tiles.keys().next().value);
        return p;
    };

    async function update() {
        if (!on) return;
        const z = map.getZoom(), minQ = MIN_Q(z), b = map.getBounds();
        let f = TIER(z), want = null;
        if (f !== 10) {
            const ix = await getIndex(), rl = ix.rivers[f], TX = ix.tilesX;
            // the view and half of it again round it, in whole tiles
            const pw = (b.getEast() - b.getWest()) / 2, ph = (b.getNorth() - b.getSouth()) / 2;
            const r0 = Math.max(0, Math.floor((85 - b.getNorth() - ph) / 5)), r1 = Math.min(28, Math.floor((85 - b.getSouth() + ph) / 5));
            const c0 = Math.floor((b.getWest() - pw + 180) / 5), c1 = Math.floor((b.getEast() + pw + 180) / 5);
            want = [];
            for (let tr = r0; tr <= r1; tr++) for (let tc = c0; tc <= c1 && tc - c0 < TX; tc++) {
                const tcw = ((tc % TX) + TX) % TX;
                if (rl.set.has(tr * TX + tcw)) want.push([tr, tcw, base + rl.path + '/' + tr + '_' + tcw + '.bin.gz', rl.tile]);
            }
            if (want.length > MAX_TILES[f]) { f = 10; want = null; }
        }
        const key = f + '|' + minQ + '|' + (want ? want.map(w => w[2]).join() : '');
        if (key === shownKey) return;
        const my = ++seq;
        let parts, deg;
        if (f === 10) {
            parts = [{ rv: await getWorld(), row0: 0, col0: 0, T: L10.W }];
            deg = L10.deg;
        } else {
            const got = await Promise.all(want.map(w => getTile(w[2])));
            parts = want.map(([tr, tcw, , T], i) => ({ rv: got[i], row0: tr * T, col0: tcw * T, T }));
            deg = want.length ? 5 / want[0][3] : 1;
        }
        if (my !== seq || !on) return;
        shownKey = key;
        const t0 = performance.now(), features = lines(parts, deg, minQ);
        map.getSource('rivers').setData({ type: 'FeatureCollection', features });
        riverLog.push({ level: f, minQ, lines: features.length, ms: Math.round(performance.now() - t0) });
    }
    const go = () => update().catch(e => console.warn('rivers:', e.message));
    map.on('moveend', go);
    return {
        set(v) {
            on = v;
            map.setLayoutProperty('rivers', 'visibility', on ? 'visible' : 'none');
            if (on) go();
        },
    };
}
