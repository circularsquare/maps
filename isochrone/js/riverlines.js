// River lines for the page (T-045): the canoe's own river steps (data/canoe/rivers*, docs/canoe.md
// 4) drawn as thin lines, wider for more water. Whole world from L10 when zoomed out, else the
// L3 or L1 tiles round the view. Steps are joined into lines between confluences and smoothed
// once, so the cell grid does not show as a staircase. The files are fetched, parsed and joined in
// a worker of their own (js/riverworker.js, T-048); this side picks the tiles and draws.

// Look (T-052): faint, so the lines read as background and not as routes; the corridors draw the
// paddled stretches brightly.
const RIVER_COLOR = '#5b9bd5';
const RIVER_OPACITY = 0.3, RIVER_OPACITY_SHUT = 0.14;

const L10 = { W: 4320, deg: 1 / 12 };
// level by zoom, the smallest discharge drawn (m³/s) by zoom, and the most tiles a view may load
const TIER = z => z < 5 ? 10 : z < 7 ? 3 : 1;
const MIN_Q = z => z < 3 ? 1000 : z < 4 ? 300 : z < 6 ? 100 : z < 7 ? 30 : z < 8 ? 3 : 1;
const MAX_TILES = { 3: 40, 1: 20 };
// { level, minQ, lines, ms, workerMs } per rebuild, for checks: ms is the time on this thread
// (taking the message and handing the lines to the map), workerMs the worker's own
const riverLog = window.riverLog = [];

// Adds the source and layer (before layer `before`) and keeps them to the view while shown.
export function riverLines(map, base, before) {
    map.addSource('rivers', { type: 'geojson', data: { type: 'FeatureCollection', features: [] }, tolerance: 0.3 });
    map.addLayer({ id: 'rivers', type: 'line', source: 'rivers', layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
            'line-color': RIVER_COLOR,
            'line-opacity': ['case', ['==', ['get', 'shut'], 1], RIVER_OPACITY_SHUT, RIVER_OPACITY],
            'line-width': ['interpolate', ['linear'], ['zoom'],
                3, ['interpolate', ['linear'], ['get', 'lq'], 0, 0.4, 4, 1.6],
                9, ['interpolate', ['linear'], ['get', 'lq'], 0, 1, 4, 3.5]],
        } }, before);
    const worker = new Worker(new URL('./riverworker.js', import.meta.url), { type: 'module' });
    let on = false, shownKey = '', seq = 0;
    let index = null;
    const abs = u => new URL(u, document.baseURI).href;   // the worker lives in js/, the urls are the page's
    const pending = new Map();   // job id -> { level, minQ, key, t }
    const getIndex = () => index || (index = fetch(base + 'index.json').then(r => r.json()).then(ix => {
        for (const k in ix.rivers) ix.rivers[k].set = new Set(ix.rivers[k].tiles);
        return ix;
    }).catch(e => { index = null; throw e; }));

    worker.onmessage = ({ data }) => {
        const job = pending.get(data.id);
        pending.delete(data.id);
        if (!job) return;
        if (data.error) { console.warn('rivers:', data.error); return; }
        if (data.id !== seq || !on) return;   // a newer view has been asked for
        const t0 = performance.now();
        shownKey = job.key;
        map.getSource('rivers').setData({ type: 'FeatureCollection', features: data.features });
        riverLog.push({ level: job.level, minQ: job.minQ, lines: data.features.length,
                        ms: Math.round(performance.now() - t0), workerMs: data.ms });
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
        const id = ++seq;
        pending.set(id, { level: f, minQ, key });
        if (f === 10) {
            worker.postMessage({ id, urls: [abs(base + 'rivers.bin.gz')], row0s: [0], col0s: [0], Ts: [L10.W], deg: L10.deg, minQ });
        } else {
            worker.postMessage({ id, urls: want.map(w => abs(w[2])), row0s: want.map(w => w[0] * w[3]), col0s: want.map(w => w[1] * w[3]),
                Ts: want.map(w => w[3]), deg: want.length ? 5 / want[0][3] : 1, minQ });
        }
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
