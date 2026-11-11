// T-3157, T-3158: how lines look on a flyover's legs.
//  T-3157: in the Level view (line colour by height) a ramp's colour runs from its foot's level colour to its top's,
//  not a step: the track strokes carry their heights at both ends (`levelCode`), and the pixels drawn along the ramp
//  match the level colours blended at the height there.
//  T-3158: a line alone on single track is one width everywhere: the flyover's two legs (one on the line's path, one
//  its return run) carry the same width values (`Strokes.band`, widths), where the path leg had a band's 20% cut.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6201):
//   node trainworld/checks/t3157-look.mjs [dev url]
// About 1 minute. Pictures in artifacts/trainworld/t3157-check-*.png.
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6201/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3157-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);
const settle = async (ms = 800) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  // a double main, a station on it, a branch from there with the ramp switch on (two single legs and ramps), a line on each
  const junctions = await page.evaluate(async ([c]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, y) => ({ x: o.x + dx, y: o.y + y });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const w0 = P(-3000, 0), e0 = P(3000, 0), b0 = P(1800, 2200), s0 = P(-1200, 0);
    const before = new Set(tw.world.value.nodes.keys());
    await cl.edit({ op: 'route', from: { kind: 'free', ...w0, level: 0 }, to: { kind: 'free', ...e0, level: 0 }, pis: [], single: false });
    await pause(400);
    const main = tw.world.value.edges.find((e) => e.tracks === 2 && !before.has(e.a));
    await cl.edit({ op: 'addStation', at: { kind: 'edge', edge: main.id, ...s0 }, platform: 200, name: '' });
    await pause(300);
    const S = near(s0);
    await cl.edit({ op: 'route', from: { kind: 'node', node: S }, to: { kind: 'free', ...b0, level: 0 }, pis: [], single: false, ramp: true, design: 160 });
    await pause(500);
    const [W, E, B] = [near(w0), near(e0), near(b0)];
    for (const n of [W, E, B]) { await cl.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: '' }); await pause(200); }
    await cl.edit({ op: 'constructAll' });
    await pause(500);
    await cl.edit({ op: 'addLine', stops: [W, E], name: 'Main', colour: '#e6b800' });
    await pause(400);
    await cl.edit({ op: 'addLine', stops: [W, B], name: 'Branch', colour: '#f28c00' });
    await pause(800);
    const w = tw.world.value;
    return [...w.nodes.values()].filter((n) => !before.has(n.id) && n.ports >= 3 && n.platform === 0).map((n) => ({ lng: n.lng, lat: n.lat }));
  }, [RC]);
  check(junctions.length === 2, 'a flyover was laid', `${junctions.length} junctions`);
  await page.getByText(/Loading city data|Calculating demand/).first().waitFor({ state: 'hidden', timeout: 240000 }).catch(() => {});
  await page.waitForTimeout(1500);

  // T-3158: the strokes of lines on single track
  const widths = await page.evaluate(() => {
    const n = window.tw.renderer.net, single = new Set(window.tw.world.value.edges.filter((e) => e.tracks === 1).map((e) => e.id));
    const s = n.lines, seen = new Map();
    for (let i = 0; i < s.count; i++) {
      if (!single.has(s.edge[i])) continue;
      // the width values: its band's `1 - width share` at both ends, and the lane
      const key = [s.band ? s.band[i * 6 + 2] : 0, s.band ? s.band[i * 6 + 3] : 0, s.lane[i * 2], s.lane[i * 2 + 1]].join('/');
      const e = seen.get(s.edge[i]) ?? new Set();
      e.add(key);
      seen.set(s.edge[i], e);
    }
    return [...seen].map(([edge, keys]) => ({ edge, keys: [...keys] }));
  });
  check(widths.length >= 2, 'line strokes on both single legs', JSON.stringify(widths));
  const all = new Set(widths.flatMap((w) => w.keys));
  check(all.size === 1, 'both legs\' line strokes carry one width (and lane) value', [...all].join(' | '));

  // T-3157: the Level view
  const mid = [(junctions[0].lng + junctions[1].lng) / 2, (junctions[0].lat + junctions[1].lat) / 2];
  await page.evaluate(([c]) => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: c, zoom: 16.5 }); window.tw.overlay.invalidate(); }, [mid]);
  await settle(1500);
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).setDisplay('trackColour', 'height'); });
  await settle(1500);
  const shot = join(ARTIFACTS, 't3157-check-level.png');
  const png = await page.screenshot({ path: shot });
  const r = await page.evaluate(async (b64) => {
    const geo = await import('/src/map/geo.ts');
    const tw = window.tw, T = tw.renderer.net.track, map = tw.map;
    const rect = document.querySelector('#map').getBoundingClientRect();
    const img = new Image();
    await new Promise((res) => { img.onload = res; img.src = 'data:image/png;base64,' + b64; });
    const cv = document.createElement('canvas'); cv.width = img.width; cv.height = img.height;
    const cx = cv.getContext('2d', { willReadFrequently: true }); cx.drawImage(img, 0, 0);
    const lc = tw.renderer.levelColours;
    const colourAt = (h) => { const lh = Math.max(0, Math.min(6, 3 - h)), lo = Math.floor(lh), f = lh - lo, hi = Math.min(6, lo + 1); return [0, 1, 2].map((k) => 255 * (lc[lo * 3 + k] + (lc[hi * 3 + k] - lc[lo * 3 + k]) * f)); };
    const out = { ramp: 0, heights: [], match: 0, rows: [], whole: colourAt(0).concat(colourAt(1)) };
    for (let i = 0; i < T.count; i++) {
      const q0 = Math.floor(T.level[i] / 256), q1 = T.level[i] - q0 * 256;
      if (q0 === q1) continue;
      const h = (q0 + q1) / 2 / 32 - 4;
      const x = (T.seg[i * 4] + T.seg[i * 4 + 2]) / 2, y = (T.seg[i * 4 + 1] + T.seg[i * 4 + 3]) / 2;
      const p = map.project(geo.toLngLat(x, y));
      const px = Math.round(p.x + rect.left), py = Math.round(p.y + rect.top);
      if (px < 0 || py < 0 || px >= img.width || py >= img.height) continue;
      const d = cx.getImageData(px, py, 1, 1).data;
      const want = colourAt(h), ok = want.every((v, k) => Math.abs(v - d[k]) < 40);
      out.ramp++;
      out.heights.push(Math.round(h * 100) / 100);
      if (ok) out.match++;
      out.rows.push([Math.round(h * 100) / 100, [...d].slice(0, 3).join(','), want.map(Math.round).join(','), ok ? 1 : 0].join(' '));
    }
    return out;
  }, png.toString('base64'));
  const distinct = new Set(r.heights).size;
  check(r.ramp >= 6, 'ramp strokes carry a start and end height that differ', `${r.ramp} segments`);
  check(distinct >= 4, 'the heights along the ramp pass through several values between two levels', `${distinct} distinct`);
  check(r.ramp > 0 && r.match >= 0.8 * r.ramp, 'pixels drawn along the ramp match the colours blended at their height', `${r.match} of ${r.ramp}; whole levels ${r.whole.map(Math.round).join(',')}`);
  if (r.match < 0.8 * r.ramp) console.log(r.rows.slice(0, 20).join('\n'));
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).setDisplay('trackColour', 'line'); });
  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3157 FAILED: ${fail}` : 'T-3157 PASSED');
process.exit(fail ? 1 : 0);
