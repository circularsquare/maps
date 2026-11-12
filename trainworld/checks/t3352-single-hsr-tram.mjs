// T-3352: single track of high-speed and tram type. On a new game in New York, headless Chrome:
//  A the track bar: Single stays picked through Regular, High speed and Tram (shot per type, light and dark);
//  B a single high-speed run and a single tram run drawn through the track model's edit (blueprint): every piece
//    is one track and keeps its type; One way sets on the high-speed piece (the Track inspector's switch shows
//    for it) and on the tram piece;
//  C the inspector of the single high-speed piece, shot.
// Screenshots artifacts/trainworld/t3352-*.png. From the maps root, on a Vite of your own:
//   node trainworld/checks/t3352-single-hsr-tram.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6197/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const C = [-73.93, 40.668];

const chrome = await launchChrome({ prefix: 'tw-t3352-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { if (!sessionStorage.getItem('t3352')) { sessionStorage.setItem('t3352', '1'); localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { tram: false, roads: false, level: 1, keep: true, tracks: 2 } })); } } catch {} });
const errors = [];
collectErrors(page, errors);
const ready = () => page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
const ui = async (f) => { await page.evaluate(async () => { window.__ui ??= await import('/src/game/ui.ts'); }); return page.evaluate(`(${f.toString()})(window.__ui)`); };
const state = () => ui((u) => ({ type: u.trackType.value, single: u.singleTrack.value, quad: u.quadTrack.value }));
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3352-${name}.png`) });
const barShot = async (name) => {
  const b = await page.locator('.track-bar').boundingBox();
  await page.screenshot({ path: join(ARTIFACTS, `t3352-${name}.png`), clip: { x: Math.max(0, b.x - 16), y: Math.max(0, b.y - 12), width: b.width + 32, height: b.height + 24 } });
};
const draw = (xs, dy, opts) => page.evaluate(async ([c, xs, dy, opts]) => {
  const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
  const k = 1 / Math.cos((c[1] * Math.PI) / 180);
  const P = (dx) => ({ x: o.x + dx * k, y: o.y + dy * k });
  const r = await cl.edit({ op: 'route', from: { kind: 'free', ...P(xs[0]), level: 0 }, to: { kind: 'free', ...P(xs[1]), level: 0 }, pis: [], single: true, ...opts });
  await new Promise((d) => setTimeout(d, 600));
  return { ok: r.ok, issues: JSON.stringify(r.issues ?? '').slice(0, 200) };
}, [C, xs, dy, opts]);
try {
  await page.goto(`${BASE}?debug=1&new=1&city=nyc&paused=1&demandWorkers=1&v=${Date.now()}`);
  await ready();
  await ui((u) => { u.tool.value = 'track'; });
  await wait(800);
  const bar = page.locator('.track-bar');
  const btn = (label) => bar.locator(`button[aria-label="${label}"]`);
  const type = (name) => bar.locator('.tb-wide button', { hasText: new RegExp(`^${name}$`) });
  await btn('Single track').click();
  for (const [name, want] of [['High speed', 'hsr'], ['Tram', 'tram'], ['Regular', 2]]) {
    await type(name).click();
    const s = await state();
    check(s.type === want && s.single, `A: ${name} with Single picked`, JSON.stringify(s));
    if (name !== 'Regular') await barShot(`bar-single-${want}`);
  }
  await ui((u) => { u.shade.value = 'dark'; u.setTrackType('hsr'); });
  await wait(400);
  await barShot('bar-single-hsr-dark');
  await ui((u) => { u.shade.value = 'light'; u.tool.value = 'select'; });

  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 13.5 }), C);
  const h = await draw([-3000, 0], 400, { hsr: true });
  check(h.ok, 'B: single high-speed route drawn', h.issues);
  const t = await draw([-3000, 0], -400, { tram: true });
  check(t.ok, 'B: single tram route drawn', t.issues);
  const edges = await page.evaluate(() => [...window.tw.world.value.edges].map((e) => ({ id: Number(e.id), tracks: e.tracks, hsr: !!e.hsr, kind: e.kind ?? 0 })));
  const hs = edges.filter((e) => e.hsr), tr = edges.filter((e) => e.kind);
  check(hs.length > 0 && hs.every((e) => e.tracks === 1), 'B: every high-speed piece is single', JSON.stringify(hs));
  check(tr.length > 0 && tr.every((e) => e.tracks === 1), 'B: every tram piece is single', JSON.stringify(tr));
  for (const [what, e] of [['high-speed', hs[0]], ['tram', tr[0]]]) {
    const way = await page.evaluate(async (id) => {
      const { setOneWay } = await import('/src/game/oneWay.ts');
      await setOneWay(id, 1);
      await new Promise((d) => setTimeout(d, 600));
      return window.tw.world.value.edges.find((x) => Number(x.id) === id)?.oneWay;
    }, e.id);
    check(!!way, `B: One way sets on single ${what} track`, String(way));
  }
  await shot('map');
  await page.evaluate(async (id) => { const u = await import('/src/game/ui.ts'); u.tool.value = 'select'; u.selection.value = { kind: 'track', edge: id }; }, hs[0].id);
  await wait(800);
  const sw = await page.locator('text=One way').count();
  check(sw > 0, 'C: the Track inspector shows One way for single high-speed track', String(sw));
  await shot('inspector');
} catch (e) {
  fail++;
  console.log('FAIL exception', e?.stack ?? e);
}
const bad = errors.filter((e) => !/favicon|AJAX|openfreemap/.test(e));
check(bad.length === 0, 'no page errors', bad.slice(0, 3).join(' | '));
await chrome.close();
console.log(fail ? `${fail} FAILED` : 'ALL PASS');
process.exit(fail ? 1 : 0);
