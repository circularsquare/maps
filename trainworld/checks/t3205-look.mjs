// T-3171 look: pictures of a hand-laid single track leaving the upper track of a double main 40 m before its end, zoomed in (no asserts; judge the curve leaves its track cleanly). t3191-z20.png, t3191-z18.png. Vite of your own on 6112.
// T-3162: with single track picked, each track of double track drawn as two tracks is a branch point across its
// whole drawn width, at any zoom: a pointer anywhere over the outer track (and over the inner one) starts a branch
// from that track (`Snap.join`), the pointer just past the drawn pair does not. T-3163: the branch dot is 15 px.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6112):
//   node trainworld/checks/t3162-branch-hit.mjs [dev url]
// Pictures in artifacts/trainworld/t3162-*.png and t3163-*.png.
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6112/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3162-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `${name}.png`) });
const settle = async (ms = 600) => {
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(ms);
};

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(([c]) => {
    const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + 3000 + dy });
    const ll = (p) => { const q = tw.packToLngLat(p.x, p.y); return Array.isArray(q) ? q : [q.lng, q.lat]; };
    window.__j = { P, ll };
  }, [RC]);
  await page.evaluate(async () => {
    const { P } = window.__j, cl = window.tw.client;
    await cl.edit({ op: 'route', from: { kind: 'free', ...P(-3000, 0), level: 0 }, to: { kind: 'free', ...P(3000, 0), level: 0 }, pis: [], single: false });
  });
  await wait(800);
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'track'; ui.buildLevel.value = 0; ui.keepPlacing.value = false; ui.quadTrack.value = false;
    ui.singleTrack.value = true; ui.tramTrack.value = false; ui.highSpeed.value = false; ui.designSpeed.value = 60;
  });
  const box = await (await page.$('#map')).boundingBox();
  const lanes = () => page.evaluate(async () => (({ gap, t, share }) => ({ gap, t, share }))((await import('/src/map/renderer.ts')).laneState(window.tw.map.getZoom())));
  const hoverAt = async (x, y) => {
    await page.mouse.move(x - 25, y, { steps: 2 });
    await page.mouse.move(x, y, { steps: 2 });
    await wait(250);
    return page.evaluate(() => {
      const h = window.tw.tools.hover;
      if (!h) return null;
      const p = h.join ? window.tw.map.project(window.tw.toLngLat(h.join[0], h.join[1])) : null;
      const c = window.tw.map.project(window.tw.toLngLat(h.lx, h.ly));
      return { kind: h.kind, side: p ? (p.y < c.y ? 'up' : 'down') : null };
    });
  };
  // T-3205: a single main; a double track drawn from open ground onto it, the pointer on the single and off it
  await page.evaluate(async () => {
    const { P } = window.__j, cl = window.tw.client, tw = window.tw;
    await cl.edit({ op: 'removeEdge', edge: tw.world.value.edges.find((x) => x.tracks === 2).id });
    await new Promise((r) => setTimeout(r, 600));
    await cl.edit({ op: 'route', from: { kind: 'free', ...P(-3000, 0), level: 0 }, to: { kind: 'free', ...P(3000, 0), level: 0 }, pis: [], single: true });
    await new Promise((r) => setTimeout(r, 800));
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'track'; ui.buildLevel.value = 0; ui.keepPlacing.value = false; ui.quadTrack.value = false;
    ui.singleTrack.value = false; ui.tramTrack.value = false; ui.highSpeed.value = false; ui.designSpeed.value = 60;
  });
  const proj = (p) => page.evaluate((p) => { const q = window.tw.map.project(window.__j.ll(window.__j.P(p[0], p[1]))); return [q.x, q.y]; }, p).then(([x, y]) => [box.x + x, box.y + y]);
  const jump = async (p, z) => { await page.evaluate(([p, z]) => window.tw.map.jumpTo({ center: window.__j.ll(window.__j.P(p[0], p[1])), zoom: z }), [p, z]); await settle(); };
  // the start: open ground south of the single, so the double runs up to it
  await jump([0, -150], 16);
  const [sx, sy] = await proj([0, -250]);
  await page.mouse.move(sx - 20, sy, { steps: 2 });
  await page.mouse.move(sx, sy, { steps: 2 });
  await wait(300);
  await page.mouse.click(sx, sy);
  await wait(300);
  const lane = () => page.evaluate(() => { const a = window.tw.tools.lastPreview; return a ? { ok: a.ok, lanes: a.lanes ? [...a.lanes].slice(0, 3).concat([...a.lanes].slice(-3)) : null } : null; });
  const [ex, ey] = await proj([0, 0]);
  for (const [name, dy] of [['on', 0], ['off', 40]]) {
    await page.mouse.move(ex - 10, ey + dy - 10, { steps: 2 });
    await page.mouse.move(ex, ey + dy, { steps: 2 });
    await wait(1500);
    console.log(name, JSON.stringify(await lane()), 'hover', await page.evaluate(() => { const h = window.tw.tools.hover; return h ? h.kind : null; }));
    await jump([0, 15], 19);
    await shot('t3205-' + name + '-z18');
    await jump([0, -150], 16);
    await page.mouse.move(ex - 10, ey + dy - 10, { steps: 2 });
    await page.mouse.move(ex, ey + dy, { steps: 2 });
    await wait(800);
  }
} catch (e) {
  await shot('t3191-died');
  console.log(e.stack);
} finally {
  await chrome.close();
}
process.exit(0);
