// T-3134: the placement ghost draws each segment with its own track count. With Single track picked the ghost
// carries lane 0 (one track close in), with double and quad 255; the sim's preview answer says it per segment
// (`PreviewAnswer.lanes`): single, double and quad routes, a flyover's plain leg and crossing leg (single) on a
// double route, a reshape and a node drag (`partLanes`).
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6191):
//   node trainworld/checks/t3134-single-ghost.mjs [dev url]
// About 1 minute. Pictures in artifacts/trainworld/t3134-check-*.png.
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6191/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3134-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3134-check-${name}.png`) });

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const box = await (await page.$('#map')).boundingBox();
  await page.evaluate(([c]) => window.tw.map.jumpTo({ center: c, zoom: 16 }), [RC]);
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(1500);

  // 1. the drawing tool's ghost, single then double then quad, by the mouse
  const mx = box.x + box.width / 2 - 250, my = box.y + box.height / 2;
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'track'; ui.buildLevel.value = 0; ui.keepPlacing.value = false;
    ui.quadTrack.value = false; ui.singleTrack.value = true;
  });
  await page.mouse.move(mx, my, { steps: 3 });
  await wait(300);
  await page.mouse.click(mx, my);
  await wait(300);
  await page.mouse.move(mx + 300, my - 60, { steps: 6 });
  const ghostLanes = () => page.evaluate(() => { const s = window.tw.renderer.previewSrc; return s ? { n: s.count, lanes: [...new Set(s.lane ?? [])] } : null; });
  const settleGhost = async (want) => {
    const t0 = Date.now();
    while (Date.now() - t0 < 15000) {
      const g = await ghostLanes();
      if (g && g.n > 0 && g.lanes.length === 1 && g.lanes[0] === want) return g;
      await wait(150);
    }
    return ghostLanes();
  };
  let g = await settleGhost(0);
  check(g?.n > 0 && g.lanes.length === 1 && g.lanes[0] === 0, 'Single track: the ghost is drawn single (lane 0)', JSON.stringify(g));
  await shot('single');
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).singleTrack.value = false; });
  g = await settleGhost(255);
  check(g?.lanes?.length === 1 && g.lanes[0] === 255, 'Double track: the ghost is drawn double (lane 255)', JSON.stringify(g));
  await shot('double');
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).quadTrack.value = true; });
  g = await settleGhost(255);
  check(g?.lanes?.length === 1 && g.lanes[0] === 255, 'Quad track: the ghost is drawn double pairs (lane 255)', JSON.stringify(g));
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.quadTrack.value = false; ui.tool.value = 'select'; });
  await page.keyboard.press('Escape');

  // 2. the sim's answer per segment, asked straight: a double main, then routes off it
  const r = await page.evaluate(async ([c]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, y) => ({ x: o.x + dx, y: o.y + 3000 + y, level: 0 });
    const ask = (req) => new Promise((res) => cl.preview(req, res));
    const sum = (a) => ({ ok: a.ok, issues: a.issues, n: a.speeds?.length, lanes: a.lanes ? [...a.lanes] : null, speeds: a.speeds ? [...a.speeds] : null });
    const out = {};
    await cl.edit({ op: 'route', from: { kind: 'free', ...P(-3000, 0) }, to: { kind: 'free', ...P(5000, 0) }, pis: [], single: false });
    await new Promise((d) => setTimeout(d, 500));
    const main = tw.world.value.edges.find((e) => e.tracks === 2);
    const to = { kind: 'free', ...P(3500, 1500) };
    const from = { kind: 'edge', edge: main.id, ...P(1000, 0) };
    for (const [name, single, quad] of [['single', true, false], ['double', false, false], ['quad', false, true]]) {
      out[name] = sum(await ask({ from: { kind: 'free', ...P(-2500, -2500) }, to: { kind: 'free', ...P(-500, -1800) }, pis: [], single, quad, through: true }));
    }
    out.flyover = sum(await ask({ from, to, pis: [], single: false, ramp: true, through: true }));
    out.flat = sum(await ask({ from, to, pis: [], single: false, through: true }));
    out.singleBranch = sum(await ask({ from, to, pis: [], single: true, ramp: true, through: true }));
    // a blueprint single edge, reshaped and dragged
    await cl.edit({ op: 'route', from: { kind: 'free', ...P(-2000, -3500) }, to: { kind: 'free', ...P(0, -3500) }, pis: [], single: true });
    await new Promise((d) => setTimeout(d, 500));
    const se = tw.world.value.edges.find((e) => e.tracks === 1 && !e.built);
    out.reshape = se ? sum(await new Promise((res) => cl.previewEdge(se.id, [{ x: o.x - 1000, y: o.y - 200, level: 0 }], res))) : null;
    out.drag = se ? await new Promise((res) => cl.previewNode(se.a, o.x - 2000, o.y - 300, (a) => res({ ok: a.ok, issues: a.issues, partLanes: a.partLanes ? [...a.partLanes] : null, parts: a.parts ? [...a.parts] : null }))) : null;
    return out;
  }, [RC]);
  const only = (s, v) => s?.ok && s.lanes?.length === s.n && s.n > 0 && s.lanes.every((l, i) => s.speeds[i] < 0 || l === v);
  check(only(r.single, 0), 'a single route: every segment single', JSON.stringify(r.single?.lanes?.slice(0, 6)));
  check(only(r.double, 255), 'a double route: every segment double', JSON.stringify(r.double?.lanes?.slice(0, 6)));
  check(only(r.quad, 255), 'a quad route: every segment double pairs', JSON.stringify(r.quad?.lanes?.slice(0, 6)));
  const f = r.flyover;
  const gap = f?.speeds?.findIndex((v) => v < 0) ?? -1;
  check(f?.ok && gap > 0 && f.lanes.length === f.n, 'a flyover route has a crossing leg after a step', `gap at ${gap} of ${f?.n}`);
  check(gap > 0 && f.lanes.slice(gap + 1).every((l) => l === 0), 'the crossing leg is single');
  const main = gap > 0 ? f.lanes.slice(0, gap) : [];
  check(main.includes(0) && main.includes(255), 'the route is single from the junction to J3, double after', `${main.filter((l) => l === 0).length} single of ${main.length} segments`);
  check(r.flat?.ok && r.flat.lanes?.every((l) => l === 255), 'flat (ramps off): the whole route is double');
  check(r.singleBranch?.ok && r.singleBranch.lanes?.every((l) => l === 0), 'a single branch off double track with ramps on: all single (no flyover)');
  check(only(r.reshape, 0), 'a single blueprint edge reshaped: single', JSON.stringify(r.reshape?.issues));
  check(r.drag?.ok && r.drag.partLanes?.length === r.drag.parts.length - 1 && r.drag.partLanes.every((l) => l === 0), 'a node drag on single track: each part single', JSON.stringify(r.drag));
  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3134 FAILED: ${fail}` : 'T-3134 PASSED');
process.exit(fail ? 1 : 0);
