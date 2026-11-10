// T-2991: one-way arrows. On a new game in New York, T-2981's train ring in Brooklyn (four stations, built with the
// edit API, its stops drawn anticlockwise) and a bus loop round a block of Manhattan (one way when new). Checks that
// the bus loop has arrows and the ring, run both ways, no chevrons; that switching the ring to One way gives it
// chevrons and the bus loop to Both ways takes its arrows away; and pictures, light and dark, near in
// artifacts/trainworld/t2991-*.png. From the maps root, on a Vite of your own:
//   node trainworld/checks/t2991-arrows.mjs [dev url]
// About a minute (no demand solve).
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6480/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const RC = [-73.945, 40.672], H = 1200;
const BUS = [[-73.9930, 40.7390], [-73.9770, 40.7390], [-73.9770, 40.7520], [-73.9930, 40.7520]];
const BC = [(BUS[0][0] + BUS[1][0]) / 2, (BUS[0][1] + BUS[2][1]) / 2];

const chrome = await launchChrome({ prefix: 'tw-t2991-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const settle = async (ms = 900) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
const shade = (s) => page.evaluate(async (s) => { (await import('/src/game/ui.ts')).shade.value = s; window.tw.overlay.invalidate(); }, s);
const counts = () => page.evaluate(() => ({ chevrons: window.tw.renderer.oneWay.size, arrows: window.tw.renderer.buses.arrows }));
const shot = async (file, c, zoom, clip) => {
  await page.evaluate(([c, z]) => { window.tw.map.jumpTo({ center: c, zoom: z }); window.tw.overlay.invalidate(); }, [c, zoom]);
  await settle(1200);
  await page.screenshot({ path: join(ARTIFACTS, file), ...(clip ? { clip } : {}) });
};
const CLIP = { x: 300, y: 150, width: 800, height: 600 };

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}${process.env.TW_EXTRA ?? ""}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.bus, undefined, { timeout: 180000 });

  const ring = await page.evaluate(async ([c, h]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const mids = [P(h, 0), P(0, h), P(-h, 0), P(0, -h)], corners = [P(h, h), P(-h, h), P(-h, -h), P(h, -h)];
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const nodes = [];
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const lv = -1, pi = (p) => [{ x: p.x, y: p.y, level: lv }];
    const res = [];
    res.push(await cl.edit({ op: 'route', from: { kind: 'free', x: mids[0].x, y: mids[0].y, level: lv }, to: { kind: 'free', x: mids[1].x, y: mids[1].y, level: lv }, pis: pi(corners[0]), single: false }));
    await pause(300);
    nodes.push(near(mids[0]), near(mids[1]));
    for (const k of [2, 3]) {
      res.push(await cl.edit({ op: 'route', from: { kind: 'node', node: nodes[k - 1] }, to: { kind: 'free', x: mids[k].x, y: mids[k].y, level: lv }, pis: pi(corners[k - 1]), single: false }));
      await pause(300);
      nodes.push(near(mids[k]));
    }
    res.push(await cl.edit({ op: 'route', from: { kind: 'node', node: nodes[3] }, to: { kind: 'node', node: nodes[0] }, pis: pi(corners[3]), single: false }));
    await pause(500);
    for (const n of nodes) { res.push(await cl.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: '' })); await pause(200); }
    res.push(await cl.edit({ op: 'constructAll' }));
    await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: [...nodes, nodes[0]], name: 'Ring', colour: '#0a7d3b' });
    await pause(500);
    return { line: line.line, ok: line.ok, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 80)) };
  }, [RC, H]);
  check(ring.ok && !ring.refused.length, 'the train ring is built', ring.refused.join(' | '));
  await edit({ op: 'setTrainCount', line: ring.line, trains: [8, 6, 4] });
  const bus = await page.evaluate(async (pts) => {
    const tw = window.tw;
    const route = [...pts, pts[0]].flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y, 1]; });
    const r = await tw.client.edit({ op: 'addBusLine', route, name: '', colour: '#c0392b' });
    const c = r.ok ? await tw.client.edit({ op: 'constructLine', line: r.line }) : null;
    return { line: r.line, ok: r.ok && c?.ok };
  }, BUS);
  check(bus.ok, 'the bus loop is added and constructed');
  await page.waitForTimeout(1500);

  await shot('t2991-bus-z15-light.png', BC, 15, CLIP);
  let n = await counts();
  check(n.arrows >= 40 && n.arrows <= 50 && n.chevrons === 0, 'a new bus loop has arrows every 125 m (about 45), the ring run both ways no chevrons', JSON.stringify(n));
  await shot('t2991-bus-z14-light.png', BC, 14, CLIP);

  await edit({ op: 'lineOneWay', line: ring.line, oneWay: true });
  await page.waitForTimeout(1500);
  await shot('t2991-ring-z14-light.png', RC, 14.2, CLIP);
  n = await counts();
  check(n.chevrons > 20, 'the ring run one way has chevrons', JSON.stringify(n));
  await shot('t2991-ring-z15-light.png', [RC[0] - 0.0141, RC[1]], 15.5, CLIP);
  await shot('t2991-ring-z13-light.png', RC, 13.2, CLIP);
  await shade('dark');
  await shot('t2991-ring-z14-dark.png', RC, 14.2, CLIP);
  await shot('t2991-bus-z15-dark.png', BC, 15, CLIP);
  await shot('t2991-bus-z14-dark.png', BC, 14, CLIP);
  await shade('light');

  await edit({ op: 'lineOneWay', line: bus.line, oneWay: false });
  await edit({ op: 'lineOneWay', line: ring.line, oneWay: false });
  await page.waitForTimeout(1500);
  await shot('t2991-bus-both-z15-light.png', BC, 15, CLIP);
  n = await counts();
  check(n.arrows === 0 && n.chevrons === 0, 'both ways, neither has marks', JSON.stringify(n));
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't2991-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2991 FAILED: ${fail}` : 'T-2991 PASSED');
process.exit(fail ? 1 : 0);
