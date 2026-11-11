// T-3164: a badge where one-way single track no train can use meets. Two single pieces W and E meet at a plain
// joint B on a new game: pointing into each other shows the converging badge, pointing away the diverging one, one
// way through none, and a station at B none. Hover tags read on the map. Pictures in artifacts/trainworld/t3164-*.png.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6114):
//   node trainworld/checks/t3164-unusable-oneway.mjs [dev url]
// About a minute.
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6114/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];
// markData.ts
const M_CONVERGE = 5, M_DIVERGE = 6;

const chrome = await launchChrome({ prefix: 'tw-t3164-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3164-${name}.png`) });

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(([c]) => {
    const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + 3000 + dy });
    const ll = (p) => { const q = tw.packToLngLat(p.x, p.y); return Array.isArray(q) ? q : [q.lng, q.lat]; };
    const node = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    window.__u = { P, ll, node };
  }, [RC]);

  const net = await page.evaluate(async () => {
    const tw = window.tw, cl = tw.client, { P, node } = window.__u;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const res = [];
    res.push(await cl.edit({ op: 'route', from: { kind: 'free', ...P(-1500, 0), level: 0 }, to: { kind: 'free', ...P(0, 0), level: 0 }, pis: [], single: true, design: 60 }));
    await pause(400);
    const B = node(P(0, 0)), before = new Set(tw.world.value.edges.map((e) => e.id));
    res.push(await cl.edit({ op: 'route', from: { kind: 'node', node: B }, to: { kind: 'free', ...P(1500, 0), level: 0 }, pis: [], single: true, design: 60 }));
    await pause(400);
    res.push(await cl.edit({ op: 'constructAll' }));
    await pause(800);
    const w = tw.world.value, singles = w.edges.filter((e) => e.tracks === 1);
    const E = singles.find((e) => !before.has(e.id)), W = singles.find((e) => before.has(e.id));
    return { refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues)), B, W: W?.id, E: E?.id, wInto: W && (W.b === B ? 1 : 2), eInto: E && (E.b === B ? 1 : 2), pieces: singles.length };
  });
  check(!net.refused.length && net.pieces === 2 && net.W !== undefined && net.E !== undefined, 'two single pieces meet at a plain joint', JSON.stringify(net));

  const setWay = (edge, dir) => page.evaluate(async ([edge, dir]) => { const r = await window.tw.client.edit({ op: 'oneWay', edge, dir }); await new Promise((d) => setTimeout(d, 700)); return r.ok; }, [edge, dir]);
  // the node's flag and the marks drawn there
  const at = () => page.evaluate((B) => {
    const tw = window.tw, n = tw.client.lastNodes.find((x) => x.id === B), m = tw.renderer.net?.marks ?? [];
    const shapes = [];
    for (let k = 0; k < m.length; k += 6) if (m[k + 2] >= 5) shapes.push(m[k + 2]);
    return { unusable: n?.unusable ?? 0, shapes };
  }, net.B);
  const settle = async () => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {}); await wait(700); };
  await page.waitForFunction(() => !document.body.innerText.includes('Loading city data'), undefined, { timeout: 120000 }).catch(() => {});
  await page.evaluate(() => window.tw.map.jumpTo({ center: window.__u.ll(window.__u.P(0, 0)), zoom: 16.5 }));
  await settle();
  const box = await (await page.$('#map')).boundingBox();
  const tag = async () => {
    const [x, y] = await page.evaluate(() => { const q = window.tw.map.project(window.__u.ll(window.__u.P(0, 0))); return [q.x, q.y]; });
    await page.mouse.move(box.x + x - 25, box.y + y - 25, { steps: 2 });
    await page.mouse.move(box.x + x, box.y + y, { steps: 3 });
    await wait(1000);
    return page.evaluate(() => document.body.innerText);
  };

  check(await setWay(net.W, net.wInto), 'W set one way into B');
  let s = await at();
  check(s.unusable === 0 && !s.shapes.length, 'one way into a two-way piece: no badge', JSON.stringify(s));
  check(await setWay(net.E, 3 - net.eInto), 'E set one way out of B');
  s = await at();
  check(s.unusable === 0 && !s.shapes.length, 'one way straight through: no badge', JSON.stringify(s));

  check(await setWay(net.E, net.eInto), 'E set one way into B');
  s = await at();
  check(s.unusable === 1 && s.shapes.length === 1 && s.shapes[0] === M_CONVERGE, 'pointing into each other: the converging badge', JSON.stringify(s));
  let text = await tag();
  check(text.includes('Unusable converging one-way track'), 'its tag');
  await shot('converging');
  await page.mouse.move(box.x + 5, box.y + 300);

  check(await setWay(net.W, 3 - net.wInto) && await setWay(net.E, 3 - net.eInto), 'both set one way out of B');
  s = await at();
  check(s.unusable === 2 && s.shapes.length === 1 && s.shapes[0] === M_DIVERGE, 'pointing away from each other: the diverging badge', JSON.stringify(s));
  text = await tag();
  check(text.includes('Unusable diverging one-way track'), 'its tag');
  await shot('diverging');
  await page.mouse.move(box.x + 5, box.y + 300);

  // a station at B says nothing: a line may end there
  await page.evaluate(async (B) => { await window.tw.client.edit({ op: 'addStation', at: { kind: 'node', node: B }, platform: 100, name: '' }); await new Promise((d) => setTimeout(d, 900)); }, net.B);
  s = await at();
  check(s.unusable === 0 && !s.shapes.length, 'a station at B: no badge', JSON.stringify(s));

  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3164 FAILED: ${fail}` : 'T-3164 PASSED');
process.exit(fail ? 1 : 0);
