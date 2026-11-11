// T-3137 C4: the track inspector's one-way switch on single track shows the way (inferred included), pins both ways
// (stored 3), goes back to automatic, and setting a way moves the piece's end join to the track that carries it.
// A hand-laid stub off a double main's upper track: its way is inferred. Pictures in artifacts/trainworld/t3137-way-*.png.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6203):
//   node trainworld/checks/t3137-way-switch.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6203/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3137w-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3137-way-${name}.png`) });

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(([c]) => {
    const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]);
    window.__w = { P: (dx, dy) => ({ x: o.x + dx, y: o.y + 3000 + dy }) };
  }, [RC]);
  const stub = await page.evaluate(async () => {
    const tw = window.tw, cl = tw.client, { P } = window.__w;
    await cl.edit({ op: 'route', from: { kind: 'free', ...P(-3000, 0), level: 0 }, to: { kind: 'free', ...P(3000, 0), level: 0 }, pis: [], single: false });
    await new Promise((d) => setTimeout(d, 600));
    const w0 = tw.world.value, y0 = P(0, 0).y, X = P(-900, 0).x;
    const main = w0.edges.find((e) => { const a = w0.nodes.get(e.a), b = w0.nodes.get(e.b); return e.tracks === 2 && Math.abs(a.y - y0) < 1 && Math.abs(b.y - y0) < 1 && Math.min(a.x, b.x) < X && X < Math.max(a.x, b.x); })?.id;
    const before = new Set(w0.edges.map((e) => e.id)), j = P(-900, 2);
    const r = await cl.edit({ op: 'route', from: { kind: 'edge', edge: main, ...P(-900, 0) }, to: { kind: 'free', ...P(-600, 250), level: 0 }, pis: [], single: true, design: 60, join: [j.x, j.y, null, null] });
    await new Promise((d) => setTimeout(d, 800));
    const s = tw.world.value.edges.find((e) => e.tracks === 1 && !before.has(e.id));
    return { ok: r.ok, id: s?.id, a: s?.a };
  });
  check(stub.ok && stub.id !== undefined, 'a stub laid off the main\'s upper track', JSON.stringify(stub));
  const row = () => page.evaluate((id) => { const e = window.tw.world.value.edges.find((x) => x.id === id); return { way: e.oneWay ?? 0, set: e.setWay ?? 0, auto: e.autoWay ?? 0, join: e.join ?? 0 }; }, stub.id);
  const box = () => page.evaluate(() => { const c = document.querySelector('label.check input'); const names = [...document.querySelectorAll('.btn.text')].map((b) => b.textContent.trim()); return { on: c?.checked ?? null, buttons: names }; });
  const jump = async (z) => {
    await page.evaluate(async ([id, z]) => {
      const tw = window.tw, w = tw.world.value, e = w.edges.find((x) => x.id === id), a = w.nodes.get(e.a), b = w.nodes.get(e.b);
      (await import('/src/game/ui.ts')).selection.value = { kind: 'track', edge: id };
      const q = tw.packToLngLat(a.x + (b.x - a.x) * 0.12, a.y + (b.y - a.y) * 0.12), [lng, lat] = Array.isArray(q) ? q : [q.lng, q.lat];
      tw.map.jumpTo({ center: [lng, lat], zoom: z });
    }, [stub.id, z]);
    await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
    await wait(700);
  };
  await page.waitForFunction(() => !document.body.innerText.includes("Loading city data"), undefined, { timeout: 120000 }).catch(() => {});
  await jump(18.6);

  const r0 = await row(), b0 = await box();
  check(r0.way !== 0 && r0.set === 0 && r0.join !== 0, 'the stub runs one way by itself, nothing stored', JSON.stringify(r0));
  check(b0.on === true && !b0.buttons.includes('Use automatic way'), 'the switch is on and there is no Automatic button', JSON.stringify(b0));
  await shot('1-inferred');

  // off: pinned both ways (3)
  await page.locator('label.check').first().click();
  await wait(900);
  const r1 = await row(), b1 = await box();
  check(r1.set === 3 && r1.way === 0 && r1.join === r0.join, 'switch off stores 3 (both ways pinned), the join stays', JSON.stringify(r1));
  check(b1.on === false && b1.buttons.includes('Use automatic way'), 'the switch is off and the Automatic button appears', JSON.stringify(b1));
  await shot('2-pinned-both');

  // automatic
  await page.getByText('Use automatic way', { exact: true }).click();
  await wait(900);
  const r2 = await row(), b2 = await box();
  check(r2.set === 0 && r2.way === r0.way, 'Automatic stores 0 and the inferred way returns', JSON.stringify(r2));
  check(b2.on === true && !b2.buttons.includes('Use automatic way'), 'the switch is on again, no Automatic button', JSON.stringify(b2));

  // reverse: the way flips and the end join moves to the other track
  await page.getByText('Reverse direction', { exact: true }).click();
  await wait(900);
  const r3 = await row();
  check(r3.set === 3 - r0.way && r3.way === 3 - r0.way, 'Reverse stores the other way', JSON.stringify(r3));
  check(r3.join !== 0 && r3.join !== r0.join, 'the join at the junction moved to the other track', `${r0.join} -> ${r3.join}`);
  await shot('3-reversed');

  // switch on from both-ways: stores the automatic way
  await page.locator('label.check').first().click();
  await wait(900);
  const r4 = await row();
  check(r4.set === 3 && r4.way === 0, 'off again pins both ways', JSON.stringify(r4));
  await page.locator('label.check').first().click();
  await wait(900);
  const r5 = await row();
  check(r5.way !== 0 && r5.set === r5.way, 'switching on stores a way', JSON.stringify(r5));

  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3137 C4 FAILED: ${fail}` : 'T-3137 C4 PASSED');
process.exit(fail ? 1 : 0);
