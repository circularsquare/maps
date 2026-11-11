// T-3123: Manual bus stops on routes with a loop, and no other line's stop markers in a bus line's stop editing.
// On Anita's Amsterdam save: a tram line's stops are edited (its markers show), then a bus route that loops back is
// made beside it and set to Manual: none of the tram's markers stay up, before or after the map pans. A click on the
// loop adds a stop there once, a click on the way out a stop on both passes, a click on that stop takes both out,
// and the loop's last stop between offers no removal. Then a ring: a click adds a stop. Screenshots
// artifacts/trainworld/t3123-*.png.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3123-busloop-stops.mjs [dev url]
// About two minutes.
import { ARTIFACTS, collectErrors, demandFinal, launchChrome } from './lib.mjs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6185/';
const SAVE = process.env.SAVE ?? join(ARTIFACTS, 't3099-anita-amsterdam.save');
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = await launchChrome({ prefix: 'tw-t3123-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
collectErrors(page, errors);
const ready = () => page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });

/** the train and tram stop markers on screen: shown, inside the map */
const foreign = () => page.evaluate(() => {
  const box = window.tw.tools.stops.box, r = window.tw.map.getContainer().getBoundingClientRect();
  if (getComputedStyle(box).display === 'none') return 0;
  return [...box.children].filter((e) => { if (getComputedStyle(e).display === 'none') return false; const b = e.getBoundingClientRect(); return b.width > 0 && b.right > r.left && b.left < r.right && b.bottom > r.top && b.top < r.bottom; }).length;
});
/** a bus line's stops as [lng, lat], and whether it is by hand */
const busOf = (num) => page.evaluate((num) => {
  const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), b = l && w.lineStats[l.id]?.bus;
  return b ? { id: l.id, hand: b.hand, stops: b.stops.map((s) => [s.lng, s.lat]), route: Array.from(b.routeLL) } : null;
}, num);
const until = async (num, test) => { for (let k = 0; k < 100; k++) { const s = await busOf(num); if (s && test(s)) return s; await wait(200); } return busOf(num); };
const screen = (lng, lat) => page.evaluate(([lng, lat]) => { const p = window.tw.map.project([lng, lat]); const r = window.tw.map.getContainer().getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; }, [lng, lat]);
const click = async (lng, lat) => { const [x, y] = await screen(lng, lat); await page.mouse.move(x, y); await wait(250); await page.mouse.click(x, y); };
const tagText = () => page.evaluate(() => [...document.querySelectorAll('.stop-tag')].filter((t) => t.style.display !== 'none').map((t) => t.innerText).join(' / '));
const mode = (name) => page.locator('.h .group button').filter({ hasText: new RegExp(`^${name}$`) });
/** the point `f` of the way from route point i to i + 1 */
const along = (route, i, f) => [route[2 * i] + f * (route[2 * i + 2] - route[2 * i]), route[2 * i + 1] + f * (route[2 * i + 3] - route[2 * i + 1])];
const makeBus = (route, name) => page.evaluate(async ([route, name]) => {
  const a = await window.tw.client.edit({ op: 'addBusLine', route, name, colour: '#8e44ad' });
  return { ok: a.ok, line: a.line, issues: a.issues };
}, [route, name]);
const manual = async (num) => {
  const id = (await busOf(num)).id;
  await page.evaluate((id) => { window.tw.tools.exit(); window.tw.selection.value = { kind: 'line', line: id }; }, id);
  await wait(800);
  await mode('Manual').click();
  const s = await until(num, (x) => x.hand);
  await page.waitForFunction(() => document.documentElement.classList.contains('stop-editing'), undefined, { timeout: 10000 }).catch(() => null);
  await wait(800);
  return s;
};
const frame = (route) => page.evaluate((route) => {
  const lngs = route.filter((_, i) => i % 2 === 0), lats = route.filter((_, i) => i % 2 === 1);
  window.tw.map.fitBounds([[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]], { padding: 220, duration: 0 });
}, route);

try {
  await page.goto(`${BASE}?debug=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await ready();
  const bytes = await readFile(SAVE);
  await page.evaluate(async (data) => { const m = await import('/src/workers/clockClient.ts'); await m.importGame(new File([Uint8Array.from(data)], 'game.save')); }, Array.from(bytes));
  await page.waitForURL(/load=file/, { timeout: 120000 });
  await ready();
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000, polling: 100 });
  await demandFinal(page, { timeout: 600000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await wait(1000);

  // ---- a tram line's stops edited: its markers show
  const tram = await page.evaluate(async () => {
    const tw = window.tw, w = tw.world.value, ui = await import('/src/game/ui.ts');
    const l = w.save.lines.find((x) => w.lineStats[x.id]?.tram && x.stops.length >= 4);
    const n = w.nodes.get(Number(l.stops[Math.floor(l.stops.length / 2)]));
    tw.selection.value = { kind: 'line', line: l.id };
    ui.tool.value = 'line';
    ui.lineDraft.value = { line: l.num, stops: l.stops.map(Number) };
    return { num: l.num, name: l.name, x: n.x, y: n.y, lng: n.lng, lat: n.lat };
  });
  console.log('tram', tram.name);
  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 14.6 }), [tram.lng, tram.lat]);
  await wait(2500);
  const shownTram = await foreign();
  check(shownTram > 0, `${tram.name}'s stop markers show while its stops are edited`, String(shownTram));
  await page.screenshot({ path: join(ARTIFACTS, 't3123-tram-editing.png') });

  // ---- a bus route beside it that loops back: out, round a square, back
  const { x, y } = tram;
  const P = [[250, 250], [850, 250], [1450, 250], [1450, 850], [850, 850], [850, 250], [250, 250]];
  const made = await makeBus(P.flatMap(([dx, dy]) => [x + dx, y + dy, 1]), 'Loop back');
  check(made.ok, 'a bus route that loops back is made', JSON.stringify(made));
  const back = await manual(made.line);
  check(back.hand && back.stops.length === 3, 'Manual leaves the closing stop and one half way round', String(back.stops.length));
  await frame(back.route);
  await wait(1500);
  let n = await foreign();
  check(n === 0, 'no train or tram stop markers in the bus line\'s stop editing', String(n));
  await page.evaluate(() => window.tw.map.panBy([180, 120], { duration: 0 }));
  await wait(1200);
  n = await foreign();
  check(n === 0, 'nor after the map pans', String(n));
  await page.screenshot({ path: join(ARTIFACTS, 't3123-bus-manual-panned.png') });
  await frame(back.route);
  await wait(1200);

  // a click on the loop adds a stop there once; one on the way out, a stop on both passes
  const R = back.route;
  await click(...along(R, 2, 0.5));
  let s = await until(made.line, (b) => b.stops.length === 4);
  check(s.stops.length === 4, 'a click on the loop adds a stop, the one half way round kept', String(s.stops.length));
  await click(...along(R, 0, 0.5));
  s = await until(made.line, (b) => b.stops.length === 6);
  check(s.stops.length === 6, 'a click on the way out adds a stop on the way out and on the way back', String(s.stops.length));
  await page.mouse.move(700, 880);
  await wait(600);
  await page.screenshot({ path: join(ARTIFACTS, 't3123-loopback-stops.png') });
  const stem = s.stops[1];
  await click(...stem);
  s = await until(made.line, (b) => b.stops.length === 4);
  check(s.stops.length === 4, 'a click on that stop takes it out of both passes', String(s.stops.length));
  await click(...s.stops[1]);
  s = await until(made.line, (b) => b.stops.length === 3);
  check(s.stops.length === 3, 'and the loop\'s stop goes', String(s.stops.length));
  const [lx, ly] = await screen(...s.stops[1]);
  await page.mouse.move(lx, ly);
  await wait(500);
  const lastTag = await tagText();
  check(!/remove/i.test(lastTag), 'the last stop between offers no removal', lastTag);

  // ---- a ring: the first click adds a stop
  const ring = [[250, -250], [850, -250], [850, -850], [250, -850], [250, -250]];
  const madeRing = await makeBus(ring.flatMap(([dx, dy]) => [x + dx, y + dy, 1]), 'Ring');
  check(madeRing.ok, 'a bus ring is made', JSON.stringify(madeRing));
  const r0 = await manual(madeRing.line);
  await frame(r0.route);
  await wait(1500);
  check((await foreign()) === 0, 'no foreign markers on the ring either');
  await click(...along(r0.route, 0, 0.5));
  s = await until(madeRing.line, (b) => b.stops.length === 4);
  check(s.stops.length === 4, 'a click on a ring adds a stop', `${r0.stops.length} -> ${s.stops.length}`);
  await page.mouse.move(700, 880);
  await wait(600);
  await page.screenshot({ path: join(ARTIFACTS, 't3123-ring-stops.png') });
  await page.evaluate(() => window.tw.tools.exit());
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't3123-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
}
check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
await chrome.close();
console.log(fail ? `T-3123 FAILED (${fail})` : 'T-3123 PASSED');
process.exit(fail ? 1 : 0);
