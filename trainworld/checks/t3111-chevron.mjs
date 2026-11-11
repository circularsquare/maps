// T-3111: a one-way stop's heading as a chevron in its line's colour: in the station hover beside the line's bullet,
// and on the marker while that line's stops are edited; two-way stops get none. Amsterdam save, Line 1.
// From the maps root, on a Vite of your own:   node trainworld/checks/t3111-chevron.mjs [dev url]   (SHADE=dark for dark)
import { ARTIFACTS, collectErrors, demandFinal, launchChrome } from './lib.mjs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6188/';
const SAVE = process.env.SAVE ?? join(ARTIFACTS, 't3099-anita-amsterdam.save');
const SHADE = process.env.SHADE ?? 'light';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t3111-', size: [1400, 900], deviceScaleFactor: 2 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, SHADE);
collectErrors(page, errors);
const ready = () => page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });
const centre = (id, zoom) => page.evaluate(([id, zoom]) => {
  const tw = window.tw, n = tw.world.value.nodes.get(Number(id));
  tw.map.jumpTo({ center: tw.packToLngLat ? tw.packToLngLat(n.x, n.y) : [n.lng, n.lat], zoom });
}, [id, zoom]);
const chev = () => page.evaluate(() => [...document.querySelectorAll('.tool-tag .heading-chevron, .stop-mark > .heading-chevron')].filter((e) => e.offsetParent && getComputedStyle(e).display !== 'none').map((e) => ({ colour: getComputedStyle(e).color, tf: e.style.transform, tag: !!e.closest('.tool-tag') })));

try {
  await page.goto(`${BASE}?debug=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await ready();
  const bytes = await readFile(SAVE);
  await page.evaluate(async (data) => { const m = await import('/src/workers/clockClient.ts'); await m.importGame(new File([Uint8Array.from(data)], 'game.save')); }, Array.from(bytes));
  await page.waitForURL(/load=file/, { timeout: 120000 });
  await ready();
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000, polling: 100 });
  await demandFinal(page, { timeout: 600000 });
  await page.waitForTimeout(1000);
  const info = await page.evaluate(async () => {
    const w = window.tw.world.value, l = w.save.lines.find((x) => x.name === 'Line 1'), st = w.lineStats[l.id];
    // T-3122: the sim's one-way stops
    const ow = new Map(Object.entries(st.oneWayAt)), nm = new Map(w.save.stations.map((s) => [s.id, s.name]));
    const two = l.stops.find((s) => !ow.has(s) && l.stops.filter((x) => x === s).length > 1);
    const one = [...ow.keys()].find((s) => nm.get(s) === 'Apollobuurt') ?? [...ow.keys()][0];
    return { colour: l.colour, num: l.num, one, oneName: nm.get(one), two, twoName: nm.get(two), count: ow.size };
  });
  console.log(JSON.stringify(info));
  check(!!info.one && !!info.two, 'Line 1 has one-way and two-way stops', JSON.stringify(info));
  let mid = { x: 700, y: 450 };
  const hoverAt = async (id) => { await centre(id, 14.5); await page.waitForTimeout(2500); mid = await page.evaluate((id) => { const tw = window.tw, n = tw.world.value.nodes.get(Number(id)), p = tw.map.project(tw.packToLngLat(n.x, n.y)), r = tw.map.getContainer().getBoundingClientRect(); return { x: r.left + p.x, y: r.top + p.y }; }, id); await page.mouse.move(mid.x + 3, mid.y + 3); await page.mouse.move(mid.x, mid.y); await page.waitForTimeout(800); };
  await hoverAt(info.one);
  let c = await chev();
  check(c.some((e) => e.tag), `hover on ${info.oneName} shows a chevron`, JSON.stringify(c));
  await page.screenshot({ path: join(ARTIFACTS, `t3124-hover-oneway-${SHADE}.png`), clip: { x: Math.max(0, mid.x - 150), y: Math.max(0, mid.y - 120), width: 500, height: 300 }, scale: "device" });
  await hoverAt(info.two);
  c = await chev();
  check(!c.some((e) => e.tag), `hover on two-way ${info.twoName} shows none`, JSON.stringify(c));
  await page.screenshot({ path: join(ARTIFACTS, `t3124-hover-twoway-${SHADE}.png`), clip: { x: Math.max(0, mid.x - 150), y: Math.max(0, mid.y - 120), width: 500, height: 300 }, scale: "device" });

  await page.evaluate(async (num) => {
    const tw = window.tw, ui = await import('/src/game/ui.ts');
    const l = tw.world.value.save.lines.find((x) => x.num === num);
    tw.selection.value = { kind: 'line', line: l.id };
    ui.tool.value = 'line';
    ui.lineDraft.value = { line: num, stops: l.stops.map(Number) };
  }, info.num);
  await page.waitForTimeout(2500);
  await centre(info.one, 14.0);
  await page.waitForTimeout(2500);
  c = await chev();
  check(c.filter((e) => !e.tag).length >= 1, 'editing Line 1 shows chevrons by its one-way stops', String(c.length));
  const want = SHADE === 'dark' ? 'rgb(21, 23, 26)' : 'rgb(255, 255, 255)';
  check(c.filter((e) => !e.tag).every((e) => e.colour === want), `map chevrons are ${want}`, c.filter((e) => !e.tag)[0]?.colour);

  const marks = await page.evaluate(() => [...window.tw.tools.stops.marks].filter(([, m]) => m.shown).map(([id, m]) => [id, !!m.chev && m.chev.style.display !== 'none']));
  const bad = marks.filter(([id, has]) => has && id === undefined);
  check(bad.length === 0 && marks.some(([, h]) => !h), 'two-way markers have none', `${marks.filter(([, h]) => h).length} with, ${marks.filter(([, h]) => !h).length} without`);
  await centre(info.one, 16);
  await page.waitForTimeout(2500);
  await page.screenshot({ path: join(ARTIFACTS, `t3124-editing-${SHADE}.png`) });
} catch (e) {
  console.log('FAIL', e.message);
  fail++;
}
check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
await chrome.close();
console.log(fail ? `T-3111 FAILED (${fail})` : 'T-3111 PASSED');
process.exit(fail ? 1 : 0);
