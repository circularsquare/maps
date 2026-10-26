// T-1373 (T-264 stage 2): a superholiday's going-out and coming-back days.
// Node: the calendar gives going-out days kind 4 and coming-back days kind 5 (China's Spring Festival
// and Golden Week, the US's Thanksgiving).
// Page: a save (default the real save; SAVE=<path> for another, the China network is
// t1156-all-china.save at the maps root, COUNTRY=CN) on BASE (default its own Vite on PORT, 6201): the
// clock set two days before the country's next superholiday, the going-out and coming-back views
// are made; against the weekday and Sunday every purpose's riders are the same but the longer trips',
// which are RUSH_LONGER times a weekday's (a Sunday's timing coming back), give or take their lean;
// the lines' longer riders lean one way going out and the other coming back. Rail trips a day on each
// day, the made views' time, and screenshots of the Day choice and the calendar into
// artifacts/trainworld/t1373/ (SHOTS=0 skips them).
//   node trainworld/checks/t1373-rush-days.mjs [base]      (from the maps root; about 3 minutes, China 6-8)
import { mkdir, readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { ARTIFACTS, REAL_SAVE, NOISE, MAPS, launchChrome, collectErrors, importSave, demandFinal, startVite, loadApp } from './lib.mjs';

let failed = 0;
const report = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name} ${detail}`); };

const SAVE = process.env.SAVE ? resolve(MAPS, process.env.SAVE) : REAL_SAVE;
const COUNTRY = process.env.COUNTRY ?? 'US';
const TAG = process.env.TAG ?? (process.env.SAVE ? 'china' : 'real');
const OUT = join(ARTIFACTS, 't1373');
await mkdir(OUT, { recursive: true });

// ---- the calendar
const cal = await loadApp('src/game/holidayCalendar.ts');
const NEW = JSON.parse(await readFile(new URL('../app/src/game/holidays.json', import.meta.url), 'utf8'));
for (const [c, want] of [['CN', ['Spring Festival', 'Golden Week']], ['US', ['Thanksgiving']]]) {
  let ok = true;
  const k = cal.kindOfIn(NEW, c);
  for (let y = 0; y < 7; y++) {
    for (const r of cal.superRuns(NEW, c, 1 + 365 * y, 366 + 365 * y).filter((r) => want.some((w) => r.name.includes(w)))) {
      for (let d = r.out; d < r.end; d++) if (k(d) !== (d < r.away ? 4 : d < r.back ? 3 : 5)) ok = false;
      if (k(r.out - 1) >= 3 || k(r.end) >= 3) ok = false;
    }
  }
  report(`calendar: ${c} ${want.join(' and ')} go out as kind 4 and come back as kind 5`, ok);
}

// ---- the page
const port = Number(process.env.PORT ?? 6201);
let vite = null, base = process.argv[2];
if (!base) { vite = await startVite(port); base = vite.url; }
const { ctx, close } = await launchChrome({ prefix: 'tw-t1373-', deviceScaleFactor: 2 });
const errors = [];
const LONGER_SUNDAY = 1.48;

/** each purpose's riders and the rail trips of a day's view, and per line its longer riders each way */
const figures = (page, k) => page.evaluate((k) => {
  const v = window.twDemand.days()[k].view;
  const by = {};
  for (const [p, d] of v.purposes ?? []) { by[p] = 0; for (const l of d.lines.values()) by[p] += l.ridersPerDay; }
  const longer = v.purposes?.get('longer'), runs = {};
  for (const [id, l] of longer?.lines ?? []) {
    const n = l.boardings.length;
    let a = 0, b = 0;
    for (const seg of l.loads) for (let i = 0; i < n - 1; i++) { a += seg[i]; b += seg[n - 1 + i]; }
    if (a + b > 0) runs[id] = [a, b];
  }
  return { rail: v.railTripsPerDay, trips: v.tripsPerDay, by, runs, version: v.version };
}, k);

try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.goto(`${base}${base.includes('?') ? '&' : '?'}debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => {
    const c = window.twDemand.client, inSums = c.inSums.bind(c);
    c.inSums = (q, ...rest) => { if (q.kind === 'days') window.__daysInput = q.input; return inSums(q, ...rest); };
  });
  const t0 = Date.now();
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await demandFinal(page, { polling: 500, timeout: 1200000 });
  await page.waitForFunction(() => { const d = window.twDemand.days(); return d.every((x) => x.view?.final && x.view.version === d[0].view.version); }, undefined, { timeout: 120000 });
  console.log(`demand final ${Math.round((Date.now() - t0) / 1000)} s after the import`);
  const plain = await page.evaluate(async () => { const d = window.twDemand.days(), near = (await import('/src/game/days.ts')).rushRun.value; return [d[4].view === d[0].view, d[5].view === d[2].view, near]; });
  if (plain[2]) report('a superholiday is near at the save\'s date: its rush views are made', !plain[0] && !plain[1], `(${plain[2]})`);
  else report('no superholiday near: going out and coming back are the weekday and Sunday', plain[0] && plain[1]);
  const wd = await figures(page, 0), sun = await figures(page, 2);

  // the clock two days before the country's next run's going-out day
  await page.evaluate(() => window.tw.clock.setRunning(false));
  const run = await page.evaluate(async (cc) => {
    const h = await import('/src/game/holidays.ts'), c = await import('/src/game/holidayCalendar.ts');
    h.loadHolidays();
    for (let i = 0; i < 100 && !h.data.value; i++) await new Promise((r) => setTimeout(r, 100));
    const today = Math.floor(window.tw.clock.now() / 86400);
    return c.superRuns(h.data.value, cc, today + 3, today + 400)[0] ?? null;
  }, COUNTRY);
  if (!run) throw new Error(`no superholiday run in ${COUNTRY}`);
  const setDay = (day) => page.evaluate((d) => window.tw.clock.set(d * 86400 + 9 * 3600 - (window.tw.viewShift?.value ?? 0)), day);
  await setDay(run.out - 2);
  const made = Date.now();
  await page.waitForFunction(() => { const d = window.twDemand.days(); return d[4].view && d[4].view !== d[0].view && d[5].view !== d[2].view; }, undefined, { timeout: 120000 });
  console.log(`${run.name}: out day ${run.out}, away ${run.away}, back ${run.back}, end ${run.end}; rush views made ${Date.now() - made} ms after the clock moved`);
  const out = await figures(page, 4), back = await figures(page, 5);
  report('rush views are of the same solve', out.version === wd.version && back.version === wd.version);

  const same = (a, b) => Object.entries(a.by).filter(([p]) => p !== 'longer').every(([p, n]) => Math.abs((b.by[p] ?? 0) - n) <= 1e-6 * Math.max(1, n));
  report('going out: every purpose but the longer trips is a weekday\'s', same(wd, out));
  report('coming back: every purpose but the longer trips is a Sunday\'s', same(sun, back));
  const ro = (out.by.longer ?? 0) / (wd.by.longer || 1), rb = (back.by.longer ?? 0) / ((sun.by.longer || 1) / LONGER_SUNDAY);
  report('longer trips about twice a weekday\'s', Math.abs(ro - 2) < 0.15 && Math.abs(rb - 2) < 0.15, `(going out ${ro.toFixed(3)}, coming back ${rb.toFixed(3)} times a weekday's; ${Math.round(wd.by.longer ?? 0)} longer riders on a weekday)`);
  // the lean: per line the share of its longer riders on run 0, weekday against going out and coming back
  let leaned = 0, opposite = 0, biggest = null;
  for (const [id, [a, b]] of Object.entries(wd.runs)) {
    const o = out.runs[id], k = back.runs[id];
    if (!o || !k || a + b < 50) continue;
    const s0 = a / (a + b), so = o[0] / (o[0] + o[1]), sk = k[0] / (k[0] + k[1]);
    if (Math.abs(so - s0) <= 0.005) continue;
    leaned++;
    if ((so - s0) * (sk - s0) < 0) opposite++;
    if (!biggest || Math.abs(so - s0) > biggest.d) biggest = { id, d: Math.abs(so - s0), s0, so, sk };
  }
  report('lines lean one way going out and the other coming back', leaned > 0 && opposite >= leaned * 0.8, `(${leaned} lines lean by more than half a point going out, ${opposite} of them the other way coming back${biggest ? `; most: line ${biggest.id}, run 0's share of its longer riders ${(100 * biggest.s0).toFixed(1)}% on a weekday, ${(100 * biggest.so).toFixed(1)}% going out, ${(100 * biggest.sk).toFixed(1)}% coming back` : ''})`);
  report('rail trips a day', true, `(weekday ${Math.round(wd.rail)}, going out ${Math.round(out.rail)} (${(100 * (out.rail / wd.rail - 1)).toFixed(2)}%), Sunday ${Math.round(sun.rail)}, coming back ${Math.round(back.rail)} (${(100 * (back.rail / sun.rail - 1)).toFixed(2)}%); longer riders ${Math.round(wd.by.longer ?? 0)}, ${Math.round(out.by.longer ?? 0)}, ${Math.round(sun.by.longer ?? 0)}, ${Math.round(back.by.longer ?? 0)})`);

  // the day views with the rush days and without, same input, five runs each
  const t = await page.evaluate(async () => {
    const c = window.twDemand.client, input = window.__daysInput;
    if (!input) return null;
    const run = async (days) => { const a = performance.now(); await c.inSums({ kind: 'days', input: { ...input, days } }, [], () => null); return performance.now() - a; };
    const med = async (days) => { const xs = []; for (let i = 0; i < 5; i++) xs.push(await run(days)); return xs.sort((a, b) => a - b)[2]; };
    return { all: await med([0, 1, 2, 3, 4, 5, 6]), five: await med([0, 1, 2, 3, 6]) };
  });
  report('day views timed', !!t, t ? `(${t.five.toFixed(0)} ms without the rush days, ${t.all.toFixed(0)} ms with them, in the sums worker)` : '(no input caught)');

  // the Day choice and the calendar
  const dayChoice = await page.evaluate(async () => {
    const u = await import('/src/game/ui.ts');
    u.tab.value = 'settings';
    await new Promise((r) => setTimeout(r, 500));
    return [...document.querySelectorAll('.day-away')].map((e) => e.dataset.tip);
  });
  report('the Day choice offers going out, away and coming back', dayChoice.length === 3, JSON.stringify(dayChoice));
  await setDay(run.out - 1);
  await page.waitForTimeout(800);
  const had = await page.evaluate(() => window.tw.display.value.dayShown);
  await page.locator('.clock .cal').click();
  const after = await page.evaluate(() => window.tw.display.value.dayShown);
  const shownOut = await page.evaluate(async () => (await import('/src/game/ui.ts')).shownDay.value);
  report('the calendar\'s click shows the going-out day', after === 'out' && shownOut === 4, `${had} -> ${after}`);
  if (process.env.SHOTS !== '0') {
    for (const shade of ['light', 'dark']) {
      await page.evaluate((s) => { document.documentElement.dataset.shade = s; window.tw.shade.value = s; }, shade);
      await page.waitForTimeout(600);
      await page.mouse.move(2, 2);
      const seg = await page.locator('.day-away').first().boundingBox();
      if (seg) await page.screenshot({ path: join(OUT, `${TAG}-day-choice-${shade}.png`), clip: { x: 0, y: Math.max(0, seg.y - 80), width: 440, height: 140 } });
      await page.locator('.clock .cal').hover();
      await page.waitForTimeout(600);
      await page.screenshot({ path: join(OUT, `${TAG}-going-out-${shade}.png`) });
    }
    await page.evaluate(() => { document.documentElement.dataset.shade = 'light'; window.tw.shade.value = 'light'; });
  }
  await page.close();
} catch (e) {
  report('page', false, e.message.split('\n')[0]);
} finally {
  await close();
  vite?.child.kill();
}
report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
process.exit(failed ? 1 : 0);
