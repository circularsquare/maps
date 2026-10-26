// T-1476, T-1477, T-1478 on the real save: the bare calendar icon (only within 2 days of a
// superholiday or during one), the map's rows (track above station ridership, no Drives row, a Drives
// setting), and the "+N daily rides" indicator on the map (moved there by T-1499) after a line's stops change. Screenshots light and dark into
// artifacts/trainworld/t1476/. From the maps root, with a running dev server:
//   node trainworld/checks/t1476-controls.mjs <port>
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, GL, NOISE, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1476');
await mkdir(OUT, { recursive: true });
const BASE = `http://localhost:${process.argv[2] ?? 6183}/`;
const { ctx, close } = await launchChrome({ prefix: 'tw-t1476-', size: [1400, 900], args: GL, deviceScaleFactor: 2 });
const errors = [];
let fails = 0;
const check = (name, ok, detail = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'average' }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 20, undefined, { timeout: 120000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await demandFinal(page, { polling: 250, lines: 20 });
  await page.waitForTimeout(1500);
  const both = async (name) => {
    for (const shade of ['light', 'dark']) {
      await page.evaluate((s) => { document.documentElement.dataset.shade = s; window.tw.shade.value = s; }, shade);
      await page.waitForTimeout(500);
      await name(shade);
    }
    await page.evaluate(() => { document.documentElement.dataset.shade = 'light'; window.tw.shade.value = 'light'; });
  };

  // ---- T-1476: the calendar
  await page.evaluate(() => window.tw.clock.setRunning(false));
  const run = await page.evaluate(async () => {
    const h = await import('/src/game/holidays.ts'), c = await import('/src/game/holidayCalendar.ts');
    h.loadHolidays();
    for (let i = 0; i < 100 && !h.data.value; i++) await new Promise((r) => setTimeout(r, 100));
    const w = window.tw.world.value;
    const country = [...h.railCountries.value][0]?.split(' ') ?? [];
    const out = [];
    for (const k of h.railCountries.value) {
      const [cc] = k.split(' ');
      for (const r of c.superRuns(h.data.value, cc, 400, 800)) out.push({ cc, ...r });
    }
    return { countries: [...h.railCountries.value], out, country };
  });
  console.log('rail countries and runs', JSON.stringify(run));
  const r0 = run.out[0];
  if (!r0) throw new Error('no superholiday run found for the real save');
  const setDay = async (day) => {
    await page.evaluate((d) => { const c = window.tw.clock; c.set(d * 86400 + 9 * 3600 - (window.tw.viewShift?.value ?? 0)); }, day);
    await page.waitForTimeout(1200);
  };
  const cal = () => page.locator('.clock .cal').count();
  await setDay(r0.out - 10);
  check('10 days before a superholiday: no calendar icon', (await cal()) === 0);
  const dayOpt = await page.evaluate(async () => { const u = await import('/src/game/ui.ts'); u.tab.value = 'settings'; await new Promise((r) => setTimeout(r, 400)); return !!document.querySelector('.day-away'); });
  check('10 days before: the Settings Day choice still offers the calendar day', dayOpt);
  await setDay(r0.out - 3);
  check('3 days before: no calendar icon', (await cal()) === 0);
  await setDay(r0.out - 2);
  check('2 days before: calendar icon shows', (await cal()) === 1);
  await setDay(r0.out - 1);
  check('1 day before: calendar icon shows', (await cal()) === 1);
  const style = await page.evaluate(() => {
    const b = document.querySelector('.clock .cal'), s = getComputedStyle(b), d = getComputedStyle(b.querySelector('.cal-dot'));
    const r = b.getBoundingClientRect(), dr = b.querySelector('.cal-dot').getBoundingClientRect();
    return { border: s.borderTopWidth, bg: s.backgroundColor, dot: d.backgroundColor, tip: b.dataset.tip, dotInside: dr.left >= r.left && dr.right <= r.right && dr.top >= r.top && dr.bottom <= r.bottom, w: r.width, h: r.height };
  });
  console.log('calendar', JSON.stringify(style));
  check('no border and no background', style.border === '0px' && style.bg === 'rgba(0, 0, 0, 0)');
  check('a pink dot inside it', style.dot === 'rgb(255, 95, 162)' && style.dotInside);
  check('its hover lists the run', /in \d+ days|tomorrow|now/.test(style.tip ?? ''), (style.tip ?? '').split('\n')[0]);
  await setDay(r0.out + 1);
  check('during the run: calendar icon shows', (await cal()) === 1);
  await both(async (shade) => {
    const box = await page.locator('.clock').boundingBox();
    await page.mouse.move(2, 2);
    await page.screenshot({ path: join(OUT, `calendar-${shade}.png`), clip: { x: box.x - 16, y: Math.max(0, box.y - 6), width: box.width + 40, height: box.height + 14 } });
  });
  // the click and hover are T-1372's: click toggles the away day when one is offered
  await setDay(r0.away - 1);
  const had = await page.evaluate(() => window.tw.display.value.dayShown);
  await page.locator('.clock .cal').click();
  const after = await page.evaluate(() => window.tw.display.value.dayShown);
  check('click shows demand on the day away', after === 'away', `${had} -> ${after}`);
  await page.locator('.clock .cal').click();
  check('click again goes back', (await page.evaluate(() => window.tw.display.value.dayShown)) === had);
  await page.locator('.clock .cal').hover();
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(OUT, 'calendar-hover-light.png'), clip: { x: 700, y: 0, width: 700, height: 260 } });
  await setDay(r0.end + 3);
  check('after the run: no calendar icon', (await cal()) === 0);

  // ---- T-1477: the map rows
  await page.evaluate(async () => { const u = await import('/src/game/ui.ts'); u.tab.value = 'lines'; });
  await setDay(r0.out - 30);
  const rows = await page.evaluate(() => [...document.querySelectorAll('.map-top-left > .demand-selector, .map-top-row .demand-selector')].map((e) => e.getAttribute('aria-label') ?? e.querySelector('[aria-label]')?.getAttribute('aria-label')));
  console.log('rows', JSON.stringify(rows));
  check('rows: demand, track display, station ridership', JSON.stringify(rows) === JSON.stringify(['Demand on the map', 'Track display', 'Station ridership']), JSON.stringify(rows));
  check('no Drives row on the map', (await page.locator('[aria-label="Where people drive"]').count()) === 0);
  await both(async (shade) => {
    await page.mouse.move(2, 2);
    const box = await page.locator('.map-top-left').boundingBox();
    await page.screenshot({ path: join(OUT, `rows-${shade}.png`), clip: { x: Math.max(0, box.x - 10), y: Math.max(0, box.y - 10), width: Math.max(box.width + 20, 460), height: box.height + 20 } });
  });
  // folding still works for the rows that remain
  await page.locator('[aria-label="Station ridership"] .hint-cell').click();
  check('folding the station ridership row hides its buttons', (await page.locator('[aria-label="Station ridership"] .btn').count()) === 0);
  await page.locator('[aria-label="Station ridership"] .hint-cell').click();
  await page.locator('[aria-label="Track display"] .hint-cell').click();
  check('folding the track row hides its buttons', (await page.locator('[aria-label="Track display"] .btn').count()) === 0);
  await page.locator('[aria-label="Track display"] .hint-cell').click();
  check('both open again', (await page.locator('[aria-label="Track display"] .btn').count()) === 4 && (await page.locator('[aria-label="Station ridership"] .btn').count()) === 4);
  // Drives in Settings > Show
  await page.evaluate(async () => { const u = await import('/src/game/ui.ts'); u.tab.value = 'settings'; });
  await page.waitForTimeout(500);
  const drives = page.locator('.checks label', { hasText: 'Drives' });
  check('Settings > Show has a Drives check', (await drives.count()) === 1);
  const off = await page.evaluate(() => window.tw.display.value.drives);
  await drives.locator('input').check();
  check('ticking it turns the drives on', (await page.evaluate(() => window.tw.display.value.drives)) === true && off === false);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(OUT, 'drives-setting-light.png') });
  await drives.locator('input').uncheck();
  check('unticking turns them off', (await page.evaluate(() => window.tw.display.value.drives)) === false);

  // ---- T-1478: the rides note
  await setDay(r0.out - 30);
  await page.evaluate(async () => { const u = await import('/src/game/ui.ts'); u.tab.value = 'lines'; });
  const pick = await page.evaluate(() => {
    const w = window.tw.world.value;
    return w.save.lines.filter((l) => w.lineStats[l.id]?.status === 'running' && l.stops.length > 5).map((l) => ({ id: l.id, num: l.num, name: l.name, stops: l.stops }))[0];
  });
  console.log('line', pick.name, pick.stops.length, 'stops');
  const noteOf = () => page.evaluate((name) => {
    const n = [...document.querySelectorAll('.rides-notes .rides-note')].find((e) => e.querySelector('.chip')?.dataset.tip === name);
    return n ? { text: n.textContent.trim(), tip: n.dataset.tip, cls: n.className } : null;
  }, pick.name);
  check('no legend beside the track buttons', (await page.locator('.track-key').count()) === 0);
  check('no note before any edit', (await noteOf()) === null);
  const dropped = pick.stops.slice(0, -1);
  await page.evaluate(({ num, stops }) => window.tw.client.edit({ op: 'setStops', line: num, stops: stops.map(Number) }), { num: pick.num, stops: dropped });
  const seen = [];
  const t0 = Date.now();
  while (Date.now() - t0 < 90000) {
    const n = await noteOf();
    if (n) { seen.push([Date.now() - t0, n.text]); if (seen.length === 1) { await page.waitForTimeout(1300); var first = await noteOf(); } }
    if (seen.length && Date.now() - t0 > seen[0][0] + 6000) break;
    await page.waitForTimeout(150);
  }
  console.log('after removing the last stop', JSON.stringify(seen.slice(0, 8)), seen.length);
  check('a note appears after the last stop is removed and demand answers', seen.length > 0, first?.text ?? '');
  check('and is gone after a few seconds', (await noteOf()) === null);
  if (first) console.log('note', JSON.stringify(first));

  // add the stop back, light and dark screenshots while it shows
  await page.evaluate(({ num, stops }) => window.tw.client.edit({ op: 'setStops', line: num, stops: stops.map(Number) }), { num: pick.num, stops: pick.stops });
  let up = null;
  const t1 = Date.now();
  while (Date.now() - t1 < 90000 && !up) { up = await noteOf(); if (!up) await page.waitForTimeout(150); }
  check('a note appears after the stop is added back', !!up, up?.text ?? '');
  if (up) {
    await page.waitForTimeout(1200);
    await page.mouse.move(2, 2);
    const dock = await page.locator('#map').boundingBox();
    await page.screenshot({ path: join(OUT, 'note-light.png'), clip: { x: dock.x + dock.width / 2 - 200, y: dock.y, width: 400, height: 120 } });
    await page.evaluate(() => { document.documentElement.dataset.shade = 'dark'; window.tw.shade.value = 'dark'; });
    await page.waitForTimeout(300);
    await page.screenshot({ path: join(OUT, 'note-dark.png'), clip: { x: dock.x + dock.width / 2 - 200, y: dock.y, width: 400, height: 120 } });
    await page.evaluate(() => { document.documentElement.dataset.shade = 'light'; window.tw.shade.value = 'light'; });
  }
  check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(fails ? 1 : 0);
