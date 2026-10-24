// T-263: attractions. The real save reaches a final with the attractions purpose in it (its rail
// trips, solve time, attraction boardings); the marks show in Manhattan at zoom 13 and the Met's
// hover gives its name, visits a day and a share by train; the purpose filter on Attractions with
// Destinations bubbles; then a new game with a short line at Ueno, the Tokyo National Museum's
// hover. Screenshots to artifacts/trainworld/t263/ (light, or SHADE=dark). About four minutes.
//   node trainworld/checks/t263-attractions.mjs [url]   (from the maps root; default localhost:6123)
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { launchChrome, demandFinal, importSave, acrossLine, collectErrors, REAL_SAVE, ARTIFACTS, NOISE } from './lib.mjs';

const BASE = (process.argv[2] ?? 'http://localhost:6123/').replace(/\/?$/, '/');
const SHADE = process.env.SHADE ?? 'light';
const OUT = `${ARTIFACTS}t263/`;
const MET = [-73.9632, 40.7794];
const UENO = [139.7764, 35.7188];
const TNM = [139.7766, 35.7188];
await mkdir(OUT, { recursive: true });

const { ctx, close } = await launchChrome({ prefix: 'tw-t263-' });
const results = [], log = {}, errors = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const strip = (l) => l.replace(/<[^>]+>/g, '');

/** The hover at an attraction's place, with the tag drawn for a screenshot around it. */
async function hoverShot(page, ll, file) {
  const at = await page.evaluate((ll) => window.tw.map.project(ll), ll);
  const lines = await page.evaluate(([x, y]) => window.twViews.hover(x, y), [at.x, at.y]);
  const box = await page.evaluate(() => window.tw.map.getContainer().getBoundingClientRect().toJSON());
  await page.mouse.move(box.left + at.x - 4, box.top + at.y - 4);
  await page.mouse.move(box.left + at.x, box.top + at.y, { steps: 3 });
  await page.waitForTimeout(900);
  await page.screenshot({ path: `${OUT}${file}`, clip: { x: box.left + at.x - 380, y: box.top + at.y - 260, width: 760, height: 520 } });
  return (lines ?? []).map(strip).filter(Boolean);
}
const marksShown = (page) => page.evaluate(() => [...document.querySelectorAll('.attraction-mark')].filter((m) => m.style.display !== 'none').length);

try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE);
  await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, SHADE);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.twViews, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  const t0 = Date.now();
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 100, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 250, lines: 100, timeout: 600000 });
  const solveS = (Date.now() - t0) / 1000;
  log.real = await page.evaluate(() => {
    const v = window.twDemand.view(), by = {};
    for (const [k, p] of v.purposes ?? []) by[k] = [...p.lines.values()].reduce((s, l) => s + l.ridersPerDay, 0);
    return { rail: Math.round(v.railTripsPerDay), round: v.round, boardings: by };
  });
  log.real.solveS = solveS;
  report('real save final', log.real.rail > 0, `(${log.real.rail.toLocaleString('en-US')} rail trips a day, import to final ${solveS.toFixed(1)} s, attraction boardings ${Math.round(log.real.boardings.attractions ?? 0).toLocaleString('en-US')})`);
  report('attractions board trains', (log.real.boardings.attractions ?? 0) > 0);

  // marks in Manhattan, and the Met's hover
  await page.evaluate((ll) => void window.tw.map.jumpTo({ center: ll, zoom: 13.5 }), MET);
  await page.waitForTimeout(8000);
  const n = await marksShown(page);
  report('marks in Manhattan', n > 5, `(${n})`);
  log.met = await hoverShot(page, MET, `t263-met-${SHADE}.png`);
  report('the Met\'s hover', /Metropolitan/.test(log.met[0] ?? '') && log.met.some((l) => /visits per weekday/.test(l)) && log.met.some((l) => /% by train/.test(l)), JSON.stringify(log.met));

  // the purpose filter on Attractions with Destinations bubbles
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    const t = window.tw;
    t.display.value = { ...t.display.value, commuters: true, commuterEnd: 'work' };
    ui.purposeFilter.value = 'attractions';
  });
  await page.evaluate(() => void window.tw.map.jumpTo({ center: [-73.975, 40.755], zoom: 11.6 }));
  await page.waitForTimeout(12000);
  await page.mouse.move(5, 5);
  await page.screenshot({ path: `${OUT}t263-filter-${SHADE}.png` });
  report('marks with the filter at zoom 11.6', (await marksShown(page)) > 3);

  // Tokyo: a new game, a short line at Ueno
  const tyo = await ctx.newPage();
  collectErrors(tyo, errors, NOISE);
  await tyo.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await tyo.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.twViews, undefined, { timeout: 180000 });
  await tyo.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 12 }), UENO);
  if ((await acrossLine(tyo, UENO, 3)) === null) throw new Error('no line at Ueno');
  await tyo.waitForFunction(() => { const v = window.twDemand.view(); return v && v.final && v.lines.size === 1; }, undefined, { timeout: 300000, polling: 500 });
  await tyo.evaluate((ll) => void window.tw.map.jumpTo({ center: ll, zoom: 14 }), TNM);
  await tyo.waitForTimeout(8000);
  log.tnm = await hoverShot(tyo, TNM, `t263-ueno-${SHADE}.png`);
  report('the Tokyo National Museum\'s hover', /Tokyo National Museum/.test(log.tnm[0] ?? '') && log.tnm.some((l) => /visits per weekday/.test(l)), JSON.stringify(log.tnm));
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await writeFile(`${OUT}t263-attractions-${SHADE}.json`, JSON.stringify(log, null, 1));
  await close();
}
process.exit(results.length && results.every(Boolean) ? 0 : 1);
