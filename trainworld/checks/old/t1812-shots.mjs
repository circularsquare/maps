// T-1812: screenshots of the five panel and map changes, light and dark, on the real save. From the
// maps root:
//   node trainworld/checks/t1812-shots.mjs <before|after> [url]   (about a minute)
// Writes artifacts/trainworld/t1812/<phase>-<what>-<shade>.png.
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, GL, launchChrome, collectErrors, importSave } from '../lib.mjs';

const PHASE = process.argv[2] ?? 'after';
const BASE = process.argv[3] ?? 'http://localhost:6187/';
const OUT = join(ARTIFACTS, 't1812');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const errors = [];
const chrome = await launchChrome({ prefix: 'tw-t1812-', size: [1400, 900], args: GL });
const ui = (shade) => (s) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: s, display: { dayShown: 'weekday' }, v: 2 })); } catch {} };
try {
  // 1. the start menu's logo
  for (const shade of ['light', 'dark']) {
    const page = await chrome.ctx.newPage();
    await page.addInitScript(ui(shade), shade);
    collectErrors(page, errors);
    await page.goto(`${BASE}?v=${Date.now()}`);
    await page.waitForSelector('canvas.start-logo', { timeout: 60000 });
    await page.waitForFunction(() => window.twLogo?.frames > 20, undefined, { timeout: 30000 });
    await page.waitForTimeout(800);
    await page.screenshot({ path: join(OUT, `${PHASE}-logo-${shade}.png`) });
    await page.close();
  }
  // the game
  const page = await chrome.ctx.newPage();
  await page.addInitScript(ui('light'), 'light');
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(4000);
  const shade = async (s) => { await page.evaluate((s) => { document.documentElement.dataset.shade = s; window.tw.shade.value = s; }, s); await page.waitForTimeout(500); };
  const shoot = async (name, clip) => {
    for (const s of ['light', 'dark']) { await shade(s); await page.mouse.move(700, 2); await page.screenshot({ path: join(OUT, `${PHASE}-${name}-${s}.png`), ...(clip ? { clip } : {}) }); }
    await shade('light');
  };
  const lineByName = (re) => page.evaluate((src) => { const l = window.tw.world.value.save.lines.find((x) => new RegExp(src).test(x.name)); return l ? { id: l.id, num: l.num, stops: l.stops } : null; }, re.source);
  const fit = (stops) => page.evaluate(async (stops) => { const loc = await import('/src/ui/locate.ts'); loc.locateLine(stops); }, stops);
  // 2. a selected line, and a selected group
  const four = await lineByName(/(^|\s)4$/), five = await lineByName(/(^|\s)5$/);
  check(!!four && !!five, 'lines 4 and 5 found');
  await fit(four.stops);
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, four.id);
  await page.waitForTimeout(1500);
  await shoot('map-line');
  // closer in, over the middle of the line
  await page.evaluate(async (stops) => {
    const wd = await import('/src/game/world.ts'), at = wd.stopsById(window.tw.world.value).get(stops[Math.floor(stops.length / 2)]);
    window.tw.map.jumpTo({ center: [at.lng, at.lat], zoom: 12.6 });
  }, four.stops);
  await page.waitForTimeout(1200);
  await shoot('map-line-close', { x: 302, y: 90, width: 1098, height: 810 });
  await page.evaluate(async ([a, b]) => { await window.tw.client.edit({ op: 'lineGroup', lines: [a, b], group: 'new' }); }, [four.num, five.num]);
  await page.waitForTimeout(1000);
  const g = await page.evaluate((n) => window.tw.world.value.save.lines.find((l) => l.num === n)?.group, four.num);
  check(!!g, `group made (${g})`);
  await page.evaluate((g) => { window.tw.selection.value = { kind: 'group', group: g }; }, g);
  await page.waitForTimeout(1500);
  await shoot('map-group');
  // 4. the line panel scrolled down: the title stays
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, four.id);
  await page.waitForTimeout(1200);
  const dock = { x: 0, y: 0, width: 560, height: 900 };
  await shoot('panel-line-top', dock);
  const sc = await page.evaluate(() => { const e = document.getElementById('inspector'); e.scrollTop = 500; return [e.scrollTop, e.scrollHeight, e.clientHeight]; });
  console.log('scroll', sc);
  check(sc[0] > 0, 'the line panel scrolls');
  await page.waitForTimeout(300);
  await shoot('panel-line-scrolled', dock);
  const titleTop = await page.evaluate(() => { const t = document.querySelector('#inspector .title'), i = document.getElementById('inspector'); return [t.getBoundingClientRect().top, i.getBoundingClientRect().top]; });
  check(Math.abs(titleTop[0] - titleTop[1]) < 20, `line title at the top of the panel when scrolled: ${titleTop.map((x) => Math.round(x))}`);
  // 5. a station
  const st = await page.evaluate(async () => {
    const wd = await import('/src/game/world.ts'), cv = await import('/src/game/complexView.ts');
    const w = window.tw.world.value;
    const best = w.save.stations.map((s) => [s.id, cv.complexOf(s.id).length, wd.linesAt(s.id).length]).sort((a, b) => b[1] - a[1] || b[2] - a[2])[0];
    window.tw.selection.value = { kind: 'station', station: best[0] };
    return best;
  });
  console.log('station', st);
  await page.waitForTimeout(2500);
  await page.evaluate(() => { document.getElementById('inspector').scrollTop = 0; });
  await page.waitForTimeout(300);
  await shoot('panel-station-complex', dock);
  const alone = await page.evaluate(async () => { const cv = await import('/src/game/complexView.ts'); const s = window.tw.selection.value; const m = cv.complexOf(s.station); cv.openAlone(m[0]); return m[0]; });
  await page.waitForTimeout(2000);
  await page.evaluate(() => { document.getElementById('inspector').scrollTop = 0; });
  await page.waitForTimeout(300);
  await shoot('panel-station-alone', dock);
  await page.evaluate(() => { const e = document.getElementById('inspector'); e.scrollTop = 400; });
  await page.waitForTimeout(300);
  await shoot('panel-station-alone-scrolled', dock);
  const text = await page.evaluate(() => document.getElementById('inspector').innerText);
  const gone = ['Trip from here', 'Commuters', 'Live nearby', 'Where they', 'Inter-city trips'];
  for (const t of gone) check(PHASE === 'before' || !text.includes(t), `station panel has no "${t}"`);
  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
