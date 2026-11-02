// T-1792: a line with no trains at the level in force has its bullet dimmed everywhere. On the real
// save one line ('subway 2') is set to 0 trains at the level now; the Lines tab, a station panel
// listing it and the related lines view of a line crossing it are shot light and dark, and the
// bullet's class and hover are checked. From the maps root:
//   node trainworld/checks/t1792-idle-bullets.mjs [url]   (about a minute)
// Screenshots to artifacts/trainworld/t1792/.
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, GL, launchChrome, collectErrors, importSave } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6159/';
const OUT = join(ARTIFACTS, 't1792');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t1792-', size: [1400, 1800], args: GL });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(4000);
  const shade = async (s) => { await page.evaluate((s) => { document.documentElement.dataset.shade = s; window.tw.shade.value = s; }, s); await page.waitForTimeout(400); };
  const info = await page.evaluate(async () => {
    const tw = window.tw, w = await import('/src/game/world.ts'), ck = await import('/src/game/clock.ts'), ty = await import('/src/game/types.ts'), ui = await import('/src/game/ui.ts');
    const lines = tw.world.value.save.lines;
    const l = lines.find((x) => x.name === 'subway 2') ?? lines.find((x) => /(^|\s)2$/.test(x.name)) ?? lines[1];
    const st = tw.world.value.lineStats[l.id];
    const lev = ck.demandAt(ck.lineSeconds(st));
    w.setTrainCount(l.id, lev, 0);
    ui.tab.value = 'lines';
    return { id: l.id, name: l.name, lev, before: st.trainsNeeded, sample: lines.slice(0, 5).map((x) => x.name) };
  });
  await page.waitForFunction((id) => window.tw.world.value.lineStats[id].trainsNeeded.some((n) => n === 0), info.id, { timeout: 60000 });
  await page.waitForTimeout(1500);
  console.log('line', info);
  const idleChips = () => page.$$eval('.chip.idle', (els) => els.map((e) => e.getAttribute('data-tip')));
  const tips = await idleChips();
  check(tips.length > 0, `Lines tab: ${tips.length} dimmed bullet(s)`);
  check(tips.every((t) => t === null || /Not running$/.test(t)), `hovers end with "Not running": ${JSON.stringify([...new Set(tips)])}`);
  const dim = await page.evaluate(() => { const e = document.querySelector('.chip.idle'); return e && getComputedStyle(e).opacity; });
  check(dim === '0.7', `idle opacity ${dim}`);
  // a running line's bullets are not dimmed
  const others = await page.$$eval('.chip:not(.idle)', (els) => els.length);
  check(others > 0, `${others} undimmed bullets`);
  const shotAll = async (name) => { for (const s of ['light', 'dark']) { await shade(s); await page.mouse.move(2, 2); await page.screenshot({ path: join(OUT, `${name}-${s}.png`), clip: { x: 0, y: 0, width: 420, height: 1800 } }); } await shade('light'); };
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tab.value = 'lines'; });
  await page.waitForTimeout(500);
  await shotAll('lines-tab');
  // a stop of line 2 in a complex with the most other lines (stops are linked into complexes)
  const station = await page.evaluate(async (id) => {
    const w = window.tw.world.value, wd = await import('/src/game/world.ts'), l2 = w.save.lines.find((l) => l.id === id);
    const best = l2.stops.map((sid) => [sid, wd.linesAt(sid).length]).sort((a, b) => b[1] - a[1])[0];
    window.tw.selection.value = { kind: 'station', station: best[0] };
    return { id: best[0], lines: best[1], name: w.save.stations.find((s) => s.id === best[0])?.name };
  }, info.id);
  console.log('station', station);
  check(station.lines >= 1, 'a station of line 2');
  await page.waitForTimeout(1500);
  check((await idleChips()).length > 0, 'station panel: a dimmed bullet');
  await shotAll('station-panel');
  // the related lines view of a running line, one that draws line 2
  let related = null;
  for (const n of ['4', '5', '3', '6', '1', '7']) {
    const ok = await page.evaluate((n) => { const l = window.tw.world.value.save.lines.find((x) => x.name === n); if (!l) return false; window.tw.selection.value = { kind: 'line', line: l.id }; return true; }, n);
    if (!ok) continue;
    await page.waitForSelector('button.related-toggle', { timeout: 30000 });
    if (!(await page.$('button.related-toggle.on'))) await page.click('button.related-toggle');
    await page.waitForSelector('.related .pill', { timeout: 30000 });
    await page.waitForTimeout(1200);
    if ((await page.$$eval('.related .chip.idle', (e) => e.length)) > 0) { related = n; break; }
  }
  console.log('related of', related);
  check(related !== null, 'related lines view: a dimmed bullet');
  await shotAll('related-lines');
  // the selected line's own panel
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, info.id);
  await page.waitForTimeout(1500);
  await shotAll('line-panel');
  // map tag: hover of a station tag builds lineBullets; check the DOM helper directly
  const tag = await page.evaluate(async (id) => {
    const tr = await import('/src/map/trackTagRows.ts');
    const l = window.tw.world.value.save.lines;
    return [tr.lineBullet(l.find((x) => x.id === id)).className, tr.lineBullet(l.find((x) => x.id !== id)).className];
  }, info.id);
  check(/idle/.test(tag[0]) && !/idle/.test(tag[1]), `map tag bullets: ${tag.join(' | ')}`);
  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
