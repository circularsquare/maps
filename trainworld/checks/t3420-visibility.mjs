// T-3420: hiding lines and track levels on the map. On the real save: the Lines tab's eye button shows a
// toggle per row; hiding lines and a group takes their strokes, trains and picks off the map and shows
// "Show all" (laid out all along, so nothing moves); Settings' Show levels hides a level's track, lines,
// stations and picks; the hidden lines survive the save file and its load, the levels do not. Shots of
// the Lines tab, Settings and the map, light and dark. From the maps root:
//   node trainworld/checks/t3420-visibility.mjs [url]   (about two minutes; default localhost:6113)
// Screenshots to artifacts/trainworld/t3420/.
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, GL, launchChrome, collectErrors, importSave } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6113/';
const OUT = join(ARTIFACTS, 't3420');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3420-', size: [1400, 1000], args: GL, deviceScaleFactor: 2 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { if (!sessionStorage.getItem('t3420')) { sessionStorage.setItem('t3420', '1'); localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 14 })); } } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 4, undefined, { timeout: 120000 });
  await page.waitForTimeout(3000);
  await page.evaluate(async () => {
    window.__m = { v: await import('/src/game/lineVisibility.ts'), ui: await import('/src/game/ui.ts'), w: await import('/src/game/world.ts'), live: await import('/src/map/live.ts'), pick: await import('/src/map/pick.ts'), vis: await import('/src/map/visibility.ts'), groups: await import('/src/game/lineGroups.ts') };
  });
  const S = (f, arg) => page.evaluate(`(async () => (${f.toString()})({ ...window.__m, tw: window.tw }, ${JSON.stringify(arg ?? null)}))()`);
  const shade = async (s) => { await page.evaluate((s) => { window.tw.shade.value = s; }, s); await page.waitForTimeout(500); };
  const frame = () => page.waitForTimeout(700);

  // a few lines: the first two, and a group if there is one
  const info = await S(({ tw, groups, vis }) => {
    const w = tw.world.value, ls = w.save.lines;
    const g = [...groups.lineGroups(ls).entries()][0];
    const levels = {};
    const n = tw.renderer?.net ?? null;
    return { ids: ls.slice(0, 8).map((l) => [l.id, l.name]), group: g ? [g[0], g[1].map((l) => l.id)] : null, nodeLevels: [...w.nodes.values()].reduce((m, x) => ((m[x.level] = (m[x.level] ?? 0) + 1), m), levels) };
  });
  console.log('lines', JSON.stringify(info.ids), 'group', JSON.stringify(info.group), 'node levels', JSON.stringify(info.nodeLevels));
  await S(({ ui, tw }) => { ui.tab.value = 'lines'; tw.selection.value = null; });
  await frame();
  const actions = async () => page.evaluate(() => { const r = document.querySelector('.line-list-actions').getBoundingClientRect(); return [r.top, r.height, document.querySelector('.lines-box').getBoundingClientRect().top]; });
  const before = await actions();
  await page.screenshot({ path: join(OUT, 'lines-plain.png'), clip: { x: 0, y: 0, width: 440, height: 560 } });
  await page.click('.list-vis .btn.icon');
  await frame();
  check(await page.$$eval('.lines-box .vis-toggle', (b) => b.length) > 4, 'the eye button shows a toggle on each row');
  // hide two lines by their rows' eyes and a group's
  const rows = await page.$$('.lines-box .rows > li:not(.group-row) .vis-toggle');
  await rows[0].click();
  await rows[2].click();
  const gEye = await page.$('.lines-box .rows > li.group-row .vis-toggle');
  if (gEye) await gEye.click();
  await frame();
  const hid = await S(({ v }) => [...v.hiddenLines.peek()]);
  console.log('hidden', JSON.stringify(hid));
  check(hid.length >= 2, `lines hidden: ${hid.length}`);
  const after = await actions();
  check(JSON.stringify(before) === JSON.stringify(after), `Show all appears without moving anything: ${before} / ${after}`);
  check(await page.$eval('.show-all', (b) => getComputedStyle(b).visibility === 'visible'), 'Show all is seen');
  for (const s of ['light', 'dark']) {
    await shade(s);
    await page.mouse.move(700, 900);
    await page.screenshot({ path: join(OUT, `lines-hidden-${s}.png`), clip: { x: 0, y: 0, width: 440, height: 560 } });
  }
  await shade('light');

  // the renderer: a hidden line's strokes are passed over by a pick
  const pickTest = await S(({ tw, live, pick, vis }, id) => {
    const ov = live.live.overlay, n = ov.renderer.net, L = n.lines;
    let i = -1;
    for (let k = 0; k < L.count; k++) if (String(L.colour[k]) === id) { i = k; break; }
    if (i < 0) return null;
    return { skipped: vis.pickSkip.get(L.seg)?.(i) ?? false, hiddenNums: [...vis.hidden.lines] };
  }, hid[0]);
  console.log('pick', JSON.stringify(pickTest));
  check(pickTest?.skipped === true, 'a hidden line\'s strokes are skipped by picks');
  // fit the map to the first hidden line's stops and shoot it
  await S(({ tw }, id) => { const l = tw.world.value.save.lines.find((x) => x.id === id); tw.selection.value = null; return l?.stops.length; }, hid[0]);
  await page.mouse.move(1300, 900);
  await page.screenshot({ path: join(OUT, 'map-hidden.png') });
  await S(({ v }) => v.showAllLines());
  await frame();
  await page.screenshot({ path: join(OUT, 'map-shown.png') });
  check(await page.$eval('.show-all', (b) => getComputedStyle(b).visibility === 'hidden'), 'Show all goes once every line shows');
  await S(({ v }, ids) => v.setLinesHidden(ids, true), hid);
  await frame();

  // levels
  await S(({ ui }) => { ui.tab.value = 'settings'; });
  await frame();
  await page.evaluate(() => document.querySelector('.level-show')?.scrollIntoView({ block: 'center' }));
  await frame();
  const lvBtns = await page.$$('.level-show .btn.lv');
  check(lvBtns.length === 7, 'Show levels has seven buttons');
  check(await page.$$eval('.level-show .btn.lv.on', (b) => b.length) === 7, 'all levels start shown');
  await lvBtns[4].click(); // -1
  await frame();
  const lv = await S(({ v, vis }) => ({ levels: [...v.hiddenLevels.peek()], bits: vis.hidden.levels }));
  check(lv.levels.length === 1 && lv.levels[0] === -1 && lv.bits === 1 << 2, `level -1 hidden: ${JSON.stringify(lv)}`);
  const sbox = await page.evaluate(() => { const r = document.querySelector('.level-show').getBoundingClientRect(); return { x: 0, y: Math.max(0, r.top - 260), width: 440, height: 340 }; });
  for (const s of ['light', 'dark']) {
    await shade(s);
    await page.mouse.move(700, 900);
    await page.screenshot({ path: join(OUT, `settings-${s}.png`), clip: sbox });
  }
  await shade('light');
  // track at -1 is skipped by picks, at 0 not
  const trackSkip = await S(({ live, vis }) => {
    const n = live.live.overlay.renderer.net, T = n.track, f = vis.pickSkip.get(T.seg);
    let under = 0, underSkipped = 0, ground = 0, groundSkipped = 0;
    for (let i = 0; i < T.count; i++) {
      const c = T.level[i], q0 = Math.floor(c / 256), q1 = c - q0 * 256;
      if (q0 === q1 && q0 === 96) { under++; if (f?.(i)) underSkipped++; }
      if (q0 === q1 && q0 === 64) { ground++; if (f?.(i)) groundSkipped++; }
    }
    return { under, underSkipped, ground, groundSkipped, stationsHidden: [...Array(n.stationCount).keys()].filter((i) => live.live.overlay.renderer.stationHidden(i)).length, stations: n.stationCount };
  });
  console.log('levels', JSON.stringify(trackSkip));
  check(trackSkip.under === 0 || trackSkip.underSkipped === trackSkip.under, 'every piece at -1 is skipped by picks');
  check(trackSkip.ground > 0 && trackSkip.groundSkipped === 0, `no piece at -2 is skipped (${trackSkip.ground})`);
  await S(({ ui }) => { ui.tab.value = 'lines'; });
  await page.mouse.move(1300, 900);
  await frame();
  await page.screenshot({ path: join(OUT, 'map-level-hidden.png') });
  await S(({ v }) => v.toggleLevel(-1));
  await frame();
  await page.screenshot({ path: join(OUT, 'map-level-shown.png') });

  // hidden lines in the save file and back with it
  const bytes = await page.evaluate(async () => { const b = await (await import('/src/workers/clockClient.ts')).exportForShare(); return b ? Array.from(b) : null; });
  const header = await page.evaluate(async (data) => (await import('/src/workers/saveFile.ts')).decodeSave(Uint8Array.from(data)), bytes).then((r) => r?.header ?? r, () => null);
  console.log('header hidden', JSON.stringify(header?.hidden));
  check(JSON.stringify([...(header?.hidden ?? [])].sort()) === JSON.stringify([...hid].sort()), 'the save holds the hidden lines');
  await S(({ v }) => { v.takeHiddenLines([]); v.hiddenLevels.value = new Set([-2]); });
  const was = await page.evaluate(() => window.tw.client.lastLoad);
  await importSave(page, Buffer.from(bytes));
  await page.waitForFunction((was) => window.tw.client.lastLoad && window.tw.client.lastLoad !== was, was, { timeout: 120000 });
  await page.waitForTimeout(1500);
  const back = await S(({ v }) => [...v.hiddenLines.peek()]);
  check(JSON.stringify(back.sort()) === JSON.stringify([...hid].sort()), `hidden lines come back with the load: ${JSON.stringify(back)}`);
} finally {
  await chrome.close();
}
const real = errors.filter((e) => !/Failed to load resource/.test(e));
check(real.length === 0, `no page errors${real.length ? ': ' + real.slice(0, 5).join(' | ') : ''}`);
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
