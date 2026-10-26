// T-1499 on the real save: no colour key beside the Speed and Elevation track views, and the "+N daily
// rides" indicator at the map's top centre (not in the Lines tab). Screenshots light and dark into
// artifacts/trainworld/t1499/. From the maps root, with a running dev server:
//   node trainworld/checks/t1499-legends-note.mjs <port>
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, GL, NOISE, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1499');
await mkdir(OUT, { recursive: true });
const BASE = `http://localhost:${process.argv[2] ?? 6188}/`;
const { ctx, close } = await launchChrome({ prefix: 'tw-t1499-', size: [1400, 900], args: GL, deviceScaleFactor: 2 });
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
  const shade = (s) => page.evaluate((x) => { document.documentElement.dataset.shade = x; window.tw.shade.value = x; }, s);
  const both = async (fn) => { for (const s of ['light', 'dark']) { await shade(s); await page.waitForTimeout(500); await fn(s); } await shade('light'); };

  // ---- the track rows
  for (const mode of ['speed', 'height']) {
    await page.evaluate((m) => { window.tw.display.value = { ...window.tw.display.value, trackColour: m }; }, mode);
    await page.waitForTimeout(800);
    check(`${mode} view: no colour key`, (await page.locator('.track-key, .speed-key, .levels-key').count()) === 0);
    const bar = await page.locator('[aria-label="Track display"]').boundingBox();
    check(`${mode} view: the track row is the four buttons and the fold icon`, (await page.locator('[aria-label="Track display"] .btn').count()) === 4 && bar.width < 330, `width ${Math.round(bar.width)}`);
    await both(async (s) => {
      await page.mouse.move(2, 2);
      const box = await page.locator('.map-top-left').boundingBox();
      await page.screenshot({ path: join(OUT, `rows-${mode}-${s}.png`), clip: { x: Math.max(0, box.x - 10), y: Math.max(0, box.y - 10), width: Math.max(box.width + 20, 460), height: box.height + 20 } });
    });
  }
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, trackColour: 'line' }; });

  // ---- the indicator
  await page.evaluate(async () => { const u = await import('/src/game/ui.ts'); u.tab.value = 'lines'; });
  const pick = await page.evaluate(() => {
    const w = window.tw.world.value;
    return w.save.lines.filter((l) => w.lineStats[l.id]?.status === 'running' && l.stops.length > 5).map((l) => ({ id: l.id, num: l.num, name: l.name, stops: l.stops }))[0];
  });
  const noteOf = () => page.evaluate(() => {
    const n = document.querySelector('.rides-notes .rides-note');
    return n ? { text: n.querySelector(':scope > span:not(.chip)').textContent.trim(), tip: n.dataset.tip, cls: n.className, chip: n.querySelector('.chip')?.dataset.tip } : null;
  });
  const inLines = () => page.evaluate(() => document.querySelectorAll('#dock .rides-note').length);
  const waitNote = async (ms) => { const t = Date.now(); let n = null; while (Date.now() - t < ms && !n) { n = await noteOf(); if (!n) await page.waitForTimeout(150); } return n; };
  check('no indicator before any edit', (await noteOf()) === null);
  const edits = [['drop-last', pick.stops.slice(0, -1)], ['restore', pick.stops]];
  for (const [what, stops] of edits) {
    await page.evaluate(({ num, stops }) => window.tw.client.edit({ op: 'setStops', line: num, stops: stops.map(Number) }), { num: pick.num, stops });
    const n = await waitNote(90000);
    const sign = n?.text.startsWith('+') ? 'gain' : 'fall';
    check(`${what}: a note shows, signed, quiet when it is a fall`, !!n && /^[+−][\d,]+ daily rides?$/.test(n.text) && n.cls.includes('down') === (sign === 'fall'), n?.text ?? '');
    check(`${what}: its chip is the line`, n?.chip === pick.name);
    check(`${what}: its hover gives from and to`, /Riders a day went from [\d.,]+[KM]? to [\d.,]+[KM]?/.test(n?.tip ?? ''), n?.tip ?? '');
    check(`${what}: nothing in the Lines tab`, (await inLines()) === 0);
    if (n) {
      await page.waitForTimeout(1000);
      await page.mouse.move(2, 2);
      const m = await page.locator('#map').boundingBox();
      const bar = await page.locator('.rides-notes').boundingBox();
      const dock = await page.locator('#dock').boundingBox();
      check(`${what}: it sits inside the map, clear of the dock`, bar.x >= m.x && bar.x + bar.width <= m.x + m.width && (dock.x >= bar.x + bar.width || dock.x + dock.width <= bar.x));
      for (const s of ['light', 'dark']) {
        await shade(s);
        await page.screenshot({ path: join(OUT, `note-${sign}-${s}.png`), clip: { x: m.x + m.width / 2 - 350, y: m.y, width: 700, height: 160 } });
        if (sign === 'gain') await page.screenshot({ path: join(OUT, `note-screen-${s}.png`) });
      }
      await shade('light');
    }
    await page.waitForTimeout(6000);
    check(`${what}: the note is gone after 5 s`, (await noteOf()) === null);
  }
  check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(fails ? 1 : 0);
