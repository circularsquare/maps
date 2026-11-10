// T-3068: screenshots of the top bar's mode split hover and a trip panel, on the real New York save. From the maps root:
//   node trainworld/checks/t3068-transit-shots.mjs [url]
import { readFile } from 'node:fs/promises';
import { ARTIFACTS, collectErrors, demandFinal, importSave, launchChrome, REAL_SAVE } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6195/';
const chrome = await launchChrome({ prefix: 'tw-t3068-', size: [1400, 1000] });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${BASE}?debug=1&new=1&demandWorkers=2&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: '#debug, .save-notice { display: none !important; }' });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { lines: 100, timeout: 600000 });
  const split = await page.evaluate(() => { const e = document.querySelector('.split-bar, .top-split, [class*=split]'); return e ? e.getBoundingClientRect().toJSON() : null; });
  if (split) {
    await page.mouse.move(split.left + 10, split.top + split.height / 2);
    await page.mouse.move(split.left + 20, split.top + split.height / 2);
    await page.waitForTimeout(800);
    await page.screenshot({ path: join(ARTIFACTS, 't3068-split-hover.png'), clip: { x: Math.max(0, split.left - 60), y: 0, width: 720, height: 300 } });
  } else console.log('no split bar found');
  await page.mouse.move(5, 600);
  const st = await page.evaluate(() => { const s = window.tw.world.value.save.stations; return { n: s.length, keys: Object.keys(s[0]), a: s[0], b: s[Math.floor(s.length / 2)] }; });
  console.log(JSON.stringify(st).slice(0, 400));
  const p = (s) => s.pos ?? [s.x, s.y];
  await page.evaluate(({ a, b }) => { window.tw.selection.value = { kind: 'trip', from: a, to: b }; }, { a: p(st.a), b: p(st.b) });
  await page.waitForTimeout(6000);
  await page.screenshot({ path: join(ARTIFACTS, 't3068-trip.png') });
  console.log(errors.length ? 'errors: ' + errors.slice(0, 3).join(' | ') : 'no errors');
} finally {
  await chrome.close();
}
