// T-884: the loading card's bar keeps an even pace on the paths Anita enters a game by. Through the start
// menu, in headless Chrome with a temporary profile: a new game in New York; the real New York save
// (T-007) in a slot, picked twice (the first load with nothing remembered, the second with this slot's
// last wait); Anita's world save (artifacts/trainworld/world-map-day73.save, Toronto to Tokyo) the same.
// The bar's fill is read every 100 ms from the card. FAIL when over 25% of a wait is spent above 90%, the
// bar goes back, or its fill at 10/30/50/70/90% of the wait is over 30 points from that share (the old bar
// stood at 83% for two minutes of a Japan save's load); also prints the longest flat stretch.
//   node trainworld/checks/t884-loadbar.mjs [url]     (from the maps root; default localhost:6150)
// About 5 minutes. Writes ../artifacts/trainworld/t884-loadbar.json.
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, importSave, REAL_SAVE, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6150/';
const ART = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));
const WORLD = join(ART, 'world-map-day73.save');
const LIMIT = 0.25;

let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const out = [];

/** one profile: optionally a save put in its slot, then `loads` picks from the start menu */
async function run(label, save, loads) {
  const { ctx, close } = await launchChrome({ prefix: 'tw-t884-', size: [1300, 900], args: ANGLE });
  await ctx.addInitScript(() => {
    window.__lb = [];
    setInterval(() => {
      const fill = document.querySelector('.loading-card .load-fill');
      window.__lb.push([performance.now(), document.querySelector('.loading-card') ? 1 : 0, fill ? parseFloat(fill.style.width) : null]);
    }, 100);
  });
  try {
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    if (save) {
      await page.goto(`${BASE}?debug=1&continue=1&paused=1`);
      await page.waitForFunction(() => window.tw?.world?.value && window.tw.lastSaved, undefined, { timeout: 180000 });
      const bytes = await readFile(save);
      const before = await page.evaluate(() => window.tw.lastSaved.peek() ?? 0);
      await importSave(page, bytes);
      await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 240000 });
      await page.waitForTimeout(3000);
      await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
      await page.waitForFunction((b) => (window.tw.lastSaved.peek() ?? 0) > b, before, { timeout: 60000 });
      // the import's own wait is not one a player sees from the menu: forget it
      await page.evaluate(() => { localStorage.removeItem('anitabuilder-load-wait'); localStorage.removeItem('anitabuilder-load-solve'); });
    }
    for (let i = 0; i < loads; i++) {
      await page.goto(BASE);
      if (save) {
        await page.waitForSelector('.start-menu .start-save', { timeout: 60000 });
        await page.evaluate(() => { window.__pick = performance.now(); document.querySelector('.start-menu .start-save').click(); });
      } else {
        await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
        await page.click('.start-menu .start-new');
        await page.click('.start-menu .new-game-cities [data-city="nyc"]');
        await page.evaluate(() => { window.__pick = performance.now(); document.querySelector('.start-menu .new-game .start-new').click(); });
      }
      await page.waitForFunction(() => { const s = window.__lb; const i = s.findIndex((x) => x[1]); return i >= 0 && s.slice(i).some((x) => !x[1]); }, undefined, { timeout: 360000, polling: 200 });
      const s = await page.evaluate(() => window.__lb);
      const i0 = s.findIndex((x) => x[1]), i1 = s.findIndex((x, i) => i > i0 && !x[1]);
      const t0 = s[i0][0], t1 = s[i1][0], dur = t1 - t0;
      // time-weighted: each sample holds until the next (a blocked main thread shows the fill it froze at)
      let above = 0, back = 0, flat = 0, run = 0;
      for (let k = i0; k < i1; k++) {
        const f = s[k][2] ?? 0, dt = s[k + 1][0] - s[k][0];
        if (f >= 90) above += dt;
        if (k > i0 && f < (s[k - 1][2] ?? 0) - 0.01) back = Math.max(back, (s[k - 1][2] ?? 0) - f);
        if (k > i0 && f === s[k - 1][2]) { run += dt; flat = Math.max(flat, run); } else run = 0;
      }
      const at = (q) => { let f = 0; for (let k = i0; k < i1 && s[k][0] <= t0 + q * dur; k++) f = s[k][2] ?? 0; return Math.round(f); };
      const name = `${label}, load ${i + 1}`;
      const share = above / dur;
      out.push({ name, seconds: dur / 1000, above90: share, back, flat: flat / 1000, fill: [0.1, 0.3, 0.5, 0.7, 0.9].map(at) });
      report(`${name}: time above 90%`, share <= LIMIT, `(${(100 * share).toFixed(0)}% of ${(dur / 1000).toFixed(1)} s; fill ${[0.1, 0.3, 0.5, 0.7, 0.9].map(at).join('/')}; longest flat ${(flat / 1000).toFixed(1)} s)`);
      report(`${name}: never goes back`, back === 0, back ? `(fell ${back.toFixed(0)} points)` : '');
      const off = Math.max(...[0.1, 0.3, 0.5, 0.7, 0.9].map((q) => Math.abs(at(q) - 100 * q)));
      report(`${name}: even pace`, off <= 30, `(fill at most ${off} points from the share of the wait gone)`);
    }
    report(`${label}: no page errors`, errors.length === 0, errors.slice(0, 2).join(' | '));
  } finally {
    await close();
  }
}

await run('new game in New York', null, 1);
await run('real New York save', REAL_SAVE, 2);
if (existsSync(WORLD)) await run('world save', WORLD, 2);
else console.log(`SKIP  world save  (${WORLD} not found)`);
await writeFile(join(ART, 't884-loadbar.json'), JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);
