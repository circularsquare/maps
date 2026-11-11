// T-3186: long ferry lines in the game itself. Two ports are placed with the Ferry tool far apart (Stockholm to
// Mariehamn, 135 km, and Kobe to Takamatsu, 111 km: both once answered "No water path found" because the leg's
// tiles did not fit the router's limit) and the route must come back whole. Pictures go in
// artifacts/trainworld/t3186-*.png. From the maps root, on a Vite of your own, claim cores by hand first:
//   node trainworld/checks/t3186-ferry-long-game.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6117/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const CASES = [
  { name: 'stockholm-mariehamn', a: [18.1000, 59.3200], b: [19.9370, 60.0990], min: 130, max: 220 },
  { name: 'kobe-takamatsu', a: [135.1930, 34.6760], b: [134.0510, 34.3500], min: 108, max: 200 },
];

const chrome = await launchChrome({ prefix: 'tw-t3186-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.ferry, undefined, { timeout: 180000 });
  const box = await (await page.$('#map')).boundingBox();
  const settle = async () => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(800); };
  const at = async (p) => { const [x, y] = await page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p); return { x: Math.round(box.x + x), y: Math.round(box.y + y) }; };
  const state = () => page.evaluate(() => { const f = window.tw.tools.ferry; return { ports: f.ports.length, route: f.route ? { ok: f.route.ok, m: f.route.m, refused: f.route.legs.filter((l) => l.reason).map((l) => l.reason) } : null }; });
  await page.click('button.tab[aria-label="Build"]');
  for (const c of CASES) {
    await page.evaluate(([a, b]) => window.tw.map.fitBounds([[Math.min(a[0], b[0]), Math.min(a[1], b[1])], [Math.max(a[0], b[0]), Math.max(a[1], b[1])]], { padding: 120, duration: 0 }), [c.a, c.b]);
    await settle();
    await page.click('button.tool-ferry');
    await page.waitForTimeout(300);
    for (const [k, p] of [c.a, c.b].entries()) {
      const q = await at(p);
      await page.mouse.move(q.x - 4, q.y - 4);
      await page.mouse.move(q.x, q.y, { steps: 4 });
      await page.waitForTimeout(2500);
      await page.mouse.click(q.x, q.y);
      await page.waitForFunction((n) => window.tw.tools.ferry.ports.length >= n, k + 1, { timeout: 40000 }).catch(() => {});
    }
    check((await state()).ports === 2, `${c.name}: two ports placed`, `${(await state()).ports}`);
    await page.waitForFunction(() => { const f = window.tw.tools.ferry; return f.route && f.routeKey === f.berthKey(); }, undefined, { timeout: 120000 }).catch(() => {});
    const s = await state(), km = (s.route?.m ?? 0) / 1000;
    check(!!s.route?.ok && s.route.refused.length === 0 && km >= c.min && km <= c.max, `${c.name}: the route comes back whole`, `${km.toFixed(1)} km ${s.route?.refused.join(',') ?? 'no route'}`);
    await page.screenshot({ path: join(ARTIFACTS, `t3186-${c.name}.png`), clip: box });
    await page.keyboard.press('Escape');
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);
  }
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
console.log(fail ? `FAIL ${fail}` : 'PASS all');
process.exit(fail ? 1 : 0);
