// T-2421: Liberty Island is its own land part, so trips off it have no car and no walk. From the maps root:
//   node trainworld/checks/t2421-liberty.mjs [url]
// A new game in New York; the trip explorer from Liberty Island to Battery Park and to Liberty State Park across
// the water: a No route row, "No road between these two places" under By car and By walking, no parking. Uses the
// Vite on the url (dev-nowatch.mjs, port 6196 by default); 2 demand workers, claim 3 cores. About 3 minutes.
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2421');
await mkdir(OUT, { recursive: true });
const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6196/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const LIBERTY = [-74.0450, 40.6893];
const ACROSS = [['Battery Park', [-74.0160, 40.7035]], ['Liberty State Park', [-74.0540, 40.7010]]];

const chrome = await launchChrome({ prefix: 'tw-t2421-', size: [1200, 800], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${BASE}?debug=1&new=1&city=nyc&demandWorkers=2&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.twDemand?.cities?.()?.get('nyc'), undefined, { timeout: 300000, polling: 1000 });
  await page.evaluate((c) => window.tw.map.jumpTo({ center: c, zoom: 13.5 }), LIBERTY);
  for (const [name, to] of ACROSS) {
    await page.evaluate(() => { window.tw.selection.value = null; });
    await page.evaluate(([a, b]) => {
      const p = window.tw.lngLatToPack(a[0], a[1]), q = window.tw.lngLatToPack(b[0], b[1]);
      window.tw.selection.value = { kind: 'trip', from: [p.x, p.y], to: [q.x, q.y] };
    }, [LIBERTY, to]);
    await page.waitForFunction(() => /By car/.test(document.querySelector('#inspector, .inspector')?.innerText ?? ''), undefined, { timeout: 120000, polling: 500 });
    await page.waitForTimeout(800);
    const text = await page.evaluate(() => document.querySelector('#inspector, .inspector')?.innerText ?? '');
    const noRoad = (text.match(/No road between these two places/g) ?? []).length;
    check(/No route/.test(text) && noRoad === 2 && !/Parking/.test(text), `Liberty Island to ${name}: a No route row, no road by car or on foot`,
      text.split('\n').filter((s) => /No route|No road|Train|Walk|Drive/.test(s)).join(' | ').slice(0, 300));
    await page.screenshot({ path: join(OUT, `liberty-${name.replace(/ /g, '-').toLowerCase()}.png`) });
  }
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2421 FAILED: ${fail}` : 'T-2421 PASSED');
process.exit(fail ? 1 : 0);
