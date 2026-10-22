// T-872: a public holiday runs as a weekend day for the lines in its country. Imports the real save
// (New York) into a ?new=1 game and moves the clock to the end of Wednesday, Thursday (the next
// Thanksgiving) and Saturday of that week, reading each day's boarding account (every
// line's past 24 hours) and, after Sunday's start, the money ledger. Thanksgiving's riders and fares
// should match Saturday's, not Wednesday's, and fares should follow riders. The calendar itself:
// the same day is a weekday in Canada and the UK.
//   node trainworld/checks/t872-holidays.mjs [url]     (from the maps root; default localhost:6145)
// Prints PASS or FAIL lines and exits 1 on failure. About two minutes, most of it demand.
import { readFile, mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const BASE = process.argv[2] ?? 'http://localhost:6145/';
const SAVE = fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright');
}

const { chromium } = await playwright();
const profile = await mkdtemp(join(tmpdir(), 'tw-t872-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1200, height: 800 }, args: ['--use-angle=d3d11'] });
let ok = true;
const check = (pass, text) => { console.log(`${pass ? 'PASS' : 'FAIL'}  ${text}`); ok &&= pass; };
try {
  const page = await ctx.newPage();
  const wait = (ms) => page.waitForTimeout(ms);
  await page.goto(`${BASE}?debug=1&new=1&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });

  const bytes = await readFile(SAVE);
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForFunction(() => { const d = window.twDemand, v = d?.view?.(); return v && v.final && !d.client.waiting && !window.tw.clock.held.peek(); }, undefined, { timeout: 300000, polling: 500 });
  await page.evaluate(() => window.tw.clock.setRunning(false));

  // the next Thanksgiving after the save's clock, and the same day in Canada and the UK
  const cal = await page.evaluate(async () => {
    const c = await import('/src/game/holidayCalendar.ts');
    const d = (await import('/src/game/holidays.json')).default;
    let thu = Math.floor(window.tw.clock.now() / 86400) + 1;
    while (!/Thanksgiving/.test(c.holidayName(d, 'US', thu) ?? '')) thu++;
    return { thu, kinds: ['US', 'CA', 'GB'].map((k) => c.kindOfIn(d, k)(thu)), name: c.holidayName(d, 'US', thu) };
  });
  const THU = cal.thu, WED = THU - 1, SAT = THU + 2;
  check(cal.kinds.join() === '1,0,0', `game day ${THU} is ${cal.name}: a weekend day in US, CA, GB ${cal.kinds.join(', ')}`);

  // every line's past 24 hours at 23:50 of `day`: riders and fares summed (the account and the money)
  const dayAt = (day) => page.evaluate(async (day) => {
    window.tw.clock.set(day * 86400 + 23 * 3600 + 3000);
    await new Promise((r) => setTimeout(r, 1500));
    const { queryClock } = await import('/src/workers/clockClient.ts');
    const { readToday } = await import('/src/game/today.ts');
    const t = readToday(await queryClock({ what: 'today' }));
    let riders = 0, fares = 0;
    for (const v of t.values()) { riders += v.riders; fares += v.fares; }
    return { riders, fares };
  }, day);
  const wed = await dayAt(WED), thu = await dayAt(THU), sat = await dayAt(SAT);
  for (const [k, v] of Object.entries({ wed, thu, sat })) console.log(`  ${k}: riders ${Math.round(v.riders)}, fares $${Math.round(v.fares)}`);
  const near = (a, b, tol) => Math.abs(a / b - 1) < tol;
  check(near(thu.riders, sat.riders, 0.02) && !near(thu.riders, wed.riders, 0.1), `Thanksgiving's boarding account is Saturday's: riders ${Math.round(thu.riders)} against ${Math.round(sat.riders)} (Wednesday ${Math.round(wed.riders)})`);
  check(near(thu.fares / thu.riders, sat.fares / sat.riders, 0.03) && near(thu.fares / wed.fares, thu.riders / wed.riders, 0.05), `fares follow the account: $/rider Thursday ${(thu.fares / thu.riders).toFixed(3)}, Saturday ${(sat.fares / sat.riders).toFixed(3)}, Wednesday ${(wed.fares / wed.riders).toFixed(3)}`);

  await page.evaluate((t) => window.tw.clock.set(t), (SAT + 1) * 86400 + 1800);
  await wait(1500);
  const led = await page.evaluate(() => window.tw.moneyState.value.ledger.map((d) => ({ day: d.day, fares: d.fares, rides: d.rides })));
  const by = new Map(led.map((d) => [d.day, d]));
  for (let d = WED; d <= SAT; d++) { const e = by.get(d); if (e) console.log(`  ledger day ${d}: fares ${e.fares.toFixed(3)} US$M, rides ${Math.round(e.rides)}`); }
  const [lw, lt, lf, ls] = [WED, THU, THU + 1, SAT].map((d) => by.get(d));
  check(!!(lw && lt && lf && ls) && near(lt.fares, ls.fares, 0.03) && near(lw.fares, lf.fares, 0.03) && !near(lt.fares, lw.fares, 0.1), `the ledger's Thanksgiving is a weekend day, Wednesday and Friday weekdays`);
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
process.exit(ok ? 0 : 1);
