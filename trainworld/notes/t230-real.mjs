// T-230 on New York's real network (T-007 save): for every line, demand's forecast figure (the
// busiest hour's average train on its fullest stretch, `fullestPct`) against the per-train
// account's day (`lineAccount`): its fullest train, trains leaving full and when, riders left on
// the platform. Lines the forecast put under 100% but whose trains leave riders behind are the
// ones whose panels disagreed. Needs a Vite dev server (BASE, default http://localhost:6932) and
// Playwright (PLAYWRIGHT_MODULE); PROFILE_DIR for the throwaway profile.
//   node trainworld/notes/t230-real.mjs   ->  artifacts/trainworld/notes-outputs/t230-real.json, artifacts/trainworld/T-230-real-line.png
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const ART = join(ROOT, '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:6932';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't230r-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const out = {};
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round === 3, undefined, { timeout: 300000 });
  const v0 = await page.evaluate(() => window.twDemand.view().version);
  const bytes = await readFile(join(ROOT, 'T-007-real-nyc.save'));
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view().round === 3 && window.twDemand.view().lines.size > 100, v0, { timeout: 600000 });
  await page.waitForTimeout(4000);
  out.lines = await page.evaluate(async () => {
    const tw = window.tw, w = tw.world.value, dv = window.twDemand.view(), rows = [];
    const hm = s => { const m = Math.round(s / 60) % 1440; return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); };
    for (const l of w.save.lines) {
      const d = dv.lines.get(l.id);
      const a = await tw.client.query({ what: 'lineAccount', line: l.num });
      if (!d || !a || a.length < 5) continue;
      const spans = [];
      for (let k = 0; k < a[4]; k++) spans.push(hm(a[5 + 2 * k]) + '-' + hm(a[6 + 2 * k]));
      rows.push({ name: l.name, id: l.id, forecastPct: d.fullestPct, accountPct: Math.round(100 * a[0]), fullTrains: a[1], leftBehind: Math.round(a[2]), longestWaitMin: Math.round(a[3] / 60), spans: spans.join(', ') });
    }
    return rows;
  });
  const disagree = out.lines.filter(r => r.forecastPct < 100 && r.leftBehind >= 1).sort((a, b) => b.leftBehind - a.leftBehind);
  out.summary = { lines: out.lines.length, forecastUnder100: out.lines.filter(r => r.forecastPct < 100).length, ofThoseLeavingRiders: disagree.length, forecastUnder100ButFull: out.lines.filter(r => r.forecastPct < 100 && r.fullTrains > 0).length };
  out.disagree = disagree.slice(0, 15);
  const pick = disagree.find(r => r.name.startsWith('subway')) ?? disagree[0];
  if (pick) {
    await page.evaluate(id => window.tw.select({ kind: 'line', line: id }), pick.id);
    await page.waitForTimeout(1500);
    await page.evaluate(() => { const d = document.querySelector('#dock-root'); const h = [...d.querySelectorAll('.h')].find(e => e.textContent.startsWith('Demand forecast')); h?.scrollIntoView(); });
    await page.waitForTimeout(500);
    await page.screenshot({ path: join(ART, 'T-230-real-line.png') });
    out.picked = pick;
  }
  out.errors = errors.slice(0, 10);
} finally {
  await ctx.close();
}
await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', 't230-real.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify({ summary: out.summary, disagree: out.disagree, picked: out.picked }, null, 1));
