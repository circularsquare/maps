// T-1431: load a save on a dev server, wait for demand final, and record rail trips a weekday, the
// longer trips' riders, the time from import to final, Chrome's private memory (sampled), and on the
// China save the explorer's Beijing - Shanghai and two more pairs.
// Usage (from the tree's maps root): node <this> <base url> <save> <label>
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
const TREE = process.env.TREE ?? 'C:/Users/anita/projects/maps/.tw-trees/t1521/trainworld/checks/lib.mjs';
const { ARTIFACTS, launchChrome, collectErrors, importSave, demandFinal, registry, NOISE } = await import('file:///' + TREE);

const [BASE, SAVE, LABEL, NODES] = process.argv.slice(2);
const OUT = 'C:/Users/anita/projects/maps/artifacts/trainworld/T-1521';
await mkdir(OUT, { recursive: true });
const reg = registry();
const place = (id) => reg.cities.find((c) => c.id === id)?.place;
const MB = 1048576;
function chromeMB(profile) {
  const needle = profile.replace(/'/g, "''");
  const ps = `Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like '*${needle}*' } | ForEach-Object { $_.PrivatePageCount }`;
  try {
    const out = execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], { encoding: 'utf8' });
    return Math.round(out.trim().split(/\r?\n/).filter(Boolean).reduce((a, b) => a + Number(b), 0) / MB);
  } catch { return -1; }
}

const { ctx, close, profile } = await launchChrome({ prefix: 'tw-t1521-' });
const errors = [];
const out = { label: LABEL, save: SAVE, base: BASE };
let peak = 0, sampling = true;
const sampler = (async () => { while (sampling) { peak = Math.max(peak, chromeMB(profile)); await new Promise((r) => setTimeout(r, 3000)); } })();
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  out.solve_ms = { m: 0, o: 0, n: 0 };
  page.on('console', (msg) => { const m = /^T1431TMP (m|o) ([\d.]+)/.exec(msg.text()); if (m) { out.solve_ms[m[1]] += Number(m[2]); out.solve_ms.n++; } });
  await page.goto(`${BASE}${BASE.includes('?') ? '&' : '?'}debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const t0 = Date.now();
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  if (NODES) {
    await page.evaluate((nodes) => window.tw.client.edit({ op: 'fares', nodes }), JSON.parse(NODES));
    await page.waitForTimeout(3000);
  }
  out.nodes = await page.evaluate(() => window.tw.moneyState.value.fares.nodes);
  await demandFinal(page, { polling: 500, timeout: 2400000 });
  await page.waitForFunction(() => { const d = window.twDemand.days(); return d[0].view?.final; }, undefined, { timeout: 600000 });
  out.final_s = Math.round((Date.now() - t0) / 1000);
  out.peak_mb = peak;
  out.after_mb = chromeMB(profile);
  Object.assign(out, await page.evaluate(() => {
    const v = window.twDemand.days()[0].view;
    const by = {};
    for (const [p, d] of v.purposes ?? []) { by[p] = 0; for (const l of d.lines.values()) by[p] += l.ridersPerDay; }
    return { rail: v.railTripsPerDay, trips: v.tripsPerDay, by };
  }));
  const pairs = [['sha', 'bei'], ['sha', 'tia'], ['nan', 'bei'], ['wuh', 'cha']].filter(([a, b]) => place(a) != null && place(b) != null);
  if (/china/i.test(SAVE)) {
    out.pairs = {};
    for (const [a, b] of pairs) {
      out.pairs[`${a}-${b}`] = await page.evaluate(async ({ near, far }) => {
        const ps = await window.twDemand.client.marketTrip('home', new Uint32Array(0), far, -1, { place: near, gid: -1 });
        return ps && ps.map((p) => p && { trips: p.trips, rail: p.rail, served: p.served, why: p.why });
      }, { near: place(a), far: place(b) });
    }
  }
} catch (e) {
  out.error = String(e);
} finally {
  sampling = false;
  await sampler;
  out.errors = errors.slice(0, 20);
  await writeFile(join(OUT, `measure-${LABEL}.json`), JSON.stringify(out, null, 1));
  console.log(JSON.stringify({ ...out, pairs: out.pairs ? Object.fromEntries(Object.entries(out.pairs).map(([k, v]) => [k, v?.[1]])) : undefined }, null, 1));
  await close();
}
