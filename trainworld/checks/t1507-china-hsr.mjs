// T-1507: the China network's figures on the old test save and the high-speed one, side by side.
// Each save is loaded in headless Chrome (a sandbox, weekday view after demand final, as T-1504
// measured): rail trips a day, the market's longer-trip boardings (all lines and the high-speed
// lines 113-122), and the trip explorer's served share and rail share for Beijing - Shanghai,
// Guangzhou - Shenzhen and Wuhan - Changsha (each way, summed over the day's periods; "served" is
// the biggest period's share of trips with rail on offer). JSON per save in
// artifacts/trainworld/T-1507/<tag>.json. Starts its own Vite on PORT (default 8307) unless a base
// url is given. About 10-12 minutes a save.
//   node trainworld/checks/t1507-china-hsr.mjs [base url]          (from the maps root)
//   SAVES=t1507-all-china-hsr.save TAG=after-t1521 node ...         (one save; TAG suffixes the files)
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { basename, join } from 'node:path';
import { ARTIFACTS, MAPS, NOISE, launchChrome, collectErrors, importSave, demandFinal, registry, startVite } from './lib.mjs';

const OUT = join(ARTIFACTS, 'T-1507');
await mkdir(OUT, { recursive: true });
const SAVES = (process.env.SAVES ?? 't1156-all-china.save,t1507-all-china-hsr.save').split(',');
// the saves live at the main tree's maps root; an agent tree (maps/.tw-trees/<name>) looks there too
const savePath = (s) => [s, join(MAPS, s), join(MAPS, '..', '..', s)].find((p) => existsSync(p)) ?? s;
const place = (id) => registry().cities.find((c) => c.id === id).place;
const PAIRS = [['bei', 'sha', 'Beijing - Shanghai'], ['gua', 'shz', 'Guangzhou - Shenzhen'], ['wuh', 'chs', 'Wuhan - Changsha']].map(([a, b, name]) => ({ name, near: place(a), far: place(b) }));

let vite = null;
let BASE = process.argv[2];
if (!BASE) { vite = await startVite(Number(process.env.PORT ?? 8307)); BASE = vite.url; }

const measure = (page, pairs) => page.evaluate(async (PAIRS) => {
  const v = window.twDemand.days()[0].view;
  const longer = { all: 0, hsr: 0, perLine: {} };
  for (const [p, d] of v.purposes ?? []) {
    if (p !== 'longer') continue;
    for (const [id, l] of d.lines) { longer.all += l.ridersPerDay; if (Number(id) >= 113) { longer.hsr += l.ridersPerDay; longer.perLine[id] = Math.round(l.ridersPerDay); } }
  }
  const pairs = [];
  for (const { name, near, far } of PAIRS) {
    const ps = await window.twDemand.client.marketTrip('home', new Uint32Array(0), far, -1, { place: near, gid: -1 });
    const day = { trips: 0, rail: 0, air: 0, car: 0 };
    let big = null;
    for (const p of ps ?? []) {
      if (!p) continue;
      day.trips += p.trips; day.rail += p.rail; day.air += p.air; day.car += p.car;
      if (!big || p.trips > big.trips) big = p;
    }
    pairs.push({ name, trips: day.trips, rail: day.rail, air: day.air, car: day.car, railShare: day.trips ? day.rail / day.trips : 0,
      railOfRailCarAir: day.rail + day.car + day.air ? day.rail / (day.rail + day.car + day.air) : 0,
      served: big?.served ?? 0, railPerceived: big?.railPerceived ?? null, carPerceived: big?.carPerceived ?? null, why: big?.why ?? null,
      route: big?.routes?.[0]?.legs?.filter((l) => l.kind === 'ride').map((l) => l.line) ?? [] });
  }
  return { rail: v.railTripsPerDay, longer, pairs };
}, pairs);

const { ctx, close } = await launchChrome({ prefix: 'tw-t1507m-' });
const results = [];
try {
  for (const save of SAVES) {
    const errors = [];
    const page = await ctx.newPage();
    collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
    await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
    await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    const t0 = Date.now();
    await importSave(page, await readFile(savePath(save)));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
    await page.waitForTimeout(5000);
    await demandFinal(page, { polling: 1000, timeout: 1800000 });
    await page.waitForTimeout(2000);
    await demandFinal(page, { polling: 1000, timeout: 600000 });
    const r = { save, loadSecs: Math.round((Date.now() - t0) / 1000), ...(await measure(page, PAIRS)), errors: errors.slice(0, 10) };
    const tag = basename(save).replace(/\.save$/, '') + (process.env.TAG ? `-${process.env.TAG}` : '');
    await writeFile(join(OUT, `${tag}.json`), JSON.stringify(r, null, 1));
    results.push(r);
    console.log(`${save}: final in ${r.loadSecs} s, rail ${Math.round(r.rail).toLocaleString('en')} a day, longer ${Math.round(r.longer.all).toLocaleString('en')} (high speed ${Math.round(r.longer.hsr).toLocaleString('en')}), ${errors.length} page errors`);
    for (const p of r.pairs) console.log(`  ${p.name}: ${Math.round(p.trips)} trips, rail ${p.rail.toFixed(1)} (${(100 * p.railShare).toFixed(2)}%), served ${(100 * p.served).toFixed(1)}%${p.why ? `, ${p.why}` : ''}`);
    await page.close();
  }
} finally {
  await close();
  vite?.child.kill();
}
