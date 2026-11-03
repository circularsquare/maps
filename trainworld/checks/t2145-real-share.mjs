// T-2145: each real network's train share of the trips within its city (the city view's Within bar:
// rail / (rail + walk + drive)), on a weekday, Saturday, Sunday and the average day, which
// app/src/game/realShare.ts marks on that bar. Rerun after a demand change that moves the shares, and
// paste the printed rows into REAL_SHARE.
//   node trainworld/checks/t2145-real-share.mjs [url] [id ...]     (from the maps root)
// Each data/packs/real/<id>.save (sdj tor seo lon par tyo nyc; name some to run fewer) is imported into a
// `?new=1&opened=all&demandWorkers=1&paused=1` game, waited to the average day's demand final, and every
// zone with train trips read. With no url it starts its own Vite (port T2145_PORT or 6173). About 25
// minutes for all seven on one worker (New York and Tokyo take most); a worker needs about 2.5 GB.
// Writes artifacts/trainworld/t2143/realshare-<id>.json and prints REAL_SHARE's rows.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { MAPS, ARTIFACTS, launchChrome, importSave, demandFinal, startVite } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2143');
const ALL = ['sdj', 'tor', 'seo', 'lon', 'par', 'tyo', 'nyc'];
const DAYS = [[0, 'weekday'], [1, 'saturday'], [2, 'sunday'], [6, 'average']];

const args = process.argv.slice(2);
const url = args.find((a) => /^https?:/.test(a));
const named = args.filter((a) => !/^https?:/.test(a));
const ids = named.length ? named : ALL;
await mkdir(OUT, { recursive: true });
const t0 = Date.now();
const log = (s) => console.log(`${((Date.now() - t0) / 1000).toFixed(0).padStart(5)} s  ${s}`);
const vite = url ? null : await startVite(Number(process.env.T2145_PORT ?? 6173));
const base = url ?? vite.url;
const rows = [];
try {
  for (const id of ids) {
    const { ctx, close } = await launchChrome({ prefix: `tw-t2145-${id}-` });
    try {
      const page = ctx.pages()[0] ?? await ctx.newPage();
      await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'average' }, v: 2 })); } catch { /* none */ } });
      await page.goto(`${base}${base.includes('?') ? '&' : '?'}debug=1&new=1&opened=all&demandWorkers=1&paused=1`);
      await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
      await importSave(page, await readFile(join(MAPS, `trainworld/data/packs/real/${id}.save`)));
      await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 900000 });
      log(`${id}: save in, waiting for demand`);
      const t = Date.now();
      await demandFinal(page, { timeout: 4 * 3600000, polling: 5000, lines: 3 });
      const solveS = (Date.now() - t) / 1000;
      const g = await page.evaluate((DAYS) => {
        const days = window.twDemand.days(), out = {};
        for (const [k, name] of DAYS) {
          const d = days[k];
          if (!d?.cities) continue;
          for (const [zone, v] of d.cities) {
            const train = v.railTripsPerDay, walk = v.tripsPerDay * v.split.walk / 100, drive = v.tripsPerDay * v.split.drive / 100;
            const all = train + walk + drive;
            if (!(train > 0)) continue;
            (out[zone] ??= {})[name] = { share: all > 0 ? 100 * train / all : 0, split: v.split, train, all, final: d.view?.final ?? null };
          }
        }
        return { zones: out, lines: window.tw.world.value.save.lines.length, stations: window.tw.world.value.save.stations.length };
      }, DAYS);
      await writeFile(join(OUT, `realshare-${id}.json`), JSON.stringify({ id, at: new Date().toISOString(), solveS, ...g }, null, 1) + '\n');
      const z = g.zones[id];
      if (z) rows.push(`  ${id}: [${DAYS.map(([, n]) => Math.round((z[n]?.share ?? 0) * 10) / 10).join(', ')}],`);
      const others = Object.entries(g.zones).filter(([k, v]) => k !== id && v.average?.share >= 0.5).map(([k, v]) => `${k} ${v.average.share.toFixed(1)}%`);
      log(`${id}: final in ${solveS.toFixed(0)} s; ${id} ${z?.average?.share.toFixed(1) ?? '-'}% on the average day${others.length ? `; other zones over 0.5%: ${others.join(', ')}` : ''}`);
    } catch (e) {
      log(`${id}: FAILED ${e.message.split('\n')[0]}`);
    } finally {
      await close();
    }
  }
} finally {
  vite?.child.kill();
}
console.log('\nREAL_SHARE rows (weekday, Saturday, Sunday, average day):');
for (const r of rows) console.log(r);
