// T-279: the real save's figures on a dev server (rail trips a weekday, the market's, and named
// corridors' trips by mode), for a before/after of the routed drives. Start the server with and
// without TW_MARKET_DRIVES and run this against each:
//   node trainworld/notes/T-279/save_figures.mjs http://localhost:6150/ out.json
import { writeFile } from 'node:fs/promises';
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, launchChrome, importSave, demandFinal, registry } from '../../checks/lib.mjs';

const [BASE = 'http://localhost:6150/', OUT] = process.argv.slice(2);
const reg = registry();
const named = (n) => reg.market.places.findIndex((p) => p.name === n);
const PAIRS = [['New York', 'Boston'], ['New York', 'Philadelphia'], ['New York', 'Washington'], ['New York', 'Albany'], ['Boston', 'Philadelphia']]
  .map(([a, b]) => [a, b, named(a), named(b)]).filter((p) => p[2] >= 0 && p[3] >= 0);
const { ctx, close } = await launchChrome({ prefix: 'tw-t279-' });
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
  await page.click('.start-menu .start-new');
  await page.click('.start-menu .new-game-cities [data-city="nyc"]');
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu'), undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 500, lines: 100, timeout: 900000 });
  await page.waitForTimeout(2000);
  const out = await page.evaluate((pairs) => {
    const d = window.twDemand.days()[0], v = d.view, ic = d.intercity, c = ic?.corridors;
    const fig = (a, b) => {
      if (!c?.table) return null;
      const t = c.table;
      const lr = new Map();
      c.railAt.forEach((j, k) => lr.set(j, [c.longerRail[k], c.commuteRail[k]]));
      for (let j = 0; j < t.n; j++) {
        if (!((t.from[j] === a && t.to[j] === b) || (t.from[j] === b && t.to[j] === a))) continue;
        const [longRail, comRail] = lr.get(j) ?? [0, 0];
        const tl = t.longer[j] * c.longerScale, cm = t.commutes[j] * c.commuteScale, rest = tl - longRail;
        return { longer: tl, longerRail: longRail, air: rest * t.air[j], car: rest * t.car[j], commutes: cm, commuteRail: comRail };
      }
      return null;
    };
    return {
      rail: v.railTripsPerDay, trips: v.tripsPerDay, round: v.round,
      inter: ic && { trips: ic.tripsPerDay, rail: ic.railTripsPerDay, commutes: ic.commuteTripsPerDay, commuteRail: ic.commuteRailPerDay },
      pairs: Object.fromEntries(pairs.map(([na, nb, a, b]) => [`${na} - ${nb}`, fig(a, b)])),
      workers: window.twDemand.start.workers,
    };
  }, PAIRS);
  console.log(JSON.stringify(out, null, 1));
  if (OUT) await writeFile(OUT, JSON.stringify(out, null, 1));
} finally {
  await close();
}
