// T-409: two networks with no track between them, a line in Boston and one in Philadelphia, both
// within the regional market's reach. LINES picks which are built (bos, phl or both) in a new game;
// the run records every line's figures and the market's per period, then (with both) times an edit
// in each network to its final. QS="&perfOff=split" gives the market as one node (before T-409);
// SAVE=real builds them onto the real New York save.
//   BASE=http://localhost:6109/ LINES=bos,phl LABEL=both OUT=<dir> node trainworld/checks/t409-split.mjs
import { writeFile, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, collectErrors, importSave, demandFinal, buildLine, REAL_SAVE } from './lib.mjs';
// Playwright as smoke.mjs finds it: PLAYWRIGHT_MODULE, else `playwright`, else the npx cache
const OUT = process.env.OUT ?? '.';
const BASE = process.env.BASE ?? 'http://localhost:6109/';
const QS = process.env.QS ?? '';
const LABEL = process.env.LABEL ?? 'run';
const LINES = (process.env.LINES ?? 'bos,phl').split(',');
const REPS = Number(process.env.REPS ?? 2);
/** SAVE=real: the lines go onto the real New York save instead of a new game */
const REAL = process.env.SAVE === 'real';
const NETS = {
  bos: { name: 'Boston test', centre: [-71.08, 42.35], pts: [[-71.0552, 42.3523], [-71.0752, 42.3474], [-71.1001, 42.3480], [-71.1210, 42.3510]] },
  phl: { name: 'Philadelphia test', centre: [-75.16, 39.955], pts: [[-75.1817, 39.9557], [-75.1636, 39.9526], [-75.1530, 39.9510], [-75.1340, 39.9690]] },
};
const { ctx, close } = await launchChrome({ profile: join(OUT, `profile-${LABEL}`) });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), LABEL, ...a);
const page = await ctx.newPage();
const errors = [];
collectErrors(page, errors, null);

const waitSolved = (timeout = 240000) => demandFinal(page, { timeout, polling: 50 });

/** each line's figures by name, and the market's per period from the final results */
const figures = () => page.evaluate(() => {
  const d = window.twDemand, s = d.log.at(-1), v = d.view(), lines = {};
  for (const l of window.tw.world.value.save.lines) {
    const f = v.lines.get(l.id);
    lines[l.name] = f ? JSON.parse(JSON.stringify(f, (k, x) => (x instanceof Map ? [...x] : ArrayBuffer.isView(x) ? Array.from(x) : x))) : null;
    // the line as the solve was given it (run times, trains an hour), to tell input noise from figures
    const sl = s.svc.lines.find((x) => x.id === l.id);
    if (lines[l.name]) lines[l.name].input = sl ? JSON.stringify(sl, (k, x) => (ArrayBuffer.isView(x) ? Array.from(x) : x)) : null;
  }
  const market = s.results.map((r) => r && { summary: r.intercitySummary.map((x, i) => (i === 4 ? 0 : x)), parts: r.intercityParts, corridorShares: r.corridorShares, corridorRail: r.corridorRail });
  return { lines, market, clusters: s.clusters ?? null, cities: s.cities.join('+'), rail: v.railTripsPerDay };
});

/** an edit, then ms to the final of the solve it started */
const timed = (edit) => page.evaluate(async (edit) => {
  const d = window.twDemand, n0 = d.log.length;
  const t0 = performance.now();
  const r = await window.tw.client.edit(edit);
  const end = performance.now() + 240000;
  while (performance.now() < end) {
    const s = d.log.at(-1), v = d.view();
    if (d.log.length > n0 && s && v && v.version === s.version && v.final && !d.client.waiting) {
      return { ok: r.ok, final: Math.round(s.refinedAt - t0), clusters: s.clusters ?? null };
    }
    await new Promise((res) => setTimeout(res, 20));
  }
  return { timeout: true };
}, edit);

const build = async (net) => (await buildLine(page, net.pts, { name: net.name, trains: [10, 8, 6] })).line;

const out = { label: LABEL, qs: QS, lines: LINES, edits: {}, errors };
try {
  if (REAL) {
    // the real New York save first (as t326-clusters.mjs), in a fresh profile's slot
    await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    const bytes = await readFile(REAL_SAVE);
    const before = await page.evaluate(() => window.tw.lastSaved.peek());
    await importSave(page, bytes);
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 240000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
    await page.goto(`${BASE}?debug=1&continue=1${QS}&v=${Date.now()}`);
  } else await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1${QS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
  await waitSolved();
  const built = {};
  for (const id of LINES) {
    await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 12 }), NETS[id].centre);
    await page.waitForTimeout(2500);
    await waitSolved();
    built[id] = await build(NETS[id]);
    await page.waitForTimeout(1000);
    await waitSolved();
  }
  await page.waitForTimeout(1500);
  await waitSolved();
  out.figures = await figures();
  log('figures', JSON.stringify({ clusters: out.figures.clusters, cities: out.figures.cities, rail: out.figures.rail, market: out.figures.market.map((m) => m?.summary.slice(1, 3)) }));
  for (const id of LINES) {
    out.edits[id] = [];
    await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 12 }), NETS[id].centre);
    await page.waitForTimeout(1500);
    await waitSolved();
    for (let i = 0; i < REPS; i++) {
      await page.waitForTimeout(1500);
      const t = await timed({ op: 'setTrainCount', line: built[id], trains: i % 2 ? [10, 8, 6] : [12, 9, 6] });
      out.edits[id].push(t);
      log(`${id} edit`, JSON.stringify(t));
    }
  }
  await page.waitForTimeout(1500);
  await waitSolved();
  out.after = await figures();
} finally {
  await close();
}
await writeFile(join(OUT, `t409-${LABEL}.json`), JSON.stringify(out, null, 1));
log('errors', JSON.stringify(errors.slice(0, 5)));
