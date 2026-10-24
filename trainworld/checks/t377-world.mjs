// T-377: the world market in the browser. The real New York save loaded and reloaded (startup to
// its final timed, the demand final and the market's rail read, WASM memory per worker); then a
// line in Toronto and one in London (three clusters), an edit in each timed to its final with the
// clusters it kept and solved; then every London cell selected as homes: its longer trips by place.
//   BASE=http://localhost:6111/ LABEL=after OUT=<dir> PLAYWRIGHT_MODULE=... node trainworld/checks/t377-world.mjs
// (writes OUT/world-LABEL.json; a fresh browser profile there). Run against 8800 for the last commit.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, collectErrors, importSave, demandFinal, buildLine, REAL_SAVE } from './lib.mjs';
const OUT = process.env.OUT ?? '.';
const BASE = process.env.BASE ?? 'http://localhost:6111/';
const LABEL = process.env.LABEL ?? 'run';
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ profile: join(OUT, `profile-${LABEL}`) });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), LABEL, ...a);
const page = await ctx.newPage();
const errors = [];
collectErrors(page, errors, null);

const waitSolved = (timeout = 300000) => demandFinal(page, { timeout, polling: 50 });
const figures = () => page.evaluate(() => {
  const d = window.twDemand, s = d.log.at(-1), v = d.view(), ic = d.intercity();
  return { rail: Math.round(v.railTripsPerDay), lines: v.lines.size, cities: s.cities.join('+'), markets: (s.markets ?? (s.market ? ['market'] : [])).join('+'),
    longer: ic && { trips: Math.round(ic.tripsPerDay), rail: Math.round(ic.railTripsPerDay), commutes: Math.round(ic.commuteTripsPerDay), commuteRail: Math.round(ic.commuteRailPerDay) },
    memMiB: s.memBytes.map((b) => Math.round(b / 1048576)), clusters: s.clusters ?? null };
});
const timed = (edit) => page.evaluate(async (edit) => {
  const d = window.twDemand, n0 = d.log.length, t0 = performance.now();
  const r = await window.tw.client.edit(edit);
  const end = performance.now() + 300000;
  while (performance.now() < end) {
    const s = d.log.at(-1), v = d.view();
    if (d.log.length > n0 && s && v && v.version === s.version && v.final && !d.client.waiting) return { ok: r.ok, final: Math.round(s.refinedAt - t0), clusters: s.clusters ?? null, cities: s.cities.join('+') };
    await new Promise((res) => setTimeout(res, 20));
  }
  return { timeout: true };
}, edit);
/** a line through `pts` (lng, lat), its stations at each point */
const build = async (pts, name) => (await buildLine(page, pts, { name, trains: [10, 8, 6] })).line;
const go = async (zone, center, zoom = 12) => {
  await page.evaluate(([c, z]) => void window.tw.map.jumpTo({ center: c, zoom: z }), [center, zoom]);
  await page.waitForFunction((zone) => window.twDemand.activeCity.peek() === zone, zone, { timeout: 60000, polling: 50 });
  await page.waitForTimeout(2500);
  await waitSolved();
};

const out = { label: LABEL, errors };
try {
  await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
  const bytes = await readFile(REAL_SAVE);
  const saved = await page.evaluate(() => window.tw.lastSaved.peek());
  await importSave(page, bytes);
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 240000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, saved, { timeout: 60000 });
  // startup: the reload to the real save's demand final
  const t0 = Date.now();
  await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
  await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
  await waitSolved();
  out.startupMs = Date.now() - t0;
  out.real = await figures();
  log('real save', out.startupMs, 'ms', JSON.stringify(out.real));
  const nyLine = await page.evaluate(() => window.tw.world.value.save.lines[0].num);
  out.realEdit = [];
  for (let i = 0; i < 2; i++) { await page.waitForTimeout(1500); out.realEdit.push(await timed({ op: 'setTrainCount', line: nyLine, trains: i % 2 ? [20, 12, 6] : [22, 14, 7] })); }
  log('real save edits', JSON.stringify(out.realEdit));
  // Toronto (Yonge, Union to Eglinton) and London (Oxford Circus to Liverpool Street)
  await go('tor', [-79.39, 43.67]);
  const tor = await build([[-79.3806, 43.6453], [-79.3832, 43.6560], [-79.3866, 43.6710], [-79.3904, 43.6870], [-79.3985, 43.7055]], 'Yonge test');
  await page.waitForTimeout(1000); await waitSolved();
  await go('lon', [-0.11, 51.515]);
  const lon = await build([[-0.1415, 51.5152], [-0.1300, 51.5165], [-0.1119, 51.5180], [-0.0977, 51.5146], [-0.0823, 51.5178]], 'Central test');
  await page.waitForTimeout(1000); await waitSolved();
  out.three = await figures();
  log('three cities', JSON.stringify(out.three));
  out.edits = {};
  for (const [name, line, trains] of [['lon', lon, [12, 9, 6]], ['tor', tor, [12, 9, 6]], ['nyc', nyLine, [21, 13, 7]]]) {
    out.edits[name] = [];
    for (let i = 0; i < 2; i++) { await page.waitForTimeout(1500); out.edits[name].push(await timed({ op: 'setTrainCount', line, trains: i % 2 ? [10, 8, 6] : trains })); }
    log(name, 'edits', JSON.stringify(out.edits[name]));
  }
  out.threeAfter = await figures();
  // every London cell as homes: the longer trips by place
  await go('lon', [-0.11, 51.515], 10);
  out.london = await page.evaluate(async () => {
    const d = window.twDemand, cells = await d.cityCells();
    if (!cells) return null;
    const sel = await d.intercitySelection('home', Uint32Array.from({ length: cells.n }, (_, i) => i));
    if (!sel) return null;
    const places = (f) => f.places.slice(0, 12).map((p) => ({ place: p.place, all: Math.round(p.all), rail: Math.round(p.rail) }));
    return { starts: { all: Math.round(sel.starts.all), rail: Math.round(sel.starts.rail), places: places(sel.starts) }, ends: { all: Math.round(sel.ends.all), places: places(sel.ends) } };
  });
  log('london homes', JSON.stringify(out.london));
} finally {
  await close();
}
await writeFile(join(OUT, `world-${LABEL}.json`), JSON.stringify(out, null, 1));
log('errors', JSON.stringify(errors.slice(0, 5)));
