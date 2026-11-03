// T-1849: how long each demand worker takes to make its corridor table the first time it has market
// figures. Needs the temporary `self.__t1849` marks in demand.worker.ts (removed after the task).
// MODE=fresh: a new game, a 3-station line in ZONE (lines.json of T-2122: jak, mai, tyo); MODE=save:
// SAVE (default the real save) imported. env: BASE (http://localhost:6179), WORKERS (2), TAG, SCR.
// Run from the tree's maps root: node trainworld/notes/T-1849/timing.mjs
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, demandFinal, importSave, REAL_SAVE } from '../../checks/lib.mjs';

const BASE = process.env.BASE ?? 'http://localhost:6179';
const MODE = process.env.MODE ?? 'fresh';
const ZONE = process.env.ZONE ?? 'jak';
const TAG = process.env.TAG ?? `${MODE}-${ZONE}`;
const SCR = process.env.SCR ?? 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/76ebbfee-75f0-49e0-be85-230cb2ef8d79/scratchpad/t1849';
const OUT = process.env.OUT ?? 'C:/Users/anita/projects/maps/artifacts/trainworld/t1849';
const W = process.env.WORKERS ?? 2;
const { ctx, close } = await launchChrome({ prefix: join(SCR, `prof-${TAG}-`), args: ['--enable-unsafe-swiftshader'] });
const page = await ctx.newPage();
page.on('pageerror', (e) => console.log('pageerror', e.message.slice(0, 200)));
const t0 = Date.now();
if (MODE === 'fresh') {
  const Z = JSON.parse(await readFile(new URL('../T-2122/lines.json', import.meta.url), 'utf8'))[ZONE];
  await page.goto(`${BASE}/?debug=1&paused=1&mode=sandbox&opened=all&new=1&demandWorkers=${W}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round >= 0, undefined, { timeout: 240000 });
  await page.evaluate((ll) => { window.tw.map.jumpTo({ center: ll, zoom: 11.2 }); }, Z.centre);
  await page.waitForFunction((id) => window.twDemand.client.start.loads.find((l) => l.item === id)?.openedAt, ZONE, { timeout: 300000 });
  await new Promise((r) => setTimeout(r, 4000));
  const v = await page.evaluate(() => window.twDemand.view()?.version ?? 0);
  await page.evaluate(async (L) => {
    const t = window.tw, run = async (op) => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
    const pts = L.stops.map((s) => t.lngLatToPack(s[1], s[2]));
    const tail = (a, b) => { const dx = a.x - b.x, dy = a.y - b.y, l = Math.hypot(dx, dy); return { x: a.x + 300 * dx / l, y: a.y + 300 * dy / l }; };
    const path = [tail(pts[0], pts[1]), ...pts, tail(pts.at(-1), pts.at(-2))];
    const nodes = [];
    for (let i = 1; i < path.length; i++) {
      const from = i === 1 ? { kind: 'free', x: path[0].x, y: path[0].y, level: L.level } : { kind: 'node', node: nodes.at(-1) };
      await run({ op: 'route', from, to: { kind: 'free', x: path[i].x, y: path[i].y, level: L.level }, pis: [], single: false });
      const e = t.world.value.edges.at(-1);
      if (i === 1) nodes.push(e.a);
      nodes.push(e.b);
    }
    const at = nodes.slice(1, -1);
    for (let i = 0; i < L.stops.length; i++) await run({ op: 'addStation', at: { kind: 'node', node: at[i] }, platform: 160, name: L.stops[i][0] });
    const line = (await run({ op: 'addLine', stops: at, name: 't1849', colour: '#2a7ab0' })).line;
    await run({ op: 'setTrainCount', line, trains: [10, 10, 5] });
    await run({ op: 'constructAll' });
  }, Z.lines[0]);
  await demandFinal(page, { after: v, timeout: 600000 });
} else {
  await page.goto(`${BASE}/?debug=1&paused=1&new=1&demandWorkers=${W}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 240000 });
  await importSave(page, await readFile(process.env.SAVE ?? REAL_SAVE));
  if (process.env.FINAL === '0') {
    // the tables alone: every solving worker's marks, not the save's whole solve
    for (let k = 0; ; k++) {
      let n = 0;
      for (const w of page.workers()) if ((await w.evaluate(() => self.__t1849?.length ?? 0).catch(() => 0)) >= 2) n++;
      if (n >= Number(W)) break;
      if (k > 600) throw Error('no tables');
      await new Promise((r) => setTimeout(r, 1500));
    }
  } else await demandFinal(page, { timeout: 900000, lines: 0 });
}
await new Promise((r) => setTimeout(r, 3000));
const workers = [];
for (const w of page.workers()) {
  const marks = await w.evaluate(() => self.__t1849 ?? null).catch(() => null);
  if (marks) workers.push(marks);
}
const rail = await page.evaluate(() => Math.round(window.twDemand.view()?.railTripsPerDay ?? -1));
const tables = await page.evaluate(() => [...window.twDemand.client.corridorTables.keys()]);
// per worker: [pairs, trips, commutes, modes ms, rows], then [columns, corridor_rows ms, at]
const out = { tag: TAG, wall: (Date.now() - t0) / 1000, rail, tables, workers };
console.log(JSON.stringify(out));
await writeFile(join(OUT, `timing-${TAG}.json`), JSON.stringify(out, null, 1));
await Promise.race([close(), new Promise((r) => setTimeout(r, 30000))]);
process.exit(0);
