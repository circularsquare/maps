// T-1283: how long service edits take in the clock worker on the real save. The real save, its
// line with the most Medium trains switched to headways, its Medium headway stepped down and up a
// minute at a time, then train counts set by hand; each edit waited for. Prints the worker's own
// times per edit (`apply`: the track model's edit; `total`: with the state it sends).
//   node trainworld/checks/t1283-edit-time.mjs [url] [--max-ms N]   (default http://localhost:6141/)
// Fails if any edit's total passes --max-ms (default 1000). Output: ../artifacts/trainworld/t1283/edit-time.json.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, collectErrors, importSave, demandFinal, REAL_SAVE, ARTIFACTS } from './lib.mjs';

const args = process.argv.slice(2);
const BASE = args.find((a) => /^https?:/.test(a)) ?? 'http://localhost:6141/';
const MAX = Number(args[args.indexOf('--max-ms') + 1]) || 1000;
const OUT = join(ARTIFACTS, 't1283');
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1283-', size: [1400, 900] });
const page = await ctx.newPage();
const errors = [];
collectErrors(page, errors);
const out = { base: BASE, errors, edits: [] };
let failed = false;
try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(1000);
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page);
  log('demand final');
  const line = await page.evaluate(() => {
    const w = window.tw.world.value;
    const lines = w.save.lines.filter((l) => { const s = w.lineStats[l.id]; return s && s.status === 'running' && !s.loop && l.stops.length > 2; });
    lines.sort((a, b) => w.lineStats[b.id].trainsNeeded[1] - w.lineStats[a.id].trainsNeeded[1]);
    window.tw.clock.setSpeed(1);
    window.tw.clock.setRunning(true);
    return { id: lines[0].id, name: lines[0].name, trains: w.lineStats[lines[0].id].trainsNeeded };
  });
  out.line = line;
  log('line', JSON.stringify(line));
  const ops = [
    ['headways', { op: 'setHeadway', line: Number(line.id), seconds: [300, 600, 1200] }],
    ...[540, 480, 420, 360, 420, 480, 540, 600].map((m) => [`medium ${m} s`, { op: 'setHeadway', line: Number(line.id), seconds: [300, m, 1200] }]),
    ['counts by hand', { op: 'setHeadway', line: Number(line.id), seconds: null }],
    ...[[32, 20, 9], [32, 21, 9], [32, 22, 9], [32, 12, 9]].map((t) => [`trains ${t.join('/')}`, { op: 'setTrainCount', line: Number(line.id), trains: t }]),
  ];
  for (const [what, op] of ops) {
    const r = await page.evaluate(async (op) => {
      const t = performance.now();
      const a = await window.tw.client.edit(op);
      const e = window.tw.client.editLog.at(-1);
      return { ok: a.ok, wall: Math.round(performance.now() - t), apply: Math.round(e?.worker?.apply ?? -1), total: Math.round(e?.worker?.total ?? -1) };
    }, op);
    out.edits.push({ what, ...r });
    log(what.padEnd(16), `apply ${r.apply} ms  total ${r.total} ms  answered ${r.wall} ms  ok ${r.ok}`);
    if (!r.ok || r.total > MAX) failed = true;
    await page.waitForTimeout(400);
  }
  const totals = out.edits.map((e) => e.total).sort((a, b) => a - b);
  out.median = totals[totals.length >> 1];
  out.max = totals.at(-1);
  log('total: median', out.median, 'ms, max', out.max, 'ms');
} catch (e) {
  failed = true;
  out.error = String(e?.stack ?? e);
  log('error', out.error);
} finally {
  await close();
}
await writeFile(join(OUT, 'edit-time.json'), JSON.stringify(out, null, 1));
if (errors.length) { failed = true; log('page errors', errors.slice(0, 5)); }
console.log(`${failed ? 'FAIL' : 'PASS'}  t1283 service edits on the real save within ${MAX} ms in the worker`);
process.exit(failed ? 1 : 0);
