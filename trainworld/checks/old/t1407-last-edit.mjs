// T-1407: every edit's answer arrives with nothing after it (the clock worker's follow-up runs on
// its own, never waiting for a later message), in headless Chrome.
//   node trainworld/checks/t1407-last-edit.mjs [page url]      (from the maps root)
// A new game; a three-station line built and constructed; its trains set; then, each as the page's
// last action with the page idle: a train count, a headway, the line removed (the only line), an
// undo, a blueprint segment sent `early`. Each answer must come within LIMIT ms (default 20000).
import { launchChrome, acrossLine, ANGLE } from '../lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6183/';
const LIMIT = Number(process.env.LIMIT ?? 20000);

const { ctx, close } = await launchChrome({ prefix: 'tw-t1407l-', size: [1200, 800], args: ANGLE });
let failed = 0;
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools && window.twDemand?.client, undefined, { timeout: 180000 });
  const centre = [-73.985, 40.752];
  await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 13 }), centre);
  const line = await acrossLine(page, centre, 1.5);
  await page.evaluate(() => window.tw.client.edit({ op: 'constructAll' }));
  const final = () => page.waitForFunction(() => { const d = window.twDemand, v = d?.view?.(); return !!v?.final && !d.client.waiting; }, undefined, { timeout: 300000, polling: 500 });
  await final();
  await page.waitForTimeout(3000);
  // one edit as the last thing the page does; its answer's ms, or null past LIMIT
  const last = (op, early = false) => page.evaluate(async ([op, early, LIMIT]) => {
    const t0 = performance.now();
    const a = await Promise.race([window.tw.client.edit(op, early), new Promise((r) => setTimeout(() => r(null), LIMIT))]);
    return a ? { ok: a.ok, early: !!a.early, ms: Math.round(performance.now() - t0) } : null;
  }, [op, early, LIMIT]);
  const n = line;
  const cases = [
    ['train count', { op: 'setTrainCount', line: n, trains: [8, 6, 4] }],
    ['headway', { op: 'setHeadway', line: n, seconds: [300, 300, 300] }],
    ['headway off', { op: 'setHeadway', line: n, seconds: null }],
    ['the only line removed', { op: 'removeLine', line: n }],
    ['undo', { op: 'undo' }],
  ];
  for (const [what, op] of cases) {
    const r = await last(op);
    if (!r) failed++;
    console.log(`${r ? 'PASS' : 'FAIL'}  ${what}: ${r ? JSON.stringify(r) : `no answer in ${LIMIT} ms`}`);
    // the solve it starts, and the page idle again, before the next
    await page.waitForTimeout(1500);
    await final().catch(() => console.log(`  (no demand final after ${what})`));
    await page.waitForTimeout(3000);
  }
  // a blueprint segment answered early, its state on its own a moment later
  const seg = await page.evaluate(async ([c, LIMIT]) => {
    const t = window.tw, a = t.lngLatToPack(c[0], c[1] + 0.01), b = t.lngLatToPack(c[0] + 0.01, c[1] + 0.01);
    const edges = t.world.value.edges.length, t0 = performance.now();
    const r = await Promise.race([t.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false }, true), new Promise((r) => setTimeout(() => r(null), LIMIT))]);
    if (!r) return null;
    const ms = performance.now() - t0;
    while (performance.now() - t0 < LIMIT && t.world.value.edges.length === edges) await new Promise((r) => setTimeout(r, 5));
    return { ok: r.ok, early: !!r.early, ms: Math.round(ms), stateMs: t.world.value.edges.length === edges ? null : Math.round(performance.now() - t0) };
  }, [centre, LIMIT]);
  const segOk = seg && seg.ok && seg.stateMs !== null;
  if (!segOk) failed++;
  console.log(`${segOk ? 'PASS' : 'FAIL'}  blueprint segment early: ${JSON.stringify(seg)}`);
} finally {
  await close();
}
console.log(failed ? `T-1407 LAST EDIT FAILED: ${failed}` : 'T-1407 LAST EDIT PASSED');
process.exitCode = failed ? 1 : 0;
