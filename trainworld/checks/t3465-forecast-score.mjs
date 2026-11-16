// T-3465: a region's points are its forecast weekday's, shown and sent everywhere. On the real save: the score equals the
// weekday row of pointsPerDay; it does not move while the clock runs (game hours pass, so ledgers and the day kind change);
// an edit changes it once the solve answers, and a reload (the save imported again) gives the same score back.
//   node trainworld/checks/t3465-forecast-score.mjs [url]     (about 8 minutes: three demand finals on the real save)
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, ANGLE, REAL_SAVE } from './lib.mjs';
const BASE = process.argv[2] ?? 'http://localhost:6144/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const { ctx, close } = await launchChrome({ prefix: 'tw-t3465-', size: [1300, 850], args: ANGLE });
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&demandWorkers=3&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const save = await readFile(REAL_SAVE);
  const read = () => page.evaluate(async () => {
    const p = await import('/src/game/points.ts');
    const a = p.pointsScore.value, b = p.pointsPerDay.value;
    let sum = 0, same = true;
    for (const [z, v] of a) { sum += v; if (b.size && Math.abs(v - (b.get(z)?.[0] ?? 0)) > 1) same = false; }
    return { sum: Math.round(sum), regions: a.size, same, hour: window.tw.moneyState.value?.hours.at(-1)?.hour ?? null };
  });
  await importSave(page, save);
  await demandFinal(page, { timeout: 400000, lines: 50 });
  await page.waitForTimeout(3000);
  const s0 = await read();
  check(s0.sum > 0 && s0.same, `score is the weekday row of the solve: ${s0.sum} points, ${s0.regions} regions`);
  await page.evaluate(() => { window.tw.clock.setMultiple(100); window.tw.clock.setRunning(true); });
  await page.waitForTimeout(25000);
  await page.evaluate(() => window.tw.clock.setRunning(false));
  await page.waitForTimeout(2000);
  const s1 = await read();
  check(s1.hour - s0.hour >= 24 && s1.sum === s0.sum, `clock moved (hour ${s0.hour} -> ${s1.hour}), score ${s0.sum} -> ${s1.sum}`);
  const v0 = await page.evaluate(() => window.twDemand.view().version);
  await page.evaluate(async () => {
    const w = window.tw.world.value;
    const l = w.save.lines.filter((x) => w.lineStats[x.id]?.status === 'running').sort((a, b) => w.lineStats[b.id].km - w.lineStats[a.id].km)[0];
    const tph = [l.tph.high, l.tph.medium, l.tph.low];
    await window.tw.client.edit({ op: 'setHeadway', line: Number(l.num), seconds: tph.map((n) => (n > 0 ? Math.round((3600 / n) * 3) : 0)) });
  });
  await demandFinal(page, { timeout: 400000, after: v0 });
  await page.waitForTimeout(3000);
  const s2 = await read();
  check(s2.sum !== s1.sum && s2.same, `an edit (a longest line's headway tripled) moves the score ${s1.sum} -> ${s2.sum}`);
  const v1 = await page.evaluate(() => window.twDemand.view().version);
  await importSave(page, save);
  await demandFinal(page, { timeout: 400000, after: v1 });
  await page.waitForTimeout(3000);
  const s3 = await read();
  check(Math.abs(s3.sum - s0.sum) <= Math.max(5, s0.sum * 1e-4), `the save imported again gives the same score: ${s0.sum} -> ${s3.sum}`);
} finally { await close(); }
process.exit(fail ? 1 : 0);
