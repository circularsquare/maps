// Run from maps: node trainworld/checks/t3302-player-saves.mjs [url]      (default http://localhost:6159; about 3 minutes)
// Two Tokyo saves a player sent (T-3302, fixtures/t3302-tokyo-day60.save and -day77.save) load and play. Day 77 hung
// on v0.3.0 at "Loading 17%": the clock worker never left the settle in `load` (T-3238's runaway wait behind a frequent
// express). Each save, on a fresh test page: the load answers within 2 minutes without an error, with every line; a
// demand final for it within 4 minutes; a train count edit on its busiest line answers within 90 s; the clock runs on
// at full speed; the game exports. Fails on any page error.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { launchChrome, importSave, demandFinal, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6159';
const FIX = (f) => fileURLToPath(new URL(`./fixtures/${f}`, import.meta.url));
const SAVES = [{ file: 't3302-tokyo-day60.save', lines: 22 }, { file: 't3302-tokyo-day77.save', lines: 27 }];

const { ctx, close } = await launchChrome({ prefix: 'tw-t3302-', size: [1200, 800], acceptDownloads: true });
let failed = false;
try {
  for (const s of SAVES) {
    const errors = [];
    const page = await ctx.newPage();
    collectErrors(page, errors);
    try {
      await page.goto(`${url}/?debug=1&new=1&paused=1&v=${Date.now()}`);
      await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client && window.twDemand?.view?.(), undefined, { timeout: 180000 });
      const before = await page.evaluate(() => { window.tw.client.lastLoad = null; return window.twDemand.view().version; });
      const t0 = Date.now();
      await importSave(page, readFileSync(FIX(s.file)));
      await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file', undefined, { timeout: 120000 })
        .catch(() => { throw new Error(`no load answer in 120 s (a hang in the clock worker's load)`); });
      const load = await page.evaluate(() => ({ ...window.tw.client.lastLoad, lines: window.tw.world.value.save.lines.length }));
      assert.ok(!load.error, `load error: ${load.error} ${load.why ?? ''}`);
      assert.equal(load.lines, s.lines, 'lines loaded');
      const loadS = (Date.now() - t0) / 1000;
      await demandFinal(page, { after: before, timeout: 240000 });
      const finalS = (Date.now() - t0) / 1000;
      const riders = await page.evaluate(() => [...window.twDemand.view().lines.values()].reduce((a, l) => a + (l.ridersPerDay ?? 0), 0));
      assert.ok(riders > 0, 'riders a day on the lines');
      // an edit re-settles the network: the busiest line one train more at each level
      const ed = await page.evaluate(async () => {
        const w = window.tw.world.value;
        let best = null;
        for (const l of w.save.lines) { const st = w.lineStats[l.id]; if (st?.trainsNeeded && (!best || st.trainsNeeded[0] > best.n[0])) best = { num: l.num, name: l.name, n: st.trainsNeeded }; }
        const t = performance.now();
        const r = await Promise.race([window.tw.client.edit({ op: 'setTrainCount', line: best.num, trains: best.n.map((x) => x + 1) }), new Promise((go) => setTimeout(() => go('timeout'), 90000))]);
        return { line: best.name, from: best.n, ms: performance.now() - t, timeout: r === 'timeout' };
      });
      assert.ok(!ed.timeout, `train count edit on ${ed.line} never answered`);
      const ran = await page.evaluate(async () => {
        const c = window.tw.clock, t = c.now();
        c.setSpeed(2); c.setRunning(true);
        await new Promise((go) => setTimeout(go, 15000));
        return c.now() - t;
      });
      assert.ok(ran > 600, `the clock ran ${ran.toFixed(0)} game s in 15 s`);
      const dl = page.waitForEvent('download', { timeout: 60000 });
      await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
      assert.ok((await (await dl).path()), 'exported');
      assert.deepEqual(errors, [], 'page errors');
      console.log(`PASS  ${s.file}  (${load.lines} lines; loaded ${loadS.toFixed(1)} s, demand final ${finalS.toFixed(1)} s, ${Math.round(riders).toLocaleString('en')} riders a day on its lines; ${ed.line} ${ed.from.join('/')} +1 answered in ${(ed.ms / 1000).toFixed(1)} s; clock ran ${(ran / 3600).toFixed(1)} game h; exported)`);
    } catch (e) {
      failed = true;
      console.log(`FAIL  ${s.file}  ${e.message.split('\n')[0]}`);
    } finally { await page.close(); }
  }
} finally { await close(); }
console.log(failed ? 't3302 player saves: FAILED' : 't3302 player saves: PASSED');
process.exit(failed ? 1 : 0);
