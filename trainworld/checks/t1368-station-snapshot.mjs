// T-1368: an edit that changes the built stations but no line's service still reaches demand. A new
// game builds a line in Manhattan; the line is removed, its stations removed one at a time with pauses
// longer than the clock's snapshot debounce (on a big save each edit takes that long), then a station on
// no line is built. After each, the network demand holds (`client.latest()`) must have as many stations
// as the save. Usage (maps root): node trainworld/checks/t1368-station-snapshot.mjs [dev server url].
// About a minute.
import { NOISE, launchChrome, collectErrors, acrossLine, demandFinal } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6141/';
let failed = 0;
const report = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1368-' });
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /glyphs/);
  await page.goto(`${BASE}${BASE.includes('?') ? '&' : '?'}debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const at = [-73.98, 40.75];
  await page.evaluate(([at]) => window.tw.map.jumpTo({ center: at, zoom: 13 }), [at]);
  const line = await acrossLine(page, at);
  if (line === null) throw new Error('no line built');
  await demandFinal(page, { lines: 0 });
  const counts = () => page.evaluate(() => ({ save: window.tw.world.value.save.stations.length, demand: window.twDemand.client.latest()?.stations.length ?? 0 }));
  const built = await counts();
  report('line built, demand has its stations', built.save === 3 && built.demand === 3, JSON.stringify(built));

  // the line first, then each of its stations on its own, each after the snapshot of the edit before
  await page.evaluate(async (line) => {
    const c = window.tw.client, pause = (ms) => new Promise((r) => setTimeout(r, ms));
    await c.edit({ op: 'removeLine', line });
    await pause(1500);
    for (const s of [...window.tw.world.value.save.stations]) { await c.edit({ op: 'removeStation', node: s.num }); await pause(1200); }
    await pause(1500);
  }, line);
  const removed = await counts();
  report('stations removed after their line reach demand', removed.save === 0 && removed.demand === removed.save, JSON.stringify(removed));

  // a station on no line, built
  await page.evaluate(async ([at]) => {
    const tw = window.tw, c = tw.client, pause = (ms) => new Promise((r) => setTimeout(r, ms));
    const a = tw.lngLatToPack(at[0] - 0.01, at[1]), b = tw.lngLatToPack(at[0] + 0.01, at[1]);
    await c.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: -1 }, to: { kind: 'free', x: b.x, y: b.y, level: -1 }, pis: [], single: false });
    await pause(1500);
    let best = null, bd = 1e12;
    for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - a.x, n.y - a.y); if (d < bd) { bd = d; best = n.id; } }
    await c.edit({ op: 'addStation', at: { kind: 'node', node: best }, platform: 200, name: '' });
    await pause(1500);
    await c.edit({ op: 'constructAll' });
    await pause(2000);
  }, [at]);
  const lone = await counts();
  report('a station on no line, built, reaches demand', lone.save === 1 && lone.demand === lone.save, JSON.stringify(lone));
  report('no console errors', errors.length === 0, errors.slice(0, 5).join(' | '));
} catch (e) {
  report('ran', false, e.message.split('\n')[0]);
} finally {
  await close();
}
console.log(failed ? `T1368 FAILED: ${failed}` : 'T1368 PASSED');
process.exit(failed ? 1 : 0);
