// T-2606: the network a save sends its demand workers, and the market packs they open, as plain
// JSON for the native dump (t2606.rs). t756-reach.mjs's capture, cut short: it stops once the whole
// solve's network has gone out and the packs have stopped opening (the figures are not needed).
//   URL=http://localhost:8800/trainworld/ node trainworld/notes/T-2606/capture.mjs <save> <out.json>
import { readFileSync, writeFileSync } from 'node:fs';
import { launchChrome, importSave } from '../../checks/lib.mjs';

const [save, out] = process.argv.slice(2);
const base = process.env.URL ?? 'http://localhost:8800/trainworld/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t2606-', size: [1200, 800], args: [] });
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    window.__c = { solves: [], markets: new Set(), packs: new Set(), lastOpen: 0 };
    const orig = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (m, t) {
      try {
        if (m && (m.kind === 'init' || m.kind === 'open') && !m.bare && !m.aux) {
          for (const p of m.packs ?? []) window.__c.packs.add(p.id);
          for (const p of m.intercity ?? []) window.__c.markets.add(p.id);
          window.__c.lastOpen = performance.now();
        }
        if (m && m.kind === 'solve' && !m.only?.length && m.net?.lineN?.length) {
          const n = m.net, c = (v) => Array.from(v);
          window.__c.solves.push({ at: performance.now(), version: m.version,
            net: { frame: n.frame, fare: n.fare, stXY: c(n.stXY), complex: c(n.complex), lineN: c(n.lineN), stops: c(n.stops), times: c(n.times), tph: c(n.tph), cars: c(n.cars), hopKm: c(n.hopKm) },
            ids: m.ids ? JSON.parse(JSON.stringify(m.ids, (k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v))) : null });
        }
      } catch (e) { console.error('t2606 hook', e); }
      return orig.call(this, m, t);
    };
  });
  await page.goto(`${base}?debug=1&new=1&demandWorkers=1&perfOff=focus&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);
  await importSave(page, readFileSync(save));
  const t0 = Date.now();
  for (;;) {
    const s = await page.evaluate(() => ({ n: window.__c.solves.length, since: performance.now() - Math.max(window.__c.lastOpen, window.__c.solves.at(-1)?.at ?? 0), m: window.__c.markets.size }));
    if (s.n && s.since > 25000) break;
    if (Date.now() - t0 > 8 * 60000) throw new Error('no settled network');
    await page.waitForTimeout(3000);
  }
  const last = await page.evaluate(() => ({ ...window.__c.solves.at(-1), markets: [...window.__c.markets], packs: [...window.__c.packs] }));
  writeFileSync(out, JSON.stringify({ save, frame: last.net.frame, fare: last.net.fare, markets: last.markets, packs: last.packs, ids: last.ids, ...Object.fromEntries(['stXY', 'complex', 'lineN', 'stops', 'times', 'tph', 'cars', 'hopKm'].map((k) => [k, last.net[k]])) }));
  console.log(`captured ${last.net.stXY.length / 2} stations, ${last.net.lineN.length} lines, ${last.packs.length} city packs, ${last.markets.length} market packs, in ${Math.round((Date.now() - t0) / 1000)} s`);
} finally {
  await close();
}
