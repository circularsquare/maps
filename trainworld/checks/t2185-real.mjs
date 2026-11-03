// T-2185: the merges in saves (shared track lines join from two or more edges, wire kind 12) and
// their waits at high demand: every line's merge rows from `lineDelays`, and the map's merge dots
// (75% and up). From the maps root:
//   node trainworld/checks/t2185-real.mjs [url] [save ...]
// Writes artifacts/trainworld/t2185/real-<save>.json.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2185');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6184/';
const saves = process.argv.length > 3 ? process.argv.slice(3) : [REAL_SAVE];
for (const save of saves) {
  const chrome = await launchChrome({ prefix: 'tw-t2185-', size: [1200, 800] });
  const errors = [];
  try {
    const page = await chrome.ctx.newPage();
    collectErrors(page, errors);
    await page.goto(`${url}?debug=1&new=1&demandWorkers=1`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
    await importSave(page, await readFile(save));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 0 && window.tw.renderer.tripCount > 0, undefined, { timeout: 300000 });
    await page.waitForFunction(() => !window.tw.clock.held.peek() && !window.tw.clock.capped.peek(), undefined, { timeout: 900000, polling: 1000 });
    const r = await page.evaluate(async () => {
      const tw = window.tw, w = tw.world.value;
      const at = new Map(); // node: { s per train, lines }
      for (const l of w.save.lines) {
        const d = await tw.client.query({ what: 'lineDelays', line: l.num, level: 0 });
        for (let k = 0; k + 4 < (d?.length ?? 0); k += 5) {
          if (d[k] !== 12) continue;
          const v = at.get(d[k + 1]) ?? { node: d[k + 1], s: 0, lines: [] };
          v.s = Math.max(v.s, d[k + 4]);
          v.lines.push(l.name);
          at.set(d[k + 1], v);
        }
      }
      const merges = [...at.values()].sort((a, b) => b.s - a.s).map((v) => { const n = w.nodes.get(v.node); return { ...v, s: +v.s.toFixed(1), lngLat: n ? [+n.lng.toFixed(5), +n.lat.toFixed(5)] : null }; });
      const dots = (w.capacity[0] ?? []).filter((m) => m.kind === 12).map((m) => ({ rho: +m.rho.toFixed(3), delayS: +m.delayS.toFixed(1), tph: +m.tph.toFixed(1), lngLat: tw.toLngLat(m.x, m.y).map((x) => +x.toFixed(5)) }));
      const kinds = {};
      for (const m of w.capacity[0] ?? []) kinds[m.kind] = (kinds[m.kind] ?? 0) + 1;
      return { lines: w.save.lines.length, merges, dots, dotKinds: kinds };
    });
    const name = save.replace(/.*[\\/]/, '').replace(/\W+/g, '_');
    console.log(name, 'lines', r.lines, 'merges', r.merges.length, 'merge dots', r.dots.length, 'dots by kind', JSON.stringify(r.dotKinds));
    console.log('  busiest', JSON.stringify(r.merges.slice(0, 5)));
    console.log('  dots', JSON.stringify(r.dots.slice(0, 8)));
    await writeFile(join(OUT, `real-${name}.json`), JSON.stringify({ save, ...r, errors }, null, 1));
    if (errors.length) console.log('  errors: ' + errors.slice(0, 3).join(' | '));
  } finally {
    await chrome.close();
  }
}
