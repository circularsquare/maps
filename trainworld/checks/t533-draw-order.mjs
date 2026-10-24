// T-533: where lines share track the shorter line is drawn on top (and wins the click) if it runs
// trains; a line with no trains goes under all that do. On the real save (T-007-real-nyc.save, 109
// lines): where a short service and a longer one run over the same ground, the short one wins the
// click; with its fleet set to zero it drops under; given trains again it is back on top. Saves
// artifacts/trainworld/t533-short-on-top.png and t533-no-trains-under.png.
//   node trainworld/checks/t533-draw-order.mjs [url]     (from the maps root; default localhost:6130)
// Prints PASS or FAIL lines and exits 1 on failure. About a minute.
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, REAL_SAVE, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6130/';

const { ctx, close } = await launchChrome({ prefix: 'tw-t533-', size: [1200, 800], args: ANGLE });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twViews && window.twDemand?.client, undefined, { timeout: 180000 });
  const bytes = await readFile(REAL_SAVE);
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(2500);

  // Ground two lines both cover (the real save gives each line its own edges, so overlap is found
  // on the drawn polylines, 6 m cells): the pair whose shorter line is shortest, both running
  // trains, the longer at least 1.5 times as long. `pts`: up to 8 places on it, as lng/lat.
  const pick = await page.evaluate(() => {
    const w = window.tw.world.value, st = w.lineStats, pts = window.tw.renderer.net.edgePts;
    const runs = (id) => st[id].trainsNeeded.reduce((a, b) => a + b, 0) > 0;
    const CELL = 6, cells = new Map();
    for (const e of w.edges) {
      const p = pts.get(e.id);
      if (!p || !e.lines.length) continue;
      for (let i = 0; i + 1 < p.length / 3; i++) {
        const x0 = p[i * 3], y0 = p[i * 3 + 1], x1 = p[i * 3 + 3], y1 = p[i * 3 + 4];
        const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / CELL));
        for (let k = 0; k < n; k++) {
          const x = x0 + ((x1 - x0) * k) / n, y = y0 + ((y1 - y0) * k) / n;
          const key = Math.round(x / CELL) + ',' + Math.round(y / CELL);
          const c = cells.get(key) ?? { x, y, lines: new Set() };
          for (const l of e.lines) c.lines.add(l);
          cells.set(key, c);
        }
      }
    }
    const pairs = new Map();
    for (const c of cells.values()) {
      if (c.lines.size !== 2) continue;
      const [a, b] = [...c.lines].sort((p, q) => st[p].km - st[q].km);
      if (!runs(a) || !runs(b) || !(st[a].km > 1 && st[b].km > 1.5 * st[a].km)) continue;
      const k = a + ' ' + b, v = pairs.get(k) ?? { short: a, long: b, cells: [] };
      v.cells.push(c);
      pairs.set(k, v);
    }
    const ok = [...pairs.values()].filter((v) => v.cells.length >= 12);
    ok.sort((p, q) => st[p.short].km - st[q.short].km || q.cells.length - p.cells.length);
    const best = ok[0];
    if (!best) return { pairs: pairs.size };
    const step = Math.max(1, Math.floor(best.cells.length / 8));
    const at = best.cells.filter((_, i) => i % step === 0).slice(0, 8).map((c) => window.tw.toLngLat(c.x, c.y));
    return { short: best.short, long: best.long, shortKm: st[best.short].km, longKm: st[best.long].km, overlapCells: best.cells.length, at, names: [st[best.short], st[best.long]].map((_, i) => w.save.lines.find((l) => l.id === [best.short, best.long][i]).name) };
  });
  if (!pick.short) throw new Error('no overlapping pair found: ' + JSON.stringify(pick));
  console.log('pair', JSON.stringify({ ...pick, at: undefined }));
  await page.evaluate((p) => void window.tw.map.jumpTo({ center: p.at[Math.floor(p.at.length / 2)], zoom: 15 }), pick);
  await page.waitForTimeout(2500);
  const box = await page.evaluate(() => window.tw.map.getContainer().getBoundingClientRect().toJSON());

  /** the lines a click wins at those places (where the click is a line) */
  const winners = () => page.evaluate((p) => {
    const out = [];
    for (const ll of p.at) {
      const q = window.tw.map.project(ll);
      const s = window.tw.pick(window.tw.overlay, q.x, q.y, false);
      if (s?.kind === 'line') out.push(String(s.line));    }
    return out;
  }, pick);
  const settled = async () => {
    await demandFinal(page, { strict: false });
    await page.waitForFunction(() => !document.querySelector('.loading-card'), undefined, { timeout: 300000, polling: 500 });
    await page.waitForTimeout(1500);
  };
  const shot = async (name) => {
    await settled();
    const c = await page.evaluate((p) => { const q = window.tw.map.project(p.at[Math.floor(p.at.length / 2)]); return { x: q.x, y: q.y }; }, pick);
    await page.screenshot({ path: `artifacts/trainworld/${name}`, clip: { x: Math.max(0, box.left + c.x - 250), y: Math.max(0, box.top + c.y - 190), width: 500, height: 380 } });
  };
  const detail = (w) => `(short ${pick.short} ${pick.names[0]} ${pick.shortKm.toFixed(1)} km, long ${pick.long} ${pick.names[1]} ${pick.longKm.toFixed(1)} km, ${pick.overlapCells} overlapping cells, clicks ${w.join(',')})`;

  /** the drawn order (bottom first) holds: lines with no trains first, then in each group longest to shortest, ties by id */
  const orderOk = () => page.evaluate(() => {
    const w = window.tw.world.value, st = w.lineStats, ids = window.tw.renderer.net.lineIds;
    const key = (id) => [st[id].trainsNeeded.some((t) => t > 0) ? 1 : 0, -st[id].km, Number(id)];
    let bad = 0;
    for (let i = 1; i < ids.length; i++) {
      const a = key(ids[i - 1]), b = key(ids[i]);
      if (a[0] > b[0] || (a[0] === b[0] && (a[1] > b[1] + 1e-6 || (Math.abs(a[1] - b[1]) <= 1e-6 && a[2] > b[2])))) bad++;
    }
    return { lines: ids.length, bad, idle: ids.filter((id) => !st[id].trainsNeeded.some((t) => t > 0)).length };
  });

  const first = await winners();
  const o1 = await orderOk();
  report('drawn order: no-train lines under, then longest to shortest', o1.bad === 0, `(${o1.lines} lines, ${o1.idle} without trains, ${o1.bad} out of order)`);
  report('short line wins the click over a long line', first.length > 0 && first.every((id) => id === pick.short), detail(first));
  await shot('t533-short-on-top.png');

  // its fleet to zero: it goes under the long line
  const ok = await page.evaluate(async (p) => {
    const { edit } = await import('/src/game/world.ts');
    const r = await edit({ op: 'setTrainCount', line: Number(p.short), trains: [0, 0, 0] });
    return r.ok;
  }, pick);
  await page.waitForFunction((p) => window.tw.world.value.lineStats[p.short].trainsNeeded.every((t) => t === 0), pick, { timeout: 30000 });
  await page.waitForTimeout(2500);
  const second = await winners();
  const o2 = await orderOk();
  const rank = await page.evaluate((p) => window.tw.renderer.net.lineIds.indexOf(p.short), pick);
  report('a line with no trains drops under', ok && second.length > 0 && !second.includes(pick.short) && o2.bad === 0 && o2.idle > 0 && rank < o2.idle, `(edit ${ok}, drawn ${rank} of ${o2.lines}, ${o2.idle} without trains, ${o2.bad} out of order, clicks ${second.join(',')})`);
  await shot('t533-no-trains-under.png');

  // trains again: back on top
  await page.evaluate(async (p) => {
    const { edit } = await import('/src/game/world.ts');
    await edit({ op: 'setTrainCount', line: Number(p.short), trains: [2, 2, 2] });
  }, pick);
  await page.waitForFunction((p) => window.tw.world.value.lineStats[p.short].trainsNeeded.some((t) => t > 0), pick, { timeout: 30000 });
  await page.waitForTimeout(2500);
  const third = await winners();
  const o3 = await orderOk();
  report('given trains again it is back on top', third.length > 0 && third.every((id) => id === pick.short) && o3.bad === 0, `(${o3.bad} out of order, clicks ${third.join(',')})`);
} finally {
  await close();
}
process.exit(results.length && results.every(Boolean) ? 0 : 1);
