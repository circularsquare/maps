// T-1285: the world's coarse water mask (pipeline/world_water.py) outside the zones' own masks.
// In a new game through the edit API: a line from Long Island to Lisbon drawn at ground level is
// bridged and priced over water (high-speed, about US$1.5T), and lowering it to the ground is refused;
// a short chord across a bay inside New York's pack (Raritan Bay) and one outside every zone (Bahia de
// Banderas, Mexico) both get a bridge, and both refuse the ground. With a second url (a server
// without the file, TW_WORLD_WATER pointing nowhere) the same Atlantic line as before the mask, and
// the real save's track priced on both, edge by edge. About two minutes.
// Usage: node trainworld/checks/t1285-world-water.mjs <url with the mask> [<url without>]

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const WITH = process.argv[2] ?? 'http://localhost:6185/';
const WITHOUT = process.argv[3];
const OUT = join(ARTIFACTS, 't1285');
mkdirSync(OUT, { recursive: true });
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) failed++; };

const ATLANTIC = [[-73.2, 40.8], [-9.2, 38.78]];
const RARITAN = [[-74.205, 40.43], [-74.15, 40.548]];
const BANDERAS = JSON.parse(process.env.T1285_BAY ?? '[[-105.40, 20.78], [-105.20, 20.62]]');

async function open(ctx, base) {
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  const sep = base.includes('?') ? '&' : '?';
  await page.goto(`${base}${sep}debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  return { page, errors };
}

/** A straight route between two lon, lat points at ground level; the new edges' US$M, km, levels.
 * Then each new edge lowered to the ground in turn, and whether that was refused. */
const route = (page, a, b, hsr) => page.evaluate(async ([a, b, hsr]) => {
  const tw = window.tw, c = tw.client;
  tw.map.jumpTo({ center: a, zoom: 12 });
  await new Promise((d) => setTimeout(d, 1500));
  const before = new Set(tw.world.value.edges.map((e) => e.id));
  const p = tw.lngLatToPack(a[0], a[1]), q = tw.lngLatToPack(b[0], b[1]);
  const r = await c.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false, hsr });
  if (!r.ok) return { error: r.issues.join(', ') };
  const mine = () => tw.world.value.edges.filter((e) => !before.has(e.id));
  let last = -1;
  for (let k = 0; k < 40; k++) {
    const cost = mine().reduce((s, e) => s + e.cost, 0);
    if (cost === last && k > 5) break;
    last = cost;
    await new Promise((d) => setTimeout(d, 250));
  }
  const es = mine();
  const out = { cost: es.reduce((s, e) => s + e.cost, 0) / 1e6, km: es.reduce((s, e) => s + e.lengthM, 0) / 1000, edges: es.length, ground: [] };
  for (const e of es) {
    const g = await c.edit({ op: 'edgeLevel', edge: e.id, level: 0 });
    out.ground.push({ ok: g.ok, issues: g.issues });
    if (g.ok) await c.edit({ op: 'undo' }).catch(() => {});
  }
  return out;
}, [a, b, hsr]);

/** The real save imported: each edge's price, US$ (to the cent). */
async function realPrices(page) {
  await importSave(page, readFileSync(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(3000);
  return page.evaluate(() => Object.fromEntries(window.tw.world.value.edges.map((e) => [e.id, Math.round(e.cost * 100) / 100])));
}

const { ctx, close } = await launchChrome({ prefix: 'tw-t1285-' });
const report = {};
try {
  const { page, errors } = await open(ctx, WITH);
  const atl = await route(page, ATLANTIC[0], ATLANTIC[1], true);
  report.atlantic = atl;
  check(!atl.error && atl.cost > 1.3e6 && atl.cost < 1.7e6, `Long Island to Lisbon, high-speed, drawn at ground level: ${atl.error ?? `US$${(atl.cost / 1e6).toFixed(3)}T over ${atl.km?.toFixed(0)} km in ${atl.edges} edges`}`);
  check(atl.ground?.some((g) => !g.ok && g.issues.some((i) => /GroundOverWater|Water/.test(i))), `lowering its bridge to the ground is refused (${JSON.stringify(atl.ground?.map((g) => g.ok ? 'ok' : g.issues.join('/')))})`);
  for (const [name, [a, b]] of [['Raritan Bay (inside the New York pack)', RARITAN], ['Bahia de Banderas (outside every zone)', BANDERAS]]) {
    const r = await route(page, a, b, false);
    report[name] = r;
    check(!r.error && r.edges >= 1 && r.cost / r.km > 46, `${name}: ${r.error ?? `${r.km.toFixed(2)} km, US$${r.cost.toFixed(0)}M, ${(r.cost / r.km).toFixed(1)} a km (45 on dry flat ground)`}`);
    check(r.ground?.some((g) => !g.ok), `${name}: the bridge refuses the ground (${JSON.stringify(r.ground?.map((g) => g.ok ? 'ok' : g.issues.join('/')))})`);
  }
  check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
  if (WITHOUT) {
    const old = await open(ctx, WITHOUT);
    const o = await route(old.page, ATLANTIC[0], ATLANTIC[1], true);
    report.atlanticWithout = o;
    console.log(`INFO without the mask the same line: ${o.error ?? `US$${(o.cost / 1e6).toFixed(3)}T over ${o.km.toFixed(0)} km`}`);
    const [pa, pb] = [await realPrices((await open(ctx, WITH)).page), await realPrices(old.page)];
    const ids = Object.keys(pb);
    const diff = ids.filter((id) => pa[id] !== pb[id]);
    const sum = (p) => Object.values(p).reduce((s, v) => s + v, 0);
    report.real = { edges: ids.length, differ: diff.length, totalWith: sum(pa), totalWithout: sum(pb), diffs: diff.slice(0, 20).map((id) => [id, pa[id], pb[id]]) };
    check(diff.length === 0 && ids.length === Object.keys(pa).length, `the real save: ${ids.length} edges, ${diff.length} priced differently; total US$${(sum(pa) / 1e6).toFixed(2)}M with the mask, US$${(sum(pb) / 1e6).toFixed(2)}M without`);
  }
} finally {
  writeFileSync(join(OUT, 'report.json'), JSON.stringify(report, null, 1));
  await close();
}
process.exit(failed ? 1 : 0);
