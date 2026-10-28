// T-1616: each city's (zone's) money a steady weekday on the all-Japan save, US$M a game day, worked
// out from the lines (t1616/summary.json, lines.json) the way game/regionMoney.ts splits it, simplified:
// running cost and upkeep by route km, each stretch half to each end stop's zone; fares half by the
// boardings at each stop (the base) and half by route km (the per-km part; no per-stretch loads kept).
// The clock's own region ledger (regions.json) covered only a few game hours of day 1, so it is not used.
import { readFileSync } from 'node:fs';
const DIR = 'C:/Users/anita/projects/maps/artifacts/trainworld/t1616/';
const d = JSON.parse(readFileSync(DIR + 'lines.json', 'utf8'));
const sum = new Map(JSON.parse(readFileSync(DIR + 'summary.json', 'utf8')).map((r) => [r.id, r]));
const registry = JSON.parse(readFileSync('C:/Users/anita/projects/maps/trainworld/app/src/game/registry.json', 'utf8'));
const names = new Map();
const walk = (o) => { if (Array.isArray(o)) o.forEach(walk); else if (o && typeof o === 'object') { if (typeof o.id === 'string' && typeof o.name === 'string') names.set(o.id, o.name); Object.values(o).forEach(walk); } };
walk(registry);
const Z = new Map();
const at = (z) => { if (!Z.has(z)) Z.set(z, { fares: 0, running: 0, upkeep: 0, long: 0 }); return Z.get(z); };
for (const l of d.lines) {
  const r = sum.get(l.id);
  if (!r) continue;
  const km = l.hopKm, tot = km.reduce((a, v) => a + v, 0) || 1;
  const kmShare = new Map();
  km.forEach((h, k) => { for (const z of [l.zones[k], l.zones[k + 1]]) kmShare.set(z, (kmShare.get(z) ?? 0) + h / 2 / tot); });
  const b = l.boardings ?? [], bt = b.reduce((a, v) => a + v, 0);
  for (const [z, s] of kmShare) { const e = at(z); e.running += r.running * s; e.upkeep += r.upkeep * s; e.fares += r.fares * s * (bt ? 0.5 : 1); if (r.km >= 200) e.long += (r.fares - r.running - r.upkeep) * s; }
  if (bt) b.forEach((v, k) => { at(l.zones[k]).fares += r.fares * 0.5 * v / bt; });
}
const rows = [...Z].map(([z, e]) => ({ z, name: names.get(z) ?? (z || '(outside every zone)'), ...e, net: e.fares - e.running - e.upkeep })).sort((a, b) => a.net - b.net);
const M = (v) => (v / 1e6).toFixed(0);
console.log(`${rows.length} zones, ${rows.filter((r) => r.net < 0).length} losing; net ${M(rows.reduce((a, r) => a + r.net, 0))}`);
console.log('zone, fares, running, upkeep, net, of which lines over 200 km');
for (const r of rows) console.log(`${r.z.padEnd(5)} ${String(r.name).slice(0, 20).padEnd(20)} ${M(r.fares).padStart(6)} ${M(r.running).padStart(6)} ${M(r.upkeep).padStart(5)} ${M(r.net).padStart(6)} ${M(r.long).padStart(6)}`);
