// T-1504 probe command: experiment 1. Link every high-speed stop (lines 113-122) to the nearest
// stop of another line within 1 km (a passage, as a player would build), then wait for demand
// final and report the weekday's riders by purpose and the explorer on the named pairs.
const w0 = window.tw.world.value;
const st = w0.save.stations;
const sample = st[0];
const stopLine = new Map();
for (const l of w0.save.lines) for (const s of l.stops) { const k = String(s); if (!stopLine.has(k)) stopLine.set(k, new Set()); stopLine.get(k).add(Number(l.id)); }
const pos = (s) => { const n = w0.nodes.get(s) ?? w0.nodes.get(Number(s)); return n ? [n.x, n.y, n.lat] : null; };
// a stop is a node; its station is the one at its place
const byNode = { get(k) { const p = pos(k); if (!p) return null; let b = null, bd = 80; for (const s of st) { const d = Math.hypot(s.x - p[0], s.y - p[1]); if (d < bd) { bd = d; b = s; } } return b; } };
const others = [...stopLine.keys()].filter((k) => [...stopLine.get(k)].some((id) => id < 113)).map((k) => [k, pos(k)]).filter((e) => e[1]);
const made = [];
for (const l of w0.save.lines.filter((l) => Number(l.id) >= 113)) for (const s of l.stops) {
  const p = pos(s); if (!p) continue;
  const k = Math.cos(p[2] * Math.PI / 180);
  let best = null;
  for (const [o, q] of others) { const d = Math.hypot(q[0] - p[0], q[1] - p[1]) * k; if (!best || d < best.d) best = { d, o }; }
  if (!best || best.d > 990) continue;
  const sa = byNode.get(String(s)), sb = byNode.get(best.o);
  if (!sa || !sb) { made.push({ s, o: best.o, d: Math.round(best.d), missing: true }); continue; }
  const r = await window.tw.client.edit({ op: 'link', a: sa.num, b: sb.num, on: true });
  made.push({ s, o: best.o, d: Math.round(best.d), ok: r?.ok });
}
// wait for a new final
const v0 = window.twDemand.view().version;
const t0 = performance.now();
for (;;) {
  await new Promise((r) => setTimeout(r, 2000));
  const d = window.twDemand, s = d.log.at(-1), v = d.view();
  if (v && v.version > v0 && v.final && s && v.version === s.version && !d.client.waiting) break;
  if (performance.now() - t0 > 1500000) return { made, timeout: true };
}
await new Promise((r) => setTimeout(r, 3000));
const v = window.twDemand.days()[0].view;
const purposes = {};
for (const [p, d] of v.purposes ?? []) { let sum = 0; const hsr = {}; for (const [id, l] of d.lines) { sum += l.ridersPerDay; if (Number(id) >= 113) hsr[id] = Math.round(l.ridersPerDay); } purposes[p] = { sum: Math.round(sum), hsr }; }
const P = { bei: 2688, sha: 17, gua: 2594, shz: 2598, wuh: 2624, chs: 2675, tia: 2701, nan: 2694, zhe: 2710 };
const pairs = {};
for (const [a, b] of [['bei', 'sha'], ['gua', 'shz'], ['wuh', 'chs'], ['bei', 'tia'], ['sha', 'nan'], ['bei', 'wuh'], ['bei', 'zhe'], ['wuh', 'gua'], ['bei', 'gua']]) pairs[`${a}-${b}`] = await window.__t.ask(P[a], P[b]);
return { sampleStation: Object.keys(sample), made, rail: v.railTripsPerDay, secs: Math.round((performance.now() - t0) / 1000), purposes, pairs };
