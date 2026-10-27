// T-1504 probe command: for each high-speed stop, the nearest stop of any other line (metres) and
// that line; also the world's and save's keys (to find station complexes).
const w = window.tw.world.value;
const stopLine = new Map();
for (const l of w.save.lines) for (const s of l.stops) { const k = String(s); if (!stopLine.has(k)) stopLine.set(k, new Set()); stopLine.get(k).add(l.id); }
const pos = (s) => { const n = w.nodes.get(s) ?? w.nodes.get(Number(s)); return n ? [n.x, n.y, n.lat] : null; };
const all = [...stopLine.keys()].map((k) => [k, pos(k)]).filter((e) => e[1]);
const out = { worldKeys: Object.keys(w), saveKeys: Object.keys(w.save), lines: [] };
for (const l of w.save.lines.filter((l) => Number(l.id) >= 113)) {
  out.lines.push({ id: l.id, stops: l.stops.map((s) => {
    const p = pos(s); if (!p) return null;
    const k = Math.cos(p[2] * Math.PI / 180);
    let best = null;
    for (const [o, q] of all) {
      if (o === String(s) || [...stopLine.get(o)].every((id) => Number(id) >= 113)) continue;
      const d = Math.hypot(q[0] - p[0], q[1] - p[1]) * k;
      if (!best || d < best.d) best = { d: Math.round(d), stop: o, lines: [...stopLine.get(o)] };
    }
    return { stop: String(s), lines: [...stopLine.get(String(s))], nearest: best };
  }) });
}
return out;
