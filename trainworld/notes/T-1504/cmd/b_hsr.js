// T-1504 probe command: the high-speed lines (ids 113-122): tph, and per stop its node, demand
// station, position and the weekday view's station figures.
const w = window.tw.world.value, toLL = window.tw.packToLngLat;
const v = window.twDemand.days()[0].view;
const out = { stationKeys: null, lines: [] };
const st0 = v.stations && (v.stations.values ? v.stations.values().next().value : Object.values(v.stations)[0]);
out.stationKeys = st0 ? Object.keys(st0) : null;
out.sampleStation = st0 ? JSON.parse(JSON.stringify(st0, (k, x) => (x instanceof Map ? Object.fromEntries(x) : ArrayBuffer.isView(x) ? Array.from(x).slice(0, 8) : x))) : null;
for (const l of w.save.lines.filter((l) => Number(l.id) >= 113)) {
  const stops = l.stops.map((s) => {
    const n = w.nodes.get(s) ?? w.nodes.get(Number(s));
    const st = v.stations?.get ? v.stations.get(s) ?? v.stations.get(String(s)) ?? v.stations.get(Number(s)) : null;
    return { node: s, ll: n ? toLL(n.x, n.y).map((x) => +x.toFixed(4)) : null, nodeKeys: n ? Object.keys(n) : null, station: n?.station ?? null,
      st: st ? JSON.parse(JSON.stringify(st, (k, x) => (x instanceof Map ? Object.fromEntries(x) : ArrayBuffer.isView(x) ? Array.from(x).slice(0, 8) : x))) : null };
  });
  out.lines.push({ id: l.id, name: l.name, tph: l.tph, stops });
}
return out;
