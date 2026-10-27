// T-1507: the page command (run by probe.mjs in this folder on t1156-all-china.save) that turns the
// ten high-speed lines (113-122) into a service worth measuring, through the game's own edits:
// 1. Lines 114-122 started (and 120 ended) one station short of their first city: the track ran on
//    to a loose end on top of another line's station (the builder's station there was refused). A
//    station goes on that last piece of track where it comes nearest a metro station, at least
//    600 m back from the loose end, and the rest of the stub is removed.
// 2. Line 116 ran through Wuhan without a stop (a plain joint on 114's station): a station on its
//    track there, nearest a metro station.
// 3. A stop more than 1 km from any metro station moves along its line's track (up to 12 km) to the
//    point nearest a metro station, when that is within 1 km; the old station is removed.
// 4. Each stop is joined by a passage to the nearest metro station within 1 km, and to another
//    line's high-speed station within 1 km.
// 5. High-speed trains on every line (the track was high speed already) and headways by demand
//    level. Cars stay at 20 (400 m, the platforms' most).
const c = window.tw.client;
const pause = (ms) => new Promise((d) => setTimeout(d, ms));
const W = () => window.tw.world.value;
const log = [];
const ok = (what, r) => { log.push([what, r?.ok ? 'ok' : r?.issues ?? r]); return r?.ok; };
const H = (id) => Number(id) >= 113;
const HEADWAY = { 113: [600, 900, 1800], 114: [600, 900, 1800], 115: [900, 1200, 2400], 116: [900, 1200, 2400], 120: [900, 1200, 2400] };
const HEADWAY_REST = [1200, 1800, 3600];

const node = (id) => W().nodes.get(Number(id));
const k = (lat) => Math.cos((lat * Math.PI) / 180);
const ground = (p, q) => Math.hypot(p.x - q.x, p.y - q.y) * k(p.lat ?? q.lat);
const metro = new Map();
for (const l of W().save.lines) if (!H(l.id)) for (const s of l.stops) metro.set(Number(s), node(s));
const nearestMetro = (p) => { let b = null; for (const [id, n] of metro) { const d = ground(p, n); if (!b || d < b.d) b = { id, d }; } return b; };
const edgesAt = (n) => W().edges.filter((e) => e.a === Number(n) || e.b === Number(n));
const stationIds = () => new Set(W().save.stations.map((s) => Number(s.id)));

/** the edge's polyline in pack metres from node `from`: its ends and PIs */
async function poly(e, from) {
  const raw = Array.from((await c.query({ what: 'edgePis', edge: e.id })) ?? []);
  const pts = [node(e.a)];
  for (let i = 1; i + 8 <= raw.length; i += 8) pts.push({ x: raw[i], y: raw[i + 1], lat: node(e.a).lat });
  pts.push(node(e.b));
  return e.a === Number(from) ? pts : pts.reverse();
}
/** points every 50 m along a polyline from its start, between `min` and `max` metres of ground */
function samples(pts, min, max) {
  const out = [];
  let run = 0;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i], L = ground(a, b);
    for (let s = Math.ceil(run / 50) * 50; s <= run + L; s += 50) {
      if (s < min || s > max) continue;
      const t = (s - run) / L;
      out.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, lat: a.lat, along: s });
    }
    run += L;
  }
  return out;
}
/** the best place for a station along the given polylines: nearest a metro station */
function bestPlace(polys, min, max) {
  let best = null;
  for (const { pts, edge } of polys) for (const p of samples(pts, min, max)) {
    const m = nearestMetro(p);
    if (!best || m.d < best.metro.d) best = { p, edge, metro: m };
  }
  return best;
}
/** add a station on `edge` near `p`; its node id */
async function addStationAt(edge, p) {
  const before = stationIds();
  if (!ok(`station on edge ${edge}`, await c.edit({ op: 'addStation', at: { kind: 'edge', edge, x: p.x, y: p.y }, platform: 400, name: '' }))) return null;
  await pause(200);
  return [...stationIds()].find((id) => !before.has(id)) ?? null;
}

const lines = W().save.lines.filter((l) => H(l.id)).map((l) => ({ id: Number(l.id), stops: l.stops.map(Number) }));
const removeLater = [];
const moves = [];

// 1. stubs at the ends
for (const l of lines) for (const end of ['first', 'last']) {
  const s = end === 'first' ? l.stops[0] : l.stops.at(-1);
  const stub = edgesAt(s).find((e) => !(e.lines ?? []).includes(String(l.id)));
  if (!stub) continue;
  const loose = stub.a === s ? stub.b : stub.a;
  if (edgesAt(loose).length !== 1 || stationIds().has(loose)) continue;
  const pts = await poly(stub, loose);
  const best = bestPlace([{ pts, edge: stub.id }], 600, 15000);
  const sid = await addStationAt(stub.id, best.p);
  if (sid === null) continue;
  const rest = edgesAt(loose)[0];
  if (rest) ok(`remove stub ${rest.id}`, await c.edit({ op: 'removeEdge', edge: rest.id }));
  if (end === 'first') l.stops.unshift(sid); else l.stops.push(sid);
  moves.push({ line: l.id, kind: 'end', end, station: sid, back: best.p.along, metro: best.metro });
}

// 2. high-speed joints on top of another line's station (116 through Wuhan)
const hsrStation = (n) => W().save.stations.find((s) => Number(s.id) !== Number(n) && s.level === 2 && ground(s, node(n)) < 100);
for (const l of lines) {
  const joints = new Set();
  for (const e of W().edges) if ((e.lines ?? []).includes(String(l.id))) for (const n of [e.a, e.b]) if (!stationIds().has(n) && edgesAt(n).length === 2 && hsrStation(n)) joints.add(n);
  for (const j of joints) {
    const polys = [];
    for (const e of edgesAt(j)) polys.push({ pts: await poly(e, j), edge: e.id });
    const best = bestPlace(polys, 600, 5000);
    const sid = await addStationAt(best.edge, best.p);
    if (sid === null) continue;
    // between the two stops whose straight segment passes nearest the joint
    const J = node(j);
    let at = 1, bd = Infinity;
    for (let i = 1; i < l.stops.length; i++) {
      const a = node(l.stops[i - 1]), b = node(l.stops[i]);
      const d = ground(a, J) + ground(J, b) - ground(a, b);
      if (d < bd) { bd = d; at = i; }
    }
    l.stops.splice(at, 0, sid);
    moves.push({ line: l.id, kind: 'joint', joint: j, station: sid, along: best.p.along, metro: best.metro });
  }
}

// 3. stops more than 1 km from the metro move along the track
for (const l of lines) for (const [i, s] of l.stops.entries()) {
  if (moves.some((m) => m.station === s)) continue;
  const m0 = nearestMetro(node(s));
  if (m0.d <= 1000 || m0.d > 15000) continue;
  const polys = [];
  for (const e of edgesAt(s).filter((e) => (e.lines ?? []).includes(String(l.id)))) polys.push({ pts: await poly(e, s), edge: e.id });
  const best = bestPlace(polys, 600, 12000);
  if (!best || best.metro.d > 1000 || best.metro.d >= m0.d) { moves.push({ line: l.id, kind: 'left', station: s, metro: m0, best: best?.metro }); continue; }
  const sid = await addStationAt(best.edge, best.p);
  if (sid === null) continue;
  l.stops[i] = sid;
  removeLater.push(s);
  moves.push({ line: l.id, kind: 'moved', from: s, station: sid, along: best.p.along, metro: best.metro, was: m0 });
}

ok('construct', await c.edit({ op: 'constructAll' }));
await pause(500);
for (const l of lines) ok(`stops ${l.id}`, await c.edit({ op: 'setStops', line: l.id, stops: l.stops }));
for (const s of removeLater) ok(`remove station ${s}`, await c.edit({ op: 'removeStation', node: s }));

// 4. passages
const links = [];
const finalStops = new Set(lines.flatMap((l) => l.stops));
for (const s of finalStops) {
  const m = nearestMetro(node(s));
  if (m.d <= 1000) links.push([s, m.id, Math.round(m.d), 'metro']);
}
const done = new Set();
for (const s of finalStops) for (const t of finalStops) {
  if (s >= t || done.has(`${s}-${t}`)) continue;
  const ls = lines.filter((l) => l.stops.includes(s)).map((l) => l.id), lt = lines.filter((l) => l.stops.includes(t)).map((l) => l.id);
  if (ls.some((x) => lt.includes(x))) continue;
  const d = ground(node(s), node(t));
  if (d <= 1000) { links.push([s, t, Math.round(d), 'high speed']); done.add(`${s}-${t}`); }
}
for (const [a, b, d, kind] of links) ok(`link ${a}-${b} ${d} m ${kind}`, await c.edit({ op: 'link', a, b, on: true }));

// 5. trains: high speed, then headways (set twice: switching to headways keeps the counts)
for (const l of lines) {
  ok(`fast ${l.id}`, await c.edit({ op: 'lineFast', line: l.id, fast: true }));
  const h = HEADWAY[l.id] ?? HEADWAY_REST;
  ok(`headway ${l.id}`, await c.edit({ op: 'setHeadway', line: l.id, seconds: h }));
  ok(`headway again ${l.id}`, await c.edit({ op: 'setHeadway', line: l.id, seconds: h.map((x) => x - 1) }));
  ok(`headway set ${l.id}`, await c.edit({ op: 'setHeadway', line: l.id, seconds: h }));
}
await pause(1500);

const summary = lines.map((l) => {
  const st = W().lineStats[l.id] ?? W().lineStats.get?.(l.id);
  return { id: l.id, status: st.status, km: Math.round(st.km), oneWayMin: Math.round(st.fromStartS.at(-1) / 60), roundTripMin: Math.round(st.roundTripS / 60), trains: st.trainsNeeded, autoS: st.autoS, fast: st.fast, cars: st.cars,
    limits: [...new Set(st.stretches.map((x) => Math.round(x.limit)))], stops: l.stops.map((s) => { const m = nearestMetro(node(s)); return [s, Math.round(m.d)]; }) };
});
return { moves, links, summary, log, savedLinks: W().save.links };
