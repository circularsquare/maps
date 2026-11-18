// T-3537, T-3538: a street-snapped tram line that is double at its ends and splits in the middle into two
// one-way single tracks, Anita's Caracas build (board 20261117-105027, 20261117-105713).
// T-3537, through the pointer and the real tool, street snap on, single track picked, at ZOOM (default 15):
//   two doubles on one street, D1 west and D2 east, loose ends facing; from D1's east end a single 5 px left of
//   its centre line joins D1's left track and ends 5 px left of D2's west end on D2's left track; a second single
//   from the same end (no longer loose) on the right tracks; a third, turning off at the first corner, on the track
//   of its own side (right; T-3609), and the same turn from a loose end on the track under the pointer (left);
//   one from the end's centre merges both tracks. Each time the ghost's joins agree with the built piece's.
// T-3538, on track laid by edits: a tram double, two one-way singles 500 m apart, double again; a line run out over
//   one leg and back over the other (a loop that turns back at both ends, its stops on both legs) has no turn-back
//   mark at either end and no turn icon in its stop list; a line that turns back partway (N, S, L) and a loop that
//   turns back at two stops besides its first keep theirs.
// T-3541, on Anita's save (artifacts/trainworld/t3541-caracas.save, skipped when absent): Line 2's return run turns at
//   the loose end just past Avenida Arismendi; blueprint double track drawn on from that end leaves it running, and
//   so does building it.
// Pictures (SHOTS=before|after): artifacts/trainworld/t3537_<shots>_end.png (the split at D1's end, zoomed in),
// t3537_<shots>_corner.png, t3538_<shots>.png (the line selected).
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6122):
//   [ZOOM=15] [OFF=5] node trainworld/checks/t3537-tram-split.mjs [dev url]
// About 4 minutes.
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6122/';
const SHOTS = process.env.SHOTS ?? 'after';
const Z = Number(process.env.ZOOM ?? 15), OFF = Number(process.env.OFF ?? 5);
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
// downtown Caracas: a point on a straight street running east-south-east, and that street's way on screen
const AT = [-66.9146, 10.5061];
const STREET = [840, 450], DIR = [0.98, 0.197];

const chrome = await launchChrome({ prefix: 'tw-t3537-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const open = async () => {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&city=caa&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  return page;
};

try {
  // ------------------------------------------------------------------ T-3537
  const page = await open();
  const settle = async (ms = 600) => {
    await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
    await wait(ms);
  };
  /** the tram track tool with street snap on, single or double */
  const tramTool = (single) => page.evaluate(async (single) => {
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'track'; ui.setTrackType('tram'); ui.setSingleTrack(single); ui.snapStreets.value = true; ui.keepPlacing.value = false;
  }, single);
  const edgeCount = () => page.evaluate(() => window.tw.world.value.edges.length);
  const waitEdges = (n) => page.waitForFunction((n) => window.tw.world.value.edges.length > n, n, { timeout: 30000 }).then(() => wait(800), () => {});
  const streetDone = () => page.waitForFunction(() => ['done', 'none', 'off'].includes(window.tw.tools.street.phase), undefined, { timeout: 30000 }).catch(() => {});
  // the doubles' ends on the street, found at zoom 16 and kept as lng, lat
  await page.evaluate(([c]) => window.tw.map.jumpTo({ center: c, zoom: 16 }), [AT]);
  await tramTool(false);
  await settle(5000);
  const ends = await page.evaluate(([s, d]) => {
    const tw = window.tw, st = tw.tools.street;
    const at = (t) => { const p = tw.map.unproject([s[0] + d[0] * t, s[1] + d[1] * t]); return st.landing(p.lng, p.lat); };
    return { d1: [at(-260), at(-60)], d2: [at(190), at(330)] };
  }, [STREET, DIR]);
  // the doubles, plain tram track between points on the street (a double's loose end is what the tool starts from)
  const made = await page.evaluate(async (ends) => {
    const tw = window.tw, P = ([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return { kind: 'free', x: p.x, y: p.y, level: 0 }; };
    const out = [];
    for (const [a, b] of [ends.d1, ends.d2]) {
      out.push((await tw.client.edit({ op: 'route', from: P(a), to: P(b), pis: [], single: false, tram: true })).ok);
      await new Promise((r) => setTimeout(r, 600));
    }
    const w = tw.world.value, near = ([lng, lat]) => [...w.nodes.values()].sort((p, q) => Math.hypot(p.lng - lng, p.lat - lat) - Math.hypot(q.lng - lng, q.lat - lat))[0].id;
    return { ok: out, e1: near(ends.d1[1]), w2: near(ends.d2[0]) };
  }, ends);
  check(made.ok.every(Boolean), 'two doubles laid on the street', JSON.stringify(made));
  await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [ends.d1[1], Z]);
  await tramTool(true);
  await settle(4000);

  /** a node's place on screen and the way out of it along its double (unit, screen), with left = (uy, -ux) */
  const endAt = (node) => page.evaluate((node) => {
    const tw = window.tw, w = tw.world.value, n = w.nodes.get(node), r = tw.map.getContainer().getBoundingClientRect();
    const e = w.edges.find((e) => e.tracks === 2 && (e.a === node || e.b === node));
    const pts = tw.renderer.net.edgePts.get(e.id), k = e.a === node ? 0 : pts.length - 3, k2 = e.a === node ? 3 : pts.length - 6;
    const p = tw.map.project([n.lng, n.lat]), q0 = tw.map.project(tw.toLngLat(pts[k], pts[k + 1])), q1 = tw.map.project(tw.toLngLat(pts[k2], pts[k2 + 1]));
    const l = Math.hypot(q0.x - q1.x, q0.y - q1.y);
    return { x: r.left + p.x, y: r.top + p.y, ux: (q0.x - q1.x) / l, uy: (q0.y - q1.y) / l };
  }, node);
  const hoverAt = async (x, y) => {
    await page.mouse.move(x, y, { steps: 3 });
    await wait(500);
    return page.evaluate(() => { const h = window.tw.tools.hover; return h ? { kind: h.kind, node: h.node, join: !!h.join, merge: !!h.merge } : null; });
  };
  const e1 = await endAt(made.e1), w2 = await endAt(made.w2);
  /** single track from D1's end at `off` px left of its centre line to `to` (screen), clicked; the ghost and the built piece's joins */
  const single = async (off, to, toOff = null) => {
    const h = await hoverAt(e1.x + e1.uy * off, e1.y - e1.ux * off);
    const n = await edgeCount();
    await page.mouse.down(); await page.mouse.up();
    await wait(600);
    await page.mouse.move(to[0] - 30, to[1], { steps: 6 });
    await page.mouse.move(to[0], to[1], { steps: 4 });
    await streetDone();
    await wait(1000);
    const hEnd = toOff === null ? null : await page.evaluate(() => { const h = window.tw.tools.hover; return h ? { kind: h.kind, join: !!h.join, merge: !!h.merge } : null; });
    const ghost = await page.evaluate(() => { const p = window.tw.tools.lastPreview; return p ? { ok: p.ok, joins: Array.from(p.joins ?? []).slice(0, 2), street: window.tw.tools.street.phase } : null; });
    await page.mouse.down(); await page.mouse.up();
    await waitEdges(n);
    const got = await page.evaluate(([n, a, b]) => {
      const w = window.tw.world.value, made = w.edges.slice(n).filter((e) => e.tracks === 1);
      const at = (node) => { const e = made.find((e) => e.a === node || e.b === node); return e ? (e.a === node ? e.join & 3 : (e.join >> 2) & 3) : null; };
      return { made: made.length, start: at(a), end: at(b), kinds: [...new Set(made.map((e) => e.kind))] };
    }, [n, made.e1, made.w2]);
    return { h, hEnd, ghost, ...got };
  };
  // D2's west end, its way out pointing back west: its left is D1's right
  const d2 = (off) => [w2.x + w2.uy * off, w2.y - w2.ux * off];
  // the first corner: the street crossing south of D1's end, found at the screen's centre on D1's end
  const cornerPoint = () => page.evaluate(([x, y, ux, uy]) => {
    const tw = window.tw, r = tw.map.getContainer().getBoundingClientRect();
    // 70 px on and 90 px to the right of the way out, on the nearest street
    const p = tw.map.unproject([x - r.left + ux * 70 - uy * 90, y - r.top + uy * 70 + ux * 90]), q = tw.tools.street.landing(p.lng, p.lat), s = tw.map.project(q);
    return [r.left + s.x, r.top + s.y];
  }, [e1.x, e1.y, e1.ux, e1.uy]);
  // turning off at the first corner from the loose end: a press 5 px left, the turn to the right; the pointer's track wins
  const cornerLoose = await single(OFF, await cornerPoint());
  check(cornerLoose.start === 1 && cornerLoose.ghost?.joins?.[0] === 1, 'a single turning off at the first corner from a loose end joins the track under the pointer (left)', JSON.stringify(cornerLoose));
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  await wait(1200);
  const left = await single(OFF, d2(-OFF), -OFF);
  check(left.h?.join && !left.h.merge, `hover ${OFF} px left of D1's end picks a track`, JSON.stringify(left.h));
  check(left.hEnd?.join, `hover ${OFF} px off D2's end picks a track`, JSON.stringify(left.hEnd));
  check(left.ghost?.joins?.[0] === 1 && left.ghost?.joins?.[1] !== 0, 'the ghost joins the left tracks', JSON.stringify(left.ghost));
  check(left.start === 1 && left.end !== 0 && left.end === left.ghost?.joins?.[1] && left.ghost?.street === 'done', 'the click, a street route, joins D1 and D2 on those tracks', JSON.stringify(left));
  const right = await single(-OFF, d2(OFF), OFF);
  check(right.h?.join && !right.h.merge, 'from the same end (no longer loose) the hover picks the right track', JSON.stringify(right.h));
  check(right.start === 2 && right.end !== 0 && right.end !== left.end && right.ghost?.joins?.[0] === 2 && right.ghost?.joins?.[1] === right.end, 'the second single joins the right tracks at both ends, the ghost the same', JSON.stringify(right));
  // each single's way as the direction it runs: east (+1) or west (-1)
  const ways = await page.evaluate(() => {
    const w = window.tw.world.value;
    return w.edges.filter((e) => e.tracks === 1).map((e) => {
      const way = e.autoWay ?? e.oneWay ?? 0, a = w.nodes.get(e.a), b = w.nodes.get(e.b);
      return way ? (way === 1) === (b.lng > a.lng) ? 1 : -1 : 0;
    });
  });
  check(ways.length === 2 && ways.includes(1) && ways.includes(-1), 'the two singles run one way each, opposite ways', JSON.stringify(ways));
  // zoomed in on the split
  await page.evaluate(([c]) => window.tw.map.jumpTo({ center: c, zoom: 18.3 }), [ends.d1[1]]);
  await settle(2500);
  await page.screenshot({ path: join(ARTIFACTS, `t3537_${SHOTS}_end.png`), clip: { x: 500, y: 250, width: 700, height: 400 } });
  await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [ends.d1[1], Z]);
  await settle(1500);
  // the same press and turn where the double has split into two singles (T-3609): the single takes the track on its own side, the right
  const cornerTo = await cornerPoint();
  const corner = await single(OFF, cornerTo);
  check(corner.start === 2 && corner.ghost?.joins?.[0] === 2, 'at the split a single turning right takes the right track whatever side was pressed', JSON.stringify(corner));
  await page.screenshot({ path: join(ARTIFACTS, `t3537_${SHOTS}_corner.png`), clip: { x: 450, y: 200, width: 700, height: 500 } });
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  await wait(1200);
  // from the centre the single takes both tracks; on a loose end only, so D2's east end
  const e2 = await page.evaluate(([n]) => {
    const w = window.tw.world.value, deg = new Map();
    for (const e of w.edges) for (const x of [e.a, e.b]) deg.set(x, (deg.get(x) ?? 0) + 1);
    return w.edges.filter((e) => e.tracks === 2).flatMap((e) => [e.a, e.b]).find((x) => deg.get(x) === 1 && x !== n);
  }, [made.e1]);
  const c2 = await endAt(e2);
  const h = await hoverAt(c2.x, c2.y);
  const n = await edgeCount();
  await page.mouse.down(); await page.mouse.up();
  await wait(600);
  await page.mouse.move(c2.x + c2.ux * 150, c2.y + c2.uy * 150, { steps: 8 });
  await streetDone();
  await wait(1000);
  const ghost = await page.evaluate(() => Array.from(window.tw.tools.lastPreview?.joins ?? []).slice(0, 1));
  await page.mouse.down(); await page.mouse.up();
  await waitEdges(n);
  const mid = await page.evaluate(([n, node]) => { const e = window.tw.world.value.edges.slice(n).find((e) => e.a === node || e.b === node); return e ? (e.a === node ? e.join & 3 : (e.join >> 2) & 3) : null; }, [n, e2]);
  check(h?.merge && ghost[0] === 0 && mid === 0, 'a click at a loose end\'s centre merges both tracks into the single', JSON.stringify({ h, ghost, mid }));
  await page.close();

  // ------------------------------------------------------------------ T-3538
  const p2 = await open();
  const net = await p2.evaluate(async ([c]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const node = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const free = (p) => ({ kind: 'free', x: p.x, y: p.y, level: 0 });
    const bad = [];
    const ed = async (op) => { const r = await cl.edit(op); if (!r.ok) bad.push(op.op + ':' + (r.issues ?? []).join(',')); await pause(400); return r; };
    // N(0,2000) double to A(0,1000); two singles A to B(0,0), each joined to one track so each runs one way; B double to S(-1500,0)
    await ed({ op: 'route', from: free(P(0, 2000)), to: free(P(0, 1000)), pis: [], single: false, tram: true });
    await ed({ op: 'route', from: free(P(0, 0)), to: free(P(-1500, 0)), pis: [], single: false, tram: true });
    const A = node(P(0, 1000)), B = node(P(0, 0));
    await ed({ op: 'route', from: { kind: 'node', node: A }, to: { kind: 'node', node: B }, pis: [{ ...P(-250, 500), level: 0 }], single: true, tram: true, join: [o.x - 2, o.y + 1000, o.x, o.y + 2] });
    await ed({ op: 'route', from: { kind: 'node', node: A }, to: { kind: 'node', node: B }, pis: [{ ...P(250, 500), level: 0 }], single: true, tram: true, join: [o.x + 2, o.y + 1000, o.x, o.y - 2] });
    const st = {};
    for (const [k, dx, dy, end] of [['N', 0, 2000, true], ['A', 0, 1300, false], ['W', -250, 500, false], ['E', 250, 500, false], ['L', -700, 0, false], ['S', -1500, 0, true]]) {
      const p = P(dx, dy);
      let at = { kind: 'node', node: node(p) };
      if (!end) {
        let best = null, bd = 1e12;
        for (const e of tw.world.value.edges) {
          const pts = tw.renderer.net.edgePts.get(e.id);
          for (let i = 0; pts && i < pts.length; i += 3) { const [lng, lat] = tw.toLngLat(pts[i], pts[i + 1]); const q = tw.lngLatToPack(lng, lat), d = Math.hypot(q.x - p.x, q.y - p.y); if (d < bd) { bd = d; best = e.id; } }
        }
        at = { kind: 'edge', edge: best, x: p.x, y: p.y };
      }
      const r = await ed({ op: 'addStation', at, platform: 60, name: { N: 'El Silencio', A: 'Avenida Este 12', W: 'Pasaje Calasanz', E: 'Pasaje Cala', L: 'Calle Loyola', S: 'Avenida Arismendi' }[k] });
      st[k] = r.ok ? node(p) : null;
    }
    const ways = tw.world.value.edges.filter((e) => e.tracks === 1).map((e) => e.autoWay ?? e.oneWay ?? 0);
    const line = async (name, steps) => {
      const a = await ed({ op: 'addLine', stops: [], name, colour: '#1c6dd0' });
      for (const s of steps) {
        if (s === '+') { const l = tw.world.value.save.lines.find((x) => x.num === a.line); await ed({ op: 'setStops', line: a.line, stops: [...l.stops.map(Number), Number(l.stops[0])] }); }
        else if (s === '1') await ed({ op: 'lineOneWay', line: a.line, oneWay: true });
        else if (Array.isArray(s)) { await ed({ op: 'setStops', line: a.line, stops: s.map((k) => st[k]) }); await ed({ op: 'orderStops', line: a.line, stops: s.map((k) => st[k]) }); }
        else await ed({ op: 'insertStop', line: a.line, station: st[s] });
      }
      return a.line;
    };
    // the right click's loop on the line's last stop, run one way, its stops on both legs added
    const split = await line('Split', ['N', 'A', 'L', 'S', '+', '1', 'W', 'E']);
    await pause(1500);
    const { reversalsOf } = await import('/src/map/reversalMarks.ts');
    const name = (id) => Object.entries(st).find((e) => e[1] === Number(id))?.[0] ?? id;
    const info = (num) => {
      const w = tw.world.value, marks = reversalsOf(w), l = w.save.lines.find((x) => x.num === num), s = w.lineStats[l.id];
      return { id: l.id, stops: l.stops.map(name).join(''), loop: s.loop, oneWay: s.oneWay, rev: s.reversals.map((k) => name(l.stops[k])), marked: marks.filter((m) => m.lines.some((x) => x.id === l.id)).map((m) => name(m.station)) };
    };
    window.t3538 = { line, info, pause };
    return { bad, ways, st, split: info(split) };
  }, [AT]);
  check(!net.bad.length && net.ways.every((w) => w), 'a double, two one-way singles, a double; stations on both legs', JSON.stringify({ bad: net.bad, ways: net.ways }));
  const s = net.split;
  check(s.loop === 'balloon' && s.rev.length === 1 && /W/.test(s.stops) && /E/.test(s.stops), 'the split line runs out on one leg and back on the other, a loop that turns at both ends', JSON.stringify(s));
  check(s.marked.length === 0, 'no turn-back mark at either end of the split line', JSON.stringify(s.marked));
  // the line panel's stop list: no turn icon on the split line, nor a mark on the map
  await p2.evaluate(async ([id, c]) => {
    const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]), q = tw.packToLngLat(o.x - 400, o.y + 900), [lng, lat] = Array.isArray(q) ? q : [q.lng, q.lat];
    (await import('/src/game/ui.ts')).selection.value = { kind: 'line', line: id };
    tw.map.jumpTo({ center: [lng, lat], zoom: 14.6 });
  }, [s.id, AT]);
  await p2.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(3000);
  await p2.evaluate(() => document.querySelector('[data-stop]')?.scrollIntoView({ block: 'start' }));
  await wait(500);
  const icons = await p2.evaluate(() => ({ list: document.querySelectorAll('.rev-stop').length, map: [...document.querySelectorAll('.rev-mark')].filter((e) => e.style.display !== 'none').length }));
  check(icons.list === 0 && icons.map === 0, 'no turn icon in the split line\'s stop list, no mark on the map', JSON.stringify(icons));
  const lap = await p2.evaluate(() => document.querySelector('.inspector, aside, body').innerText);
  check(!/Reversing partway/.test(lap) && /Turning at the end/.test(lap), 'its lap has both turns as turning at the end, none partway', (lap.match(/(Turning at the end|Reversing partway)\s*\S+/g) ?? []).join(', '));
  await p2.screenshot({ path: join(ARTIFACTS, `t3538_${SHOTS}.png`) });
  // lines that do turn back partway keep their marks
  const ctl = await p2.evaluate(async () => {
    const { line, info, pause } = window.t3538;
    const back = await line('Switchback', [['N', 'S', 'L']]);
    const two = await line('Two turns', [['A', 'S', 'N', 'A']]);
    await pause(1500);
    return { back: info(back), two: info(two) };
  });
  check(ctl.back.rev.length === 1 && ctl.back.marked.join() === ctl.back.rev.join(), 'a line turning back partway keeps its mark', JSON.stringify(ctl.back));
  check(ctl.two.rev.length >= 2 && ctl.two.marked.length === ctl.two.rev.length, 'a loop turning back at two stops keeps both marks', JSON.stringify(ctl.two));

  // ------------------------------------------------------------------ T-3541
  const SAVE = join(ARTIFACTS, 't3541-caracas.save');
  if (!existsSync(SAVE)) console.log('SKIP T-3541: no ' + SAVE);
  else {
    const p3 = await open();
    await importSave(p3, readFileSync(SAVE));
    await p3.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
    await wait(3000);
    const r = await p3.evaluate(async () => {
      const tw = window.tw, pause = (ms) => new Promise((d) => setTimeout(d, ms));
      const status = () => tw.world.value.lineStats[tw.world.value.save.lines.find((l) => l.name === 'Line 2').id].status;
      const w = tw.world.value, ar = w.save.stations.find((s) => s.name === 'Avenida Arismendi');
      const ends = new Map();
      for (const e of w.edges) for (const n of [e.a, e.b]) ends.set(n, [...(ends.get(n) ?? []), e]);
      // the loose end nearest the station, and the way on past it
      const loose = [...w.nodes.values()].filter((n) => ends.get(n.id)?.length === 1).sort((a, b) => Math.hypot(a.lng - ar.lng, a.lat - ar.lat) - Math.hypot(b.lng - ar.lng, b.lat - ar.lat))[0];
      const e = ends.get(loose.id)[0], o = w.nodes.get(e.a === loose.id ? e.b : e.a);
      const p = tw.lngLatToPack(loose.lng, loose.lat), q = tw.lngLatToPack(o.lng, o.lat), L = Math.hypot(p.x - q.x, p.y - q.y);
      const out = { before: status(), far: Math.round(Math.hypot((loose.lng - ar.lng) * 109000, (loose.lat - ar.lat) * 111000)) };
      const x = await tw.client.edit({ op: 'route', from: { kind: 'node', node: loose.id }, to: { kind: 'free', x: p.x + ((p.x - q.x) / L) * 300, y: p.y + ((p.y - q.y) / L) * 300, level: 0 }, pis: [], single: false, tram: true });
      await pause(2500);
      out.extended = x.ok ? status() : 'refused ' + (x.issues ?? []).join(',');
      const c = await tw.client.edit({ op: 'constructAll' });
      await pause(2500);
      out.built = c.ok ? status() : 'refused ' + (c.issues ?? []).join(',');
      return out;
    });
    check(r.before === 'running' && r.far < 60, 'her Line 2 runs, a loose end just past Avenida Arismendi', JSON.stringify(r));
    check(r.extended === 'running', 'blueprint double drawn on from that end leaves Line 2 running', JSON.stringify(r));
    check(r.built === 'running', 'built, Line 2 still runs', JSON.stringify(r));
  }

  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3537/T-3538/T-3541 FAILED: ${fail}` : 'T-3537/T-3538/T-3541 PASSED');
process.exit(fail ? 1 : 0);
