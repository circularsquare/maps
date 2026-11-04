// T-2162: the network diagram's names (app/src/diagram/names.ts) on the real New York save and Tokyo's real
// network. At every zoom level, and between levels, it counts names overlapping a line's stroke, a pill or
// another name, and names nearer another station's pill than their own; round Times Sq and Shinjuku (a
// 1600 x 1000 px view at a pitch of 7 and 14 px) the same, and every interchange must be named at 14 px
// (at 7 px a figure only); and layout, drawing and
// naming together must stay under 5 s (New York) and 8 s (Tokyo). `nyc-game` and `tyo-game` are the game's own
// graphs with the names as the page measured them (diagram-game.mjs; run it first), and their time the page's.
//   node trainworld/checks/t2162-diagram-names.mjs [nyc,tyo,nyc-game,tyo-game]      (about two minutes; figures in
//   artifacts/trainworld/t2162/names.json)
import { mkdirSync, writeFileSync } from 'node:fs';
import { openSave } from './diagram-saves.mjs';
import { gameGraph, splitCity } from './diagram-game.mjs';
import { ARTIFACTS, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const A = await loadApp(`export * from '@app/src/diagram/layout.ts'; export { drawing, PILL_R, PILL_OUTLINE } from '@app/src/diagram/draw.ts'; export * from '@app/src/diagram/names.ts';`);
const which = (process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : 'nyc,tyo,nyc-game,tyo-game').split(',');
/** `--svg`: each view drawn to artifacts/trainworld/t2162/<save>-<pitch>px.svg */
const SVG = process.argv.includes('--svg');
mkdirSync(`${ARTIFACTS}t2162`, { recursive: true });
const CENTRE = { nyc: 'Times Sq-42 St', tyo: '新宿' };
const LIMIT_MS = { nyc: 5000, tyo: 8000, jpn: 60000 };
const FONT = 12, VIEW = [1600, 1000], ZOOMS = [7, 14];
let failed = 0;
const out = {};
const line = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };
const boxesMeet = (a, b) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1];

for (const name of which) {
  const [city, game] = splitCity(name);
  let g, text, width, width2, font = FONT, cap = null;
  if (game) {
    cap = gameGraph(city);
    if (!cap) { console.log(`SKIP ${name}: no capture; run diagram-game.mjs first`); continue; }
    ({ g, text, width, width2 } = cap);
    font = cap.fontPx;
  } else {
    const s = openSave(city);
    ({ g } = s.graph());
    const stopsAt = new Map();
    for (const l of s.lines) for (const st of l.stops) stopsAt.set(st, (stopsAt.get(st) ?? 0) + 1);
    text = [...g.key].map((k) => {
      const c = s.cx.of.get(String(k));
      const members = c === undefined ? [String(k)] : s.cx.members[c];
      return A.pickName(members.map((id) => ({ name: s.name(Number(id)), lines: stopsAt.get(Number(id)) ?? 0 })));
    });
    s.api.free();
    width = text.map((t) => A.guessWidth(t, FONT));
    width2 = text.map((t) => { const p = A.splitName(t); return p ? Math.max(A.guessWidth(p[0], FONT), A.guessWidth(p[1], FONT)) : 0; });
  }
  const t0 = performance.now();
  const lay = A.layoutDiagram(g, A.LAYOUT);
  const t1 = performance.now();
  const d = A.drawing(g, lay);
  const t2 = performance.now();
  const plan = A.placeNames(g, lay, d, text, width, width2, font);
  const t3 = performance.now();
  let ms = { layout: t1 - t0, drawing: t2 - t1, names: t3 - t2, total: t3 - t0 };
  // the game's graph: the page worker's own times, when the capture is of this layout code
  if (cap?.fresh) { const p = cap.page.ms; ms = { layout: p.total, drawing: p.drawing, names: p.names, total: p.total + p.drawing + p.names, page: true }; }
  else if (cap) console.log(`INFO ${name}: the capture (${cap.captured}) is older than the layout code; times are Node's; run diagram-game.mjs`);

  // the obstacles, filed by cell
  const pitch = d.pitch, W = d.stroke, R = (A.PILL_R + A.PILL_OUTLINE) * W, cell = 4 * pitch;
  const segs = [], grid = new Map();
  const file = (x0, y0, x1, y1, r, kind, owner) => {
    const i = segs.length;
    segs.push([x0, y0, x1, y1, r, kind, owner]);
    for (let gx = Math.floor((Math.min(x0, x1) - r) / cell); gx <= Math.floor((Math.max(x0, x1) + r) / cell); gx++)
      for (let gy = Math.floor((Math.min(y0, y1) - r) / cell); gy <= Math.floor((Math.max(y0, y1) + r) / cell); gy++) {
        const k = `${gx},${gy}`;
        if (!grid.has(k)) grid.set(k, []);
        grid.get(k).push(i);
      }
  };
  for (const p of d.ribbons) for (let j = 0; j + 3 < p.length; j += 2) file(p[j], p[j + 1], p[j + 2], p[j + 3], W / 2, 0, -1);
  d.pills.forEach((p, i) => file(p[0], p[1], p[2], p[3], R + p[4], 1, d.pillOf[i]));
  for (const q of d.passages) file(q[0], q[1], q[2], q[3], (A.PILL_R * 0.45 + A.PILL_OUTLINE) * W, 2, -2);
  const near = (b, pad) => {
    const seen = new Set();
    for (let gx = Math.floor((b[0] - pad) / cell); gx <= Math.floor((b[2] + pad) / cell); gx++)
      for (let gy = Math.floor((b[1] - pad) / cell); gy <= Math.floor((b[3] + pad) / cell); gy++) for (const i of grid.get(`${gx},${gy}`) ?? []) seen.add(i);
    return [...seen].map((i) => segs[i]);
  };

  const colours = Array.from(g.key, () => new Set());
  for (const ln of g.lines) ln.chain.forEach((k, i) => { if (ln.stop[i]) colours[k].add(ln.colour); });
  const hasPill = new Set(d.pillOf);
  const centre = text.indexOf(CENTRE[city]);

  /** the names of level `li` drawn at a pitch of `P` px: their boxes and what they meet */
  const measure = (li, P, win) => {
    const sc = P / pitch, L = plan.at[li];
    const boxes = [];
    for (let i = 0; i < L.node.length; i++) boxes.push([...A.placedBox(plan, li, i, sc), L.node[i]]);
    const inWin = (b) => !win || (b[2] > win[0] && b[0] < win[2] && b[3] > win[1] && b[1] < win[3]);
    const r = { names: 0, nameName: 0, nameLine: 0, namePill: 0, nearerOther: 0, examples: [] };
    const ex = (what, k) => { if (r.examples.length < 6) r.examples.push(`${what}: ${text[k]}`); };
    const bgrid = new Map();
    boxes.forEach((b, i) => {
      const k = `${Math.floor(b[0] / (40 / sc))},${Math.floor(b[1] / (40 / sc))}`;
      if (!bgrid.has(k)) bgrid.set(k, []);
      bgrid.get(k).push(i);
    });
    boxes.forEach((b, i) => {
      if (!inWin(b)) return;
      r.names++;
      const k = b[4];
      const bx = Math.floor(b[0] / (40 / sc)), by = Math.floor(b[1] / (40 / sc)), span = Math.ceil((b[2] - b[0]) / (40 / sc)) + 1, spanY = Math.ceil((b[3] - b[1]) / (40 / sc)) + 1;
      let hitName = false;
      for (let gx = bx - span; gx <= bx + span && !hitName; gx++) for (let gy = by - spanY; gy <= by + spanY && !hitName; gy++)
        for (const j of bgrid.get(`${gx},${gy}`) ?? []) if (j !== i && boxesMeet(b, boxes[j])) { hitName = true; break; }
      if (hitName) (r.nameName++, ex('name', k));
      let own = Infinity, other = Infinity, hitLine = false, hitPill = false;
      for (const sg of near(b, 2 * R + 20 / sc + 3 * pitch)) {
        const dist = A.segBoxDist(sg[0], sg[1], sg[2], sg[3], b) - sg[4];
        if (sg[5] === 0) { if (dist < 0) hitLine = true; continue; }
        if (dist < 0) hitPill = true;
        if (sg[5] !== 1) continue;
        if (sg[6] === k) own = Math.min(own, dist);
        else other = Math.min(other, dist);
      }
      if (hitLine) (r.nameLine++, ex('line', k));
      if (hitPill) (r.namePill++, ex('pill', k));
      if (other < own) (r.nearerOther++, ex('nearer another pill', k));
    });
    return { r, named: new Set(boxes.map((b) => b[4])) };
  };

  /** the view as an SVG: lines, pills, name boxes and names; unnamed interchanges ringed red */
  const svg = (win, sc, li, missing) => {
    const X = (x) => ((x - win[0]) * sc).toFixed(1), Y = (y) => ((win[3] - y) * sc).toFixed(1);
    const o = [`<svg xmlns="http://www.w3.org/2000/svg" width="${VIEW[0]}" height="${VIEW[1]}" font-family="sans-serif" font-size="${FONT}"><rect width="100%" height="100%" fill="#fbf5ef"/>`];
    d.ribbons.forEach((p, i) => {
      const pts = [];
      for (let j = 0; j + 1 < p.length; j += 2) pts.push(`${X(p[j])},${Y(p[j + 1])}`);
      o.push(`<polyline points="${pts.join(' ')}" fill="none" stroke="${d.lines[i].colour}" stroke-width="${(W * sc).toFixed(2)}" stroke-linejoin="round"/>`);
    });
    for (const p of d.pills) o.push(`<line x1="${X(p[0])}" y1="${Y(p[1])}" x2="${X(p[2])}" y2="${Y(p[3])}" stroke="#2b2b2b" stroke-width="${(2 * (R + p[4]) * sc).toFixed(2)}" stroke-linecap="round"/><line x1="${X(p[0])}" y1="${Y(p[1])}" x2="${X(p[2])}" y2="${Y(p[3])}" stroke="#fff" stroke-width="${(2 * (R - A.PILL_OUTLINE * W + p[4]) * sc).toFixed(2)}" stroke-linecap="round"/>`);
    const L = plan.at[li], esc = (t) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;');
    for (let i = 0; i < L.node.length; i++) {
      const b = A.placedBox(plan, li, i, sc), parts = L.wrap[i] ? A.splitName(text[L.node[i]]) : [text[L.node[i]]];
      o.push(`<rect x="${X(b[0])}" y="${Y(b[3])}" width="${((b[2] - b[0]) * sc).toFixed(1)}" height="${((b[3] - b[1]) * sc).toFixed(1)}" fill="none" stroke="#999" stroke-width="0.5"/>`);
      parts.forEach((t, j) => o.push(`<text x="${X(b[0])}" y="${(+Y(b[3]) + FONT * (A.NAME_LINE / 2 + j * A.NAME_LINE2) + 4).toFixed(1)}">${esc(t)}</text>`));
    }
    for (const k of missing) o.push(`<circle cx="${X(lay.x[k])}" cy="${Y(lay.y[k])}" r="12" fill="none" stroke="red" stroke-width="2"/><text x="${+X(lay.x[k]) + 14}" y="${Y(lay.y[k])}" fill="red">${text[k]}</text>`);
    o.push('</svg>');
    return o.join('\n');
  };

  const levels = [];
  let worst = 0, prevNamed = null, lost = 0;
  plan.levels.forEach((P, li) => {
    const at = measure(li, P), mid = measure(li, P * 1.18);
    worst += at.r.nameName + at.r.nameLine + at.r.namePill + at.r.nearerOther + mid.r.nameName + mid.r.nameLine + mid.r.namePill + mid.r.nearerOther;
    if (prevNamed) for (const k of prevNamed) if (!at.named.has(k)) lost++;
    prevNamed = at.named;
    levels.push({ pitchPx: +P.toFixed(2), ...at.r, between: mid.r });
  });
  const views = {};
  for (const Z of ZOOMS) {
    const li = A.levelAt(plan, Z), sc = Z / pitch;
    const hw = VIEW[0] / 2 / sc, hh = VIEW[1] / 2 / sc, cx = lay.x[centre], cy = lay.y[centre];
    const win = [cx - hw, cy - hh, cx + hw, cy + hh];
    const m = measure(li, Z, win);
    const margin = 40 / sc;
    const inter = [];
    for (let k = 0; k < g.key.length; k++) {
      if (colours[k].size < 2 || !hasPill.has(k) || !text[k]) continue;
      if (lay.x[k] < win[0] + margin || lay.x[k] > win[2] - margin || lay.y[k] < win[1] + margin || lay.y[k] > win[3] - margin) continue;
      inter.push(k);
    }
    const missing = inter.filter((k) => !m.named.has(k)).map((k) => text[k]);
    if (SVG) writeFileSync(`${ARTIFACTS}t2162/${name}-${Z}px.svg`, svg(win, sc, li, new Set(inter.filter((k) => !m.named.has(k)))));
    let stations = 0;
    for (const k of hasPill) if (lay.x[k] >= win[0] && lay.x[k] <= win[2] && lay.y[k] >= win[1] && lay.y[k] <= win[3]) stations++;
    views[Z] = { ...m.r, stations, interchanges: inter.length, missing };
    line(m.r.nameName + m.r.nameLine + m.r.namePill === 0, `${name}: no name round ${CENTRE[city]} at a pitch of ${Z} px meets a line, pill or name`, `${m.r.names} names of ${stations} stations; names ${m.r.nameName}, lines ${m.r.nameLine}, pills ${m.r.namePill}${m.r.examples.length ? '; ' + m.r.examples.join(', ') : ''}`);
    // at the opening zoom the busiest knots leave no room (notes/T-2162.md): a figure, not a test
    const info = `${inter.length - missing.length} of ${inter.length}${missing.length ? '; missing ' + missing.slice(0, 8).join(', ') : ''}`;
    if (Z === ZOOMS[0]) console.log(`INFO ${name}: interchanges round ${CENTRE[city]} named at ${Z} px: ${info}`);
    else line(!missing.length, `${name}: every interchange round ${CENTRE[city]} named at ${Z} px`, info);
    line(m.r.nearerOther === 0, `${name}: no name round ${CENTRE[city]} at ${Z} px nearer another station's pill than its own`, `${m.r.nearerOther}`);
  }
  line(worst === 0, `${name}: no overlaps anywhere, at any level or between levels`, `${worst}`);
  line(ms.total < LIMIT_MS[city], `${name}: layout, drawing and names under ${LIMIT_MS[city] / 1000} s${ms.page ? " in the page's worker" : ''}`, `layout ${(ms.layout / 1000).toFixed(2)} s, drawing ${(ms.drawing / 1000).toFixed(2)} s, names ${(ms.names / 1000).toFixed(2)} s`);
  console.log(`     ${name}: names per level ${levels.map((l) => `${l.pitchPx}:${l.names}`).join(' ')}; ${lost} lost going up a level`);
  out[name] = { ms, levels, views, lost };
}
mkdirSync(`${ARTIFACTS}t2162`, { recursive: true });
writeFileSync(`${ARTIFACTS}t2162/names.json`, JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);
