// T-2405: no line of the network diagram drawn under another. For every drawn stretch (a line's ribbon
// between two consecutive nodes of its chain), the share of its length whose middle lies within half a
// stroke of another line's middle, that line not in the same stroke by the order's rule (lines of one colour
// interlining on the same track share one, docs/ui-diagram.md "Order"). A stretch 90% covered or more
// counts as hidden; there should be none. Also: the stretches half covered, and the layout's time.
// `nyc-game`, `l8-game` and `xp-game` are the game's own graphs (diagram-game.mjs; run it first); `xp` is
// t2405-express-game.mjs's express beside a local.
//   node trainworld/checks/t2405-diagram-hidden.mjs [nyc,l8,xp,nyc-game,l8-game,xp-game]      (about a minute;
//   figures in artifacts/trainworld/t2405/hidden.json)
import { mkdirSync, writeFileSync } from 'node:fs';
import { openSave, waterOf, gameSea } from './diagram-saves.mjs';
import { gameGraph, splitCity } from './diagram-game.mjs';
import { ARTIFACTS, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const A = await loadApp(`export * from '@app/src/diagram/layout.ts'; export { drawing } from '@app/src/diagram/draw.ts';`);
const which = (process.argv[2] ?? 'nyc,l8,xp,nyc-game,l8-game,xp-game').split(',');
/** a stretch this share covered is hidden; a point is covered by a middle this many strokes away */
const HIDDEN = 0.9, NEAR_W = 0.5, STEP_P = 0.2;
/** `--svg`: the drawing round the first hidden stretches, to artifacts/trainworld/t2405/<name>-<i>.svg, the
 * hidden ones marked by a thin magenta line (and the whole of a small one, <name>-all.svg); `--half` marks the
 * half covered too */
const SVG = process.argv.includes('--svg'), HALF = process.argv.includes('--half');
let failed = 0;
const out = {};

function svgAt(g, d, names, cx, cy, half, marks) {
  const sc = 600 / (2 * half), X = (x) => ((x - cx + half) * sc).toFixed(1), Y = (y) => ((cy + half - y) * sc).toFixed(1);
  const poly = (r) => { let s = ''; for (let j = 0; j < r.length; j += 2) s += `${X(r[j])},${Y(r[j + 1])} `; return s; };
  const o = [`<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" font-family="sans-serif" font-size="10"><rect width="100%" height="100%" fill="#fbf5ef"/>`];
  for (const r of d.ribbons) o.push(`<polyline points="${poly(r)}" fill="none" stroke="#fbf5ef" stroke-width="${(d.pitch * sc).toFixed(2)}" stroke-linejoin="round"/>`);
  d.ribbons.forEach((r, i) => {
    const l = d.lines[i];
    o.push(`<polyline points="${poly(r)}" fill="none" stroke="${l.colour}" stroke-width="${(d.stroke * sc).toFixed(2)}" stroke-linejoin="round" stroke-linecap="round"/>`);
    if (l.fast) o.push(`<polyline points="${poly(r)}" fill="none" stroke="#fbf5ef" stroke-width="${(0.34 * d.stroke * sc).toFixed(2)}"/>`);
  });
  for (const p of d.pills) o.push(`<line x1="${X(p[0])}" y1="${Y(p[1])}" x2="${X(p[2])}" y2="${Y(p[3])}" stroke="#222" stroke-width="${(2.6 * d.stroke * sc).toFixed(2)}" stroke-linecap="round"/><line x1="${X(p[0])}" y1="${Y(p[1])}" x2="${X(p[2])}" y2="${Y(p[3])}" stroke="#fff" stroke-width="${(1.9 * d.stroke * sc).toFixed(2)}" stroke-linecap="round"/>`);
  for (const m of marks) o.push(`<polyline points="${poly(m)}" fill="none" stroke="#f0f" stroke-width="1.5" stroke-dasharray="4 3"/>`);
  for (let k = 0; k < g.key.length; k++) {
    const lx = d.pills.find((p, i) => d.pillOf[i] === k);
    if (lx && Math.abs(lx[0] - cx) < half && Math.abs(lx[1] - cy) < half) o.push(`<text x="${(+X(lx[0]) + 8).toFixed(1)}" y="${(+Y(lx[1]) - 6).toFixed(1)}">${(names[k] ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')}</text>`);
  }
  o.push('</svg>');
  return o.join('\n');
}
const line = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

/** per piece and line, the lines drawn in its stroke there (itself included) */
function strokeMates(g, lay) {
  const n = g.key.length, mates = new Map();
  for (const pc of lay.pieces) {
    // pieces as T-2405 makes them carry their strokes' lines; before, one stroke per colour
    const groups = pc.strokes ?? (() => {
      const e = [...Array(g.ea.length).keys()].find((k) => g.ea[k] === Math.min(pc.nodes[0], pc.nodes[1]) && g.eb[k] === Math.max(pc.nodes[0], pc.nodes[1]));
      if (e === undefined) return [];
      const by = new Map();
      for (let k = g.first[e]; k < g.first[e + 1]; k++) { const c = g.lines[g.on[k]].colour; if (!by.has(c)) by.set(c, []); by.get(c).push(g.on[k]); }
      return [...by.values()];
    })();
    for (let i = 0; i + 1 < pc.nodes.length; i++) {
      const hk = Math.min(pc.nodes[i], pc.nodes[i + 1]) * n + Math.max(pc.nodes[i], pc.nodes[i + 1]);
      for (const grp of groups) for (const l of grp) mates.set(hk * 65536 + l, new Set(grp));
    }
  }
  return mates;
}

for (const name of which) {
  const [city, game] = splitCity(name);
  let g, names, sea = null;
  if (game) {
    const cap = gameGraph(city);
    if (!cap) { console.log(`SKIP ${name}: no capture; run diagram-game.mjs first`); continue; }
    ({ g } = cap);
    names = cap.text;
    sea = gameSea(cap);
  } else {
    const s = openSave(city);
    ({ g } = s.graph());
    names = [...g.key].map((k) => s.name(k) ?? '');
    ({ sea } = waterOf(s, g));
    s.api.free();
  }
  // with the sea on the ground, as the page's worker lays it out (T-2541)
  A.layoutDiagram(g, A.LAYOUT, null, undefined, sea); // the code compiled
  const t0 = performance.now();
  const lay = A.layoutDiagram(g, A.LAYOUT, null, undefined, sea);
  const ms = performance.now() - t0;
  const d = A.drawing(g, lay);
  const n = g.key.length, W = d.stroke, near = NEAR_W * W, step = STEP_P * d.pitch;
  const mates = strokeMates(g, lay);
  // every ribbon segment in a grid
  const cell = 2 * d.pitch, grid = new Map();
  d.ribbons.forEach((r, i) => {
    for (let j = 0; j + 3 < r.length; j += 2) {
      const x0 = Math.floor((Math.min(r[j], r[j + 2]) - near) / cell), x1 = Math.floor((Math.max(r[j], r[j + 2]) + near) / cell);
      const y0 = Math.floor((Math.min(r[j + 1], r[j + 3]) - near) / cell), y1 = Math.floor((Math.max(r[j + 1], r[j + 3]) + near) / cell);
      for (let gx = x0; gx <= x1; gx++) for (let gy = y0; gy <= y1; gy++) {
        const k = gx * 1e6 + gy;
        if (!grid.has(k)) grid.set(k, []);
        grid.get(k).push(i, j);
      }
    }
  });
  const segDist = (px, py, r, j) => {
    const ax = r[j], ay = r[j + 1], dx = r[j + 2] - ax, dy = r[j + 3] - ay, l2 = dx * dx + dy * dy;
    const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0;
    return Math.hypot(ax + dx * t - px, ay + dy * t - py);
  };
  /** the drawn lines (not `self`, not its stroke's mates) whose middle is within `near` of the point */
  const coverers = (px, py, self, allowed) => {
    const hit = new Set();
    for (const v of [grid.get(Math.floor(px / cell) * 1e6 + Math.floor(py / cell))]) if (v) for (let q = 0; q < v.length; q += 2) {
      const i = v[q];
      if (i === self || hit.has(i) || allowed?.has(d.lineOf[i])) continue;
      if (segDist(px, py, d.ribbons[i], v[q + 1]) < near) hit.add(i);
    }
    return hit;
  };
  let stretches = 0, hidden = 0, half = 0, coveredLen = 0, totalLen = 0;
  const examples = [], halves = [], byPair = new Map(), spots = [], marks = [];
  d.ribbons.forEach((r, i) => {
    const li = d.lineOf[i], ln = g.lines[li], at = d.chainAt[i];
    // stretches: runs of points with one chain index (the hop from chain[c] to chain[c + 1])
    let j0 = 0;
    for (let j = 1; j <= at.length; j++) {
      if (j < at.length && at[j] === at[j0]) continue;
      const c = at[j0], hopEnd = Math.min(j, at.length - 1);
      if (c + 1 < ln.chain.length) {
        const a = ln.chain[c], b = ln.chain[c + 1], hk = Math.min(a, b) * n + Math.max(a, b);
        const allowed = mates.get(hk * 65536 + li);
        let len = 0, cov = 0;
        const who = new Map();
        for (let q = j0; q < hopEnd; q++) {
          const x0 = r[q * 2], y0 = r[q * 2 + 1], x1 = r[q * 2 + 2], y1 = r[q * 2 + 3], sl = Math.hypot(x1 - x0, y1 - y0);
          const k = Math.max(1, Math.ceil(sl / step));
          for (let s = 0; s < k; s++) {
            const f = (s + 0.5) / k, h = coverers(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, i, allowed);
            len += sl / k;
            if (h.size) {
              cov += sl / k;
              for (const o of h) who.set(o, (who.get(o) ?? 0) + sl / k);
            }
          }
        }
        if (len > 1e-9 * d.pitch) {
          stretches++;
          totalLen += len;
          coveredLen += cov;
          if (cov >= 0.5 * len) {
            half++;
            const top = [...who.entries()].sort((p, q) => q[1] - p[1])[0];
            const other = top ? g.lines[d.lineOf[top[0]]] : null;
            const pk = `${ln.colour}${ln.fast ? ' (high speed)' : ''} under ${other?.colour}${other?.fast ? ' (high speed)' : ''}`;
            const ex = `${names[a]} - ${names[b]}: ${pk}, ${Math.round((100 * cov) / len)}%`;
            if (cov >= HIDDEN * len) {
              hidden++;
              byPair.set(pk, (byPair.get(pk) ?? 0) + 1);
              if (examples.length < 12) examples.push(ex);
            } else if (halves.length < 20) halves.push(ex);
            if (cov >= HIDDEN * len || HALF) {
              const mid = j0 + ((hopEnd - j0) >> 1);
              if (!spots.some(([x, y]) => Math.hypot(x - r[mid * 2], y - r[mid * 2 + 1]) < 20 * d.pitch)) spots.push([r[mid * 2], r[mid * 2 + 1]]);
              marks.push(Array.from(r.subarray(j0 * 2, hopEnd * 2 + 2)));
            }
          }
        }
      }
      j0 = j;
    }
  });
  if (SVG && n < 200) {
    const [x0, y0, x1, y1] = d.box;
    writeFileSync(`${ARTIFACTS}t2405/${name}-all.svg`, svgAt(g, d, names, (x0 + x1) / 2, (y0 + y1) / 2, Math.max(x1 - x0, y1 - y0) / 2 + 4 * d.pitch, marks));
  }
  if (SVG) for (const [q, [cx, cy]] of spots.slice(0, 6).entries()) writeFileSync(`${ARTIFACTS}t2405/${name}-${q}.svg`, svgAt(g, d, names, cx, cy, 30 * d.pitch, marks));
  out[name] = { complexes: n, lines: g.lines.length, ms: Math.round(ms), layoutMs: lay.ms, stretches, hidden, half, coveredShare: coveredLen / totalLen, byPair: Object.fromEntries(byPair), examples, halves, strokes: lay.pieces.reduce((s, pc) => s + (pc.strokes ?? pc.cols).length, 0) };
  line(hidden === 0, `${name}: no drawn stretch hidden under another line`, `${hidden} of ${stretches} stretches ${HIDDEN * 100}% covered or more, ${half} half covered, ${(100 * coveredLen / totalLen).toFixed(2)}% of all line length covered${examples.length ? '; e.g. ' + examples.slice(0, 4).join('; ') : ''}`);
  console.log(`     ${name}: ${n} complexes, ${g.lines.length} lines, laid out in ${(ms / 1000).toFixed(2)} s in Node (A ${(lay.ms.a / 1000).toFixed(2)}, B ${(lay.ms.b / 1000).toFixed(2)}, routing ${(lay.ms.route / 1000).toFixed(2)}); ${out[name].strokes} strokes over ${lay.pieces.length} pieces`);
}
mkdirSync(`${ARTIFACTS}t2405`, { recursive: true });
writeFileSync(`${ARTIFACTS}t2405/hidden${process.env.TAG ? '-' + process.env.TAG : ''}.json`, JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);
