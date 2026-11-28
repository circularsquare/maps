// T-1381: the track and line strokes as the clock worker posts them (map/strokePost.ts) against the
// strokes laid end to end whole, after random series of edits: a piece (an edge's or a line's strokes)
// added at the end, inserted, replaced by one of the same size (it stays where it is) or another,
// removed, moved (the runs after an insert or removal are copied within the page's arrays, T-1509), the last few replaced (blueprint track), nothing changed, many added at once (past
// the room after the last segment), everything removed, and a state the page dropped (the next is
// whole). Each post goes through a real structured clone with its buffers transferred, and the
// segments of every earlier network that the posts since did not write over must still be intact. T-4401:
// `mergeNet` leaves the drawn network's arrays as they are until the merged one is drawn (`netIn`).
//   node trainworld/checks/t1381-strokes.mjs [seeds]      (from the maps root)
import { loadApp } from './lib.mjs';

const { StrokeSender, mergeStrokes, planStrokes, strokesOfPieces, strokeRuns, mergeNet, netIn } = await loadApp(
  "export { StrokeSender, mergeStrokes, planStrokes, strokesOfPieces } from '@app/src/map/strokePost.ts'; export { strokeRuns, mergeNet, netIn } from '@app/src/map/network.ts';");

const SEEDS = Number(process.argv[2] ?? 40);
let fails = 0, steps = 0;
const seen = { track: { whole: 0, same: 0, patch: 0 }, lines: { whole: 0, same: 0, patch: 0 } };

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

function piece(rnd, kind, n = 1 + Math.floor(rnd() * 25)) {
  const f = (k) => Float32Array.from({ length: n * k }, () => Math.fround((rnd() - 0.5) * 1e5));
  const u = (k) => Uint32Array.from({ length: n * k }, () => Math.floor(rnd() * 2 ** 32));
  const strokes = { seg: f(4), colour: u(1), edge: u(1), level: f(1), flags: f(1), dist: f(1), count: n };
  // a piece without lanes is all double track (255)
  if (rnd() < 0.9) strokes.lane = Uint8Array.from({ length: n * 2 }, () => Math.floor(rnd() * 256));
  if (kind === 'lines') strokes.stretch = f(1);
  return kind === 'track' ? { strokes, speed: f(1), train: f(1) } : { strokes };
}
const empty = (kind) => {
  const p = piece(() => 0.5, kind, 0);
  return p;
};

const bytes = (v) => new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
const WIDTH = { seg: 4, lane: 2 };
/** the first difference between two sets, ignoring the segments in `dirty` (start, count pairs) */
function differs(a, b, dirty = []) {
  if (a.count !== b.count) return `count ${a.count} vs ${b.count}`;
  const skip = new Uint8Array(a.count);
  for (let k = 0; k < dirty.length; k += 2) skip.fill(1, dirty[k], Math.min(a.count, dirty[k] + dirty[k + 1]));
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    if (k === 'count') continue;
    const x = a[k], y = b[k];
    if (!x || !y) return `${k} missing`;
    if (x.constructor !== y.constructor || x.length !== y.length) return `${k} length ${x.length} vs ${y.length}`;
    const w = (WIDTH[k] ?? 1) * x.BYTES_PER_ELEMENT, p = bytes(x), q = bytes(y);
    for (let i = 0; i < a.count; i++) {
      if (skip[i]) continue;
      for (let j = i * w; j < (i + 1) * w; j++) if (p[j] !== q[j]) return `${k} at segment ${i}`;
    }
  }
  return null;
}

for (const kind of ['track', 'lines']) {
  for (let seed = 1; seed <= SEEDS; seed++) {
    const rnd = rng(seed * 7919 + (kind === 'track' ? 1 : 2));
    const sender = new StrokeSender();
    let pieces = Array.from({ length: Math.floor(rnd() * 30) }, () => piece(rnd, kind));
    let last = null, merged = null;
    const count = (ps) => ps.reduce((n, p) => n + p.strokes.count, 0);
    // every network the page has had, with the strokes whole as they were then, and the segments written over since
    const history = [];
    for (let step = 0; step < 150; step++) {
      const q = rnd();
      if (step === 0) { /* the first is whole */ }
      else if (q < 0.20) pieces = [...pieces, piece(rnd, kind)];
      else if (q < 0.25) pieces = [...pieces, ...Array.from({ length: 1 + Math.floor(rnd() * 40) }, () => piece(rnd, kind))];
      else if (q < 0.30) { const i = Math.floor(rnd() * (pieces.length + 1)); pieces = [...pieces.slice(0, i), piece(rnd, kind), ...pieces.slice(i)]; }
      else if (q < 0.40 && pieces.length) { const i = Math.floor(rnd() * pieces.length); pieces = pieces.map((p, j) => (j === i ? piece(rnd, kind) : p)); }
      else if (q < 0.55 && pieces.length) {
        // replaced by one of the same size, a few of them: nothing moves
        const hit = new Set(Array.from({ length: 1 + Math.floor(rnd() * 3) }, () => Math.floor(rnd() * pieces.length)));
        pieces = pieces.map((p, j) => (hit.has(j) ? piece(rnd, kind, p.strokes.count) : p));
      }
      else if (q < 0.60 && pieces.length) { const i = Math.floor(rnd() * pieces.length); pieces = pieces.filter((_, j) => j !== i); }
      else if (q < 0.64 && pieces.length) pieces = pieces.slice(0, -1);
      else if (q < 0.76 && pieces.length) { const cut = 1 + Math.floor(rnd() * Math.min(3, pieces.length)); pieces = [...pieces.slice(0, -cut), ...Array.from({ length: Math.floor(rnd() * 4) }, () => piece(rnd, kind))]; } // the tail replaced
      else if (q < 0.80 && pieces.length > 1) { const p = [...pieces], [a] = p.splice(Math.floor(rnd() * p.length), 1); p.splice(Math.floor(rnd() * (p.length + 1)), 0, a); pieces = p; }
      else if (q < 0.82) pieces = [];
      else if (q < 0.84) pieces = [...pieces, empty(kind)];
      // else: nothing changed (the same pieces)
      const dropped = step > 0 && rnd() < 0.03;
      if (dropped) { sender.forget(); last = null; merged = null; }
      const n = count(pieces);
      const runs = last ? strokeRuns(pieces, (p) => p.strokes.count, last.pieces) : null;
      const post = sender.post(kind, pieces, n, runs, last ? last.count : 0);
      seen[kind][post.mode]++;
      const cloned = structuredClone(post.strokes, { transfer: [...new Set(post.buffers)] });
      const got = mergeStrokes(kind, merged ?? undefined, cloned, post.mode, post.ranges, post.moves);
      steps++;
      const whole = strokesOfPieces(kind, pieces);
      const why = !got ? `mergeStrokes gave null for ${post.mode}` : differs(got, whole);
      if (why) { fails++; if (fails <= 10) console.log(`FAIL ${kind} seed ${seed} step ${step} (${post.mode}): ${why}`); break; }
      // the networks before this one are as they were, but where this post wrote
      if (post.mode === 'patch') {
        // T-1509: and where the runs that moved were put
        const written = [...post.ranges];
        for (let k = 0; k + 2 < post.moves.length; k += 3) written.push(post.moves[k + 1], post.moves[k + 2]);
        for (const h of history) h.dirty.push(...written);
      }
      history.push({ got, whole, dirty: [] });
      const bad = history.slice(-8, -1).map((h) => differs(h.got, h.whole, h.dirty)).find(Boolean);
      if (bad) { fails++; if (fails <= 10) console.log(`FAIL ${kind} seed ${seed} step ${step} (${post.mode}): an earlier network changed: ${bad}`); break; }
      merged = got;
      last = { pieces, count: n };
    }
  }
}

// a post the page cannot make whole is refused, not guessed, and writes nothing
const probe = new StrokeSender();
const first = [piece(rng(1), 'lines', 10)];
const w = probe.post('lines', first, 10, null, 0);
const own = structuredClone(w.strokes, { transfer: [...new Set(w.buffers)] });
const before = Array.from(own.seg);
const tailOf = (n) => strokesOfPieces('lines', [piece(rng(2), 'lines', n)]);
const ranges = (...r) => Int32Array.from(r);
const refusals = {
  'patch with nothing before it': () => planStrokes('lines', undefined, { ...tailOf(2), count: 12 }, 'patch', ranges(10, 2)),
  'same with another count': () => planStrokes('lines', own, { count: 11 }, 'same', ranges()),
  'a range past the end': () => planStrokes('lines', own, { ...tailOf(2), count: 11 }, 'patch', ranges(10, 2)),
  'a gap before new segments': () => planStrokes('lines', own, { ...tailOf(1), count: 12 }, 'patch', ranges(11, 1)),
  'new segments missing at the end': () => planStrokes('lines', own, { ...tailOf(1), count: 12 }, 'patch', ranges(3, 1)),
  'ranges out of order': () => planStrokes('lines', own, { ...tailOf(2), count: 12 }, 'patch', ranges(5, 1, 3, 1)),
  'ranges overlapping': () => planStrokes('lines', own, { ...tailOf(4), count: 10 }, 'patch', ranges(0, 2, 1, 2)),
  'an empty range': () => planStrokes('lines', own, { ...tailOf(1), count: 10 }, 'patch', ranges(2, 0, 4, 1)),
  'a tail of another length': () => planStrokes('lines', own, { ...tailOf(3), count: 12 }, 'patch', ranges(10, 2)),
  'a move from past the end': () => planStrokes('lines', own, { ...tailOf(1), count: 10 }, 'patch', ranges(0, 1), ranges(9, 0, 2)),
  'a move to past the end': () => planStrokes('lines', own, { ...tailOf(1), count: 10 }, 'patch', ranges(0, 1), ranges(2, 9, 2)),
  'an empty move': () => planStrokes('lines', own, { ...tailOf(1), count: 10 }, 'patch', ranges(0, 1), ranges(2, 3, 0)),
  'room for fewer segments': () => planStrokes('lines', own, { ...tailOf(2990), count: 3000 }, 'patch', ranges(10, 2990)),
};
// ... and one that fits is taken
if (!planStrokes('lines', own, { ...tailOf(2), count: 12 }, 'patch', ranges(10, 2))) { fails++; console.log('FAIL a post that fits was refused'); }
for (const [what, f] of Object.entries(refusals)) if (f() !== null) { fails++; console.log(`FAIL a post with ${what} was taken`); }
if (Array.from(own.seg).join() !== before.join()) { fails++; console.log('FAIL a refused post changed the arrays'); }

// T-4401: a merged network's segments go into the last one's arrays only once it is drawn (`netIn`), since the page
// draws the last one until then (a state held for its trips); two merged before either is drawn are written in order
function netsCase(drawFirst) {
  const send = { track: new StrokeSender(), lines: new StrokeSender() }, r = rng(drawFirst ? 11 : 12);
  let serial = 0;
  const post = (parts, last) => {
    const n = { serial: ++serial, edgePts: new Map() }, counts = {};
    const patch = last ? { base: last.serial, oldTrack: last.counts.track, oldLines: last.counts.lines, same: [], edgeIds: new Uint32Array(0), modes: {}, ranges: {}, moves: {} } : null;
    for (const kind of ['track', 'lines']) {
      const ps = parts[kind], c = ps.reduce((s, p) => s + p.strokes.count, 0);
      const runs = last ? strokeRuns(ps, (p) => p.strokes.count, last.parts[kind]) : null;
      const p = send[kind].post(kind, ps, c, runs, last ? last.counts[kind] : 0);
      n[kind] = structuredClone(p.strokes, { transfer: [...new Set(p.buffers)] });
      counts[kind] = c;
      if (patch) Object.assign(patch, { [kind]: runs }), (patch.modes[kind] = p.mode), (patch.ranges[kind] = p.ranges), (patch.moves[kind] = p.moves);
    }
    if (patch) n.patch = patch;
    return { n, serial: n.serial, counts, parts };
  };
  const fail = (why) => { fails++; console.log(`FAIL merged networks (${drawFirst ? 'each drawn' : 'the second drawn alone'}): ${why}`); };
  const p0 = { track: Array.from({ length: 6 }, () => piece(r, 'track')), lines: Array.from({ length: 6 }, () => piece(r, 'lines')) };
  // each edit puts a piece first, so every segment after it moves
  const p1 = { track: [piece(r, 'track'), ...p0.track], lines: [piece(r, 'lines'), ...p0.lines] };
  const p2 = { track: [piece(r, 'track'), ...p1.track], lines: [piece(r, 'lines'), ...p1.lines] };
  const a = post(p0, null), b = post(p1, a), c = post(p2, b);
  if (!mergeNet(null, a.n)) return fail('the whole one was refused');
  const was = { track: strokesOfPieces('track', p0.track), lines: strokesOfPieces('lines', p0.lines) };
  if (!mergeNet(a.n, b.n) || !mergeNet(b.n, c.n)) return fail('a patch was refused');
  if (b.n.patch.modes.lines !== 'patch' || !b.n.patch.moves.lines.length) return fail(`the posts were not patches that move segments (${b.n.patch.modes.lines})`);
  for (const kind of ['track', 'lines']) { const why = differs(a.n[kind], was[kind]); if (why) return fail(`the drawn network's ${kind} changed before the next was drawn: ${why}`); }
  if (drawFirst) {
    netIn(b.n);
    for (const kind of ['track', 'lines']) { const why = differs(b.n[kind], strokesOfPieces(kind, p1[kind])); if (why) return fail(`the first drawn differs in ${kind}: ${why}`); }
  }
  netIn(c.n);
  netIn(b.n);
  for (const kind of ['track', 'lines']) { const why = differs(c.n[kind], strokesOfPieces(kind, p2[kind])); if (why) return fail(`the second drawn differs in ${kind}: ${why}`); }
}
netsCase(true);
netsCase(false);

for (const kind of ['track', 'lines']) {
  const s = seen[kind];
  if (!s.whole || !s.same || !s.patch) { fails++; console.log(`FAIL ${kind}: not every kind of post came up`, JSON.stringify(s)); }
}
console.log(`${fails ? 'FAIL' : 'PASS'}  t1381 strokes: ${steps} steps, track ${JSON.stringify(seen.track)}, lines ${JSON.stringify(seen.lines)}`);
process.exit(fails ? 1 : 0);
