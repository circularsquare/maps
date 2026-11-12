// T-3297: left-hand running is the mirror image of right-hand running. A left-hand game of a mirrored layout must
// decide everything a right-hand game does and look like it flipped.
//   1. The sim: `lanes_parity`'s `mirror-right` (its small game without quad track) against `mirror-left` (that
//      game laid with x negated, keeping left): every path, route search, badge, junction inspector, guard pair,
//      capacity resource and figure, line figure and stop offset the same, numbers compared by size (x and a
//      heading's cosine change sign; a junction's crossing pair comes in either order), and `game-left` against
//      `mirror-right` differing (the check can fail).
//   2. The map: a new sandbox game keeping right (A) and one keeping left with the layout mirrored about a
//      north-south line (B), and for a control the mirrored layout keeping right (C): a double main with a through
//      station and dead-end termini, a flat double Y, a flyover, a two-way single branch, three lines on no common
//      piece (lines sharing track are drawn side by side in an order a mirror does not keep), one turning back at
//      the through station, a train every 3 minutes, all built, the clock stopped at 08:30. Close-in views with the
//      basemap, the trains' loads (riders come from people, who are not mirrored) and every page element but the
//      map's and the network's canvases hidden: B flipped matches A (pixels apart by more than a little, nearest
//      within 1 px, under 0.5% of the drawn ones), and C flipped does not where there are trains or a turn-back
//      crossover. Pictures in artifacts/trainworld/t3297-mirror-*.png, `-diff-AB-*` with the differences red.
// From the maps root (or the tree's), on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6169):
//   node trainworld/checks/t3297-mirror.mjs [dev url] [--sim-only]
// About 4 minutes.
import { execFileSync, spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { ARTIFACTS, HERE, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6169/';
const SIM_ONLY = process.argv.includes('--sim-only');
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

// ---- 1. the sim
const SIM = join(HERE, '..', 'sim');
const OUT = join(ARTIFACTS, 't3297-mirror');
mkdirSync(OUT, { recursive: true });
const env = { ...process.env, CARGO_BUILD_JOBS: process.env.CARGO_BUILD_JOBS ?? '3', PATH: `${join(homedir(), '.cargo', 'bin')};${process.env.PATH}` };
execFileSync('cargo', ['build', '--release', '--bin', 'lanes_parity'], { cwd: SIM, env, stdio: 'inherit' });
const bin = join(SIM, 'target', 'release', process.platform === 'win32' ? 'lanes_parity.exe' : 'lanes_parity');
const print = (mode) => {
  const out = join(OUT, `${mode}.txt`);
  const r = spawnSync(bin, [mode, out], { env, encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`lanes_parity ${mode} exited ${r.status}: ${r.stderr}`);
  return readFileSync(out, 'utf8').split('\n');
};
const NUM = /-?\d+(?:\.\d+)?(?:e-?\d+)?/gi;
/** rows of `a` and `b` that differ beyond a mirror: numbers by size, and in `nodes_info` (whose last column is
 * the node's heading h, mirrored to pi - h) the heading's sine */
function mirrorDiff(a, b) {
  const bad = [];
  let section = '';
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const x = a[i] ?? '', y = b[i] ?? '';
    if (x.startsWith('## ')) section = x.slice(3);
    if (x === y) continue;
    const nx = x.match(NUM) ?? [], ny = y.match(NUM) ?? [];
    let ok = x.replace(NUM, '#') === y.replace(NUM, '#') && nx.length === ny.length;
    for (let k = 0; ok && k < nx.length; k++) {
      let p = Math.abs(+nx[k]), q = Math.abs(+ny[k]);
      if (section === 'nodes_info' && k === nx.length - 1) { p = Math.sin(+nx[k]); q = Math.sin(+ny[k]); }
      if (Math.abs(p - q) > 1e-6 * Math.max(1, p, q)) ok = false;
    }
    if (!ok) bad.push(`${section} row ${i}: ${x.slice(0, 160)} | ${y.slice(0, 160)}`);
  }
  return bad;
}
/** the capacity section with its resources (a heading and its users) and each one's users in a sorted order:
 * the mirror swaps the two moves of a junction's crossing pair, so which of its resources and which of their
 * users come first */
function sortCapacity(rows) {
  const at = rows.indexOf('## capacity'), end = rows.findIndex((r, i) => i > at && r.startsWith('## '));
  const blocks = [];
  for (const r of rows.slice(at + 1, end)) if (r.startsWith('  ') && blocks.length) blocks[blocks.length - 1].push(r); else blocks.push([r]);
  for (const b of blocks) b.splice(1, b.length - 1, ...b.slice(1).sort());
  const key = (b) => b.join('\n').replace(NUM, (n) => Math.abs(+n).toPrecision(6));
  blocks.sort((p, q) => (key(p) < key(q) ? -1 : key(p) > key(q) ? 1 : 0));
  return [...rows.slice(0, at + 1), ...blocks.flat(), ...rows.slice(end)];
}
const right = sortCapacity(print('mirror-right')), left = sortCapacity(print('mirror-left'));
const sims = mirrorDiff(right, left);
check(sims.length === 0, 'the sim decides the mirrored left-hand game as the right-hand one', sims.length ? `${sims.length} rows: ${sims[0]}` : `${right.length} rows`);
const control = mirrorDiff(right, print('game-left'));
check(control.length > 50, 'and the same game keeping left, not mirrored, decides differently', `${control.length} rows differ`);

// ---- 2. the map
if (!SIM_ONLY) {
  const RC = [-73.93, 40.675];
  const chrome = await launchChrome({ prefix: 'tw-t3297m-', size: [1200, 800], deviceScaleFactor: 1 });
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
  const errors = [];
  collectErrors(page, errors);
  const settle = async (ms = 800) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {}); await page.waitForTimeout(ms); };

  // one game: `side` its running side, `sx` -1 for the layout mirrored about RC's meridian
  const game = async (side, sx) => {
    await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&side=${side}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
    return page.evaluate(async ([c, sx]) => {
      const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
      const P = (dx, dy) => ({ x: o.x + sx * dx, y: o.y + dy });
      const pause = (ms) => new Promise((d) => setTimeout(d, ms));
      const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
      const edgeAt = (p) => { let best = null, bd = 1e12; for (const [id, pts] of tw.renderer.net.edgePts) for (let k = 0; k + 2 < pts.length; k += 3) { const d = Math.hypot(pts[k] - p.lx, pts[k + 1] - p.ly); if (d < bd) { bd = d; best = id; } } return best; };
      const res = [];
      const go = async (op, ms = 300) => { const r = await cl.edit(op); res.push([op.op + (op.name ? ' ' + op.name : ''), r]); await pause(ms); return r; };
      const W = P(-3000, 0), E = P(3000, 0), M = P(-1200, 0), Y = P(800, 0), Yend = P(-400, 1500), F = P(-200, -2400), S = P(-2200, 0), Send = P(-2700, 1200);
      await go({ op: 'route', from: { kind: 'free', ...W, level: 0 }, to: { kind: 'free', ...E, level: 0 }, pis: [], single: false }, 500);
      const main = () => tw.world.value.edges.find((e) => e.tracks === 2);
      await go({ op: 'addStation', at: { kind: 'edge', edge: main().id, ...M }, platform: 200, name: '' });
      // the Y and the single branch leave from the main's piece under their point
      const pieceAt = (p) => {
        const loc = tw.toLocal(...(() => { const q = tw.packToLngLat(p.x, p.y); return Array.isArray(q) ? q : [q.lng, q.lat]; })());
        return edgeAt({ lx: loc[0], ly: loc[1] });
      };
      await go({ op: 'route', from: { kind: 'edge', edge: pieceAt(Y), ...Y }, to: { kind: 'free', ...Yend, level: 0 }, pis: [], single: false }, 500);
      await go({ op: 'route', from: { kind: 'node', node: near(M) }, to: { kind: 'free', ...F, level: 0 }, pis: [], single: false, ramp: true, design: 160, flyover: 20 }, 600);
      await go({ op: 'route', from: { kind: 'edge', edge: pieceAt(S), ...S }, to: { kind: 'free', ...Send, level: 0 }, pis: [], single: true }, 500);
      const ends = [W, E, Yend, F, Send].map(near);
      for (const n of ends) await go({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: '' }, 200);
      await go({ op: 'constructAll' }, 500);
      const [nW, nE, nY, nF, nS] = ends, nM = near(M);
      const colours = ['#e6b800', '#3a7bd5', '#c2185b'];
      // no two lines on one piece: lines sharing track are drawn side by side in an order the mirror does not keep
      const lines = [[nW, nM], [nM, nF], [nY, nE]];
      for (const [k, stops] of lines.entries()) {
        const r = await go({ op: 'addLine', stops, name: `L${k}`, colour: colours[k] }, 500);
        // a train every 3 minutes, so there are trains both ways in every view
        if (r.ok) await go({ op: 'setHeadway', line: r.line, seconds: [180, 180, 180] }, 500);
      }
      // the same moment in every game, the clock stopped: trains both ways
      tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600);
      await pause(1500);
      const w = tw.world.value;
      const ll = (p) => { const q = tw.packToLngLat(p.x, p.y); return Array.isArray(q) ? q : [q.lng, q.lat]; };
      return { refused: res.filter(([, r]) => !r.ok).map(([op, r]) => `${op}: ${JSON.stringify(r.issues ?? r).slice(0, 100)}`), lines: w.lines?.length ?? w.save?.lines?.length,
        views: { M: ll(M), J: ll(P(-700, -40)), Y: ll(Y), S: ll(S), W: ll(W), E: ll(E) },
        badges: tw.client.lastNodes.filter((n) => n.crosses).length, rightHand: tw.renderer.net.keep };
    }, [RC, sx]);
  };
  // the map alone: the basemap's layers hidden, and every page element but the map's and the network's canvases
  const bare = () => page.evaluate(() => {
    const map = window.tw.map;
    for (const l of map.getStyle().layers) if (l.type !== 'custom' && l.type !== 'background') map.setLayoutProperty(l.id, 'visibility', 'none');
    if (!document.getElementById('t3297-bare')) {
      const s = document.createElement('style');
      s.id = 't3297-bare';
      s.textContent = '* { visibility: hidden !important; } canvas.maplibregl-canvas, canvas.overlay { visibility: visible !important; }';
      document.head.appendChild(s);
    }
    window.tw.selection.value = null;
    // riders come from the people where the track is, who are not mirrored: trains drawn without their loads, for
    // good (demand answers late)
    const r = window.tw.renderer;
    if (!r.__noLoads) {
      const set = r.setLayers.bind(r);
      r.setLayers = () => set(null);
      r.__noLoads = true;
      set(null);
    }
  });
  const VIEWS = [['M', 17.6], ['J', 17.2], ['Y', 17.4], ['S', 17.4], ['W', 17.8], ['E', 17.8]];
  const shots = async (tag, built, mirrored) => {
    await bare();
    const out = {};
    for (const [v, zoom] of VIEWS) {
      let [lng, lat] = built.views[v];
      await page.evaluate(([c, z]) => { window.tw.map.jumpTo({ center: c, zoom: z, bearing: 0, pitch: 0 }); window.tw.overlay.invalidate(); }, [[lng, lat], zoom]);
      await settle(1500);
      const file = join(ARTIFACTS, `t3297-mirror-${tag}-${v}.png`);
      // the map's own box, whose middle is the view's centre, so a flip of it is the mirror about the centre
      const box = await page.locator('canvas.maplibregl-canvas').boundingBox();
      const w = 2 * Math.floor(box.width / 2);
      out[v] = (await page.screenshot({ path: file, clip: { x: box.x, y: box.y, width: w, height: box.height } })).toString('base64');
    }
    return out;
  };
  // the share of drawn pixels (not the background in either) where `a` and `b` flipped are apart by more than
  // `tol` in a channel at every pixel within 1 px; a picture of where, red
  const compare = (a, b, name) => page.evaluate(async ([a, b, tol]) => {
    const img = async (s) => { const i = new Image(); i.src = 'data:image/png;base64,' + s; await i.decode(); return i; };
    const [ia, ib] = await Promise.all([img(a), img(b)]);
    const W = ia.width, H = ia.height;
    const read = (i, flip) => { const c = new OffscreenCanvas(W, H), g = c.getContext('2d'); if (flip) { g.translate(W, 0); g.scale(-1, 1); } g.drawImage(i, 0, 0); return g.getImageData(0, 0, W, H).data; };
    const A = read(ia, false), B = read(ib, true);
    const bg = [A[0], A[1], A[2]];
    const isBg = (D, k) => Math.abs(D[k] - bg[0]) + Math.abs(D[k + 1] - bg[1]) + Math.abs(D[k + 2] - bg[2]) < 12;
    let drawn = 0, off = 0;
    const diff = new OffscreenCanvas(W, H), dg = diff.getContext('2d'), dd = dg.createImageData(W, H);
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const k = (y * W + x) * 4;
      if (isBg(A, k) && isBg(B, k)) continue;
      drawn++;
      let best = 1e9;
      for (let dy = -1; dy <= 1 && best > tol; dy++) for (let dx = -1; dx <= 1; dx++) {
        const q = ((y + dy) * W + x + dx) * 4;
        best = Math.min(best, Math.max(Math.abs(A[k] - B[q]), Math.abs(A[k + 1] - B[q + 1]), Math.abs(A[k + 2] - B[q + 2])));
      }
      const bad = best > tol;
      if (bad) off++;
      dd.data[k] = bad ? 255 : A[k] * 0.3 + 180; dd.data[k + 1] = bad ? 0 : A[k + 1] * 0.3 + 180; dd.data[k + 2] = bad ? 0 : A[k + 2] * 0.3 + 180; dd.data[k + 3] = 255;
    }
    dg.putImageData(dd, 0, 0);
    const blob = await diff.convertToBlob();
    const buf = new Uint8Array(await blob.arrayBuffer());
    let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode(...buf.subarray(i, i + 0x8000));
    return { drawn, off, share: drawn ? off / drawn : 0, png: btoa(s) };
  }, [a, b, 40]).then((r) => { writeFileSync(join(ARTIFACTS, `t3297-mirror-diff-${name}.png`), Buffer.from(r.png, 'base64')); return r; });

  try {
    const A = await game('right', 1);
    check(!A.refused.length && A.rightHand === 1, 'A: the layout keeping right, built', A.refused.join(' | '));
    const shotsA = await shots('A', A);
    const B = await game('left', -1);
    check(!B.refused.length && B.rightHand === -1, 'B: the layout mirrored keeping left, built', B.refused.join(' | '));
    check(A.badges === B.badges, 'B has the same flat-junction badges as A', `${A.badges} and ${B.badges}`);
    const shotsB = await shots('B', B);
    const C = await game('right', -1);
    check(!C.refused.length && C.rightHand === 1, 'C: the layout mirrored keeping right, built', C.refused.join(' | '));
    const shotsC = await shots('C', C);
    let worstB = 0, bestC = 1;
    // the views with a train in them or a turn-back crossover; the single branch has no line, and at the dead end
    // (W) no train stands at that moment and its scissors are the same either way
    const sided = new Set(['M', 'J', 'Y', 'S', 'E']);
    for (const [v] of VIEWS) {
      const b = await compare(shotsA[v], shotsB[v], `AB-${v}`), c = await compare(shotsA[v], shotsC[v], `AC-${v}`);
      console.log(`  view ${v}: B flipped ${(100 * b.share).toFixed(2)}% of ${b.drawn} drawn pixels apart, C flipped ${(100 * c.share).toFixed(2)}%`);
      worstB = Math.max(worstB, b.share);
      if (sided.has(v)) bestC = Math.min(bestC, c.share);
      check(b.drawn > 2000, `view ${v} has something drawn`, `${b.drawn} pixels`);
    }
    check(worstB < 0.005, 'every view of B flipped matches A', `worst ${(100 * worstB).toFixed(2)}%`);
    check(bestC > Math.max(0.002, 2 * worstB), 'and C flipped (the mirror keeping right) does not, in every view with trains', `least ${(100 * bestC).toFixed(2)}%`);
    const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
    check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
  } catch (e) {
    console.log('FAIL', e?.stack ?? e);
    fail++;
  } finally {
    await chrome.close?.();
  }
}
console.log(fail ? `FAIL: ${fail}` : 'PASS: all');
process.exit(fail ? 1 : 0);
