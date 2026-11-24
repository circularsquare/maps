// T-4139: the edit bench. One fixed set of edits replayed on a real save in a real game (checks/lib/play.mjs: the
// real toolbar, pointer and panels), each timed from the input to the frame that shows it, on cores claimed and
// pinned, so two builds can be compared and two runs of one build agree. Every edit-speed task reports its numbers
// before and after. How to read it: notes/T-4139.md.
//
//   node trainworld/tools/edit_bench.mjs [--url http://localhost:6158/] [--save asia|nyc|<file>[,..]] [--runs 3]
//        [--calculating] [--no-ledger] [--label L] [--compare <hash or run id>] [--only a,b] [--cores 4]
//        [--baseline] [--heavy "<reason>"] [--dry-run]
//
//   The budget (trainworld/CLAUDE.md, T-4213; tools/bench_gate.mjs). Inside .tw-trees/<name>/: 2 invocations in all
//   (counted in <tree>/.bench-runs.json), --runs 1 (the default there), the default save is nyc, the asia save (and
//   any tokyo/belgrade file) needs --heavy "<reason>" (still counted), --compare is refused: the run is diffed against
//   master's stored lines (editbench/baseline.json) by itself. --dry-run checks the gates and exits. On the main
//   tree nothing is refused; the orchestrator writes the baseline with --baseline (replaces the file with this run's
//   lines). A run with more than 4 other cores claimed at its start or end is stamped NOISY (n) on its lines.
//
//   (default)        paused clock, trips final first: the edits' own wait
//   --calculating    also a second pass in a fresh game with the edits made while trips are calculating
//   --compare <hash or run id> the change per action against one run: the run id (printed at the end of a run, in the ledger's `run`) or the newest run of that build hash; only that run's lines are read
//   --show [hash or run id]    print one run's table from the ledger, no run (the newest run without an argument)
//   --steps          each edit's steps, ms from the input: posted, word, drawn ahead, rows, network in, taken, drawn (T-4097)
//
// Actions (times in ms, median of --runs repeats; shows = the first frame that changes, built = the frame that
// draws the edit as built): ghost trail (px the drawn ghost trails the pointer at a steady speed, and the answer
// time of each move), a track piece, a station, Construct a line, a stop click (removed, added back), a stop
// reorder, undo, deleting a line. The ledger is the main tree's notes/edit-ledger.jsonl, from any tree.
//
// Steady numbers: it claims cores in the root ledger (cores.py) and pins the Chrome and its workers to them, holds
// a lock so two benches never run together, and records how busy those cores were before it started (`loaded`).
import { spawn, spawnSync } from 'node:child_process';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync, unlinkSync } from 'node:fs';
import { cpus } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { TREE_NAME, refuse, gateRun, noiseWatch } from './bench_gate.mjs';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const TREE = resolve(HERE, '../..');
const MAPS = TREE.replace(/[\\/]\.tw-trees[\\/][^\\/]+$/, '');
const LEDGER = join(MAPS, 'trainworld', 'notes', 'edit-ledger.jsonl');
const HOME = join(MAPS, 'artifacts', 'trainworld', 'editbench');
const LOCK = join(HOME, 'bench.lock');
const BASELINE = join(HOME, 'baseline.json');                           // master's lines of one run

const SAVES = {
  asia: join(MAPS, 'artifacts', 'trainworld', 't3600-asianita-1852.save'),
  nyc: join(TREE, 'trainworld', 'T-007-real-nyc.save'),
};

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : (args[i + 1] === undefined || args[i + 1].startsWith('--') ? true : args[i + 1]); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const r1 = (x, d = 0) => (typeof x === 'number' ? Math.round(x * 10 ** d) / 10 ** d : x);
const pct = (a, p) => { const s = a.filter((x) => typeof x === 'number' && !Number.isNaN(x)).sort((x, y) => x - y); if (!s.length) return null; return s[Math.min(s.length - 1, Math.max(0, Math.ceil(p * s.length) - 1))]; };
const median = (a) => { const s = a.filter((x) => typeof x === 'number' && !Number.isNaN(x)).sort((x, y) => x - y); if (!s.length) return null; const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const git = (...a) => spawnSync('git', ['-C', TREE, ...a], { encoding: 'utf8', windowsHide: true }).stdout?.trim() ?? '';
const log = (...a) => console.log(...a);

// ------------------------------------------------------------------------------------------ cores, lock, load

function claim(want, name, what, min) {
  const r = spawnSync('python', [join(MAPS, 'cores.py'), 'claim', String(want), name, '--pid', String(process.pid), '--min', String(min), '--timeout', '1800', what], { encoding: 'utf8', windowsHide: true, cwd: MAPS, stdio: ['ignore', 'pipe', 'inherit'] });
  const n = Number((r.stdout ?? '').trim().split('\n').at(-1));
  return r.status === 0 && n >= 1 ? n : null;
}
const release = (name) => spawnSync('python', [join(MAPS, 'cores.py'), 'release', name], { stdio: 'ignore', windowsHide: true, cwd: MAPS });
const alive = (pid) => { try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; } };

/** one bench at a time: waits (up to 40 minutes) for a live holder, takes over a dead one's lock */
async function takeLock() {
  mkdirSync(HOME, { recursive: true });
  const until = Date.now() + 40 * 60000;
  for (;;) {
    let holder = null;
    try { holder = JSON.parse(readFileSync(LOCK, 'utf8')); } catch { /* none */ }
    if (!holder || !alive(holder.pid)) break;
    if (Date.now() > until) throw new Error('another edit bench (pid ' + holder.pid + ') held the lock for 40 minutes');
    log('waiting for the edit bench of pid ' + holder.pid + ' (' + holder.what + ')');
    await sleep(15000);
  }
  writeFileSync(LOCK, JSON.stringify({ pid: process.pid, what: opt('label', git('rev-parse', '--short=8', 'HEAD')), at: new Date().toISOString() }));
  const drop = () => { try { if (JSON.parse(readFileSync(LOCK, 'utf8')).pid === process.pid) unlinkSync(LOCK); } catch { /* gone */ } };
  process.on('exit', drop);
  return drop;
}

const coreTimes = () => cpus().map((c) => ({ ...c.times }));
/** busy percent of the given core numbers between two snapshots */
function busyOn(a, b, cores) {
  let busy = 0, all = 0;
  for (const i of cores) {
    const d = Object.fromEntries(Object.keys(a[i]).map((k) => [k, b[i][k] - a[i][k]]));
    const t = d.user + d.nice + d.sys + d.irq + d.idle;
    busy += t - d.idle; all += t;
  }
  return all ? Math.round((100 * busy) / all) : 0;
}

/** The game's Chrome runs above normal priority on its own cores, so the below-normal builds and checks other sessions
 * leave on the same cores take a smaller share of them (reserve_cores.ps1 keeps sessions' children below normal). */
function boost(profile) {
  const ps = "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($env:TW_BOOST) } | ForEach-Object { try { (Get-Process -Id $_.ProcessId -ErrorAction Stop).PriorityClass = 'AboveNormal' } catch {} }";
  const go = () => { try { spawn('powershell', ['-NoProfile', '-NonInteractive', '-Command', ps], { windowsHide: true, stdio: 'ignore', env: { ...process.env, TW_BOOST: profile.replace(/\//g, '\\') } }).on('error', () => {}); } catch { /* best effort */ } };
  go();
  return setInterval(go, 5000);
}

// ------------------------------------------------------------------------------------------ in the page

/** Installed once per game. Stamps what the edit path does, in page time: inputs, messages both ways, the state
 * taken, the renderer's calls and the frames drawn. Each wrap checks that the thing is there, so a build that has
 * renamed one loses that mark and no more. */
function INSTALL() {
  if (window.__eb) return;
  const eb = window.__eb = { ev: [], ptr: null, ghost: null, ghostDrawn: null, answers: [], trail: [], trailOn: false, dirty: false };
  const now = () => performance.now();
  const push = (k, o = {}) => { eb.ev.push({ t: now(), k, ...o }); eb.dirty = true; };
  const tw = window.tw, c = tw.client, r = tw.renderer;
  if (typeof c.post === 'function') {
    const post = c.post.bind(c);
    c.post = (msg, tr) => { if (msg.kind !== 'epoch') push('post', { kind: msg.kind, op: msg.op?.op, seq: msg.seq }); return post(msg, tr); };
  }
  if (typeof c.arrive === 'function') {
    const arrive = c.arrive.bind(c);
    c.arrive = (msg) => {
      if (msg.kind === 'edited') push('arrive', { kind: 'edited', seq: msg.seq, ok: msg.ok, ms: msg.ms });
      else if (msg.kind === 'applied') push('arrive', { kind: 'applied', seq: msg.seq });
      else if (msg.kind === 'state') { const s = msg.state; push('arrive', { kind: 'state', rows: !!s.rows, net: !!s.net, service: !!s.service }); }
      return arrive(msg);
    };
  }
  if (typeof c.takeStateNow === 'function') {
    const tsn = c.takeStateNow.bind(c);
    c.takeStateNow = (msg) => { const t = now(); try { return tsn(msg); } finally { push('take', { ms: now() - t, rows: !!msg.state?.rows, net: !!msg.state?.net }); } };
  }
  for (const m of ['setNetwork', 'setService', 'keepPreviewAsPlaced', 'addPlaced', 'setTrips', 'flushStaged']) {
    if (typeof r[m] !== 'function') continue;
    const f = r[m].bind(r);
    r[m] = (...a) => { const t = now(); try { return f(...a); } finally { push('r.' + m, { ms: now() - t }); } };
  }
  // T-4145: an edit drawn ahead of the network (game/pending.ts); `build` false is a network clearing it, not a show
  if (typeof r.setAhead === 'function') {
    const sa = r.setAhead.bind(r);
    // (only a model that draws something: one emptied by a refusal or a network is no show)
    const drawsSome = (m) => m.lines.size || m.hideEdges.size || m.edges.size || m.hideStations.size || m.stations.length || m.colours.size || m.track.length || m.lineStrokes.length || m.hideLines?.size || m.rawStations?.length;
    r.setAhead = (m, build = true) => { const t = now(); try { return sa(m, build); } finally { if (build && drawsSome(m)) push('r.setAhead', { ms: now() - t }); } };
  }
  if (typeof r.stage === 'function') {
    const st = r.stage.bind(r);
    r.stage = (v, at, put, lf) => { push('r.stage', { v }); return st(v, at, () => { const t = now(); put(); push('r.staged', { ms: now() - t }); }, lf); };
  }
  if (typeof r.draw === 'function') {
    const draw = r.draw.bind(r);
    r.draw = (...a) => {
      const t = now();
      try { return draw(...a); } finally {
        eb.ghostDrawn = eb.ghost;
        if (eb.dirty) { eb.dirty = false; eb.ev.push({ t, k: 'draw', ms: now() - t }); }
      }
    };
  }
  for (const typ of ['pointerdown', 'keydown']) addEventListener(typ, (e) => eb.ev.push({ t: e.timeStamp, k: 'input', typ, key: e.key }), true);
  addEventListener('pointermove', (e) => { eb.ptr = { t: e.timeStamp, x: e.clientX, y: e.clientY }; }, true);
  const tl = tw.tools;
  if (tl && typeof tl.showPreview === 'function') {
    const sp = tl.showPreview.bind(tl);
    tl.showPreview = (a, key) => { const p = a?.pts; if (p && p.length >= 6) eb.ghost = { x: p[p.length - 3], y: p[p.length - 2] }; return sp(a, key); };
  }
  if (typeof c.preview === 'function') {
    const pv = c.preview.bind(c);
    c.preview = (req, cb) => { const t = now(); return pv(req, (a) => { eb.answers.push({ t, ms: now() - t }); return cb(a); }); };
  }
  // T-4147: an edit's guess drawn in a panel or the bus layer is a show too (game/pending.ts bumps its version for each)
  import('/src/game/pending.ts').then((m) => { let first = true; m.pendingVersion.subscribe(() => { if (first) { first = false; return; } push('r.setAhead', { ms: 0 }); requestAnimationFrame(() => eb.ev.push({ t: now(), k: 'draw', ms: 0 })); }); }).catch(() => {});
  eb.trailStart = () => {
    eb.trail = []; eb.answers = []; eb.ghost = null; eb.ghostDrawn = null; eb.trailOn = true;
    const loop = () => {
      if (!eb.trailOn) return;
      const p = eb.ptr, g = eb.ghostDrawn;
      if (p && g) {
        const rect = tw.map.getCanvas().getBoundingClientRect();
        const [sx, sy] = tw.tools.screen(g.x, g.y);
        eb.trail.push({ t: now(), d: Math.hypot(rect.left + sx - p.x, rect.top + sy - p.y) });
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  };
  eb.trailStop = () => { eb.trailOn = false; return { trail: eb.trail, answers: eb.answers, lastMove: eb.ptr?.t ?? null }; };
  /** resolves when the client has nothing in flight and no edit-path event happened for quietMs (or after maxMs: -1) */
  eb.quiet = async (maxMs = 25000, quietMs = 1500) => {
    const t0 = now();
    while (now() - t0 < maxMs) {
      await new Promise((res) => setTimeout(res, 100));
      const last = eb.ev.filter((e) => e.k !== 'input').at(-1)?.t ?? 0;
      if ((c.pending?.size ?? 0) === 0 && now() - last > quietMs) return Math.round(now() - t0);
    }
    return -1;
  };
}

// ------------------------------------------------------------------------------------------ reading the events

/** one record per edit message: input to the frames that show it */
function editsOf(ev) {
  const out = [];
  const r0 = (x) => (x == null ? null : Math.round(x));
  for (const [i, p] of ev.entries()) {
    if (!(p.k === 'post' && p.kind === 'edit')) continue;
    const inp = ev.slice(0, i).reverse().find((e) => e.k === 'input' && e.t <= p.t && p.t - e.t < 3000);
    const t0 = inp?.t ?? p.t;
    const after = (j, pred) => { for (let k = j; k < ev.length; k++) if (pred(ev[k])) return k; return -1; };
    const frameAfter = (k) => { if (k < 0) return null; const d = ev[after(k, (e) => e.k === 'draw')]; return d ? d.t + (d.ms ?? 0) - t0 : null; };
    const placed = after(i, (e) => e.k === 'r.keepPreviewAsPlaced' || e.k === 'r.addPlaced');
    // (an edit shown ahead is drawn at its click, before its message)
    const aheadAt = after(inp ? ev.indexOf(inp) : Math.max(0, i - 6), (e) => e.k === 'r.setAhead' && e.t >= t0 - 30);
    const take = after(i, (e) => e.k === 'take' && e.rows);
    const net = after(i, (e) => e.k === 'r.setNetwork' || e.k === 'r.setService' || e.k === 'r.staged');
    const ans = after(i, (e) => e.k === 'arrive' && e.kind === 'edited' && e.seq === p.seq);
    const word = after(i, (e) => e.k === 'arrive' && e.kind === 'applied' && e.seq === p.seq);
    const netIn = after(i, (e) => e.k === 'arrive' && e.kind === 'state' && e.net);
    const netTake = after(i, (e) => e.k === 'take' && e.net);
    const rows = frameAfter(take), built = frameAfter(net), drawnAhead = aheadAt >= 0 && (net < 0 || aheadAt < net) ? frameAfter(aheadAt) : null;
    // (T-4210: the track tool's stand-in for a placed segment is drawn ahead too)
    const standIn = [frameAfter(placed), drawnAhead].filter((x) => x != null).reduce((m, x) => Math.min(m, x), Infinity);
    const shows = [standIn, rows, built, drawnAhead].filter((x) => x != null && Number.isFinite(x));
    out.push({ op: p.op, seq: p.seq, post: r0(p.t - t0), answer: ans >= 0 ? r0(ev[ans].t - t0) : null, ok: ans >= 0 ? ev[ans].ok : null,
      standIn: Number.isFinite(standIn) ? r0(standIn) : null, rows: r0(rows), built: r0(built), shows: shows.length ? r0(Math.min(...shows)) : null, work: ans >= 0 ? ev[ans].ms ?? null : null,
      // the steps between (T-4097): the edit's word in, the drawn-ahead frame, the network's state in and taken
      word: word >= 0 ? r0(ev[word].t - t0) : null, ahead: r0(drawnAhead), netIn: netIn >= 0 ? r0(ev[netIn].t - t0) : null, netTake: netTake >= 0 ? r0(ev[netTake].t + (ev[netTake].ms ?? 0) - t0) : null });
  }
  return out;
}

// ------------------------------------------------------------------------------------------ the game and its places

/** Places that depend on the save, from the save itself: a stretch of open ground off the end of its biggest rail
 * line (the same on every run of one save), and that line's stops. */
function PLACES() {
  const s = window.tw.world.value.save, w = window.tw.world.value;
  const byId = new Map(s.stations.map((x) => [String(x.id), x]));
  const km = (a, b) => Math.hypot((b.lng - a.lng) * Math.cos(a.lat * Math.PI / 180), b.lat - a.lat) * 111;
  const rail = s.lines.filter((l) => !(l.bus || l.ferry || l.kind === 'bus' || l.kind === 'ferry' || w.lineStats?.[l.id]?.bus));
  const L = rail.sort((a, b) => b.stops.length - a.stops.length || a.id - b.id)[0];
  const ends = [[L.stops.at(-1), L.stops.at(-2)], [L.stops[0], L.stops[1]]].map(([e, p]) => ({ e: byId.get(String(e)), p: byId.get(String(p)) }));
  const crowd = (st) => s.stations.filter((b) => km(st, b) < 3).length;
  ends.sort((a, b) => crowd(a.e) - crowd(b.e) || a.e.id - b.e.id);
  const { e: E, p: P } = ends[0];
  const kx = 1 / (111 * Math.cos(E.lat * Math.PI / 180)), ky = 1 / 111;
  let dx = (E.lng - P.lng) / kx, dy = (E.lat - P.lat) / ky; const dl = Math.hypot(dx, dy); dx /= dl; dy /= dl;
  const rot = (a) => [dx * Math.cos(a) - dy * Math.sin(a), dx * Math.sin(a) + dy * Math.cos(a)];
  const at = (o, v, d) => [o[0] + v[0] * d * kx, o[1] + v[1] * d * ky];
  const e0 = [E.lng, E.lat], u = rot(Math.PI / 3), perp = rot(Math.PI / 2);
  const sites = Array.from({ length: 8 }, (_, k) => k).map((k) => {
    const p0 = at(at(e0, u, 2.0), perp, 1.4 * k);
    const pts = [0, 1, 2].map((i) => at(p0, [dx, dy], 0.85 * i));
    const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    return { pts, st: [mid(pts[0], pts[1]), mid(pts[1], pts[2])] };
  });
  const n = L.stops.length;
  const pick = (i) => { const st = byId.get(String(L.stops[i])); return { id: st.id, name: st.name, lng: st.lng, lat: st.lat }; };
  return { line: { num: L.num, name: L.name, n, id: L.id }, sites, ghost: at(at(e0, u, 2.0), perp, -1.6), mid: pick(n >> 1), swaps: Array.from({ length: 8 }, (_, k) => Math.min((n >> 1) + 2 * k, n - 3)) };
}

/** pacing: `fn(i)` every 16.7 ms on an absolute clock */
async function paced(n, fn) {
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    const wait = t0 + i * (1000 / 60) - performance.now();
    if (wait > 1) await sleep(wait);
    await fn(i);
  }
}

// ------------------------------------------------------------------------------------------ one game, all actions

/** Plays every action on `g` (an open game), pushing samples into `acc` as `acc[name] = [ms, ...]`. */
async function play(g, runs, only, acc, note) {
  const page = g.page;
  const add = (name, v) => { if (typeof v === 'number') (acc[name] ??= []).push(v); };
  const want = (a) => !only || only.includes(a);
  await page.evaluate(INSTALL);
  const C = await page.evaluate(PLACES);
  note('places: line ' + C.line.name + ' (' + C.line.n + ' stops), ' + Math.min(runs, C.sites.length) + ' sites');
  const quiet = (ms = 25000, q = 1500) => page.evaluate(([a, b]) => window.__eb.quiet(a, b), [ms, q]);
  const mark = () => page.evaluate(() => window.__eb.ev.length);
  const since = async (m) => editsOf(await page.evaluate((m) => window.__eb.ev.slice(m), m));
  const sample = async (fn, ops, keys, label) => {
    await quiet(20000, 1000);
    const m = await mark();
    let err = null;
    try { await fn(); } catch (e) { err = e.message.split('\n')[0]; }
    await quiet(40000, 1500);
    const seen = await page.evaluate((m) => window.__eb.ev.slice(m).filter((x) => x.k === 'post' || x.k === 'input').map((x) => x.k === 'post' ? x.kind + ':' + (x.op ?? '') : 'input:' + x.typ), m);
    const edits = (await since(m)).filter((e) => !ops || ops.includes(e.op));
    const e = edits[0];
    if (!e || err) { note('  ' + label + ': no sample' + (err ? ' (' + err + ')' : ' (no edit message; saw ' + (seen.join(', ') || 'nothing') + ')')); return null; }
    if (e.ok === false) note('  ' + label + ': refused');
    if (opt('steps', false)) note('  ' + label + ': ' + JSON.stringify(e));
    for (const [k, name] of Object.entries(keys)) add(name, e[k]);
    return e;
  };

  const track = { snap: { streets: false, real: false, roads: false, beside: false } };
  const setTrack = async () => { try { await g.act.tool('track', track); } catch { await g.act.tool('track'); } };

  // ghost: a fixed path at a steady speed over open ground
  if (want('ghost')) for (let k = 0; k < runs; k++) {
    await g.act.key('Escape');
    await g.act.camera(C.ghost, 15);
    await setTrack();
    const [c0] = await g.toScreen([C.ghost]);
    const S = { x: Math.round(c0.x), y: Math.round(c0.y) };
    const way = [[0, 0], [260, 0], [260, 160], [0, 160], [0, 0], [-120, 90]].map(([a, b]) => [S.x + a, S.y + b]);
    const steps = [];
    for (let i = 1; i < way.length; i++) {
      const [ax, ay] = way[i - 1], [bx, by] = way[i], l = Math.hypot(bx - ax, by - ay), n = Math.max(1, Math.round(l / 6));
      for (let j = 1; j <= n; j++) steps.push([ax + ((bx - ax) * j) / n, ay + ((by - ay) * j) / n]);
    }
    await g.driver.move(S.x, S.y, { steps: 2 });
    await g.driver.press(S.x, S.y);
    await sleep(600);
    await quiet(15000, 800);
    await page.evaluate(() => window.__eb.trailStart());
    await paced(steps.length, (i) => g.driver.move(steps[i][0], steps[i][1], { steps: 1 }));
    await sleep(1500);
    const r = await page.evaluate(() => window.__eb.trailStop());
    const moving = r.trail.filter((x) => x.t <= r.lastMove + 5).map((x) => x.d);
    const rest = r.trail.filter((x) => x.t > r.lastMove + 1200).map((x) => x.d);
    const answers = r.answers.map((x) => x.ms);
    if (moving.length < 20) note('  ghost: only ' + moving.length + ' frames with a ghost');
    add('ghost trail px median', median(moving)); add('ghost trail px p90', pct(moving, 0.9));
    add('ghost answer ms median', median(answers)); add('ghost answer ms p90', pct(answers, 0.9));
    if (rest.length) add('ghost rest px', median(rest));
    const caught = r.trail.find((x) => x.t > r.lastMove && x.d <= (median(rest) ?? 0) + 3);
    add('ghost catch-up ms', caught ? caught.t - r.lastMove : null);
    await g.act.key('Escape');
  }

  // sites: a track piece, a station, a line, Construct
  const names = [];
  for (const [k, site] of C.sites.slice(0, runs).entries()) {
    const [a, b, c] = site.pts;
    await g.act.key('Escape');
    await g.act.camera(site.pts[1], 14.5);
    if (want('track') || want('station') || want('construct') || want('delete')) {
      await setTrack();
      await sample(() => g.act.drawTrack([a, b]), ['route'], { standIn: 'track piece: stand-in', built: 'track piece: built' }, 'track piece ' + (k + 1));
      await quiet(20000, 1000);
      await g.act.drawTrack([b, c]);
      await quiet(20000, 1000);
      await g.act.tool('station');
      await sample(() => g.act.placeStation(site.st[0]), ['addStation', 'addBrtStation'], { shows: 'station: shows', built: 'station: built' }, 'station ' + (k + 1));
      await quiet(20000, 1000);
      await g.act.placeStation(site.st[1]);
      await quiet(20000, 1000);
      await g.act.key('Escape');
      const name = 'EB' + (k + 1);
      await g.act.newLine(site.st, { name, zoom: 15 });
      await quiet(20000, 1000);
      names.push(name);
      await g.act.openLine(name);
      await sample(() => g.act.click('Construct', { within: '#inspector', exact: false }), ['constructLine'], { shows: 'construct a line: shows', built: 'construct a line: built' }, 'construct ' + (k + 1));
      await g.act.key('Escape');
    }
  }

  // T-4148: the select tool's edits that change how track and stations look, on blueprint laid at the later sites
  const doEdit = (op) => page.evaluate(async (op) => {
    let edit;
    try { edit = (await import('/src/game/world.ts')).edit; } catch { edit = (o) => window.tw.client.edit(o); }
    return edit(op, true);
  }, op);
  const settle = async () => { await quiet(20000, 1000); };
  // --shots <dir>: a picture before an edit, just after its message (before the network), and once it settled
  const SHOTS = opt('shots', false) ? String(opt('shots')) : null;
  const shot = async (name) => { if (SHOTS) await page.screenshot({ path: join(SHOTS, name + '.png'), clip: { x: 250, y: 150, width: 900, height: 600 } }); };
  const shotEdit = async (op, name) => {
    await shot(name + '-0');
    const done = page.evaluate(async (op) => { const { edit } = await import('/src/game/world.ts'); return edit(op, true); }, op);
    await sleep(Number(opt('shot-wait', 60)));
    await shot(name + '-1');
    const a = await done;
    await quiet(20000, 1000);
    await shot(name + '-2');
    return a;
  };
  if (want('pieces')) for (let k = 0; k < runs; k++) {
    const site = C.sites[runs + k];
    if (!site) { note('  pieces: no site ' + (runs + k)); break; }
    const [a, b, c] = site.pts;
    await g.act.key('Escape');
    await g.act.camera(site.pts[1], 14.5);
    const was = await page.evaluate(() => { const w = window.tw.world.value; return { e: w.edges.map((x) => x.id), s: w.save.stations.map((x) => x.num) }; });
    await setTrack();
    await g.act.drawTrack([a, b]); await settle();
    await g.act.drawTrack([b, c]); await settle();
    await g.act.tool('station');
    await g.act.placeStation(site.st[0]); await settle();
    await g.act.placeStation(site.st[1]); await settle();
    await g.act.key('Escape');
    const F = await page.evaluate((was) => { const w = window.tw.world.value; return { edges: w.edges.filter((e) => !was.e.includes(e.id)).map((e) => e.id), stations: w.save.stations.filter((s) => !was.s.includes(s.num)).map((s) => s.num) }; }, was);
    if (!F.edges.length) { note('  pieces ' + (k + 1) + ': no new edges'); continue; }
    const up = async () => { await g.act.key('Control+Z'); await settle(); };
    const plain = async (op) => { await doEdit(op); await settle(); };
    await plain({ op: 'edgesTracks', edges: F.edges, tracks: 2 });
    await sample(() => (SHOTS && k === 0 ? shotEdit({ op: 'edgesKind', edges: F.edges, kind: 2 }, 'kind') : doEdit({ op: 'edgesKind', edges: F.edges, kind: 2 })), ['edgesKind'], { shows: 'track type: shows', built: 'track type: built' }, 'track type ' + (k + 1));
    await plain({ op: 'edgesKind', edges: F.edges, kind: 1 });
    await sample(() => (SHOTS && k === 0 ? shotEdit({ op: 'doubleUp', edge: F.edges[0] }, 'double') : doEdit({ op: 'doubleUp', edge: F.edges[0] })), ['doubleUp'], { shows: 'double up: shows', built: 'double up: built' }, 'double up ' + (k + 1));
    await up();
    await plain({ op: 'edgesTracks', edges: F.edges, tracks: 1 });
    await sample(() => (SHOTS && k === 0 ? shotEdit({ op: 'oneWays', edges: F.edges, dirs: F.edges.map(() => 1) }, 'oneway') : doEdit({ op: 'oneWays', edges: F.edges, dirs: F.edges.map(() => 1) })), ['oneWays'], { shows: 'one way on: shows', built: 'one way on: built' }, 'one way on ' + (k + 1));
    await sample(() => (SHOTS && k === 0 ? shotEdit({ op: 'oneWays', edges: F.edges, dirs: F.edges.map(() => 0) }, 'oneway-off') : doEdit({ op: 'oneWays', edges: F.edges, dirs: F.edges.map(() => 0) })), ['oneWays'], { shows: 'one way off: shows', built: 'one way off: built' }, 'one way off ' + (k + 1));
    await plain({ op: 'edgesTracks', edges: F.edges, tracks: 2 });
    if (F.stations.length) {
      await sample(() => (SHOTS && k === 0 ? shotEdit({ op: 'blueprintOff', edges: [], stations: F.stations, off: true }, 'stationoff') : doEdit({ op: 'blueprintOff', edges: [], stations: F.stations, off: true })), ['blueprintOff'], { shows: 'station off: shows', built: 'station off: built' }, 'station off ' + (k + 1));
      await plain({ op: 'blueprintOff', edges: [], stations: F.stations, off: false });
    }
    await plain({ op: 'construct', edges: F.edges, stations: F.stations });
    await sample(() => (SHOTS && k === 0 ? shotEdit({ op: 'rebuild', edges: F.edges, stations: [], level: null, kind: 2, tracks: 0 }, 'rebuild') : doEdit({ op: 'rebuild', edges: F.edges, stations: [], level: null, kind: 2, tracks: 0 })), ['rebuild'], { shows: 'rebuild: shows', built: 'rebuild: built' }, 'rebuild ' + (k + 1));
  }

  // stops, reorder, undo on the save's biggest line
  if (want('stop') || want('reorder') || want('undo')) {
    await g.act.key('Escape');
    await g.act.openLine(C.line.name);
    await g.act.click('Edit stops', { exact: false });
    await g.act.camera([C.mid.lng, C.mid.lat], 15);
    await sleep(1500);
    const stopOps = ['toggleStop', 'insertStop', 'dropStop', 'setStops', 'stopPassed'];
    for (let k = 0; k < runs && want('stop'); k++) {
      await sample(() => g.act.press([C.mid.lng, C.mid.lat]), stopOps, { shows: 'stop click, removed: shows', rows: 'stop click, removed: panel rows', built: 'stop click, removed: line path' }, 'stop remove ' + (k + 1));
      await sample(() => g.act.press([C.mid.lng, C.mid.lat]), stopOps, { shows: 'stop click, added: shows', rows: 'stop click, added: panel rows', built: 'stop click, added: line path' }, 'stop add ' + (k + 1));
    }
    await g.act.click('Stop editing', { exact: false }).catch(() => {});
    await g.act.key('Escape');
    for (let k = 0; k < runs && (want('reorder') || want('undo')); k++) {
      const i = C.swaps[k];
      const e = await sample(() => page.evaluate(async ([num, i]) => {
        let edit;
        try { edit = (await import('/src/game/world.ts')).edit; } catch { edit = (op) => window.tw.client.edit(op); }
        const l = window.tw.world.value.save.lines.find((x) => x.num === num);
        const s = l.stops.map(Number); const t = s[i]; s[i] = s[i + 1]; s[i + 1] = t;
        return edit({ op: 'setStops', line: num, stops: s });
      }, [C.line.num, i]), ['setStops'], { shows: 'stop reorder: shows', built: 'stop reorder: built' }, 'reorder ' + (k + 1));
      if (e) await sample(() => g.act.key('Control+Z'), ['undo'], { shows: 'undo: shows', built: 'undo: built' }, 'undo ' + (k + 1));
    }
  }

  // T-4147: stop order and the group panel's stop edits, line groups, bus and ferry panels, on the save's own lines (each
  // put back by an undo or the opposite edit)
  const live = (fn, arg) => page.evaluate(fn, arg);
  if (want('order') && C.line.n >= 8) {
    const nOf = C.line.n;
    await g.act.key('Escape');
    await g.act.camera([C.mid.lng, C.mid.lat], 14);
    const undo = async () => { await g.act.key('Control+Z'); await settle(); };
    for (let k = 0; k < runs; k++) {
      const i = C.swaps[k];
      const swapped = () => live(([num, i]) => { const l = window.tw.world.value.save.lines.find((x) => x.num === num); const s = l.stops.map(Number); const t = s[i]; s[i] = s[i + 1]; s[i + 1] = t; return s; }, [C.line.num, i]);
      const order = await swapped();
      await sample(() => doEdit({ op: 'orderStops', line: C.line.num, stops: order }), ['orderStops'], { shows: 'stops: order changed, shows', built: 'stops: order changed, network' }, 'orderStops ' + (k + 1));
      await undo();
      await sample(() => doEdit({ op: 'dropStop', line: C.line.num, index: 0 }), ['dropStop'], { shows: 'group panel: drop an end stop, shows', built: 'group panel: drop an end stop, network' }, 'dropStop end ' + (k + 1));
      await undo();
      await sample(() => doEdit({ op: 'dropStop', line: C.line.num, index: (nOf >> 1) + k }), ['dropStop'], { shows: 'group panel: drop a middle stop, shows', built: 'group panel: drop a middle stop, network' }, 'dropStop middle ' + (k + 1));
      await undo();
      await sample(() => doEdit({ op: 'endLineAt', line: C.line.num, index: nOf - 4 - k }), ['endLineAt'], { shows: 'group panel: end line here, shows', built: 'group panel: end line here, network' }, 'endLineAt ' + (k + 1));
      await undo();
    }
  }
  if (want('groups')) {
    const F = await live((num) => { const w = window.tw.world.value; const L = w.save.lines.filter((l) => l.num !== num && !w.lineStats[l.id]?.bus && !l.group); return L.slice(0, 2).map((l) => l.num); }, C.line.num);
    if (F.length < 2) note('  groups: no two ungrouped rail lines');
    else for (let k = 0; k < runs; k++) {
      await g.act.key('Escape');
      await sample(() => doEdit({ op: 'groupMembers', lines: F, groups: [], group: 'new', parent: 0 }), ['groupMembers'], { shows: 'groups: merge two lines, shows', rows: 'groups: merge two lines, rows' }, 'groupMembers ' + (k + 1));
      await settle();
      const gr = await live((n) => window.tw.world.value.save.lines.find((l) => l.num === n)?.group ?? 0, F[0]);
      if (!gr) { note('  groups: the merge made no group'); break; }
      await sample(() => doEdit({ op: 'groupName', group: gr, name: 'Group bench ' + k }), ['groupName'], { shows: 'groups: name, shows', rows: 'groups: name, rows' }, 'groupName ' + (k + 1));
      await settle();
      await sample(() => doEdit({ op: 'groupLook', group: gr, symbol: 'GB', colour: '#cc3300' }), ['groupLook'], { shows: 'groups: look, shows', rows: 'groups: look, rows' }, 'groupLook ' + (k + 1));
      await settle();
      await sample(() => doEdit({ op: 'groupMembers', lines: F, groups: [], group: 0, parent: 0 }), ['groupMembers'], { shows: 'groups: split, shows', rows: 'groups: split, rows' }, 'groupMembers split ' + (k + 1));
      await settle();
    }
  }
  for (const kind of ['bus', 'ferry'].filter(want)) {
    const B = await live((kind) => {
      const w = window.tw.world.value;
      const l = w.save.lines.find((x) => { const b = w.lineStats[x.id]?.bus; return b && (kind === 'ferry' ? b.ferry : !b.ferry && !b.hand && !b.brt); });
      if (!l) return null;
      const b = w.lineStats[l.id].bus;
      return { num: l.num, built: b.built, spacing: b.spacing, hand: b.hand, boat: b.ferry ? (b.ferry.boat ?? 0) : null, route: b.route, kinds: b.kinds, ports: b.ferry ? b.ferry.ports.map((p) => ({ node: p.node })) : null, x: b.stops[Math.max(1, b.stops.length >> 1)] };
    }, kind);
    if (!B) note('  ' + kind + ': no line on this save');
    else {
      await g.act.key('Escape');
      const ll = await live((num) => { const b = window.tw.world.value.lineStats[window.tw.world.value.save.lines.find((l) => l.num === num).id].bus; const s = b.stops[b.stops.length >> 1] ?? b.stops[0]; return s ? [s.lng, s.lat] : null; }, B.num);
      if (ll) await g.act.camera(ll, 14);
      const undo = async () => { await g.act.key('Control+Z'); await settle(); };
      const route3 = B.route.length && B.kinds ? Array.from({ length: B.kinds.length }, (_, i) => [B.route[2 * i], B.route[2 * i + 1], B.kinds[i]]).flat() : null;
      for (let k = 0; k < runs; k++) {
        if (kind === 'bus') {
          await sample(() => doEdit({ op: 'busSpacing', line: B.num, spacing: 400 + 100 * k }), ['busSpacing'], { shows: 'bus: stop spacing, shows', rows: 'bus: stop spacing, rows', built: 'bus: stop spacing, network' }, 'busSpacing ' + (k + 1));
          await settle();
          if (route3) {
            await sample(() => doEdit({ op: 'busRoute', line: B.num, route: route3 }), ['busRoute'], { shows: 'bus: route, shows', rows: 'bus: route, rows', built: 'bus: route, network' }, 'busRoute ' + (k + 1));
            await settle();
          }
          await sample(() => doEdit({ op: 'busHand', line: B.num }), ['busHand'], { shows: 'bus: place stops manually, shows', rows: 'bus: place stops manually, rows', built: 'bus: place stops manually, network' }, 'busHand ' + (k + 1));
          await settle();
          await undo(); await undo();
          await sample(() => doEdit({ op: 'copyLine', line: B.num }), ['copyLine'], { shows: 'bus: copy line, shows', rows: 'bus: copy line, rows', built: 'bus: copy line, network' }, 'bus copy ' + (k + 1));
          await settle();
          await undo(); await undo();
        } else {
          await sample(() => doEdit({ op: 'ferryBoat', line: B.num, boat: (B.boat + 1) % 3 }), ['ferryBoat'], { shows: 'ferry: boat size, shows', rows: 'ferry: boat size, rows', built: 'ferry: boat size, network' }, 'ferryBoat ' + (k + 1));
          await settle();
          await undo();
        }
      }
    }
  }

  // delete the lines made above
  if (want('delete')) for (const name of names) {
    await g.act.key('Escape');
    await g.act.openLine(name);
    await sample(() => g.act.click('Delete line', { within: '#inspector', exact: false }), ['removeLine'], { shows: 'delete a line: shows', built: 'delete a line: gone' }, 'delete ' + name);
  }
}

// ------------------------------------------------------------------------------------------ main

function tableFrom(rows) {
  const w = Math.max(...rows.map((r) => r.action.length));
  return rows.map((r) => r.action.padEnd(w) + '  ' + String(r.median).padStart(7) + '  p90 ' + String(r.p90).padStart(7) + '  n ' + r.n).join('\n');
}

function readLedger() { return existsSync(LEDGER) ? readFileSync(LEDGER, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)) : []; }

function hashNow() {
  const dirty = git('status', '--porcelain', '--', 'trainworld/app', 'trainworld/sim');
  return git('rev-parse', '--short=8', 'HEAD') + (dirty ? '+changes' : '');
}

/** a run's id: lines before the ids existed share their hash, label and core mask */
const runOf = (r) => r.run ?? [r.hash, r.label ?? '', r.mask ?? ''].join('|');

/** the lines of one run, found by its run id or else by a build hash (then the newest run of that hash); the ledger is
 * shared with other agents, so lines of any other run are never read */
function linesOf(rows, id) {
  const hit = rows.filter((r) => r.run === id);
  const run = hit.length ? id : rows.findLast((r) => r.hash === id || r.hash?.startsWith(id)) && runOf(rows.findLast((r) => r.hash === id || r.hash?.startsWith(id)));
  const m = new Map();
  if (run) for (const r of rows) if (runOf(r) === run) m.set(r.save + '|' + r.pass + '|' + r.action, r);
  return m;
}

function compare(rows, cur, base) {
  const b = linesOf(rows, base);
  const out = [];
  for (const r of cur) {
    const o = b.get(r.save + '|' + r.pass + '|' + r.action);
    if (!o || !o.median || !r.median) { out.push(r.save + ' ' + r.pass + ' ' + r.action + ': no line for ' + base); continue; }
    const ch = Math.round((100 * (r.median - o.median)) / o.median);
    out.push(r.save + ' ' + r.pass.padEnd(5) + ' ' + r.action + ': ' + o.median + ' -> ' + r.median + ' (' + (ch > 0 ? '+' : '') + ch + '%)');
  }
  return out;
}

async function main() {
  if (opt('show', false)) {
    const rows = readLedger();
    const h = typeof opt('show') === 'string' ? opt('show') : runOf(rows.at(-1) ?? {});
    for (const [, r] of linesOf(rows, h)) log(r.save.padEnd(5), r.pass.padEnd(5), r.action.padEnd(36), String(r.median).padStart(7), 'p90', String(r.p90).padStart(7), 'n', r.n, r.loaded ? 'loaded' : '');
    return;
  }
  const url = String(opt('url', 'http://localhost:6158/'));
  const saveNames = String(opt('save', TREE_NAME ? 'nyc' : 'asia,nyc')).split(',');
  if (TREE_NAME && opt('baseline', false)) refuse('--baseline is the orchestrator\'s, on master');
  const runs = Number(opt('runs', TREE_NAME ? 1 : 3));
  gateRun({ tool: 'edit_bench', saves: saveNames.map((s) => (SAVES[s] ? s + ' ' + SAVES[s] : s)), runs, compare: !!opt('compare', false), heavy: opt('heavy', false), dryRun: !!opt('dry-run', false) });
  const only = opt('only', null) ? String(opt('only')).split(',') : null;
  const passes = opt('calculating', false) ? ['quiet', 'calc'] : ['quiet'];
  const hash = String(opt('build', hashNow()));
  const name = 'editbench-' + process.pid;
  if (spawnSync('python', [join(MAPS, 'cores.py'), 'total'], { stdio: 'ignore', windowsHide: true, cwd: MAPS }).status !== 0) throw new Error('the cores ledger is over its cap: not measured (no measuring on an overbooked machine)');
  const drop = await takeLock();
  const want = Number(opt('cores', 4));
  const cores = claim(want, name, 'edit_bench ' + hash, Math.min(want, 3));
  if (!cores) throw new Error('no free cores within 30 minutes: not measured');
  process.env.TW_CORES = String(cores);
  // the quietest cores (above Anita's two), so another session's work on the machine lands on them least
  const q0 = coreTimes(); await sleep(3000); const q1 = coreTimes();
  const quietest = [...Array(q0.length).keys()].filter((i) => i >= 2).map((i) => [i, busyOn(q0, q1, [i])]).sort((a, b) => a[1] - b[1] || b[0] - a[0]).slice(0, cores).map(([i]) => i);
  if (!process.env.TW_CORE_MASK) process.env.TW_CORE_MASK = String(quietest.reduce((m, i) => m | (1 << i), 0));
  const { openGame } = await import('../checks/lib/play.mjs');
  const { corePlan } = await import('../checks/cores.mjs');
  const plan = corePlan();
  const noise = noiseWatch(cores);
  noise.sample();
  const since = Math.floor(Date.now() / 1000);
  const RUN = hash + '@' + new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '') + '-' + process.pid;
  const all = [];
  try {
    for (const sn of saveNames) {
      const file = SAVES[sn] ?? resolve(sn);
      if (!existsSync(file)) throw new Error('no save ' + file);
      for (const pass of passes) {
        const t = Date.now();
        // how busy the claimed cores are with nothing of ours running
        const c0 = coreTimes(); await sleep(2500);
        const pre = plan ? busyOn(c0, coreTimes(), plan.cores) : null;
        const c1 = coreTimes();
        log('\n' + sn + ' / ' + pass + ': cores ' + (plan?.cores.join(',') ?? 'unpinned') + ', those cores ' + pre + '% busy before the game');
        const g = await openGame({ url, save: file, saveVia: 'post', paused: true, prefix: 'tw-editbench-', size: [1400, 900] });
        const acc = {};
        let final = null;
        const boosting = boost(g.chrome.profile ?? g.chrome.dir ?? 'tw-editbench-');
        try {
          if (pass === 'quiet') {
            await g.waitDemand({ timeout: Number(opt('final-wait', 900)) * 1000 }).then(() => { final = true; }, () => { final = false; });
            if (!final) log('  trips were still calculating after ' + opt('final-wait', 900) + ' s: the quiet pass ran with them calculating');
          } else {
            await sleep(3000);
            final = !!(await g.state.demand()).final;
            if (final) log('  trips were already final: this pass is the quiet pass again');
          }
          await play(g, runs, only, acc, (s) => log(s));
          const busy = plan ? busyOn(c1, coreTimes(), plan.cores) : null;
          const rows = Object.entries(acc).map(([action, v]) => ({ action, median: r1(median(v), action.includes('px') ? 1 : 0), p90: r1(pct(v, 0.9), action.includes('px') ? 1 : 0), n: v.length, all: v.map((x) => r1(x, 1)) }));
          const line = { run: RUN, date: new Date().toISOString().slice(0, 16), hash, label: opt('label', null), save: sn, pass, final, cores: plan?.n ?? null, mask: plan?.cores.join(',') ?? null, loaded: pre != null && pre >= 15, preBusy: pre, busy, workers: plan?.workers ?? null };
          for (const r of rows) all.push({ ...line, ...r });
          log(tableFrom(rows));
          log('  ' + Math.round((Date.now() - t) / 1000) + ' s, those cores ' + busy + '% busy over the game' + (g.pageErrors().length ? ', ' + g.pageErrors().length + ' page errors: ' + String(g.pageErrors()[0]).slice(0, 120) : ''));
        } finally { clearInterval(boosting); await g.close().catch(() => {}); }
      }
    }
    noise.sample();
  } finally { release(name); drop(); }
  const noisy = noise.stamp();
  if (noisy) { log('\nNOISY: ' + noisy + ' other cores were claimed during the run (limit 4): report these numbers as noisy and stop'); for (const r of all) r.noisy = noisy; }
  if (TREE_NAME) for (const r of all) r.tree = TREE_NAME;
  const fair = spawnSync('python', [join(MAPS, 'cores.py'), 'total', '--since', String(since)], { stdio: 'ignore', windowsHide: true, cwd: MAPS }).status === 0;
  if (!fair) log('\nTHE CORES LEDGER WENT OVER ITS CAP DURING THE RUN: nothing appended to the ledger');
  const prior = readLedger();
  if (opt('baseline', false)) {
    if (fair && !noisy) { writeFileSync(BASELINE, JSON.stringify(all)); log('\nbaseline written (' + all.length + ' lines): ' + BASELINE); }
    else log('\nBASELINE NOT WRITTEN: ' + (noisy ? 'the run was noisy' : 'the cores ledger went over its cap'));
  }
  if (TREE_NAME && existsSync(BASELINE)) {
    const bl = JSON.parse(readFileSync(BASELINE, 'utf8'));
    if (bl.length) { log('\nagainst master\'s baseline (' + bl[0].hash + ', run ' + bl[0].run + '):'); for (const l of compare(bl, all, bl[0].run)) log('  ' + l); }
  } else if (TREE_NAME) log('\nNO MASTER BASELINE (' + BASELINE + '): nothing to diff against');
  if (opt('compare', false)) { { const b = linesOf(prior, String(opt('compare'))); log('\nagainst ' + opt('compare') + (b.size ? ' (run ' + runOf([...b.values()][0]) + ')' : '') + ':'); } for (const l of compare(prior, all, String(opt('compare')))) log('  ' + l); }
  if (!opt('no-ledger', false) && fair) {
    mkdirSync(join(MAPS, 'trainworld', 'notes'), { recursive: true });
    appendFileSync(LEDGER, all.map((r) => JSON.stringify(r)).join('\n') + '\n');
    log('\n' + all.length + ' lines in notes/edit-ledger.jsonl for build ' + hash + ', run ' + RUN);
  }
}

main().then(() => process.exit(0), (e) => { console.error(e); process.exit(1); });
