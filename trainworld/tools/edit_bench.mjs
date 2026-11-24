// T-4139: the edit bench. One fixed set of edits replayed on a real save in a real game (checks/lib/play.mjs: the
// real toolbar, pointer and panels), each timed from the input to the frame that shows it, on cores claimed and
// pinned, so two builds can be compared and two runs of one build agree. Every edit-speed task reports its numbers
// before and after. How to read it: notes/T-4139.md.
//
//   node trainworld/tools/edit_bench.mjs [--url http://localhost:6158/] [--save asia|nyc|<file>[,..]] [--runs 3]
//        [--calculating] [--no-ledger] [--label L] [--compare <hash>] [--only a,b] [--cores 4]
//
//   (default)        paused clock, trips final first: the edits' own wait
//   --calculating    also a second pass in a fresh game with the edits made while trips are calculating
//   --compare <hash> the change per action against the newest ledger lines of that build hash (a percent of the wait)
//   --show [hash]    print a build's table from the ledger, no run
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

const HERE = fileURLToPath(new URL('.', import.meta.url));
const TREE = resolve(HERE, '../..');
const MAPS = TREE.replace(/[\\/]\.tw-trees[\\/][^\\/]+$/, '');
const LEDGER = join(MAPS, 'trainworld', 'notes', 'edit-ledger.jsonl');
const HOME = join(MAPS, 'artifacts', 'trainworld', 'editbench');
const LOCK = join(HOME, 'bench.lock');

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
    const aheadAt = after(inp ? ev.indexOf(inp) : i, (e) => e.k === 'r.setAhead');
    const take = after(i, (e) => e.k === 'take' && e.rows);
    const net = after(i, (e) => e.k === 'r.setNetwork' || e.k === 'r.setService' || e.k === 'r.staged');
    const ans = after(i, (e) => e.k === 'arrive' && e.kind === 'edited' && e.seq === p.seq);
    const standIn = frameAfter(placed), rows = frameAfter(take), built = frameAfter(net), drawnAhead = aheadAt >= 0 && (net < 0 || aheadAt < net) ? frameAfter(aheadAt) : null;
    const shows = [standIn, rows, built, drawnAhead].filter((x) => x != null);
    out.push({ op: p.op, seq: p.seq, post: r0(p.t - t0), answer: ans >= 0 ? r0(ev[ans].t - t0) : null, ok: ans >= 0 ? ev[ans].ok : null,
      standIn: r0(standIn), rows: r0(rows), built: r0(built), shows: shows.length ? r0(Math.min(...shows)) : null, work: ans >= 0 ? ev[ans].ms ?? null : null });
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
    const edits = (await since(m)).filter((e) => !ops || ops.includes(e.op));
    const e = edits[0];
    if (!e || err) { note('  ' + label + ': no sample' + (err ? ' (' + err + ')' : ' (no edit message)')); return null; }
    if (e.ok === false) note('  ' + label + ': refused');
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

  // stops, reorder, undo on the save's biggest line
  if (want('stop') || want('reorder') || want('undo')) {
    await g.act.key('Escape');
    await g.act.openLine(C.line.name);
    await g.act.click('Edit stops', { exact: false });
    await g.act.camera([C.mid.lng, C.mid.lat], 15);
    await sleep(1500);
    const stopOps = ['insertStop', 'dropStop', 'setStops', 'stopPassed'];
    for (let k = 0; k < runs && want('stop'); k++) {
      await sample(() => g.act.press([C.mid.lng, C.mid.lat]), stopOps, { rows: 'stop click, removed: panel rows', built: 'stop click, removed: line path' }, 'stop remove ' + (k + 1));
      await sample(() => g.act.press([C.mid.lng, C.mid.lat]), stopOps, { rows: 'stop click, added: panel rows', built: 'stop click, added: line path' }, 'stop add ' + (k + 1));
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

/** the newest ledger lines of a build hash (one per save, pass and action) */
function linesOf(rows, hash) {
  const m = new Map();
  for (const r of rows) if (r.hash === hash || r.hash?.startsWith(hash)) m.set(r.save + '|' + r.pass + '|' + r.action, r);
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
    const h = typeof opt('show') === 'string' ? opt('show') : rows.at(-1)?.hash;
    for (const [, r] of linesOf(rows, h)) log(r.save.padEnd(5), r.pass.padEnd(5), r.action.padEnd(36), String(r.median).padStart(7), 'p90', String(r.p90).padStart(7), 'n', r.n, r.loaded ? 'loaded' : '');
    return;
  }
  const url = String(opt('url', 'http://localhost:6158/'));
  const saveNames = String(opt('save', 'asia,nyc')).split(',');
  const runs = Number(opt('runs', 3));
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
  const since = Math.floor(Date.now() / 1000);
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
          const line = { date: new Date().toISOString().slice(0, 16), hash, label: opt('label', null), save: sn, pass, final, cores: plan?.n ?? null, mask: plan?.cores.join(',') ?? null, loaded: pre != null && pre >= 15, preBusy: pre, busy, workers: plan?.workers ?? null };
          for (const r of rows) all.push({ ...line, ...r });
          log(tableFrom(rows));
          log('  ' + Math.round((Date.now() - t) / 1000) + ' s, those cores ' + busy + '% busy over the game' + (g.pageErrors().length ? ', ' + g.pageErrors().length + ' page errors: ' + String(g.pageErrors()[0]).slice(0, 120) : ''));
        } finally { clearInterval(boosting); await g.close().catch(() => {}); }
      }
    }
  } finally { release(name); drop(); }
  const fair = spawnSync('python', [join(MAPS, 'cores.py'), 'total', '--since', String(since)], { stdio: 'ignore', windowsHide: true, cwd: MAPS }).status === 0;
  if (!fair) log('\nTHE CORES LEDGER WENT OVER ITS CAP DURING THE RUN: nothing appended to the ledger');
  const prior = readLedger();
  if (opt('compare', false)) { log('\nagainst ' + opt('compare') + ':'); for (const l of compare(prior, all, String(opt('compare')))) log('  ' + l); }
  if (!opt('no-ledger', false) && fair) {
    mkdirSync(join(MAPS, 'trainworld', 'notes'), { recursive: true });
    appendFileSync(LEDGER, all.map((r) => JSON.stringify(r)).join('\n') + '\n');
    log('\n' + all.length + ' lines in notes/edit-ledger.jsonl for build ' + hash);
  }
}

main().then(() => process.exit(0), (e) => { console.error(e); process.exit(1); });
