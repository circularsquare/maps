// T-3666: a game kept open for probes. `start` opens it once in a headless Chrome of its own (a background
// process, one per Vite port, its cores claimed from the ledger); each later command runs in that open game and
// answers with one line of JSON in a few seconds, waiting on conditions, never fixed times.
//
//   node trainworld/checks/play.mjs start <city|file.save> --port 6195 [--cores 3] [--mode sandbox] [--run]
//        [--query 'opened=all&x=1'] [--size 1400x900] [--touch] [--via panel] [--idle 30] [--timeout 100] [--base trainworld] (default on :8800, the production build)
//   node trainworld/checks/play.mjs do '<calls>' [--timeout 60] [--full]      or  do --file probe.js
//   node trainworld/checks/play.mjs sim <read> ['<json arg>']                  or  sim '<page expression>'
//   node trainworld/checks/play.mjs edits <file.json>
//   node trainworld/checks/play.mjs reload [<city|file.save>]                  (after restarting your Vite)
//   node trainworld/checks/play.mjs status | stop
//
// `--port` is your Vite's; leave it out when this tree has one game open. Quoting: from Git Bash the calls in
// single quotes, JS strings inside in double ones; from PowerShell the other way round (it drops inner double
// quotes); anything long in a file (`do --file`). `start` returns once the game has loaded (a new game ~11 s, the
// real save ~30 s); its trips are calculated after that (`do 'wait.final()'`, ~2 minutes for the real save on 3
// cores). Past `--timeout` it says still loading, and the next command waits for the load. `--timeout` is at most
// 170 s (a tool call is moved to the background at 180 s): a wait that times out is simply asked again. The game
// stops by itself after `--idle` minutes unused (30).
//
// `do` runs JS in the open game; a lone expression is its answer, else end with `return`. In scope:
//   act, see, state      the harness (lib/play.mjs, lib/actions.mjs): act.camera(lngLat, zoom), act.click(name),
//                        act.press(lngLat), act.key(k), act.tool(...), drawTrack, placeStation, newLine, see.panel(),
//                        see.notes(), state.counts(), state.trips(), state.bubbles({ kind }), ...
//   wait.final()         the demand final, then state.trips(); after `edits` it waits for a solve newer than them
//   wait.panel(text)     a panel shown (holding `text`);  wait.selection(kind)  a selection (of that kind)
//   wait.bubbles({ kind, n })  demand bubbles drawn (each with its lngLat to press)
//   wait.lines(n)        n lines or more;  wait.days(n)  the clock n game days on (it must run; ~90 s a day at speed 2)
//   wait.until(get, ok)  any condition (`get` async, `ok` on its value)
//   clock({ run, speed }) start/stop the clock, speed step 0-2;  sim(read, arg) as the sim command;  shot(name, opts)
//   g, page              the game object and the Playwright page, for anything else
// A wait gives up after 20 s ({ timeout: ms } for longer), the final and days at the call's --timeout (60 s). The answer:
// {"ok":true,"ms":..,"result":..}, cut to 3000 characters unless --full; a failure exits 1 with the error, the
// tool notes and the status tag; page errors since the last call come as "pageErrors".
//
// `sim` reads the simulation: a named read from lib/reads.mjs STATE (counts, trips, lines, stations, clock, demand,
// money, selection, view, bubbles, ...) with an optional JSON argument, or any page expression (await allowed; Maps
// come back as objects, long arrays cut).
//
// `edits <file>`: a JSON list run in order, each waiting for the world to change: a raw edit op (as tw.client.edit
// takes; `{"$node": [lng, lat]}` becomes the nearest node's id, `{"$free": [lng, lat], "level": -1}` a free
// point), `{"line": [[lng, lat], ...], "name", "trains": [h, m, l]}` (track, a station at each point, built, a line),
// or `{"across": [lng, lat], "km": 2}` (a three-station line east to west, 6/4/3 trains). It answers `after`, the
// demand version before the edits, for `do 'wait.final({ after: N })'`.
// Snapshot and restore were tried and not kept (trainworld/notes/T-3666.md).
import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, openSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { randomBytes } from 'node:crypto';
import { tmpdir } from 'node:os';
import { isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ME = fileURLToPath(import.meta.url);
const TREE = fileURLToPath(new URL('../../', import.meta.url)).replace(/[\\/]$/, '');
const DIR = join(tmpdir(), 'tw-play');
const statePath = (port) => join(DIR, `${port}.json`);
const logPath = (port) => join(DIR, `${port}.log`);
const OUT_CAP = 3000;
const MAX_S = 170; // a tool call still running at 180 s is moved to the background

const alive = (pid) => { try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; } };
const readState = (port) => { try { return JSON.parse(readFileSync(statePath(port), 'utf8')); } catch { return null; } };
// (Windows refuses the rename while `start` has the file open to read it: try again a moment later)
const writeState = (s) => {
  mkdirSync(DIR, { recursive: true });
  const tmp = statePath(s.port) + '.tmp';
  writeFileSync(tmp, JSON.stringify(s));
  for (let i = 0; ; i++) {
    try { return renameSync(tmp, statePath(s.port)); } catch (e) { if (i >= 40 || e.code !== 'EPERM' && e.code !== 'EBUSY' && e.code !== 'EACCES') throw e; }
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 25);
  }
};
/** the pid is alive and still a game process (pids are reused, and a state file outlives a crash or a reboot) */
const ours = (pid) => {
  if (!alive(pid)) return false;
  if (process.platform !== 'win32') return true;
  const r = spawnSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', `(Get-CimInstance Win32_Process -Filter "ProcessId=${Number(pid)}").CommandLine`], { encoding: 'utf8', windowsHide: true });
  return /play\.mjs"?\s+serve\b/.test(r.stdout ?? '');
};
/** the open games, a stale state file removed */
const games = () => (existsSync(DIR) ? readdirSync(DIR) : []).filter((f) => /^\d+\.json$/.test(f)).map((f) => readState(f.slice(0, -5))).filter((s) => {
  if (s && ours(s.pid)) return true;
  if (s) rmSync(statePath(s.port), { force: true });
  return false;
});

function parseArgs(argv) {
  const pos = [], opt = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const k = a.slice(2), next = argv[i + 1];
      if (['run', 'touch', 'full', 'all'].includes(k)) opt[k] = true;
      else { opt[k] = next; i++; }
    } else pos.push(a);
  }
  return { pos, opt };
}

function fail(msg, code = 1) { console.log(JSON.stringify({ ok: false, error: msg })); process.exit(code); }

/** this tree's one open game, or the one on --port */
function pick(opt) {
  if (opt.port) {
    const s = readState(opt.port);
    if (!s || !ours(s.pid)) fail(`no game open on :${opt.port}; node trainworld/checks/play.mjs start <city|save> --port ${opt.port}`);
    return s;
  }
  const mine = games().filter((s) => s.tree === TREE);
  if (mine.length === 1) return mine[0];
  if (!mine.length) fail('no game open from this tree; node trainworld/checks/play.mjs start <city|save> --port <your Vite>');
  fail(`${mine.length} games open from this tree (ports ${mine.map((s) => s.port).join(', ')}); pass --port`);
}

/** a JSON line, cut to the cap unless `full` */
function emit(obj, full) {
  let s = JSON.stringify(obj, jsonable);
  if (!full && s.length > OUT_CAP) s = s.slice(0, OUT_CAP) + ` ...(${s.length} characters; --full for all)`;
  console.log(s);
}
function jsonable(_k, v) {
  if (v instanceof Map) return Object.fromEntries(v);
  if (v instanceof Set) return [...v];
  return v;
}

async function call(s, path, body, timeoutS) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), (timeoutS + 15) * 1000);
  try {
    const r = await fetch(`http://127.0.0.1:${s.ctl}/${path}`, { method: 'POST', headers: { 'x-token': s.token }, body: JSON.stringify(body ?? {}), signal: ctl.signal });
    return await r.json();
  } catch (e) {
    return { ok: false, error: `the game on :${s.port} did not answer (${e.name === 'AbortError' ? 'timed out' : e.message}); see ${s.log}` };
  } finally { clearTimeout(timer); }
}

const secs = (opt, dflt) => { const t = Number(opt.timeout ?? dflt); if (!(t > 0)) fail('--timeout wants seconds'); if (t > MAX_S) fail(`--timeout ${t} is over ${MAX_S} s (a tool call is moved to the background at 180 s): wait again in a second call`); return t; };

// ---- the commands

async function start({ pos, opt }) {
  const port = Number(opt.port);
  if (!port) fail('start wants --port <your Vite port>');
  const what = pos[0];
  if (!what) fail('start wants a city id (nyc) or a .save file');
  const save = /\.save$/i.test(what) || existsSync(what) ? resolve(what) : null;
  if (save && !existsSync(save)) fail(`no such save: ${save}`);
  const old = readState(port);
  if (old && ours(old.pid)) {
    if (old.tree !== TREE) fail(`:${port} has a game from another tree (${old.tree}); pick your own Vite port`);
    fail(`a game is already open on :${port} (${old.what}); play.mjs stop, or play.mjs reload <city|save>`);
  }
  // 8800 (serve.py) serves the game under /trainworld/; Git Bash rewrites a leading-slash argument into a Windows path, so --base is given as `trainworld`
  const base = String(opt.base ?? (port === 8800 ? 'trainworld' : '')).replace(/^\/+|\/+$/g, '');
  const url = `http://localhost:${port}/${base && base + '/'}`;
  // T-4210: a dev Vite serves the packs at /packs only: under a base every pack, water mask and boundary comes back as
  // the page ("New York could not load", each ghost waiting on a mask that never comes)
  if (base && port !== 8800) fail(`--base is for 8800's build; a Vite on :${port} serves the game at its root`);
  try { await fetch(url, { signal: AbortSignal.timeout(5000) }); } catch { fail(`no Vite on :${port}: node trainworld/checks/dev-nowatch.mjs ${port}`); }
  mkdirSync(DIR, { recursive: true });
  rmSync(statePath(port), { force: true });
  const fd = openSync(logPath(port), 'w');
  const conf = { port, url, what, save, mode: opt.mode ?? null, run: !!opt.run, query: opt.query ?? '', size: opt.size ?? null, touch: !!opt.touch, via: opt.via ?? 'post', idle: Number(opt.idle ?? 30) };
  const env = { ...process.env };
  if (opt.cores) env.TW_CORES_WANT = String(opt.cores);
  // cwd outside the tree, so the open game never holds a tree a landing wants to remove
  const child = spawn(process.execPath, [ME, 'serve', JSON.stringify(conf)], { cwd: DIR, detached: true, stdio: ['ignore', fd, fd], windowsHide: true, env });
  child.unref();
  const t0 = Date.now(), limit = secs(opt, 100) * 1000;
  for (;;) {
    const s = readState(port);
    if (s?.phase === 'ready' || s?.phase === 'failed') {
      if (s.phase === 'failed') { console.log(`failed: ${s.error}\n${tail(logPath(port))}`); process.exit(1); }
      console.log(`ready: ${s.what} on :${port} in ${s.loadS} s; ${s.cores} cores, pid ${s.pid}, log ${s.log}`);
      return;
    }
    if (!alive(child.pid)) { console.log(`failed: the game process ended\n${tail(logPath(port))}`); process.exit(1); }
    if (Date.now() - t0 > limit) { console.log(`still loading after ${Math.round((Date.now() - t0) / 1000)} s (pid ${child.pid}); the next play command waits for it`); return; }
    await new Promise((r) => setTimeout(r, 250));
  }
}

function tail(file, n = 12) { try { return readFileSync(file, 'utf8').trimEnd().split('\n').slice(-n).join('\n'); } catch { return ''; } }

async function client(cmd, { pos, opt }) {
  if (cmd === 'status') {
    const all = games();
    for (const g of all) console.log(`:${g.port} ${g.what} ${g.phase}, pid ${g.pid}, ${g.cores} cores, since ${new Date(g.startedAt).toLocaleTimeString()}, tree ${g.tree}`);
    if (!all.length) console.log('no games open');
    return;
  }
  const s = pick(opt);
  const full = !!opt.full;
  if (cmd === 'stop') {
    const r = await call(s, 'stop', {}, 20);
    for (let i = 0; i < 80 && alive(s.pid); i++) await new Promise((res) => setTimeout(res, 250));
    if (ours(s.pid)) {
      // the game process did not end: stop it (only that pid) and its Chrome by profile path
      try { process.kill(s.pid); } catch {}
      killChrome(s.profile);
      rmSync(statePath(s.port), { force: true });
      console.log(JSON.stringify({ ok: true, forced: true, note: r.error ?? null }));
    } else console.log(JSON.stringify({ ok: true, stopped: `:${s.port} ${s.what}` }));
    return;
  }
  const timeout = secs(opt, 60);
  let body;
  if (cmd === 'do') {
    const code = opt.file ? readFileSync(opt.file, 'utf8') : pos[0];
    if (!code) fail("do wants the calls: do 'see.panel()'");
    body = { code };
  } else if (cmd === 'sim') {
    if (!pos[0]) fail("sim wants a read (counts, trips, lines, clock, ...) or a page expression");
    body = { read: pos[0], arg: pos[1] ? JSON.parse(pos[1]) : undefined };
  } else if (cmd === 'edits') {
    if (!pos[0]) fail('edits wants a JSON file');
    body = { items: JSON.parse(readFileSync(pos[0], 'utf8')) };
  } else if (cmd === 'reload') {
    const what = pos[0];
    body = what ? { what, save: /\.save$/i.test(what) || existsSync(what) ? resolve(what) : null } : {};
  } else fail(`unknown command ${cmd}`);
  const r = await call(s, cmd, { ...body, timeout }, timeout);
  emit(r, full);
  if (!r.ok) process.exit(1);
}

function killChrome(profile) {
  if (!profile || process.platform !== 'win32') return;
  const ps = `Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($env:TW_PROFILE) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`;
  spawnSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', ps], { env: { ...process.env, TW_PROFILE: profile.replace(/\//g, '\\') }, stdio: 'ignore', windowsHide: true });
  try { rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }); } catch { /* left for staleTempProfiles */ }
}

// ---- the game process

async function serve(conf) {
  const { launchGames } = await import('./lib/play.mjs');
  const { STATE } = await import('./lib/reads.mjs');
  const { corePlan } = await import('./cores.mjs');
  const state = { port: conf.port, pid: process.pid, tree: TREE, what: conf.what, phase: 'starting', startedAt: Date.now(), log: logPath(conf.port), token: randomBytes(12).toString('hex'), ctl: null, cores: null, profile: null };
  const save = (patch) => { Object.assign(state, patch); writeState(state); };
  const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
  save({});

  let gamesOf = null, g = null, lastErr = 0, lastUsed = Date.now(), busy = null, loaded, editedAfter = null;
  const ready = new Promise((r) => { loaded = r; });
  const [w, h] = (conf.size ?? '1400x900').split('x').map(Number);

  async function open(what, savePath) {
    const t0 = Date.now();
    const city = savePath ? null : what;
    g = await gamesOf.open({ url: conf.url, city, save: savePath, saveVia: conf.via, mode: conf.mode, paused: !conf.run, query: conf.query });
    lastErr = 0;
    g.page.on('crash', () => { log('page crashed'); shutdown(1); });
    const loadS = Math.round((Date.now() - t0) / 100) / 10;
    log(`opened ${what} in ${loadS} s`);
    return loadS;
  }

  let stopping = false;
  async function shutdown(code = 0) {
    if (stopping) return;
    stopping = true;
    log('stopping');
    try { await gamesOf?.close(); } catch (e) { log('close:', e.message); }
    rmSync(statePath(conf.port), { force: true });
    process.exit(code);
  }

  // the control port opens at once, so a command sent while loading waits for the load
  const server = createServer((req, res) => {
    let raw = '';
    req.on('data', (d) => { raw += d; });
    req.on('end', async () => {
      const reply = (o) => { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(o, jsonable)); };
      if (req.headers['x-token'] !== state.token) { res.writeHead(403); res.end(); return; }
      lastUsed = Date.now();
      const what = req.url.slice(1);
      let body;
      try { body = JSON.parse(raw || '{}'); } catch { return reply({ ok: false, error: 'bad request' }); }
      if (what === 'stop') { reply({ ok: true }); setTimeout(() => shutdown(0), 50); return; }
      const t0 = Date.now(), deadline = t0 + (body.timeout ?? 60) * 1000;
      try { await Promise.race([ready, timeoutAt(deadline, 'the game to load')]); } catch (e) { return reply({ ok: false, ms: Date.now() - t0, error: e.message }); }
      // one call at a time: a later one waits for the earlier to end
      while (busy) {
        if (Date.now() > deadline) return reply({ ok: false, ms: Date.now() - t0, error: `busy: a ${busy.what} call started ${Math.round((Date.now() - busy.at) / 1000)} s ago is still running` });
        await Promise.race([busy.p, new Promise((r) => setTimeout(r, 250))]);
      }
      let done;
      const mine = { what, at: Date.now(), p: new Promise((r) => { done = r; }) };
      busy = mine;
      let out;
      try {
        const run = handle(what, body, deadline);
        // a body still running at its timeout keeps the game for 10 s more at most
        run.finally(() => { if (busy === mine) { busy = null; done(); } }).catch(() => {});
        setTimeout(() => { if (busy === mine) { log(`${what}: still running 10 s past its timeout; the next call goes ahead`); busy = null; done(); } }, deadline - Date.now() + 10000);
        out = { ok: true, ms: 0, result: await Promise.race([run, timeoutAt(deadline, `the ${what} call (its --timeout)`)]) };
      } catch (e) {
        out = { ok: false, ms: 0, error: String(e?.message ?? e).split('\n').slice(0, 3).join(' ') };
        try { out.notes = (await g.see.notes()).map((n) => n.text); out.status = await g.see.status(); } catch {}
      }
      out.ms = Date.now() - t0;
      const errs = g?.pageErrors() ?? [];
      if (errs.length > lastErr) { out.pageErrors = errs.slice(lastErr).map((s) => s.slice(0, 300)); lastErr = errs.length; }
      reply(out);
    });
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  save({ ctl: server.address().port });

  const timeoutAt = (deadline, what) => new Promise((_, rej) => setTimeout(() => rej(new Error(`timed out waiting for ${what}`)), Math.max(0, deadline - Date.now())));

  async function handle(what, body, deadline) {
    if (what === 'do') return runDo(body.code, deadline);
    if (what === 'sim') return sim(body.read, body.arg);
    if (what === 'edits') {
      // `after`: the demand version before the edits, for wait.final({ after }) once they are in
      const after = (await g.state.trips())?.version ?? 0;
      const out = [];
      for (const item of body.items) {
        if (Date.now() > deadline) throw new Error(`timed out after ${out.length} of ${body.items.length} edits`);
        out.push(await g.page.evaluate(EDIT, item));
      }
      // a plain wait.final() then wants a solve newer than the edits (none comes when every edit was refused)
      if (out.some((e) => e?.ok !== false)) editedAfter = after;
      return { after, edits: out };
    }
    if (what === 'reload') {
      const target = body.what ?? state.what, savePath = body.what ? body.save : conf.save;
      await g.close().catch(() => {});
      const loadS = await open(target, savePath);
      conf.save = savePath; save({ what: target });
      return { reopened: target, loadS };
    }
    throw new Error(`unknown call ${what}`);
  }

  async function sim(read, arg) {
    if (STATE[read]) return g.page.evaluate(STATE[read], arg ?? {});
    return g.page.evaluate(PAGE_EXPR, read);
  }

  async function runDo(code, deadline) {
    // a wait ends before the call does, so its own message (what it waited for, the last value) is the answer;
    // all but the final give up after 20 s unless given { timeout } in ms
    const left = (cap = Infinity) => Math.max(500, Math.min(cap, deadline - Date.now() - 2000));
    const until = (get, ok, o = {}) => g.until(get, ok, { ...o, timeout: left(o.timeout ?? 20000) });
    const wait = {
      final: async (o = {}) => { const t = await g.waitDemand({ polling: 250, after: editedAfter, ...o, timeout: left(o.timeout) }).then(() => g.state.trips()); if (editedAfter !== null && t.version > editedAfter) editedAfter = null; return t; },
      panel: (text = null) => until(() => g.see.panel(), (p) => !!p?.text && (!text || p.text.includes(text)), { what: `a panel${text ? ` with "${text}"` : ''}` }),
      selection: (kind = null) => until(() => g.state.selection(), (s) => !!s && (!kind || s.kind === kind), { what: `a selection${kind ? ` of kind ${kind}` : ''}` }),
      lines: (n) => until(() => g.state.counts(), (c) => c.lines >= n, { what: `${n} lines` }),
      bubbles: (o = {}) => until(() => g.state.bubbles(o), (b) => b.length >= (o.n ?? 1), { what: 'demand bubbles drawn', every: 300 }),
      async days(n, o = {}) {
        // a clock just started can be held a moment
        const c0 = await until(() => g.state.clock(), (c) => c.running && !c.held, { timeout: 5000, what: 'the clock to run (clock({ run: true }) first)' });
        return until(() => g.state.clock(), (c) => c.now >= c0.now + n * 86400, { what: `${n} game days`, every: 500, timeout: Infinity, ...o });
      },
      until,
    };
    const clock = ({ run, speed } = {}) => g.page.evaluate(({ run, speed }) => {
      const c = window.tw.clock;
      if (run !== undefined) c.setRunning(!!run);
      if (speed !== undefined) c.setSpeed(speed);
      return { running: c.running.peek(), speed: c.speed.peek() };
    }, { run, speed });
    const names = ['g', 'page', 'act', 'see', 'state', 'wait', 'clock', 'sim', 'shot'];
    const vals = [g, g.page, g.act, g.see, g.state, wait, clock, sim, g.shot];
    const AsyncFunction = (async () => {}).constructor;
    let fn;
    try { fn = new AsyncFunction(...names, `return (${code}\n);`); } catch { fn = new AsyncFunction(...names, code); }
    return fn(...vals);
  }


  // stop after `idle` minutes unused, so a forgotten game does not hold its cores
  setInterval(() => { if (Date.now() - lastUsed > conf.idle * 60000) { log(`unused for ${conf.idle} min`); shutdown(0); } }, 30000).unref();
  for (const sig of ['SIGINT', 'SIGTERM', 'SIGBREAK']) process.on(sig, () => shutdown(0));

  try {
    const t0 = Date.now();
    gamesOf = await launchGames({ prefix: `tw-play-${conf.port}-`, size: [w || 1400, h || 900], touch: conf.touch });
    const plan = corePlan();
    save({ profile: gamesOf.chrome.profile, cores: plan?.n ?? null });
    await open(conf.what, conf.save);
    loaded();
    save({ phase: 'ready', loadS: Math.round((Date.now() - t0) / 100) / 10 });
    lastUsed = Date.now();
  } catch (e) {
    log('failed:', e.message);
    for (const p of gamesOf?.chrome?.ctx?.pages?.() ?? []) log('page', p.url(), JSON.stringify(await p.evaluate(() => ({ tw: !!window.tw, world: !!window.tw?.world?.value, tools: !!window.tw?.tools, demand: !!window.twDemand?.client, lastLoad: window.tw?.client?.lastLoad ?? null, lines: window.tw?.world?.value?.save?.lines?.length })).catch((x) => x.message)));
    save({ phase: 'failed', error: e.message.split('\n')[0] });
    await gamesOf?.close().catch(() => {});
    setTimeout(() => process.exit(1), 300);
  }
}

/** a page expression's value as JSON text: Maps as objects, Sets and typed arrays as arrays, long lists cut */
const PAGE_EXPR = async (src) => {
  const v = await new Function(`return (async () => (${src}\n))();`)();
  const seen = new WeakSet();
  const cut = (x, depth) => {
    if (x === null || typeof x !== 'object') return typeof x === 'function' ? '[function]' : typeof x === 'bigint' ? Number(x) : x;
    if (seen.has(x)) return '[seen]';
    seen.add(x);
    if (depth > 6) return '[deeper]';
    if (ArrayBuffer.isView(x)) x = Array.from(x.slice(0, 50));
    if (x instanceof Map) x = Object.fromEntries([...x.entries()].slice(0, 200));
    else if (x instanceof Set) x = [...x];
    if (Array.isArray(x)) { const a = x.slice(0, 200).map((y) => cut(y, depth + 1)); if (x.length > 200) a.push(`...${x.length} in all`); return a; }
    const o = {};
    for (const [k, y] of Object.entries(x)) o[k] = cut(y, depth + 1);
    return o;
  };
  return cut(v, 0);
};

/** one item of an edits file, run in the page; waits for the world to change after each edit */
const EDIT = async (item) => {
  const tw = window.tw, c = tw.client;
  // `skip`: node ids left out (a line's own track is found among the nodes its routes added, not the network's)
  const nearNode = ([lng, lat], skip = null) => { const p = tw.lngLatToPack(lng, lat); let best = null, bd = Infinity; for (const n of tw.world.value.nodes.values()) { if (skip?.has(n.id)) continue; const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
  const free = ([lng, lat], level = -1) => { const p = tw.lngLatToPack(lng, lat); return { kind: 'free', x: p.x, y: p.y, level }; };
  const res = (v) => Array.isArray(v) ? v.map(res) : v && typeof v === 'object' ? ('$node' in v ? nearNode(v.$node) : '$free' in v ? free(v.$free, v.level ?? -1) : Object.fromEntries(Object.entries(v).map(([k, x]) => [k, res(x)]))) : v;
  const send = async (op) => {
    const w0 = tw.world.value, t = performance.now();
    const a = await c.edit(op);
    while (tw.world.value === w0 && performance.now() - t < 3000) await new Promise((r) => setTimeout(r, 30));
    const s = JSON.stringify(a ?? null);
    return s.length < 400 ? a : { ok: a?.ok, issues: a?.issues?.slice?.(0, 5), line: a?.line, keys: Object.keys(a) };
  };
  const line = async (pts, name, trains) => {
    const old = new Set([...tw.world.value.nodes.values()].map((n) => n.id)), mine = (p) => nearNode(p, old);
    // every step's refusal, named, so the first cause shows (a station on a crossing, then the line's path)
    const issues = [];
    const step = async (what, op) => { const a = await send(op); if (a?.ok === false) issues.push(`${what}: ${(a.issues ?? []).join(', ')}`); return a; };
    await step('track 1', { op: 'route', from: free(pts[0]), to: free(pts[1]), pis: [], single: false });
    for (let k = 2; k < pts.length; k++) await step(`track ${k}`, { op: 'route', from: { kind: 'node', node: mine(pts[k - 1]) }, to: free(pts[k]), pis: [], single: false });
    for (const [k, p] of pts.entries()) await step(`station ${k + 1}`, { op: 'addStation', at: { kind: 'node', node: mine(p) }, platform: 200, name: '' });
    await step('construct', { op: 'constructAll' });
    const r = await step('line', { op: 'addLine', stops: pts.map(mine), name: name ?? 'Check', colour: '#0a7d3b' });
    if (trains && r?.ok !== false && r?.line !== undefined) await step('trains', { op: 'setTrainCount', line: r.line, trains });
    return { ok: !issues.length, line: r?.ok !== false ? r?.line ?? null : null, issues };
  };
  if (item.op) return send(res(item));
  if (item.line) return line(item.line, item.name, item.trains);
  if (item.across) {
    const [lng, lat] = item.across, km = item.km ?? 2, d = km / (111.32 * Math.cos((lat * Math.PI) / 180));
    return line([-1, 0, 1].map((k) => [lng + k * d, lat]), item.name, item.trains ?? [6, 4, 3]);
  }
  throw new Error(`an edit wants "op", "line" or "across": ${JSON.stringify(item).slice(0, 100)}`);
};

const [cmd, ...rest] = process.argv.slice(2);
if (cmd === 'serve') await serve(JSON.parse(rest[0]));
else if (cmd === 'start') await start(parseArgs(rest));
else if (['do', 'sim', 'edits', 'reload', 'status', 'stop'].includes(cmd)) await client(cmd, parseArgs(rest));
else {
  console.log(readFileSync(ME, 'utf8').split('\n').filter((l) => l.startsWith('//')).slice(0, 14).map((l) => l.slice(3)).join('\n'));
  process.exit(cmd ? 1 : 0);
}
