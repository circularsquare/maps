// T-4025: the performance bench. A fixed set of saves measured the same way on any build: load, trips calculated,
// one fixed edit, memory, bytes downloaded, and counts of the work done that do not depend on how busy the machine
// is. `run` appends one line per save to notes/perf-ledger.jsonl and flags anything over 10% worse than the last
// publish. How to read it and when to run which saves: notes/T-4025.md.
//
//   node trainworld/tools/perf_bench.mjs build <commit | tree folder> [--label L]   a production build, printed path
//   node trainworld/tools/perf_bench.mjs run <dist> --label L [--saves nyc,tokyo,belgrade] [--runs 1]
//        [--published vX.Y.Z] [--no-ledger] [--edit-trips] [--min-cores 3]       measure, append to the ledger
//   node trainworld/tools/perf_bench.mjs compare <base dist> <head dist> [--saves ...] [--runs 2]
//                                                     base and head interleaved on the same cores: the only fair wall-clock
//   node trainworld/tools/perf_bench.mjs check [--label L]                          flags of the ledger's lines for L (default the last)
//   node trainworld/tools/perf_bench.mjs published <label> <vX.Y.Z>                 after a publish: its lines become the reference
//
// From the maps root or a tree; the ledger is always the main tree's. Builds go through checks/agent-tree.mjs; dists,
// Cargo target dirs, saves and each run's full record live in the main tree's artifacts/trainworld/perfbench/. Each run
// is a fresh headless Chrome with its own profile and 3 demand workers whatever the cores, so the work counts compare
// however busy the machine is; the cores claimed in cores.py only decide how fast it goes.
import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { createReadStream, existsSync, statSync, readFileSync, writeFileSync, appendFileSync, mkdirSync, rmSync, cpSync } from 'node:fs';
import { cpus } from 'node:os';
import { join, resolve, extname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const TREE = resolve(HERE, '../..');                                    // the tree this file is in
const MAPS = TREE.replace(/[\\/]\.tw-trees[\\/][^\\/]+$/, '');          // the main tree: ledger of cores, data, builds
const HOME = join(MAPS, 'artifacts', 'trainworld', 'perfbench');
const LEDGER = join(MAPS, 'trainworld', 'notes', 'perf-ledger.jsonl');   // the main tree's, from any tree
const MB = 1048576;
const WORSE = 0.10;
const WORKERS = 3;                                                     // demand workers in every run, whatever the cores
const WANT_CORES = 5;
/** --edit-trips: also wait for the trips after the edit (about the load's whole solve again on NYC); off by default */
let EDIT_TRIPS = false;                                                  // the page, the clock worker and 3 demand workers

const args = process.argv.slice(2);
const cmd = args[0];
const pos = args.filter((a, i) => i > 0 && !a.startsWith('--') && !args[i - 1].startsWith('--'));
const opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : (args[i + 1] === undefined || args[i + 1].startsWith('--') ? true : args[i + 1]); };

/** The bench. \`touches\`: what a change to it would show, for picking one save for a small change. */
const SAVES = {
  nyc: { file: join(TREE, 'trainworld', 'T-007-real-nyc.save'), what: 'the real New York save: one zone, 109 lines, the subway' },
  tokyo: { file: join(HOME, 'saves', 't3939-standin-13zones.save'), from: join(MAPS, 'artifacts', 'trainworld', 't3939-standin-13zones.save'),
    what: "Tokyo's real network with 13 zones open (T-3939's stand-in for the iPhone crash saves): many zones, the big solve" },
  belgrade: { file: join(HOME, 'saves', 'belgrade-df9whrkk-r1.save'), url: 'https://market.anita.garden/trainworld/community/df9whrkk/1/save',
    what: 'Belgrade (real world, traced, detailed), a Community save: 256 lines, mostly buses, 9 zones', timeoutS: 4800 },
};

const median = (a) => { const s = a.filter((x) => typeof x === 'number' && !Number.isNaN(x)).sort((x, y) => x - y); if (!s.length) return null; const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const r1 = (x, d = 1) => (typeof x === 'number' ? Math.round(x * 10 ** d) / 10 ** d : x);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const git = (dir, ...a) => spawnSync('git', ['-C', dir, ...a], { encoding: 'utf8', windowsHide: true }).stdout?.trim() ?? '';

// ---------------------------------------------------------------------------------------------- cores

/** cores from the ledger: `want`, or fewer once at least `min` are free; cores.py waits (never past the cap) up to
 * 30 minutes. Null when it gave up: the caller measures nothing rather than run on one core. */
function claim(want, name, what, min = want) {
  const r = spawnSync('python', [join(MAPS, 'cores.py'), 'claim', String(want), name, '--pid', String(process.pid), '--min', String(min), '--timeout', '1800', what], { encoding: 'utf8', windowsHide: true, cwd: MAPS });
  const n = Number((r.stdout ?? '').trim().split('\n').at(-1));
  return r.status === 0 && n >= 1 ? n : null;
}
/** cores.py total: false when the ledger is over the cap now, or was at any point since `since` (epoch s) */
const withinCap = (since) => spawnSync('python', [join(MAPS, 'cores.py'), 'total', ...(since ? ['--since', String(since)] : [])], { stdio: 'ignore', windowsHide: true, cwd: MAPS }).status === 0;
const release = (name) => spawnSync('python', [join(MAPS, 'cores.py'), 'release', name], { stdio: 'ignore', windowsHide: true, cwd: MAPS });
const cpuSnap = () => cpus().map((c) => c.times);
function busyPct(a, b) {
  let busy = 0, all = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Object.fromEntries(Object.keys(a[i]).map((k) => [k, b[i][k] - a[i][k]]));
    const t = d.user + d.nice + d.sys + d.irq + d.idle;
    busy += t - d.idle; all += t;
  }
  return all ? Math.round((100 * busy) / all) : null;
}

// ---------------------------------------------------------------------------------------------- build

/** agent-tree.mjs, which makes and removes trees with the shared junctions safely (it unlinks them first) */
const agentTree = (...a) => { const r = spawnSync('node', [join(MAPS, 'trainworld', 'checks', 'agent-tree.mjs'), ...a], { stdio: 'inherit', windowsHide: true, cwd: MAPS }); if (r.status !== 0) throw new Error('agent-tree.mjs ' + a.join(' ') + ' failed (' + r.status + ')'); };

/** A production build (shipped WASM profile, Vite) of a commit or of a tree's working copy, into HOME/dist/<label>. */
function build(src, label) {
  const out = join(HOME, 'dist', label);
  const sh = (c, a, o = {}) => { const r = spawnSync(c, a, { stdio: 'inherit', windowsHide: true, shell: process.platform === 'win32', ...o }); if (r.status !== 0) throw new Error(c + ' ' + a.join(' ') + ' failed (' + r.status + ')'); };
  let tree, temp = null, commit;
  if (existsSync(src) && statSync(src).isDirectory()) {
    tree = resolve(src);
    commit = git(tree, 'rev-parse', 'HEAD') + (git(tree, 'status', '--porcelain', '--', 'trainworld/app', 'trainworld/sim') ? '+changes' : '');
  } else {
    temp = 'bench-' + label.replace(/[^\w-]/g, '-');
    tree = join(MAPS, '.tw-trees', temp);
    if (existsSync(tree)) agentTree('remove', temp, '--force');
    agentTree('new', temp, src);
    commit = git(tree, 'rev-parse', 'HEAD');
    // agent-tree.mjs copies the main tree's uncommitted app/src/game/*.json in (the packs were built for them)
    const copied = git(tree, 'status', '--porcelain', '--', 'trainworld').split('\n').filter(Boolean).map((l) => l.slice(3));
    if (copied.length) { commit += '+main-json'; console.log('with the main tree\'s uncommitted ' + copied.join(', ')); }
  }
  const tt = join(HOME, 'target-track'), td = join(HOME, 'target-demand');
  if (!existsSync(tt) && existsSync(join(MAPS, '.tw-snap-target'))) cpSync(join(MAPS, '.tw-snap-target'), tt, { recursive: true });
  const name = 'perfbench-build-' + label;
  if (!claim(4, name, 'perf_bench build ' + label)) throw new Error('no 4 free cores within 30 minutes: not built');
  const env = { ...process.env, CARGO_BUILD_JOBS: '2', TW_LINK_PACKS: '1', PATH: join(process.env.USERPROFILE ?? '', '.cargo', 'bin') + ';' + process.env.PATH };
  const t = Date.now();
  try {
    const app = join(tree, 'trainworld', 'app');
    if (existsSync(out)) rmSync(out, { recursive: true, force: true });
    sh('python', [join(tree, 'trainworld', 'tools', 'snap_wasm.py'), tt, td], { cwd: app, env });
    sh('npx', ['vite', 'build', '--outDir', out, '--emptyOutDir'], { cwd: app, env });
  } finally {
    release(name);
    if (temp) agentTree('remove', temp, '--force');
  }
  writeFileSync(join(out, 'perf-build.json'), JSON.stringify({ label, commit, src, builtAt: new Date().toISOString(), buildS: Math.round((Date.now() - t) / 1000) }) + '\n');
  console.log('built ' + commit.slice(0, 8) + ' into ' + out + ' in ' + Math.round((Date.now() - t) / 1000) + ' s');
  return out;
}

// ---------------------------------------------------------------------------------------------- server

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.geojson': 'application/geo+json', '.wasm': 'application/wasm', '.png': 'image/png', '.svg': 'image/svg+xml', '.pbf': 'application/x-protobuf', '.woff2': 'font/woff2', '.ttf': 'font/ttf', '.webmanifest': 'application/manifest+json' };
/** checks/static-serve.mjs (Range, no-store) with the bytes it sent counted, packs apart */
function serve(dist) {
  const st = { bytes: 0, packs: 0, requests: 0 };
  const server = createServer((req, res) => {
    let p = decodeURIComponent((req.url ?? '/').split('?')[0]);
    if (p.endsWith('/')) p += 'index.html';
    const f = join(dist, p);
    if (!f.startsWith(dist) || !existsSync(f) || statSync(f).isDirectory()) { res.statusCode = 404; return res.end(); }
    const size = statSync(f).size;
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Content-Type', TYPES[extname(f)] ?? 'application/octet-stream');
    const m = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? '');
    let a = 0, b = size - 1;
    if (m) { a = Number(m[1]); b = m[2] ? Math.min(Number(m[2]), size - 1) : size - 1; res.statusCode = 206; res.setHeader('Content-Range', 'bytes ' + a + '-' + b + '/' + size); }
    res.setHeader('Content-Length', b - a + 1);
    st.requests++;
    res.on('finish', () => { st.bytes += b - a + 1; if (p.startsWith('/packs/')) st.packs += b - a + 1; });
    if (req.method === 'HEAD') return res.end();
    createReadStream(f, { start: a, end: b }).pipe(res);
  });
  return new Promise((r) => server.listen(0, '127.0.0.1', () => r({ st, url: 'http://127.0.0.1:' + server.address().port + '/', close: () => server.close() })));
}

// ---------------------------------------------------------------------------------------------- measuring

/** One PowerShell for the whole run: every second, the private bytes and CPU seconds of each process of the Chrome
 * whose command line holds \`profile\`, by process type. Peaks kept; \`cpu()\` reads the latest CPU seconds by type. */
function procWatch(profile) {
  const needle = profile.replace(/\//g, '\\').replace(/'/g, "''");
  const ps = "while ($true) { $o = @(); Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('" + needle + "') } | ForEach-Object { $t = if ($_.CommandLine -match '--type=(\\S+)') { $matches[1] } else { 'browser' }; if ($t -eq 'renderer' -and $_.CommandLine -match '--extension-process') { $t = 'ext' }; $o += ('{0}:{1}:{2}:{3}' -f $t, $_.PrivatePageCount, ($_.KernelModeTime + $_.UserModeTime), $_.ProcessId) }; [Console]::Out.WriteLine('S ' + ($o -join ' ')); [Console]::Out.Flush(); Start-Sleep -Milliseconds 1000 }";
  const p = spawn('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], { windowsHide: true });
  const w = { peak: { renderer: 0, total: 0, gpu: 0 }, last: null, samples: 0 };
  let buf = '';
  p.stdout.on('data', (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
      if (!line.startsWith('S')) continue;
      const by = {}, cpu = {};
      let total = 0, big = 0, pid = null;
      for (const part of line.slice(1).trim().split(' ').filter(Boolean)) {
        const [t, priv, c, id] = part.split(':');
        by[t] = (by[t] ?? 0) + Number(priv) / MB; cpu[t] = (cpu[t] ?? 0) + Number(c) / 1e7; total += Number(priv) / MB;
        if (t === 'renderer' && Number(priv) > big) { big = Number(priv); pid = Number(id); }
      }
      w.last = { by, cpu, total, pid, at: Date.now() };
      w.samples++;
      w.peak.renderer = Math.max(w.peak.renderer, by.renderer ?? 0);
      w.peak.gpu = Math.max(w.peak.gpu, by['gpu-process'] ?? 0);
      w.peak.total = Math.max(w.peak.total, total);
    }
  });
  return {
    w,
    /** the next sample's CPU seconds by type (waits for one taken after now) */
    async cpu() { const t = Date.now(); for (let k = 0; k < 100 && !(w.last && w.last.at > t); k++) await sleep(100); return w.last?.cpu ?? {}; },
    stop: () => { try { p.kill(); } catch { /* gone */ } },
  };
}

/** CPU seconds of each thread of the tab's process (tools/perf_threads.ps1), by thread id. Chrome's workers are threads
 * of the renderer, and CDP gives a worker no CPU time, so a worker's CPU is its thread's. */
function threadCpu(pid) {
  if (!pid) return {};
  const r = spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', join(HERE, 'perf_threads.ps1'), String(pid)], { encoding: 'utf8', windowsHide: true });
  const out = {};
  for (const l of (r.stdout ?? '').split(/\r?\n/)) { const m = /^([\d.,]+) (\d+)/.exec(l.trim()); if (m) out[m[2]] = Number(m[1].replace(/,/g, '')); }
  return out;
}
/** the threads' CPU between two readings, the busiest first: the top ones of a load are the demand workers */
const threadDelta = (a, b) => Object.entries(b).map(([id, s]) => s - (a[id] ?? 0)).filter((x) => x >= 0.05).sort((x, y) => y - x);

/** a raw CDP connection to the browser, so each worker's session can be spoken to */
async function rawCdp(port) {
  let v;
  for (let k = 0; k < 100 && !v; k++) v = await fetch('http://127.0.0.1:' + port + '/json/version').then((r) => r.json()).catch(() => sleep(200));
  const ws = new WebSocket(v.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0;
  const wait = new Map();
  ws.onmessage = (ev) => { const m = JSON.parse(ev.data); const w = m.id && wait.get(m.id); if (!w) return; wait.delete(m.id); if (m.error) w.j(new Error(m.error.message)); else w.r(m.result); };
  const send = (method, params = {}, sessionId) => new Promise((r, j) => {
    const i = ++id;
    wait.set(i, { r, j });
    ws.send(JSON.stringify({ id: i, method, params, ...(sessionId ? { sessionId } : {}) }));
    setTimeout(() => { if (wait.delete(i)) j(new Error('timeout ' + method)); }, 30000);
  });
  return { send, close: () => ws.close() };
}

const ctxName = (t) => t.type === 'page' ? 'page' : (t.url.replace(/^.*\//, '').replace(/\?.*$/, '').replace(/-[A-Za-z0-9_-]{8}\.js$/, '').replace(/\.js$/, '') || 'worker');

/** The page and each of its workers: heap used and CPU seconds (CDP ThreadTime); \`deep\` collects garbage first and adds
 * WASM memory. Workers of one script are numbered in the order they were met. */
function contexts(raw, base) {
  const known = new Map();
  return async function probe(deep = false) {
    const { targetInfos } = await raw.send('Target.getTargets', { filter: [{}] });
    const out = {};
    for (const t of targetInfos) {
      if (t.type !== 'page' && t.type !== 'worker') continue;
      if (t.type === 'page' && !t.url.startsWith(base)) continue;
      try {
        let k = known.get(t.targetId);
        if (!k) {
          const s = (await raw.send('Target.attachToTarget', { targetId: t.targetId, flatten: true })).sessionId;
          await raw.send('Performance.enable', { timeDomain: 'threadTicks' }, s).catch(() => raw.send('Performance.enable', {}, s).catch(() => {}));
          const n = ctxName(t);
          const same = [...known.values()].filter((x) => x.base === n).length;
          k = { s, base: n, name: same ? n + '#' + (same + 1) : n };
          known.set(t.targetId, k);
        }
        if (deep) await raw.send('HeapProfiler.collectGarbage', {}, k.s).catch(() => {});
        const h = await raw.send('Runtime.getHeapUsage', {}, k.s);
        const m = await raw.send('Performance.getMetrics', {}, k.s).catch(() => null);
        const metric = (n) => m?.metrics?.find((x) => x.name === n)?.value;
        const row = { heap: h.usedSize / MB, cpu: metric('ThreadTime') ?? null };
        if (deep) row.wasm = await wasmOf(raw, k.s);
        out[k.name] = row;
      } catch { /* a worker that just ended */ }
    }
    return out;
  };
}
async function wasmOf(raw, s) {
  try {
    const { result: proto } = await raw.send('Runtime.evaluate', { expression: 'WebAssembly.Memory.prototype' }, s);
    const { objects } = await raw.send('Runtime.queryObjects', { prototypeObjectId: proto.objectId }, s);
    const { result } = await raw.send('Runtime.callFunctionOn', { objectId: objects.objectId, functionDeclaration: 'function () { return this.reduce((a, m) => a + m.buffer.byteLength, 0); }', returnByValue: true }, s);
    return result.value / MB;
  } catch { return null; }
}

/** In the page: marks in ms since \`t0\` for the imported game: its lines drawn, its trips calculated (the newest solve's
 * final, sent after t0), the clock running. Also each demand worker's WASM, the most seen. */
const PROBE = ({ t0, edit }) => {
  const b = (window.__b = { t0, marks: {}, wasm: window.__b?.wasm ?? [] });
  const mark = (k) => { if (!(k in b.marks)) b.marks[k] = Math.round(performance.now() - t0); };
  const id = setInterval(() => {
    const w = window.tw?.world?.value, d = window.twDemand, v = d?.view?.();
    try { for (const m of d.client.memory?.() ?? []) b.wasm[m.worker] = Math.max(b.wasm[m.worker] ?? 0, m.wasm); } catch { /* older build */ }
    if (edit != null) { if (w?.save?.stations && !w.save.stations.some((s) => s.id === edit)) mark('shown'); }
    else if ((w?.save?.lines?.length ?? 0) > 0) mark('drawn');
    const s = d?.log?.findLast?.((x) => !d.client.fversions?.has(x.version));
    if ((edit != null ? 'shown' : 'drawn') in b.marks && s && s.arrivedAt >= t0 && v?.version === s.version && v.final && !d.client.waiting) mark('final');
    if ('final' in b.marks && window.tw?.clock?.held?.peek() === false && !window.tw.clock.capped?.peek()) { mark('playable'); clearInterval(id); }
  }, 20);
};

/** Counts of the work the load's solve did, read from twDemand's log; none depends on how busy the machine is. */
const COUNTS = (t0) => {
  const d = window.twDemand, v = d.view(), w = window.tw.world.value;
  const mine = d.log.filter((x) => x.arrivedAt >= t0);
  const s = mine.findLast((x) => !d.client.fversions?.has(x.version)) ?? mine.at(-1);
  const sum = (a) => (a ?? []).reduce((x, y) => x + (y ?? 0), 0);
  return {
    lines: w.save.lines.length, stations: w.save.stations.length,
    zones: s?.cities?.length ?? null, markets: s?.markets?.length ?? null, rounds: s?.byRound?.length ?? v.round ?? null,
    solves: mine.length, periodsSolved: s?.periods ? sum(s.periods.map((p) => p[1])) : null, clustersSolved: s?.clusters?.[1] ?? null,
    periodMs: s?.periodMs ? Math.round(sum(s.periodMs.flat())) : null, setupMs: s?.setupMs ? Math.round(sum(s.setupMs)) : null,
    rows: v.lines?.size ?? null, railTripsPerDay: Math.round(v.railTripsPerDay ?? 0),
  };
};

/** the station removed by the fixed edit: the middle stop of the line with the most stops (ties: the lowest id) */
const PICK = () => {
  const w = window.tw.world.value.save;
  const ids = new Map(w.stations.map((s) => [s.id, s]));
  const mid = (l) => l.stops[l.stops.length >> 1];
  const line = w.lines.filter((l) => l.stops.length >= 3 && ids.has(mid(l))).sort((a, b) => b.stops.length - a.stops.length || a.id - b.id)[0];
  return line ? { node: mid(line), line: line.id, stops: line.stops.length, name: ids.get(mid(line)).name } : null;
};

async function runOnce(save, base, srv, log) {
  const { launchChrome, collectErrors, GL } = await import('../checks/lib.mjs');
  const port = 9300 + Math.floor(Math.random() * 600);
  const chrome = await launchChrome({ prefix: 'tw-perf-', size: [1400, 900], args: [...GL, '--remote-debugging-port=' + port, '--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1'] });
  const procs = procWatch(chrome.profile);
  const errors = [];
  const raw = await rawCdp(port);
  const probe = contexts(raw, base);
  const ctxPeak = {};
  let sampling = true;
  const sampler = (async () => { while (sampling) { try { for (const [k, v] of Object.entries(await probe(false))) ctxPeak[k] = Math.max(ctxPeak[k] ?? 0, v.heap); } catch { /* closing */ } await sleep(2000); } })();
  const rec = { errors, last: null };
  try {
    const page = await chrome.ctx.newPage();
    collectErrors(page, errors, /favicon|openfreemap|AJAXError|ERR_NAME_NOT_RESOLVED|Failed to load resource/);
    srv.st.bytes = 0; srv.st.packs = 0; srv.st.requests = 0;
    await page.goto(base + '?debug=1&new=1&demandWorkers=' + WORKERS + '&v=' + Date.now());
    await page.waitForFunction(() => { const d = window.twDemand, v = d?.view?.(); return window.tw?.world?.value && v?.final && !d.client.waiting && window.tw.clock?.held?.peek() === false; }, undefined, { timeout: 240000, polling: 100 });
    rec.boot = await page.evaluate(() => Math.round(performance.now()));
    rec.bootBytes = srv.st.bytes;
    await page.waitForTimeout(1500);
    const bytes = readFileSync(save.file);
    const cpu0 = await procs.cpu(), ctx0 = await probe(false);
    const tabPid = procs.w.last?.pid, th0 = threadCpu(tabPid);
    srv.st.bytes = 0; srv.st.packs = 0; srv.st.requests = 0;
    const machine0 = cpuSnap();
    const t0 = await page.evaluate(({ data, PROBE }) => { const b = Uint8Array.from(data).buffer; const t = performance.now(); (0, eval)('(' + PROBE + ')')({ t0: t, edit: null }); window.tw.client.post({ kind: 'import', bytes: b }, [b]); return t; }, { data: Array.from(bytes), PROBE: PROBE.toString() });
    const state = () => page.evaluate(() => { const d = window.twDemand, v = d?.view?.(), s = d?.log?.at(-1); return { marks: window.__b?.marks, round: v?.round, final: v?.final, version: v?.version, logV: s?.version, waiting: !!d?.client?.waiting, held: window.tw?.clock?.held?.peek(), capped: window.tw?.clock?.capped?.peek?.() ?? null, lines: window.tw?.world?.value?.save?.lines?.length, solves: d?.log?.length,
      log: (d?.log ?? []).map((x) => ({ v: x.version, rough: !!x.rough, f: d.client.fversions?.has(x.version) ?? null, pub: x.published, rounds: x.byRound?.length, first: x.firstAt ? Math.round(x.firstAt) : null, refined: x.refinedAt ? Math.round(x.refinedAt) : null, arrived: Math.round(x.arrivedAt), zones: x.cities?.length, newZones: !!x.newZones })),
      tag: document.querySelector('.status-tag, .statusTag, [class*=status]')?.textContent?.slice(0, 120) ?? null, now: Math.round(performance.now()),
      loads: (() => { const l = d?.start?.loads ?? []; return { n: l.length, open: l.filter((x) => x.openedAt).length, got: l.filter((x) => x.downloadedAt).length, waiting: l.filter((x) => !x.openedAt).map((x) => x.item).slice(0, 12) }; })(),
      mem: (() => { try { return d.client.memory().map((m) => [m.wasm, m.heap, m.aux ? 'aux' : '']); } catch { return null; } })(),
      work: (() => { const s = d?.log?.at(-1); return s?.workLog ? s.workLog.slice(-4).map((w) => [w.i, w.stage, w.done, w.of]) : s?.work ?? null; })() }; }).catch((e) => ({ error: String(e).slice(0, 100) }));
    const ticker = setInterval(async () => { rec.last = await state(); if (process.env.PERF_VERBOSE) { console.log('    ' + JSON.stringify(rec.last)); await page.screenshot({ path: join(HOME, 'last.png') }).catch(() => {}); } }, 30000);
    try { await page.waitForFunction(() => window.__b?.marks?.playable !== undefined, undefined, { timeout: Number(process.env.PERF_TIMEOUT_S ?? save.timeoutS ?? 1200) * 1000, polling: 500 }); }
    finally { clearInterval(ticker); }
    await page.waitForTimeout(2000);
    const cpu1 = await procs.cpu(), th1 = threadCpu(tabPid), ctx1 = await probe(true);
    rec.threadsLoad = threadDelta(th0, th1);
    rec.busy = busyPct(machine0, cpuSnap());
    rec.marks = await page.evaluate(() => window.__b.marks);
    rec.bytes = { all: srv.st.bytes, packs: srv.st.packs, requests: srv.st.requests };
    rec.counts = await page.evaluate(COUNTS, t0);
    rec.demandWasm = await page.evaluate(() => window.__b.wasm);
    rec.cpuLoad = { renderer: (cpu1.renderer ?? 0) - (cpu0.renderer ?? 0), gpu: (cpu1['gpu-process'] ?? 0) - (cpu0['gpu-process'] ?? 0),
      byContext: Object.fromEntries(Object.entries(ctx1).map(([k, v]) => [k, v.cpu !== null && ctx0[k]?.cpu != null ? v.cpu - ctx0[k].cpu : v.cpu])) };
    rec.ctxEnd = ctx1;
    // the fixed edit
    const pick = await page.evaluate(PICK);
    rec.edit = pick;
    if (pick) {
      const e0 = await procs.cpu(), te0 = threadCpu(tabPid);
      await page.evaluate(({ PROBE, node }) => { const t = performance.now(); (0, eval)('(' + PROBE + ')')({ t0: t, edit: node }); window.__e0 = t; window.tw.client.edit({ op: 'removeStation', node }); }, { PROBE: PROBE.toString(), node: pick.node });
      if (EDIT_TRIPS) await page.waitForFunction(() => window.__b?.marks?.final !== undefined, undefined, { timeout: 600000, polling: 250 }).catch(() => {});
      else await page.waitForFunction(() => window.__b?.marks?.shown !== undefined, undefined, { timeout: 120000, polling: 50 }).catch(() => {});
      await page.waitForTimeout(EDIT_TRIPS ? 1000 : 3000);
      const e1 = await procs.cpu(), te1 = threadCpu(tabPid);
      rec.threadsEdit = threadDelta(te0, te1);
      rec.editMarks = await page.evaluate(() => window.__b.marks);
      rec.editCounts = await page.evaluate(() => { const d = window.twDemand, s = d.log.filter((x) => x.arrivedAt >= window.__e0); const sum = (a) => (a ?? []).reduce((x, y) => x + (y ?? 0), 0); const last = s.findLast((x) => !d.client.fversions?.has(x.version)); return { solves: s.length, periodsSolved: last?.periods ? sum(last.periods.map((p) => p[1])) : null, units: last?.work ? sum(last.work.map((w) => w ? w.done[0] + w.done[1] : 0)) : null }; });
      rec.cpuEdit = (e1.renderer ?? 0) - (e0.renderer ?? 0);
    }
    rec.ctxPeak = ctxPeak;
    rec.procPeak = { ...procs.w.peak };
  } catch (e) {
    rec.failed = String(e.message ?? e).slice(0, 300) + (rec.last ? ' | last ' + JSON.stringify(rec.last) : '');
  } finally {
    sampling = false; await sampler; procs.stop(); raw.close(); await chrome.close().catch(() => {});
  }
  log?.(rec);
  return rec;
}

/** a run's record as the ledger keeps it: seconds, MB, the work counts */
function summarise(rec) {
  if (rec.failed) return { failed: rec.failed, errors: rec.errors.length };
  const s = (ms) => (ms == null ? null : r1(ms / 1000));
  const ctx = rec.ctxEnd ?? {};
  const named = (re) => Object.entries(ctx).filter(([k]) => re.test(k));
  const clock = named(/clock/i)[0]?.[1];
  const dw = (rec.demandWasm ?? []).filter((x) => x != null);
  return {
    wall: { boot: s(rec.boot), firstMap: s(rec.marks?.drawn), trips: s(rec.marks?.final), playable: s(rec.marks?.playable), editShows: s(rec.editMarks?.shown), editTrips: s(rec.editMarks?.final) },
    cpu: { load: r1(rec.cpuLoad?.renderer), edit: r1(rec.cpuEdit), page: r1(rec.cpuLoad?.byContext?.page), threads: (rec.threadsLoad ?? []).slice(0, 6).map((x) => r1(x)), editThreads: (rec.threadsEdit ?? []).slice(0, 4).map((x) => r1(x)) },
    mem: { tabPeak: Math.round(rec.procPeak?.renderer ?? 0), chromePeak: Math.round(rec.procPeak?.total ?? 0), pageHeap: Math.round(ctx.page?.heap ?? 0), pageHeapPeak: Math.round(rec.ctxPeak?.page ?? 0),
      clockHeap: Math.round(clock?.heap ?? 0), clockWasm: Math.round(clock?.wasm ?? 0), demandWasm: dw.map((x) => Math.round(x)), demandWasmSum: Math.round(dw.reduce((a, b) => a + b, 0)) },
    bytes: { bootMB: r1(rec.bootBytes / MB), allMB: r1(rec.bytes?.all / MB), packsMB: r1(rec.bytes?.packs / MB), requests: rec.bytes?.requests },
    work: { ...rec.counts, edit: rec.editCounts ?? null },
    edit: rec.edit, busy: rec.busy, errors: rec.errors.length,
  };
}

/** the median of several runs' summaries, number by number */
function combine(sums) {
  const ok = sums.filter((x) => !x.failed);
  if (!ok.length) return sums[0];
  const walk = (vals) => {
    const v0 = vals[0];
    if (Array.isArray(v0)) return v0.map((_, i) => walk(vals.map((v) => v?.[i])));
    if (v0 && typeof v0 === 'object') return Object.fromEntries(Object.keys(v0).map((k) => [k, walk(vals.map((v) => v?.[k]))]));
    return typeof v0 === 'number' ? r1(median(vals)) : v0;
  };
  const out = walk(ok);
  out.runs = ok.length;
  if (ok.length > 1) out.spread = { trips: r1(Math.max(...ok.map((x) => x.wall.trips)) / Math.min(...ok.map((x) => x.wall.trips)), 2), cpuLoad: r1(Math.max(...ok.map((x) => x.cpu.load)) / Math.min(...ok.map((x) => x.cpu.load)), 2), tabPeak: r1(Math.max(...ok.map((x) => x.mem.tabPeak)) / Math.min(...ok.map((x) => x.mem.tabPeak)), 2) };
  if (ok.length < sums.length) out.failedRuns = sums.length - ok.length;
  return out;
}

// ---------------------------------------------------------------------------------------------- the ledger and its flags

/** What is judged (higher is worse) and the least change that counts. Wall-clock is never judged from the ledger. */
const JUDGED = [
  ['cpu.load', 1], ['cpu.edit', 5], ['mem.tabPeak', 30], ['mem.chromePeak', 50], ['mem.pageHeap', 10], ['mem.clockWasm', 10], ['mem.demandWasmSum', 20],
  ['bytes.bootMB', 0.5], ['bytes.allMB', 1], ['work.periodMs', 1000], ['work.setupMs', 500],
];
const CHANGED = ['work.solves', 'work.zones', 'work.markets', 'work.rounds', 'work.rows', 'work.railTripsPerDay', 'work.lines', 'work.stations'];
const get = (o, p) => p.split('.').reduce((a, k) => a?.[k], o);

function flags(row, ref) {
  const out = [];
  if (!ref) return out;
  for (const [p, floor] of JUDGED) {
    const a = get(ref, p), b = get(row, p);
    if (typeof a !== 'number' || typeof b !== 'number') continue;
    if (b > a * (1 + WORSE) && b - a > floor) out.push(p + ' ' + a + ' -> ' + b + ' (+' + Math.round((100 * (b - a)) / Math.max(a, 1e-9)) + '%)');
  }
  for (const p of CHANGED) { const a = get(ref, p), b = get(row, p); if (a != null && b != null && a !== b) out.push('changed ' + p + ' ' + a + ' -> ' + b); }
  return out;
}
const readLedger = () => existsSync(LEDGER) ? readFileSync(LEDGER, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)) : [];
const lastPublished = (rows, save, before) => rows.filter((r) => r.save === save && r.published && r !== before).at(-1) ?? null;

function line(r) {
  if (r.failed) return r.save.padEnd(9) + ' FAILED ' + r.failed;
  const w = r.wall, c = r.cpu, m = r.mem, k = r.work;
  return [r.save.padEnd(9), 'map ' + w.firstMap + 's', 'trips ' + w.trips + 's', 'edit ' + w.editShows + '/' + w.editTrips + 's', 'cpu ' + c.load + '+' + c.edit + 's',
    'tab ' + m.tabPeak + 'MB', 'page ' + m.pageHeap, 'clock ' + m.clockWasm, 'demand ' + m.demandWasmSum, r.bytes.allMB + 'MB down',
    k.zones + ' zones', 'solver ' + r1((k.periodMs ?? 0) / 1000) + 's', k.solves + ' solves', k.railTripsPerDay + ' rail/day'].join('  ');
}

// ---------------------------------------------------------------------------------------------- commands

function fetchSaves(names) {
  mkdirSync(join(HOME, 'saves'), { recursive: true });
  for (const n of names) {
    const s = SAVES[n];
    if (!s) throw new Error('no save ' + n + ' (' + Object.keys(SAVES).join(', ') + ')');
    if (existsSync(s.file)) continue;
    if (s.from && existsSync(s.from)) { cpSync(s.from, s.file); continue; }
    if (s.url) { const r = spawnSync('curl', ['-s', '-f', '-m', '120', '-o', s.file, s.url], { windowsHide: true }); if (r.status === 0) continue; }
    throw new Error('the ' + n + ' save is missing: ' + s.file);
  }
}

function buildInfo(dist) {
  const b = existsSync(join(dist, 'build.json')) ? JSON.parse(readFileSync(join(dist, 'build.json'), 'utf8')) : {};
  const p = existsSync(join(dist, 'perf-build.json')) ? JSON.parse(readFileSync(join(dist, 'perf-build.json'), 'utf8')) : {};
  return { build: b.build ?? null, commit: p.commit ?? null };
}

async function measure(dist, names, runs, name) {
  const cores = claim(WANT_CORES, name, 'perf_bench ' + names.join(','), Number(opt('min-cores', 3)));
  if (!cores) throw new Error('no ' + opt('min-cores', 3) + ' free cores within 30 minutes: not measured');
  process.env.TW_CORES = String(cores);  // lib.mjs pins Chrome to them and claims no more
  const srv = await serve(resolve(dist));
  const out = {};
  try {
    for (let i = 0; i < runs; i++) for (const n of names) {
      const t = Date.now();
      const rec = await runOnce(SAVES[n], srv.url, srv);
      mkdirSync(join(HOME, 'runs'), { recursive: true });
      writeFileSync(join(HOME, 'runs', name.replace(/^perfbench-/, '') + '-' + n + '-' + (i + 1) + '.json'), JSON.stringify(rec));
      (out[n] ??= []).push(summarise(rec));
      console.log('  ' + n + ' run ' + (i + 1) + ': ' + Math.round((Date.now() - t) / 1000) + ' s' + (rec.failed ? ' FAILED ' + rec.failed : ''));
    }
  } finally { srv.close(); release(name); }
  return { cores, out };
}

async function main() {
  if (cmd === 'build') {
    if (!pos[0]) throw new Error('build <commit | tree folder> [--label L]');
    build(pos[0], opt('label', basename(pos[0]).replace(/[^\w.-]/g, '').slice(0, 20) || 'build'));
    return;
  }
  EDIT_TRIPS = !!opt('edit-trips', false);
  const names = String(opt('saves', 'nyc,tokyo,belgrade')).split(',');
  fetchSaves(names);
  if (cmd === 'run') {
    const dist = pos[0];
    if (!dist || !existsSync(join(dist, 'index.html'))) throw new Error('run <dist folder with index.html> --label L');
    const label = opt('label', basename(resolve(dist)));
    const published = opt('published', null);
    const runs = Number(opt('runs', 1));
    const info = buildInfo(dist);
    const t = Date.now();
    const since = Math.floor(Date.now() / 1000);
    if (!withinCap()) throw new Error('the cores ledger is over its cap: not measured (Anita: no measuring on an overbooked machine)');
    const { cores, out } = await measure(dist, names, runs, 'perfbench-' + label);
    const fair = withinCap(since);
    if (!fair) console.log('THE CORES LEDGER WENT OVER ITS CAP DURING THE RUN: nothing appended to the ledger');
    const ledger = readLedger();
    const rows = names.map((n) => ({ date: new Date().toISOString().slice(0, 16), label, ...info, published: published === true ? null : published, save: n, cores, workers: WORKERS, ...combine(out[n]) }));
    for (const r of rows) {
      const ref = lastPublished(ledger, r.save, r);
      r.flags = flags(r, ref);
      r.vs = ref ? ref.label : null;
      console.log(line(r));
      if (!ref && !r.published) console.log('    NO PUBLISHED LINE FOR ' + r.save + ' IN ' + LEDGER + ': nothing to judge against');
      for (const f of r.flags) console.log('    ' + f + '  (vs ' + ref.label + ')');
      if (!opt('no-ledger', false) && fair) appendFileSync(LEDGER, JSON.stringify(r) + '\n');
    }
    console.log('bench ' + Math.round((Date.now() - t) / 1000) + ' s on ' + cores + ' cores' + (opt('no-ledger', false) || !fair ? '' : ', ' + rows.length + ' lines in notes/perf-ledger.jsonl'));
    return;
  }
  if (cmd === 'compare') {
    const [a, b] = pos;
    if (!a || !b) throw new Error('compare <base dist> <head dist>');
    const runs = Number(opt('runs', 2));
    const name = 'perfbench-compare';
    const cores = claim(WANT_CORES, name, 'perf_bench compare', Number(opt('min-cores', 3)));
    if (!cores) throw new Error('no free cores within 30 minutes: not compared');
    process.env.TW_CORES = String(cores);
    if (!withinCap()) throw new Error('the cores ledger is over its cap: not compared');
    const since = Math.floor(Date.now() / 1000);
    const sa = await serve(resolve(a)), sb = await serve(resolve(b));
    const res = { base: {}, head: {} };
    try {
      for (let i = 0; i < runs; i++) for (const n of names) {
        const order = i % 2 ? [['head', sb], ['base', sa]] : [['base', sa], ['head', sb]];
        for (const [k, srv] of order) { (res[k][n] ??= []).push(summarise(await runOnce(SAVES[n], srv.url, srv))); console.log('  ' + n + ' ' + k + ' run ' + (i + 1)); }
      }
    } finally { sa.close(); sb.close(); release(name); }
    if (!withinCap(since)) console.log('THE CORES LEDGER WENT OVER ITS CAP DURING THE COMPARE: its wall-clock is not fair');
    for (const n of names) {
      const ra = { save: n, ...combine(res.base[n]) }, rb = { save: n, ...combine(res.head[n]) };
      console.log('base ' + line(ra)); console.log('head ' + line(rb));
      for (const f of flags(rb, ra)) console.log('    ' + f);
    }
    const f = join(HOME, 'compare-' + new Date().toISOString().slice(0, 16).replace(/[:T]/g, '') + '.json');
    writeFileSync(f, JSON.stringify({ base: a, head: b, cores, res }, null, 1));
    console.log('runs in ' + f);
    return;
  }
  if (cmd === 'published') {
    const [label, version] = pos;
    if (!label || !version) throw new Error('published <label> <vX.Y.Z>');
    const rows = readLedger();
    let n = 0;
    for (const r of rows) if (r.label === label) { r.published = version; n++; }
    if (!n) throw new Error('no ledger lines labelled ' + label);
    writeFileSync(LEDGER, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
    console.log(n + ' lines of ' + label + ' marked published as ' + version);
    return;
  }
  if (cmd === 'check') {
    const rows = readLedger();
    const label = opt('label', rows.at(-1)?.label);
    for (const r of rows.filter((x) => x.label === label)) {
      const ref = lastPublished(rows, r.save, r);
      console.log(line(r));
      for (const f of flags(r, ref)) console.log('    ' + f + '  (vs ' + ref.label + ')');
    }
    return;
  }
  console.log(readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').filter((l) => l.startsWith('//')).join('\n'));
}

main().then(() => process.exit(0), (e) => { console.error(e); process.exit(1); });
