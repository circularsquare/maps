// Shared plumbing for the standing checks (T-1114): finding Playwright, a headless Chrome with a
// throwaway profile, a page's error log, the real save, the demand-final wait, a built test line, a
// Vite of its own, and the demand worker run in Node. Import it from a check next to it:
//   import { playwright, launchChrome, collectErrors } from './lib.mjs';
import { spawn } from 'node:child_process';
import { readdir, mkdtemp, mkdir, rm, stat } from 'node:fs/promises';
import { existsSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { isAbsolute, join } from 'node:path';
import { Worker } from 'node:worker_threads';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const HERE = fileURLToPath(new URL('.', import.meta.url));
export const MAPS = fileURLToPath(new URL('../../', import.meta.url));
/** where checks write screenshots and outputs */
export const ARTIFACTS = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));
/** the real New York save, read from this tree (an agent worktree has its own) */
export const REAL_SAVE = fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));
/** smoke's figures by url (T-1135), for checks that compare against the current data */
export const SMOKE_FIGURES = fileURLToPath(new URL('../../artifacts/trainworld/smoke-figures.json', import.meta.url));
export const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
/** Chrome flags that give the page a real GL context (`ANGLE` alone for the checks that never added the blocklist flag) */
export const ANGLE = ['--use-angle=d3d11'];
export const GL = [...ANGLE, '--ignore-gpu-blocklist'];
/** network noise that is not the game's: the basemap's tiles and fonts when offline, the favicon */
export const NOISE = /favicon|openfreemap|AJAXError/;

/** PLAYWRIGHT_MODULE (a file URL of playwright/index.mjs), else `playwright`, else the npx cache */
export async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright: set PLAYWRIGHT_MODULE to a file URL of playwright/index.mjs');
}

const TEMP_PROFILES = new Set();
let EXIT_HOOKED = false;
/** removes every live temp profile, synchronously (a profile Chrome still holds is retried; T-1213) */
function removeTempProfiles() {
  for (const dir of TEMP_PROFILES) {
    try { rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }); TEMP_PROFILES.delete(dir); } catch { /* left for the next run's smoke warning */ }
  }
}

/**
 * A new throwaway --user-data-dir for headless Chrome (T-1213): a folder named `prefix` plus random
 * characters in the temp folder (a full path prefix goes where it says), removed by `remove()` and
 * otherwise when the process exits: normally, on an uncaught error, process.exit() or Ctrl-C.
 * @returns {Promise<{ dir: string, remove: () => Promise<void> }>}
 */
export async function tempProfile(prefix = 'tw-check-') {
  const dir = await mkdtemp(isAbsolute(prefix) ? prefix : join(tmpdir(), prefix));
  TEMP_PROFILES.add(dir);
  if (!EXIT_HOOKED) {
    EXIT_HOOKED = true;
    process.on('exit', removeTempProfiles);
    for (const [sig, code] of [['SIGINT', 130], ['SIGTERM', 143], ['SIGHUP', 129], ['SIGBREAK', 149]]) process.on(sig, () => process.exit(code));
  }
  return { dir, remove: async () => { await rm(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }); TEMP_PROFILES.delete(dir); } };
}

/** how many `tw-*` folders in the temp folder are over a day old: profiles an older check left behind */
export async function staleTempProfiles(days = 1) {
  const cutoff = Date.now() - days * 86400000;
  const names = await readdir(tmpdir(), { withFileTypes: true }).catch(() => []);
  let n = 0;
  for (const d of names) {
    if (!d.isDirectory() || !d.name.startsWith('tw-')) continue;
    const s = await stat(join(tmpdir(), d.name)).catch(() => null);
    if (s && s.mtimeMs < cutoff) n++;
  }
  return n;
}

/**
 * Headless Chrome with a persistent context. `prefix` names a throwaway profile (see `tempProfile`),
 * deleted by `close()` or at exit. `profile` names a folder to use instead: kept afterwards, and
 * emptied first unless `fresh` is false (a profile that holds the last run's save). `size` is
 * [width, height]; `args` default to GL; any other option goes to Playwright.
 * @returns {Promise<{ ctx: import('playwright').BrowserContext, close: () => Promise<void>, profile: string }>}
 */
export async function launchChrome({ prefix = 'tw-check-', profile, fresh = true, size = [1400, 900], args = GL, ...options } = {}) {
  const { chromium } = await playwright();
  const temp = profile ? null : await tempProfile(prefix);
  if (temp) profile = temp.dir;
  else if (fresh) await rm(profile, { recursive: true, force: true });
  try {
    const ctx = await chromium.launchPersistentContext(profile, { executablePath: CHROME, headless: true, viewport: { width: size[0], height: size[1] }, args, ...options });
    return { ctx, profile, close: async () => { try { await ctx.close(); } finally { await temp?.remove().catch(() => {}); } } };
  } catch (e) {
    await temp?.remove().catch(() => {});
    throw e;
  }
}

/** push a page's page errors and console errors onto `errors`, except console errors whose text matches `ignore` (null: none) or whose source url matches `ignoreUrl` */
export function collectErrors(page, errors, ignore = NOISE, ignoreUrl = null) {
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !ignore?.test(m.text()) && !ignoreUrl?.test(m.location()?.url ?? '')) errors.push(m.text()); });
}

/** post the bytes of a save to the page as an import; the page's performance.now() as it posts */
export const importSave = (page, bytes) => page.evaluate((data) => { const b = Uint8Array.from(data).buffer; const t = performance.now(); window.tw.client.post({ kind: 'import', bytes: b }, [b]); return t; }, Array.from(bytes));

/**
 * Wait for a demand final. Strict: the newest solve's final, none waiting. `strict: false` takes the
 * view as it is; `after` wants a version above it; `held` also waits for the clock to start;
 * `lines` wants more lines than that.
 */
export const demandFinal = (page, { timeout = 300000, polling = 500, strict = true, after = null, held = false, lines = null } = {}) =>
  page.waitForFunction(({ strict, after, held, lines }) => {
    const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
    return !!v && (!strict || (s && v.version === s.version)) && v.final && !d.client.waiting
      && (after === null || v.version > after) && (!held || !window.tw.clock.held.peek()) && (lines === null || v.lines.size > lines);
  }, { strict, after, held, lines }, { timeout, polling });

/**
 * A line of stations through `pts` ([lng, lat] each), built with the edit API: track between them,
 * a station at each, the track constructed, the line added. `wait` overrides the pauses in ms
 * (`route`, `station`, `settle`); `trains` sets the line's trains an hour by level when given.
 * Returns the addLine result: `.line` its number, `.ok` whether it was made.
 */
export const buildLine = (page, pts, { name = 'Check', colour = '#0a7d3b', trains = null, wait = {} } = {}) => page.evaluate(async ([PTS, name, colour, trains, wait]) => {
  const tw = window.tw, c = tw.client;
  const xy = PTS.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
  const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
  const pause = (ms) => new Promise((d) => setTimeout(d, ms));
  await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
  for (let k = 2; k < xy.length; k++) {
    await pause(wait.route ?? 300);
    await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[k - 1]) }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: -1 }, pis: [], single: false });
  }
  await pause(wait.settle ?? 500);
  for (const p of xy) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(wait.station ?? 200); }
  await c.edit({ op: 'constructAll' });
  await pause(wait.settle ?? 500);
  const r = await c.edit({ op: 'addLine', stops: xy.map(near), name, colour });
  if (trains && r?.line !== undefined) await c.edit({ op: 'setTrainCount', line: r.line, trains });
  return r;
}, [pts, name, colour, trains, wait]);

/** a built line of three stations `km` apart, east to west across [lng, lat], three trains set; its number or null */
export const acrossLine = async (page, [lng, lat], km = 2) => {
  const dLng = km / (111.32 * Math.cos((lat * Math.PI) / 180));
  const r = await buildLine(page, [-1, 0, 1].map((k) => [lng + k * dLng, lat]), { trains: [6, 4, 3], wait: { route: 300, station: 150, settle: 300 } });
  return r?.line ?? null;
};

/** this check's own Vite (dev-nowatch.mjs) on `port`; `child.kill()` stops it */
export async function startVite(port) {
  const child = spawn(process.execPath, [join(HERE, 'dev-nowatch.mjs'), String(port)], { stdio: ['ignore', 'pipe', 'pipe'] });
  let out = '';
  child.stdout.on('data', (d) => { out += d; });
  child.stderr.on('data', (d) => { out += d; });
  const deadline = Date.now() + 60000;
  while (!/localhost:/.test(out)) {
    if (child.exitCode !== null || Date.now() > deadline) throw new Error('Vite did not start: ' + out.slice(-500));
    await new Promise((r) => setTimeout(r, 200));
  }
  return { child, url: `http://localhost:${port}/` };
}

// ---- app code and the demand worker in Node

/**
 * App code bundled by Vite and imported, for the Node checks. `spec` is a path under app/
 * ('src/map/network.ts'), or the source of a virtual entry whose `@app` stands for app's folder:
 * "export { a } from '@app/src/x.ts';".
 */
export async function loadApp(spec) {
  const { build } = await import(pathToFileURL(join(HERE, '../app/node_modules/vite/dist/node/index.js')).href);
  const app = fileURLToPath(new URL('../app', import.meta.url)).replace(/\\/g, '/');
  const virtual = /\bexport\b/.test(spec);
  const bundle = await build({
    configFile: false, logLevel: 'silent',
    plugins: virtual ? [{ name: 'app-entry', resolveId(id) { if (id.endsWith('app-entry')) return '\0app-entry'; }, load(id) { if (id === '\0app-entry') return spec.replaceAll('@app', app); } }] : [],
    build: { write: false, minify: false, lib: { entry: virtual ? 'app-entry' : `${app}/${spec}`, formats: ['es'] } },
  });
  const code = (Array.isArray(bundle) ? bundle[0] : bundle).output.find((o) => o.type === 'chunk').code;
  return import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
}

/**
 * Bundle the demand worker into artifacts/trainworld/<name>/ with a Node adapter (worker messages,
 * a fetch that serves the WASM file); the adapter's path, for `new Worker`.
 */
export async function bundleDemandWorker(name) {
  const dir = join(ARTIFACTS, name);
  await mkdir(dir, { recursive: true });
  const compiled = join(dir, 'demand-worker.mjs'), wrapper = join(dir, 'worker-adapter.mjs');
  const wasm = fileURLToPath(new URL('../app/src/wasm/demand/sim_bg.wasm', import.meta.url));
  const { rolldown } = await import(pathToFileURL(join(HERE, '../app/node_modules/rolldown/dist/index.mjs')).href);
  const bundle = await rolldown({ input: fileURLToPath(new URL('../app/src/workers/demand.worker.ts', import.meta.url)), platform: 'browser', transform: { define: { 'import.meta.env.DEV': 'false' } } });
  await bundle.write({ file: compiled, format: 'esm', codeSplitting: false });
  await bundle.close();
  writeFileSync(wrapper, `import {parentPort} from 'node:worker_threads';
import {readFileSync} from 'node:fs';
globalThis.onmessage=null;
globalThis.location={search:'',hash:'',href:'http://localhost/'};
globalThis.postMessage=(data,options={})=>parentPort.postMessage(data,options.transfer??[]);
globalThis.fetch=async()=>new Response(readFileSync(${JSON.stringify(wasm)}),{headers:{'Content-Type':'application/wasm'}});
await import(${JSON.stringify(pathToFileURL(compiled).href)});
parentPort.on('message',data=>globalThis.onmessage({data}));
parentPort.postMessage({kind:'boot'});
`);
  return wrapper;
}

/**
 * A demand worker: `messages` it has sent, `next(predicate, from)` the first message to match (from
 * index `from`), `ask(query)` a request with a fresh id and its answer. An error in the worker ends
 * the check.
 */
export function startWorker(wrapper, { timeout = 300000 } = {}) {
  const w = new Worker(wrapper), messages = [], waiters = [];
  w.on('message', (msg) => { messages.push(msg); for (const f of [...waiters]) f(msg); });
  w.on('error', (e) => { console.error(e); process.exit(1); });
  const next = (predicate, from = 0) => new Promise((resolve, reject) => {
    const existing = messages.slice(from).find(predicate); if (existing) return resolve(existing);
    const timer = setTimeout(() => reject(new Error('worker timed out; had ' + messages.map((m) => m.kind + (m.error ? ':' + m.error : '')).join(','))), timeout);
    const f = (m) => { if (predicate(m)) { clearTimeout(timer); waiters.splice(waiters.indexOf(f), 1); resolve(m); } }; waiters.push(f);
  });
  let nextId = 1;
  const ask = (q) => { const id = nextId++; w.postMessage({ ...q, id }); return next((m) => m.kind === q.kind && m.id === id); };
  return { w, messages, next, ask };
}

const FRAME = { kind: 'equirect', lon: -73.985, lat: 40.758 }, R = 6371008.8, RAD = Math.PI / 180;
const xyOf = ([lon, lat]) => [R * (lon - FRAME.lon) * RAD * Math.cos(FRAME.lat * RAD), R * (lat - FRAME.lat) * RAD];
const LINES = {
  bos: [[-71.0552, 42.3523], [-71.0752, 42.3474], [-71.1001, 42.3480], [-71.1210, 42.3510]],
  phl: [[-75.1817, 39.9557], [-75.1636, 39.9526], [-75.1530, 39.9510], [-75.1340, 39.9690]],
};
/** a network of Boston's and Philadelphia's test lines (`ids`), each 4 stations, 3 minutes a hop, `tph` trains an hour, in New York's frame */
export function network(ids, tph = { bos: [12, 10, 6], phl: [12, 10, 6] }) {
  const stXY = [], lineN = [], stops = [], times = [], tphs = [], cars = [], hopKm = [];
  for (const id of ids) {
    const s0 = stXY.length / 2, pts = LINES[id].map(xyOf), m = pts.length;
    for (const p of pts) stXY.push(...p);
    lineN.push(m);
    for (let k = 0; k < m; k++) stops.push(s0 + k);
    const t = new Array(6 * m).fill(0);
    for (let lev = 0; lev < 3; lev++) for (let k = 0; k < m - 1; k++) { t[lev * m + k] = 180; t[(3 + lev) * m + k + 1] = 180; }
    times.push(...t);
    tphs.push(...tph[id]);
    cars.push(6);
    const km = pts.slice(1).map((p, k) => Math.hypot(p[0] - pts[k][0], p[1] - pts[k][1]) / 1000);
    hopKm.push(...km, 0, ...km.slice().reverse(), 0);
  }
  return { frame: FRAME, stXY: new Float32Array(stXY), complex: new Uint32Array(stXY.length / 2).map((_, i) => i), lineN: new Uint32Array(lineN), stops: new Uint32Array(stops), times: new Float32Array(times), tph: new Float32Array(tphs), cars: new Float32Array(cars), fare: { base: 2.9, perKm: 0.1 }, hopKm: new Float32Array(hopKm) };
}

let REGISTRY = null;
/** app/src/game/registry.json */
export const registry = () => REGISTRY ??= JSON.parse(readFileSync(new URL('../app/src/game/registry.json', import.meta.url)));
/** a city pack as the page sends it to the worker */
export const workerPack = (id) => ({ id, header: readFileSync(new URL(`../data/packs/${id}.json`, import.meta.url), 'utf8'), bin: Uint8Array.from(readFileSync(new URL(`../data/packs/${id}.bin`, import.meta.url))).buffer });
/** the zones' market packs, as the page sends them */
export const marketPacks = (ids) => ids.map((id) => ({ id, pack: readFileSync(new URL(`../data/packs/${id}.market.json`, import.meta.url), 'utf8') }));
/** the world market's rows, which the page sends with the first market pack (T-776) */
export const marketRows = () => readFileSync(new URL('../app/src/game/market-rows.json', import.meta.url), 'utf8');
