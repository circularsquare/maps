// T-1020: what the player feels while laying track (notes/T-1001.md "T-1020").
//   node trainworld/checks/t1020-click-latency.mjs <page url>      (from the maps root)
// Per setup, the track tool on open ground near the network, keep placing on: a click for the start,
// then CLICKS segments, each reached by MOVES pointer moves about 16 ms apart, then a click. Measured
// from the browser's own event times: click to drawn (the press to the first frame after the world
// changed), ghost latency (each move to the first frame showing a preview asked at or after it), frame
// gaps while moving, and long tasks (over 50 ms) in the 1.5 s after each click. Prints one line per
// setup and writes OUT/<LABEL>-<setup>.json.
// SETUPS: new (a new game with five short lines), nyc (the real save, 109 lines), japan (Japan's real
// network, ../artifacts/trainworld/t737-japan.save; demand is not waited for, it takes ~19 min),
// repro (Anita's report: a new game in London, 4 pieces there, then Shanghai and London again with the
// run still open, so the first click in each is a piece from the other city; it must be refused at
// once with no edit and no long task, then a run of its own is drawn there).
// Exits 1 when a setup places nothing or the repro's far click is not refused at once.
// Env: SETUPS (default "repro,new,nyc"), CLICKS (default 8), MOVES (default 20), LABEL, OUT (default
// ../artifacts/trainworld/t1020), PROFILE=1 (a CPU profile of the main thread over the clicks, its top
// functions in the json), SHOTS=1 (nyc: screenshots after a fixed series of edits, and the same views
// after a full rebuild of the drawn network, compared pixel by pixel), DEMAND_WORKERS (default 2).
import { readFile, readdir, mkdtemp, rm, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const BASE = process.argv[2] ?? 'http://localhost:6135/';
const SETUPS = (process.env.SETUPS ?? 'repro,new,nyc').split(',');
const CLICKS = Number(process.env.CLICKS ?? 8);
const MOVES = Number(process.env.MOVES ?? 20);
const LABEL = process.env.LABEL ?? 'run';
const OUT = process.env.OUT ?? fileURLToPath(new URL('../../artifacts/trainworld/t1020/', import.meta.url));
const NYC = fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));
const JAPAN = process.env.JAPAN ?? fileURLToPath(new URL('../../artifacts/trainworld/t737-japan.save', import.meta.url));
// where the track is drawn: open ground beside each network (lng, lat, zoom)
const AT = { new: [-74.13, 40.585, 14], nyc: [-74.13, 40.585, 14], japan: [140.0, 35.9, 14],
  london: [-0.20, 51.565, 14], shanghai: [121.40, 31.25, 14], london2: [-0.12, 51.585, 14] };
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright');
}

/** inclusive ms per function (name @ file:line) from a CPU profile */
function profileTop(prof, n = 30) {
  const byId = new Map(prof.nodes.map((x) => [x.id, x]));
  const parent = new Map();
  for (const x of prof.nodes) for (const c of x.children ?? []) parent.set(c, x.id);
  const selfUs = new Map();
  for (let i = 0; i < prof.samples.length; i++) selfUs.set(prof.samples[i], (selfUs.get(prof.samples[i]) ?? 0) + (prof.timeDeltas[i + 1] ?? 0));
  const name = (x) => `${x.callFrame.functionName || '(anon)'} ${x.callFrame.url.replace(/.*\//, '').replace(/\?.*/, '')}:${x.callFrame.lineNumber + 1}`;
  const self = new Map(), incl = new Map();
  for (const [id, us] of selfUs) {
    const x = byId.get(id);
    self.set(name(x), (self.get(name(x)) ?? 0) + us);
    const seen = new Set();
    for (let c = id; c !== undefined; c = parent.get(c)) { const k = name(byId.get(c)); if (!seen.has(k)) { seen.add(k); incl.set(k, (incl.get(k) ?? 0) + us); } }
  }
  const top = (m) => [...m].filter(([k]) => !/^\((root|program|idle|garbage)/.test(k)).sort((a, b) => b[1] - a[1]).slice(0, n).map(([k, us]) => [k, Math.round(us / 1000)]);
  return { self: top(self), inclusive: top(incl) };
}

const pct = (a, p) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return Math.round(s[Math.min(s.length - 1, Math.floor(p * s.length))]); };

const { chromium } = await playwright();
await mkdir(OUT, { recursive: true });
let failed = false;

for (const setup of SETUPS) {
  const profile = await mkdtemp(join(tmpdir(), 'tw-t1020-'));
  const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
  const errors = [];
  const out = { label: LABEL, setup, base: BASE, clicks: CLICKS, moves: MOVES, errors };
  try {
    const page = await ctx.newPage();
    await page.addInitScript(() => {
      try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch { /* */ }
      const m = (window.__m = { frames: [], lt: [], moves: [], downs: [], previews: [], worlds: [] });
      const tick = (t) => { m.frames.push(t); requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
      try { new PerformanceObserver((l) => { for (const e of l.getEntries()) m.lt.push([e.startTime, e.duration]); }).observe({ type: 'longtask', buffered: true }); } catch { /* */ }
      // the browser's own event times, so a blocked main thread still counts from the press
      addEventListener('pointermove', (e) => { if (m.on) m.moves.push(e.timeStamp); }, true);
      addEventListener('pointerdown', (e) => { if (m.on) m.downs.push({ t: e.timeStamp, world: 0, frame: 0 }); }, true);
      // the frame after something changed: rAF, then a task after it (the frame's work done)
      m.afterFrame = (f) => requestAnimationFrame(() => { const ch = new MessageChannel(); ch.port1.onmessage = () => f(performance.now()); ch.port2.postMessage(0); });
      const hook = setInterval(() => {
        const tw = window.tw;
        if (!tw?.world || !tw.client || !tw.renderer) return;
        clearInterval(hook);
        let last = tw.world.peek();
        tw.world.subscribe((w) => {
          if (w === last) return;
          last = w;
          const d = m.downs.at(-1);
          if (!m.on || !d || d.world) return;
          d.world = performance.now();
          m.afterFrame((t) => { d.frame = t; });
        });
        // each clock message while measuring: when it arrived, its kind, the main thread's ms on it
        const c = tw.client, recv = c.receive.bind(c);
        m.msgs = [];
        c.receive = (msg) => { const t = performance.now(); try { return recv(msg); } finally { if (m.on && msg.kind !== 'cash') m.msgs.push([t, msg.kind, performance.now() - t]); } };
        const edit = c.edit.bind(c);
        c.edit = (op) => { if (m.on) m.msgs.push([performance.now(), 'editSent', 0]); return edit(op); };
        // each preview asked: the newest move it answers; drawn when the tool set it on the renderer
        const ask = c.preview.bind(c);
        c.preview = (req, cb) => {
          const p = { move: m.moves.length - 1, asked: performance.now(), answered: 0, drawn: 0 };
          return ask(req, (a) => {
            p.answered = performance.now();
            const before = tw.renderer.previewSrc;
            cb(a);
            if (m.on && tw.renderer.previewSrc !== before) { m.previews.push(p); m.afterFrame((t) => { p.drawn = t; }); }
          });
        };
      }, 5);
    });
    page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
    page.on('console', (msg) => { if (msg.type() === 'error' && !/favicon|openfreemap|AJAXError|^\w{1,3}$/.test(msg.text()) && !/maplibre-gl/.test(msg.location()?.url ?? '')) errors.push(msg.text().slice(0, 300)); });
    const workers = `demandWorkers=${process.env.DEMAND_WORKERS ?? 2}`;
    const solved = (timeout) => page.waitForFunction(() => { const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.(); return s && v && v.version === s.version && v.final && !d.client.waiting; }, undefined, { timeout, polling: 250 });
    await page.goto(`${BASE}?debug=1&new=1${setup === 'repro' ? '&city=lon' : ''}&${workers}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw.client.preview.length === 2, undefined, { timeout: 180000 });
    const t0 = Date.now();
    if (setup === 'new') {
      await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.99, 40.72], zoom: 12 }));
      await page.waitForTimeout(3000);
      const build = (pts, name, colour) => page.evaluate(async ([PTS, name, colour]) => {
        const tw = window.tw, c = tw.client;
        const xy = PTS.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
        const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
        await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
        for (let k = 2; k < xy.length; k++) {
          await new Promise((d) => setTimeout(d, 200));
          await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[k - 1]) }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: -1 }, pis: [], single: false });
        }
        await new Promise((d) => setTimeout(d, 300));
        for (const p of xy) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await new Promise((d) => setTimeout(d, 100)); }
        await c.edit({ op: 'constructAll' });
        await new Promise((d) => setTimeout(d, 300));
        return (await c.edit({ op: 'addLine', stops: xy.map(near), name, colour })).ok;
      }, [pts, name, colour]);
      const built = [
        await build([[-74.0110, 40.7128], [-73.9969, 40.7306], [-73.9903, 40.7506], [-73.9819, 40.7681], [-73.9680, 40.7990]], 'Eighth', '#e4002b'),
        await build([[-73.9897, 40.6926], [-73.9776, 40.6840], [-73.9615, 40.6712], [-73.9442, 40.6501]], 'Flatbush', '#0060a8'),
        await build([[-73.9500, 40.8100], [-73.9380, 40.7950], [-73.9300, 40.7800]], 'Harlem', '#00933c'),
        await build([[-73.9600, 40.7150], [-73.9400, 40.7100], [-73.9200, 40.7050], [-73.9000, 40.7000]], 'Canarsie', '#a7a9ac'),
        await build([[-73.9000, 40.7500], [-73.8800, 40.7480], [-73.8600, 40.7460]], 'Flushing', '#b933ad'),
      ];
      out.linesBuilt = built.filter(Boolean).length;
      await solved(300000).catch(() => log('new: no final in 5 min'));
    } else if (setup === 'repro') {
      await page.waitForTimeout(3000);
      await solved(300000).catch(() => log('repro: no final in 5 min'));
    } else {
      const bytes = await readFile(setup === 'japan' ? JAPAN : NYC);
      await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
      await page.waitForFunction((n) => (window.tw.world.value?.save?.lines?.length ?? 0) >= n, setup === 'japan' ? 500 : 100, { timeout: 600000, polling: 250 });
      if (setup === 'nyc') await solved(600000).catch(() => log('nyc: no final in 10 min'));
      else await page.waitForTimeout(Number(process.env.JAPAN_WAIT ?? 90) * 1000);
    }
    out.readyS = Math.round((Date.now() - t0) / 1000);
    out.network = await page.evaluate(() => { const w = window.tw.world.value, n = window.tw.renderer.net; return { lines: w.save.lines.length, stations: w.save.stations.length, edges: w.edges.length, trackSegs: n?.track.count, lineSegs: n?.lines.count }; });
    log(setup, 'ready', JSON.stringify(out.network), `${out.readyS} s`);

    if (process.env.SHOTS === '1' && setup === 'nyc') out.shots = await shots(page);

    if (setup === 'repro') {
      // T-1020 (Anita, 2026-10-23): smooth in London, one piece in Shanghai froze, London slow after
      out.phases = {};
      for (const [name, at, n] of [['london', AT.london, Math.min(4, CLICKS)], ['shanghai', AT.shanghai, 2], ['london2', AT.london2, Math.min(4, CLICKS)]]) {
        const r = (out.phases[name] = await draw(page, ctx, at, n, `${setup}-${name}`, name !== 'london'));
        if (!r.summary.clicksPlaced || (r.farClick && (r.farClick.edits || r.farClick.longTasks.some((ms) => ms > 200)))) failed = true;
      }
    } else {
      const r = await draw(page, ctx, AT[setup], CLICKS, setup);
      Object.assign(out, r);
      if (!r.summary.clicksPlaced) failed = true;
    }
  } catch (e) {
    failed = true;
    console.log(`${setup} FAILED ${e.message}`);
  } finally {
    await ctx.close();
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  }
  if (errors.length) log(setup, 'errors', errors.slice(0, 5));
  await writeFile(join(OUT, `${LABEL}-${setup}.json`), JSON.stringify(out, null, 1));
}

/** The track tool, keep placing on, at `at` (lng, lat, zoom): a start, then `clicks` segments, each
 * reached by MOVES pointer moves. Returns the timings (see the top of the file). */
async function draw(page, ctx, at, segments, setup, far = false) {
  const out = {};
  const CLICKS = segments;
  await page.evaluate(() => { const m = window.__m; m.downs = []; m.moves = []; m.previews = []; m.msgs = []; });
  const [lng, lat, zoom] = at;
  await page.evaluate(async ([lng, lat, zoom]) => {
    const ui = await import('/src/game/ui.ts');
    ui.keepPlacing.value = true;
    ui.tool.value = 'track';
    window.tw.map.jumpTo({ center: [lng, lat], zoom });
  }, [lng, lat, zoom]);
  await page.waitForTimeout(4000);
  const cdp = await ctx.newCDPSession(page);
  if (process.env.PROFILE === '1') {
    await cdp.send('Profiler.enable');
    await cdp.send('Profiler.setSamplingInterval', { interval: 250 });
    await cdp.send('Profiler.start');
  }
  const box = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; });
  // a zigzag left to right, 70 px a segment
  const pt = (k) => [box[0] - 320 + 70 * k, box[1] + (k % 2 ? 50 : -50)];
  await page.evaluate(() => { window.__m.on = true; });
  if (far) {
    // the run from the last city still open: a click here would be a piece from there (T-1020), which
    // the tool refuses at once; then Esc, and a run of its own here
    const t = await page.evaluate(() => performance.now());
    await page.mouse.move(...pt(0));
    await page.mouse.down(); await page.mouse.up();
    await page.waitForTimeout(1500);
    out.farClick = await page.evaluate((t) => ({ edits: window.__m.msgs.filter(([s, k]) => s >= t && k === 'editSent').length,
      previews: window.__m.previews.length, longTasks: window.__m.lt.filter(([s]) => s >= t).map(([, d]) => Math.round(d)) }), t);
    log(setup, 'far click', JSON.stringify(out.farClick));
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);
    await page.evaluate(() => { const m = window.__m; m.downs = []; m.moves = []; m.previews = []; m.msgs = []; });
  }
  const startLines = await page.evaluate(() => window.tw.world.value.edges.length);
  await page.mouse.move(...pt(0));
  await page.waitForTimeout(100);
  await page.mouse.down(); await page.mouse.up();
  await page.waitForTimeout(400);
  for (let k = 1; k <= CLICKS; k++) {
    const [ax, ay] = pt(k - 1), [bx, by] = pt(k);
    for (let i = 1; i <= MOVES; i++) {
      await page.mouse.move(ax + ((bx - ax) * i) / MOVES, ay + ((by - ay) * i) / MOVES);
      await page.waitForTimeout(16);
    }
    await page.waitForTimeout(250);
    const answered = await page.evaluate(() => window.__m.msgs.filter(([, k]) => k === 'edited').length);
    await page.mouse.down(); await page.mouse.up();
    // a slow edit is waited for (up to 60 s), so the next click is not queued behind it
    await page.waitForFunction((n) => window.__m.msgs.filter(([, k]) => k === 'edited').length > n, answered, { timeout: 60000, polling: 50 }).catch(() => log(setup, 'no answer to click', k));
    await page.waitForTimeout(1000);
  }
  await page.waitForTimeout(1000);
  await page.evaluate(() => { window.__m.on = false; });
  if (process.env.PROFILE === '1') {
    const { profile: prof } = await cdp.send('Profiler.stop');
    await writeFile(join(OUT, `${LABEL}-${setup}.cpuprofile`), JSON.stringify(prof));
    out.profile = profileTop(prof);
  }
  out.edgesAdded = (await page.evaluate(() => window.tw.world.value.edges.length)) - startLines;
  const raw = await page.evaluate(() => {
    const m = window.__m;
    return { frames: m.frames, lt: m.lt, moves: m.moves, downs: m.downs, previews: m.previews, msgs: m.msgs, editLog: window.tw.client.editLog };
  });
  // the first press is the start (no edit); the others each place a segment
  const clicks = raw.downs.slice(1).map((d, j) => {
    const end = raw.downs[j + 2]?.t ?? Infinity;
    const first = (k) => raw.msgs.find(([t, kind]) => kind === k && t >= d.t && t < end);
    const st = first('state'), tr = first('trips'), ed = first('edited'), es = first('editSent');
    return { world: d.world ? Math.round(d.world - d.t) : null, drawn: d.frame ? Math.round(d.frame - d.t) : null, editSent: es ? Math.round(es[0] - d.t) : null,
      stateIn: st ? Math.round(st[0] - d.t) : null, stateMs: st ? Math.round(st[2]) : null, tripsIn: tr ? Math.round(tr[0] - d.t) : null, tripsMs: tr ? Math.round(tr[2]) : null,
      editedIn: ed ? Math.round(ed[0] - d.t) : null, editedMs: ed ? Math.round(ed[2]) : null,
      lt: raw.lt.filter(([s]) => s >= d.t - 5 && s < Math.min(end, (ed?.[0] ?? d.t) + 1000)).map(([, ms]) => Math.round(ms)) };
  });
  out.editLog = raw.editLog.slice(-CLICKS);
  const drawn = clicks.map((c) => c.drawn).filter((x) => x !== null);
  // ghost: per move, the first preview drawn that answers it or a later move
  const ghost = [];
  for (let j = 0; j < raw.moves.length; j++) {
    let best = Infinity;
    for (const p of raw.previews) if (p.move >= j && p.drawn && p.drawn >= raw.moves[j]) best = Math.min(best, p.drawn - raw.moves[j]);
    // moves past the last segment's end, or with no preview yet (the start's own moves), are left out
    if (best < Infinity) ghost.push(best);
  }
  const gaps = [];
  const f = raw.frames.filter((t) => t >= (raw.downs[0]?.t ?? 0));
  for (let i = 1; i < f.length; i++) gaps.push(f[i] - f[i - 1]);
  const lts = clicks.flatMap((c) => c.lt);
  out.raw = { clicks, previews: raw.previews.length, moves: raw.moves.length };
  out.summary = {
    clicksPlaced: drawn.length, clickToDrawn: { p50: pct(drawn, 0.5), max: pct(drawn, 1) },
    ghost: { n: ghost.length, p50: pct(ghost, 0.5), p90: pct(ghost, 0.9), max: pct(ghost, 1) },
    frames: { over50: gaps.filter((g) => g > 50).length, maxGap: pct(gaps, 1) },
    longTasksPerClick: +(lts.length / Math.max(1, clicks.length)).toFixed(2), longestTask: Math.max(0, ...lts),
    longTaskMsPerClick: Math.round(lts.reduce((a, b) => a + b, 0) / Math.max(1, clicks.length)),
    // medians: when the state arrived after the press and the main thread's ms on it, the same for
    // its trips and the edit's answer, and the worker's ms by step (max beside each)
    steps: Object.fromEntries(['editSent', 'stateIn', 'stateMs', 'tripsIn', 'tripsMs', 'editedIn', 'editedMs'].map((k) => { const v = clicks.map((c) => c[k]).filter((x) => x !== null); return [k, `${pct(v, 0.5)}/${pct(v, 1)}`]; })),
    worker: Object.fromEntries([...new Set(out.editLog.flatMap((e) => Object.keys(e.worker ?? {})))].map((k) => { const v = out.editLog.map((e) => e.worker?.[k] ?? 0); return [k, `${pct(v, 0.5)}/${pct(v, 1)}`]; })),
  };
  out.network = await page.evaluate(() => { const w = window.tw.world.value, n = window.tw.renderer.net; return { lines: w.save.lines.length, edges: w.edges.length, trackSegs: n?.track.count }; });
  console.log(`${setup.padEnd(6)} ${JSON.stringify(out.network)} click to drawn p50 ${out.summary.clickToDrawn.p50} max ${out.summary.clickToDrawn.max} ms (${drawn.length}/${clicks.length} placed); ghost p50 ${out.summary.ghost.p50} p90 ${out.summary.ghost.p90} max ${out.summary.ghost.max} ms (${ghost.length} moves); frames over 50 ms ${out.summary.frames.over50}, longest gap ${out.summary.frames.maxGap}; long tasks a click ${out.summary.longTasksPerClick}, ${out.summary.longTaskMsPerClick} ms, longest ${out.summary.longestTask}`);
  console.log(`       steps ${JSON.stringify(out.summary.steps)} worker ${JSON.stringify(out.summary.worker)}`);
  if (out.profile) log('inclusive', JSON.stringify(out.profile.inclusive.slice(0, 25)));
  return out;
}

/** The real save after a fixed series of edits (a route, its end dragged, a station, an undo, a
 * station removed, a line's train count), the clock stopped at 08:30: screenshots of four views, then
 * the same views after the drawn network is rebuilt whole from the renderer's buffers; pixels that
 * differ by more than 6 in sum are counted. */
async function shots(page) {
  await page.evaluate(() => { const tw = window.tw; tw.clock.setRunning(false); tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600); });
  const edits = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client, res = [];
    const near = (lng, lat) => { const p = tw.lngLatToPack(lng, lat); let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    const a = tw.lngLatToPack(-73.93, 40.69), b = tw.lngLatToPack(-73.915, 40.70);
    const r = await c.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: -1 }, to: { kind: 'free', x: b.x, y: b.y, level: -1 }, pis: [], single: false });
    res.push(r.ok); await wait(300);
    if (r.node !== undefined) { const n = tw.world.value.nodes.get(r.node); res.push((await c.edit({ op: 'moveNode', node: r.node, x: n.x + 150, y: n.y - 80 })).ok); await wait(300); }
    res.push((await c.edit({ op: 'addStation', at: { kind: 'node', node: near(-73.93, 40.69) }, platform: 200, name: '' })).ok); await wait(300);
    res.push((await c.edit({ op: 'addStation', at: { kind: 'node', node: r.node }, platform: 200, name: '' })).ok); await wait(300);
    res.push((await c.edit({ op: 'undo' })).ok); await wait(300);
    const st = tw.world.value.save.stations[5];
    res.push((await c.edit({ op: 'removeStation', node: st.num })).ok); await wait(300);
    const line = tw.world.value.save.lines[3];
    res.push((await c.edit({ op: 'setTrainCount', line: line.num, trains: [6, 4, 2] })).ok);
    return res;
  });
  log('shot edits', JSON.stringify(edits));
  await page.waitForTimeout(4000);
  const views = [[-73.99, 40.73, 15], [-73.925, 40.695, 14], [-73.98, 40.75, 12], [-73.95, 40.7, 10]];
  const res = [];
  for (let i = 0; i < views.length; i++) {
    const [lng, lat, z] = views[i];
    await page.evaluate(([lng, lat, z]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: z }), [lng, lat, z]);
    await page.waitForTimeout(3500);
    const a = await page.screenshot({ path: join(OUT, `shot-${LABEL}-${i}.png`) });
    // the same network uploaded whole, as a fresh load would
    await page.evaluate(() => { const tw = window.tw, r = tw.renderer; r.setNetwork(r.net); if (r.layersSrc) r.setLayers(r.layersSrc); tw.overlay.invalidate(); tw.map.triggerRepaint(); });
    await page.waitForTimeout(1500);
    const b = await page.screenshot({ path: join(OUT, `shot-${LABEL}-${i}-whole.png`) });
    const px = await page.evaluate(async (urls) => {
      const data = await Promise.all(urls.map(async (u) => { const im = new Image(); im.src = u; await im.decode(); const c = document.createElement('canvas'); c.width = im.width; c.height = im.height; const g = c.getContext('2d'); g.drawImage(im, 0, 0); return g.getImageData(0, 0, c.width, c.height).data; }));
      let n = 0; const [x, y] = data;
      for (let i = 0; i < x.length; i += 4) if (Math.abs(x[i] - y[i]) + Math.abs(x[i + 1] - y[i + 1]) + Math.abs(x[i + 2] - y[i + 2]) > 6) n++;
      return n;
    }, [a, b].map((x) => 'data:image/png;base64,' + x.toString('base64')));
    res.push({ view: i, differ: px });
  }
  log('shots', JSON.stringify(res));
  return { edits, views: res };
}

process.exit(failed ? 1 : 0);
