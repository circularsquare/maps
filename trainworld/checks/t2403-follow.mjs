// T-2388, T-2403, T-2404: how fast the track tool's ghost follows the pointer, right after a click
// and in steady movement. The pointer is driven by CDP on a fixed 16 ms schedule that never waits for
// the page (a hand keeps moving while the page is busy); clicks fall mid-movement. Also the tool's own
// move handler, and every grid query the snap and pick make (built, being built, carried, too wide).
//   node trainworld/checks/t2403-follow.mjs <url> <setup>   setup: new | nyc | nycdense | japan | na
// (japan: t737-japan-trains.save; na: Anita's North America day 383, t2606/na383.save, beside its
// densest stations unless NA_AT=lng,lat,zoom). Before the run, demand is left to settle after the
// camera's jump (SETTLE_S, default 600).
// Env: SEGS (clicks, default 8), SEG_MS (ms between clicks, default 1600), STEADY_MS (moving with no
// click first, default 4000), TRACE=1 (a Chrome trace over the clicks), PROFILE=1 (a CPU profile of
// the main thread over the clicks; PROFILE=steady from the start click), STOPS=1 (the line tool's stop
// clicks after), OUT, LABEL, DW (demand workers), EXTRA (more URL parameters, e.g. patch=check)
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, importSave, demandFinal, REAL_SAVE, ANGLE, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6191/';
const SETUP = process.argv[3] ?? 'nyc';
const SEGS = Number(process.env.SEGS ?? 8), SEG_MS = Number(process.env.SEG_MS ?? 1600), STEADY_MS = Number(process.env.STEADY_MS ?? 4000);
const OUT = process.env.OUT ?? join(ARTIFACTS, 't2403');
const LABEL = process.env.LABEL ?? SETUP;
const JAPAN = join(ARTIFACTS, 't737-japan-trains.save');
const NA = join(ARTIFACTS, 't2606', 'na383.save');
const BIG = { japan: JAPAN, na: NA };
const AT = { new: [-74.13, 40.585, 14], nyc: [-74.13, 40.585, 14], nycdense: [-73.955, 40.68, 14], japan: [139.95, 35.87, 14], na: (process.env.NA_AT ?? '-87.75,41.80,14').split(',').map(Number) };
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const pct = (a, p) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return Math.round(s[Math.min(s.length - 1, Math.floor(p * s.length))]); };
await mkdir(OUT, { recursive: true });

/** inclusive and self ms per function (name @ file:line) from a CPU profile (t1020's) */
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

const { ctx, close } = await launchChrome({ prefix: 'tw-follow-', size: [1400, 900], args: ANGLE });
const errors = [];
const out = { setup: SETUP, base: BASE, segs: SEGS, segMs: SEG_MS, errors };
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  // ?patch=check: a difference found in the clock worker is a console error there
  page.on('worker', (w) => { try { w.on('console', (m) => { if (m.type() === 'error') errors.push('worker: ' + m.text().slice(0, 300)); }); } catch { /* an older Playwright */ } });
  page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError/.test(m.text())) errors.push(m.text().slice(0, 200)); });
  await page.addInitScript(() => {
    try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch { /* */ }
    const m = (window.__m = { on: false, frames: [], lt: [], moves: [], downs: [], previews: [], msgs: [], posts: [], moveMs: [], grid: [], objs: [] });
    let lastNodes = null, lastNet = null, lastWorld = null;
    const tick = (t) => {
      if (m.on) {
        m.frames.push(t);
        const tw = window.tw, w = tw?.world?.value, n = tw?.renderer?.net;
        if (w && (w !== lastWorld || w.nodes !== lastNodes || n !== lastNet)) { m.objs.push([t, w !== lastWorld ? 1 : 0, w.nodes !== lastNodes ? 1 : 0, n !== lastNet ? 1 : 0]); lastWorld = w; lastNodes = w.nodes; lastNet = n; }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    m.ltAll = [];
    try { new PerformanceObserver((l) => { for (const e of l.getEntries()) { if (m.on) m.lt.push([e.startTime, e.duration]); m.ltAll.push([e.startTime, e.duration]); if (m.ltAll.length > 500) m.ltAll.shift(); } }).observe({ type: 'longtask', buffered: false }); } catch { /* */ }
    // the event's own time (when the browser made it) and when this thread got to it
    addEventListener('pointermove', (e) => { if (m.on) m.moves.push([e.timeStamp, performance.now(), e.getCoalescedEvents ? e.getCoalescedEvents().length : 1]); }, true);
    addEventListener('pointerdown', (e) => { if (m.on) m.downs.push([e.timeStamp, performance.now()]); }, true);
    m.afterFrame = (f) => requestAnimationFrame((rt) => { const ch = new MessageChannel(); ch.port1.onmessage = () => f(rt, performance.now()); ch.port2.postMessage(0); });
    const hook = setInterval(() => {
      const tw = window.tw;
      if (!tw?.world || !tw.client || !tw.renderer) return;
      clearInterval(hook);
      const tl = tw.tools;
      if (tl) { const mv = tl.move.bind(tl); tl.move = (e) => { const t = performance.now(); try { return mv(e); } finally { if (m.on) m.moveMs.push([t, performance.now() - t]); } }; }
      const c = tw.client, recv = c.receive.bind(c), post = c.post.bind(c);
      c.receive = (msg) => { const t = performance.now(); try { return recv(msg); } finally { if (m.on && msg.kind !== 'cash') m.msgs.push([t, msg.kind === 'state' && msg.state.ms ? `state[w ${msg.state.ms.map((x) => Math.round(x)).join('+')}]` : msg.kind, performance.now() - t]); } };
      c.post = (msg, tr) => { if (m.on && msg.kind !== 'epoch') m.posts.push([performance.now(), msg.kind + (msg.op ? ':' + msg.op.op : '')]); return post(msg, tr); };
      const ask = c.preview.bind(c);
      c.preview = (req, cb) => {
        const p = { move: m.moves.length - 1, asked: performance.now(), busy: c.previewBusy, answered: 0, cbMs: 0, drawn: 0, rafAt: 0, stale: false };
        if (m.on) m.previews.push(p);
        return ask(req, (a) => {
          p.answered = performance.now();
          const before = tw.renderer.previewSrc;
          cb(a);
          p.cbMs = performance.now() - p.answered;
          if (tw.renderer.previewSrc !== before) m.afterFrame((rt, t) => { p.rafAt = rt; p.drawn = t; });
        });
      };
    }, 5);
  });
  const dw = process.env.DW ?? '1';
  await page.goto(`${BASE}?debug=1&new=1&demandWorkers=${dw}${process.env.EXTRA ? '&' + process.env.EXTRA : ''}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw.tools, undefined, { timeout: 180000 });
  const t0 = Date.now();
  if (SETUP === 'nyc' || SETUP === 'nycdense' || BIG[SETUP]) {
    await importSave(page, await readFile(BIG[SETUP] ?? REAL_SAVE));
    await page.waitForFunction((n) => (window.tw.world.value?.save?.lines?.length ?? 0) >= n, SETUP === 'japan' ? 300 : 100, { timeout: 900000, polling: 500 });
    if (!BIG[SETUP]) await demandFinal(page, { timeout: 900000 }).catch(() => log('no demand final'));
    else {
      // settled: the clock worker answers a trivial query in under 50 ms and the main thread has had
      // no long task for 5 s, checked every 5 s, at most JAPAN_WAIT s
      const until = Date.now() + Number(process.env.JAPAN_WAIT ?? 900) * 1000;
      for (;;) {
        const s = await page.evaluate(async () => {
          const t = performance.now();
          await window.tw.client.query({ what: 'today' });
          const rtt = performance.now() - t;
          await new Promise((r) => setTimeout(r, 5000));
          const lt = window.__m.ltAll.filter(([st]) => st > performance.now() - 5000);
          return { rtt: Math.round(rtt), lt: lt.length, longest: Math.round(Math.max(0, ...lt.map(([, d]) => d))) };
        });
        log('settling', JSON.stringify(s));
        if ((s.rtt < 50 && s.lt === 0) || Date.now() > until) break;
      }
    }
  } else {
    await page.waitForTimeout(4000);
    await demandFinal(page, { timeout: 120000, strict: false }).catch(() => log('new: no final'));
  }
  out.readyS = Math.round((Date.now() - t0) / 1000);
  out.network = await page.evaluate(() => { const w = window.tw.world.value, n = window.tw.renderer.net; return { lines: w.save.lines.length, stations: w.save.stations.length, edges: w.edges.length, nodes: w.nodes.size, trackSegs: n?.track.count, lineSegs: n?.lines.count }; });
  log(SETUP, 'ready', JSON.stringify(out.network), `${out.readyS} s`);
  // the achievements' figures (T-2403: the same before and after its caching)
  const figures = () => page.evaluate(async () => {
    const a = await import('/src/game/achievements.ts');
    const nf = a.networkFigures.value, df = a.demandFigures.value;
    const r = (d) => d && { ...d, v: d.v === undefined ? undefined : +d.v.toPrecision(9) };
    return JSON.stringify({ cities: nf?.cities, running: nf?.running, nfFirsts: Object.fromEntries([...(nf?.firsts ?? [])].map(([k, d]) => [k, r(d)])), trips: df && +df.trips.toPrecision(9), station: r(df?.station), line: r(df?.line), dfFirsts: Object.fromEntries([...(df?.firsts ?? [])].map(([k, d]) => [k, r(d)])), covered: df?.covered, met: Object.keys(a.metNow.value.met).sort() });
  });
  out.figuresReady = await figures();
  log('figures at ready', out.figuresReady);

  if (SETUP === 'na' && !process.env.NA_AT) {
    // beside the densest cluster of stations: the one with most others within 15 km, 3 km east of it
    AT.na = await page.evaluate(() => {
      const st = window.tw.world.value.save.stations;
      let best = st[0], bn = -1;
      for (let i = 0; i < st.length; i += 3) { let c = 0; for (const o of st) if (Math.abs(o.lat - st[i].lat) < 0.135 && Math.abs(o.lng - st[i].lng) < 0.18) c++; if (c > bn) (bn = c), (best = st[i]); }
      return [best.lng + 0.036, best.lat, 14, best.name, bn];
    });
    log('na at', JSON.stringify(AT.na));
  }
  const [lng, lat, zoom] = AT[SETUP];
  await page.evaluate(async ([lng, lat, zoom]) => {
    const ui = await import('/src/game/ui.ts');
    ui.keepPlacing.value = true;
    ui.tool.value = 'track';
    window.tw.map.jumpTo({ center: [lng, lat], zoom });
  }, [lng, lat, zoom]);
  await page.waitForTimeout(5000);
  // the jump makes demand solve the view's area: wait until no solve has been logged for 10 s
  {
    const until = Date.now() + Number(process.env.SETTLE_S ?? 600) * 1000;
    let last = '';
    for (;;) {
      const s = await page.evaluate(async () => { const d = window.twDemand, v = d?.view?.(), a = await import('/src/game/achievements.ts'); return `${d?.log?.length}/${v?.version}/${(a.demandFigures.value?.trips ?? 0) > 0}/${v?.final}/${!!d?.client?.waiting}`; });
      if (s.includes('/false/')) { last = ''; if (Date.now() > until) { log('demand not settled', s); break; } await page.waitForTimeout(5000); continue; }
      if (s === last && /\/true\/false$/.test(s)) break;
      if (Date.now() > until) { log('demand not settled', s); break; }
      last = s;
      await page.waitForTimeout(10000);
    }
    log('demand settled', last);
  }
  // whether the snap's grids answer (null: every node and track point is scanned on each move)
  const probe = () => page.evaluate(async () => {
    const tw = window.tw, si = await import('/src/map/snapIndex.ts'), geo = await import('/src/map/geo.ts');
    const c = tw.map.getCenter(), [x, y] = geo.toLocal(c.lng, c.lat);
    const t = performance.now();
    const nn = si.nodesNear(tw.world.value, x, y, 200), en = si.edgesNear(tw.renderer.net, x, y, 200);
    return { nodes: nn === null ? 'scan' : nn.length, edges: en === null ? 'scan' : en.length, ms: +(performance.now() - t).toFixed(1), version: tw.world.value.version };
  });
  // the camera, and every grid query from here on: built, still being built, or too wide
  out.camera = await page.evaluate(async () => {
    const tw = window.tw, pick = await import('/src/map/pick.ts'), ci = await import('/src/map/cellIndex.ts');
    const ov = tw.overlay ?? tw.renderer?.overlay, mm = ov?.camMatrix, cv = tw.map.getCanvas();
    const at = ov ? pick.pointerLocal(ov, cv.clientWidth / 2, cv.clientHeight / 2) : null;
    const P = ci.CellLevels.prototype, orig = P.for;
    P.for = function (reach) {
      const sides = [256, 2048, 16384, 131072], L = sides.findIndex((s) => 2 * reach <= s);
      const st = L < 0 ? 'wide' : this.levels[L] === undefined ? (this.from ? 'carry?' : 'none') : this.levels[L] === 'asked' ? 'asked' : 'built';
      const t = performance.now(), g = orig.call(this, reach), m = window.__m;
      if (m.on) m.grid.push([t, this.count, L, st, g ? 1 : 0, performance.now() - t]);
      return g;
    };
    return { globe: ov?.globe, pitch: tw.map.getPitch(), bearing: tw.map.getBearing(), zoom: tw.map.getZoom(), m3: mm?.[3], m7: mm?.[7], at: at && at.map((v) => +v.toFixed(3)) };
  });
  log('camera', JSON.stringify(out.camera));
  out.gridProbe = [await probe()];
  await page.waitForTimeout(3000);
  out.gridProbe.push(await probe());
  log('grid probe', JSON.stringify(out.gridProbe));
  const box = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2, r.width, r.height]; });
  const cdp = await ctx.newCDPSession(page);
  const origin = await page.evaluate(() => performance.timeOrigin);
  // the hand: a slow loop left to right and back, ~2 px a move; position by time
  const W = Math.min(Number(process.env.HAND_W ?? 900), box[2] - 200), H = Math.min(Number(process.env.HAND_H ?? 500), box[3] - 200);
  const pos = (ms) => { const a = ms / 1000 * 0.9; return [Math.round(box[0] + (W / 2) * Math.sin(a)), Math.round(box[1] + (H / 2) * Math.sin(a * 1.7))]; };
  const sent = [], clicks = [];
  let traceDone = null;
  if (process.env.TRACE === '1') {
    const cats = ['devtools.timeline', 'disabled-by-default-devtools.timeline', 'disabled-by-default-devtools.timeline.frame', 'v8.execute', 'toplevel', 'blink.user_timing', 'disabled-by-default-v8.cpu_profiler', 'gpu', 'viz', 'latencyInfo', 'input', 'benchmark'];
    await cdp.send('Tracing.start', { traceConfig: { includedCategories: cats, recordMode: 'recordContinuously' }, transferMode: 'ReturnAsStream', streamFormat: 'json' });
    traceDone = new Promise((r) => cdp.once('Tracing.tracingComplete', r));
  }
  await page.evaluate(() => { window.__m.on = true; });
  const start = Date.now();
  // first: a start click, then STEADY_MS moving with no click, then a click every SEG_MS
  const plan = [];
  plan.push({ at: 300, click: true, kind: 'start' });
  for (let k = 0; k < SEGS; k++) plan.push({ at: 300 + STEADY_MS + k * SEG_MS, click: true, kind: 'seg' });
  const end = 300 + STEADY_MS + SEGS * SEG_MS + 1200;
  let pi = 0, profiling = false;
  await new Promise((resolve) => {
    const step = () => {
      const t = Date.now() - start;
      const [x, y] = pos(t);
      if (pi < plan.length && t >= plan[pi].at) {
        if (((process.env.PROFILE === '1' && plan[pi].kind === 'seg') || process.env.PROFILE === 'steady') && !profiling) {
          profiling = true;
          cdp.send('Profiler.enable').then(() => cdp.send('Profiler.setSamplingInterval', { interval: 200 })).then(() => cdp.send('Profiler.start')).catch((e) => log('profiler', e.message));
        }
        clicks.push({ epoch: Date.now(), kind: plan[pi].kind, x, y });
        cdp.send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', buttons: 1, clickCount: 1 }).catch(() => {});
        cdp.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', buttons: 0, clickCount: 1 }).catch(() => {});
        pi++;
      } else {
        sent.push({ epoch: Date.now(), x, y });
        cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y, buttons: 0 }).catch(() => {});
      }
      if (t >= end) return resolve();
      setTimeout(step, 16);
    };
    step();
  });
  await page.waitForTimeout(800);
  await page.evaluate(() => { window.__m.on = false; });
  out.figuresAfter = await figures();
  log('figures after the clicks', out.figuresAfter);
  if (profiling) {
    const { profile: prof } = await cdp.send('Profiler.stop');
    await writeFile(join(OUT, `${LABEL}.cpuprofile`), JSON.stringify(prof));
    out.profile = profileTop(prof, 45);
    log('profile inclusive', JSON.stringify(out.profile.inclusive));
    log('profile self', JSON.stringify(out.profile.self.slice(0, 25)));
  }
  if (traceDone) {
    await cdp.send('Tracing.end');
    const { stream } = await traceDone;
    const parts = [];
    for (;;) { const r = await cdp.send('IO.read', { handle: stream, size: 1 << 22 }); parts.push(r.base64Encoded ? Buffer.from(r.data, 'base64').toString() : r.data); if (r.eof) break; }
    await cdp.send('IO.close', { handle: stream });
    await writeFile(join(OUT, `${LABEL}.trace.json`), parts.join(''));
    log('trace written');
  }
  const raw = await page.evaluate(() => { const m = window.__m; return { frames: m.frames, lt: m.lt, moves: m.moves, downs: m.downs, previews: m.previews, msgs: m.msgs, posts: m.posts, moveMs: m.moveMs, grid: m.grid, objs: m.objs, editLog: window.tw.client.editLog.slice(-20), edges: window.tw.world.value.edges.length }; });
  {
    // the tool's own move handler, steady (before the first segment click) and after the clicks
    const firstSeg = raw.downs[1]?.[0] ?? 1e12;
    const st0 = (a) => ({ n: a.length, p50: pct(a, 0.5), p95: pct(a, 0.95), max: pct(a, 1) });
    out.moveHandler = { steady: st0(raw.moveMs.filter(([t]) => t < firstSeg).map(([, d]) => d * 10).map((x) => x / 10)), after: st0(raw.moveMs.filter(([t]) => t >= firstSeg).map(([, d]) => d)) };
    out.moveHandlerTenths = { steady: raw.moveMs.filter(([t]) => t < firstSeg).map(([, d]) => Math.round(d * 10)).sort((a, b) => a - b).filter((_, i, a) => i % Math.max(1, Math.floor(a.length / 20)) === 0) };
    const byKind = {};
    for (const [, count, L, st, ok, ms] of raw.grid) { const k = `${count}/L${L}/${st}/${ok}`; (byKind[k] ??= [0, 0])[0]++; byKind[k][1] += ms; }
    out.gridQueries = Object.fromEntries(Object.entries(byKind).map(([k, [n, ms]]) => [k, `${n} q, ${ms.toFixed(1)} ms`]));
    out.objChanges = raw.objs.map(([t, w, nd, ne]) => `${Math.round(t - (raw.downs[0]?.[0] ?? 0))}:${w}${nd}${ne}`);
    log('move handler', JSON.stringify(out.moveHandler), 'tenths', JSON.stringify(out.moveHandlerTenths));
    log('grid queries', JSON.stringify(out.gridQueries));
    log('world/nodes/net changes (ms after start click)', out.objChanges.length, out.objChanges.slice(0, 60).join(' '));
  }
  // page times to epoch
  const E = (t) => origin + t;
  const prevs = raw.previews.filter((p) => p.drawn && p.move >= 0).map((p) => ({ ...p, moveTs: E(raw.moves[p.move][0]), drawnE: E(p.drawn), askedE: E(p.asked), answeredE: E(p.answered) }));
  const downsE = raw.downs.map(([ts]) => E(ts));
  // per hand sample: the first ghost drawn from a move made at or after it
  const samples = [];
  // a move that asked no preview (the pointer on the start, or a move the tool skips) has no ghost to wait for
  const askedMoves = new Set(raw.previews.map((p) => p.move));
  const moveE = raw.moves.map(([ts]) => E(ts));
  let skipped = 0;
  for (const s of sent) {
    const j = moveE.findIndex((t) => t >= s.epoch - 1);
    if (j < 0 || !askedMoves.has(j)) { skipped++; continue; }
    let best = Infinity, via = null;
    for (const p of prevs) if (p.moveTs >= s.epoch - 1 && p.drawnE >= s.epoch && p.drawnE - s.epoch < best) { best = p.drawnE - s.epoch; via = p; }
    if (best === Infinity) continue;
    const lastDown = downsE.filter((d) => d <= s.epoch).pop();
    const since = lastDown === undefined ? null : s.epoch - lastDown;
    samples.push({ epoch: s.epoch, lat: best, since, segs: downsE.filter((d) => d <= s.epoch).length });
  }
  // buckets: the steady stretch after the start click (from 1 s on, before the first segment), and by time since a segment click
  const steady = samples.filter((s) => s.segs === 1 && s.since > 1000).map((s) => s.lat);
  const segSamples = samples.filter((s) => s.segs >= 2);
  const bucket = (lo, hi) => segSamples.filter((s) => s.since >= lo && s.since < hi).map((s) => s.lat);
  const st = (a) => ({ n: a.length, p50: pct(a, 0.5), p95: pct(a, 0.95), max: pct(a, 1) });
  out.summary = { steady: st(steady), '0-100': st(bucket(0, 100)), '100-300': st(bucket(100, 300)), '300-1000': st(bucket(300, 1000)), '1000+': st(bucket(1000, 1e9)) };
  // preview round trips: ask to answer, by whether one was in flight
  const rt = prevs.map((p) => p.answeredE - p.askedE);
  out.previewRtt = st(rt);
  out.previewCb = st(prevs.map((p) => p.cbMs));
  out.answerToDrawn = st(prevs.map((p) => p.drawnE - p.answeredE));
  out.moveDispatch = st(raw.moves.map(([ts, h]) => h - ts));
  // frames: gaps over the whole run, and in the first 1000 ms after segment clicks
  const gaps = [];
  for (let i = 1; i < raw.frames.length; i++) gaps.push([raw.frames[i - 1], raw.frames[i] - raw.frames[i - 1]]);
  const segDowns = raw.downs.slice(1).map(([ts]) => ts);
  const nearClick = (t) => segDowns.some((d) => t >= d && t < d + 1000);
  out.frames = { all: st(gaps.map((g) => g[1])), afterClick: st(gaps.filter((g) => nearClick(g[0])).map((g) => g[1])), steady: st(gaps.filter((g) => !nearClick(g[0]) && g[0] > (raw.downs[0]?.[0] ?? 0) + 1000 && g[0] < (segDowns[0] ?? 1e12)).map((g) => g[1])) };
  out.longTasks = { afterClick: raw.lt.filter(([s]) => nearClick(s)).map(([, d]) => Math.round(d)), elsewhere: raw.lt.filter(([s]) => !nearClick(s)).map(([, d]) => Math.round(d)) };
  // per click: the messages in the 1000 ms after it (ms after the press, kind, main-thread ms)
  out.perClick = segDowns.map((d) => ({
    msgs: raw.msgs.filter(([t]) => t >= d && t < d + 1000).map(([t, k, ms]) => `${Math.round(t - d)} ${k} ${ms.toFixed(1)}`),
    posts: raw.posts.filter(([t]) => t >= d && t < d + 1000 && !/preview/.test(t)).filter(([, k]) => !/^preview$/.test(k)).map(([t, k]) => `${Math.round(t - d)} ${k}`),
    worst: Math.max(0, ...segSamples.filter((s) => E(0) && s.since !== null && Math.abs((s.epoch - s.since) - E(d)) < 5).map((s) => s.lat)),
    lt: raw.lt.filter(([s]) => s >= d && s < d + 1000).map(([s, du]) => `${Math.round(s - d)}+${Math.round(du)}`),
  }));
  out.editLog = raw.editLog;
  out.edgesAdded = raw.edges - out.network.edges;
  out.samples = samples.map((s) => [Math.round(s.since ?? -1), Math.round(s.lat), s.segs]);
  out.previews = prevs.map((p) => [Math.round(p.askedE - (downsE.filter((d) => d <= p.askedE).pop() ?? p.askedE)), Math.round(p.answeredE - p.askedE), +p.cbMs.toFixed(1), Math.round(p.drawnE - p.answeredE), p.busy ? 1 : 0]);
  out.skipped = skipped;
  log(SETUP, 'edges added', out.edgesAdded, 'samples', samples.length, 'skipped', skipped, 'previews', prevs.length);
  log('ghost latency', JSON.stringify(out.summary));
  log('preview rtt', JSON.stringify(out.previewRtt), 'cb', JSON.stringify(out.previewCb), 'answer->drawn', JSON.stringify(out.answerToDrawn), 'move dispatch', JSON.stringify(out.moveDispatch));
  log('frames', JSON.stringify(out.frames));
  log('long tasks after click', JSON.stringify(out.longTasks.afterClick), 'elsewhere', JSON.stringify(out.longTasks.elsewhere));
  for (const [i, c] of out.perClick.entries()) log(`click ${i + 1}`, c.msgs.join(' | '), ' LT', c.lt.join(' '));
  if (process.env.STOPS === '1') {
    // the line tool: a new line's stops added one at a time, as its clicks do (tools.toggleStop):
    // each from the call to the edit's answer, the state taken, and the frame after
    out.stops = await page.evaluate(async () => {
      const tw = window.tw, ui = await import('/src/game/ui.ts'), c = tw.client;
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      const w = tw.world.value;
      // stations in Brooklyn, west to east
      // the stops of a running line with the most stops, so each added stop has track to it
      const src = w.save.lines.filter((l) => w.lineStats[l.id]?.status === 'running' && !w.lineStats[l.id]?.bus).sort((a, b) => b.stops.length - a.stops.length)[0];
      const ids = [...new Set((src?.stops ?? []).map(Number))];
      const at = w.save.stations.find((s) => Number(s.num) === ids[4]);
      if (at) tw.map.jumpTo({ center: [at.lng, at.lat], zoom: 13 });
      if (ids.length < 4) return { error: 'few stations ' + ids.length };
      const r = await c.edit({ op: 'addLine', stops: ids.slice(0, 2), name: 'Lag check', colour: '#aa22aa' });
      if (!r.ok || r.line === undefined) return { error: 'addLine ' + JSON.stringify(r.issues) };
      ui.lineDraft.value = { line: r.line, stops: ids.slice(0, 2) };
      ui.tool.value = 'line';
      await sleep(1500);
      const res = [];
      for (const id of ids.slice(2, 9)) {
        const t0 = performance.now();
        let stateAt = 0, stateMs = 0, editedAt = 0, frameAt = 0;
        const recv = c.receive;
        c.receive = (msg) => { const t = performance.now(); const o = recv.call(c, msg); if (msg.kind === 'state' && !stateAt) { stateAt = t; stateMs = performance.now() - t; window.__m.afterFrame((rt, ft) => { frameAt = ft; }); } if (msg.kind === 'edited' && !editedAt) editedAt = t; return o; };
        tw.tools.toggleStop(id);
        const until = performance.now() + 5000;
        while ((!frameAt || !editedAt) && performance.now() < until) await sleep(5);
        c.receive = recv;
        res.push({ stops: tw.world.value.save.lines.find((l) => Number(l.id) === r.line)?.stops.length, edited: Math.round(editedAt - t0), state: Math.round(stateAt - t0), stateMs: Math.round(stateMs), drawn: Math.round(frameAt - t0), worker: c.editLog.at(-1)?.worker });
        await sleep(800);
      }
      ui.tool.value = 'select';
      return res;
    });
    log('stops', JSON.stringify(out.stops));
  }
} catch (e) {
  console.log('FAILED', e.stack);
} finally {
  await close();
}
if (errors.length) log('errors', JSON.stringify(errors.slice(0, 8)));
await writeFile(join(OUT, `${LABEL}.json`), JSON.stringify(out, null, 1));
process.exit(0);
