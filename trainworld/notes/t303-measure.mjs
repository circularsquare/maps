// T-303 stage 1: where demand loading time and memory go, on the served build (8800). notes/T-303.md.
//   MODE=prepare | timeline | clicks | toronto | pan    BASE=http://localhost:8800/trainworld/
//   OUT=<scratch folder> PLAYWRIGHT_MODULE=file:///.../playwright/index.mjs node notes/t303-measure.mjs
// One persistent Chrome profile (OUT/PROFILE) so the NYC save stays in its IndexedDB between
// modes: run MODE=prepare first, and again after `toronto` (its line is autosaved) or NEW=1.
// timeline: RUNS, SECS, NEW=1, BUBBLES=off|home|work. clicks: AT=5,15,30,60, PAN=lon,lat,zoom,s.
// toronto: SETTLE=1 waits for Toronto to open before building. pan: WAY=[[name,lon,lat,zoom],...].
// Outputs OUT/<MODE>-<LABEL>.json.
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const HERE = process.env.OUT ?? '.';
const BASE = process.env.BASE ?? 'http://localhost:8800/trainworld/';
const MODE = process.env.MODE ?? 'timeline';
const LABEL = process.env.LABEL ?? 'run';
const RUNS = Number(process.env.RUNS ?? 2);
const BUBBLES = process.env.BUBBLES ?? 'work'; // 'off' | 'home' | 'work'
const NEW = process.env.NEW === '1';
const profile = join(HERE, process.env.PROFILE ?? 'profile');
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

// bubbles setting before the page's modules read it; a timeline sampler from navigation
await ctx.addInitScript(([bubbles]) => {
  try {
    const s = JSON.parse(localStorage.getItem('trainworld-ui') || '{}');
    s.v = 2;
    s.display = { ...(s.display || {}), commuters: bubbles !== 'off', commuterEnd: bubbles === 'home' ? 'home' : 'work', dayShown: 'weekday' };
    localStorage.setItem('trainworld-ui', JSON.stringify(s));
  } catch { /* */ }
  window.__ev = [];
  const ev = (k, v) => window.__ev.push([Math.round(performance.now()), k, v]);
  let lastView = '', lastCity = '', lastStep = '', world = false, clock = false, nLoads = 0;
  setInterval(() => {
    if (!world && window.tw?.world?.value) { world = true; ev('world'); }
    if (!clock && world && window.tw?.clock?.held && window.tw.clock.held.peek() === false) { clock = true; ev('clock'); }
    const d = window.twDemand;
    if (!d) return;
    const v = d.view?.();
    const key = v ? `${v.version}:${v.round}:${v.final}` : '';
    if (key !== lastView) { lastView = key; ev('view', key); }
    const c = d.activeCity?.peek?.();
    if (c && c !== lastCity) { lastCity = c; ev('focus', c); }
    const loads = d.start?.loads ?? [];
    for (; nLoads < loads.length; nLoads++) ev('request', loads[nLoads].item);
    const bar = document.querySelector('.top-bar')?.textContent?.match(/(Updating demand[^A-Z]*|Estimate[^A-Z]*|Loading[^A-Z]*)/)?.[0] ?? '';
    if (bar !== lastStep) { lastStep = bar; ev('bar', bar); }
  }, 20);
}, [BUBBLES]);

const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
const requests = [];
page.on('requestfinished', async (r) => {
  const u = r.url();
  if (!/packs\/|\.wasm/.test(u)) return;
  const t = r.timing();
  let size = 0;
  try { size = (await r.sizes()).responseBodySize; } catch { /* */ }
  requests.push({ url: u.replace(/.*\/trainworld\//, ''), range: r.headers().range ?? null, start: Math.round(t.startTime), end: Math.round(t.responseEnd), size, worker: !!r.frame ? false : true });
});

const dump = () => page.evaluate(() => {
  const d = window.twDemand, c = d.client;
  return {
    hc: navigator.hardwareConcurrency, computeN: c.computeN, nav: Math.round(performance.timeOrigin),
    ev: window.__ev,
    loads: d.start.loads.map((l) => ({ item: l.item, req: Math.round(l.requestedAt), dl: l.downloadedAt && Math.round(l.downloadedAt), open: l.openedAt && Math.round(l.openedAt) })),
    workers: d.start.workers.map((w) => ({ wasmMs: Math.round(w.wasmMs), openMs: Math.round(w.openMs), warmMs: Math.round(w.warmMs), memMB: +(w.memBytes / 1e6).toFixed(1) })),
    solves: d.log.map((s) => ({ v: s.version, cities: s.cities.join('+'), arrived: Math.round(s.arrivedAt), sent: Math.round(s.sentAt), first: s.firstAt && Math.round(s.firstAt), refined: s.refinedAt && Math.round(s.refinedAt), published: s.published, est: s.showEstimates, periodMs: s.periodMs.map((p) => p.map((x) => Math.round(x))), setupMs: s.setupMs.map((x) => Math.round(x)), memMB: s.memBytes.map((b) => +(b / 1e6).toFixed(1)), lines: s.svc.lines.length, stations: s.svc.stations.length })),
    opened: [...c.opened], focus: d.activeCity.peek(), zoom: window.tw.map.getZoom(), center: window.tw.map.getCenter(),
    qlog: c.qlog.slice(-60).map((q) => ({ kind: q.kind, sent: Math.round(q.sent - performance.timeOrigin), replies: q.replies.map((p) => ({ w: p.w, wait: Math.round(p.start - q.sent), ms: Math.round(p.ms), at: Math.round(p.at - performance.timeOrigin) })) })),
  };
});

/** biggest bubbles on screen by the hover tag: [pageX, pageY, legs] */
const findBubbles = () => page.evaluate(() => {
  const hits = [];
  if (!window.twViews) return hits;
  const r = window.tw.map.getContainer().getBoundingClientRect();
  for (let y = 120; y < r.height - 120; y += 20) for (let x = 120; x < r.width - 120; x += 20) {
    const h = window.twViews.hover(x, y);
    if (h && /homes|destinations/i.test(h[0]) && !/longer|inter/i.test(h[0])) {
      const m = h[0].match(/^([\d.,]+)\s*([kM]?)/);
      const n = m ? parseFloat(m[1].replace(/,/g, '')) * (m[2] === 'k' ? 1e3 : m[2] === 'M' ? 1e6 : 1) : 0;
      hits.push([x + r.left, y + r.top, n]);
    }
  }
  hits.sort((a, b) => b[2] - a[2]);
  return hits;
});

/** a click on a bubble: ms from the mouse down to the far end handed over and drawn */
const click = async (x, y) => {
  await page.evaluate(() => { window.tw.selection.value = null; });
  await page.waitForTimeout(200);
  await page.evaluate(() => {
    window.__t0 = 0;
    window.addEventListener('mousedown', () => { window.__t0 = performance.now(); }, { capture: true, once: true });
  });
  await page.mouse.click(x, y);
  return page.evaluate(async () => {
    const st = window.twViews.stats.commuters, t0 = window.__t0;
    const deadline = performance.now() + 90000;
    while (!(st.farAt > t0) && performance.now() < deadline) await new Promise((d) => setTimeout(d, 5));
    if (!(st.farAt > t0)) return { at: Math.round(t0), timeout: true, sel: window.tw.selection.value?.kind ?? null };
    await new Promise((d) => requestAnimationFrame(() => requestAnimationFrame(d)));
    const v = window.twDemand.view();
    return { at: Math.round(t0), ms: Math.round(performance.now() - t0), sel: window.tw.selection.value?.kind, view: `${v.version}:${v.round}:${v.final}` };
  });
};

const waitSolved = (timeout = 180000) => page.waitForFunction(() => {
  const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
  return s && v && v.version === s.version && v.final && !d.client.waiting;
}, undefined, { timeout, polling: 100 });

const out = { mode: MODE, label: LABEL, base: BASE, bubbles: BUBBLES, runs: [], errors, requests };
try {
  if (MODE === 'prepare') {
    await page.goto(`${BASE}?debug=1&new=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    const bytes = await readFile('C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save');
    const before = await page.evaluate(() => window.tw.lastSaved.peek());
    await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 240000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
    log('save stored');
  } else if (MODE === 'timeline') {
    const SECS = Number(process.env.SECS ?? 75);
    for (let i = 0; i < RUNS; i++) {
      requests.length = 0;
      await page.goto(`${BASE}?debug=1${NEW ? '&new=1' : ''}&v=${Date.now()}`);
      await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
      await page.waitForTimeout(SECS * 1000);
      const r = await dump();
      r.requests = requests.slice();
      out.runs.push(r);
      log('run', i, r.solves.map((s) => `v${s.v}[${s.cities}] sent ${s.sent} first ${s.first} final ${s.refined}`).join(' | '));
    }
  } else if (MODE === 'clicks') {
    const AT = (process.env.AT ?? '5,15,30,60').split(',').map(Number);
    for (let i = 0; i < RUNS; i++) for (const at of AT) {
      await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
      await page.waitForFunction(() => !!window.twDemand && !!window.twViews, undefined, { timeout: 60000 });
      // PAN="lon,lat,zoom,seconds": the view moves there that many seconds after navigation
      if (process.env.PAN) {
        const [lon, lat, zoom, pat] = process.env.PAN.split(',').map(Number);
        await page.waitForFunction((t) => performance.now() >= t * 1000 && window.tw?.map, pat, { timeout: 120000, polling: 50 });
        await page.evaluate(([lon, lat, zoom]) => window.tw.map.jumpTo({ center: [lon, lat], zoom }), [lon, lat, zoom]);
      }
      await page.waitForFunction((at) => performance.now() >= at * 1000, at, { timeout: 120000, polling: 50 });
      // the first bubble on screen at or after `at` s
      let hits = [], firstBubbles = null;
      for (let k = 0; k < 400 && hits.length < 3; k++) {
        hits = await findBubbles();
        if (hits.length < 3) await page.waitForTimeout(100);
      }
      firstBubbles = await page.evaluate(() => Math.round(performance.now()));
      const r = hits.length ? await click(hits[0][0], hits[0][1]) : { none: true };
      r.target = at; r.bubblesAt = firstBubbles; r.legs = hits[0]?.[2];
      const d = await dump();
      r.solves = d.solves.map((s) => ({ v: s.v, cities: s.cities, sent: s.sent, first: s.first, refined: s.refined }));
      r.qlog = d.qlog.filter((q) => q.sent >= r.at - 5);
      r.ev = d.ev;
      out.runs.push(r);
      log('click at', at, 's:', r.ms ?? (r.timeout ? 'timeout' : 'none'), 'ms; clicked at', r.at, 'view', r.view);
    }
  } else if (MODE === 'toronto') {
    // the NYC save solved; then to Toronto and a new line there (Yonge St, Union to Eglinton).
    // SETTLE=1: wait for Toronto to open before building; else build straight away.
    await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
    await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
    await waitSolved();
    await page.waitForTimeout(2000);
    const n0 = (await dump()).solves.length;
    const tJump = await page.evaluate(() => { window.tw.map.jumpTo({ center: [-79.39, 43.67], zoom: 12 }); return Math.round(performance.now()); });
    if (process.env.SETTLE === '1') {
      await page.waitForTimeout(1500);
      await waitSolved(240000);
      await page.waitForTimeout(2000);
    } else await page.waitForTimeout(800);
    const PTS = [[-79.3806, 43.6453], [-79.3832, 43.6560], [-79.3866, 43.6710], [-79.3904, 43.6870], [-79.3985, 43.7055]];
    const built = await page.evaluate(async (PTS) => {
      const tw = window.tw, c = tw.client;
      const xy = PTS.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
      const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return [best, bd]; };
      const issues = [];
      let r = await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
      issues.push(r.issues);
      for (let k = 2; k < xy.length; k++) {
        await new Promise((d) => setTimeout(d, 300));
        const [from] = near(xy[k - 1]);
        r = await c.edit({ op: 'route', from: { kind: 'node', node: from }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: -1 }, pis: [], single: false });
        issues.push(r.issues);
      }
      await new Promise((d) => setTimeout(d, 500));
      const nodes = xy.map((p) => near(p));
      for (const [id] of nodes) { r = await c.edit({ op: 'addStation', at: { kind: 'node', node: id }, platform: 200, name: '' }); issues.push(r.issues); await new Promise((d) => setTimeout(d, 200)); }
      await new Promise((d) => setTimeout(d, 500));
      r = await c.edit({ op: 'constructAll' }); issues.push(r.issues);
      await new Promise((d) => setTimeout(d, 500));
      const stops = xy.map((p) => near(p)[0]);
      const t0 = performance.now();
      r = await c.edit({ op: 'addLine', stops, name: 'Yonge test', colour: '#0a7d3b' });
      issues.push(r.issues);
      window.__line = r.line;
      const r2 = await c.edit({ op: 'setTrainCount', line: r.line, trains: [10, 8, 6] });
      issues.push(r2.issues);
      window.__lineInfo = JSON.stringify(window.tw.world.value.save.lines.find((l) => String(l.num ?? l.id) === String(r.line)) ?? null).slice(0, 400);
      return { t0: Math.round(t0), line: r.line, ok: r.ok, issues, nodes, info: window.__lineInfo };
    }, PTS);
    log('built', JSON.stringify(built));
    // when the new line first has riders in the shown result, and when that result is final
    const tl = await page.evaluate(async (t0) => {
      const out = [];
      const id = String(window.__line);
      let last = '';
      const end = performance.now() + 150000;
      while (performance.now() < end) {
        const v = window.twDemand.view(), l = v?.lines?.get(id);
        const tor = window.twDemand.cities().get('tor');
        const key = `${v?.version}:${v?.round}:${v?.final}:${l ? Math.round(l.ridersPerDay) : 'none'}:${tor ? 'tor' : '-'}`;
        if (key !== last) { last = key; out.push([Math.round(performance.now() - t0), key]); }
        if (l && v.final && l.ridersPerDay > 0 && window.twDemand.log.at(-1).version === v.version) break;
        await new Promise((d) => setTimeout(d, 50));
      }
      return out;
    }, built.t0);
    const d = await dump();
    out.toronto = { settle: process.env.SETTLE === '1', tJump, built, timeline: tl, loads: d.loads.map((l) => ({ ...l, req: l.req - built.t0, dl: l.dl && l.dl - built.t0, open: l.open && l.open - built.t0 })), solves: d.solves.slice(n0).map((s) => ({ ...s, arrived: s.arrived - built.t0, sent: s.sent - built.t0, first: s.first && s.first - built.t0, refined: s.refined && s.refined - built.t0, periodMs: s.periodMs, setupMs: s.setupMs })) };
    log('timeline', JSON.stringify(tl));
    log('solves', out.toronto.solves.map((s) => `v${s.v}[${s.cities}] arrived ${s.arrived} sent ${s.sent} first ${s.first} final ${s.refined}`).join(' | '));
  } else if (MODE === 'pan') {
    // the NYC save, solved, then a pan: Trenton, Philadelphia, Worcester, Toronto, then round the Northeast
    await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
    await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
    await waitSolved();
    await page.waitForTimeout(3000);
    const memNow = () => page.evaluate(() => {
      const d = window.twDemand, s = d.log.at(-1);
      return { solve: s.memBytes.map((b) => +(b / 1e6).toFixed(1)), opened: d.start.workers.map((w) => +(w.memBytes / 1e6).toFixed(1)) };
    });
    const steps = [];
    const base = await dump();
    steps.push({ name: 'start (Midtown)', mem: await memNow(), opened: base.opened, solves: base.solves.length });
    const WAY = JSON.parse(process.env.WAY ?? 'null') ?? [
      ['Trenton', -74.76, 40.22, 10.5], ['Philadelphia', -75.16, 39.95, 11], ['Worcester', -71.80, 42.26, 10.5], ['Toronto', -79.38, 43.65, 11],
      ['New Haven', -72.93, 41.31, 10.5], ['Hartford', -72.68, 41.76, 10.5], ['Springfield', -72.59, 42.10, 10.5], ['Providence', -71.41, 41.82, 10.5],
      ['Boston', -71.06, 42.36, 11], ['Baltimore', -76.61, 39.29, 11], ['Washington', -77.04, 38.90, 11], ['New London', -72.10, 41.36, 10.5], ['Litchfield', -73.19, 41.75, 10.5],
    ];
    for (const [name, lon, lat, zoom] of WAY) {
      const before = await dump();
      const nLoads = before.loads.length, nSolves = before.solves.length;
      const t0 = await page.evaluate(([lon, lat, zoom]) => { const t = performance.now(); window.tw.map.jumpTo({ center: [lon, lat], zoom }); return Math.round(t); }, [lon, lat, zoom]);
      // settle: nothing downloading or opening, and the newest solve final (or nothing to solve)
      await page.waitForTimeout(1500);
      await page.waitForFunction((n) => {
        const d = window.twDemand, c = d.client;
        const loads = d.start.loads;
        const pending = loads.some((l) => l.openedAt === undefined);
        const s = d.log.at(-1), v = d.view();
        return !pending && !c.waiting && s && v && v.version === s.version && v.final;
      }, nLoads, { timeout: 240000, polling: 100 }).catch(() => log('settle timeout', name));
      const after = await dump();
      // a click right away on the focused zone's biggest bubble, if any
      const hits = await findBubbles();
      const st = { name, t0, focus: after.focus, newLoads: after.loads.slice(nLoads).map((l) => ({ ...l, req: l.req - t0, dl: l.dl && l.dl - t0, open: l.open && l.open - t0 })), newSolves: after.solves.slice(nSolves).map((s) => ({ ...s, sent: s.sent - t0, first: s.first && s.first - t0, refined: s.refined && s.refined - t0 })), mem: await memNow(), opened: after.opened, bubbles: hits.length };
      steps.push(st);
      log(name, 'focus', st.focus, 'loads', st.newLoads.map((l) => `${l.item} dl ${l.dl} open ${l.open}`).join(', '), '| solves', st.newSolves.map((s) => `v${s.v} sent ${s.sent} final ${s.refined} [${s.cities}]`).join(', '), '| mem', st.mem.solve.join('/'));
    }
    // back to Midtown, a re-solve of the same network with every pack open, and clicks during it
    await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.975, 40.745], zoom: 12.3 }));
    await page.waitForTimeout(1500);
    await waitSolved(240000).catch(() => {});
    await page.waitForTimeout(2000);
    const n0 = (await dump()).solves.length;
    await page.evaluate(() => window.twDemand.resubmit());
    await waitSolved(240000);
    const re = (await dump()).solves.slice(n0);
    steps.push({ name: 'resubmit at Midtown, all opened', solves: re, mem: await memNow() });
    log('resubmit', re.map((s) => `sent ${s.sent} first ${s.first - s.sent} final ${s.refined - s.sent} [${s.cities}]`).join(' | '));
    const hits = await findBubbles();
    if (hits.length) {
      const idle = await click(hits[0][0], hits[0][1]);
      await page.evaluate(() => window.twDemand.resubmit());
      await page.waitForTimeout(1500);
      const during = await click(hits[0][0], hits[0][1]);
      steps.push({ name: 'clicks with all open', idle, during });
      log('click idle', idle.ms, 'during re-solve', during.ms);
      await waitSolved(240000).catch(() => {});
    }
    out.steps = steps;
    out.final = await dump();
  }
} finally {
  await ctx.close();
}
await writeFile(join(HERE, `${MODE}-${LABEL}.json`), JSON.stringify(out, null, 1));
log('written', `${MODE}-${LABEL}.json`, 'errors', errors.slice(0, 5));
