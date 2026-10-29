// T-1805: the placement lag, by case, in headless Chrome with the track tool (keep placing on).
//   node trainworld/checks/t1805-placement.mjs [page url] [save | fresh]      (from the maps root)
// For each of EDGES constructed edges on screen: (a) a run of three segments starting on the edge
// mid-way (the first splits the built edge), then (b) a run of three branching off the middle of the
// run's second segment (the first splits a blueprint edge); each segment's previews (pointer moves a
// frame apart, ask-to-answer ms) and its click to the next segment's first ghost. Segments 2 and 3
// of each run are (c). Also whether a click started a demand solve (a network snapshot sent) and the
// clock worker's own ms on the edit. Env: EDGES (default 6), MOVES (default 12), OUT (json).
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, importSave, acrossLine, ANGLE, REAL_SAVE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6175/';
const SAVE = process.argv[3] ?? REAL_SAVE;
const EDGES = Number(process.env.EDGES ?? 6), MOVES = Number(process.env.MOVES ?? 12);

const { ctx, close } = await launchChrome({ prefix: 'tw-t1805-', size: [1200, 800], args: ANGLE });
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  page.on('console', (m) => m.text().startsWith('t1805') && console.log(m.text()));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools && window.twDemand?.client, undefined, { timeout: 180000 });
  let centre = [-73.985, 40.752];
  if (SAVE === 'fresh') {
    // a small hand-built network: two crossing lines of three stations in Midtown
    await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 13 }), centre);
    await acrossLine(page, centre, 1.5);
    await acrossLine(page, [centre[0], centre[1] + 0.012], 1.5);
  } else {
    await importSave(page, await readFile(SAVE));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) >= 20, undefined, { timeout: 300000, polling: 250 });
  }
  await page.evaluate((c) => window.tw.map.jumpTo({ center: c, zoom: 15 }), centre);
  // the load's (or the build's) demand solve over, so it does not count against a click
  await page.waitForFunction(() => { const d = window.twDemand, v = d?.view?.(); return !!v?.final && !d.client.waiting; }, undefined, { timeout: 300000, polling: 500 }).catch(() => console.log('t1805 no demand final before the runs'));
  if (process.env.TIMELINE) await page.evaluate(() => { window.__TL = true; });
  // PROFILE=1: the main thread's self time by function, from each click to its next ghost
  if (process.env.PROFILE) {
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('Profiler.enable');
    await cdp.send('Profiler.setSamplingInterval', { interval: 200 });
    await page.exposeFunction('__prof', async (what, label) => {
      if (what === 'start') return void await cdp.send('Profiler.start');
      const { profile } = await cdp.send('Profiler.stop');
      const self = new Map(), dt = profile.timeDeltas, byId = new Map(profile.nodes.map((n) => [n.id, n]));
      profile.samples.forEach((id, k) => {
        const f = byId.get(id).callFrame, key = `${f.functionName || '(anon)'} ${f.url.split('/').pop()}:${f.lineNumber + 1}`;
        self.set(key, (self.get(key) ?? 0) + (dt[k] ?? 0) / 1000);
      });
      const top = [...self].filter(([k]) => !k.startsWith('(idle)') && !k.startsWith('(program)')).sort((x, y) => y[1] - x[1]).slice(0, 14);
      console.log(`profile ${label}: ${top.map(([k, v]) => `${v.toFixed(1)} ${k}`).join(' ; ')}`);
    });
  }
  await page.waitForTimeout(3000);
  const r = await page.evaluate(async ([EDGES, MOVES]) => {
    const tw = window.tw, map = tw.map, tools = tw.tools, c = tw.client;
    const ui = await import('/src/game/ui.ts');
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
    const ev = (x, y) => ({ point: { x, y }, lngLat: map.unproject([x, y]), originalEvent: { detail: 1, shiftKey: false } });
    const until = async (pred, ms = 15000) => { const t = performance.now(); while (performance.now() - t < ms) { if (pred()) return true; await sleep(2); } return false; };
    const asks = [];
    let editMs = null, editedAt = 0, snapshots = 0, states = 0, stateMs = 0;
    const recv = c.receive.bind(c), post = c.post.bind(c);
    // every message both ways after a click, for TIMELINE=1
    let tl = null;
    c.post = (msg, tr) => { tl?.push(`${Math.round(performance.now() - tl.at)} > ${msg.kind}${msg.op ? ':' + msg.op.op : ''}`); return post(msg, tr); };
    c.receive = (msg) => {
      tl?.push(`${Math.round(performance.now() - tl.at)} < ${msg.kind}`);
      if (msg.kind === 'edited') { editedAt = performance.now(); if (msg.ms) editMs = msg.ms; }
      if (msg.kind === 'snapshot') snapshots++;
      if (msg.kind === 'state') { states++; const t = performance.now(); const out = recv(msg); stateMs += performance.now() - t; return out; }
      return recv(msg);
    };
    const ask = c.preview.bind(c);
    c.preview = (req, cb) => { const p = { t: performance.now(), ms: -1 }; asks.push(p); return ask(req, (a) => { p.ms = performance.now() - p.t; cb(a); }); };
    ui.keepPlacing.value = true;
    ui.tool.value = 'track';
    const n = tw.renderer.net, W = map.getCanvas().clientWidth, H = map.getCanvas().clientHeight;
    const built = new Set(tw.world.value.edges.filter((e) => e.built).map((e) => e.id));
    const picks = [];
    for (const [edge, pts] of n.edgePts) {
      if (!built.has(edge)) continue;
      for (let k = 0; k + 5 < pts.length; k += 3) {
        const [xa, ya] = tw.overlay.toScreen(pts[k], pts[k + 1]), [xb, yb] = tw.overlay.toScreen(pts[k + 3], pts[k + 4]);
        const x0 = (xa + xb) / 2, y0 = (ya + yb) / 2;
        if (x0 < 300 || x0 > W - 300 || y0 < 200 || y0 > H - 200 || Math.hypot(xb - xa, yb - ya) < 8) continue;
        picks.push({ edge, x: x0, y: y0, h: Math.atan2(yb - ya, xb - xa) });
        break;
      }
    }
    // one segment: moves from (x, y) out by `len` px at angle a, then a click; its figures
    const segment = async (x, y, a0, len) => {
      let px = x, py = y, ms = [], ok = false, a = a0;
      // the first angle whose ghost is accepted (dense track refuses some)
      // and the first level (a branch off a subway line wants to stay under ground)
      const lv0 = ui.buildLevel.peek();
      const tries = [lv0, -1, -2, 0, 1].filter((v, k, xs) => xs.indexOf(v) === k).flatMap((lv) => [0, 30, -30, 60, -60].map((d) => [lv, d]));
      for (const [lv, d] of tries) {
        ui.buildLevel.value = lv;
        a = a0 + d * Math.PI / 180;
        asks.length = 0;
        for (let m = 1; m <= MOVES; m++) {
          px = x + Math.cos(a) * len * m / MOVES; py = y + Math.sin(a) * len * m / MOVES;
          tools.move(ev(px, py));
          await frame();
        }
        await until(() => tools.wantKey && tools.wantKey === tools.previewKey, 5000);
        ms = [...ms, ...asks.filter((q) => q.ms >= 0).map((q) => q.ms)];
        ok = !!tools.lastPreview?.ok;
        if (ok) break;
      }
      if (!ok) return { ok, placed: false, issues: tools.lastPreview?.issues };
      const chain0 = tools.chain.length;
      editMs = null; states = 0; stateMs = 0;
      const snap0 = snapshots, solves0 = window.twDemand.log.length;
      const clickAt = performance.now();
      tl = Object.assign([], { at: clickAt });
      if (window.__prof) await window.__prof('start');
      tools.click(ev(px, py));
      const placed = await until(() => !tools.placing && tools.chain.length > chain0);
      const placedMs = performance.now() - clickAt;
      // the next segment's first ghost, from the new end
      tools.move(ev(px + Math.cos(a) * 60, py + Math.sin(a) * 60));
      const next = placed && await until(() => tools.wantKey && tools.wantKey === tools.previewKey);
      const nextMs = performance.now() - clickAt, q = asks.find((q) => q.t > clickAt), timeline = tl.join(' | ');
      tl = null;
      if (window.__prof) await window.__prof('stop', `${chain0 ? 'c' : 'first'} ${Math.round(nextMs)} ms`);
      await sleep(1500); // a snapshot is sent after a debounce
      return { ok, placed, maxAskMs: Math.round(Math.max(0, ...ms)), meanAskMs: Math.round(ms.reduce((s, v) => s + v, 0) / Math.max(1, ms.length)), firstAskMs: Math.round(ms[0] ?? -1), placedMs: Math.round(placedMs), editedMs: Math.round(editedAt - clickAt), nextAskAt: q ? Math.round(q.t - clickAt) : null, nextAskMs: q ? Math.round(q.ms) : null, nextGhostMs: next ? Math.round(nextMs) : null, states, mainStateMs: Math.round(stateMs), snapshot: snapshots > snap0, solves: window.twDemand.log.length - solves0, worker: editMs && Object.fromEntries(Object.entries(editMs).map(([k, v]) => [k, Math.round(v)])), px, py, a, ...(window.__TL ? { timeline } : {}) };
    };
    const out = [];
    for (let i = 0; i < picks.length && out.length < EDGES; i++) {
      const p = picks[i];
      tools.reset();
      await frame();
      tools.move(ev(p.x, p.y));
      if (tools.hover?.kind !== 'edge') continue;
      let undos = 0;
      const before = new Set(tw.renderer.net.edgePts.keys());
      // (a): off the built edge
      tools.click(ev(p.x, p.y));
      const a = p.h + (i % 2 ? 1 : -1) * 35 * Math.PI / 180;
      const runA = [];
      let x = p.x, y = p.y, ang = a;
      for (let k = 0; k < 3; k++) {
        const s = await segment(x, y, ang, 120);
        runA.push(s);
        if (!s.placed) break;
        undos++;
        x = s.px; y = s.py; ang = s.a;
      }
      tools.reset();
      await frame();
      // (b): off the middle of the run's second segment (blueprint)
      const runB = [];
      if (runA.length >= 2 && runA[1].placed) {
        // a blueprint edge the run made, at its middle sample, on screen
        await sleep(200);
        let bx = 0, by = 0;
        for (const [edge, pts] of tw.renderer.net.edgePts) {
          if (before.has(edge) || tw.world.value.edges.find((e) => e.id === edge)?.built) continue;
          const k = Math.floor(pts.length / 6) * 3, [sx, sy] = tw.overlay.toScreen(pts[k], pts[k + 1]);
          if (sx < 150 || sx > W - 150 || sy < 150 || sy > H - 150) continue;
          bx = sx; by = sy;
          break;
        }
        tools.move(ev(bx, by));
        await frame();
        const hv = tools.hover;
        const onBlue = hv?.kind === 'edge' && !tw.world.value.edges.find((e) => e.id === hv.edge)?.built;
        if (onBlue) {
          tools.click(ev(bx, by));
          const b = a + (i % 2 ? -1 : 1) * 60 * Math.PI / 180;
          let x2 = bx, y2 = by; ang = b;
          for (let k = 0; k < 3; k++) {
            const s = await segment(x2, y2, ang, 120);
            runB.push(s);
            if (!s.placed) break;
            undos++;
            x2 = s.px; y2 = s.py; ang = s.a;
          }
        } else runB.push({ skipped: hv?.kind ?? 'none' });
      }
      tools.reset();
      for (let k = 0; k < undos; k++) await c.edit({ op: 'undo' }, true).catch(() => {});
      // the undos' demand solve over before the next run, so it does not count against its click
      await sleep(800);
      await until(() => { const d = window.twDemand, v = d.view(); return !!v?.final && !d.client.waiting; }, 120000);
      // and the transfers that follow a final (about half a second in the clock worker on shared track)
      await sleep(2500);
      out.push({ edge: p.edge, a: runA, b: runB });
    }
    console.log(`t1805 picks ${picks.length}`);
    return out;
  }, [EDGES, MOVES]);
  for (const e of r) console.log(JSON.stringify(e));
  // the table: a = first segment of run a, b = first of run b, c = later segments of either
  const rows = { a: [], b: [], c: [] };
  for (const e of r) {
    e.a.forEach((s, k) => s.placed && rows[k ? 'c' : 'a'].push(s));
    e.b.forEach((s, k) => s.placed && rows[k ? 'c' : 'b'].push(s));
  }
  const mean = (xs) => xs.length ? Math.round(xs.reduce((s, v) => s + v, 0) / xs.length) : '-';
  const mx = (xs) => xs.length ? Math.max(...xs) : '-';
  for (const [k, xs] of Object.entries(rows)) {
    const w = (f) => mean(xs.map((s) => s.worker?.[f] ?? 0));
    console.log(`${k}: n ${xs.length}, click to next ghost mean ${mean(xs.map((s) => s.nextGhostMs))} max ${mx(xs.map((s) => s.nextGhostMs))} ms, click to placed ${mean(xs.map((s) => s.placedMs))}, to edited ${mean(xs.map((s) => s.editedMs))}, next ask sent ${mean(xs.map((s) => s.nextAskAt ?? 0))} answered in ${mean(xs.map((s) => s.nextAskMs ?? 0))},preview ask mean ${mean(xs.map((s) => s.meanAskMs))} max ${mx(xs.map((s) => s.maxAskMs))} ms, worker apply ${w('apply')} state ${w('state')} buffers ${w('buffers')} post ${w('post')} board ${w('board')} total ${w('total')}, main state ${mean(xs.map((s) => s.mainStateMs))} ms, snapshots ${xs.filter((s) => s.snapshot).length}, solves ${xs.reduce((s, v) => s + v.solves, 0)}`);
  }
  if (process.env.OUT) await writeFile(process.env.OUT, JSON.stringify(r, null, 1));
} finally {
  await close();
}
