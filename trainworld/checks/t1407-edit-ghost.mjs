// T-1407: how long the track tool's ghost waits after a service edit, in headless Chrome.
//   node trainworld/checks/t1407-edit-ghost.mjs [page url] [save]      (from the maps root)
// Loads the save and waits for demand's final, then ROUNDS times edits the busiest running line
// (setTrainCount two more and back, then setHeadway 240 s and back, in turn) with the track tool's
// ghost moved every frame from the edit on for WATCH ms. Per edit: the first preview asked after
// the edit (ask-to-answer ms), the worst in the watch, the edit's answer, and the clock worker's
// own ms. Then the edit's demand solve is waited out. Ends with each kind's mean and p95.
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, ANGLE, REAL_SAVE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6183/';
const SAVE = process.argv[3] ?? REAL_SAVE;
const ROUNDS = Number(process.env.ROUNDS ?? 8), WATCH = Number(process.env.WATCH ?? 1500);

const { ctx, close } = await launchChrome({ prefix: 'tw-t1407-', size: [1200, 800], args: ANGLE });
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&city=${process.env.CITY ?? 'nyc'}&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) >= 5, undefined, { timeout: 300000, polling: 250 });
  await page.waitForFunction(() => { const d = window.twDemand, v = d?.view?.(); return !!v?.final && !d.client.waiting; }, undefined, { timeout: 600000, polling: 500 });
  await page.waitForTimeout(4000);
  console.log('t1407 load done');
  const rows = [];
  for (let round = 0; round < ROUNDS; round++) {
    const r = await page.evaluate(async ([round, WATCH]) => {
      const tw = window.tw, map = tw.map, tools = tw.tools, c = tw.client;
      const ui = await import('/src/game/ui.ts');
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
      const ev = (x, y) => ({ point: { x, y }, lngLat: map.unproject([x, y]), originalEvent: { detail: 1, shiftKey: false } });
      const ask = c.__t1407ask ??= c.preview.bind(c);
      const asks = [];
      c.preview = (req, cb) => { const p = { t: performance.now(), ms: -1 }; asks.push(p); return ask(req, (a) => { p.ms = performance.now() - p.t; cb(a); }); };
      ui.tool.value = 'track';
      tools.reset();
      const W = map.getCanvas().clientWidth, H = map.getCanvas().clientHeight;
      tools.click(ev(W / 2, H / 2));
      let k = 0;
      for (; k < 10; k++) { tools.move(ev(W / 2 + 80 * Math.cos(k / 7), H / 2 + 80 * Math.sin(k / 7))); await frame(); }
      await sleep(300);
      const w = tw.world.value;
      window.__t1407line ??= (() => { const l = w.save.lines.filter((l) => w.lineStats[l.id]?.status === 'running').sort((a, b) => (w.lineStats[b.id].riders ?? 0) - (w.lineStats[a.id].riders ?? 0))[0]; return { num: l.num, t: w.lineStats[l.id].trainsNeeded }; })();
      const L = window.__t1407line;
      const kind = round % 4 < 2 ? 'count' : 'headway';
      const op = kind === 'count' ? { op: 'setTrainCount', line: L.num, trains: round % 2 ? L.t : L.t.map((n) => n + 2) } : { op: 'setHeadway', line: L.num, seconds: round % 2 ? null : [240, 240, 240] };
      const v0 = window.twDemand.view();
      asks.length = 0;
      const t0 = performance.now();
      let editedAt = 0, ms = null;
      const recv = c.receive.bind(c);
      const trips = [];
      c.receive = (msg) => {
        if (msg.kind === 'edited' && !editedAt) { editedAt = performance.now(); ms = msg.ms; }
        if (msg.kind === 'trips' && msg.board && trips.length < 4) trips.push(`${Math.round(performance.now() - t0)}:${Math.round(msg.boardMs)}ms/${msg.redone}`);
        return recv(msg);
      };
      const done = c.edit(op).catch(() => {});
      while (performance.now() - t0 < WATCH) {
        k++;
        tools.move(ev(W / 2 + 80 * Math.cos(k / 7), H / 2 + 80 * Math.sin(k / 7)));
        await frame();
      }
      await done;
      c.receive = recv;
      tools.reset();
      const got = asks.filter((q) => q.ms >= 0);
      // the edit's solve over before the next round
      const t1 = performance.now();
      while (performance.now() - t1 < 120000) { const v = window.twDemand.view(); if (v && v !== v0 && v.final && !window.twDemand.client.waiting) break; await sleep(200); }
      await sleep(3000);
      // `later`: the worst wait of a ghost asked from 150 ms after the edit on (a pointer back on the map from the panel)
      return { kind, first: got[0]?.ms ?? -1, firstAt: got[0] ? got[0].t - t0 : -1, max: Math.max(0, ...got.map((q) => q.ms)), later: Math.max(0, ...got.filter((q) => q.t - t0 >= 150).map((q) => q.ms)), asks: got.length, edited: editedAt - t0, trips: trips.join(' '), worker: ms && Object.fromEntries(Object.entries(ms).map(([a, b]) => [a, Math.round(b)])) };
    }, [round, WATCH]);
    rows.push(r);
    console.log(`t1407 round ${round}: ${JSON.stringify(Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === 'number' ? Math.round(v) : v])))}`);
  }
  const p95 = (xs) => { const s = xs.slice().sort((a, b) => a - b); return Math.round(s[Math.min(s.length - 1, Math.ceil(s.length * 0.95) - 1)]); };
  const mean = (xs) => Math.round(xs.reduce((s, v) => s + v, 0) / Math.max(1, xs.length));
  for (const kind of ['count', 'headway']) {
    const xs = rows.filter((r) => r.kind === kind);
    if (!xs.length) continue;
    console.log(`t1407 ${kind}: n ${xs.length}, first ghost after the edit mean ${mean(xs.map((r) => r.first))} p95 ${p95(xs.map((r) => r.first))} ms, worst in the watch mean ${mean(xs.map((r) => r.max))} p95 ${p95(xs.map((r) => r.max))}, worst asked from 150 ms on mean ${mean(xs.map((r) => r.later))} p95 ${p95(xs.map((r) => r.later))}, edited mean ${mean(xs.map((r) => r.edited))}`);
  }
} finally {
  await close();
}
