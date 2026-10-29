// T-1810: the clock worker's work after a demand solve (riders, then transfers), and how long track
// previews wait meanwhile, in headless Chrome.
//   node trainworld/checks/t1810-transfers.mjs [page url] [save]      (from the maps root)
// Loads the save, waits for demand's final and its transfers, then ROUNDS times sets a running
// line's trains (alternately two more and back) so demand solves again, with the track tool's ghost moved every
// frame from the edit to SETTLE ms after the solve's transfers went to the clock worker. Per round:
// the preview asks' ask-to-answer ms (mean, max, the max from 1 s before the transfers on, the
// three worst with ms from the transfers) and a hash of each drawn hour's last boarding slice, to
// compare runs bit for bit. A save of another city needs CITY=<its zone id> (CITY=tyo for tyo.save): the
// page opens that zone, since its demand asks for the zone in view (T-1814; in a New York page a
// Tokyo save gets no transfers).
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, ANGLE, REAL_SAVE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6191/';
const SAVE = process.argv[3] ?? REAL_SAVE;
const ROUNDS = Number(process.env.ROUNDS ?? 3), SETTLE = Number(process.env.SETTLE ?? 4000);

const { ctx, close } = await launchChrome({ prefix: 'tw-t1810-', size: [1200, 800], args: ANGLE });
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&city=${process.env.CITY ?? 'nyc'}&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => {
    const c = window.tw.client, recv = c.receive.bind(c);
    window.__t1810board = new Map();
    c.receive = (msg) => { if (msg.kind === 'trips' && msg.board) window.__t1810board.set(msg.epoch, msg.board.slice()); return recv(msg); };
    const post = c.post.bind(c);
    window.__t1810xfer = 0;
    c.post = (msg, tr) => { if (msg.kind === 'transfers') window.__t1810xfer++; return post(msg, tr); };
  });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) >= 5, undefined, { timeout: 300000, polling: 250 });
  await page.waitForFunction(() => { const d = window.twDemand, v = d?.view?.(); return !!v?.final && !d.client.waiting; }, undefined, { timeout: 600000, polling: 500 });
  await page.waitForFunction(() => window.__t1810xfer > 0, undefined, { timeout: 60000, polling: 250 }).catch(() => console.log('t1810 no transfers after the load'));
  console.log('t1810 load done');
  for (let round = 0; round < ROUNDS; round++) {
    const r = await page.evaluate(async ([round, SETTLE]) => {
      const tw = window.tw, map = tw.map, tools = tw.tools, c = tw.client;
      const ui = await import('/src/game/ui.ts');
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
      const ev = (x, y) => ({ point: { x, y }, lngLat: map.unproject([x, y]), originalEvent: { detail: 1, shiftKey: false } });
      let xferAt = 0;
      const ask = c.__t1810ask ??= c.preview.bind(c);
      const asks = [];
      c.preview = (req, cb) => { const p = { t: performance.now(), ms: -1 }; asks.push(p); return ask(req, (a) => { p.ms = performance.now() - p.t; cb(a); }); };
      // a ghost from the middle of the screen, moved back and forth
      ui.tool.value = 'track';
      tools.reset();
      const W = map.getCanvas().clientWidth, H = map.getCanvas().clientHeight;
      tools.click(ev(W / 2, H / 2));
      const t0 = performance.now();
      // the busiest running line's trains up by two, then back
      const w = tw.world.value;
      window.__t1810line ??= (() => { const l = w.save.lines.filter((l) => w.lineStats[l.id]?.status === 'running').sort((a, b) => (w.lineStats[b.id].riders ?? 0) - (w.lineStats[a.id].riders ?? 0))[0]; return { num: l.num, t: w.lineStats[l.id].trainsNeeded }; })();
      const L = window.__t1810line;
      await c.edit({ op: 'setTrainCount', line: L.num, trains: round % 2 ? L.t : L.t.map((n) => n + 2) }).catch(() => {});
      let k = 0, finalAt = 0, xfers = 0;
      const v0 = window.twDemand.view();
      while (performance.now() - t0 < 300000) {
        k++;
        tools.move(ev(W / 2 + 80 * Math.cos(k / 7), H / 2 + 80 * Math.sin(k / 7)));
        await frame();
        // the edit's solve shown final, then the transfers asked after it
        if (!finalAt) { const v = window.twDemand.view(); if (v && v !== v0 && v.final && !window.twDemand.client.waiting) { finalAt = performance.now(); xfers = window.__t1810xfer ?? 0; } }
        if (finalAt && !xferAt && window.__t1810xfer > xfers) xferAt = performance.now();
        if (xferAt && performance.now() - xferAt > SETTLE) break;
      }
      tools.reset();
      // the account as drawn once settled: a hash of each hour's last boarding slice
      await sleep(3000);
      const hashes = Object.fromEntries([...(window.__t1810board ?? new Map())].map(([e, b]) => {
        const u = new Uint32Array(b.buffer, b.byteOffset, b.length);
        let h = 0x811c9dc5;
        for (let i = 0; i < u.length; i++) h = Math.imul(h ^ u[i], 0x01000193) >>> 0;
        return [e, `${b.length}:${h.toString(16)}`];
      }));
      const done = asks.filter((q) => q.ms >= 0);
      const after = done.filter((q) => xferAt && q.t >= xferAt - 1000);
      return { asks: done.length, mean: done.reduce((s, q) => s + q.ms, 0) / Math.max(1, done.length), max: Math.max(0, ...done.map((q) => q.ms)), maxNearXfer: Math.max(0, ...after.map((q) => q.ms)), worst: done.slice().sort((a, b) => b.ms - a.ms).slice(0, 3).map((q) => `${Math.round(q.ms)} ms at ${Math.round(q.t - (xferAt || t0))}`).join(', '), finalAfterMs: finalAt ? finalAt - t0 : null, xferAfterMs: xferAt ? xferAt - t0 : null, hashes };
    }, [round, SETTLE]);
    console.log(`t1810 round ${round}: ${JSON.stringify(Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === 'number' ? Math.round(v) : v])))}`);
  }
} finally {
  await close();
}
