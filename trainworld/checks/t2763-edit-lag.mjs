// T-2763: click -> visible result for edits that wait on the clock worker, in headless Chrome.
//   node trainworld/checks/t2763-edit-lag.mjs [url] [real|new]      (from the maps root)
//   BUSY=1: run while the load's demand solve is still going (default: after it is final)
//   CASES=station,bp,station-del (default all)   TL=1: print every message both ways per case
// Cases: a station placed on built track with the station tool (click -> the station in `world`);
// a blueprint segment removed (removeThing -> gone from `world`); a built station removed (the
// question, then its answer); a stop added to a line that the cash cannot pay for (-> the notice).
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, acrossLine, ANGLE, REAL_SAVE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6131/';
const SAVE = process.argv[3] ?? 'real';
const BUSY = process.env.BUSY === '1';
const CASES = (process.env.CASES ?? 'station,bp,station-del').split(',');

const { ctx, close } = await launchChrome({ prefix: 'tw-t2763-', size: [1200, 800], args: ANGLE });
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  page.on('console', (m) => m.text().startsWith('t2763') && console.log(m.text()));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools && window.twDemand?.client, undefined, { timeout: 180000 });
  const centre = [-73.985, 40.752];
  if (SAVE === 'new') {
    await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 13 }), centre);
    await acrossLine(page, centre, 1.5);
    await acrossLine(page, [centre[0], centre[1] + 0.012], 1.5);
    await page.evaluate(() => window.tw.client.edit({ op: 'constructAll' }));
  } else {
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) >= 20, undefined, { timeout: 300000, polling: 250 });
  }
  await page.evaluate((c) => window.tw.map.jumpTo({ center: c, zoom: 15 }), centre);
  if (!BUSY) await page.waitForFunction(() => { const d = window.twDemand, v = d?.view?.(); return !!v?.final && !d.client.waiting; }, undefined, { timeout: 300000, polling: 500 }).catch(() => console.log('t2763 no demand final'));
  else await page.waitForTimeout(2500);
  await page.waitForTimeout(3000);
  const r = await page.evaluate(async ([CASES, BUSY, TL]) => {
    const tw = window.tw, map = tw.map, tools = tw.tools, c = tw.client;
    const ui = await import('/src/game/ui.ts');
    const rm = await import('/src/game/remove.ts');
    const wm = await import('/src/game/world.ts');
    const mm = await import('/src/game/money.ts');
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const ev = (x, y) => ({ point: { x, y }, lngLat: map.unproject([x, y]), originalEvent: { detail: 1, shiftKey: false } });
    const solving = () => { const d = window.twDemand, v = d?.view?.(); return !(v?.final && !d.client.waiting); };
    let tl = null;
    const recv = c.receive.bind(c), post = c.post.bind(c);
    c.post = (msg, tr) => { tl?.push(`${Math.round(performance.now() - tl.at)} > ${msg.kind}${msg.op ? ':' + msg.op.op : ''}${msg.q ? ':' + msg.q.what : ''}`); return post(msg, tr); };
    c.receive = (msg) => { tl?.push(`${Math.round(performance.now() - tl.at)} < ${msg.kind}${msg.kind === 'edited' ? ' ' + JSON.stringify({ ok: msg.ok, ms: msg.ms, issues: msg.issues }) : ''}`); return recv(msg); };
    const out = [];
    const settle = async () => { await sleep(500); if (BUSY) { await sleep(200); return; } const t = performance.now(); while (performance.now() - t < 120000 && solving()) await sleep(100); await sleep(1500); };
    // time from `start()` until `done()` holds, polled every ms
    const measure = async (label, start, done, limit = 20000, seen = null) => {
      const busy = solving();
      tl = []; tl.at = performance.now();
      const t0 = tl.at;
      const p = start();
      let ms = null, visibleMs = null;
      while (performance.now() - t0 < limit) { if (seen && visibleMs === null && seen()) visibleMs = Math.round(performance.now() - t0); if (done()) { ms = Math.round(performance.now() - t0); break; } await sleep(1); }
      if (TL) await sleep(900);
      await p?.catch?.(() => {});
      const o = { label, ms, ...(seen ? { visibleMs } : {}), busy, ...(TL ? { tl: tl.slice(0, 30) } : { msgs: tl.length }) };
      tl = null;
      out.push(o);
      return o;
    };
    const W = map.getCanvas().clientWidth, H = map.getCanvas().clientHeight;
    const w0 = () => tw.world.value;
    // a built edge's midpoint on screen
    const pickBuilt = (skip = 0) => {
      const n = tw.renderer.net, built = new Set(w0().edges.filter((e) => e.built).map((e) => e.id));
      let k0 = 0;
      for (const [edge, pts] of n.edgePts) {
        if (!built.has(edge)) continue;
        for (let k = 0; k + 5 < pts.length; k += 3) {
          const [xa, ya] = tw.overlay.toScreen(pts[k], pts[k + 1]), [xb, yb] = tw.overlay.toScreen(pts[k + 3], pts[k + 4]);
          const x = (xa + xb) / 2, y = (ya + yb) / 2;
          if (x < 300 || x > W - 300 || y < 200 || y > H - 200 || Math.hypot(xb - xa, yb - ya) < 12) continue;
          if (k0++ < skip) break;
          return { x, y, edge };
        }
      }
      return null;
    };
    if (CASES.includes('station')) {
      ui.tool.value = 'station';
      for (let i = 0; i < 3; i++) {
        const p = pickBuilt(i * 10);
        if (!p) { out.push({ label: 'station', err: 'no built edge on screen' }); break; }
        tools.move(ev(p.x, p.y));
        await sleep(800); // the ghost, as when a player points before clicking
        const n0 = w0().save.stations.length;
        await measure('station placed', () => tools.click(ev(p.x, p.y)), () => w0().save.stations.length > n0, 20000, () => !!document.querySelector('.station-ghost.placed') || w0().save.stations.length > n0);
        await settle();
      }
      ui.tool.value = null;
    }
    if (CASES.includes('bp')) {
      for (let i = 0; i < 3; i++) {
        const a = tw.lngLatToPack(-73.985 + i * 0.002, 40.752 + 0.02), b = tw.lngLatToPack(-73.985 + i * 0.002 + 0.006, 40.752 + 0.02);
        const before = new Set(w0().edges.map((e) => e.id));
        const ans = await c.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false });
        await settle();
        const e = w0().edges.find((x) => !before.has(x.id) && !x.built);
        if (!e) { out.push({ label: 'bp delete', err: 'no blueprint edge made ' + JSON.stringify(ans) }); continue; }
        await measure('blueprint track removed', () => rm.removeThing({ kind: 'track', edge: e.id }), () => !w0().edges.some((x) => x.id === e.id), 20000, () => tw.renderer.hide.includes(e.id) || !w0().edges.some((x) => x.id === e.id));
        await settle();
      }
    }
    if (CASES.includes('station-del')) {
      for (let i = 0; i < 2; i++) {
        const w = w0();
        const inLine = new Set(w.save.lines.flatMap((l) => l.stops.map(Number)));
        const s = w.save.stations.filter((x) => x.built && !inLine.has(Number(x.id)))[i] ?? w.save.stations.filter((x) => x.built)[i];
        if (!s) { out.push({ label: 'station delete', err: 'no built station' }); break; }
        ui.ask.value = null;
        await measure('built station: question', () => rm.removeThing({ kind: 'station', node: Number(s.id) }), () => !!ui.ask.value || !w0().save.stations.some((x) => x.id === s.id));
        if (ui.ask.value) await measure('built station: answered', () => ui.ask.value.run(), () => !w0().save.stations.some((x) => x.id === s.id));
        await settle();
      }
    }
    return out;
  }, [CASES, BUSY, !!process.env.TL]);
  for (const o of r) console.log(JSON.stringify(o));
} finally {
  await close();
}
