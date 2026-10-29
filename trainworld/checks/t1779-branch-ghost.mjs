// T-1779, T-1780: branching off a line with the track tool on a real save, in headless Chrome.
//   node trainworld/checks/t1779-branch-ghost.mjs [page url] [save]      (from the maps root)
// For EDGES edges on screen: a start on the line (a branch), then MOVES pointer moves a frame apart
// out to 400 px at 25 degrees off the line; each preview's ask-to-answer ms, how many moves the
// ghost was behind when the pointer stopped, and whether the last ghost answers the last position;
// then a click (keep placing on) and the ms from it to the next segment's first ghost. The save
// defaults to the shared-track New York (../artifacts/trainworld/t1305-shared/nyc_real.save).
// Env: EDGES (default 12), MOVES (default 30), OUT (a json of every figure).
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { launchChrome, importSave, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6199/';
const SAVE = process.argv[3] ?? fileURLToPath(new URL('../../artifacts/trainworld/t1305-shared/nyc_real.save', import.meta.url));
const EDGES = Number(process.env.EDGES ?? 12), MOVES = Number(process.env.MOVES ?? 30);

const { ctx, close } = await launchChrome({ prefix: 'tw-t1779-', size: [1200, 800], args: ANGLE });
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  page.on('console', (m) => m.text().startsWith('picks') && console.log(m.text()));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) >= 50, undefined, { timeout: 300000, polling: 250 });
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.985, 40.752], zoom: 15 }));
  await page.waitForTimeout(4000);
  const r = await page.evaluate(async ([EDGES, MOVES]) => {
    const tw = window.tw, map = tw.map, tools = tw.tools, c = tw.client;
    const ui = await import('/src/game/ui.ts');
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
    const ev = (x, y) => ({ point: { x, y }, lngLat: map.unproject([x, y]), originalEvent: { detail: 1, shiftKey: false } });
    const settle = async (pred, ms = 15000) => { const t = performance.now(); while (performance.now() - t < ms) { if (pred()) return true; await sleep(5); } return false; };
    const asks = [];
    // the clock worker's own ms on each edit (`edited`'s ms: apply, the state, total)
    let editMs = null;
    const recv = c.receive.bind(c);
    c.receive = (msg) => { if (msg.kind === 'edited' && msg.ms) editMs = msg.ms; return recv(msg); };
    const ask = c.preview.bind(c);
    c.preview = (req, cb) => { const p = { t: performance.now(), ms: -1 }; asks.push(p); return ask(req, (a) => { p.ms = performance.now() - p.t; cb(a); }); };
    ui.keepPlacing.value = true;
    ui.tool.value = 'track';
    const n = tw.renderer.net, W = map.getCanvas().clientWidth, H = map.getCanvas().clientHeight;
    const picks = [];
    for (const [edge, pts] of n.edgePts) {
      // the middle of the edge's first leg well inside the view
      for (let k = 0; k + 5 < pts.length; k += 3) {
        const [xa, ya] = tw.overlay.toScreen(pts[k], pts[k + 1]), [xb, yb] = tw.overlay.toScreen(pts[k + 3], pts[k + 4]);
        const x0 = (xa + xb) / 2, y0 = (ya + yb) / 2;
        if (x0 < 380 || x0 > W - 380 || y0 < 100 || y0 > H - 100 || Math.hypot(xb - xa, yb - ya) < 8) continue;
        picks.push({ edge, x: x0, y: y0, h: Math.atan2(yb - ya, xb - xa) });
        break;
      }
    }
    const out = [];
    for (let i = 0; i < picks.length && out.length < EDGES; i++) {
      const p = picks[i];
      tools.reset();
      await frame();
      tools.move(ev(p.x, p.y));
      if (tools.hover?.kind !== 'edge') { (window.__skip ??= []).push(tools.hover?.kind); continue; }
      tools.click(ev(p.x, p.y));
      asks.length = 0;
      const a = p.h + (i % 2 ? 1 : -1) * 25 * Math.PI / 180;
      let x = p.x, y = p.y;
      const t0 = performance.now();
      for (let m = 1; m <= MOVES; m++) {
        x = p.x + Math.cos(a) * 400 * m / MOVES; y = p.y + Math.sin(a) * 400 * m / MOVES;
        tools.move(ev(x, y));
        await frame();
      }
      const moved = performance.now() - t0;
      const caught = await settle(() => tools.wantKey && tools.wantKey === tools.previewKey);
      const behind = performance.now() - t0 - moved;
      const ms = asks.filter((q) => q.ms >= 0).map((q) => q.ms);
      const pv = tools.lastPreview;
      // the click: place, and the next segment's first ghost from the new end
      const clickAt = performance.now();
      editMs = null;
      tools.click(ev(x, y));
      const placed = await settle(() => !tools.placing && tools.chain.length > 0);
      const placedMs = performance.now() - clickAt;
      tools.move(ev(x + Math.cos(a) * 150, y + Math.sin(a) * 150));
      const next = await settle(() => tools.chain.length > 0 && tools.wantKey && tools.wantKey === tools.previewKey);
      const nextMs = performance.now() - clickAt;
      out.push({ edge: p.edge, asked: asks.length, answered: ms.length, maxAskMs: Math.round(Math.max(0, ...ms)), meanAskMs: Math.round(ms.reduce((s, v) => s + v, 0) / Math.max(1, ms.length)), behindMs: Math.round(behind), caught, ok: !!pv?.ok, issues: pv?.issues ?? [], placed: placed && tools.chain.length > 0, placedMs: Math.round(placedMs), nextGhostMs: next ? Math.round(nextMs) : null, worker: editMs && Object.fromEntries(Object.entries(editMs).map(([k, v]) => [k, Math.round(v)])) });
      tools.reset();
      await c.edit({ op: 'undo' }, true).catch(() => {});
      await sleep(300);
    }
    console.log(`picks ${picks.length}, skipped ${JSON.stringify(window.__skip ?? [])}`);
    return out;
  }, [EDGES, MOVES]);
  for (const e of r) console.log(JSON.stringify(e));
  const avg = (k) => Math.round(r.reduce((s, e) => s + (e[k] ?? 0), 0) / Math.max(1, r.length));
  const placed = r.filter((e) => e.placed), pavg = (k) => Math.round(placed.reduce((s, e) => s + (e[k] ?? 0), 0) / Math.max(1, placed.length));
  console.log(`${r.length} branches: max ask ${Math.max(0, ...r.map((e) => e.maxAskMs))} ms, mean ask ${avg('meanAskMs')} ms, behind after the last move ${avg('behindMs')} ms mean (${Math.max(0, ...r.map((e) => e.behindMs))} max), click to placed ${pavg('placedMs')} ms, click to next ghost ${pavg('nextGhostMs')} ms mean (placed ones)`);
  if (process.env.OUT) await writeFile(process.env.OUT, JSON.stringify(r, null, 1));
} finally {
  await close();
}
