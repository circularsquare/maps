// T-4509: T-4508's two bugs went unseen for two days because nothing edited while zoomed out and then zoomed in, or
// deleted the last line in the draw order. A new NYC game with ?patch=check, played through lib/play.mjs's edit calls:
//   1 (new lines vanishing zoomed in) lines A, C (short) and B (longer) built, the full set drawn zoomed in (culling
//     boxes made) and a zoomed-out copy made; a stop of A dropped and E (no trains, drawn first) built, both while
//     zoomed out, so the boxes of the edit before are what the next one would wrongly keep; zooming in over E must give no "T-4262 check" error (the boxes kept differ from whole ones)
//   2 (deleted lines staying zoomed out) C, the last line in the draw order, removed while its zoomed-out copy is
//     made: no copy of the lines or the track may hold a segment past the set's end, and each copy is made again
// Fails on any console error. Fails with T-4508's fix reverted (notes/T-4509.md).
// From the tree root, on a Vite of your own:   node trainworld/checks/t4509-zoom-edits.mjs [dev url]   (about 1 minute)
import { openGame } from './lib/play.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6138/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const A = [[-73.99, 40.66], [-73.96, 40.65], [-73.93, 40.66], [-73.90, 40.67]];
const B = [[-73.93, 40.82], [-73.90, 40.84], [-73.87, 40.86], [-73.85, 40.88], [-73.83, 40.89]];
const C = [[-73.88, 40.74], [-73.86, 40.745]];
const E = [[-73.82, 40.70], [-73.79, 40.71], [-73.76, 40.72]];
const mid = (pts) => pts[Math.floor(pts.length / 2)];

const g = await openGame({ url: BASE, city: 'nyc', mode: 'sandbox', query: { patch: 'check' }, ui: { shade: 'light', v: 2 }, prefix: 'tw-t4509-' });
const page = g.page;
try {
  /** a line built the way the edits file does: track, a station at each point, construct, the line, its trains */
  const build = (pts, name, trains) => page.evaluate(async ([pts, name, trains]) => {
    const tw = window.tw, c = tw.client;
    const nearNode = ([lng, lat], skip) => { const p = tw.lngLatToPack(lng, lat); let best = null, bd = Infinity; for (const n of tw.world.value.nodes.values()) { if (skip?.has(n.id)) continue; const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const free = ([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return { kind: 'free', x: p.x, y: p.y, level: -1 }; };
    const send = async (op) => { const w0 = tw.world.value, t = performance.now(); const a = await c.edit(op); while (tw.world.value === w0 && performance.now() - t < 3000) await new Promise((r) => setTimeout(r, 30)); return a; };
    const old = new Set([...tw.world.value.nodes.values()].map((n) => n.id)), mine = (p) => nearNode(p, old), bad = [];
    const step = async (what, op) => { const a = await send(op); if (a?.ok === false) bad.push(`${what}: ${(a.issues ?? []).join(', ')}`); return a; };
    await step('track 1', { op: 'route', from: free(pts[0]), to: free(pts[1]), pis: [], single: false });
    for (let k = 2; k < pts.length; k++) await step(`track ${k}`, { op: 'route', from: { kind: 'node', node: mine(pts[k - 1]) }, to: free(pts[k]), pis: [], single: false });
    for (const [k, p] of pts.entries()) await step(`station ${k + 1}`, { op: 'addStation', at: { kind: 'node', node: mine(p) }, platform: 200, name: '' });
    await step('construct', { op: 'constructAll' });
    const r = await step('line', { op: 'addLine', stops: pts.map(mine), name, colour: '#0a7d3b' });
    if (trains && r?.line !== undefined) await step('trains', { op: 'setTrainCount', line: r.line, trains });
    return { line: r?.line ?? null, bad };
  }, [pts, name, trains]);
  const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
  /** the lines and track sets drawn at this zoom: their full set's boxes made, or a zoomed-out copy for its band */
  const view = async (at, zoom) => {
    await g.act.camera(at, zoom);
    const band = Math.ceil(zoom - 1e-6);
    await page.waitForFunction(([zoom, band]) => {
      const r = window.tw.renderer, st = r.lodOf.get(r.lines);
      return !!st && (zoom >= 15 ? !!st.boxes : st.bands.has(band));
    }, [zoom, band], { timeout: 20000 }).catch(() => {});
    await page.waitForTimeout(400);
  };
  /** segments held by a copy past the end of its set, and the copies there are */
  const stale = () => page.evaluate(() => {
    const r = window.tw.renderer, out = {};
    for (const [name, s] of [['lines', r.lines], ['track', r.track]]) {
      const st = r.lodOf.get(s);
      out[name] = [...(st?.bands.values() ?? [])].map((b) => Math.max(-1, ...b.first) >= s.count);
    }
    return out;
  });

  const a = await build(A, 'A', [4, 3, 2]), c = await build(C, 'C', [4, 3, 2]);
  check(!a.bad.length && !c.bad.length, 'lines A and C built', [...a.bad, ...c.bad].join(' | '));
  await view(mid(A), 16);
  await view(mid(A), 11.4);
  const b = await build(B, 'B', [4, 3, 2]);
  check(!b.bad.length, 'line B built (longer than A, so drawn first)', b.bad.join(' | '));
  await view(mid(B), 15.3);
  await view(mid(B), 11.4);
  const drop = await edit({ op: 'dropStop', line: a.line, index: 1 });
  check(drop?.ok !== false, 'a stop of A dropped', JSON.stringify(drop).slice(0, 120));
  await view(mid(B), 11.4);
  const e = await build(E, 'E', null);
  check(!e.bad.length, 'line E built while zoomed out', e.bad.join(' | '));
  await view(mid(E), 15.3);
  check(!g.pageErrors().some((x) => /T-4262 check|T-1028|T-2403/.test(x)), 'zooming in over the new line E gives no check error', g.pageErrors().slice(0, 3).join(' | '));

  // 2: the last line in the draw order removed while its copy is made
  await view(mid(C), 11.4);
  const before = await stale();
  check(before.lines.length > 0 && !before.lines.some(Boolean), 'zoomed out, the lines copy is made and holds nothing stale', JSON.stringify(before));
  const rm = await edit({ op: 'removeLine', line: c.line });
  check(rm?.ok !== false, 'line C removed', JSON.stringify(rm).slice(0, 120));
  await page.waitForTimeout(800);
  await view(mid(C), 11.4);
  const after = await stale();
  check(!after.lines.some(Boolean) && !after.track.some(Boolean), 'after the removal no copy holds a segment past its set', JSON.stringify(after));
  check(g.pageErrors().length === 0, 'no console errors', g.pageErrors().slice(0, 3).join(' | '));
} finally {
  await g.close();
}
process.exit(fail ? 1 : 0);
