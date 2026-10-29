// T-1811: the track tool's runs and node drag, with real mouse events in headless Chrome.
//   node trainworld/checks/t1811-track-tool.mjs [url] [fresh]      (from the maps root; about a minute)
// On the real save (or a fresh game with `fresh`) a blueprint piece is drawn by Central Park. With
// keep placing on: a run from open ground to open ground carries on; one ending on the piece's loose
// end, and one ending mid-piece, end. Then a press on the piece's end moved 40 px drags the node
// (one undo puts it back), a short click on it starts a branch there, and the squares show.
// Screenshots to artifacts/trainworld/t1811/. One PASS or FAIL line each.
import { readFile, mkdir } from 'node:fs/promises';
import { launchChrome, importSave, REAL_SAVE, GL, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6183/';
const FRESH = process.argv[3] === 'fresh';
const OUT = `${ARTIFACTS}t1811/`;
await mkdir(OUT, { recursive: true });
const tag = FRESH ? 'fresh' : 'real';
const { ctx, close } = await launchChrome({ prefix: 'tw-t1811-', size: [1200, 800], args: GL });
let failed = false;
const report = (ok, what, detail = '') => {
  failed ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${tag}: ${what}${detail ? '  (' + detail + ')' : ''}`);
};
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  if (!FRESH) {
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  }
  await page.waitForTimeout(2000);
  const A = [-73.968, 40.778], B = [-73.960, 40.782];
  const edge = await page.evaluate(async ([A, B]) => {
    const tw = window.tw, a = tw.lngLatToPack(...A), b = tw.lngLatToPack(...B);
    const before = new Set(tw.world.value.edges.map((e) => e.id));
    const r = await tw.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false });
    return r?.ok ? tw.world.value.edges.find((e) => !before.has(e.id))?.id ?? null : null;
  }, [A, B]);
  if (edge === null) throw new Error('the blueprint piece was not placed');
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    ui.keepPlacing.value = true;
    ui.snapReal.value = false;
    ui.snapBeside.value = false;
    ui.tool.value = 'track';
    window.tw.map.jumpTo({ center: [-73.964, 40.780], zoom: 15 });
  });
  await page.waitForTimeout(1200);
  const st = () => page.evaluate(async () => { const t = window.tw.tools, { notice } = await import('/src/game/world.ts'); return { start: t.start ? t.start.end : null, chain: t.chain.length, placing: t.placing, hover: t.hover?.kind ?? null, edges: window.tw.world.value.edges.length, said: notice.value?.text ?? '' }; });
  const ends = () => page.evaluate((id) => {
    const w = window.tw.world.value, e = w.edges.find((x) => x.id === id), m = window.tw.map;
    if (!e) return null;
    // page px: the map's own px plus where the map sits on the page
    const r = m.getContainer().getBoundingClientRect();
    const pt = (n) => { const q = m.project([w.nodes.get(n).lng, w.nodes.get(n).lat]); return { node: n, x: q.x + r.left, y: q.y + r.top }; };
    const pts = window.tw.renderer.net.edgePts.get(id), k = Math.floor(pts.length / 6) * 3;
    const mid = window.tw.overlay.toScreen(pts[k], pts[k + 1]);
    return { a: pt(e.a), b: pt(e.b), mid: { x: mid[0] + r.left, y: mid[1] + r.top } };
  }, edge);
  const idle = () => page.waitForFunction(() => !window.tw.tools.placing, undefined, { timeout: 30000 });
  const click = async (x, y) => {
    await page.mouse.move(x - 3, y - 2);
    await page.mouse.move(x, y);
    await page.waitForTimeout(250);
    await page.mouse.click(x, y);
    await page.waitForTimeout(700); // past the double-click window, so the next click counts one
    await idle();
  };
  const esc = async () => { await page.evaluate(() => window.tw.tools.reset()); await page.waitForTimeout(200); };
  let E = await ends();
  // a perpendicular off the piece, on open ground
  const dx = E.b.x - E.a.x, dy = E.b.y - E.a.y, L = Math.hypot(dx, dy), nx = -dy / L, ny = dx / L;
  const off = (p, d) => [p.x + nx * d, p.y + ny * d];
  // along the piece (a to b) by `t` px and off it by `d`: a join comes in nearly along the track,
  // since a segment meeting it square on is refused as too tight a curve
  const at = (p, t, d) => [p.x + (dx / L) * t + nx * d, p.y + (dy / L) * t + ny * d];

  // 1a: open ground to open ground carries on
  const e0 = (await st()).edges;
  await click(...at(E.mid, -30, 140));
  await click(...at(E.mid, -180, 140));
  let s = await st();
  report(s.start !== null && s.chain === 1 && s.edges === e0 + 1, 'a run on open ground carries on with keep placing', `start ${JSON.stringify(s.start)}, chain ${s.chain}, edges +${s.edges - e0}, said "${s.said}"`);
  await esc();

  // 1b: a run ending on the loose end stops
  const e1 = (await st()).edges;
  await click(...at(E.b, 250, 20));
  await page.mouse.move(E.b.x + 2, E.b.y + 1);
  await page.waitForTimeout(250);
  const hb = (await st()).hover;
  await click(E.b.x, E.b.y);
  s = await st();
  await page.screenshot({ path: `${OUT}${tag}-joined-end.png` });
  report(s.start === null && s.edges === e1 + 1, 'a run ending on a loose end stops with keep placing on', `hover there ${hb}, start ${JSON.stringify(s.start)}, edges +${s.edges - e1}, said "${s.said}"`);
  await esc();

  // 2: squares show in the track tool
  await page.mouse.move(600, 400);
  await page.waitForTimeout(300);
  const marks = await page.evaluate(() => {
    const vis = (sel) => [...document.querySelectorAll(sel)].filter((m) => m.style.display !== 'none').length;
    return { squares: vis('.tool-marks > .pi-mark'), loose: vis('.tool-marks > .pi-mark.loose'), rings: vis('.tool-marks > .end-mark') };
  });
  await page.screenshot({ path: `${OUT}${tag}-squares.png` });
  report(marks.squares > 0 && marks.loose > 0, 'the track tool shows drag squares at blueprint points', JSON.stringify(marks));

  // 2a: press and drag the piece's first end 40 px
  E = await ends();
  const before = await page.evaluate((n) => { const p = window.tw.world.value.nodes.get(n); return [p.lng, p.lat]; }, E.a.node);
  const e3 = (await st()).edges;
  await page.mouse.move(E.a.x, E.a.y);
  await page.waitForTimeout(200);
  await page.mouse.down();
  for (let i = 1; i <= 10; i++) { await page.mouse.move(E.a.x + 4 * i, E.a.y + 3 * i); await page.waitForTimeout(30); }
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${OUT}${tag}-dragging.png` });
  await page.mouse.up();
  await page.waitForTimeout(900);
  const after = await page.evaluate((n) => { const p = window.tw.world.value.nodes.get(n); return p ? [p.lng, p.lat] : null; }, E.a.node);
  s = await st();
  const moved = after && Math.hypot(after[0] - before[0], after[1] - before[1]) > 1e-4;
  report(!!moved && s.edges === e3 && s.start === null, 'a press dragged 40 px on a blueprint end moves it and starts no run', `moved ${moved}, edges +${s.edges - e3}, start ${JSON.stringify(s.start)}`);
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }, true));
  await page.waitForTimeout(600);
  const back = await page.evaluate((n) => { const p = window.tw.world.value.nodes.get(n); return [p.lng, p.lat]; }, E.a.node);
  report(Math.hypot(back[0] - before[0], back[1] - before[1]) < 1e-7, 'one undo puts the dragged end back');

  // 2b: a short click on the end starts a run from it, and the next click branches
  E = await ends();
  const e4 = (await st()).edges;
  await click(E.a.x, E.a.y);
  const s1 = await st();
  await click(...off(E.a, 170));
  s = await st();
  await page.screenshot({ path: `${OUT}${tag}-branched.png` });
  report(s1.start?.kind === 'node' && s1.start.node === E.a.node && s.edges === e4 + 1, 'a short click on a blueprint end starts track from it', `start ${JSON.stringify(s1.start)}, edges +${s.edges - e4}`);
  await esc();

  // 1c (last: it cuts the piece in two): a run ending mid-piece stops
  E = await ends();
  const e2 = (await st()).edges;
  await click(...at(E.mid, -400, -80));
  await page.mouse.move(E.mid.x + 1, E.mid.y + 1);
  await page.waitForTimeout(250);
  const hm = (await st()).hover;
  await click(E.mid.x, E.mid.y);
  s = await st();
  report(s.start === null && s.edges > e2, 'a run ending mid-piece stops with keep placing on', `hover there ${hm}, start ${JSON.stringify(s.start)}, edges +${s.edges - e2}, said "${s.said}"`);
} catch (e) {
  failed = true;
  console.log(`FAIL  ${tag}: track tool check  ${e.message}`);
} finally {
  await close();
}
process.exitCode = failed ? 1 : 0;
