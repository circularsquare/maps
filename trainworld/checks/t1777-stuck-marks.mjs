// Marks drawn over the map in screen space that stay put while the map moves (T-1777). On the real
// save a blueprint piece is drawn in Manhattan; with each of the select tool (drag squares at its
// ends), the track tool and the station
// tool, the pointer rests by the piece, the wheel zooms out, the view jumps to Singapore at zoom
// 10.5 and pans. Any element in the map's own DOM layers shown both before and after the pan with
// the same transform is stuck. Run on the globe (New York is then on its far side, where a point
// projects to NaN) and the flat map. PASS when nothing is stuck.
//   node trainworld/checks/t1777-stuck-marks.mjs [url]   (from the maps root; about 1.5 minutes)
// Screenshots and the list to artifacts/trainworld/t1777/.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { launchChrome, importSave, REAL_SAVE, GL, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6196/';
const OUT = `${ARTIFACTS}t1777/`;
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1777-', size: [1200, 800], args: GL });
const out = {};
let failed = false;
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(3000);
  const A = [-73.968, 40.778], B = [-73.960, 40.782];
  const edge = await page.evaluate(async ([A, B]) => {
    const tw = window.tw, a = tw.lngLatToPack(...A), b = tw.lngLatToPack(...B);
    const before = new Set(tw.world.value.edges.map((e) => e.id));
    const r = await tw.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false });
    await new Promise((d) => setTimeout(d, 800));
    return r?.ok ? tw.world.value.edges.find((e) => !before.has(e.id))?.id ?? null : null;
  }, [A, B]);
  if (edge === null) throw new Error('the blueprint piece was not placed');
  const raf = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r()))));
  // every shown element with a transform in the map container's own layers (not MapLibre's)
  const shown = () => page.evaluate(() => {
    const box = window.tw.map.getContainer(), res = [];
    for (const el of box.querySelectorAll('*')) {
      if (!el.style.transform || el.closest('.maplibregl-canvas-container, .maplibregl-control-container')) continue;
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden') continue;
      let p = el.parentElement, hidden = false;
      while (p && p !== box) { if (getComputedStyle(p).display === 'none') { hidden = true; break; } p = p.parentElement; }
      const r = el.getBoundingClientRect();
      if (hidden || (r.width === 0 && r.height === 0)) continue;
      res.push({ cls: el.getAttribute('class') ?? el.tagName, t: el.style.transform });
    }
    return res;
  });
  const run = async (proj, name, toolName, selectEdge) => {
    await page.evaluate(async ([proj, t, e]) => {
      const tw = window.tw, ui = await import('/src/game/ui.ts');
      tw.display.value = { ...tw.display.peek(), projection: proj };
      ui.tool.value = t;
      tw.select(e === null ? null : { kind: 'track', edge: e });
      tw.map.jumpTo({ center: [-73.964, 40.780], zoom: 15 });
    }, [proj, toolName, selectEdge ? edge : null]);
    await raf(); await page.waitForTimeout(300);
    const p = await page.evaluate((A) => { const q = window.tw.map.project(A); return [q.x, q.y]; }, A);
    await page.mouse.move(p[0] + 20, p[1] + 20);
    await page.mouse.move(p[0] + 5, p[1] + 5);
    await raf(); await page.waitForTimeout(300);
    const near = (await shown()).length;
    for (let i = 0; i < 12; i++) { await page.mouse.wheel(0, 400); await page.waitForTimeout(60); }
    await page.waitForTimeout(800);
    await page.evaluate(() => window.tw.map.jumpTo({ center: [103.85, 1.33], zoom: 10.5 }));
    await raf(); await page.waitForTimeout(400);
    await page.mouse.move(600, 400);
    await raf(); await page.waitForTimeout(300);
    const s1 = await shown();
    await page.evaluate(() => window.tw.map.panBy([180, 90], { animate: false }));
    await raf(); await page.waitForTimeout(400);
    const s2 = await shown();
    const stuck = s2.filter((e) => s1.some((f) => f.cls === e.cls && f.t === e.t));
    await page.screenshot({ path: `${OUT}${proj}-${name}.png` });
    return { near, shown: s2.length, stuck };
  };
  for (const proj of ['globe', 'mercator']) {
    for (const [name, t, sel] of [['select', 'select', false], ['track', 'track', false], ['station', 'station', false]]) {
      const r = out[`${proj}-${name}`] = await run(proj, name, t, sel);
      const ok = r.near > 0 && r.stuck.length === 0;
      failed ||= !ok;
      console.log(`${ok ? 'PASS' : 'FAIL'}  ${proj}, ${name}: nothing stuck after zooming out and panning  (${r.near} marks by the piece; ${r.stuck.length} stuck${r.stuck.length ? ': ' + r.stuck.slice(0, 4).map((s) => `${s.cls} ${s.t}`).join('; ') : ''})`);
    }
  }
  await writeFile(`${OUT}stuck-marks.json`, JSON.stringify(out, null, 1));
} catch (e) {
  failed = true;
  console.log('FAIL  stuck marks check  ' + e.message);
} finally {
  await close();
}
process.exitCode = failed ? 1 : 0;
