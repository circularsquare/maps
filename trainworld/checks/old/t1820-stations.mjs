// T-1820: the station tool places stations on track only, and snaps to track from 15 px.
//   node trainworld/checks/old/t1820-stations.mjs [url] [fresh]      (from the maps root; about a minute)
// A blueprint piece is drawn by Central Park (on the real save, or a fresh game with `fresh`). With the
// station tool: a click on open ground places nothing and the tag says "Place stations on track"; a
// click 17 px off the track (outside the radius) places nothing; one 13 px off (inside it, outside
// the earlier 10 px) places a station on the piece. Screenshots to artifacts/trainworld/t1820/.
import { readFile, mkdir } from 'node:fs/promises';
import { launchChrome, importSave, REAL_SAVE, GL, ARTIFACTS } from '../lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6195/';
const FRESH = process.argv[3] === 'fresh';
const OUT = `${ARTIFACTS}t1820/`;
await mkdir(OUT, { recursive: true });
const tag = FRESH ? 'fresh' : 'real';
const { ctx, close } = await launchChrome({ prefix: 'tw-t1820-', size: [1200, 800], args: GL });
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
    ui.tool.value = 'station';
    window.tw.map.jumpTo({ center: [-73.964, 40.780], zoom: 15 });
  });
  await page.waitForTimeout(1500);
  const mid = await page.evaluate((id) => {
    const m = window.tw.map, r = m.getContainer().getBoundingClientRect();
    const w = window.tw.world.value, e = w.edges.find((x) => x.id === id);
    const q = (n) => m.project([w.nodes.get(n).lng, w.nodes.get(n).lat]);
    const a = q(e.a), b = q(e.b), L = Math.hypot(b.x - a.x, b.y - a.y);
    return { x: (a.x + b.x) / 2 + r.left, y: (a.y + b.y) / 2 + r.top, nx: -(b.y - a.y) / L, ny: (b.x - a.x) / L, len: L };
  }, edge);
  const state = () => page.evaluate(async () => {
    const { notice } = await import('/src/game/world.ts');
    const t = document.querySelector('.tool-tag.build-tag'), vis = t && t.style.display !== 'none';
    return { stations: window.tw.world.value.save.stations.length, hover: window.tw.tools.hover?.kind ?? null, tag: vis ? t.textContent : '', bad: !!vis && t.classList.contains('bad'), said: notice.value?.text ?? '' };
  });
  const hoverAt = async (x, y) => {
    await page.mouse.move(x - 3, y - 2);
    await page.mouse.move(x, y);
    await page.waitForTimeout(500);
  };
  const off = (d, along = 0) => [mid.x + mid.nx * d + (-mid.ny) * along, mid.y + mid.ny * d + mid.nx * along];
  report(mid.len > 120, 'the blueprint piece is long enough to click beside', `${mid.len.toFixed(0)} px`);

  // open ground, well away
  const s0 = await state();
  await hoverAt(...off(200));
  const sh = await state();
  await page.screenshot({ path: `${OUT}${tag}-open-ground.png` });
  report(sh.hover === 'free' && sh.tag === 'Place stations on track' && sh.bad, 'the ghost on open ground is refused with its reason', JSON.stringify(sh));
  await page.mouse.click(...off(200));
  await page.waitForTimeout(1500);
  const s1 = await state();
  report(s1.stations === s0.stations, 'a click on open ground places nothing', `stations ${s0.stations} -> ${s1.stations}, said "${s1.said}"`);

  // one just outside the radius (15 px)
  await hoverAt(...off(17));
  const so = await state();
  await page.mouse.click(...off(17));
  await page.waitForTimeout(1500);
  const s2 = await state();
  report(so.hover === 'free' && s2.stations === s0.stations, 'a click 17 px off the track places nothing', `hover ${so.hover}, stations ${s2.stations}`);

  // inside the new radius, outside the old one (10 px)
  await hoverAt(...off(13));
  const si = await state();
  await page.screenshot({ path: `${OUT}${tag}-near.png` });
  await page.mouse.click(...off(13));
  await page.waitForTimeout(2000);
  const s3 = await state();
  report(si.hover === 'edge' && s3.stations === s0.stations + 1, 'a click 13 px off the track places a station on it', `hover ${si.hover}, stations ${s0.stations} -> ${s3.stations}, said "${s3.said}"`);
} catch (e) {
  failed = true;
  console.log(`FAIL  ${tag}: stations check  ${e.message}`);
} finally {
  await close();
}
process.exitCode = failed ? 1 : 0;
