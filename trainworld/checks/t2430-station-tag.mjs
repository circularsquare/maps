// T-2430: the station tool's pointer tag has "12k trips within 1km" over its bar on track as soon as the
// pointer gets there, the figure asked for when the tool is picked and when the demand shown changes.
//   node trainworld/checks/t2430-station-tag.mjs [url] [fresh] [--save <file>] [--piece lng,lat;lng,lat]
//   (from the maps root; one to three minutes for New York, more for Tokyo)
// The station tool is picked as the save loads, a blueprint piece is drawn (by Central Park unless
// --piece), demand is waited for, and the pointer moves from open ground onto the piece: within 300 ms
// the tag must say the trips near (before T-2430 they were asked for only then, and came a second or
// more later, or with a big zone's solve running not until it ended). Open ground keeps its refusal,
// and after placing one the next spot has the distance and the trips. Screenshots to
// artifacts/trainworld/t2430/.
import { readFile, mkdir } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, REAL_SAVE, GL, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2]?.startsWith('http') ? process.argv[2] : 'http://localhost:6192/';
const FRESH = process.argv.includes('fresh');
const opt = (k) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : null; };
const SAVE = opt('--save') ?? REAL_SAVE;
const [A, B] = (opt('--piece') ?? '-73.968,40.778;-73.952,40.786').split(';').map((p) => p.split(',').map(Number));
const OUT = `${ARTIFACTS}t2430/`;
await mkdir(OUT, { recursive: true });
const tag = FRESH ? 'fresh' : opt('--save') ? 'save' : 'real';
const { ctx, close } = await launchChrome({ prefix: 'tw-t2430-', size: [1200, 800], args: GL });
let failed = false;
const report = (ok, what, detail = '') => {
  failed ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${tag}: ${what}${detail ? '  (' + detail + ')' : ''}`);
};
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  if (!FRESH) {
    await importSave(page, await readFile(SAVE));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  }
  await page.evaluate(([A, B]) => window.tw.map.jumpTo({ center: [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2], zoom: 14.5 }), [A, B]);
  const edge = await page.evaluate(async ([A, B]) => {
    const tw = window.tw, a = tw.lngLatToPack(...A), b = tw.lngLatToPack(...B);
    const before = new Set(tw.world.value.edges.map((e) => e.id));
    const r = await tw.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false });
    return r?.ok ? tw.world.value.edges.find((e) => !before.has(e.id))?.id ?? null : null;
  }, [A, B]);
  if (edge === null) throw new Error('the blueprint piece was not placed');
  // a spot a fraction f of the way from A to B, `side` px off the piece
  const at = (f, side = 0) => page.evaluate(([A, B, f, side]) => {
    const m = window.tw.map, r = m.getContainer().getBoundingClientRect();
    const p = m.project([A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f]), q = m.project(B), o = m.project(A);
    const L = Math.hypot(q.x - o.x, q.y - o.y);
    return { x: p.x + r.left - ((q.y - o.y) / L) * side, y: p.y + r.top + ((q.x - o.x) / L) * side };
  }, [A, B, f, side]);
  // the station tool by its key, as a player picks it, before demand is in
  const ground = await at(0.15, 200);
  await page.mouse.move(ground.x, ground.y);
  await page.keyboard.press('2');
  await demandFinal(page, { timeout: 300000 });
  await page.waitForTimeout(3000);
  const state = () => page.evaluate(() => {
    const t = document.querySelector('.tool-tag.build-tag'), vis = !!t && t.style.display !== 'none';
    return { hover: window.tw.tools.hover?.kind ?? null, tag: vis ? t.textContent : '', bad: vis && t.classList.contains('bad'), bar: vis && !!t.querySelector('.size-bar') };
  });
  // open ground: the refusal, as before
  await page.mouse.move(ground.x + 2, ground.y + 2);
  await page.waitForTimeout(300);
  const g = await state();
  report(g.hover === 'free' && g.tag === 'Place stations on track' && g.bad, 'open ground has the refusal', JSON.stringify(g));
  // onto the piece: the trips near at once
  const p = await at(0.15);
  await page.mouse.move(p.x - 3, p.y - 2);
  await page.mouse.move(p.x, p.y);
  await page.waitForTimeout(300);
  const s = await state();
  await page.screenshot({ path: `${OUT}${tag}-on-track.png` });
  report(s.hover === 'edge' && /trips within 1/.test(s.tag) && s.bar && !s.bad, 'on track the tag says the trips near within 300 ms, over its bar', JSON.stringify(s));
  // place one; the next spot has the distance from it and the trips near
  const before = await page.evaluate(() => window.tw.world.value.save.stations.length);
  await page.mouse.click(p.x, p.y);
  await page.waitForFunction((n) => window.tw.world.value.save.stations.length > n, before, { timeout: 20000 });
  await page.waitForTimeout(1000);
  const p2 = await at(0.85);
  let s2 = null;
  for (let i = 0; i < 20 && !/ from .*trips within 1/.test(s2?.tag ?? ''); i++) {
    await page.mouse.move(p2.x - 3 + (i % 2), p2.y - 2);
    await page.mouse.move(p2.x, p2.y);
    await page.waitForTimeout(500);
    s2 = await state();
  }
  await page.screenshot({ path: `${OUT}${tag}-second.png` });
  report(s2.hover === 'edge' && / from .*trips within 1/.test(s2.tag) && s2.bar, 'the next spot has the distance and the trips near', JSON.stringify(s2));
} catch (e) {
  failed = true;
  console.log(`FAIL  ${tag}: station tag check  ${e.message}`);
} finally {
  await close();
}
process.exitCode = failed ? 1 : 0;
