// T-2628: the hill shading never shows on water, whatever loads first. A new game paused, Terrain on,
// labels off, the basemap's vector tiles (its water and land) held 12 s each while the terrain tiles come:
//   1. off New England (the shelf edge and its canyons, no zone) and on Lake Superior (its floor in the
//      Terrain Tiles): the pixels that are water once all is in (bluer than red, 2 px in from any other
//      colour) show no relief at 3, 6, 9 and 11 s (under RELIEF_MAX of them a luminance step of RELIEF_STEP
//      or more from a neighbour), while the water is still on its way;
//   2. at Innsbruck with nothing held, the shading on land once all is in (`land-<tag>.png`; with a tag
//      other than `before`, compared with `land-before.png` when it is there: the land shade unchanged).
// Screenshots to artifacts/trainworld/hillwater/<place>-<when>-<tag>.png. About two minutes.
// Usage: node trainworld/checks/t2628-shade-water.mjs [url] [tag] (default http://localhost:6551/, after)

import { mkdirSync, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6551/';
const TAG = process.argv[3] ?? 'after';
const sep = BASE.includes('?') ? '&' : '?';
const OUT = join(ARTIFACTS, 'hillwater');
mkdirSync(OUT, { recursive: true });
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) failed++; };
const hold = (ms) => new Promise((d) => setTimeout(d, ms));
/** bare ground and water are flat in colour; relief steps between neighbours in most pixels */
const RELIEF_STEP = 2;
const RELIEF_MAX = 0.002;
const VECTOR_MS = 12000;
const AT = [3000, 6000, 9000, 11000];

const WATER = [
  { name: 'shelf', center: [-69.8, 40.3], zoom: 7.2 },
  { name: 'superior', center: [-87.6, 47.6], zoom: 7.6 },
];
const LAND = { center: [11.39, 47.27], zoom: 10.5 };

const net = { vector: 'pass' };
const { ctx, close } = await launchChrome({ prefix: 'tw-t2628-', size: [1200, 800] });
try {
  await ctx.route(/tiles\.openfreemap\.org\/.*\.pbf/, async (route) => {
    try {
      if (net.vector === 'slow') await hold(VECTOR_MS);
      await route.continue();
    } catch { /* the page went */ }
  });
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&demandWorkers=1&opened=all&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: '#debug-root { display: none !important; }' });
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    ui.display.value = { ...ui.display.value, hillshade: true };
  });
  const labelsOff = () => page.evaluate(() => {
    const m = window.tw.map;
    for (const l of m.getStyle().layers) if (l.type === 'symbol') m.setLayoutProperty(l.id, 'visibility', 'none');
  });
  const settled = () => page.waitForFunction(() => window.tw.map.areTilesLoaded() && !window.tw.map.isMoving(), undefined, { timeout: 90000, polling: 250 }).then(() => hold(1000));
  await settled();

  /** per shot, the share of `done`'s water pixels with relief; and how many water pixels there are */
  const reliefOverWater = (done, shots) => page.evaluate(async ([done, shots, step]) => {
    const px = async (b64) => {
      const img = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
      const cv = new OffscreenCanvas(img.width, img.height), cx = cv.getContext('2d');
      cx.drawImage(img, 0, 0);
      return { w: img.width, h: img.height, d: cx.getImageData(0, 0, img.width, img.height).data };
    };
    const D = await px(done), { w, h } = D;
    // the map only: right of the side panel, below the top bar and the buttons
    const x0 = 520, y0 = 200;
    const wet = new Uint8Array(w * h);
    for (let y = y0; y < h; y++) for (let x = x0; x < w; x++) { const i = y * w + x; wet[i] = D.d[4 * i + 2] > D.d[4 * i] + 10 ? 1 : 0; }
    const inner = new Uint8Array(w * h);
    let n = 0;
    for (let y = y0 + 2; y < h - 3; y++)
      for (let x = x0 + 2; x < w - 3; x++) {
        let all = 1;
        for (let dy = -2; dy <= 3 && all; dy++) for (let dx = -2; dx <= 3; dx++) if (!wet[(y + dy) * w + x + dx]) { all = 0; break; }
        if (all) inner[y * w + x] = 1, n++;
      }
    const out = [];
    for (const s of [...shots, done]) {
      const { d } = await px(s);
      const L = (i) => 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2];
      let hit = 0;
      for (let i = 0; i < w * h; i++) if (inner[i] && (Math.abs(L(i) - L(i + 1)) >= step || Math.abs(L(i) - L(i + w)) >= step)) hit++;
      out.push(hit / Math.max(1, n));
    }
    return { n, out };
  }, [done.toString('base64'), shots.map((s) => s.toString('base64')), RELIEF_STEP]);

  // 1. the water late: the view goes where nothing is loaded yet, with the basemap held
  for (const p of WATER) {
    net.vector = 'slow';
    await labelsOff();
    await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [p.center, p.zoom]);
    const t0 = Date.now(), shots = [];
    for (const at of AT) {
      await hold(at - (Date.now() - t0));
      shots.push(await page.screenshot({ path: join(OUT, `${p.name}-${at / 1000}s-${TAG}.png`) }));
    }
    net.vector = 'pass';
    await settled();
    await labelsOff();
    await hold(500);
    const done = await page.screenshot({ path: join(OUT, `${p.name}-done-${TAG}.png`) });
    // the shade did come: under the water, seen with the water hidden
    const water = (v) => page.evaluate((v) => { for (const id of ['water', 'water_over_shade']) window.tw.map.setLayoutProperty(id, 'visibility', v); }, v);
    await water('none');
    await hold(500);
    const under = await page.screenshot({ path: join(OUT, `${p.name}-under-${TAG}.png`) });
    await water('visible');
    const r = await reliefOverWater(done, [...shots, under]);
    const pct = (v) => (v * 100).toFixed(2) + '%';
    const during = r.out.slice(0, AT.length);
    check(r.n > 50000 && Math.max(...during) < RELIEF_MAX,
      `${p.name}: no relief on the water while it is on its way (${during.map(pct).join(', ')} of ${r.n} water pixels at ${AT.map((t) => t / 1000).join(', ')} s; under ${pct(RELIEF_MAX)})`);
    check(r.out.at(-1) < RELIEF_MAX, `${p.name}: none once it is in (${pct(r.out.at(-1))})`);
    check(r.out[AT.length] > RELIEF_MAX, `${p.name}: the shade is there under the water once it is in (${pct(r.out[AT.length])} with the water hidden)`);
  }

  // 2. land shading as before
  net.vector = 'pass';
  await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [LAND.center, LAND.zoom]);
  await settled();
  await hold(2500); // a late class grid redraws its tiles flattened
  await settled();
  await labelsOff();
  await hold(500);
  const lpng = await page.screenshot({ path: join(OUT, `land-${TAG}.png`) });
  const before = join(OUT, 'land-before.png');
  if (TAG !== 'before' && existsSync(before)) {
    const diff = await page.evaluate(async ([a64, b64]) => {
      const px = async (b) => {
        const img = await createImageBitmap(await (await fetch(`data:image/png;base64,${b}`)).blob());
        const cv = new OffscreenCanvas(img.width, img.height), cx = cv.getContext('2d');
        cx.drawImage(img, 0, 0);
        return cx.getImageData(0, 0, img.width, img.height).data;
      };
      const [a, b] = [await px(a64), await px(b64)];
      let s = 0, big = 0;
      for (let i = 0; i < a.length; i += 4) {
        const d = (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2])) / 3;
        s += d;
        if (d > 8) big++;
      }
      return { mean: s / (a.length / 4), bigShare: big / (a.length / 4) };
    }, [readFileSync(before).toString('base64'), lpng.toString('base64')]);
    check(diff.mean < 1 && diff.bigShare < 0.01, `land: the shade as before (mean difference ${diff.mean.toFixed(3)} of 255, ${(diff.bigShare * 100).toFixed(2)}% of pixels over 8)`);
  }
  check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
