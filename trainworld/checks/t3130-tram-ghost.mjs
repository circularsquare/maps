// T-3130: a street tram ghost in Bilbao, in Spanish, with the Profit view on. The route onto a bridge over the
// Nervión at a sharp corner can be built (its corner piece over water was refused, GroundOverWater); the ghost is
// not painted the Profit view's deep red (a ghost has no profit); with the pointer off the map the tag stays
// with its ghost, price and all.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3130-tram-ghost.mjs [dev url]
// About 1 minute. Pictures in artifacts/trainworld/t3130-check-*.png.
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6189/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
/** the Profit view's colour at no profit (`PROFIT_STOPS[0]`) */
const DEEP_RED = [0xb2, 0x18, 0x2b];
// from a riverside street east of Abando onto a bridge over the Nervión, turning sharply onto it
const A = [-2.9338978634796153, 43.256250527227174], B = [-2.928463375394896, 43.267956872893215];

const chrome = await launchChrome({ prefix: 'tw-t3130-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { tram: true, streets: true, level: 0, keep: true } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3130-check-${name}.png`) });

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&lang=es&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const box = await (await page.$('#map')).boundingBox();
  const screen = async (p) => { const [x, y] = await page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p); return [box.x + x, box.y + y]; };
  await page.evaluate(([c]) => window.tw.map.jumpTo({ center: c, zoom: 15 }), [[(A[0] + B[0]) / 2, (A[1] + B[1]) / 2]]);
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving() && window.tw.map.areTilesLoaded(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(1500);
  // the Profit view, with a profit table (no profit anywhere) so the view paints
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    ui.display.value = { ...ui.display.value, trackColour: 'profit' };
    window.tw.renderer.setProfit(new Float32Array(64));
    ui.tool.value = 'track'; ui.tramTrack.value = true; ui.snapStreets.value = true; ui.buildLevel.value = 0; ui.keepPlacing.value = true;
  });
  await wait(500);
  /** pixels of the map near the Profit view's deep red */
  const reds = async () => {
    const b64 = (await page.screenshot({ clip: { x: box.x, y: box.y + 100, width: box.width, height: box.height - 100 } })).toString('base64');
    return page.evaluate(async ([b64, c]) => {
      const img = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
      const cv = new OffscreenCanvas(img.width, img.height), cx = cv.getContext('2d');
      cx.drawImage(img, 0, 0);
      const d = cx.getImageData(0, 0, img.width, img.height).data;
      let n = 0;
      for (let i = 0; i < d.length; i += 4) if (Math.abs(d[i] - c[0]) < 24 && Math.abs(d[i + 1] - c[1]) < 24 && Math.abs(d[i + 2] - c[2]) < 24) n++;
      return n;
    }, [b64, DEEP_RED]);
  };
  const before = await reds();
  await page.mouse.move(...(await screen(A)), { steps: 3 });
  await wait(300);
  await page.mouse.click(...(await screen(A)));
  await wait(300);
  const [x0, y0] = await screen(A), [x1, y1] = await screen(B);
  await page.mouse.move(x0 + (x1 - x0) * 0.9, y0 + (y1 - y0) * 0.9, { steps: 4 });
  await page.mouse.move(x1, y1, { steps: 4 });
  await page.waitForFunction(() => { const s = window.tw.tools.street; return (s.phase === 'done' && window.tw.tools.lastPreview) || s.phase === 'none'; }, undefined, { timeout: 20000 }).catch(() => {});
  await wait(800);
  const r = await page.evaluate(() => { const T = window.tw.tools, pv = T.lastPreview; return { phase: T.street.phase, ok: pv?.ok, issues: pv?.issues, water: pv?.ground?.[4] ?? 0, tag: T.tag.style.display !== 'none' ? T.tag.innerText : '' }; });
  check(r.phase === 'done' && r.ok === true, 'the street route onto the bridge can be built', JSON.stringify(r));
  check(r.water > 1, 'it crosses water', `${r.water?.toFixed?.(0)} m`);
  check(/M/.test(r.tag) && /km/.test(r.tag), 'the tag gives its length and price', JSON.stringify(r.tag));
  const during = await reds();
  await shot('profit');
  check(during - before < 200, 'the ghost is not painted the Profit view\'s deep red', `${before} deep red pixels before the ghost, ${during} with it`);
  // the pointer off the map (onto the top bar): the ghost stays, and its tag with it
  await page.mouse.move(x1, 20, { steps: 6 });
  await wait(500);
  const off = await page.evaluate(() => { const T = window.tw.tools; return { tag: T.tag.style.display !== 'none' ? T.tag.innerText : '', ghost: !!window.tw.renderer.preview }; });
  await shot('pointer-off-map');
  check(off.ghost && /M/.test(off.tag), 'with the pointer off the map the ghost keeps its tag', JSON.stringify(off));
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3130 FAILED: ${fail}` : 'T-3130 PASSED');
process.exit(fail ? 1 : 0);
