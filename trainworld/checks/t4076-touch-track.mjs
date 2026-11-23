// T-4076: track drawn by taps on a tablet reaches the blueprint, also when the browser follows each tap with a mouse move on the spot (Firefox for Android).
// From the maps root, on a Vite of your own (about 1 minute):
//   node trainworld/checks/t4076-touch-track.mjs [dev url]
// Firefox for Android follows a page change with a mouse move at the last mouse spot, the tap's own (Mozilla bugs
// 1973014, 1292041); that took the hover back from the finger after every tap. Playwright's Firefox makes no such
// move, so the check makes it by hand after each tap, in Chrome with touch emulated.
import { collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6199/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t4076-', size: [1280, 800], hasTouch: true, isMobile: true, deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
collectErrors(page, errors);
const at = async (p) => { const b = await page.locator('#map').boundingBox(); const [x, y] = await page.evaluate(([a, c]) => { const q = window.tw.map.project([a, c]); return [q.x, q.y]; }, p); return { x: Math.round(b.x + x), y: Math.round(b.y + y) }; };
const tapEl = async (sel) => { const b = await page.locator(sel).first().boundingBox(); await page.touchscreen.tap(b.x + b.width / 2, b.y + b.height / 2); await page.waitForTimeout(300); };
const tap = async (p) => {
  await page.touchscreen.tap(p.x, p.y);
  await page.waitForTimeout(150);
  await page.evaluate(([x, y]) => {
    const el = document.elementFromPoint(x, y);
    el.dispatchEvent(new PointerEvent('pointermove', { pointerType: 'mouse', pointerId: 1, isPrimary: true, clientX: x, clientY: y, bubbles: true }));
    el.dispatchEvent(new MouseEvent('mousemove', { clientX: x, clientY: y, bubbles: true }));
  }, [p.x, p.y]);
};
const edges = () => page.evaluate(() => window.tw.world.value.edges.length);
const bar = () => page.locator('.touch-bar').count();

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.map.fitBounds([[-73.998, 40.725], [-73.972, 40.755]], { padding: 100, duration: 0 }));
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 });
  await tapEl('button.tab[aria-label="Build"]');
  await tapEl('.tool-pick button:has(.tool-label:text-is("Track"))');
  const A = await at([-73.990, 40.731]), B = await at([-73.980, 40.748]), C = await at([-73.988, 40.752]);
  await tap(A);
  await page.waitForTimeout(800);
  await tap(B);
  await page.waitForTimeout(1500);
  check((await bar()) === 1, 'the end tapped: the touch bar shows');
  await tap(B);
  await page.waitForFunction(() => window.tw.world.value.edges.length >= 1, undefined, { timeout: 20000 }).catch(() => {});
  check((await edges()) === 1, 'a second tap on the end places the segment', `${await edges()} edges`);
  await tap(C);
  await page.waitForTimeout(1500);
  if (await page.locator('.touch-bar .tb.ok').count()) await tapEl('.touch-bar .tb.ok');
  await page.waitForFunction(() => window.tw.world.value.edges.length >= 2, undefined, { timeout: 20000 }).catch(() => {});
  check((await edges()) === 2, 'the bar\'s tick places the next segment', `${await edges()} edges`);
  await page.mouse.move(700, 300);
  await page.mouse.move(720, 320, { steps: 3 });
  await page.waitForTimeout(400);
  check((await bar()) === 0, 'a real mouse moving takes the hover back (the bar goes)');
  check(errors.length === 0, 'no page errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
