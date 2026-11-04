// T-2439: the trip-length graph's hover cost. Sweeps the pointer across the graph in the city view (a new game
// in New York) with synthetic pointermoves and times each, up to the next macrotask and a forced layout: the
// graph's re-render, the tooltip's rebuild and the style/layout they cause. Reports ms per move and DOM
// mutations per move. Fails above 2 ms (median) per move.
//   node trainworld/checks/t2439-hover-perf.mjs [url]     (from the maps root; default localhost:6184)
import { ARTIFACTS, collectErrors, demandFinal, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6184/';
const ZONE = 'nyc';
const chrome = await launchChrome({ prefix: 'tw-t2439-', size: [1400, 1000] });
const page = await chrome.ctx.newPage();
const errors = [];
collectErrors(page, errors);
let fail = 0;
try {
  await page.goto(`${BASE}?debug=1&new=1&city=${ZONE}&demandWorkers=2&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: '#debug, .save-notice { display: none !important; }' });
  await demandFinal(page, { strict: false });
  await page.evaluate((z) => { window.tw.selection.value = { kind: 'city', city: z }; }, ZONE);
  await page.waitForFunction(() => document.querySelector('svg.trip-graph'), undefined, { timeout: 120000, polling: 500 });
  await page.waitForTimeout(1500);
  const r = await page.evaluate(async () => {
    const rect = document.querySelector('.tg-hover');
    rect.scrollIntoView({ block: 'center' });
    const svg = document.querySelector('svg.trip-graph');
    let muts = 0;
    new MutationObserver((l) => { muts += l.length; }).observe(svg, { subtree: true, attributes: true, childList: true });
    const tick = () => new Promise((res) => { const c = new MessageChannel(); c.port1.onmessage = () => res(); c.port2.postMessage(0); });
    const b = rect.getBoundingClientRect();
    const times = [];
    const y = b.top + b.height / 2;
    for (let pass = 0; pass < 3; pass++) {
      for (let x = b.left + 2; x < b.right - 2; x += 3) {
        const t0 = performance.now();
        rect.dispatchEvent(new PointerEvent('pointermove', { clientX: x, clientY: y, bubbles: true, pointerType: 'mouse' }));
        await tick();
        void document.body.offsetHeight;
        void document.querySelector('.ui-tag')?.offsetHeight;
        times.push(performance.now() - t0);
      }
    }
    times.sort((a, c) => a - c);
    const q = (p) => times[Math.floor(p * (times.length - 1))];
    return { n: times.length, median: q(0.5), p90: q(0.9), max: times[times.length - 1], mean: times.reduce((s, v) => s + v, 0) / times.length, muts: muts / times.length };
  });
  console.log(JSON.stringify(r));
  if (r.median > 2) { console.log('FAIL median over 2 ms'); fail++; }
  if (errors.length) { console.log('FAIL console errors', errors.slice(0, 3).join(' | ')); fail++; }
} catch (e) {
  console.log('FAIL', e.stack ?? e.message);
  fail++;
} finally {
  await chrome.close();
}
void ARTIFACTS;
console.log(fail ? 'T-2439 FAILED' : 'T-2439 PASSED');
process.exit(fail ? 1 : 0);
