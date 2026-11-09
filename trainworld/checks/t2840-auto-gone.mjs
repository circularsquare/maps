// T-2840: Map detail has no Auto. A stored "auto" (and none stored) comes up High: all the street layers and the
// device's pixel ratio; a stored "on" is still Low (fewer layers, ratio 1); the Settings row has two options.
//   TW_CORES=2 node trainworld/checks/t2840-auto-gone.mjs http://localhost:6177/
import { GL, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6177/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const read = async (stored, tab) => {
  const chrome = await launchChrome({ prefix: 'tw-t2840a-', args: GL, size: [1400, 900], deviceScaleFactor: 2 });
  try {
    await chrome.ctx.addInitScript(([s, tab]) => {
      const display = s === null ? {} : { lighterMap: s };
      localStorage.setItem('trainworld-ui', JSON.stringify({ v: 12, tab, display }));
    }, [stored, tab]);
    const page = await chrome.ctx.newPage();
    await page.goto(`${BASE}?debug=1&new=1&demandWorkers=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.map, undefined, { timeout: 240000 });
    await page.waitForTimeout(4000);
    const r = await page.evaluate(() => {
      const map = window.tw.map;
      return { setting: window.tw.display.value.lighterMap, ratio: map.getPixelRatio(), layers: map.getStyle().layers.filter((l) => map.getLayoutProperty(l.id, 'visibility') !== 'none').length };
    });
    const options = tab === 'settings' ? await page.$$eval('.slider .group .btn', (b) => b.map((x) => x.textContent.trim())).catch(() => []) : [];
    return { ...r, options };
  } finally { await chrome.close(); }
};
const high = await read('auto', 'settings');
console.log('stored auto:', JSON.stringify(high));
const none = await read(null, 'lines');
const low = await read('on', 'lines');
console.log('none:', JSON.stringify(none), 'on:', JSON.stringify(low));
check(high.setting === 'off', 'a stored "auto" reads as High');
check(high.ratio === 2, 'High: the device pixel ratio (2)');
check(high.layers > 38, `High: the full layers (${high.layers})`);
check(none.setting === 'off' && none.layers > 38, 'nothing stored: High, full layers');
check(low.setting === 'on' && low.ratio === 1, 'a stored "on" is still Low (ratio 1)');
check(low.layers < high.layers - 8, `Low: fewer layers (${low.layers} against ${high.layers})`);
check(high.options.length >= 2 && !high.options.some((o) => /auto/i.test(o)), `the Settings row has Low and High only (${high.options.join(', ')})`);
process.exitCode = fail ? 1 : 0;
