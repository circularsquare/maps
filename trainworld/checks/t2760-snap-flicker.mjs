// T-2760: the track tool's tooltip with the real-track snap on and the pointer still. Counts the
// tag's show/hide/content changes over 5 s. From the maps root:
//   node trainworld/checks/t2760-snap-flicker.mjs [url] [lng lat]
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, launchChrome, collectErrors, importSave } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6150/';
const CENTER = process.argv[4] ? [+process.argv[3], +process.argv[4]] : [-73.985, 40.758];
const chrome = await launchChrome({ prefix: 'tw-t2760-', size: [1200, 800] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  page.on('console', (m) => { if (/ROADS/.test(m.text())) console.log('  ', m.text()); });
  await page.goto(`${url}?debug=1&new=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  if (process.env.SAVE) {
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 0, undefined, { timeout: 300000 });
  }
  await page.evaluate(async ([c]) => {
    const ui = await import('/src/game/ui.ts');
    ui.keepPlacing.value = true;
    ui.snapReal.value = true;
    ui.tool.value = 'track';
    window.tw.map.jumpTo({ center: c, zoom: 13 });
  }, [CENTER]);
  await page.waitForTimeout(4000);
  const find = (skip) => page.evaluate((skip) => {
    const tw = window.tw, r = tw.map.getContainer().getBoundingClientRect();
    let n = 0;
    for (let y = 300; y < r.height - 60; y += 6) for (let x = 60; x < r.width - 100; x += 6) {
      const q = tw.tools.real.ring(x, y);
      if (q && n++ >= skip) return [r.left + x, r.top + y];
    }
    return null;
  }, skip);
  const a = await find(0);
  console.log('first ring point', a);
  if (!a) throw new Error('no real track drawn here');
  await page.mouse.move(...a);
  await page.waitForTimeout(300);
  await page.mouse.down(); await page.mouse.up();
  await page.waitForTimeout(500);
  console.log('after click', JSON.stringify(await page.evaluate(() => ({ from: window.tw.tools.real.from, start: !!window.tw.tools.start, tool: window.tw.tools.constructor.name }))));
  await page.screenshot({ path: 'artifacts/trainworld/t2760-shot.png' });
  // a second point some way along
  let b = null;
  for (const skip of [400, 800, 1600, 300, 150, 60]) {
    b = await find(skip);
    if (b && Math.hypot(b[0] - a[0], b[1] - a[1]) > 80) break;
  }
  console.log('second ring point', b);
  for (let i = 1; i <= 12; i++) { await page.mouse.move(a[0] + ((b[0] - a[0]) * i) / 12, a[1] + ((b[1] - a[1]) * i) / 12); await page.waitForTimeout(16); }
  if (process.env.CALLS) await page.evaluate(() => {
    const r = window.tw.tools.real; window.__calls = [];
    for (const k of ['move', 'again', 'drop', 'ask', 'reset', 'begin']) { const o = r[k].bind(r); r[k] = (...a) => { window.__calls.push(k + ' ' + (new Error().stack.split('\n').slice(2, 5).map((s) => s.trim().replace(/\(.*\//, '(')).join(' < '))); return o(...a); }; }
  });
  await page.evaluate(() => {
    const tag =document.querySelector('.tool-tag.build-tag');
    window.__flips = [];
    const t0 = performance.now();
    let last = '';
    const snap = () => (tag.style.display === 'none' ? 'hidden' : 'shown:' + tag.textContent + ':' + tag.className);
    last = snap();
    window.__flipObs = new MutationObserver(() => {
      const s = snap();
      if (s !== last) { window.__flips.push([Math.round(performance.now() - t0), s]); last = s; }
    });
    window.__flipObs.observe(tag, { attributes: true, childList: true, characterData: true, subtree: true });
  });
  await page.waitForTimeout(5000);
  const r = await page.evaluate(() => ({ flips: window.__flips, spot: window.tw.tools.real.spot?.phase, ring: window.tw.tools.real.ringAt }));
  if (process.env.CALLS) console.log((await page.evaluate(() => window.__calls)).slice(0, 12).join('\n'));
  const vis = r.flips.filter((f, i) => (f[1] === 'hidden') !== ((r.flips[i - 1]?.[1] ?? 'x') === 'hidden')).length;
  console.log('changes in 5 s:', r.flips.length, 'show/hide flips:', vis, 'phase', r.spot);
  console.log(JSON.stringify(r.flips.slice(0, 20)));
  // moving still works
  const c = await find(200);
  for (let i = 1; i <= 12; i++) { await page.mouse.move(b[0] + ((c[0] - b[0]) * i) / 12, b[1] + ((c[1] - b[1]) * i) / 12); await page.waitForTimeout(16); }
  await page.waitForTimeout(1500);
  const m = await page.evaluate(() => { const tag = document.querySelector('.tool-tag.build-tag'); return { display: tag.style.display, text: tag.textContent, ring: window.tw.tools.real.ringAt, phase: window.tw.tools.real.spot?.phase }; });
  console.log('after moving:', JSON.stringify(m));
  if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
