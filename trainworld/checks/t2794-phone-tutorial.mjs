// T-2794: the tutorial walked on an emulated phone (Pixel size, touch, DPR 3, Android user agent) from a fresh
// profile to the end of its second card; building goes through the edit API as in t2123-tutorial.mjs, the rest by
// tapping. Prints the renderer's private memory as it goes. Run against your own Vite (it imports app modules).
//   TW_CORES=3 node trainworld/checks/t2794-phone-tutorial.mjs http://localhost:6133/   (about 5 minutes)
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { mkdir } from 'node:fs/promises';
import { ARTIFACTS, GL, launchChrome, collectErrors, playwright } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6133/';
const OUT = join(ARTIFACTS, 't2794');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const MB = 1048576;
function rendererMB(profile) {
  const needle = profile.replace(/'/g, "''");
  const ps = `(Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like '*${needle}*' -and $_.CommandLine -match '--type=renderer' } | Measure-Object -Property PrivatePageCount -Sum).Sum`;
  return Math.round(Number(execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], { encoding: 'utf8' }).trim()) / MB);
}
const { devices } = await playwright();
const phone = { ...devices['Pixel 7'], deviceScaleFactor: 3 };
delete phone.defaultBrowserType;
const size = [phone.viewport.width, phone.viewport.height];
delete phone.viewport;
const chrome = await launchChrome({ prefix: 'tw-t2794t-', args: GL, size, ...phone });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  let crashed = false;
  page.on('crash', () => { crashed = true; });
  collectErrors(page, errors, /favicon|openfreemap|AJAXError|noritetsu/, /maplibre-gl/);
  await page.goto(`${BASE}?debug=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .new-game', { timeout: 180000 });
  await page.tap('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu') && document.querySelector('.guide'), undefined, { timeout: 240000 });
  await page.addStyleTag({ content: '#debug-root { display: none !important; }' });
  await page.waitForFunction(() => window.twDemand?.view?.()?.final, undefined, { timeout: 240000 });
  await page.waitForTimeout(2000);
  console.log('game up, renderer MB', rendererMB(chrome.profile));

  const stepNow = () => page.$eval('.guide-steps li.now', (l) => l.dataset.step).catch(() => null);
  const waitStep = (id, timeout = 20000) => page.waitForFunction((id) => document.querySelector('.guide-steps li.now')?.dataset.step === id, id, { timeout }).then(() => true, () => false);
  const tap = async (sel) => { await page.tap(sel, { timeout: 8000, force: true }).catch((e) => console.log('tap failed', sel, String(e).slice(0, 120))); };
  const mod = (fn, arg) => page.evaluate(async ([src, arg]) => (0, eval)(src)(await import('/src/game/world.ts'), arg), [fn.toString(), arg]);

  // card 1
  check((await stepNow()) === 'split', `first step is split (${await stepNow()})`);
  await tap('#topbar .split-row');
  check(await waitStep('bubble', 6000), `tapping the split ticks it (now ${await stepNow()})`);
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.94, 40.7], zoom: 12 }));
  await page.waitForFunction(() => window.twDemand?.view?.()?.final, undefined, { timeout: 120000 }).catch(() => {});
  await page.waitForTimeout(2500);
  // a bubble: a few spots until one selects
  let picked = false;
  const [W, H] = size;
  for (const [fx, fy] of [[0.5, 0.35], [0.6, 0.4], [0.4, 0.45], [0.7, 0.3], [0.3, 0.3], [0.55, 0.25]]) {
    await page.touchscreen.tap(W * fx, H * fy);
    await page.waitForTimeout(1200);
    if ((await page.evaluate(() => window.tw.selection.value?.kind)) === 'commuters') { picked = true; break; }
  }
  check(picked && await waitStep('trip', 5000), `tapping a bubble ticks (now ${await stepNow()})`);
  // a trip's two ends, through the selection when the tool is out of reach
  await page.keyboard.press('Escape');
  await page.keyboard.press('t');
  await page.touchscreen.tap(W * 0.5, H * 0.35);
  await page.waitForTimeout(700);
  await page.touchscreen.tap(W * 0.35, H * 0.2);
  check(await waitStep('destinations', 8000), `a trip ticks (now ${await stepNow()})`);
  await page.keyboard.press('Escape');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(800);
  await tap('[data-guide="destinations"]');
  check(await waitStep('track', 5000), `Destinations ticks (now ${await stepNow()})`);
  const stops = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const pts = [[-73.99, 40.73], [-73.98, 40.75], [-73.97, 40.77], [-73.955, 40.79]].map(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y]; });
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    await c.edit({ op: 'route', from: { kind: 'free', x: pts[0][0], y: pts[0][1], level: -1 }, to: { kind: 'free', x: pts[1][0], y: pts[1][1], level: -1 }, pis: [], single: false });
    for (let k = 2; k < pts.length; k++) { await pause(400); await c.edit({ op: 'route', from: { kind: 'node', node: near(pts[k - 1]) }, to: { kind: 'free', x: pts[k][0], y: pts[k][1], level: -1 }, pis: [], single: false }); }
    await pause(700);
    for (const p of pts.slice(0, 3)) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(300); }
    await c.edit({ op: 'constructAll' });
    await pause(800);
    return pts.slice(0, 3).map(near);
  });
  check(await waitStep('line', 20000), `track, stations and construct tick (now ${await stepNow()})`);
  const made = await page.evaluate((stops) => window.tw.client.edit({ op: 'addLine', stops, name: 'Tutorial', colour: '#0a7d3b' }), stops);
  const L = String(made.line);
  await page.waitForTimeout(1200);
  await page.waitForFunction(() => document.querySelector('.guide-end .btn')?.textContent === 'Continue', undefined, { timeout: 20000 }).catch(() => {});
  check((await page.textContent('.guide-end .btn').catch(() => '')) === 'Continue', `card one ends with Continue (now ${await stepNow()})`);
  console.log('card 1 done, renderer MB', rendererMB(chrome.profile));
  await tap('.guide-end .btn');
  await page.waitForSelector('.guide[data-card="2"]', { timeout: 8000 }).catch(() => {});

  // card 2
  await page.evaluate((id) => window.tw.select({ kind: 'line', line: id }), L);
  await mod((w, id) => w.setTrainCount(id, 'high', 4), L);
  check(await waitStep('play', 12000), `trains ticks (now ${await stepNow()})`);
  await tap('[data-guide="play"]');
  check(await waitStep('crowd', 12000), `play ticks (now ${await stepNow()})`);
  await page.waitForSelector('[data-guide="crowd"]', { timeout: 120000 }).catch(() => {});
  const seen = await page.$eval('[data-guide="crowd"]', (e) => { const b = e.getBoundingClientRect(), t = document.elementFromPoint(b.x + b.width / 2, b.y + b.height / 2); return { box: [b.x, b.y, b.width, b.height], top: t ? `${t.tagName}.${t.className}` : null, panel: !!e.closest('.dock, #dock') }; }).catch(() => null);
  console.log('crowd dot on the page', JSON.stringify(seen));
  await page.screenshot({ path: join(OUT, 'phone-tutorial-crowd.png') });
  await page.evaluate(() => { window.__ev = []; for (const k of ['pointerover', 'pointerdown', 'touchstart', 'click']) addEventListener(k, (e) => window.__ev.push(`${k} ${e.pointerType ?? ''} ${e.target?.tagName}.${e.target?.className} tip=${!!e.target?.closest?.('[data-tip]')}`), { capture: true, passive: true }); });
  await tap('[data-guide="crowd"]');
  await page.waitForTimeout(500);
  console.log('events on the tap', JSON.stringify(await page.evaluate(() => window.__ev.slice(0, 12))));
  if (process.env.STOP_AT_CROWD) throw new Error('stopped at the crowd step');
  const crowdTapped = await waitStep('levels', 8000);
  check(crowdTapped, `tapping the crowding dot ticks (now ${await stepNow()})`);
  if (!crowdTapped) await page.$eval('[data-guide="crowd"]', (e) => e.dispatchEvent(new PointerEvent('pointerover', { bubbles: true })));
  await mod((w, id) => { w.setTrainCount(id, 'medium', 2); w.setTrainCount(id, 'low', 1); }, L);
  check(await waitStep('traffic', 12000), `levels tick (now ${await stepNow()})`);
  await tap('[data-guide="traffic"]');
  check(await waitStep('star', 8000), `Traffic ticks (now ${await stepNow()})`);
  await page.screenshot({ path: join(OUT, 'phone-tutorial-star.png') });
  // the last step asks for a 1% train share: let the game run
  // (a short line rarely gets to 1% this fast, and t2123 stops here too: said, not failed)
  const done = await page.waitForFunction(() => document.querySelector('.guide-end .btn')?.textContent === 'Done', undefined, { timeout: 60000 }).then(() => true, () => false);
  const count = await page.$eval('.guide-steps li[data-step="star"] .guide-count', (e) => e.textContent).catch(() => '');
  console.log(done ? 'the star step finished' : `the star step did not reach 1% in 60 s (now ${await stepNow()}, ${count})`);
  check(!/NaN/.test(count), `the star step's counter is a number (${count})`);
  console.log('tutorial end, renderer MB', rendererMB(chrome.profile));
  check(!crashed, 'page did not crash');
} catch (e) {
  check(false, 'threw: ' + String(e).slice(0, 300));
} finally {
  await chrome.close();
}
check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 4).join(' | ') : ''));
process.exitCode = fail ? 1 : 0;
