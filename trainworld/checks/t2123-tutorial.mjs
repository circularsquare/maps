// T-2123: the tutorial walked in a fresh profile. Run 1: the start menu's default (New York), the
// card's first steps with the view rows folded, the ripple repeating until its control is clicked and
// moving from the Lines tab to New line, the close confirmation, a line built and given trains at High
// (Medium and Low stay 0: T-2092's fill is gone), the crowding and every-level steps, the game paused
// from the start until the last step's play button (T-2142), Continue, the
// second card's steps, and Start again in Settings. Run 2: Choose on map, whose first step is a zoom.
// Screenshots in artifacts/trainworld/t2123/.
//   node trainworld/checks/t2123-tutorial.mjs [url]     (about 4 minutes)
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, GL, NOISE, launchChrome, collectErrors, buildLine } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6161/';
const OUT = join(ARTIFACTS, 't2123');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const errors = [];

/** the box of what the ripple canvas has drawn now, CSS px, or null */
const ringBox = (page) => page.evaluate(() => {
  const c = document.querySelector('canvas.guide-ripple');
  if (!c || !c.width) return null;
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data, k = c.width / innerWidth;
  let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1;
  for (let y = 0; y < c.height; y += 2) for (let x = 0; x < c.width; x += 2) {
    if (d[(y * c.width + x) * 4 + 3] > 0) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  }
  return x1 < 0 ? null : { x0: x0 / k, y0: y0 / k, x1: x1 / k, y1: y1 / k };
});
/** whether the ripple is round the element `sel` (its centre inside the rings' box, the box within 24 px of it) */
const ringOn = async (page, sel) => {
  const b = await ringBox(page);
  const r = await page.evaluate((s) => { const e = document.querySelector(s); if (!e) return null; const q = e.getBoundingClientRect(); return { x0: q.left, y0: q.top, x1: q.right, y1: q.bottom }; }, sel);
  if (!b || !r) return false;
  const cx = (r.x0 + r.x1) / 2, cy = (r.y0 + r.y1) / 2;
  return cx > b.x0 && cx < b.x1 && cy > b.y0 && cy < b.y1 && b.x0 > r.x0 - 24 && b.x1 < r.x1 + 24;
};
const steps = (page) => page.$$eval('.guide-steps li', (ls) => ls.map((l) => `${l.dataset.step}:${l.className}`));
const stepNow = (page) => page.$eval('.guide-steps li.now', (l) => l.dataset.step).catch(() => null);
const waitStep = (page, id, timeout = 60000) => page.waitForFunction((id) => document.querySelector('.guide-steps li.now')?.dataset.step === id, id, { timeout }).then(() => true, () => false);
const shotCard = (page, name) => page.screenshot({ path: join(OUT, name + '.png') });
const startGame = async (page) => {
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu') && document.querySelector('.guide'), undefined, { timeout: 180000 });
  // the debug figures sit over the card's x and would be in every picture
  await page.addStyleTag({ content: '#debug-root { display: none !important; }' });
};

// ---- run 1: New York by default ----
{
  const chrome = await launchChrome({ prefix: 'tw-t2123-', size: [1400, 900], args: GL });
  try {
    const page = await chrome.ctx.newPage();
    collectErrors(page, errors, NOISE);
    await page.goto(`${BASE}?debug=1&demandWorkers=1&v=${Date.now()}`);
    await page.waitForSelector('.start-menu .new-game', { timeout: 180000 });
    const label = await page.textContent('.start-menu .new-game .start-new');
    check(label === 'Start in New York', `the start button reads "${label}"`);
    check(!!(await page.$('.new-game-cities [data-city="nyc"].on')) && !(await page.$('.new-game-map.on')), 'New York is lit, Choose on map is not');
    await page.click('.new-game-cities [data-city="nyc"]');
    check((await page.textContent('.start-menu .new-game .start-new')) === 'Start in New York', 'a click on the lit city keeps it');
    await page.screenshot({ path: join(OUT, '01-start-menu.png') });
    await startGame(page);
    await page.waitForTimeout(2500);
    const ui = (k) => page.evaluate(async (k) => (await import('/src/game/ui.ts'))[k].value, k);
    check((await ui('tab')) === 'build', 'the game opens on Build');
    check((await page.evaluate(() => window.tw.clock.running.value)) === false, 'a new game starts paused (T-2142)');
    check(!(await page.$('[data-guide="homes"]')) && !(await page.$('.demand-selector .btn')), 'the view rows are folded while the first card is up');
    console.log('card 1', JSON.stringify(await steps(page)));
    check((await stepNow(page)) === 'track', 'the step in hand is track');
    check(!(await page.$('.guide-steps li[data-step="zoom"]')), 'no zoom step on a city start');
    // the ripple keeps going: at 1.5 s, 6 s and 11 s it is round the Track tool
    await page.waitForTimeout(1500);
    const r1 = await ringOn(page, '[data-guide="track"]');
    await page.waitForTimeout(4500);
    const r2 = await ringOn(page, '[data-guide="track"]');
    await shotCard(page, '02-card-first-step');
    await page.waitForTimeout(5000);
    const r3 = await ringOn(page, '[data-guide="track"]');
    check(r1 && r2 && r3, `the ripple repeats round the Track tool (${r1} ${r2} ${r3})`);
    // switch to Lines: the ripple goes to the Build tab
    await page.click('[data-guide-tab="lines"]');
    await page.waitForTimeout(1200);
    check(await ringOn(page, '[data-guide-tab="build"]'), 'with Lines open the ripple is on the Build tab');
    await page.click('[data-guide-tab="build"]');
    await page.waitForTimeout(1200);
    check(await ringOn(page, '[data-guide="track"]'), 'Build opened: the ripple is back on the Track tool');
    await page.click('[data-guide="track"]');
    await page.waitForTimeout(1800);
    check(!(await ringBox(page)), 'a click on the Track tool stops the ripple');
    await page.keyboard.press('Escape');

    // the close asks first
    await page.click('.guide-x');
    await page.waitForSelector('dialog.confirmation', { timeout: 5000 });
    const dlg = await page.$eval('dialog.confirmation', (d) => ({ text: d.querySelector('#confirmation-text').textContent, buttons: [...d.querySelectorAll('button')].map((b) => b.textContent) }));
    console.log('close dialog', JSON.stringify(dlg));
    check(dlg.text === 'You can restart the tutorial in Settings.' && dlg.buttons.join('|') === 'Keep going|Close tutorial', 'the close asks, with Keep going and Close tutorial');
    await page.screenshot({ path: join(OUT, '03-close-confirm.png') });
    await page.click('dialog.confirmation button:text("Keep going")');
    await page.waitForTimeout(500);
    check(!!(await page.$('.guide')), 'Keep going keeps the card');

    // track, stations and construct through the edit API, then the line step's ripple
    await page.evaluate(() => window.tw.clock.setRunning(false));
    const c = await page.evaluate(async () => {
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
    check(await waitStep(page, 'line', 20000), `track, stations and construct tick (now ${await stepNow(page)})`);
    await page.waitForTimeout(1500);
    check(await ringOn(page, '[data-guide-tab="lines"]'), 'the line step ripples the Lines tab while Build is open');
    await page.click('[data-guide-tab="lines"]');
    await page.waitForTimeout(1200);
    check(await ringOn(page, '[data-guide="line"]'), 'Lines opened: the ripple moves to New line');
    await shotCard(page, '04-new-line-pulse');

    // the line, its stops, then trains at High
    const made = await page.evaluate(async (stops) => window.tw.client.edit({ op: 'addLine', stops, name: 'Tutorial', colour: '#0a7d3b' }), c);
    const L = String(made.line);
    await page.waitForTimeout(1000);
    await page.evaluate((id) => window.tw.select({ kind: 'line', line: id }), L);
    check(await waitStep(page, 'trains', 20000), `line and stops tick (now ${await stepNow(page)})`);
    const words = await page.$eval('.guide-steps li[data-step="stops"] .guide-words', (e) => e.textContent);
    check(words === 'Assign stops for the line by clicking them on the map', `the stops step reads "${words}"`);
    const mod = (fn, arg) => page.evaluate(async ([src, arg]) => (0, eval)(src)(await import('/src/game/world.ts'), arg), [fn.toString(), arg]);
    await mod((w, id) => w.setTrainCount(id, 'high', 4), L);
    await page.waitForTimeout(1500);
    const t1 = await page.evaluate((id) => window.tw.world.value.lineStats[id]?.trainsNeeded, L);
    check(JSON.stringify(t1) === '[4,0,0]', `4 trains at High leave Medium and Low at 0 (${t1})`);
    check(await waitStep(page, 'crowd', 10000), `trains ticks (now ${await stepNow(page)})`);
    await page.waitForSelector('[data-guide="crowd"]', { timeout: 120000 }).catch(() => {});
    await page.waitForTimeout(1500);
    check(await ringOn(page, '[data-guide="crowd"]'), 'the crowding step ripples the dot by High');
    await shotCard(page, '05-crowd-step');
    await page.hover('[data-guide="crowd"]');
    check(await waitStep(page, 'levels', 5000), `hovering the dot ticks (now ${await stepNow(page)})`);
    await page.mouse.move(700, 450);
    await page.waitForTimeout(1500);
    check(await ringOn(page, '[data-guide="levels"]'), 'the levels step ripples the Medium stepper');
    await shotCard(page, '06-levels-step');
    await mod((w, id) => { w.setTrainCount(id, 'medium', 2); w.setTrainCount(id, 'low', 1); }, L);
    check(await waitStep(page, 'play', 10000), `trains at every level tick (now ${await stepNow(page)})`);
    check((await page.evaluate(() => window.tw.clock.running.value)) === false, 'still paused at the play step');
    await page.waitForTimeout(1500);
    check(await ringOn(page, '[data-guide="play"]'), 'the play step ripples the play button');
    await shotCard(page, '06b-play-step');
    await page.click('[data-guide="play"]');
    await page.waitForFunction(() => document.querySelector('.guide-end .btn')?.textContent === 'Continue', undefined, { timeout: 10000 }).catch(() => {});
    check((await page.textContent('.guide-end .btn').catch(() => '')) === 'Continue', 'the first card ends with Continue');
    await shotCard(page, '07-card-one-finished');

    // the second card
    await page.click('.guide-end .btn');
    await page.waitForSelector('.guide[data-card="2"]', { timeout: 5000 });
    await page.waitForTimeout(800);
    check(!!(await page.$('[data-guide="homes"]')), 'the second card unfolds the view rows');
    console.log('card 2', JSON.stringify(await steps(page)));
    check(await ringOn(page, '[data-guide="homes"]'), 'the Homes step ripples Homes');
    await shotCard(page, '08-card-two');
    await page.click('[data-guide="homes"]');
    check(await waitStep(page, 'bubble', 5000), `Homes ticks (now ${await stepNow(page)})`);
    await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.94, 40.70], zoom: 12 }));
    await page.waitForFunction(() => window.twDemand?.view?.()?.final, undefined, { timeout: 120000 }).catch(() => {});
    await page.waitForTimeout(2500);
    // a bubble: a few spots in Brooklyn and Queens until one selects
    let picked = false;
    for (const [x, y] of [[760, 470], [820, 520], [700, 560], [880, 430], [640, 500], [900, 600]]) {
      await page.mouse.click(x, y);
      await page.waitForTimeout(1200);
      if ((await page.evaluate(() => window.tw.selection.value?.kind)) === 'commuters') { picked = true; break; }
      await page.keyboard.press('Escape');
    }
    check(picked && await waitStep(page, 'trip', 5000), `a bubble click ticks (now ${await stepNow(page)})`);
    await page.waitForTimeout(3000);
    // its destinations: one of the far bubbles, by the trip tool's two clicks if none is hit
    await page.keyboard.press('Escape');
    await page.keyboard.press('t');
    await page.mouse.click(760, 470);
    await page.waitForTimeout(500);
    await page.mouse.click(560, 300);
    check(await waitStep(page, 'star', 10000), `a trip ticks (now ${await stepNow(page)})`);
    await page.waitForTimeout(1500);
    await shotCard(page, '09-card-two-star');
    const star = await page.$eval('.guide-steps li[data-step="star"]', (l) => l.textContent);
    console.log('star step', star);

    // closing, then Start again in Settings
    await page.click('.guide-x');
    await page.click('dialog.confirmation button:text("Close tutorial")');
    await page.waitForTimeout(500);
    check(!(await page.$('.guide')), 'Close tutorial closes it');
    await page.evaluate(async () => { (await import('/src/game/ui.ts')).tab.value = 'settings'; });
    await page.waitForSelector('[data-guide-restart]', { timeout: 5000 });
    await page.click('[data-guide-restart]');
    await page.waitForSelector('.guide[data-card="1"]', { timeout: 5000 }).catch(() => {});
    check(!!(await page.$('.guide[data-card="1"]')), 'Start again in Settings brings back the first card');
  } finally {
    await chrome.close();
  }
}

// ---- run 2: Choose on map ----
{
  const chrome = await launchChrome({ prefix: 'tw-t2123b-', size: [1400, 900], args: GL });
  try {
    const page = await chrome.ctx.newPage();
    collectErrors(page, errors, NOISE);
    await page.goto(`${BASE}?debug=1&demandWorkers=1&v=${Date.now()}`);
    await page.waitForSelector('.start-menu .new-game', { timeout: 180000 });
    await page.click('.new-game-map');
    check((await page.textContent('.start-menu .new-game .start-new')) === 'Start', 'Choose on map: the button reads Start');
    await startGame(page);
    await page.waitForTimeout(4000);
    console.log('map start', JSON.stringify(await steps(page)));
    check((await page.evaluate(() => window.tw.clock.running.value)) === false, 'a map start is paused too');
    check((await stepNow(page)) === 'zoom', `a map start's first step is the zoom (${await stepNow(page)})`);
    check(await ringOn(page, '[data-guide="zoom"]'), 'the zoom step ripples the find button');
    await shotCard(page, '10-zoom-step');
    await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.98, 40.75], zoom: 11.5 }));
    check(await waitStep(page, 'track', 180000), `zooming into New York ticks it (now ${await stepNow(page)})`);
    await shotCard(page, '11-zoomed-in');
  } finally {
    await chrome.close();
  }
}

check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
process.exitCode = fail ? 1 : 0;
