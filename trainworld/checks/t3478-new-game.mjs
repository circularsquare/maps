// T-3478: New game from the saves panel, from a game in hand, saves it, goes to the start menu (`?next=new`) and
// starts the new game from there; it never ends back in the game left. Each case starts a New York game with a
// line, then picks New game, Choose on map, Start, and must reach the new game on the world map (an empty
// network in a slot of its own, the camera zoomed out) within the time given:
//   1. the terrain files failing at once (as offline): 20 s;
//   2. the terrain files stalled (no answer for 70 s), the old game reopened from the menu and New game picked
//      while it is still opening: 25 s (it waited out the 60 s fetch timeout before);
//   3. the terrain working: 20 s;
//   4. `?next=new` with no new game waiting in storage, the game with the line remembered as the last opened:
//      a new game still, not that one.
// About four minutes.
//   TW_CORES=4 node trainworld/checks/t3478-new-game.mjs http://localhost:6160/
import { GL, launchChrome, acrossLine } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6160/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const hold = (ms) => new Promise((d) => setTimeout(d, ms));
const lastSlot = (page) => page.evaluate(() => localStorage.getItem('anitabuilder-save-slot'));
const gameUp = (page, timeout) => page.waitForFunction(() => window.tw?.world?.value && !document.querySelector('.start-menu') && !document.querySelector('.loading-card'), undefined, { timeout });

async function run(mode, reopen) {
  const chrome = await launchChrome({ prefix: 'tw-t3478-', args: GL, size: [1400, 900] });
  try {
    await chrome.ctx.route(/\/terrain-world\/([0-9_-]+\.terrain|index\.bin)/, async (route) => {
      try {
        if (mode === 'block') return await route.abort('internetdisconnected');
        if (mode === 'stall') { await hold(70000); return await route.abort('timedout'); }
        await route.continue();
      } catch { /* the page went */ }
    });
    const page = await chrome.ctx.newPage();
    const urls = [];
    page.on('framenavigated', (f) => { if (f === page.mainFrame()) urls.push(f.url()); });
    await page.goto(`${BASE}?debug=1&demandWorkers=1&v=${Date.now()}`);
    await page.waitForSelector('.start-menu .new-game', { timeout: 120000 });
    await page.click('.start-menu .new-game-cities [data-city="nyc"]');
    await page.click('.start-menu .new-game .start-new');
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu'), undefined, { timeout: 240000 });
    await hold(3000);
    await acrossLine(page, [-73.98, 40.75], 2);
    await hold(2000);
    if (reopen) {
      await page.click('.tab.gear.save');
      await page.waitForSelector('.save-files .save-main-menu', { timeout: 60000 });
      await page.click('.save-files .save-main-menu');
      await page.waitForSelector('.start-menu .start-save', { timeout: 60000 });
      await hold(1000);
      await page.click('.start-menu .start-save');
      await page.waitForFunction(() => !document.querySelector('.start-menu') && window.tw?.client, undefined, { timeout: 60000 });
      await hold(500);
    }
    const old = await lastSlot(page);
    await page.click('.tab.gear.save');
    await page.waitForSelector('.save-new', { timeout: 60000 });
    await page.click('.save-new');
    await page.waitForSelector('.new-game-map', { timeout: 10000 });
    await page.click('.new-game-map');
    const t0 = Date.now();
    urls.length = 0;
    await page.click('.new-game-start');
    const limit = reopen ? 25000 : 20000;
    const reached = await page.waitForFunction((old) => {
      const w = window.tw?.world?.value;
      return w && !document.querySelector('.start-menu') && localStorage.getItem('anitabuilder-save-slot') !== old && w.edges.length === 0 && window.tw.map.getZoom() < 3;
    }, old, { timeout: limit }).then(() => true, () => false);
    const took = ((Date.now() - t0) / 1000).toFixed(1);
    const s = await page.evaluate(() => ({ menu: !!document.querySelector('.start-menu'), edges: window.tw?.world?.value?.edges.length, zoom: window.tw?.map?.getZoom().toFixed(1), slot: localStorage.getItem('anitabuilder-save-slot'), q: location.search })).catch((e) => ({ err: String(e) }));
    const label = `${mode}${reopen ? ', old game still opening' : ''}`;
    check(reached, `${label}: the new game on the world map in ${took} s (limit ${limit / 1000}); ${JSON.stringify(s)}`);
    check(s.slot !== old, `${label}: not the game left (${old} -> ${s.slot})`);
    check(urls.some((u) => /[?&]next=new/.test(u)), `${label}: by way of the start menu (${urls.map((u) => new URL(u).search).join(' , ')})`);
    check(!/next=|load=/.test(s.q ?? ''), `${label}: the address bar is left plain, so a reload goes to the menu (${s.q})`);
    if (mode === 'pass') {
      // 4. the new game's spec missing from storage
      // the game left is the one a page opens when it is given no other
      const before = old;
      await page.evaluate((k) => localStorage.setItem('anitabuilder-save-slot', k), old);
      await page.goto(`${BASE}?debug=1&demandWorkers=1&next=new&mode=normal&v=${Date.now()}`);
      const ok = await page.waitForFunction((b) => { const w = window.tw?.world?.value; return w && !document.querySelector('.start-menu') && localStorage.getItem('anitabuilder-save-slot') !== b && w.edges.length === 0; }, before, { timeout: 60000 }).then(() => true, () => false);
      check(ok, `no spec in storage: a new game still, not the game opened last (${before} -> ${await lastSlot(page)})`);
    }
  } finally { await chrome.close(); }
}

await run('block', false);
await run('stall', true);
await run('pass', false);
process.exitCode = fail ? 1 : 0;
