// T-2232: passing tracks in the panels. A new sandbox game with T-2228's test line (180 km of
// high-speed ground track, 7 stations 30 km apart, a local at all of them and an express at the
// first, middle and last, both every 10 minutes): the express panel's button adds passing tracks at
// the five middle stations in one edit at the price it shows, then the express loses its "Behind
// slower trains" and the local gains "Waiting for faster trains". Screenshots light and dark in
// artifacts/trainworld/t2232/. From the maps root:
//   node trainworld/checks/t2232-passing-panels.mjs [url] [local headway s] [express headway s]
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2232');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6188/';
const HL = Number(process.argv[3] ?? 600), HX = Number(process.argv[4] ?? 600);
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const LAT = 40.3, LNG0 = -79.6, N = 7, STEP_KM = 30, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180));
const PTS = Array.from({ length: N }, (_, k) => [LNG0 + k * STEP_KM * KM, LAT]);

/** the inspector's rows as [label, value] pairs, its buttons' text and hovers */
const readPanel = (page) => page.evaluate(() => {
  const el = document.querySelector('#inspector');
  const rows = [...el.querySelectorAll('ul.rows > li')].map((li) => [li.querySelector('.grow')?.textContent?.trim() ?? li.firstElementChild?.textContent?.trim() ?? '', li.lastElementChild?.textContent?.trim() ?? '']);
  const buttons = [...el.querySelectorAll('button')].map((b) => ({ text: b.textContent.trim(), tip: b.dataset.tip ?? '' }));
  return { rows, buttons };
});
const shade = (page, s) => page.evaluate((x) => { document.documentElement.dataset.shade = x; window.tw.shade.value = x; }, s);
const select = (page, sel) => page.evaluate((s) => window.tw.select(s), sel);
const toBottom = (page) => page.evaluate(() => { const el = document.querySelector('#inspector'); let s = el; while (s && s.scrollHeight <= s.clientHeight + 1) s = s.parentElement; if (s) s.scrollTop = s.scrollHeight; });
async function shot(page, name, hover) {
  await toBottom(page);
  await page.waitForTimeout(300);
  if (hover) { await page.locator(hover).first().hover(); await page.waitForTimeout(700); }
  else await page.mouse.move(5, 450);
  const box = await page.locator('#inspector').boundingBox();
  await page.screenshot({ path: join(OUT, name), clip: { x: Math.max(0, box.x - 8), y: 0, width: Math.min(box.width + 300, 1400 - box.x + 8), height: 900 } });
}

const chrome = await launchChrome({ prefix: 'tw-t2232-', size: [1400, 900] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(1000);
  const built = await page.evaluate(async ([PTS, HL, HX]) => {
    const tw = window.tw, c = tw.client;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = PTS.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const log = [];
    const r0 = await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: 0 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: 0 }, pis: [], single: false, hsr: true, design: 320 });
    log.push(r0?.ok);
    for (let k = 2; k < xy.length; k++) {
      await pause(250);
      const r = await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[k - 1]) }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: 0 }, pis: [], single: false, hsr: true, design: 320 });
      log.push(r?.ok);
    }
    await pause(500);
    for (const [k, p] of xy.entries()) { const r = await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 400, name: 'ABCDEFG'[k] }); log.push(r?.ok); await pause(150); }
    const k = await c.edit({ op: 'constructAll' });
    log.push(k?.ok);
    await pause(600);
    const stops = xy.map(near);
    const local = (await c.edit({ op: 'addLine', stops, name: 'Local', colour: '#0a7d3b', autoFast: true })).line;
    await c.edit({ op: 'lineFast', line: local, fast: true });
    const express = (await c.edit({ op: 'copyLine', line: local })).line;
    await c.edit({ op: 'setStops', line: express, stops: [stops[0], stops[3], stops[6]] });
    await c.edit({ op: 'lineFast', line: express, fast: true });
    await c.edit({ op: 'lineLook', line: express, name: 'Express', colour: '#d9467a', letter: 'X' });
    await c.edit({ op: 'setHeadway', line: local, seconds: [HL, HL, 0] });
    await c.edit({ op: 'setHeadway', line: express, seconds: [HX, HX, 0] });
    await pause(2500);
    return { log, stops, local, express };
  }, [PTS, HL, HX]);
  console.log('built', JSON.stringify(built));
  check(built.log.every(Boolean), 'the test line was built');
  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 8.5 }), PTS[3]);

  // the express before: following the local, and the button
  await select(page, { kind: 'line', line: String(built.express) });
  await page.waitForFunction(() => [...document.querySelectorAll('#inspector button')].some((b) => b.textContent.startsWith('Add passing tracks')), undefined, { timeout: 20000 }).catch(() => {});
  const before = await readPanel(page);
  const behind0 = before.rows.find((r) => r[0] === 'Behind slower trains');
  const btn = before.buttons.find((b) => b.text.startsWith('Add passing tracks'));
  console.log('express before', JSON.stringify(before.rows), JSON.stringify(btn));
  check(!!behind0, `the express shows "Behind slower trains" before (${behind0?.[1]})`);
  check(!!btn && /at 5 stations for \$[\d.]+[MB]$/.test(btn.text), `the button reads "${btn?.text}"`);
  check(btn?.tip === 'At B, C, D, E and F', `its hover names the five stations ("${btn?.tip}")`);
  for (const sh of ['light', 'dark']) {
    await shade(page, sh);
    await shot(page, `express-before-${sh}.png`, '#inspector button:has-text("Add passing tracks")');
  }
  await shade(page, 'light');

  // one click: one edit, all five, the station value up by the price shown (the charge also buys
  // the trains the local now needs for its headway)
  const priceShown = Number(btn?.text.match(/\$([\d.]+)M$/)?.[1] ?? NaN);
  const value0 = await page.evaluate(() => window.tw.world.value.money.builtValue);
  await page.evaluate(() => {
    const c = window.tw.client;
    window.__edits = [];
    const orig = c.edit.bind(c);
    c.edit = (op, ...rest) => { const p = orig(op, ...rest); window.__edits.push({ op: op.op, nodes: op.nodes, bit: op.bit }); void p.then((a) => { window.__edits[window.__edits.length - 1].charge = a?.charge; }); return p; };
  });
  await page.locator('#inspector button:has-text("Add passing tracks")').first().click();
  await page.waitForFunction(() => window.__edits?.length && window.__edits[0].charge !== undefined, undefined, { timeout: 20000 });
  await page.waitForTimeout(2500);
  const edits = await page.evaluate(() => window.__edits);
  console.log('edits', JSON.stringify(edits));
  check(edits.length === 1 && edits[0].op === 'stationUpgrade' && edits[0].bit === 8 && edits[0].nodes?.length === 5, `one stationUpgrade edit with five stations (${edits.length} edits)`);
  const value1 = await page.evaluate(() => window.tw.world.value.money.builtValue);
  const tracks = (value1 - value0) / 1e6;
  check(Math.abs(tracks - priceShown) < 0.05, `passing tracks cost $${tracks.toFixed(1)}M against $${priceShown}M shown (charged $${edits[0]?.charge?.toFixed(1)}M with trains)`);
  const bits = await page.evaluate((stops) => { const st = window.tw.world.value.save.stations; return stops.map((n) => (st.find((s) => s.num === n)?.upgrades ?? 0) & 8); }, built.stops);
  check(bits.join(',') === '0,8,8,8,8,8,0', `passing tracks at the five middle stations (${bits.join(',')})`);
  const notice = await page.evaluate(() => [...document.querySelectorAll('body *')].map((e) => e.childElementCount === 0 ? e.textContent : '').find((t) => t.startsWith('Passing tracks built')) ?? '');
  check(notice.startsWith(`Passing tracks built at 5 stations for $${btn?.text.match(/\$([\d.]+M)$/)?.[1]}.`), `the notice reads "${notice}"`);

  // the express after: no following, no button
  await page.waitForTimeout(1500);
  const after = await readPanel(page);
  console.log('express after', JSON.stringify(after.rows));
  const behind1 = after.rows.find((r) => r[0] === 'Behind slower trains');
  check(!behind1, `"Behind slower trains" is gone from the express${behind1 ? ' (' + behind1[1] + ')' : ''}`);
  check(!after.buttons.some((b) => b.text.startsWith('Add passing tracks')), 'the button is gone');
  await shot(page, 'express-after-light.png');

  // the local after: its waits, and where
  await select(page, { kind: 'line', line: String(built.local) });
  await page.waitForFunction(() => [...document.querySelectorAll('#inspector ul.rows > li')].some((li) => li.textContent.startsWith('Waiting for faster trains')), undefined, { timeout: 20000 }).catch(() => {});
  const local = await readPanel(page);
  console.log('local after', JSON.stringify(local.rows));
  const waitRow = local.rows.find((r) => r[0] === 'Waiting for faster trains');
  const places = local.rows.filter((r) => /^At .+, for Express trains$/.test(r[0]));
  check(!!waitRow, `the local shows "Waiting for faster trains" (${waitRow?.[1]})`);
  check(places.length > 0, `and where: ${places.map((r) => r[0] + ' ' + r[1]).join('; ')}`);
  const wi = local.rows.indexOf(waitRow);
  check(wi >= 0 && local.rows.slice(wi + 1).every((r) => /, for Express trains$/.test(r[0])), 'only its own places sit under "Waiting for faster trains"');
  for (const sh of ['light', 'dark']) {
    await shade(page, sh);
    await shot(page, `local-after-${sh}.png`);
  }
  await shade(page, 'light');

  // the station panel's fourth upgrade
  await select(page, { kind: 'station', station: String(built.stops[2]) });
  await page.waitForTimeout(800);
  const up = await page.evaluate(() => { const b = document.querySelector('#inspector button[aria-label="Passing tracks"]'); return b && { tip: b.dataset.tip, pressed: b.getAttribute('aria-pressed') }; });
  console.log('station upgrade', JSON.stringify(up));
  check(!!up && up.pressed === 'true' && up.tip.startsWith('Passing tracks\nAllow faster trains to pass slower trains that are stopped here.\nCosts $'), 'the station panel shows passing tracks built, with its hover');
  await page.evaluate(() => { let s = document.querySelector('#inspector'); while (s && s.scrollHeight <= s.clientHeight + 1) s = s.parentElement; if (s) s.scrollTop = 0; });
  await page.locator('#inspector button[aria-label="Passing tracks"]').hover();
  await page.waitForTimeout(700);
  const box = await page.locator('#inspector').boundingBox();
  await page.screenshot({ path: join(OUT, 'station-light.png'), clip: { x: Math.max(0, box.x - 8), y: 0, width: Math.min(box.width + 300, 1400 - box.x + 8), height: 900 } });

  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
