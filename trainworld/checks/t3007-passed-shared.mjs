// T-3007: "Add N passed stations" includes stations another line stops at. A sandbox track of 7
// stations; a local stops at all seven, an express at the first and last. In the express's stop editor
// (related lines on, the default) the passed button counts all five middle stations, one click adds
// them, and an undo takes them off again. From the maps root:
//   node trainworld/checks/t3007-passed-shared.mjs [url]
import { launchChrome, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6510/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const LAT = 40.3, LNG0 = -79.6, N = 7, STEP_KM = 30, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180));
const PTS = Array.from({ length: N }, (_, k) => [LNG0 + k * STEP_KM * KM, LAT]);

const chrome = await launchChrome({ prefix: 'tw-t3007-', size: [1400, 900] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(1000);
  const built = await page.evaluate(async (PTS) => {
    const tw = window.tw, c = tw.client;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = PTS.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const log = [];
    log.push((await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: 0 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: 0 }, pis: [], single: false, hsr: true, design: 320 }))?.ok);
    for (let k = 2; k < xy.length; k++) {
      await pause(250);
      log.push((await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[k - 1]) }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: 0 }, pis: [], single: false, hsr: true, design: 320 }))?.ok);
    }
    await pause(500);
    for (const [k, p] of xy.entries()) { log.push((await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 400, name: 'ABCDEFG'[k] }))?.ok); await pause(150); }
    log.push((await c.edit({ op: 'constructAll' }))?.ok);
    await pause(600);
    const stops = xy.map(near);
    const local = (await c.edit({ op: 'addLine', stops, name: 'Local', colour: '#0a7d3b' })).line;
    const express = (await c.edit({ op: 'copyLine', line: local })).line;
    await c.edit({ op: 'setStops', line: express, stops: [stops[0], stops[6]] });
    await c.edit({ op: 'lineLook', line: express, name: 'Express', colour: '#d9467a', letter: 'X' });
    await pause(2000);
    return { log, stops, local, express };
  }, PTS);
  check(built.log.every(Boolean), 'the test line was built');
  const stopCount = (num) => page.evaluate((n) => window.tw.world.value.save.lines.find((l) => l.num === n)?.stops.length, num);
  check((await stopCount(built.express)) === 2 && (await stopCount(built.local)) === 7, 'express has 2 stops, local 7');

  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 8.5 }), PTS[3]);
  await page.evaluate((l) => window.tw.select({ kind: 'line', line: String(l) }), built.express);
  await page.locator('#inspector button:has-text("Edit stops")').first().click();
  const passedBtn = page.locator('#inspector li.passed button');
  await passedBtn.first().waitFor({ timeout: 20000 }).catch(() => {});
  const label = (await passedBtn.first().textContent().catch(() => '')) ?? '';
  console.log('button:', label.trim());
  check(/5 stations passed/.test(label), `the button counts all five middle stations, shared ones included ("${label.trim()}")`);
  await passedBtn.first().click();
  const addBtn = page.locator('#inspector li.passed.confirm button').first();
  await addBtn.waitFor({ timeout: 10000 });
  check(/^Add 5 stations$/.test((await addBtn.textContent()).trim()), 'its Add button reads "Add 5 stations"');
  await addBtn.click();
  await page.waitForFunction(([n]) => window.tw.world.value.save.lines.find((l) => l.num === n)?.stops.length === 7, [built.express], { timeout: 20000 }).catch(() => {});
  check((await stopCount(built.express)) === 7, `the express now stops at all seven (${await stopCount(built.express)})`);
  await page.evaluate(async () => (await import('/src/game/world.ts')).undo());
  await page.waitForTimeout(1500);
  check((await stopCount(built.express)) === 2, `undo brings it back to two stops (${await stopCount(built.express)})`);

  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
