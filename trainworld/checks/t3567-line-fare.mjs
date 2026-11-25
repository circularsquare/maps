// T-3567, T-4253: a line's own fare multiple, set through its panel. On the real save a busy rail line's panel shows a
// Fare multiple row at 1.00x, right above the revenue and expenses chart (T-3566, the panel's last section).
// Typed 1.2 it reads 1.20x and, once demand's final is in, the line's fare a ride is higher and its riders fewer;
// 15 is held to 10.00x, 0 reads 0.00x with the fare a ride below the start's (none at all since T-3578); the undo button puts it back to
// 1.00x and every figure is as it started (rail trips, each line's riders and revenue, to the bit). At 10.00x
// again the game is exported and opened again: the row still reads 10.00x and the save's fares carry it. Raising the
// base fare to US$250 allows 4.00x ($1,000) and refuses higher multiples with the panel message. Shots of
// the panel's bottom, light and dark, in artifacts/trainworld/. From the maps root (or a tree's root):
//   node trainworld/checks/t3567-line-fare.mjs [url] [line number]   (about six minutes; default localhost:6144, line 14)
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, importSave } from './lib.mjs';
import { openGame } from './lib/play.mjs';

const QUICK = process.argv.includes('--quick');
const args = process.argv.slice(2).filter((a) => a !== '--quick');
const BASE = args[0] ?? 'http://localhost:6144/';
const NUM = Number(args[1] ?? (QUICK ? 0 : 14));
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const g = await openGame(QUICK
  ? { url: BASE, city: 'nyc', mode: 'sandbox', prefix: 'tw-t3567-', ui: { shade: 'light', v: 15 } }
  : { url: BASE, save: REAL_SAVE, saveVia: 'post', prefix: 'tw-t3567-', ui: { shade: 'light', v: 15 } });
const page = g.page;
/** the network's and line NUM's figures: rail trips a day, each line's riders and fare revenue a day (US$M) */
const figures = () => page.evaluate(async (num) => {
  const money = await import('/src/game/money.ts'), periods = await import('/src/game/periods.ts');
  const hours = periods.PERIODS.map((p) => p.to - p.from);
  const v = window.twDemand.view(), w = window.tw.world.value, inc = money.incomeByDay.value[0];
  const day = (m) => (m ? m.perHour.reduce((a, x, p) => a + x * hours[p], 0) : null);
  const lines = {};
  for (const l of w.save.lines) lines[l.num] = [v.lines.get(l.id)?.ridersPerDay ?? null, day(inc?.lines.get(l.id))];
  const [riders, revenue] = lines[num];
  return { version: v.version, rail: v.railTripsPerDay, lines, riders, revenue, perRide: revenue * 1e6 / 100 / riders, own: money.moneyState.value?.fares?.lines ?? null };
}, NUM);
const row = async () => (await g.see.panel())?.rows['Fare multiple'] ?? null;
/** type into the row's value, then wait for demand's final on the new fares */
const typeFare = async (text) => {
  const v = await page.evaluate(() => window.twDemand.view().version);
  await g.act.type('Type fare multiple', text, { within: '#inspector' });
  await g.waitDemand({ after: v });
  await page.waitForTimeout(2000);
};
/** the panel scrolled to its bottom (the fare row and the chart), shot */
const shot = async (file) => {
  await page.evaluate(() => { const el = document.querySelector('#inspector svg.line-money')?.nextElementSibling; el?.scrollIntoView({ block: 'end' }); });
  await page.waitForTimeout(500);
  return g.shot(file, { element: '#inspector' });
};
const fmt = (f) => `${Math.round(f.riders).toLocaleString('en-US')} riders, $${f.revenue.toFixed(2)}M a day, $${f.perRide.toFixed(3)} a ride`;
try {
  if (QUICK) {
    // Fast T-4253 check on a new NY game: build one line, test 10x save/load and the US$1,000 base fare refusal
    const lineNum = await page.evaluate(async () => {
      const a = window.tw.lngLatToPack(-73.99, 40.75), b = window.tw.lngLatToPack(-73.97, 40.75);
      const r = await window.tw.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: -1 }, to: { kind: 'free', x: b.x, y: b.y, level: -1 }, pis: [], single: false });
      if (!r.ok) throw Error('route: ' + JSON.stringify(r));
      const nodes = window.tw.world.value.save.stations;
      const edge = window.tw.world.value.edges[0];
      const s1 = await window.tw.client.edit({ op: 'addStation', at: { kind: 'node', node: edge.a }, platform: 160, name: 'West' });
      const s2 = await window.tw.client.edit({ op: 'addStation', at: { kind: 'node', node: edge.b }, platform: 160, name: 'East' });
      if (!s1.ok || !s2.ok) throw Error('stations: ' + JSON.stringify([s1, s2]));
      await window.tw.client.edit({ op: 'constructAll' });
      const ln = await window.tw.client.edit({ op: 'addLine', stops: [edge.a, edge.b], name: 'Fare Test', colour: '#2563eb' });
      if (!ln.ok) throw Error('addLine: ' + JSON.stringify(ln));
      await window.tw.client.edit({ op: 'setTrainCount', line: ln.line, trains: [4, 3, 2] });
      return ln.line;
    });
    await g.act.openLine('Fare Test');
    check((await row()) === '1.00x', `line ${lineNum} starts at ${await row()}`);

    await g.act.type('Type fare multiple', '15', { within: '#inspector' });
    await page.waitForFunction((num) => window.tw.moneyState.value?.fares?.lines?.[String(num)] === 10, lineNum, { timeout: 10000 });
    check((await row()) === '10.00x', `typed 15: held to ${await row()}`);

    const dl = page.waitForEvent('download', { timeout: 30000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
    const bytes = await readFile(await (await dl).path());
    const saved = await page.evaluate(async (data) => { const s = await import('/src/workers/saveFile.ts'); return (await s.decodeSave(Uint8Array.from(data))).header.fares.lines ?? null; }, Array.from(bytes));
    check(saved?.[lineNum] === 10, `exported save carries 10x: ${JSON.stringify(saved)}`);
    await importSave(page, bytes);
    await page.waitForFunction((num) => window.tw.client.lastLoad?.from === 'file' && window.tw.moneyState.value?.fares?.lines?.[String(num)] === 10, lineNum, { timeout: 30000 });
    await g.act.openLine('Fare Test');
    check((await row()) === '10.00x', `after save and load: the row still reads ${await row()}`);

    // Raise base fare to US$250 in NYC: 4x ($1,000) allowed, 4.05x and 5x refused with message
    await page.evaluate(async () => {
      const f = window.tw.moneyState.value.fares;
      const d = await import('/src/game/fareDefault.ts');
      const sc = d.baseFareIn(f, 'nyc') / (f.nodes?.[0]?.[1] ?? f.base);
      const b = 250 / (sc || 1);
      await window.tw.client.edit({ op: 'fares', nodes: [[0, b], [20, b * 1.2], [100, b * 2]] });
    });
    await page.waitForFunction(async () => {
      const d = await import('/src/game/fareDefault.ts');
      return Math.abs(d.baseFareIn(window.tw.moneyState.value.fares, 'nyc') - 250) < 1;
    }, undefined, { timeout: 10000 });
    // Stepping down from 10x (over cap after base raise) still works
    await g.act.click('Less fare multiple', { within: '#inspector' });
    await page.waitForFunction((num) => window.tw.moneyState.value?.fares?.lines?.[String(num)] === 9.95, lineNum, { timeout: 10000 });
    check((await row()) === '9.95x', `stepping down from over-cap multiple works (${await row()})`);
    await g.act.click('Back to 1.00x', { within: '#inspector' });
    await page.waitForFunction((num) => window.tw.moneyState.value?.fares?.lines?.[String(num)] === undefined, lineNum, { timeout: 10000 });
    await g.act.type('Type fare multiple', '4', { within: '#inspector' });
    await page.waitForFunction((num) => window.tw.moneyState.value?.fares?.lines?.[String(num)] === 4, lineNum, { timeout: 10000 });
    check((await row()) === '4.00x', `at $250 base, 4.00x ($1,000) is allowed (${await row()})`);
    await g.act.click('More fare multiple', { within: '#inspector' });
    const capMsg = await page.locator('#inspector .line-fare-cap').textContent();
    check(capMsg === 'Base fare can be at most $1,000' && (await row()) === '4.00x', `stepping over $1,000 refused: "${capMsg}", row ${await row()}`);
    await g.act.type('Type fare multiple', '5', { within: '#inspector' });
    const capMsg2 = await page.locator('#inspector .line-fare-cap').textContent();
    check(capMsg2 === 'Base fare can be at most $1,000' && (await row()) === '4.00x', `typing 5 over $1,000 refused: "${capMsg2}", row ${await row()}`);
    const workerRefused = await page.evaluate(async (num) => {
      const r = await window.tw.client.edit({ op: 'fareLine', line: num, mul: 5 });
      return !r.ok && window.tw.moneyState.value?.fares?.lines?.[String(num)] === 4;
    }, lineNum);
    check(workerRefused, 'clock worker also refuses fareLine over $1,000');
    const errs = g.pageErrors().filter((e) => !/WebGL|GPU|ResizeObserver/.test(e));
    check(errs.length === 0, `no page errors${errs.length ? ': ' + errs.slice(0, 3).join(' | ') : ''}`);
  } else {
  await g.waitDemand({ lines: 50 });
  // the save's line days are older than its clock: a few days run on (the paused clock moved, which settles the
  // hours on the way) give the chart some
  const days = async () => page.evaluate((num) => (window.tw.moneyState.value?.lineDays?.[String(num)]?.length ?? 0) / 3, NUM);
  if ((await days()) < 4) {
    const t0 = await page.evaluate(() => window.tw.clock.now());
    for (let k = 1; k <= 5; k++) { await page.evaluate((t) => window.tw.clock.set(t), t0 + k * 86400); await page.waitForTimeout(2500); }
    await page.waitForFunction((num) => (window.tw.moneyState.value?.lineDays?.[String(num)]?.length ?? 0) >= 12, NUM, { timeout: 60000, polling: 500 }).catch(() => {});
    await g.waitDemand({ lines: 50 });
  }
  console.log(`  line ${NUM} has ${await days()} days kept`);
  await page.waitForTimeout(2000);
  const name = await page.evaluate((num) => window.tw.world.value.save.lines.find((l) => l.num === num)?.name, NUM);
  await g.act.openLine(name);
  check((await row()) === '1.00x', `line ${NUM} (${name}): the row reads ${await row()}`);
  // the row, then the chart's heading, the chart and its dates, and nothing after
  const order = await page.evaluate(() => {
    const ul = [...document.querySelectorAll('#inspector ul.rows')].find((u) => u.textContent.includes('Fare multiple'));
    const out = [];
    for (let e = ul; e; e = e.nextElementSibling) out.push(e.tagName.toLowerCase() + '.' + [...e.classList].join('.'));
    return out.join(' ');
  });
  check(order === 'ul.rows div.h svg.hist.line-money div.hist-x.small.muted', `the fare row sits right above the chart, at the panel's bottom (${order})`);
  const f0 = await figures();
  console.log(`  at 1.00x: ${fmt(f0)}`);

  await typeFare('1.2');
  const f12 = await figures();
  check((await row()) === '1.20x' && f12.own?.[NUM] === 1.2, `typed 1.2: the row reads ${await row()}, saved fares ${JSON.stringify(f12.own)}`);
  check(f12.perRide > f0.perRide && f12.riders < f0.riders, `at 1.20x: ${fmt(f12)}`);
  await shot('t3567_line_light.png');

  await typeFare('15');
  const f10 = await figures();
  check((await row()) === '10.00x' && f10.own?.[NUM] === 10, `typed 15: held to ${await row()}`);
  check(f10.riders < f12.riders && (f10.riders === 0 || f10.perRide > f12.perRide), `at 10.00x: ${fmt(f10)}`);

  await typeFare('0');
  const fz = await figures();
  check((await row()) === '0.00x', `typed 0: the row reads ${await row()}`);
  check(!(fz.perRide >= f0.perRide) && fz.riders > f0.riders && fz.revenue >= 0, `at 0.00x (T-3578: a free line takes no fare, base included): ${fmt(fz)}`);

  const v = await page.evaluate(() => window.twDemand.view().version);
  await g.act.click('Back to 1.00x', { within: '#inspector' });
  await g.waitDemand({ after: v });
  await page.waitForTimeout(2000);
  const f1 = await figures();
  check((await row()) === '1.00x' && f1.own === null, `the undo button: back to ${await row()}, no multiples kept (${JSON.stringify(f1.own)})`);
  check(f1.rail === f0.rail && JSON.stringify(f1.lines) === JSON.stringify(f0.lines), `back at 1.00x every figure is as it started (rail ${f1.rail} vs ${f0.rail})`);

  // save and open again at 10.00x (T-4253)
  await typeFare('10');
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());
  const saved = await page.evaluate(async (data) => { const s = await import('/src/workers/saveFile.ts'); return (await s.decodeSave(Uint8Array.from(data))).header.fares.lines ?? null; }, Array.from(bytes));
  check(saved?.[NUM] === 10, `the exported save's fares carry it: ${JSON.stringify(saved)}`);
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 50, undefined, { timeout: 300000 });
  await g.waitDemand({ lines: 50 });
  await page.waitForTimeout(2000);
  await g.act.openLine(name);
  const back = await figures();
  check((await row()) === '10.00x' && back.own?.[NUM] === 10, `opened again: the row reads ${await row()}`);
  check(Math.abs(back.revenue - f10.revenue) <= 1e-6 * Math.max(1, f10.revenue), `opened again: ${fmt(back)}`);

  // T-4253: a multiple whose base fare times it is over US$1,000 is refused with a message
  await page.evaluate(async () => {
    const f = window.tw.moneyState.value.fares;
    const d = await import('/src/game/fareDefault.ts');
    const sc = d.baseFareIn(f, 'nyc') / (f.nodes?.[0]?.[1] ?? f.base);
    const b = 250 / (sc || 1);
    await window.tw.client.edit({ op: 'fares', nodes: [[0, b], [20, b * 1.2], [100, b * 2]] });
  });
  await page.waitForFunction(async () => {
    const d = await import('/src/game/fareDefault.ts');
    return Math.abs(d.baseFareIn(window.tw.moneyState.value.fares, 'nyc') - 250) < 1;
  }, undefined, { timeout: 10000 });
  await g.act.click('Back to 1.00x', { within: '#inspector' });
  await page.waitForFunction((num) => window.tw.moneyState.value?.fares?.lines?.[String(num)] === undefined, NUM, { timeout: 10000 });
  await g.act.type('Type fare multiple', '4', { within: '#inspector' });
  await page.waitForFunction((num) => window.tw.moneyState.value?.fares?.lines?.[String(num)] === 4, NUM, { timeout: 10000 });
  check((await row()) === '4.00x', `at $250 base, 4.00x ($1,000) is allowed (${await row()})`);
  await g.act.click('More fare multiple', { within: '#inspector' });
  const capMsg = await page.locator('#inspector .line-fare-cap').textContent();
  check(capMsg === 'Base fare can be at most $1,000' && (await row()) === '4.00x', `stepping over $1,000 refused: "${capMsg}", row ${await row()}`);
  await g.act.type('Type fare multiple', '5', { within: '#inspector' });
  const capMsg2 = await page.locator('#inspector .line-fare-cap').textContent();
  check(capMsg2 === 'Base fare can be at most $1,000' && (await row()) === '4.00x', `typing 5 over $1,000 refused: "${capMsg2}", row ${await row()}`);
  const workerRefused = await page.evaluate(async (num) => {
    const r = await window.tw.client.edit({ op: 'fareLine', line: num, mul: 5 });
    return !r.ok && window.tw.moneyState.value?.fares?.lines?.[String(num)] === 4;
  }, NUM);
  check(workerRefused, 'clock worker also refuses fareLine over $1,000');

  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await page.waitForTimeout(600);
  await shot('t3567_line_dark.png');
  const errs = g.pageErrors().filter((e) => !/WebGL|GPU|ResizeObserver/.test(e));
  check(errs.length === 0, `no page errors${errs.length ? ': ' + errs.slice(0, 3).join(' | ') : ''}`);
  console.log(`  shots: ${join(ARTIFACTS, 't3567_line_light.png')}, t3567_line_dark.png`);
  }
} catch (e) {
  check(false, `ran to the end: ${e.stack ?? e.message}`);
} finally {
  await g.close();
}
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
