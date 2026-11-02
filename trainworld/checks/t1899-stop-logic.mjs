// T-1899: the line tool's stop logic in a fresh city, clicked as a player clicks the stop markers.
//   node trainworld/checks/t1899-stop-logic.mjs [url]     (from the maps root; default http://localhost:6120/)
// The city is ../artifacts/trainworld/t1899/fresh-city.save (`node trainworld/checks/t1899-stop-orders.mjs
// build` writes it): a 14 km main line with a way round
// 2.4 times as long through three towns, a branch, a 60 km high-speed line off its west end, and a
// station of its own 80 m from Centre. Checks, with a picture each:
//  1. clicks along the way round in an order that reversed before give a straight line along it;
//  2. a new line's first stops on the high-speed line give it high-speed trains, and taking the
//     high-speed stop off again gives it ordinary ones; a line edited later keeps its type;
//  3. on a new line a click where Centre (served) and Centre Market (no trains) overlap takes Centre Market.
import { readFile } from 'node:fs/promises';
import { mkdirSync } from 'node:fs';
import { ARTIFACTS, NOISE, launchChrome, collectErrors, importSave } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6120/';
const DIR = ARTIFACTS + 't1899/';
mkdirSync(DIR, { recursive: true });
const results = [];
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};

const { ctx, close } = await launchChrome({ prefix: 'tw-t1899-' });
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(DIR + 'fresh-city.save'));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.stations.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(1500);

  const S = await page.evaluate(() => Object.fromEntries(window.tw.world.value.save.stations.map((s) => [s.name, { id: s.id, num: s.num, lng: s.lng, lat: s.lat }])));
  const line = (num) => page.evaluate((num) => {
    const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num);
    const st = l && w.lineStats[l.id];
    const name = new Map(w.save.stations.map((s) => [s.id, s.name]));
    return l && { stops: l.stops.map((s) => name.get(s)), fast: st.fast, reversals: st.reversals.length + st.endTurns.length };
  }, num);
  const view = (names, zoom) => page.evaluate(([pts, zoom]) => {
    const lng = pts.reduce((s, p) => s + p.lng, 0) / pts.length, lat = pts.reduce((s, p) => s + p.lat, 0) / pts.length;
    window.tw.map.jumpTo({ center: [lng, lat], zoom });
  }, [names.map((n) => S[n]), zoom]);
  /** a new line by the N key, its number */
  const newLine = async () => {
    await page.evaluate(() => { document.activeElement?.blur?.(); window.tw.tools.exit(); window.tw.selection.value = null; });
    const before = await page.evaluate(() => Math.max(-1, ...window.tw.world.value.save.lines.map((l) => l.num)));
    await page.keyboard.press('n');
    await page.waitForFunction((b) => window.tw.world.value.save.lines.some((l) => l.num > b), before, { timeout: 30000 });
    await page.evaluate(() => document.activeElement?.blur?.());
    await page.waitForTimeout(800);
    return page.evaluate((b) => Math.max(...window.tw.world.value.save.lines.filter((l) => l.num > b).map((l) => l.num)), before);
  };
  /** a click on a station's marker, as the player clicks it; waits for the stops to change */
  const click = async (num, name, at = S[name]) => {
    const before = JSON.stringify((await line(num)).stops);
    const xy = await page.evaluate((p) => { const r = window.tw.map.getContainer().getBoundingClientRect(), q = window.tw.map.project([p.lng, p.lat]); return [r.left + q.x, r.top + q.y]; }, at);
    await page.mouse.move(xy[0], xy[1]);
    await page.mouse.click(xy[0], xy[1]);
    for (let k = 0; k < 40; k++) {
      await page.waitForTimeout(250);
      if (JSON.stringify((await line(num)).stops) !== before) break;
    }
    await page.waitForTimeout(600); // the markers' tiers from the worker
  };
  const shot = async (file) => { await page.waitForTimeout(800); await page.screenshot({ path: DIR + file }); };

  // the main line, running, so Centre has trains
  const main = await page.evaluate(async (ids) => {
    const r = await window.tw.client.edit({ op: 'addLine', stops: ids, name: 'Main', colour: '#0a7d3b' });
    await window.tw.client.edit({ op: 'setTrainCount', line: r.line, trains: [4, 4, 2] });
    return r.line;
  }, ['Westgate', 'Mill Lane', 'Centre', 'Eastfield', 'Station 14'].map((n) => S[n].num));
  await page.waitForTimeout(1000);
  const ms = await page.evaluate((n) => { const w = window.tw.world.value, st = w.lineStats[w.save.lines.find((l) => l.num === n).id]; return { status: st.status, trains: st.trainsNeeded }; }, main);
  report('main line runs with trains', ms.status === 'running' && ms.trains[0] > 0, JSON.stringify(ms));

  // 1. the way round, clicked ends first
  const order = ['Westgate', 'Station 14', 'Mill Lane', 'Northby', 'Upton', 'Hillcrest', 'Eastfield'];
  await view(['Westgate', 'Station 14', 'Upton'], 11.6);
  const round = await newLine();
  for (const n of order) await click(round, n);
  const r1 = await line(round);
  const want = ['Westgate', 'Mill Lane', 'Northby', 'Upton', 'Hillcrest', 'Eastfield', 'Station 14'];
  report('way round runs straight', r1.reversals === 0 && (r1.stops.join() === want.join() || r1.stops.join() === want.slice().reverse().join()), `(${r1.stops.join(' > ')}, ${r1.reversals} reversals)`);
  await shot('way-round.png');

  // 2. high speed on a new line, and back
  await view(['Lakeport', 'Westgate'], 9.6);
  const hs = await newLine();
  await click(hs, 'Ridgeway');
  await click(hs, 'Westgate');
  const h1 = await line(hs);
  report('new line on high-speed track gets high-speed trains', h1.fast, `(${h1.stops.join(' > ')}, ${h1.fast ? 'high-speed' : 'ordinary'})`);
  await shot('high-speed-new.png');
  await view(['Westgate', 'Far End'], 11.2);
  await click(hs, 'Far End');
  const h2 = await line(hs);
  report('still mostly high-speed', h2.fast, `(${h2.stops.join(' > ')})`);
  await view(['Lakeport', 'Westgate'], 9.6);
  await click(hs, 'Ridgeway');
  const h3 = await line(hs);
  report('high-speed stop off again: ordinary trains', !h3.fast, `(${h3.stops.join(' > ')}, ${h3.fast ? 'high-speed' : 'ordinary'})`);
  await shot('high-speed-back.png');
  // an established line keeps its type: stop editing ends, then opens again
  await page.evaluate(() => window.tw.tools.exit());
  await page.waitForTimeout(500);
  await page.evaluate((n) => { const l = window.tw.world.value.save.lines.find((x) => x.num === n); window.tw.selection.value = { kind: 'line', line: l.id }; document.activeElement?.blur?.(); }, hs);
  await page.keyboard.press('s');
  await page.waitForTimeout(800);
  await click(hs, 'Ridgeway');
  const h4 = await line(hs);
  report('established line keeps ordinary trains on high-speed track', !h4.fast && h4.stops.includes('Ridgeway'), `(${h4.stops.join(' > ')}, ${h4.fast ? 'high-speed' : 'ordinary'})`);

  // 3. overlapping markers: the station with no trains
  await view(['Centre'], 12.4);
  const tie = await newLine();
  await click(tie, 'Centre');
  const t1 = await line(tie);
  report('new line: a click where Centre and Centre Market overlap takes Centre Market', t1.stops.join() === 'Centre Market', `(${t1.stops.join(' > ')})`);
  await shot('overlap-pick.png');
} finally {
  await close();
}
report('no console or page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
process.exit(results.every(Boolean) ? 0 : 1);
