// T-2220: changes inside a station building count 0.8 (fixed minutes and walk). On the real New York save:
// `baseline` records demand's weekday figures with no building (run it on the code before a demand change);
// `check` loads the save again, compares with the baseline to the bit (a building-free save solves as
// before), then builds the rectangle the upgrade row offers round Times Sq-42 St (as t2218 does) and checks
// that no line's riders move more than 5% and the building's stations gain transfers. Writes
// artifacts/trainworld/t2220/<mode>.json. Uses the Vite already on the url (dev-nowatch.mjs). From the maps root:
//   node trainworld/checks/t2220-building-change.mjs <url> baseline|check
import { ARTIFACTS, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const OUT = join(ARTIFACTS, 't2220');
await mkdir(OUT, { recursive: true });
const [BASE = 'http://localhost:6196/', MODE = 'check'] = process.argv.slice(2);
/** lines with fewer riders a day are reported but not held to the 5% */
const MIN_RIDERS = 1000;
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

/** demand's weekday figures as the view has them */
const figures = (page) => page.evaluate(() => {
  const w = window.tw.world.value, v = window.twDemand.view();
  const lines = {}, stations = {};
  for (const l of w.save.lines) { const d = v.lines.get(l.id); if (d) lines[l.id] = { name: l.name, riders: d.ridersPerDay }; }
  for (const s of w.save.stations) { const d = v.stations.get(s.id); if (d) stations[s.id] = { name: s.name, boardings: d.boardings, transfers: d.transfers }; }
  return { version: v.version, railTripsPerDay: v.railTripsPerDay, tripsPerDay: v.tripsPerDay, lines, stations };
});

const chrome = await launchChrome({ prefix: 'tw-t2220-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const out = { mode: MODE, at: new Date().toISOString() };
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.building && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { timeout: 900000, polling: 1000 });
  const before = await figures(page);
  out.before = before;
  if (MODE === 'baseline') {
    check(Object.keys(before.lines).length > 100, 'baseline recorded', `${Object.keys(before.lines).length} lines, ${Math.round(before.railTripsPerDay)} rail trips/day`);
  } else {
    // ---- no building: as the baseline, to the bit
    const base = JSON.parse(await readFile(join(OUT, 'baseline.json'), 'utf8')).before;
    const diff = [];
    for (const k of ['railTripsPerDay', 'tripsPerDay']) if (!Object.is(base[k], before[k])) diff.push(`${k} ${base[k]} -> ${before[k]}`);
    for (const [id, l] of Object.entries(base.lines)) if (!Object.is(l.riders, before.lines[id]?.riders)) diff.push(`line ${l.name} ${l.riders} -> ${before.lines[id]?.riders}`);
    for (const [id, s] of Object.entries(base.stations)) for (const k of ['boardings', 'transfers']) if (!Object.is(s[k], before.stations[id]?.[k])) diff.push(`station ${s.name} ${k} ${s[k]} -> ${before.stations[id]?.[k]}`);
    out.sameAsBaseline = diff;
    check(diff.length === 0, 'with no building, every line and station figure is the baseline\'s to the bit', diff.slice(0, 4).join('; ') || `${Object.keys(base.lines).length} lines, ${Object.keys(base.stations).length} stations`);

    // ---- the building the upgrade row offers round Times Sq-42 St
    const box = await (await page.$('#map')).boundingBox();
    const st = await page.evaluate(() => {
      const s = window.tw.world.value.save.stations.find((x) => x.name.startsWith('Times Sq') && x.built);
      return s && { id: s.id, name: s.name, lng: s.lng, lat: s.lat };
    });
    await page.evaluate((s) => { window.tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 15.6 }); window.tw.selection.value = { kind: 'station', station: s.id }; }, st);
    await page.waitForTimeout(1500);
    await page.click('button[aria-label="Station building"]');
    await page.waitForTimeout(500);
    await page.mouse.move(box.x + 30, box.y + 200);
    await page.waitForFunction(() => { const b = window.tw.tools.building, c = b.quoted(); return b.quote && c && b.quote.key === c.join(); }, undefined, { timeout: 30000 });
    const quote = await page.evaluate(() => ({ m2: window.tw.tools.building.quote.m2, members: window.tw.tools.building.quote.members }));
    await page.click('#tool-notes button:has-text("Construct")');
    await page.waitForFunction(() => (window.tw.world.value.save.buildings?.length ?? 0) === 1, undefined, { timeout: 30000 });
    await demandFinal(page, { timeout: 900000, polling: 1000, after: before.version });
    // the building's members as the panels have them (game/buildingView.ts, the dev server's module)
    const members = await page.evaluate(async () => {
      const m = await import('/src/game/buildingView.ts');
      return [...(m.buildingMembersById.peek().get(window.tw.world.value.save.buildings[0].id) ?? [])];
    });
    const after = await figures(page);
    out.after = after;
    out.building = { ...quote, members };
    const moves = [];
    for (const [id, l] of Object.entries(before.lines)) {
      const a = after.lines[id]?.riders ?? 0;
      moves.push({ id, name: l.name, before: l.riders, after: a, change: l.riders > 0 ? a / l.riders - 1 : (a > 0 ? Infinity : 0) });
    }
    moves.sort((x, y) => Math.abs(y.change) - Math.abs(x.change));
    const held = moves.filter((m) => m.before >= MIN_RIDERS);
    out.moves = moves;
    const top = held.slice(0, 5).map((m) => `${m.name} ${Math.round(m.before)} -> ${Math.round(m.after)} (${(100 * m.change).toFixed(2)}%)`).join('; ');
    check(held.length > 50 && Math.abs(held[0].change) <= 0.05, `no line with ${MIN_RIDERS}+ riders a day moves more than 5%`, top);
    const small = moves.filter((m) => m.before < MIN_RIDERS && Math.abs(m.change) > 0.05);
    console.log(`  lines under ${MIN_RIDERS} riders a day moving more than 5%: ${small.length}${small.length ? ' (' + small.slice(0, 5).map((m) => `${m.name} ${Math.round(m.before)} -> ${Math.round(m.after)}`).join('; ') + ')' : ''}`);
    const sum = (f, k) => members.reduce((s, id) => s + (f.stations[id]?.[k] ?? 0), 0);
    const tb = sum(before, 'transfers'), ta = sum(after, 'transfers');
    out.transfers = { before: tb, after: ta };
    check(ta > tb, 'the building\'s stations gain transfers', `${members.length} stations, ${Math.round(tb)} -> ${Math.round(ta)} transfers a day (${(100 * (ta / tb - 1)).toFixed(1)}%), ${Math.round(quote.m2)} m²`);
    console.log(`  rail trips a day ${Math.round(before.railTripsPerDay)} -> ${Math.round(after.railTripsPerDay)} (${(100 * (after.railTripsPerDay / before.railTripsPerDay - 1)).toFixed(3)}%)`);
  }
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
await writeFile(join(OUT, `${MODE}.json`), JSON.stringify(out, null, 1));
console.log(fail ? `T-2220 FAILED: ${fail}` : 'T-2220 PASSED');
process.exit(fail ? 1 : 0);
