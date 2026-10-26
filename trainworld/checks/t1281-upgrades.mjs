// T-1281: station upgrades on the real save. Rail trips a day as it loads, then with a waiting room
// at every station, then with a park and ride at every station as well; and screenshots of the
// station panel's upgrade row and of a terminal's sidings.
//   node trainworld/checks/t1281-upgrades.mjs [url]     (from the maps root; a running dev server)
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, NOISE, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6185/';
const UP = { terminal: 1, parkRide: 2, waitingRoom: 4 };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1281-', size: [1500, 950] });
const errors = [];
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  const rail = async (label) => {
    const before = await page.evaluate(() => window.twDemand.view()?.version ?? -1);
    await demandFinal(page, { polling: 500, lines: 100, after: label === 'as saved' ? null : before });
    const r = await page.evaluate(() => window.twDemand.view().railTripsPerDay);
    console.log(`${label}: ${Math.round(r).toLocaleString('en-US')} rail trips a day`);
    return r;
  };
  // one edit for every constructed station
  const upgradeAll = (bit) => page.evaluate(async (bit) => {
    const nodes = window.tw.world.value.save.stations.filter((s) => s.built).map((s) => s.num);
    const a = await window.tw.client.edit({ op: 'stationUpgrade', nodes, bit, on: true });
    return a.ok ? nodes.length : 0;
  }, bit);
  // SHOTS_ONLY=1: the screenshots alone
  if (!process.env.SHOTS_ONLY) {
    const base = await rail('as saved');
    console.log(`waiting rooms built: ${await upgradeAll(UP.waitingRoom)}`);
    const room = await rail('a waiting room at every station');
    console.log(`park and rides built: ${await upgradeAll(UP.parkRide)}`);
    const both = await rail('and a park and ride at every station');
    console.log(`change: waiting rooms ${(100 * (room / base - 1)).toFixed(1)}%, both ${(100 * (both / base - 1)).toFixed(1)}%`);
  }

  // a line's last stop with track on past it, else a stop along a line (the real save's lines end
  // at dead ends): give it a terminal and a park and ride, and look at it
  const st = await page.evaluate(() => {
    const w = window.tw.world.value, nodes = w.nodes, byId = new Map(w.save.stations.map((x) => [x.id, x]));
    const through = (id) => { const s = byId.get(id); return s && (nodes.get(s.num)?.ports ?? 0) >= 2 ? s : null; };
    const pick = (s) => ({ num: s.num, id: s.id, lng: s.lng, lat: s.lat, name: s.name, heading: s.heading, length: s.length });
    for (const l of w.save.lines) { const s = through(l.stops[l.stops.length - 1]); if (s) return pick(s); }
    for (const l of w.save.lines) for (const id of l.stops.slice(1, -1)) { const s = through(id); if (s) return pick(s); }
    return null;
  });
  if (!st) throw new Error('no through station on a line');
  if (process.env.SHOTS_ONLY) await page.evaluate((num) => window.tw.client.edit({ op: 'stationUpgrade', nodes: [num], bit: 2, on: true }), st.num);
  const t = await page.evaluate((num) => window.tw.client.edit({ op: 'stationUpgrade', nodes: [num], bit: 1, on: true }), st.num);
  console.log(`terminal at ${st.name}: ${t.ok ? 'built' : 'refused'}`);
  await page.evaluate((s) => { window.tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 17.2 }); window.tw.select({ kind: 'station', station: s.id }); }, st);
  await page.waitForTimeout(1000);
  // in a complex, its own line's row opens it alone (station view, where the whole row shows)
  const own = await page.$('.complex-stations li.on');
  if (own) await own.click();
  await demandFinal(page, { timeout: 150000 }).catch(() => {});
  await page.waitForTimeout(3000);
  const row = await page.$('.upgrades');
  const shot = join(ARTIFACTS, 't1281-upgrades.png');
  await page.screenshot({ path: shot });
  if (row) await (await row.$('button')).hover();
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(ARTIFACTS, 't1281-upgrades-hover.png') });
  console.log(`row shown: ${!!row}; screenshots in ${ARTIFACTS}`);
  // the sidings close in, past the platform's north end
  const tb = await page.evaluate((s) => {
    window.tw.select(null);
    const d = s.length / 2 + 100, lat = s.lat + Math.sin(s.heading) * d / 111320, lng = s.lng + Math.cos(s.heading) * d / (111320 * Math.cos(s.lat * Math.PI / 180));
    window.tw.map.jumpTo({ center: [lng, lat], zoom: 17.6 });
    return window.tw.renderer?.turnback?.count ?? null;
  }, st);
  await page.waitForTimeout(2500);
  await page.screenshot({ path: join(ARTIFACTS, 't1281-terminal.png') });
  console.log(`turn-back strokes drawn: ${tb}`);
  // and with the terminal taken away again, to compare
  await page.evaluate((num) => window.tw.client.edit({ op: 'stationUpgrade', nodes: [num], bit: 1, on: false }), st.num);
  await page.waitForTimeout(2500);
  await page.screenshot({ path: join(ARTIFACTS, 't1281-terminal-off.png') });
  console.log(errors.length ? `console errors:\n${errors.join('\n')}` : 'no console errors');
} finally {
  await close();
}
