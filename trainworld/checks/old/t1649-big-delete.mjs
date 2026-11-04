// T-1649: a delete of more than 50 km of track, or a tenth of the network's, asks first. On the real
// save the Select tool's box (zoomed out, so no stations are picked) plus Delete asks "Delete N km of
// built track?", Keep it changes nothing, and a small delete asks as before. On a new game a big
// blueprint delete asks, and one undo brings track, stations and the line back; a small one is quiet.
// Uses the Vite already on `port` (dev-nowatch.mjs). From the maps root:
//   node trainworld/checks/t1649-big-delete.mjs [port]
import { readFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave } from './lib.mjs';

const port = Number(process.argv[2] ?? 6151);
const SHOT = join(ARTIFACTS, 't1649-confirm.png');
await mkdir(ARTIFACTS, { recursive: true });
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t1649-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const ask = (page) => page.evaluate(async () => { const { ask } = await import('/src/game/ui.ts'); const a = ask.peek(); return a && { text: a.text, yes: a.yes, no: a.no }; });
const state = (page) => page.evaluate(() => { const w = window.tw.world.value; return { lines: w.save.lines.length, stations: w.save.stations.length, km: w.edges.reduce((s, e) => s + e.lengthM, 0) / 1000 }; });
const settle = (page, ms = 1500) => page.waitForTimeout(ms);
const open = (p) => p.goto(`http://localhost:${port}/?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`).then(() => p.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 }));
try {
  // ---- the real save
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await open(page);
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await settle(page);
  const s0 = await state(page);
  check(Math.abs(s0.km - 4498.6) < 1 && s0.lines === 109, 'real save loaded', JSON.stringify(s0));

  // small delete: one built piece asks as before, in the old words
  const small = await page.evaluate(async () => {
    const { bpPicked } = await import('/src/game/blueprintPick.ts');
    const { removePicked } = await import('/src/game/blueprintPick.ts');
    const e = window.tw.world.value.edges.filter((x) => x.built && !x.crossover && x.lengthM > 800).sort((a, b) => a.lengthM - b.lengthM)[0];
    bpPicked.value = { edges: [e.id], stations: [], built: true };
    await removePicked();
    return null;
  });
  const qs = await ask(page);
  check(qs && /^Remove [\d.]+ km of built track\? Nothing is refunded\.$/.test(qs.text) && qs.yes === 'Remove' && qs.no === 'Keep', 'a small built delete asks in the old words', JSON.stringify(qs));
  await page.evaluate(async () => { const { ask } = await import('/src/game/ui.ts'); ask.value = null; const { clearBpPicked } = await import('/src/game/blueprintPick.ts'); clearBpPicked(); });

  // the Select tool's box over the whole network, zoomed out so no station is picked
  await page.evaluate(async () => { const { tool } = await import('/src/game/ui.ts'); tool.value = 'blueprint'; window.tw.map.jumpTo({ center: [-74.0, 40.75], zoom: 7.6 }); });
  await page.waitForFunction(() => window.tw.map.loaded());
  await settle(page, 800);
  const box = await (await page.$('#map')).boundingBox();
  await page.mouse.move(box.x + 4, box.y + 4);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 6 });
  await page.mouse.move(box.x + box.width - 4, box.y + box.height - 4, { steps: 6 });
  await page.mouse.up();
  await settle(page, 600);
  const picked = await page.evaluate(async () => { const { pickedState } = await import('/src/game/blueprintPick.ts'); const p = pickedState.peek(); return { edges: p.edges, stations: p.stations, km: p.lengthM / 1000, built: p.built }; });
  check(picked.edges > 1000 && picked.stations === 0, 'the box picks the track and no stations', JSON.stringify(picked));
  await page.keyboard.press('Delete');
  await settle(page, 800);
  const q = await ask(page);
  check(q && /^Delete [\d,]+ km of built track\? Nothing is refunded\.$/.test(q.text) && q.yes === 'Delete track' && q.no === 'Keep it', 'Delete over the box asks with the amount', JSON.stringify(q));
  await page.screenshot({ path: SHOT });
  const rows = await page.evaluate(() => document.querySelectorAll('dialog.confirmation li').length);
  check(rows <= 6, 'the line list is capped', rows + ' rows');
  await page.click('dialog.confirmation button:has-text("Keep it")');
  await settle(page, 800);
  const sKeep = await state(page);
  const alive = await page.evaluate(async () => { const { lineEffects } = await import('/src/game/remove.ts'); return (await lineEffects([window.tw.world.value.edges[0].id], [])).length; });
  check(Math.abs(sKeep.km - s0.km) < 0.01 && sKeep.lines === 109 && sKeep.stations === s0.stations && alive > 0, 'Keep it changes nothing, in the model too', JSON.stringify(sKeep) + ' alive ' + alive);
  await page.keyboard.press('Delete');
  await settle(page, 500);
  check(!!(await ask(page)), 'Delete asks again');
  await page.click('dialog.confirmation button:has-text("Delete track")');
  await page.waitForFunction(() => window.tw.world.value.edges.length === 0, undefined, { timeout: 30000 }).catch(() => {});
  await settle(page, 500);
  const sGone = await state(page);
  check(sGone.km < 1 && sGone.stations === s0.stations, 'confirming deletes the track and leaves the stations', JSON.stringify(sGone));
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  await settle(page, 2000);
  const sUndo = await state(page);
  check(Math.abs(sUndo.km - s0.km) < 0.1 && sUndo.lines === 109 && sUndo.stations === s0.stations, 'one undo brings the built track back, 4498.6 km and 109 lines', JSON.stringify(sUndo));

  // ---- a blueprint, where undo works
  const g = await chrome.ctx.newPage();
  collectErrors(g, errors);
  await open(g);
  await g.evaluate(() => window.tw.map.jumpTo({ center: [-73.9, 40.75], zoom: 9 }));
  await g.waitForFunction(() => window.tw.map.loaded());
  const made = await g.evaluate(async () => {
    const tw = window.tw, c = tw.client, pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = (a, b) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y]; };
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const pts = [-74.6, -74.2, -73.8, -73.4, -73.0].map((lng) => xy(lng, 40.6));
    await c.edit({ op: 'route', from: { kind: 'free', x: pts[0][0], y: pts[0][1], level: -1 }, to: { kind: 'free', x: pts[1][0], y: pts[1][1], level: -1 }, pis: [], single: false });
    for (let k = 2; k < pts.length; k++) { await pause(300); await c.edit({ op: 'route', from: { kind: 'node', node: near(pts[k - 1]) }, to: { kind: 'free', x: pts[k][0], y: pts[k][1], level: -1 }, pis: [], single: false }); }
    await pause(500);
    for (const p of [pts[0], pts[4]]) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(200); }
    const l = await c.edit({ op: 'addLine', stops: [near(pts[0]), near(pts[4])], name: 'Check', colour: '#0a7d3b' });
    await pause(500);
    // a 2 km piece apart from it
    const a = xy(-73.9, 40.9), b = xy(-73.9, 40.918);
    await c.edit({ op: 'route', from: { kind: 'free', x: a[0], y: a[1], level: -1 }, to: { kind: 'free', x: b[0], y: b[1], level: -1 }, pis: [], single: false });
    await pause(500);
    return l?.ok ?? l;
  });
  await settle(g, 1000);
  const b0 = await state(g);
  check(b0.km > 100 && b0.lines === 1 && b0.stations === 2, 'a blueprint line and a spur were made', `${JSON.stringify(b0)} ${made}`);
  // small: the spur (shorter than 5 km) goes without a question
  const spur = await g.evaluate(async () => {
    const { bpPicked, removePicked } = await import('/src/game/blueprintPick.ts');
    const e = window.tw.world.value.edges.filter((x) => x.lengthM < 5000).sort((a, b) => a.lengthM - b.lengthM)[0];
    if (!e) return null;
    bpPicked.value = { edges: [e.id], stations: [], built: false };
    await removePicked();
    const { ask } = await import('/src/game/ui.ts');
    return { km: e.lengthM / 1000, asked: !!ask.peek() };
  });
  await settle(g, 1000);
  const b1 = await state(g);
  check(spur && !spur.asked && b1.km < b0.km - 1, 'a small blueprint delete does not ask', JSON.stringify(spur));
  await g.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  await settle(g, 1000);
  // big: everything
  await g.evaluate(async () => {
    const { bpPicked, removePicked } = await import('/src/game/blueprintPick.ts');
    const w = window.tw.world.value;
    bpPicked.value = { edges: w.edges.map((e) => e.id), stations: w.save.stations.map((s) => s.num) };
    void removePicked();
  });
  await settle(g, 800);
  const gq = await ask(g);
  check(gq && /^Delete [\d,.]+ km of track and 2 stations\?$/.test(gq.text) && gq.yes === 'Delete track', 'a big blueprint delete asks', JSON.stringify(gq));
  await g.evaluate(async () => { const { ask } = await import('/src/game/ui.ts'); ask.value = null; });
  const b2 = await state(g);
  check(Math.abs(b2.km - (await state(g)).km) < 0.01 && b2.lines === 1 && b2.km > 100, 'declining keeps everything', JSON.stringify(b2));
  await g.evaluate(async () => { const { removePicked } = await import('/src/game/blueprintPick.ts'); void removePicked(); });
  await settle(g, 600);
  await g.click('dialog.confirmation button:has-text("Delete track")');
  await settle(g, 2000);
  const b3 = await state(g);
  check(b3.km < 1 && b3.stations === 0, 'confirming deletes track and stations', JSON.stringify(b3));
  await g.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  await settle(g, 2000);
  const b4 = await state(g);
  check(Math.abs(b4.km - b2.km) < 0.01 && b4.lines === b2.lines && b4.stations === 2, 'one undo brings track, stations and the line back', JSON.stringify(b4));
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
