// T-2125: "what this edit did" at the map's upper left. On the real save and on a new New York game, a
// train count change, a stop change and a new line each show the note; its figures are printed beside
// the Lines tab's and the demand result's before and after, and screenshots go to
// artifacts/trainworld/t2125/ light and dark. From the maps root, with a running dev server:
//   node trainworld/checks/old/t2125-edit-note.mjs <port> [real|new|both]
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, GL, NOISE, REAL_SAVE, buildLine, collectErrors, demandFinal, importSave, launchChrome } from '../lib.mjs';

const OUT = join(ARTIFACTS, 't2125');
await mkdir(OUT, { recursive: true });
const BASE = `http://localhost:${process.argv[2] ?? 6168}/`;
const which = process.argv[3] ?? 'both';
const { ctx, close } = await launchChrome({ prefix: 'tw-t2125-', size: [1400, 900], args: GL, deviceScaleFactor: 2 });
const errors = [];
let fails = 0;
const check = (name, ok, detail = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };

/** what the panels and the demand result say now: the network's and one line's riders and money */
const figures = (page, id) => page.evaluate(async (id) => {
  const w = window.tw.world.value, dv = window.twDemand.view();
  const { today, runningToday } = await import('/src/game/today.ts');
  const { moneyState } = await import('/src/game/money.ts');
  const t = today.value, m = moneyState.value;
  let dvRiders = 0, tabRiders = 0, tabMoney = 0, shares = 0;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st || st.status !== 'running') continue;
    if (!st.bus) dvRiders += dv?.lines.get(l.id)?.ridersPerDay ?? 0;
    const d = t.get(l.id);
    if (d) { if (!st.bus) tabRiders += d.riders; tabMoney += d.fares - runningToday(l, st) - st.upkeep; }
    shares += st.upkeep;
  }
  // the network's money as the Lines tab's lines add up, plus the upkeep no line carries
  const netMoney = tabMoney + shares - (m.upkeep.track + m.upkeep.stations + m.upkeep.buses) * 1e6;
  const l = id && w.save.lines.find((x) => x.id === id), st = l && w.lineStats[id], d = l && t.get(id);
  return {
    dvRiders: Math.round(dvRiders), tabRiders: Math.round(tabRiders), netMoney: Math.round(netMoney),
    line: l ? { dv: Math.round(dv?.lines.get(id)?.ridersPerDay ?? 0), tab: d ? Math.round(d.riders) : null, money: d ? Math.round(d.fares - runningToday(l, st) - st.upkeep) : null } : null,
  };
}, id ?? null);

const noteNow = (page) => page.evaluate(async () => {
  const { editNote } = await import('/src/game/editNote.ts');
  const n = editNote.value, el = document.querySelector('.map-top-left .edit-note');
  return n ? { ...n, rows: el ? [...el.querySelectorAll('.edit-row')].map((r) => ({ text: r.textContent.trim(), tip: r.dataset.tip, cls: r.className, arrow: r.querySelector('.edit-arrow')?.getAttribute('class') })) : null } : null;
});
const waitNote = async (page, ms = 300000) => {
  const t = Date.now();
  while (Date.now() - t < ms) {
    const n = await noteNow(page);
    if (n) return { n, s: (Date.now() - t) / 1000 };
    await page.waitForTimeout(250);
  }
  return { n: null, s: ms / 1000 };
};
const shade = (page, s) => page.evaluate((x) => { document.documentElement.dataset.shade = x; window.tw.shade.value = x; }, s);
const fmt = (n) => Math.round(n).toLocaleString('en-US');

async function one(page, tag, what, id, doEdit) {
  const before = await figures(page, id);
  await doEdit();
  const { n, s } = await waitNote(page);
  check(`${tag} ${what}: a note shows`, !!n, `after ${s.toFixed(1)} s`);
  if (!n) return;
  await page.waitForTimeout(1200); // the count-up
  const after = await figures(page, id ?? n.line?.id);
  const rows = (await noteNow(page))?.rows ?? [];
  console.log(`      note: ${rows.map((r) => r.text).join(' | ')}`);
  console.log(`      hovers: ${rows.map((r) => JSON.stringify(r.tip)).join(' | ')}`);
  console.log(`      riders ${fmt(n.riders.from)} -> ${fmt(n.riders.to)}; demand result ${fmt(before.dvRiders)} -> ${fmt(after.dvRiders)}; Lines tab ${fmt(before.tabRiders)} -> ${fmt(after.tabRiders)}`);
  console.log(`      profit ${fmt(n.money.from)} -> ${fmt(n.money.to)} (${fmt(n.money.to - n.money.from)}); Lines tab sum ${fmt(before.netMoney)} -> ${fmt(after.netMoney)} (${fmt(after.netMoney - before.netMoney)})`);
  if (n.line) console.log(`      line ${n.line.id}: riders ${fmt(n.line.riders.from)} -> ${fmt(n.line.riders.to)} (tab ${before.line?.tab ?? '-'} -> ${after.line?.tab ?? '-'}), money ${fmt(n.line.money.from)} -> ${fmt(n.line.money.to)} (tab ${before.line?.money ?? '-'} -> ${after.line?.money ?? '-'})`);
  check(`${tag} ${what}: riders to = the demand result's`, Math.abs(n.riders.to - after.dvRiders) <= Math.max(1, 1e-5 * after.dvRiders), `${n.riders.to} vs ${after.dvRiders}`);
  check(`${tag} ${what}: texts are signed rates`, rows.length > 0 && rows.every((r) => /^[+−]([\d,]+|\d+(\.\d+)?[kM]) riders\/day$|^[+−]\$\d+(\.\d+)?[kMB]? profit\/day$/.test(r.text)), rows.map((r) => r.text).join(' | '));
  check(`${tag} ${what}: arrows follow the signs`, rows.every((r) => r.arrow?.includes(r.text.startsWith('+') ? 'up' : 'down')));
  check(`${tag} ${what}: a fall in profit is red`, rows.every((r) => !r.text.includes('profit') || r.cls.includes('money-neg') === r.text.startsWith('−')));
  const box = await page.locator('.map-top-left').boundingBox();
  for (const sh of ['light', 'dark']) {
    await shade(page, sh);
    await page.waitForTimeout(300);
    await page.screenshot({ path: join(OUT, `${tag}-${what}-${sh}.png`), clip: { x: Math.max(0, box.x - 10), y: Math.max(0, box.y - 10), width: 520, height: box.height + 20 } });
  }
  if (what === 'trains') await page.screenshot({ path: join(OUT, `${tag}-screen-light.png`) });
  await shade(page, 'light');
}

async function realSave() {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 20, undefined, { timeout: 120000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await demandFinal(page, { polling: 500, lines: 20, timeout: 600000 });
  await page.evaluate(async () => { const u = await import('/src/game/ui.ts'); u.tab.value = 'lines'; });
  await page.waitForTimeout(3000);
  check('real: no note before any edit', (await noteNow(page)) === null);
  const lines = await page.evaluate(() => {
    const w = window.tw.world.value, dv = window.twDemand.view();
    return w.save.lines.filter((l) => w.lineStats[l.id]?.status === 'running' && !w.lineStats[l.id].bus && l.stops.length > 5)
      .map((l) => ({ id: l.id, num: l.num, name: l.name, stops: l.stops, trains: w.lineStats[l.id].trainsNeeded, riders: dv.lines.get(l.id)?.ridersPerDay ?? 0 }))
      .sort((a, b) => b.riders - a.riders);
  });
  const a = lines[2], b = lines[5];
  console.log(`real: trains on ${a.name} (${a.trains.join('/')}), stops on ${b.name}, new line on ${lines[8].name}'s first stops`);
  await one(page, 'real', 'trains', a.id, () => page.evaluate(async ({ num, trains }) => (await import('/src/game/world.ts')).edit({ op: 'setTrainCount', line: num, trains: trains.map((t) => Math.max(1, Math.round(t / 2))) }), a));
  await one(page, 'real', 'stops', b.id, () => page.evaluate(async ({ num, stops }) => (await import('/src/game/world.ts')).edit({ op: 'setStops', line: num, stops: stops.slice(0, -2).map(Number) }), b));
  const c = lines[8];
  await one(page, 'real', 'newline', null, () => page.evaluate(async ({ stops }) => {
    const r = await (await import('/src/game/world.ts')).edit({ op: 'addLine', stops: stops.slice(0, 4).map(Number), name: 'Check', colour: '#0a7d3b' });
    if (r?.line !== undefined) await (await import('/src/game/world.ts')).edit({ op: 'setTrainCount', line: r.line, trains: [8, 6, 3] });
  }, c));
  // goes after NOTE_MS and on the next edit
  const t = Date.now();
  while ((await noteNow(page)) && Date.now() - t < 12000) await page.waitForTimeout(250);
  check('real: the note goes after about 8 s', (await noteNow(page)) === null, `${((Date.now() - t) / 1000).toFixed(1)} s after the screenshots`);
  await page.close();
}

async function newGame() {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => void window.tw.map.jumpTo({ center: [-73.982, 40.752], zoom: 12.5 }));
  await page.evaluate(async () => { const u = await import('/src/game/ui.ts'); u.tab.value = 'lines'; });
  await page.waitForTimeout(8000);
  const pts = [[-73.995, 40.728], [-73.988, 40.742], [-73.981, 40.756], [-73.974, 40.770], [-73.967, 40.784]];
  // lib's buildLine, through the game's edit() (the client's own bypasses the edit bookkeeping)
  await one(page, 'new', 'newline', null, () => page.evaluate(async (PTS) => {
    const tw = window.tw, { edit } = await import('/src/game/world.ts');
    const xy = PTS.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    await edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
    for (let k = 2; k < xy.length; k++) { await pause(300); await edit({ op: 'route', from: { kind: 'node', node: near(xy[k - 1]) }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: -1 }, pis: [], single: false }); }
    await pause(500);
    for (const p of xy) { await edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(200); }
    await edit({ op: 'constructAll' });
    await pause(500);
    const r = await edit({ op: 'addLine', stops: xy.map(near), name: 'Check', colour: '#0a7d3b' });
    if (r?.line !== undefined) await edit({ op: 'setTrainCount', line: r.line, trains: [6, 4, 3] });
  }, pts));
  const id = await page.evaluate(() => window.tw.world.value.save.lines[0]?.id);
  const num = await page.evaluate(() => window.tw.world.value.save.lines[0]?.num);
  await one(page, 'new', 'trains', id, () => page.evaluate(async (num) => (await import('/src/game/world.ts')).edit({ op: 'setTrainCount', line: num, trains: [12, 8, 4] }), num));
  await one(page, 'new', 'stops', id, () => page.evaluate(async () => {
    const l = window.tw.world.value.save.lines[0];
    return (await import('/src/game/world.ts')).edit({ op: 'setStops', line: l.num, stops: l.stops.slice(0, -1).map(Number) });
  }));
  await page.close();
}

try {
  if (which !== 'new') await realSave();
  if (which !== 'real') await newGame();
  check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(fails ? 1 : 0);
