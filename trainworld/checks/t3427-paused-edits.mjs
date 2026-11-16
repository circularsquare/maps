// T-3427: "Recalculate trips while paused" (Settings, Performance). With it off and the game paused, edits start
// no demand solve (none dispatched, nothing updating, the status tag says trips recalculate on unpause); pressing
// play starts exactly one, whose final has every held edit in it. With it on, a paused edit solves at once.
// From the maps root (or a worktree's), with a running dev server:
//   node trainworld/checks/t3427-paused-edits.mjs [url]
// About 2 minutes in New York; screenshots of the tag and the Settings row in artifacts/trainworld/t3427/.
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, GL, NOISE, acrossLine, collectErrors, demandFinal, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6117/';
const OUT = join(ARTIFACTS, 't3427');
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t3427-', size: [1400, 900], args: GL });
const errors = [];
let fails = 0;
const check = (name, ok, detail = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };
const NYC = [-73.985, 40.748];

/** solves dispatched so far (the focus pool's left out), and what demand shows */
const state = (page) => page.evaluate(async () => {
  const d = window.twDemand, dem = await import('/src/game/demand.ts');
  const solves = d.log.filter((x) => !d.client.fversions?.has(x.version)).length;
  const v = d.view();
  return { solves, version: v?.version ?? null, final: !!v?.final, lines: v?.lines.size ?? 0, updating: dem.demandUpdating.value, held: dem.demandHeld.value,
    tag: document.querySelector('.status-tag .demand-busy')?.textContent ?? null, tip: document.querySelector('.status-tag')?.dataset.tip ?? null,
    bar: !!document.querySelector('.status-tag .load-bar, .status-tag [class*="bar"]'), running: window.tw.clock.running.peek() };
});
const setRecalc = (page, on) => page.evaluate(async (on) => (await import('/src/game/ui.ts')).setDisplay('recalcWhilePaused', on), on);
const shift = (km) => [NYC[0] + km / (111.32 * Math.cos((NYC[1] * Math.PI) / 180)), NYC[1]];

const page = await ctx.newPage();
collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 12 }), NYC);
  const def = await page.evaluate(() => window.tw.display.value.recalcWhilePaused);
  check('on by default', def === true);

  // on (the default), paused: an edit solves at once
  if ((await acrossLine(page, NYC)) === null) throw new Error('no first line');
  await demandFinal(page, { timeout: 180000, lines: 0 });
  await page.waitForTimeout(3000); // (a zone's fine solve may follow its first)
  await demandFinal(page, { timeout: 180000, lines: 0 });
  const a = await state(page);
  check('on: a paused edit solves', a.final && a.lines === 1 && a.solves >= 1, JSON.stringify(a));

  // off, paused: edits start nothing
  await setRecalc(page, false);
  const line2 = await acrossLine(page, shift(-6));
  const line3 = await acrossLine(page, shift(6));
  if (line2 === null || line3 === null) throw new Error('no second or third line');
  await page.evaluate(async (n) => window.tw.client.edit({ op: 'setTrainCount', line: n, trains: [8, 6, 4] }), line2);
  await page.waitForTimeout(6000);
  const b = await state(page);
  check('off: paused edits dispatch no solve', b.solves === a.solves && !b.updating, `solves ${a.solves} -> ${b.solves}, updating ${b.updating}`);
  check('off: the figures stay the last final', b.version === a.version && b.lines === 1, `version ${b.version}, lines ${b.lines}`);
  check('off: held, and the tag says so with a hover and no bar', b.held && b.tag === 'Trips recalculate when you unpause' && b.tip === 'Recalculate trips while paused is off in Settings' && !b.bar, JSON.stringify({ held: b.held, tag: b.tag, tip: b.tip, bar: b.bar }));
  const box = await page.locator('.map-top-left').boundingBox().catch(() => null);
  if (box) await page.screenshot({ path: join(OUT, 'tag.png'), clip: { x: Math.max(0, box.x - 10), y: Math.max(0, box.y - 10), width: 700, height: box.height + 20 } });

  // play: exactly one solve, covering every held edit
  await page.evaluate(() => window.tw.clock.setRunning(true));
  await page.waitForTimeout(1500);
  await page.evaluate(() => window.tw.clock.setRunning(false));
  await demandFinal(page, { timeout: 180000, after: a.version });
  await page.waitForTimeout(3000);
  const c = await state(page);
  const why = await page.evaluate((v0) => { const d = window.twDemand; const L = d.log.filter((x) => x.version > v0); return L.map((s, i) => ({ version: s.version, cities: s.cities.join(','), markets: s.markets.length, coarse: (s.coarse ?? []).join(','), sameNet: i > 0 && s.svc === L[i - 1].svc, sinceArrive: Math.round(s.sentAt - s.arrivedAt), at: Math.round(s.sentAt) })); }, a.version);
  // (a network whose stations reach new market areas is solved again once those open, as after any edit)
  const nets = why.filter((s) => !s.sameNet).length, again = why.filter((s) => s.sameNet);
  check('play: exactly one network solved, every held edit in it', nets === 1 && again.every((s) => s.markets > why[0].markets), `${why.length} solves, ${nets} network(s)${again.length ? ', then the same network with the market areas it reaches' : ''}`);
  check('play: its final has all three lines', c.final && c.lines === 3 && !c.held && c.tag !== 'Trips recalculate when you unpause', JSON.stringify(c));

  // off, paused, then the setting turned back on: the held edit goes at once
  const before = c.solves;
  await page.evaluate(async (n) => window.tw.client.edit({ op: 'setTrainCount', line: n, trains: [4, 3, 2] }), line3);
  await page.waitForTimeout(3000);
  const d = await state(page);
  check('off again: a paused edit waits', d.solves === before && d.held, `solves ${d.solves}, held ${d.held}`);
  await setRecalc(page, true);
  await page.waitForTimeout(1500);
  const e = await state(page);
  check('turned on while paused: one solve starts', e.solves === before + 1 && !e.held, `solves ${e.solves}, held ${e.held}`);
  await demandFinal(page, { timeout: 180000 });

  // the Settings row
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).tab.value = 'settings'; });
  await page.waitForTimeout(500);
  const row = page.locator('label', { hasText: 'Recalculate trips while paused' });
  check('Settings has the checkbox', (await row.count()) === 1);
  await row.scrollIntoViewIfNeeded();
  const rb = await row.boundingBox();
  const dock = await page.locator('.checks', { has: row }).boundingBox();
  if (rb && dock) await page.screenshot({ path: join(OUT, 'settings.png'), clip: { x: Math.max(0, dock.x - 20), y: Math.max(0, dock.y - 60), width: dock.width + 40, height: dock.height + 160 } });
} catch (e) {
  check('ran to the end', false, e.message.split('\n')[0]);
}
check('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
await close();
console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails ? 1 : 0);
