// T-1803: which edits hold the clock at 4x while demand updates (game/editCap.ts). On the real save,
// each case makes one edit through the game's own edit calls and reads whether the clock was capped
// while the update ran. A case ends with demandUpdating forced off (the real solve takes minutes on
// one worker; the next edit starts a new solve anyway). From the maps root:
//   node trainworld/checks/t1803-cap.mjs [url]   (a few minutes; one demand worker)
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, GL, launchChrome, collectErrors, importSave, demandFinal } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6173/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t1803-', size: [1400, 900], args: GL });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { held: true });
  await page.waitForTimeout(1500);
  console.log('demand final');

  // lines to edit: running, not loops or buses, with 3 to 12 trains at the level now in force
  const lines = await page.evaluate(async () => {
    const w = window.tw.world.value, ck = await import('/src/game/clock.ts'), ty = await import('/src/game/types.ts');
    const out = [];
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      if (!st || st.status !== 'running' || st.loop || st.bus || st.autoS || l.stops.length < 5) continue;
      const f = ty.DEMAND_INDEX[ck.demandAt(ck.lineSeconds(st))];
      if (st.trainsNeeded[f] >= 3 && st.trainsNeeded[f] <= 12) out.push({ id: l.id, num: l.num, f, n: st.trainsNeeded, stops: l.stops.length });
    }
    return out;
  });
  console.log('candidate lines', lines.length);
  if (lines.length < 9) throw new Error('not enough candidate lines');
  const [L1, L2, L3, L4, L5, L6, L7, L8, L9] = lines;
  const DEM = ['high', 'medium', 'low'];

  const settle = async () => {
    for (let k = 0; k < 8; k++) {
      await page.evaluate(async () => { (await import('/src/game/demand.ts')).demandUpdating.value = false; });
      await page.waitForTimeout(1200);
      const up = await page.evaluate(async () => (await import('/src/game/demand.ts')).demandUpdating.value);
      if (!up) return;
    }
    console.log('  (demand would not settle)');
  };
  const sample = () => page.evaluate(async () => {
    const d = await import('/src/game/demand.ts'), ck = await import('/src/game/clock.ts'), ld = await import('/src/game/loading.ts');
    return { upd: d.demandUpdating.value, capByEdit: ck.clock.capped.value && !ld.loadProgress.value && !d.refiningZone.value, tip: document.querySelector('.status-tag')?.getAttribute('data-tip') ?? null };
  });
  /** run `fn` (page code, in a page.evaluate) and watch the update it starts */
  const run = async (label, expectCap, fn, arg) => {
    await settle();
    await page.evaluate(fn, arg);
    let sawUpd = false, sawCap = false, tipQuiet = null, tipCap = null;
    const t0 = Date.now();
    let after = 0;
    while (Date.now() - t0 < 45000 && after < 6) {
      const s = await sample();
      if (s.upd) { sawUpd = true; after++; if (s.capByEdit) { sawCap = true; tipCap = s.tip; } else tipQuiet = s.tip; }
      await page.waitForTimeout(300);
    }
    if (!sawUpd) console.log('  world after the edit:', JSON.stringify(await page.evaluate(([id]) => ({ trains: window.tw.world.value.lineStats[id]?.trainsNeeded, tph: window.tw.world.value.save.lines.find((l) => l.id === id)?.tph, key: window.twDemand.client.waiting ? 'waiting' : 'none' }), [arg?.[0] ?? arg])));
    const ok = sawUpd && sawCap === expectCap;
    check(ok, `${label}: ${sawUpd ? (sawCap ? 'capped' : 'not capped') : 'no update seen'} (expected ${expectCap ? 'capped' : 'not capped'})`);
    if (ok && !expectCap) check(!/clock runs/.test(tipQuiet ?? ''), `${label}: status hover says nothing of the clock`);
    if (ok && expectCap) check(/clock runs at 4x/.test(tipCap ?? ''), `${label}: status hover names the cap`);
    return { sawUpd, sawCap };
  };
  const countFn = async ([id, dem, to]) => (await import('/src/game/world.ts')).setTrainCount(id, dem, to);
  const other = (L) => (L.f + 1) % 3;

  // train counts
  await run('count at a level not in force, to 0', false, countFn, [L1.id, DEM[other(L1)], 0]);
  await run('count at the level in force, one fewer (under half)', false, countFn, [L2.id, DEM[L2.f], L2.n[L2.f] - 1]);
  await run('count at the level in force, down to 1 (over half)', true, countFn, [L3.id, DEM[L3.f], 1]);
  await run('count at the level in force, to 0', true, countFn, [L4.id, DEM[L4.f], 0]);
  await run('count at the level in force, up from 0', true, countFn, [L4.id, DEM[L4.f], 2]);

  // several edits in a row: a quiet one, then one that caps
  const quietThenLoud = async (a, b) => {
    await settle();
    await page.evaluate(countFn, [a.id, DEM[a.f], a.n[a.f] - 1]);
    let sawQuiet = false;
    for (let k = 0; k < 100 && !sawQuiet; k++) { const s = await sample(); if (s.upd) sawQuiet = !s.capByEdit; await page.waitForTimeout(250); }
    check(sawQuiet, 'a quiet edit first: updating, not capped');
    await page.evaluate(countFn, [b.id, DEM[b.f], 1]);
    let capped = false;
    for (let k = 0; k < 60 && !capped; k++) { capped = (await sample()).capByEdit; await page.waitForTimeout(250); }
    check(capped, 'then an edit that caps: capped');
  };
  await quietThenLoud(L5, L6);

  // headways, on a line that keeps a headway
  const headwayFn = async ([id, dem, mul]) => {
    const w = await import('/src/game/world.ts');
    const base = w.headwayShown(id);
    const was = base[{ high: 0, medium: 1, low: 2 }[dem]];
    // mul 0: the headway that runs one train fewer
    const lev = { high: 0, medium: 1, low: 2 }[dem], st = w.world.value.lineStats[id];
    w.setHeadway(id, dem, mul ? Math.round(was * mul / 5) * 5 : Math.ceil(st.roundTripByLevel[lev] / (st.trainsNeeded[lev] - 1) / 5) * 5);
  };
  await settle();
  await page.evaluate(async (id) => (await import('/src/game/world.ts')).setAutoTrains(id, true), L7.id);
  await page.waitForFunction((id) => window.tw.world.value.lineStats[id].autoS, L7.id, { timeout: 60000 });
  await run('headway at the level in force, one train fewer (under half)', false, headwayFn, [L7.id, DEM[L7.f], 0]);
  await run('headway at a level not in force, 2.5 times longer', false, headwayFn, [L7.id, DEM[other(L7)], 2.5]);
  await run('headway at the level in force, 2.5 times longer', true, headwayFn, [L7.id, DEM[L7.f], 2.5]);

  // stops
  await run('train counts of a line set to 0 at every level', true, async ([id, DEM]) => {
    const w = await import('/src/game/world.ts');
    w.setTrainCount(id, 'high', 0); w.setTrainCount(id, 'medium', 0); w.setTrainCount(id, 'low', 0);
  }, [L8.id, DEM]);
  await page.waitForFunction((id) => window.tw.world.value.lineStats[id].trainsNeeded.every((n) => n === 0), L8.id, { timeout: 60000 });
  await run('a stop dropped from a line with no trains', false, async (n) => (await import('/src/game/world.ts')).edit({ op: 'dropStop', line: n, index: 1 }), L8.num);
  await run('a stop dropped from a line with trains', true, async (n) => (await import('/src/game/world.ts')).edit({ op: 'dropStop', line: n, index: 1 }), L9.num);

  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
