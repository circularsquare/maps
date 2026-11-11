// T-3165 (after T-3152): a line's train count can be set to one train more than the track lets run, so one train is
// held and the held messages, the issue and the red headway show. On the real New York save:
//  - typing the whole trains the track runs leaves the headway plain; plus at that gives one more, with that train held
//    (`lineHeld` waits) and the headway red; another plus, and a typed number above, are refused with the shake;
//  - a stored count far above the cap is lowered to it when the game loads (set high by an edit, downloaded, opened again).
// Screenshots artifacts/trainworld/t3165-*.png.
// From the maps root, on a Vite of your own:   node trainworld/checks/t3165-cap.mjs [dev url]
import { ARTIFACTS, REAL_SAVE, collectErrors, demandFinal, launchChrome } from './lib.mjs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6113/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t3165-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const ready = () => page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3165-${name}.png`) });
const loaded = async () => {
  await page.waitForURL(/load=file/, { timeout: 120000 });
  await ready();
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000, polling: 100 });
  await demandFinal(page, { timeout: 600000 });
  await page.waitForTimeout(1500);
};

try {
  await page.goto(`${BASE}?debug=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await ready();
  const bytes = await readFile(REAL_SAVE);
  await page.evaluate(async (data) => { const m = await import('/src/workers/clockClient.ts'); await m.importGame(new File([Uint8Array.from(data)], 'game.save')); }, Array.from(bytes));
  await loaded();

  // a running rail line whose track limits its level-0 count to W whole trains, with room to type W
  const line = await page.evaluate(async () => {
    const { queryClock } = await import('/src/workers/clockClient.ts');
    const w = window.tw.world.value;
    let best = null;
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      if (!st || st.status !== 'running' || st.bus || st.loop || st.autoS || !(st.trainsNeeded[0] >= 2 && st.trainsNeeded[0] < 40)) continue;
      const d = await queryClock({ what: 'lineHeld', line: l.num });
      const s = d[21], trip = st.roundTripByLevel[0];
      if (!Number.isFinite(s) || !(s > 0)) continue;
      const room = Math.floor(trip / s + 0.01);
      if (room >= st.trainsNeeded[0] + 3 && room < 300 && (!best || room < best.room)) best = { id: l.id, num: l.num, room, trains: st.trainsNeeded.slice(), name: l.name };
    }
    if (best) window.tw.selection.value = { kind: 'line', line: best.id };
    return best;
  });
  console.log('line', JSON.stringify(line));
  if (!line) throw new Error('no line with a track limit and room');
  const W = line.room;
  await page.waitForTimeout(1500);
  const typedBtn = page.locator('button.typed[aria-label^="Type trains on the line"]').first();
  const count = () => page.evaluate((id) => window.tw.world.value.lineStats[id].trainsNeeded[0], line.id);
  const waits = () => page.evaluate(async (num) => { const { queryClock } = await import('/src/workers/clockClient.ts'); const d = await queryClock({ what: 'lineHeld', line: num }); return d[0]; }, line.num);
  const type = async (text) => {
    await typedBtn.click();
    await page.keyboard.press('Control+A');
    await page.keyboard.type(text);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(2500);
  };
  const plus = async () => { await page.locator('.stepper:has(button.typed[aria-label^="Type trains on the line"]) button[aria-label^="More"]').first().click(); await page.waitForTimeout(2500); };
  const redHeadway = () => page.evaluate(() => [...document.querySelectorAll('table td .bad')].some((e) => /\d/.test(e.textContent ?? '') && /min|h|:|\//.test(e.textContent ?? '')));
  await page.evaluate(() => { window.__shook = 0; new MutationObserver((ms) => { for (const m of ms) if (m.target.classList?.contains('stepper') && m.target.classList.contains('shake')) window.__shook++; }).observe(document.body, { subtree: true, attributes: true, attributeFilter: ['class'] }); });
  const shook = () => page.evaluate(() => window.__shook > 0);

  await type(String(W));
  check((await count()) === W, `typing the ${W} whole trains the track runs sets it`, `${await count()}`);
  check(!(await redHeadway()) && !((await waits()) > 0), 'at that count no train is held and the headway is not red');
  await plus();
  check((await count()) === W + 1, 'plus at the running count gives one more', `${await count()}`);
  const held = await waits();
  check(held > 0, 'that train is held (lineHeld reports a wait)', `${Math.round(held)} s`);
  check(await redHeadway(), 'the headway is red');
  await shot('held');
  await page.evaluate(() => { window.__shook = 0; });
  await plus();
  check((await count()) === W + 1 && await shook(), 'one more plus is refused with the shake', `${await count()}`);
  await page.evaluate(() => { window.__shook = 0; });
  await type(String(W + 8));
  check((await count()) === W + 1 && await shook(), 'a typed number above is clamped to the cap with the shake', `${await count()}`);

  // load: a stored count far above the cap
  await page.evaluate(async ([num, n]) => { const w = await import('/src/game/world.ts'); await w.edit({ op: 'setTrainCount', line: num, trains: [n, n, n] }, true, true); }, [line.num, W + 9]);
  await page.waitForTimeout(3000);
  const high = await count();
  check(high > W + 1, 'a stored count above the cap is set up for the load test', `${high}`);
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(async () => { (await import('/src/workers/clockClient.ts')).exportGame(); });
  const file = await (await dl).createReadStream();
  const chunks = [];
  for await (const c of file) chunks.push(c);
  const saved = Buffer.concat(chunks);
  await page.evaluate(async (data) => { const m = await import('/src/workers/clockClient.ts'); await m.importGame(new File([Uint8Array.from(data)], 'over.save')); }, Array.from(saved));
  await page.waitForFunction(([num, to]) => { const w = window.tw?.world?.value; const l = w?.save?.lines?.find((x) => x.num === num); return !!l && w.lineStats[l.id].trainsNeeded[0] === to; }, [line.num, W + 1], { timeout: 180000, polling: 500 }).catch(() => {});
  const after = await page.evaluate((num) => { const w = window.tw.world.value; const l = w.save.lines.find((x) => x.num === num); return w.lineStats[l.id].trainsNeeded.slice(); }, line.num);
  check(after[0] === W + 1, 'the count is lowered to the cap when the game loads', `${high} -> ${after[0]}`);
  check(after[1] <= high && after[2] <= high, 'other levels are not above what was stored', JSON.stringify(after));
  check(errors.length === 0, 'no page errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  console.log('ERROR', e.stack ?? e);
  fail++;
} finally {
  await chrome.close();
}
console.log(fail ? `${fail} FAILED` : 'ALL PASS');
process.exit(fail ? 1 : 0);
