// T-3152, T-3154 (and T-3151's button colour): on the real New York save,
//  - the trouble list button is orange (--warn), not red (--bad);
//  - a line's train count typed in the schedule: a number and Enter sets it, a number past the track's limit is clamped to it,
//    Escape cancels, text that is no number reverts;
//  - a station complex's view lists its stations, each with the terminal button; a click builds a terminal at that
//    station only (the same edit as the station view's).
// Screenshots artifacts/trainworld/t3151-*.png.
// From the maps root, on a Vite of your own:   node trainworld/checks/t3151-todo.mjs [dev url]   (SHADE=dark)
import { ARTIFACTS, REAL_SAVE, collectErrors, demandFinal, launchChrome } from './lib.mjs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6200/';
const SHADE = process.env.SHADE ?? 'light';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t3151-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, SHADE);
collectErrors(page, errors);
const ready = () => page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3151-${name}-${SHADE}.png`) });

try {
  await page.goto(`${BASE}?debug=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await ready();
  const bytes = await readFile(REAL_SAVE);
  await page.evaluate(async (data) => { const m = await import('/src/workers/clockClient.ts'); await m.importGame(new File([Uint8Array.from(data)], 'game.save')); }, Array.from(bytes));
  await page.waitForURL(/load=file/, { timeout: 120000 });
  await ready();
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000, polling: 100 });
  await demandFinal(page, { timeout: 600000 });
  await page.waitForTimeout(1000);

  // ---- T-3151: the trouble button
  const btn = await page.evaluate(() => {
    const el = document.querySelector('.trouble-btn');
    if (!el) return null;
    const probe = (v) => { const p = document.createElement('i'); p.style.color = `var(${v})`; document.body.appendChild(p); const c = getComputedStyle(p).color; p.remove(); return c; };
    return { colour: getComputedStyle(el).color, warn: probe('--warn'), bad: probe('--bad') };
  });
  check(!!btn && btn.colour === btn.warn && btn.colour !== btn.bad, 'the trouble list button is the warning orange, not the red', JSON.stringify(btn));
  await shot('trouble');

  // ---- T-3152: type the train count of a running rail line
  // a running rail line whose track limits its level-0 count: the most trains it takes is the whole trains that fit
  // (round trip over the shortest headway, `lineHeld`'s second block), as the schedule works it out
  const line = await page.evaluate(async () => {
    const { queryClock } = await import('/src/workers/clockClient.ts');
    const w = window.tw.world.value;
    let best = null;
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      if (!st || st.status !== 'running' || st.bus || st.loop || !(st.trainsNeeded[0] >= 2 && st.trainsNeeded[0] < 40)) continue;
      const d = await queryClock({ what: 'lineHeld', line: l.num });
      const s = d[21], trip = st.roundTripByLevel[0];
      if (!Number.isFinite(s) || !(s > 0)) continue;
      const cap = Math.floor(trip / s + 0.01);
      if (cap >= st.trainsNeeded[0] + 5 && cap < 300 && (!best || cap < best.cap)) best = { id: l.id, num: l.num, cap, trains: st.trainsNeeded.slice() };
    }
    best ??= (() => { const l = w.save.lines.find((x) => { const st = w.lineStats[x.id]; return st && st.status === 'running' && !st.bus && !st.loop && st.trainsNeeded[0] >= 2 && st.trainsNeeded[0] < 40; }); return { id: l.id, num: l.num, cap: null, trains: w.lineStats[l.id].trainsNeeded.slice() }; })();
    window.tw.selection.value = { kind: 'line', line: best.id };
    return best;
  });
  console.log('line', JSON.stringify(line));
  await page.waitForTimeout(1500);
  const typedBtn = page.locator('button.typed[aria-label^="Type trains on the line"]').first();
  const count = () => page.evaluate((id) => window.tw.world.value.lineStats[id].trainsNeeded[0], line.id);
  const shown = () => page.evaluate(() => { const el = document.querySelector('button.typed[aria-label^="Type trains on the line"], input.typing[aria-label^="trains on the line"]'); return el ? (el.tagName === 'INPUT' ? el.value : el.textContent).trim() : null; });
  check(await typedBtn.count() > 0, 'the schedule\'s train count is a button to type in', JSON.stringify(line.trains));
  const type = async (text, key) => {
    await typedBtn.click();
    await page.keyboard.press('Control+A');
    await page.keyboard.type(text);
    await page.keyboard.press(key);
    await page.waitForTimeout(2500);
  };
  const first = line.trains[0];
  await type(String(first + 3), 'Enter');
  const pending = await shown();
  const after = await count();
  check(after === first + 3 && pending === String(first + 3), 'a number and Enter sets the count', `was ${first}, now ${after}, shown ${pending}`);
  await typedBtn.click();
  await page.keyboard.press('Control+A');
  await page.keyboard.type('12');
  await shot('train-count-typing');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(1500);
  check((await count()) === first + 3, 'Escape leaves the count as it was', `${await count()}`);
  await type('abc', 'Enter');
  check((await count()) === first + 3, 'text that is no number leaves the count as it was', `${await count()}`);
  // T-3152: the limit. A typed number past it is clamped to the limit, the stepper's plus at the limit is refused,
  // and neither leaves more trains than run
  const shook = () => page.evaluate(() => !!document.querySelector('.stepper.shake') || window.__shook > 0);
  const cap = line.cap ?? 1000;
  await page.evaluate(() => { window.__shook = 0; new MutationObserver((ms) => { for (const m of ms) if (m.target.classList?.contains('stepper') && m.target.classList.contains('shake')) window.__shook++; }).observe(document.body, { subtree: true, attributes: true, attributeFilter: ['class'] }); });
  await type('99999', 'Enter');
  check((await count()) === cap, `a number past the limit is clamped to it (${line.cap === null ? 'no track limit here: 1000' : cap})`, `${await count()}`);
  check(await shook(), 'the clamp shakes the stepper');
  await page.evaluate(() => { window.__shook = 0; });
  await page.locator('.stepper:has(button.typed[aria-label^="Type trains on the line"]) button[aria-label^="More"]').first().click();
  await page.waitForTimeout(2500);
  check((await count()) === cap && await shook(), 'plus at the limit is refused with the shake', `${await count()}`);
  await page.evaluate(() => { window.__shook = 0; });
  await type(String(cap + 7), 'Enter');
  check((await count()) === cap && await shook(), 'typing more at the limit is refused with the shake', `${await count()}`);
  await type(String(cap - 2), 'Enter');
  check((await count()) === cap - 2, 'typing fewer is allowed', `${await count()}`);
  await type('0', 'Enter');
  check((await count()) === 0, 'zero sets the line to no trains', `${await count()}`);

  // ---- T-3154: a complex's stations, one row each with its line chips and terminal button (no row per line)
  const cx = await page.evaluate(async () => {
    const { complexOf } = await import('/src/game/complexView.ts');
    const w = window.tw.world.value;
    let best = null;
    for (const s of w.save.stations) {
      const m = complexOf(s.id);
      if (m.length < 2 || m.length > 6 || m[0] !== s.id) continue;
      const st = m.map((id) => w.save.stations.find((x) => x.id === id));
      if (!st.every((x) => !x.tram && !x.port && !(x.upgrades & 1))) continue;
      const nl = w.save.lines.filter((l) => l.stops.some((x) => m.includes(x))).length;
      if (!best || nl > best.nl) best = { id: s.id, members: m, names: st.map((x) => x.name), nl };
    }
    return best;
  });
  check(!!cx, 'a complex without terminals found in the save', JSON.stringify(cx));
  await page.evaluate(async (id) => {
    const ui = await import('/src/game/complexView.ts');
    window.tw.selection.value = { kind: 'station', station: id };
    ui.singleStation.value = null;
  }, cx.id);
  await page.waitForTimeout(1500);
  const rows = page.locator('ul.complex-stations > li:not(.cols)');
  check((await rows.count()) === cx.members.length, 'one row per station of the complex', `${await rows.count()} rows, ${cx.members.length} stations, ${cx.nl} lines`);
  const rowNames = await rows.evaluateAll((els) => els.map((e) => e.querySelector('.grow')?.textContent.trim()).sort());
  check(JSON.stringify(rowNames) === JSON.stringify([...cx.names].sort()), 'the rows are the stations\' names, not line names', JSON.stringify(rowNames));
  const chips = await page.locator('ul.complex-stations .station-chips .chip').count();
  check(chips >= cx.nl, 'each station row has the chips of its lines', `${chips} chips for ${cx.nl} lines`);
  const buttons = page.locator('ul.complex-stations button[aria-label="Terminal"]');
  check((await buttons.count()) === cx.members.length, 'each row has the terminal button', `${await buttons.count()}`);
  await shot('complex-stations');
  const upg = (id) => page.evaluate((id) => window.tw.world.value.save.stations.find((x) => x.id === id).upgrades & 1, id);
  await buttons.nth(1).click();
  await page.waitForTimeout(2500);
  const built = await Promise.all(cx.members.map(upg));
  check(built.filter(Boolean).length === 1, 'a click on one row builds a terminal at that station only', JSON.stringify(built));
  const done = built.findIndex(Boolean);
  const tip = await buttons.nth(1).getAttribute('data-tip');
  check(/Terminal/.test(tip ?? '') && (await buttons.nth(1).getAttribute('aria-pressed')) === 'true', 'its button now shows built', tip?.replace(/\n/g, ' | '));
  await shot('complex-stations-built');
  await buttons.nth(1).click();
  await page.waitForTimeout(2500);
  check((await upg(cx.members[done])) === 0, 'a second click takes it away again');
  await page.locator('ul.complex-stations .station-chips .chip').first().click();
  await page.waitForTimeout(500);
  check(await page.evaluate(() => window.tw.selection.value?.kind === 'line'), 'a line chip in a row selects that line');
} catch (e) {
  console.log('FAIL', e.message);
  fail++;
}
check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
await chrome.close();
console.log(fail ? `T-3151..4 FAILED (${fail})` : 'T-3151..4 PASSED');
process.exit(fail ? 1 : 0);
