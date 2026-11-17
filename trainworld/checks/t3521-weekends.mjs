// T-3521: weekend trains in the schedule panel. On the real save, with no `?timetables`: a line's
// panel opens from the line panel's icon and shows a Weekdays and a Weekends section, the percent at
// 100%; stepping the percent down to 70% follows the weekdays (counts rounded up) and the sim agrees;
// a weekday change moves the weekends with it; a weekend count stepped by hand greys the percent and
// survives a weekday change; Apply to all lines asks first and sets every line; Escape closes the
// panel. At 50% on every line: no demand solve; a Saturday and a public holiday run fewer trips, the
// Lines tab counts the weekend's, the Weekends row is bold, an hour costs less. A headway line at
// 15/20/30 min runs 70%, 50% and 150% weekends at the weekday headway over the percent, down to a
// whole minute (the panel shows it), its weekdays unchanged. A fresh game first:
// the icon opens the panel there too. Shots of both states, light. From the maps
// root (or a tree's root):
//   node trainworld/checks/t3521-weekends.mjs [url] [shots dir]   (about two minutes; default localhost:6168)
import { existsSync } from 'node:fs';
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, MAPS, REAL_SAVE, GL, launchChrome, collectErrors, importSave, acrossLine } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6168/';
const OUT = process.argv[3] ?? ARTIFACTS;
// a tree has no copy of the real save: the main tree's
const SAVE = existsSync(REAL_SAVE) ? REAL_SAVE : join(MAPS, 'trainworld', 'T-007-real-nyc.save');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3521-', size: [1400, 900], args: GL, deviceScaleFactor: 1 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { if (!sessionStorage.getItem('t3521')) { sessionStorage.setItem('t3521', '1'); localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 15 })); } } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });

  // a fresh game: a line built in Manhattan, its panel's icon opens the schedule panel (no `?timetables`)
  await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 12 }), [-73.985, 40.758]);
  const freshLine = await acrossLine(page, [-73.985, 40.758]);
  check(freshLine !== null, `a fresh game's line built (${freshLine})`);
  await page.waitForFunction((n) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === n); return l && w.lineStats[l.id]?.status === 'running'; }, freshLine, { timeout: 60000 }).catch(() => {});
  await page.evaluate((n) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === n); window.tw.selection.value = { kind: 'line', line: l.id }; }, freshLine);
  const icon = await page.waitForSelector('button[data-tip="Detailed schedule"]', { timeout: 30000 }).catch(() => null);
  check(!!icon, "a fresh game's line panel has the schedule icon");
  if (icon) {
    await icon.click();
    const open = await page.waitForSelector('dialog.sched-panel[open]', { timeout: 10000 }).catch(() => null);
    check(!!open, 'the icon opens the schedule panel on a fresh game');
    await page.waitForTimeout(800);
    await (await page.$('dialog.sched-panel .saves-inner'))?.screenshot({ path: join(OUT, 't3521_fresh.png') });
    await page.click('dialog.sched-panel .sp-close');
    await page.waitForTimeout(400);
    check(!(await page.$('dialog.sched-panel[open]')), 'its close button closes it');
  }

  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 4, undefined, { timeout: 120000 });
  await page.waitForTimeout(4000);
  await page.evaluate(async () => {
    window.__m = { w: await import('/src/game/world.ts'), cc: await import('/src/workers/clockClient.ts') };
  });
  const S = (f, arg) => page.evaluate(`(async () => (${f.toString()})({ ...window.__m, tw: window.tw }, ${JSON.stringify(arg ?? null)}))()`);
  const settle = async (f, arg, timeout = 60000) => page.waitForFunction(f, arg, { timeout, polling: 200 });

  // a running count line with trains at every level, no twin
  const pick = await S(({ tw }) => {
    const w = tw.world.value;
    const ok = (l) => { const st = w.lineStats[l.id]; return st && !st.bus && !st.twin && !l.twinOf && !st.autoS && st.status === 'running' && l.stops.length >= 4 && st.trainsNeeded.every((n) => n >= 3); };
    const l = w.save.lines.filter(ok).sort((a, b) => b.trainsNeeded?.[0] - a.trainsNeeded?.[0])[0];
    return l ? { id: l.id, num: l.num, name: l.name, trains: w.lineStats[l.id].trainsNeeded } : null;
  });
  console.log('line', JSON.stringify(pick));
  check(!!pick, 'a running count line to show');

  // open from the line panel's icon, no ?timetables in the address
  await S(({ tw }, id) => { tw.selection.value = { kind: 'line', line: id }; }, pick.id);
  await page.waitForSelector('button[data-tip="Detailed schedule"]', { timeout: 30000 });
  await page.click('button[data-tip="Detailed schedule"]');
  await page.waitForSelector('dialog.sched-panel[open]', { timeout: 10000 });
  await page.waitForTimeout(800);
  const heads = await page.$$eval('dialog.sched-panel tr.sched-day th', (ths) => ths.map((t) => t.textContent));
  check(heads.length === 2 && heads[0].startsWith('Weekdays') && heads[1].startsWith('Weekends'), `two sections: ${JSON.stringify(heads)}`);
  const pct0 = await page.$eval('dialog.sched-panel .sched-day-row .stepper', (e) => e.textContent);
  check(/100%/.test(pct0), `the percent starts at 100% (${pct0})`);

  // percent down to 70% with six clicks on its minus
  const minus = 'dialog.sched-panel .sched-day-row .stepper button:first-child';
  for (let i = 0; i < 6; i++) { await page.click(minus); await page.waitForTimeout(250); }
  await settle((id) => window.tw.world.value.lineStats[id]?.weekend?.[0] === 70, pick.id).catch(() => {});
  const at70 = await S(({ tw }, id) => { const st = tw.world.value.lineStats[id]; return { weekend: st.weekend, trains: st.trainsNeeded, restTph: st.restTph, tph: tw.world.value.save.lines.find((l) => l.id === id).tph }; }, pick.id);
  console.log('at 70%', JSON.stringify(at70));
  const want = at70.trains.map((n) => Math.ceil(n * 0.7 - 1e-9));
  check(at70.weekend?.[0] === 70 && at70.weekend[1] === 0 && JSON.stringify(at70.weekend.slice(2, 5)) === JSON.stringify(want), `70%: weekend counts ${JSON.stringify(at70.weekend?.slice(2, 5))}, want ${JSON.stringify(want)}`);
  check(at70.restTph[0] < at70.tph.high, `weekend high rate ${at70.restTph[0]?.toFixed(2)} under the weekday's ${at70.tph.high?.toFixed(2)}`);
  await page.waitForTimeout(800);
  const panel = await page.$('dialog.sched-panel .saves-inner');
  await (await page.$('dialog.sched-panel .sp-left'))?.screenshot({ path: join(OUT, 't3521_percent.png') });
  await panel?.screenshot({ path: join(OUT, 't3521_panel_percent.png') });

  // a weekday change moves the weekends
  await S(({ w }, p) => w.setTrainCount(p.id, 'high', p.trains[0] + 4), { id: pick.id, trains: at70.trains });
  await settle((a) => window.tw.world.value.lineStats[a.id]?.trainsNeeded[0] === a.n, { id: pick.id, n: at70.trains[0] + 4 }).catch(() => {});
  await settle((a) => window.tw.world.value.lineStats[a.id]?.weekend?.[2] === Math.ceil(a.n * 0.7 - 1e-9), { id: pick.id, n: at70.trains[0] + 4 }).catch(() => {});
  const moved = await S(({ tw }, id) => tw.world.value.lineStats[id].weekend, pick.id);
  check(moved?.[2] === Math.ceil((at70.trains[0] + 4) * 0.7 - 1e-9), `a weekday change moves the weekend: ${JSON.stringify(moved)}`);

  // a weekend count by hand: the percent greys, and a weekday change leaves it
  const rows = await page.$$('dialog.sched-panel table.sched tr');
  let weekendRow = null, seen = false;
  for (const r of rows) {
    if (await r.evaluate((e) => e.classList.contains('sched-day') && e.textContent.includes('Weekends'))) { seen = true; continue; }
    if (seen && !weekendRow) weekendRow = r;
  }
  await (await weekendRow.$('td:nth-child(2) .stepper button:first-child')).click();
  await settle((id) => window.tw.world.value.lineStats[id]?.weekend?.[1] === 1, pick.id).catch(() => {});
  const hand = await S(({ tw }, id) => tw.world.value.lineStats[id].weekend, pick.id);
  check(hand?.[1] === 1 && hand[0] === 70, `a weekend count by hand: ${JSON.stringify(hand)}`);
  await page.waitForTimeout(800);
  const grey = await page.$eval('dialog.sched-panel .sched-day-row .stepper .muted', (e) => getComputedStyle(e).color).catch(() => null);
  check(!!grey, `the percent greys (${grey})`);
  await (await page.$('dialog.sched-panel .sp-left'))?.screenshot({ path: join(OUT, 't3521_custom.png') });
  await S(({ w }, p) => w.setTrainCount(p.id, 'high', p.n), { id: pick.id, n: at70.trains[0] + 6 });
  await settle((a) => window.tw.world.value.lineStats[a.id]?.trainsNeeded[0] === a.n, { id: pick.id, n: at70.trains[0] + 6 }).catch(() => {});
  await page.waitForTimeout(1500);
  const kept = await S(({ tw }, id) => tw.world.value.lineStats[id].weekend, pick.id);
  check(JSON.stringify(kept) === JSON.stringify(hand), `counts by hand stay through a weekday change: ${JSON.stringify(kept)}`);

  // Apply to all lines: a question first, then every line at the percent
  await page.click('dialog.sched-panel .sched-day-row button.btn.text');
  await page.waitForSelector('dialog.confirmation[open]', { timeout: 5000 });
  const ask = await page.$eval('dialog.confirmation #confirmation-text', (e) => e.textContent);
  check(/70%/.test(ask), `the question: ${ask}`);
  await page.screenshot({ path: join(OUT, 't3521_ask.png') });
  await page.click('dialog.confirmation .btn.danger');
  await settle(() => { const w = window.tw.world.value; return w.save.lines.filter((l) => !l.twinOf && w.lineStats[l.id]?.status === 'running').every((l) => w.lineStats[l.id].weekend?.[0] === 70 && w.lineStats[l.id].weekend[1] === 0); }, null, 120000).catch(() => {});
  const all = await S(({ tw }) => { const w = tw.world.value; const ls = w.save.lines.filter((l) => !l.twinOf && w.lineStats[l.id]?.status === 'running'); return { n: ls.length, at70: ls.filter((l) => w.lineStats[l.id].weekend?.[0] === 70 && w.lineStats[l.id].weekend[1] === 0).length }; });
  check(all.n > 0 && all.at70 === all.n, `every running line at 70%: ${all.at70}/${all.n}`);
  await page.waitForTimeout(800);
  await (await page.$('dialog.sched-panel .sp-left'))?.screenshot({ path: join(OUT, 't3521_after_all.png') });

  // undo puts the lines back
  await S(({ w }) => w.edit({ op: 'undo' }));
  await settle(() => { const w = window.tw.world.value; return w.save.lines.some((l) => w.lineStats[l.id]?.weekend?.[1] === 1); }, null).catch(() => {});
  const undone = await S(({ tw }, id) => tw.world.value.lineStats[id].weekend, pick.id);
  check(undone?.[1] === 1, `undo brings the line's counts by hand back: ${JSON.stringify(undone)}`);

  await page.keyboard.press('Escape');
  await page.waitForTimeout(400);
  check(!(await page.$('dialog.sched-panel[open]')), 'Escape closes the panel');

  // the days: every line at 50%, no demand solve for it; then a Saturday and a public holiday run
  // fewer trains, count fewer in the Lines tab, bold the Weekends row and cost less an hour. The clock
  // only moves forward (the ledger's hours settle as it goes).
  await page.evaluate(async () => {
    window.__m.lr = await import('/src/game/lineRunning.ts');
    window.__m.hol = await import('/src/game/holidays.ts');
    window.__m.days = await import('/src/game/days.ts');
    window.__m.money = await import('/src/game/money.ts');
  });
  const solves0 = await page.evaluate(() => window.twDemand.log.length);
  await S(({ w }) => w.setWeekendShareAll(50));
  await settle(() => { const w = window.tw.world.value; return w.save.lines.filter((l) => !l.twinOf && w.lineStats[l.id]?.status === 'running').every((l) => w.lineStats[l.id].weekend?.[0] === 50 && w.lineStats[l.id].weekend[1] === 0); }, null, 120000).catch(() => {});
  await page.waitForTimeout(5000);
  const solves1 = await page.evaluate(() => window.twDemand.log.length);
  check(solves1 === solves0, `weekend edits start no demand solve (${solves0} solves before, ${solves1} after)`);
  const days = await S(({ tw, hol, days }, id) => {
    const w = tw.world.value, l = w.save.lines.find((x) => x.id === id), st = w.lineStats[id];
    const c = hol.lineCountry(l.stops), shift = st.shift ?? 0, today = Math.floor((tw.clock.now() + shift) / 86400);
    let sat = null, hol1 = null;
    for (let d = today + 2; d < today + 400 && (sat === null || hol1 === null); d++) {
      if (sat === null && hol.kindIn(c, d) === 1 && hol.rowIn(c, d) === 1 && hol.rowIn(c, d - 1) === 0) sat = d;
      if (hol1 === null && d > today + 9 && hol.kindIn(c, d) === 2 && days.dayKindOf(d) === 0 && hol.rowIn(c, d) === 1 && hol.rowIn(c, d - 7) === 0) hol1 = d;
    }
    return { c, shift, sat, hol: hol1, num: l.num };
  }, pick.id);
  console.log('days', JSON.stringify(days));
  const at = (d, h) => d * 86400 + h * 3600 - days.shift;
  const trips = (t0) => S(async ({ cc }, a) => (await cc.queryClock({ what: 'lineTrips', line: a.num, t0: a.t0, t1: a.t0 + 3600 }))?.length / 4, { num: days.num, t0 });
  const goTo = async (t) => { await page.evaluate((t) => window.tw.clock.set(t), t); await page.waitForTimeout(2500); };
  // an hour's running cost (game hour `t`'s): the clock moved past it, which settles the hours on the
  // way (a headless page's clock does not run)
  const cost = async (t) => {
    const h = Math.floor(t / 3600);
    await goTo(t + 5400);
    await page.waitForFunction((h) => (window.__m.money.moneyState.value?.hours ?? []).some((x) => x.hour > h), h, { timeout: 60000, polling: 200 }).catch(() => {});
    return page.evaluate((h) => window.__m.money.moneyState.value?.hours?.find((x) => x.hour === h)?.running ?? null, h);
  };
  if (days.sat === null) check(false, 'a Saturday ahead');
  else {
    const fri = await cost(at(days.sat - 1, 13)), sat = await cost(at(days.sat, 13));
    check(fri > 0 && sat > 0 && sat < fri * 0.7, `an hour's running cost: Saturday ${sat?.toFixed(4)}, Friday ${fri?.toFixed(4)} (US$M)`);
    await goTo(at(days.sat, 15) + 600);
    const [before, on] = [await trips(at(days.sat - 1, 13)), await trips(at(days.sat, 13))];
    check(on < before * 0.75 && on > 0, `Saturday (day ${days.sat}): ${on} trips leave 13:00-14:00, ${before} on the Friday`);
    const now = await S(({ lr, tw }, id) => ({ n: lr.trainsNowAt(id, 0), wk: tw.world.value.lineStats[id].weekend }), pick.id);
    check(now.n === now.wk?.[2], `Saturday: the Lines tab counts ${now.n} trains now, the weekend's ${now.wk?.[2]}`);
    // the panel: the bold row is in the Weekends section
    await S(({ tw }, id) => { tw.selection.value = { kind: 'line', line: id }; }, pick.id);
    await page.click('button[data-tip="Detailed schedule"]');
    await page.waitForSelector('dialog.sched-panel[open]', { timeout: 10000 });
    await page.waitForTimeout(800);
    const bold = await page.$$eval('dialog.sched-panel table.sched tr', (rs) => { let sec = ''; for (const r of rs) { if (r.classList.contains('sched-day')) sec = r.textContent.startsWith('Weekends') ? 'weekends' : 'weekdays'; if (r.classList.contains('now')) return sec + ' ' + r.textContent; } return null; });
    check(/^weekends High/.test(bold ?? ''), `on a Saturday the level now is bold under Weekends: ${bold}`);
    await (await page.$('dialog.sched-panel .sp-left'))?.screenshot({ path: join(OUT, 't3521_saturday.png') });
    await page.keyboard.press('Escape');
  }
  if (days.hol === null) check(false, 'a public holiday ahead');
  else {
    // a public holiday's calendar reaches the trips only near it: the clock goes to the day before
    await goTo(at(days.hol - 1, 13));
    const [before, on] = [await trips(at(days.hol - 7, 13)), await trips(at(days.hol, 13))];
    check(on < before * 0.75 && on > 0, `public holiday (day ${days.hol}): ${on} trips leave 13:00-14:00, ${before} a week before`);
    await goTo(at(days.hol, 13));
    const now = await S(({ lr, tw }, id) => ({ n: lr.trainsNowAt(id, 0), wk: tw.world.value.lineStats[id].weekend }), pick.id);
    check(now.n === now.wk?.[2], `public holiday: the Lines tab counts ${now.n} trains now, the weekend's ${now.wk?.[2]}`);
  }

  // a headway line: its weekends run the weekday headway over the percent, down to a whole minute,
  // and its weekdays stay as they were (15/20/30 min: at 70% 21/28/42, at 50% 30/40/60, at 150%
  // 10/13/20); the panel's Weekends rows show it
  const hl = await S(({ tw }) => {
    const w = tw.world.value;
    const ok = (l) => { const st = w.lineStats[l.id]; return st && !st.bus && !st.twin && !l.twinOf && st.status === 'running' && st.trainsNeeded.every((n) => n >= 4); };
    const l = w.save.lines.filter(ok)[6];
    return l ? { id: l.id, num: l.num } : null;
  });
  check(!!hl, `a line for headways (${hl?.num})`);
  const hwState = (id) => S(({ tw }, id) => { const w = tw.world.value, st = w.lineStats[id], l = w.save.lines.find((x) => x.id === id); return { weekend: st.weekend, auto: st.autoS, rest: st.restTph, week: [l.tph.high, l.tph.medium, l.tph.low] }; }, id);
  await S(({ w }, num) => w.edit({ op: 'setHeadway', line: num, seconds: [900, 1200, 1800] }), hl.num);
  await settle((id) => JSON.stringify(window.tw.world.value.lineStats[id]?.autoS) === '[900,1200,1800]', hl.id).catch(() => {});
  await page.waitForTimeout(4000);
  const hw0 = await hwState(hl.id);
  console.log('headway line', JSON.stringify(hw0));
  check(hw0.week.every((t, i) => Math.abs(3600 / t - [900, 1200, 1800][i]) < 10), `weekdays every 15/20/30 min: ${hw0.week.map((t) => (3600 / t).toFixed(0))} s`);
  for (const [share, want] of [[70, [1260, 1680, 2520]], [50, [1800, 2400, 3600]], [150, [600, 780, 1200]]]) {
    await S(({ w }, a) => w.edit({ op: 'weekendShare', line: a.num, share: a.share }), { num: hl.num, share });
    await settle((a) => window.tw.world.value.lineStats[a.id]?.weekend?.[0] === a.share, { id: hl.id, share }).catch(() => {});
    await page.waitForTimeout(2500);
    const s = await hwState(hl.id);
    const runs = s.rest.map((t) => (t > 0 ? 3600 / t : 0));
    check(runs.every((h, i) => h >= want[i] - 1 && h <= want[i] * 1.005 + 1), `${share}%: weekends every ${runs.map((h) => h.toFixed(0))} s, want ${want}`);
    check(JSON.stringify(s.week) === JSON.stringify(hw0.week), `${share}%: weekdays as they were (${s.week.map((t) => (3600 / t).toFixed(0))} s)`);
    if (share === 70) {
      await S(({ tw }, id) => { tw.selection.value = { kind: 'line', line: id }; }, hl.id);
      await page.click('button[data-tip="Detailed schedule"]');
      await page.waitForSelector('dialog.sched-panel[open]', { timeout: 10000 });
      await page.waitForTimeout(800);
      const shown = await page.$$eval('dialog.sched-panel table.sched tr', (rs) => { let sec = '', out = []; for (const r of rs) { if (r.classList.contains('sched-day')) { sec = r.textContent.startsWith('Weekends') ? 'weekends' : 'weekdays'; continue; } if (sec === 'weekends') out.push([...r.cells].map((c) => c.textContent).join('|')); } return out; });
      const mins = shown.map((x) => +(/\|(\d+(?:\.\d+)?) min/.exec(x)?.[1] ?? NaN));
      check(mins.length === 3 && mins.every((m, i) => Math.abs(m - want[i] / 60) < 0.2), `70%: the panel's Weekends rows show ${JSON.stringify(shown)}`);
      await (await page.$('dialog.sched-panel .sp-left'))?.screenshot({ path: join(OUT, 't3521_headway.png') });
      await page.keyboard.press('Escape');
      await page.waitForTimeout(400);
    }
  }

  // dark
  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await S(({ tw }, id) => { tw.selection.value = { kind: 'line', line: id }; }, pick.id);
  await page.click('button[data-tip="Detailed schedule"]');
  await page.waitForSelector('dialog.sched-panel[open]', { timeout: 10000 });
  await page.waitForTimeout(1000);
  await (await page.$('dialog.sched-panel .sp-left'))?.screenshot({ path: join(OUT, 't3521_custom_dark.png') });
} finally {
  await chrome.close();
}
const real = errors.filter((e) => !/WebGL|GPU|ResizeObserver/.test(e));
check(real.length === 0, `no page errors${real.length ? ': ' + real.slice(0, 3).join(' | ') : ''}`);
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
