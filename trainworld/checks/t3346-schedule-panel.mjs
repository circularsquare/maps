// T-3346 stage 1: the schedule panel. On the real save: a line's panel opens from the line panel's
// schedule icon; the departures board's times at three stations match the timetable chart's trains
// (each trip's leave time worked out from the profile the chart draws) to the second, and the board
// shows one minute for each; a stepper in the panel changes the line and the board follows; Escape
// closes it. Then two lines sharing track are grouped and the group's chart opens from the group
// view with the second line drawn on it. Shots light and dark. From the maps root (or a tree's root):
//   node trainworld/checks/t3346-schedule-panel.mjs [url] [shots dir]   (about two minutes; default localhost:6180)
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, GL, launchChrome, collectErrors, importSave, buildLine } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6180/';
const OUT = process.argv[3] ?? ARTIFACTS;
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3346-', size: [1400, 900], args: GL, deviceScaleFactor: 1 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { if (!sessionStorage.getItem('t3346')) { sessionStorage.setItem('t3346', '1'); localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 14 })); } } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?timetables&debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 4, undefined, { timeout: 120000 });
  await page.waitForTimeout(4000);
  await page.evaluate(async () => {
    let prof = null;
    try { prof = await import('/src/map/profiles.ts'); } catch {}
    window.__m = { sp: await import('/src/ui/inspector/SchedulePanel.tsx'), share: await import('/src/ui/inspector/mareyShare.ts'), live: await import('/src/map/live.ts'),
      cc: await import('/src/workers/clockClient.ts'), clock: await import('/src/game/clock.ts'), w: await import('/src/game/world.ts'), ui: await import('/src/game/ui.ts'), prof };
  });
  // the page's CSP refuses eval, so each step goes as an expression through the protocol
  const S = (f, arg) => page.evaluate(`(async () => (${f.toString()})({ ...window.__m, tw: window.tw }, ${JSON.stringify(arg ?? null)}))()`);
  const shade = async (s) => { await page.evaluate((s) => { window.tw.shade.value = s; }, s); await page.waitForTimeout(500); };

  // 1. a line: running rail, no twin, the most stops up to 30; and the running line sharing most of its track
  const pick = await S(({ tw, live, share }) => {
    const w = tw.world.value, net = live.live.overlay.renderer.net;
    const ok = (l) => { const st = w.lineStats[l.id]; return st && !st.bus && !st.twin && !l.twinOf && st.status === 'running' && l.stops.length >= 4 && l.stops.length <= 30 && st.trainsNeeded.some((n) => n > 0); };
    const ls = w.save.lines.filter(ok).sort((a, b) => b.stops.length - a.stops.length);
    const byNum = new Map(w.save.lines.map((l) => [l.num, l]));
    let best = null, seen = 0;
    for (const l of ls) {
      const st = w.lineStats[l.id], m = share.sharedStretches(net, l.num, st.stopS[0][0], st.stopS[0][l.stops.length - 1]);
      seen += m.size;
      for (const [o, list] of m) {
        const ol = byNum.get(o);
        if (!ol || !ok(ol) || ol.group || l.group) continue;
        const len = list.reduce((s, x) => s + x.b - x.a, 0);
        if (!best || len > best.len) best = { a: l.id, b: ol.id, len };
      }
    }
    const a = ls[0];
    return { line: a.id, num: a.num, name: a.name, stops: a.stops.length, pair: best, candidates: ls.length, shared: seen, grouped: w.save.lines.filter((l) => l.group).length };
  });
  console.log('line', JSON.stringify(pick));
  check(!!pick.line, `a running line to show: ${pick.name} (${pick.stops} stops)`);

  // 2. open the panel from the line panel's icon
  await S(({ tw }, id) => { tw.selection.value = { kind: 'line', line: id }; }, pick.line);
  await page.waitForSelector('button[data-tip="Detailed schedule"]', { timeout: 30000 });
  await page.click('button[data-tip="Detailed schedule"]');
  await page.waitForSelector('dialog.sched-panel[open]', { timeout: 10000 });
  await page.waitForSelector('.sp-board .sp-row', { timeout: 30000 });
  await page.waitForTimeout(800);

  // 3. the board against the chart's trains at three stations: the first, a middle one, the last
  const cmp = await S(async ({ sp, cc, clock, live, prof, tw, w: wm }, id) => {
    const w = tw.world.value, l = w.save.lines.find((x) => x.id === id), st = w.lineStats[id], net = live.live.overlay.renderer.net;
    const n = l.stops.length, shift = st.shift ?? 0;
    const now = clock.clock.minute.value * 60, t0 = Math.floor((now + shift) / 3600) * 3600 - shift, t1 = t0 + 86400;
    const runOf = (p) => (prof?.profileRun ? prof.profileRun(p) : Math.floor(p / 3) % 2);
    const trips = await cc.queryClock({ what: 'lineTrips', line: l.num, t0, t1 });
    const out = [];
    for (const station of [l.stops[0], l.stops[Math.floor(n / 2)], l.stops[n - 1]]) {
      const board = await sp.stationDepartures(l, st, station, t0, t1);
      const chart = [];
      for (let i = 0; i < trips.length; i += 4) {
        const p = trips[i], dep = trips[i + 1], run = runOf(p);
        const first = net.meta[p * 4], count = net.meta[p * 4 + 1], tripS = net.meta[p * 4 + 2];
        for (let j = 0; j < n; j++) {
          if (l.stops[j] !== station) continue;
          const k = run === 0 ? j : n - 1 - j;
          if (k === n - 1) continue;
          const off = st.stopS[run][k];
          // the train leaves as it first moves on from the stop: after its stand there, or at once from the first stop
          let leave = null, at = k === 0;
          for (let q = first; q < first + count && leave === null; q++) {
            const s = net.phases[q * 4 + 1], v = net.phases[q * 4 + 2], a = net.phases[q * 4 + 3];
            const rest = Math.abs(v) < 0.01 && a <= 0;
            if (!at && rest && Math.abs(s - off) < 1.5) at = true;
            else if (at && !rest) leave = net.phases[q * 4];
          }
          if (leave !== null && dep + leave >= t0 && dep + leave < t1) chart.push(dep + leave);
        }
      }
      chart.sort((a, b) => a - b);
      const bt = (board ?? []).map((d) => d.t);
      let worst = 0;
      for (let i = 0; i < Math.min(bt.length, chart.length); i++) worst = Math.max(worst, Math.abs(bt[i] - chart[i]));
      out.push({ station, name: wm.stationNames(w).get(station) ?? station, board: bt.length, chart: chart.length, worst });
    }
    return out;
  }, pick.line);
  for (const c of cmp) check(c.board > 0 && c.board === c.chart && c.worst < 1, `${c.name}: board ${c.board} departures, chart ${c.chart}, worst gap ${c.worst.toFixed(2)} s`);

  // the board's minutes as shown, station by station through the picker
  for (const c of cmp) {
    await page.selectOption('.sp-tools select', c.station);
    await page.waitForTimeout(700);
    const shown = await page.$$eval('.sp-board .sp-mins > span', (xs) => xs.length);
    check(shown === c.board, `${c.name}: ${shown} minutes on the board for ${c.board} departures`);
  }
  await page.selectOption('.sp-tools select', cmp[1].station);
  await page.waitForTimeout(700);
  for (const s of ['light', 'dark']) {
    await shade(s);
    await page.mouse.move(5, 5);
    await page.screenshot({ path: join(OUT, `t3346-s1-board-${s}.png`) });
  }
  await shade('light');

  // 4. a stepper in the panel: one more train at the level now, and the board follows
  const before = await page.$$eval('.sp-board .sp-mins > span', (xs) => xs.length);
  const steppers = await page.$$('.sched-panel table.sched tr.now .stepper button');
  if (steppers.length >= 2) {
    await steppers[steppers.length - 1].click();
    await page.waitForFunction((b) => document.querySelectorAll('.sp-board .sp-mins > span').length !== b, before, { timeout: 30000, polling: 300 }).catch(() => {});
    const after = await page.$$eval('.sp-board .sp-mins > span', (xs) => xs.length);
    check(after > before, `a train added in the panel: the board goes from ${before} to ${after} departures`);
    const back = await page.$$('.sched-panel table.sched tr.now .stepper button');
    await back[0].click();
    await page.waitForFunction((b) => document.querySelectorAll('.sp-board .sp-mins > span').length === b, before, { timeout: 30000, polling: 300 }).catch(() => {});
  } else check(false, 'the panel has the level steppers');

  // 5. the chart at the panel's width
  await page.click('.sp-tabs .btn:has-text("Chart")');
  await page.waitForSelector('.sched-panel .marey-chart use.run', { timeout: 30000 });
  await page.waitForTimeout(800);
  const cw = await page.$eval('.sched-panel .marey-chart', (e) => [e.getBoundingClientRect().width, e.closest('.sp-right').getBoundingClientRect().width]);
  check(Math.abs(cw[0] - cw[1]) < 2, `the chart is the panel's right side wide: ${Math.round(cw[0])} of ${Math.round(cw[1])} px`);
  for (const s of ['light', 'dark']) {
    await shade(s);
    await page.mouse.move(5, 5);
    await page.screenshot({ path: join(OUT, `t3346-s1-chart-${s}.png`) });
  }
  await shade('light');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(400);
  check(!(await page.$('dialog.sched-panel[open]')), 'Escape closes the panel');
  check(await S(({ tw }) => tw.selection.value?.kind === 'line'), 'and leaves the line selected');

  // 6. a group of two lines sharing track: its chart from the group view. The real save's lines each
  // have their own track, so a line of five stations on Staten Island and a short one over its first three.
  if (!pick.pair) {
    const dLng = 2 / (111.32 * Math.cos((40.58 * Math.PI) / 180));
    const r = await buildLine(page, [0, 1, 2, 3, 4].map((k) => [-74.17 + k * dLng, 40.58]), { name: 'Long', colour: '#2a6fdb', trains: [8, 6, 4], wait: { route: 300, station: 150, settle: 600 } });
    const pair = await S(async ({ tw }, num) => {
      const w = tw.world.value, a = w.save.lines.find((l) => l.num === num);
      const nodes = a.stops.slice(0, 3).map((id) => w.save.stations.find((s) => s.id === id)?.num ?? Number(id));
      const r2 = await tw.client.edit({ op: 'addLine', stops: nodes, name: 'Short', colour: '#e07a1f' });
      if (r2?.line !== undefined) await tw.client.edit({ op: 'setTrainCount', line: r2.line, trains: [6, 4, 2] });
      await new Promise((d) => setTimeout(d, 2500));
      const b = tw.world.value.save.lines.find((l) => l.num === r2?.line);
      return b ? { a: a.id, b: b.id, len: 0 } : null;
    }, r?.line ?? -1);
    console.log('built pair', JSON.stringify(r?.ok), JSON.stringify(pair));
    pick.pair = pair;
  }
  if (pick.pair) {
    const g = await S(async ({ w, tw }, pair) => {
      const ls = tw.world.value.save.lines, a = ls.find((l) => l.id === pair.a), b = ls.find((l) => l.id === pair.b);
      await w.edit({ op: 'lineGroup', lines: [a.num, b.num], group: 'new' });
      await new Promise((r) => setTimeout(r, 1500));
      const gl = tw.world.value.save.lines.find((l) => l.id === pair.a);
      return { group: gl.group ?? null, a: a.name, b: b.name };
    }, pick.pair);
    console.log('group', JSON.stringify(g), 'shared km', (pick.pair.len / 1000).toFixed(1));
    check(g.group !== null, `lines ${g.a} and ${g.b} grouped`);
    await S(({ tw }, group) => { tw.selection.value = { kind: 'group', group }; }, g.group);
    await page.waitForSelector('button[data-tip="Timetable chart"]', { timeout: 30000 });
    await page.click('button[data-tip="Timetable chart"]');
    await page.waitForSelector('dialog.sched-panel[open] .marey-chart use.run', { timeout: 30000 });
    await page.waitForTimeout(1500);
    const runs = await page.$$eval('.sched-panel .marey-layer use.run', (xs) => ({ own: xs.filter((x) => !x.style.getPropertyValue('--run')).length, group: xs.filter((x) => x.style.getPropertyValue('--run') && !x.classList.contains('other')).length }));
    check(runs.own > 0 && runs.group > 0, `the group chart draws the axis line's trains (${runs.own}) and the other line's at full strength (${runs.group})`);
    const gw = await page.$eval('.sched-panel .marey-chart', (e) => [e.getBoundingClientRect().width, e.closest('.sp-group').getBoundingClientRect().width, e.closest('.saves-inner').getBoundingClientRect().width]);
    check(Math.abs(gw[0] - gw[1]) < 2 && gw[1] > gw[2] - 60, `the group chart is the panel wide: ${Math.round(gw[0])} of ${Math.round(gw[2])} px`);
    for (const s of ['light', 'dark']) {
      await shade(s);
      await page.mouse.move(5, 5);
      await page.screenshot({ path: join(OUT, `t3346-s1-group-${s}.png`) });
    }
    await shade('light');
    await page.keyboard.press('Escape');
  } else check(false, 'two running lines sharing track to group');

  // 7. on a phone: the two sides one under the other
  await page.setViewportSize({ width: 420, height: 860 });
  await page.waitForTimeout(800);
  await S(({ sp }, id) => { sp.scheduleShown.value = { kind: 'line', line: id }; }, pick.line);
  await page.waitForSelector('dialog.sched-panel[open] .sp-tabs', { timeout: 30000 });
  // the Chart choice of step 5 is kept for the session
  await page.click('.sp-tabs .btn:has-text("Departures")');
  await page.waitForSelector('dialog.sched-panel[open] .sp-board .sp-row', { timeout: 30000 });
  await page.waitForTimeout(800);
  const phone = await page.$eval('dialog.sched-panel', (d) => ({ w: d.getBoundingClientRect().width, scroll: d.querySelector('.saves-inner').scrollWidth - d.querySelector('.saves-inner').clientWidth }));
  check(phone.w <= 420 && phone.scroll <= 1, `on a phone the panel fits the width (${Math.round(phone.w)} px, ${phone.scroll} px sideways)`);
  await page.screenshot({ path: join(OUT, 't3346-s1-phone-light.png') });
} finally {
  await chrome.close();
}
const real = errors.filter((e) => !/Failed to load resource/.test(e));
check(real.length === 0, `no page errors${real.length ? ': ' + real.slice(0, 5).join(' | ') : ''}`);
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
