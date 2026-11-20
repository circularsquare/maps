// T-3603: a weekend or weekday headway edit in the timetables window never shows a decimal headway ("6.34 min")
// on the way to the sim's stepped one. A 3-stop line on Headway; every animation frame the window's headways are
// read; weekend headway steps, percent steps and weekday headway steps are pressed at all three levels.
//   node trainworld/checks/t3603-weekend-flash.mjs [dev url]     (default localhost:6201)
import { openGame } from './lib/play.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6201/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const C = [-73.93, 40.66];
const P = (dx, dy) => [C[0] + dx / (111.32 * Math.cos((C[1] * Math.PI) / 180)), C[1] + dy / 111.32]; // km east, north

const g = await openGame({ url: BASE, city: 'nyc', mode: 'sandbox', ui: { shade: 'light', v: 2 }, prefix: 'tw-t3603-' });
try {
  await g.act.camera(C, 12.5);
  await g.act.tool('track', { type: 'regular', tracks: 2, snap: { real: false, roads: false, beside: false }, keepPlacing: false });
  const A = P(-4, 0), B = P(0, 0.3), D = P(4, 0);
  const run = await g.act.drawTrack([A, B, D]);
  check(run.ok, 'track drawn');
  for (const p of [A, B, D]) await g.act.placeStation(p);
  await g.act.construct();
  const line = await g.act.newLine([A, B, D], { name: 'Headway', zoom: 15 });
  check(line?.stops.length === 3, 'a 3-stop line built');
  await g.act.setTrains([8, 6, 4]);

  await g.page.evaluate(async () => {
    window.__m = { cc: await import('/src/workers/clockClient.ts'), hol: await import('/src/game/holidays.ts'), days: await import('/src/game/days.ts'), w: await import('/src/game/world.ts') };
  });
  const S = (f, arg) => g.page.evaluate(`(async () => (${f.toString()})({ ...window.__m, tw: window.tw }, ${JSON.stringify(arg ?? null)}))()`);
  const who = await S(({ tw }) => { const l = tw.world.value.save.lines.at(-1); return { id: l.id, num: l.num }; });
  await g.page.click('button[data-tip="Detailed schedule"]');
  await g.page.waitForSelector('dialog.sched-panel[open]', { timeout: 10000 });
  await g.page.locator('dialog.sched-panel button:text-is("Headway")').first().click();
  await g.page.waitForFunction((id) => window.tw.world.value.lineStats[id]?.autoS, who.id, { timeout: 30000 });
  await g.page.waitForTimeout(2500);

  // every animation frame, the headways the dialog shows: any with a decimal ("6.34 min") is the flash
  await g.page.evaluate(() => {
    window.__seen = new Set(); window.__frames = 0; window.__log = []; let last = '';
    const tick = () => {
      const d = document.querySelector('dialog.sched-panel');
      if (d) {
        const txt = d.innerText.replace(/\s+/g, ' ');
        for (const m of txt.matchAll(/\d+\.\d+ min/g)) window.__seen.add(m[0] + ' @' + window.__label);
        const k = (txt.match(/Weekdays High.*?Headways for new/) || [''])[0];
        if (k !== last) { last = k; window.__log.push(window.__label + ' | ' + k); }
      }
      window.__frames++;
      window.__raf = requestAnimationFrame(tick);
    };
    tick();
  });
  const press = async (name, nth = 0) => { await g.page.evaluate((x) => { window.__label = x; }, name + '#' + nth); await g.act.click(name, { within: 'dialog.sched-panel', nth }); await g.page.waitForTimeout(1500); };
  const lvl = [0, 1, 2];
  for (const more of [true, true, true, false, false, false, false, false]) for (const nth of lvl) await press(`${more ? 'More' : 'Less'} Headway on weekends`, nth);
  for (const more of [false, false, true, true, true]) await press(`${more ? 'More' : 'Less'} Weekend service as a share of weekdays`);
  for (const more of [true, false, false, true]) for (const nth of lvl) await press(`${more ? 'More' : 'Less'} headway`, nth);
  for (const more of [true, false]) await press(`${more ? 'More' : 'Less'} Headway on weekends`, 0);
  const res = await g.page.evaluate(() => { cancelAnimationFrame(window.__raf); return { seen: [...window.__seen], frames: window.__frames, log: window.__log }; });
  console.log('frames sampled', res.frames, 'decimal headways seen', JSON.stringify(res.seen));
  if (process.env.LOG) console.log(res.log.join(String.fromCharCode(10)));
  check(res.frames > 100, 'sampled every frame');
  check(res.seen.length === 0, 'no decimal headway shown after any edit', res.seen.join(', '));
  const errs = g.pageErrors();
  check(errs.length === 0, 'no console errors', errs.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await g.close();
}
console.log(fail ? `T-3603 FAILED: ${fail}` : 'T-3603 PASSED');
process.exit(fail ? 1 : 0);
