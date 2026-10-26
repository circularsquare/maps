// T-1372 (T-264 stage 1): superholidays.
// Node: the calendar against `REF`'s (default HEAD) over every country and the seven game years, where
// only days away and Japan's Obon may change kind; China's Spring Festival and Golden Week and Japan's
// Golden Week and Obon have days away, which run as the away kind (3).
// Page: the real save on BASE (default its own Vite on PORT, 6172) and on BASELINE (default 8800, the
// last commit's build; BASELINE=none skips it): the weekday, Saturday, Sunday and average figures equal
// the baseline's to the bit; the away day's riders by purpose are a Sunday's times days.rs `AWAY_RATE`;
// the day views' time with the away day and without; the final solve's time on both.
//   node trainworld/checks/t1372-superholidays.mjs [base]      (from the maps root; about 3 minutes)
import { readFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { MAPS, REAL_SAVE, NOISE, launchChrome, collectErrors, importSave, demandFinal, startVite, loadApp } from './lib.mjs';

let failed = 0;
const report = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name} ${detail}`); };

// ---- the calendar
const cal = await loadApp('src/game/holidayCalendar.ts');
const { dateName } = await loadApp('src/game/days.ts');
const NEW = JSON.parse(readFileSync(new URL('../app/src/game/holidays.json', import.meta.url), 'utf8'));
const OLD = JSON.parse(execFileSync('git', ['show', `${process.env.REF ?? 'HEAD'}:trainworld/app/src/game/holidays.json`], { cwd: MAPS, maxBuffer: 1 << 27 }).toString());
const YEARS = 7 * 365;
let away = 0, obon = 0;
const other = [];
for (const c of Object.keys(NEW.countries)) {
  const k1 = cal.kindOfIn(NEW, c), k0 = cal.kindOfIn(OLD, c);
  for (let d = 1; d <= YEARS; d++) {
    const a = k1(d), b = k0(d);
    if (a === b) continue;
    if (a === 3) away++;
    else if (c === 'JP' && cal.holidayName(NEW, c, d)?.includes('Obon')) obon++;
    else if (other.length < 10) other.push(`${c} day ${d}: ${b} -> ${a}`);
  }
}
report('calendar: only days away and Obon change kind', other.length === 0, `(${away} country-days away, ${obon} Obon days over 7 years in ${Object.keys(NEW.supers).length} countries${other.length ? '; ' + other.join(', ') : ''})`);
const show = (r) => `${r.name} out ${dateName(r.out)}, away ${dateName(r.away)} to ${dateName(r.back - 1)}, back ${dateName(r.back)}${r.end - r.back > 1 ? ' and ' + dateName(r.end - 1) : ''}`;
for (const [c, want] of [['CN', ['Spring Festival', 'Golden Week']], ['JP', ['Golden Week', 'Obon']]]) {
  let ok = true;
  const seen = [];
  for (let y = 0; y < 7; y++) {
    const runs = cal.superRuns(NEW, c, 1 + 365 * y, 366 + 365 * y);
    for (const name of want) {
      const r = runs.find((x) => x.name === name);
      if (!r || r.back <= r.away) { ok = false; continue; }
      for (let d = r.away; d < r.back; d++) if (cal.kindOfIn(NEW, c)(d) !== 3) ok = false;
      for (const d of [r.out, r.back]) if (cal.kindOfIn(NEW, c)(d) === 3) ok = false;
      if (y === 0) seen.push(show(r));
    }
  }
  report(`calendar: ${c} ${want.join(' and ')} run as days away`, ok, `(first year: ${seen.join('; ')})`);
}

// ---- the page
const AWAY_RATE = { work: 1, shopping: 0.7, social: 0.7, school: 0.7, airport: 1, medical: 0.7, attractions: 1.5, longer: 1 };
const port = Number(process.env.PORT ?? 6172);
let vite = null, base = process.argv[2];
if (!base) { vite = await startVite(port); base = vite.url; }
const BASELINE = process.env.BASELINE ?? 'http://localhost:8800/trainworld/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t1372-' });
const errors = [];

async function realSave(url, mine) {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  // map glyphs the font server lacks answer 404, as in smoke
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.goto(`${url}${url.includes('?') ? '&' : '?'}debug=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
  if (await page.$('.start-menu .start-new:not(.new-game-start)')) await page.click('.start-menu .start-new:not(.new-game-start)'); // T-1441: a first visit opens on the sheet
  await page.click('.start-menu .new-game-cities [data-city="nyc"]');
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu'), undefined, { timeout: 180000 });
  if (mine) {
    // the day views' input, to time them with the away day and without
    await page.evaluate(() => {
      const c = window.twDemand.client, inSums = c.inSums.bind(c);
      c.inSums = (q, ...rest) => { if (q.kind === 'days') window.__daysInput = q.input; return inSums(q, ...rest); };
    });
  }
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 250, lines: 100, timeout: 600000 });
  await page.waitForFunction(() => { const d = window.twDemand.days(); return d.every((x) => x.view?.final && x.view.version === d[0].view.version); }, undefined, { timeout: 60000 });
  return page;
}

/** every figure of a day's view, as one string of exact numbers, and a few totals */
const figures = (page, ks) => page.evaluate((ks) => {
  const s = (x) => (typeof x === 'number' ? String(x) : x && typeof x.length === 'number' ? Array.from(x, s).join(',') : '');
  return ks.map((k) => {
    const v = window.twDemand.days()[k].view, parts = [s(v.railTripsPerDay), s(v.tripsPerDay), s([v.split.train, v.split.walk, v.split.drive])];
    for (const [id, l] of [...v.lines].sort()) parts.push(id, s(l.ridersPerDay), s(l.boardings), ...l.loads.map(s));
    for (const [id, st] of [...v.stations].sort()) parts.push(id, s(st.boardings), s(st.alightings));
    const by = {};
    for (const [p, d] of [...(v.purposes ?? [])].sort()) {
      by[p] = 0;
      for (const [id, l] of [...d.lines].sort()) { parts.push(p, id, s(l.ridersPerDay)); by[p] += l.ridersPerDay; }
    }
    return { all: parts.join('|'), rail: v.railTripsPerDay, by };
  });
}, ks);

const solveMs = (page) => page.evaluate(() => { const s = window.twDemand.log.at(-1); return s.refinedAt !== undefined ? Math.round(s.refinedAt - s.sentAt) : null; });

try {
  const page = await realSave(base, true);
  const mine = await figures(page, [0, 1, 2, 4, 3]);
  const mineMs = await solveMs(page);
  // the away day against Sunday, purpose by purpose
  const sun = mine[2].by, aw = mine[4].by, bad = [];
  for (const [p, n] of Object.entries(sun)) {
    const want = n * (AWAY_RATE[p] ?? NaN);
    if (!(Math.abs((aw[p] ?? 0) - want) <= 1e-3 * Math.max(1, want))) bad.push(`${p} ${aw[p]} vs ${want}`);
  }
  const ratio = Object.entries(sun).map(([p, n]) => `${p} ${n > 0 ? ((aw[p] ?? 0) / n).toFixed(3) : '-'}`).join(', ');
  report('away day: a Sunday by purpose times AWAY_RATE', bad.length === 0 && Object.keys(sun).length > 3, `(${ratio}; rail ${Math.round(mine[2].rail)} on Sunday, ${Math.round(mine[4].rail)} away${bad.length ? '; ' + bad.join(', ') : ''})`);
  // the day views with and without the away day, same input, five runs each
  const t = await page.evaluate(async () => {
    const c = window.twDemand.client, input = window.__daysInput;
    if (!input) return null;
    const run = async (days) => { const a = performance.now(); await c.inSums({ kind: 'days', input: { ...input, days } }, [], () => null); return performance.now() - a; };
    const med = async (days) => { const xs = []; for (let i = 0; i < 5; i++) xs.push(await run(days)); return xs.sort((a, b) => a - b)[2]; };
    return { five: await med([0, 1, 2, 3, 4]), four: await med([0, 1, 2, 4]) };
  });
  report('day views timed', !!t, t ? `(${t.four.toFixed(0)} ms without the away day, ${t.five.toFixed(0)} ms with it, in the sums worker)` : '(no input caught)');
  if (BASELINE !== 'none') {
    const bpage = await realSave(BASELINE, false);
    const was = await figures(bpage, [0, 1, 2, 3]);
    const wasMs = await solveMs(bpage);
    const names = ['weekday', 'Saturday', 'Sunday', 'average'];
    const diff = names.filter((_, i) => was[i].all !== mine[i].all);
    report('ordinary days equal the baseline to the bit', diff.length === 0, `(${names.map((n, i) => `${n} rail ${Math.round(mine[i].rail)}`).join(', ')}${diff.length ? '; differ: ' + diff.join(', ') : ''})`);
    report('final solve time', true, `(${mineMs} ms here, ${wasMs} ms on the baseline; one run each)`);
    await bpage.close();
  }
  await page.close();
} catch (e) {
  report('page', false, e.message.split('\n')[0]);
} finally {
  await close();
  vite?.child.kill();
}
report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
process.exit(failed ? 1 : 0);
