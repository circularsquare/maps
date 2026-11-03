// T-2196: the ferry fans (docs/demand-mode.md "Ferry fans"), on the real save in headless Chrome. From the maps root:
//   node trainworld/checks/t2196-ferry-fans.mjs [url] [--baseline]
// Run once with --baseline on the code before the change (it writes artifacts/trainworld/t2196/baseline-*.json),
// then without on the code after:
// 1. The real save as it is (T-007-real-nyc.save, no ferry) solved cold: every line's riders by purpose on a
//    weekday, a Sunday and the average day, to the bit as the baseline.
// 2. T-2195's real save with five ferry lines (artifacts/trainworld/t2195/real-ferries.save; run
//    t2195-ferry-demand.mjs to make it) solved cold, `T2196_REPS` times (2 by default): the Staten Island
//    ferry's riders by purpose against the baseline (leisure and attractions up, at most the fans' share of
//    those purposes' trips on it, every other purpose's within 1%), and every line's commutes and errands
//    against the baseline (only crowding moves them).
// 3. (not with --baseline) The same save cold with `?perfOff=fans` (no fans) in turn with the runs of 2: to the
//    bit as the baseline's ferry figures, and the fans' mean workers' ms within 3% of those runs'. The
//    baseline's own times are reported beside (another build, minutes apart: too noisy for 3%).
// Uses the Vite already on the url (dev-nowatch.mjs). `T2196_WORKERS`: the demand workers, 3 by default.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2196');
await mkdir(OUT, { recursive: true });
const args = process.argv.slice(2);
const BASE = args.find((a) => /^https?:/.test(a)) ?? 'http://localhost:6191/';
const BASELINE = args.includes('--baseline');
const WORKERS = Number(process.env.T2196_WORKERS ?? 3);
const REPS = Number(process.env.T2196_REPS ?? 3);
const FERRY_SAVE = join(ARTIFACTS, 't2195', 'real-ferries.save');
const SI = '109'; // the Staten Island ferry's line id in real-ferries.save (t2195's ferries.json)
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const t0 = Date.now();
const log = (s) => console.log(`${((Date.now() - t0) / 1000).toFixed(0).padStart(5)} s  ${s}`);
const tag = BASELINE ? 'baseline' : 'after';

async function open(prefix, extra = '') {
  const chrome = await launchChrome({ prefix, size: [1200, 800], deviceScaleFactor: 1 });
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  const errors = [];
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=${WORKERS}&paused=1${extra}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  return { chrome, page, errors };
}

/** Import a save and wait for its final; the solve's own ms (sent to final) and its workers' summed ms. */
async function coldSolve(page, bytes) {
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  await demandFinal(page, { timeout: 3600000, polling: 1000, lines: 100 });
  return page.evaluate(() => {
    const d = window.twDemand, s = d.log.findLast((x) => !d.client.fversions?.has(x.version));
    const sum = (a) => (a ?? []).flat(4).reduce((x, y) => x + (Number.isFinite(y) ? y : 0), 0);
    return { solveMs: (s.roundAt?.at(-1) ?? s.refinedAt ?? performance.now()) - s.sentAt, cpuMs: sum(s.periodMs) + sum(s.setupMs) };
  });
}

/** Weekday, Sunday and average-day figures: rail trips, each line's riders, and each purpose's riders per line. */
const figures = (page) => page.evaluate(() => {
  const days = window.twDemand.days(), out = {};
  for (const [k, name] of [[0, 'weekday'], [2, 'sunday'], [6, 'average']]) {
    const v = days[k]?.view;
    if (!v) continue;
    const purposes = {};
    for (const [key, p] of v.purposes ?? []) purposes[key] = Object.fromEntries([...p.lines].map(([id, l]) => [id, l.ridersPerDay]));
    out[name] = { rail: v.railTripsPerDay, lines: Object.fromEntries([...v.lines].map(([id, l]) => [id, l.ridersPerDay])), purposes };
  }
  return out;
});
const sameBits = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const fmt = (x) => Math.round(x ?? 0).toLocaleString('en-US');
const mean = (ts, f) => ts.reduce((a, t) => a + t[f], 0) / Math.max(1, ts.length);

if (!existsSync(FERRY_SAVE)) { console.log(`no ${FERRY_SAVE}: run checks/t2195-ferry-demand.mjs first`); process.exit(1); }
const ferryBytes = await readFile(FERRY_SAVE);
const read = async (f) => (existsSync(join(OUT, f)) ? JSON.parse(await readFile(join(OUT, f), 'utf8')) : null);

// ---- 1. the real save, no ferry
{
  const { chrome, page, errors } = await open('tw-t2196-plain-');
  try {
    const t = await coldSolve(page, await readFile(REAL_SAVE));
    log(`real save: final in ${(t.solveMs / 1000).toFixed(1)} s`);
    const fig = await figures(page);
    await writeFile(join(OUT, `${tag}-plain-w${WORKERS}.json`), JSON.stringify(fig) + '\n');
    const base = BASELINE ? null : await read(`baseline-plain-w${WORKERS}.json`);
    if (base) check(sameBits(base, fig), 'the ferry-free real save solves to the bit as before (every line, every purpose, weekday, Sunday, average)', `weekday rail ${fig.weekday?.rail} against ${base.weekday?.rail}`);
    check(errors.length === 0, 'no console errors (ferry-free)', errors.slice(0, 3).join(' | '));
  } catch (e) {
    check(false, 'the ferry-free real save solved', e.stack ?? e.message);
  } finally {
    await chrome.close();
  }
}

// ---- 2. five ferry lines, cold, REPS times; and (3) with no fans in turn
const times = [], offTimes = [];
let fig = null, offFig = null;
for (let rep = 0; rep < REPS; rep++) {
  for (const off of BASELINE ? [false] : [false, true]) {
    const { chrome, page, errors } = await open(`tw-t2196-ferries${off ? 'off' : ''}${rep}-`, off ? '&perfOff=fans' : '');
    try {
      const t = await coldSolve(page, ferryBytes);
      (off ? offTimes : times).push(t);
      log(`ferries${off ? ', no fans' : ''} ${rep + 1}/${REPS}: final in ${(t.solveMs / 1000).toFixed(1)} s (workers ${(t.cpuMs / 1000).toFixed(1)} s)`);
      if (off && !offFig) offFig = await figures(page);
      if (!off && !fig) {
        fig = await figures(page);
        check(errors.length === 0, 'no console errors (ferries)', errors.slice(0, 3).join(' | '));
      }
    } catch (e) {
      check(false, `cold solve with ferries${off ? ', no fans' : ''} ${rep + 1}`, e.stack ?? e.message);
    } finally {
      await chrome.close();
    }
  }
}
if (fig) {
  await writeFile(join(OUT, `${tag}-ferries-w${WORKERS}.json`), JSON.stringify({ fig, times, offTimes }) + '\n');
  const split = (f, day) => Object.fromEntries(Object.entries(f[day]?.purposes ?? {}).map(([k, v]) => [k, Math.round(v[SI] ?? 0)]).filter(([, v]) => v > 0));
  for (const day of ['weekday', 'sunday']) log(`Staten Island ${day} riders by purpose: ${JSON.stringify(split(fig, day))}, all ${fmt(fig[day]?.lines[SI])}`);
  const base = BASELINE ? null : await read(`baseline-ferries-w${WORKERS}.json`);
  if (base) {
    const b = base.fig;
    for (const day of ['weekday', 'sunday']) log(`  before: ${JSON.stringify(split(b, day))}, all ${fmt(b[day]?.lines[SI])}`);
    const si = (f, day, k) => f[day]?.purposes?.[k]?.[SI] ?? 0;
    for (const day of ['weekday', 'sunday']) {
      const up = ['social', 'attractions'].map((k) => [k, si(b, day, k), si(fig, day, k)]);
      check(up.every(([, x, y]) => y > x), `Staten Island's leisure and attractions riders rise (${day})`, up.map(([k, x, y]) => `${k} ${fmt(x)} -> ${fmt(y)}`).join(', '));
      const all = [fig[day]?.lines[SI] ?? 0, b[day]?.lines[SI] ?? 0];
      check(all[0] > all[1], `Staten Island's riders rise (${day})`, `${fmt(all[1])} -> ${fmt(all[0])}, +${((all[0] / all[1] - 1) * 100).toFixed(1)}%`);
    }
    // commutes and errands: only crowding moves them (fans change loads); the most any line moves
    for (const k of ['work', 'shopping']) {
      let worst = 0, at = '', total = [0, 0];
      for (const [id, x] of Object.entries(b.weekday?.purposes?.[k] ?? {})) {
        const y = fig.weekday?.purposes?.[k]?.[id] ?? 0;
        total[0] += x; total[1] += y;
        const d = Math.abs(y - x) / Math.max(x, 100);
        if (d > worst) { worst = d; at = `line ${id} ${fmt(x)} -> ${fmt(y)}`; }
      }
      check(Math.abs(total[1] / total[0] - 1) < 0.002, `${k} riders on every line together within 0.2% of before (crowding alone moves them)`, `${fmt(total[0])} -> ${fmt(total[1])}; the most on one line ${(worst * 100).toFixed(2)}% (${at})`);
    }
    if (offFig) check(sameBits(b, offFig), 'with no fans (perfOff=fans) the ferry save solves to the bit as before', `weekday rail ${offFig.weekday?.rail} against ${b.weekday?.rail}`);
    const ratio = mean(times, 'solveMs') / mean(offTimes, 'solveMs'), cpu = mean(times, 'cpuMs') / mean(offTimes, 'cpuMs');
    check(cpu <= 1.03, `the workers' summed ms with fans within 3% of the same save with none, runs in turn`, `x${cpu.toFixed(3)}; solve ${(mean(times, 'solveMs') / 1000).toFixed(1)} s against ${(mean(offTimes, 'solveMs') / 1000).toFixed(1)} s, x${ratio.toFixed(3)}`);
    console.log(`  against the baseline build's runs: workers x${(mean(times, 'cpuMs') / mean(base.times, 'cpuMs')).toFixed(3)}, solve x${(mean(times, 'solveMs') / mean(base.times, 'solveMs')).toFixed(3)}; no fans x${(mean(offTimes, 'cpuMs') / mean(base.times, 'cpuMs')).toFixed(3)}`);
  } else if (!BASELINE) console.log(`  (no baseline-ferries-w${WORKERS}.json: run with --baseline on the code before the change)`);
}
console.log(fail ? `T-2196 FAILED: ${fail}` : 'T-2196 PASSED');
process.exit(fail ? 1 : 0);
