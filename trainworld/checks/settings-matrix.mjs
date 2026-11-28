// T-4426: the game under non-default settings, which the standing checks (all on the computer defaults) never
// reach. Each scenario is a fresh browser profile with the settings stored (or the device faked), a sandbox New York
// game, one built line, and a demand final. A scenario passes when the settings read back as set, the demand pool
// is the size those settings give, the final comes without a failure, a "Demand stopped" note or a page error,
// and trips are counted. About 1 minute a scenario, 3 cores.
//
//   node trainworld/checks/settings-matrix.mjs [port] [--only name,name] [--force] [--no-ledger]
//   node trainworld/checks/settings-matrix.mjs --last          when it last ran, and whether one is due
//
// Run it after a cpu, memory, loading or worker change: about weekly, at most daily (Anita, 11-27). The date of each
// full run is appended to the main tree's trainworld/notes/settings-check-ledger.jsonl. Within 24 hours of the last
// run it stops (--force overrides); within 7 days it warns and runs. A partial run (--only) is not logged.
// Find more settings in game/ui.ts (DisplaySettings) and workers/deviceMemory.ts; add a scenario below.
import { appendFileSync, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, collectErrors, demandFinal, acrossLine, startVite, NOISE } from './lib.mjs';

const TREE = fileURLToPath(new URL('../../', import.meta.url));
const MAPS = TREE.replace(/[\\/]\.tw-trees[\\/][^\\/]+[\\/]?$/, '/');
const LEDGER = join(MAPS, 'trainworld', 'notes', 'settings-check-ledger.jsonl');
const GLYPH_404 = /\/fonts\/[^/]+\/\d+-\d+\.pbf$/;
const NYC = [-73.98, 40.75];
const DAY = 86400000;

const args = process.argv.slice(2);
const flag = (f) => args.includes(f);
const only = args.includes('--only') ? args[args.indexOf('--only') + 1].split(',') : null;
const port = Number(args.find((a, i) => /^\d+$/.test(a) && args[i - 1] !== '--only') ?? 6196);

function lastRun() {
  if (!existsSync(LEDGER)) return null;
  const rows = readFileSync(LEDGER, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  return rows.at(-1) ?? null;
}
const ago = (r) => (Date.now() - Date.parse(r.at)) / DAY;
const last = lastRun();
if (flag('--last')) {
  if (!last) console.log('settings-matrix: never run; due');
  else console.log(`settings-matrix: last run ${last.at} (${ago(last).toFixed(1)} days ago, ${last.passed ? 'passed' : 'FAILED'}); ${ago(last) >= 7 ? 'due' : 'not due until 7 days'}`);
  process.exit(0);
}
if (last && !only) {
  const d = ago(last);
  if (d < 1 && !flag('--force')) { console.log(`settings-matrix: ran ${(d * 24).toFixed(1)} hours ago; at most daily (--force to run anyway)`); process.exit(3); }
  if (d < 7) console.log(`WARNING settings-matrix: last ran ${d.toFixed(1)} days ago; about weekly is enough`);
}

const PHONE_UA = 'Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36';
/** settings: stored display settings; pool: the solving workers they give (a number, or 'cores:N' for the slider's N
 * within the machine's most); device: how the browser is faked; query: extra page query; expect: display values read back */
const SCENARIOS = [
  { name: 'defaults', note: 'control: the computer defaults, Cores on Auto', settings: {}, pool: 'auto' },
  { name: 'cores-1', note: 'Cores slider at 1', settings: { cores: 1 }, pool: 'cores:1' },
  { name: 'cores-5', note: 'Cores slider at 5', settings: { cores: 5 }, pool: 'cores:5' },
  { name: 'low-memory', note: 'a browser reporting 2 GB of device memory', settings: {}, device: 'lowmem', pool: 1 },
  { name: 'phone', note: 'an Android phone: touch, mobile user agent, its own defaults (intercity off, coarse on)', settings: {}, device: 'phone', pool: 1, expect: { intercityTrips: false, coarseDemand: true } },
  { name: 'intercity-off', note: 'Calculate intercity trips off', settings: { intercityTrips: false }, pool: 'auto', expect: { intercityTrips: false } },
  { name: 'coarse-demand', note: 'coarse demand on', settings: { coarseDemand: true }, pool: 'auto', expect: { coarseDemand: true } },
  { name: '32-bit', note: 'a 32-bit browser (?bits32=1)', settings: {}, query: 'bits32=1', pool: 1 },
];

/** runs in the page before the game: the faked device */
const lowMemInit = () => { Object.defineProperty(Navigator.prototype, 'deviceMemory', { get: () => 2, configurable: true }); };
/** the browser pin adds ?demandWorkers (cores.mjs), which would override the Cores slider and Auto's rule: take it off */
const unpinInit = () => { try { const u = new URL(location.href); if (u.searchParams.has('demandWorkers')) { u.searchParams.delete('demandWorkers'); history.replaceState(history.state, '', u); } } catch {} };

async function runScenario(sc, BASE) {
  const t0 = Date.now();
  const phone = sc.device === 'phone';
  const { ctx, close } = await launchChrome({ prefix: 'tw-settings-', size: phone ? [412, 915] : [1400, 900],
    ...(phone ? { userAgent: PHONE_UA, hasTouch: true, isMobile: true, deviceScaleFactor: 2 } : {}) });
  const errors = [], failed = [];
  try {
    await ctx.addInitScript(unpinInit);
    if (sc.device === 'lowmem') await ctx.addInitScript(lowMemInit);
    await ctx.addInitScript((display) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday', ...display }, v: 16 })); } catch {} }, sc.settings);
    const page = await ctx.newPage();
    collectErrors(page, errors, NOISE, GLYPH_404);
    page.on('requestfailed', (r) => failed.push(r.url().slice(0, 120)));
    const q = sc.query ? '&' + sc.query : '';
    await page.goto(`${BASE}?debug=1&new=1&city=nyc&mode=sandbox&paused=1${q}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    const info = await page.evaluate(() => {
      const d = window.twDemand, s = window.tw.display.value;
      return { display: { cores: s.cores, intercityTrips: s.intercityTrips, coarseDemand: s.coarseDemand }, pool: d.poolSize(), poolN: d.client.poolN, coresMax: d.cores.max(), marketOff: d.marketOff(), hc: navigator.hardwareConcurrency, mem: navigator.deviceMemory ?? null };
    });
    const problems = [];
    for (const [k, v] of Object.entries({ ...(sc.settings.cores !== undefined ? { cores: sc.settings.cores } : {}), ...(sc.expect ?? {}) })) {
      if (info.display[k] !== v) problems.push(`setting ${k} is ${info.display[k]}, wanted ${v}`);
    }
    const wantPool = sc.pool === 'auto' ? null : typeof sc.pool === 'number' ? sc.pool : Math.min(Number(sc.pool.split(':')[1]), info.coresMax);
    if (wantPool !== null && info.pool !== wantPool) problems.push(`pool is ${info.pool}, wanted ${wantPool} (Cores max here ${info.coresMax})`);
    if (info.poolN !== info.pool) problems.push(`the running pool is ${info.poolN}, poolSize says ${info.pool}`);
    if (sc.settings.intercityTrips === false && !info.marketOff) problems.push('intercity trips still on');
    const line = await acrossLine(page, NYC);
    if (line === null) problems.push('no line built');
    else {
      await demandFinal(page, { lines: 0, timeout: 240000 }).catch((e) => { problems.push('no demand final: ' + e.message.split('\n')[0]); });
      const end = await page.evaluate(() => {
        const d = window.twDemand, v = d.view();
        return { status: String(d.client.status), stopped: !!document.querySelector('.demand-stopped-note'), final: !!v?.final, lines: v?.lines?.size ?? 0, trips: v?.tripsPerDay ?? null, rail: v?.railTripsPerDay ?? null };
      });
      if (end.status.startsWith('failed')) problems.push('demand ' + end.status);
      if (end.stopped) problems.push('Demand stopped note shown');
      if (!end.final || end.lines < 1) problems.push(`not final with the line (${JSON.stringify(end)})`);
      if (!(end.trips > 0)) problems.push(`no trips counted (${end.trips})`);
      info.end = end;
    }
    if (errors.length) problems.push('page errors: ' + errors.slice(0, 3).join(' | ') + (failed.length ? ' (failed requests: ' + [...new Set(failed)].slice(0, 3).join(' ') + ')' : ''));
    return { name: sc.name, ok: problems.length === 0, problems, s: Math.round((Date.now() - t0) / 1000), info };
  } catch (e) {
    return { name: sc.name, ok: false, problems: [e.message.split('\n')[0], ...errors.slice(0, 2)], s: Math.round((Date.now() - t0) / 1000) };
  } finally {
    await close();
  }
}

const vite = await startVite(port);
const results = [];
try {
  for (const sc of SCENARIOS) {
    if (only && !only.includes(sc.name)) continue;
    const r = await runScenario(sc, vite.url);
    results.push(r);
    const i = r.info;
    console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${sc.name} (${sc.note}; ${r.s} s)${i ? ` pool ${i.pool}, trips/day ${Math.round(i.end?.trips ?? 0)}, rail ${Math.round(i.end?.rail ?? 0)}` : ''}${r.problems.length ? '\n      ' + r.problems.join('\n      ') : ''}`);
  }
} finally {
  vite.child.kill();
}
const passed = results.every((r) => r.ok);
if (!only && !flag('--no-ledger')) {
  appendFileSync(LEDGER, JSON.stringify({ at: new Date().toISOString(), passed, scenarios: Object.fromEntries(results.map((r) => [r.name, r.ok ? 'pass' : r.problems[0]])) }) + '\n');
  console.log('ledger: ' + LEDGER);
}
console.log(`settings-matrix: ${passed ? 'PASSED' : 'FAILED'} (${results.filter((r) => r.ok).length}/${results.length})`);
process.exit(passed ? 0 : 1);
