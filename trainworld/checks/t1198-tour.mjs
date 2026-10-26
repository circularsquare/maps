// T-1198: a tour of unbuilt cities in a new game leaves no zone open, and memory settles.
//   node trainworld/checks/t1198-tour.mjs [url]      (from the maps root)
// With no url it starts its own Vite (checks/dev-nowatch.mjs, port TOUR_PORT or 6194). A new game; for
// each of the cities in TOUR (N of them, default 16): the camera to it at zoom 10.5, its bare figures
// waited for, the zone selected (its market opens), on to the next. Then the camera out to zoom 6 over
// the sea, and SETTLE_S seconds (default 75, past the 60 s close) for the sweep. Per stop and at the end:
// the Chrome processes' private memory (Windows, by profile path), the page's JS heap, each worker's
// WASM memory, and what the demand client still holds. Fails when, after settling, any zone or market
// is still open bare or in every worker, or a solving worker's WASM passed WASM_MAX MB (default 1300:
// two market compositions of about 450 MB and room; before T-1198 it passed 1,400 by the sixth city
// and kept growing).
// QS adds URL switches (QS=&perfOff=close: nothing closes). Writes ../artifacts/trainworld/t1198/tour-LABEL.json.
import { writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { launchChrome, collectErrors, startVite } from './lib.mjs';

const OUT = process.env.OUT ?? fileURLToPath(new URL('../../artifacts/trainworld/t1198/', import.meta.url));
const LABEL = process.env.LABEL ?? 'run';
const QS = process.env.QS ?? '';
const SETTLE_S = Number(process.env.SETTLE_S ?? 75);
const TOUR = [
  ['London', -0.12, 51.5], ['Paris', 2.35, 48.86], ['Brussels', 4.35, 50.85], ['Amsterdam', 4.9, 52.37],
  ['Cologne', 6.96, 50.94], ['Frankfurt', 8.68, 50.11], ['Munich', 11.58, 48.14], ['Vienna', 16.37, 48.21],
  ['Prague', 14.42, 50.08], ['Berlin', 13.4, 52.52], ['Hamburg', 10.0, 53.55], ['Copenhagen', 12.57, 55.68],
  ['Zurich', 8.54, 47.37], ['Milan', 9.19, 45.46], ['Lyon', 4.84, 45.76], ['Barcelona', 2.17, 41.39],
  ['Madrid', -3.7, 40.42], ['Lisbon', -9.14, 38.72], ['Rome', 12.5, 41.9], ['Warsaw', 21.01, 52.23],
].slice(0, Number(process.env.N ?? 16));
const SEA = [-20, 47];
const MB = 1048576;

/** private bytes and working set of the Chrome processes started on `profile`, by type */
function chromeMemory(profile) {
  const needle = profile.replace(/'/g, "''");
  const ps = `Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like '*${needle}*' } | ForEach-Object { $t = if ($_.CommandLine -match '--type=(\\S+)') { $matches[1] } else { 'browser' }; if ($_.CommandLine -match '--utility-sub-type=(\\S+)') { $t = $t + ':' + $matches[1] }; '{0} {1} {2} {3}' -f $_.ProcessId, $t, $_.PrivatePageCount, $_.WorkingSetSize }`;
  const out = execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], { encoding: 'utf8' });
  const procs = out.trim().split(/\r?\n/).filter(Boolean).map((l) => { const [pid, type, priv, ws] = l.trim().split(' '); return { pid: +pid, type, privMB: Math.round(+priv / MB), wsMB: Math.round(+ws / MB) }; });
  const by = {};
  for (const p of procs) { const k = p.type.replace(/:.*$/, (m) => (/network|storage|audio/.test(m) ? ':other' : m)); by[k] = (by[k] ?? 0) + p.privMB; }
  return { privMB: procs.reduce((a, p) => a + p.privMB, 0), renderer: procs.filter((p) => p.type === 'renderer').reduce((a, p) => a + p.privMB, 0), gpu: by['gpu-process'] ?? 0, by, procs };
}

async function run(BASE) {
  const { ctx, close, profile } = await launchChrome({ prefix: 'tw-t1198-', size: [1400, 900] });
  const errors = [];
  const out = { label: LABEL, base: BASE, qs: QS, stops: [], errors };
  try {
    const page = await ctx.newPage();
    await page.addInitScript(() => {
      try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday', commuters: true, commuterEnd: 'home' }, v: 2 })); } catch { /* */ }
    });
    collectErrors(page, errors, /favicon|openfreemap|AJAXError|noritetsu/, /maplibre-gl/);
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('Performance.enable');
    await page.goto(`${BASE}?new=1&debug=1${QS}&v=${Date.now()}#10.5/${TOUR[0][2]}/${TOUR[0][1]}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw?.map, undefined, { timeout: 180000 });
    await page.waitForFunction(() => window.twDemand.client.status === 'ready' || window.twDemand.client.bareOpen?.size > 0, undefined, { timeout: 180000 }).catch(() => {});
    await page.waitForTimeout(5000);
    const measure = async (where) => {
      await cdp.send('HeapProfiler.collectGarbage').catch(() => {});
      const { metrics } = await cdp.send('Performance.getMetrics');
      const m = (n) => Math.round((metrics.find((x) => x.name === n)?.value ?? 0) / MB);
      const st = await page.evaluate(() => {
        const c = window.twDemand.client;
        const size = (x) => (x?.size ?? x?.length ?? 0);
        return {
          wasmMB: c.start.workers.map((w) => Math.round((w?.memBytes ?? 0) / 1048576)),
          opened: [...c.opened], bare: [...c.bareOpen.keys()], bareKept: size(c.bareKept), cells: size(c.cells ?? c.cityCellsCache),
          loads: c.start.loads.length, view: window.twDemand.activeCity.peek(),
          held: Object.fromEntries(Object.entries(c).filter(([, v]) => v instanceof Map || v instanceof Set || Array.isArray(v)).map(([k, v]) => [k, size(v)]).filter(([, n]) => n > 0)),
        };
      });
      const proc = profile ? chromeMemory(profile) : null;
      const r = { where, jsHeapMB: m('JSHeapUsedSize'), jsHeapTotalMB: m('JSHeapTotalSize'), ...st, proc: proc && { privMB: proc.privMB, renderer: proc.renderer, gpu: proc.gpu, by: proc.by } };
      out.stops.push(r);
      console.log(`${where.padEnd(14)} chrome ${String(proc?.privMB ?? '?').padStart(5)} MB (renderer ${proc?.renderer ?? '?'}, gpu ${proc?.gpu ?? '?'})  js ${r.jsHeapMB}/${r.jsHeapTotalMB}  wasm ${r.wasmMB.join(',')}  open ${r.opened.length} bare ${r.bare.length} kept ${r.bareKept}`);
      return r;
    };
    await measure('start');
    for (const [name, lng, lat] of TOUR.slice(1)) {
      await page.evaluate(([at]) => { window.tw.map.jumpTo({ center: at, zoom: 10.5 }); }, [[lng, lat]]);
      // its bare figures, or a minute
      await page.waitForFunction(() => { const c = window.twDemand.client, id = window.twDemand.activeCity.peek(); return c.bareOpen.has(id) || c.opened.has(id); }, undefined, { timeout: 60000 }).catch(() => console.log(`  ${name}: no bare figures in a minute`));
      const id = await page.evaluate(() => window.twDemand.activeCity.peek());
      await page.evaluate((city) => window.tw.select({ kind: 'city', city }), id);
      await page.waitForTimeout(6000);
      await measure(name);
    }
    await page.evaluate(() => window.tw.select(null));
    await page.evaluate(([at]) => { window.tw.map.jumpTo({ center: at, zoom: 6 }); }, [SEA]);
    await page.waitForTimeout(SETTLE_S * 1000);
    out.end = await measure('settled');
  } finally {
    await close();
  }
  return out;
}

const given = process.argv[2];
const vite = given ? null : await startVite(Number(process.env.TOUR_PORT ?? 6194));
let out;
try {
  out = await run(given ?? vite.url);
} finally {
  if (vite) vite.child.kill();
}
await mkdir(OUT, { recursive: true });
await writeFile(join(OUT, `tour-${LABEL}.json`), JSON.stringify(out, null, 1));
const e = out.end;
const WASM_MAX = Number(process.env.WASM_MAX ?? 1300);
const solvingMax = Math.max(...out.stops.flatMap((s) => s.wasmMB.slice(0, -1)));
const bad = [
  ...(solvingMax > WASM_MAX ? [`a solving worker reached ${solvingMax} MB of WASM`] : []),
  ...(e.opened.length ? [`still open in every worker: ${e.opened.join(', ')}`] : []),
  ...(e.bare.length ? [`still open bare: ${e.bare.join(', ')}`] : []),
  ...out.errors.slice(0, 3),
];
console.log(`${bad.length ? 'FAIL' : 'PASS'}  a tour of ${TOUR.length} unbuilt cities closes every zone (chrome ${out.stops[0].proc?.privMB} -> max ${Math.max(...out.stops.map((s) => s.proc?.privMB ?? 0))} -> ${e.proc?.privMB} MB; solving worker WASM at most ${solvingMax} MB)${bad.length ? ': ' + bad.join('; ') : ''}`);
process.exit(bad.length ? 1 : 0);
