// T-1205: a demand worker well above its memory need is restarted when idle, and gives the memory back
// with the figures unchanged to the trip. A save (default the all-China one at the maps root) is loaded
// to its weekday final; a short line is built in each of a few big zones it does not reach (Tokyo, Osaka,
// Seoul), so they open in every worker, and solved; the edits are undone, the network solved again, and
// the sweep closes the zones (60 s). The pool should then restart the workers well above their need and
// solve once more. Passes when that restart came, none began while a solve ran, the workers' WASM after
// is well below that before, the rail trips a day after equal those of the undone network to the trip
// (and the load's), a station is answered, and no console errors.
// Usage (maps root): node trainworld/checks/t1205-memory.mjs [dev server url] [save]. QS adds URL
// switches (QS=&perfOff=cap keeps every solving worker). The all-China save takes about 20 minutes.
// Writes ../artifacts/trainworld/t1205/memory-LABEL.json.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { MAPS, ARTIFACTS, NOISE, launchChrome, collectErrors, importSave, acrossLine } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6125/';
const SAVE = process.argv[3] ?? join(MAPS, 't1156-all-china.save');
const QS = process.env.QS ?? '';
const LABEL = process.env.LABEL ?? 'run';
const ZONES = [[139.70, 35.69], [135.50, 34.69], [126.98, 37.57]];
const MB = 1048576;
let failed = 0;
const report = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

function chromeMB(profile) {
  const needle = profile.replace(/'/g, "''");
  const ps = `Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like '*${needle}*' } | ForEach-Object { $_.PrivatePageCount }`;
  const out = execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], { encoding: 'utf8' });
  return Math.round(out.trim().split(/\r?\n/).filter(Boolean).reduce((a, b) => a + Number(b), 0) / MB);
}

const { ctx, close, profile } = await launchChrome({ prefix: 'tw-t1205-' });
const errors = [], infos = [], out = { label: LABEL, base: BASE, save: SAVE, qs: QS, stops: [], errors, infos };
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0);
const restarts = () => infos.filter((s) => /restarted/.test(s)).length;
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /glyphs/);
  page.on('console', (m) => { if (/restarted to give back memory|solving workers to|panicked/.test(m.text())) { infos.push(`${secs()} s ${m.text()}`); console.log(`      ${secs()} s: ${m.text().slice(0, 240)}`); } });
  await page.goto(`${BASE}${BASE.includes('?') ? '&' : '?'}debug=1&new=1&paused=1${QS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  // a planned restart that begins while the solve in hand has not published its final, or one waits
  await page.evaluate(() => {
    const c = window.twDemand.client;
    window.__t1205 = { inSolve: [] };
    const orig = console.info;
    console.info = (...a) => {
      const s = c.log.at(-1), v = window.twDemand.view();
      if (/restarted to give back memory/.test(String(a[0])) && s && (s.published < 3 || c.waiting || v?.version !== s.version)) window.__t1205.inSolve.push({ version: s.version, published: s.published, waiting: !!c.waiting });
      orig.apply(console, a);
    };
  });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  const final = (after) => page.waitForFunction((after) => {
    const d = window.twDemand, s = d.log.at(-1), v = d.view();
    return String(d.client.status).startsWith('failed') || (!!v && s && v.version === s.version && v.final && !d.client.waiting && v.version > after);
  }, after, { timeout: 1800000, polling: 1000 });
  const state = async (where) => {
    const st = await page.evaluate(async () => {
      const d = window.twDemand, v = d.view(), c = d.client;
      const station = window.tw.world.value.save.stations[37]?.id;
      const riders = station ? await d.stationRiders(station) : null;
      return { version: v?.version, rail: Math.round(v?.railTripsPerDay ?? 0), status: String(c.status), workers: c.computeN, memory: c.memory(), station: !!riders, opened: [...c.opened], stations: window.tw.world.value.save.stations.length };
    });
    const r = { where, at: Number(secs()), chromeMB: chromeMB(profile), ...st };
    out.stops.push(r);
    console.log(`      ${where.padEnd(16)} ${secs()} s  chrome ${r.chromeMB} MB  rail ${r.rail.toLocaleString('en-US')}  v${r.version}  ${r.workers} solving  open ${r.opened.length}  wasm ${r.memory.map((m) => m.wasm).join(',')}  heap ${r.memory.map((m) => m.heap).join(',')}  peak ${r.memory.map((m) => m.solvePeak).join(',')}  need ${r.memory.map((m) => m.need).join(',')}`);
    return r;
  };
  await final(-1);
  const first = await state('loaded');

  // a short line in each big zone: they open in every worker
  const nums = await page.evaluate(() => window.tw.world.value.save.stations.map((s) => s.num));
  const home = await page.evaluate(() => { const c = window.tw.map.getCenter(); return { center: [c.lng, c.lat], zoom: window.tw.map.getZoom() }; });
  const lines = [];
  for (const at of ZONES) {
    await page.evaluate(([at]) => window.tw.map.jumpTo({ center: at, zoom: 12 }), [at]);
    const line = await acrossLine(page, at);
    if (line === null) throw new Error(`no line built at ${at}`);
    lines.push(line);
  }
  await page.waitForFunction((n) => window.twDemand.client.opened.size > n, first.opened.length, { timeout: 600000, polling: 1000 });
  await final(first.version);
  const built = await state('lines built');
  const added = built.opened.filter((x) => !first.opened.includes(x));

  // the lines and their stations removed (the track stays, which opens nothing in the solving workers),
  // the camera back over the network, solved again, then the zones closed by the sweep
  out.removed = await page.evaluate(async ([lines, nums]) => {
    const c = window.tw.client, log = [];
    for (const line of lines) log.push(['line', line, (await c.edit({ op: 'removeLine', line }))?.ok]);
    for (let k = 0; k < 5; k++) {
      const left = window.tw.world.value.save.stations.filter((s) => !nums.includes(s.num));
      if (!left.length) break;
      for (const s of left) log.push(['station', s.num, (await c.edit({ op: 'removeStation', node: s.num }))?.ok]);
      await new Promise((r) => setTimeout(r, 2000));
    }
    // (on the all-China save the demand network kept two zones' removed stations, 2026-10-26: an edit
    // on a line of the save's, made and undone, sends the network again)
    await new Promise((r) => setTimeout(r, 5000));
    if (window.twDemand.client.latest().stations.length > window.tw.world.value.save.stations.length) {
      const w = window.tw.world.value, l = w.save.lines.find((l) => w.lineStats[l.id]?.status === 'running');
      const t = w.lineStats[l.id].trainsNeeded;
      log.push(['nudge', l.num, (await c.edit({ op: 'setTrainCount', line: l.num, trains: t.map((n) => n + 1) }))?.ok, (await c.edit({ op: 'setTrainCount', line: l.num, trains: t }))?.ok]);
    }
    return log;
  }, [lines, nums]);
  await page.evaluate((home) => window.tw.map.jumpTo(home), home);
  await page.waitForFunction((added) => { const r = window.twDemand.client.knownReach(); return !!r && ![...r].some((x) => added.includes(x)); }, added, { timeout: 600000, polling: 1000 }).catch(async (e) => {
    out.reachStuck = await page.evaluate((added) => {
      const c = window.twDemand.client, svc = c.latest(), near = c.nearOf(svc);
      const lls = []; for (let k = 0; k < near.ll.length; k += 2) lls.push([near.ll[k], near.ll[k + 1]]);
      const far = lls.filter(([lng, lat]) => lng > 125 && lat < 40);
      return { reach: [...(c.knownReach() ?? [])].filter((x) => added.includes(x)), stations: window.tw.world.value.save.stations.length, svcStations: svc.stations.length, svcLines: svc.lines.length, near: near.near.filter((x) => added.includes(x)), eastOf125: far.slice(0, 10), waiting: !!c.waiting, lastKeyLen: c.lastKey.length };
    }, added);
    console.log('      reach stuck: ' + JSON.stringify(out.reachStuck) + ' removed ' + JSON.stringify(out.removed));
    throw e;
  });
  await final(built.version);
  const undone = await state('undone');
  const before = restarts();
  await page.waitForFunction((added) => added.every((x) => !window.twDemand.client.opened.has(x)), added, { timeout: 300000, polling: 1000 }).catch(async (e) => {
    out.stuck = await page.evaluate((added) => {
      const c = window.twDemand.client, keep = c.keepSet();
      return { added: added.filter((x) => c.opened.has(x)), keep: added.filter((x) => keep?.has(x)), keepNull: keep === null, reach: [...(c.knownReach() ?? [])].filter((x) => added.includes(x)), nearby: [...window.twDemand.nearbyZones?.peek?.() ?? []], job: !!c.job, hold: !!c.hold, held: !!c.held, ready: c.ready, workers: c.workers.length, status: c.status, stations: window.tw.world.value.save.stations.length };
    }, added);
    console.log('      stuck: ' + JSON.stringify(out.stuck));
    throw e;
  });
  // a worker's heap still holds the closed zones' deferred frees and the old market parts until its next
  // solve (T-1367), so the restart waits for one: an edit on a save's line made and undone
  const v0 = await page.evaluate(() => window.twDemand.log.at(-1).version);
  await page.evaluate(async () => {
    const c = window.tw.client, w = window.tw.world.value, l = w.save.lines.find((l) => w.lineStats[l.id]?.status === 'running'), t = w.lineStats[l.id].trainsNeeded;
    await c.edit({ op: 'setTrainCount', line: l.num, trains: t.map((n) => n + 1) });
    await c.edit({ op: 'setTrainCount', line: l.num, trains: t });
  });
  await final(v0);
  const closed = await state('zones closed');
  // the restart: MEM_QUIET_MS after the last stir, once nothing is in flight
  await page.waitForFunction(() => window.twDemand.client.respawn.size > 0, undefined, { timeout: 300000, polling: 500 }).catch(() => {});
  const began = restarts() > before;
  const at = await page.evaluate(() => window.twDemand.log.at(-1)?.version ?? 0);
  if (began) {
    await page.waitForFunction(() => window.twDemand.client.respawn.size === 0, undefined, { timeout: 1800000, polling: 1000 });
    await final(at);
  }
  const after = await state('after restart');
  out.inSolve = await page.evaluate(() => window.__t1205.inSolve);

  const sum = (r) => r.memory.reduce((a, m) => a + m.wasm, 0);
  report('restart once the zones closed', began, infos.filter((s) => /restarted/.test(s)).join(' | ').slice(0, 500));
  report('no restart inside a solve', out.inSolve.length === 0, JSON.stringify(out.inSolve));
  report('memory given back', sum(after) < sum(closed) - 256, `(workers' WASM ${sum(first)} MB loaded, ${sum(built)} with ${added.length} more items open, ${sum(closed)} closed, ${sum(after)} after; Chrome ${first.chromeMB}, ${built.chromeMB}, ${closed.chromeMB}, ${after.chromeMB} MB)`);
  report('figures unchanged', after.rail === undone.rail && !after.status.startsWith('failed'), `(${after.rail.toLocaleString('en-US')} rail trips a day after, ${undone.rail.toLocaleString('en-US')} undone, ${first.rail.toLocaleString('en-US')} loaded)`);
  report('station answered as at the load', after.station === first.station, `(${after.station})`);
  report('no console errors', errors.length === 0, errors.slice(0, 5).join(' | '));
} catch (e) {
  report('ran', false, e.message.split('\n')[0]);
} finally {
  const dir = join(ARTIFACTS, 't1205');
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, `memory-${LABEL}.json`), JSON.stringify(out, null, 1));
  await close();
}
console.log(failed ? `T1205 FAILED: ${failed}` : 'T1205 PASSED');
process.exit(failed ? 1 : 0);
