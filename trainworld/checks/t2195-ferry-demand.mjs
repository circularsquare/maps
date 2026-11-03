// T-2195: ferries in demand as lines, on the real save (T-007-real-nyc.save) in headless Chrome. From the maps root:
//   node trainworld/checks/t2195-ferry-demand.mjs [url] [--baseline]
// 1. The real save as it is, solved cold: its weekday and average-day figures (rail trips, every line's riders)
//    are written to artifacts/trainworld/t2195/ferry-free.json; with --baseline that is all (run it on the code
//    before a change), otherwise they must equal baseline-w<workers>.json there to the bit when it exists.
// 2. Five ferry lines added on that page through the app's own water router (the Ferry tool's worker): the
//    Staten Island ferry and four of NYC Ferry's East River routes, constructed. Staten Island's weekday riders
//    between 20,000 and 100,000; rail trips up (ferry trips are the network's); the Ferry fare multiple at 2.00x
//    loses Staten Island riders. The game is saved to artifacts/trainworld/t2195/real-ferries.save.
// 3. Cold solves of the real save and of real-ferries.save, each in a fresh profile, in turn (`T2195_PAIRS`,
//    2 by default): the ferries' mean solve within 10% of the plain one's (the solve's own time, from
//    sending to its final). With 2 workers the night period's one extra crowding round (the default
//    boat every 30 minutes crowds at Staten Island) falls on the busier worker and costs about 15%.
// Uses the Vite already on the url (dev-nowatch.mjs). `T2195_WORKERS`: the demand workers, 3 by default
// (the game's own on a machine of 12 threads or more); claim 5 cores.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2195');
await mkdir(OUT, { recursive: true });
const args = process.argv.slice(2);
const BASE = args.find((a) => /^https?:/.test(a)) ?? 'http://localhost:6191/';
const BASELINE = args.includes('--baseline');
const WORKERS = Number(process.env.T2195_WORKERS ?? 3);
const PAIRS = Number(process.env.T2195_PAIRS ?? 2);
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const t0 = Date.now();
const log = (s) => console.log(`${((Date.now() - t0) / 1000).toFixed(0).padStart(5)} s  ${s}`);

/** Ports by name, [lng, lat] on the shore by each landing (the router snaps them), and the five lines. */
const PORT = {
  whitehall: [-74.0130, 40.7012], stgeorge: [-74.0715, 40.6437],
  pier11: [-74.0068, 40.7041], dumbo: [-73.9937, 40.7030], schaefer: [-73.9693, 40.7093], nwburg: [-73.9617, 40.7202],
  greenpoint: [-73.9609, 40.7311], hpsouth: [-73.9593, 40.7420], e34: [-73.9730, 40.7445], lic: [-73.9590, 40.7488],
  rooseveltis: [-73.9530, 40.7566], astoria: [-73.9353, 40.7717], pier6: [-73.9985, 40.6935], redhook: [-74.0115, 40.6855],
  armyterm: [-74.0245, 40.6460], bayridge: [-74.0355, 40.6355], e90: [-73.9440, 40.7770], soundview: [-73.8510, 40.8065],
};
const LINES = [
  ['Staten Island', ['whitehall', 'stgeorge']],
  ['East River', ['pier11', 'dumbo', 'schaefer', 'nwburg', 'greenpoint', 'hpsouth', 'e34']],
  ['Astoria', ['pier11', 'e34', 'lic', 'rooseveltis', 'astoria']],
  ['South Brooklyn', ['pier11', 'dumbo', 'pier6', 'redhook', 'armyterm', 'bayridge']],
  ['Soundview', ['pier11', 'e34', 'e90', 'soundview']],
];

async function open(prefix) {
  const chrome = await launchChrome({ prefix, size: [1200, 800], deviceScaleFactor: 1 });
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  const errors = [];
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=${WORKERS}&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw.tools?.ferry, undefined, { timeout: 180000 });
  return { chrome, page, errors };
}

/** Import a save and wait for its final; the solve's own ms (sent to final) and its workers' summed ms. */
async function coldSolve(page, bytes) {
  const t = Date.now();
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  await demandFinal(page, { timeout: 3600000, polling: 1000, lines: 100 });
  const s = await page.evaluate(() => {
    const d = window.twDemand, s = d.log.findLast((x) => !d.client.fversions?.has(x.version));
    const sum = (a) => (a ?? []).flat(4).reduce((x, y) => x + (Number.isFinite(y) ? y : 0), 0);
    return { solveMs: (s.roundAt?.at(-1) ?? s.refinedAt ?? performance.now()) - s.sentAt, cpuMs: sum(s.periodMs) + sum(s.setupMs), stops: s.layout.reduce((a, l) => a + l.n, 0) };
  });
  return { ...s, wallMs: Date.now() - t };
}

/** Weekday and average-day figures: rail trips and each line's riders, as the view has them. */
const figures = (page) => page.evaluate(() => {
  const days = window.twDemand.days(), out = {};
  for (const [k, name] of [[0, 'weekday'], [6, 'average']]) {
    const v = days[k]?.view;
    if (!v) continue;
    out[name] = { rail: v.railTripsPerDay, lines: Object.fromEntries([...v.lines].map(([id, l]) => [id, l.ridersPerDay])) };
  }
  return out;
});
const weekdayRiders = (page, id) => page.evaluate((id) => window.twDemand.days()[0]?.view?.lines.get(id)?.ridersPerDay ?? null, id);
const sameBits = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const realBytes = await readFile(REAL_SAVE);
const times = { plain: [], ferries: [] };
// ---- 1. the real save as it is
{
  const { chrome, page, errors } = await open('tw-t2195-a-');
  try {
    const t = await coldSolve(page, realBytes);
    times.plain.push(t);
    log(`real save: final in ${(t.solveMs / 1000).toFixed(1)} s (workers ${(t.cpuMs / 1000).toFixed(1)} s, ${t.stops} stops)`);
    const fig = await figures(page);
    await writeFile(join(OUT, BASELINE ? `baseline-w${WORKERS}.json` : 'ferry-free.json'), JSON.stringify(fig) + '\n');
    if (BASELINE) {
      log(`baseline written: weekday rail ${Math.round(fig.weekday?.rail ?? 0).toLocaleString('en-US')}`);
      await chrome.close();
      process.exit(0);
    }
    const basePath = join(OUT, `baseline-w${WORKERS}.json`);
    if (existsSync(basePath)) {
      const base = JSON.parse(await readFile(basePath, 'utf8'));
      check(sameBits(base, fig), 'the ferry-free real save solves to the bit as the baseline', `weekday rail ${fig.weekday?.rail} against ${base.weekday?.rail}`);
    } else console.log(`  (no baseline-w${WORKERS}.json: run with --baseline on the code before the change to compare)`);

    // ---- 2. five ferry lines
    await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
    const before = fig.weekday?.rail ?? 0;
    const plainVersion = await page.evaluate(() => window.twDemand.view().version);
    const made = await page.evaluate(async ([PORT, LINES]) => {
      const tw = window.tw, water = tw.tools.ferry.water, out = [], nodes = {};
      const snap = async ([lng, lat]) => {
        const k = 1 / (111320 * Math.cos((lat * Math.PI) / 180)), j = 1 / 110540;
        for (const r of [0, 50, 100, 150]) for (let a = 0; a < (r ? 8 : 1); a++) {
          const s = await water.port([lng + r * Math.cos((a * Math.PI) / 4) * k, lat + r * Math.sin((a * Math.PI) / 4) * j]);
          if (s?.ok) return s;
        }
        return null;
      };
      for (const [name, stops] of LINES) {
        const ports = [], berths = [];
        for (const p of stops) {
          if (nodes[p]) { ports.push({ node: nodes[p].node }); berths.push(nodes[p].berth); continue; }
          const s = await snap(PORT[p]);
          if (!s) return { error: `${name}: no port at ${p}` };
          const a = tw.lngLatToPack(s.landing[0], s.landing[1]), b = tw.lngLatToPack(s.berth[0], s.berth[1]);
          ports.push({ x: a.x, y: a.y, bx: b.x, by: b.y, name: p });
          berths.push(s.berth);
        }
        const r = await water.line(berths.map((at) => ({ at, kind: 1 })), 'check');
        if (!r?.route.ok) return { error: `${name}: no route (${r?.route.legs.map((l) => l.reason ?? 'ok').join(', ')})` };
        const route = [];
        for (let i = 0; i < r.route.kinds.length; i++) { const q = tw.lngLatToPack(r.route.ll[2 * i], r.route.ll[2 * i + 1]); route.push(q.x, q.y, r.route.kinds[i], r.route.speeds[i]); }
        const e = await tw.client.edit({ op: 'addFerryLine', ports, route, name, colour: '#1f77b4' });
        if (!e?.ok || e.line === undefined) return { error: `${name}: not added (${(e?.issues ?? []).join(', ')})` };
        const c = await tw.client.edit({ op: 'constructLine', line: e.line });
        if (!c?.ok) return { error: `${name}: not constructed (${(c?.issues ?? []).join(', ')})` };
        await new Promise((d) => setTimeout(d, 300));
        const st = tw.world.value.lineStats[String(e.line)];
        st.bus.ferry.ports.forEach((p, i) => { nodes[stops[i]] ??= { node: p.node, berth: p.berth }; });
        out.push({ name, id: String(e.line), km: r.route.m / 1000, minutes: (st.fromStartS.at(-1) ?? 0) / 60, boats: Math.max(...st.trainsNeeded) });
      }
      return { lines: out };
    }, [PORT, LINES]);
    if (made.error) throw new Error(made.error);
    check(made.lines.length === 5, 'five ferry lines added and constructed', made.lines.map((l) => `${l.name} ${l.km.toFixed(1)} km ${l.minutes.toFixed(0)} min ${l.boats} boats`).join('; '));
    await demandFinal(page, { timeout: 1200000, polling: 1000, after: plainVersion });
    log('the ferries\' final is in');
    const withF = await figures(page);
    const si = made.lines[0].id;
    const riders = Object.fromEntries(made.lines.map((l) => [l.name, Math.round(withF.weekday?.lines[l.id] ?? 0)]));
    log(`weekday riders: ${JSON.stringify(riders)}`);
    const siR = withF.weekday?.lines[si] ?? 0;
    check(siR >= 20000 && siR <= 100000, 'Staten Island ferry carries 20,000-100,000 riders a weekday (real about 50,000-70,000)', Math.round(siR).toLocaleString('en-US'));
    check((withF.weekday?.rail ?? 0) > before, 'trips that ride ferries count as rail', `weekday rail ${Math.round(before).toLocaleString('en-US')} -> ${Math.round(withF.weekday?.rail ?? 0).toLocaleString('en-US')}`);
    const panel = await page.evaluate(async (id) => {
      window.tw.selection.value = { kind: 'line', line: id };
      await new Promise((d) => setTimeout(d, 2500));
      return document.querySelector('#inspector, .inspector')?.innerText ?? document.body.innerText;
    }, si);
    check(/Changing to trains/.test(panel) && /Average load/.test(panel), 'the ferry panel has Changing to trains and Average load', panel.split('\n').filter((s) => /Riders|Changing|Average load|leaving full/.test(s)).join(' | '));
    await page.evaluate(() => { window.tw.selection.value = null; });

    // the Ferry fare multiple: dearer ferries, fewer riders
    let v = await page.evaluate(() => window.twDemand.view().version);
    await page.evaluate(() => window.tw.client.edit({ op: 'fareFerry', mul: 2 }));
    await demandFinal(page, { timeout: 1200000, polling: 1000, after: v });
    log('the dear ferries\' final is in');
    const dear = await weekdayRiders(page, si);
    check(dear !== null && dear < siR, 'the Ferry fare multiple at 2.00x loses Staten Island riders', `${Math.round(siR).toLocaleString('en-US')} -> ${Math.round(dear ?? 0).toLocaleString('en-US')}`);
    v = await page.evaluate(() => window.twDemand.view().version);
    await page.evaluate(() => window.tw.client.edit({ op: 'fareFerry', mul: 1 }));
    await demandFinal(page, { timeout: 1200000, polling: 1000, after: v });

    const dl = page.waitForEvent('download', { timeout: 60000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
    await writeFile(join(OUT, 'real-ferries.save'), await readFile(await (await dl).path()));
    await writeFile(join(OUT, 'ferries.json'), JSON.stringify({ made: made.lines, riders, dear, rail: [before, withF.weekday?.rail] }, null, 1) + '\n');
    check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
  } catch (e) {
    check(false, 'the real save with ferries ran to the end', e.stack ?? e.message);
  } finally {
    await chrome.close();
  }
}

// ---- 3. cold solves, in turn
const ferryBytes = existsSync(join(OUT, 'real-ferries.save')) ? await readFile(join(OUT, 'real-ferries.save')) : null;
if (ferryBytes) {
  times.plain = []; // part 1's came first, with the pages' code not yet warm in the disk cache
  for (let rep = 0; rep < PAIRS; rep++) {
    for (const [kind, bytes] of [['plain', realBytes], ['ferries', ferryBytes]]) {
      const { chrome, page } = await open(`tw-t2195-${kind}${rep}-`);
      try {
        const t = await coldSolve(page, bytes);
        times[kind].push(t);
        log(`${kind}: final in ${(t.solveMs / 1000).toFixed(1)} s (workers ${(t.cpuMs / 1000).toFixed(1)} s, ${t.stops} stops)`);
      } catch (e) {
        check(false, `cold solve ${kind}`, e.message.split('\n')[0]);
      } finally {
        await chrome.close();
      }
    }
  }
  const mean = (k, f) => times[k].reduce((a, t) => a + t[f], 0) / Math.max(1, times[k].length);
  const ratio = mean('ferries', 'solveMs') / mean('plain', 'solveMs'), cpu = mean('ferries', 'cpuMs') / mean('plain', 'cpuMs');
  check(ratio <= 1.10, `the real save with five ferry lines solves within 10% of the plain one (${WORKERS} workers)`, `mean ${(mean('ferries', 'solveMs') / 1000).toFixed(1)} s against ${(mean('plain', 'solveMs') / 1000).toFixed(1)} s, x${ratio.toFixed(3)}; workers' ms x${cpu.toFixed(3)}`);
  await writeFile(join(OUT, 'times.json'), JSON.stringify(times, null, 1) + '\n');
}
console.log(fail ? `T-2195 FAILED: ${fail}` : 'T-2195 PASSED');
process.exit(fail ? 1 : 0);
