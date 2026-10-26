// T-328: the demand cache on reload (docs/architecture.md 11 "Demand cache"). Times and checks a
// reload of the real New York save, with a persistent profile so the save and its cache stay.
//   MODE=prepare | reload | stale    BASE=http://localhost:6151/    OUT=<folder>    RUNS=3
//   node trainworld/checks/t328-cache-check.mjs            (from the maps root)
// prepare: imports T-007-real-nyc.save into the profile's autosave slot.
// reload: RUNS page loads; each records when the network was drawn, when the clock was released
//   (T-347, T-421: not before the fresh solve's first estimate, with or without the cache),
//   every change of the shown view (version:round:final, version 0 = the cache) and a signature of
//   every figure in the first final view shown and in the fresh solve's final. PASS when the
//   fresh final's signature is the same on every run and the cached view's equals it.
// stale: tampers with the stored cache (its build, a pack's digest), reloads, and checks that the
//   cache is not shown; then a changed network (one line's trains) must not show it either.
// Outputs OUT/t328-<MODE>-<LABEL>.json. Playwright from PLAYWRIGHT_MODULE, else the npx cache.
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, collectErrors, importSave, demandFinal, REAL_SAVE } from './lib.mjs';

const OUT = process.env.OUT ?? '.';
const BASE = process.env.BASE ?? 'http://localhost:6151/';
const MODE = process.env.MODE ?? 'reload';
const LABEL = process.env.LABEL ?? 'run';
const RUNS = Number(process.env.RUNS ?? 3);
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const { ctx, close } = await launchChrome({ profile: join(OUT, 'profile'), fresh: false });

await ctx.addInitScript(([bubbles]) => {
  try {
    const s = JSON.parse(localStorage.getItem('trainworld-ui') || '{}');
    s.v = 2;
    s.display = { ...(s.display || {}), dayShown: 'weekday', commuters: bubbles !== 'off', commuterEnd: 'work' };
    localStorage.setItem('trainworld-ui', JSON.stringify(s));
  } catch { /* */ }
  // a signature of every figure a view holds: lines, stations, purposes, the split, city views, the market
  window.__sig = () => {
    const d = window.twDemand, v = d.view(), nums = [], marks = [];
    const put = (x) => { if (x == null) nums.push(-1e300); else if (typeof x === 'number') nums.push(x); else for (const y of x) put(y); };
    const mark = (k) => marks.push([k, nums.length]);
    mark('view');
    put([v.split.train, v.split.walk, v.split.drive, v.tripsPerDay, v.railTripsPerDay, v.peakHourFactor, v.trainShareByPeriod.map((p) => p[1])]);
    mark('lines');
    for (const [, l] of [...v.lines].sort((a, b) => +a[0] - +b[0])) put([l.ridersPerDay, l.fullestPct, l.boardings, ...l.loads, ...(l.board ?? []), ...(l.alight ?? [])]);
    mark('stations');
    for (const [, s] of [...v.stations].sort((a, b) => +a[0] - +b[0])) put([s.boardings, s.alightings]);
    mark('purposes');
    for (const [k, p] of [...(v.purposes ?? [])].sort()) {
      nums.push(k.length);
      for (const [, l] of [...p.lines].sort((a, b) => +a[0] - +b[0])) put([l.ridersPerDay, l.boardings, ...l.loads]);
      for (const [, s] of [...p.stations].sort((a, b) => +a[0] - +b[0])) put([s.boardings, s.alightings]);
    }
    mark('cities');
    const cities = [...d.cities()].sort();
    for (const [, c] of cities) put([c.split.train, c.split.walk, c.split.drive, c.tripsPerDay, c.railTripsPerDay, c.trainShareByPeriod.map((p) => p[1])]);
    const ic = d.intercity();
    if (ic) {
      mark('market');
      put([ic.tripsPerDay, ic.railTripsPerDay, ic.commuteTripsPerDay ?? 0, ic.commuteRailPerDay ?? 0]);
      const c = ic.corridors;
      mark('corridorTable');
      nums.push(c.table.n, c.longerScale, c.commuteScale); for (const ch of c.table.key) nums.push(ch.charCodeAt(0));
      mark('railAt'); put([...c.railAt]);
      mark('longerRail'); put([...c.longerRail]);
      mark('commuteRail'); put([...c.commuteRail]);
    }
    mark('end');
    const f = new Float64Array(nums), b = new Uint32Array(f.buffer);
    const fnv = (lo, hi) => { let h = 0x811c9dc5; for (let i = 2 * lo; i < 2 * hi; i++) h = Math.imul(h ^ b[i], 0x01000193); return (h >>> 0).toString(16); };
    // each section's own hash, so two differing hashes point at the figures that differ
    const sections = Object.fromEntries(marks.slice(0, -1).map(([k, at], i) => [k, fnv(at, marks[i + 1][1])]));
    window.__lastNums = nums; // DUMP=1 reads these to diff two runs
    return { hash: fnv(0, nums.length), sections, marks, n: nums.length, lines: v.lines.size, stations: v.stations.size, cities: cities.map((c) => c[0]).join('+'), train: +v.split.train.toFixed(3), rail: Math.round(v.railTripsPerDay) };
  };
  // the bubbles' legs per cell (weekday, all purposes) of the view shown
  window.__modesSig = async () => {
    const m = await window.twDemand.cellModes();
    if (!m) return null;
    let h = 0x811c9dc5, n = 0;
    for (const a of [m.homeRail, m.homeWalk, m.homeDrive, m.workRail, m.workWalk, m.workDrive]) {
      const b = new Uint32Array(a.buffer, a.byteOffset, a.length);
      for (let i = 0; i < b.length; i++) h = Math.imul(h ^ b[i], 0x01000193);
      n += a.length;
    }
    return { version: m.version, hash: (h >>> 0).toString(16), n };
  };
  window.__ev = [];
  const ev = (k, v) => window.__ev.push([Math.round(performance.now()), k, v]);
  let lastView = '', world = false, clock = false, firstFinal = false, lastBar = '', bubblesSeen = false;
  setInterval(() => {
    if (!bubblesSeen && window.twViews?.stats?.commuters?.builds > 0) { bubblesSeen = true; ev('bubbles'); }
    if (!world && window.tw?.world?.value) { world = true; ev('world'); }
    if (!clock && world && window.tw?.clock?.held && window.tw.clock.held.peek() === false) { clock = true; ev('clock'); }
    const d = window.twDemand;
    if (!d) return;
    const v = d.view?.();
    const key = v ? `${v.version}:${v.round}:${v.final}` : '';
    if (key !== lastView) { lastView = key; ev('view', key); }
    if (v?.final && !firstFinal) { firstFinal = true; window.__firstSig = window.__sig(); if (v.version === 0) window.__modesSig().then((m) => { window.__firstModes = m; }); }
    const bar = document.querySelector('#topbar')?.textContent?.match(/(Updating demand[^A-Z]*|Estimate[^A-Z]*|Loading[^A-Z]*)/)?.[0] ?? '';
    if (bar !== lastBar) { lastBar = bar; ev('bar', bar); }
  }, 20);
}, [process.env.BUBBLES ?? 'work']);

const page = await ctx.newPage();
const errors = [];
collectErrors(page, errors, null);

const waitSolved = (timeout = 180000) => demandFinal(page, { timeout, polling: 100 });

let dumps = 0;
/** DUMP=1: the shown figures as float64 in OUT/t328-dump-<n>.f64 (sections: the `marks` of the run's json), to diff two runs */
async function dump() {
  if (!process.env.DUMP) return;
  const b64 = await page.evaluate(() => {
    const f = new Float64Array(window.__sig() && window.__lastNums), u = new Uint8Array(f.buffer);
    let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
    return btoa(s);
  });
  await writeFile(join(OUT, `t328-dump-${dumps++}.f64`), Buffer.from(b64, 'base64'));
}
/** one page load: events, the first final view's signature, the fresh final's */
async function reload() {
  await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
  await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
  await waitSolved();
  await page.waitForTimeout(1500); // the cache write after the final
  await dump();
  return page.evaluate(async () => {
    const c = window.twDemand.client;
    return { ev: window.__ev, first: window.__firstSig, fresh: window.__sig(), cache: c.cacheLog ?? null, firstModes: window.__firstModes ?? null, freshModes: await window.__modesSig(),
      solves: window.twDemand.log.map((s) => ({ v: s.version, sent: Math.round(s.sentAt), first: s.firstAt && Math.round(s.firstAt), refined: s.refinedAt && Math.round(s.refinedAt), est: s.showEstimates })) };
  });
}

const at = (r, k, v) => r.ev.find((e) => e[1] === k && (v === undefined || e[2] === v))?.[0];
const summary = (r) => {
  const firstFinal = r.ev.find((e) => e[1] === 'view' && /:true$/.test(e[2]));
  return { world: at(r, 'world'), clock: at(r, 'clock'), bubbles: at(r, 'bubbles'), firstModes: r.firstModes, freshModes: r.freshModes, firstView: r.ev.find((e) => e[1] === 'view' && e[2])?.[0], firstFinal: firstFinal?.[0], firstFinalKey: firstFinal?.[2], freshFinal: r.solves.at(-1)?.refined, cached: r.ev.some((e) => e[1] === 'view' && /^0:/.test(e[2])), views: r.ev.filter((e) => e[1] === 'view').map((e) => `${e[0]} ${e[2]}`).join(', ') };
};

const out = { mode: MODE, label: LABEL, base: BASE, runs: [], errors };
let ok = true;
try {
  if (MODE === 'prepare') {
    await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    const bytes = await readFile(REAL_SAVE);
    const before = await page.evaluate(() => window.tw.lastSaved.peek());
    await importSave(page, bytes);
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 240000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
    await waitSolved();
    await page.waitForTimeout(2000);
    const sig = await page.evaluate(() => window.__sig());
    await dump();
    log(`save stored; first final after the import: ${sig.hash} rail ${sig.rail} n ${sig.n} ${JSON.stringify(sig.sections)}`);
  } else if (MODE === 'reload') {
    for (let i = 0; i < RUNS; i++) {
      const r = await reload();
      const s = summary(r);
      out.runs.push({ ...s, first: r.first, fresh: r.fresh, cache: r.cache, solves: r.solves });
      log(`run ${i}: bubbles ${s.bubbles} (modes cached ${s.firstModes?.hash ?? '-'} fresh ${s.freshModes?.hash ?? '-'}) world ${s.world} clock ${s.clock} first view ${s.firstView} first final ${s.firstFinal} (${s.firstFinalKey}) fresh final ${s.freshFinal} cached ${s.cached} | first ${r.first?.hash} fresh ${r.fresh.hash} n ${r.fresh.n} train ${r.fresh.train}% rail ${r.fresh.rail}`);
    }
    const fresh = new Set(out.runs.map((r) => r.fresh.hash));
    const cachedSame = out.runs.filter((r) => r.cached).every((r) => r.first.hash === r.fresh.hash && (!r.firstModes || r.firstModes.hash === r.freshModes?.hash));
    ok = fresh.size === 1 && cachedSame;
    console.log(`${fresh.size === 1 ? 'PASS' : 'FAIL'}  fresh finals agree on every run (${[...fresh].join(', ')})`);
    if (fresh.size > 1) {
      const names = Object.keys(out.runs[0].fresh.sections);
      console.log('  sections that differ between runs: ' + (names.filter((k) => new Set(out.runs.map((r) => r.fresh.sections[k])).size > 1).join(', ') || 'none'));
    }
    console.log(`${cachedSame ? 'PASS' : 'FAIL'}  the cached view equals the fresh final (${out.runs.filter((r) => r.cached).length} cached runs)`);
    // T-347: the clock stays held, the card up, through the cached view and the fresh solve's first
    // estimate; it runs once the zone in view has its fresh final (T-421), which can come a few hundred
    // ms before the solve's last round (`refined`)
    const heldToFinal = out.runs.every((r) => r.clock !== undefined && r.freshFinal !== undefined && r.clock >= r.solves.at(-1).first && r.clock <= r.freshFinal + 1000);
    ok &&= heldToFinal;
    console.log(`${heldToFinal ? 'PASS' : 'FAIL'}  the clock is held until the zone in view has its fresh final (clock ${out.runs.map((r) => r.clock).join('/')}, first estimate ${out.runs.map((r) => r.solves.at(-1).first).join('/')}, final ${out.runs.map((r) => r.freshFinal).join('/')})`);
  } else if (MODE === 'stale') {
    // the cache entry, as the page stores it (IndexedDB "anitabuilder", store "saves", keys "demand:...")
    const tamper = (field) => page.evaluate((field) => new Promise((res, rej) => {
      const o = indexedDB.open('anitabuilder', 1);
      o.onsuccess = () => {
        const st = o.result.transaction('saves', 'readwrite').objectStore('saves');
        const k = st.getAllKeys();
        k.onsuccess = () => {
          const keys = k.result.filter((x) => typeof x === 'string' && x.startsWith('demand:'));
          if (!keys.length) return res(0);
          let n = 0;
          for (const key of keys) {
            const g = st.get(key);
            g.onsuccess = () => {
              const v = g.result;
              if (field === 'build') v.build = 'x' + v.build;
              else v.packs[Object.keys(v.packs)[0]] = 'x';
              st.put(v, key).onsuccess = () => { if (++n === keys.length) res(n); };
            };
          }
        };
      };
      o.onerror = () => rej(o.error);
    }), field);
    for (const field of ['build', 'pack']) {
      let r = await reload(); // writes a good cache again if the last run left none
      if (!summary(r).cached) r = await reload();
      const n = await tamper(field);
      r = await reload();
      const s = summary(r);
      const pass = n > 0 && !s.cached;
      ok &&= pass;
      out.runs.push({ field, tampered: n, ...s, cache: r.cache });
      console.log(`${pass ? 'PASS' : 'FAIL'}  a changed ${field} shows no cache (${n} entries tampered; views ${s.views}; ${JSON.stringify(r.cache)})`);
    }
    // a changed network (the fare, which demand reads), never solved to a cache (writes off), then
    // a reload; the fare is put back at the end
    let r = await reload();
    await page.evaluate(() => { window.twDemand.client.cacheWrites = false; });
    const edited = await page.evaluate(async () => {
      const f = window.twDemand.log.at(-1).svc.fare, before = window.tw.lastSaved.peek();
      await window.tw.client.edit({ op: 'fares', base: f.base + 0.25, perKm: f.perKm });
      return { fare: f, before };
    });
    await page.waitForTimeout(1500);
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, edited.before, { timeout: 60000 }).catch(() => {});
    r = await reload();
    const s = summary(r);
    const pass = !s.cached;
    ok &&= pass;
    out.runs.push({ field: 'network', edited, ...s, cache: r.cache });
    console.log(`${pass ? 'PASS' : 'FAIL'}  a changed network shows no cache (fare +0.25; views ${s.views}; ${JSON.stringify(r.cache)})`);
    const back = await page.evaluate(async (f) => { const before = window.tw.lastSaved.peek(); await window.tw.client.edit({ op: 'fares', base: f.base, perKm: f.perKm }); return before; }, edited.fare);
    await page.waitForTimeout(1500);
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, back, { timeout: 60000 }).catch(() => {});
  }
} catch (e) {
  ok = false;
  console.log('FAIL  ' + (e?.message ?? e));
}
const errs = errors.filter((e) => !/favicon|openfreemap|AJAXError/.test(e));
if (errs.length) { ok = false; console.log('FAIL  console errors: ' + errs.slice(0, 3).join(' | ')); }
await writeFile(join(OUT, `t328-${MODE}-${LABEL}.json`), JSON.stringify(out, null, 1));
await close();
process.exit(ok ? 0 : 1);
