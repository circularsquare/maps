// T-328: the demand cache on reload (docs/architecture.md 11 "Demand cache"). Times and checks a
// reload of the real New York save, with a persistent profile so the save and its cache stay.
//   MODE=prepare | reload | stale | edit    BASE=http://localhost:6151/    OUT=<folder>    RUNS=3
//   node trainworld/checks/t328-cache-check.mjs            (from the maps root)
// prepare: imports T-007-real-nyc.save into the profile's autosave slot.
// reload: RUNS page loads; each records when the network was drawn, when the clock was released
//   (T-2560: as the cached view shows; without the cache, T-347, T-421: at the zone in view's final),
//   every change of the shown view (version:round:final, version 0 = the cache) and a signature of
//   every figure in the first final view shown and in the fresh solve's final. PASS when the
//   fresh final's signature is the same on every run and the cached view's equals it.
// stale: tampers with the stored cache (its build, a pack's digest), reloads, and checks that the
//   cache is not shown; then a changed network (one line's trains) must not show it either.
// edit (T-2560): a reload showing the cache, one line's trains +1 the moment the clock runs (its network
//   still being solved again), that edit's final; then a reload missing the cache (perfOff=nocache)
//   solves the edited network cold: the figures must be the same. The trains are put back after.
// Outputs OUT/t328-<MODE>-<LABEL>.json. Playwright from PLAYWRIGHT_MODULE, else the npx cache.
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, collectErrors, importSave, demandFinal, REAL_SAVE } from './lib.mjs';

const OUT = process.env.OUT ?? '.';
const BASE = process.env.BASE ?? 'http://localhost:6151/';
const MODE = process.env.MODE ?? 'reload';
const LABEL = process.env.LABEL ?? 'run';
const RUNS = Number(process.env.RUNS ?? 3);
const QUERY = process.env.QUERY ? `&${process.env.QUERY}` : ''; // more URL parameters (perfOff=...)
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
    const cityAt = [];
    for (const [id, c] of cities) { cityAt.push([id, nums.length]); put([c.split.train, c.split.walk, c.split.drive, c.tripsPerDay, c.railTripsPerDay, c.trainShareByPeriod.map((p) => p[1])]); }
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
    // each zone's own hash, so a zone shown in one view only is told from a zone whose figures differ
    const zones = Object.fromEntries(cityAt.map(([id, at], i) => [id, fnv(at, cityAt[i + 1]?.[1] ?? marks.find(([k]) => k !== 'view' && k !== 'lines' && k !== 'stations' && k !== 'purposes' && k !== 'cities')?.[1] ?? nums.length)]));
    window.__lastNums = nums; // DUMP=1 reads these to diff two runs
    return { hash: fnv(0, nums.length), sections, zones, marks, n: nums.length, lines: v.lines.size, stations: v.stations.size, cities: cities.map((c) => c[0]).join('+'), train: +v.split.train.toFixed(3), rail: Math.round(v.railTripsPerDay) };
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

const waitSolved = (timeout = 600000) => demandFinal(page, { timeout, polling: 100 });

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
  await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}${QUERY}`);
  await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
  await waitSolved();
  await page.waitForTimeout(1500); // the cache write after the final
  await dump();
  return page.evaluate(async () => {
    const c = window.twDemand.client;
    return { ev: window.__ev, first: window.__firstSig, fresh: window.__sig(), cache: c.cacheLog ?? null, firstModes: window.__firstModes ?? null, freshModes: await window.__modesSig(),
      solves: window.twDemand.log.filter((s) => !s.rough && !c.fversions?.has(s.version)).map((s) => ({ v: s.version, sent: Math.round(s.sentAt), first: s.firstAt && Math.round(s.firstAt), refined: s.refinedAt && Math.round(s.refinedAt), est: s.showEstimates })) };
  });
}

/** two signatures hold the same figures: every section alike, and every zone both show alike (T-2560: a zone
 * no station reaches opens bare whenever it comes near the view, so the zones shown can differ) */
const sameFigures = (a, b) => !!a && !!b && Object.keys(b.sections).every((k) => k === 'cities' || a.sections[k] === b.sections[k])
  && Object.keys(a.zones ?? {}).every((k) => !(k in (b.zones ?? {})) || a.zones[k] === b.zones[k]);
const at = (r, k, v) => r.ev.find((e) => e[1] === k && (v === undefined || e[2] === v))?.[0];
const summary = (r) => {
  const firstFinal = r.ev.find((e) => e[1] === 'view' && /:true$/.test(e[2]));
  return { world: at(r, 'world'), clock: at(r, 'clock'), bubbles: at(r, 'bubbles'), firstModes: r.firstModes, freshModes: r.freshModes, firstView: r.ev.find((e) => e[1] === 'view' && e[2])?.[0], firstFinal: firstFinal?.[0], firstFinalKey: firstFinal?.[2], freshFinal: r.solves.at(-1)?.refined, cached: r.ev.some((e) => e[1] === 'view' && /^0:/.test(e[2])), views: r.ev.filter((e) => e[1] === 'view').map((e) => `${e[0]} ${e[2]}`).join(', ') };
};

const out = { mode: MODE, label: LABEL, base: BASE, runs: [], errors };
let ok = true;
try {
  if (MODE === 'prepare') {
    await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}${QUERY}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    const bytes = await readFile(process.env.SAVE ?? REAL_SAVE); // (SAVE: another save of over 100 lines)
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
      if (s.cached && r.first && r.first.hash !== r.fresh.hash) {
        const z1 = r.first.zones ?? {}, z2 = r.fresh.zones ?? {};
        log(`run ${i}: cached and fresh differ in ${Object.keys(r.fresh.sections).filter((k) => r.first.sections[k] !== r.fresh.sections[k]).join(', ')}; zones only cached ${Object.keys(z1).filter((k) => !(k in z2)).join(',') || '-'}, only fresh ${Object.keys(z2).filter((k) => !(k in z1)).join(',') || '-'}, differing ${Object.keys(z1).filter((k) => k in z2 && z1[k] !== z2[k]).join(',') || '-'}`);
      }
      log(`run ${i}: bubbles ${s.bubbles} (modes cached ${s.firstModes?.hash ?? '-'} fresh ${s.freshModes?.hash ?? '-'}) world ${s.world} clock ${s.clock} first view ${s.firstView} first final ${s.firstFinal} (${s.firstFinalKey}) fresh final ${s.freshFinal} cached ${s.cached} | first ${r.first?.hash} fresh ${r.fresh.hash} n ${r.fresh.n} train ${r.fresh.train}% rail ${r.fresh.rail}`);
    }
    const fresh = new Set(out.runs.map((r) => Object.entries(r.fresh.sections).filter(([k]) => k !== 'cities').map(([, h]) => h).join('')));
    const cachedSame = out.runs.filter((r) => r.cached).every((r) => sameFigures(r.first, r.fresh) && (!r.firstModes || r.firstModes.hash === r.freshModes?.hash));
    ok = fresh.size === 1 && cachedSame;
    console.log(`${fresh.size === 1 ? 'PASS' : 'FAIL'}  fresh finals agree on every run, zones shown aside (${out.runs.map((r) => r.fresh.hash).join(', ')})`);
    if (fresh.size > 1) {
      const names = Object.keys(out.runs[0].fresh.sections);
      console.log('  sections that differ between runs: ' + (names.filter((k) => new Set(out.runs.map((r) => r.fresh.sections[k])).size > 1).join(', ') || 'none'));
    }
    console.log(`${cachedSame ? 'PASS' : 'FAIL'}  the cached view's figures equal the fresh final's (${out.runs.filter((r) => r.cached).length} cached runs)`);
    // T-347: without the cache the clock stays held, the card up, through the fresh solve's first
    // estimate; it runs once the zone in view has its fresh final (T-421), which can come a few hundred
    // ms before the solve's last round (`refined`). T-2560: with the cache it runs as the cache shows,
    // or as the network is drawn when the cache came ahead of it (T-1539), before the fresh solve's final
    const released = (r) => r.clock !== undefined && r.freshFinal !== undefined && (r.cached
      ? r.firstFinalKey?.startsWith('0:') && r.clock <= Math.max(r.firstFinal, r.world) + 2000 && r.clock < r.freshFinal
      : r.clock >= r.solves.at(-1).first && r.clock <= r.freshFinal + 1000);
    const heldRight = out.runs.every(released);
    ok &&= heldRight;
    console.log(`${heldRight ? 'PASS' : 'FAIL'}  the clock runs at the cached view, else when the zone in view has its fresh final (clock ${out.runs.map((r) => r.clock).join('/')}, cached view ${out.runs.map((r) => (r.cached ? r.firstFinal : '-')).join('/')}, first estimate ${out.runs.map((r) => r.solves.at(-1).first).join('/')}, final ${out.runs.map((r) => r.freshFinal).join('/')})`);
  } else if (MODE === 'edit') {
    // T-2560: an edit made while the cached result's network is solved again, the clock already running,
    // ends at the figures a load of the edited network solves from cold (a reload that misses the cache)
    let r = await reload();
    if (!summary(r).cached) r = await reload(); // (the real save's first reload misses: T-2552's fare rounding)
    await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}${QUERY}`);
    await page.waitForFunction(() => window.tw?.clock?.held && window.tw.clock.held.peek() === false && window.twDemand?.view?.()?.version === 0, undefined, { timeout: 120000, polling: 50 });
    const pre = await page.evaluate(() => { const c = window.twDemand.client; return { published: c.cur?.published ?? null, waiting: !!c.waiting, cacheSolve: !!c.cacheSolve }; });
    const solving = pre.waiting || (pre.published ?? -1) < 3;
    const pick = await page.evaluate(() => {
      const w = window.tw.world.value;
      const lines = w.save.lines.filter((l) => w.lineStats?.[l.id]?.trainsNeeded?.[0] > 0);
      lines.sort((a, b) => w.lineStats[b.id].trainsNeeded[0] - w.lineStats[a.id].trainsNeeded[0]);
      const l = lines[Math.floor(lines.length / 2)];
      return { id: l.num, trains: w.lineStats[l.id].trainsNeeded.slice(0, 3), version: window.twDemand.client.version, saved: window.tw.lastSaved.peek() };
    });
    await page.evaluate((p) => window.tw.client.edit({ op: 'setTrainCount', line: p.id, trains: p.trains.map((t) => t + 1) }), pick);
    await page.waitForFunction((v) => window.twDemand.client.version > v, pick.version, { timeout: 60000, polling: 100 });
    await demandFinal(page, { timeout: 300000, polling: 100, after: pick.version });
    await page.waitForTimeout(1500);
    const edited = await page.evaluate(() => ({ sig: window.__sig(), views: window.__ev.filter((e) => e[1] === 'view').map((e) => e[2]) }));
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, pick.saved, { timeout: 60000 }).catch(() => {});
    await page.goto(`${BASE}?debug=1&continue=1&perfOff=nocache&v=${Date.now()}${QUERY}`);
    await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
    await waitSolved();
    const cold = await page.evaluate(() => ({ sig: window.__sig(), cached: window.__ev.some((e) => e[1] === 'view' && /^0:/.test(e[2])) }));
    if (edited.sig.hash !== cold.sig.hash) {
      const z1 = edited.sig.zones, z2 = cold.sig.zones;
      log(`edit and cold differ in ${Object.keys(cold.sig.sections).filter((k) => edited.sig.sections[k] !== cold.sig.sections[k]).join(', ')}; zones only after the edit ${Object.keys(z1).filter((k) => !(k in z2)).join(',') || '-'}, only cold ${Object.keys(z2).filter((k) => !(k in z1)).join(',') || '-'}, differing ${Object.keys(z1).filter((k) => k in z2 && z1[k] !== z2[k]).join(',') || '-'}`);
    }
    const pass = solving && pre.cacheSolve && !cold.cached && sameFigures(edited.sig, cold.sig);
    ok &&= pass;
    out.runs.push({ pre, pick, edited, cold });
    console.log(`${pass ? 'PASS' : 'FAIL'}  an edit while the cache's network is solved again (clock running, solve ${pre.waiting ? 'waiting for packs' : 'at round ' + pre.published}) ends at the cold solve's figures (edit ${edited.sig.hash}, cold ${cold.sig.hash}${cold.cached ? ', but the cold load showed a cache' : ''}; views ${edited.views.join(' ')})`);
    // the trains put back, for the next run
    await page.evaluate(async (p) => { await window.tw.client.edit({ op: 'setTrainCount', line: p.id, trains: p.trains }); }, pick);
    await page.waitForTimeout(1500);
    const saved = await page.evaluate(() => window.tw.lastSaved.peek());
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, saved, { timeout: 60000 }).catch(() => {});
  } else if (MODE === 'stale') {
    // the cache entry, as the page stores it (IndexedDB "anitabuilder", store "saves", keys "demand:...")
    const tamper = (field) => page.evaluate((field) => new Promise((res, rej) => {
      const o = indexedDB.open('anitabuilder', 1);
      o.onsuccess = () => {
        const tx = o.result.transaction('saves', 'readwrite'), st = tx.objectStore('saves');
        // resolved once committed: a reload right after a put's success could abort it (T-1539)
        let n = 0;
        tx.oncomplete = () => res(n);
        const k = st.getAllKeys();
        k.onsuccess = () => {
          const keys = k.result.filter((x) => typeof x === 'string' && x.startsWith('demand:'));
          for (const key of keys) {
            const g = st.get(key);
            g.onsuccess = () => {
              const v = g.result;
              if (field === 'build') v.build = 'x' + v.build;
              else v.packs[Object.keys(v.packs)[0]] = 'x';
              st.put(v, key).onsuccess = () => { n++; };
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
