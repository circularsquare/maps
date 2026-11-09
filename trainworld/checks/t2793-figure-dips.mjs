// T-2793: between an edit and its final, does any line's shown riders, fares or booked money fall on a temporary answer?
//   TW_CORES=3 node trainworld/checks/t2793-figure-dips.mjs <save> [url] [edits]     (from the maps root)
// TW_TAG names the output files; TW_WORKERS sets the demand workers (3 by default, a 16-thread desktop); TW_PERFOFF=keepFigures
// runs the old behaviour for a before/after comparison of the demand route.
// edits: comma list of headwayFast, headwayLong, trains, dropStop, headwayCity, twoEdits, zone, openZone, burst, burstView (default the first seven and openZone).
// Every 250 ms, and on each demand publish, each running line's figures: riders as the Lines tab shows them, fares a
// day, the riders a day in the demand view, and the fare income posted to the clock. Prints per edit the lines that fell
// by more than half against the smaller of their figure before the edit and after the final.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { launchChrome, importSave, ANGLE, ARTIFACTS } from './lib.mjs';

const SAVE = process.argv[2];
const BASE = process.argv[3] ?? 'http://localhost:6103/';
const ONLY = process.argv[4]?.split(',') ?? ['headwayFast', 'headwayLong', 'trains', 'dropStop', 'headwayCity', 'twoEdits', 'zone', 'openZone'];
const WORKERS = process.env.TW_WORKERS ?? '3'; // as on a 16-thread desktop, where the market periods are shared between workers (T-1607)
const OUT = ARTIFACTS + `t2793/${process.env.TW_TAG ?? 'run'}-`;
await mkdir(OUT, { recursive: true });

const { ctx, close } = await launchChrome({ prefix: 'tw-t2793-', size: [1300, 850], args: ANGLE });
const stamp = () => new Date().toISOString().slice(11, 19);
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  page.on('console', (m) => { if (m.type() === 'error') console.log('console.error', m.text().slice(0, 300)); });
  // TW_CACHED=1: the save goes in a slot and the page is opened again, so an edit lands on a cached result (T-2533, T-2560);
  // TW_DURINGLOAD=1: the first edit is made while the load's own estimates are shown
  const CACHED = !!process.env.TW_CACHED, DURING_LOAD = !!process.env.TW_DURINGLOAD;
  const QUERY = `?debug=1&${CACHED ? 'continue=1' : 'new=1'}&demandWorkers=${WORKERS}${process.env.TW_PERFOFF ? `&perfOff=${process.env.TW_PERFOFF}` : ''}`;
  await page.goto(`${BASE}${QUERY}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const install = () => page.evaluate(async () => {
    // the recorder: everything is sampled in the page, in plain arrays
    const [td, lf, mm, dm, ui] = await Promise.all(['today', 'lineFigures', 'money', 'demand', 'ui'].map((m) => import(`/src/game/${m}.ts`)));
    const tw = window.tw, R = (window.__rec = { td, lf, mm, dm, ui, posted: null, postedCount: 0, pubs: [], on: false, t0: 0, tick: null });
    const post = tw.client.post.bind(tw.client);
    tw.client.post = (msg, tr) => {
      if (msg.kind === 'income') {
        R.postedCount++;
        R.posted = new Map(msg.lines.map((l) => [String(l.line), l.perHour[0].reduce((a, b) => a + b, 0)]));
      }
      return post(msg, tr);
    };
    // every kind of day of every publish, each line against its figure when the edit began (a dip can be shorter than a sample)
    R.baseDays = null;
    R.dayRiders = (days) => days.map((d) => new Map([...(d.view?.lines ?? [])].map(([id, l]) => [id, l.ridersPerDay ?? 0])));
    dm.demandByDay.subscribe((days) => {
      if (!R.on) return;
      const v = days[0]?.view;
      if (!v) return R.pubs.push({ t: Math.round(performance.now() - R.t0), none: true });
      let sum = 0;
      for (const l of v.lines.values()) sum += l.ridersPerDay ?? 0;
      const rec = { t: Math.round(performance.now() - R.t0), version: v.version, round: v.round, final: !!v.final, lines: v.lines.size, sum: Math.round(sum), est: [...v.lines.values()].filter((l) => l.estimate).length };
      if (R.baseDays) {
        let worst = 1, at = '', absent = 0;
        days.forEach((d, k) => { const b = R.baseDays[k]; if (!b || !d.view) return; for (const [id, r0] of b) { if (r0 < 50) continue; const l = d.view.lines.get(id); if (!l) { absent++; continue; } const q = (l.ridersPerDay ?? 0) / r0; if (q < worst) { worst = q; at = `${k}:${id}`; } } });
        rec.worst = +worst.toFixed(3); rec.at = at; rec.absent = absent;
      }
      R.pubs.push(rec);
    });
    R.line = (l) => {
      const w = tw.world.value, st = w.lineStats[l.id];
      if (st?.status !== 'running') return null;
      const d = td.today.value.get(l.id), k = td.pendingOf(l);
      const riders = k ? td.pendingRiders(l) : d?.riders ?? 0;
      return [riders ?? 0, d?.dayFares ?? 0, dm.demandView.value?.lines.get(l.id)?.ridersPerDay ?? 0, R.posted?.get(l.id) ?? 0, k ? 1 : 0, d?.riders ?? 0, d?.fares ?? 0];
    };
    R.snap = () => {
      const o = {};
      for (const l of tw.world.value.save.lines) { const x = R.line(l); if (x) o[l.id] = x; }
      return o;
    };
    R.busy = () => { const d = window.twDemand, v = d?.view?.(); return !(v?.final && !d.client.waiting) || dm.demandUpdating.value || d.client.respawn?.size > 0; };
    // which publishing route ran when, for reading a dip against
    R.calls = [];
    const c = window.twDemand.client;
    for (const m of ['publish', 'publishCluster', 'publishNewLines', 'publishFocus', 'publishRefine', 'fillNewZones', 'selectedRound', 'workerDied', 'dispatch', 'applyCache', 'dropEarly']) {
      const f = c[m];
      if (typeof f !== 'function') continue;
      c[m] = function (...a) { R.calls.push([Math.round(performance.now() - R.t0), m, a[0]?.version ?? a[0]?.cities?.length ?? null, typeof a[1] === 'number' ? a[1] : (typeof a[3] === 'number' ? a[3] : null), a[2]?.local ? 'local' : '']); return f.apply(this, a); };
    }
  });
  await install();
  await importSave(page, await readFile(SAVE));
  console.log(stamp(), 'imported');
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) >= 20, undefined, { timeout: 300000, polling: 250 });
  // the load's own figures: from the moment the world is there
  const loadT = await page.evaluate(() => {
    const R = window.__rec; R.on = true; R.t0 = performance.now(); R.load = []; R.ui.tab.value = 'lines';
    R.timer = setInterval(() => R.load.push({ t: Math.round(performance.now() - R.t0), busy: R.busy(), lines: R.snap(), card: !!document.querySelector('.load-card, .loading-card') }), 500);
    return window.tw.world.value.save.lines.length;
  });
  console.log(stamp(), 'lines', loadT);
  const settled = () => page.waitForFunction(() => {
    const R = window.__rec; const ok = !R.busy() && !R.lf.linesPending.value.size && R.td.today.value.size > 5;
    if (ok) R.quiet = (R.quiet ?? 0) + 1; else R.quiet = 0;
    return R.quiet > 8;
  }, undefined, { timeout: 900000, polling: 500 });
  if (DURING_LOAD) {
    // as soon as there is a first estimate and the Lines tab reads something, a train added to the busiest line
    await page.waitForFunction(() => window.__rec.td.today.value.size > 5 || window.__rec.dm.demandView.value, undefined, { timeout: 400000, polling: 200 });
    const r = await page.evaluate(async () => {
      const w = window.tw.world.value, R = window.__rec;
      const l = w.save.lines.filter((x) => w.lineStats[x.id]?.status === 'running').sort((a, b) => w.lineStats[b.id].km - w.lineStats[a.id].km)[0];
      const a = await window.tw.client.edit({ op: 'setTrainCount', line: Number(l.num), trains: w.lineStats[l.id].trainsNeeded.map((n) => (n > 0 ? n + 1 : 0)) });
      return { name: l.name, ok: a?.ok, view: R.dm.demandView.value?.final ? 'final' : 'estimate', t: Math.round(performance.now() - R.t0) };
    });
    console.log(stamp(), 'edit during the load', JSON.stringify(r));
  }
  await settled();
  const load = await page.evaluate(() => { const R = window.__rec; clearInterval(R.timer); const a = R.load; R.load = null; return a; });
  await writeFile(OUT + 'load.json', JSON.stringify(load));
  console.log(stamp(), 'loaded and final; load samples', load.length);
  // the clock runs, as in play
  await page.evaluate(() => { window.tw.clock.setSpeed(1); });
  await page.waitForTimeout(8000);
  await settled();

  const runEdit = async (name) => {
    const r = await page.evaluate(async (name) => {
      const R = window.__rec, tw = window.tw, { td, lf, dm } = R, w = () => tw.world.value;
      const sleep = (ms) => new Promise((f) => setTimeout(f, ms));
      R.ui.selection.value = null; R.ui.tab.value = 'lines';
      const running = w().save.lines.filter((l) => w().lineStats[l.id]?.status === 'running' && !w().lineStats[l.id]?.bus);
      const rid = (l) => td.today.value.get(l.id)?.riders ?? 0;
      const byKm = [...running].sort((a, b) => w().lineStats[b.id].km - w().lineStats[a.id].km);
      const fast = running.filter((l) => w().lineStats[l.id].fast).sort((a, b) => rid(b) - rid(a));
      const pick = { headwayFast: fast[0] ?? byKm[0], headwayLong: byKm[0], trains: byKm[1], dropStop: byKm.find((l) => l.stops.length > 6) ?? byKm[2], headwayCity: running.filter((l) => w().lineStats[l.id].km < 40).sort((a, b) => rid(b) - rid(a))[0] }[name];
      const headwayOp = (l, f = 1.15) => {
        const tph = [l.tph.high, l.tph.medium, l.tph.low];
        return { op: 'setHeadway', line: Number(l.num), seconds: tph.map((n) => (n > 0 ? Math.round((3600 / n) * f) : 0)) };
      };
      let ops = [], note;
      if (name === 'trains') { const t = w().lineStats[pick.id].trainsNeeded; ops = [{ op: 'setTrainCount', line: Number(pick.num), trains: t.map((n) => (n > 0 ? n + 1 : 0)) }]; note = `${pick.name} ${w().lineStats[pick.id].km | 0} km`; }
      else if (name === 'dropStop') { ops = [{ op: 'dropStop', line: Number(pick.num), index: pick.stops.length >> 1 }]; note = `${pick.name} ${w().lineStats[pick.id].km | 0} km, stops ${pick.stops.length}`; }
      else if (name === 'twoEdits') { ops = [headwayOp(byKm[2]), headwayOp(byKm[4])]; note = `${byKm[2].name}, ${byKm[4].name}`; }
      else if (!['zone', 'openZone', 'roughView', 'forceRough', 'restart', 'restartEdit', 'editDuring', 'stopThenSchedule', 'lod', 'lodEdit', 'fast', 'lineKind', 'lineCars'].includes(name) && !name.startsWith('burst')) { ops = [headwayOp(pick)]; note = `${pick.name} ${w().lineStats[pick.id].km | 0} km fast=${w().lineStats[pick.id].fast}`; }
      const base = R.snap(), samples = [];
      R.pubs = []; R.calls = []; R.t0 = performance.now(); R.baseDays = R.dayRiders(dm.demandByDay.value);
      let edited = false;
      const out = { name, note, ops };
      if (name === 'forceRough') {
        // a rough solve of the zone in view, which shows no figures (the state she can reach while a pack loads), forced
        const { activeCity } = await import('/src/game/cities.ts');
        const first = w().save.stations.find((s) => s.id === byKm[0].stops[0]);
        tw.map.jumpTo({ center: [first.lng, first.lat], zoom: 9.5 });
        await sleep(4000);
        const c = window.twDemand.client, id = activeCity.peek(), svc = c.latest();
        out.note = `rough solve of ${id}, viewed ${c.viewedSet().has(id)}, open ${c.opened.has(id)}`;
        dm.demandByDay.value = dm.demandByDay.value.map((d) => ({ ...d, cities: new Map([...d.cities].filter(([k]) => k !== id)) }));
        c.rough = null;
        out.sent = c.roughSoon.length;
        c.roughSoon(svc, c.solveNeed(svc));
        out.rough = !!c.cur?.rough;
      } else if (name === 'roughView') {
        // the zone in view has no figures while a schedule edit waits for another pack: a rough solve of that zone alone
        const { CITIES } = await import('/src/game/cities.ts');
        const c = window.twDemand.client, a = w().save.stations.find((s) => s.id === byKm[0].stops[0]);
        const dist = (x) => (x.center[0] - a.lng) ** 2 + (x.center[1] - a.lat) ** 2;
        const by = CITIES.filter((x) => !c.isOpen(x.id) && !c.fetching.has(x.id) && x.center).sort((p, q) => dist(p) - dist(q));
        const [y, z] = by;
        out.note = `in view ${y.id}, waiting on ${z.id}`;
        tw.map.jumpTo({ center: y.center, zoom: 10 });
        await sleep(2500);
        out.active = (await import('/src/game/cities.ts')).activeCity.peek();
        c.want(y.id, true); c.want(z.id, true);
        const l = byKm[2];
        await tw.client.edit(headwayOp(l, 1.07));
        R.waitOpen = z.id;
      } else if (name === 'openZone') {
        // a zone joins the solve with figures already shown everywhere (a line built or extended into it, a pack that took
        // long): the nearest zone not open, opened in every worker
        const { CITIES } = await import('/src/game/cities.ts');
        const c = window.twDemand.client, a = w().save.stations.find((s) => s.id === byKm[0].stops[0]);
        const by = CITIES.filter((x) => !c.isOpen(x.id) && !c.fetching.has(x.id) && x.center).sort((p, q) => ((p.center[0] - a.lng) ** 2 + (p.center[1] - a.lat) ** 2) - ((q.center[0] - a.lng) ** 2 + (q.center[1] - a.lat) ** 2));
        out.note = `opened ${by[0]?.id}, near ${a.name}`;
        c.want(by[0].id, true);
        R.waitOpen = by[0].id;
      } else if (name === 'restart' || name === 'restartEdit') {
        // every worker restarted to give back its memory (T-1205), as `memCheck` does; the same network is solved again from nothing
        const c = window.twDemand.client, n = c.computeN;
        out.note = `${n} workers restarted${name === 'restartEdit' ? ', then a headway edit' : ''}`;
        c.debugRestart(...Array.from({ length: n }, (_, i) => i));
        if (name === 'restartEdit') {
          await sleep(4000);
          const l = fast[0] ?? byKm[0];
          out.answers = [(await tw.client.edit(headwayOp(l, 1.1)))?.ok];
          out.note += ` on ${l.name}`;
        }
      } else if (name === 'editDuring') {
        // a second and a third change while the first one's solve runs, the same long line and another
        const l = fast[0] ?? byKm[0], m = byKm[1];
        out.note = `${l.name}, then again, then ${m.name}`;
        out.answers = [(await tw.client.edit(headwayOp(l, 1.08)))?.ok];
        await sleep(6000);
        out.answers.push((await tw.client.edit(headwayOp(w().save.lines.find((x) => x.id === l.id), 0.95)))?.ok);
        await sleep(9000);
        out.answers.push((await tw.client.edit(headwayOp(m, 1.08)))?.ok);
      } else if (name === 'stopThenSchedule') {
        // a stop dropped, the timetable of the same line changed while its riders are worked out
        const l = byKm.find((x) => x.stops.length > 6) ?? byKm[0];
        out.note = `${l.name} ${w().lineStats[l.id].km | 0} km`;
        out.answers = [(await tw.client.edit({ op: 'dropStop', line: Number(l.num), index: l.stops.length >> 1 }))?.ok];
        await sleep(4000);
        out.answers.push((await tw.client.edit(headwayOp(w().save.lines.find((x) => x.id === l.id), 1.1)))?.ok);
      } else if (name === 'lod') {
        // zoomed far in on one zone and out again (a level of detail refined, then another)
        const a = w().save.stations.find((s) => s.id === byKm[0].stops[byKm[0].stops.length >> 1]);
        out.note = `zoom at ${a.name}`;
        for (const z of [12, 6, 12.5, 7]) { tw.map.jumpTo({ center: [a.lng, a.lat], zoom: z }); await sleep(9000); }
      } else if (name === 'lineKind' || name === 'lineCars') {
        // the run times of a whole long line change, so the local solve opens a window of the market (T-1735): the train kind flipped; or its cars
        const l = fast[0] ?? byKm[0], st = w().lineStats[l.id];
        const op = name === 'lineKind' ? { op: 'lineFast', line: Number(l.num), fast: !st.fast } : { op: 'lineCars', line: Number(l.num), cars: st.cars + 2 };
        out.note = `${l.name} ${st.km | 0} km ${JSON.stringify(op)}`;
        out.answers = [(await tw.client.edit(op))?.ok];
      } else if (name === 'lodEdit') {
        // a stop dropped, then the view straight to a coarse zone on the line (its level refined aside, T-2605, while the edit's solve is in hand)
        const l = byKm.find((x) => x.stops.length > 6) ?? byKm[0], c = window.twDemand.client;
        const stops = l.stops.map((id) => w().save.stations.find((s) => s.id === id));
        out.answers = [(await tw.client.edit({ op: 'dropStop', line: Number(l.num), index: l.stops.length >> 1 }))?.ok];
        out.note = `${l.name} ${w().lineStats[l.id].km | 0} km, then zoomed to its stops`;
        for (const k of [0.15, 0.85, 0.5, 0.3]) {
          const s = stops[Math.floor(k * stops.length)];
          tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 11.5 });
          await sleep(5000);
        }
        out.frefine = c.frefine ?? null;
      } else if (name === 'fast') {
        // the clock at its fastest with a schedule change
        const l = fast[0] ?? byKm[0];
        tw.clock.setSpeed(2);
        out.note = `fast clock, ${l.name}`;
        out.answers = [(await tw.client.edit(headwayOp(l, 1.1)))?.ok];
      } else if (name === 'zone') {
        // to the other end of the network and back: zones opened or refined for the view
        const ls = w().save.lines; const sOf = (l) => w().save.stations.find((s) => s.id === l.stops[0]);
        const a = sOf(byKm[0]), b = sOf(byKm[Math.min(8, byKm.length - 1)]);
        tw.map.jumpTo({ center: [a.lng, a.lat], zoom: 9.5 }); await sleep(12000);
        tw.map.jumpTo({ center: [b.lng, b.lat], zoom: 9.5 }); await sleep(12000);
        out.note = `${a.name}, ${b.name}, of ${ls.length} lines`;
      } else if (name.startsWith('burst')) {
        // the kind of thing she does: a few schedule tweaks on the big lines, irregularly, the camera moving meanwhile
        const big = (fast.length >= 3 ? fast : [...running].sort((a, b) => rid(b) - rid(a))).slice(0, 4);
        const gaps = [300, 700, 1500, 3500, 8000, 11000, 25000];
        tw.clock.setSpeed(2);
        out.note = big.map((l) => l.name).join(', ');
        out.answers = [];
        for (let i = 0; i < gaps.length; i++) {
          const l = w().save.lines.find((x) => x.id === big[i % big.length].id);
          void tw.client.edit(headwayOp(l, i % 2 ? 0.92 : 1.09)).then((a) => out.answers.push(a?.ok));
          if (name === 'burstView' && i % 2) {
            const s = w().save.stations.find((st) => st.id === l.stops[Math.floor(Math.random() * l.stops.length)]);
            tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 8 + 3 * Math.random() });
          }
          await sleep(gaps[i]);
        }
      } else {
        out.answers = [];
        for (let i = 0; i < ops.length; i++) {
          if (i) await sleep(1500);
          void tw.client.edit(ops[i]).then((a) => out.answers.push(a?.ok));
        }
        edited = true;
      }
      let quiet = 0;
      while (performance.now() - R.t0 < 420000) {
        const s = { t: Math.round(performance.now() - R.t0), busy: R.busy(), npend: lf.linesPending.value.size, lines: R.snap(), posted: R.postedCount };
        samples.push(s);
        if (R.waitOpen && window.twDemand.client.isOpen(R.waitOpen)) { out.openedAt = s.t; R.waitOpen = null; }
        quiet = !s.busy && !s.npend && s.t > 10000 && !R.waitOpen ? quiet + 1 : 0;
        if (quiet > 60) break;
        await sleep(250);
      }
      void edited;
      Object.assign(out, { base, samples, pubs: R.pubs, calls: R.calls });
      return out;
    }, name);
    await writeFile(OUT + `${name}.json`, JSON.stringify(r));
    report(r);
    await settled();
  };

  const failures = [];
  const SERIES = [['shown riders', 0], ['fares/day', 1], ['demand riders', 2], ['posted income', 3]];
  function report(r) {
    const fin = r.samples.at(-1);
    console.log(`\n=== ${r.name}: ${r.note} answers ${JSON.stringify(r.answers ?? [])} samples ${r.samples.length} settled at ${fin.t} ms, max pending ${Math.max(...r.samples.map((s) => s.npend))}`);
    console.log('publishes:', r.pubs.map((p) => p.none ? `${p.t}:none` : `${p.t}:v${p.version}r${p.round}${p.final ? 'F' : ''} n${p.lines} s${p.sum} e${p.est} w${p.worst ?? '-'}@${p.at ?? ''} a${p.absent ?? 0}`).join('  '));
    for (const [label, k] of SERIES) {
      const dipped = [];
      for (const id of Object.keys(r.base)) {
        const b = r.base[id][k], f = fin.lines[id]?.[k];
        if (!(b > 0) || !(f >= 0)) continue;
        const ref = Math.min(b, f);
        if (ref <= 0) continue;
        let lo = Infinity, at = 0, pend = 0;
        for (const s of r.samples) { const x = s.lines[id]; if (x && x[k] < lo) { lo = x[k]; at = s.t; pend = x[4]; } }
        const dip = 1 - lo / ref;
        if (dip > 0.1) dipped.push({ id, b: +b.toPrecision(4), f: +f.toPrecision(4), lo: +lo.toPrecision(4), at, pend, dip: +dip.toFixed(2) });
      }
      dipped.sort((a, b) => b.dip - a.dip);
      console.log(`  ${label}: over 50%: ${dipped.filter((d) => d.dip > 0.5).length}, over 10%: ${dipped.length}; worst ${JSON.stringify(dipped.slice(0, 6))}`);
      if (dipped.some((d) => d.dip > 0.5)) failures.push(`${r.name} ${label}`);
    }
  }

  if (DURING_LOAD) console.log(stamp(), '(see the load samples for the edit during the load)');
  if (CACHED) {
    // the save was imported into the slot: flush, open it again, and edit as soon as the cached figures are shown
    await page.evaluate(() => { window.tw.client.post({ kind: 'flush' }); });
    await page.waitForTimeout(5000);
    await page.goto(`${BASE}${QUERY}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await install();
    await page.waitForFunction(() => window.tw.clock.held.peek() === false && window.twDemand?.view?.()?.version === 0 && window.__rec.td.today.value.size > 5, undefined, { timeout: 300000, polling: 100 }).catch((e) => console.log('no cached view:', e.message.slice(0, 200)));
    console.log(stamp(), 'reopened; view version', await page.evaluate(() => window.twDemand.view()?.version));
    await page.evaluate(() => { window.__rec.on = true; window.__rec.ui.tab.value = 'lines'; });
    await page.waitForTimeout(1000);
  }
  for (const name of ONLY) { console.log(stamp(), 'edit', name); await runEdit(name); }
  console.log(stamp(), 'done');
  if (failures.length) { console.log('FAIL', failures.join('; ')); process.exitCode = 1; } else console.log('PASS: no line fell by more than half in any series');
} finally {
  await close();
}
