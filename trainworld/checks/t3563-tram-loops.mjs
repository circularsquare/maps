// T-3563: trams use passing loops as trains do. A new game, four made-up lines, each 12 km of two-way single track
// underground with a stop every 3 km, the three between the ends placed as passing stations (loop) or as stops with no
// passing loop (noloop), on rail track and on tram track. Every line is held to 60 km/h so rail and tram run alike.
// Per line, at 1 to 4 trains: round trip, trains an hour, the limiting headway and its kind (the single track's), the
// train cap. Then:
//  - tram loop: single-track stretches of 3 km, so a shorter limiting headway than tram noloop's one of 12 km, and
//    more trams fit (they meet at the loops);
//  - tram loop against rail loop, and tram noloop against rail noloop: the same cap and dwell (30 s, T-3921); the
//    headways are equal (T-3921: a tram brakes at a train's 0.9 m/s2 and stands 30 s; at 0.8 a tram's was 1.2 s a stop longer).
// With BEFORE=1 it only prints (on the old code tram loop and tram noloop come out the same).
// Screenshot artifacts/trainworld/t3563-tram-loop.png: the tram loop stop with the line selected.
// From the maps root (or a tree root), on a Vite of your own:   node trainworld/checks/t3563-tram-loops.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6142/';
const BEFORE = process.env.BEFORE === '1';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3563-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });
  await page.evaluate(() => {
    const tw = window.tw, c = tw.client;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const at = (lng, lat) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; };
    // 12 km of single track east from [lng, lat], a stop every 3 km; those between the ends with no loop when `noloop`
    window.__build = async (lng, lat, tram, noloop, name) => {
      const dLng = 1 / (111.32 * Math.cos((lat * Math.PI) / 180));
      const xy = [0, 3, 6, 9, 12].map((km) => at(lng + km * dLng, lat));
      for (let i = 0; i + 1 < xy.length; i++) {
        const from = i === 0 ? { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 } : { kind: 'node', node: near(xy[i]) };
        const r = await c.edit({ op: 'route', from, to: { kind: 'free', x: xy[i + 1][0], y: xy[i + 1][1], level: -1 }, pis: [], single: true, tram, street: false });
        if (!r.ok) throw new Error(`${name}: route refused ${JSON.stringify(r.issues ?? r)}`);
        await pause(400);
      }
      for (const [k, p] of xy.entries()) {
        const one = k > 0 && k + 1 < xy.length && noloop ?{ one: { x: p[0], y: p[1] - 10 } } : {};
        const s = await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p), ...one }, platform: 80, name: `${name} ${k + 1}` });
        if (!s.ok) throw new Error(`${name}: station ${k + 1} refused ${JSON.stringify(s.issues ?? s)}`);
        await pause(250);
      }
      await c.edit({ op: 'constructAll' });
      await pause(600);
      const l = await c.edit({ op: 'addLine', stops: xy.map(near), name, colour: tram ? '#c0392b' : '#0a7d3b' });
      if (!l.ok) throw new Error(`${name}: line refused ${JSON.stringify(l.issues ?? l)}`);
      await c.edit({ op: 'lineLimit', line: l.line, kmh: 60 });
      await pause(1500);
      const line = tw.world.value.save.lines.find((x) => x.num === l.line);
      const mid = near(xy[1]);
      const st = tw.world.value.save.stations.find((s) => String(s.num) === String(mid) || s.id === String(mid));
      return { num: l.line, id: line.id, stops: xy.map(near), tram: !!tw.world.value.lineStats[line.id]?.tram, dwell: tw.world.value.lineStats[line.id]?.dwellS, one: !!st?.one, mid: st ? { id: st.id, lng: st.lng, lat: st.lat } : null };
    };
    window.__info = async (num) => {
      const { queryClock } = await import('/src/workers/clockClient.ts');
      const { trainCap } = await import('/src/game/world.ts');
      const w = tw.world.value, l = w.save.lines.find((x) => x.num === num), st = w.lineStats[l.id];
      const d = await queryClock({ what: 'lineHeld', line: num });
      const trip = st.roundTripByLevel[0];
      return { n: st.trainsNeeded[0], trip: Math.round(trip), tph: +(l.tph?.high ?? 0).toFixed(2), limitS: d[21] > 0 ? Math.round(d[21]) : null, kind: d[21] > 0 ? d[22] : null, capS: d[36] > 0 ? Math.round(d[36]) : null, cap: trainCap(st, trip, d[36] > 0 ? d[36] : d[21] > 0 ? d[21] : null) };
    };
  });
  const info = (num) => page.evaluate((x) => window.__info(x), num);
  const settle = async (num, want) => {
    for (let k = 0; k < 30; k++) { await page.waitForTimeout(500); if (want === undefined || (await info(num)).n === want) break; }
    await page.waitForTimeout(1500);
    return info(num);
  };
  const apiSet = async (num, n) => { await page.evaluate(([num, n]) => window.tw.client.edit({ op: 'setTrainCount', line: num, trains: [n, n, n] }), [num, n]); return settle(num, n); };

  const cases = [
    { key: 'tram loop', tram: true, noloop: false, lat: 42.30 },
    { key: 'tram noloop', tram: true, noloop: true, lat: 42.34 },
    { key: 'rail loop', tram: false, noloop: false, lat: 42.38 },
    { key: 'rail noloop', tram: false, noloop: true, lat: 42.42 },
  ];
  const got = {};
  for (const c of cases) {
    const b = await page.evaluate(([lat, tram, noloop, name]) => window.__build(-71.06, lat, tram, noloop, name), [c.lat, c.tram, c.noloop, c.key]);
    const rows = [];
    for (const n of [1, 2, 3, 4]) rows.push(await apiSet(b.num, n));
    got[c.key] = { b, rows, head: rows[1].limitS, kind: rows[1].kind, cap: rows[1].cap };
    console.log(`${c.key}: tram ${b.tram}, mid stop one-track ${b.one}`);
    for (const r of rows) console.log(`  ${r.n} trains: round trip ${r.trip} s, ${r.tph}/h, limit ${r.limitS} s (kind ${r.kind}), cap headway ${r.capS} s, cap ${r.cap}`);
    check(b.tram === c.tram && b.one === c.noloop, `${c.key}: built as asked`);
  }
  const [tl, tn, rl, rn] = ['tram loop', 'tram noloop', 'rail loop', 'rail noloop'].map((k) => got[k]);
  if (!BEFORE) {
    check(tl.kind === 1 && tn.kind === 1 && rl.kind === 1 && rn.kind === 1, 'every line is limited by its single track', [tl, tn, rl, rn].map((x) => x.kind).join(', '));
    check(tl.head < tn.head - 60, 'tram loop: the trams meet at the loop, a shorter headway than with no loop', `${tl.head} s against ${tn.head} s`);
    check(tl.cap > tn.cap, 'tram loop: more trams fit than with no loop', `${tl.cap} against ${tn.cap}`);
    // T-3921: a new tram line stands a train's 30 s; braking is a train's, so the headways are equal
    check([tl, tn, rl, rn].every((x) => x.b.dwell === 30), 'new tram and train lines both stand 30 s', [tl, tn, rl, rn].map((x) => x.b.dwell).join(', '));
    check(tl.cap === rl.cap && tl.head === rl.head, 'tram loop runs as rail loop', `cap ${tl.cap} / ${rl.cap}, headway ${tl.head} s / ${rl.head} s`);
    check(tn.cap === rn.cap && tn.head === rn.head, 'tram noloop runs as rail noloop', `cap ${tn.cap} / ${rn.cap}, headway ${tn.head} s / ${rn.head} s`);
  }
  // the tram loop stop, its line selected
  await page.evaluate(([id, s]) => { const tw = window.tw; tw.selection.value = { kind: 'line', line: id }; if (s) tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 15.5 }); }, [tl.b.id, tl.b.mid]);
  await page.waitForTimeout(2500);
  await page.screenshot({ path: join(ARTIFACTS, BEFORE ? 't3563-tram-loop-before.png' : 't3563-tram-loop.png') });
  check(errors.length === 0, 'no page errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  console.log('ERROR', e.stack ?? e);
  fail++;
} finally {
  await chrome.close();
}
console.log(fail ? `${fail} FAILED` : 'ALL PASS');
process.exit(fail ? 1 : 0);
