// T-3526: a line alone on two-way single track does not queue against its own trains (its two directions are one
// stream in the wait), so the trains past what the stretch carries are held at the ends as at a terminal, and the
// panel's train count stops at the trains that fit plus one (T-3165). A new game, three made-up lines:
//  A one 5 km single piece, two stations, 100 km/h;  B 5 km double, 5 single, 5 double, 100 km/h;
//  C 20 km double then 1 km single to the end, 160 km/h.
// Per case: round trip, trains an hour and held trains by count (the API), then the panel's own control: typing 20 on
// a line of one train and pressing + from one train both stop at the cap (A 2, B 2, C 3). Then a shuttle over B's
// single stretch: two lines on one stretch still wait for each other.
// Screenshot artifacts/trainworld/t3526-B-cap.png.
// From the maps root, on a Vite of your own:   node trainworld/checks/t3526-single-cap.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6113/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3526-', size: [1400, 900], deviceScaleFactor: 1 });
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
    // pieces east from [lng, lat], each [km, single]; a named station at every node
    window.__build = async (lng, lat, pieces, name, kmh) => {
      const dLng = 1 / (111.32 * Math.cos((lat * Math.PI) / 180));
      const xy = [at(lng, lat)];
      let km = 0;
      for (const [k] of pieces) { km += k; xy.push(at(lng + km * dLng, lat)); }
      for (let i = 0; i < pieces.length; i++) {
        const from = i === 0 ? { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 } : { kind: 'node', node: near(xy[i]) };
        await c.edit({ op: 'route', from, to: { kind: 'free', x: xy[i + 1][0], y: xy[i + 1][1], level: -1 }, pis: [], single: pieces[i][1] });
        await pause(400);
      }
      for (const [k, p] of xy.entries()) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: `${name} ${k + 1}` }); await pause(250); }
      await c.edit({ op: 'constructAll' });
      await pause(600);
      const r = await c.edit({ op: 'addLine', stops: xy.map(near), name, colour: '#0a7d3b' });
      if (kmh) await c.edit({ op: 'lineLimit', line: r.line, kmh });
      await pause(1500);
      const l = tw.world.value.save.lines.find((x) => x.num === r.line);
      return { num: r.line, id: l.id, stops: xy.map(near) };
    };
    window.__info = async (num) => {
      const { queryClock } = await import('/src/workers/clockClient.ts');
      const { trainCap } = await import('/src/game/world.ts');
      const w = tw.world.value, l = w.save.lines.find((x) => x.num === num), st = w.lineStats[l.id];
      const d = await queryClock({ what: 'lineHeld', line: num });
      const waits = Array.from(await queryClock({ what: 'lineDelays', line: num, level: 0 }));
      let single = 0;
      for (let i = 0; i + 4 < waits.length; i += 5) if (waits[i] === 1) single += waits[i + 4];
      const trip = st.roundTripByLevel[0];
      return { n: st.trainsNeeded[0], trip: Math.round(trip), tph: +(l.tph?.high ?? 0).toFixed(2), held: +(d[0] || 0).toFixed(2), limitS: d[21] > 0 ? Math.round(d[21]) : null, kind: d[21] > 0 ? d[22] : null, capS: d[36] > 0 ? Math.round(d[36]) : null, cap: trainCap(st, trip, d[36] > 0 ? d[36] : d[21] > 0 ? d[21] : null), singleWait: Math.round(single) };
    };
  });
  const info = (num) => page.evaluate((x) => window.__info(x), num);
  const settle = async (num, want) => {
    for (let k = 0; k < 30; k++) { await page.waitForTimeout(500); if (want === undefined || (await info(num)).n === want) break; }
    await page.waitForTimeout(1500);
    return info(num);
  };
  const apiSet = async (num, n) => { await page.evaluate(([num, n]) => window.tw.client.edit({ op: 'setTrainCount', line: num, trains: [n, n, n] }), [num, n]); return settle(num, n); };
  const typedBtn = page.locator('button.typed[aria-label^="Type trains on the line"]').first();
  const plusBtn = page.locator('.stepper:has(button.typed[aria-label^="Type trains on the line"]) button[aria-label^="More"]').first();
  const type = async (text) => { await typedBtn.click(); await page.keyboard.press('Control+A'); await page.keyboard.type(text); await page.keyboard.press('Enter'); };
  const red = () => page.evaluate(() => [...document.querySelectorAll('table td .bad')].some((e) => /\d/.test(e.textContent ?? '')));
  const alert = () => page.evaluate(() => document.body.innerText.match(/[^\n]*Single track[^\n]*is full[^\n]*/g) ?? []);

  // free: the round trip of one train; head: the stretch's headway with the margin; cap: the trains that fit plus one
  const cases = [
    { name: 'A', lng: -71.06, lat: 42.30, pieces: [[5, true]], kmh: 100, cap: 2 },
    { name: 'B', lng: -71.06, lat: 42.40, pieces: [[5, false], [5, true], [5, false]], kmh: 100, cap: 2 },
    { name: 'C', lng: -71.06, lat: 42.50, pieces: [[10, false], [10, false], [1, true]], kmh: 0, cap: 3 },
  ];
  let B = null;
  for (const c of cases) {
    const b = await page.evaluate(([lng, lat, pieces, name, kmh]) => window.__build(lng, lat, pieces, name, kmh), [c.lng, c.lat, c.pieces, c.name, c.kmh]);
    if (c.name === 'B') B = b;
    await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, b.id);
    const rows = [];
    for (const n of [1, 2, 3, 4]) rows.push(await apiSet(b.num, n));
    for (const r of rows) console.log(`  ${c.name} ${r.n} trains: round trip ${r.trip} s, ${r.tph}/h, held ${r.held}, waits at single track ${r.singleWait} s, limit ${r.limitS} s (kind ${r.kind}), cap headway ${r.capS} s, cap ${r.cap}`);
    const one = rows[0], head = rows[1].limitS, most = 3600 / head;
    check(rows.every((r) => r.singleWait <= 5), `${c.name}: the line alone never waits at its single track`, rows.map((r) => r.singleWait).join(', '));
    check(rows.slice(1).every((r) => Math.abs(r.trip - rows[1].trip) <= 10) && Math.abs(rows[1].trip - one.trip) <= 45, `${c.name}: the round trip stays put as trains are added`, rows.map((r) => r.trip).join(', '));
    const full = rows.find((r) => r.n === c.cap);
    check(Math.abs(full.tph - most) < 0.03, `${c.name}: at ${c.cap} trains it runs the stretch's headway`, `${full.tph}/h against ${most.toFixed(2)}/h, every ${head} s`);
    check(full.held > 0 && full.held < 1.5, `${c.name}: at ${c.cap} trains part of a train is held at the ends`, `${full.held}`);
    check(rows.slice(1).every((r) => r.cap === c.cap) && one.capS === head, `${c.name}: the cap is ${c.cap} at every count, the lone train's taken with the margin`, `caps ${rows.map((r) => r.cap).join(', ')}; one train's cap headway ${one.capS} s, two trains' limit ${head} s`);
    // the panel's control
    await apiSet(b.num, 1);
    await page.waitForTimeout(1500);
    check(!(await red()) && (await alert()).length === 0, `${c.name}: one train: no red headway, no full alert (T-3201)`);
    await type('20');
    const typed = await settle(b.num);
    check(typed.n === c.cap, `${c.name}: typing 20 on a line of one train stops at ${c.cap}`, `${typed.n}`);
    await apiSet(b.num, 1);
    await page.waitForTimeout(1500);
    const seq = [];
    for (let k = 0; k < 6; k++) {
      await plusBtn.click();
      seq.push((await settle(b.num)).n);
      if (seq.length > 1 && seq.at(-1) === seq.at(-2)) break;
    }
    check(seq.at(-1) === c.cap, `${c.name}: + from one train stops at ${c.cap}`, seq.join(' '));
    check(await red(), `${c.name}: at the cap the headway is red`);
    const tip = await plusBtn.getAttribute('data-tip');
    check(/^Single track (near|at) .+ takes one train every/.test(tip ?? ''), `${c.name}: + at the cap names the single track`, (tip ?? '').split('\n')[0]);
    if (c.name === 'B') await page.screenshot({ path: join(ARTIFACTS, 't3526-B-cap.png') });
    await apiSet(b.num, 2);
  }
  // two lines on one stretch still wait for each other
  const s = await page.evaluate(async (stops) => window.tw.client.edit({ op: 'addLine', stops: [stops[1], stops[2]], name: 'Shuttle', colour: '#aa3300' }), B.stops);
  await page.evaluate((num) => window.tw.client.edit({ op: 'lineLimit', line: num, kmh: 100 }), s.line);
  await apiSet(s.line, 1);
  const both = await apiSet(B.num, 1);
  const sh = await info(s.line);
  console.log(`  B with the shuttle: B ${JSON.stringify(both)}; shuttle ${JSON.stringify(sh)}`);
  check(both.singleWait > 0 && sh.singleWait > 0, 'B and a shuttle on its single stretch wait for each other there', `${both.singleWait} s, ${sh.singleWait} s`);
  check(errors.length === 0, 'no page errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  console.log('ERROR', e.stack ?? e);
  fail++;
} finally {
  await chrome.close();
}
console.log(fail ? `${fail} FAILED` : 'ALL PASS');
process.exit(fail ? 1 : 0);
