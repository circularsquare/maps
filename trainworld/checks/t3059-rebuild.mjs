// T-3059: rebuilding built track in place. On a new game in New York: a straight 8 km run with a station at
// each end and one in the middle, built, with a line over it. Checks: a quote names the pieces a rebuild would
// refuse (a pick without the middle station cannot ramp to it) and a rebuild leaves them alone; high speed on
// one piece changes only that piece; a viaduct rebuild of the whole run charges exactly its quote, keeps every
// station's id and name and the line's stops, leaves the line running and clears undo; then the same through
// the Select tool's bar: the level button, the Rebuild hover's price, the confirm and its yes. T-3088: the run,
// its stations and its line rebuilt as tram track (tram stops, a tram line, still running), through the bar's
// Tram button, and back to rail. T-3087: a run drawn at ground across the East River and Roosevelt Island (pieces
// at 0 and on bridges at +1) rebuilt to -1 is at -1 everywhere, with nothing refused. Screenshots
// artifacts/trainworld/t3059-*.png (the refused piece outlined in red).
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3059-rebuild.mjs [dev url]
// About two minutes.
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6182/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.668];

const chrome = await launchChrome({ prefix: 'tw-t3059-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);

/** the track, stations and lines as the page's world holds them */
const snap = () => page.evaluate(() => {
  const w = window.tw.world.value;
  const nodes = new Map([...w.nodes.values()].map((n) => [Number(n.id), n]));
  return {
    cash: w.save.cash,
    edges: w.edges.map((e) => ({ id: Number(e.id), built: e.built, hsr: !!e.hsr, kind: e.kind ?? 0, lo: e.levelMin, hi: e.levelMax, len: e.lengthM, cost: e.cost })),
    stations: w.save.stations.map((s) => ({ num: Number(s.num), name: s.name, built: s.built, platform: nodes.get(Number(s.num))?.platform, tram: !!nodes.get(Number(s.num))?.kind })),
    lines: w.save.lines.map((l) => ({ num: Number(l.num), stops: l.stops.map(Number), status: w.lineStats[l.id]?.status, tram: !!w.lineStats[l.id]?.tram })),
  };
});
const quote = (edges, stations, level, kind) => page.evaluate(async ([edges, stations, level, kind]) => {
  const { queryClock } = await import('/src/workers/clockClient.ts');
  const d = await queryClock({ what: 'rebuildQuote', edges, stations, level, kind });
  const refused = d[3];
  return { price: d[0], changed: Array.from(d.subarray(5 + refused), Number), stations: d[2], refused: Array.from(d.subarray(5, 5 + refused), Number), red: d[4] };
}, [edges, stations, level, kind]);
const rebuild = (op) => page.evaluate((op) => window.tw.client.edit({ op: 'rebuild', ...op }), op);

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const built = await page.evaluate(async (c) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const free = (p) => ({ kind: 'free', x: p.x, y: p.y, level: 0 });
    const at = (n) => ({ kind: 'node', node: n });
    const res = [];
    const A = P(-4000, 0), M = P(0, 0), B = P(4000, 0);
    res.push(await cl.edit({ op: 'route', from: free(A), to: free(M), pis: [], single: false })); await pause(300);
    const nA = near(A), nM = near(M);
    res.push(await cl.edit({ op: 'route', from: at(nM), to: free(B), pis: [], single: false })); await pause(300);
    const nB = near(B);
    for (const n of [nA, nM, nB]) { res.push(await cl.edit({ op: 'addStation', at: at(n), platform: 160, name: '' })); await pause(200); }
    res.push(await cl.edit({ op: 'constructAll' })); await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: [nA, nM, nB], name: 'Rebuild check', colour: '#7b3fb3' });
    await pause(800);
    return { n: { A: nA, M: nM, B: nB }, line: line.line, ok: line.ok, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 140)) };
  }, C);
  check(built.ok && !built.refused.length, 'the run, its stations and its line are built', built.refused.join(' | '));
  if (!built.ok) throw new Error('no line');
  let s0 = await snap();
  const edges = s0.edges.filter((e) => e.built).map((e) => e.id);
  const stations = s0.stations.filter((s) => s.built).map((s) => s.num);
  check(edges.length >= 2 && stations.length === 3, 'built track and three stations', `${edges.length} pieces`);
  const names0 = s0.stations.map((s) => `${s.num}:${s.name}`).sort();
  const line0 = s0.lines[0];

  // a pick without the stations cannot ramp to their platforms: named by the quote, left alone by the rebuild
  const q1 = await quote(edges, [], 1, 0);
  check(q1.refused.length > 0, 'a viaduct pick without its stations names pieces it would refuse', JSON.stringify(q1));
  check(q1.refused.every((e) => edges.includes(e)), 'the refused pieces are in the pick');
  const r1 = await rebuild({ edges, stations: [], level: 1, kind: 0 });
  const s1 = await snap();
  check(r1.skipped === q1.refused.length, 'the rebuild skips the pieces the quote named', `skipped ${r1.skipped}, quoted ${q1.refused.length}`);
  for (const id of q1.refused) {
    const was = s0.edges.find((e) => e.id === id), now = s1.edges.find((e) => e.id === id);
    check(now.lo === was.lo && now.hi === was.hi, `refused piece ${id} stays as it was`, `${was.lo}..${was.hi} to ${now.lo}..${now.hi}`);
  }
  if (r1.ok) check(s1.cash < s0.cash, 'what did change was charged');
  else check(s1.cash === s0.cash, 'a rebuild that changes nothing charges nothing');

  // high speed on one piece
  const one = edges[0];
  const q2 = await quote([one], [], null, 2);
  check(q2.price > 0 && q2.changed.length === 1 && !q2.refused.length, 'high speed on one piece is quoted', JSON.stringify(q2));
  const c2 = (await snap()).cash;
  const r2 = await rebuild({ edges: [one], stations: [], level: null, kind: 2 });
  const s2 = await snap();
  check(r2.ok, 'high speed on one piece is rebuilt', JSON.stringify(r2.issues));
  check(s2.edges.find((e) => e.id === one).hsr && s2.edges.filter((e) => e.hsr).length === 1, 'only that piece is high speed now');
  check(Math.abs(c2 - s2.cash - q2.price * 1e6) < 2, 'the cash taken is the full new price of the piece', `taken ${c2 - s2.cash}, quoted ${q2.price * 1e6}`);
  check(s2.edges.find((e) => e.id === one).cost > s0.edges.find((e) => e.id === one).cost, 'it is priced as high speed now');

  // the Select tool: the whole run as a viaduct through the bar
  await page.evaluate(async (c) => {
    const tw = window.tw, ui = await import('/src/game/ui.ts');
    ui.tool.value = 'blueprint';
    tw.map.jumpTo({ center: [c[0] + 0.001, c[1]], zoom: 12.4 });
  }, C);
  await page.waitForTimeout(1500);
  // first a pick that leaves a station out, to see the refused piece drawn
  const partial = edges.slice(0, Math.max(1, edges.length - 1));
  await page.evaluate(async ([partial]) => {
    const pick = await import('/src/game/blueprintPick.ts');
    pick.bpPicked.value = { edges: partial, stations: [], built: true };
  }, [partial]);
  await page.locator('.blueprint-bar .btn.lv', { hasText: '+1' }).first().click();
  await page.waitForTimeout(1200);
  const tipPartial = await page.locator('.blueprint-bar button', { hasText: 'Rebuild' }).getAttribute('data-tip');
  console.log('hover with some pieces refused:', JSON.stringify(tipPartial));
  check(/can't change|can’t change/.test(tipPartial ?? ''), 'the Rebuild hover counts the pieces that cannot change');
  await page.screenshot({ path: join(ARTIFACTS, 't3059-refused.png') });

  const s3 = await snap();
  await page.evaluate(async ([edges, stations]) => {
    const pick = await import('/src/game/blueprintPick.ts');
    pick.bpPicked.value = { edges, stations, built: true };
  }, [edges, stations]);
  await page.waitForTimeout(500);
  await page.locator('.blueprint-bar .btn.lv', { hasText: '+1' }).first().click();
  await page.waitForTimeout(1200);
  const tip = await page.locator('.blueprint-bar button', { hasText: 'Rebuild' }).getAttribute('data-tip');
  console.log('hover:', JSON.stringify(tip));
  const price = Number(/\$([\d.]+)M/.exec(tip ?? '')?.[1]);
  check(price > 0 && !/change/.test(tip ?? ''), 'the Rebuild hover has a price and no refusal line', tip ?? '');
  const q3 = await quote(edges, stations, 1, 0);
  check(q3.refused.length === 0 && q3.stations === 3, 'the whole run with its stations has nothing refused', JSON.stringify(q3));
  await page.locator('.blueprint-bar button', { hasText: 'Rebuild' }).click();
  await page.waitForSelector('dialog.confirmation[open]', { timeout: 5000 });
  const ask = await page.locator('dialog.confirmation #confirmation-text').innerText();
  console.log('confirm:', ask);
  check(/Rebuild .* to level \+1 for \$/.test(ask) && /Nothing is refunded/.test(ask), 'the confirm names the work, the level and the price');
  await page.locator('dialog.confirmation .btn.danger').click();
  await page.waitForTimeout(2500);
  const s4 = await snap();
  check(Math.abs(s3.cash - s4.cash - q3.price * 1e6) < 2, 'the cash taken is exactly the quoted full price', `taken ${s3.cash - s4.cash}, quoted ${q3.price * 1e6}`);
  check(s4.edges.filter((e) => e.built).every((e) => e.lo === 1 && e.hi === 1), 'every piece is a viaduct now', JSON.stringify(s4.edges.map((e) => [e.id, e.lo, e.hi])));
  check(JSON.stringify(s4.stations.map((s) => `${s.num}:${s.name}`).sort()) === JSON.stringify(names0), 'every station keeps its id and name');
  const line4 = s4.lines.find((l) => l.num === line0.num);
  check(line4 && JSON.stringify(line4.stops) === JSON.stringify(line0.stops), 'the line keeps its stops');
  check(line4?.status === line0.status && line4.status !== 'broken', 'the line is still running', `${line0.status} to ${line4?.status}`);
  const undo = await page.evaluate(() => window.tw.world.value.canUndo ?? window.tw.world.value.save?.canUndo);
  check(!undo, 'undo is cleared', String(undo));
  const note = await page.evaluate(async () => (await import('/src/game/world.ts')).notice.value?.text ?? '');
  console.log('notice:', note);
  check(/pieces rebuilt to level \+1/.test(note), 'the notice says what was rebuilt', note);
  await page.screenshot({ path: join(ARTIFACTS, 't3059-after.png') });

  // T-3088: the run, its stations and its line as tram track through the bar's Tram button, then back to rail
  const s5 = await snap();
  const q5 = await quote(edges, stations, null, 3);
  check(q5.price > 0 && q5.changed.length === edges.length && !q5.refused.length && q5.stations === 3, 'tram track for the whole run is quoted', JSON.stringify(q5));
  await page.evaluate(async ([edges, stations]) => {
    const pick = await import('/src/game/blueprintPick.ts');
    pick.bpPicked.value = { edges, stations, built: true };
  }, [edges, stations]);
  await page.waitForTimeout(500);
  await page.locator('.blueprint-bar .group button', { hasText: /^Tram$/ }).click();
  await page.waitForTimeout(1200);
  const tipTram = await page.locator('.blueprint-bar button', { hasText: 'Rebuild' }).getAttribute('data-tip');
  console.log('tram hover:', JSON.stringify(tipTram));
  check(/to tram track/.test(tipTram ?? ''), 'the Rebuild hover says tram track', tipTram ?? '');
  await page.locator('.blueprint-bar button', { hasText: 'Rebuild' }).click();
  await page.waitForSelector('dialog.confirmation[open]', { timeout: 5000 });
  console.log('tram confirm:', await page.locator('dialog.confirmation #confirmation-text').innerText());
  await page.locator('dialog.confirmation .btn.danger').click();
  await page.waitForTimeout(2500);
  const s6 = await snap();
  check(s6.edges.filter((e) => e.built).every((e) => e.kind === 3), 'every piece is tram track', JSON.stringify(s6.edges.map((e) => [e.id, e.kind])));
  check(s6.stations.every((s) => s.tram && s.platform === 80), 'the stations are tram stops cut to 80 m', JSON.stringify(s6.stations));
  check(JSON.stringify(s6.stations.map((s) => `${s.num}:${s.name}`).sort()) === JSON.stringify(names0), 'the stops keep their ids and names');
  const line6 = s6.lines.find((l) => l.num === line0.num);
  check(line6?.tram && JSON.stringify(line6.stops) === JSON.stringify(line0.stops), 'the line is a tram line with the same stops', JSON.stringify(line6));
  check(line6?.status === line0.status, 'the tram line runs', `${line0.status} to ${line6?.status}`);
  check(s5.cash - s6.cash >= q5.price * 1e6 - 2, 'at least the quote is taken (trams too)', `taken ${s5.cash - s6.cash}, quoted ${q5.price * 1e6}`);
  await page.screenshot({ path: join(ARTIFACTS, 't3059-tram.png') });
  const r7 = await rebuild({ edges, stations, level: null, kind: 1 });
  await page.waitForTimeout(1500);
  const s7 = await snap();
  check(r7.ok && s7.edges.filter((e) => e.built).every((e) => e.kind === 0 && !e.hsr), 'and back to regular track', JSON.stringify(r7.issues));
  const line7 = s7.lines.find((l) => l.num === line0.num);
  check(line7 && !line7.tram && line7.status === line0.status && s7.stations.every((s) => !s.tram), 'a train line again, running, on stations', JSON.stringify(line7));

  // T-3087: a run at ground across the East River and Roosevelt Island, so at 0 with bridges at +1, rebuilt to -1
  const W = { lat: 40.7555, lngs: [-73.985, -73.975, -73.94, -73.925] };
  await page.evaluate(async () => { (await import('/src/game/blueprintPick.ts')).bpPicked.value = { edges: [], stations: [] }; });
  await page.evaluate((W) => window.tw.map.jumpTo({ center: [-73.955, W.lat], zoom: 13 }), W);
  await page.waitForTimeout(4000);
  const before = new Set((await snap()).edges.map((e) => e.id));
  const wet = await page.evaluate(async (W) => {
    const tw = window.tw, cl = tw.client;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const pts = W.lngs.map((lng) => tw.lngLatToPack(lng, W.lat));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const res = [];
    let from = { kind: 'free', x: pts[0].x, y: pts[0].y, level: 0 };
    for (const p of pts.slice(1)) {
      res.push(await cl.edit({ op: 'route', from, to: { kind: 'free', x: p.x, y: p.y, level: 0 }, pis: [], single: false })); await pause(300);
      from = { kind: 'node', node: near(p) };
    }
    const ends = [near(pts[0]), near(pts[3])];
    for (const n of ends) { res.push(await cl.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 160, name: '' })); await pause(200); }
    res.push(await cl.edit({ op: 'constructAll' })); await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: ends, name: 'Under the river', colour: '#1f6fd1' });
    await pause(800);
    return { ends, ok: line.ok, line: line.line, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 140)) };
  }, W);
  check(wet.ok && !wet.refused.length, 'the run across the river is built with its line', wet.refused.join(' | '));
  const s8 = await snap();
  const wetEdges = s8.edges.filter((e) => e.built && !before.has(e.id));
  console.log('pieces across the river (id, lowest, highest level):', JSON.stringify(wetEdges.map((e) => [e.id, e.lo, e.hi])));
  check(wetEdges.length >= 3 && wetEdges.some((e) => e.hi === 1) && wetEdges.some((e) => e.lo === 0), 'drawn at ground with bridges at +1');
  const wetIds = wetEdges.map((e) => e.id);
  const q8 = await quote(wetIds, wet.ends, -1, 0);
  check(!q8.refused.length && q8.changed.length === wetIds.length, 'nothing is refused going down to -1', JSON.stringify(q8));
  const r8 = await rebuild({ edges: wetIds, stations: wet.ends, level: -1, kind: 0 });
  await page.waitForTimeout(1500);
  const s9 = await snap();
  const after = s9.edges.filter((e) => wetIds.includes(e.id));
  console.log('after (id, lowest, highest level):', JSON.stringify(after.map((e) => [e.id, e.lo, e.hi])));
  check(r8.ok && r8.skipped === 0, 'the run is rebuilt', JSON.stringify(r8.issues));
  check(after.length === wetIds.length && after.every((e) => e.lo === -1 && e.hi === -1), 'every piece is at -1, none at 0 over the water');
  const lineW = s9.lines.find((l) => l.num === wet.line);
  check(lineW && lineW.status !== 'broken', 'the line under the river runs', lineW?.status);
  await page.evaluate(async (ids) => { (await import('/src/game/blueprintPick.ts')).bpPicked.value = { edges: ids, stations: [], built: true }; }, wetIds);
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(ARTIFACTS, 't3087-under-river.png') });
} catch (e) {
  fail++;
  console.log('FAIL exception', e?.stack ?? e);
}
check(!errors.length, 'no console or page errors', errors.slice(0, 3).join(' | '));
await chrome.close();
console.log(fail ? `FAIL: ${fail} failed` : 'PASS: all checks');
process.exit(fail ? 1 : 0);
