// T-3059: rebuilding built track in place. On a new game in New York: a straight 8 km run with a station at
// each end and one in the middle, built, with a line over it. Checks: a quote names the pieces a rebuild would
// refuse (a pick without the middle station cannot ramp to it) and a rebuild leaves them alone; high speed on
// one piece changes only that piece; a viaduct rebuild of the whole run charges exactly its quote, keeps every
// station's id and name and the line's stops, leaves the line running and clears undo; then the same through
// the Select tool's bar: the level button, the Rebuild hover's price, the confirm and its yes. Screenshots
// artifacts/trainworld/t3059-*.png (the refused piece outlined in red).
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3059-rebuild.mjs [dev url]
// About a minute.
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
  return {
    cash: w.save.cash,
    edges: w.edges.map((e) => ({ id: Number(e.id), built: e.built, hsr: !!e.hsr, lo: e.levelMin, hi: e.levelMax, len: e.lengthM, cost: e.cost })),
    stations: w.save.stations.map((s) => ({ num: Number(s.num), name: s.name, built: s.built })),
    lines: w.save.lines.map((l) => ({ num: Number(l.num), stops: l.stops.map(Number), status: w.lineStats[l.id]?.status })),
  };
});
const quote = (edges, stations, level, kind) => page.evaluate(async ([edges, stations, level, kind]) => {
  const { queryClock } = await import('/src/workers/clockClient.ts');
  const d = await queryClock({ what: 'rebuildQuote', edges, stations, level, kind });
  const refused = d[3];
  return { price: d[0], changed: Array.from(d.subarray(4 + refused), Number), stations: d[2], refused: Array.from(d.subarray(4, 4 + refused), Number) };
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
} catch (e) {
  fail++;
  console.log('FAIL exception', e?.stack ?? e);
}
check(!errors.length, 'no console or page errors', errors.slice(0, 3).join(' | '));
await chrome.close();
console.log(fail ? `FAIL: ${fail} failed` : 'PASS: all checks');
process.exit(fail ? 1 : 0);
