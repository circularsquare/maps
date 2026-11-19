// T-3408, T-3746: rebuilding one built piece clicked while plain selecting. The click makes the piece the pick, so the
// selection bar at the top of the map rebuilds it; the track panel shows information only. On a new game in New
// York, built with the edit API: a straight 8 km run of double track A-M-B with a station at each stop and a line over
// it; a lone piece of double track (nothing joined to it); and a blueprint stub. From the bar, with each piece
// clicked, the M-B piece goes high speed, back to regular, single, back to double and up to level +1, and the lone
// piece goes tram and back to regular, each by its buttons, the Rebuild hover and the confirm. Checks each step's cash
// taken is the quote, the piece ends as asked, the line keeps running, undo is cleared (the rebuild is final), the
// panel has no controls, and a clicked blueprint piece's bar has Construct and no Rebuild. Screenshots of the bar,
// light and dark: artifacts/trainworld/t3408_*.png.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3408-panel-rebuild.mjs [dev url]
// About two minutes.
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6107/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.668];

const chrome = await launchChrome({ prefix: 'tw-t3408-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);

const piece = (id) => page.evaluate((id) => {
  const w = window.tw.world.value, e = w.edges.find((x) => x.id === id);
  return e && { tracks: e.tracks, hsr: !!e.hsr, kind: e.kind ?? 0, lo: e.levelMin, hi: e.levelMax, cost: Math.round(e.cost), built: e.built, cash: w.save.cash, canUndo: !!w.canUndo, status: Object.values(w.lineStats)[0]?.status };
}, id);
/** the quote for one piece: [price US$M, edges changed, stations charged, edges refused] */
const quote = (edge, c) => page.evaluate(async ([edge, c]) => {
  const { queryClock } = await import('/src/workers/clockClient.ts');
  const a = await queryClock({ what: 'rebuildQuote', edges: [edge], stations: [], level: c.level ?? null, kind: c.kind ?? 0, tracks: c.tracks ?? 0 }), d = a.subarray(1, 1 + a[0]);
  return { price: d[0], changed: d[1], stations: d[2], refused: d[3] };
}, [edge, c]);
const waitFor = async (fn, what) => {
  for (let k = 0; k < 60; k++) {
    if (await fn()) return true;
    await page.waitForTimeout(400);
  }
  check(false, `waited for ${what}`);
  return false;
};
// a click on the piece while plain selecting, as the map's click sets it (main.ts `selectAt`)
const selectEdge = async (edge) => { await page.evaluate((edge) => { window.tw.selection.value = { kind: 'track', edge }; }, edge); await page.waitForTimeout(1200); };
const bar = () => page.locator('.blueprint-bar').first();
const barShot = async (file) => { await page.waitForTimeout(600); await bar().screenshot({ path: join(ARTIFACTS, file) }); };
const shade = (s) => page.evaluate(async (s) => { (await import('/src/game/ui.ts')).shade.value = s; }, s);
const levelBtn = (root, label) => root.locator('.group.levels .btn.lv', { hasText: new RegExp(`^${label.replace('+', '\\+')}$`) }).first();

// the steps: the piece, the button to press in the bar, the quote's choice, and what the piece becomes
const STEPS = [
  { name: 'high speed', on: 'mb', button: (r) => r.locator('button', { hasText: /^High speed$/ }), c: { kind: 2 }, want: (p) => p.tracks === 2 && p.hsr },
  { name: 'regular', on: 'mb', button: (r) => r.locator('button', { hasText: /^Regular$/ }), c: { kind: 1 }, want: (p) => p.tracks === 2 && !p.hsr },
  { name: 'single track', on: 'mb', button: (r) => r.locator('button[aria-label="Single track"]'), c: { tracks: 1 }, want: (p) => p.tracks === 1 },
  { name: 'double track', on: 'mb', button: (r) => r.locator('button[aria-label="Double track"]'), c: { tracks: 2 }, want: (p) => p.tracks === 2 },
  { name: 'level +1', on: 'mb', button: (r) => levelBtn(r, '+1'), c: { level: 1 }, want: (p) => p.lo === 1 && p.hi === 1 },
  { name: 'tram', on: 'iso', button: (r) => r.locator('button', { hasText: /^Tram$/ }), c: { kind: 3 }, want: (p) => p.kind !== 0 },
  { name: 'back to rail', on: 'iso', button: (r) => r.locator('button', { hasText: /^Regular$/ }), c: { kind: 1 }, want: (p) => p.kind === 0 && !p.hsr },
];

/** One step through the bar: press the choice, read the Rebuild hover, confirm. */
async function step(s, edge) {
  const before = await piece(edge);
  const q = await quote(edge, s.c);
  await s.button(bar()).click();
  const rebuildBtn = () => bar().locator('button', { hasText: /Rebuild/ });
  await waitFor(async () => !(await rebuildBtn().isDisabled()), `${s.name}: Rebuild to be offered`);
  const tip = await rebuildBtn().getAttribute('data-tip');
  if (s.name === 'level +1') {
    await barShot('t3408_levels_light.png');
    await shade('dark');
    await barShot('t3408_levels_dark.png');
    await shade('light');
  }
  await rebuildBtn().click();
  await page.waitForSelector('dialog.confirmation[open]', { timeout: 5000 });
  const ask = await page.locator('dialog.confirmation #confirmation-text').innerText();
  await page.locator('dialog.confirmation .btn.danger').click();
  await waitFor(async () => s.want(await piece(edge)), `${s.name}: the piece to change`);
  await page.waitForTimeout(600);
  const after = await piece(edge);
  const taken = before.cash - after.cash;
  check(s.want(after), `${s.name} rebuilt`, JSON.stringify(after));
  check(Math.abs(taken - q.price * 1e6) < 2, `${s.name} takes the quote`, `taken ${(taken / 1e6).toFixed(3)}M, quoted ${q.price.toFixed(3)}M`);
  check(!after.canUndo, `${s.name} clears undo (final)`);
  check(after.status && after.status !== 'broken', `the line still runs after ${s.name}`, after.status);
  console.log(`   hover ${JSON.stringify(tip)}\n   ask   ${JSON.stringify(ask)}`);
  return { tip, taken };
}

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
    const edgeOf = (a, b) => tw.world.value.edges.find((e) => (e.a === a && e.b === b) || (e.a === b && e.b === a))?.id;
    const res = [];
    // the run A-M-B with its stations
    const A = P(-4000, 0), M = P(0, 0), B = P(4000, 0);
    res.push(await cl.edit({ op: 'route', from: free(A), to: free(M), pis: [], single: false })); await pause(300);
    const nA = near(A), nM = near(M);
    res.push(await cl.edit({ op: 'route', from: at(nM), to: free(B), pis: [], single: false })); await pause(300);
    const nB = near(B);
    for (const n of [nA, nM, nB]) { res.push(await cl.edit({ op: 'addStation', at: at(n), platform: 160, name: '' })); await pause(200); }
    // the lone piece
    res.push(await cl.edit({ op: 'route', from: free(P(-4000, 3000)), to: free(P(0, 3000)), pis: [], single: false })); await pause(300);
    const iso = edgeOf(near(P(-4000, 3000)), near(P(0, 3000)));
    res.push(await cl.edit({ op: 'constructAll' })); await pause(500);
    // a blueprint stub
    res.push(await cl.edit({ op: 'route', from: free(P(-4000, 6000)), to: free(P(0, 6000)), pis: [], single: false })); await pause(300);
    const stub = edgeOf(near(P(-4000, 6000)), near(P(0, 6000)));
    const line = await cl.edit({ op: 'addLine', stops: [nA, nM, nB], name: 'Panel rebuild', colour: '#7b3fb3' });
    await pause(800);
    return { ok: line.ok, mb: edgeOf(nM, nB), iso, stub, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 140)) };
  }, C);
  check(built.ok && [built.mb, built.iso, built.stub].every((x) => x !== undefined) && !built.refused.length, 'the run with its stations and line, the lone piece and a blueprint stub are built', JSON.stringify(built));
  if (!built.ok || built.mb === undefined) throw new Error('no line');
  const edges = { mb: built.mb, iso: built.iso };
  await page.evaluate((c) => { window.tw.map.jumpTo({ center: c, zoom: 12.4 }); }, C);
  const p0 = await piece(built.mb);
  check(p0.built && p0.tracks === 2 && !p0.hsr, 'the M-B piece is built double regular track', JSON.stringify(p0));

  // a clicked blueprint piece: the bar has Construct and no Rebuild
  await selectEdge(built.stub);
  await bar().waitFor({ timeout: 10000 });
  const stubBar = (await bar().innerText()).replace(/\s+/g, ' ');
  check(/Construct for/.test(stubBar) && !/Rebuild/.test(stubBar), 'a clicked blueprint piece\'s bar has Construct, no Rebuild', stubBar);

  // a clicked built piece: the panel informs, the bar acts
  await selectEdge(built.mb);
  await bar().waitFor({ timeout: 10000 });
  const pick = await page.evaluate(async () => { const p = (await import('/src/game/blueprintPick.ts')).bpPicked.value; return { edges: p.edges, built: !!p.built }; });
  check(pick.edges.length === 1 && pick.edges[0] === built.mb && pick.built, 'the click makes the piece the pick', JSON.stringify(pick));
  const panelText = (await page.locator('#inspector').innerText()).replace(/\s+/g, ' ');
  check((await page.locator('#inspector button.btn.text').count()) === 0 && !/Rebuild|Remove|Use the Select tool/.test(panelText), 'the track panel has no controls', panelText.slice(0, 300));
  check(await bar().locator('button', { hasText: /Rebuild/ }).isDisabled(), 'Rebuild is greyed until a choice is made');
  await barShot('t3408_bar_light.png');
  const done = [];
  for (const s of STEPS) {
    await selectEdge(edges[s.on]);
    await bar().waitFor({ timeout: 10000 });
    done.push(await step(s, edges[s.on]));
  }
  await selectEdge(built.mb);
  await page.waitForTimeout(1000);
  check(await levelBtn(bar(), '+1').evaluate((b) => b.classList.contains('on')), 'after the steps the bar shows M-B at +1');
  // a choice made for one piece is dropped when another piece is clicked
  await bar().locator('button', { hasText: /^High speed$/ }).click();
  await selectEdge(built.iso);
  await page.waitForTimeout(600);
  check(await bar().locator('button', { hasText: /Rebuild/ }).isDisabled(), 'clicking another piece drops the choice');
  await shade('dark');
  await barShot('t3408_bar_dark.png');
  await shade('light');
  check(done[0].taken > 0 && done[2].taken === 0 && /Free/.test(done[2].tip ?? ''), 'high speed costs, single track is free');
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't3408_died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3408 FAILED: ${fail}` : 'T-3408 PASSED');
process.exit(fail ? 1 : 0);
