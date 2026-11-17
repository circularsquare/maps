// T-3408: rebuilding one built piece from the track panel, against the Select tool's bar on the same piece. On a
// new game in New York, built with the edit API: a straight 8 km run of double track A-M-B with a station at each
// stop and a line over it; a lone piece of double track (nothing joined to it); a straight run of three pieces with
// no stations, whose middle piece is tied at both ends; and a blueprint stub. The game is exported, then from the
// panel the M-B piece goes high speed, back to regular, single, back to double and up to level +1, and the lone
// piece goes tram and back to regular, each by its buttons, the Rebuild hover and the confirm; then the export is
// loaded again and the same steps go through the bar with that one piece picked. Checks each step's hover, question,
// quote and cash taken are the same both ways and the cash taken is the quote, the piece ends as asked, the line
// keeps running, undo is cleared as for the bar (the rebuild is final). Also: the panel offers a level or Tram only
// where the quote for that one piece refuses nothing (M-B: levels, no Tram; the lone piece: Tram; the tied piece:
// no levels, no Tram), the hint to the Select tool, and no rebuild on a blueprint piece. Screenshots of the panel,
// light and dark: artifacts/trainworld/t3408_*.png.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3408-panel-rebuild.mjs [dev url]
// About three minutes.
import { ARTIFACTS, collectErrors, importSave, launchChrome } from './lib.mjs';
import { readFile } from 'node:fs/promises';
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
const whole = (q) => q.refused === 0 && q.changed + q.stations > 0;
const waitFor = async (fn, what) => {
  for (let k = 0; k < 60; k++) {
    if (await fn()) return true;
    await page.waitForTimeout(400);
  }
  check(false, `waited for ${what}`);
  return false;
};
const selectEdge = async (edge) => { await page.evaluate((edge) => { window.tw.selection.value = { kind: 'track', edge }; }, edge); await page.waitForTimeout(1500); };
const panel = () => page.locator('#inspector:has(.piece-rebuild), .inspector:has(.piece-rebuild)').first();
const shot = async (file) => {
  const p = await page.$('#inspector, .inspector');
  await page.waitForTimeout(600);
  await (p ?? page).screenshot({ path: join(ARTIFACTS, file) });
};
const shade = (s) => page.evaluate(async (s) => { (await import('/src/game/ui.ts')).shade.value = s; }, s);
const levelBtn = (root, label) => root.locator('.group.levels .btn.lv', { hasText: new RegExp(`^${label.replace('+', '\\+')}$`) }).first();

// the steps: the piece, the button to press in the panel or the bar, the quote's choice, and what the piece becomes
const STEPS = [
  { name: 'high speed', on: 'mb', button: (r) => r.locator('button', { hasText: /^High speed$/ }), c: { kind: 2 }, want: (p) => p.tracks === 2 && p.hsr },
  { name: 'regular', on: 'mb', button: (r) => r.locator('button', { hasText: /^Regular$/ }), c: { kind: 1 }, want: (p) => p.tracks === 2 && !p.hsr },
  { name: 'single track', on: 'mb', button: (r) => r.locator('button[aria-label="Single track"]'), c: { tracks: 1 }, want: (p) => p.tracks === 1 },
  { name: 'double track', on: 'mb', button: (r) => r.locator('button[aria-label="Double track"]'), c: { tracks: 2 }, want: (p) => p.tracks === 2 },
  { name: 'level +1', on: 'mb', button: (r) => levelBtn(r, '+1'), c: { level: 1 }, want: (p) => p.lo === 1 && p.hi === 1 },
  { name: 'tram', on: 'iso', button: (r) => r.locator('button', { hasText: /^Tram$/ }), c: { kind: 3 }, want: (p) => p.kind !== 0 },
  { name: 'back to rail', on: 'iso', button: (r) => r.locator('button', { hasText: /^Regular$/ }), c: { kind: 1 }, want: (p) => p.kind === 0 && !p.hsr },
];

/** One step through `root` (the panel or the bar): press the choice, read the Rebuild hover, confirm. */
async function step(root, s, edge, label) {
  const before = await piece(edge);
  const q = await quote(edge, s.c);
  await s.button(root()).click();
  const rebuildBtn = () => root().locator('button', { hasText: /Rebuild/ });
  await waitFor(async () => !(await rebuildBtn().isDisabled()), `${label} ${s.name}: Rebuild to be offered`);
  const tip = await rebuildBtn().getAttribute('data-tip');
  if (label === 'panel' && s.name === 'level +1') {
    await shot('t3408_levels_light.png');
    await shade('dark');
    await shot('t3408_levels_dark.png');
    await shade('light');
  }
  await rebuildBtn().click();
  await page.waitForSelector('dialog.confirmation[open]', { timeout: 5000 });
  const ask = await page.locator('dialog.confirmation #confirmation-text').innerText();
  await page.locator('dialog.confirmation .btn.danger').click();
  await waitFor(async () => s.want(await piece(edge)), `${label} ${s.name}: the piece to change`);
  await page.waitForTimeout(600);
  const after = await piece(edge);
  if (label === 'panel' && s.name === 'single track') await shot('t3408_single_light.png');
  const taken = before.cash - after.cash;
  check(s.want(after), `${label}: ${s.name} rebuilt`, JSON.stringify(after));
  check(Math.abs(taken - q.price * 1e6) < 2, `${label}: ${s.name} takes the quote`, `taken ${(taken / 1e6).toFixed(3)}M, quoted ${q.price.toFixed(3)}M`);
  check(!after.canUndo, `${label}: ${s.name} clears undo (final, as the bar's)`);
  check(after.status && after.status !== 'broken', `${label}: the line still runs after ${s.name}`, after.status);
  return { tip, ask, taken: Math.round(taken), price: q.price, cost: after.cost, changed: q.changed, refused: q.refused };
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
    // three pieces in a row, the middle one tied at both ends
    const T = [P(-4000, -3000), P(-2000, -3000), P(0, -3000), P(2000, -3000)];
    res.push(await cl.edit({ op: 'route', from: free(T[0]), to: free(T[1]), pis: [], single: false })); await pause(300);
    for (let k = 1; k < 3; k++) { res.push(await cl.edit({ op: 'route', from: at(near(T[k])), to: free(T[k + 1]), pis: [], single: false })); await pause(300); }
    const tied = edgeOf(near(T[1]), near(T[2]));
    res.push(await cl.edit({ op: 'constructAll' })); await pause(500);
    // a blueprint stub, for the blueprint panel's check
    res.push(await cl.edit({ op: 'route', from: free(P(-4000, 6000)), to: free(P(0, 6000)), pis: [], single: false })); await pause(300);
    const stub = edgeOf(near(P(-4000, 6000)), near(P(0, 6000)));
    const line = await cl.edit({ op: 'addLine', stops: [nA, nM, nB], name: 'Panel rebuild', colour: '#7b3fb3' });
    await pause(800);
    return { ok: line.ok, mb: edgeOf(nM, nB), iso, tied, stub, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 140)) };
  }, C);
  check(built.ok && [built.mb, built.iso, built.tied, built.stub].every((x) => x !== undefined) && !built.refused.length, 'the run with its stations and line, the lone piece, the tied piece and a blueprint stub are built', JSON.stringify(built));
  if (!built.ok || built.mb === undefined) throw new Error('no line');
  const edges = { mb: built.mb, iso: built.iso };
  await page.evaluate((c) => { window.tw.map.jumpTo({ center: c, zoom: 12.4 }); }, C);
  const p0 = await piece(built.mb);
  check(p0.built && p0.tracks === 2 && !p0.hsr, 'the M-B piece is built double regular track', JSON.stringify(p0));

  // the blueprint stub's panel has no rebuild
  await selectEdge(built.stub);
  await page.locator('.top-speed').first().waitFor({ timeout: 20000 });
  check((await page.locator('.piece-rebuild').count()) === 0, 'a blueprint piece\'s panel has no rebuild');

  // which levels and kinds the panel offers: those whose quote for the piece alone refuses nothing
  const offered = async (edge) => {
    await selectEdge(edge);
    await panel().waitFor({ timeout: 20000 });
    await page.waitForTimeout(1500);
    const levels = (await panel().locator('.group.levels .btn.lv').allInnerTexts()).map((s) => s.trim());
    const tram = (await panel().locator('button', { hasText: /^Tram$/ }).count()) > 0;
    const open = [];
    for (const level of [-3, -2, -1, 1, 2, 3]) if (whole(await quote(edge, { level }))) open.push(level);
    return { levels, tram, open, tramOpen: whole(await quote(edge, { kind: 3 })) };
  };
  const label = (l) => (l > 0 ? `+${l}` : l < 0 ? `−${-l}` : '0');
  const sameLevels = (o) => JSON.stringify(o.levels.filter((x) => x !== '0').sort()) === JSON.stringify(o.open.map(label).sort());
  const oMb = await offered(built.mb), oIso = await offered(built.iso), oTied = await offered(built.tied);
  console.log('offered:', JSON.stringify({ mb: oMb, iso: oIso, tied: oTied }));
  check(oMb.levels.length > 1 && sameLevels(oMb), 'M-B offers its own level and every level its quote takes whole', JSON.stringify(oMb));
  check(!oMb.tram && !oMb.tramOpen, 'M-B, joined to rail, offers no Tram, and its tram quote refuses it');
  check(oIso.tram && oIso.tramOpen, 'the lone piece offers Tram');
  check(oTied.levels.length === 0 && oTied.open.length === 0 && !oTied.tram && !oTied.tramOpen, 'the tied piece offers no level and no Tram, and no level quote takes it', JSON.stringify(oTied));
  await shot('t3408_tied_light.png');

  // the save as it is now, for the bar's run
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());

  // the panel
  await selectEdge(built.mb);
  await panel().waitFor({ timeout: 20000 });
  const text = (await panel().innerText()).replace(/\s+/g, ' ');
  check(/Use the Select tool \(4\) to rebuild longer stretches/.test(text), 'the panel has the hint to the Select tool (4)', text.slice(0, 300));
  check((await panel().locator('button', { hasText: /Rebuild/ }).isDisabled()), 'Rebuild is greyed until a choice is made');
  await shot('t3408_panel_light.png');
  const viaPanel = [];
  for (const s of STEPS) {
    if (s.on === 'iso' && s.name === 'tram') { await selectEdge(edges.iso); await panel().waitFor({ timeout: 20000 }); }
    viaPanel.push(await step(panel, s, edges[s.on], 'panel'));
  }
  await selectEdge(built.mb);
  await panel().waitFor({ timeout: 20000 });
  await page.waitForTimeout(1000);
  check(await levelBtn(panel(), '+1').evaluate((b) => b.classList.contains('on')), 'after the steps the panel shows M-B at +1');
  await panel().locator('button', { hasText: /^High speed$/ }).click();
  await panel().locator('button', { hasText: /^Regular$/ }).click(); // what the piece has: drops the choice
  await page.waitForTimeout(400);
  check(await panel().locator('button', { hasText: /Rebuild/ }).isDisabled(), 'picking what the piece has drops the choice');
  await shade('dark');
  await shot('t3408_panel_dark.png');
  await shade('light');

  // the bar, from the same save
  await importSave(page, bytes);
  await waitFor(async () => { const p = await piece(built.mb); return p && p.tracks === 2 && !p.hsr && p.lo === 0 && Math.abs(p.cash - p0.cash) < 1; }, 'the save to load again');
  await page.waitForTimeout(2500);
  const pickOne = (edge) => page.evaluate(async (edge) => {
    const ui = await import('/src/game/ui.ts'), pick = await import('/src/game/blueprintPick.ts');
    window.tw.selection.value = null;
    ui.tool.value = 'blueprint';
    pick.bpPicked.value = { edges: [edge], stations: [], built: true };
  }, edge);
  const bar = () => page.locator('.blueprint-bar').first();
  const viaBar = [];
  for (const s of STEPS) {
    await pickOne(edges[s.on]);
    await bar().waitFor({ timeout: 10000 });
    await page.waitForTimeout(300);
    viaBar.push(await step(bar, s, edges[s.on], 'bar'));
  }

  for (let k = 0; k < STEPS.length; k++) {
    const a = viaPanel[k], b = viaBar[k];
    console.log(`${STEPS[k].name}: panel ${a.taken / 1e6}M (quote ${a.price.toFixed(3)}M, piece cost ${a.cost / 1e6}M) | bar ${b.taken / 1e6}M (quote ${b.price.toFixed(3)}M, piece cost ${b.cost / 1e6}M)`);
    console.log(`   hover ${JSON.stringify(a.tip)}\n   ask   ${JSON.stringify(a.ask)}`);
    check(a.tip === b.tip && a.ask === b.ask, `${STEPS[k].name}: the same hover and question from the panel and the bar`, a.tip === b.tip ? '' : `${JSON.stringify(a.tip)} vs ${JSON.stringify(b.tip)}`);
    check(a.taken === b.taken && a.cost === b.cost && a.changed === b.changed && a.refused === b.refused, `${STEPS[k].name}: the same charge and result from the panel and the bar`, `${a.taken} vs ${b.taken}`);
  }
  check(viaPanel[0].taken > 0 && viaPanel[2].taken === 0 && /Free/.test(viaPanel[2].tip ?? ''), 'high speed costs, single track is free');
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't3408_died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3408 FAILED: ${fail}` : 'T-3408 PASSED');
process.exit(fail ? 1 : 0);
